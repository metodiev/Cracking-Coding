/**
 * Task 668 - Implement priority inheritance protocol simulation for priority inversion.
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Discrete-tick simulator of a preemptive fixed-priority uniprocessor with mutexes. A task is a list
 * of operations - cpu(time), lock(name), unlock(name) - so critical sections are explicit and the
 * blocking of a high-priority task by a low-priority one can be measured exactly. Lock/unlock happen
 * in a *prepare* phase at the start of each tick (in declaration order) and cost no tick; only cpu
 * operations consume ticks. A task needing a held mutex records `blockedFor` and stops; the scheduler
 * then runs one tick of the runnable task with the highest *effective* priority, where
 *   pip(P) = max(P.priority, max { pip(W) : W is blocked on a mutex held by P })
 * is iterated to a fixed point, which makes the priority inheritance protocol (PIP) transitive. With
 * priorityInheritance false the same simulator reproduces plain fixed priority, i.e. unbounded
 * priority inversion.
 *
 * Verified: (1) the textbook inversion scenario blocks HIGH for the whole MED computation without PIP
 * and for exactly the critical section with PIP; (2) the invariant "nobody waits for a mutex held by
 * a lower effective priority task" holds on every tick under PIP and fails on every inversion tick
 * without it; (3) a brute-force BFS over the scheduler state space brackets the PIP completion
 * (optimum <= PIP <= plain); (4) two-mutex chains show transitive inheritance; (5) PIP is shown not
 * to prevent the A/B lock-order deadlock, with the invariant intact at the stall; (6) with no locks
 * the run matches the hand-computed fixed-priority schedule; (7) alternative tie-breaks and a seeded
 * PRNG keep the invariants and repeated runs are bit-identical.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed668n;

/** Deterministic 64-bit splitmix64, used for the seeded tie-break experiment. */
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------- task model
/** {kind:"cpu", time} | {kind:"lock", name} | {kind:"unlock", name} */
const cpu = (time) => ({ kind: "cpu", time });
const lock = (name) => ({ kind: "lock", name });
const unlock = (name) => ({ kind: "unlock", name });
const task = (id, priority, ops) => ({ id, priority, ops });

/** Load the work counter of the operation the task's cursor points at. */
function enterOp(run) {
  const op = run.ops[run.cursor];
  run.remaining = op && op.kind === "cpu" ? op.time : 0;
}

function cloneRun(tasks) {
  const run = tasks.map((t) => ({ id: t.id, priority: t.priority, ops: t.ops, cursor: 0,
    remaining: 0, blockedFor: null, done: false, finish: null, blockedTicks: 0 }));
  for (const t of run) enterOp(t); // the cursor already sits on the first operation
  return run;
}

/**
 * Prepare phase: resolve lock/unlock operations (instantaneous) until every task reaches a cpu
 * operation, finishes, or blocks on a mutex somebody else holds.
 */
function prepareTick(run, locks, tick) {
  for (const t of run) {
    if (t.done) continue;
    for (;;) {
      const op = t.ops[t.cursor];
      if (!op || op.kind === "cpu") break;
      if (op.kind === "lock") {
        if (locks.has(op.name)) {
          t.blockedFor = op.name;
          break;
        }
        locks.set(op.name, t.id);
        t.blockedFor = null;
      } else if (locks.get(op.name) === t.id) {
        locks.delete(op.name);
      }
      t.cursor += 1;
      enterOp(t);
    }
    if (t.cursor >= t.ops.length) {
      t.done = true;
      if (t.finish === null) t.finish = tick + 1;
    }
  }
}

/**
 * Effective priorities. The inheritance step is iterated to a fixed point: a holder inherits the
 * (possibly already inherited) priority of every waiter, which makes chains of blocking transitive.
 */
function effectivePriorities(run, locks, inherit) {
  const eff = new Map(run.map((t) => [t.id, t.priority]));
  if (!inherit) return eff;
  let changed = true;
  while (changed) {
    changed = false;
    for (const waiter of run) {
      if (waiter.done || !waiter.blockedFor) continue;
      const holderId = locks.get(waiter.blockedFor);
      if (holderId === undefined) continue;
      const holder = run.find((t) => t.id === holderId);
      if (holder && !holder.done && eff.get(holder.id) < eff.get(waiter.id)) {
        eff.set(holder.id, eff.get(waiter.id));
        changed = true;
      }
    }
  }
  return eff;
}

/** Runnable tasks, highest effective priority first, with the requested tie-break rule. */
function readyOrder(run, locks, options) {
  const eff = effectivePriorities(run, locks, options.priorityInheritance);
  const ready = run.filter((t) => !t.done && !t.blockedFor && t.cursor < t.ops.length);
  ready.sort((a, b) => {
    const delta = eff.get(b.id) - eff.get(a.id);
    if (delta !== 0) return delta;
    if (options.tieBreak === "high-id") return a.id < b.id ? 1 : -1;
    if (options.tieBreak === "seeded") return (a._tie - b._tie) || (a.id < b.id ? -1 : 1);
    return a.id < b.id ? -1 : 1;
  });
  return { eff, ready };
}

/** Run one tick of `chosen`'s current cpu operation; mutexes are touched only in the prepare phase. */
function executeTick(chosen, tick) {
  const op = chosen.ops[chosen.cursor];
  if (!op) return;
  if (op.kind === "cpu") {
    chosen.remaining -= 1;
    if (chosen.remaining <= 0) {
      chosen.cursor += 1;
      enterOp(chosen);
    }
  }
  if (chosen.cursor >= chosen.ops.length) {
    chosen.done = true;
    chosen.finish = tick + 1;
  }
}

function stateKey(run, locks) {
  const progress = run.map((t) => `${t.cursor}.${t.remaining}.${t.done ? 1 : 0}`).join(",");
  const held = [...locks.entries()].sort().map(([name, id]) => `${name}=${id}`).join(";");
  return `${progress}|${held}`;
}

// -------------------------------------------------------------------- scheduler and brute force
function simulate(tasks, options = {}) {
  const opts = { priorityInheritance: true, tieBreak: "low-id", tickLimit: 500, ...options };
  const run = cloneRun(tasks);
  if (opts.tieBreak === "seeded") {
    const rng = new SplitMix64(opts.seed ?? DEFAULT_SEED);
    for (const t of run) t._tie = rng.nextInt(4);
  }
  const locks = new Map();
  const finish = new Map();
  const blocked = new Map();
  const violations = [];
  const trace = [];
  const maxEffective = new Map(run.map((t) => [t.id, t.priority]));
  let tick = 0;
  let idleTicks = 0;
  let deadlocked = false;

  while (tick < opts.tickLimit && run.some((t) => !t.done)) {
    prepareTick(run, locks, tick);
    const { eff, ready } = readyOrder(run, locks, opts);
    for (const t of run) maxEffective.set(t.id, Math.max(maxEffective.get(t.id), eff.get(t.id)));
    // The invariant PIP promises: a waiter's blocker is never less urgent than the waiter.
    for (const waiter of run) {
      const blocked = waiter.done || !waiter.blockedFor;
      const holderId = blocked ? undefined : locks.get(waiter.blockedFor);
      if (holderId === undefined) continue;
      if (eff.get(holderId) < eff.get(waiter.id)) {
        violations.push({ tick, waiter: waiter.id, holder: holderId,
          holderPriority: run.find((h) => h.id === holderId).priority });
      }
    }
    if (ready.length === 0) {
      deadlocked = run.some((t) => !t.done);
      if (deadlocked) idleTicks += 1;
      break;
    }
    const chosen = ready[0];
    executeTick(chosen, tick);
    trace.push({ tick, id: chosen.id });
    // Blocked time is booked after the runnable task consumed the tick.
    for (const t of run) if (!t.done && t.blockedFor) t.blockedTicks += 1;
    tick += 1;
  }
  for (const t of run) {
    if (t.done) finish.set(t.id, t.finish);
    blocked.set(t.id, t.blockedTicks);
  }
  const makespan = finish.size ? Math.max(...finish.values()) : 0;
  return { finish, blocked, makespan, ticks: tick, idleTicks, deadlocked, violations, trace,
    maxEffective };
}

/**
 * Brute force reference: BFS over the whole scheduler state space (any runnable task may be picked on
 * any tick) giving the minimum number of ticks after which `targetId` finishes - the best any
 * uniprocessor scheduler could achieve for that task.
 */
function optimalCompletionTime(tasks, targetId, options = {}) {
  const opts = { priorityInheritance: true, tieBreak: "low-id", maxTicks: 120, ...options };
  const start = { run: cloneRun(tasks), locks: new Map() };
  const visited = new Set([stateKey(start.run, start.locks)]);
  let frontier = [start];
  let explored = 1;
  for (let tick = 0; tick < opts.maxTicks; tick++) {
    const next = [];
    for (const node of frontier) {
      prepareTick(node.run, node.locks, tick);
      if (node.run.find((t) => t.id === targetId).done) return { ticks: tick + 1, explored };
      for (const candidate of readyOrder(node.run, node.locks, opts).ready) {
        const child = { run: node.run.map((t) => ({ ...t })), locks: new Map(node.locks) };
        const chosen = child.run.find((t) => t.id === candidate.id);
        executeTick(chosen, tick);
        const key = stateKey(child.run, child.locks);
        if (visited.has(key)) continue;
        visited.add(key);
        explored += 1;
        if (chosen.done) return { ticks: tick + 1, explored };
        next.push(child);
      }
    }
    if (next.length === 0) break;
    frontier = next;
  }
  return { ticks: Infinity, explored };
}

// -------------------------------------------------------------------------------------- scenarios
/** Textbook inversion: LOW holds the mutex, HIGH needs it, MED is an unrelated CPU hog. */
const inversionScenario = () => [
  task("LOW", 1, [lock("mutex"), cpu(10), unlock("mutex")]),
  task("MED", 2, [cpu(30)]),
  task("HIGH", 3, [cpu(5), lock("mutex"), cpu(5), unlock("mutex")]),
];

/** Two-mutex chain: HIGH waits on B (held by MID), MID waits on A (held by LOW). */
const transitiveInheritance = () => [
  task("LOW", 1, [lock("A"), cpu(8), unlock("A")]),
  task("MID", 2, [lock("B"), cpu(4), lock("A"), cpu(6), unlock("A"), unlock("B")]),
  task("HIGH", 3, [cpu(2), lock("B"), cpu(4), unlock("B")]),
];

/** A/B lock-order inversion: no protocol ordering avoids the circular wait. */
const deadlockScenario = () => [
  task("T1", 1, [lock("A"), cpu(2), lock("B"), cpu(2), unlock("B"), unlock("A")]),
  task("T2", 2, [lock("B"), cpu(2), lock("A"), cpu(2), unlock("A"), unlock("B")]),
];

function main() {
  // (1) Inversion scenario: blocking of HIGH with and without inheritance.
  const plain = simulate(inversionScenario(), { priorityInheritance: false });
  const pip = simulate(inversionScenario(), { priorityInheritance: true });
  const cs = 10;
  assert.ok(plain.blocked.get("HIGH") >= 30, `without inheritance HIGH is blocked ${plain.blocked.get("HIGH")} ticks: the whole MED run plus the critical section`);
  assert.strictEqual(pip.blocked.get("HIGH"), cs, "with PIP the inversion is bounded by the section");
  assert.ok(pip.finish.get("HIGH") < plain.finish.get("HIGH"),
    `HIGH finishes at ${pip.finish.get("HIGH")} with PIP vs ${plain.finish.get("HIGH")} without`);
  assert.ok(pip.finish.get("MED") > plain.finish.get("MED"),
    `trade-off: MED is delayed (${pip.finish.get("MED")} vs ${plain.finish.get("MED")}): the boosted ` +
    "low-priority holder preempts it");
  assert.strictEqual(pip.maxEffective.get("LOW"), 3, "LOW reaches HIGH's priority while holding");
  assert.strictEqual(plain.maxEffective.get("LOW"), 1, "and stays at 1 without inheritance");

  // (2) The PIP invariant, checked on every tick.
  assert.strictEqual(pip.violations.length, 0, "with PIP nobody waits on a lower-priority holder");
  assert.ok(plain.violations.length >= 30, `plain fixed priority violates that on ${plain.violations.length} ticks`);
  assert.ok(plain.violations.every((v) => v.holder === "LOW" && v.waiter === "HIGH"), "every violation is HIGH waiting for the LOW holder");

  // (3) Brute force over all schedulers brackets the PIP completion time.
  const optimum = optimalCompletionTime(inversionScenario(), "HIGH");
  assert.ok(optimum.ticks >= 10 && Number.isFinite(optimum.ticks),
    `the optimal scheduler needs ${optimum.ticks} ticks for HIGH, at least its own 10 units`);
  assert.ok(optimum.ticks <= pip.finish.get("HIGH"), `PIP (${pip.finish.get("HIGH")}) is never better than the optimum (${optimum.ticks})`);
  assert.ok(optimum.explored > 50, `the brute-force search visited ${optimum.explored} states`);

  // (4) Transitive inheritance through a two-mutex chain.
  const chainPlain = simulate(transitiveInheritance(), { priorityInheritance: false });
  const chainPip = simulate(transitiveInheritance(), { priorityInheritance: true });
  assert.strictEqual(chainPip.maxEffective.get("LOW"), 3, "LOW inherits transitively via MID's blocked mutex");
  assert.strictEqual(chainPlain.maxEffective.get("LOW"), 1);
  assert.strictEqual(chainPip.violations.length, 0);
  assert.ok(chainPlain.violations.length > 0, `documented: the chain also inverts without PIP (${chainPlain.violations.length} tick(s))`);
  assert.ok(chainPip.finish.get("LOW") <= chainPlain.finish.get("LOW"),
    "the boosted holder finishes no later than under plain scheduling");

  // (5) PIP does not fix deadlock.
  assert.strictEqual(simulate(deadlockScenario()).deadlocked, true,
    "plain fixed priority deadlocks on A/B lock ordering");
  const deadPip = simulate(deadlockScenario(), { priorityInheritance: true });
  assert.strictEqual(deadPip.deadlocked, true, "PIP bounds inversion but cannot break a circular wait");
  assert.strictEqual(deadPip.violations.length, 0, "the invariant holds at the deadlock itself");
  assert.strictEqual(deadPip.finish.size, 0, "no task completes in the deadlocked run");
  assert.ok(deadPip.trace.length < 10, "the deadlock is detected after only a few ticks");

  // (6) No locks: the simulator reproduces the hand-computed fixed-priority schedule.
  const fm = simulate([task("A", 3, [cpu(4)]), task("B", 2, [cpu(6)]), task("C", 1, [cpu(2)])]);
  assert.deepStrictEqual([fm.finish.get("A"), fm.finish.get("B"), fm.finish.get("C")], [4, 10, 12],
    "A runs 0..3, B 4..9, C 10..11 - pure fixed-priority preemption");
  assert.strictEqual(fm.makespan, 12, "makespan equals total work when nobody blocks");
  assert.ok([...fm.blocked.values()].every((v) => v === 0), "no lock waits, so no blocked ticks");
  assert.strictEqual(fm.idleTicks, 0, "the CPU is never idle here");

  // (7) Tick accounting: one trace entry per tick, and all work executed exactly once.
  const work = inversionScenario().reduce((acc, t) => acc + t.ops
    .filter((o) => o.kind === "cpu").reduce((s, o) => s + o.time, 0), 0);
  assert.strictEqual(pip.trace.length, pip.ticks, "one trace entry per simulated tick");
  assert.strictEqual(pip.makespan, pip.ticks, "the run stops at the last completion");
  assert.strictEqual(pip.makespan, work, `all ${work} units of work run exactly once, no idle tick`);

  // (8) Alternative tie-breaks and a seeded PRNG keep the invariants; runs are reproducible.
  for (const tieBreak of ["low-id", "high-id", "seeded"]) {
    const a = simulate(inversionScenario(), { tieBreak, seed: DEFAULT_SEED });
    const b = simulate(inversionScenario(), { tieBreak, seed: DEFAULT_SEED });
    assert.strictEqual(a.violations.length, 0, `tie-break ${tieBreak} keeps the PIP invariant`);
    assert.strictEqual(a.blocked.get("HIGH"), cs, `tie-break ${tieBreak} bounds blocking by ${cs}`);
    assert.deepStrictEqual([a.finish, a.trace], [b.finish, b.trace],
      `tie-break ${tieBreak} is reproducible`);
  }
  assert.notDeepStrictEqual(simulate(transitiveInheritance()).trace.slice(0, 3), simulate(inversionScenario()).trace.slice(0, 3), "different scenarios schedule differently");
}

module.exports = { SplitMix64, DEFAULT_SEED, cpu, lock, unlock, task, enterOp, cloneRun, prepareTick,
  effectivePriorities, readyOrder, executeTick, stateKey, simulate, optimalCompletionTime,
  inversionScenario, transitiveInheritance, deadlockScenario };

if (require.main === module) {
  main();
  console.log("PASS");
}
