/** Task 639 - Implement two-phase locking simulator with deadlock detection. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: a lock manager and a cooperative scheduler over explicit transaction programs (no
 * threads, no timers, no wall-clock). Programs are step lists - lock, unlock, read, write, bump, commit,
 * abort - and the scheduler runs them round robin: a transaction whose lock request cannot be granted is
 * parked with the set of lock holders it waits for and retried in the next round. When a whole round
 * makes no progress every live transaction is blocked, so the waits-for graph is built from the pending
 * requests and searched for a cycle; a cycle means deadlock, and the *youngest* transaction (latest
 * registration) is aborted and all its locks released, which lets the survivors finish.
 *
 * RULES ENFORCED: lock modes S (shared) and X (exclusive) with the usual conflict matrix; upgrades are
 * allowed only in place; requests are served strictly in FIFO order, except that a transaction already
 * holding a lock on the key may upgrade without queueing behind newcomers. Reads require a shared or
 * exclusive lock and writes an exclusive one, so the simulator refuses to run a program that touches an
 * unlocked item. Two-phase locking itself is checked: a transaction that unlocks and later locks again
 * is a violation (its shrinking phase already began), so it is reported in `violations` and aborted
 * instead of being allowed to run; strict 2PL runs keep every lock until commit or abort.
 *
 * WHAT IS VERIFIED: the lock matrix and FIFO behaviour; deadlocks are detected in 2- and 3-cycle shapes
 * with the right victim, the survivors commit and the victim's buffered writes are rolled back; a mere
 * wait is not reported as a deadlock (no false positives) and the waiter reads the committed value (no
 * dirty read); two-phase violations are rejected; and for every strict-2PL run the committed
 * transactions replayed in *commit order* reproduce every recorded read and the final store state, which
 * a brute-force permutation oracle confirms is a genuine serial order.
 */
"use strict";

const CONFLICT = { S: { S: false, X: true }, X: { S: true, X: true } };

class LockManager {
  constructor() {
    this.locks = new Map(); // key -> {owners: Map<txn, mode>, queue: [{txn, mode}]}
  }
  _entry(key) {
    let entry = this.locks.get(key);
    if (!entry) {
      entry = { owners: new Map(), queue: [] };
      this.locks.set(key, entry);
    }
    return entry;
  }
  acquire(txn, key, mode) {
    const entry = this._entry(key);
    const held = entry.owners.get(txn);
    if (held === "X" || (held === "S" && mode === "S")) return { granted: true, already: true };
    const blockedBy = [...entry.owners].filter(([other, otherMode]) => other !== txn && CONFLICT[mode][otherMode]);
    // A fresh request must respect FIFO; an upgrade only has to be compatible with the other holders.
    const headBlocks = held === undefined && entry.queue.length > 0 && entry.queue[0].txn !== txn;
    if (blockedBy.length === 0 && !headBlocks) {
      entry.owners.set(txn, mode);
      entry.queue = entry.queue.filter((request) => request.txn !== txn);
      return { granted: true, already: false };
    }
    const queued = entry.queue.find((request) => request.txn === txn);
    if (queued) queued.mode = mode;
    else entry.queue.push({ txn, mode });
    return { granted: false, waitingFor: blockedBy.map(([other]) => other).sort() };
  }
  release(txn, key) {
    const entry = this.locks.get(key);
    if (!entry || !entry.owners.has(txn)) throw new Error(`transaction ${txn} does not hold a lock on ${key}`);
    entry.owners.delete(txn);
    this._grantQueued(key);
  }
  releaseAll(txn) {
    const released = [];
    for (const [key, entry] of this.locks) {
      if (!entry.owners.has(txn)) continue;
      entry.owners.delete(txn);
      this._grantQueued(key);
      released.push(key);
    }
    for (const entry of this.locks.values()) entry.queue = entry.queue.filter((request) => request.txn !== txn);
    return released.sort();
  }
  _grantQueued(key) {
    const entry = this.locks.get(key);
    while (entry.queue.length > 0) {
      const request = entry.queue[0];
      const blocked = [...entry.owners].some(([other, mode]) => other !== request.txn && CONFLICT[request.mode][mode]);
      if (blocked) return;
      entry.queue.shift();
      entry.owners.set(request.txn, request.mode);
    } }
  holders(key) {
    return [...this._entry(key).owners.keys()].sort();
  }
  waitingQueue(key) {
    return this._entry(key).queue.map((request) => `${request.txn}:${request.mode}`);
  } }

class LockSimulator {
  constructor(initialValues = {}) {
    this.store = new Map(Object.entries(initialValues));
    this.locks = new LockManager();
    this.txns = new Map();
    this.order = [];
    this.log = [];
    this.violations = [];
    this.deadlocks = [];
    this.rounds = 0;
  }
  register(id, program) {
    if (this.txns.has(id)) throw new Error(`transaction ${id} is already registered`);
    this.txns.set(id, {
      id, program, pc: 0, state: "active", held: new Map(), pending: new Map(), env: new Map(),
      reads: [], writes: [], waiting: null, released: false, reason: null, commitOrder: null,
    });
    this.order.push(id);
    return id;
  }
  run(maxRounds = 400) {
    for (this.rounds = 0; this.rounds < maxRounds; this.rounds++) {
      let progress = false;
      for (const id of this.order) {
        const txn = this.txns.get(id);
        if (txn.state !== "active") continue;
        txn.waiting = null;
        if (this._execute(txn)) progress = true;
      }
      if (this.order.every((id) => this.txns.get(id).state !== "active")) return this;
      if (!progress) {
        const cycle = this.findDeadlock();
        if (!cycle) throw new Error("no progress and no cycle: the simulator should not be able to stall");
        const victim = cycle.slice().sort((a, b) => this.order.indexOf(b) - this.order.indexOf(a))[0];
        this.deadlocks.push({ cycle, victim });
        this._finish(victim, "aborted", `deadlock victim in cycle ${cycle.join(" -> ")}`);
      } }
    throw new Error(`the simulation did not settle within ${maxRounds} rounds`);
  }
  _execute(txn) {
    const step = txn.program[txn.pc];
    if (!step) throw new Error(`transaction ${txn.id} ran off the end of its program`);
    const [kind, key, arg] = step;
    if (kind === "lock") {
      if (txn.released) { // two-phase locking is broken: the shrinking phase already began
        this.violations.push(`transaction ${txn.id} requested ${arg}(${key}) after releasing a lock`);
        this._finish(txn.id, "aborted", "two-phase locking violation");
        return true;
      }
      const result = this.locks.acquire(txn.id, key, arg);
      if (!result.granted) {
        txn.waiting = { key, mode: arg, blockers: result.waitingFor };
        this.log.push(`wait ${txn.id} ${arg}(${key}) for ${result.waitingFor.join(",")}`);
        return false;
      }
      txn.held.set(key, arg);
      txn.pc++;
      this.log.push(`lock ${txn.id} ${arg}(${key})`);
      return true;
    }
    if (kind === "unlock") {
      this.locks.release(txn.id, key);
      txn.held.delete(key);
      txn.released = true;
      txn.pc++;
      this.log.push(`unlock ${txn.id} ${key}`);
      return true;
    }
    if (kind === "read") {
      if (!txn.held.has(key)) throw new Error(`transaction ${txn.id} read ${key} without a lock`);
      const value = txn.pending.has(key) ? txn.pending.get(key) : this.store.get(key);
      txn.env.set(key, value);
      txn.reads.push({ key, value });
      txn.pc++;
      this.log.push(`read ${txn.id} ${key}=${value}`);
      return true;
    }
    if (kind === "write" || kind === "bump") {
      if (txn.held.get(key) !== "X") throw new Error(`transaction ${txn.id} wrote ${key} without an exclusive lock`);
      const value = kind === "bump" ? txn.env.get(key) + 1 : arg;
      txn.pending.set(key, value);
      txn.pc++;
      this.log.push(`write ${txn.id} ${key}=${value}`);
      return true;
    }
    if (kind === "commit") {
      this.locks.releaseAll(txn.id);
      for (const [item, value] of txn.pending) this.store.set(item, value);
      txn.writes = [...txn.pending].map(([item, value]) => ({ key: item, value }));
      this._finish(txn.id, "committed", "committed");
      return true;
    }
    if (kind === "abort") {
      this._finish(txn.id, "aborted", "explicit abort");
      return true;
    }
    throw new Error(`unknown operation ${kind} in the program of ${txn.id}`);
  }
  _finish(id, state, reason) {
    const txn = this.txns.get(id);
    this.locks.releaseAll(id);
    txn.state = state;
    txn.reason = reason;
    txn.pending = new Map();
    txn.held = new Map();
    txn.waiting = null;
    if (state === "committed") txn.commitOrder = this.order.filter((other) => {
      const t = this.txns.get(other);
      return t.state === "committed" && t.commitOrder !== null;
    }).length + 1;
    this.log.push(`${state} ${id} (${reason})`);
  }
  findDeadlock() {
    const edges = new Map();
    for (const id of this.order) {
      const txn = this.txns.get(id);
      const blockers = txn.state === "active" && txn.waiting ? txn.waiting.blockers : [];
      edges.set(id, blockers.filter((other) => this.txns.get(other).state === "active"));
    }
    const state = new Map(this.order.map((id) => [id, 0]));
    const stack = [];
    const visit = (id) => {
      state.set(id, 1);
      stack.push(id);
      for (const next of edges.get(id)) {
        if (state.get(next) === 1) return stack.slice(stack.indexOf(next));
        if (state.get(next) === 0) {
          const found = visit(next);
          if (found) return found;
        } }
      stack.pop();
      state.set(id, 2);
      return null;
    };
    for (const id of this.order) {
      if (state.get(id) !== 0) continue;
      const found = visit(id);
      if (found) return found;
    }
    return null;
  }
  entries() {
    return this.order.map((id) => this.txns.get(id)).map((txn) => ({
      id: txn.id, state: txn.state, reason: txn.reason, reads: txn.reads, writes: txn.writes,
      commitOrder: txn.commitOrder, held: [...txn.held.keys()].sort(),
    }));
  }
  liveCount() {
    return this.entries().filter((entry) => entry.state === "active").length;
  } }

/** Replay a candidate serial order over the recorded reads; null when a read does not match. */
function replayInOrder(order, entries, initial) {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const state = new Map(Object.entries(initial));
  for (const id of order) {
    const txn = byId.get(id);
    for (const r of txn.reads) {
      if ((state.has(r.key) ? state.get(r.key) : undefined) !== r.value) return null;
    }
    for (const w of txn.writes) state.set(w.key, w.value);
  }
  return state;
}

function commitOrderOf(entries) {
  return entries.filter((entry) => entry.state === "committed")
    .sort((a, b) => a.commitOrder - b.commitOrder).map((entry) => entry.id);
}

/** Brute-force oracle: is any permutation of the committed transactions a valid serial order? */
function findAnySerialOrder(entries, initial) {
  const ids = commitOrderOf(entries);
  if (ids.length > 6) throw new Error("brute force is limited to six transactions");
  const done = commitOrderOf(entries).concat();
  void done;
  const used = new Array(ids.length).fill(false);
  const order = [];
  const search = () => {
    if (order.length === ids.length) return replayInOrder(order, entries, initial) ? order.slice() : null;
    for (let i = 0; i < ids.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      order.push(ids[i]);
      const found = search();
      order.pop();
      used[i] = false;
      if (found) return found;
    }
    return null;
  };
  return search();
}

module.exports = { LockManager, LockSimulator, replayInOrder, commitOrderOf, findAnySerialOrder };

if (require.main === module) {
  const assert = require("node:assert");
  // Lock matrix and FIFO queueing.
  const locks = new LockManager();
  assert.strictEqual(locks.acquire("t1", "x", "S").granted, true);
  assert.strictEqual(locks.acquire("t2", "x", "S").granted, true, "shared locks are compatible");
  const queued = locks.acquire("t3", "x", "X");
  assert.strictEqual(queued.granted, false);
  assert.deepStrictEqual(queued.waitingFor, ["t1", "t2"]);
  assert.deepStrictEqual(locks.waitingQueue("x"), ["t3:X"]);
  assert.strictEqual(locks.acquire("t1", "x", "X").granted, false, "upgrade blocked by t2");
  locks.release("t2", "x");
  assert.strictEqual(locks.acquire("t1", "x", "X").granted, true, "the upgrade wins over the queue");
  assert.strictEqual(locks.acquire("t3", "x", "X").granted, false);
  assert.strictEqual(locks.acquire("t4", "x", "S").granted, false, "FIFO: t3 is ahead of t4");
  locks.release("t1", "x");
  assert.strictEqual(locks.acquire("t3", "x", "X").granted, true, "granted from the queue");
  assert.deepStrictEqual(locks.holders("x"), ["t3"]);
  assert.throws(() => locks.release("t4", "x"), /does not hold a lock/);
  assert.deepStrictEqual(locks.releaseAll("t3"), ["x"]);
  // Strict 2PL run: t2 waits for t1 and then reads the committed value - no dirty read, no deadlock.
  const strict = new LockSimulator({ a: 10, b: 20 });
  strict.register("t1", [
    ["lock", "a", "X"], ["read", "a"], ["bump", "a"], ["lock", "b", "X"], ["read", "b"], ["bump", "b"], ["commit"], ]);
  strict.register("t2", [["lock", "b", "X"], ["read", "b"], ["bump", "b"], ["commit"]]);
  strict.run();
  assert.deepStrictEqual(strict.deadlocks, []);
  assert.deepStrictEqual(strict.violations, []);
  assert.ok(strict.log.some((line) => /^wait t1 X\(b\) for t2/.test(line)), "t1 waits for the holder of b");
  assert.strictEqual(strict.store.get("a"), 11);
  assert.strictEqual(strict.store.get("b"), 22, "t1 bumped the value t2 had committed");
  const entries = strict.entries();
  assert.deepStrictEqual(commitOrderOf(entries), ["t2", "t1"], "t1 blocked while t2 finished first");
  assert.deepStrictEqual(entries.map((entry) => entry.state), ["committed", "committed"]);
  assert.deepStrictEqual(entries.map((entry) => entry.held), [[], []], "strict 2PL holds locks until commit");
  const replay = replayInOrder(commitOrderOf(entries), entries, { a: 10, b: 20 });
  assert.notStrictEqual(replay, null, "commit order is a valid serial order");
  assert.deepStrictEqual([...replay], [...strict.store]);
  assert.deepStrictEqual(findAnySerialOrder(entries, { a: 10, b: 20 }), ["t2", "t1"]);
  // A shared-lock reader blocks on the writer and therefore never sees a dirty value.
  const reader = new LockSimulator({ a: 10 });
  reader.register("w", [["lock", "a", "X"], ["read", "a"], ["bump", "a"], ["commit"]]);
  reader.register("r", [["lock", "a", "S"], ["read", "a"], ["commit"]]);
  reader.run();
  assert.ok(reader.log.some((line) => /^wait r S\(a\) for w/.test(line)));
  assert.deepStrictEqual(reader.deadlocks, [], "a plain wait is not a deadlock");
  assert.deepStrictEqual(reader.entries().find((entry) => entry.id === "r").reads,
    [{ key: "a", value: 11 }], "the reader saw the committed value, not the uncommitted one");
  // Deadlock: t1 holds a and wants b, t2 holds b and wants a. The youngest (t2) is the victim, its
  // buffered write is rolled back, and the survivor completes.
  const dead = new LockSimulator({ a: 1, b: 2 });
  dead.register("t1", [["lock", "a", "X"], ["read", "a"], ["lock", "b", "X"], ["read", "b"], ["write", "a", 111], ["commit"]]);
  dead.register("t2", [["lock", "b", "X"], ["read", "b"], ["write", "b", 222], ["lock", "a", "X"], ["read", "a"], ["commit"]]);
  dead.run();
  assert.strictEqual(dead.deadlocks.length, 1);
  assert.deepStrictEqual(dead.deadlocks[0].cycle, ["t1", "t2"]);
  assert.strictEqual(dead.deadlocks[0].victim, "t2", "the youngest transaction is chosen");
  const deadEntries = dead.entries();
  assert.deepStrictEqual(deadEntries.map((entry) => entry.state), ["committed", "aborted"]);
  assert.match(deadEntries[1].reason, /deadlock victim in cycle t1 -> t2/);
  assert.strictEqual(dead.store.get("a"), 111);
  assert.strictEqual(dead.store.get("b"), 2, "the victim's write was rolled back");
  assert.deepStrictEqual(commitOrderOf(deadEntries), ["t1"]);
  assert.deepStrictEqual(replayInOrder(["t1"], deadEntries, { a: 1, b: 2 }).get("a"), 111);
  assert.strictEqual(dead.liveCount(), 0);
  assert.deepStrictEqual(findAnySerialOrder(deadEntries, { a: 1, b: 2 }), ["t1"]);
  // A three-transaction cycle: each holds one key and wants the next one.
  const ring = new LockSimulator({ a: 1, b: 2, c: 3 });
  ring.register("t1", [["lock", "a", "X"], ["read", "a"], ["lock", "b", "X"], ["commit"]]);
  ring.register("t2", [["lock", "b", "X"], ["read", "b"], ["lock", "c", "X"], ["commit"]]);
  ring.register("t3", [["lock", "c", "X"], ["read", "c"], ["lock", "a", "X"], ["commit"]]);
  ring.run();
  assert.strictEqual(ring.deadlocks.length, 1);
  assert.deepStrictEqual(ring.deadlocks[0].cycle, ["t1", "t2", "t3"]);
  assert.strictEqual(ring.deadlocks[0].victim, "t3");
  assert.deepStrictEqual(ring.entries().map((entry) => entry.state), ["committed", "committed", "aborted"]);
  assert.strictEqual(ring.liveCount(), 0);
  // Two-phase locking is enforced: unlocking before the end of the transaction and locking again is
  // reported as a violation, and the offending transaction cannot continue.
  const bad = new LockSimulator({ a: 1 });
  bad.register("t1", [["lock", "a", "X"], ["read", "a"], ["unlock", "a"], ["lock", "a", "X"], ["commit"]]);
  bad.run();
  assert.deepStrictEqual(bad.violations, ["transaction t1 requested X(a) after releasing a lock"]);
  assert.deepStrictEqual(bad.entries().map((entry) => entry.state), ["aborted"]);
  assert.strictEqual(bad.entries()[0].reason, "two-phase locking violation");
  assert.strictEqual(bad.store.get("a"), 1, "nothing was written");
  // Programming errors in a program are refused rather than silently tolerated.
  const sloppy = new LockSimulator({ a: 1 });
  sloppy.register("t1", [["read", "a"], ["commit"]]);
  assert.throws(() => sloppy.run(), /without a lock/);
  const shared = new LockSimulator({ a: 1 });
  shared.register("t1", [["lock", "a", "S"], ["write", "a", 2], ["commit"]]);
  assert.throws(() => shared.run(), /without an exclusive lock/);
  const truncated = new LockSimulator({ a: 1 });
  truncated.register("t1", [["lock", "a", "X"], ["read", "a"]]);
  assert.throws(() => truncated.run(), /ran off|not settle/);
  console.log(`lock manager: ${locks.waitingQueue("x").length} queued, simulator rounds ${strict.rounds}`);
  console.log("PASS");
}
