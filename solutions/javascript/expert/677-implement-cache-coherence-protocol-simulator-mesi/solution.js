/**
 * Task 677 - Implement cache coherence protocol simulator (MESI).
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * A bus-based MESI simulator: N private caches, one shared memory, and a snooping bus whose
 * transactions are counted (BusRd on a read miss, BusRdX on a write miss or upgrade). Each cache line
 * carries one of the four MESI states, and the transitions follow the usual rules:
 *
 *   read miss   - BusRd; another cache in M flushes its data and demotes to S; caches in E demote to S;
 *                 if nobody held a valid copy the requester installs the line in E (exclusive), which is
 *                 why a lone cache can write afterwards without any bus traffic at all;
 *   write hit   - M stays silent; E promotes to M silently; S needs a BusRdX upgrade that invalidates
 *                 every other copy and flushes a modified one back to memory;
 *   write miss  - BusRdX, invalidating (and flushing) every other copy.
 *
 * `coherenceProblems` re-checks the protocol invariants after every step - at most one M copy, no
 * M/E copy alongside an S copy, all valid copies of a line agree on the value - and `checkShadow`
 * compares what every access returned with an independent shadow map of the last value written, while
 * `flushAll` (write-back and invalidate everything) is the state in which memory must equal the shadow,
 * because no modified copy may still be hiding a newer value. The counted transactions and flushes of
 * the hand-written trace (7 BusRd/BusRdX requests, 2 flushes) are asserted exactly, as are the silent
 * write paths (M hit, and E->M promotion in a single-cache machine).
 */
"use strict";

const assert = require("node:assert");

const INVALID = "I";
const SHARED = "S";
const EXCLUSIVE = "E";
const MODIFIED = "M";

function makeMachine(lines, cacheCount, busLatency = 1) {
  return {
    lines: [...lines],
    caches: Array.from({ length: cacheCount }, () => new Map()),
    memory: new Map(),
    shadow: new Map(lines.map((line) => [line, 0])),
    transactions: [],
    flushes: 0,
    busCycles: 0,
    busLatency,
  };
}

const stateOf = (machine, p, line) => machine.caches[p].get(line)?.state ?? INVALID;
const valueOf = (machine, p, line) => machine.caches[p].get(line)?.value ?? null;
const request = (machine, type, line, requester) => {
  machine.transactions.push({ type, line, requester });
  machine.busCycles += machine.busLatency;
};

function read(machine, p, line) {
  const self = machine.caches[p];
  const entry = self.get(line);
  if (entry !== undefined && entry.state !== INVALID) return { value: entry.value, hit: true, traffic: false };
  request(machine, "BusRd", line, p);
  let value = machine.memory.get(line) ?? machine.shadow.get(line) ?? 0;
  let supplier = null;
  let sharers = 0;
  machine.caches.forEach((cache, q) => {
    if (q === p) return;
    const other = cache.get(line);
    if (other === undefined || other.state === INVALID) return;
    sharers += 1;
    if (other.state === MODIFIED) {
      value = other.value;
      machine.memory.set(line, other.value);
      machine.flushes += 1;
      supplier = q;
      other.state = SHARED;
    } else if (other.state === EXCLUSIVE) other.state = SHARED;
  });
  // No other copy exists: the requester may keep the line exclusive and write it later for free.
  self.set(line, { state: sharers === 0 ? EXCLUSIVE : SHARED, value });
  machine.memory.set(line, value);
  return { value, hit: false, traffic: true, supplier };
}

function write(machine, p, line, value) {
  const self = machine.caches[p];
  const entry = self.get(line);
  const previous = entry?.state ?? INVALID;
  if (previous === MODIFIED || previous === EXCLUSIVE) {
    // Silent: no bus traffic when the cache already owns the line exclusively.
    self.set(line, { state: MODIFIED, value });
    machine.shadow.set(line, value);
    return { hit: previous === MODIFIED, silent: true, traffic: false };
  }
  request(machine, previous === SHARED ? "BusRdX-upgrade" : "BusRdX", line, p);
  machine.caches.forEach((cache, q) => {
    if (q === p) return;
    const other = cache.get(line);
    if (other === undefined || other.state === INVALID) return;
    if (other.state === MODIFIED) {
      machine.memory.set(line, other.value);
      machine.flushes += 1;
    }
    other.state = INVALID;
  });
  self.set(line, { state: MODIFIED, value });
  machine.shadow.set(line, value);
  return { hit: false, silent: false, traffic: true };
}

/** Protocol invariants, re-checked after every step. */
function coherenceProblems(machine) {
  const bad = [];
  for (const line of machine.lines) {
    const states = machine.caches.map((cache) => stateOf(machine, machine.caches.indexOf(cache), line));
    const modified = states.filter((s) => s === MODIFIED).length;
    if (modified > 1) bad.push(`${modified} modified copies of ${line}`);
    if (modified > 0 && states.some((s) => s === SHARED || s === EXCLUSIVE)) {
      bad.push(`a modified copy of ${line} coexists with shared/exclusive copies`);
    }
    if (states.includes(SHARED) && states.some((s) => s === MODIFIED || s === EXCLUSIVE)) {
      bad.push(`a shared copy of ${line} coexists with an exclusive copy`);
    }
    const values = machine.caches.map((cache) => cache.get(line))
      .filter((entry) => entry !== undefined && entry.state !== INVALID).map((entry) => entry.value);
    if (new Set(values).size > 1) bad.push(`valid copies of ${line} disagree: ${values}`);
    if (states.every((s) => s !== MODIFIED) && values.length > 0
        && (machine.memory.get(line) ?? machine.shadow.get(line)) !== values[0]) {
      bad.push(`memory is stale for ${line} although nobody holds it modified`);
    }
  }
  return bad;
}

/** Independent reference: every read must have returned the shadow's value at that moment. */
function checkShadow(machine, log) {
  const bad = [];
  for (const entry of log) {
    if (entry.value !== entry.expected) bad.push(`${entry.line} read ${entry.value}, expected ${entry.expected}`);
  }
  return bad;
}

/** Write back and invalidate everything; then memory must equal the shadow for every line. */
function flushAll(machine) {
  machine.caches.forEach((cache) => {
    for (const [line, entry] of cache) {
      if (entry.state === MODIFIED) {
        machine.memory.set(line, entry.value);
        machine.flushes += 1;
      }
    }
    cache.clear();
  });
  return machine;
}

function memoryProblems(machine) {
  const bad = [];
  for (const line of machine.lines) {
    if ((machine.memory.get(line) ?? machine.shadow.get(line)) !== machine.shadow.get(line)) {
      bad.push(`memory holds ${machine.memory.get(line)} for ${line}, expected ${machine.shadow.get(line)}`);
    }
  }
  return bad;
}

/** The hand-checked trace: three caches, one line, alternating readers and writers. */
function runTrace(machine) {
  const log = [];
  const record = (result) => log.push({ value: result.value, line: "L", expected: machine.shadow.get("L") });
  record(read(machine, 0, "L"));
  record(read(machine, 1, "L"));
  record(read(machine, 0, "L"));
  write(machine, 0, "L", 7);
  write(machine, 0, "L", 9);
  record(read(machine, 1, "L"));
  write(machine, 1, "L", 12);
  record(read(machine, 0, "L"));
  record(read(machine, 2, "L"));
  return log;
}

function main() {
  // (1) The hand-written two-cache trace: transactions, flushes and per-step states.
  const machine = makeMachine(["L"], 3);
  const before = machine.transactions.length;
  const log = runTrace(machine);
  assert.strictEqual(before, 0, "the machine starts quiet");
  assert.deepStrictEqual(machine.transactions.map((t) => `${t.type}@${t.requester}`),
    ["BusRd@0", "BusRd@1", "BusRdX-upgrade@0", "BusRd@1", "BusRdX-upgrade@1", "BusRd@0", "BusRd@2"],
    "read misses and write upgrades are the only bus traffic");
  assert.strictEqual(machine.flushes, 2, "two modified copies had to be flushed on the bus");
  assert.deepStrictEqual(log.map((entry) => entry.value), [0, 0, 0, 9, 12, 12],
    "every read returns the last value written to the line");
  assert.strictEqual(machine.busCycles, 7 * machine.busLatency, "each bus request costs one bus cycle");
  assert.deepStrictEqual(checkShadow(machine, log), [], "every read matched the shadow value of its moment");
  assert.deepStrictEqual(log.map((entry) => entry.expected), [0, 0, 0, 9, 12, 12], "and those shadows are hand-checkable");
  assert.strictEqual(valueOf(machine, 2, "L"), 12, "the third cache also sees the newest value");

  // (2) The silent paths: an M hit and an E -> M promotion produce no traffic at all.
  const quiet = makeMachine(["L"], 1);
  read(quiet, 0, "L");
  assert.strictEqual(stateOf(quiet, 0, "L"), EXCLUSIVE, "a lone reader installs the line exclusively");
  const promotions = quiet.transactions.length;
  write(quiet, 0, "L", 5);
  write(quiet, 0, "L", 6);
  assert.deepStrictEqual([quiet.transactions.length, promotions, stateOf(quiet, 0, "L")], [1, 1, MODIFIED],
    "E -> M and M hit both stay silent, so the transaction count does not move");

  // (3) State transitions across the trace, step by step.
  const states = makeMachine(["L"], 2);
  read(states, 0, "L");
  assert.deepStrictEqual([stateOf(states, 0, "L"), stateOf(states, 1, "L")], [EXCLUSIVE, INVALID],
    "the first read makes the line exclusive in the only cache");
  read(states, 1, "L");
  assert.deepStrictEqual([stateOf(states, 0, "L"), stateOf(states, 1, "L")], [SHARED, SHARED],
    "the second reader downgrades the owner and shares the line");
  write(states, 1, "L", 3);
  assert.deepStrictEqual([stateOf(states, 0, "L"), stateOf(states, 1, "L")], [INVALID, MODIFIED],
    "the upgrade invalidates the other copy");
  read(states, 0, "L");
  assert.deepStrictEqual([stateOf(states, 0, "L"), stateOf(states, 1, "L")], [SHARED, SHARED],
    "and a later read turns the modified copy back into a shared one");

  // (4) Invariants after every single step of a longer interleaved run.
  const interleaved = makeMachine(["A", "B"], 3);
  const operations = [[0, "read", "A"], [1, "write", "A"], [2, "read", "A"], [0, "write", "B"],
    [1, "read", "B"], [2, "write", "B"], [0, "read", "B"], [2, "read", "A"], [1, "write", "A"]];
  let steps = 0;
  for (const [p, kind, line] of operations) {
    const value = 10 * p + steps;
    if (kind === "read") read(interleaved, p, line);
    else write(interleaved, p, line, value);
    steps += 1;
    assert.deepStrictEqual(coherenceProblems(interleaved), [], `invariants broke after step ${steps}`);
  }
  assert.strictEqual(steps, operations.length, "every operation ran");
  flushAll(interleaved);
  assert.deepStrictEqual(memoryProblems(interleaved), [], "after flushing, memory holds every latest value");
  assert.ok([...interleaved.caches].every((cache) => cache.size === 0), "and all caches are empty");

  // (5) Single-line ownership: writing where the line is shared costs exactly one upgrade.
  const upgrade = makeMachine(["L"], 2);
  read(upgrade, 0, "L");
  read(upgrade, 1, "L");
  const baseline = upgrade.transactions.length;
  write(upgrade, 0, "L", 1);
  assert.deepStrictEqual([upgrade.transactions.length - baseline, upgrade.transactions[2].type], [1, "BusRdX-upgrade"],
    "an S -> M transition is one upgrade request");
  assert.strictEqual(stateOf(upgrade, 1, "L"), INVALID, "which invalidates the other copy");

  // (6) Determinism: the same trace gives byte-identical buses, states and values.
  const replayA = makeMachine(["L"], 3);
  const replayB = makeMachine(["L"], 3);
  runTrace(replayA);
  runTrace(replayB);
  assert.deepStrictEqual([replayA.transactions, replayA.flushes, replayA.busCycles, [...replayA.caches[0]]],
    [replayB.transactions, replayB.flushes, replayB.busCycles, [...replayB.caches[0]]], "the simulator is deterministic");

  // (7) Boundary: a machine with a single line and no accesses stays cold, and reads of untouched
  // lines are misses that install the line without any supplier.
  const cold = makeMachine(["X"], 2);
  assert.deepStrictEqual([cold.transactions.length, coherenceProblems(cold)], [0, []], "no accesses, no traffic");
  const fresh = read(cold, 0, "X");
  assert.deepStrictEqual([fresh.hit, fresh.supplier, stateOf(cold, 0, "X")], [false, null, EXCLUSIVE],
    "a cold read has no supplier and can take the line exclusively");
  assert.strictEqual(read(cold, 0, "X").hit, true, "and the second read of the same line hits");
}

module.exports = { INVALID, SHARED, EXCLUSIVE, MODIFIED, makeMachine, stateOf, valueOf, read, write,
  coherenceProblems, checkShadow, flushAll, memoryProblems, runTrace };

if (require.main === module) {
  main();
  console.log("PASS");
}
