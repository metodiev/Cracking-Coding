/** Task 640 - Implement optimistic concurrency control simulation with validation phase. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: Kung-Robinson optimistic concurrency control over explicit transaction programs
 * (no threads, no timers). Every transaction has the three classic phases - a read phase that records
 * the read set (key -> value seen) and buffers the write set, a validation phase, and a write phase that
 * installs the buffered writes - and the simulator runs the programs round robin, so interleavings are
 * explicit and reproducible. A transaction that fails validation is restarted with a fresh read phase
 * (its attempt counter grows); after `maxAttempts` it is reported as starved and gives up, so a run
 * always terminates.
 *
 * TWO VALIDATION VARIANTS (both are real OCC variants, the difference is what they compare against):
 *  - "backward": at validation time, compare against the transactions that committed *during* our
 *    execution - fail if one of them wrote something we read (our reads were invalidated) or something we
 *    are about to write (lost update).
 *  - "forward": at validation time, compare against the transactions that are still *active* - on a clash
 *    the younger of the two backs off: the committing transaction restarts if the conflicting active
 *    transaction is older, otherwise the active one is marked invalidated and restarted before its next    step.
 * Reads observe committed data (plus our own buffered writes), which is what makes validation necessary.
 *
 * WHAT IS VERIFIED: disjoint transactions commit on the first attempt; the classic lost-update schedule
 * (two transactions bumping the same counter) ends with an exact total, the loser restarted once, and the
 * victim differs between the two validation modes (backward fails the *older* transaction that was still
 * running, forward makes the younger one back off); a read-only transaction commits without a write set;
 * the starvation limit terminates a run instead of looping; unknown operations, reading after the end of
 * a program and uncommitted transactions are refused; and over a sweep of noop-padded variants the
 * committed history always replays exactly in commit order (the OCC serializability invariant) and a
 * brute-force permutation oracle agrees, with no increment ever being lost.
 */
"use strict";

class OccEngine {
  constructor(initialValues = {}, options = {}) {
    this.mode = options.mode === "forward" ? "forward" : "backward";
    this.store = new Map(Object.entries(initialValues));
    this.clock = 0;
    this.txns = new Map();
    this.order = [];
    this.history = []; // committed and aborted records in the order they finished
    this.invalidations = []; // every restart with its reason
    this.log = [];
    this.rounds = 0;
  }
  register(id, program, options = {}) {
    if (this.txns.has(id)) throw new Error(`transaction ${id} is already registered`);
    this.txns.set(id, {
      id, program, pc: 0, state: "active", attempt: 1, restarts: 0,
      maxAttempts: options.maxAttempts === undefined ? 3 : options.maxAttempts,
      start: ++this.clock, validated: false, invalidated: false, reason: null,
      readSet: new Map(), writeSet: new Map(), reads: [], writes: [], writtenKeys: [], commitTime: null,
    });
    this.order.push(id);
    return id;
  }
  activeIds() {
    return this.order.filter((id) => this.txns.get(id).state === "active");
  }
  run(maxRounds = 400) {
    for (this.rounds = 1; this.rounds <= maxRounds; this.rounds++) {
      for (const id of this.order) {
        const txn = this.txns.get(id);
        if (txn.state === "active") this._step(txn);
      }
      if (this.activeIds().length === 0) return this;
    }
    throw new Error(`OCC simulation did not settle within ${maxRounds} rounds`);
  }
  _step(txn) {
    if (txn.invalidated) { // an invalidation takes effect before the next step, never mid-round
      txn.invalidated = false;
      this._restart(txn, `invalidated by ${txn.invalidatedBy}`);
      return;
    }
    const step = txn.program[txn.pc];
    if (!step) throw new Error(`transaction ${txn.id} ran off the end of its program without committing`);
    const [kind, key, arg] = step;
    if (kind === "noop") {
      txn.pc++;
      return;
    }
    if (kind === "read") {
      const value = txn.writeSet.has(key) ? txn.writeSet.get(key) : this.store.get(key);
      txn.readSet.set(key, value);
      txn.reads.push({ key, value });
      txn.pc++;
      this.log.push(`read ${txn.id} ${key}=${value}`);
      return;
    }
    if (kind === "write" || kind === "bump") {
      if (kind === "write") txn.writeSet.set(key, arg);
      else txn.writeSet.set(key, (txn.readSet.has(key) ? txn.readSet.get(key) : this.store.get(key)) + 1);
      txn.pc++;
      this.log.push(`write ${txn.id} ${key}=${txn.writeSet.get(key)}`);
      return;
    }
    if (kind === "validate") {
      this.validate(txn);
      if (txn.state === "active") txn.pc++;
      return;
    }
    if (kind === "commit") {
      if (!txn.validated && this.validate(txn) !== true) return;
      this._install(txn);
      return;
    }
    if (kind === "abort") {
      this._finish(txn, "aborted", "explicit abort");
      return;
    }
    throw new Error(`unknown operation ${kind} in the program of ${txn.id}`);
  }
  /** Validation phase; returns true when the transaction may enter its write phase. */
  validate(txn) {
    if (this.mode === "backward") {
      for (const other of this.txns.values()) {
        if (other.id === txn.id || other.state !== "committed" || other.commitTime === null) continue;
        if (other.commitTime <= txn.start) continue;
        for (const key of other.writtenKeys) {
          if (txn.readSet.has(key)) return this._restart(txn, `read ${key} invalidated by ${other.id}`);
          if (txn.writeSet.has(key)) return this._restart(txn, `write ${key} clashes with ${other.id}`);
        } }
      return this._pass(txn);
    }
    for (const other of this.txns.values()) {
      if (other.id === txn.id || other.state !== "active") continue;
      const reads = [...other.readSet.keys()];
      const writes = [...other.writeSet.keys()];
      const clash = reads.some((key) => txn.writeSet.has(key)) || writes.some((key) => txn.writeSet.has(key));
      if (!clash) continue;
      if (other.start < txn.start) return this._restart(txn, `older active transaction ${other.id} conflicts`);
      other.invalidated = true;
      other.invalidatedBy = txn.id;
      this.invalidations.push({ txn: other.id, kind: "invalidated", reason: `younger than ${txn.id}, which writes a clashing key` });
      this.log.push(`invalidate ${other.id} by ${txn.id}`);
    }
    return this._pass(txn);
  }
  _pass(txn) {
    if (!txn.validated) this.log.push(`validate ${txn.id} ok (attempt ${txn.attempt})`);
    txn.validated = true;
    return true;
  }
  _restart(txn, reason) {
    txn.restarts++;
    this.invalidations.push({ txn: txn.id, kind: "restart", reason });
    if (txn.restarts >= txn.maxAttempts) {
      this._finish(txn, "aborted", `gave up after ${txn.maxAttempts} attempts (${reason})`);
      return false;
    }
    txn.attempt = txn.restarts + 1;
    txn.pc = 0;
    txn.validated = false;
    txn.readSet = new Map();
    txn.writeSet = new Map();
    txn.reads = [];
    txn.start = ++this.clock;
    this.log.push(`restart ${txn.id} attempt ${txn.attempt}: ${reason}`);
    return false;
  }
  _install(txn) {
    txn.commitTime = ++this.clock;
    txn.writtenKeys = [...txn.writeSet.keys()];
    for (const [key, value] of txn.writeSet) this.store.set(key, value);
    txn.writes = [...txn.writeSet].map(([key, value]) => ({ key, value }));
    this._finish(txn, "committed", "committed");
  }
  _finish(txn, state, reason) {
    txn.state = state;
    txn.reason = reason;
    this.history.push({
      id: txn.id, state, reason, attempt: txn.attempt, commitTime: txn.commitTime,
      reads: txn.reads.slice(), writes: txn.writes.slice(), restarts: txn.restarts,
    });
    txn.writeSet = new Map();
    this.log.push(`${state} ${txn.id} (${reason})`);
  }
  record(id) {
    const txn = this.txns.get(id);
    return {
      id, state: txn.state, reason: txn.reason, attempt: txn.attempt, restarts: txn.restarts,
      reads: txn.reads, writes: txn.writes, commitTime: txn.commitTime,
    };
  } }

/** Replay the committed transactions of a history in commit order; null if a read does not replay. */
function replayInCommitOrder(history, initial) {
  const committed = history.filter((entry) => entry.state === "committed")
    .sort((a, b) => a.commitTime - b.commitTime);
  const state = new Map(Object.entries(initial));
  for (const txn of committed) {
    for (const r of txn.reads) {
      if ((state.has(r.key) ? state.get(r.key) : undefined) !== r.value) return null;
    }
    for (const w of txn.writes) state.set(w.key, w.value);
  }
  return state;
}

/** Brute-force oracle: is any permutation of the committed transactions a valid serial order? */
function findAnySerialOrder(history, initial) {
  const ids = history.filter((entry) => entry.state === "committed").map((entry) => entry.id);
  if (ids.length > 6) throw new Error("brute force is limited to six transactions");
  const byId = new Map(history.map((entry) => [entry.id, entry]));
  const replay = (order) => {
    const state = new Map(Object.entries(initial));
    for (const id of order) {
      const txn = byId.get(id);
      for (const r of txn.reads) {
        if ((state.has(r.key) ? state.get(r.key) : undefined) !== r.value) return null;
      }
      for (const w of txn.writes) state.set(w.key, w.value);
    }
    return state;
  };
  const used = new Array(ids.length).fill(false);
  const order = [];
  const search = () => {
    if (order.length === ids.length) return replay(order) ? order.slice() : null;
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

const COUNTER = [["read", "k"], ["bump", "k"], ["commit"]];
const counterWithNoops = (noops) => [["read", "k"], ...Array.from({ length: noops }, () => ["noop"]), ["bump", "k"], ["commit"]];

module.exports = { OccEngine, replayInCommitOrder, findAnySerialOrder, COUNTER, counterWithNoops };

if (require.main === module) {
  const assert = require("node:assert");
  // Disjoint transactions: no validation ever fails.
  const disjoint = new OccEngine({ a: 10, b: 20 });
  disjoint.register("t1", [["read", "a"], ["write", "a", 11], ["commit"]]);
  disjoint.register("t2", [["read", "b"], ["write", "b", 21], ["commit"]]);
  disjoint.run();
  assert.deepStrictEqual(disjoint.invalidations, []);
  assert.deepStrictEqual(disjoint.record("t1").restarts, 0);
  assert.deepStrictEqual([...disjoint.store], [["a", 11], ["b", 21]]);
  assert.deepStrictEqual(disjoint.history.map((entry) => entry.state), ["committed", "committed"]);
  assert.deepStrictEqual(findAnySerialOrder(disjoint.history, { a: 10, b: 20 }), ["t1", "t2"]);
  // The classic lost-update schedule. Backward validation fails the still-running older transaction,
  // so the total is exact and no increment is lost.
  const backward = new OccEngine({ k: 0 });
  backward.register("t1", counterWithNoops(2));
  backward.register("t2", COUNTER);
  backward.run();
  assert.strictEqual(backward.store.get("k"), 2, "both increments were applied");
  assert.strictEqual(backward.record("t1").restarts, 1);
  assert.strictEqual(backward.record("t2").restarts, 0, "the first committer keeps its attempt");
  assert.match(backward.invalidations[0].reason, /^read k invalidated by t2$/);
  assert.strictEqual(backward.record("t1").attempt, 2);
  // The same schedule under forward validation: the younger committer backs off instead.
  const forward = new OccEngine({ k: 0 }, { mode: "forward" });
  forward.register("t1", counterWithNoops(2));
  forward.register("t2", COUNTER);
  forward.run();
  assert.strictEqual(forward.store.get("k"), 2, "the total is exact in both modes");
  assert.strictEqual(forward.record("t1").restarts, 0, "the older transaction never restarts");
  assert.strictEqual(forward.record("t2").restarts, 2, "the younger one backs off, then yields again");
  assert.ok(forward.invalidations.some((entry) => /older active transaction t1 conflicts/.test(entry.reason)));
  assert.ok(forward.invalidations.some((entry) => /younger than t1/.test(entry.reason)));
  // A write-write clash is caught by backward validation and the loser retries successfully. Here the
  // second transaction reads an unrelated key, so it is the write check that fires.
  const clash = new OccEngine({ k: 0, j: 1 });
  clash.register("t1", [["read", "k"], ["write", "k", 5], ["commit"]]);
  clash.register("t2", [["read", "j"], ["write", "k", 7], ["commit"]]);
  clash.run();
  assert.strictEqual(clash.record("t2").restarts, 1);
  assert.match(clash.invalidations[0].reason, /^write k clashes with t1$/);
  assert.strictEqual(clash.store.get("k"), 7, "the retried write wins");
  assert.strictEqual(clash.record("t2").attempt, 2);
  // Read-only transactions commit without entering the write phase.
  const readOnly = new OccEngine({ a: 1 });
  readOnly.register("reader", [["read", "a"], ["noop"], ["commit"]]);
  readOnly.register("writer", [["read", "a"], ["write", "a", 2], ["commit"]]);
  readOnly.run();
  assert.strictEqual(readOnly.record("reader").state, "committed");
  assert.deepStrictEqual(readOnly.history.find((entry) => entry.id === "reader").writes, []);
  // Starvation: with a single attempt allowed the loser gives up and the run still terminates.
  const starved = new OccEngine({ k: 0 });
  starved.register("t1", counterWithNoops(2), { maxAttempts: 1 });
  starved.register("t2", COUNTER, { maxAttempts: 1 });
  starved.run();
  assert.strictEqual(starved.store.get("k"), 1, "only the winning increment survived");
  const giveUp = starved.history.find((entry) => entry.state === "aborted");
  assert.match(giveUp.reason, /^gave up after 1 attempts \(read k invalidated by t2\)$/);
  assert.deepStrictEqual(giveUp.writes, []);
  // Invalid programs are refused rather than silently tolerated.
  const bogus = new OccEngine({ a: 1 });
  bogus.register("t1", [["teleport", "a"], ["commit"]]);
  assert.throws(() => bogus.run(), /unknown operation/);
  const unfinished = new OccEngine({ a: 1 });
  unfinished.register("t1", [["read", "a"], ["noop"]]);
  assert.throws(() => unfinished.run(), /ran off the end of its program without committing/);
  const empty = new OccEngine();
  empty.register("t1", []);
  assert.throws(() => empty.run(), /ran off the end/);
  assert.throws(() => bogus.register("t1", []), /already registered/);
  // Sweep of noop-padded counter variants: whatever OCC accepts must replay exactly in commit order,
  // the permutation oracle must agree, and no increment may be lost.
  let schedules = 0;
  let restarted = 0;
  let clean = 0;
  let proactiveForward = 0;
  let proactiveBackward = 0;
  for (let a = 0; a <= 1; a++) for (let lead = 0; lead <= 3; lead++) for (const mode of ["backward", "forward"]) {
    const engine = new OccEngine({ k: 0 }, { mode });
    engine.register("t1", counterWithNoops(a));
    engine.register("t2", [
      ...Array.from({ length: lead }, () => ["noop"]), ["read", "k"], ["bump", "k"], ["commit"], ]);
    engine.run();
    schedules++;
    assert.strictEqual(engine.activeIds().length, 0, `a=${a} lead=${lead} ${mode}`);
    const committed = engine.history.filter((entry) => entry.state === "committed").length;
    assert.strictEqual(engine.store.get("k"), committed, `every increment is applied exactly once (${mode})`);
    const replay = replayInCommitOrder(engine.history, { k: 0 });
    assert.notStrictEqual(replay, null, `commit order replays (a=${a} lead=${lead} ${mode})`);
    assert.deepStrictEqual([...replay], [...engine.store]);
    assert.notStrictEqual(findAnySerialOrder(engine.history, { k: 0 }), null, `serializable (${mode})`);
    const totalRestarts = engine.record("t1").restarts + engine.record("t2").restarts;
    restarted += totalRestarts > 0 ? 1 : 0;
    clean += totalRestarts === 0 ? 1 : 0;
    assert.strictEqual(engine.invalidations.filter((entry) => entry.kind === "restart").length, totalRestarts);
    const marks = engine.invalidations.filter((entry) => entry.kind === "invalidated").length;
    if (mode === "forward") proactiveForward += marks; else proactiveBackward += marks;
    assert.strictEqual(engine.history.filter((entry) => entry.state === "aborted").length, 0);
  }
  assert.ok(clean > 0 && restarted > 0, "the sweep covers both conflict-free and conflicting runs");
  assert.ok(proactiveForward > 0, "forward validation proactively invalidates younger transactions");
  assert.strictEqual(proactiveBackward, 0, "backward validation only reacts to committed transactions");
  console.log(`OCC sweep: ${schedules} schedules, ${clean} conflict-free, ${restarted} with a restart`);
  console.log("PASS");
}
