/** Task 637 - Implement SNAPSHOT isolation anomalies demonstrator (write skew). (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMULATED: an in-memory multi-version store driven by an explicit schedule (no threads, no
 * timers, no wall-clock), so every run is deterministic. Snapshot isolation is modelled the way a real
 * engine does it: `begin` takes a snapshot (the commit clock at that instant), reads are served by the
 * newest version whose commit number is <= the snapshot (repeatable reads for free), writes stay
 * buffered in the transaction, and `commit` applies first-committer-wins - if a written key was updated
 * after our snapshot the transaction aborts, otherwise it installs one new version per key.
 *
 * THE ANOMALY (write skew): SI only forbids write-write conflicts, so two transactions reading the same
 * snapshot may write *different* keys and both commit. The demo is the doctor-on-call constraint ("at
 * least one doctor must stay on call"): both transactions read two doctors on call, each decides it may
 * go off, both commit, and the constraint breaks although no key was written twice. Interleaved and
 * strictly serial executions of the *same* two bodies are contrasted (see DOCTOR_* below).
 *
 * WHAT IS VERIFIED: (1) the anomaly happens under SI and disappears under the serial order; (2) the SI
 * history has a cycle in the multiversion serialization graph (wr, rw anti-dependency incl. phantoms,
 * ww) while the serial history stays acyclic - cross-checked against a brute-force oracle that searches
 * every permutation of the committed transactions for a serial order matching all observed reads;
 * (3) whenever a history is acyclic, its topological order is itself a valid serial order (verified by
 * replaying the recorded observations, not by re-asking the graph); (4) repeatable read,
 * first-committer-wins (no lost update), invisible aborted writes and read-only commits hold, committed
 * write sets are pairwise disjoint, and every write-write victim names its conflicting keys. Claims (2)
 * and the disjointness invariant are exercised on every interleaving of four pairs of three-step
 * programs, enumerated exhaustively instead of sampled.
 */
"use strict";

const ON_CALL = "on-call";
const OFF = "off";
const DOCTORS = ["alice", "bob"];
const MISSING = undefined; // a read of a row that does not exist yet

class SnapshotDb {
  constructor(initialValues = {}) {
    this.versions = new Map(); // key -> [{key, version, value, writer}] ascending by version
    this.clock = 0; // commit clock; initial values are version 0
    this.transactions = new Map();
    this.history = []; // committed/aborted transaction records, in the order they finished
    for (const [key, value] of Object.entries(initialValues)) {
      this.versions.set(key, [{ key, version: 0, value, writer: null }]);
    } }
  begin(id) {
    if (this.isActive(id)) throw new Error(`transaction ${id} is already active`);
    this.transactions.set(id, { id, snapshot: this.clock, state: "active", reads: [], writes: new Map() });
    return id;
  }
  isActive(id) {
    const txn = this.transactions.get(id);
    return Boolean(txn) && txn.state === "active";
  }
  activeCount() {
    return [...this.transactions.values()].filter((t) => t.state === "active").length;
  }
  _active(id) {
    const txn = this.transactions.get(id);
    if (!txn || txn.state !== "active") throw new Error(`transaction ${id} is not active`);
    return txn;
  }
  /** Snapshot read: newest version committed at or before this transaction's snapshot. */
  read(id, key) {
    const txn = this._active(id);
    const chain = this.versions.get(key) || [];
    let found = null;
    for (let i = chain.length - 1; i >= 0; i--) if (chain[i].version <= txn.snapshot) { found = chain[i]; break; }
    txn.reads.push({ key, version: found ? found.version : -1, value: found ? found.value : MISSING });
    return found ? found.value : MISSING;
  }
  write(id, key, value) {
    const txn = this._active(id);
    if (value === MISSING) throw new Error("a write may not store an absent value");
    txn.writes.set(key, value);
  }
  abort(id) {
    this._finish(this._active(id), "aborted", [], [], "explicit abort");
    return { committed: false, reason: "explicit abort", conflicts: [] };
  }
  commit(id) {
    const txn = this._active(id);
    const conflicts = [];
    for (const key of txn.writes.keys()) {
      const chain = this.versions.get(key) || [];
      const latest = chain[chain.length - 1];
      if (latest && latest.version > txn.snapshot) conflicts.push(key);
    }
    if (conflicts.length > 0) {
      this._finish(txn, "aborted", [], conflicts, "write-write conflict");
      return { committed: false, reason: "write-write conflict", conflicts };
    }
    this.clock += 1;
    const written = [];
    for (const [key, value] of txn.writes) {
      const entry = { key, version: this.clock, value, writer: id };
      const chain = this.versions.get(key);
      if (chain) chain.push(entry); else this.versions.set(key, [entry]);
      written.push(entry);
    }
    this._finish(txn, "committed", written, [], "committed");
    return { committed: true, reason: "committed", version: this.clock, written };
  }
  _finish(txn, state, written, conflicts, reason) {
    txn.state = state;
    this.history.push({
      id: txn.id, state, reason, snapshot: txn.snapshot, reads: txn.reads.slice(), writes: written,
      attemptedKeys: [...txn.writes.keys()], conflicts,
    });
    txn.writes = new Map();
  }
  /** Committed state, i.e. what a brand-new transaction would see right now. */
  committedState() {
    const state = new Map();
    for (const [key, chain] of this.versions) state.set(key, chain[chain.length - 1].value);
    return state;
  }
  versionCount(key) {
    return (this.versions.get(key) || []).length;
  } }

/** Multiversion serialization graph: wr, rw (anti-dependency, covers phantoms) and ww edges. */
function buildMvGraph(history) {
  const committed = history.filter((h) => h.state === "committed");
  const edges = new Map(committed.map((h) => [h.id, new Set()]));
  const addEdge = (from, to) => {
    if (from !== to && edges.has(from) && edges.has(to)) edges.get(from).add(to);
  };
  for (const ti of committed) for (const tj of committed) {
    if (ti.id === tj.id) continue;
    // wr: tj read exactly a version ti wrote; rw: ti read a version (or found the row absent) that tj
    // later overwrote; ww: ti's version of a key precedes tj's version of the same key.
    if (tj.reads.some((r) => ti.writes.some((w) => w.key === r.key && w.version === r.version))) addEdge(ti.id, tj.id);
    if (ti.reads.some((r) => tj.writes.some((w) => w.key === r.key && w.version > r.version))) addEdge(ti.id, tj.id);
    if (ti.writes.some((wi) => tj.writes.some((w) => w.key === wi.key && w.version > wi.version))) addEdge(ti.id, tj.id);
  }
  return { nodes: committed.map((h) => h.id), edges };
}

/** Returns a cycle as an id path (first id repeated) when the graph is cyclic, else null. */
function findMvCycle(history) {
  const { nodes, edges } = buildMvGraph(history);
  const colour = new Map(nodes.map((n) => [n, 0]));
  const stack = [];
  const visit = (node) => {
    colour.set(node, 1);
    stack.push(node);
    for (const next of edges.get(node)) {
      if (colour.get(next) === 1) return stack.slice(stack.indexOf(next)).concat(next);
      const found = colour.get(next) === 0 ? visit(next) : null;
      if (found) return found;
    }
    stack.pop();
    colour.set(node, 2);
    return null;
  };
  for (const node of nodes) if (colour.get(node) === 0) {
    const found = visit(node);
    if (found) return found;
  }
  return null;
}

function isConflictSerializable(history) {
  return findMvCycle(history) === null;
}

function topologicalOrder(history) {
  const { nodes, edges } = buildMvGraph(history);
  const indegree = new Map(nodes.map((n) => [n, 0]));
  for (const from of nodes) for (const to of edges.get(from)) indegree.set(to, indegree.get(to) + 1);
  const ready = nodes.filter((n) => indegree.get(n) === 0);
  const order = [];
  while (ready.length > 0) {
    const node = ready.shift();
    order.push(node);
    for (const to of edges.get(node)) {
      indegree.set(to, indegree.get(to) - 1);
      if (indegree.get(to) === 0) ready.push(to);
    } }
  return order.length === nodes.length ? order : null;
}

/** Serial-order check: do the recorded observations match the state at that point in the order? */
function orderMatchesObservations(order, history, initial) {
  const committed = new Map(history.filter((h) => h.state === "committed").map((h) => [h.id, h]));
  const state = new Map(Object.entries(initial));
  for (const id of order) {
    const txn = committed.get(id);
    for (const r of txn.reads) {
      if ((state.has(r.key) ? state.get(r.key) : MISSING) !== r.value) return false;
    }
    for (const w of txn.writes) state.set(w.key, w.value);
  }
  return true;
}

/** Brute-force oracle: try every permutation of the committed transactions (small histories only). */
function isSerializableByBruteForce(history, initial) {
  const ids = history.filter((h) => h.state === "committed").map((h) => h.id);
  if (ids.length > 5) throw new Error("brute force is limited to five transactions");
  const used = new Array(ids.length).fill(false);
  const order = [];
  const search = () => {
    if (order.length === ids.length) return orderMatchesObservations(order, history, initial);
    for (let i = 0; i < ids.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      order.push(ids[i]);
      const found = search();
      order.pop();
      used[i] = false;
      if (found) return true;
    }
    return false;
  };
  return search();
}

/** Schedule DSL: "begin t1" / "read t1 k" / "guard t1 self" / "write t1 k 11" / "abort t1" / "commit t1". */
function runSteps(db, steps) {
  const events = [];
  for (const step of steps) {
    const [op, txn, key, value] = step.split(" ");
    if (op === "begin") db.begin(txn);
    else if (op === "read") events.push(`read ${txn} ${key}=${db.read(txn, key)}`);
    else if (op === "write") db.write(txn, key, /^-?\d+$/.test(value) ? Number(value) : value);
    else if (op === "abort") db.abort(txn);
    else if (op === "guard") {
      // The doctor-on-call rule: leave the roster only if a second doctor is still on call.
      const onCall = DOCTORS.filter((d) => db.read(txn, d) === ON_CALL).length;
      if (onCall < 2) db.abort(txn); else db.write(txn, key, OFF);
      events.push(`guard ${txn} ${onCall < 2 ? "abort" : "leave"}`);
    } else if (op === "commit") {
      if (!db.isActive(txn)) events.push(`commit ${txn} skipped`);
      else {
        const result = db.commit(txn);
        events.push(`commit ${txn} ${result.committed ? "ok" : result.reason}`);
      }
    } else throw new Error(`unknown step: ${step}`);
  }
  return events;
}

/** The same two transaction bodies, once interleaved and once strictly back to back. */
const DOCTOR_INTERLEAVED = ["begin t1", "begin t2", "guard t1 alice", "guard t2 bob", "commit t1", "commit t2"];
const DOCTOR_SERIAL = ["begin t1", "guard t1 alice", "commit t1", "begin t2", "guard t2 bob", "commit t2"];
const doctorInitial = { alice: ON_CALL, bob: ON_CALL };

function onCallCount(db) {
  const state = db.committedState();
  return DOCTORS.filter((d) => state.get(d) === ON_CALL).length;
}

/** All interleavings of two step lists that keep each transaction's own step order. */
function interleavings(a, b) {
  if (a.length === 0) return [b.slice()];
  if (b.length === 0) return [a.slice()];
  return [
    ...interleavings(a.slice(1), b).map((rest) => [a[0], ...rest]),
    ...interleavings(a, b.slice(1)).map((rest) => [b[0], ...rest]), ];
}

module.exports = {
  SnapshotDb, buildMvGraph, findMvCycle, isConflictSerializable, topologicalOrder,
  orderMatchesObservations, isSerializableByBruteForce, runSteps, interleavings, onCallCount,
  doctorInitial, DOCTOR_INTERLEAVED, DOCTOR_SERIAL, ON_CALL, OFF,
};

if (require.main === module) {
  const assert = require("node:assert");
  const initial = { a: 10, b: 20 };
  // Empty and single-transaction histories are trivially serializable.
  const empty = new SnapshotDb();
  assert.strictEqual(empty.history.length, 0);
  assert.strictEqual(isConflictSerializable(empty.history), true);
  assert.strictEqual(topologicalOrder(empty.history).length, 0);
  assert.deepStrictEqual(runSteps(empty, ["begin solo", "write solo x 1", "commit solo"]), ["commit solo ok"]);
  assert.strictEqual(empty.committedState().get("x"), 1);
  assert.deepStrictEqual(topologicalOrder(empty.history), ["solo"]);
  // Write skew under SI: both doctors leave, so the on-call constraint breaks.
  const skew = new SnapshotDb(doctorInitial);
  assert.deepStrictEqual(runSteps(skew, DOCTOR_INTERLEAVED), [
    "guard t1 leave", "guard t2 leave", "commit t1 ok", "commit t2 ok", ]);
  assert.strictEqual(onCallCount(skew), 0, "both doctors went off call");
  assert.strictEqual(skew.committedState().get("alice"), OFF);
  assert.strictEqual(skew.committedState().get("bob"), OFF);
  assert.strictEqual(isConflictSerializable(skew.history), false, "write skew is not serializable");
  assert.deepStrictEqual(findMvCycle(skew.history), ["t1", "t2", "t1"]);
  assert.strictEqual(topologicalOrder(skew.history), null);
  assert.strictEqual(isSerializableByBruteForce(skew.history, doctorInitial), false);
  skew.begin("observer");
  assert.strictEqual(skew.read("observer", "alice"), OFF, "no serial order sees this state");
  assert.ok(skew.commit("observer").committed, "a read-only transaction always commits");
  // The same bodies run serially: the second guard sees the first commit and aborts.
  const serial = new SnapshotDb(doctorInitial);
  assert.deepStrictEqual(runSteps(serial, DOCTOR_SERIAL), [
    "guard t1 leave", "commit t1 ok", "guard t2 abort", "commit t2 skipped", ]);
  assert.strictEqual(onCallCount(serial), 1, "one doctor stays on call");
  assert.strictEqual(isConflictSerializable(serial.history), true);
  assert.deepStrictEqual(topologicalOrder(serial.history), ["t1"]);
  assert.ok(orderMatchesObservations(["t1"], serial.history, doctorInitial));
  assert.ok(isSerializableByBruteForce(serial.history, doctorInitial));
  assert.deepStrictEqual(serial.history[1].writes, [], "the aborted transaction installed no version");
  assert.strictEqual(serial.activeCount(), 0);
  // Repeatable read of a frozen snapshot, plus an absent row that the reader itself fills in.
  const readDb = new SnapshotDb({ x: 1 });
  assert.deepStrictEqual(runSteps(readDb, [
    "begin reader", "begin writer", "read reader x", "write writer x 2", "commit writer",
    "read reader x", "read reader y", "write reader y 5", "commit reader",
  ]), ["read reader x=1", "commit writer ok", "read reader x=1", "read reader y=undefined", "commit reader ok"]);
  assert.strictEqual(readDb.committedState().get("x"), 2);
  assert.strictEqual(readDb.versionCount("x"), 2);
  assert.strictEqual(readDb.versionCount("y"), 1);
  assert.throws(() => readDb.read("reader", "y"), /not active/, "finished transactions are dead");
  readDb.begin("bad");
  assert.throws(() => readDb.write("bad", "y", MISSING), /absent value/);
  assert.throws(() => readDb.begin("bad"), /already active/);
  assert.throws(() => runSteps(readDb, ["nonsense t1"]), /unknown step/);
  // First-committer-wins: no lost update, and the loser is told which key clashed.
  const lost = new SnapshotDb({ k: 10 });
  assert.deepStrictEqual(runSteps(lost, [
    "begin a", "begin b", "read a k", "read b k", "write a k 11", "write b k 12", "commit a", "commit b",
  ]), ["read a k=10", "read b k=10", "commit a ok", "commit b write-write conflict"]);
  assert.strictEqual(lost.committedState().get("k"), 11, "the late writer did not win");
  assert.strictEqual(lost.versionCount("k"), 2);
  assert.deepStrictEqual(lost.history[1].conflicts, ["k"]);
  assert.deepStrictEqual(lost.history[1].attemptedKeys, ["k"]);
  assert.strictEqual(lost.activeCount(), 0);
  // An explicit abort leaves nothing behind, not even a version.
  const doomed = new SnapshotDb({ m: 1 });
  runSteps(doomed, ["begin doomed", "write doomed m 99", "abort doomed"]);
  assert.strictEqual(doomed.committedState().get("m"), 1);
  assert.strictEqual(doomed.versionCount("m"), 1);
  assert.strictEqual(doomed.history[0].reason, "explicit abort");
  assert.strictEqual(isConflictSerializable(doomed.history), true, "aborted work is ignored");
  // Every interleaving of four pairs of three-step programs (each ending in its own commit): the graph
  // criterion must agree with the brute-force oracle, the topological order must replay, and two
  // committed transactions must never share a written key.
  const variants = [
    [["read t1 a", "write t1 a 1", "commit t1"], ["read t2 b", "write t2 b 2", "commit t2"]], // disjoint
    [["read t1 c", "write t1 c 3", "commit t1"], ["read t2 c", "write t2 c 4", "commit t2"]], // same key
    [["read t1 b", "write t1 a 1", "commit t1"], ["read t2 a", "write t2 b 2", "commit t2"]], // write skew
    [["read t1 c", "write t1 d 7", "commit t1"], ["read t2 d", "write t2 c 8", "commit t2"]], // phantom
  ];
  let serializable = 0;
  let cyclic = 0;
  for (const [first, second] of variants) {
    for (const merged of interleavings(first.slice(), second.slice())) {
      const db = new SnapshotDb(initial);
      runSteps(db, ["begin t1", "begin t2", ...merged]);
      assert.strictEqual(db.activeCount(), 0, "every transaction resolved");
      const committed = db.history.filter((h) => h.state === "committed");
      const acyclic = isConflictSerializable(db.history);
      assert.strictEqual(acyclic, isSerializableByBruteForce(db.history, initial), merged.join(" | "));
      for (let i = 0; i < committed.length; i++) for (let j = i + 1; j < committed.length; j++) {
        const overlap = committed[i].attemptedKeys.filter((k) => committed[j].attemptedKeys.includes(k));
        assert.deepStrictEqual(overlap, [], "committed write sets are disjoint");
      }
      for (const victim of db.history.filter((h) => h.state === "aborted" && h.conflicts.length > 0)) {
        assert.strictEqual(victim.reason, "write-write conflict");
        assert.ok(victim.conflicts.every((k) => victim.attemptedKeys.includes(k)));
      }
      if (acyclic) {
        serializable++;
        assert.ok(orderMatchesObservations(topologicalOrder(db.history), db.history, initial));
      } else cyclic++;
    } }
  assert.ok(serializable > 0 && cyclic > 0, "the sample covers both outcomes");
  console.log(`schedules: ${serializable} serializable, ${cyclic} with a cycle in the MV graph`);
  console.log("PASS");
}
