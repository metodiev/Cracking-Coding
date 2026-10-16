/** Task 638 - Implement serializable snapshot isolation tester. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMPLIFIED: the engine is an in-memory multi-version store driven by explicit schedules (no
 * threads, no timers, no wall-clock), and the SSI guard is the *conservative* form of Cahill's rule. A
 * committing transaction T tentatively installs its versions, the rw-antidependencies (reader -> writer
 * edges: "the reader saw a version the writer later replaced") of the whole live transaction set are
 * recomputed, and T is refused if the resulting graph contains a pivot - a transaction with both an
 * incoming and an outgoing rw edge. Textbook SSI only refuses the pivot when the outgoing end committed
 * first, so it aborts fewer transactions; the invariant verified here is the one that matters ("every
 * accepted history is conflict-serializable"), which the stricter guard also guarantees, and the aborts
 * are reported with their pivots so the extra conservatism is visible.
 *
 * WHY THE EDGES ARE ALL VISIBLE AT COMMIT TIME: a reader that will point at T must have begun before
 * T's commit (otherwise its snapshot already contains T's version), so it is live while T commits and
 * its recorded reads are available. Symmetrically, an edge T -> U that can complete a cycle is
 * re-examined when the reader end commits, because T's own recorded reads are then still in memory.
 *
 * WHAT IS VERIFIED: (a) the doctor-on-call write skew is rejected under SSI (and accepted under plain
 * SI, where the tester's oracles then prove the history non-serializable, so the checks have teeth);
 * (b) over every interleaving of several three-step program pairs, each accepted history is
 * conflict-serializable according to TWO independent oracles - a brute-force search for a serial order
 * of the committed transactions that replays every recorded read, and the acyclicity of the multiversion
 * serialization graph - and the final state equals the state of that serial replay; (c) the accepted
 * history never contains a pivot, and never two committed transactions with overlapping write sets;
 * (d) read-only transactions are never refused, and first-committer-wins still aborts a write-write
 * clash with the conflicting key reported.
 */
"use strict";

const ON_CALL = "on-call";
const OFF = "off";
const DOCTORS = ["alice", "bob"];
const MISSING = undefined;

/** rw-antidependencies (reader -> writer) among the given node views. */
function rwEdges(nodes) {
  const edges = [];
  for (const reader of nodes) for (const writer of nodes) {
    if (reader.id === writer.id) continue;
    for (const [key, version] of reader.readVersions) {
      const written = writer.writtenVersions.get(key);
      if (written !== undefined && written > version) { edges.push([reader.id, writer.id]); break; }
    } }
  return edges;
}

/** Pivots: nodes with an incoming and an outgoing rw edge - the middle of a dangerous structure. */
function pivotIds(nodes, edges) {
  const hasIn = new Set();
  const hasOut = new Set();
  for (const [from, to] of edges) { hasOut.add(from); hasIn.add(to); }
  return nodes.filter((n) => hasIn.has(n.id) && hasOut.has(n.id)).map((n) => n.id).sort();
}

class SsiEngine {
  constructor(initialValues = {}, options = {}) {
    this.serializable = options.serializable !== false; // false -> plain snapshot isolation
    this.versions = new Map(); // key -> [{key, version, value}] ascending
    this.clock = 0;
    this.transactions = new Map();
    this.history = []; // finished transaction records (committed and aborted)
    for (const [key, value] of Object.entries(initialValues)) {
      this.versions.set(key, [{ key, version: 0, value }]);
    } }
  begin(id) {
    if (this.isActive(id)) throw new Error(`transaction ${id} is already active`);
    this.transactions.set(id, {
      id, snapshot: this.clock, state: "active", reads: [], writes: new Map(), readVersions: new Map(), writtenVersions: new Map(),
    });
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
  read(id, key) {
    const txn = this._active(id);
    const chain = this.versions.get(key) || [];
    let found = null;
    for (let i = chain.length - 1; i >= 0; i--) if (chain[i].version <= txn.snapshot) { found = chain[i]; break; }
    const version = found ? found.version : -1;
    const value = found ? found.value : MISSING;
    txn.reads.push({ key, version, value });
    const previous = txn.readVersions.get(key);
    if (previous === undefined || version < previous) txn.readVersions.set(key, version);
    return value;
  }
  write(id, key, value) {
    const txn = this._active(id);
    if (value === MISSING) throw new Error("a write may not store an absent value");
    txn.writes.set(key, value);
  }
  abort(id, reason = "explicit abort") {
    this._finish(this._active(id), "aborted", [], [], reason);
    return { committed: false, reason, conflicts: [] };
  }
  /** The pivot guard: would installing this transaction's writes create a pivot worth refusing? */
  _blockingPivots(txn) {
    const others = [...this.transactions.values()].filter((t) => t.state !== "aborted" && t.id !== txn.id);
    const tentativeWrites = new Map([...txn.writes.keys()].map((key) => [key, this.clock + 1]));
    const nodes = [...others, { id: txn.id, readVersions: txn.readVersions, writtenVersions: tentativeWrites }];
    const committedIds = new Set(others.filter((t) => t.state === "committed").map((t) => t.id));
    return pivotIds(nodes, rwEdges(nodes)).filter((id) => id === txn.id || committedIds.has(id));
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
      return { committed: false, reason: "write-write conflict", conflicts, pivots: [] };
    }
    const pivots = this.serializable ? this._blockingPivots(txn) : [];
    if (pivots.length > 0) {
      this._finish(txn, "aborted", [], [], `dangerous structure (rw pivots ${pivots.join(",")})`);
      return { committed: false, reason: "dangerous structure", conflicts: [], pivots };
    }
    this.clock += 1;
    const written = [];
    for (const [key, value] of txn.writes) {
      const entry = { key, version: this.clock, value };
      const chain = this.versions.get(key);
      if (chain) chain.push(entry); else this.versions.set(key, [entry]);
      written.push(entry);
      txn.writtenVersions.set(key, this.clock);
    }
    this._finish(txn, "committed", written, [], "committed");
    return { committed: true, reason: "committed", conflicts: [], pivots: [], version: this.clock };
  }
  _finish(txn, state, written, conflicts, reason) {
    txn.state = state;
    this.history.push({
      id: txn.id, state, reason, snapshot: txn.snapshot, reads: txn.reads.slice(), writes: written,
      attemptedKeys: [...txn.writes.keys()], conflicts,
    });
    txn.writes = new Map();
  }
  committedState() {
    const state = new Map();
    for (const [key, chain] of this.versions) state.set(key, chain[chain.length - 1].value);
    return state;
  }
  versionCount(key) {
    return (this.versions.get(key) || []).length;
  } }

/** Multiversion serialization graph of a history: wr, rw (anti-dependency) and ww edges. */
function buildMvGraph(history) {
  const committed = history.filter((h) => h.state === "committed");
  const edges = new Map(committed.map((h) => [h.id, new Set()]));
  const addEdge = (from, to) => {
    if (from !== to && edges.has(from) && edges.has(to)) edges.get(from).add(to);
  };
  for (const ti of committed) for (const tj of committed) {
    if (ti.id === tj.id) continue;
    if (tj.reads.some((r) => ti.writes.some((w) => w.key === r.key && w.version === r.version))) addEdge(ti.id, tj.id);
    if (ti.reads.some((r) => tj.writes.some((w) => w.key === r.key && w.version > r.version))) addEdge(ti.id, tj.id);
    if (ti.writes.some((wi) => tj.writes.some((w) => w.key === wi.key && w.version > wi.version))) addEdge(ti.id, tj.id);
  }
  return { nodes: committed.map((h) => h.id), edges };
}

function findCycle(history) {
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
  for (const node of nodes) {
    const found = colour.get(node) === 0 ? visit(node) : null;
    if (found) return found;
  }
  return null;
}

function graphIsSerializable(history) {
  return findCycle(history) === null;
}

/** Replay a candidate serial order over the recorded observations; null if a read does not match. */
function replayOrder(order, history, initial) {
  const byId = new Map(history.filter((h) => h.state === "committed").map((h) => [h.id, h]));
  const state = new Map(Object.entries(initial));
  for (const id of order) {
    const txn = byId.get(id);
    for (const r of txn.reads) {
      if ((state.has(r.key) ? state.get(r.key) : MISSING) !== r.value) return null;
    }
    for (const w of txn.writes) state.set(w.key, w.value);
  }
  return state;
}

function sameState(a, b) {
  if (!a || a.size !== b.size) return false;
  for (const [key, value] of a) if (b.get(key) !== value) return false;
  return true;
}

/** Brute-force oracle: the first serial order that replays every recorded read exactly. */
function findSerialOrder(history, initial) {
  const ids = history.filter((h) => h.state === "committed").map((h) => h.id);
  if (ids.length > 6) throw new Error("brute force is limited to six transactions");
  const used = new Array(ids.length).fill(false);
  const order = [];
  const search = () => {
    if (order.length === ids.length) return replayOrder(order, history, initial) ? order.slice() : null;
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

/** The doctor-on-call rule: go off call only if a second doctor is still on call. */
function doctorGuard(engine, id, self, roster = DOCTORS) {
  const seen = roster.filter((doctor) => engine.read(id, doctor) === ON_CALL).length;
  if (seen < 2) { engine.abort(id, "constraint: not enough doctors on call"); return "aborted"; }
  engine.write(id, self, OFF);
  return "leaving";
}

/** Tiny schedule DSL: "begin t1" / "read t1 k" / "guard t1 self" / "write t1 k 1" / "abort t1" / "commit t1". */
function runSteps(engine, steps) {
  const events = [];
  for (const step of steps) {
    const [op, txn, key, value] = step.split(" ");
    if (op === "begin") engine.begin(txn);
    else if (op === "read") events.push(`read ${txn} ${key}=${engine.read(txn, key)}`);
    else if (op === "write") engine.write(txn, key, /^-?\d+$/.test(value) ? Number(value) : value);
    else if (op === "abort") events.push(`abort ${txn} ${engine.abort(txn).reason}`);
    else if (op === "guard") events.push(`guard ${txn} ${doctorGuard(engine, txn, key) === "aborted" ? "abort" : "leave"}`);
    else if (op === "commit") {
      if (!engine.isActive(txn)) { events.push(`commit ${txn} skipped`); continue; }
      const result = engine.commit(txn);
      events.push(`commit ${txn} ${result.committed ? "ok" : result.reason.split(" (")[0]}`);
    } else throw new Error(`unknown step: ${step}`);
  }
  return events;
}

/** All interleavings of two step lists that keep each transaction's own step order. */
function interleavings(a, b) {
  if (a.length === 0) return [b.slice()];
  if (b.length === 0) return [a.slice()];
  return [
    ...interleavings(a.slice(1), b).map((rest) => [a[0], ...rest]),
    ...interleavings(a, b.slice(1)).map((rest) => [b[0], ...rest]), ];
}

const WRITE_SKEW = ["begin t1", "guard t1 alice", "begin t2", "guard t2 bob", "commit t1", "commit t2"];

module.exports = {
  SsiEngine, rwEdges, pivotIds, buildMvGraph, findCycle, graphIsSerializable, replayOrder,
  sameState, findSerialOrder, doctorGuard, runSteps, interleavings, WRITE_SKEW, ON_CALL, OFF,
};

if (require.main === module) {
  const assert = require("node:assert");
  const initial = { a: 10, b: 20 };
  const roster = { alice: ON_CALL, bob: ON_CALL };
  const onCall = (engine) => DOCTORS.filter((d) => engine.committedState().get(d) === ON_CALL).length;
  // Plain SI accepts the write skew; the tester's oracles then prove that history non-serializable.
  const si = new SsiEngine(roster, { serializable: false });
  assert.deepStrictEqual(runSteps(si, WRITE_SKEW), ["guard t1 leave", "guard t2 leave", "commit t1 ok", "commit t2 ok"]);
  assert.strictEqual(onCall(si), 0, "SI lets the on-call constraint break");
  assert.strictEqual(graphIsSerializable(si.history), false);
  assert.strictEqual(findSerialOrder(si.history, roster), null, "no serial order replays the reads");
  // The same schedule under SSI: the second committer is refused as the pivot of a dangerous structure.
  const ssi = new SsiEngine(roster, { serializable: true });
  assert.deepStrictEqual(runSteps(ssi, WRITE_SKEW), ["guard t1 leave", "guard t2 leave", "commit t1 ok", "commit t2 dangerous structure"]);
  assert.strictEqual(onCall(ssi), 1, "the constraint survives SSI");
  assert.deepStrictEqual(ssi.history[1].reason, "dangerous structure (rw pivots t1,t2)");
  assert.deepStrictEqual(ssi.history[1].writes, [], "the refused transaction installed nothing");
  assert.strictEqual(graphIsSerializable(ssi.history), true);
  assert.deepStrictEqual(findSerialOrder(ssi.history, roster), ["t1"]);
  assert.ok(sameState(replayOrder(["t1"], ssi.history, roster), ssi.committedState()));
  // Back-to-back execution needs no refusal at all: the guard itself stops the second transaction.
  const seq = new SsiEngine(roster, { serializable: true });
  assert.deepStrictEqual(runSteps(seq, ["begin t1", "guard t1 alice", "commit t1", "begin t2", "guard t2 bob", "commit t2"]),
    ["guard t1 leave", "commit t1 ok", "guard t2 abort", "commit t2 skipped"]);
  assert.strictEqual(onCall(seq), 1);
  assert.strictEqual(graphIsSerializable(seq.history), true);
  // Read-only transactions install no version, so they can never be a pivot; both commit here.
  const ro = new SsiEngine(initial, { serializable: true });
  assert.deepStrictEqual(runSteps(ro, [
    "begin reader", "read reader a", "begin writer", "read writer a", "write writer b 21", "commit writer", "commit reader",
  ]), ["read reader a=10", "read writer a=10", "commit writer ok", "commit reader ok"]);
  assert.strictEqual(ro.committedState().get("b"), 21);
  assert.strictEqual(ro.history.filter((h) => h.state === "aborted").length, 0);
  // Disjoint concurrent work is accepted, and first-committer-wins still reports its key.
  const safe = new SsiEngine(initial, { serializable: true });
  assert.deepStrictEqual(runSteps(safe, [
    "begin s1", "begin s2", "read s1 a", "read s2 b", "write s1 b 21", "write s2 c 5", "commit s1", "commit s2",
  ]), ["read s1 a=10", "read s2 b=20", "commit s1 ok", "commit s2 ok"]);
  const lost = new SsiEngine({ k: 10 }, { serializable: true });
  assert.deepStrictEqual(runSteps(lost, [
    "begin x", "begin y", "read x k", "read y k", "write x k 11", "write y k 12", "commit x", "commit y",
  ]), ["read x k=10", "read y k=10", "commit x ok", "commit y write-write conflict"]);
  assert.strictEqual(lost.committedState().get("k"), 11);
  assert.deepStrictEqual(lost.history[1].conflicts, ["k"]);
  assert.throws(() => runSteps(safe, ["begin z", "begin z"]), /already active/);
  assert.throws(() => runSteps(safe, ["noop z"]), /unknown step/);
  // Exhaustive interleavings of four program pairs: whatever SSI accepts must be serializable by both
  // oracles, must replay to the engine state, must contain no pivot and no overlapping write sets.
  const programs = [
    [["read t1 a", "write t1 a 1", "commit t1"], ["read t2 b", "write t2 b 2", "commit t2"]],
    [["read t1 a", "write t1 a 1", "commit t1"], ["read t2 a", "write t2 b 2", "commit t2"]],
    [["read t1 b", "write t1 a 1", "commit t1"], ["read t2 a", "write t2 b 2", "commit t2"]],
    [["read t1 a", "write t1 b 1", "commit t1"], ["read t2 b", "write t2 c 2", "commit t2"]], ];
  let commits = 0;
  let aborts = 0;
  for (const [first, second] of programs) {
    for (const merged of interleavings(first.slice(), second.slice())) {
      const engine = new SsiEngine(initial, { serializable: true });
      runSteps(engine, ["begin t1", "begin t2", ...merged]);
      assert.strictEqual(engine.activeCount(), 0, "every transaction is resolved");
      const label = merged.join(" | ");
      assert.strictEqual(graphIsSerializable(engine.history), true, `accepted a cyclic history: ${label}`);
      const committed = engine.history.filter((h) => h.state === "committed");
      const order = findSerialOrder(engine.history, initial);
      assert.notStrictEqual(order, null, `oracles disagree: ${label}`);
      assert.ok(sameState(replayOrder(order, engine.history, initial), engine.committedState()), `state: ${label}`);
      const live = committed.map((h) => ({ id: h.id, readVersions: new Map(), writtenVersions: new Map() }));
      for (const node of live) {
        const record = committed.find((h) => h.id === node.id);
        for (const r of record.reads) node.readVersions.set(r.key, r.version);
        for (const w of record.writes) node.writtenVersions.set(w.key, w.version);
      }
      assert.deepStrictEqual(pivotIds(live, rwEdges(live)), [], `a committed pivot remains: ${label}`);
      for (let i = 0; i < committed.length; i++) for (let j = i + 1; j < committed.length; j++) {
        const overlap = committed[i].attemptedKeys.filter((k) => committed[j].attemptedKeys.includes(k));
        assert.deepStrictEqual(overlap, [], `overlapping writes accepted: ${label}`);
      }
      commits += committed.length;
      aborts += engine.history.filter((h) => h.state === "aborted").length;
    } }
  assert.ok(commits > 0 && aborts > 0, "the sample exercises both outcomes");
  console.log(`SSI sweep: ${commits} commits, ${aborts} refusals`);
  // Plain SI over the same sweep: the oracles must now find non-serializable histories, which is what
  // proves the machinery above - and the SSI refusals - are not vacuous.
  let siAnomalies = 0;
  for (const [first, second] of programs) {
    for (const merged of interleavings(first.slice(), second.slice())) {
      const engine = new SsiEngine(initial, { serializable: false });
      runSteps(engine, ["begin t1", "begin t2", ...merged]);
      if (!graphIsSerializable(engine.history)) {
        siAnomalies++;
        assert.strictEqual(findSerialOrder(engine.history, initial), null, "both oracles agree");
      } }
  }
  assert.ok(siAnomalies > 0, "plain SI does produce anomalies on this sweep");
  console.log(`plain SI sweep: ${siAnomalies} non-serializable histories`);
  console.log("PASS");
}
