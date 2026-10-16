/** Task 663 - Implement snapshot-based log compaction and recovery for replicated log. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A replicated log whose state machine is a small key/value store (Raft's `set`/`del`), with the
 * two mechanisms that keep such a log from growing forever:
 *
 *   * compaction — once an entry has been applied, the whole prefix up to the applied index can
 *     be replaced by a *snapshot* of the state machine. The log then only holds entries after
 *     the snapshot, and `firstIndex = snapshotIndex + 1`.
 *   * installSnapshot — a replica that has fallen behind the compacted prefix cannot be caught up
 *     by log entries any more, so the leader ships the snapshot instead; a snapshot older than
 *     the replica's own is rejected rather than rolling the state back.
 *
 * The invariant that makes this safe, and the one this file hammers on: for every replica and any
 * moment, `replay the log from index 1` and `restore the snapshot + replay the retained suffix`
 * produce exactly the same state machine state. Recovery of a fresh replica from a snapshot plus
 * the retained tail must therefore equal the leader's state, and compaction must never drop an
 * entry that has not been applied yet.
 *
 * Simplifications: single-term log (no elections), no log-mismatch repair, snapshots are whole
 * copies of a tiny state machine, and installSnapshot transfers the state directly instead of
 * streaming chunks.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
/** Tiny deterministic state machine: key/value store with set and delete. */
class KeyValueState {
  constructor(entries = []) {
    this.data = new Map(entries);
  }
  apply(command) {
    if (command === null || typeof command !== "object") throw new TypeError("a command must be an object");
    if (command.op === "set") {
      if (typeof command.key !== "string") throw new TypeError("set needs a string key");
      this.data.set(command.key, command.value);
    } else if (command.op === "del") {
      this.data.delete(command.key);
    } else if (command.op === "incr") {
      const current = this.data.get(command.key) ?? 0;
      if (typeof current !== "number") throw new TypeError("incr needs a numeric value");
      this.data.set(command.key, current + (command.by ?? 1));
    } else {
      throw new TypeError(`unknown command ${command.op}`);
    }
    return this;
  }
  get(key) {
    return this.data.get(key);
  }
  size() {
    return this.data.size;
  }
  clone() {
    return new KeyValueState(this.data);
  }
  toSortedList() {
    return [...this.data.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  }
  equals(other) {
    return JSON.stringify(this.toSortedList()) === JSON.stringify(other.toSortedList());
  }
}
/**
 * A replica: applied state machine state, a log of entries after the snapshot, and the snapshot
 * bookkeeping (`snapshotIndex`/`snapshotTerm`).
 */
class Replica {
  constructor(id) {
    this.id = id;
    this.state = new KeyValueState();
    this.entries = [];        // { index, term, command }, contiguous, index === firstIndex + i
    this.snapshotIndex = 0;
    this.snapshotTerm = 0;
    this.appliedIndex = 0;
    this.compactions = 0;
    this.installedSnapshots = 0;
  }
  firstIndex() {
    return this.snapshotIndex + 1;
  }
  lastIndex() {
    return this.snapshotIndex + this.entries.length;
  }
  entryAt(index) {
    if (index < this.firstIndex() || index > this.lastIndex()) return null;
    return this.entries[index - this.firstIndex()];
  }
  /** Appends a command and returns its log index. */
  append(term, command) {
    const index = this.lastIndex() + 1;
    this.entries.push({ index, term, command: structuredClone(command) });
    this.assertStructuralInvariant();
    return index;
  }
  /** Applies every entry up to `index` (idempotent: re-applying an index is a no-op). */
  applyUpTo(index) {
    if (index > this.lastIndex()) throw new RangeError(`cannot apply ${index}: log ends at ${this.lastIndex()}`);
    let applied = 0;
    while (this.appliedIndex < index) {
      const next = this.appliedIndex + 1;
      const entry = this.entryAt(next);
      if (entry !== null) this.state.apply(entry.command);
      this.appliedIndex = next; // a compacted index is already inside the snapshot
      applied++;
    }
    return applied;
  }
  /** Replaces the applied prefix by a snapshot of the state machine and drops those entries. */
  compact() {
    const index = this.appliedIndex;
    if (index <= this.snapshotIndex) {
      return { compacted: false, snapshotIndex: this.snapshotIndex, dropped: 0 };
    }
    const term = this.entryAt(index)?.term ?? this.snapshotTerm;
    const dropped = this.entries.filter((entry) => entry.index <= index).length;
    this.entries = this.entries.filter((entry) => entry.index > index);
    this.snapshotIndex = index;
    this.snapshotTerm = term;
    this.compactions++;
    this.assertStructuralInvariant();
    return { compacted: true, snapshotIndex: index, dropped };
  }
  takeSnapshot() {
    return { state: this.state.clone(), index: this.snapshotIndex, term: this.snapshotTerm };
  }
  /**
   * Installs a leader snapshot. Older snapshots are refused so the state never rolls back, and
   * entries already covered by the snapshot are dropped.
   */
  installSnapshot(snapshot) {
    if (snapshot.index <= this.snapshotIndex) {
      return { installed: false, reason: "stale snapshot" };
    }
    this.state = snapshot.state.clone();
    this.snapshotIndex = snapshot.index;
    this.snapshotTerm = snapshot.term;
    this.entries = this.entries.filter((entry) => entry.index > snapshot.index);
    this.appliedIndex = Math.max(this.appliedIndex, snapshot.index);
    this.installedSnapshots++;
    this.assertStructuralInvariant();
    return { installed: true, index: snapshot.index, retained: this.entries.length };
  }
  /** The leader's *retained* log (what a lagging follower can still receive as entries). */
  retainedCommands() {
    return this.entries.map((entry) => structuredClone(entry.command));
  }
  assertStructuralInvariant() {
    if (this.snapshotIndex > this.appliedIndex) throw new Error("snapshot ahead of the applied index");
    for (let i = 0; i < this.entries.length; i++) {
      if (this.entries[i].index !== this.firstIndex() + i) throw new Error("log is not contiguous");
      if (this.entries[i].term < this.snapshotTerm) throw new Error("entry term below the snapshot term");
    }
    return true;
  }
}
/** Reference recovery: replay every command from scratch into a fresh state machine. */
function replayFromScratch(commands) {
  const machine = new KeyValueState();
  for (const command of commands) machine.apply(command);
  return machine;
}
/** Recovery from a snapshot plus the retained log suffix — must equal the full replay. */
function recoverFromSnapshot(snapshotCommands, snapshotIndex, suffixCommands) {
  const machine = new KeyValueState();
  for (let i = 0; i < snapshotIndex; i++) machine.apply(snapshotCommands[i]);
  for (const command of suffixCommands) machine.apply(command);
  return machine;
}
module.exports = {
  KeyValueState, Replica, replayFromScratch, recoverFromSnapshot,
};
if (require.main === module) {
  // -- state machine basics --------------------------------------------------------
  const machine = new KeyValueState();
  machine.apply({ op: "set", key: "a", value: 1 }).apply({ op: "incr", key: "a", by: 4 });
  strictEqual(machine.get("a"), 5);
  machine.apply({ op: "del", key: "a" });
  strictEqual(machine.get("a"), undefined);
  throws(() => machine.apply({ op: "nope" }), TypeError, "unknown commands must be rejected");
  throws(() => machine.apply(null), TypeError, "a command must be an object");
  throws(() => new KeyValueState([["x", "text"]]).apply({ op: "incr", key: "x" }), TypeError,
    "incr on a non-numeric value must be rejected");
  strictEqual(new KeyValueState().apply({ op: "incr", key: "fresh" }).get("fresh"), 1,
    "incr treats a missing key as zero");
  // -- appending and applying ------------------------------------------------------
  const timeline = [
    { op: "set", key: "x", value: 1 },
    { op: "set", key: "y", value: 2 },
    { op: "incr", key: "x", by: 10 },
    { op: "del", key: "y" },
    { op: "set", key: "z", value: 7 },
    { op: "incr", key: "z", by: 3 },
    { op: "set", key: "x", value: 99 },
    { op: "set", key: "w", value: 0 },
  ];
  const leader = new Replica("leader");
  for (const command of timeline) leader.append(1, command);
  strictEqual(leader.firstIndex(), 1, "an uncompacted log starts at index 1");
  strictEqual(leader.lastIndex(), timeline.length);
  strictEqual(leader.entryAt(3).command.op, "incr");
  strictEqual(leader.entryAt(0), null, "index 0 is below the log range");
  strictEqual(leader.entryAt(99), null, "an index past the end is not an entry");
  throws(() => leader.entryAt(3) && leader.applyUpTo(99), RangeError, "applying past the log end is invalid");
  leader.applyUpTo(leader.lastIndex());
  deepStrictEqual(leader.state.toSortedList(), [["w", 0], ["x", 99], ["z", 10]]);
  const beforeReapply = leader.state.count ?? leader.state.toSortedList().length;
  strictEqual(leader.applyUpTo(leader.lastIndex()), 0, "re-applying a committed prefix applies nothing");
  strictEqual(leader.state.toSortedList().length, beforeReapply);
  // -- compaction keeps the suffix and the state ------------------------------------
  const compacted = leader.compact();
  ok(compacted.compacted && compacted.dropped === timeline.length, "compaction drops the applied prefix");
  strictEqual(leader.entries.length, 0, "nothing is left after compacting the whole log");
  strictEqual(leader.snapshotIndex, timeline.length);
  strictEqual(leader.firstIndex(), timeline.length + 1, "firstIndex follows the snapshot");
  ok(leader.assertStructuralInvariant());
  deepStrictEqual(leader.compact(), { compacted: false, snapshotIndex: timeline.length, dropped: 0 },
    "compacting twice is a no-op");
  // -- snapshot + suffix == full replay, at every prefix ----------------------------
  for (let cut = 0; cut <= timeline.length; cut++) {
    const suffix = timeline.slice(cut);
    const recovered = recoverFromSnapshot(timeline, cut, suffix);
    ok(recovered.equals(replayFromScratch(timeline)), `snapshot at ${cut} plus suffix must match a full replay`);
  }
  // -- honest compaction on a partially applied log ---------------------------------
  const partial = new Replica("partial");
  for (const command of timeline) partial.append(1, command);
  partial.applyUpTo(3);
  const partialCompaction = partial.compact();
  strictEqual(partialCompaction.snapshotIndex, 3);
  strictEqual(partial.entries.length, timeline.length - 3, "unapplied entries are never dropped");
  strictEqual(partial.entries[0].index, 4, "the suffix starts right after the snapshot");
  partial.applyUpTo(partial.lastIndex());
  ok(partial.state.equals(replayFromScratch(timeline)), "state after compaction equals a full replay");
  strictEqual(partial.applyUpTo(2), 0, "applying a lower index is a no-op, never a rollback");
  strictEqual(partial.appliedIndex, partial.lastIndex(), "the applied index never moves backwards");
  // -- leader hands a lagging replica a snapshot plus its retained tail -------------
  const follower = new Replica("follower");
  follower.append(1, timeline[0]);
  follower.append(1, timeline[1]);
  follower.applyUpTo(2);
  strictEqual(follower.lastIndex(), 2, "the follower is two entries behind");
  const installed = follower.installSnapshot(leader.takeSnapshot());
  ok(installed.installed && installed.index === timeline.length, "the follower takes the leader's snapshot");
  strictEqual(follower.entries.length, 0, "entries covered by the snapshot are dropped");
  for (const command of leader.retainedCommands()) follower.append(1, command);
  follower.applyUpTo(follower.lastIndex());
  ok(follower.state.equals(leader.state), "the caught-up follower matches the leader exactly");
  ok(follower.state.equals(replayFromScratch(timeline)), "and matches the from-scratch replay");
  // -- stale snapshots never roll the state back ------------------------------------
  const stale = follower.installSnapshot({ state: new KeyValueState([["bogus", 1]]), index: 2, term: 1 });
  deepStrictEqual(stale, { installed: false, reason: "stale snapshot" });
  ok(follower.state.equals(leader.state), "a stale snapshot must not change the state");
  strictEqual(follower.snapshotIndex, timeline.length);
  deepStrictEqual(new Replica("bad").installSnapshot({ state: new KeyValueState(), index: -1, term: 0 }),
    { installed: false, reason: "stale snapshot" }, "a snapshot beyond the applied index is refused");
  // -- a replica that only ever sees a snapshot recovers the same state -------------
  const late = new Replica("late");
  late.installSnapshot({ state: replayFromScratch(timeline.slice(0, 5)), index: 5, term: 1 });
  for (const command of timeline.slice(5)) late.append(1, command);
  late.applyUpTo(late.lastIndex());
  ok(late.state.equals(replayFromScratch(timeline)), "snapshot-only recovery equals the reference replay");
  strictEqual(late.compactions, 0, "no local compaction was needed");
  // -- compaction keeps working after a snapshot install ----------------------------
  late.compact();
  strictEqual(late.snapshotIndex, timeline.length);
  late.append(1, { op: "set", key: "fresh", value: 1 });
  late.applyUpTo(late.lastIndex());
  strictEqual(late.state.get("fresh"), 1, "the log keeps working after compaction");
  ok(late.assertStructuralInvariant());
  const expectedAfterAppend = replayFromScratch([...timeline, { op: "set", key: "fresh", value: 1 }]);
  ok(late.state.equals(expectedAfterAppend), "and the recovered state still equals the reference");
  console.log(`entries=${timeline.length} dropped=${compacted.dropped} followerInstalls=${follower.installedSnapshots} ` +
    `compactions=${leader.compactions + partial.compactions + late.compactions}`);
  console.log("PASS");
}
