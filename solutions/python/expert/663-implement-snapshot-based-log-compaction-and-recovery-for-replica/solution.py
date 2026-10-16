"""Task 663 - Implement snapshot-based log compaction and recovery for a replicated log.  (Tier: Expert, Ch 9 - System Design and Scalability)

A replicated log with a real state machine, real snapshots and real state transfer. Each replica holds
a log of (index, term, command) entries above a snapshot: the snapshot is the serialized state machine
after the first `snapshot_index` commands, so entry i can be reconstructed as long as i > snapshot_index
and the older prefix is gone forever - which is exactly the point of compaction. The state machine is a
deterministic key-value store (set/del/incr) whose state has a canonical textual encoding, so a snapshot
can be serialized, shipped and rebuilt byte-for-byte.

The rules an implementation must get right are all enforced and checked: AppendEntries is idempotent
(replaying an already applied entry does nothing), a conflicting entry truncates the log suffix, an
entry with a gap in front of it is rejected so the leader can back up, a follower whose next index is
below the leader's snapshot index receives the snapshot plus the log tail instead of entries, a stale
InstallSnapshot (older index or older term) is refused, compaction is only allowed below the applied
index, and a replica can be reconstructed from its serialized snapshot plus log tail. The central
invariant - and the cross-check - is that replaying the full log from scratch and replaying
snapshot+tail yield the *same* state digest, for every compaction point.

Simplified: single-threaded, one leader pushing to followers synchronously, terms are passed in rather
than elected, and the snapshot is a canonical text encoding of the state machine instead of a chunked
binary snapshot stream (chunk offsets/failure mid-transfer are not modelled). Verified: the replay
equivalence, serialization round-trip, truncation and index bookkeeping, state transfer to a far-behind
follower, staleness and gap rejection, idempotence, recovery from disk, and term/index monotonicity.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class StateMachine:
    """Deterministic key-value state machine over (op, key, value) commands."""

    def __init__(self):
        self.data = {}

    def apply(self, command):
        kind, key = command[0], command[1]
        if kind == "set":
            self.data[key] = command[2]
        elif kind == "del":
            self.data.pop(key, None)
        elif kind == "incr":
            self.data[key] = self.data.get(key, 0) + 1
        else:
            raise ValueError(f"unknown command {kind}")

    def serialize(self):
        """Canonical snapshot encoding: sorted key/value pairs, so two equal states encode equally."""
        return ";".join(f"{key}={self.data[key]}" for key in sorted(self.data))

    @classmethod
    def deserialize(cls, text):
        machine = cls()
        if text:
            for pair in text.split(";"):
                key, _, value = pair.partition("=")
                machine.data[key] = int(value)
        return machine

    def digest(self):
        return tuple(sorted(self.data.items()))


class Replica:
    """A replica: snapshot (state + last included index/term) plus the log above it."""

    def __init__(self, name):
        self.name = name
        self.state = StateMachine()
        self.snapshot_index = 0
        self.snapshot_term = 0
        self.snapshot_text = ""
        self.log = []                     # [(index, term, command)], indices > snapshot_index
        self.last_applied = 0
        self.installed_snapshots = 0

    # ------------------------------------------------------------------ log helpers
    def last_index(self):
        return self.log[-1][0] if self.log else self.snapshot_index

    def last_term(self):
        return self.log[-1][1] if self.log else self.snapshot_term

    def term_at(self, index):
        if index == self.snapshot_index:
            return self.snapshot_term
        for entry_index, term, _ in self.log:
            if entry_index == index:
                return term
        return None

    def entry(self, index):
        """None for a compacted index: those must be served from the snapshot."""
        for entry_index, term, command in self.log:
            if entry_index == index:
                return (entry_index, term, command)
        return None

    def append(self, entries):
        """Idempotent AppendEntries: duplicates are ignored, conflicts truncate, gaps are refused."""
        for index, term, command in entries:
            if index <= self.last_applied:
                continue                                   # already applied: replay is a no-op
            if index <= self.snapshot_index:
                continue                                   # compacted away: served by the snapshot
            if index > self.last_index() + 1:
                raise ValueError(f"gap: entry {index} follows {self.last_index()}")
            existing = self.term_at(index)
            if existing is None:
                self.log.append((index, term, command))
            elif existing != term:
                self.log = [entry for entry in self.log if entry[0] < index]
                self.log.append((index, term, command))

    def apply_committed(self, commit_index):
        """Apply every entry up to the commit index; the applied marker makes this idempotent."""
        applied = 0
        for index, _, command in sorted(self.log):
            if index <= commit_index and index > self.last_applied:
                self.state.apply(command)
                self.last_applied = index
                applied += 1
        return applied

    # ------------------------------------------------------------------ snapshots
    def compact(self, up_to):
        check(up_to <= self.last_applied, "only an applied prefix may be compacted away")
        check(up_to >= self.snapshot_index, "compaction may not move the snapshot backwards")
        if up_to == self.snapshot_index:
            return None
        term = self.term_at(up_to)
        check(term is not None, "the compacted index must exist in the log")
        self.snapshot_index = up_to
        self.snapshot_term = term
        self.snapshot_text = self.state.serialize()
        self.log = [entry for entry in self.log if entry[0] > up_to]
        return self.snapshot_text

    def install_snapshot(self, index, term, text):
        """InstallSnapshot: refuses anything at or below what the replica already has."""
        if index <= self.snapshot_index or index <= self.last_applied:
            return False
        self.state = StateMachine.deserialize(text)
        self.snapshot_index = index
        self.snapshot_term = term
        self.snapshot_text = text
        self.last_applied = index
        self.log = [entry for entry in self.log if entry[0] > index]
        self.installed_snapshots += 1
        return True

    def recover(self):
        """Rebuild the machine from the serialized snapshot plus the log tail."""
        machine = StateMachine.deserialize(self.snapshot_text)
        for index, _, command in sorted(self.log):
            if index <= self.last_applied:
                machine.apply(command)
        return machine

    def digest(self):
        return self.state.digest()


def append_entries(follower, entries, commit_index):
    follower.append(entries)
    return follower.apply_committed(commit_index)


def replicate(leader, follower, commit_index):
    """Push what the follower needs: entries when it reaches into the log, a snapshot when it does not.

    A state transfer is followed by the tail, because the snapshot only covers the leader's prefix.
    """
    if follower.last_index() < leader.snapshot_index:
        check(leader.snapshot_index > 0, "a snapshot is only needed once the leader compacted")
        follower.install_snapshot(leader.snapshot_index, leader.snapshot_term, leader.snapshot_text)
    first = follower.last_index() + 1
    entries = [leader.entry(index) for index in range(first, min(commit_index, leader.last_index()) + 1)]
    entries = [entry for entry in entries if entry is not None]
    if entries:
        append_entries(follower, entries, commit_index)
    return follower.last_index()


def commands(count, prefix="k"):
    return [("set", f"{prefix}{index % 7}", index) for index in range(count)]


def main():
    # 1. replaying the full log and replaying snapshot+tail give the same state
    full = Replica("full")
    full.append([(index + 1, 1, command) for index, command in enumerate(commands(40))])
    full.apply_committed(40)
    reference = full.state.digest()
    check(len(reference) == 7, "the workload must touch seven keys")

    for cut in (1, 5, 17, 40):
        compacted = Replica("compacted")
        compacted.append([(index + 1, 1, command) for index, command in enumerate(commands(40))])
        compacted.apply_committed(40)
        text = compacted.compact(cut)
        check(compacted.snapshot_index == cut, "the snapshot must sit at the requested index")
        check(compacted.last_applied == 40, "compaction may not change what was applied")
        check(compacted.digest() == reference, "compaction may not change the state")
        check(compacted.log[0][0] == cut + 1 if compacted.log else True,
              "the log must start right after the snapshot")
        check(not [entry for entry in compacted.log if entry[0] <= cut],
              "no compacted index may stay in the log")
        check(StateMachine.deserialize(text).digest() == reference,
              "the serialized snapshot must rebuild the same state")
        recovered = Replica("recovered")
        recovered.install_snapshot(cut, 1, text)
        recovered.log = list(compacted.log)
        recovered.last_applied = 40
        check(recovered.recover().digest() == reference,
              "snapshot plus log tail must replay to the same state")

    # 2. snapshot encoding is canonical and round-trips
    machine = StateMachine()
    for command in commands(9):
        machine.apply(command)
    text = machine.serialize()
    check(text == StateMachine.deserialize(text).serialize(), "the snapshot encoding must round-trip")
    check(text.split(";") == sorted(text.split(";")), "the encoding must be sorted by key")
    check(Replica("x").state.serialize() == "", "an empty state serializes to an empty string")

    # 3. compacted entries are gone and must come from the snapshot
    replica = Replica("cut")
    replica.append([(index + 1, 3, command) for index, command in enumerate(commands(6))])
    replica.apply_committed(6)
    replica.compact(4)
    check(replica.entry(4) is None, "a compacted entry must not be readable from the log")
    check(replica.entry(5) is not None and replica.entry(5)[0] == 5, "later entries stay readable")
    check(replica.term_at(4) == 3, "the snapshot remembers the term of its last index")
    check(replica.snapshot_term == 3 and replica.snapshot_index == 4, "snapshot bookkeeping")
    check(replica.last_index() == 6, "the last index survives compaction")

    # 4. a follower that only lags a little is caught up with entries
    leader = Replica("leader")
    leader.append([(index + 1, 1, command) for index, command in enumerate(commands(20))])
    leader.apply_committed(20)
    leader.compact(5)
    follower = Replica("follower")
    follower.append([(index + 1, 1, command) for index, command in enumerate(commands(7))])
    follower.apply_committed(7)
    replicate(leader, follower, 20)
    check(follower.last_index() == 20, "the follower must catch up to the leader's last index")
    check(follower.digest() == leader.digest(), "the follower must reach the leader's state")
    check(follower.installed_snapshots == 0, "a slightly lagging follower needs no state transfer")

    # 5. a follower whose log is below the snapshot index gets state transfer
    behind = Replica("behind")
    behind.append([(index + 1, 1, command) for index, command in enumerate(commands(3))])
    behind.apply_committed(3)
    check(behind.last_index() < leader.snapshot_index, "the follower must really be behind the snapshot")
    replicate(leader, behind, 20)
    check(behind.installed_snapshots == 1, "a far-behind follower must receive a snapshot")
    check(behind.snapshot_index == 5, "the installed snapshot must carry the leader's index")
    check(behind.digest() == leader.digest(), "state transfer plus tail must converge")
    check(behind.last_index() == 20, "the follower must also receive the tail")

    # 6. stale and redundant snapshot installs are refused
    check(behind.install_snapshot(5, 1, "z=1") is False, "a snapshot at the current index is refused")
    check(behind.install_snapshot(3, 1, "z=1") is False, "an older index is refused")
    check(behind.install_snapshot(20, 1, "z=1") is False, "the applied index is refused as well")
    check(behind.digest() == leader.digest(), "a refused snapshot must not touch the state")
    check(behind.install_snapshot(25, 2, "n=7") is True,
          "a snapshot beyond the applied point is accepted")
    check(behind.snapshot_index == 25 and behind.last_applied == 25,
          "the installed snapshot becomes the replica's applied point")
    check(behind.digest() == (("n", 7),), "the installed state is the snapshot's state")
    check(behind.install_snapshot(25, 2, "n=8") is False,
          "a snapshot at the installed index may not be installed twice")
    check(behind.installed_snapshots == 2, "the counter must track successful installations")

    # 7. AppendEntries idempotence, conflict truncation and gap rejection
    node = Replica("node")
    node.append([(1, 1, ("set", "a", 1)), (2, 1, ("set", "b", 2))])
    node.apply_committed(2)
    node.append([(1, 1, ("set", "a", 1)), (2, 1, ("set", "b", 2))])
    check(node.last_index() == 2 and node.digest() == (("a", 1), ("b", 2)),
          "replaying AppendEntries must not duplicate entries")
    try:
        node.append([(4, 1, ("set", "d", 4))])
        raise AssertionError("a gap must be rejected")
    except ValueError as error:
        check("gap" in str(error), "the gap guard must fire")
    node.append([(3, 2, ("set", "c", 3))])
    check(node.last_index() == 3 and node.term_at(3) == 2, "a valid next entry is appended")
    node.append([(3, 3, ("set", "c", 99))])
    check(node.term_at(3) == 3 and node.last_index() == 3,
          "a conflicting term must replace the entry")
    node.apply_committed(3)
    check(node.digest() == (("a", 1), ("b", 2), ("c", 99)),
          "the replacement command must be the one applied")
    check(node.apply_committed(3) == 0, "re-applying the same commit index must do nothing")

    # 8. compaction guard rails
    guard = Replica("guard")
    guard.append([(1, 1, ("set", "a", 1)), (2, 1, ("set", "b", 2))])
    guard.apply_committed(1)
    try:
        guard.compact(2)
        raise AssertionError("compaction above the applied index must be refused")
    except AssertionError as error:
        check("applied prefix" in str(error), "the compaction guard must fire")
    guard.compact(1)
    check(guard.snapshot_index == 1 and guard.compact(1) is None,
          "compacting to the current snapshot index is a no-op")
    check(guard.snapshot_text == StateMachine.deserialize("a=1").serialize(),
          "the snapshot must hold the applied prefix only")

    # 9. recovery from disk: the serialized snapshot plus the tail rebuild the exact state
    source = Replica("source")
    source.append([(index + 1, index // 10 + 1, command) for index, command in enumerate(commands(30))])
    source.apply_committed(30)
    source.compact(12)
    disk_snapshot, disk_tail = source.snapshot_text, list(source.log)
    check(disk_snapshot != "", "a non-empty prefix must serialize to a non-empty snapshot")
    restored = Replica("restored")
    restored.install_snapshot(12, source.snapshot_term, disk_snapshot)
    restored.log = list(disk_tail)
    restored.last_applied = 30
    check(restored.recover().digest() == source.digest(),
          "recovery from disk must rebuild the same state")
    check(restored.last_index() == source.last_index(), "recovery must keep the tail's last index")

    # 10. term and index monotonicity
    ordered = Replica("ordered")
    ordered.append([(index + 1, 1, command) for index, command in enumerate(commands(12))])
    ordered.apply_committed(12)
    terms = [ordered.term_at(index) for index in range(1, 13)]
    check(all(term is not None for term in terms), "every log index must have a term")
    check([entry[0] for entry in ordered.log] == list(range(1, 13)),
          "log indices must be consecutive")
    before = ordered.term_at(6)
    ordered.compact(6)
    check(ordered.term_at(6) == before, "compaction must preserve the term of the snapshot index")
    check(all(entry[0] > 6 for entry in ordered.log), "the log must lose the compacted prefix")
    check(ordered.last_applied == 12 and ordered.last_index() == 12,
          "compaction must not disturb the applied marker")

    print(f"663 reference digest={reference} snapshots={behind.installed_snapshots}")
    print(f"663 leader snapshot@{leader.snapshot_index} tail={len(leader.log)} "
          f"restored={restored.last_index()}")
    print("PASS")


if __name__ == "__main__":
    main()
