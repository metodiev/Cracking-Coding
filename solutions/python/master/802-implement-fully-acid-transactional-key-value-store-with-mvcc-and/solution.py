"""Task 802 - Implement fully ACID transactional key-value store with MVCC and 2PC.  (Tier: Master, CTCI Ch 9 - System Design and Scalability)

A distributed transactional key/value store in one process.  Keys are partitioned over shards
("participants"); every shard keeps a multi-version store, i.e. one version chain per key whose
versions carry (start_ts, commit_ts, value, tombstone).  Timestamps come from a single monotone
coordinator counter, and the commit_ts of a version is its write timestamp, so a snapshot read at
ts sees exactly the newest version whose commit_ts <= ts - this is snapshot isolation, and it is
also what makes an old reader keep seeing a consistent, never-changing picture while writers move
on.

Writes are not applied at write time: a transaction buffers them and hands them to the two-phase
commit protocol.  Phase one asks every participant that owns one of the write keys to *prepare*:
the participant takes a per-key lock and votes abort if the key is locked by somebody else
(no-wait locking, hence 2PL is deadlock free) or if the key was written at a timestamp later than
the reader's start_ts (first-committer-wins).  If every participant votes commit the coordinator
appends the decision record - the commit point - to its write-ahead log and only then does phase
two ship "commit" messages, which install the versions at the decision timestamp and release the
locks.  A participant that voted abort, or a decision of abort, releases the locks without
installing anything.  Recovery replays the coordinator log: transactions with a decision are
re-shipped (idempotently, the participants detect replays), transactions that were prepared but
never decided are aborted.

Verified invariants (each against an independently written oracle where one exists):
  atomicity      a transaction is applied on every participant or on none - checked with a
                 participant that votes abort, and with a coordinator crash both before and after
                 the commit point; after recovery every participant agrees with the decision and
                 no lock is left behind;
  durability     rebuilding both participants purely by replaying their write-ahead logs
                 reproduces the live version chains, the applied-decision map and the visible
                 state exactly;
  isolation      the visible-at-infinity state equals a plain-dict serial replay of the committed
                 transactions in commit order, and every committed read-only transaction's read
                 set equals the serial replay truncated at its own start_ts (brute-force oracle);
  consistency    a batch of interleaved transfer transactions preserves the total balance, and
                 no aborted transaction's write is ever visible;
  monotonicity   start/commit timestamps are unique and strictly increasing, and every commit_ts
                 is greater than its own start_ts;
  GC safety      reclaiming versions below a watermark leaves every snapshot at ts >= watermark
                 unchanged (brute force over all timestamps), and a watermark that is above the
                 oldest live snapshot is rejected instead of silently corrupting a reader.

Simplified versus a production system (CockroachDB/Percolator/Spanner): one process, no network -
messages are an explicit outbox drained on demand; participants never time out, the crash and
recovery points are scripted instead of driven by wall-clock timing; there is no commitment
atomicity across *transaction log* and distributed state beyond the single coordinator log; range
scans and predicate locks are omitted, so phantom protection is per key; the storage engine is an
in-memory dict rather than an LSM tree.  All schedules are explicit lists of actions, which makes
every run bit-for-bit reproducible.
"""
MAX_TS = 1 << 62


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Tombstone:
    """Delete marker stored in a version chain; keeps the old value readable to old snapshots."""

    def __repr__(self):
        return "TOMBSTONE"


TOMBSTONE = Tombstone()


class Version:
    __slots__ = ("start_ts", "commit_ts", "value", "tombstone")

    def __init__(self, start_ts, commit_ts, value, tombstone):
        self.start_ts = start_ts
        self.commit_ts = commit_ts
        self.value = value
        self.tombstone = tombstone

    def key(self):
        return (self.start_ts, self.commit_ts, self.value, self.tombstone)


# --------------------------------------------------------------------------- participant
class Shard:
    def __init__(self, name):
        self.name = name
        self.versions = {}     # key -> [Version], ordered by commit_ts
        self.locks = {}        # key -> txn id holding the write lock
        self.prepared = {}     # txn id -> (start_ts, {key: value|TOMBSTONE})
        self.decided = {}      # txn id -> commit_ts, for idempotent replay detection
        self.wal = []          # everything needed to rebuild the shard from scratch

    # ------------------------------------------------------------------ snapshot read
    def visible(self, key, ts):
        """Value visible to a snapshot taken at `ts`; None means the key does not exist."""
        best = None
        for version in self.versions.get(key, ()):
            if version.commit_ts <= ts:
                best = version
        if best is None or best.tombstone:
            return None
        return best.value

    def scan(self, ts):
        return {key: value for key, value in
                ((key, self.visible(key, ts)) for key in sorted(self.versions))
                if value is not None}

    def chains(self):
        return {key: [version.key() for version in chain]
                for key, chain in sorted(self.versions.items())}

    # ------------------------------------------------------------------------ phase 1
    def prepare(self, txn_id, start_ts, writes):
        """Take the write locks and vote; returns (vote, reason)."""
        if txn_id in self.prepared:
            return True, "already prepared"
        for key in sorted(writes):
            holder = self.locks.get(key)
            if holder is not None and holder != txn_id:
                return False, "key %s locked by %s" % (key, holder)
            newest = 0
            for version in self.versions.get(key, ()):
                newest = max(newest, version.commit_ts)
            if newest > start_ts:
                return False, "key %s written at ts %d after start ts %d" % (key, newest, start_ts)
        self.prepared[txn_id] = (start_ts, dict(writes))
        for key in sorted(writes):
            self.locks[key] = txn_id
        self.wal.append(("lock", txn_id, start_ts, tuple(sorted(writes.items()))))
        return True, "commit"

    # ------------------------------------------------------------------------ phase 2
    def apply(self, txn_id, commit_ts):
        """Install the prepared versions; idempotent so a re-sent decision is harmless."""
        if txn_id in self.decided:
            return False
        start_ts, writes = self.prepared.get(txn_id, (commit_ts, {}))
        for key in sorted(writes):
            value = writes[key]
            self.wal.append(("apply", txn_id, start_ts, commit_ts, key, value,
                             value is TOMBSTONE))
            self.versions.setdefault(key, []).append(
                Version(start_ts, commit_ts, value, value is TOMBSTONE))
        self.decided[txn_id] = commit_ts
        self.wal.append(("applied", txn_id, commit_ts))
        self.release(txn_id)
        return True

    def release(self, txn_id):
        for key in sorted(self.prepared.get(txn_id, (None, {}))[1]):
            if self.locks.get(key) == txn_id:
                del self.locks[key]
        self.prepared.pop(txn_id, None)
        self.wal.append(("unlock", txn_id))

    # ----------------------------------------------------------------------------- GC
    def gc(self, watermark):
        """Keep the newest version at or below the watermark plus everything above it."""
        removed = 0
        for key in list(self.versions):
            chain = self.versions[key]
            keep, rest = None, []
            for version in chain:
                if version.commit_ts <= watermark:
                    keep = version
                else:
                    rest.append(version)
            kept = ([] if keep is None else [keep]) + rest
            removed += len(chain) - len(kept)
            if kept:
                self.versions[key] = kept
            else:
                del self.versions[key]
        return removed


def rebuild_shard(name, wal):
    """Independent durability oracle: replay a write-ahead log into a fresh participant."""
    shard = Shard(name)
    for record in wal:
        if record[0] == "lock":
            _, txn_id, start_ts, writes = record
            shard.prepared[txn_id] = (start_ts, dict(writes))
            for key, _ in writes:
                shard.locks[key] = txn_id
        elif record[0] == "apply":
            _, txn_id, start_ts, commit_ts, key, value, tombstone = record
            shard.versions.setdefault(key, []).append(
                Version(start_ts, commit_ts, value, tombstone))
        elif record[0] == "applied":
            shard.decided[record[1]] = record[2]
        elif record[0] == "unlock":
            shard.release(record[1])
    return shard


# --------------------------------------------------------------------------- transaction
class Transaction:
    def __init__(self, txn_id, start_ts):
        self.id = txn_id
        self.start_ts = start_ts
        self.writes = {}
        self.reads = {}
        self.read_history = {}
        self.votes = {}
        self.decision = None
        self.commit_ts = 0
        self.reason = None


class Coordinator:
    """Assigns timestamps, drives two-phase commit, keeps the authoritative write-ahead log."""

    def __init__(self):
        self.timestamp = 0
        self.wal = []
        self.alive = True
        self.outbox = []
        self.prepared = {}       # txn id -> {shard name: voted commit}

    def next_ts(self):
        self.timestamp += 1
        return self.timestamp

    def note_vote(self, txn_id, shard_name, vote):
        self.prepared.setdefault(txn_id, {})[shard_name] = vote
        self.wal.append(("vote", txn_id, shard_name, vote))


class System:
    def __init__(self, shard_map, shards):
        self.shard_map = dict(shard_map)
        self.shards = dict(shards)
        self.coordinator = Coordinator()
        self.transactions = {}
        self.committed = []      # (commit_ts, txn id, writes) in commit order - oracle input
        self.deliveries = 0

    # ------------------------------------------------------------------------ client API
    def begin(self, txn_id):
        txn = Transaction(txn_id, self.coordinator.next_ts())
        self.transactions[txn_id] = txn
        return txn.start_ts

    def read(self, txn_id, key):
        txn = self.transactions[txn_id]
        if key in txn.writes:
            value = txn.writes[key]                       # read your own writes
        else:
            value = self.shards[self.shard_map[key]].visible(key, txn.start_ts)
        txn.reads[key] = value
        txn.read_history.setdefault(key, []).append(value)
        return value

    def write(self, txn_id, key, value):
        self.transactions[txn_id].writes[key] = value

    def delete(self, txn_id, key):
        self.transactions[txn_id].writes[key] = TOMBSTONE

    def participants(self, txn):
        return sorted({self.shard_map[key] for key in txn.writes})

    # --------------------------------------------------------------------- two phase commit
    def prepare(self, txn_id):
        txn = self.transactions[txn_id]
        self.require_alive()
        for shard_name in self.participants(txn):
            owned = {key: value for key, value in txn.writes.items()
                     if self.shard_map[key] == shard_name}
            vote, reason = self.shards[shard_name].prepare(txn_id, txn.start_ts, owned)
            txn.votes[shard_name] = vote
            if reason != "already prepared" and not vote:
                txn.reason = "%s: %s" % (shard_name, reason)
            self.coordinator.note_vote(txn_id, shard_name, vote)
        return all(txn.votes.values())

    def decide(self, txn_id):
        """Append the decision record; this write is the commit point of the transaction."""
        txn = self.transactions[txn_id]
        self.require_alive()
        participants = self.participants(txn)
        commit = all(txn.votes.values()) if participants else True
        if commit:
            txn.commit_ts = self.coordinator.next_ts()
        txn.decision = "commit" if commit else "abort"
        self.coordinator.wal.append(("decision", txn_id, txn.decision, txn.commit_ts))
        if commit:
            self.committed.append((txn.commit_ts, txn_id, dict(txn.writes)))
        return txn.decision

    def finish(self, txn_id):
        txn = self.transactions[txn_id]
        targets = self.participants(txn)
        if not targets:
            return 0
        self.coordinator.outbox.append((txn_id, txn.decision, txn.commit_ts, tuple(targets)))
        return len(targets)

    def commit(self, txn_id):
        self.prepare(txn_id)
        decision = self.decide(txn_id)
        self.finish(txn_id)
        self.quiesce()
        return decision

    def require_alive(self):
        if not self.coordinator.alive:
            raise AssertionError("the coordinator is crashed")

    # ------------------------------------------------------------------------ delivery
    def quiesce(self):
        """Deliver every pending phase-2 message; messages vanish while the coordinator is down."""
        delivered = 0
        while self.coordinator.outbox:
            txn_id, decision, commit_ts, targets = self.coordinator.outbox.pop(0)
            if not self.coordinator.alive:
                continue
            for shard_name in targets:
                shard = self.shards[shard_name]
                if decision == "commit":
                    shard.apply(txn_id, commit_ts)
                else:
                    shard.release(txn_id)
                delivered += 1
        self.deliveries += delivered
        return delivered

    def crash_coordinator(self):
        self.coordinator.alive = False
        self.coordinator.outbox = []          # in-flight phase-2 messages are lost

    def restart_coordinator(self):
        """Recovery: replay the log, finish decided transactions, abort the in-doubt ones."""
        self.coordinator.alive = True
        highest = 0
        decided = {}
        for record in self.coordinator.wal:
            if record[0] == "vote" and record[3]:
                self.coordinator.prepared.setdefault(record[1], {})[record[2]] = True
            elif record[0] == "decision":
                decided[record[1]] = (record[2], record[3])
                highest = max(highest, record[3])
        self.coordinator.timestamp = max(self.coordinator.timestamp, highest)
        resent, aborted = 0, 0
        for txn_id in sorted(self.coordinator.prepared):
            if txn_id in decided:
                decision, commit_ts = decided[txn_id]
            else:
                decision, commit_ts = "abort", 0
                self.coordinator.wal.append(("decision", txn_id, "abort", 0))
                aborted += 1
            for shard_name in sorted(self.coordinator.prepared[txn_id]):
                self.coordinator.outbox.append((txn_id, decision, commit_ts, (shard_name,)))
                resent += 1
        for commit_ts, txn_id, writes in self.committed:
            if commit_ts > highest:
                raise AssertionError("recovery must not lose the commit point ordering")
        self.quiesce()
        return resent, aborted

    # -------------------------------------------------------------------------- inspection
    def visible_state(self, ts=MAX_TS):
        state = {}
        for shard in self.shards.values():
            state.update(shard.scan(ts))
        return state

    def locks_held(self):
        return {name: dict(shard.locks) for name, shard in sorted(self.shards.items())
                if shard.locks}

    def gc(self, watermark):
        """Reclaim dead versions; refuse a watermark that would break a live snapshot."""
        live = [txn.start_ts for txn in self.transactions.values() if txn.decision is None]
        if live and watermark > min(live):
            raise ValueError("watermark %d is below the oldest live snapshot %d"
                             % (watermark, min(live)))
        return sum(shard.gc(watermark) for shard in self.shards.values())


# ------------------------------------------------------------------------------- oracles
def replay_serial(committed, upto=MAX_TS):
    """Plain-dict serial execution of the committed transactions in commit-ts order."""
    state = {}
    for commit_ts, _, writes in sorted(committed):
        if commit_ts > upto:
            break
        for key in sorted(writes):
            if writes[key] is TOMBSTONE:
                state.pop(key, None)
            else:
                state[key] = writes[key]
    return state


def run_schedule(system, schedule):
    results = []
    for action in schedule:
        kind = action[0]
        if kind == "begin":
            results.append(system.begin(action[1]))
        elif kind == "read":
            results.append(system.read(action[1], action[2]))
        elif kind == "write":
            results.append(system.write(action[1], action[2], action[3]))
        elif kind == "delete":
            results.append(system.delete(action[1], action[2]))
        elif kind == "prepare":
            results.append(system.prepare(action[1]))
        elif kind == "decide":
            results.append(system.decide(action[1]))
        elif kind == "finish":
            results.append(system.finish(action[1]))
        elif kind == "commit":
            results.append(system.commit(action[1]))
        elif kind == "crash":
            results.append(system.crash_coordinator())
        elif kind == "restart":
            results.append(system.restart_coordinator())
        elif kind == "quiesce":
            results.append(system.quiesce())
        else:
            raise ValueError("unknown action %r" % (action,))
    return results


def generated_schedule(seed, steps, keys=("a", "b", "c", "d")):
    """Deterministic pseudo-random multi-transaction schedule (no library randomness)."""
    schedule = []
    state = seed
    open_ids = []
    next_id = 0
    for _ in range(steps):
        state, raw = splitmix64(state)
        pick = raw % 100
        if not open_ids or (pick < 35 and len(open_ids) < 4):
            next_id += 1
            txn_id = "g%d" % next_id
            open_ids.append(txn_id)
            schedule.append(("begin", txn_id))
        elif pick < 75:
            txn_id = open_ids[raw // 100 % len(open_ids)]
            key = keys[(raw >> 8) % len(keys)]
            schedule.append(("write", txn_id, key, int((raw >> 16) % 50)))
        else:
            txn_id = open_ids.pop(raw // 100 % len(open_ids))
            schedule.append(("commit", txn_id))
    for txn_id in open_ids:
        schedule.append(("commit", txn_id))
    return schedule


def fresh_store():
    """Two shards: s1 owns a/b, s2 owns c/d."""
    return System({"a": "s1", "b": "s1", "c": "s2", "d": "s2"},
                  {"s1": Shard("s1"), "s2": Shard("s2")})


def main():
    # --- 1. snapshot isolation, read-your-writes, repeatable read, tombstones -------
    store = fresh_store()
    run_schedule(store, [("begin", "t1"), ("read", "t1", "a"), ("write", "t1", "a", 10),
                         ("commit", "t1")])
    check(store.visible_state() == {"a": 10}, "a committed write must become visible")
    check(store.transactions["t1"].read_history["a"] == [None],
          "an unwritten key must read as absent")

    run_schedule(store, [("begin", "t2"), ("write", "t2", "b", 20), ("begin", "t3"),
                         ("read", "t3", "b")])
    check(store.transactions["t3"].reads["b"] is None,
          "a snapshot must not see another transaction's buffered write")
    check(store.read("t2", "b") == 20, "a transaction must read its own writes")
    run_schedule(store, [("commit", "t2")])
    check(store.read("t3", "b") is None,
          "repeatable read: a snapshot must not change after another transaction commits")
    check(store.transactions["t3"].read_history["b"] == [None, None],
          "both reads of the snapshot must return the same value")
    run_schedule(store, [("begin", "t4"), ("read", "t4", "b"), ("commit", "t4")])
    check(store.transactions["t4"].reads["b"] == 20,
          "a transaction that starts after the commit must see the new value")
    run_schedule(store, [("begin", "t5"), ("delete", "t5", "a"), ("commit", "t5")])
    check(store.visible_state() == {"b": 20}, "a delete must remove the key from the store")
    a_start = store.transactions["t1"].start_ts
    check(store.shards["s1"].visible("a", a_start) is None
          and store.shards["s1"].visible("a", store.transactions["t5"].commit_ts) is None,
          "the tombstone must hide the key from both old and new snapshots")
    old_version = store.shards["s1"].versions["a"][0]
    check(old_version.value == 10 and not old_version.tombstone,
          "the old value must survive inside the version chain for historical snapshots")
    check(store.shards["s1"].visible("a", old_version.commit_ts) == 10,
          "a snapshot taken at the old commit timestamp must still read the old value")

    # --- 2. write/write conflict: first committer wins, loser aborts ----------------
    conflict = fresh_store()
    schedule = [("begin", "w1"), ("begin", "w2"), ("write", "w1", "a", 1),
                ("write", "w2", "a", 2), ("prepare", "w1"), ("prepare", "w2")]
    votes = run_schedule(conflict, schedule)
    check(votes[4] is True and votes[5] is False,
          "the first preparer must win the key and the second must be refused")
    check("locked by w1" in conflict.transactions["w2"].reason,
          "the loser must record why it was refused, got %r"
          % conflict.transactions["w2"].reason)
    run_schedule(conflict, [("decide", "w1"), ("finish", "w1"), ("decide", "w2"),
                            ("finish", "w2"), ("quiesce",)])
    check(conflict.transactions["w2"].decision == "abort",
          "a transaction that failed to prepare must be aborted")
    check(conflict.visible_state() == {"a": 1},
          "the aborted write must never become visible: %s" % (conflict.visible_state(),))
    check(conflict.locks_held() == {}, "an aborted transaction must release every lock")
    check([entry[1] for entry in conflict.committed] == ["w1"],
          "only the winner may appear in the commit order")
    check(conflict.shards["s1"].decided.get("w2") is None,
          "an aborted transaction must leave no decided record")
    check(all(version.value != 2 for version in conflict.shards["s1"].versions["a"]),
          "no version of the losing transaction may be installed")

    # --- 3. cross-shard atomicity when one participant votes abort -------------------
    cross = fresh_store()
    run_schedule(cross, [("begin", "x1"), ("begin", "seed"), ("write", "seed", "c", 100),
                         ("commit", "seed")])
    results = run_schedule(cross, [("write", "x1", "a", 1), ("write", "x1", "c", 2),
                                   ("prepare", "x1")])
    check(results[2] is False,
          "a write committed after the start timestamp must abort the whole transaction")
    check(sorted(cross.shards["s1"].locks) == ["a"] and sorted(cross.shards["s2"].locks) == [],
          "the stale participant must not take the lock")
    run_schedule(cross, [("decide", "x1"), ("finish", "x1"), ("quiesce",)])
    check(cross.shards["s1"].visible("a", MAX_TS) is None
          and cross.shards["s2"].visible("c", MAX_TS) == 100,
          "an aborted cross-shard transaction must leave both participants untouched")
    check(cross.locks_held() == {}, "the abort must release the lock taken on the good shard")

    # --- 4. crash after the commit point: recovery completes on every participant ---
    crash = fresh_store()
    run_schedule(crash, [("begin", "d1"), ("write", "d1", "a", 7), ("write", "d1", "c", 8),
                         ("prepare", "d1"), ("decide", "d1")])
    decision = crash.transactions["d1"].decision
    check(decision == "commit", "both participants voted commit, so the decision is commit")
    crash.crash_coordinator()
    check(crash.quiesce() == 0, "phase-2 messages must be lost while the coordinator is down")
    check(crash.visible_state() == {},
          "without phase two nothing may be applied: %s" % (crash.visible_state(),))
    resent, aborted = crash.restart_coordinator()
    check(resent == 2 and aborted == 0,
          "recovery must re-ship both decisions, resent %d aborted %d" % (resent, aborted))
    check(crash.visible_state() == {"a": 7, "c": 8},
          "after recovery the committed transaction must be applied everywhere: %s"
          % (crash.visible_state(),))
    check(crash.deliveries == 2 and crash.locks_held() == {},
          "recovery must deliver exactly the missing messages and drop every lock")

    # --- 5. crash before the commit point: recovery aborts the in-doubt transaction --
    indoubt = fresh_store()
    run_schedule(indoubt, [("begin", "i1"), ("write", "i1", "a", 5), ("write", "i1", "d", 6),
                           ("prepare", "i1")])
    indoubt.crash_coordinator()
    check(sorted(indoubt.locks_held().get("s1", {})) == ["a"],
          "the prepared locks must be held while the outcome is unknown")
    resent, aborted = indoubt.restart_coordinator()
    check(aborted == 1 and resent >= 1,
          "an in-doubt transaction must be aborted by recovery, aborted %d" % aborted)
    check(indoubt.visible_state() == {}, "an in-doubt transaction must not be applied")
    check(indoubt.locks_held() == {}, "recovery must release the locks of the aborted transaction")
    check(all(shard.decided.get("i1") is None for shard in indoubt.shards.values()),
          "an aborted in-doubt transaction must leave no decided record")

    # --- 6. durability: rebuild the participants from their logs ---------------------
    durable = fresh_store()
    run_schedule(durable, generated_schedule(4242, 90))
    rebuilt = {name: rebuild_shard(name, shard.wal)
               for name, shard in durable.shards.items()}
    for name in sorted(durable.shards):
        check(rebuilt[name].chains() == durable.shards[name].chains(),
              "replaying the log of %s must reproduce every version chain" % name)
        check(rebuilt[name].decided == durable.shards[name].decided,
              "replaying the log of %s must reproduce the decided map" % name)
    rebuilt_state = {}
    for shard in rebuilt.values():
        rebuilt_state.update(shard.scan(MAX_TS))
    check(rebuilt_state == durable.visible_state(),
          "the replayed store must expose exactly the same state")
    check(durable.visible_state() == replay_serial(durable.committed),
          "the store must equal a serial replay of the committed transactions")
    check(len(durable.committed) >= 8,
          "the generated schedule must commit a meaningful number of transactions, got %d"
          % len(durable.committed))

    # --- 7. isolation oracle: read-only snapshots vs the serial order ---------------
    oracle = fresh_store()
    run_schedule(oracle, generated_schedule(777, 60))
    reads = run_schedule(oracle, [("begin", "r1"), ("read", "r1", "a"), ("read", "r1", "b"),
                                 ("read", "r1", "c"), ("read", "r1", "d"), ("commit", "r1")])
    start_ts = reads[0]
    expected = replay_serial(oracle.committed, upto=start_ts)
    observed = {key: value for key, value in oracle.transactions["r1"].reads.items()
                if value is not None}
    check(observed == expected,
          "a snapshot must equal the serial state as of its start_ts: %s vs %s"
          % (observed, expected))
    check(oracle.transactions["r1"].decision == "commit",
          "a read-only transaction must commit even with an empty write set")
    check(oracle.transactions["r1"].commit_ts > start_ts,
          "every commit timestamp must be greater than the transaction's start timestamp")

    # --- 8. invariant: interleaved transfers preserve the total --------------------
    bank = fresh_store()
    run_schedule(bank, [("begin", "init"), ("write", "init", "a", 100),
                        ("write", "init", "b", 100), ("write", "init", "c", 100),
                        ("write", "init", "d", 100), ("commit", "init")])
    opening = sum(bank.visible_state().values())
    for step in range(40):
        txn_id = "m%d" % step
        source = ("a", "b", "c", "d")[step % 4]
        target = ("a", "b", "c", "d")[(step * 3 + 1) % 4]
        amount = 5 + step % 7
        run_schedule(bank, [("begin", txn_id)])
        source_value = bank.read(txn_id, source) or 0       # snapshot read, then blind write
        target_value = bank.read(txn_id, target) or 0
        actions = [("write", txn_id, source, source_value - amount),
                   ("write", txn_id, target, target_value + amount)]
        actions.append(("prepare" if step in (3, 27) else "commit", txn_id))
        run_schedule(bank, actions)
    for txn_id in ("m3", "m27"):                            # resolve the delayed transfers
        run_schedule(bank, [("decide", txn_id), ("finish", txn_id), ("quiesce",)])
    accepted = [entry[1] for entry in bank.committed if entry[1].startswith("m")]
    refused = [txn_id for txn_id, txn in bank.transactions.items()
               if txn_id.startswith("m") and txn.decision == "abort"]
    check(len(accepted) >= 10, "most interleaved transfers must commit, got %d" % len(accepted))
    check(len(refused) >= 1,
          "some transfers must lose a lock or a timestamp race, got %d" % len(refused))
    check(len(accepted) + len(refused) == 40, "every transfer must have a decision")
    check(sum(bank.visible_state().values()) == opening,
          "no committed transfer may change the total: %d vs %d"
          % (sum(bank.visible_state().values()), opening))
    check(bank.visible_state() == replay_serial(bank.committed),
          "the bank state must equal the serial replay of the accepted transfers")
    check(all(txn_id not in [entry[1] for entry in bank.committed] for txn_id in refused),
          "an aborted transfer must not appear in the commit order")
    check(all(bank.visible_state().get(key, 0) >= 0 for key in ("a", "b", "c", "d")),
          "the transfer amounts must never drive a committed balance negative")

    # --- 9. garbage collection safety ----------------------------------------------
    gc_store = fresh_store()
    run_schedule(gc_store, generated_schedule(31337, 120))
    sequential = []
    for step in range(30):                     # serial writers pile up version chains
        txn_id = "p%d" % step
        sequential += [("begin", txn_id), ("write", txn_id, "a", step),
                       ("write", txn_id, "c", step), ("commit", txn_id)]
    run_schedule(gc_store, sequential)
    highest = max(entry[0] for entry in gc_store.committed)
    watermark = max(1, highest // 2)
    before = {ts: gc_store.visible_state(ts) for ts in range(0, highest + 1)}
    versions_before = sum(len(chain) for shard in gc_store.shards.values()
                          for chain in shard.versions.values())
    reclaimed = gc_store.gc(watermark)
    versions_after = sum(len(chain) for shard in gc_store.shards.values()
                         for chain in shard.versions.values())
    check(versions_after < versions_before,
          "collection must reclaim dead versions: %d -> %d" % (versions_before, versions_after))
    for ts in range(watermark, highest + 1):
        check(gc_store.visible_state(ts) == before[ts],
              "collection must not change the snapshot at ts %d" % ts)
    check(reclaimed == versions_before - versions_after,
          "the reported count must match the number of reclaimed versions")
    gc_store.gc(highest + 1)
    check(gc_store.visible_state() == replay_serial(gc_store.committed),
          "a full collection must still expose the committed state")

    guarded = fresh_store()
    run_schedule(guarded, [("begin", "g0"), ("read", "g0", "a")])
    run_schedule(guarded, [("begin", "g1"), ("write", "g1", "a", 1), ("commit", "g1")])
    raised = False
    try:
        guarded.gc(guarded.transactions["g1"].commit_ts)
    except ValueError:
        raised = True
    check(raised, "a watermark above the oldest live snapshot must be refused")
    check(guarded.gc(guarded.transactions["g0"].start_ts) == 0,
          "a watermark at the oldest live snapshot is safe and reclaims nothing below it")

    # --- 10. idempotent redelivery and empty transactions ---------------------------
    replay = fresh_store()
    run_schedule(replay, [("begin", "s1"), ("write", "s1", "a", 9), ("prepare", "s1"),
                          ("decide", "s1")])
    commit_ts = replay.transactions["s1"].commit_ts
    replay.coordinator.outbox.append(("s1", "commit", commit_ts, ("s1",)))
    replay.quiesce()
    replay.coordinator.outbox.append(("s1", "commit", commit_ts, ("s1",)))
    replay.quiesce()
    check(len(replay.shards["s1"].versions["a"]) == 1,
          "a re-sent commit decision must not install the version twice")
    check(replay.visible_state() == {"a": 9}, "redelivery must leave the state unchanged")
    run_schedule(replay, [("begin", "empty"), ("commit", "empty")])
    check(replay.transactions["empty"].decision == "commit"
          and replay.transactions["empty"].commit_ts > replay.transactions["empty"].start_ts,
          "an empty transaction commits with a fresh commit timestamp and no participants")
    check(replay.deliveries == 2, "only real phase-2 deliveries may be counted, got %d"
          % replay.deliveries)

    # --- 11. reproducibility -------------------------------------------------------
    first = fresh_store()
    run_schedule(first, generated_schedule(2024, 70))
    second = fresh_store()
    run_schedule(second, generated_schedule(2024, 70))
    check(first.visible_state() == second.visible_state()
          and first.committed == second.committed,
          "identical schedules must produce identical committed state")

    print("generated schedule: %d committed, %d aborted, %d keys, %d version chains"
          % (len(durable.committed),
             sum(1 for txn in durable.transactions.values() if txn.decision == "abort"),
             len(durable.visible_state()),
             sum(len(shard.versions) for shard in durable.shards.values())))
    print("gc watermark %d: %d -> %d versions, snapshots above the watermark unchanged"
          % (watermark, versions_before, versions_after))
    print("bank: %d transfers, %d committed, total preserved at %d"
          % (len(bank.transactions), len(bank.committed),
             sum(bank.visible_state().values())))
    print("PASS")


if __name__ == "__main__":
    main()
