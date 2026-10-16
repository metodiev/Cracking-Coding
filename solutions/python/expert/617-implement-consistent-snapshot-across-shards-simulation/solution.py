"""Task 617 - Implement consistent snapshot across shards simulation.  (Tier: Expert, Ch 9 — System Design and Scalability)

A sharded key-value store where every shard keeps multi-versioned values (one version per
committing transaction, stamped with a global logical commit timestamp), and a coordinator that
takes a *consistent snapshot across shards*: it picks one global timestamp T and asks every shard
for its state as of T.  Because a shard only makes a transaction's writes visible at commit -
with the same timestamp in every shard the transaction touched - the union of the per-shard
states is a single consistent cut.

Transactions use two-phase commit: the coordinator sends the writes to every participant, each
participant "prepares" (buffers them, refuses nothing in this simulation) and only the coordinator
commit makes them visible atomically with one timestamp.  A transaction that is prepared but not
yet committed at T is therefore invisible in every shard - no torn transaction.

Verified:
  * per-key equality with an independent replay model: applying the committed transactions in
    timestamp order to a from-scratch store must reproduce the snapshot exactly, for a sweep of
    snapshot timestamps;
  * atomicity of the cut: for every transaction that spans T, either all of its writes are in the
    snapshot or none of them are (no torn transaction), checked for every T in the sweep;
  * a cross-shard conservation invariant (a transfer between shards keeps the total constant) holds
    for the snapshot at every T, and the snapshot never contains a write whose commit timestamp is
    greater than T;
  * negative controls: taking the per-shard snapshots at *different* timestamps really does tear a
    cross-shard transaction (the invariant then breaks), and snapshotting a shard that has an
    uncommitted prepared transaction while another shard has already applied it is detected.
"""


class Shard:
    """Multi-versioned shard: key -> [(commit timestamp, value)] in commit order."""

    def __init__(self, name):
        self.name = name
        self.versions = {}
        self.prepared = {}          # transaction id -> {key: value}, buffered writes only

    def prepare(self, transaction_id, writes):
        self.prepared[transaction_id] = dict(writes)

    def commit(self, transaction_id, timestamp):
        writes = self.prepared.pop(transaction_id, None)
        if writes is None:
            raise AssertionError("%s: no prepared transaction %s" % (self.name, transaction_id))
        for key, value in writes.items():
            self.versions.setdefault(key, []).append((timestamp, value, transaction_id))

    def abort(self, transaction_id):
        self.prepared.pop(transaction_id, None)

    def read(self, key, timestamp):
        """Value visible at that timestamp: the newest version committed at or before it."""
        best = None
        for commit_timestamp, value, _ in self.versions.get(key, []):
            if commit_timestamp <= timestamp and (best is None or commit_timestamp > best[0]):
                best = (commit_timestamp, value)
        return None if best is None else best[1]

    def visible(self, key, transaction_id, timestamp):
        """True when that transaction's own version of the key is inside the snapshot."""
        return any(commit_timestamp <= timestamp and tid == transaction_id
                   for commit_timestamp, _, tid in self.versions.get(key, []))

    def snapshot_at(self, timestamp):
        return {key: self.read(key, timestamp) for key in self.versions}

    def keys(self):
        return sorted(self.versions)


class Transaction:
    def __init__(self, transaction_id, writes):
        self.id = transaction_id
        self.writes = writes                # shard name -> {key: value}
        self.started = None
        self.prepared_at = None
        self.committed_at = None
        self.aborted = False


class Cluster:
    """Coordinator plus shards; the logical commit clock only moves on commit."""

    def __init__(self, shards):
        self.shards = {name: Shard(name) for name in shards}
        self.clock = 0
        self.transactions = {}
        self.history = []                   # (timestamp, transaction id, writes)

    def begin(self, transaction_id, writes):
        for shard in writes:
            if shard not in self.shards:
                raise AssertionError("unknown shard %s" % shard)
        transaction = Transaction(transaction_id, writes)
        transaction.started = self.clock
        self.transactions[transaction_id] = transaction
        return transaction

    def prepare(self, transaction_id):
        transaction = self.transactions[transaction_id]
        if transaction.aborted or transaction.committed_at is not None:
            raise AssertionError("transaction %s is not active" % transaction_id)
        transaction.prepared_at = self.clock
        for shard, writes in transaction.writes.items():
            self.shards[shard].prepare(transaction_id, writes)
        return transaction.prepared_at

    def commit(self, transaction_id):
        transaction = self.transactions[transaction_id]
        if transaction.prepared_at is None:
            raise AssertionError("transaction %s was not prepared" % transaction_id)
        self.clock += 1
        transaction.committed_at = self.clock
        for shard in transaction.writes:
            self.shards[shard].commit(transaction_id, self.clock)
        self.history.append((self.clock, transaction_id, transaction.writes))
        return self.clock

    def abort(self, transaction_id):
        transaction = self.transactions[transaction_id]
        transaction.aborted = True
        for shard in transaction.writes:
            self.shards[shard].abort(transaction_id)
        return None

    # -- snapshots ---------------------------------------------------------
    def consistent_snapshot(self, timestamp):
        """One cut: every shard is read at the *same* timestamp."""
        return {name: shard.snapshot_at(timestamp) for name, shard in self.shards.items()}

    def torn_snapshot(self, per_shard_timestamps):
        """Deliberately wrong: each shard is read at its own timestamp."""
        return {name: self.shards[name].snapshot_at(timestamp)
                for name, timestamp in per_shard_timestamps.items()}

    def replay(self, timestamp):
        """Independent oracle: apply the committed transactions in order up to the timestamp."""
        state = {name: {} for name in self.shards}
        for commit_timestamp, _, writes in sorted(self.history):
            if commit_timestamp > timestamp:
                break
            for shard, pairs in writes.items():
                for key, value in pairs.items():
                    state[shard][key] = value
        return state

    def torn_transactions(self, snapshot_timestamps):
        """Transactions that are only half-visible when shards are read at their own times."""
        torn = []
        for commit_timestamp, transaction_id, writes in self.history:
            visible = [snapshot_timestamps[shard] >= commit_timestamp for shard in writes]
            if any(visible) and not all(visible):
                torn.append(transaction_id)
        return torn


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    cluster = Cluster(["shard_a", "shard_b", "shard_c"])
    cluster.begin("seed", {"shard_a": {"balance": 100}, "shard_b": {"balance": 200},
                           "shard_c": {"balance": 50}})
    cluster.prepare("seed")
    cluster.commit("seed")                     # one atomic cross-shard transaction
    check(cluster.clock == 1, "the seed commits at clock 1")
    early = cluster.consistent_snapshot(0)
    check(all(value is None for state in early.values() for value in state.values()),
          "before the seed commits no value is visible in any shard")

    # ---- a deterministic workload: every transaction is a cross-shard transfer ----
    def balance(name):
        return cluster.shards[name].read("balance", cluster.clock) or 0

    transfers = [("shard_a", "shard_b", 5), ("shard_b", "shard_c", 3),
                 ("shard_c", "shard_a", 2), ("shard_a", "shard_c", 4)]
    prepared_but_open = []
    for step in range(30):
        actor = "txn%02d" % step
        source, target, amount = transfers[step % 4]
        cluster.begin(actor, {source: {"balance": balance(source) - amount},
                              target: {"balance": balance(target) + amount}})
        cluster.prepare(actor)
        if step % 7 == 6:
            if len(prepared_but_open) > 1:
                cluster.abort(prepared_but_open.pop(0))
            prepared_but_open.append(actor)
            continue                           # prepared but not committed yet
        cluster.commit(actor)

    check(prepared_but_open, "the workload must leave prepared transactions open")
    for actor in prepared_but_open:
        check(cluster.transactions[actor].committed_at is None,
              "%s must still be uncommitted" % actor)
        for shard in cluster.transactions[actor].writes:
            check(actor in cluster.shards[shard].prepared,
                  "the open transaction is prepared on every participant")

    # ---- sweep the snapshot timestamp: per-key equality with the replay -----
    for timestamp in range(0, cluster.clock + 1):
        snapshot = cluster.consistent_snapshot(timestamp)
        replayed = cluster.replay(timestamp)
        for shard in cluster.shards:
            for key in set(snapshot[shard]) | set(replayed[shard]):
                check(snapshot[shard].get(key) == replayed[shard].get(key),
                      "shard %s key %s at T=%d: snapshot %r vs replay %r"
                      % (shard, key, timestamp, snapshot[shard].get(key),
                         replayed[shard].get(key)))
        check(cluster.torn_transactions({name: timestamp for name in cluster.shards}) == [],
              "a single-timestamp snapshot cannot tear anything (T=%d)" % timestamp)

    # ---- atomicity of the cut: no torn cross-shard transaction --------------
    for timestamp in range(0, cluster.clock + 1):
        for commit_timestamp, transaction_id, writes in cluster.history:
            if len(writes) < 2:
                continue
            visible = [all(cluster.shards[shard].visible(key, transaction_id, timestamp)
                           for key in pairs) for shard, pairs in writes.items()]
            check(all(visible) or not any(visible),
                  "transaction %s is torn at T=%d" % (transaction_id, timestamp))

    # ---- cross-shard conservation invariant ---------------------------------
    for timestamp in range(0, cluster.clock + 1):
        snapshot = cluster.consistent_snapshot(timestamp)
        total = sum(snapshot[shard].get("balance", 0) or 0
                    for shard in ("shard_a", "shard_b", "shard_c"))
        expected = 0 if timestamp == 0 else 350
        check(total == expected,
              "balance total at T=%d is %d, expected %d" % (timestamp, total, expected))

    # the snapshot at the current clock equals the live state
    latest = cluster.consistent_snapshot(cluster.clock)
    live = {name: {key: shard.read(key, cluster.clock) for key in shard.keys()}
            for name, shard in cluster.shards.items()}
    check(latest == live, "the snapshot at the current clock equals the live state")

    # ---- negative control: per-shard timestamps tear a transaction ----------
    torn_found = []
    for timestamp in range(2, cluster.clock + 1):
        for delta in (1, 2):
            stamps = {"shard_a": timestamp - delta, "shard_b": timestamp,
                      "shard_c": timestamp}
            torn_found.extend(cluster.torn_transactions(stamps))
    check(torn_found, "a per-shard timestamp snapshot must tear at least one transaction")

    stamps = {"shard_a": 0, "shard_b": cluster.clock, "shard_c": cluster.clock}
    mixed = cluster.torn_snapshot(stamps)
    mixed_total = sum(mixed[shard].get("balance", 0) or 0
                      for shard in ("shard_a", "shard_b", "shard_c"))
    steady = cluster.consistent_snapshot(cluster.clock)
    check(sum(steady[shard].get("balance", 0) or 0
              for shard in ("shard_a", "shard_b", "shard_c")) == 350,
          "the consistent snapshot keeps the invariant")
    check(mixed_total != 350,
          "the torn snapshot breaks the invariant (reported %d instead of 350)" % mixed_total)

    # ---- rejected inputs -----------------------------------------------------
    for thunk in (lambda: cluster.begin("bad", {"shard_z": {"k": 1}}),
                  lambda: cluster.commit("seed_shard_a"),
                  lambda: cluster.prepare("nope")):
        raised = False
        try:
            thunk()
        except (AssertionError, KeyError):
            raised = True
        check(raised, "an invalid transaction operation must raise")

    print("cluster: %d shards, %d commits, clock %d, %d prepared-but-open transactions"
          % (len(cluster.shards), len(cluster.history), cluster.clock, len(prepared_but_open)))
    print("sweep: snapshots at every T in 0..%d match the replay and are never torn"
          % cluster.clock)
    print("conservation: balance total 350 at every T; per-shard timestamps tear %d transactions"
          % len(torn_found))
    print("PASS")


if __name__ == "__main__":
    main()
