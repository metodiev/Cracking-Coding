"""Task 637 - Implement SNAPSHOT isolation anomalies demonstrator (write skew).  (Tier: Expert, Ch 14 — Databases)

A deterministic MVCC snapshot-isolation engine driven by an *explicit* schedule of
transaction steps - no threads, no timing, no randomness.  A transaction is a generator
body that reads through the snapshot it was created with, buffers its writes and installs
all of them under a single commit timestamp, so every row of one transaction becomes
visible atomically.  Commit uses first-committer-wins: a transaction aborts when some row
it wrote was updated by somebody else after its snapshot was taken (write-write conflict).
That is precisely what snapshot isolation promises - and nothing more.

The demonstrator *proves* the anomaly instead of asserting it.  The classic doctors
schedule commits both transactions, the final state breaks the business invariant ("at
least one doctor stays on call"), and - the decisive property - that state is equal to no
serial execution of the very same transaction bodies, hence the history is not
serializable.  A lost-update schedule shows first-committer-wins aborting the second
writer, and a repeatable-read probe shows a snapshot stays stable while another
transaction commits.

Invariants verified: (a) a committed transaction never overwrites a row changed after its
snapshot, (b) reads only ever see versions with commit_ts <= snapshot, (c) the schedule
whose outcome matches no serial order is exactly the write-skew schedule.
"""

import itertools


class MVCCStore:
    """Version store: `history[key]` is a list of (commit_ts, value) in commit order."""

    def __init__(self):
        self.clock = 0
        self.history = {}

    def read_at(self, snapshot, key):
        value = None
        for ts, val in self.history.get(key, ()):
            if ts <= snapshot:
                value = val
        return value

    def latest_ts(self, key):
        chain = self.history.get(key, ())
        return chain[-1][0] if chain else -1

    def install(self, writes):
        """Write phase of a committing transaction: one timestamp for all of its rows."""
        self.clock += 1
        for key in sorted(writes):
            self.history.setdefault(key, []).append((self.clock, writes[key]))
        return self.clock

    def snapshot_state(self, keys):
        return {key: self.read_at(self.clock, key) for key in keys}


class Tx:
    def __init__(self, name, store):
        self.name = name
        self.store = store
        self.snapshot = store.clock          # commit_ts visible to this transaction
        self.reads = {}
        self.writes = {}
        self.conflicts = []
        self.status = "running"
        self.commit_ts = None

    def read(self, key):
        if key in self.writes:               # read-your-own-writes
            value = self.writes[key]
        else:
            value = self.store.read_at(self.snapshot, key)
        self.reads[key] = value
        return value

    def write(self, key, value):
        self.writes[key] = value             # buffered until commit

    def commit(self):
        self.conflicts = [k for k in sorted(self.writes)
                          if self.store.latest_ts(k) > self.snapshot]
        if self.conflicts:
            self.status = "aborted"          # first-committer-wins
            return False
        self.commit_ts = self.store.install(self.writes)
        self.status = "committed"
        return True


def seed_store(initial):
    store = MVCCStore()
    for key in sorted(initial):
        store.history[key] = [(0, initial[key])]
    return store


def run_schedule(store, bodies, schedule):
    """Execute an explicit event list: ("step"|"commit"|"abort", txn_name).

    Transactions are instantiated lazily at their first event so that a schedule where one
    transaction starts only after another has committed really does get the later snapshot.
    """
    txs, gens = {}, {}

    def tx_of(name):
        if name not in txs:
            txs[name] = Tx(name, store)
            gens[name] = bodies[name](txs[name].read, txs[name].write)
        return txs[name]

    trace = []
    for kind, name in schedule:
        tx = tx_of(name)
        if kind == "step":
            if tx.status != "running":
                trace.append((kind, name, False))
                continue
            try:
                next(gens[name])
                trace.append((kind, name, True))
            except StopIteration:
                trace.append((kind, name, False))
        elif kind == "commit":
            if tx.status == "running":
                try:
                    next(gens[name])            # a commit may not truncate the body
                except StopIteration:
                    pass
                else:
                    raise AssertionError("%s committed with body work pending" % name)
            trace.append((kind, name, tx.commit() if tx.status == "running" else False))
        elif kind == "abort":
            tx.status = "aborted"
            trace.append((kind, name, True))
        else:
            raise AssertionError("unknown event %r" % (kind,))
    return txs, trace


def committed_names(txs):
    return [name for name in txs if txs[name].status == "committed"]


def run_serially(bodies, order, initial):
    """Oracle: run the *same* bodies one at a time against a plain key/value state."""
    state = dict(initial)
    for name in order:
        gen = bodies[name](lambda k: state.get(k), lambda k, v: state.__setitem__(k, v))
        for _ in gen:
            pass
    return state


def is_serializable(bodies, names, outcome, initial):
    return any(run_serially(bodies, order, initial) == outcome
               for order in itertools.permutations(sorted(names)))


def doctor_body(name, other):
    """Curried body: leaves the shift only when two doctors were on call when it looked."""
    def body(read, write):
        on_call = sum(1 for d in (name, other) if read(d) == "on")
        yield "read"                         # both doctors read before either writes
        if on_call >= 2:
            write(name, "off")
    return body


def increment_body(read, write):
    value = read("counter")
    yield "read"
    write("counter", value + 1)


def probe_reader_body(observed):
    def body(read, write):
        observed.append(read("counter"))
        yield "read"
        observed.append(read("counter"))     # second read must repeat the first
    return body


def write_skew_demo():
    initial = {"alice": "on", "bob": "on"}
    bodies = {"T1": doctor_body("alice", "bob"), "T2": doctor_body("bob", "alice")}
    schedule = [("step", "T1"), ("step", "T2"), ("step", "T1"), ("step", "T2"),
                ("commit", "T1"), ("commit", "T2")]
    store = seed_store(initial)
    txs, _ = run_schedule(store, bodies, schedule)
    outcome = store.snapshot_state(sorted(initial))
    serial = [run_serially(bodies, order, initial)
              for order in itertools.permutations(sorted(bodies))]
    return dict(store=store, txs=txs, outcome=outcome, serial=serial, initial=initial,
                bodies=bodies)


def lost_update_demo():
    bodies = {"U1": increment_body, "U2": increment_body}
    overlapping = [("step", "U1"), ("step", "U2"), ("step", "U1"), ("step", "U2"),
                   ("commit", "U1"), ("commit", "U2")]
    store = seed_store({"counter": 0})
    txs, _ = run_schedule(store, bodies, overlapping)
    sequential = [("step", "U1"), ("step", "U1"), ("commit", "U1"),
                  ("step", "U2"), ("step", "U2"), ("commit", "U2")]
    store2 = seed_store({"counter": 0})
    txs2, _ = run_schedule(store2, bodies, sequential)
    return dict(overlapping=(store, txs), sequential=(store2, txs2), bodies=bodies)


def repeatable_read_demo():
    observed = []
    bodies = {"R": probe_reader_body(observed), "W": increment_body}
    schedule = [("step", "R"), ("step", "W"), ("commit", "W"), ("step", "R"), ("commit", "R")]
    store = seed_store({"counter": 5})
    txs, _ = run_schedule(store, bodies, schedule)
    return dict(store=store, txs=txs, observed=observed, bodies=bodies)


def anomaly_report():
    """Run every demo and classify what the engine produced."""
    skew = write_skew_demo()
    lost = lost_update_demo()
    repeat = repeatable_read_demo()
    on_call = sum(1 for v in skew["outcome"].values() if v == "on")
    store, txs = lost["overlapping"]
    return {
        "write_skew_demonstrated": on_call == 0,
        "non_serializable": not is_serializable(skew["bodies"], ["T1", "T2"],
                                                skew["outcome"], skew["initial"]),
        "invariant_holds_in_every_serial_order":
            all(sum(1 for v in st.values() if v == "on") == 1 for st in skew["serial"]),
        "lost_update_prevented": txs["U2"].status == "aborted",
        "repeatable_read": repeat["observed"] == [5, 5],
    }


def main():
    skew = write_skew_demo()
    txs, outcome = skew["txs"], skew["outcome"]

    check(txs["T1"].status == "committed" and txs["T2"].status == "committed",
          "both disjoint writers must commit under snapshot isolation")
    check(outcome == {"alice": "off", "bob": "off"},
          "snapshot isolation let both doctors go off call: %r" % (outcome,))
    check(not is_serializable(skew["bodies"], ["T1", "T2"], outcome, skew["initial"]),
          "the write-skew outcome must match no serial order")
    check(all(st != outcome for st in skew["serial"]),
          "every serial order must differ from the snapshot-isolation outcome")
    check(all(sum(1 for v in st.values() if v == "on") == 1 for st in skew["serial"]),
          "the business invariant holds in every serial order")
    check(sum(1 for v in outcome.values() if v == "on") == 0,
          "the business invariant is violated by the concurrent schedule")

    store = skew["store"]
    check(len(store.history["alice"]) == 2 and len(store.history["bob"]) == 2,
          "each committed write appended exactly one version")
    check(store.history["alice"][-1][0] == txs["T1"].commit_ts
          and store.history["bob"][-1][0] == txs["T2"].commit_ts,
          "each transaction installed all of its rows at its own commit timestamp")
    check(txs["T1"].commit_ts != txs["T2"].commit_ts,
          "commit timestamps are unique")
    check(txs["T1"].snapshot == txs["T2"].snapshot == 0,
          "both transactions read the same snapshot")
    check(txs["T1"].reads == {"alice": "on", "bob": "on"},
          "T1 observed the pre-anomaly state, not the concurrent write")

    lost = lost_update_demo()
    store_lu, txs_lu = lost["overlapping"]
    check(txs_lu["U1"].status == "committed", "the first writer commits")
    check(txs_lu["U2"].status == "aborted" and txs_lu["U2"].conflicts == ["counter"],
          "the second overlapping writer is aborted by first-committer-wins")
    check(store_lu.snapshot_state(["counter"]) == {"counter": 1},
          "only the committed increment is visible")
    check(len(store_lu.history["counter"]) == 2,
          "the aborted transaction installed no version")
    check(run_serially(lost["bodies"], ["U1", "U2"], {"counter": 0}) == {"counter": 2},
          "the same bodies run serially produce two increments")
    store_seq, txs_seq = lost["sequential"]
    check(txs_seq["U1"].status == "committed" and txs_seq["U2"].status == "committed",
          "a non-overlapping schedule must not abort anybody")
    check(txs_seq["U2"].snapshot == 1 and store_seq.snapshot_state(["counter"]) == {"counter": 2},
          "the late transaction took a later snapshot and saw the first increment")

    repeat = repeatable_read_demo()
    check(repeat["observed"] == [5, 5],
          "both reads inside one snapshot must return the same version: %r" % (repeat["observed"],))
    check(repeat["txs"]["R"].snapshot == 0 and repeat["txs"]["W"].commit_ts == 1,
          "the reader's snapshot predates the writer's commit")
    check(repeat["store"].read_at(repeat["store"].clock, "counter") == 6,
          "a transaction starting later sees the committed update")

    report = anomaly_report()
    for name, ok in sorted(report.items()):
        check(ok, "anomaly report entry failed: %s" % name)
    print("[637] write skew: committed both, invariant broken, non-serializable; "
          "lost update aborted; repeatable read holds")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
