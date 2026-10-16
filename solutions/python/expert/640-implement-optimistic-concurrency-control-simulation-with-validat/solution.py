"""Task 640 - Implement optimistic concurrency control simulation with validation phase.  (Tier: Expert, Ch 14 — Databases)

A deterministic Kung-Robinson OCC engine with the three canonical phases driven by an
explicit schedule:

  * read phase    - the body runs up to its first `yield`, reading the committed state and
                    recording the read set; every write is buffered in the transaction.
  * validation    - the read set and the buffered write set are compared with every
                    transaction validated before this one.  A transaction is valid when all
                    earlier validators either finished (validation + write phase) before it
                    started reading, or touch none of the rows it read or wrote.
  * write phase   - the buffered writes become visible atomically and the transaction is
                    appended to the validation list.

Because validation order is the commit order, that order *is* the equivalent serial order,
which the checks exploit: the final state must equal a plain serial execution of the
surviving bodies.  Documented simplification: the conservative validation rule above is
stricter than the paper's condition (b) - it never aborts less, it may abort a read-only
transaction that overlaps a concurrent writer, and it does not implement transaction
restart (the retry is simply scheduled as a new transaction).
"""


class OccTransaction:
    def __init__(self, name, start_time):
        self.name = name
        self.start = start_time        # when the read phase began
        self.read_end = None
        self.end = None                # set when validation succeeded
        self.reads = {}
        self.writes = {}
        self.status = "running"
        self.conflict_with = []


class OccEngine:
    def __init__(self, initial):
        self.data = dict(initial)
        self.time = 0
        self.validated = []            # validation order == the equivalent serial order
        self.aborts = []
        self.trace = []

    def reader(self, tx):
        def read(key):
            value = tx.writes.get(key, self.data.get(key))
            tx.reads[key] = value
            return value
        return read

    def writer(self, tx):
        def write(key, value):
            tx.writes[key] = value
        return write

    def validate(self, tj):
        for ti in self.validated:
            if ti.end <= tj.start:
                continue               # ti was completely done before tj began reading
            if set(ti.writes) & (set(tj.reads) | set(tj.writes)):
                tj.conflict_with.append(ti.name)
        return not tj.conflict_with

    def run(self, bodies, schedule):
        """schedule events: ("read", name) | ("validate", name)."""
        txs, gens = {}, {}
        for kind, name in schedule:
            self.time += 1
            if name not in txs:
                tx = OccTransaction(name, self.time)
                txs[name] = tx
                gens[name] = bodies[name](self.reader(tx), self.writer(tx))
            tx = txs[name]
            if tx.status != "running":
                continue
            if kind == "read":
                try:
                    next(gens[name])       # stop at the first yield: end of read phase
                except StopIteration:
                    pass
                tx.read_end = self.time
                self.trace.append(("read", name, self.time))
            elif kind == "validate":
                if self.validate(tx):
                    for _ in gens[name]:   # write phase: drain the remainder of the body
                        pass
                    for key in sorted(tx.writes):
                        self.data[key] = tx.writes[key]
                    tx.end = self.time
                    tx.status = "committed"
                    self.validated.append(tx)
                    self.trace.append(("commit", name, self.time))
                else:
                    tx.status = "aborted"
                    self.aborts.append(name)
                    self.trace.append(("abort", name, self.time))
            else:
                raise AssertionError("unknown event %r" % (kind,))
        return txs


def serial_run(bodies, order, initial):
    """Oracle: run the same bodies one at a time against a plain key/value state."""
    state = dict(initial)
    for name in order:
        gen = bodies[name](lambda k: state.get(k), lambda k, v: state.__setitem__(k, v))
        for _ in gen:
            pass
    return state


def increment_body(read, write):
    value = read("x")
    yield "read"
    write("x", value + 1)


def doctor_body(name, other):
    def body(read, write):
        on_call = sum(1 for d in (name, other) if read(d) == "on")
        yield "read"
        if on_call >= 2:
            write(name, "off")
    return body


def pair_body(read_key, write_key, value):
    def body(read, write):
        read(read_key)
        yield "read"
        write(write_key, value)
    return body


def lost_update_scenario():
    bodies = {"T1": increment_body, "T2": increment_body, "T3": increment_body}
    schedule = [("read", "T1"), ("read", "T2"), ("validate", "T1"), ("validate", "T2"),
                ("read", "T3"), ("validate", "T3")]
    engine = OccEngine({"x": 0})
    txs = engine.run(bodies, schedule)
    return engine, txs, bodies


def write_skew_scenario():
    bodies = {"A": doctor_body("alice", "bob"), "B": doctor_body("bob", "alice")}
    schedule = [("read", "A"), ("read", "B"), ("validate", "A"), ("validate", "B")]
    engine = OccEngine({"alice": "on", "bob": "on"})
    txs = engine.run(bodies, schedule)
    return engine, txs, bodies


def disjoint_scenario():
    bodies = {"D1": pair_body("a", "x", 1), "D2": pair_body("b", "y", 2)}
    schedule = [("read", "D1"), ("read", "D2"), ("validate", "D1"), ("validate", "D2")]
    engine = OccEngine({"a": 0, "b": 0, "x": 0, "y": 0})
    txs = engine.run(bodies, schedule)
    return engine, txs, bodies


def serial_scenario():
    bodies = {"S1": increment_body, "S2": increment_body}
    schedule = [("read", "S1"), ("validate", "S1"), ("read", "S2"), ("validate", "S2")]
    engine = OccEngine({"x": 0})
    txs = engine.run(bodies, schedule)
    return engine, txs, bodies


def main():
    # lost update: the overlapping second validator conflicts with the first one
    engine, txs, bodies = lost_update_scenario()
    check(txs["T2"].status == "aborted", "the overlapping writer is aborted")
    check(engine.aborts == ["T2"], "exactly one abort: %r" % (engine.aborts,))
    check(txs["T2"].conflict_with == ["T1"], "the conflict is attributed to T1")
    check(txs["T1"].status == "committed" and txs["T3"].status == "committed",
          "the first writer and the retry both commit")
    check([tx.name for tx in engine.validated] == ["T1", "T3"],
          "validation order is the commit order")
    check(engine.data == {"x": 2}, "no update was lost: %r" % (engine.data,))
    check(engine.data == serial_run(bodies, ["T1", "T3"], {"x": 0}),
          "the result equals the serial execution of the surviving transactions")
    check(txs["T1"].writes == {"x": 1} and txs["T3"].writes == {"x": 2},
          "the retry read the state produced by the first writer")
    check(txs["T1"].end <= txs["T3"].start,
          "the retry started after the first writer had finished")

    # write skew: disjoint write sets, so snapshot isolation would commit both
    engine_ws, txs_ws, bodies_ws = write_skew_scenario()
    t1, t2 = txs_ws["A"], txs_ws["B"]
    check(not (set(t1.writes) & set(t2.writes)),
          "the two doctors write different rows: no write-write conflict")
    check(set(t1.writes) & set(t2.reads) != set(),
          "the read-write conflict is what OCC detects")
    check(txs_ws["B"].status == "aborted" and t1.status == "committed",
          "OCC aborts the write-skew partner")
    check(engine_ws.data == {"alice": "off", "bob": "on"},
          "the invariant survives: %r" % (engine_ws.data,))
    check(sum(1 for v in engine_ws.data.values() if v == "on") == 1, "one doctor stays on call")
    check(engine_ws.data == serial_run(bodies_ws, ["A"], {"alice": "on", "bob": "on"}),
          "the OCC outcome is serializable")

    # disjoint concurrent transactions: no false aborts
    engine_d, txs_d, bodies_d = disjoint_scenario()
    check(engine_d.aborts == [], "disjoint transactions never conflict")
    check(engine_d.data == {"a": 0, "b": 0, "x": 1, "y": 2},
          "both disjoint writers installed their rows")
    check(engine_d.data == serial_run(bodies_d, ["D1", "D2"], {"a": 0, "b": 0, "x": 0, "y": 0}),
          "disjoint execution matches the serial oracle")
    check(len(engine_d.validated) == 2, "both disjoint transactions validated")

    # non-overlapping transactions on the same row: serialized, so allowed
    engine_s, txs_s, bodies_s = serial_scenario()
    check(engine_s.aborts == [], "sequential transactions are never aborted")
    check(txs_s["S1"].end <= txs_s["S2"].start,
          "S2 began reading after S1 finished its write phase")
    check(engine_s.data == {"x": 2}, "two serial increments: %r" % (engine_s.data,))
    check(engine_s.data == serial_run(bodies_s, ["S1", "S2"], {"x": 0}),
          "serial execution matches the oracle")
    check([tx.end for tx in engine_s.validated] == sorted(tx.end for tx in engine_s.validated),
          "validation timestamps are strictly increasing")

    # a read-only transaction that touches nothing the writer touched commits
    ro_bodies = {"RO": lambda read, write: (read("z"), (yield "read"), read("z"))[2],
                 "W": pair_body("k", "k", 5)}
    ro_schedule = [("read", "RO"), ("read", "W"), ("validate", "W"), ("validate", "RO")]
    engine_ro = OccEngine({"k": 0, "z": 9})
    txs_ro = engine_ro.run(ro_bodies, ro_schedule)
    check(txs_ro["RO"].status == "committed", "a non-overlapping read-only transaction commits")
    check(txs_ro["W"].status == "committed" and engine_ro.data["k"] == 5,
          "the writer installed its row")
    check(txs_ro["RO"].writes == {}, "the read-only transaction buffered no writes")

    # counters: validated + aborted == every transaction that reached validation
    for eng, tx_map in ((engine, txs), (engine_ws, txs_ws), (engine_d, txs_d), (engine_s, txs_s)):
        check(len(eng.validated) + len(eng.aborts) == len(tx_map),
              "every transaction either validated or aborted")
    check(engine.trace == [("read", "T1", 1), ("read", "T2", 2), ("commit", "T1", 3),
                           ("abort", "T2", 4), ("read", "T3", 5), ("commit", "T3", 6)],
          "the event trace records every phase in order: %r" % (engine.trace,))
    print("[640] OCC: %d commit / %d abort on write skew; lost update prevented; "
          "validation order is the serial order"
          % (len(engine_ws.validated), len(engine_ws.aborts)))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
