"""Task 638 - Implement serializable snapshot isolation tester.  (Tier: Expert, Ch 14 — Databases)

An SSI tester over explicit transaction schedules.  Transactions read at the snapshot they
were created with and buffer their writes, so write-write conflicts are resolved by
first-committer-wins; the commit path additionally builds the *dependency graph* of the
concurrent transactions and forbids the one structure SSI has to prevent:

    T1 --rw--> T2 --rw--> T3

"Ti --rw--> Tj" means "Tj read a key that Ti overwrote, and Ti committed after Tj's
snapshot was taken".  Two consecutive rw edges (a *dangerous structure*) are what lets a
non-serializable snapshot schedule commit, so the committing transaction that closes such
a structure is aborted.  Documented simplification of full SSI (Cahill et al.): only the
structure passing through the committing transaction is tested, edges are evaluated at
commit time, and read-only transactions - which never acquire an outgoing rw edge - are
never chosen as victim.

Invariants verified: (a) the doctors write-skew schedule commits both transactions under
plain snapshot isolation but aborts exactly one under SSI, (b) the surviving state obeys
the business invariant and equals a serial execution of the surviving bodies, (c) serial,
disjoint and read-only schedules produce zero false aborts, (d) SSI is deterministic.
"""

import itertools


class SsiTransaction:
    def __init__(self, name, start_seq):
        self.name = name
        self.start_seq = start_seq       # sequence number of both start and snapshot
        self.reads = set()
        self.writes = {}
        self.status = "running"
        self.commit_seq = None


class SsiEngine:
    def __init__(self, initial):
        self.seq = 0                                     # event clock: start/commit order
        self.versions = {k: [(0, v)] for k, v in initial.items()}
        self.txns = []
        self.rw_edges = []
        self.dangerous = []
        self.aborts = []

    def read(self, tx, key):
        if key in tx.writes:
            return tx.writes[key]
        tx.reads.add(key)
        value = None
        for seq, val in self.versions.get(key, ()):
            if seq <= tx.start_seq:
                value = val
        return value

    def state(self):
        return {key: chain[-1][1] for key, chain in sorted(self.versions.items())}

    def record_edge(self, frm, to):
        if (frm, to) not in self.rw_edges:
            self.rw_edges.append((frm, to))

    def incoming_rw(self, tj):
        """Transactions that overwrote something tj read after tj's snapshot was taken."""
        out = []
        for ti in self.txns:
            if ti is tj or ti.commit_seq is None:
                continue
            if ti.commit_seq > tj.start_seq and ti.writes.keys() & tj.reads:
                out.append(ti)
        return out

    def outgoing_rw(self, tj):
        """Concurrent transactions that read a key tj is writing right now."""
        out = []
        for tk in self.txns:
            if tk is tj:
                continue
            concurrent = tk.commit_seq is None or tk.commit_seq > tj.start_seq
            if concurrent and tj.writes.keys() & tk.reads:
                out.append(tk)
        return out

    def commit(self, tx, ssi=True):
        if tx.status != "running":
            return False
        stale = [k for k in sorted(tx.writes)
                 if self.versions.get(k, [(0, None)])[-1][0] > tx.start_seq]
        if stale:
            tx.status = "aborted"             # first-committer-wins
            self.aborts.append(tx.name)
            return False
        if ssi:
            incoming = self.incoming_rw(tx)
            outgoing = self.outgoing_rw(tx)
            for ti in incoming:
                self.record_edge(ti.name, tx.name)
            for tk in outgoing:
                self.record_edge(tx.name, tk.name)
            if incoming and outgoing:
                self.dangerous.append((incoming[0].name, tx.name, outgoing[0].name))
                tx.status = "aborted"
                self.aborts.append(tx.name)
                return False
        self.seq += 1
        tx.commit_seq = self.seq
        for key in sorted(tx.writes):
            self.versions.setdefault(key, []).append((self.seq, tx.writes[key]))
        tx.status = "committed"
        return True

    def run(self, bodies, schedule, ssi=True):
        seen, gens, trace = {}, {}, []
        for kind, name in schedule:
            self.seq += 1
            if name not in seen:
                tx = SsiTransaction(name, self.seq)
                seen[name] = tx
                self.txns.append(tx)
                gens[name] = bodies[name](
                    lambda k, t=tx: self.read(t, k),
                    lambda k, v, t=tx: t.writes.__setitem__(k, v))
            tx = seen[name]
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
                        next(gens[name])
                    except StopIteration:
                        pass
                    else:
                        raise AssertionError("%s committed with body work pending" % name)
                trace.append((kind, name, self.commit(tx, ssi)))
            elif kind == "abort":
                tx.status = "aborted"
                self.aborts.append(name)
                trace.append((kind, name, True))
            else:
                raise AssertionError("unknown event %r" % (kind,))
        return seen, trace


def doctor_body(name, other):
    """Leaves the shift only when two doctors were on call when it looked."""
    def body(read, write):
        on_call = sum(1 for d in (name, other) if read(d) == "on")
        yield "read"
        if on_call >= 2:
            write(name, "off")
    return body


def reader_writer(read_key, write_key, value):
    def body(read, write):
        read(read_key)
        yield "read"
        write(write_key, value)
    return body


def swap_body(mine, theirs):
    """Updates the row the other writer read: builds a pair of rw antidependencies."""
    def body(read, write):
        value = read(theirs)
        yield "read"
        write(mine, value + 1)
    return body


def read_only_body(*keys):
    def body(read, write):
        for key in keys:
            read(key)
        yield "read"
        read(keys[0])
    return body


def serial_run(bodies, order, initial):
    state = dict(initial)
    for name in order:
        gen = bodies[name](lambda k: state.get(k), lambda k, v: state.__setitem__(k, v))
        for _ in gen:
            pass
    return state


def is_serializable(bodies, names, outcome, initial):
    return any(serial_run(bodies, order, initial) == outcome
               for order in itertools.permutations(sorted(names)))


def doctors_schedule():
    return [("step", "T1"), ("step", "T2"), ("step", "T1"), ("step", "T2"),
            ("commit", "T1"), ("commit", "T2")]


def run_doctors(ssi):
    initial = {"alice": "on", "bob": "on"}
    bodies = {"T1": doctor_body("alice", "bob"), "T2": doctor_body("bob", "alice")}
    engine = SsiEngine(initial)
    seen, trace = engine.run(bodies, doctors_schedule(), ssi=ssi)
    return engine, seen, bodies, initial


def main():
    engine_si, seen_si, _, _ = run_doctors(ssi=False)
    check(all(tx.status == "committed" for tx in seen_si.values()),
          "snapshot isolation alone commits both doctors")
    si_outcome = engine_si.state()
    check(si_outcome == {"alice": "off", "bob": "off"},
          "plain snapshot isolation breaks the invariant: %r" % (si_outcome,))

    engine, seen, bodies, initial = run_doctors(ssi=True)
    check(seen["T1"].status == "committed", "the first committer survives SSI")
    check(seen["T2"].status == "aborted",
          "SSI aborts the transaction that closes the dangerous structure")
    check(engine.aborts == ["T2"], "exactly one abort: %r" % (engine.aborts,))
    check(engine.dangerous == [("T1", "T2", "T1")],
          "detected structure T1 ->rw T2 ->rw T1: %r" % (engine.dangerous,))
    check(sorted(engine.rw_edges) == [("T1", "T2"), ("T2", "T1")],
          "both rw antidependencies recorded once: %r" % (engine.rw_edges,))
    outcome = engine.state()
    check(outcome == {"alice": "off", "bob": "on"},
          "the surviving state keeps one doctor on call: %r" % (outcome,))
    check(sum(1 for v in outcome.values() if v == "on") == 1, "business invariant holds")
    check(outcome == serial_run(bodies, ["T1"], initial),
          "the SSI outcome equals the serial execution of the surviving transaction")
    check(is_serializable(bodies, ["T1"], outcome, initial), "the SSI outcome is serializable")
    check(engine.versions["bob"] == [(0, "on")], "the aborted transaction installed nothing")
    check(all(tx.commit_seq is None for tx in seen.values() if tx.status == "aborted"),
          "an aborted transaction never receives a commit sequence number")

    again, seen_again, _, _ = run_doctors(ssi=True)
    check([t.status for t in seen_again.values()] == [t.status for t in seen.values()],
          "SSI decisions are deterministic")
    check(again.aborts == engine.aborts and again.dangerous == engine.dangerous,
          "identical aborts and dangerous structures on a second run")

    # a pure rw cycle between two writers: W1 writes what W2 read and vice versa
    bodies_swap = {"W1": swap_body("b", "a"), "W2": swap_body("a", "b")}
    schedule_swap = [("step", "W1"), ("step", "W2"), ("step", "W1"), ("step", "W2"),
                     ("commit", "W1"), ("commit", "W2")]
    engine_swap = SsiEngine({"a": 0, "b": 0})
    seen_swap, _ = engine_swap.run(bodies_swap, schedule_swap)
    check(seen_swap["W1"].status == "committed" and seen_swap["W2"].status == "aborted",
          "the second transaction of an rw cycle is the victim")
    check(engine_swap.state() == {"a": 0, "b": 1}
          and engine_swap.dangerous == [("W1", "W2", "W1")],
          "the victim's buffered writes are discarded: %r" % (engine_swap.state(),))

    # disjoint concurrent writers: no edges, no false abort
    bodies_disjoint = {"D1": reader_writer("a", "x", 1), "D2": reader_writer("b", "y", 2)}
    schedule_disjoint = [("step", "D1"), ("step", "D2"), ("step", "D1"), ("step", "D2"),
                         ("commit", "D1"), ("commit", "D2")]
    engine_disjoint = SsiEngine({"a": 0, "b": 0, "x": 0, "y": 0})
    seen_disjoint, _ = engine_disjoint.run(bodies_disjoint, schedule_disjoint)
    check(engine_disjoint.aborts == [] and not engine_disjoint.dangerous,
          "no conflicts means no abort and no dangerous structure")
    check(engine_disjoint.state()["x"] == 1 and engine_disjoint.state()["y"] == 2,
          "both disjoint writers installed their rows")

    # fully serial chain: every reader saw the previous commit, so no rw edge exists
    bodies_chain = {"S1": reader_writer("x", "y", 1), "S2": reader_writer("y", "z", 1),
                    "S3": reader_writer("z", "w", 1)}
    schedule_chain = [("step", "S1"), ("step", "S1"), ("commit", "S1"),
                      ("step", "S2"), ("step", "S2"), ("commit", "S2"),
                      ("step", "S3"), ("step", "S3"), ("commit", "S3")]
    engine_chain = SsiEngine({"x": 0, "y": 0, "z": 0, "w": 0})
    seen_chain, _ = engine_chain.run(bodies_chain, schedule_chain)
    check(all(tx.status == "committed" for tx in seen_chain.values()),
          "a serial chain must not trigger a false positive")
    check(engine_chain.rw_edges == [],
          "no antidependency across commits: %r" % (engine_chain.rw_edges,))
    check(engine_chain.state() == {"x": 0, "y": 1, "z": 1, "w": 1},
          "the serial chain applied every write in order")

    # read-only transaction: it can only be the target of rw edges, never the source
    bodies_ro = {"R1": reader_writer("p", "p", 1), "R2": reader_writer("q", "q", 1),
                 "RO": read_only_body("p", "q")}
    schedule_ro = [("step", "R1"), ("step", "R2"), ("step", "RO"),
                   ("step", "R1"), ("step", "R2"), ("step", "RO"),
                   ("commit", "R1"), ("commit", "R2"), ("commit", "RO")]
    engine_ro = SsiEngine({"p": 0, "q": 0})
    seen_ro, _ = engine_ro.run(bodies_ro, schedule_ro)
    check(seen_ro["RO"].status == "committed", "a read-only transaction is never the victim")
    check(seen_ro["R1"].status == "committed" and seen_ro["R2"].status == "committed",
          "both writers commit: no dangerous structure without an outgoing edge")
    check(sorted(engine_ro.rw_edges) == [("R1", "RO"), ("R2", "RO")],
          "rw edges point away from the writers: %r" % (engine_ro.rw_edges,))
    check(seen_ro["RO"].reads == {"p", "q"}, "the read-only transaction recorded its read set")
    print("[638] SSI: 1 abort on write skew; 0 aborts on disjoint, serial and read-only schedules")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
