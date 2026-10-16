"""Task 639 - Implement two-phase locking simulator with deadlock detection.  (Tier: Expert, Ch 14 — Databases)

A deterministic 2PL locker.  Transactions are scripts of lock requests - ("S", item) for a
shared read lock, ("X", item, value_spec) for an exclusive write lock plus the write itself
- and the simulator executes them round-robin over an explicit lock table with a wait queue
per item.  A request is granted when it is compatible (shared/shared, or the requester is
the only holder, which also covers the S -> X upgrade); otherwise the transaction waits and
its blockers are recorded.  Every transaction holds all locks until commit, i.e. the
simulator implements *strict* 2PL, so the commit order is the serial order.

When a whole round makes no progress the wait-for graph is built from the current blockers
and searched for a cycle; the youngest member of the cycle is aborted, its writes are
rolled back in reverse order (the undo log) and its locks released, after which the
survivors can proceed.  Deadlock detection is thus a real wait-for cycle search, not a
timeout.

Simplifications documented in the header: no lock escalation, no timeouts, no restarts (an
aborted transaction simply stays aborted - a restarted one would be a fresh script), and
the plans are fixed-length scripts rather than an interactive workload.
"""


class LockTable:
    """Shared/exclusive locks plus the blocker set recorded for each waiting transaction."""

    def __init__(self):
        self.mode = {}          # item -> "S" | "X"
        self.holders = {}       # item -> {transaction, ...}
        self.waits = {}         # transaction -> {blocking transactions, ...}

    def holders_of(self, item):
        return set(self.holders.get(item, ()))

    def request(self, name, item, mode):
        holders = self.holders_of(item)
        if not holders or holders == {name}:
            self._take(name, item, mode)
            return True
        if mode == "S" and self.mode.get(item) == "S":
            self._take(name, item, mode)
            return True
        self.waits.setdefault(name, set()).update(holders - {name})
        return False

    def _take(self, name, item, mode):
        holders = self.holders.setdefault(item, set())
        holders.add(name)
        self.mode[item] = "X" if (mode == "X" or self.mode.get(item) == "X") else "S"

    def release_all(self, name):
        released = []
        for item in sorted(self.holders):
            if name in self.holders[item]:
                self.holders[item].discard(name)
                released.append(item)
        for item in released:
            if not self.holders[item]:
                del self.holders[item]
                self.mode.pop(item, None)
        return released

    def snapshot(self):
        return {item: sorted(holders) for item, holders in sorted(self.holders.items())}


class TwoPhaseTransaction:
    def __init__(self, name, plan, age):
        self.name = name
        self.plan = list(plan)
        self.age = age
        self.cursor = 0
        self.status = "running"
        self.locks = []          # ("S"|"X", item) in acquisition order
        self.released = []
        self.writes = []         # (item, value) applied, for the serial oracle
        self.undo = []           # (item, previous value, existed) for rollback
        self.blocked_rounds = 0
        self.commit_round = None

    def done(self):
        return self.cursor >= len(self.plan)

    def trace(self):
        ops = [("acquire", mode, item) for mode, item in self.locks]
        ops += [("release", item) for item in self.released]
        return ops


def resolve_value(spec, data):
    if isinstance(spec, tuple):
        kind, ref, delta = spec
        if kind != "add":
            raise AssertionError("unknown value spec %r" % (spec,))
        return data.get(ref, 0) + delta
    return spec


def rollback(tx, data):
    for item, previous, existed in reversed(tx.undo):
        if existed:
            data[item] = previous
        else:
            data.pop(item, None)
    tx.undo = []


def find_cycle(graph):
    """First cycle of a wait-for graph, found by a deterministic DFS over sorted nodes."""
    def dfs(node, path):
        for nxt in sorted(graph.get(node, ())):
            if nxt in path:
                return path[path.index(nxt):]
            found = dfs(nxt, path + [nxt])
            if found:
                return found
        return None

    visited = set()
    for start in sorted(graph):
        if start in visited:
            continue
        visited.add(start)
        found = dfs(start, [start])
        if found:
            return found
        visited.update(graph.get(start, ()))
    return None


def two_phase_ok(trace):
    """A trace obeys 2PL when no lock is acquired after the first release."""
    shrinking = False
    for op in trace:
        if op[0] == "acquire":
            if shrinking:
                return False
        elif op[0] == "release":
            shrinking = True
        else:
            raise AssertionError("unknown trace op %r" % (op,))
    return True


def run_2pl(plans, initial, max_rounds=40):
    table = LockTable()
    data = dict(initial)
    txs = {name: TwoPhaseTransaction(name, plan, age)
           for age, (name, plan) in enumerate(plans.items())}
    history, deadlocks, aborted, snapshots = [], [], [], []
    rounds = 0
    for round_no in range(max_rounds):
        rounds = round_no + 1
        table.waits = {}
        progressed = False
        for name in plans:
            tx = txs[name]
            if tx.status != "running":
                continue
            if tx.done():
                tx.released = table.release_all(name)
                tx.status = "committed"
                tx.commit_round = round_no
                history.append(("commit", name, round_no))
                progressed = True
                continue
            op = tx.plan[tx.cursor]
            if op[0] == "S":
                item = op[1]
                if table.request(name, item, "S"):
                    tx.locks.append(("S", item))
                    tx.cursor += 1
                    history.append(("read", name, item, data.get(item), round_no))
                    progressed = True
                else:
                    tx.blocked_rounds += 1
            elif op[0] == "X":
                item, spec = op[1], op[2]
                if table.request(name, item, "X"):
                    tx.locks.append(("X", item))
                    value = resolve_value(spec, data)
                    tx.undo.append((item, data.get(item), item in data))
                    data[item] = value
                    tx.writes.append((item, value))
                    tx.cursor += 1
                    history.append(("write", name, item, value, round_no))
                    progressed = True
                else:
                    tx.blocked_rounds += 1
            else:
                raise AssertionError("unknown op %r" % (op,))
        snapshots.append(table.snapshot())
        if all(tx.status != "running" for tx in txs.values()):
            break
        if not progressed:
            cycle = find_cycle(table.waits)
            if cycle is None:
                break                        # nothing blocked, nothing to do
            deadlocks.append(tuple(cycle))
            victim = max(cycle, key=lambda n: txs[n].age)
            rollback(txs[victim], data)
            table.release_all(victim)
            txs[victim].status = "aborted"
            aborted.append(victim)
            history.append(("abort", victim, round_no))
    return dict(data=data, txs=txs, history=history, deadlocks=deadlocks,
                aborted=aborted, rounds=rounds, snapshots=snapshots, table=table)


def run_serially(plans, order, initial):
    """Oracle: apply the transactions one after another; only the X ops touch the data."""
    data = dict(initial)
    for name in order:
        for op in plans[name]:
            if op[0] == "X":
                value = resolve_value(op[2], data)
                data[op[1]] = value
    return data


def max_concurrent_holders(snapshots, item):
    return max((len(snap[item]) for snap in snapshots if item in snap), default=0)


def main():
    # lock compatibility matrix, exercised directly on the lock table
    table = LockTable()
    check(table.request("A", "x", "S"), "first shared lock is granted")
    check(table.request("B", "x", "S"), "shared/shared is compatible")
    check(table.snapshot() == {"x": ["A", "B"]}, "both readers hold the shared lock")
    check(not table.request("C", "x", "X"), "exclusive request blocked by shared holders")
    check(table.waits["C"] == {"A", "B"}, "the blocked reader records both holders")
    check(table.request("A", "y", "X"), "an exclusive lock on a free item is granted")
    check(not table.request("B", "y", "S"), "shared request blocked by an exclusive holder")
    check(not table.request("B", "y", "X"), "exclusive request blocked by an exclusive holder")
    check(table.release_all("A") == ["x", "y"], "release removes both of A's locks")
    check(table.request("B", "y", "X"), "the released item becomes available")

    check(find_cycle({"A": {"B"}, "B": {"A"}}) == ["A", "B"], "two-node wait cycle detected")
    check(find_cycle({"A": {"B"}, "B": {"C"}, "C": {"A"}}) == ["A", "B", "C"],
          "three-node wait cycle detected")
    check(find_cycle({"A": {"B"}, "B": set()}) is None, "a chain is not a cycle")
    check(find_cycle({}) is None, "an empty wait-for graph has no cycle")
    check(find_cycle({"A": {"B"}, "B": {"A"}, "C": {"D"}, "D": {"C"}}) == ["A", "B"],
          "the first cycle in sorted order wins")

    check(two_phase_ok([("acquire", "S", "x"), ("acquire", "X", "y"), ("release", "x"),
                        ("release", "y")]), "a growing-then-shrinking trace is 2PL")
    check(two_phase_ok([("acquire", "S", "x")]), "a trace without releases is 2PL")
    check(not two_phase_ok([("acquire", "S", "x"), ("release", "x"), ("acquire", "X", "y")]),
          "an acquire after a release violates 2PL")

    # classic two-transaction deadlock: each holds the item the other wants next
    deadlock_plans = {
        "T1": [("S", "A"), ("X", "A", ("add", "A", 1)), ("S", "B"), ("X", "B", ("add", "B", 1))],
        "T2": [("S", "B"), ("X", "B", ("add", "B", 1)), ("S", "A"), ("X", "A", ("add", "A", 1))],
    }
    initial = {"A": 10, "B": 20}
    result = run_2pl(deadlock_plans, initial)
    check(len(result["deadlocks"]) == 1, "exactly one deadlock is detected")
    check(result["deadlocks"][0] == ("T1", "T2"), "the cycle contains both transactions")
    check(result["aborted"] == ["T2"], "the youngest transaction is the victim")
    check(result["txs"]["T2"].status == "aborted" and result["txs"]["T1"].status == "committed",
          "the survivor commits, the victim stays aborted")
    check(result["data"] == run_serially(deadlock_plans, ["T1"], initial),
          "the outcome equals the serial execution of the surviving transaction")
    check(result["data"] == {"A": 11, "B": 21}, "the victim's write was rolled back")
    check(any(ev[0] == "abort" for ev in result["history"]),
          "the abort is recorded in the history")
    check(("write", "T2", "B", 21, 1) in result["history"],
          "the victim really had written before it was aborted")
    check(max_concurrent_holders(result["snapshots"], "A") == 1,
          "strict 2PL never lets two writers hold the same item")

    # shared readers are genuinely concurrent, and never deadlock
    shared_plans = {"R1": [("S", "A")], "R2": [("S", "A")]}
    shared = run_2pl(shared_plans, {"A": 1})
    check(shared["deadlocks"] == [] and shared["aborted"] == [],
          "two shared readers never deadlock")
    check(max_concurrent_holders(shared["snapshots"], "A") == 2,
          "both readers held the shared lock at the same time")
    check(shared["txs"]["R1"].blocked_rounds == 0 and shared["txs"]["R2"].blocked_rounds == 0,
          "compatible requests are never blocked")

    # a writer blocks a reader until it commits; the reader then sees the new value
    writer_plans = {"W": [("X", "A", 7), ("X", "B", 1)], "R": [("S", "A")]}
    blocked = run_2pl(writer_plans, {"A": 1, "B": 2})
    check(blocked["deadlocks"] == [] and blocked["aborted"] == [], "no deadlock here")
    check(blocked["txs"]["R"].blocked_rounds >= 2,
          "the reader waited while the writer held the exclusive lock")
    check(blocked["txs"]["R"].blocked_rounds <= blocked["rounds"],
          "the reader did not wait longer than the run lasted")
    write_at = [i for i, ev in enumerate(blocked["history"]) if ev[:2] == ("write", "W")]
    read_at = [i for i, ev in enumerate(blocked["history"]) if ev[:2] == ("read", "R")]
    check(write_at and read_at and max(write_at) < min(read_at),
          "the read is ordered after the write that it waited for")
    check(blocked["history"][read_at[0]][3] == 7, "the reader observed the committed value")
    check(blocked["txs"]["W"].writes == [("A", 7), ("B", 1)], "the writer applied both rows")

    # the simulator only ever produces 2PL traces
    for name, tx in result["txs"].items():
        check(two_phase_ok(tx.trace()), "%s violated two-phase locking" % name)
    check(all(tx.released for tx in result["txs"].values() if tx.status == "committed"),
          "every committed transaction released its locks")
    print("[639] 2PL: deadlock cycle %r detected, victim %r, %d rounds"
          % (result["deadlocks"][0], result["aborted"][0], result["rounds"]))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
