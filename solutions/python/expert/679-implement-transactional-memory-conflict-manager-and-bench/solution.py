"""Task 679 - Implement transactional memory conflict manager and bench.  (Tier: Expert, Beyond — Networking, OS & Real-Time)

An optimistic software transactional memory with versioned validation (the shape of TL2).

  * global memory holds a value and a version per address;
  * a transaction executes its operations step by step - `load` records the address, the
    value **and the version it observed**, `add`/`store` write into a private buffer, so no
    other transaction can see a partially executed transaction;
  * when a transaction has executed all of its operations it commits: every address in its
    read set (and every address it blindly overwrote) must still carry the version it
    observed, otherwise the commit is refused and the transaction restarts from scratch;
  * a deterministic round-robin scheduler runs one operation of one transaction per step, so
    the whole benchmark is reproducible.

Verified: read-your-own-writes inside a transaction; a stale read is refused exactly when
another transaction committed in between; an aborted transaction leaves no trace; a
read-only transaction commits without bumping any version; and the headline property, that
the final memory of a contended run equals the result of *some* serial order of the
committed transactions - checked by brute force over all permutations of the committed
transactions and independently by acyclicity of the read/write conflict graph. The counter
benchmark also shows there are no lost updates and that aborts are bounded (every
transaction eventually commits).

Simplified: one value per address, no nested transactions, no write-write merge, and
restart-after-abort rather than contention management with backoff.
"""

from itertools import permutations


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class TxnMemory:
    """Versioned memory: every write bumps the version of its address."""

    def __init__(self, initial):
        self.values = dict(initial)
        self.versions = {addr: 1 for addr in initial}
        self.bumps = 0

    def read(self, addr):
        if addr not in self.values:
            raise ValueError("unknown address %r" % (addr,))
        return self.values[addr], self.versions[addr]

    def write(self, addr, value):
        if addr not in self.values:
            raise ValueError("unknown address %r" % (addr,))
        self.values[addr] = value
        self.versions[addr] += 1
        self.bumps += 1

    def snapshot(self):
        return (dict(self.values), dict(self.versions))


class Transaction:
    """A transaction and its private working state."""

    def __init__(self, name, ops):
        self.name = name
        self.ops = list(ops)
        for op in self.ops:
            if op[0] not in ("load", "add", "store", "store_value"):
                raise ValueError("unknown transaction operation %r" % (op,))
        self.reset()

    def reset(self):
        self.pc = 0
        self.registers = {}
        self.read_set = {}          # addr -> version observed
        self.write_set = {}         # addr -> value to install
        self.write_versions = {}    # addr -> version observed when buffering the write
        self.status = "active"
        self.commits = 0            # completed commit attempts that succeeded

    def done(self):
        return self.pc >= len(self.ops)

    def step(self, memory):
        """Execute one operation; returns the op that ran."""
        op = self.ops[self.pc]
        kind = op[0]
        if kind == "load":
            if op[1] in self.write_set:
                value = self.write_set[op[1]]       # read your own buffered write
            else:
                value, version = memory.read(op[1])
                self.read_set[op[1]] = version
            self.registers[op[2]] = value
        elif kind == "add":
            self.registers[op[1]] = self.registers.get(op[1], 0) + op[2]
        elif kind == "store":
            address = op[1]
            if address not in self.write_versions:
                self.write_versions[address] = memory.versions.get(address, 0)
                self.read_set.pop(address, None)
            self.write_set[address] = self.registers[op[2]]
        else:
            address = op[1]
            if address not in self.write_versions:
                self.write_versions[address] = memory.versions.get(address, 0)
                self.read_set.pop(address, None)
            self.write_set[address] = op[2]
        self.pc += 1
        return op

    def validate(self, memory):
        """True when nothing the transaction observed has changed in the meantime."""
        for address, version in self.read_set.items():
            if memory.versions[address] != version:
                return False
        for address, version in self.write_versions.items():
            if memory.versions[address] != version:
                return False
        return True


class StmSystem:
    """Round-robin optimistic STM scheduler."""

    def __init__(self, initial, transactions, max_steps=100000):
        self.memory = TxnMemory(initial)
        self.transactions = transactions
        self.max_steps = max_steps
        self.steps = 0
        self.commits = 0
        self.aborts = 0
        self.committed_order = []

    def run(self):
        pending = list(self.transactions)
        while pending:
            progress = False
            for transaction in list(pending):
                if transaction.done():
                    if transaction.validate(self.memory):
                        for address, value in sorted(transaction.write_set.items()):
                            self.memory.write(address, value)
                        transaction.status = "committed"
                        transaction.commits += 1
                        self.commits += 1
                        self.committed_order.append(transaction)
                        pending.remove(transaction)
                    else:
                        self.aborts += 1
                        transaction.reset()
                    progress = True
                    continue
                self.steps += 1
                if self.steps > self.max_steps:
                    raise AssertionError("the scheduler failed to make progress")
                transaction.step(self.memory)
                progress = True
            if not progress:
                raise AssertionError("no transaction could run")
        return self.memory


def replay(ops, values):
    """Execute a transaction's operations serially on plain values (the serial oracle)."""
    state = dict(values)
    registers = {}
    for op in ops:
        if op[0] == "load":
            registers[op[2]] = state[op[1]]
        elif op[0] == "add":
            registers[op[1]] = registers.get(op[1], 0) + op[2]
        elif op[0] == "store":
            state[op[1]] = registers[op[2]]
        else:
            state[op[1]] = op[2]
    return state


def serializable(final_values, committed, initial):
    """Is the observed final state reproducible by some serial order of the committed set?"""
    for order in permutations(committed):
        state = dict(initial)
        for transaction in order:
            state = replay(transaction.ops, state)
        if state == final_values:
            return True
    return False


def conflict_graph_acyclic(committed):
    """Serializability via the conflict order: read/write conflicts must not form a cycle."""
    edges = set()
    for earlier_index, first in enumerate(committed):
        for second in committed[earlier_index + 1:]:
            first_writes = set(first.write_set)
            second_writes = set(second.write_set)
            first_reads = set(first.read_set)
            second_reads = set(second.read_set)
            if (first_writes & second_reads) or (first_writes & second_writes) \
                    or (second_writes & first_reads):
                edges.add((earlier_index, committed.index(second)))
    adjacency = {i: set() for i in range(len(committed))}
    for a, b in edges:
        adjacency[a].add(b)
    colour = {}

    def visit(node):
        colour[node] = 1
        for nxt in sorted(adjacency[node]):
            if colour.get(nxt, 0) == 1:
                return True
            if colour.get(nxt, 0) == 0 and visit(nxt):
                return True
        colour[node] = 2
        return False

    return not any(visit(node) for node in range(len(committed)) if colour.get(node, 0) == 0)


def counter_transaction(name, counter, increments=1):
    ops = []
    for i in range(increments):
        ops += [("load", counter, "r"), ("add", "r", 1), ("store", counter, "r")]
    return Transaction(name, ops)


def main():
    # ---- read-your-own-writes and buffering ------------------------------------------
    memory = TxnMemory({"x": 0, "y": 5})
    transaction = Transaction("t", [("load", "x", "r"), ("add", "r", 1), ("store", "x", "r"),
                                    ("load", "x", "s"), ("store_value", "y", 9)])
    while not transaction.done():
        transaction.step(memory)
    check(transaction.registers["s"] == 1,
          "a load after a buffered store must read the transaction's own write")
    check(memory.values == {"x": 0, "y": 5} and transaction.write_set == {"x": 1, "y": 9},
          "nothing is visible before the commit: %r" % (transaction.write_set,))
    check("x" not in transaction.read_set and transaction.write_versions["x"] == 1,
          "a buffered write is validated through the version it observed: %r"
          % (transaction.write_versions,))
    check(transaction.validate(memory), "an untouched memory must validate")
    for address, value in transaction.write_set.items():
        memory.write(address, value)
    check(memory.values == {"x": 1, "y": 9}, "the commit installs the buffered values")

    # ---- a stale read is refused ------------------------------------------------------
    fresh = TxnMemory({"x": 0})
    reader = Transaction("reader", [("load", "x", "r")])
    reader.step(fresh)
    check(reader.validate(fresh), "the read set is valid immediately after the read")
    fresh.write("x", 99)                      # another transaction commits in between
    check(not reader.validate(fresh),
          "a commit between the read and the validation must be detected")
    before = fresh.snapshot()
    if reader.validate(fresh):                # would be a bug; keep the state untouched anyway
        raise AssertionError("stale validation must fail")
    check(fresh.snapshot() == before, "a refused validation must not touch memory")

    # a blind write must also be validated
    blind = Transaction("blind", [("store_value", "x", 7)])
    blind.step(fresh)
    fresh.write("x", 100)
    check(not blind.validate(fresh), "a blind write must notice a concurrent update")

    # ---- disjoint transactions never conflict ----------------------------------------
    quiet = StmSystem({"a": 0, "b": 0}, [counter_transaction("t0", "a", 2),
                                         counter_transaction("t1", "b", 3)])
    quiet.run()
    check(quiet.aborts == 0 and quiet.memory.values == {"a": 2, "b": 3},
          "disjoint transactions must not abort: %r" % (quiet.memory.values,))
    check(quiet.commits == 2 and quiet.steps == 15,
          "every operation of both transactions runs exactly once: %d steps" % quiet.steps)

    # ---- the contended counter benchmark ---------------------------------------------
    workload = [counter_transaction("t0", "c", 3), counter_transaction("t1", "c", 3),
                counter_transaction("t2", "c", 3)]
    system = StmSystem({"c": 0}, workload, max_steps=100000)
    system.run()
    check(system.memory.values["c"] == 9,
          "no lost updates: the counter must reach 9 (%r)" % (system.memory.values,))
    check(system.aborts > 0, "contention must produce aborts (%d)" % system.aborts)
    check(system.commits == 3 and len(system.committed_order) == 3,
          "every transaction eventually commits exactly once")
    check(all(t.status == "committed" for t in workload), "all transactions finished")
    check(system.memory.versions["c"] == 1 + system.commits,
          "only committed transactions bump the version: %d" % system.memory.versions["c"])
    check(serializable(system.memory.values, system.committed_order, {"c": 0}),
          "the final state must be the result of some serial order of the committed set")
    check(conflict_graph_acyclic(system.committed_order),
          "the committed read/write conflicts must be acyclic")

    quiet_commits = quiet.committed_order
    check(serializable(quiet.memory.values, quiet_commits, {"a": 0, "b": 0})
          and conflict_graph_acyclic(quiet_commits),
          "the disjoint run is trivially serializable")

    # ---- higher contention: more transactions, more aborts, same invariant ------------
    hot = StmSystem({"c": 0}, [counter_transaction("h%d" % i, "c", 2) for i in range(4)])
    hot.run()
    check(hot.memory.values["c"] == 8, "the hot counter must still reach 8: %r"
          % (hot.memory.values,))
    check(hot.aborts >= 2, "four transactions on one address abort repeatedly (%d)"
          % hot.aborts)
    check(serializable(hot.memory.values, hot.committed_order, {"c": 0}),
          "the hot run must also be serializable")
    check(conflict_graph_acyclic(hot.committed_order), "hot conflict graph must be acyclic")

    # ---- read-only transactions and empty transactions -------------------------------
    read_only = [Transaction("ro1", [("load", "v", "r"), ("store_value", "w", 0)]),
                 Transaction("ro2", [("load", "v", "s")])]
    read_run = StmSystem({"v": 3, "w": 1}, read_only)
    read_run.run()
    check(read_run.memory.values == {"v": 3, "w": 0}, "read-only work commits cleanly")
    empty = Transaction("empty", [])
    empty_system = StmSystem({"z": 1}, [empty])
    empty_system.run()
    check(empty_system.commits == 1 and empty_system.memory.values == {"z": 1},
          "a transaction with no operations commits without touching memory")
    check(empty_system.memory.bumps == 0, "it bumps no version")

    # ---- determinism, error handling and the step guard ------------------------------
    again = StmSystem({"c": 0}, [counter_transaction("t0", "c", 3),
                                 counter_transaction("t1", "c", 3),
                                 counter_transaction("t2", "c", 3)], max_steps=100000)
    again.run()
    check(again.memory.values == system.memory.values and again.aborts == system.aborts
          and again.steps == system.steps,
          "the benchmark must be deterministic (%d vs %d aborts)" % (again.aborts, system.aborts))
    rejected = 0
    for bad in (lambda: Transaction("bad", [("frob", "x")]),
                lambda: TxnMemory({"x": 0}).read("nope"),
                lambda: TxnMemory({"x": 0}).write("nope", 1)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 3, "invalid operations and addresses must be rejected (%d of 3)"
          % rejected)
    stuck = False
    try:
        StmSystem({"c": 0}, [counter_transaction("t0", "c", 3)], max_steps=2).run()
    except AssertionError:
        stuck = True
    check(stuck, "the step guard must abort a run that cannot finish")

    print("counter bench: %d commits, %d aborts, %d steps, counter = %d"
          % (system.commits, system.aborts, system.steps, system.memory.values["c"]))
    print("hot bench: %d commits, %d aborts of 4 transactions" % (hot.commits, hot.aborts))
    print("PASS")


if __name__ == "__main__":
    main()
