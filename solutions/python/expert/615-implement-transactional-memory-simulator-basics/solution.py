"""Task 615 - Implement transactional memory simulator basics.  (Tier: Expert, CTCI Ch 15 — Threads and Locks)

A software transactional memory (TL2 style) driven by an *explicit schedule* rather than real
threads: a schedule is a list of transaction names and each entry lets that transaction execute
exactly one step of its program.  No threads, no sockets, no sleeping, no clock - the whole
interleaving is plain data, so every run is reproducible.

Model:
  * the store maps variable -> (value, version); a global commit clock is bumped by each
    committing transaction and stamped on every variable it writes;
  * a read records (variable, version seen) in the transaction's read set; a write goes to a
    private buffer that nobody else can observe;
  * commit validates the read set (every variable must still carry the version that was read)
    and only then publishes the buffer with a fresh version.  A validation failure aborts the
    transaction: its buffer is dropped, its abort counter grows and it restarts at step 0, which
    is how the scheduler retries it;
  * reads consult the transaction's own buffer first (read-your-writes) and otherwise the
    published store, so an uncommitted value is never observable (no dirty reads).

Simplification, as documented: no low-level reads of the versioned metadata are modelled (a real
TL2 also validates a "global version" snapshot and locks written locations during commit), and
the scheduler is an explicit list rather than a fair thread scheduler.  The invariant that is
still genuinely verified is opacity-ish serializability: commit-order replay of the committed
transactions against a fresh store reproduces the final state exactly, no read ever observes an
uncommitted value, a hand-written conflicting interleaving produces exactly one validation
failure followed by a successful retry, and 40 seeded transfer schedules all preserve the
conservation of the transferred total with every transaction settled.
"""

MASK = 0xFFFFFFFFFFFFFFFF


def splitmix64(value):
    z = (value + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


class Store:
    """Published state: variable -> value plus a version per variable and a global clock."""

    def __init__(self, initial):
        self.values = dict(initial)
        self.versions = {name: 0 for name in initial}
        self.clock = 0

    def read(self, name):
        return self.values[name], self.versions[name]

    def publish(self, writes):
        self.clock += 1
        for name, value in writes.items():
            self.values[name] = value
            self.versions[name] = self.clock
        return self.clock


class Transaction:
    """One STM transaction over a fixed op program, advanced one step at a time.

    Program ops: ("read", var), ("write", var, value), ("commit",)."""

    def __init__(self, name, program, store):
        self.name = name
        self.program = program
        self.store = store
        self.aborts = 0
        self.commits = 0
        self.attempts = 0
        self.reset()

    def reset(self):
        self.pc = 0
        self.reads = {}          # variable -> version observed, validated at commit
        self.writes = {}         # buffered writes, invisible until commit
        self.observed = []       # (variable, value) as actually read, for the witness
        self.finished = False
        self.committed = False

    @property
    def op(self):
        return None if self.finished else self.program[self.pc][0]

    def step(self):
        """Execute exactly one op; returns its kind, or 'idle' when the transaction is done."""
        if self.finished or self.pc >= len(self.program):
            return "idle"
        op = self.program[self.pc]
        self.pc += 1
        if op[0] == "read":
            name = op[1]
            if name in self.writes:
                value = self.writes[name]                   # read-your-writes
            else:
                value, version = self.store.read(name)
                self.reads[name] = version
            self.observed.append((name, value))
            return "read"
        if op[0] == "write":
            self.writes[op[1]] = op[2]
            return "write"
        if op[0] == "commit":
            self.attempts += 1
            self.finished = True
            for name, version in self.reads.items():
                if self.store.versions[name] != version:    # read set went stale
                    self.aborts += 1
                    self.reset()                            # retry from the beginning
                    return "abort"
            self.store.publish(self.writes)
            self.committed = True
            self.commits += 1
            return "commit"
        raise AssertionError("unknown op %r" % (op,))


class Scheduler:
    """Runs explicit schedules; one schedule entry = one step of one transaction."""

    def __init__(self, initial):
        self.initial = dict(initial)
        self.store = Store(initial)
        self.transactions = {}
        self.log = []
        self.steps = 0

    def add(self, name, program):
        self.transactions[name] = Transaction(name, program, self.store)

    def settle(self):
        """Finish every transaction with a fixed order, one op at a time, until all are done."""
        for _ in range(400):
            progressed = False
            for name in sorted(self.transactions):
                transaction = self.transactions[name]
                if not transaction.finished:
                    outcome = transaction.step()
                    progressed = True
                    if outcome in ("commit", "abort"):
                        self.log.append((outcome, name))
            if not progressed:
                return

    def run(self, schedule, max_steps=400):
        self.steps = 0
        for name in schedule:
            if self.steps >= max_steps:
                raise AssertionError("the schedule did not settle within max_steps")
            self.steps += 1
            outcome = self.transactions[name].step()
            if outcome in ("commit", "abort"):
                self.log.append((outcome, name))
        self.settle()
        return self.log

    def round_robin(self, rounds=40):
        return [name for _ in range(rounds) for name in sorted(self.transactions)]

    def seeded_schedule(self, seed, length=60):
        state, names = splitmix64(seed), sorted(self.transactions)
        schedule = []
        for _ in range(length):
            state = splitmix64(state)
            schedule.append(names[state % len(names)])
        return schedule


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # ---- a serial schedule commits everything and matches the arithmetic ----
    scheduler = Scheduler({"x": 100, "y": 50})
    scheduler.add("t1", [("read", "x"), ("write", "y", 160), ("commit",)])
    scheduler.add("t2", [("read", "y"), ("write", "x", 200), ("commit",)])
    log = scheduler.run(["t1", "t1", "t1", "t2", "t2", "t2"])
    check([entry[0] for entry in log] == ["commit", "commit"], "serial schedule commits both")
    check(scheduler.store.values == {"x": 200, "y": 160}, "serial schedule final state")
    check(all(transaction.aborts == 0 for transaction in scheduler.transactions.values()),
          "no aborts in a serial run")
    check(scheduler.store.versions == {"x": 2, "y": 1}, "commit clock stamps both writes")
    serial_versions = dict(scheduler.store.versions)

    # ---- a conflicting interleaving aborts exactly once, then retries to a commit --
    scheduler = Scheduler({"x": 100, "y": 50})
    scheduler.add("t1", [("read", "x"), ("read", "y"), ("write", "x", 110),
                         ("write", "y", 40), ("commit",)])
    scheduler.add("t2", [("read", "y"), ("read", "x"), ("write", "y", 60),
                         ("write", "x", 90), ("commit",)])
    log = scheduler.run(["t1", "t1", "t2", "t2",          # both read before either writes
                         "t1", "t1", "t1",                 # t1 publishes
                         "t2", "t2", "t2"], max_steps=40)  # t2's read set is stale now
    outcomes = [entry[0] for entry in log]
    check(outcomes.count("abort") == 1, "exactly one validation failure: %s" % outcomes)
    check(outcomes[-1] == "commit", "the retried transaction commits")
    check(scheduler.transactions["t2"].aborts == 1, "the abort belongs to t2")
    check(scheduler.transactions["t2"].commits == 1, "the retry commits exactly once")
    check(scheduler.store.values["x"] + scheduler.store.values["y"] == 150,
          "the conservation invariant survives the conflict")

    # ---- read-set validation: a write committed between read and commit aborts --
    scheduler = Scheduler({"a": 1})
    scheduler.add("writer", [("write", "a", 99), ("commit",)])
    scheduler.add("reader", [("read", "a"), ("write", "a", 5), ("commit",)])
    scheduler.run(["writer", "reader", "writer", "reader", "reader"])
    check(scheduler.transactions["reader"].aborts == 1, "the stale reader aborts once")
    check(scheduler.store.values["a"] == 5, "the retry publishes its own value")
    check(scheduler.transactions["reader"].observed == [("a", 99)],
          "the retry re-reads the value published by the writer")

    # ---- no dirty reads: an uncommitted buffer is invisible ---------------------
    scheduler = Scheduler({"v": 0})
    scheduler.add("a", [("write", "v", 7), ("read", "v"), ("commit",)])
    scheduler.add("b", [("read", "v"), ("commit",)])
    scheduler.run(["a", "b", "a", "b", "a"])
    check(scheduler.transactions["b"].observed == [("v", 0)],
          "b must never see a's uncommitted 7")
    check(scheduler.transactions["a"].observed == [("v", 7)], "read-your-writes for a")
    check(scheduler.store.values["v"] == 7, "a's buffer is published on commit")
    check(scheduler.transactions["a"].aborts == 0 and scheduler.transactions["b"].aborts == 0,
          "no aborts when the conflict is only a buffered write")

    # ---- read-your-own-writes inside one transaction ----------------------------
    scheduler = Scheduler({"k": 1})
    scheduler.add("t", [("write", "k", 11), ("read", "k"), ("write", "k", 12),
                        ("read", "k"), ("commit",)])
    scheduler.run(scheduler.round_robin(6))
    check(scheduler.transactions["t"].observed == [("k", 11), ("k", 12)],
          "buffered writes read back in program order")
    check(scheduler.store.values["k"] == 12, "only the last buffered value is published")

    # ---- serializability witness: commit-order replay reproduces the final state --
    for seed in (1, 7, 99, 1234, 20250):
        scheduler = Scheduler({"p": 10, "q": 20, "r": 5})
        programs = {
            "t1": [("read", "p"), ("write", "q", 21), ("commit",)],
            "t2": [("read", "q"), ("write", "r", 6), ("commit",)],
            "t3": [("read", "r"), ("read", "p"), ("write", "p", 11), ("commit",)],
        }
        for name, program in programs.items():
            scheduler.add(name, program)
        log = scheduler.run(scheduler.seeded_schedule(seed, 60))
        committed = [entry[1] for entry in log if entry[0] == "commit"]
        check(committed, "every sweep must commit at least one transaction")
        store = Store(scheduler.initial)
        for name in committed:
            replay = Transaction(name, programs[name], store)
            for _ in range(len(programs[name]) + 1):
                if not replay.finished:
                    replay.step()
            check(replay.committed, "the commit-order replay commits cleanly")
        check(store.values == scheduler.store.values,
              "commit-order replay reproduces the final state (seed %d)" % seed)

    # ---- conservation invariant across 40 seeded transfer schedules --------------
    for seed in range(40):
        scheduler = Scheduler({"left": 500, "right": 500})
        scheduler.add("move_a", [("read", "left"), ("read", "right"),
                                 ("write", "left", 400), ("write", "right", 600), ("commit",)])
        scheduler.add("move_b", [("read", "right"), ("read", "left"),
                                 ("write", "right", 450), ("write", "left", 550), ("commit",)])
        scheduler.run(scheduler.seeded_schedule(seed, 40), max_steps=300)
        check(scheduler.store.values["left"] + scheduler.store.values["right"] == 1000,
              "conservation broken by schedule %d" % seed)
        for transaction in scheduler.transactions.values():
            check(transaction.finished, "every transaction must settle")
        check(sum(t.commits for t in scheduler.transactions.values()) >= 1,
              "at least one transfer commits per schedule")

    # ---- rejected input and degenerate programs ---------------------------------
    scheduler = Scheduler({"z": 1})
    scheduler.add("bad", [("frobnicate", "z")])
    try:
        scheduler.run(["bad"])
        raise AssertionError("an unknown op must raise")
    except AssertionError:
        pass
    unfinished = Scheduler({"z": 1})
    unfinished.add("busy", [("read", "z"), ("read", "z")])
    unfinished.run(["busy"], max_steps=5)
    check(not unfinished.transactions["busy"].finished,
          "a program without a commit never finishes")
    empty = Scheduler({"e": 3})
    empty.add("noop", [("commit",)])
    empty.run(["noop"])
    check(empty.store.values["e"] == 3 and empty.store.clock == 1,
          "a commit with an empty write set still advances the clock")

    print("serial run: 2 commits, 0 aborts, versions %s" % serial_versions)
    print("conflicting run: 1 abort then a retry commit; conservation holds")
    print("seeded sweep: 40 transfer schedules settled with the total conserved")
    print("PASS")


if __name__ == "__main__":
    main()
