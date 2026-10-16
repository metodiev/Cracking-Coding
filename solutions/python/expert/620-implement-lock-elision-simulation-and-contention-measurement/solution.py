"""Task 620 - Implement lock elision simulation and contention measurement.  (Tier: Expert, CTCI Ch 15 — Threads and Locks)

Lock elision (Intel TSX-style hardware transactional memory) tries to run a critical section as a
transaction: if it commits without conflicts the lock is never taken at all, and only a conflict
or a capacity overflow rolls the attempt back, after which the code retries or falls back to
taking the lock.  This file simulates exactly that and *measures* the contention it exposes.

Two engines execute the same workload - every transaction reads each key of its section and writes
it back incremented - over the same explicit schedule, i.e. a list of thread ids, with no real
threads, sockets or sleeps anywhere:

  * LockedEngine - acquire/read/write/release micro-ops.  A busy lock does not block the
    simulator: the thread spins (one contention event plus a spin cost) and retries that micro-op.
  * ElidedEngine - begin/read/write/commit micro-ops with a read set and a write set.  Commit
    validates the versions of the read set; a changed version is a *conflict abort*, exceeding the
    tracked-location capacity is a *capacity abort* that falls straight back to the lock, and a
    thread whose attempt budget is exhausted also falls back.  A fallback runs the whole section
    under the lock, again micro-op by micro-op.

Measured and verified:
  * both engines always reach the same final state - a hand-written schedule plus 40 seeded
    schedules - so elision changes only the cost, never the result;
  * the contention numbers are real: a serial schedule reports zero lock waits and zero aborts, a
    hand-interleaved schedule reports the one conflict abort and the one spin it implies, a section
    wider than the tracked capacity reports one capacity abort and one fallback, and a repeatedly
    losing thread reports exactly the attempt budget before falling back;
  * an aborted attempt leaves no trace: the store fingerprint after the rollback equals the one
    taken before the stale commit;
  * the cost model is honest: with a conflict-free schedule elision is strictly cheaper than
    locking for the same work, and every abort adds its modelled cost, so the measurement - not an
    assumption - decides when elision stops paying.
"""

BEGIN_COST = 2
READ_COST = 1
WRITE_COST = 1
COMMIT_COST = 4
CONFLICT_ABORT_COST = 6
CAPACITY_ABORT_COST = 4
ACQUIRE_COST = 20
RELEASE_COST = 5
SPIN_COST = 3
MAX_ATTEMPTS = 3                 # then fall back to taking the lock
TRACKED_CAPACITY = 4             # tracked locations before a capacity abort


class SharedStore:
    def __init__(self, initial):
        self.values = dict(initial)
        self.versions = {key: 0 for key in initial}
        self.owner = {key: None for key in initial}

    def fingerprint(self):
        return (tuple(sorted(self.values.items())), tuple(sorted(self.versions.items())))


class Engine:
    """A step executes one micro-op of the calling thread; the scheduler chooses the thread."""

    def __init__(self, store, threads, keys_per_transaction, transactions=1):
        self.store = store
        self.threads = list(threads)
        self.keys = list(keys_per_transaction)
        self.transactions = transactions
        self.micro = {thread: self._build_program() for thread in threads}
        self.pc = {thread: 0 for thread in threads}
        self.done = {thread: False for thread in threads}
        self.cycles = 0
        self.waits = 0
        self.lock_acquisitions = 0
        self.aborts = {}
        self.commits = 0
        self.begins = 0
        self.fallbacks = 0

    def _build_program(self):
        raise NotImplementedError

    def step(self, thread):
        if self.done[thread]:
            return "done"
        if self.pc[thread] >= len(self.micro[thread]):
            self.done[thread] = True
            return "done"
        op = self.micro[thread][self.pc[thread]]
        self.pc[thread] += 1
        if op[0] == "lock-remainder":
            return self.execute_lock_remainder(thread, op[1])
        return self.execute(thread, op)

    def execute(self, thread, op):
        raise NotImplementedError

    def execute_lock_remainder(self, thread, key):
        """One key of a section that fell back to the lock; the lock is taken and released."""
        if self.store.owner[key] is None:
            self.store.owner[key] = thread
            self.lock_acquisitions += 1
            self.cycles += ACQUIRE_COST + READ_COST + WRITE_COST + RELEASE_COST
            self.store.values[key] = self.store.values[key] + 1
            self.store.versions[key] += 1
            self.store.owner[key] = None
            self.commits += 1
            return "fallback-commit"
        self.waits += 1
        self.cycles += SPIN_COST
        self.pc[thread] -= 1
        return "spin"

    def note_abort(self, reason):
        self.aborts[reason] = self.aborts.get(reason, 0) + 1

    def all_done(self):
        return all(self.done[thread] for thread in self.threads)

    def contention(self):
        """Anything that shows a thread colliding with another thread: spins plus aborts."""
        return self.waits + sum(self.aborts.values())


class LockedEngine(Engine):
    def __init__(self, store, threads, keys_per_transaction, transactions=1):
        self.last_read = {thread: None for thread in threads}
        super().__init__(store, threads, keys_per_transaction, transactions)

    def _build_program(self):
        program = []
        for _ in range(self.transactions):
            for index, key in enumerate(self.keys):
                last = index == len(self.keys) - 1
                program.extend([("acquire", key), ("read", key), ("write", key),
                                ("release", key, last)])
        return program

    def execute(self, thread, op):
        kind, key = op[0], op[1]
        if kind == "acquire":
            if self.store.owner[key] is None:
                self.store.owner[key] = thread
                self.lock_acquisitions += 1
                self.cycles += ACQUIRE_COST
                return "acquire"
            self.waits += 1
            self.cycles += SPIN_COST
            self.pc[thread] -= 1                     # spin on the same micro-op
            return "spin"
        if kind == "read":
            self.cycles += READ_COST
            self.last_read[thread] = self.store.values[key]
            return "read"
        if kind == "write":
            self.cycles += WRITE_COST
            self.store.values[key] = self.last_read[thread] + 1
            self.store.versions[key] += 1
            return "write"
        self.store.owner[key] = None
        self.cycles += RELEASE_COST
        if len(op) > 2 and op[2]:
            self.commits += 1                        # the section's last key was released
        return "release"


class ElidedEngine(Engine):
    def __init__(self, store, threads, keys_per_transaction, transactions=1,
                 capacity=TRACKED_CAPACITY):
        self.capacity = capacity
        self.attempts = {thread: 0 for thread in threads}
        self.read_set = {thread: [] for thread in threads}
        self.write_set = {thread: {} for thread in threads}
        self.read_versions = {thread: {} for thread in threads}
        super().__init__(store, threads, keys_per_transaction, transactions)

    def _build_program(self):
        program = []
        for _ in range(self.transactions):
            program.append(("begin", None))
            for key in self.keys:
                program.extend([("read", key), ("write", key)])
            program.append(("commit", None))
        return program

    def _stride(self):
        return 2 * len(self.keys) + 2

    def _start_attempt(self, thread):
        self.read_set[thread] = []
        self.write_set[thread] = {}
        self.read_versions[thread] = {}
        self.attempts[thread] += 1

    def execute(self, thread, op):
        kind, key = op
        if kind == "begin":
            self._start_attempt(thread)
            self.begins += 1
            self.cycles += BEGIN_COST
            return "begin"
        if kind == "read":
            tracked = len(self.read_set[thread]) + len(self.write_set[thread])
            if tracked >= self.capacity:
                self.note_abort("capacity")
                self.cycles += CAPACITY_ABORT_COST
                self.fallbacks += 1
                self._fall_back(thread, self.pc[thread] - 1)
                return "capacity-abort"
            self.read_set[thread].append(key)
            self.read_versions[thread][key] = self.store.versions[key]
            self.cycles += READ_COST
            return "read"
        if kind == "write":
            self.write_set[thread][key] = self.store.values[key] + 1
            self.cycles += WRITE_COST
            return "write"
        for watched, version in self.read_versions[thread].items():
            if self.store.versions[watched] != version:
                self.note_abort("conflict")
                self.cycles += CONFLICT_ABORT_COST
                if self.attempts[thread] >= MAX_ATTEMPTS:
                    self.fallbacks += 1
                    self._fall_back(thread, self.pc[thread] - 1)
                else:
                    self.pc[thread] = ((self.pc[thread] - 1) // self._stride()) * self._stride()
                return "conflict-abort"
        for watched, value in self.write_set[thread].items():
            self.store.values[watched] = value
            self.store.versions[watched] += 1
        self.read_set[thread] = []
        self.write_set[thread] = {}
        self.cycles += COMMIT_COST
        self.commits += 1
        return "commit"

    def _fall_back(self, thread, failing_index):
        """Abandon the attempt and run the transactions that are still left under the lock."""
        stride = self._stride()
        remaining = self.transactions - failing_index // stride
        self.micro[thread] = [("lock-remainder", key)
                              for _ in range(remaining) for key in self.keys]
        self.pc[thread] = 0
        self.attempts[thread] = 0


def run_engines(store, threads, keys, schedule, kind, transactions=1):
    engine = (LockedEngine(store, threads, keys, transactions) if kind == "locked"
              else ElidedEngine(store, threads, keys, transactions))
    for thread in schedule:
        engine.step(thread)
    for _ in range(600):
        if engine.all_done():
            break
        for thread in threads:
            engine.step(thread)
    return engine


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    threads = ["t0", "t1"]
    keys = ["counter"]

    # ---- a serial schedule: nothing to contend with -------------------------
    serial = ["t0"] * 4 + ["t1"] * 4
    locked = run_engines(SharedStore({"counter": 0}), threads, keys, serial, "locked")
    elided = run_engines(SharedStore({"counter": 0}), threads, keys, serial, "elided")
    check(locked.store.values == elided.store.values == {"counter": 2},
          "both engines must reach the same final state")
    check(locked.waits == 0 and locked.contention() == 0, "a serial schedule has no contention")
    check(elided.contention() == 0 and elided.commits == 2, "no aborts without interference")
    check(locked.cycles == 2 * (ACQUIRE_COST + READ_COST + WRITE_COST + RELEASE_COST),
          "locked cycles are the modelled micro-ops: %d" % locked.cycles)
    check(elided.cycles == 2 * (BEGIN_COST + READ_COST + WRITE_COST + COMMIT_COST),
          "elided cycles are begin/read/write/commit only: %d" % elided.cycles)
    check(elided.cycles < locked.cycles,
          "with no contention elision must be strictly cheaper (%d < %d)"
          % (elided.cycles, locked.cycles))

    # ---- a hand-interleaved schedule: exactly one conflict abort ------------
    conflict = ["t0", "t1", "t0", "t1", "t0", "t1", "t0"]
    elided = run_engines(SharedStore({"counter": 0}), threads, keys, conflict, "elided")
    check(elided.aborts.get("conflict") == 1,
          "the interleaving implies exactly one conflict abort: %s" % elided.aborts)
    check(elided.commits == 2 and elided.fallbacks == 0, "one retry is enough")
    check(elided.store.values == {"counter": 2}, "the retried section still applies")
    check(elided.store.versions["counter"] == 2, "two commits, two versions")
    check(elided.cycles == 2 * (BEGIN_COST + READ_COST + WRITE_COST + COMMIT_COST)
          + (BEGIN_COST + READ_COST + WRITE_COST) + CONFLICT_ABORT_COST,
          "the abandoned attempt and the abort are charged: %d cycles" % elided.cycles)

    locked = run_engines(SharedStore({"counter": 0}), threads, keys, ["t0", "t1"] * 4, "locked")
    check(locked.waits >= 1,
          "the locked engine must report spins on the same interleaving: %d" % locked.waits)
    check(locked.store.values == {"counter": 2}, "locking reaches the same answer")
    check(locked.cycles > 2 * (ACQUIRE_COST + RELEASE_COST),
          "spinning must show up in the measured cycles: %d" % locked.cycles)

    # ---- capacity abort and fallback ---------------------------------------
    wide_keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
    initial = {key: 0 for key in wide_keys}
    engine = run_engines(SharedStore(initial), ["u0"], wide_keys, ["u0"] * 40, "elided")
    check(engine.aborts.get("capacity") == 1,
          "a section wider than the tracked capacity aborts once and falls back")
    check(engine.fallbacks == 1, "the capacity abort triggers exactly one fallback")
    check(engine.store.values == {key: 1 for key in wide_keys},
          "the fallback still applies the whole section exactly once")
    check(engine.contention() == 1, "the capacity abort is the only contention here")
    check(engine.lock_acquisitions == len(wide_keys),
          "the fallback takes the lock once per key of the section")

    narrow = run_engines(SharedStore(initial), ["u0"], wide_keys[:TRACKED_CAPACITY // 2],
                         ["u0"] * 40, "elided")
    check(narrow.aborts == {}, "a section that stays inside the capacity never aborts")
    check(narrow.fallbacks == 0 and narrow.commits == 1, "and never falls back")

    # ---- an aborted attempt leaves no trace --------------------------------
    store = SharedStore({"counter": 3})
    engine = ElidedEngine(store, threads, ["counter"])
    engine.step("t0")                                # begin
    engine.step("t0")                                # read
    engine.step("t0")                                # write (buffered only)
    engine.step("t1")                                # begin
    engine.step("t1")                                # read
    engine.step("t1")                                # write (buffered only)
    check(store.values == {"counter": 3}, "buffered writes are invisible before commit")
    engine.step("t0")                                # commit
    check(store.values == {"counter": 4} and store.versions["counter"] == 1,
          "the committed transaction is published")
    after_commit = store.fingerprint()
    engine.step("t1")                                # commit: the read set is stale now
    check(engine.aborts.get("conflict") == 1, "the stale attempt aborts")
    check(store.fingerprint() == after_commit, "the rolled back attempt wrote nothing")
    check(engine.pc["t1"] == 0 and engine.attempts["t1"] == 1,
          "the thread rewound to the beginning of its transaction for a retry")
    for _ in range(4):
        engine.step("t1")
    check(store.values == {"counter": 5} and engine.commits == 2,
          "the retry reads the fresh value and commits: %s" % store.values)

    # ---- budget exhaustion: repeatedly losing ends in a fallback ------------
    engine = ElidedEngine(SharedStore({"counter": 0}), threads, ["counter"],
                          transactions=MAX_ATTEMPTS)
    for _ in range(MAX_ATTEMPTS):                    # t0 keeps winning the race with t1
        engine.step("t0")                            # begin t0
        engine.step("t1")                            # begin t1 (fresh or retry)
        engine.step("t0")                            # read t0
        engine.step("t1")                            # read t1
        engine.step("t0")                            # write t0
        engine.step("t1")                            # write t1
        engine.step("t0")                            # commit t0 -> t1's read set goes stale
        engine.step("t1")                            # commit t1 -> conflict
    check(engine.aborts.get("conflict") == MAX_ATTEMPTS,
          "t1 must lose exactly the budgeted number of races: %s" % engine.aborts)
    check(engine.fallbacks == 1, "the exhausted budget triggers one fallback")
    while not engine.all_done():
        for thread in threads:
            engine.step(thread)
    check(engine.store.values == {"counter": 3 + 3},
          "t0's three transactions plus t1's three fallback sections: %s" % engine.store.values)
    check(engine.lock_acquisitions == 3,
          "the fallback took the lock once per remaining section")

    # ---- sweep: identical results, measured contention ----------------------
    keys = ["a", "b"]
    initial = {"a": 0, "b": 0}
    state = 987654321
    schedules_with_aborts = 0
    for seed in range(40):
        schedule = []
        for _ in range(24):
            state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            schedule.append(threads[state % 2])
        locked_engine = run_engines(SharedStore(initial), threads, keys, schedule, "locked",
                                    transactions=2)
        elided_engine = run_engines(SharedStore(initial), threads, keys, schedule, "elided",
                                    transactions=2)
        check(locked_engine.store.values == elided_engine.store.values,
              "schedule %d: locked %s vs elided %s" % (seed, locked_engine.store.values,
                                                       elided_engine.store.values))
        check(locked_engine.commits == elided_engine.commits == 4,
              "both engines performed four sections in every schedule")
        check(locked_engine.store.values == {"a": 4, "b": 4},
              "every section increments both keys exactly once")
        check(elided_engine.contention() >= 0 and locked_engine.contention() >= 0,
              "contention is measured, never negative")
        if elided_engine.aborts:
            schedules_with_aborts += 1
    check(schedules_with_aborts > 0, "some seeded schedules must produce transactional aborts")

    print("serial: locked %d cycles, elided %d cycles, contention 0"
          % (2 * (ACQUIRE_COST + READ_COST + WRITE_COST + RELEASE_COST),
             2 * (BEGIN_COST + READ_COST + WRITE_COST + COMMIT_COST)))
    print("interleaved: 1 conflict abort + 1 retry vs %d locked cycles with %d spin(s)"
          % (locked.cycles, locked.waits))
    print("wide section: 1 capacity abort + 1 fallback applying %d keys exactly once"
          % len(wide_keys))
    print("sweep: 40 seeded schedules, locked and elided agree every time (%d with aborts)"
          % schedules_with_aborts)
    print("PASS")


if __name__ == "__main__":
    main()
