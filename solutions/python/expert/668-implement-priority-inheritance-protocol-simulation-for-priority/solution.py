"""Task 668 - Implement priority inheritance protocol simulation for priority inversion.  (Tier: Expert, Beyond — Networking, OS & Real-Time)

Deterministic event-driven simulation of the Priority Inheritance Protocol (PIP) on a
single CPU. Every task carries a program of operations - ``cpu`` bursts, ``lock`` and
``unlock`` on named binary semaphores - plus a base priority and a release time. One time
unit schedules exactly one work item; the ready task with the highest *effective* priority
wins (ties broken by declaration order, so runs are reproducible).

Effective priorities are the least fixpoint of
    eff[t] = max(base[t], max{ eff[w] : w is BLOCKED on a semaphore owned by t })
The fixpoint is what makes inheritance *transitive*: if H waits for a lock held by M and M
waits for a lock held by L, then L is boosted all the way to H's priority. On release the
semaphore is handed directly to the highest-effective-priority waiter (PIP wake-up), which
is why the woken task's queued ``lock`` operation is consumed by the handoff instead of
being retried.

The classic inversion scenario is played twice, with inheritance disabled and enabled:
without PIP an unrelated medium-priority task starves the low-priority lock holder and the
high-priority task's blocking time grows with the medium task's work; with PIP the holder
runs at the blocked task's priority and the delay is bounded by the remaining critical
section. Invariants are re-verified after every single step.

Simplified versus a real RTOS: one lock per resource, no priority ceilings, no timeouts.
Still fully verified: single-ownership of locks, transitive inheritance, priority
restoration, bounded inversion, priority wake-up selection and deadlock detection.
"""

MAX_STEPS = 20000


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Lock:
    """Binary semaphore: an owner (or None) plus an arrival-ordered wait list."""

    def __init__(self, name):
        self.name = name
        self.owner = None
        self.waiters = []


class Task:
    def __init__(self, name, base_priority, program, release=0):
        self.name = name
        self.base_priority = base_priority
        self.program = program
        self.release = release
        self.pc = 0
        self.cpu_left = 0
        self.state = "NEW"        # NEW -> READY -> RUNNING -> BLOCKED / DONE
        self.blocked_on = None
        self.held = set()
        self.blocked_time = 0
        self.eff_priority = base_priority
        self.finished_at = None


class Simulator:
    """One-CPU scheduler with optional priority inheritance."""

    def __init__(self, tasks, locks, inherit):
        self.order = [t.name for t in tasks]
        self.tasks = {t.name: t for t in tasks}
        self.locks = {l.name: l for l in locks}
        self.inherit = inherit
        self.clock = 0
        self.log = []
        self.snapshots = []       # (clock, {task: effective priority})
        self.handoffs = []        # (clock, lock, woken task)
        self.deadlock = False
        self.inversion_units = 0
        self.invariant_checks = 0

    # ---- priorities ---------------------------------------------------------
    def effective_priorities(self):
        eff = {n: t.base_priority for n, t in self.tasks.items()}
        if self.inherit:
            changed = True
            while changed:                     # monotone -> terminates
                changed = False
                for t in self.tasks.values():
                    if t.state != "BLOCKED":
                        continue
                    owner = self.locks[t.blocked_on].owner
                    if owner is not None and eff[owner] < eff[t.name]:
                        eff[owner] = eff[t.name]
                        changed = True
        for name, value in eff.items():
            self.tasks[name].eff_priority = value
        return eff

    def verify_invariants(self, runner, eff):
        self.invariant_checks += 1
        for lock in self.locks.values():
            holders = sorted(t.name for t in self.tasks.values() if lock.name in t.held)
            if lock.owner is None:
                check(holders == [], "lock %s has holders %s but no owner" % (lock.name, holders))
            else:
                check(holders == [lock.owner],
                      "lock %s owned by %s but held by %s" % (lock.name, lock.owner, holders))
        for t in self.tasks.values():
            check(t.eff_priority >= t.base_priority,
                  "task %s dropped below its base priority" % t.name)
            check(len(t.held) <= len(self.locks), "task %s holds unknown locks" % t.name)
            if t.state == "BLOCKED":
                lock = self.locks[t.blocked_on]
                check(t.name in lock.waiters or lock.owner == t.name,
                      "blocked task %s is not queued on %s" % (t.name, t.blocked_on))
        if runner is not None and self.inherit:
            for t in self.tasks.values():
                check(not (t.state == "BLOCKED" and t.eff_priority > eff[runner.name]),
                      "an inversion survived: %s blocked at %d while %s runs at %d"
                      % (t.name, t.eff_priority, runner.name, eff[runner.name]))

    # ---- primitives ---------------------------------------------------------
    def release_lock(self, task, lock):
        check(lock.owner == task.name,
              "task %s unlocked %s which it does not own" % (task.name, lock.name))
        lock.owner = None
        task.held.discard(lock.name)
        self.log.append((self.clock, task.name, "unlock:" + lock.name))
        if lock.waiters:
            eff = self.effective_priorities()
            best = max(lock.waiters, key=lambda n: (eff[n], -lock.waiters.index(n)))
            lock.waiters.remove(best)
            woken = self.tasks[best]
            lock.owner = woken.name
            woken.held.add(lock.name)
            woken.blocked_on = None
            woken.state = "READY"
            woken.pc += 1                     # handoff satisfies its queued lock op
            self.handoffs.append((self.clock, lock.name, best))

    def execute(self, task):
        task.state = "RUNNING"
        while True:
            if task.pc >= len(task.program):
                task.state = "DONE"
                task.finished_at = self.clock
                for name in sorted(task.held):
                    self.release_lock(task, self.locks[name])
                self.log.append((self.clock, task.name, "done"))
                return
            op = task.program[task.pc]
            kind = op[0]
            if kind == "cpu":
                if task.cpu_left == 0:
                    task.cpu_left = op[1]
                task.cpu_left -= 1
                if task.cpu_left == 0:
                    task.pc += 1
                task.state = "READY"
                return
            if kind == "lock":
                lock = self.locks[op[1]]
                if lock.owner is None:
                    lock.owner = task.name
                    task.held.add(lock.name)
                    task.pc += 1
                    self.log.append((self.clock, task.name, "lock:" + lock.name))
                    continue
                lock.waiters.append(task.name)
                task.blocked_on = lock.name
                task.state = "BLOCKED"
                self.log.append((self.clock, task.name, "block:" + lock.name))
                return
            check(kind == "unlock", "unknown operation %r" % (op,))
            self.release_lock(task, self.locks[op[1]])
            task.pc += 1

    # ---- main loop ----------------------------------------------------------
    def run(self):
        for _ in range(MAX_STEPS):
            for t in self.tasks.values():
                if t.state == "NEW" and self.clock >= t.release:
                    t.state = "READY"
            eff = self.effective_priorities()
            self.snapshots.append((self.clock, dict((n, t.eff_priority)
                                                    for n, t in self.tasks.items())))
            ready = [t for t in self.tasks.values() if t.state == "READY"]
            if not ready:
                if all(t.state == "DONE" for t in self.tasks.values()):
                    return self.clock
                self.deadlock = True
                return self.clock
            runner = max(ready, key=lambda t: (eff[t.name], -self.order.index(t.name)))
            blocked = [t for t in self.tasks.values() if t.state == "BLOCKED"]
            if any(t.eff_priority > eff[runner.name] for t in blocked):
                self.inversion_units += 1
            for t in blocked:
                t.blocked_time += 1
            self.clock += 1
            self.execute(runner)
            self.verify_invariants(runner, eff)
        raise AssertionError("simulation did not converge")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def inversion_scenario(inherit):
    """L grabs the lock, H blocks on it, an unrelated M becomes ready afterwards."""
    low = Task("L", 10, [("lock", "R"), ("cpu", 20), ("unlock", "R"), ("cpu", 2)], release=0)
    high = Task("H", 30, [("cpu", 1), ("lock", "R"), ("cpu", 3), ("unlock", "R")], release=1)
    medium = Task("M", 20, [("cpu", 40)], release=3)
    sim = Simulator([low, high, medium], [Lock("R")], inherit=inherit)
    sim.run()
    return sim


def transitive_scenario():
    """L holds R1, M holds R2 and waits for R1, H waits for R2 -> L inherits through M."""
    low = Task("L", 5, [("lock", "R1"), ("cpu", 4), ("unlock", "R1"), ("cpu", 1)], release=0)
    mid = Task("M", 15, [("lock", "R2"), ("cpu", 2), ("lock", "R1"),
                         ("cpu", 1), ("unlock", "R1"), ("unlock", "R2")], release=1)
    high = Task("H", 25, [("cpu", 1), ("lock", "R2"), ("cpu", 1), ("unlock", "R2")], release=2)
    sim = Simulator([low, mid, high], [Lock("R1"), Lock("R2")], inherit=True)
    sim.run()
    return sim


def wakeup_scenario():
    """Two waiters really queue on one lock; the higher-priority one must be woken."""
    holder = Task("O", 1, [("lock", "R"), ("cpu", 8), ("unlock", "R")], release=0)
    low = Task("W2", 12, [("lock", "R"), ("cpu", 1), ("unlock", "R")], release=1)
    high = Task("W3", 14, [("lock", "R"), ("cpu", 1), ("unlock", "R")], release=2)
    sim = Simulator([holder, low, high], [Lock("R")], inherit=True)
    sim.run()
    return sim


def boost_scenario():
    """A boosted holder is not preempted, so a weaker task never even joins the queue."""
    holder = Task("O", 1, [("lock", "R"), ("cpu", 5), ("unlock", "R")], release=0)
    weak = Task("W1", 8, [("lock", "R"), ("cpu", 1), ("unlock", "R")], release=1)
    strong = Task("W2", 12, [("lock", "R"), ("cpu", 1), ("unlock", "R")], release=1)
    sim = Simulator([holder, weak, strong], [Lock("R")], inherit=True)
    sim.run()
    return sim


def main():
    without = inversion_scenario(inherit=False)
    with_pip = inversion_scenario(inherit=True)

    check(not without.deadlock and not with_pip.deadlock, "inversion scenarios must terminate")
    check(all(t.state == "DONE" for t in without.tasks.values()), "all tasks run to completion")

    blocked_no = without.tasks["H"].blocked_time
    blocked_yes = with_pip.tasks["H"].blocked_time
    medium_work = 40
    check(blocked_no >= medium_work,
          "without PIP H is starved for the medium task's full work (%d < %d)"
          % (blocked_no, medium_work))
    check(blocked_yes < blocked_no,
          "inheritance must shrink the blocking window (%d !< %d)" % (blocked_yes, blocked_no))
    check(blocked_yes <= 20,
          "with PIP H is delayed only by L's critical section, not by M (%d > 20)" % blocked_yes)
    check(without.inversion_units > 0, "priority inversion must be observable without PIP")
    check(with_pip.inversion_units == 0,
          "PIP must eliminate preemption by lower-priority work (%d units)"
          % with_pip.inversion_units)
    check(without.invariant_checks > 50 and with_pip.invariant_checks > 50,
          "invariants must be rechecked on every step")

    # The medium task is what inflates the blocking window: with inheritance it is the
    # one that gets delayed (L is boosted over it), not the high-priority task.
    check(with_pip.tasks["M"].finished_at > without.tasks["M"].finished_at,
          "M must be pushed back when L inherits H's priority")
    check(with_pip.tasks["M"].finished_at - without.tasks["M"].finished_at >= blocked_yes,
          "M gives up exactly the critical section L no longer waits for")

    # Priority restoration: L is boosted to 30 while H waits and drops back to 10 after.
    boosted = [snap["L"] for _, snap in with_pip.snapshots]
    check(max(boosted) == 30, "L must inherit H's priority 30 (saw %d)" % max(boosted))
    after_unlock = with_pip.tasks["L"].eff_priority
    check(after_unlock == 10 == with_pip.tasks["L"].base_priority,
          "L's priority must revert to its immutable base 10 (got %d)" % after_unlock)

    # Transitive inheritance through M.
    trans = transitive_scenario()
    check(not trans.deadlock, "transitive scenario must terminate")
    peak = max(snap["L"] for _, snap in trans.snapshots)
    peak_m = max(snap["M"] for _, snap in trans.snapshots)
    check(peak == 25, "L must inherit 25 transitively through M (saw %d)" % peak)
    check(peak_m == 25, "M must inherit H's 25 while it holds R2 (saw %d)" % peak_m)
    # H waits at most for the chain of critical sections it is blocked behind: M's
    # remaining work while holding R2 (2 + 1 cpu units) plus L's remaining work while
    # holding R1 (3 of its 4 units), plus the units consumed by the two blocking
    # attempts themselves - i.e. the wait is bounded by critical-section work, never by
    # unrelated lower-priority work.
    chain_work = (2 + 1) + 3
    check(trans.tasks["H"].blocked_time <= chain_work + 2,
          "transitive chain must bound H's wait by CS work (%d > %d)"
          % (trans.tasks["H"].blocked_time, chain_work + 2))
    check(trans.tasks["H"].blocked_time >= 3,
          "H must really wait behind the chain (%d)" % trans.tasks["H"].blocked_time)

    # Priority wake-up hands the lock to the highest-priority waiter, not the first one
    # that happened to queue (W2 arrived earlier, W3 is stronger).
    wake = wakeup_scenario()
    check(not wake.deadlock, "wake-up scenario must terminate")
    check([h[2] for h in wake.handoffs] == ["W3", "W2"],
          "the lock must go to the strongest waiter first (%r)" % (wake.handoffs,))
    check(wake.tasks["W2"].blocked_time > wake.tasks["W3"].blocked_time,
          "the weaker waiter must wait for the stronger one")

    # A boosted holder runs instead of an unrelated weaker task, which therefore never
    # even attempts the semaphore while the lock is held.
    boost = boost_scenario()
    check(boost.tasks["W1"].blocked_time == 0,
          "the weak task must not queue behind a boosted holder (%d)"
          % boost.tasks["W1"].blocked_time)
    check(boost.tasks["W2"].blocked_time > 0, "the strong task must be the one that waits")
    peak_o = max(snap["O"] for _, snap in boost.snapshots)
    check(peak_o == 12, "the holder must be boosted to the waiter's priority (saw %d)" % peak_o)

    # Extra edge cases: empty program, single task, no locks at all.
    idle = Simulator([Task("Z", 1, [])], [], inherit=True)
    check(idle.run() == 1 and idle.tasks["Z"].finished_at == 1,
          "an empty program finishes in one scheduling unit")
    check(idle.invariant_checks == 1 and idle.tasks["Z"].blocked_time == 0,
          "trivial task bookkeeping")
    solo = Simulator([Task("A", 7, [("lock", "R"), ("cpu", 3), ("unlock", "R")])],
                     [Lock("R")], inherit=True)
    check(solo.run() == 4, "a lone task takes lock + burst + release units")

    # Deadlock: two tasks acquire the same pair of locks in opposite order. D2 is the
    # stronger task, so it grabs R2 first and then waits for R1 - which D1 holds while it
    # is busy with a long burst inside its own critical section.
    d1 = Task("D1", 5, [("lock", "R1"), ("cpu", 12), ("lock", "R2"), ("cpu", 1),
                        ("unlock", "R2"), ("unlock", "R1")], release=0)
    d2 = Task("D2", 6, [("lock", "R2"), ("cpu", 1), ("lock", "R1"), ("cpu", 1),
                        ("unlock", "R1"), ("unlock", "R2")], release=1)
    dead = Simulator([d1, d2], [Lock("R1"), Lock("R2")], inherit=True)
    dead.run()
    check(dead.deadlock, "lock-order inversion beside locks must be detected")
    check(dead.clock < MAX_STEPS, "deadlock must stop the simulation, not spin (%d)" % dead.clock)
    check(dead.tasks["D1"].blocked_on == "R2" and dead.tasks["D2"].blocked_on == "R1",
          "both tasks must be parked on opposite locks")
    check(dead.snapshots and dead.invariant_checks == dead.clock,
          "invariants hold even in the deadlocked state")

    # Invalid input is rejected rather than silently corrupting state.
    bad = Simulator([Task("X", 1, [("unlock", "R")])], [Lock("R")], inherit=True)
    rejected = False
    try:
        bad.run()
    except AssertionError as exc:
        rejected = "does not own" in str(exc)
    check(rejected, "unlocking a semaphore without holding it must raise")

    # Determinism: identical scenarios produce identical traces.
    first, second = inversion_scenario(True), inversion_scenario(True)
    check(first.log == second.log and first.snapshots == second.snapshots,
          "the simulation must be deterministic")

    print("blocked without PIP = %d, with PIP = %d (medium work = %d)"
          % (blocked_no, blocked_yes, medium_work))
    print("invariant checks = %d, inversion units = %d -> %d"
          % (without.invariant_checks, without.inversion_units, with_pip.inversion_units))
    print("PASS")


if __name__ == "__main__":
    main()
