"""Task 658 - Implement straggler mitigation via speculative execution simulation.  (Tier: Expert, Ch 9 - System Design and Scalability)

A tick-accurate mini MapReduce scheduler is simulated. The job is a list of tasks with known work
sizes; the cluster is a fixed set of slots, each advancing its current task by a fixed progress
quantum per tick (a slow machine advances by a fraction of the quantum, so a task on it takes
`slowdown` times as long). Idle slots pull tasks from a FIFO queue, and as soon as a task finishes
every other attempt of the same task is killed and its slot freed - a task therefore always produces
its result exactly once and the model reproduces the classic FIFO list schedule on a healthy cluster.

Speculative execution is the backup-task policy: at the start of every tick, a task whose single
running attempt has been running for at least `min_runtime` ticks and whose observed progress *rate*
is below `factor` times the healthy quantum (the reference rate of a healthy machine, i.e. the
progress-rate heuristic used by LATE) is treated as a straggler and a duplicate attempt is started on
an idle slot (at most `max_speculative` duplicates per task). Since an
attempt advances independently of the other slots, duplicating work can never slow the original
attempt down: the makespan with speculation is never worse than without it, which is the invariant
checked over many seeds, while the price is duplicated work, bounded by the number of backups times
the largest task.

Simplified: no network, data locality or shuffle model (any slot can run any task), progress is a
deterministic function of the tick rather than counters read from workers, and stragglers are injected
by fixed-seed slowdowns of specific machines instead of arising from real hardware variance. Verified:
the healthy run equals an independently computed FIFO list schedule, every task completes exactly once,
killed attempts always belong to already-finished tasks, at most 1 + max_speculative attempts per task,
no backup before `min_runtime`, healthy equal tasks trigger no speculation, a straggler really is
rescued, makespan(speculation) <= makespan(no speculation) for every seed, the duplicated-work bound,
an eager threshold launches at least as many backups, and determinism for a fixed seed.
"""

QUANTUM = 1000          # progress units a healthy machine applies per tick
MAX_TICKS = 200000      # safety valve: the simulation must never spin forever


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def fifo_makespan(work, slot_slowdowns):
    """Independent reference schedule: each task goes to the slot that frees up earliest."""
    free = [0] * len(slot_slowdowns)
    for size in work:
        index = free.index(min(free))
        free[index] += size * slot_slowdowns[index]
    return max(free)


class Simulator:
    """Deterministic slot scheduler with optional speculative execution."""

    def __init__(self, task_work, slot_slowdowns, speculate=True, max_speculative=1,
                 min_runtime=2, factor=0.5):
        self.task_work = list(task_work)
        self.task_count = len(self.task_work)
        self.slots = len(slot_slowdowns)
        self.speculate = speculate
        self.max_speculative = max_speculative
        self.min_runtime = min_runtime
        self.factor = factor
        self.speed = [QUANTUM // slowdown for slowdown in slot_slowdowns]

        self.slot_task = [None] * self.slots
        self.attempts = {}
        self.queue = list(range(self.task_count))
        self.completed = {}
        self.killed = []
        self.launches = []          # tick at which each backup attempt was launched
        self.births = []            # (original start tick) for every launch, for policy checks
        self.work_done = 0
        self.makespan = 0
        self.now = 0

    # ------------------------------------------------------------------ helpers
    def _running(self, task):
        return [attempt for attempt in self.attempts[task] if attempt["state"] == "running"]

    def _start(self, task, speculative):
        if speculative:
            # a backup is placed on the fastest idle machine (real schedulers blacklist slow nodes)
            idle = [index for index, current in enumerate(self.slot_task) if current is None]
            slot = max(idle, key=lambda index: (self.speed[index], -index))
        else:
            slot = self.slot_task.index(None)
        attempt = {"slot": slot, "progress": 0, "total": self.task_work[task] * QUANTUM,
                   "speed": self.speed[slot], "state": "running", "start": self.now,
                   "speculative": speculative}
        self.slot_task[slot] = task
        self.attempts.setdefault(task, []).append(attempt)
        if speculative:
            self.launches.append(self.now)
        return attempt

    def _launch_backups(self):
        """Launch a backup for any task whose observed progress rate falls below the threshold."""
        while None in self.slot_task:
            chosen = None
            for task in range(self.task_count):
                if task in self.completed:
                    continue
                running = self._running(task)
                if not running or len(running) >= 1 + self.max_speculative:
                    continue
                original = running[0]
                elapsed = self.now - original["start"]
                if elapsed < self.min_runtime or elapsed == 0:
                    continue
                rate = original["progress"] / elapsed
                if rate < self.factor * QUANTUM:
                    chosen = (task, original)
                    break
            if chosen is None:
                return
            task, original = chosen
            check(self.now - original["start"] >= self.min_runtime,
                  "a speculative copy may not start before the minimum runtime")
            self.births.append((task, original["start"]))
            self._start(task, speculative=True)

    def _finish(self, task, attempt):
        self.completed[task] = self.now
        for other in self.attempts[task]:
            if other is attempt:
                other["state"] = "won"
            elif other["state"] == "running":
                other["state"] = "killed"
                self.killed.append(other)
            self.slot_task[other["slot"]] = None

    # ------------------------------------------------------------------ main loop
    def run(self):
        while len(self.completed) < self.task_count:
            check(self.now < MAX_TICKS, "the simulation failed to finish")
            while self.queue and None in self.slot_task:
                self._start(self.queue.pop(0), speculative=False)
            if self.speculate:
                self._launch_backups()
            self.now += 1
            finished = []
            for task, attempt in ((task, attempt)
                                  for task, attempts in self.attempts.items()
                                  for attempt in attempts):
                if attempt["state"] != "running":
                    continue
                attempt["progress"] += attempt["speed"]
                self.work_done += attempt["speed"]
                if attempt["progress"] >= attempt["total"]:
                    finished.append((task, attempt))
            for task, attempt in finished:
                if task not in self.completed:
                    self._finish(task, attempt)
            self.makespan = self.now
        return self

    def wasted_work(self):
        return sum(attempt["progress"] for attempt in self.killed)


def simulate(task_work, slot_slowdowns, **policy):
    return Simulator(task_work, slot_slowdowns, **policy).run()


def random_job(seed, task_count, slow_slots, slowdown, workers, min_work=10, spread=30):
    """Task sizes and slow machines from the PRNG (deterministic for a given seed)."""
    work = []
    for _ in range(task_count):
        seed, r = splitmix64(seed)
        work.append(min_work + r % spread)
    slot_slowdowns = [1] * workers
    for _ in range(slow_slots):
        seed, r = splitmix64(seed)
        slot_slowdowns[r % workers] = slowdown
    return work, slot_slowdowns, seed


def main():
    # 1. degenerate cases
    check(simulate([], [1, 1]).makespan == 0, "a job with no tasks finishes immediately")
    single = simulate([7], [1])
    check(single.makespan == 7, "one task on one healthy slot takes exactly its work")
    check(single.completed == {0: 7}, "the task must record its finish tick")
    wide = simulate([5, 5, 5], [1, 1, 1, 1])
    check(wide.makespan == 5, "with enough slots every task runs in parallel")
    check(wide.launches == [], "no speculation is needed when slots are abundant")
    check(fifo_makespan([], [1, 1]) == 0, "the reference schedule of an empty job is zero")

    # 2. the healthy run reproduces the independent FIFO list schedule
    for work, slots in (([20] * 6, [1, 1]), ([40, 25, 18, 12, 9, 7, 5, 3], [1, 1, 1]),
                        ([3, 1, 4, 1, 5, 9, 2, 6], [1, 1, 1, 1, 1]),
                        ([9, 9, 9], [1]), ([12, 6, 6, 4, 4, 4], [1, 1, 1])):
        healthy = simulate(work, slots)
        check(healthy.makespan == fifo_makespan(work, slots),
              f"the simulation must match the FIFO list schedule for {work}")
        check(healthy.launches == [], "healthy slots must never trigger a backup")
        check(healthy.killed == [], "nothing may be killed when nothing is duplicated")
        check(healthy.work_done == sum(work) * QUANTUM, "each task's work must be done exactly once")
        check(healthy.makespan >= -(-sum(work) // len(slots)),
              "the makespan cannot beat total work over the number of slots")
    check(simulate([20] * 6, [1, 1]).makespan == 60, "six tasks of 20 on two slots take 60 ticks")

    # 3. a straggler machine is rescued by the backup attempt
    work = [20] * 4
    plain = simulate(work, [5, 1, 1, 1], speculate=False)
    rescued = simulate(work, [5, 1, 1, 1], speculate=True)
    check(plain.makespan == 100, "the slow machine alone stretches the job to 5x the work")
    check(rescued.makespan <= plain.makespan, "speculation must never make the job slower")
    check(rescued.makespan < plain.makespan, "the backup attempt must actually rescue the job")
    check(len(rescued.launches) >= 1, "a straggler must trigger at least one backup copy")
    check(all(tick >= 1 for tick in rescued.launches), "speculation must wait for the min runtime")
    check(sorted(rescued.completed) == [0, 1, 2, 3], "every task must finish exactly once")
    check(len(rescued.killed) >= 1, "the losing duplicate must be killed")
    check(rescued.wasted_work() > 0, "the killed duplicate must have burned some work")

    # 4. attempt invariants under a realistic mixed workload
    work, slot_slowdowns, seed = random_job(0x1234, 24, 3, 6, 6)
    run = simulate(work, slot_slowdowns, max_speculative=1, min_runtime=2, factor=1.0)
    check(len(slot_slowdowns) == 6, "six slots were configured")
    for task, attempts in run.attempts.items():
        check(len(attempts) <= 1 + run.max_speculative, "a task may have at most one backup attempt")
        winners = [attempt for attempt in attempts if attempt["state"] == "won"]
        check(len(winners) == 1, "exactly one attempt per task may win")
        check(task in run.completed, "a won attempt implies the task completed")
        for attempt in attempts:
            check(0 <= attempt["slot"] < run.slots, "an attempt must run in a real slot")
            if attempt["state"] == "killed":
                check(task in run.completed, "an attempt is only killed once its task finished")
                check(attempt["start"] <= run.completed[task],
                      "a killed attempt cannot start after the task finished")
    check(len(run.completed) == len(work), "every task must complete")
    check(run.killed.__len__() == len(run.launches),
          "with one backup per task every launch is either won or killed")
    check(sum(len(attempts) for attempts in run.attempts.values()) == len(work) + len(run.launches),
          "every started attempt is either an original or a launched backup")
    check(len(run.births) == len(run.launches), "every launch must record the attempt it duplicates")
    for task, start in run.births:
        check(start in [attempt["start"] for attempt in run.attempts[task]],
              "a backup must be pinned to an attempt of its own task")

    # 5. makespan bounds and the duplicated-work accounting over many seeds
    improved = 0
    for trial in range(12):
        work, slot_slowdowns, seed = random_job(0x5000 + trial, 30, 3, 8, 6)
        base = simulate(work, slot_slowdowns, speculate=False)
        fast = simulate(work, slot_slowdowns, speculate=True)
        check(fast.makespan <= base.makespan, f"speculation must never be slower (seed {trial})")
        if fast.makespan < base.makespan:
            improved += 1
        check(sorted(fast.completed) == list(range(len(work))), "every task must complete")
        check(base.work_done == sum(work) * QUANTUM,
              "without speculation each task's work is done exactly once")
        check(fast.work_done - base.work_done == fast.wasted_work(),
              "the extra work must be exactly the progress of the killed attempts")
        check(fast.wasted_work() <= len(fast.launches) * max(work) * QUANTUM,
              "duplicated work is bounded by backups times the largest task")
        check(fast.makespan >= -(-sum(work) // 6), "the makespan cannot beat the work lower bound")
    check(improved >= 9, f"speculation must help the straggler seeds (helped {improved}/12)")

    # 6. policy knobs: an eager threshold launches at least as many backups and never delays the job
    work, slot_slowdowns, seed = random_job(0x9ABC, 30, 4, 10, 6)
    lazy = simulate(work, slot_slowdowns, min_runtime=8, factor=0.05)
    eager = simulate(work, slot_slowdowns, min_runtime=1, factor=0.5)
    off = simulate(work, slot_slowdowns, speculate=False)
    check(len(eager.launches) >= len(lazy.launches),
          "an eager threshold must launch at least as many backups")
    check(eager.makespan <= lazy.makespan, "an eager threshold must not delay the job")
    check(lazy.makespan <= off.makespan, "even a lazy policy may not be slower than no speculation")
    check(all(run_task in range(len(work)) for run_task in eager.completed),
          "every task belongs to the job")
    print(f"658 slots=6 slow={len([s for s in slot_slowdowns if s > 1])} "
          f"backups lazy={len(lazy.launches)} eager={len(eager.launches)}")

    # 7. no speculation happens when every slot is healthy even with uneven task sizes
    uneven = [40, 25, 18, 12, 9, 7, 5, 3]
    diligent = simulate(uneven, [1, 1, 1])
    check(diligent.launches == [], "a healthy job with uneven tasks still needs no backup")
    check(diligent.makespan == fifo_makespan(uneven, [1, 1, 1]),
          "the healthy makespan must equal the reference schedule")
    check(diligent.makespan <= sum(uneven), "a healthy job cannot take longer than the serial time")

    # 8. determinism: identical seed, policy and cluster give identical metrics
    snapshots = []
    for _ in range(2):
        repeat = simulate(work, slot_slowdowns, max_speculative=1, min_runtime=2, factor=1.0)
        snapshots.append((repeat.makespan, repeat.work_done, tuple(repeat.launches),
                          tuple(sorted(repeat.completed.items())), len(repeat.killed),
                          repeat.wasted_work()))
    check(snapshots[0] == snapshots[1], "the simulator must be deterministic")

    # 9. max_speculative = 0 disables duplication but keeps the schedule valid
    strict = simulate(work, slot_slowdowns, max_speculative=0)
    check(strict.launches == [], "no backup may be launched when the cap is zero")
    check(strict.makespan == simulate(work, slot_slowdowns, speculate=False).makespan,
          "a zero cap must behave exactly like speculation switched off")
    generous = simulate(work, slot_slowdowns, max_speculative=2)
    check(all(len(attempts) <= 3 for attempts in generous.attempts.values()),
          "the per-task attempt cap must hold for a larger budget")

    print(f"658 rescued makespan plain={plain.makespan} speculative={rescued.makespan} "
          f"wasted={rescued.wasted_work()}")
    print(f"658 mixed job makespan plain={off.makespan} lazy={lazy.makespan} eager={eager.makespan}")
    print("PASS")


if __name__ == "__main__":
    main()
