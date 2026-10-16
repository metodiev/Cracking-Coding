"""Task 508 - Implement background job scheduler with retry/backoff semantics. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

A deterministic background job scheduler on a virtual clock. Jobs carry an attempt budget
(max_attempts), an exponential-backoff base (base_delay), a cap (max_delay) and a jitter fraction.
The scheduler keeps a binary heap keyed by (run_at, sequence number) - the sequence number breaks
ties so two jobs due at the same instant run in submission order, which makes the whole trace
reproducible - and pops events one at a time, advancing its virtual clock to the popped timestamp.
A failed attempt schedules the next one at

    now + min(max_delay, base_delay * 2^(attempt - 1)) + jitter,

where jitter = floor(fraction * delay * u) and u is a deterministic value in [0, 1) derived from a
little LCG over (job id, attempt). Jitter is therefore monotone-friendly (it never shortens a delay),
bounded by fraction * delay, and identical on every replay. A job that exhausts its budget moves to
the dead-letter queue with its attempt history and the last error; a job that succeeds stops
retrying and stops occupying the scheduler.

Documented checks: with a zero jitter fraction the run times are the exact closed form
0, +b, +3b, +7b, ... capped at max_delay (base 100 / cap 1000 gives 0, 100, 300, 700, 1500, 2500);
every scheduled delay is monotone non-decreasing until the cap and equal to the cap afterwards; the
jitter is deterministic (two identical runs share every timestamp), bounded
(0 <= jitter < fraction * delay) and actually varies across jobs and attempts; a job that is set to
fail until its k-th attempt finishes at exactly the sum of its first k - 1 delays, and a job that
never succeeds ends in the dead-letter queue with exactly max_attempts attempts and its last error;
the event trace is sorted by time and, within a time, by submission order; a job submitted later with
a short delay overtakes an earlier job with a huge cap (priority by due time, not by arrival); the
counters are conserved (attempts == successes + failures + dead-letter attempts); and the
closed-form cumulative run time matches the trace for every job.
"""

import heapq

MAX_STEPS = 1000


def lcg(seed):
    return (seed * 75 + 74) % 65537


def jitter_unit(job_id, attempt):
    """Deterministic uniform in [0, 1) for one (job, attempt) pair."""
    seed = lcg(lcg(job_id * 131 + attempt * 8191 + 17) + attempt)
    return (seed % 4096) / 4096.0


class Job:
    __slots__ = ("job_id", "base_delay", "max_delay", "jitter_fraction", "max_attempts",
                 "attempts", "next_run_at", "state", "last_error", "runs", "success_at")

    def __init__(self, job_id, base_delay=100, max_delay=10000, jitter_fraction=0.0,
                 max_attempts=5):
        self.job_id = job_id
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_fraction = jitter_fraction
        self.max_attempts = max_attempts
        self.attempts = 0
        self.next_run_at = 0
        self.state = "pending"
        self.last_error = None
        self.runs = []                       # timestamps of the attempts
        self.success_at = None

    def delay_for(self, attempt):
        """Exponential backoff, capped."""
        return min(self.max_delay, self.base_delay * 2 ** (attempt - 1))

    def jitter_for(self, attempt):
        return int(self.jitter_fraction * self.delay_for(attempt) * jitter_unit(self.job_id, attempt))

    def plan_next(self, now, attempt):
        """Timestamp of the attempt after a failure of attempt number `attempt`."""
        return now + self.delay_for(attempt) + self.jitter_for(attempt)


class Scheduler:
    """Single-threaded event loop over a (run_at, sequence) heap."""

    def __init__(self, outcomes):
        """`outcomes` maps (job_id, attempt) -> None for success or a string error."""
        self.outcomes = outcomes
        self.heap = []
        self.sequence = 0
        self.jobs = {}
        self.trace = []                      # (run_at, job_id, attempt, ok, error)
        self.dead_letter = []
        self.attempts = 0

    def submit(self, job, at=0):
        job.next_run_at = at
        job.runs.append(at)
        heapq.heappush(self.heap, (at, self.sequence, job.job_id))
        self.sequence += 1
        self.jobs[job.job_id] = job
        return job

    def run(self):
        while self.heap:
            now, _, job_id = heapq.heappop(self.heap)
            job = self.jobs[job_id]
            assert job.runs[-1] == now, "a job may only run at its scheduled timestamp"
            job.attempts += 1
            self.attempts += 1
            error = self.outcomes.get((job_id, job.attempts))
            ok = error is None
            self.trace.append((now, job_id, job.attempts, ok, error))
            if ok:
                job.state = "succeeded"
                job.success_at = now
                continue
            job.last_error = error
            if job.attempts >= job.max_attempts:
                job.state = "dead"
                self.dead_letter.append(job_id)
                continue
            next_at = job.plan_next(now, job.attempts)
            job.runs.append(next_at)
            job.next_run_at = next_at
            heapq.heappush(self.heap, (next_at, self.sequence, job_id))
            self.sequence += 1


def closed_form_runs(job, attempts):
    """Timestamps of the first `attempts` attempts, derived from the open form only."""
    times = [0]
    for attempt in range(1, attempts):
        times.append(times[-1] + job.delay_for(attempt) + job.jitter_for(attempt))
    return times


def always_fails(job_ids, max_attempts):
    return {(job_id, attempt): "boom-%d-%d" % (job_id, attempt)
            for job_id in job_ids for attempt in range(1, max_attempts + 1)}


def main():
    # --- Zero jitter: exact closed-form schedule. --------------------------------------------
    job = Job(1, base_delay=100, max_delay=1000, jitter_fraction=0.0, max_attempts=8)
    assert [job.delay_for(a) for a in range(1, 8)] == [100, 200, 400, 800, 1000, 1000, 1000]
    scheduler = Scheduler(always_fails([1], 8))
    scheduler.submit(job)
    scheduler.run()
    assert [entry[0] for entry in scheduler.trace] == [0, 100, 300, 700, 1500, 2500, 3500, 4500]
    assert [entry[0] for entry in scheduler.trace] == closed_form_runs(job, 8)
    assert job.state == "dead" and scheduler.dead_letter == [1]
    assert job.last_error == "boom-1-8" and job.attempts == 8 == job.max_attempts
    delays = [b - a for a, b in zip([entry[0] for entry in scheduler.trace],
                                    [entry[0] for entry in scheduler.trace][1:])]
    assert delays == [100, 200, 400, 800, 1000, 1000, 1000]
    for i in range(1, 4):
        assert delays[i] == 2 * delays[i - 1], "doubling before the cap"
    for i in range(4, 7):
        assert delays[i] == 1000, "capped at max_delay"

    # --- Success on the k-th attempt: the job stops at the exact closed-form time. ------------
    for success_on in (1, 2, 3, 6):
        outcomes = {(2, attempt): "flaky" for attempt in range(1, success_on)}
        job2 = Job(2, base_delay=50, max_delay=100000, jitter_fraction=0.0, max_attempts=10)
        sched2 = Scheduler(outcomes)
        sched2.submit(job2)
        sched2.run()
        assert len(sched2.trace) == success_on
        assert job2.state == "succeeded" and job2.success_at == sched2.trace[-1][0]
        assert job2.success_at == closed_form_runs(job2, success_on)[-1]
        assert job2.attempts == success_on and job2.attempts < job2.max_attempts
        assert sched2.dead_letter == []
    assert closed_form_runs(Job(3, base_delay=50, max_delay=10 ** 6, jitter_fraction=0.0), 4) == \
        [0, 50, 150, 350]

    # --- Jitter: deterministic, bounded and actually non-zero. -------------------------------
    jittered = Job(7, base_delay=200, max_delay=5000, jitter_fraction=0.5, max_attempts=6)
    fractions = [jittered.jitter_for(attempt) for attempt in range(1, 7)]
    assert fractions == [jittered.jitter_for(attempt) for attempt in range(1, 7)]
    for attempt, value in enumerate(fractions, start=1):
        assert 0 <= value < 0.5 * jittered.delay_for(attempt), (attempt, value)
    assert any(value > 0 for value in fractions), "jitter must actually jitter"
    assert len(set(fractions[1:])) > 1, "jitter must vary with the attempt"
    other = Job(8, base_delay=200, max_delay=5000, jitter_fraction=0.5, max_attempts=6)
    assert [other.jitter_for(a) for a in range(1, 7)] != fractions, "jitter must vary with the job"

    outcomes = always_fails([7], 6)
    sched_j = Scheduler(outcomes)
    sched_j.submit(jittered)
    sched_j.run()
    sched_replay = Scheduler(outcomes)
    sched_replay.submit(Job(7, base_delay=200, max_delay=5000, jitter_fraction=0.5, max_attempts=6))
    sched_replay.run()
    assert [e[0] for e in sched_j.trace] == [e[0] for e in sched_replay.trace], "replay determinism"
    plain = Job(7, base_delay=200, max_delay=5000, jitter_fraction=0.0, max_attempts=6)
    sched_plain = Scheduler(outcomes)
    sched_plain.submit(plain)
    sched_plain.run()
    jitter_times = [e[0] for e in sched_j.trace]
    plain_times = [e[0] for e in sched_plain.trace]
    assert all(j >= p for j, p in zip(jitter_times, plain_times))
    assert any(j > p for j, p in zip(jitter_times, plain_times))
    assert jitter_times == closed_form_runs(jittered, 6)

    # --- Ordering: due time first, submission order for ties. --------------------------------
    outcomes = {(101, 1): "a", (101, 2): "b", (100, 1): "c", (100, 2): "d", (100, 3): "e"}
    quick = Job(101, base_delay=10, max_delay=40, jitter_fraction=0.0, max_attempts=3)
    slow = Job(100, base_delay=10, max_delay=500, jitter_fraction=0.0, max_attempts=3)
    third = Job(105, base_delay=5, max_delay=5, jitter_fraction=0.0, max_attempts=2)
    sched3 = Scheduler(outcomes)
    sched3.submit(quick)                     # submitted first, due at 0
    sched3.submit(slow)                      # also due at 0: FIFO tie-break
    sched3.submit(third, at=12)              # later arrival, succeeds first time
    sched3.run()
    times = [entry[0] for entry in sched3.trace]
    assert times == sorted(times), "events must be processed in non-decreasing time order"
    assert times == [0, 0, 10, 10, 12, 30, 30]
    at_zero = [entry[1] for entry in sched3.trace if entry[0] == 0]
    assert at_zero == [101, 100], "ties keep submission order"
    assert sched3.trace[0] == (0, 101, 1, False, "a")
    assert (12, 105) == (sched3.trace[4][0], sched3.trace[4][1])
    assert sched3.trace[4][3] is True and third.attempts == 1
    assert quick.success_at == 30 and quick.attempts == 3
    assert slow.state == "dead" and sched3.dead_letter == [100]
    assert slow.attempts == 3 and slow.last_error == "e"
    # The 105 job only needs one slot and runs before either backoff chain can retry again.
    assert sched3.trace[4][0] < sched3.trace[5][0]

    # The counters are conserved.
    assert sched3.attempts == len(sched3.trace) == sum(
        job.attempts for job in (quick, slow, third))
    successes = sum(1 for entry in sched3.trace if entry[3])
    failures = len(sched3.trace) - successes
    assert successes == 2 and failures == 5

    # --- A cap of zero delays plus an immediate success is the degenerate case. --------------
    instant = Job(9, base_delay=0, max_delay=0, jitter_fraction=0.0, max_attempts=3)
    sched_instant = Scheduler({(9, 1): "no", (9, 2): "no"})
    sched_instant.submit(instant)
    sched_instant.run()
    assert [entry[0] for entry in sched_instant.trace] == [0, 0, 0]
    assert instant.state == "succeeded" and instant.attempts == 3
    empty = Scheduler({})
    empty.run()
    assert empty.trace == [] and empty.dead_letter == []


if __name__ == "__main__":
    main()
    print("PASS")
