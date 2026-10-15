"""Task 481 - Implement optimistic checkpointing/restart simulation. (Tier: Advanced, CTCI Ch 14)

Model (documented semantics):
- A job applies one effect per step, step ids 0..n-1. Effects are idempotent: replaying a step
  never duplicates effect id `step` (apply() is a set insert).
- Optimistic checkpointing: after every `interval` completed steps (progress % interval == 0)
  the job snapshots (progress, effects) into a single checkpoint slot, overwriting the old one.
- A crash when "about to start step k" means steps 0..k-1 executed and the checkpoint slot is
  the last one taken among them. Recovery rolls back to that slot: effects of steps that were
  never checkpointed are lost (optimistic, uncommitted), while everything at or before the last
  checkpoint survives: no committed effect is ever lost.
- Recovered progress is exactly (k // interval) * interval, so the amount of redone work is
  k - recovered < interval steps (the documented redo bound). Redone steps must be at most one
  interval's worth and the final effect set must still equal every step id exactly once.

Scripted crashes plus LCG crash points assert recovery progress, survival of committed effects,
the redo bound, and full replay to the oracle effect set.
"""


class OptimisticJob:
    def __init__(self, n, interval):
        self.n = n
        self.interval = interval
        self.progress = 0
        self.effects = []
        self.checkpoint = (0, [])
        self.checkpoint_count = 0
        self.executions = 0

    def apply(self, step):
        self.executions += 1
        if step not in self.effects:
            self.effects.append(step)

    def run_until(self, k):
        """Execute steps 0..k-1, then crash while trying to start step k."""
        while self.progress < k:
            step = self.progress
            self.apply(step)
            self.progress = step + 1
            if self.progress % self.interval == 0:
                self.checkpoint = (self.progress, list(self.effects))
                self.checkpoint_count += 1

    def recover(self):
        progress, effects = self.checkpoint
        self.progress = progress
        self.effects = list(effects)

    def finish(self):
        self.run_until(self.n)

    @property
    def redone(self):
        return self.executions - len(self.effects)


def crash_points(count, seed, limit):
    points = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        points.append(seed % limit)
    return points


def verify_crash(n, interval, k):
    """Run a fresh job to crash point k and check every documented invariant."""
    job = OptimisticJob(n, interval)
    job.run_until(k)
    assert job.progress == k
    job.recover()
    expected = (k // interval) * interval
    assert job.progress == expected, (k, job.progress, expected)
    assert job.progress <= k
    assert k - job.progress < interval
    assert set(job.effects) == set(range(expected))
    job.finish()
    assert job.progress == n
    assert sorted(job.effects) == list(range(n))
    assert job.redone == k - expected, (k, job.redone)
    return job


def main():
    # Hand case: n = 25, checkpoint every 4 steps, crashes at scripted points.
    job = OptimisticJob(25, 4)
    job.run_until(7)
    assert job.progress == 7 and sorted(job.effects) == list(range(7))
    job.recover()
    assert job.progress == 4
    assert sorted(job.effects) == [0, 1, 2, 3]
    job.run_until(16)
    job.recover()
    assert job.progress == 16
    assert sorted(job.effects) == list(range(16))
    job.run_until(23)
    job.recover()
    assert job.progress == 20
    assert sorted(job.effects) == list(range(20))
    job.finish()
    assert job.progress == 25
    assert sorted(job.effects) == list(range(25))
    assert job.executions == 7 + 12 + 7 + 5
    assert job.redone == (7 - 4) + (23 - 20)
    assert job.checkpoint_count == 6
    assert job.checkpoint == (24, list(range(24)))

    # Every scheduled crash recovers to the last interval boundary and loses nothing committed.
    for k in [0, 3, 7, 8, 16, 23]:
        job = verify_crash(25, 4, k)
        assert job.progress == 25
    assert [(k, (k // 4) * 4) for k in [3, 7, 8, 16, 23]] == \
        [(3, 0), (7, 4), (8, 8), (16, 16), (23, 20)]

    # LCG crash points: same invariants with an oracle recomputation of the expected progress.
    checked = 0
    for k in crash_points(20, 481, 26):
        expected = (k // 4) * 4
        job = verify_crash(25, 4, k)
        assert job.progress == 25 and expected == (k // 4) * 4
        checked += 1
    assert checked == 20

    # Different interval: redo bound shrinks with the checkpoint frequency.
    for interval in (1, 3, 5):
        for k in range(0, 21):
            job = verify_crash(20, interval, k)
            assert job.redone == k - (k // interval) * interval
            assert job.redone < interval
    print("PASS")


if __name__ == "__main__":
    main()
