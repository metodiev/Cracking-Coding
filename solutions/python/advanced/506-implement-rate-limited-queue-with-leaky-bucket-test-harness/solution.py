"""Task 506 - Implement rate-limited queue with leaky-bucket test harness. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Two classic rate limiters on a discrete virtual clock, plus a test harness that drives them with
scripted and LCG-generated arrival patterns. The leaky bucket is a bounded FIFO (capacity C, leak
L per tick): at the start of tick t every job still queued is advanced by up to L slots, so a job
accepted at tick t when p jobs are already ahead of it departs at tick t + 1 + p // L; arrivals are
offered after the leak and are dropped (not queued) when the bucket is full. That makes the bucket
a shaper: it never lets more than L jobs leave per tick, whatever the arrival burst looked like.
The token bucket keeps a token count refreshed by L per tick up to C; an arrival consumes a token
and departs immediately, so it is a policer: a burst of C passes at one instant and afterwards only
L per tick. Both satisfy the same window law - at most C + L * W jobs are admitted in any window of
W ticks - but the token bucket allows instantaneous bursts where the leaky bucket does not.

Documented checks: a hand-computed schedule (5 jobs offered at tick 0 with C = 5, L = 1) must be
accepted in full at tick 0 and leave at ticks 1..5, one per tick, with the queue empty from tick 6;
with an always-backlogged producer the bucket must admit exactly L = 1 job per tick after the first
burst, keep the level pinned at C, and produce departures at every tick; the window law
admitted(W) <= C + L * W is verified by brute force over every window of an LCG stream of 200 ticks
with up to 4 arrivals per tick, together with the invariants level <= C, accepted + rejected ==
offered and departures + queue == accepted; a fine-grained leak of L = 3 must space departures with
never more than 3 per tick; the token bucket must satisfy the same window law while the leaky bucket
never has two departures in the same tick during the burst (shaper versus policer), and the closed
form for a backlogged token bucket (C at tick 0, then L per tick) must hold exactly.
"""


class LeakyBucketQueue:
    """Bounded FIFO with a per-tick leak. Arrivals after the leak; full means dropped."""

    def __init__(self, capacity, leak_per_tick=1):
        if capacity <= 0 or leak_per_tick <= 0:
            raise ValueError("capacity and leak rate must be positive")
        self.capacity = capacity
        self.leak = leak_per_tick
        self.queue = []
        self.now = 0
        self.accepted = []
        self.rejected = []
        self.departures = []                   # (tick, job)
        self.accepted_at = {}                  # job -> (acceptance tick, queue position then)

    @property
    def level(self):
        return len(self.queue)

    def step(self, arrivals=()):
        """One tick: leak L from the front, then offer `arrivals`. Returns (accepted, departed)."""
        departed = [self.queue.pop(0) for _ in range(min(self.leak, len(self.queue)))]
        self.departures.extend((self.now, job) for job in departed)
        taken = []
        for job in arrivals:
            if len(self.queue) < self.capacity:
                self.accepted_at[job] = (self.now, len(self.queue))
                self.queue.append(job)
                self.accepted.append(job)
                taken.append(job)
            else:
                self.rejected.append(job)
        self.now += 1
        return taken, departed

    def departure_tick(self, accepted_tick, position):
        """Closed form: a job accepted at `accepted_tick` with `position` jobs ahead of it leaves
        at accepted_tick + ceil((position + 1) / L), because each tick leaks L from the front."""
        return accepted_tick + (position + 1 + self.leak - 1) // self.leak

    def window_max(self, window):
        """Most departures observed in any half-open window of `window` ticks."""
        ticks = [tick for tick, _ in self.departures]
        worst = 0
        for start in range(0, self.now + 1):
            worst = max(worst, sum(1 for tick in ticks if start <= tick < start + window))
        return worst


class TokenBucket:
    """Policer: C tokens, L added per tick, one token per admitted job, immediate departure."""

    def __init__(self, capacity, refill_per_tick=1):
        if capacity <= 0 or refill_per_tick <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = capacity
        self.refill = refill_per_tick
        self.tokens = capacity
        self.now = 0
        self.accepted = []
        self.rejected = []
        self.departures = []                   # (tick, job)
        self.accepted_at = {}                  # job -> (acceptance tick, queue position then)
        self.same_tick_burst = 0

    def step(self, arrivals=()):
        if self.now > 0:
            self.tokens = min(self.capacity, self.tokens + self.refill)
        taken = []
        for job in arrivals:
            if self.tokens >= 1:
                self.tokens -= 1
                self.accepted.append(job)
                self.departures.append((self.now, job))
                taken.append(job)
            else:
                self.rejected.append(job)
        self.same_tick_burst = max(self.same_tick_burst,
                                   sum(1 for tick, _ in self.departures if tick == self.now))
        self.now += 1
        return taken

    def window_max(self, window):
        ticks = [tick for tick, _ in self.departures]
        worst = 0
        for start in range(0, self.now + 1):
            worst = max(worst, sum(1 for tick in ticks if start <= tick < start + window))
        return worst


def lcg(seed):
    return (seed * 75 + 74) % 65537


def lcg_arrivals(ticks, seed):
    """Up to 4 arrivals per tick, deterministic."""
    pattern = []
    job = 0
    for _ in range(ticks):
        seed = lcg(seed)
        count = seed % 5
        pattern.append(list(range(job, job + count)))
        job += count
    return pattern


def run_stream(bucket, pattern):
    for arrivals in pattern:
        bucket.step(arrivals)
    return bucket


def main():
    # --- Hand-computed schedule: 6 jobs offered at tick 0, C = 5, L = 1. ---------------------
    bucket = LeakyBucketQueue(5, 1)
    taken, departed = bucket.step([0, 1, 2, 3, 4, 5])
    assert taken == [0, 1, 2, 3, 4] and departed == [] and bucket.level == 5
    assert bucket.rejected == [5] and bucket.accepted_at[4] == (0, 4)
    schedule = [bucket.step([])[1] for _ in range(7)]
    assert schedule == [[0], [1], [2], [3], [4], [], []], schedule
    assert bucket.level == 0 and bucket.now == 8
    assert bucket.departures == [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)], bucket.departures
    assert bucket.window_max(1) == 1 and bucket.window_max(5) == 5 and bucket.window_max(20) == 5

    # The closed form matches that schedule, including a coarse leak of L = 3.
    fresh = LeakyBucketQueue(4, 1)
    fresh.step([10, 11, 12, 13])
    for _ in range(5):
        fresh.step([])
    assert [fresh.departure_tick(0, p) for p in range(4)] == [1, 2, 3, 4] == \
        [tick for tick, _ in fresh.departures]
    assert fresh.level == 0
    fine = LeakyBucketQueue(10, 3)
    fine.step(list(range(9)))
    for _ in range(4):
        fine.step([])
    assert [fine.departure_tick(0, p) for p in range(9)] == [1, 1, 1, 2, 2, 2, 3, 3, 3] == \
        [tick for tick, _ in fine.departures]

    # --- Always-backlogged producer: exactly L admissions per tick after the burst. ----------
    backlog = LeakyBucketQueue(5, 1)
    accepted_sizes = []
    for job in range(100, 131):
        accepted_sizes.append(len(backlog.step([job])[0]))
    assert set(accepted_sizes) == {1}, "a backlogged leaky bucket admits exactly one per tick"
    assert len(backlog.departures) == 30 and backlog.level == 1
    assert backlog.window_max(10) == 10
    backlog.step([])
    assert len(backlog.departures) == 31 and backlog.level == 0
    heavy = LeakyBucketQueue(5, 1)
    heavy_sizes = []
    for job in range(0, 60, 2):
        heavy_sizes.append(len(heavy.step([job, job + 1])[0]))
    assert heavy_sizes == [2, 2, 2, 2] + [1] * 26, heavy_sizes
    assert heavy.level <= 5 and len(heavy.rejected) == 26
    assert heavy.window_max(10) <= 5 + 10
    # --- LCG stream: invariants and the window law. ------------------------------------------
    pattern = lcg_arrivals(200, seed=99)
    offerings = sum(len(a) for a in pattern)
    assert offerings > 400
    for capacity, leak in ((5, 1), (8, 2), (3, 3)):
        bucket = run_stream(LeakyBucketQueue(capacity, leak), pattern)
        # Every accepted job leaves exactly when the closed form says it does.
        departure_ticks = {job: tick for tick, job in bucket.departures}
        for job in bucket.accepted:
            accepted_tick, position = bucket.accepted_at[job]
            if job in departure_ticks:
                assert departure_ticks[job] == bucket.departure_tick(accepted_tick, position)
        assert len(bucket.accepted) - len(departure_ticks) == bucket.level
        assert len(bucket.accepted) + len(bucket.rejected) == offerings
        assert len(bucket.accepted) == len(bucket.departures) + bucket.level
        assert bucket.level <= capacity
        assert bucket.now == 200
        for window in (1, 5, 20):
            assert bucket.window_max(window) <= capacity + leak * window, (capacity, leak, window)
        assert bucket.window_max(1) <= leak, "shaper: never more than L departures in a tick"
    fine_stream = run_stream(LeakyBucketQueue(10, 3), pattern)
    assert fine_stream.window_max(1) <= 3
    # A rate below the leak rate loses nothing.
    gentle = LeakyBucketQueue(4, 2)
    for _ in range(50):
        taken = gentle.step([1])[0]
        assert taken == [1]
    assert gentle.rejected == []

    # --- Token bucket: same window law, but it allows instantaneous bursts. ------------------
    token = TokenBucket(5, 1)
    burst = token.step(list(range(20)))
    assert burst == list(range(5)), "the token bucket releases the burst at once"
    assert token.same_tick_burst == 5 and len(token.rejected) == 15
    leaky = LeakyBucketQueue(5, 1)
    leaky.step(list(range(20)))
    assert len(leaky.accepted) == 5 and len(leaky.rejected) == 15
    for _ in range(6):
        leaky.step([])
    assert leaky.window_max(1) == 1, "the leaky bucket staggers the same burst over five ticks"
    assert len({tick for tick, _ in leaky.departures}) == 5
    assert len({tick for tick, _ in token.departures}) == 1

    token_stream = run_stream(TokenBucket(6, 1), pattern)
    assert len(token_stream.accepted) + len(token_stream.rejected) == offerings
    for window in (1, 5, 20):
        assert token_stream.window_max(window) <= 6 + 1 * window
    # Backlogged token bucket: C at the first tick, then exactly L per tick.
    tokens = TokenBucket(4, 1)
    counts = [len(tokens.step([tokens.now + 500])) for _ in range(10)]
    assert counts == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert sum(counts) == 10 == len(tokens.accepted)
    refill3 = TokenBucket(7, 3)
    refill_counts = [len(refill3.step(list(range(refill3.now * 10, refill3.now * 10 + 9))))
                     for _ in range(6)]
    assert refill_counts == [7, 3, 3, 3, 3, 3], refill_counts


if __name__ == "__main__":
    main()
    print("PASS")
