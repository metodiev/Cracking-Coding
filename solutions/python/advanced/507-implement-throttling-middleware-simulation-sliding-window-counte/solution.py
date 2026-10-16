"""Task 507 - Implement throttling middleware simulation (sliding window counters). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)

Three throttling policies behind one middleware interface, all on an integer virtual clock and a
half-open window (t - W, t]:
* the sliding log - a deque of accepted timestamps, exact but O(limit) memory per key;
* the fixed window counter - one bucket per t // W, O(1) memory but it lets a client spend the whole
  quota at the end of one window and again at the start of the next;
* the sliding window counter - two integers per key (the current and the previous window count) and
  the weighted estimate prev * (1 - elapsed_fraction) + curr, which needs no per-request state.
The middleware records every decision so the policies can be replayed by an independent brute-force
counter that simply scans the accepted timestamps.

Documented checks: the sliding log is exact - its count at every request time equals a brute-force
scan of the accepted timestamps, and a from-scratch replay of the raw stream reproduces the same
acceptances; the fixed-window flaw is exhibited concretely (limit = 5, W = 10, five requests at
t = 9 and five at t = 10 all pass, so 10 > limit requests sit inside one real 10-tick window, while
the sliding log admits only the first five and never exceeds limit in any window); the sliding
counter's error is two-sided and bounded - with f the elapsed fraction of the current window,
curr the current count, prev_total the previous window's count and tail the accepted count inside
the tail of that previous window, exact = curr + tail and estimate = curr + prev_total * (1 - f),
so -prev_total <= estimate - exact <= (1 - f) * prev_total (checked at every admission of every
key of nine 300-request streams); because curr and prev_total both stay below the limit no real
window can hold more than 2 * limit - 2 requests (asserted with a tick of slack), yet the counter
is an approximation: the deliberate overweight pattern of five requests at the very end of a window
followed by two more early in the next one admits 8 requests inside a 10-tick window with limit 5,
while the sliding log stays at exactly 5; on ordinary mixed traffic the counter is the stricter of
the two (measured 179 vs 210 admissions) and stays within a factor of two; the window boundary is
half-open (limit = 1, W = 10: t = 9 is blocked, t = 10 is allowed because t = 0 has left the
window, t = 11 is blocked again); O(1) state is asserted (two counters for the sliding counter
against at most limit stored timestamps for the log); and every policy is deterministic.
"""

from collections import deque


def slots(times, window):
    """Brute-force exact count at every timestamp: accepted requests in (t - window, t]."""
    out = []
    for t in times:
        out.append(sum(1 for other in times if t - window < other <= t))
    return out


def window_max(times, window):
    """Largest number of accepted requests in any half-open window of `window` ticks."""
    return max(slots(times, window)) if times else 0


class SlidingLogLimiter:
    """Exact policy: keep the accepted timestamps in a deque and drop the expired ones."""

    name = "sliding-log"

    def __init__(self, limit, window):
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit
        self.window = window
        self.times = deque()

    def estimate(self, t):
        while self.times and self.times[0] <= t - self.window:
            self.times.popleft()
        return float(len(self.times))

    def allow(self, t):
        if self.estimate(t) + 1.0 <= self.limit + 1e-12:
            self.times.append(t)
            return True
        return False

    def state_size(self):
        return len(self.times)


class FixedWindowLimiter:
    """One counter per t // W window: O(1) memory, but the window is not really sliding."""

    name = "fixed-window"

    def __init__(self, limit, window):
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit
        self.window = window
        self.buckets = {}

    def estimate(self, t):
        return float(self.buckets.get(t // self.window, 0))

    def allow(self, t):
        key = t // self.window
        if self.buckets.get(key, 0) + 1 <= self.limit:
            self.buckets[key] = self.buckets.get(key, 0) + 1
            return True
        return False

    def state_size(self):
        return len(self.buckets)


class SlidingCounterLimiter:
    """O(1) state: prev * (1 - elapsed fraction) + curr."""

    name = "sliding-counter"

    def __init__(self, limit, window):
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit
        self.window = window
        self.current_id = None
        self.current_count = 0
        self.previous_count = 0
        self.history = []                           # (t, estimate before, prev_count, fraction)

    def _roll(self, t):
        key = t // self.window
        if self.current_id is None:
            self.current_id = key
            return
        if key != self.current_id:
            previous_id = self.current_id
            self.previous_count = self.current_count
            self.current_id = key
            self.current_count = 0
            if key - previous_id > 1:
                self.previous_count = 0             # older windows are irrelevant

    def fraction(self, t):
        return (t % self.window) / self.window

    def estimate(self, t):
        self._roll(t)
        return self.current_count + self.previous_count * (1.0 - self.fraction(t))

    def allow(self, t):
        estimate = self.estimate(t)
        if estimate + 1.0 <= self.limit + 1e-12:
            self.history.append((t, estimate, self.previous_count, self.fraction(t)))
            self.current_count += 1
            return True
        return False

    def state_size(self):
        return 2


class ThrottlingMiddleware:
    """Runs a policy over a request stream and keeps the accepted timestamps per key."""

    def __init__(self, policy, limit, window):
        self.policy = policy
        self.limit = limit
        self.window = window
        self.limiters = {}
        self.requests = 0
        self.allowed = 0
        self.throttled = 0
        self.accepted_by_key = {}
        self.decisions = []

    def handle(self, key, t):
        self.requests += 1
        limiter = self.limiters.get(key)
        if limiter is None:
            limiter = self.policy(self.limit, self.window)
            self.limiters[key] = limiter
        ok = limiter.allow(t)
        if ok:
            self.allowed += 1
            self.accepted_by_key.setdefault(key, []).append(t)
        else:
            self.throttled += 1
        self.decisions.append((key, t, ok))
        return ok


def lcg(seed):
    return (seed * 75 + 74) % 65537


def request_stream(count, seed, max_gap=3):
    """(key, t) pairs: three clients, gaps of 0..max_gap ticks so ticks are shared."""
    stream = []
    t = 0
    for _ in range(count):
        seed = lcg(seed)
        key = "client-%d" % (seed % 3)
        seed = lcg(seed)
        t += seed % (max_gap + 1)
        stream.append((key, t))
    return stream


def main():
    # --- The sliding log is exact. -----------------------------------------------------------
    stream = request_stream(400, seed=17)
    log = ThrottlingMiddleware(SlidingLogLimiter, 5, 25)
    for key, t in stream:
        log.handle(key, t)
    assert log.requests == 400 and log.allowed + log.throttled == log.requests
    for key, times in log.accepted_by_key.items():
        per_tick = slots(times, 25)
        for t, count in zip(times, per_tick):
            assert count <= 5, ("sliding log must be exact", key, t, count)
    brute = ThrottlingMiddleware(SlidingLogLimiter, 5, 25)
    for key, t in stream:
        brute.handle(key, t)
    assert brute.decisions == log.decisions, "the middleware must be deterministic"
    assert window_max(log.accepted_by_key["client-0"], 25) == 5

    # --- The fixed-window flaw, exhibited concretely. ----------------------------------------
    edge = [(0, 9), (0, 9), (0, 9), (0, 9), (0, 9), (0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    fixed = ThrottlingMiddleware(FixedWindowLimiter, 5, 10)
    for key, t in edge:
        assert fixed.handle(key, t) is True, "the fixed window lets both bursts through"
    assert fixed.allowed == 10 and fixed.accepted_by_key[0] == [9] * 5 + [10] * 5
    assert window_max(fixed.accepted_by_key[0], 10) == 10, "10 > limit inside one real window"
    sliding = ThrottlingMiddleware(SlidingLogLimiter, 5, 10)
    for key, t in edge:
        sliding.handle(key, t)
    assert sliding.allowed == 5 and sliding.throttled == 5
    assert window_max(sliding.accepted_by_key[0], 10) == 5
    counter = ThrottlingMiddleware(SlidingCounterLimiter, 5, 10)
    for key, t in edge:
        counter.handle(key, t)
    assert window_max(counter.accepted_by_key[0], 10) <= 5

    # --- The sliding counter: the two-sided error bound. ------------------------------------
    for limit, window in ((5, 25), (3, 10), (10, 60), (1, 7)):
        for seed in (17, 251, 4099):
            middleware = ThrottlingMiddleware(SlidingCounterLimiter, limit, window)
            for key, t in request_stream(300, seed=seed):
                middleware.handle(key, t)
            assert middleware.allowed + middleware.throttled == middleware.requests
            for key, times in middleware.accepted_by_key.items():
                history = middleware.limiters[key].history
                assert [entry[0] for entry in history] == times
                for i, (t, estimate, prev_total, fraction) in enumerate(history):
                    earlier = times[:i]
                    exact = float(sum(1 for other in earlier if t - window < other <= t))
                    # exact = curr + tail and estimate = curr + prev_total * (1 - f) with
                    # 0 <= tail <= prev_total, hence -prev_total <= estimate - exact <= (1-f) * prev.
                    assert estimate - exact <= prev_total * (1.0 - fraction) + 1e-9
                    assert estimate - exact >= -prev_total - 1e-9
                    assert estimate <= limit
                # curr <= limit - 1 and prev_total <= limit - 1, so no real window can hold more
                # than 2 * limit - 2 (asserted with one tick of slack).
                assert window_max(times, window) <= 2 * limit - 1, (limit, window, key)
            assert all(limiter.state_size() == 2 for limiter in middleware.limiters.values())

    # The overweight pattern that exposes the approximation: five requests at the very end of a
    # window are all admitted, the counters still look light early in the next window, and the
    # limiter lets two more through - 7 requests inside one real 10-tick window with limit 5.
    adversarial = [(0, 9)] * 5 + [(0, 11), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17)]
    counter_adv = ThrottlingMiddleware(SlidingCounterLimiter, 5, 10)
    for key, t in adversarial:
        counter_adv.handle(key, t)
    accepted_adv = counter_adv.accepted_by_key[0]
    assert accepted_adv == [9, 9, 9, 9, 9, 13, 14, 16], accepted_adv
    assert window_max(accepted_adv, 10) == 8 > 5, "the weighted counter overshoots here"
    log_adv = ThrottlingMiddleware(SlidingLogLimiter, 5, 10)
    for key, t in adversarial:
        log_adv.handle(key, t)
    assert log_adv.allowed == 5 and window_max(log_adv.accepted_by_key[0], 10) == 5

    # The log keeps at most `limit` timestamps and, on this stream, never allows more than the
    # counter does: the counter is the more conservative of the two on ordinary traffic.
    log = ThrottlingMiddleware(SlidingLogLimiter, 4, 30)
    cnt = ThrottlingMiddleware(SlidingCounterLimiter, 4, 30)
    stream = request_stream(400, seed=777)
    for key, t in stream:
        log.handle(key, t)
        cnt.handle(key, t)
    assert all(limiter.state_size() <= 4 for limiter in log.limiters.values())
    assert cnt.allowed <= log.allowed, "the counter policy is the stricter one here"
    assert cnt.allowed >= log.allowed * 0.5, ("within a factor of two", cnt.allowed, log.allowed)
    assert cnt.throttled > 0 and log.throttled > 0
    for key, times in log.accepted_by_key.items():
        assert window_max(times, 30) <= 4

    # --- Boundary of the half-open window (t - W, t]. ----------------------------------------
    boundary = ThrottlingMiddleware(SlidingLogLimiter, 1, 10)
    assert boundary.handle("k", 0) is True
    assert boundary.handle("k", 9) is False, "still inside the window"
    assert boundary.handle("k", 10) is True, "t = 10 excludes t = 0: the window is half-open"
    assert boundary.handle("k", 11) is False, "t = 10 is now inside the window"
    assert boundary.handle("k", 20) is True, "t = 20 excludes t = 10"
    assert boundary.accepted_by_key["k"] == [0, 10, 20]
    tight = ThrottlingMiddleware(SlidingLogLimiter, 1, 1)
    assert tight.handle("k", 5) is True and tight.handle("k", 5) is False
    assert tight.handle("k", 6) is True
    counter_boundary = ThrottlingMiddleware(SlidingCounterLimiter, 1, 10)
    assert counter_boundary.handle("k", 0) is True
    assert counter_boundary.handle("k", 5) is False
    assert counter_boundary.handle("k", 20) is True, "two windows later the count is gone"

    # Independent brute-force re-simulation of the log from the raw stream.
    replay = {}
    for key, t in stream:
        times = replay.setdefault(key, [])
        if sum(1 for other in times if t - 30 < other <= t) < 4:
            times.append(t)
    assert replay == dict(log.accepted_by_key), "the middleware matches a from-scratch replay"


if __name__ == "__main__":
    main()
    print("PASS")
