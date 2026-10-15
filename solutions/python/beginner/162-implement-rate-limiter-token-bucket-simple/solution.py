
"""Task 162 - Implement rate limiter (token bucket simple). (Tier: Beginner, CTCI Ch 7)

A bucket holds up to `capacity` tokens and refills at `refill_per_second`.
allow(now_ms, tokens=1) consumes tokens if available, else rejects.
Time is injected (now_ms) so behaviour is deterministic and testable.
"""


class TokenBucket:
    def __init__(self, capacity, refill_per_second):
        if capacity <= 0 or refill_per_second <= 0:
            raise ValueError("capacity and refill rate must be positive")
        self.capacity = capacity
        self.rate = refill_per_second
        self.tokens = float(capacity)
        self.last_ms = None

    def _refill(self, now_ms):
        if self.last_ms is None:
            self.last_ms = now_ms
            return
        elapsed = max(0, now_ms - self.last_ms)
        self.tokens = min(self.capacity, self.tokens + elapsed / 1000.0 * self.rate)
        self.last_ms = now_ms

    def allow(self, now_ms, cost=1):
        self._refill(now_ms)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


def main():
    bucket = TokenBucket(capacity=3, refill_per_second=1.0)
    assert bucket.allow(0) and bucket.allow(0) and bucket.allow(0)
    assert not bucket.allow(0)              # bucket drained
    assert bucket.allow(1000)               # 1 second -> 1 token
    assert not bucket.allow(1000, cost=2)   # only 1 token available
    assert bucket.allow(3000, cost=2)       # 2 more seconds -> 2 tokens
    # long idle time clamps at capacity, not beyond
    assert bucket.allow(100_000)
    assert not bucket.allow(100_000, cost=4)


if __name__ == "__main__":
    main()
    print("PASS")
