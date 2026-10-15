"""Task 333 - Design hit counter (sliding time window). (Tier: Intermediate, CTCI Ch 7)

Hits arrive with non-decreasing timestamps; get_hits(t) counts the hits h recorded so far
with t - 299 <= h <= t, i.e. the classic 300-second sliding window. The counter keeps the
timestamps in one list and answers with two binary searches (first index >= t - 299 and
first index > t): O(1) per hit, O(log n) per query. Cross-checked against a naive scan on
a deterministic LCG event stream.
"""

WINDOW = 300


class HitCounter:
    """Sliding 300-second window over non-decreasing hit timestamps."""

    def __init__(self):
        self.times = []

    def hit(self, timestamp):
        if self.times:
            assert timestamp >= self.times[-1], "timestamps must not go backwards"
        self.times.append(timestamp)

    def get_hits(self, timestamp):
        """Number of hits h with timestamp - 299 <= h <= timestamp."""
        return self._first_greater(timestamp) - self._first_at_least(timestamp - (WINDOW - 1))

    def _first_at_least(self, timestamp):
        lo, hi = 0, len(self.times)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.times[mid] < timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _first_greater(self, timestamp):
        lo, hi = 0, len(self.times)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.times[mid] <= timestamp:
                lo = mid + 1
            else:
                hi = mid
        return lo


def naive_hits(times, timestamp):
    """Oracle: linear scan of every recorded timestamp."""
    return sum(1 for h in times if timestamp - (WINDOW - 1) <= h <= timestamp)


def main():
    # Hand-checked: hits at 1, 2, 3 -> get(4) = 3; windows slide away one at a time.
    counter = HitCounter()
    for t in (1, 2, 3):
        counter.hit(t)
    assert counter.get_hits(1) == 1
    assert counter.get_hits(2) == 2
    assert counter.get_hits(3) == 3
    assert counter.get_hits(4) == 3                       # all of 1,2,3 are within [t-299, t]
    assert counter.get_hits(299) == 3
    assert counter.get_hits(300) == 3                     # 300 - 299 = 1, still counts hit 1
    assert counter.get_hits(301) == 2                     # hit 1 has left the window
    assert counter.get_hits(302) == 1
    assert counter.get_hits(303) == 0

    # A later hit at 300 joins the window: get(300) = 4, get(301) = 3, get(302) = 2.
    counter.hit(300)
    assert counter.get_hits(300) == 4
    assert counter.get_hits(301) == 3
    assert counter.get_hits(302) == 2
    assert counter.get_hits(599) == 1                     # only the hit at 300 is in [300, 599]
    assert counter.get_hits(600) == 0

    # LeetCode-flavoured sequence: hits at 1, 2, 3 and (again) at 300.
    leet = HitCounter()
    for t in (1, 2, 3, 300):
        leet.hit(t)
    assert leet.get_hits(3) == 3
    assert leet.get_hits(300) == 4
    assert leet.get_hits(301) == 3
    # Querying before the newest hit only sees hits up to that time.
    assert leet.get_hits(2) == 2
    assert leet.get_hits(0) == 0

    # Same-timestamp bursts.
    burst = HitCounter()
    for _ in range(5):
        burst.hit(7)
    assert burst.get_hits(7) == 5
    assert burst.get_hits(306) == 5
    assert burst.get_hits(307) == 0

    # LCG event stream: alternate/none hits and queries, always matching the naive scan.
    seed = 2027
    now = 0
    counter2 = HitCounter()
    history = []
    for step in range(4000):
        seed = (seed * 75 + 74) % 65537
        now += seed % 5
        seed = (seed * 75 + 74) % 65537
        if seed % 4 == 0:
            seed = (seed * 75 + 74) % 65537
            timestamp = now + seed % 3
            counter2.hit(timestamp)
            history.append(timestamp)
            now = timestamp
        else:
            query = now
            assert counter2.get_hits(query) == naive_hits(history, query), (step, query)
        if step % 97 == 0:
            assert counter2.get_hits(now) == naive_hits(history, now)
    assert naive_hits(history, now) == counter2.get_hits(now)


if __name__ == "__main__":
    main()
    print("PASS")
