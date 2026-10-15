"""Task 432 - Implement data structure for dynamic median (order-statistics tree). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Approach: two heaps with lazy deletion. Values up to the median live in a max-heap (low), the
rest in a min-heap (high); every live low value is <= every live high value. Deletions do not
search the heaps: a per-value pending count marks the removal and is consumed when a marked
value surfaces at a heap top (prune), so remove/insert are O(log n) amortised. median() returns
the lower median - the live element at 0-based index (size-1)//2 - which is exactly the top of
low, so even sizes answer with a stored value instead of a float; the empty multiset answers
None. Cross-checked against a sorted-list oracle (bisect) on LCG operation scripts that insert,
delete and re-query, including deletions until the structure empties.
"""

import heapq
from bisect import insort


class DynamicMedian:
    def __init__(self):
        self.low = []           # max-heap via negated values: ceil(size/2) smallest live values
        self.high = []          # min-heap: the remaining live values
        self.pending_low = {}   # value -> lazy deletions waiting inside low
        self.pending_high = {}  # value -> lazy deletions waiting inside high
        self.counts = {}        # value -> live multiplicity (drives remove/absent checks)
        self.total = 0
        self.low_count = 0      # live elements physically owned by low

    def size(self):
        return self.total

    def median(self):
        if self.total == 0:
            return None
        self._prune()
        return -self.low[0]

    def insert(self, value):
        self._prune()
        self.counts[value] = self.counts.get(value, 0) + 1
        if self.low and value > -self.low[0]:
            heapq.heappush(self.high, value)
        else:
            heapq.heappush(self.low, -value)
            self.low_count += 1
        self.total += 1
        self._rebalance()

    def remove(self, value):
        live = self.counts.get(value, 0)
        if live == 0:
            return False
        if live == 1:
            del self.counts[value]
        else:
            self.counts[value] = live - 1
        self._prune()
        if self.low and value <= -self.low[0]:
            self.pending_low[value] = self.pending_low.get(value, 0) + 1
            self.low_count -= 1
        else:
            self.pending_high[value] = self.pending_high.get(value, 0) + 1
        self.total -= 1
        self._rebalance()
        return True

    def _prune(self):
        while self.low and self.pending_low.get(-self.low[0], 0) > 0:
            top = -heapq.heappop(self.low)
            self.pending_low[top] -= 1
        while self.high and self.pending_high.get(self.high[0], 0) > 0:
            top = heapq.heappop(self.high)
            self.pending_high[top] -= 1

    def _rebalance(self):
        target = (self.total + 1) // 2
        while True:
            self._prune()
            if self.low_count > target:
                top = -heapq.heappop(self.low)
                self.low_count -= 1
                heapq.heappush(self.high, top)
                continue
            if self.low_count < target:
                top = heapq.heappop(self.high)
                self.low_count += 1
                heapq.heappush(self.low, -top)
                continue
            break


def oracle_median(sorted_values):
    if not sorted_values:
        return None
    return sorted_values[(len(sorted_values) - 1) // 2]


def main():
    tracker = DynamicMedian()
    assert tracker.median() is None
    assert tracker.remove(0) is False

    for value, expected in ((5, 5), (3, 3), (8, 5), (9, 5)):
        tracker.insert(value)
        assert tracker.median() == expected, value
    assert tracker.size() == 4
    assert tracker.remove(5) is True
    assert tracker.median() == 8
    assert tracker.remove(3) is True
    assert tracker.median() == 8
    assert tracker.remove(9) is True
    assert tracker.median() == 8
    assert tracker.remove(8) is True
    assert tracker.median() is None
    assert tracker.size() == 0
    assert tracker.remove(8) is False
    tracker.insert(7)
    assert tracker.median() == 7
    assert tracker.remove(7) is True
    assert tracker.median() is None

    for _ in range(3):
        tracker.insert(3)
        assert tracker.median() == 3
    assert tracker.remove(3) is True
    assert tracker.remove(3) is True
    assert tracker.median() == 3
    assert tracker.remove(3) is True
    assert tracker.median() is None

    tracker = DynamicMedian()
    oracle = []
    for value in range(1, 11):
        tracker.insert(value)
        insort(oracle, value)
    for value in (10, 1, 9, 2, 8, 3, 7, 4, 6, 5):
        assert tracker.remove(value) is True
        oracle.remove(value)
        assert tracker.median() == oracle_median(oracle), (value, oracle)
    assert tracker.median() is None

    tracker = DynamicMedian()
    oracle = []
    seed = 4320
    for _ in range(400):
        seed = (seed * 75 + 74) % 65537
        op = seed % 4
        if op <= 1 or not oracle:
            seed = (seed * 75 + 74) % 65537
            value = seed % 40 - 20
            tracker.insert(value)
            insort(oracle, value)
        elif op == 2:
            seed = (seed * 75 + 74) % 65537
            value = oracle.pop(seed % len(oracle))
            assert tracker.remove(value) is True, value
        else:
            seed = (seed * 75 + 74) % 65537
            missing = 100 + seed % 10
            assert tracker.remove(missing) is False
        assert tracker.median() == oracle_median(oracle), (seed, oracle)
        assert tracker.size() == len(oracle)
    assert tracker.size() > 0


if __name__ == "__main__":
    main()
    print("PASS")
