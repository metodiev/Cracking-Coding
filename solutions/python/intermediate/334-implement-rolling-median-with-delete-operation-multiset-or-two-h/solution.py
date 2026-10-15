"""Task 334 - Implement rolling median with delete operation (multiset or two heaps + lazy deletion). (Tier: Intermediate, CTCI Ch 7)

MedianMultiset stores the smaller half of the values in a max-heap (`low`, negated) and the
larger half in a min-heap (`high`), keeping len(low) == len(high) or len(low) == len(high)+1
in logical size. delete() cannot reach into a heap, so it marks one occurrence of the value
as pending in whichever heap holds it: the mark is consumed the next time that value
surfaces at a heap top. Each add/delete/median is O(log n) amortised. Cross-checked against
a sorted-list oracle on deterministic LCG operation sequences.
"""

import bisect
import heapq


class MedianMultiset:
    """add(value) / delete(value) / median() / size() with two heaps and lazy deletion."""

    def __init__(self):
        self.low = []                      # max-heap of the smaller half (values negated)
        self.high = []                     # min-heap of the larger half
        self.pending_low = {}              # value -> deleted occurrences still in `low`
        self.pending_high = {}
        self.pending_low_total = 0
        self.pending_high_total = 0
        self.count = 0                     # logical (active) size
        self.active = {}                   # value -> active occurrences (for presence tests)

    def size(self):
        return self.count

    def add(self, value):
        self._clean_low()
        self._clean_high()
        if self.low and value <= -self.low[0]:
            heapq.heappush(self.low, -value)
        else:
            heapq.heappush(self.high, value)
        self.count += 1
        self.active[value] = self.active.get(value, 0) + 1
        self._rebalance()

    def delete(self, value):
        """Removes one occurrence of `value`; raises ValueError when it is absent."""
        if self.active.get(value, 0) < 1:
            raise ValueError("value not present: %r" % (value,))
        self.active[value] -= 1
        self._clean_low()
        if self.low and value <= -self.low[0]:
            self.pending_low[value] = self.pending_low.get(value, 0) + 1
            self.pending_low_total += 1
        else:
            self.pending_high[value] = self.pending_high.get(value, 0) + 1
            self.pending_high_total += 1
        self.count -= 1
        self._clean_low()
        self._clean_high()
        self._rebalance()

    def median(self):
        """Returns None when empty, a float otherwise (half-integer values averaged)."""
        self._clean_low()
        self._clean_high()
        if self.count == 0:
            return None
        if self.count % 2 == 1:
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2.0

    # --- internals ------------------------------------------------------------------

    def _clean_low(self):
        while self.low and self.pending_low.get(-self.low[0], 0) > 0:
            value = -heapq.heappop(self.low)
            self.pending_low[value] -= 1
            self.pending_low_total -= 1

    def _clean_high(self):
        while self.high and self.pending_high.get(self.high[0], 0) > 0:
            value = heapq.heappop(self.high)
            self.pending_high[value] -= 1
            self.pending_high_total -= 1

    def _logical_sizes(self):
        return len(self.low) - self.pending_low_total, len(self.high) - self.pending_high_total

    def _rebalance(self):
        while True:
            low_size, high_size = self._logical_sizes()
            if low_size > high_size + 1:
                self._clean_low()
                heapq.heappush(self.high, -heapq.heappop(self.low))
            elif high_size > low_size:
                self._clean_high()
                heapq.heappush(self.low, -heapq.heappop(self.high))
            else:
                return


class SortedListOracle:
    """Reference multiset over a sorted Python list (O(n) delete, fine for tests)."""

    def __init__(self):
        self.data = []

    def add(self, value):
        bisect.insort(self.data, value)

    def delete(self, value):
        index = bisect.bisect_left(self.data, value)
        assert index < len(self.data) and self.data[index] == value
        self.data.pop(index)

    def median(self):
        n = len(self.data)
        if n == 0:
            return None
        if n % 2 == 1:
            return float(self.data[n // 2])
        return (self.data[n // 2 - 1] + self.data[n // 2]) / 2.0


def main():
    m = MedianMultiset()
    assert m.size() == 0 and m.median() is None
    m.add(5)
    assert m.median() == 5.0
    m.add(3)
    assert m.size() == 2 and m.median() == 4.0
    m.add(8)
    assert m.median() == 5.0
    m.add(1)
    assert m.median() == 4.0                                # {1,3,5,8}
    m.delete(5)
    assert m.size() == 3 and m.median() == 3.0              # {1,3,8}
    m.delete(1)
    assert m.median() == 5.5                                # {3,8}
    m.delete(3)
    assert m.median() == 8.0                                # {8}
    m.delete(8)
    assert m.size() == 0 and m.median() is None

    try:
        m.delete(7)
        raise AssertionError("delete of an absent value must raise")
    except ValueError:
        pass

    # Duplicates: every occurrence is tracked separately.
    dupes = MedianMultiset()
    for _ in range(3):
        dupes.add(4)
    assert dupes.median() == 4.0
    dupes.delete(4)
    assert dupes.median() == 4.0
    dupes.delete(4)
    assert dupes.median() == 4.0 and dupes.size() == 1
    dupes.delete(4)
    assert dupes.median() is None

    # Alternating deletes across the median boundary exercise lazy deletion on both heaps.
    seq = MedianMultiset()
    for value in range(1, 11):
        seq.add(value)
    assert seq.median() == 5.5
    for value, expected in ((5, 6.0), (6, 5.5), (10, 4.0), (1, 5.5), (7, 4.0), (2, 6.0)):
        seq.delete(value)
        assert seq.median() == expected, (value, seq.median(), expected)
    assert seq.size() == 4

    seed = 4242
    m2, oracle = MedianMultiset(), SortedListOracle()
    for step in range(2000):
        seed = (seed * 75 + 74) % 65537
        if oracle.data and seed % 10 >= 6:
            seed = (seed * 75 + 74) % 65537
            value = oracle.data[seed % len(oracle.data)]
            m2.delete(value)
            oracle.delete(value)
        else:
            seed = (seed * 75 + 74) % 65537
            value = 1 + seed % 50
            m2.add(value)
            oracle.add(value)
        assert m2.size() == len(oracle.data), step
        seed = (seed * 75 + 74) % 65537
        if seed % 5 == 0 or step % 7 == 0:
            assert m2.median() == oracle.median(), (step, m2.median(), oracle.median())

    while oracle.data:
        m2.delete(oracle.data[0])
        oracle.delete(oracle.data[0])
        assert m2.median() == oracle.median()
    assert m2.median() is None


if __name__ == "__main__":
    main()
    print("PASS")
