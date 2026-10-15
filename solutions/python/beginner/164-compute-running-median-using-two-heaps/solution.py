
"""Task 164 - Compute running median using two heaps. (Tier: Beginner, CTCI Ch 4)

low = max-heap (negated values), high = min-heap. Invariants:
len(low) == len(high) or len(low) == len(high) + 1, and max(low) <= min(high).
Each add is O(log n); median is O(1).
"""

import heapq


class RunningMedian:
    def __init__(self):
        self.low = []      # max-heap via negation
        self.high = []     # min-heap

    def add(self, value):
        if not self.low or value <= -self.low[0]:
            heapq.heappush(self.low, -value)
        else:
            heapq.heappush(self.high, value)
        # rebalance so len(low) is len(high) or len(high) + 1
        if len(self.low) > len(self.high) + 1:
            heapq.heappush(self.high, -heapq.heappop(self.low))
        elif len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def median(self):
        if not self.low:
            raise ValueError("no values yet")
        if len(self.low) == len(self.high):
            return (-self.low[0] + self.high[0]) / 2.0
        return float(-self.low[0])


def running_medians(stream):
    rm = RunningMedian()
    out = []
    for x in stream:
        rm.add(x)
        out.append(rm.median())
    return out


def main():
    assert running_medians([1, 2, 3]) == [1.0, 1.5, 2.0]
    assert running_medians([5, 15, 1, 3]) == [5.0, 10.0, 5.0, 4.0]
    assert running_medians([-1, -2, -3]) == [-1.0, -1.5, -2.0]
    assert running_medians([7]) == [7.0]
    # duplicates keep the invariant
    assert running_medians([2, 2, 2]) == [2.0, 2.0, 2.0]


if __name__ == "__main__":
    main()
    print("PASS")
