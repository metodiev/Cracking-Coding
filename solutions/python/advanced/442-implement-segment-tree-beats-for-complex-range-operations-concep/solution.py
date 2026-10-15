"""Task 442 - Implement segment tree beats for complex range operations (conceptual). (Tier: Advanced, CTCI Ch 4)

Segment tree beats (jiry_2) maintains range max/sum under range chmin (cap every element at
x) and range chmax. This file is the documented pruning variant for range chmin + range max +
range sum: recursion stops the moment a whole node already has max <= x, because capping can
then change nothing inside it, so only nodes that really lose a value are visited. Full beats
adds a second-maximum tag plus the number of maxima so an entire subtree can be capped in one
step; the amortised bound usually quoted for beats is O((n + q) log n), while this pruning-only
variant keeps the same stopping rule without the second-max shortcut and is the conceptual
simplification requested here.
"""


class SegmentTreeBeats:
    def __init__(self, values):
        self.n = len(values)
        size = 4 * max(1, self.n)
        self.node_max = [0] * size
        self.node_sum = [0] * size
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, node, lo, hi, values):
        if lo == hi:
            self.node_max[node] = values[lo]
            self.node_sum[node] = values[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self.node_max[node] = max(self.node_max[2 * node], self.node_max[2 * node + 1])
        self.node_sum[node] = self.node_sum[2 * node] + self.node_sum[2 * node + 1]

    def chmin(self, left, right, cap):
        self._chmin(1, 0, self.n - 1, left, right, cap)

    def _chmin(self, node, lo, hi, left, right, cap):
        if right < lo or hi < left or self.node_max[node] <= cap:
            return
        if lo == hi:
            self.node_max[node] = cap
            self.node_sum[node] = cap
            return
        mid = (lo + hi) // 2
        self._chmin(2 * node, lo, mid, left, right, cap)
        self._chmin(2 * node + 1, mid + 1, hi, left, right, cap)
        self.node_max[node] = max(self.node_max[2 * node], self.node_max[2 * node + 1])
        self.node_sum[node] = self.node_sum[2 * node] + self.node_sum[2 * node + 1]

    def range_max(self, left, right):
        return self._range_max(1, 0, self.n - 1, left, right)

    def _range_max(self, node, lo, hi, left, right):
        if left <= lo and hi <= right:
            return self.node_max[node]
        mid = (lo + hi) // 2
        best = -(1 << 60)
        if left <= mid:
            best = max(best, self._range_max(2 * node, lo, mid, left, right))
        if right > mid:
            best = max(best, self._range_max(2 * node + 1, mid + 1, hi, left, right))
        return best

    def range_sum(self, left, right):
        return self._range_sum(1, 0, self.n - 1, left, right)

    def _range_sum(self, node, lo, hi, left, right):
        if left <= lo and hi <= right:
            return self.node_sum[node]
        mid = (lo + hi) // 2
        total = 0
        if left <= mid:
            total += self._range_sum(2 * node, lo, mid, left, right)
        if right > mid:
            total += self._range_sum(2 * node + 1, mid + 1, hi, left, right)
        return total


def lcg_ops(count, seed, n):
    """Deterministic op script: kind 0 = chmin(lo, hi, cap), 1 = max query, 2 = sum query."""
    ops = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        lo = seed % n
        seed = (seed * 75 + 74) % 65537
        hi = lo + seed % (n - lo)
        seed = (seed * 75 + 74) % 65537
        kind = seed % 3
        seed = (seed * 75 + 74) % 65537
        cap = seed % 41 - 20
        ops.append((kind, lo, hi, cap))
    return ops


def main():
    tree = SegmentTreeBeats([1, 5, 3, 4, 2])
    assert tree.range_max(0, 4) == 5
    assert tree.range_sum(0, 4) == 15
    tree.chmin(1, 3, 3)
    assert tree.range_max(0, 4) == 3
    assert tree.range_max(1, 1) == 3
    assert tree.range_sum(0, 4) == 12
    assert tree.range_sum(2, 3) == 6
    tree.chmin(0, 4, -1)
    assert tree.range_max(0, 4) == -1
    assert tree.range_sum(0, 4) == -5
    tree.chmin(2, 2, 10)
    assert tree.range_max(0, 4) == -1
    assert tree.range_sum(0, 4) == -5

    n = 16
    seed = 1234
    base = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        base.append(seed % 50 - 25)
    tree = SegmentTreeBeats(base)
    brute = base[:]
    for kind, lo, hi, cap in lcg_ops(300, 777, n):
        if kind == 0:
            tree.chmin(lo, hi, cap)
            for i in range(lo, hi + 1):
                brute[i] = min(brute[i], cap)
        elif kind == 1:
            assert tree.range_max(lo, hi) == max(brute[lo:hi + 1]), (kind, lo, hi)
        else:
            assert tree.range_sum(lo, hi) == sum(brute[lo:hi + 1]), (kind, lo, hi)


if __name__ == "__main__":
    main()
    print("PASS")
