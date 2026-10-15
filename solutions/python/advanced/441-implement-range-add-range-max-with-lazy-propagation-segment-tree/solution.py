"""Task 441 - Implement range add + range max with lazy propagation. (Tier: Advanced, CTCI Ch 4)

Segment tree with a lazy "add" tag: range_add(lo, hi, v) adds to a whole node and parks the
delta; range_max pushes the tag down only along the visited path. O(log n) per operation.
"""

class RangeAddRangeMax:
    def __init__(self, values):
        self.n = len(values)
        self.tree = [0] * (4 * max(1, self.n))
        self.lazy = [0] * (4 * max(1, self.n))
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, node, lo, hi, values):
        if lo == hi:
            self.tree[node] = values[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def _push(self, node):
        delta = self.lazy[node]
        if delta:
            for child in (2 * node, 2 * node + 1):
                self.tree[child] += delta
                self.lazy[child] += delta
            self.lazy[node] = 0

    def add(self, left, right, value):
        self._add(1, 0, self.n - 1, left, right, value)

    def _add(self, node, lo, hi, left, right, value):
        if right < lo or hi < left:
            return
        if left <= lo and hi <= right:
            self.tree[node] += value
            self.lazy[node] += value
            return
        self._push(node)
        mid = (lo + hi) // 2
        self._add(2 * node, lo, mid, left, right, value)
        self._add(2 * node + 1, mid + 1, hi, left, right, value)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def max(self, left, right):
        return self._max(1, 0, self.n - 1, left, right)

    def _max(self, node, lo, hi, left, right):
        if left <= lo and hi <= right:
            return self.tree[node]
        self._push(node)
        mid = (lo + hi) // 2
        best = float("-inf")
        if left <= mid:
            best = max(best, self._max(2 * node, lo, mid, left, right))
        if right > mid:
            best = max(best, self._max(2 * node + 1, mid + 1, hi, left, right))
        return best


def lcg_ops(count, seed, n):
    ops = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        lo = seed % n
        seed = (seed * 75 + 74) % 65537
        hi = lo + seed % (n - lo)
        seed = (seed * 75 + 74) % 65537
        kind = seed % 2
        seed = (seed * 75 + 74) % 65537
        value = seed % 21 - 10
        ops.append((kind, lo, hi, value))
    return ops


def main():
    tree = RangeAddRangeMax([1, 3, 2, 5, 4])
    assert tree.max(0, 4) == 5
    assert tree.max(3, 3) == 5
    tree.add(1, 3, 2)
    assert tree.max(0, 4) == 7
    assert tree.max(0, 0) == 1
    assert tree.max(1, 1) == 5
    assert tree.max(2, 3) == 7
    tree.add(0, 4, -10)
    assert tree.max(0, 4) == -3
    tree.add(4, 4, 100)
    assert tree.max(0, 4) == 94

    n = 16
    seed = 1234
    base = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        base.append(seed % 50 - 25)
    tree = RangeAddRangeMax(base)
    brute = base[:]
    for kind, lo, hi, value in lcg_ops(200, 777, n):
        if kind == 0:
            tree.add(lo, hi, value)
            for i in range(lo, hi + 1):
                brute[i] += value
        else:
            assert tree.max(lo, hi) == max(brute[lo:hi + 1]), (lo, hi)


if __name__ == "__main__":
    main()
    print("PASS")
