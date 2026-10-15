"""Task 335 - Implement data structure for skyline queries (segment tree variant). (Tier: Intermediate, CTCI Ch 4)

A segment tree over the x-range supports "raise the skyline to height h on [l, r)" updates,
point queries (skyline height at x) and range-max queries (tallest point in [l, r)), each in
O(log n). Every node keeps `tag` (a height pending for its whole range) and `best` (tallest
skyline point inside it); updates only lower no heights, so applying max is monotone and no
push-down is needed. Cross-checked against a brute-force per-position max array along with
deterministic LCG update batches.
"""


class SkylineTree:
    """Range-max updates over positions [0, n) with point and range-max queries."""

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be >= 1")
        size = 1
        while size < n:
            size <<= 1
        self.n = n
        self.size = size
        self.tag = [0] * (2 * size)          # height applied to this node's whole range
        self.best = [0] * (2 * size)         # tallest point in this node's range

    def update(self, left, right, height):
        """Raises the skyline to at least `height` on the half-open range [left, right)."""
        if not 0 <= left < right <= self.n:
            raise ValueError("invalid range")
        self._update(1, 0, self.size, left, right, height)

    def point(self, index):
        """Skyline height at `index`."""
        if not 0 <= index < self.n:
            raise ValueError("index out of range")
        node, lo, hi = 1, 0, self.size
        height = 0
        while True:
            height = max(height, self.tag[node])
            if hi - lo == 1:
                return height
            mid = (lo + hi) // 2
            if index < mid:
                node, hi = 2 * node, mid
            else:
                node, lo = 2 * node + 1, mid

    def range_max(self, left, right):
        """Tallest point in [left, right); 0 for an empty range (no positions covered)."""
        if not 0 <= left <= right <= self.n:
            raise ValueError("invalid range")
        if left == right:
            return 0
        return self._range_max(1, 0, self.size, left, right)

    def skyline(self):
        return [self.point(i) for i in range(self.n)]

    # --- internals ------------------------------------------------------------------

    def _update(self, node, lo, hi, left, right, height):
        if right <= lo or hi <= left:
            return
        if left <= lo and hi <= right:
            self.tag[node] = max(self.tag[node], height)
            self.best[node] = max(self.best[node], height)
            return
        mid = (lo + hi) // 2
        self._update(2 * node, lo, mid, left, right, height)
        self._update(2 * node + 1, mid, hi, left, right, height)
        self.best[node] = max(self.tag[node], self.best[2 * node], self.best[2 * node + 1])

    def _range_max(self, node, lo, hi, left, right):
        if left >= right or right <= lo or hi <= left:
            return 0
        if left <= lo and hi <= right:
            return self.best[node]
        mid = (lo + hi) // 2
        return max(self.tag[node],
                   self._range_max(2 * node, lo, mid, left, right),
                   self._range_max(2 * node + 1, mid, hi, left, right))


def brute_schedule(n, updates):
    """Oracle: keep a height per position and apply each update directly."""
    heights = [0] * n
    for left, right, height in updates:
        for i in range(left, right):
            heights[i] = max(heights[i], height)
    return heights


def lcg_updates(count, seed, n):
    updates = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        left = seed % n
        seed = (seed * 75 + 74) % 65537
        right = left + 1 + seed % (n - left)
        seed = (seed * 75 + 74) % 65537
        updates.append((left, right, seed % 100))
    return updates, seed


def main():
    # Hand-checked: [1,4)->5, [2,3)->7, [4,6)->2 over 7 positions.
    tree = SkylineTree(7)
    for update in ((1, 4, 5), (2, 3, 7), (4, 6, 2)):
        tree.update(*update)
    assert tree.skyline() == [0, 5, 7, 5, 2, 2, 0]
    assert tree.point(2) == 7 and tree.point(6) == 0 and tree.point(0) == 0
    assert tree.range_max(0, 3) == 7
    assert tree.range_max(2, 5) == 7
    assert tree.range_max(3, 6) == 5
    assert tree.range_max(0, 7) == 7
    assert tree.range_max(5, 5) == 0                      # empty range
    assert tree.range_max(6, 7) == 0

    # Updates never lower a height: a small update under a tall one is a no-op.
    tree.update(2, 3, 4)
    assert tree.point(2) == 7
    tree.update(2, 3, 9)
    assert tree.point(2) == 9 and tree.range_max(1, 3) == 9

    # A single-position tree is the smallest legal instance.
    tiny = SkylineTree(1)
    assert tiny.point(0) == 0 and tiny.range_max(0, 1) == 0
    tiny.update(0, 1, 42)
    assert tiny.point(0) == 42 and tiny.range_max(0, 1) == 42

    # Wide updates touching every boundary, compared against the brute-force array.
    boundary = SkylineTree(4)
    for update in ((0, 4, 1), (0, 1, 2), (3, 4, 3), (1, 3, 4)):
        boundary.update(*update)
    assert boundary.skyline() == [2, 4, 4, 3]
    assert boundary.range_max(0, 4) == 4
    assert boundary.range_max(0, 2) == 4
    assert boundary.range_max(2, 4) == 4
    assert boundary.range_max(3, 4) == 3

    seed = 7788
    for trial in range(20):
        seed = (seed * 75 + 74) % 65537
        n = 1 + seed % 40
        updates, seed = lcg_updates(1 + seed % 60, seed, n)
        tree = SkylineTree(n)
        applied = []
        for update in updates:
            tree.update(*update)
            applied.append(update)
        expected = brute_schedule(n, applied)
        assert tree.skyline() == expected, (trial, applied)
        for left in range(n):
            for right in range(left, n + 1):
                assert tree.range_max(left, right) == max(expected[left:right] or [0]), \
                    (trial, left, right)


if __name__ == "__main__":
    main()
    print("PASS")
