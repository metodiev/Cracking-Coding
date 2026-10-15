"""Task 329 - Range intersection queries (interval trees basics). (Tier: Intermediate, CTCI Ch 4)

An interval tree is a binary search tree over interval starts where every node also
caches the largest end in its subtree (max_end).  A range query walks only the
subtrees that can still intersect the closed query range [lo, hi]: the left subtree
matters while left.max_end >= lo, the right subtree while node.start <= hi.
Results are cross-checked against a plain filter on deterministic interval sets.
"""


class IntervalTree:
    """Augmented BST over interval starts with subtree max-end values."""

    class Node:
        __slots__ = ("start", "end", "left", "right", "max_end")

        def __init__(self, start, end):
            self.start = start
            self.end = end
            self.left = None
            self.right = None
            self.max_end = end

    def __init__(self, intervals):
        self.root = self._build(sorted((tuple(iv) for iv in intervals), key=lambda iv: (iv[0], iv[1])))

    def _build(self, ordered):
        if not ordered:
            return None
        mid = len(ordered) // 2
        node = IntervalTree.Node(ordered[mid][0], ordered[mid][1])
        node.left = self._build(ordered[:mid])
        node.right = self._build(ordered[mid + 1:])
        node.max_end = node.end
        if node.left is not None:
            node.max_end = max(node.max_end, node.left.max_end)
        if node.right is not None:
            node.max_end = max(node.max_end, node.right.max_end)
        return node

    def query(self, lo, hi):
        """All stored intervals intersecting the closed range [lo, hi], in start order."""
        found = []
        _query(self.root, lo, hi, found)
        return found


def _query(node, lo, hi, found):
    if node is None:
        return
    if node.left is not None and node.left.max_end >= lo:
        _query(node.left, lo, hi, found)
    if node.start <= hi and node.end >= lo:
        found.append([node.start, node.end])
    if node.start <= hi:
        _query(node.right, lo, hi, found)


def naive_query(intervals, lo, hi):
    """Independent oracle: filter every interval."""
    return sorted([list(iv) for iv in intervals if iv[0] <= hi and iv[1] >= lo])


def lcg_intervals(count, seed):
    intervals = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        start = seed % 60
        seed = (seed * 75 + 74) % 65537
        end = start + 1 + seed % 12
        intervals.append([start, end])
    return intervals, seed


FIXED = [[15, 20], [10, 30], [17, 19], [5, 20], [12, 15], [30, 40]]


def main():
    tree = IntervalTree(FIXED)
    assert tree.query(6, 7) == [[5, 20]]
    assert tree.query(14, 16) == [[5, 20], [10, 30], [12, 15], [15, 20]]
    assert tree.query(30, 35) == [[10, 30], [30, 40]]
    assert tree.query(19, 19) == [[5, 20], [10, 30], [15, 20], [17, 19]]
    assert tree.query(0, 4) == []
    assert tree.query(41, 50) == []

    assert IntervalTree([]).query(0, 10) == []
    assert IntervalTree([[4, 4]]).query(4, 4) == [[4, 4]]         # point interval

    seed = 7071
    for trial in range(40):
        intervals, seed = lcg_intervals(30, seed)
        tree = IntervalTree(intervals)
        for _ in range(30):
            seed = (seed * 75 + 74) % 65537
            lo = seed % 70
            seed = (seed * 75 + 74) % 65537
            hi = lo + seed % 25
            assert tree.query(lo, hi) == naive_query(intervals, lo, hi), (intervals, lo, hi)


if __name__ == "__main__":
    main()
    print("PASS")
