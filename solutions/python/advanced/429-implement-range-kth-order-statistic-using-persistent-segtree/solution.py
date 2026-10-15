"""Task 429 - Implement range kth order statistic using persistent segtree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A persistent segment tree over compressed values: prefix version i stores how often every
rank occurs in a[0..i), so a range [l, r] is the difference of versions r+1 and l. The k-th
smallest value in the range is found by descending once, comparing left-child counts between
the two versions. Checked against a sorted-slice oracle over many LCG queries, including
duplicates and negative values.
"""


class PersistentKthSmallest:
    def __init__(self, values):
        self.sorted_values = sorted(set(values))
        self.ranks = {value: i for i, value in enumerate(self.sorted_values)}
        self.domain = len(self.sorted_values)
        self.left = [0]
        self.right = [0]
        self.count = [0]
        root = 0
        self.roots = [0]
        for value in values:
            root = self._insert(root, 0, self.domain - 1, self.ranks[value])
            self.roots.append(root)

    def _node(self, left, right, count):
        self.left.append(left)
        self.right.append(right)
        self.count.append(count)
        return len(self.count) - 1

    def _insert(self, node, lo, hi, position):
        if lo == hi:
            return self._node(0, 0, self.count[node] + 1)
        mid = (lo + hi) // 2
        if position <= mid:
            left = self._insert(self.left[node], lo, mid, position)
            right = self.right[node]
        else:
            left = self.left[node]
            right = self._insert(self.right[node], mid + 1, hi, position)
        return self._node(left, right, self.count[left] + self.count[right])

    def kth(self, left, right, k):
        """k-th smallest (1-based) value of values[left..right]."""
        low = self.roots[left]
        high = self.roots[right + 1]
        lo, hi = 0, self.domain - 1
        while lo < hi:
            mid = (lo + hi) // 2
            left_count = self.count[self.left[high]] - self.count[self.left[low]]
            if k <= left_count:
                low = self.left[low]
                high = self.left[high]
                hi = mid
            else:
                k -= left_count
                low = self.right[low]
                high = self.right[high]
                lo = mid + 1
        return self.sorted_values[lo]


def lcg_array(n, seed):
    values = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 13 - 6)
    return values


def main():
    tree = PersistentKthSmallest([5, 1, 4, 2, 3])
    assert tree.kth(0, 4, 1) == 1
    assert tree.kth(0, 4, 2) == 2
    assert tree.kth(0, 4, 3) == 3
    assert tree.kth(0, 4, 4) == 4
    assert tree.kth(0, 4, 5) == 5
    assert tree.kth(1, 3, 1) == 1
    assert tree.kth(1, 3, 2) == 2
    assert tree.kth(1, 3, 3) == 4
    assert tree.kth(2, 2, 1) == 4
    assert tree.kth(4, 4, 1) == 3
    assert tree.kth(0, 0, 1) == 5

    dupes = PersistentKthSmallest([2, 2, 2])
    assert dupes.kth(0, 2, 1) == 2
    assert dupes.kth(0, 2, 2) == 2
    assert dupes.kth(0, 2, 3) == 2
    assert dupes.kth(1, 1, 1) == 2

    negatives = PersistentKthSmallest([-3, 7, -3, 0])
    assert negatives.kth(0, 3, 1) == -3
    assert negatives.kth(0, 3, 2) == -3
    assert negatives.kth(0, 3, 3) == 0
    assert negatives.kth(0, 3, 4) == 7
    assert negatives.kth(1, 2, 1) == -3
    assert negatives.kth(1, 2, 2) == 7

    for seed in (429, 12345, 54321):
        values = lcg_array(1 + seed % 40, seed)
        tree = PersistentKthSmallest(values)
        local = seed
        for _ in range(40):
            local = (local * 75 + 74) % 65537
            left = local % len(values)
            right = (local // 7) % len(values)
            if left > right:
                left, right = right, left
            segment = sorted(values[left:right + 1])
            local = (local * 75 + 74) % 65537
            k = 1 + local % len(segment)
            got = tree.kth(left, right, k)
            assert got == segment[k - 1], (seed, left, right, k, got, segment[k - 1])


if __name__ == "__main__":
    main()
    print("PASS")
