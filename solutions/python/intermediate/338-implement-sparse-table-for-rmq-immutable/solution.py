"""Task 338 - Implement sparse table for RMQ (immutable). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Precompute row j holding the minimum of every window of length 2^j, then any range minimum
is min of two overlapping power-of-two windows: build O(n log n), query O(1) with the
floor-log table. Cross-checked for every (l, r) pair against a naive scan on deterministic
LCG arrays.
"""


class SparseTable:
    """Immutable range-minimum queries over a fixed array; query(left, right) is inclusive."""

    def __init__(self, values):
        if not values:
            raise ValueError("values must not be empty")
        n = len(values)
        self.values = list(values)
        self.log = [0] * (n + 1)
        for i in range(2, n + 1):
            self.log[i] = self.log[i // 2] + 1
        table = [self.values]
        level = 1
        while (1 << level) <= n:
            half = 1 << (level - 1)
            previous = table[level - 1]
            width = n - (1 << level) + 1
            table.append([min(previous[i], previous[i + half]) for i in range(width)])
            level += 1
        self.table = table
        self.n = n

    def query(self, left, right):
        """Minimum of values[left..right] (both ends included)."""
        if not 0 <= left <= right < self.n:
            raise ValueError("invalid range")
        level = self.log[right - left + 1]
        row = self.table[level]
        return min(row[left], row[right - (1 << level) + 1])


def naive_min(values, left, right):
    return min(values[left:right + 1])


def lcg_values(count, seed):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % 1000)
    return values, seed


def main():
    # Hand-checked array; every assertion below is verified by eye against the values.
    values = [4, 2, 7, 1, 3, 9, 0, 5]
    table = SparseTable(values)
    assert table.query(0, 7) == 0
    assert table.query(1, 3) == 1
    assert table.query(0, 0) == 4
    assert table.query(1, 1) == 2
    assert table.query(2, 2) == 7
    assert table.query(3, 7) == 0
    assert table.query(4, 6) == 0
    assert table.query(2, 5) == 1
    assert table.query(5, 7) == 0
    assert table.query(0, 2) == 2
    assert table.query(6, 6) == 0
    assert table.log[1] == 0 and table.log[2] == 1 and table.log[3] == 1
    assert table.log[4] == 2 and table.log[7] == 2 and table.log[8] == 3

    # Duplicates and extremes (including negative values) still work.
    dup = SparseTable([5, 5, 5, 5])
    assert dup.query(0, 3) == 5 and dup.query(1, 2) == 5
    mixed = SparseTable([3, -1, 4, -1, 5, -9, 2, 6, -9, 7])
    assert mixed.query(0, 9) == -9 and mixed.query(5, 8) == -9 and mixed.query(0, 3) == -1

    # Single element.
    single = SparseTable([42])
    assert single.query(0, 0) == 42
    assert single.log == [0, 0]

    seed = 8383
    for trial in range(8):
        seed = (seed * 75 + 74) % 65537
        size = 1 + seed % 64
        values, seed = lcg_values(size, seed)
        table = SparseTable(values)
        for left in range(size):
            for right in range(left, size):
                assert table.query(left, right) == naive_min(values, left, right), \
                    (trial, left, right, values)


if __name__ == "__main__":
    main()
    print("PASS")
