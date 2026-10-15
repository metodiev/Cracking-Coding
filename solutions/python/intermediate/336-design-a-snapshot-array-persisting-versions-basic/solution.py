"""Task 336 - Design a snapshot array (persisting versions basic). (Tier: Intermediate, CTCI Ch 7)

SnapshotArray keeps, per index, the growing list of (snap_id, value) writes; get(index, s)
binary-searches the last write with snap_id <= s, so set/snap are O(1) and get is O(log V)
with V writes for that index. Cross-checked against a naive implementation that stores a
full copy of every array version, on deterministic LCG operation sequences.
"""


class SnapshotArray:
    """set(index, value), snap() -> snapshot id, get(index, snap_id)."""

    def __init__(self, length):
        if length < 1:
            raise ValueError("length must be >= 1")
        self.length = length
        self.snap_count = 0
        # history[i] holds (snap_id, value) pairs with strictly increasing snap ids;
        # the entry for snap 0 defaults to 0 so reads of untouched indices are free.
        self.history = [[(0, 0)] for _ in range(length)]

    def set(self, index, value):
        if not 0 <= index < self.length:
            raise ValueError("index out of range")
        entries = self.history[index]
        if entries[-1][0] == self.snap_count:
            entries[-1] = (self.snap_count, value)      # overwrite the current (unsealed) write
        else:
            entries.append((self.snap_count, value))

    def snap(self):
        snap_id = self.snap_count
        self.snap_count += 1
        return snap_id

    def get(self, index, snap_id):
        if not 0 <= index < self.length:
            raise ValueError("index out of range")
        if not 0 <= snap_id < self.snap_count:
            raise ValueError("unknown snapshot id")
        entries = self.history[index]
        lo, hi = 0, len(entries) - 1
        while lo < hi:                                   # last entry with snap_id <= requested
            mid = (lo + hi + 1) // 2
            if entries[mid][0] <= snap_id:
                lo = mid
            else:
                hi = mid - 1
        return entries[lo][1]


class NaiveSnapshotArray:
    """Oracle: keep one full copy of the array per snapshot."""

    def __init__(self, length):
        self.current = [0] * length
        self.versions = []

    def set(self, index, value):
        self.current[index] = value

    def snap(self):
        self.versions.append(self.current[:])
        return len(self.versions) - 1

    def get(self, index, snap_id):
        return self.versions[snap_id][index]


def main():
    # Canonical LeetCode sequence.
    arr = SnapshotArray(3)
    arr.set(0, 5)
    assert arr.snap() == 0
    arr.set(0, 6)
    assert arr.get(0, 0) == 5
    assert arr.snap() == 1
    assert arr.get(0, 0) == 5
    assert arr.get(0, 1) == 6
    assert arr.get(1, 0) == 0 and arr.get(2, 1) == 0

    # Writes before the first snap are visible in snapshot 0; writes after the last snap
    # are not part of any snapshot yet.
    arr2 = SnapshotArray(2)
    arr2.set(1, 7)
    arr2.set(1, 9)                       # same snapshot 0: overwrites 7
    assert arr2.snap() == 0
    assert arr2.get(1, 0) == 9
    arr2.set(1, 11)
    assert arr2.get(1, 0) == 9           # unsealed write invisible
    assert arr2.snap() == 1
    assert arr2.get(1, 1) == 11

    # Snapshots with no writes in between repeat the previous value.
    arr3 = SnapshotArray(1)
    arr3.set(0, 1)
    first = arr3.snap()
    second = arr3.snap()
    arr3.set(0, 2)
    third = arr3.snap()
    assert (first, second, third) == (0, 1, 2)
    assert [arr3.get(0, s) for s in (0, 1, 2)] == [1, 1, 2]

    try:
        arr3.get(0, 3)
        raise AssertionError("unknown snapshot id must raise")
    except ValueError:
        pass

    # LCG operation sequences against the full-copy oracle.
    seed = 3141
    length = 8
    fast, naive = SnapshotArray(length), NaiveSnapshotArray(length)
    for step in range(3000):
        seed = (seed * 75 + 74) % 65537
        choice = seed % 10
        seed = (seed * 75 + 74) % 65537
        if choice < 5:
            index = seed % length
            seed = (seed * 75 + 74) % 65537
            value = seed % 1000
            fast.set(index, value)
            naive.set(index, value)
        elif choice < 7:
            assert fast.snap() == naive.snap()
        elif naive.versions and choice < 10:
            index = seed % length
            seed = (seed * 75 + 74) % 65537
            snap_id = seed % len(naive.versions)
            assert fast.get(index, snap_id) == naive.get(index, snap_id), (step, index, snap_id)
        for probe in range(length):
            if naive.versions:
                snap_id = len(naive.versions) - 1
                assert fast.get(probe, snap_id) == naive.get(probe, snap_id), step


if __name__ == "__main__":
    main()
    print("PASS")
