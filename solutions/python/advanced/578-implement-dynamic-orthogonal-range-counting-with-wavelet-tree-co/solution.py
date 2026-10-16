"""Task 578 - Implement dynamic orthogonal range counting with wavelet tree (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Orthogonal range counting: given an array of integers, count the positions in the index range [l, r)
whose value lies in the value range [a, b]. The static part is a full wavelet tree - every node stores
one bit per symbol of its subarray plus the prefix counts of those bits, so a range count descends one
level per bit of the value range and takes O(log sigma) steps: the count of values <= x in [l, r) is
computed by mapping l and r down the tree, adding the left-child contributions whenever the descent
goes right. The *dynamic* part is the documented simplification: a genuine dynamic wavelet tree
(pointer-machine or the Fenwick-of-trees structure) is out of scope here, so the file keeps the array
in fixed blocks of 32 elements, gives every block its own wavelet tree, rebuilds exactly one block's
tree per point update, and answers a range count as whole-block counts plus a linear scan of the two
partial blocks. The header states that trade-off plainly instead of claiming a real dynamic wavelet
tree: an update costs O(B log sigma), a query O((n/B) log sigma + B), and the number of block-tree
rebuilds is instrumented and asserted to equal the number of updates.

Checks: static counts against brute force on 200 random index/value windows on a 500-element array,
plus the boundary cases (empty index range, l = r, r = n, l = 0, a value window that misses
everything, one that covers everything, a = b, and a window narrower than a single value); the
prefix-count primitive count_less_equal(x, l, r) checked so that count_range is exactly
count_less_equal(b) - count_less_equal(a - 1); the block decomposition's whole-block/partial-block
split exercised by query windows that start and end inside blocks, align with block boundaries, and
span every block; 60 random point updates followed by a fresh query battery against the brute-force
array, with the rebuild counter asserted; the array-restore invariant (writing every changed value
back restores the array and every answer, with the rebuild counter accounting for those writes too); the empty array and single-element array; and determinism.
"""
import math

MASK64 = (1 << 64) - 1
BLOCK = 32


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


class WaveletNode:
    __slots__ = ("lo", "hi", "mid", "prefix", "left", "right", "size")

    def __init__(self, lo, hi, symbols):
        self.lo = lo
        self.hi = hi
        self.size = len(symbols)
        self.left = None
        self.right = None
        if lo == hi:
            self.mid = lo
            self.prefix = [0] * (len(symbols) + 1)
            return
        self.mid = (lo + hi) // 2
        ones = 0
        self.prefix = [0]
        for symbol in symbols:
            if symbol > self.mid:
                ones += 1
            self.prefix.append(ones)
        self.left = WaveletNode(lo, self.mid, [s for s in symbols if s <= self.mid])
        self.right = WaveletNode(self.mid + 1, hi, [s for s in symbols if s > self.mid])


class WaveletTree:
    """Static wavelet tree supporting prefix counts of values <= x over index ranges."""

    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.length = len(self.symbols)
        if self.symbols:
            self.lo, self.hi = min(self.symbols), max(self.symbols)
        else:
            self.lo, self.hi = 0, 0
        self.root = WaveletNode(self.lo, self.hi, self.symbols) if self.symbols else None

    def count_less_equal(self, value, start, end):
        """How many of symbols[start:end] are <= value."""
        if self.root is None or end <= start or value < self.lo:
            return 0
        if value >= self.hi:
            return end - start
        node = self.root
        total = 0
        while node.lo != node.hi:
            if value > node.mid:
                # Every left-child symbol in the window is <= value: add them and descend right.
                total += (end - node.prefix[end]) - (start - node.prefix[start])
                start = node.prefix[start]
                end = node.prefix[end]
                node = node.right
            else:
                start = start - node.prefix[start]
                end = end - node.prefix[end]
                node = node.left
        return total + (end - start)

    def count_range(self, low, high, start, end):
        """How many of symbols[start:end] lie in [low, high]."""
        if low > high:
            return 0
        return self.count_less_equal(high, start, end) - self.count_less_equal(low - 1, start, end)


class DynamicRangeCounter:
    """Block decomposition with one wavelet tree per block: rebuild-one-block updates."""

    def __init__(self, symbols, block=BLOCK):
        self.block = block
        self.data = list(symbols)
        self.rebuilds = 0
        self.trees = []
        self.rebuild_all()

    def rebuild_all(self):
        self.trees = []
        for start in range(0, len(self.data), self.block):
            self.trees.append(WaveletTree(self.data[start:start + self.block]))
            self.rebuilds += 1

    def _rebuild_block(self, index):
        block_index = index // self.block
        start = block_index * self.block
        self.trees[block_index] = WaveletTree(self.data[start:start + self.block])
        self.rebuilds += 1

    def update(self, index, value):
        self.data[index] = value
        self._rebuild_block(index)

    def count(self, low, high, start, end):
        if low > high or end <= start:
            return 0
        total = 0
        position = start
        while position < end:
            block_index = position // self.block
            block_start = block_index * self.block
            block_end = min(block_start + self.block, len(self.data))
            if position == block_start and block_end <= end:
                total += self.trees[block_index].count_range(low, high, 0,
                                                             block_end - block_start)
                position = block_end
            else:
                if low <= self.data[position] <= high:
                    total += 1
                position += 1
        return total


def brute_force_count(data, low, high, start, end):
    return sum(1 for value in data[start:end] if low <= value <= high)


def main():
    # --- The static tree on a small hand-checked array ----------------------------------
    hand = [3, 1, 4, 1, 5, 9, 2, 6]
    static = WaveletTree(hand)
    assert static.count_range(1, 1, 0, 8) == 2, "two ones"
    assert static.count_range(0, 10, 0, 8) == 8
    assert static.count_range(5, 9, 4, 8) == 3, "5, 9 and 6"
    assert static.count_range(2, 3, 0, 4) == 1, "only the leading 3 of 3,1,4,1 is in [2, 3]"
    assert static.count_range(7, 7, 0, 8) == 0
    assert static.count_range(0, 10, 3, 3) == 0, "an empty index range"
    for start in range(9):
        for end in range(start, 9):
            for low in range(-1, 11):
                for high in range(low, 11):
                    assert static.count_range(low, high, start, end) == brute_force_count(
                        hand, low, high, start, end), (low, high, start, end)

    # --- Random array, random windows ---------------------------------------------------
    state = 2024
    data = []
    for _ in range(500):
        state, u = uniform(state)
        data.append(int(u * 100))
    tree = WaveletTree(data)
    checked = 0
    for _ in range(200):
        state, u = uniform(state)
        state, v = uniform(state)
        state, w = uniform(state)
        start = int(u * 501)
        end = start + int(v * (501 - start))
        low = int(w * 120) - 10
        high = low + int(u * 60)
        expected = brute_force_count(data, low, high, start, end)
        assert tree.count_range(low, high, start, end) == expected
        assert tree.count_less_equal(high, start, end) - tree.count_less_equal(low - 1, start,
                                                                              end) == expected
        checked += 1
    assert checked == 200
    assert tree.count_range(-5, 200, 0, 500) == 500
    assert tree.count_range(200, 300, 0, 500) == 0
    assert tree.count_range(10, 20, 0, 500) == brute_force_count(data, 10, 20, 0, 500)
    assert tree.count_range(50, 50, 0, 500) == data.count(50)
    assert tree.count_range(0, 100, 500, 500) == 0
    assert tree.count_less_equal(-1, 0, 500) == 0
    assert tree.count_less_equal(1000, 0, 500) == 500

    # --- Degenerate arrays --------------------------------------------------------------
    empty = WaveletTree([])
    assert empty.count_range(0, 10, 0, 0) == 0 and empty.count_less_equal(5, 0, 0) == 0
    single = WaveletTree([7])
    assert single.count_range(7, 7, 0, 1) == 1 and single.count_range(8, 9, 0, 1) == 0
    assert single.count_range(7, 7, 1, 1) == 0

    # --- The dynamic block structure ----------------------------------------------------
    counter = DynamicRangeCounter(data)
    initial_rebuilds = counter.rebuilds
    assert initial_rebuilds == math.ceil(500 / BLOCK) == 16
    updates = 60
    for step in range(updates):
        state, u = uniform(state)
        state, v = uniform(state)
        index = int(u * 500)
        counter.update(index, int(v * 100))
    assert counter.rebuilds == initial_rebuilds + updates, \
        "exactly one block tree is rebuilt per update: %r" % counter.rebuilds
    for _ in range(120):
        state, u = uniform(state)
        state, v = uniform(state)
        state, w = uniform(state)
        start = int(u * 501)
        end = start + int(v * (501 - start))
        low = int(w * 120) - 10
        high = low + int(u * 40)
        assert counter.count(low, high, start, end) == brute_force_count(counter.data, low, high,
                                                                        start, end)
    # Windows that respect the block boundaries, that start inside a block, and that span all.
    for start, end in ((0, 1), (0, 32), (31, 33), (32, 64), (0, 500), (100, 400), (255, 256),
                       (480, 500), (33, 479)):
        for low, high in ((0, 100), (20, 40), (0, 0), (99, 100), (10, 9)):
            assert counter.count(low, high, start, end) == brute_force_count(counter.data, low,
                                                                            high, start, end)
    # Restoring the original values restores every answer.
    restored = 0
    for index, value in enumerate(data):
        if counter.data[index] != value:
            counter.update(index, value)
            restored += 1
    assert counter.data == data
    assert 0 < restored <= updates, "only the changed positions are written back: %r" % restored
    for _ in range(60):
        state, u = uniform(state)
        start = int(u * 500)
        assert counter.count(30, 60, start, min(500, start + 77)) == brute_force_count(
            data, 30, 60, start, min(500, start + 77))
    assert counter.rebuilds == initial_rebuilds + updates + restored, \
        "one rebuild per update, and the restores count too: %r" % counter.rebuilds

    # --- Determinism --------------------------------------------------------------------
    again = WaveletTree(data)
    assert again.count_range(25, 75, 0, 500) == tree.count_range(25, 75, 0, 500)
    assert again.count_less_equal(50, 10, 400) == tree.count_less_equal(50, 10, 400)
    fresh = DynamicRangeCounter(data)
    assert fresh.rebuilds == initial_rebuilds
    for start in range(0, 500, 90):
        assert fresh.count(0, 100, start, start + 50) == counter.count(0, 100, start, start + 50)


if __name__ == "__main__":
    main()
    print("PASS")
