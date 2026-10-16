"""Task 582 - Implement run-length encoded segment tree for binary string operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A segment tree over a binary string that carries a run-length summary in every node - the number of
ones, the length, the first and last character, and the number of runs - so that both "how many ones
in [l, r)" and "how many runs of equal characters in [l, r)" can be answered by merging O(log n)
nodes: the merge sums the run counts and subtracts one exactly when the rightmost character of the
left part equals the leftmost character of the right part. Three lazy range updates are supported -
assign 0, assign 1 and flip - with the tags composing by the rules (assign then flip is the opposite
assign; flip twice is nothing; flip then assign is the assign), and every node's summary is rebuilt
from the tag when it is pushed down, which is what makes the run-length summary stay consistent under
lazy updates. All operations are verified against a plain Python list of characters that is updated
naively, so the structure is checked on hundreds of interleaved updates rather than on a hand-picked
sequence.

Checks: 320 random operations (mixed assigns, flips, whole-range assignments and single-element
updates) where after *every* operation the one-count and the run-count of the whole string, of the
updated range, of five random sub-ranges and of six prefixes are compared with the naive list; the
documented tag composition laws exercised directly on the structure (flip twice equals doing nothing,
assign 0 after flip equals assign 1, and the two agree with the naive list); the hand-checked
structures 0000, 1111, 0011, 0101, 0001100 and 1010101 with their exact run counts (1, 1, 2, 4, 3 and
7); empty ranges returning zero runs and zero ones and single-element ranges returning one run; the
boundary ranges [0, n), [0, 1), [n-1, n); the empty string; determinism; and the number of visited
nodes per query asserted to stay within the O(log n) bound.
"""
import math

MASK64 = (1 << 64) - 1
NONE, SET_ZERO, SET_ONE, FLIP = 0, 1, 2, 3


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


class RunLengthSegmentTree:
    """ones: number of ones, length: segment length, left/right: endpoint characters,
    runs: number of maximal equal-character runs, lazy: pending tag."""

    def __init__(self, symbols):
        self.data = [1 if symbol else 0 for symbol in symbols]
        self.length = len(self.data)
        if self.length:
            self.ones = [0] * (4 * self.length)
            self.lengths = [0] * (4 * self.length)
            self.left = [0] * (4 * self.length)
            self.right = [0] * (4 * self.length)
            self.runs = [0] * (4 * self.length)
            self.lazy = [NONE] * (4 * self.length)
            self.visited = 0
            self._build(1, 0, self.length)
        else:
            self.ones, self.lengths, self.left, self.right = [], [], [], []
            self.runs, self.lazy = [], []
            self.visited = 0

    # --- construction and merging -------------------------------------------------------
    def _set(self, node, size, symbol):
        self.ones[node] = size if symbol else 0
        self.lengths[node] = size
        self.left[node] = symbol
        self.right[node] = symbol
        self.runs[node] = 1
        self.lengths[node] = size

    def _build(self, node, start, end):
        size = end - start
        self.lengths[node] = size
        if end - start == 1:
            symbol = self.data[start]
            self.ones[node] = symbol
            self.left[node] = symbol
            self.right[node] = symbol
            self.runs[node] = 1
            return
        middle = (start + end) // 2
        self._build(2 * node, start, middle)
        self._build(2 * node + 1, middle, end)
        self._pull(node)

    def _pull(self, node):
        left_child, right_child = 2 * node, 2 * node + 1
        self.ones[node] = self.ones[left_child] + self.ones[right_child]
        self.left[node] = self.left[left_child]
        self.right[node] = self.right[right_child]
        self.runs[node] = (self.runs[left_child] + self.runs[right_child]
                           - (1 if self.right[left_child] == self.left[right_child] else 0))

    def _apply(self, node, tag):
        size = self.lengths[node]
        if tag == SET_ZERO:
            self._set(node, size, 0)
        elif tag == SET_ONE:
            self._set(node, size, 1)
        elif tag == FLIP:
            self.ones[node] = size - self.ones[node]
            self.left[node] ^= 1
            self.right[node] ^= 1
        elif tag == NONE:
            return
        # Compose with the pending tag of this node.
        if tag == FLIP:
            if self.lazy[node] == SET_ZERO:
                self.lazy[node] = SET_ONE
            elif self.lazy[node] == SET_ONE:
                self.lazy[node] = SET_ZERO
            elif self.lazy[node] == FLIP:
                self.lazy[node] = NONE
            else:
                self.lazy[node] = FLIP
        else:
            self.lazy[node] = tag

    def _push(self, node):
        tag = self.lazy[node]
        if tag != NONE and self.lengths[2 * node] > 0:
            self._apply(2 * node, tag)
            self._apply(2 * node + 1, tag)
        self.lazy[node] = NONE

    # --- updates ------------------------------------------------------------------------
    def _update(self, node, start, end, low, high, tag):
        if high <= start or end <= low:
            return
        if low <= start and end <= high:
            self._apply(node, tag)
            return
        self._push(node)
        middle = (start + end) // 2
        self._update(2 * node, start, middle, low, high, tag)
        self._update(2 * node + 1, middle, end, low, high, tag)
        self._pull(node)

    def assign(self, low, high, symbol):
        if high > low and self.length:
            self._update(1, 0, self.length, low, high, SET_ONE if symbol else SET_ZERO)

    def flip(self, low, high):
        if high > low and self.length:
            self._update(1, 0, self.length, low, high, FLIP)

    # --- queries ------------------------------------------------------------------------
    def _query(self, node, start, end, low, high):
        """Returns (ones, length, left char, right char, runs) for the intersection, or None."""
        self.visited += 1
        if high <= start or end <= low:
            return None
        if low <= start and end <= high:
            return (self.ones[node], self.lengths[node], self.left[node], self.right[node],
                    self.runs[node])
        self._push(node)
        middle = (start + end) // 2
        first = self._query(2 * node, start, middle, low, high)
        second = self._query(2 * node + 1, middle, end, low, high)
        if first is None:
            return second
        if second is None:
            return first
        runs = first[4] + second[4] - (1 if first[3] == second[2] else 0)
        return (first[0] + second[0], first[1] + second[1], first[2], second[3], runs)

    def counts(self, low, high):
        """(ones, runs) in the half-open range [low, high)."""
        if high <= low or not self.length:
            return (0, 0)
        result = self._query(1, 0, self.length, low, high)
        return (result[0], result[4])

    def total_runs(self):
        return self.runs[1] if self.length else 0

    def total_ones(self):
        return self.ones[1] if self.length else 0

    def snapshot(self):
        """The current string, read out of the leaves (only used by the checks)."""
        values = []
        self._collect(1, 0, self.length, values)
        return values

    def _collect(self, node, start, end, values):
        if end - start == 1:
            values.append(self.ones[node])
            return
        self._push(node)
        middle = (start + end) // 2
        self._collect(2 * node, start, middle, values)
        self._collect(2 * node + 1, middle, end, values)


def brute_force_runs(data, low, high):
    if high <= low:
        return (0, 0)
    window = data[low:high]
    runs = 1
    for index in range(1, len(window)):
        if window[index] != window[index - 1]:
            runs += 1
    return (sum(window), runs)


def main():
    # --- Hand-checked structures --------------------------------------------------------
    cases = {"0000": (0, 1), "1111": (4, 1), "0011": (2, 2), "0101": (2, 4),
             "0001100": (2, 3), "1010101": (4, 7)}
    for text, (ones, runs) in cases.items():
        tree = RunLengthSegmentTree([int(character) for character in text])
        assert tree.total_ones() == ones and tree.total_runs() == runs, text
        assert tree.counts(0, len(text)) == (ones, runs)
        for start in range(len(text) + 1):
            for end in range(start, len(text) + 1):
                expected = brute_force_runs([int(c) for c in text], start, end)
                assert tree.counts(start, end) == expected, (text, start, end)
    assert RunLengthSegmentTree([]).total_runs() == 0
    assert RunLengthSegmentTree([]).counts(0, 5) == (0, 0)
    single = RunLengthSegmentTree([1])
    assert single.counts(0, 1) == (1, 1) and single.counts(1, 1) == (0, 0)
    assert single.counts(0, 0) == (0, 0)

    # --- Tag composition laws -----------------------------------------------------------
    tree = RunLengthSegmentTree([0, 0, 0, 1, 1, 0, 1])
    naive = [0, 0, 0, 1, 1, 0, 1]
    tree.flip(1, 5)
    naive = [1 - naive[i] if 1 <= i < 5 else naive[i] for i in range(7)]
    assert tree.snapshot() == naive
    tree.flip(1, 5)
    assert tree.snapshot() == [0, 0, 0, 1, 1, 0, 1], "flip twice is the identity"
    tree.flip(0, 7)                       # naive is now [1, 1, 1, 0, 0, 1, 0]
    tree.assign(2, 6, 1)                  # and the assignment overwrites the middle four
    naive = [1, 1, 1, 1, 1, 1, 0]
    assert tree.snapshot() == naive, "assign wins over flip"
    assert tree.counts(0, 7) == (6, 2)
    tree.assign(0, 7, 0)
    tree.flip(3, 4)
    assert tree.snapshot() == [0, 0, 0, 1, 0, 0, 0]
    tree.assign(3, 4, 0)
    assert tree.snapshot() == [0] * 7 and tree.total_runs() == 1

    # --- Randomised differential testing ------------------------------------------------
    state = 20240
    size = 256
    symbols = []
    for _ in range(size):
        state, u = uniform(state)
        symbols.append(1 if u < 0.5 else 0)
    tree = RunLengthSegmentTree(symbols)
    naive = list(symbols)
    operations = 0
    for step in range(320):
        state, u = uniform(state)
        state, v = uniform(state)
        state, w = uniform(state)
        low = int(u * size)
        high = low + 1 + int(v * (size - low))
        kind = step % 4
        if kind == 0:
            tree.assign(low, high, 0)
            naive[low:high] = [0] * (high - low)
        elif kind == 1:
            tree.assign(low, high, 1)
            naive[low:high] = [1] * (high - low)
        elif kind == 2:
            tree.flip(low, high)
            naive[low:high] = [1 - value for value in naive[low:high]]
        else:
            index = int(w * size)
            tree.flip(index, index + 1)
            naive[index] = 1 - naive[index]
        operations += 1
        assert tree.total_ones() == sum(naive), "total ones after operation %d" % operations
        assert tree.total_runs() == brute_force_runs(naive, 0, size)[1]
        assert tree.counts(low, high) == brute_force_runs(naive, low, high)
        assert tree.counts(0, size) == (sum(naive), brute_force_runs(naive, 0, size)[1])
        for _ in range(5):
            state, u = uniform(state)
            state, v = uniform(state)
            start = int(u * size)
            stop = start + int(v * (size - start))
            assert tree.counts(start, stop) == brute_force_runs(naive, start, stop), \
                "random sub-range after operation %d" % operations
        for prefix in (1, 7, 64, 100, 255, 256):
            assert tree.counts(0, prefix) == brute_force_runs(naive, 0, prefix)
        if operations % 64 == 0:
            assert tree.snapshot() == naive, "the leaves agree with the naive string"
    assert operations == 320
    assert tree.snapshot() == naive

    # --- Whole-range assignment and the visited-node bound ------------------------------
    tree.assign(0, size, 0)
    assert tree.total_runs() == 1 and tree.total_ones() == 0
    tree.assign(0, size, 1)
    assert tree.total_runs() == 1 and tree.total_ones() == size
    tree.flip(0, size)
    assert tree.total_ones() == 0 and tree.snapshot() == [0] * size
    tree.visited = 0
    tree.counts(0, size)
    bound = 4 * (math.ceil(math.log2(size)) + 1)
    assert tree.visited <= bound, "a range query visits O(log n) nodes: %r" % tree.visited
    assert tree.counts(0, 0) == (0, 0) and tree.counts(size, size) == (0, 0)
    assert tree.counts(0, 1) == (0, 1) and tree.counts(size - 1, size) == (0, 1)

    # --- Determinism --------------------------------------------------------------------
    again = RunLengthSegmentTree(symbols)
    assert again.snapshot() == symbols and again.total_runs() == RunLengthSegmentTree(symbols).total_runs()
    first = RunLengthSegmentTree([1, 0, 1, 1, 0])
    first.flip(1, 4)
    second = RunLengthSegmentTree([1, 0, 1, 1, 0])
    second.flip(1, 4)
    assert first.snapshot() == second.snapshot() and first.counts(0, 5) == second.counts(0, 5)


if __name__ == "__main__":
    main()
    print("PASS")
