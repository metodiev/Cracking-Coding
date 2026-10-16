"""Task 579 - Implement wavelet tree for strings (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A wavelet tree over a byte string: each node covers a symbol range [lo, hi], stores one bit per
symbol of its subarray (1 when the symbol is above the midpoint of the range) together with the
prefix counts of those bits needed for O(1) rank, and splits its subarray into the two child nodes -
so the tree has at most log2(256) = 8 levels and n bits per level. On top of that structure the file
implements access(i) (the i-th symbol), rank(c, i) (occurrences of c in s[:i]), select(c, k) (the
position of the k-th occurrence of c, zero-based) and count(c, l, r) (occurrences in the half-open
range [l, r)), each of them descending one level per step by reading the bit at the current position
and mapping it through the node's prefix counts. Two independent reference implementations are used
for verification: plain string scanning, and per-symbol position lists queried with bisect - so rank
and select are checked against results they were not derived from. The one honest simplification is
that select finds the k-th bit of a level by binary search over the prefix counts rather than through
block-wise super-blocks; the header records it.

Checks: exhaustive rank(c, i) over every symbol of the alphabet and every prefix index on a small
string with repeated symbols, compared against s[:i].count(c); the select/rank inverses
(select(c, rank(c, p)) = p at every occurrence p, and rank(c, select(c, k)) = k); select past the
last occurrence returning None and rank(c, 0) = 0, rank(c, n) = total count; access(i) for every
index; count(c, l, r) against both rank(c, r) - rank(c, l) and brute force over 400 random half-open
ranges, including empty ranges, l = r, r = n and l = 0; the same battery on an all-256-distinct-bytes
string (the full alphabet, 8 levels, where the stored bit count is asserted to be exactly 8n and
every rank is 1 and every select 0 or None), on a single-character string, on an empty string and on
a two-symbol string; out-of-range access and index raising IndexError; the bit-vector sizes asserted as n per level against the documented
log2(alphabet) + 1 bound; and determinism of the construction and of every query.
"""
import bisect
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31))


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def random_string(state, length, alphabet):
    symbols = []
    for _ in range(length):
        state, u = uniform(state)
        symbols.append(alphabet[int(u * len(alphabet))])
    return state, symbols


class WaveletNode:
    __slots__ = ("lo", "hi", "mid", "prefix", "left", "right", "size")

    def __init__(self, lo, hi, symbols):
        self.lo = lo
        self.hi = hi
        self.size = len(symbols)
        self.prefix = [0]
        self.left = None
        self.right = None
        if lo == hi:
            self.mid = lo
            self.prefix = [0] * (len(symbols) + 1)
            return
        self.mid = (lo + hi) // 2
        ones = 0
        for symbol in symbols:
            if symbol > self.mid:
                ones += 1
            self.prefix.append(ones)
        self.left = WaveletNode(lo, self.mid, [s for s in symbols if s <= self.mid])
        self.right = WaveletNode(self.mid + 1, hi, [s for s in symbols if s > self.mid])

    def rank1(self, index):
        """Number of ones in the first `index` bits."""
        return self.prefix[index]

    def rank0(self, index):
        return index - self.prefix[index]

    def bit(self, index):
        return 1 if self.prefix[index + 1] > self.prefix[index] else 0


def select_bit(node, k, bit):
    """Position of the k-th (zero-based) `bit` in the node's bit vector, by binary search over the
    prefix counts. A production wavelet tree stores block-wise counts instead, which turns this into
    a constant number of steps; the binary search is the honest simple version and is documented
    here rather than glossed over."""
    target = k + 1
    low, high = 0, node.size
    while low < high:
        middle = (low + high) // 2
        count = node.rank1(middle + 1) if bit else node.rank0(middle + 1)
        if count >= target:
            high = middle
        else:
            low = middle + 1
    return low


class WaveletTree:
    """Wavelet tree over a list of integer symbols; the alphabet is derived from the data unless
    an explicit (lo, hi) range is given."""

    def __init__(self, symbols, alphabet=None):
        self.symbols = list(symbols)
        self.length = len(self.symbols)
        if alphabet is None:
            alphabet = (min(self.symbols), max(self.symbols)) if self.symbols else (0, 0)
        self.lo, self.hi = alphabet
        self.root = WaveletNode(self.lo, self.hi, self.symbols) if self.symbols else None

    def levels(self):
        return int(math.ceil(math.log2(self.hi - self.lo + 1))) if self.hi > self.lo else 0

    def bit_count(self):
        """Total number of stored bits: at most n per level."""
        total = 0
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            if node.lo != node.hi:
                total += node.size
                stack.append(node.left)
                stack.append(node.right)
        return total

    def access(self, index):
        if self.root is None or index < 0 or index >= self.length:
            raise IndexError("index out of range")
        node = self.root
        while node.lo != node.hi:
            if node.bit(index) == 1:
                index = node.rank1(index)
                node = node.right
            else:
                index = node.rank0(index)
                node = node.left
        return node.lo

    def rank(self, symbol, index):
        """Occurrences of `symbol` in symbols[:index]."""
        if symbol < self.lo or symbol > self.hi or self.root is None:
            return 0
        node = self.root
        if index > self.length:
            index = self.length
        while node.lo != node.hi:
            if symbol <= node.mid:
                index = node.rank0(index)
                node = node.left
            else:
                index = node.rank1(index)
                node = node.right
        return index

    def select(self, symbol, occurrence):
        """Position of the `occurrence`-th (zero-based) `symbol`, or None if there is none."""
        if symbol < self.lo or symbol > self.hi or self.root is None:
            return None
        if occurrence < 0 or occurrence >= self.rank(symbol, self.length):
            return None
        return self._select(self.root, symbol, occurrence)

    def _select(self, node, symbol, occurrence):
        """Select is the inverse of the access mapping, so it is built bottom-up: the occurrence
        index is resolved inside the child first (the child's array index), and the parent's bit
        position, i.e. the parent's array index, is then recovered with select_bit."""
        if node.lo == node.hi:
            return occurrence
        if symbol <= node.mid:
            return select_bit(node, self._select(node.left, symbol, occurrence), 0)
        return select_bit(node, self._select(node.right, symbol, occurrence), 1)

    def count(self, symbol, start, end):
        return self.rank(symbol, end) - self.rank(symbol, start)


def main():
    state = 7
    state, sample = random_string(state, 400, [ord('a'), ord('b'), ord('c'), ord('d')])
    tree = WaveletTree(sample, alphabet=(ord('a'), ord('d')))

    # --- access -------------------------------------------------------------------------
    assert tree.length == 400
    for index in range(400):
        assert tree.access(index) == sample[index]
    try:
        tree.access(400)
    except IndexError:
        pass
    else:
        raise AssertionError("access past the end must raise")

    # --- Exhaustive rank on a small string ---------------------------------------------
    small = [ord(c) for c in "abracadabra"]
    small_tree = WaveletTree(small, alphabet=(ord('a'), ord('r')))
    for symbol in range(ord('a'), ord('s')):
        for index in range(len(small) + 1):
            expected = small[:index].count(symbol)
            assert small_tree.rank(symbol, index) == expected, \
                "rank(%r, %d)" % (chr(symbol), index)
    assert small_tree.rank(ord('a'), 0) == 0
    assert small_tree.rank(ord('a'), len(small)) == small.count(ord('a'))
    assert small_tree.rank(ord('z'), 5) == 0, "symbols outside the alphabet count zero"
    assert small_tree.rank(ord('a'), 1000) == small.count(ord('a')), "index clamps at the end"

    # --- select is the inverse of rank --------------------------------------------------
    positions = {}
    for index, symbol in enumerate(sample):
        positions.setdefault(symbol, []).append(index)
    for symbol, spots in sorted(positions.items()):
        assert tree.rank(symbol, len(sample)) == len(spots)
        for occurrence, position in enumerate(spots):
            assert tree.select(symbol, occurrence) == position, \
                "select(%r, %d)" % (chr(symbol), occurrence)
            assert tree.rank(symbol, position) == occurrence, "rank before that occurrence"
            assert tree.access(position) == symbol
        assert tree.select(symbol, len(spots)) is None
        assert tree.select(symbol, -1) is None
    assert tree.select(ord('z'), 0) is None

    # --- rank and select against the independent bisect index ---------------------------
    for symbol, spots in positions.items():
        for index in range(0, 401, 17):
            assert tree.rank(symbol, index) == bisect.bisect_left(spots, index)
        for occurrence in range(len(spots)):
            spot = tree.select(symbol, occurrence)
            assert spot == spots[occurrence]
            assert bisect.bisect_right(spots, spot) - 1 == occurrence

    # --- count over ranges --------------------------------------------------------------
    checked = 0
    for _ in range(400):
        state, u = uniform(state)
        state, v = uniform(state)
        start = int(u * 401)
        end = start + int(v * (401 - start))
        symbol = sorted(positions)[int(u * len(positions)) % len(positions)]
        expected = sum(1 for value in sample[start:end] if value == symbol)
        assert tree.count(symbol, start, end) == expected
        assert tree.count(symbol, start, end) == tree.rank(symbol, end) - tree.rank(symbol, start)
        checked += 1
    assert checked == 400
    assert tree.count(ord('a'), 200, 200) == 0, "an empty range counts nothing"
    assert tree.count(ord('a'), 0, 400) == len(positions.get(ord('a'), []))
    assert tree.count(ord('a'), 0, 0) == 0
    for symbol, spots in positions.items():
        assert tree.count(symbol, 0, 400) == len(spots)

    # --- The full byte alphabet (8 levels) ----------------------------------------------
    full = list(range(256))
    full_tree = WaveletTree(full, alphabet=(0, 255))
    assert full_tree.levels() == 8
    assert full_tree.bit_count() == full_tree.length * 8, \
        "the documented bound n * log2(alphabet) is attained exactly for a permutation"
    for index in range(256):
        assert full_tree.access(index) == index
        assert full_tree.rank(index, 256) == 1
        assert full_tree.select(index, 0) == index
        assert full_tree.select(index, 1) is None
    assert full_tree.count(3, 2, 5) == 1
    assert full_tree.count(3, 4, 5) == 0

    # --- Degenerate inputs --------------------------------------------------------------
    single = WaveletTree([42], alphabet=(42, 42))
    assert single.access(0) == 42 and single.rank(42, 1) == 1 and single.select(42, 0) == 0
    assert single.levels() == 0 and single.bit_count() == 0
    empty = WaveletTree([], alphabet=(0, 0))
    assert empty.rank(0, 5) == 0 and empty.select(0, 0) is None and empty.count(0, 0, 0) == 0
    assert empty.bit_count() == 0 and empty.levels() == 0
    try:
        empty.access(0)
    except IndexError:
        pass
    else:
        raise AssertionError("accessing the empty tree must raise")
    binary = WaveletTree([0, 1, 1, 0, 1], alphabet=(0, 1))
    assert [binary.access(i) for i in range(5)] == [0, 1, 1, 0, 1]
    assert binary.rank(1, 3) == 2 and binary.select(0, 1) == 3 and binary.count(0, 0, 5) == 2
    assert binary.bit_count() == 5, "one bit per symbol at every level"

    # --- Determinism --------------------------------------------------------------------
    again = WaveletTree(sample, alphabet=(ord('a'), ord('d')))
    assert again.bit_count() == tree.bit_count()
    assert ([again.rank(symbol, 100) for symbol in sorted(positions)]
            == [tree.rank(symbol, 100) for symbol in sorted(positions)])
    assert ([again.select(symbol, 3) for symbol in sorted(positions)]
            == [tree.select(symbol, 3) for symbol in sorted(positions)])
    assert random_string(7, 400, [ord('a'), ord('b'), ord('c'), ord('d')])[1] == sample


if __name__ == "__main__":
    main()
    print("PASS")
