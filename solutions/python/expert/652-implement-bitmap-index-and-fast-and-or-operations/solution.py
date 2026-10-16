"""Task 652 - Implement bitmap index and fast AND/OR operations.  (Tier: Expert, Ch 14 - Databases)

A bitmap index stores, for every value of a column, one bitmap with a set bit per row holding that
value. Predicates then compose with bitwise boolean algebra: an equality lookup is a dictionary
probe, a range is the OR of the equality bitmaps of the covered values (located with a binary
search over the sorted distinct values), and a conjunction is the AND of the predicate bitmaps.
Filtering rows is exactly the inverse mapping of the resulting bitmap.

Two representations of the same set are implemented and cross-validated:

* `Bitmap` - dense row ids packed 32 bits to a word. `and_`/`or_`/`and_not`/`xor` are word-parallel
  (one machine word per 32 rows) and cardinality is a sum of popcounts, so AND of two dictionaries
  costs O(words) instead of O(rows) bit tests.
* `RunBitmap` - the same set as a sorted list of disjoint [start, end) runs of set bits (the
  run-length compressed bitmap family, e.g. BBC/WAH, without the word-alignment encoding bits).
  AND/OR are linear sweeps over the two run lists, which is *cheaper* than the word form when the
  bitmaps are clustered - the reason compressed bitmaps are used in column stores.

The invariant verified everywhere is representational equivalence: word bitmap == run bitmap == the
set built straight from the rows, for equality, range, AND, OR, AND-NOT and XOR results, plus the
index answers matching brute-force row filtering. Word-parallel AND is additionally checked against a
bit-by-bit probe loop, and the clustering claim is measured as stored words (runs vs words) on a
clustered column versus a uniformly sparse one.
"""

WORD_BITS = 32
MASK32 = (1 << WORD_BITS) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def popcount(word):
    return bin(word).count("1")


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


class Bitmap:
    """Sparse-to-dense row-id bitmap packed into 32-bit words; boolean ops are word-parallel."""

    __slots__ = ("words",)

    def __init__(self, words=None):
        self.words = [0] if words is None else list(words)
        self._trim()

    def _trim(self):
        while len(self.words) > 1 and self.words[-1] == 0:
            self.words.pop()

    def copy(self):
        return Bitmap(self.words)

    def _ensure(self, bit):
        index = bit >> 5
        if index >= len(self.words):
            self.words.extend([0] * (index + 1 - len(self.words)))

    def add(self, bit):
        self._ensure(bit)
        self.words[bit >> 5] |= 1 << (bit & 31)

    def discard(self, bit):
        index = bit >> 5
        if index < len(self.words):
            self.words[index] &= MASK32 ^ (1 << (bit & 31))
            self._trim()

    def contains(self, bit):
        index = bit >> 5
        return index < len(self.words) and bool(self.words[index] >> (bit & 31) & 1)

    def cardinality(self):
        return sum(popcount(word) for word in self.words)

    def is_empty(self):
        return all(word == 0 for word in self.words)

    def iter_bits(self):
        for index, word in enumerate(self.words):
            base = index * WORD_BITS
            while word:
                lowest = word & -word
                yield base + lowest.bit_length() - 1
                word ^= lowest

    def rows(self):
        return list(self.iter_bits())

    def as_set(self):
        return set(self.iter_bits())

    def _map(self, other, op):
        size = max(len(self.words), len(other.words))
        out = [0] * size
        for index in range(size):
            left = self.words[index] if index < len(self.words) else 0
            right = other.words[index] if index < len(other.words) else 0
            out[index] = op(left, right)
        return Bitmap(out)

    def and_(self, other):
        return self._map(other, lambda a, b: a & b)

    def or_(self, other):
        return self._map(other, lambda a, b: a | b)

    def and_not(self, other):
        return self._map(other, lambda a, b: a & ~b & MASK32)

    def xor_(self, other):
        return self._map(other, lambda a, b: a ^ b)

    def __eq__(self, other):
        return isinstance(other, Bitmap) and self.words == other.words

    def __repr__(self):
        return f"Bitmap(rows={self.rows()}, words={self.words})"


def merge_runs(runs):
    """Canonicalise any list of intervals into sorted, disjoint, non-adjacent runs."""
    merged = []
    for start, end in sorted(runs):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


class RunBitmap:
    """Run-length compressed bitmap: sorted disjoint [start, end) runs of set bits."""

    __slots__ = ("runs",)

    def __init__(self, runs=()):
        self.runs = merge_runs(list(runs))

    @classmethod
    def from_rows(cls, rows):
        return cls((row, row + 1) for row in rows)

    @classmethod
    def from_bitmap(cls, bitmap):
        return cls.from_rows(bitmap.rows())

    def to_bitmap(self):
        out = Bitmap()
        for start, end in self.runs:
            for bit in range(start, end):
                out.add(bit)
        return out

    def cardinality(self):
        return sum(end - start for start, end in self.runs)

    def stored_words(self):
        """Compressed footprint: two words per run (start, length) - one word when empty."""
        return max(1, 2 * len(self.runs))

    def and_(self, other):
        out = []
        i = j = 0
        while i < len(self.runs) and j < len(other.runs):
            start = max(self.runs[i][0], other.runs[j][0])
            end = min(self.runs[i][1], other.runs[j][1])
            if start < end:
                out.append((start, end))
            if self.runs[i][1] <= other.runs[j][1]:
                i += 1
            else:
                j += 1
        return RunBitmap(out)

    def or_(self, other):
        return RunBitmap(list(self.runs) + list(other.runs))

    def and_not(self, other):
        """Sweep the complement of `other` against `self` run by run."""
        out = []
        for start, end in self.runs:
            cursor = start
            for other_start, other_end in other.runs:
                if other_end <= cursor:
                    continue
                if other_start >= end:
                    break
                if other_start > cursor:
                    out.append((cursor, min(other_start, end)))
                cursor = max(cursor, other_end)
                if cursor >= end:
                    break
            if cursor < end:
                out.append((cursor, end))
        return RunBitmap(out)

    def __eq__(self, other):
        return isinstance(other, RunBitmap) and self.runs == list(other.runs)

    def __repr__(self):
        return f"RunBitmap({self.runs})"


class BitmapIndex:
    """Equality bitmaps per column value plus AND/OR predicate composition."""

    def __init__(self, columns, row_count):
        self.row_count = row_count
        self.maps = {}
        self.sorted_values = {}
        for name, values in columns.items():
            check(len(values) == row_count, "every column must have one value per row")
            value_map = {}
            for row, value in enumerate(values):
                value_map.setdefault(value, Bitmap()).add(row)
            self.maps[name] = value_map
            self.sorted_values[name] = sorted(value_map)

    def eq(self, column, value):
        return self.maps[column].get(value, Bitmap()).copy()

    def values_in_range(self, column, low, high):
        values = self.sorted_values[column]
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] < low:
                lo = mid + 1
            else:
                hi = mid
        start = lo
        lo, hi = start, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] <= high:
                lo = mid + 1
            else:
                hi = mid
        return values[start:lo]

    def range(self, column, low, high):
        """OR of the equality bitmaps of all distinct values inside [low, high]."""
        result = Bitmap()
        for value in self.values_in_range(column, low, high):
            result = result.or_(self.maps[column][value])
        return result

    def any_of(self, column, values):
        result = Bitmap()
        for value in values:
            result = result.or_(self.maps[column].get(value, Bitmap()))
        return result

    def all_of(self, predicates):
        """AND of (column, value) equality bitmaps."""
        result = None
        for column, value in predicates:
            current = self.eq(column, value)
            result = current if result is None else result.and_(current)
        return Bitmap() if result is None else result


def brute_rows(columns, predicate):
    rows = range(len(next(iter(columns.values()))))
    return [row for row in rows
            if predicate({name: values[row] for name, values in columns.items()})]


def main():
    columns = {
        "color": ["red", "blue", "red", "green", "blue", "red", "green", "blue", "red", "green"],
        "size": [1, 2, 1, 3, 2, 1, 3, 2, 3, 1],
    }
    index = BitmapIndex(columns, 10)

    # 1. equality lookup
    red = index.eq("color", "red")
    check(red.rows() == [0, 2, 5, 8], "equality bitmap must list exactly the matching rows")
    check(red.cardinality() == 4, "cardinality must count set bits")
    check(index.eq("color", "yellow").is_empty(), "an unknown value must yield an empty bitmap")

    # 2. range lookup matches brute-force filtering
    two_or_less = index.range("size", 1, 2)
    check(two_or_less.rows() == [0, 1, 2, 4, 5, 7, 9], "range bitmap must cover values 1 and 2")
    check(two_or_less.rows() == brute_rows(columns, lambda r: 1 <= r["size"] <= 2),
          "range bitmap must equal the brute-force row filter")
    check(index.range("size", 4, 9).is_empty(), "a range with no matching value must be empty")

    # 3. conjunctive predicates = AND of equality bitmaps
    conjunction = index.all_of([("color", "red"), ("size", 1)])
    check(conjunction.rows() == [0, 2, 5], "AND of two equality bitmaps must intersect the rows")
    check(conjunction.rows() == brute_rows(columns, lambda r: r["color"] == "red" and r["size"] == 1),
          "AND result must equal brute-force filtering")
    check(index.all_of([("color", "red"), ("size", 3)]).rows() == [8],
          "intersection of red and size 3 is row 8")
    check(index.all_of([]).is_empty(), "the AND of no predicates must be the empty bitmap")

    # 4. disjunction over a value list
    picked = index.any_of("color", ["green", "never"])
    check(picked.rows() == [3, 6, 9], "OR over values must ignore unknown values")

    # 5. word-parallel AND equals a bit-by-bit probe loop
    naive = Bitmap()
    for row in range(10):
        if red.contains(row) and two_or_less.contains(row):
            naive.add(row)
    check(red.and_(two_or_less) == naive, "word-parallel AND must equal the bit-by-bit probe")
    check(red.or_(two_or_less).rows() == sorted(set(red.rows()) | set(two_or_less.rows())),
          "word-parallel OR must equal the union of rows")
    check(red.and_not(two_or_less).rows() == [8], "AND-NOT must remove the rows of the other bitmap")
    check(red.xor_(red.copy()).is_empty(), "XOR of a bitmap with itself must be empty")
    check(red.and_(Bitmap()).is_empty(), "AND with the empty bitmap must be empty")
    check(red.or_(Bitmap()) == red, "OR with the empty bitmap must be the identity")

    # 6. multi-word bitmaps and high row ids
    sparse = Bitmap()
    sparse.add(0)
    sparse.add(31)
    sparse.add(32)
    sparse.add(5000)
    check(sparse.rows() == [0, 31, 32, 5000], "set bits must survive word boundaries")
    check(len(sparse.words) == 5000 // WORD_BITS + 1, "a high row id must grow the word array")
    check(sparse.cardinality() == 4, "cardinality must count bits across words")
    sparse.discard(5000)
    check(sparse.contains(5000) is False and sparse.rows() == [0, 31, 32],
          "discarding a bit must clear it and trim the tail")

    # 7. compressed (run) bitmap equals the word bitmap for every operation
    def run_view(bitmap):
        return RunBitmap.from_bitmap(bitmap)

    pairs = [(red, two_or_less), (red, index.eq("color", "blue")), (two_or_less, index.eq("size", 3)),
             (red, Bitmap())]
    for left, right in pairs:
        lr, rr = run_view(left), run_view(right)
        check(lr.and_(rr).to_bitmap() == left.and_(right), "run AND must equal word AND")
        check(lr.or_(rr).to_bitmap() == left.or_(right), "run OR must equal word OR")
        check(lr.and_not(rr).to_bitmap() == left.and_not(right), "run AND-NOT must equal word AND-NOT")
        check(lr.cardinality() == left.cardinality(), "run cardinality must equal word cardinality")
    check(run_view(Bitmap()) == RunBitmap(), "the empty bitmap must compress to no runs")

    # 8. compression claim: clustered bitmaps collapse to few runs, uniform ones do not
    rows = []
    seed = 0xBADF00D
    for _ in range(400):
        seed, r = splitmix64(seed)
        if r % 4 == 0:
            rows.append(r % 4000)
    uniform = Bitmap()
    for row in sorted(set(rows)):
        uniform.add(row)
    clustered = Bitmap()
    for row in range(100, 140):
        clustered.add(row)
    for row in range(2000, 2040):
        clustered.add(row)
    uniform_runs, clustered_runs = RunBitmap.from_bitmap(uniform), RunBitmap.from_bitmap(clustered)
    check(uniform_runs.cardinality() == uniform.cardinality() and uniform.cardinality() > 50,
          "random bitmap must keep its cardinality through compression")
    check(uniform.cardinality() == len(set(rows)), "the random bitmap must hold its distinct rows")
    check(clustered_runs.cardinality() == 80, "clustered bitmap cardinality must be exact")
    check(clustered_runs.stored_words() < len(clustered.words),
          "clustered bitmaps must compress below their word footprint")
    check(uniform_runs.stored_words() * 4 > len(uniform.words),
          "uniformly spread bitmaps must not compress much")
    print(f"652 compressed runs clustered={len(clustered_runs.runs)} uniform={len(uniform_runs.runs)}")
    print(f"652 words clustered={len(clustered.words)} uniform={len(uniform.words)}")

    # 9. larger index cross-checked against brute force on random predicates
    big_colors = []
    big_sizes = []
    seed = 0x1234
    for _ in range(2000):
        seed, r = splitmix64(seed)
        big_colors.append(("c%d" % (r % 12)))
        seed, r = splitmix64(seed)
        big_sizes.append(r % 50)
    big = BitmapIndex({"color": big_colors, "size": big_sizes}, 2000)
    for trial in range(8):
        seed, r1 = splitmix64(seed)
        seed, r2 = splitmix64(seed)
        seed, r3 = splitmix64(seed)
        color, low = "c%d" % (r1 % 12), r2 % 50
        high = low + r3 % 10
        got = big.all_of([("color", color)]).and_(big.range("size", low, high))
        want = brute_rows({"color": big_colors, "size": big_sizes},
                          lambda row, c=color, lo=low, hi=high: row["color"] == c and lo <= row["size"] <= hi)
        check(got.rows() == want, f"indexed predicate {color} in [{low},{high}] must match brute force")
        check(RunBitmap.from_bitmap(got).to_bitmap() == got,
              "the compressed form must round-trip through runs")
    check(big.range("size", 0, 49).cardinality() == 2000, "the full range must select every row")

    # 10. determinism
    again = BitmapIndex({"color": big_colors, "size": big_sizes}, 2000)
    check(again.range("size", 10, 20) == big.range("size", 10, 20),
          "rebuilding the index must give identical bitmaps")

    print("PASS")


if __name__ == "__main__":
    main()
