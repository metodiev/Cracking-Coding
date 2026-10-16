"""Task 651 - Implement adaptive indexing (database cracking) prototype.  (Tier: Expert, Ch 14 - Databases)

Database cracking builds the index *as a side effect of querying*: nothing is pre-sorted, every
range query partially partitions the physical array around its two boundaries. A cracking column
keeps the data in place and maintains a partition of the array into pieces; a piece is a
contiguous index range plus the value interval [low, high) it is known to hold. The invariant
checked after every query is: for every piece all values v satisfy low <= v < high, the lower bound
of a piece equals the upper bound of its predecessor (so the piece value ranges grow monotonically
with physical order), the pieces tile [0, n) exactly and the array stays a permutation of the
input. Cracking around a key k (two-pointer partition of one piece) splits a straddling piece into
its (< k) and (>= k) halves which inherit the parent's bounds.

Because the piece ranges are monotone, the pieces inside a query range are located by binary search,
and a query cracks a boundary at most once, so its cost is at most two partial scans plus the answer
size - pieces already fully inside the range are read without a single element comparison. That is
exactly why repeated and overlapping queries get monotonically cheaper (the adaptive-indexing
effect) while answers stay identical to a full scan.

Simplified versus the paper: the piece table is a plain list of tuples rather than a compressed
(piece, bound) map on disk, single-threaded (no concurrent cracks), and crack keys are always query
boundaries (no learned/AVL structure over the pieces). Verified: answers cross-checked against a
brute-force scan on many random queries and columns, the piece invariant re-validated after every
single query, and the adaptation measured as comparison counts against a full-scan baseline.
"""
import math

NEG_INF = -math.inf
POS_INF = math.inf
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_column(count, high, seed):
    values = []
    for _ in range(count):
        seed, r = splitmix64(seed)
        values.append(r % high)
    return values, seed


class CrackingColumn:
    """Adaptive (cracking) column: values kept in place plus a monotone piece table."""

    def __init__(self, values):
        self.data = list(values)
        self.pieces = [(0, len(self.data), NEG_INF, POS_INF)]  # (start, end, low, high)
        self.comparisons = 0          # element comparisons spent by cracks and scans
        self.search_comparisons = 0   # comparisons spent walking the piece table
        self.moves = 0                # element swaps performed by cracks
        self.cracks = 0
        self.queries = 0

    # ------------------------------------------------------------------ piece table
    def _find_straddle(self, key):
        """Index of a piece with low < key < high, or -1 when the table is already refined there."""
        pieces = self.pieces
        lo, hi = 0, len(pieces)
        while lo < hi:                      # first piece whose upper bound exceeds key
            mid = (lo + hi) // 2
            self.search_comparisons += 1
            if pieces[mid][3] <= key:
                lo = mid + 1
            else:
                hi = mid
        if lo >= len(pieces):
            return -1
        self.search_comparisons += 1
        return lo if pieces[lo][2] < key else -1

    def _crack(self, index, key):
        """Partition piece `index` so that its values < key come first; split the piece in two."""
        start, end, low, high = self.pieces[index]
        data = self.data
        left, right = start, end - 1
        while True:
            while left <= right:
                self.comparisons += 1
                if data[left] < key:
                    left += 1
                else:
                    break
            while left <= right:
                self.comparisons += 1
                if data[right] >= key:
                    right -= 1
                else:
                    break
            if left >= right:
                break
            data[left], data[right] = data[right], data[left]
            self.moves += 2
            left += 1
            right -= 1
        self.pieces[index] = (start, left, low, key)
        self.pieces.insert(index + 1, (left, end, key, high))
        self.cracks += 1

    def _refine(self, key):
        while True:
            index = self._find_straddle(key)
            if index < 0:
                return
            self._crack(index, key)

    def _first_piece_at_least(self, low):
        lo, hi = 0, len(self.pieces)
        while lo < hi:
            mid = (lo + hi) // 2
            self.search_comparisons += 1
            if self.pieces[mid][2] < low:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _last_piece_at_most(self, high):
        lo, hi = 0, len(self.pieces)
        while lo < hi:
            mid = (lo + hi) // 2
            self.search_comparisons += 1
            if self.pieces[mid][3] <= high:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    # ------------------------------------------------------------------ queries
    def query(self, low, high):
        """Every value v with low <= v <= high, ascending. Cracks the column while answering."""
        self.queries += 1
        if high < low:
            return []
        upper = high + 1                      # piece bounds are half-open
        self._refine(low)
        self._refine(upper)
        first = self._first_piece_at_least(low)
        last = self._last_piece_at_most(upper)
        result = []
        for index in range(first, last + 1):
            start, end, _, _ = self.pieces[index]
            result.extend(self.data[start:end])
        result.sort()
        return result

    def brute(self, low, high):
        """Reference answer: a plain full scan compared twice per row (the no-index baseline)."""
        result = [value for value in self.data if low <= value <= high]
        result.sort()
        return result

    # ------------------------------------------------------------------ invariants
    def check_invariants(self):
        pieces = self.pieces
        n = len(self.data)
        check(pieces[0][0] == 0 and pieces[-1][1] == n,
              "piece table must tile the column from 0 to n")
        for index, (start, end, low, high) in enumerate(pieces):
            check(start <= end, "a piece may not start beyond its end")
            check(low < high, "a piece value interval must be non-empty")
            if index + 1 < len(pieces):
                check(end == pieces[index + 1][0], "pieces must be contiguous")
                check(high == pieces[index + 1][2], "piece bounds must chain")
            for value in self.data[start:end]:
                check(low <= value < high, "a value escaped its piece interval")
        for index in range(len(pieces) - 1):
            start, end, _, _ = pieces[index]
            next_start, next_end, next_low, _ = pieces[index + 1]
            if start < end and next_start < next_end:
                check(max(self.data[start:end]) < next_low,
                      "piece value ranges must increase with physical order")


def main():
    # 1. hand-checked example
    column = CrackingColumn([5, 3, 8, 1, 9])
    check(column.query(3, 8) == [3, 5, 8], "hand example [3,8] must return 3,5,8")
    column.check_invariants()
    check(sorted(column.data) == [1, 3, 5, 8, 9], "cracking must permute, never lose values")

    # 2. degenerate ranges
    check(column.query(7, 7) == [], "point range falling between two values")
    check(column.query(9, 9) == [9], "point range on the maximum")
    check(column.query(10, 20) == [], "range entirely above the data")
    check(column.query(2, 1) == [], "inverted range must answer empty")
    column.check_invariants()

    # 3. brute-force cross-check on random columns with repeated/overlapping queries
    seed = 0x5EED
    for _ in range(6):
        values, seed = random_column(60, 25, seed)
        cracked = CrackingColumn(values)
        for _ in range(12):
            seed, r1 = splitmix64(seed)
            seed, r2 = splitmix64(seed)
            low, high = r1 % 30, r2 % 30
            check(cracked.query(low, high) == cracked.brute(low, high),
                  f"query [{low},{high}] disagreed with the brute-force scan")
            cracked.check_invariants()
        check(len(cracked.pieces) <= 1 + 2 * cracked.queries,
              "a query may add at most two pieces")
        check(cracked.cracks <= 2 * cracked.queries,
              "a query may crack at most two boundaries")

    # 4. duplicates and constant columns
    constant = CrackingColumn([4] * 40)
    check(constant.query(4, 4) == [4] * 40, "constant column must return every row")
    check(constant.query(3, 3) == [] and constant.query(5, 5) == [],
          "constant column must answer empty outside its only value")
    constant.check_invariants()

    # 5. adaptation: one wide query covering the whole domain, then full-range reads
    values, seed = random_column(1000, 4000, 0xC0FFEE)
    cover = CrackingColumn(values)
    check(cover.query(0, 4000) == sorted(values), "full-domain query must return the sorted column")
    cover.check_invariants()
    cost_wide = cover.comparisons
    before = cover.comparisons
    for _ in range(200):
        answer = cover.query(0, 4000)
    cost_repeat = cover.comparisons - before
    check(len(answer) == 1000, "repeated full-domain query must keep answering")
    check(cost_repeat // 200 < cost_wide,
          "a cracked query must cost less than the first cracking scan")

    # 6. adaptation under a repeated narrow range: the classic cracking speed-up
    warm = CrackingColumn(values)
    warm.query(1000, 1100)
    cost_first = warm.comparisons
    before = warm.comparisons
    for _ in range(200):
        answer = warm.query(1000, 1100)
    cost_later = warm.comparisons - before
    baseline = 200 * 2 * len(values)          # 200 full scans, two comparisons per row
    check(answer == warm.brute(1000, 1100), "warm query must still match a brute-force scan")
    check(cost_first >= len(values), "the first query must pay for cracking the array")
    check(cost_later * 20 < baseline,
          "repeated queries must beat repeated full scans by more than 20x")
    check(cost_later // 200 * 10 < cost_first,
          "the amortised per-query cost must drop after cracking")
    warm.check_invariants()

    # 7. mixed workload of distinct but overlapping ranges
    work = CrackingColumn(values)
    ranges = [(500 + 100 * i, 560 + 100 * i) for i in range(10)]
    for low, high in ranges:
        work.query(low, high)
    for _ in range(20):
        for low, high in ranges:
            check(work.query(low, high) == work.brute(low, high),
                  "mixed workload answer diverged from the brute-force scan")
    mixed = work.comparisons
    mixed_baseline = 200 * 2 * len(values)
    check(mixed < mixed_baseline // 5,
          "adaptive indexing must beat 200 full scans by at least 5x")
    work.check_invariants()

    # 8. determinism: identical input and query stream give identical state and counters
    first_run, second_run = CrackingColumn(values), CrackingColumn(values)
    for low, high in ranges:
        first_run.query(low, high)
        second_run.query(low, high)
    check(first_run.data == second_run.data and first_run.comparisons == second_run.comparisons,
          "cracking must be deterministic")
    check(first_run.pieces == second_run.pieces, "identical query streams must produce identical pieces")

    # 9. narrow queries far apart do not disturb each other's pieces
    sparse = CrackingColumn(values)
    sparse.query(3000, 3010)
    pieces_after = len(sparse.pieces)
    sparse.query(3000, 3010)
    check(len(sparse.pieces) == pieces_after, "a repeated query must not add new cracks")
    check(sparse.search_comparisons > 0, "piece lookups must be counted")

    print(f"651 pieces={len(work.pieces)} cracks={work.cracks} queries={work.queries}")
    print(f"651 comparisons cracked={mixed} baseline={mixed_baseline} wide={cost_wide}")
    print("PASS")


if __name__ == "__main__":
    main()
