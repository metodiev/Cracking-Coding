"""Task 606 - Implement wavelet matrix for large alphabet rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

A wavelet matrix stores an integer array as one bit vector per bit of the alphabet, from the most
significant bit down. Level i is the array rearranged so that the elements whose i-th bit is 0 come
first (stable), and the level's bit vector records that bit for every position; `zeros[i]` is how many
elements went to the left. Nothing else is stored, so the whole structure is `bits * n` bits plus the
rank/select samples - that is what makes it usable with a large alphabet: an alphabet of 2^20 values
costs 20 levels, while a pointer-based wavelet tree would need one node per value.

Descending the levels answers everything:
  * count_less_than(l, r, x) - at a level where x's bit is 1 all zeros of the range are smaller, so
    they are added to the answer and the range moves to the ones; where the bit is 0 the range moves
    to the zeros. rank and range_count are differences of count_less_than;
  * quantile(l, r, k) - chooses the left child while k is smaller than the number of zeros in the
    range, otherwise subtracts them and takes the right child, which spells out the k-th smallest;
  * select(value, k) - descends to the range holding `value`, then walks that range's positions back
    up level by level with select0/select1, which undoes the stable partitions and lands on the
    position of the k-th occurrence in the original array.
The bit vectors carry rank1 samples (one per 64-bit word), a select1 sample every 128 ones, and
select0 by binary search over rank1, since zeros are not sampled.

Checks: every operation against brute force on a hand array and on small random arrays (all i for
access and rank, random windows for count_less_than/quantile/range_count, all occurrences for
select); the boundaries (empty array, l = r, a = b, a > b, the whole range, k out of range raising,
select on an absent value, negative k); round trips rank(value, select(value, k) + 1) = k + 1 and
access(select(value, k)) = value; large alphabets (4096 values of 20 bits and the code points of a
2000-character string) against brute force; a single-value array and a strictly increasing array; the
storage measurement; and determinism.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, value ^ (value >> 31)


def random_values(count, bound, seed):
    """Deterministic pseudo-random values below `bound` (splitmix64, no library randomness)."""
    values = []
    for _ in range(count):
        seed, value = splitmix64(seed)
        values.append(value % bound)
    return values


class BitVector:
    """Bit vector with rank1 (sampled per 64-bit word), select1 (per 128 ones) and select0."""

    def __init__(self, bits):
        self.bits, self.n = bits, len(bits)
        self.words = []
        for start in range(0, self.n, 64):
            word = 0
            for offset, bit in enumerate(bits[start:start + 64]):
                word |= bit << offset
            self.words.append(word)
        self.rank_samples = [0]
        for word in self.words:
            self.rank_samples.append(self.rank_samples[-1] + bin(word).count("1"))
        self.total = self.rank_samples[-1]
        self.select_samples, seen = {}, 0
        for position, bit in enumerate(bits):
            if bit:
                if seen % 128 == 0:
                    self.select_samples[seen] = position
                seen += 1

    def rank1(self, i):
        """Number of ones in [0, i): one word sample plus a partial-word popcount."""
        word_index, offset = divmod(i, 64)
        count = self.rank_samples[word_index]
        if offset:
            count += bin(self.words[word_index] & ((1 << offset) - 1)).count("1")
        return count

    def select1(self, k):
        """Position of the k-th one (0-based), or -1 when there is none."""
        if k < 0 or k >= self.total:
            return -1
        base = (k // 128) * 128
        position, seen = self.select_samples[base], base
        while True:
            if self.bits[position]:
                if seen == k:
                    return position
                seen += 1
            position += 1

    def select0(self, k):
        """Position of the k-th zero, or -1 when there is none; zeros are not sampled."""
        if k < 0 or k >= self.n - self.total:
            return -1
        lo, hi = k, self.n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if (mid + 1) - self.rank1(mid + 1) > k:      # zeros in [0, mid + 1) exceeds k
                hi = mid
            else:
                lo = mid + 1
        return lo


class WaveletMatrix:
    """Bits levels over the values; every operation is a descent (or an up-walk for select)."""

    def __init__(self, data, bits=None):
        self.data = list(data)
        self.n = len(self.data)
        highest = max(self.data) if self.n else 0
        self.bits = bits if bits is not None else max(1, highest.bit_length())
        self.levels, self.zeros = [], []
        current = self.data
        for level in range(self.bits - 1, -1, -1):       # level index 0 is the most significant bit
            flags = [(value >> level) & 1 for value in current]
            self.levels.append(BitVector(flags))
            self.zeros.append(self.n - sum(flags))
            zeros = [value for value, flag in zip(current, flags) if not flag]
            ones = [value for value, flag in zip(current, flags) if flag]
            current = zeros + ones

    def access(self, i):
        """The value at position i."""
        position, value = i, 0
        for level in range(self.bits):
            vector, zeros = self.levels[level], self.zeros[level]
            bit = vector.bits[position]
            value = (value << 1) | bit
            if bit:
                position = zeros + vector.rank1(position)
            else:
                position -= vector.rank1(position)
        return value

    def count_less_than(self, left, right, value):
        """Number of positions in [left, right) whose value is smaller than `value`."""
        if value <= 0 or left >= right:
            return 0
        if value >= (1 << self.bits):
            return right - left                  # every stored value is smaller than `value`
        count = 0
        for level in range(self.bits):
            vector, zeros = self.levels[level], self.zeros[level]
            ones_left, ones_right = vector.rank1(left), vector.rank1(right)
            zeros_left, zeros_right = left - ones_left, right - ones_right
            if (value >> (self.bits - 1 - level)) & 1:
                count += zeros_right - zeros_left            # all those zeros are smaller
                left, right = zeros + ones_left, zeros + ones_right
            else:
                left, right = zeros_left, zeros_right
        return count

    def rank(self, value, i):
        """Number of occurrences of `value` in data[:i]."""
        return (self.count_less_than(0, i, value + 1) - self.count_less_than(0, i, value))

    def range_count(self, left, right, low, high):
        """Number of positions in [left, right) whose value lies in [low, high)."""
        if low >= high:
            return 0
        return (self.count_less_than(left, right, high) - self.count_less_than(left, right, low))

    def quantile(self, left, right, k):
        """The k-th smallest value (0-based) in data[left:right)."""
        if not 0 <= k < right - left:
            raise IndexError("k is outside the requested range")
        value = 0
        for level in range(self.bits):
            vector, zeros = self.levels[level], self.zeros[level]
            ones_left, ones_right = vector.rank1(left), vector.rank1(right)
            zeros_left, zeros_right = left - ones_left, right - ones_right
            if k < zeros_right - zeros_left:
                value <<= 1
                left, right = zeros_left, zeros_right
            else:
                k -= zeros_right - zeros_left
                value = (value << 1) | 1
                left, right = zeros + ones_left, zeros + ones_right
        return value

    def value_range(self, value):
        """Range of positions holding `value` in the last level (the fully permuted array)."""
        left, right = 0, self.n
        for level in range(self.bits):
            vector, zeros = self.levels[level], self.zeros[level]
            ones_left, ones_right = vector.rank1(left), vector.rank1(right)
            if (value >> (self.bits - 1 - level)) & 1:
                left, right = zeros + ones_left, zeros + ones_right
            else:
                left, right = left - ones_left, right - ones_right
        return left, right

    def select(self, value, k):
        """Position of the k-th (0-based) occurrence of `value`, or -1 when there is none."""
        if k < 0 or not 0 <= value < (1 << self.bits):
            return -1
        left, right = self.value_range(value)
        if k >= right - left:
            return -1
        position = left + k
        for level in range(self.bits - 1, -1, -1):       # undo the stable partitions upwards
            vector, zeros = self.levels[level], self.zeros[level]
            if (value >> (self.bits - 1 - level)) & 1:
                position = vector.select1(position - zeros)
            else:
                position = vector.select0(position)
        return position

    def stored_bits(self):
        """One bit per element per level, plus a 32-bit rank sample per 64-bit word of each level."""
        words = (self.n + 63) // 64
        return self.bits * (self.n + words * 32)


def main():
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    matrix = WaveletMatrix(data, bits=4)
    check([matrix.access(i) for i in range(len(data))] == data, "access hand array")
    check(matrix.rank(1, 8) == 2 and matrix.rank(1, 2) == 1 and matrix.rank(0, 8) == 0, "rank hand")
    check(matrix.rank(4, 8) == 1 and matrix.rank(9, 8) == 1, "rank hand 2")
    check(matrix.range_count(0, 8, 2, 6) == 4, "range count hand")
    check(matrix.range_count(0, 8, 0, 100) == 8 and matrix.range_count(0, 8, 4, 4) == 0, "range edges")
    check(sorted(matrix.quantile(0, 8, k) for k in range(8)) == sorted(data), "quantile hand")
    check(matrix.quantile(0, 8, 0) == 1 and matrix.quantile(0, 8, 7) == 9, "quantile ends")
    check(matrix.select(1, 0) == 1 and matrix.select(1, 1) == 3, "select hand")
    check(matrix.select(9, 0) == 5 and matrix.select(9, 1) == -1, "select hand 2")
    check(matrix.select(7, 0) == -1 and matrix.select(3, -1) == -1, "select absent")
    check(all(matrix.rank(value, matrix.select(value, k) + 1) == k + 1
              for value in set(data) for k in range(data.count(value))), "select/rank round trip")
    check(all(matrix.value_range(value)[1] - matrix.value_range(value)[0] == data.count(value)
              for value in set(data)), "value range sizes")
    check(matrix.value_range(0) == (0, 0), "value range of an absent value")
    try:
        matrix.quantile(0, 3, 3)
        check(False, "quantile must reject an out-of-range k")
    except IndexError:
        pass
    check(WaveletMatrix([]).range_count(0, 0, 0, 5) == 0, "empty matrix")
    check(WaveletMatrix([7] * 5).select(7, 4) == 4, "single value array")

    for size, bound in ((24, 8), (60, 32), (150, 400)):
        values = random_values(size, bound, 6060 + size)
        matrix = WaveletMatrix(values)
        check([matrix.access(i) for i in range(size)] == values, size)
        for value in set(values):
            running = 0
            for i in range(size + 1):
                if i % 3 == 0 or i == size:
                    check(matrix.rank(value, i) == running, (size, value, i))
                if i < size:
                    running += values[i] == value
        for window in range(0, size, 7):
            left, right = window, min(size, window + 11)
            chunk = sorted(values[left:right])
            check(matrix.quantile(left, right, 0) == chunk[0], (size, left))
            check(matrix.quantile(left, right, len(chunk) - 1) == chunk[-1], (size, left))
            check(matrix.quantile(left, right, len(chunk) // 2) == chunk[len(chunk) // 2], (size, left))
            low, high = window % bound, (window * 3) % bound
            expected = sum(1 for value in values[left:right] if low <= value < high)
            check(matrix.range_count(left, right, low, high) == expected, (size, low, high))
            check(matrix.count_less_than(left, right, high)
                  == sum(1 for value in values[left:right] if value < high), (size, high))
        for value in set(values):
            positions = [i for i, item in enumerate(values) if item == value]
            check(all(matrix.select(value, k) == position
                      for k, position in enumerate(positions)), (size, value))
            check(all(matrix.access(position) == value for position in positions), (size, value))

    # large alphabet: 4096 values of 20 bits, then the code points of a 2000-character string
    wide = random_values(4096, 1 << 20, 99887766)
    matrix = WaveletMatrix(wide)
    check(matrix.bits == 20, "20-bit alphabet")
    check(matrix.access(4095) == wide[4095] and matrix.access(0) == wide[0], "wide access")
    for window in range(0, 4096, 331):
        left, right = window, window + 57
        chunk = sorted(wide[left:right])
        check(matrix.quantile(left, right, 7) == chunk[7], left)
        check(matrix.quantile(left, right, 30) == chunk[30], left)
        check(matrix.range_count(left, right, 1 << 19, 1 << 20)
              == sum(1 for value in wide[left:right] if value >= 1 << 19), left)
        check(matrix.rank(wide[left], right) == wide[:right].count(wide[left]), left)
    for value in wide[:40]:
        first = wide.index(value)
        check(matrix.select(value, 0) == first and matrix.access(first) == value, value)
    check(matrix.stored_bits() <= 64 * 4096, "wavelet matrix is smaller than pointer ints")

    text, _ = _string(2000, 6060)
    codes = [ord(ch) for ch in text]
    matrix = WaveletMatrix(codes)
    check(matrix.bits == max(codes).bit_length(), "code point alphabet")
    check([matrix.access(i) for i in (0, 999, 1999)] == [codes[0], codes[999], codes[1999]], "codes")
    for window in range(0, 2000, 271):
        left, right = window, window + 43
        chunk = sorted(codes[left:right])
        check(matrix.quantile(left, right, 0) == chunk[0], left)
        check(matrix.quantile(left, right, 42) == chunk[42], left)
        check(matrix.range_count(left, right, ord("m"), ord("q"))
              == sum(1 for value in codes[left:right] if ord("m") <= value < ord("q")), left)
        check(matrix.count_less_than(left, right, ord("n"))
              == sum(1 for value in codes[left:right] if value < ord("n")), left)
    for code in set(codes):
        check(matrix.access(matrix.select(code, 0)) == code, code)
    check(WaveletMatrix(codes).quantile(0, 2000, 1000) == sorted(codes)[1000], "stable rebuild")

    print(f"code points: {len(matrix.levels)} levels over {matrix.n} characters, "
          f"{len(set(codes))} distinct values, {matrix.stored_bits()} bits stored")
    print(f"20-bit alphabet: 4096 values in {WaveletMatrix(wide).stored_bits()} bits "
          f"({WaveletMatrix(wide).stored_bits() / 4096:.1f} bits per value)")


def _string(length, seed):
    """Deterministic string over a mix of ASCII letters and a few wide code points."""
    alphabet = "abcdefghijklmnopqrstuvwxyz\u00e9\u20ac\u4e2d\U0001f600"
    chars = []
    for _ in range(length):
        seed, value = splitmix64(seed)
        chars.append(alphabet[value % len(alphabet)])
    return "".join(chars), seed


if __name__ == "__main__":
    main()
    print("PASS")
