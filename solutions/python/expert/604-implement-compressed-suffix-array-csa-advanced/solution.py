"""Task 604 - Implement compressed suffix array (CSA) advanced. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

A compressed suffix array in the Grossi-Vitter sense: instead of the n suffix-array entries the index
stores psi - psi[i] is the rank of the suffix one position to the right of the suffix at rank i, so
SA[psi[i]] = (SA[i] + 1) mod n - plus a sparse sample of SA values, a marker bit vector with
rank1/select1 support telling which rows are sampled, and the LCP array. A missing SA value is
recovered by walking psi until a sampled row is hit: SA[i] = SA[sampled] - steps (mod n). Sampling by
*text position* (rows whose SA value is a multiple of the step) makes the walk bounded: every step
moves the text position one to the left, so a multiple of `step` is reached within `step - 1` steps -
a bound asserted on every walk. All values are bit-packed at width ceil(log2 n) instead of machine
words; stored_bits() counts n psi values + n LCP values + one sampled SA value per `step` rows + n
marker bits + coarse rank samples, far below the 64 bits per entry a plain suffix array with LCP needs.

The suffix array itself is built by prefix doubling (O(n log^2 n) with tuple keys); which construction
is used is orthogonal to the compression - the advanced tier covers SA-IS for that - and Kasai's scan
produces the LCP. Checks: doubling against a naive sort and against a manual character-by-character
comparator on hand, unary and LCG texts; Kasai against a direct scan; the defining psi identity for
every rank; full reconstruction of every SA entry from psi + samples + marker with the walk bound;
count/locate by binary search over the reconstructed array against naive search; identical answers for
six sampling steps plus the space/time trade-off between them; rank/select against direct bit counting
including out-of-range select; the storage measurement.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed=12345):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def suffix_array(text):
    """Prefix doubling: sort the suffixes by their first 2^k characters, doubling k."""
    n = len(text)
    if n == 0:
        return []
    sa, rank, k = list(range(n)), [ord(c) for c in text], 1
    while True:
        keys = {i: (rank[i], rank[i + k] if i + k < n else -1) for i in range(n)}
        sa.sort(key=keys.__getitem__)
        new_rank = [0] * n
        for idx in range(1, n):
            prev, cur = sa[idx - 1], sa[idx]
            new_rank[cur] = new_rank[prev] + (0 if keys[prev] == keys[cur] else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            return sa
        k <<= 1


def kasai_lcp(text, sa):
    """lcp[i] = longest common prefix of sa[i - 1] and sa[i] (lcp[0] = 0), in linear time."""
    n = len(text)
    rank = [0] * n
    for i, start in enumerate(sa):
        rank[start] = i
    lcp, h = [0] * n, 0
    for i in range(n):
        position = rank[i]
        if position:
            j = sa[position - 1]
            while i + h < n and j + h < n and text[i + h] == text[j + h]:
                h += 1
            lcp[position] = h
            h = max(h - 1, 0)
        else:
            h = 0
    return lcp


def compare_suffixes(text, i, j):
    """Manual comparator; the independent validity oracle for a suffix array."""
    n = len(text)
    while i < n and j < n and text[i] == text[j]:
        i += 1
        j += 1
    if i == n and j == n:
        return 0
    if i == n:
        return -1                                   # the shorter suffix is the smaller one
    if j == n:
        return 1
    return -1 if text[i] < text[j] else 1


def valid_suffix_array(text, sa):
    return (sorted(sa) == list(range(len(text)))
            and all(compare_suffixes(text, sa[k], sa[k + 1]) < 0 for k in range(len(sa) - 1)))


def pack(values, width):
    """Pack fixed-width integers into one integer: value i in bits [i*width, (i+1)*width)."""
    packed = 0
    for index, value in enumerate(values):
        packed |= value << (index * width)
    return packed


class BitVector:
    """Bit vector with rank1 (a sample per 64-bit word) and select1 (a sample per 128 ones)."""

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
        """Number of ones in [0, i)."""
        word_index, offset = divmod(i, 64)
        count = self.rank_samples[word_index]
        return count + bin(self.words[word_index] & ((1 << offset) - 1)).count("1") if offset else count

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


class CompressedSuffixArray:
    """psi + sampled SA values + marker + packed LCP. `reference_sa`/`reference_lcp` exist only so the
    checks can compare with the uncompressed arrays; they are not part of the index and are not
    counted by stored_bits()."""

    def __init__(self, text, step=8):
        n = len(text)
        self.text, self.n, self.step = text, n, step
        self.width = max(1, (n - 1).bit_length())
        self.reference_sa = sa = suffix_array(text)
        self.reference_lcp = lcp = kasai_lcp(text, sa)
        self.width_lcp = max(1, max(lcp).bit_length()) if n else 1
        rank_of = [0] * n
        for rank, position in enumerate(sa):
            rank_of[position] = rank
        # psi of the suffix at each position = the rank of the suffix one position to the right
        self.nxt = [rank_of[(position + 1) % n] for position in sa] if n else []
        self.psi_bits = pack(self.nxt, self.width)
        sampled_flags = [1 if value % step == 0 else 0 for value in sa]
        self.marker = BitVector(sampled_flags)
        self.sampled_count = sum(sampled_flags)
        self.sample_bits = pack([sa[rank] for rank in range(n) if sampled_flags[rank]], self.width)
        self.lcp_bits = pack(lcp, self.width_lcp)
        self.walk_steps = 0

    def psi(self, rank):
        return (self.psi_bits >> (rank * self.width)) & ((1 << self.width) - 1)

    def lcp(self, rank):
        return (self.lcp_bits >> (rank * self.width_lcp)) & ((1 << self.width_lcp) - 1)

    def sample_value(self, rank):
        """SA value stored for a sampled row; the row holds the rank1(rank)-th one of the marker."""
        return (self.sample_bits >> (self.marker.rank1(rank) * self.width)) & ((1 << self.width) - 1)

    def suffix_at(self, rank):
        """SA[rank], recovered by following psi until a sampled row is reached (bounded walk)."""
        steps, row = 0, rank
        while not self.marker.bits[row]:
            row = self.psi(row)
            steps += 1
            if steps > self.step:
                raise AssertionError(f"psi walk from row {rank} exceeded {self.step} steps")
        self.walk_steps += steps
        return (self.sample_value(row) - steps) % self.n

    def compare(self, rank, pattern):
        """Compare the suffix at `rank` with `pattern`: negative, zero or positive."""
        position = self.suffix_at(rank)
        limit = min(len(pattern), self.n - position)
        for offset in range(limit):
            if self.text[position + offset] != pattern[offset]:
                return -1 if self.text[position + offset] < pattern[offset] else 1
        return 0 if limit == len(pattern) else -1

    def bound(self, pattern, upper):
        """First row whose suffix is >= pattern (or > pattern when `upper`)."""
        lo, hi = 0, self.n
        while lo < hi:
            mid = (lo + hi) // 2
            verdict = self.compare(mid, pattern)
            if verdict < 0 or (upper and verdict == 0):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def count(self, pattern):
        return 0 if not pattern else self.bound(pattern, True) - self.bound(pattern, False)

    def locate(self, pattern):
        if not pattern:
            return []
        lo, hi = self.bound(pattern, False), self.bound(pattern, True)
        return sorted(self.suffix_at(rank) for rank in range(lo, hi))

    def longest_repeated_substring(self):
        """Longest interior LCP value; the substring is read at the corresponding suffix."""
        best, rank = 0, -1
        for i in range(1, self.n):
            if self.lcp(i) > best:
                best, rank = self.lcp(i), i
        return "" if rank == -1 else self.text[self.suffix_at(rank):self.suffix_at(rank) + best]

    def stored_bits(self):
        """psi + LCP + sampled SA values + marker bits + one rank sample per 64-bit word."""
        return (self.n * (self.width + self.width_lcp + 1) + self.sampled_count * self.width
                + len(self.marker.words) * self.width)


def brute_positions(text, pattern):
    out = []
    start = text.find(pattern)
    while start != -1:
        out.append(start)
        start = text.find(pattern, start + 1)
    return out


def brute_lcp(text, i, j):
    length = 0
    while i + length < len(text) and j + length < len(text) and text[i + length] == text[j + length]:
        length += 1
    return length


def main():
    csa = CompressedSuffixArray("banana", step=3)
    check(csa.count("ana") == 2 and csa.locate("ana") == [1, 3], "banana ana")
    check(csa.count("na") == 2 and csa.count("a") == 3, "banana counts")
    check(csa.count("banana") == 1 and csa.locate("banana") == [0], "banana whole")
    check(csa.count("bananas") == 0 and csa.count("x") == 0 and csa.count("") == 0, "banana absent")
    check(csa.locate("x") == [] and csa.locate("") == [], "banana absent positions")
    check(csa.suffix_at(0) == 5 and csa.suffix_at(5) == 2, "banana suffix_at")
    check(csa.psi(0) == 3 and csa.psi(4) == 0, "banana psi values")
    check(csa.marker.total == sum(1 for value in csa.reference_sa if value % 3 == 0), "samples")

    texts = ["", "a", "aaaa", "banana", "mississippi", "abracadabra", "abcabxabcd"]
    texts += [lcg_string(60, alphabet, 6040 + trial * 7919)[0]
              for alphabet in ("ab", "abc") for trial in range(8)]
    for text in texts:
        sa = suffix_array(text)
        lcp = kasai_lcp(text, sa)
        check(sa == sorted(range(len(text)), key=lambda i: text[i:]), text)
        check(not lcp or (lcp[0] == 0 and all(lcp[i] == brute_lcp(text, sa[i - 1], sa[i])
                                              for i in range(1, len(text)))), text)

    # psi identity, bounded walks and full SA reconstruction from psi + samples + marker
    for text, step in (("banana", 3), ("mississippi", 2), ("aaaa", 1), ("abcabxabcd", 7),
                       (lcg_string(400, "abc", 4711)[0], 5)):
        csa, n = CompressedSuffixArray(text, step=step), len(text)
        for rank in range(n):
            check(csa.suffix_at(rank) == csa.reference_sa[rank], (text, rank))
            check((csa.suffix_at(csa.psi(rank)) - csa.suffix_at(rank)) % n == 1, (text, rank))
        check(all(csa.lcp(i) == csa.reference_lcp[i] for i in range(1, n)), text)
        answer = csa.longest_repeated_substring()
        check(len(answer) == max(csa.reference_lcp), text)
        if answer:
            check(csa.locate(answer) == brute_positions(text, answer), (text, answer))

    # sampling steps must answer identically; a smaller step shortens every walk but stores more
    base = CompressedSuffixArray("mississippi", step=1)
    patterns = ["i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q", "mississippi"]
    for step in (1, 2, 3, 5, 8, 16):
        variant = CompressedSuffixArray("mississippi", step=step)
        for pattern in patterns:
            check(variant.count(pattern) == base.count(pattern), (step, pattern))
            check(variant.locate(pattern) == base.locate(pattern), (step, pattern))
        check(variant.stored_bits() <= 40 * len("mississippi"), step)
    fine, coarse = CompressedSuffixArray("mississippi", 2), CompressedSuffixArray("mississippi", 8)
    for pattern in patterns:
        fine.locate(pattern)
        coarse.locate(pattern)
    check(fine.walk_steps < coarse.walk_steps and fine.stored_bits() > coarse.stored_bits(), "steps")

    # rank/select primitives against direct bit counting, plus out-of-range select
    bits = [1 if value % 3 == 0 else 0 for value in range(300)]
    vector = BitVector(bits)
    ones = [i for i, bit in enumerate(bits) if bit]
    check(vector.total == len(ones), "bit vector total")
    check(all(vector.rank1(i) == sum(bits[:i]) for i in range(301)), "rank1")
    check(all(vector.select1(k) == position for k, position in enumerate(ones)), "select1")
    check(vector.select1(len(ones)) == -1 and vector.select1(-1) == -1, "select out of range")

    # medium text: the comparator oracle is exact at this size
    medium, _ = lcg_string(8000, "abc", 20260404)
    csa = CompressedSuffixArray(medium, step=11)
    check(valid_suffix_array(medium, csa.reference_sa), "medium suffix array valid")
    for rank in range(0, csa.n, 97):
        check(csa.suffix_at(rank) == csa.reference_sa[rank], rank)
        check((csa.suffix_at(csa.psi(rank)) - csa.suffix_at(rank)) % csa.n == 1, rank)
    for window in range(0, csa.n - 20, 501):
        piece = medium[window:window + 20]
        check(csa.count(piece) == len(brute_positions(medium, piece)), piece)
        check(csa.locate(piece) == brute_positions(medium, piece), piece)
    check(csa.locate(medium) == [0] and csa.count(medium) == 1, "medium whole text")
    answer = csa.longest_repeated_substring()
    check(len(answer) == max(csa.reference_lcp)
          and csa.count(answer) >= 2 and answer in medium, "medium LRS")

    print(f"banana: {CompressedSuffixArray('banana').stored_bits()} index bits versus "
          f"{2 * 6 * 64} bits for a plain suffix array with LCP")
    print(f"medium text: {csa.n} characters, {csa.stored_bits()} index bits "
          f"({csa.stored_bits() / csa.n:.1f} per character), sampling step {csa.step}, "
          f"longest repeat {len(answer)}")


if __name__ == "__main__":
    main()
    print("PASS")
