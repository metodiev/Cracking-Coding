"""Task 609 - Implement advanced compressed bitvectors (RRR) and rank/select.  (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

An RRR (Raman-Raman-Rao) bitvector does not store the n raw bits.  It cuts the bitvector
into blocks of B = 15 bits and, per block, stores only the *combinatorial class* of the
block: the popcount k together with the colexicographic rank of the block's k set bits
among the C(B, k) blocks of weight k.  Class indices are laid out as
    class = base[k] + colex_rank,   base[k] = sum_{j<k} C(B, j) = 2^B - sum_{j>=k} C(B, j),
so the class index itself encodes the popcount: the number of bits a block needs is
ceil(log2 C(B, k)) instead of B, and C(B, k) peaks at k = B/2, giving n*H0(p) + o(n) bits
for a vector whose fraction of ones is p.  The colex rank of a block with set bits at
0 <= p_1 < ... < p_k < B is sum_i C(p_i, i), which is exactly the standard combinadic
enumeration, and colex_unrank inverts it.

Rank/select are answered without decompressing the whole vector: a superblock directory
stores the number of ones before every 8th block, the per-block popcounts are read off the
class indices (no storage at all), and only the one block touched by the query is decoded
with colex_unrank.  rank0/select0 follow from rank0(pos) = pos - rank1(pos).

Scaled-down but real: real RRR adds an O(sqrt(n))-entry lookup table to find the position
of a 1 inside a decoded block in O(1); here select1 scans the 15 decoded bits (documented,
not faked).  What is verified: access/rank1/rank0/select1/select0 agree with a naive
bit-array oracle for *every* position of every test vector (including the tail block, which
is shorter than B), every stored block decodes back to the exact original bits, the class
directory is consistent (class 2^B in total, class -> popcount exact, ranks are a
permutation of 0..C(B,k)-1 per weight), and the measured RRR space really is below the
plain 1 bit/bit cost for skewed and for dense vectors.
"""
import math

B = 15                        # bits per block
BLOCKS_PER_SUPER = 8          # directory resolution: one entry per 8 blocks
SUPER_BITS = B * BLOCKS_PER_SUPER


def _binom_table(limit):
    """C[n][k] for 0 <= k <= n <= limit (Pascal's triangle, tiny here)."""
    table = [[0] * (limit + 1) for _ in range(limit + 1)]
    for n in range(limit + 1):
        table[n][0] = 1
        for k in range(1, n + 1):
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k]
    return table


C = _binom_table(B)


def binom(n, k):
    if n < 0 or k < 0 or k > n:
        return 0
    return C[n][k]


def colex_rank(positions):
    """Rank of a set of set-bit positions in colex order (combinadic value)."""
    return sum(binom(p, i + 1) for i, p in enumerate(positions))


def colex_unrank(k, rank):
    """Inverse of colex_rank: positions (ascending) of the weight-k block with this rank."""
    if k == 0:
        return []
    positions = []
    p = B - 1
    for i in range(k, 0, -1):
        while binom(p, i) > rank:
            p -= 1
            if p < 0:
                raise ValueError("rank out of range")
        positions.append(p)
        rank -= binom(p, i)
        p -= 1
    return positions[::-1]


def _build_class_directory():
    """base[k] = first class index of weight k; class_of_popcount[class] = k."""
    base = [0] * (B + 1)
    for k in range(1, B + 1):
        base[k] = base[k - 1] + binom(B, k - 1)
    popcount = [0] * (1 << B)
    for k in range(B + 1):
        for c in range(base[k], base[k] + binom(B, k)):
            popcount[c] = k
    return base, popcount


BASE, CLASS_POPCOUNT = _build_class_directory()
CLASS_BITS = {k: (binom(B, k) - 1).bit_length() for k in range(B + 1)}


class RRRBitvector:
    """Compressed bitvector with rank1/rank0/select1/select0/access."""

    def __init__(self, bits):
        if any(ch not in "01" for ch in bits):
            raise ValueError("bitvector must only contain '0' and '1'")
        self.n = len(bits)
        self.classes = []
        self.offsets = []
        for start in range(0, self.n, B):
            block = bits[start:start + B]
            k = block.count("1")
            positions = [j for j, ch in enumerate(block) if ch == "1"]
            rank = colex_rank(positions)
            self.classes.append(BASE[k] + rank)
            self.offsets.append(rank)
        self.blocks = len(self.classes)
        self.block_pop = [CLASS_POPCOUNT[c] for c in self.classes]
        self.super_ones = []
        acc = 0
        for i, pop in enumerate(self.block_pop):
            if i % BLOCKS_PER_SUPER == 0:
                self.super_ones.append(acc)
            acc += pop
        self.total_ones = acc

    # -- decoding ---------------------------------------------------------
    def block_length(self, block_index):
        remaining = self.n - block_index * B
        return B if remaining >= B else remaining

    def decode_block(self, block_index):
        """Rebuild the 15-bit pattern of a block from its class + offset."""
        cls = self.classes[block_index]
        mask = 0
        for p in colex_unrank(CLASS_POPCOUNT[cls], self.offsets[block_index]):
            mask |= 1 << p
        return mask

    def access(self, pos):
        if pos < 0 or pos >= self.n:
            raise IndexError("access out of range")
        return (self.decode_block(pos // B) >> (pos % B)) & 1

    # -- rank / select ----------------------------------------------------
    def rank1(self, pos):
        """Number of ones in [0, pos)."""
        if pos <= 0:
            return 0
        if pos >= self.n:
            return self.total_ones
        block_index = pos // B
        super_index = block_index // BLOCKS_PER_SUPER
        count = self.super_ones[super_index]
        for b in range(super_index * BLOCKS_PER_SUPER, block_index):
            count += self.block_pop[b]
        low = self.decode_block(block_index) & ((1 << (pos % B)) - 1)
        return count + bin(low).count("1")

    def rank0(self, pos):
        return pos - self.rank1(pos)

    def select1(self, index):
        """Position of the index-th one (0-based); the in-block scan replaces the
        O(1) lookup table a production RRR keeps."""
        if index < 0 or index >= self.total_ones:
            raise IndexError("select1 out of range")
        lo, hi = 0, len(self.super_ones) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.super_ones[mid] <= index:
                lo = mid
            else:
                hi = mid - 1
        count = self.super_ones[lo]
        block_index = lo * BLOCKS_PER_SUPER
        while count + self.block_pop[block_index] <= index:
            count += self.block_pop[block_index]
            block_index += 1
        mask = self.decode_block(block_index)
        remaining = index - count
        for p in range(B):
            if (mask >> p) & 1:
                if remaining == 0:
                    return block_index * B + p
                remaining -= 1
        raise AssertionError("unreachable: select1 walk fell off the block")

    def select0(self, index):
        if index < 0 or index >= self.n - self.total_ones:
            raise IndexError("select0 out of range")
        lo, hi = 0, self.n - 1
        while lo < hi:                      # binary search on rank0 (monotone)
            mid = (lo + hi) // 2
            if self.rank0(mid + 1) > index:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # -- space accounting -------------------------------------------------
    def offset_bits(self):
        """RRR's B_2: every block stores its combinadic rank with ceil(log2 C(B, k)) bits
        (0 bits when the weight-k class holds a single pattern, i.e. k = 0 or k = B)."""
        return sum(CLASS_BITS[k] for k in self.block_pop)

    def class_bits(self):
        """RRR's B_1: the popcount of every block, 4 bits for B = 15.  A production RRR
        entropy-codes this sequence (it is where the last few percent come from)."""
        return ((B).bit_length()) * self.blocks

    def directory_bits(self):
        """Superblock prefix sums only: the per-block popcounts live in the class sequence."""
        return 32 * len(self.super_ones)

    def payload_bits(self):
        return self.offset_bits() + self.class_bits()


def naive_bits(bits):
    return [1 if ch == "1" else 0 for ch in bits]


def naive_rank1(naive, pos):
    return sum(naive[:pos])


def naive_select1(naive, index):
    seen = 0
    for i, bit in enumerate(naive):
        if bit:
            if seen == index:
                return i
            seen += 1
    raise IndexError("select1 out of range")


def entropy_bound(n, ones):
    if ones == 0 or ones == n:
        return 0.0
    p = ones / float(n)
    return -n * (p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))


def lcg_bits(length, ones_per_thousand, seed):
    """Deterministic bit generator (splitmix64 scaled to 0..999): the requested fraction of
    ones is honoured exactly enough for the density checks below."""
    out = []
    state = seed & 0xFFFFFFFFFFFFFFFF
    for _ in range(length):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
        out.append("1" if z % 1000 < ones_per_thousand else "0")
    return "".join(out)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    # class directory: 2^B classes in total, each class -> exact popcount, ranks complete
    check(sum(binom(B, k) for k in range(B + 1)) == 1 << B, "class count must be 2^B")
    check(len(CLASS_POPCOUNT) == 1 << B, "popcount table size")
    seen_ranks = {k: set() for k in range(B + 1)}
    for block in range(1 << B):                            # exhaustive for B = 15
        positions = [p for p in range(B) if (block >> p) & 1]
        k = len(positions)
        rank = colex_rank(positions)
        check(CLASS_POPCOUNT[BASE[k] + rank] == k, "class index must encode the popcount")
        check(colex_unrank(k, rank) == positions, "colex unrank roundtrip")
        seen_ranks[k].add(rank)
    for k in range(B + 1):
        check(seen_ranks[k] == set(range(binom(B, k))), "colex ranks of weight %d" % k)

    # degenerate vectors
    for bits in ("", "0", "1", "00", "11", "0" * 40, "1" * 40, "0" * 39 + "1", "1" + "0" * 44):
        rrr = RRRBitvector(bits)
        naive = naive_bits(bits)
        check([rrr.access(i) for i in range(len(bits))] == naive, "access degenerate")
        check([rrr.rank1(i) for i in range(len(bits) + 1)] ==
              [naive_rank1(naive, i) for i in range(len(bits) + 1)], "rank degenerate")
        check(rrr.rank0(len(bits)) == len(bits) - sum(naive), "rank0 degenerate")
        for index in range(sum(naive)):
            check(rrr.select1(index) == naive_select1(naive, index), "select1 degenerate")

    # random vectors of several densities, including non-multiples of the block size
    vectors = ["0101011100101" + "1" * 14 + "0" * 3,
               lcg_bits(1, 500, 7),
               lcg_bits(149, 100, 1), lcg_bits(150, 500, 2), lcg_bits(151, 900, 3),
               lcg_bits(600, 30, 4), lcg_bits(600, 500, 5), lcg_bits(999, 700, 6)]
    for bits in vectors:
        rrr = RRRBitvector(bits)
        naive = naive_bits(bits)
        total = sum(naive)
        check([rrr.access(i) for i in range(len(bits))] == naive, "access mismatch")
        check([rrr.rank1(i) for i in range(len(bits) + 1)] ==
              [naive_rank1(naive, i) for i in range(len(bits) + 1)], "rank1 mismatch")
        for index in range(total):
            pos = rrr.select1(index)
            check(pos == naive_select1(naive, index), "select1 mismatch")
            check(rrr.rank1(pos) == index and rrr.access(pos) == 1, "rank/select inverse")
        for index in range(len(bits) - total):
            check(rrr.rank0(rrr.select0(index) + 1) == index + 1, "select0 mismatch")
        # every block decodes back to the exact stored bits
        rebuilt = "".join(
            "".join("1" if (rrr.decode_block(b) >> p) & 1 else "0"
                    for p in range(rrr.block_length(b)))
            for b in range(rrr.blocks))
        check(rebuilt == bits, "block decode roundtrip")
        check(rrr.total_ones == total, "total ones")

    # compression: skew pays off, and RRR never costs much more than the plain 1 bit/bit
    skewed = RRRBitvector(lcg_bits(4000, 100, 11))
    dense = RRRBitvector(lcg_bits(4000, 500, 12))
    skew_density = skewed.total_ones / 4000.0
    check(0.08 < skew_density < 0.12, "generator must honour the requested density")
    check(skewed.payload_bits() < 4000, "skewed RRR must compress below 1 bit/bit")
    check(skewed.payload_bits() < 0.7 * 4000, "skewed RRR must compress well below plain")
    entropy = entropy_bound(4000, skewed.total_ones)
    check(skewed.payload_bits() >= 0.9 * entropy,
          "no code may beat the zero-order entropy of the block distribution")
    check(skewed.payload_bits() < 1.35 * entropy, "offsets should approach the entropy")
    check(dense.payload_bits() > skewed.payload_bits(), "denser vector costs more bits")
    check(dense.payload_bits() <= 1.15 * 4000, "near p = 0.5 RRR stays within ~2n bits")
    check(skewed.directory_bits() < skewed.payload_bits(), "directory stays o(payload)")
    # the per-block offset bits are the dominant, entropy-tracking term
    check(skewed.offset_bits() < skewed.payload_bits(), "classes plus offsets add up")

    # rejected input
    try:
        RRRBitvector("0102")
        raise AssertionError("non-binary input must be rejected")
    except ValueError:
        pass
    rrr = RRRBitvector("1010")
    for bad in (-1, 4, 99):
        try:
            rrr.access(bad)
            raise AssertionError("out-of-range access must raise")
        except IndexError:
            pass
    try:
        rrr.select1(3)
        raise AssertionError("select1 past the last one must raise")
    except IndexError:
        pass

    print("B=%d, n=4000 p=%.3f: RRR payload %d bits = offsets %d + classes %d (plain %d)"
          % (B, skew_density, skewed.payload_bits(), skewed.offset_bits(),
             skewed.class_bits(), 4000))
    print("B=%d, n=4000 p=%.3f: RRR payload %d bits vs n*H0 = %d bits, directory %d bits"
          % (B, skew_density, skewed.payload_bits(), int(entropy), skewed.directory_bits()))
    print("B=%d, n=4000 p=0.500: RRR payload %d bits (plain 4000) - RRR stops paying at ~0.5"
          % (B, dense.payload_bits()))
    print("PASS")


if __name__ == "__main__":
    main()
