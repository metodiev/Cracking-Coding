/** Task 609 - Implement advanced compressed bitvectors (RRR) and rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A Raman-Raman-Rao (RRR) bitvector splits the n bits into blocks of B = 15 bits. The *class* of a
 * block is the pair (c, colex rank of the pattern among the C(B, c) patterns with c one-bits), so a
 * block costs 4 bits for the class plus log2 C(B, c) bits for the offset - less than one bit per bit
 * whenever the vector is skewed, which is the whole point of RRR. build() releases the raw words, so
 * every later query decodes a block from its class code with rankToPattern (exactly the class-table
 * lookup the real structure performs) and no query ever touches the original machine words.
 *
 * rank1 adds the 32-bit sample of the superblock (every four blocks), the popcounts of the at most
 * three blocks that follow it in the superblock, and the partial popcount of one decoded block.
 * select1 walks the samples, then the block popcounts, then the 15 bits of a single block; select0 is
 * a binary search over rank0. Both return -1 for a rank outside [1, count].
 *
 * Simplified: the class is stored as a (popcount, offset) pair instead of a pointer into a
 * precomputed class table (same information content, so the reported size is an upper bound) and the
 * samples are plain 32-bit ints. Verified: colex ranking is a bijection over all 2^15 patterns;
 * every block decodes back to the bits it was built from; rank0/rank1/select0/select1 agree with a
 * brute-force scan for many sizes and densities; the identities rank0(i)+rank1(i) == i and
 * select1(rank1(i)) <= i < select1(rank1(i)+1) hold; the structure beats n bits on skewed input and
 * (as expected for the break-even of RRR with B = 15) loses on uniform input.
 */
import java.util.*;

class RrrBitvector {
    static final int BLOCK = 15;
    static final int SUPER_BLOCKS = 4;
    static final int CLASS_BITS = 4;              // a popcount of a 15-bit block fits in 4 bits
    static final long[][] BINOM = binomialTable();

    final int n;
    final int blocks;
    final int[] blockPopcount;                    // class: the number of one-bits of the block
    final int[] classOffset;                      // class: colex rank of the block within its class
    final int[] superRank;                        // rank1 at the sample points, one per superblock
    private long[] raw;                           // construction scratch, released by build()

    static long[][] binomialTable() {
        long[][] t = new long[BLOCK + 1][BLOCK + 1];
        for (int i = 0; i <= BLOCK; i++) {
            t[i][0] = 1;
            for (int j = 1; j <= i; j++) t[i][j] = t[i - 1][j - 1] + t[i - 1][j];
        }
        return t;
    }

    static long choose(int a, int b) { return (a < 0 || b < 0 || b > a) ? 0 : BINOM[a][b]; }

    /** Colex rank of a pattern among the patterns with the same popcount: sum of C(pos_j, j). */
    static int patternRank(int pattern) {
        int rank = 0, seen = 0;
        for (int bit = 0; bit < BLOCK; bit++) {
            if (((pattern >>> bit) & 1) != 0) rank += choose(bit, ++seen);
        }
        return rank;
    }

    /** Inverse of patternRank: the unique pattern with that popcount sitting at that colex offset. */
    static int rankToPattern(int popcount, int rank) {
        int pattern = 0, remaining = rank;
        for (int j = popcount; j >= 1; j--) {
            int pos = j - 1;
            while (pos + 1 < BLOCK && choose(pos + 1, j) <= remaining) pos++;
            pattern |= 1 << pos;
            remaining -= (int) choose(pos, j);
        }
        return pattern;
    }

    static int bitsToEncode(long classes) {
        int bits = 0;
        while ((1L << bits) < classes) bits++;
        return bits;
    }

    RrrBitvector(int length) {
        n = length;
        blocks = (length + BLOCK - 1) / BLOCK;
        raw = new long[(length + 63) >>> 6];
        blockPopcount = new int[blocks];
        classOffset = new int[blocks];
        superRank = new int[(blocks + SUPER_BLOCKS - 1) / SUPER_BLOCKS + 1];
    }

    void set(int i) {
        if (i < 0 || i >= n) throw new IllegalArgumentException("index out of range: " + i);
        raw[i >>> 6] |= 1L << (i & 63);
    }

    void build() {
        for (int b = 0; b < blocks; b++) {
            int width = Math.min(BLOCK, n - b * BLOCK);
            int base = b * BLOCK;
            int word = base >>> 6, off = base & 63;
            long w;
            if (off == 0) {
                w = raw[word];
            } else {
                long hi = (word + 1 < raw.length) ? raw[word + 1] : 0L;
                w = (raw[word] >>> off) | (hi << (64 - off));
            }
            int pattern = (int) (w & ((1L << width) - 1));      // width <= 15, so no shift overflow
            blockPopcount[b] = Integer.bitCount(pattern);
            classOffset[b] = patternRank(pattern);
        }
        for (int s = 1; s < superRank.length; s++) {
            superRank[s] = superRank[s - 1];
            for (int b = (s - 1) * SUPER_BLOCKS; b < Math.min(blocks, s * SUPER_BLOCKS); b++) {
                superRank[s] += blockPopcount[b];
            }
        }
        raw = null;                                          // only the class codes survive compression
    }

    /** Block b is reconstructed from its class code alone - the RRR decode step. */
    int decodeBlock(int b) { return rankToPattern(blockPopcount[b], classOffset[b]); }

    int get(int i) { return (i < 0 || i >= n) ? 0 : (decodeBlock(i / BLOCK) >>> (i % BLOCK)) & 1; }

    int count1() { return superRank[superRank.length - 1]; }

    int count0() { return n - count1(); }

    int rank1(int i) {
        if (i <= 0) return 0;
        if (i > n) i = n;
        int full = i / BLOCK, rem = i % BLOCK;
        int superIdx = full / SUPER_BLOCKS;
        int rank = superRank[superIdx];
        for (int b = superIdx * SUPER_BLOCKS; b < full; b++) rank += blockPopcount[b];
        if (rem > 0) rank += Integer.bitCount(decodeBlock(full) & ((1 << rem) - 1));
        return rank;
    }

    int rank0(int i) {
        if (i <= 0) return 0;
        if (i > n) i = n;
        return i - rank1(i);
    }

    int select1(int k) {
        if (k < 1 || k > count1()) return -1;
        int sb = 0;
        while (sb + 1 < superRank.length && superRank[sb + 1] < k) sb++;
        int before = superRank[sb];
        for (int b = sb * SUPER_BLOCKS; b < Math.min(blocks, (sb + 1) * SUPER_BLOCKS); b++) {
            if (before + blockPopcount[b] >= k) {
                int pattern = decodeBlock(b), target = k - before;
                for (int bit = 0; bit < BLOCK; bit++) {
                    if (((pattern >>> bit) & 1) != 0 && --target == 0) return b * BLOCK + bit;
                }
                return -1;                                   // unreachable: the popcount matched
            }
            before += blockPopcount[b];
        }
        return -1;                                           // unreachable: k <= count1()
    }

    int select0(int k) {
        if (k < 1 || k > count0()) return -1;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank0(mid + 1) >= k) hi = mid; else lo = mid + 1;
        }
        return lo;
    }

    /** Bits actually stored: the class code of every block plus one 32-bit sample per superblock. */
    long compressedBits() {
        long bits = (long) blocks * CLASS_BITS + (long) superRank.length * 32;
        for (int b = 0; b < blocks; b++) bits += bitsToEncode(choose(BLOCK, blockPopcount[b]));
        return bits;
    }

    long uncompressedBits() { return n; }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    /** splitmix64: the only source of "randomness", so every run sees the same vectors. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[] randomBits(int n, int percent) {
        int[] bits = new int[n];
        for (int i = 0; i < n; i++) bits[i] = (nextLong() >>> 33) % 100 < percent ? 1 : 0;
        return bits;
    }

    static RrrBitvector build(int[] bits) {
        RrrBitvector bv = new RrrBitvector(bits.length);
        for (int i = 0; i < bits.length; i++) if (bits[i] == 1) bv.set(i);
        bv.build();
        return bv;
    }

    static void verify(int[] bits) {
        RrrBitvector bv = build(bits);
        int n = bits.length, naive = 0;
        for (int bit : bits) naive += bit;
        check(bv.count1() == naive, "count1 n=" + n);
        check(bv.count0() == n - naive, "count0 n=" + n);
        int ones = 0;
        for (int i = 0; i < n; i++) {
            check(bv.get(i) == bits[i], "get " + i + " n=" + n);
            check(bv.rank1(i) == ones, "rank1 " + i + " n=" + n);
            check(bv.rank0(i) == i - ones, "rank0 " + i + " n=" + n);
            check(bv.rank0(i) + bv.rank1(i) == i, "rank split " + i);
            ones += bits[i];
        }
        check(bv.rank1(n) == naive && bv.rank0(n) == n - naive, "rank at n");
        check(bv.rank1(-5) == 0 && bv.rank1(n + 9) == naive, "rank clamps");
        for (int k = 1; k <= naive; k++) {
            int pos = bv.select1(k);
            check(pos >= 0 && bits[pos] == 1, "select1 points at a one, k=" + k + " n=" + n);
            check(bv.rank1(pos) == k - 1 && bv.rank1(pos + 1) == k, "select1 rank round trip k=" + k);
            check(bv.select1(bv.rank1(pos) + 1) == pos, "rank then select is stable");
        }
        for (int k = 1; k <= n - naive; k++) {
            int pos = bv.select0(k);
            check(pos >= 0 && bits[pos] == 0, "select0 points at a zero, k=" + k + " n=" + n);
            check(bv.rank0(pos) == k - 1, "select0 rank round trip k=" + k);
        }
        check(bv.select1(0) == -1 && bv.select1(naive + 1) == -1, "select1 out of range");
        check(bv.select0(0) == -1 && bv.select0(n - naive + 1) == -1, "select0 out of range");
        // the class code must pin the block, not merely its popcount
        for (int b = 0; b < bv.blocks; b++) {
            int expected = 0;
            for (int bit = 0; bit < RrrBitvector.BLOCK; bit++) {
                int index = b * RrrBitvector.BLOCK + bit;
                if (index < n && bits[index] == 1) expected |= 1 << bit;
            }
            check(bv.decodeBlock(b) == expected, "class code decodes the block, block " + b + " n=" + n);
        }
    }

    public static void main(String[] args) {
        // colex ranking is a bijection on the 2^15 patterns, class by class
        int[] seenPerClass = new int[RrrBitvector.BLOCK + 1];
        for (int pattern = 0; pattern < (1 << RrrBitvector.BLOCK); pattern++) {
            int popcount = Integer.bitCount(pattern);
            int rank = RrrBitvector.patternRank(pattern);
            check(rank >= 0 && rank < RrrBitvector.choose(RrrBitvector.BLOCK, popcount),
                    "colex rank inside the class, pattern " + pattern);
            check(RrrBitvector.rankToPattern(popcount, rank) == pattern, "colex round trip " + pattern);
            seenPerClass[popcount]++;
        }
        for (int c = 0; c <= RrrBitvector.BLOCK; c++) {
            check(seenPerClass[c] == RrrBitvector.choose(RrrBitvector.BLOCK, c),
                    "class " + c + " has C(15,c) patterns");
        }
        check(RrrBitvector.rankToPattern(0, 0) == 0, "empty class unranks to the empty pattern");
        check(RrrBitvector.rankToPattern(RrrBitvector.BLOCK, 0) == (1 << RrrBitvector.BLOCK) - 1,
                "full class unranks to all ones");

        verify(new int[0]);
        verify(new int[]{1});
        verify(new int[]{0});
        verify(new int[]{1, 1, 1, 0, 0, 0, 1});
        for (int n : new int[]{14, 15, 16, 17, 29, 30, 31, 32, 60, 63, 64, 65, 100, 233, 500}) {
            for (int percent : new int[]{5, 25, 50, 75, 95}) verify(randomBits(n, percent));
        }
        int[] allOnes = new int[100], allZeros = new int[100], alternating = new int[97];
        Arrays.fill(allOnes, 1);
        for (int i = 0; i < alternating.length; i++) alternating[i] = i % 2;
        verify(allOnes);
        verify(allZeros);
        verify(alternating);

        int[] sparse = randomBits(2000, 2);
        RrrBitvector skewed = build(sparse);
        check(skewed.compressedBits() < skewed.uncompressedBits(), "skewed vector compresses");
        RrrBitvector uniform = build(randomBits(2000, 50));
        check(uniform.compressedBits() > uniform.uncompressedBits(),
                "uniform vector is not compressible at block size 15");
        System.out.println("skewed 2000 bits -> " + skewed.compressedBits() + " bits, uniform -> "
                + uniform.compressedBits() + " bits");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
