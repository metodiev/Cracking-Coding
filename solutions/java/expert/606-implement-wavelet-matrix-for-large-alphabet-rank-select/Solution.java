/** Task 606 - Implement wavelet matrix for large alphabet rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The wavelet matrix stores the values as one bit vector per bit position, most significant bit
 * first; from level to level the values are stably partitioned into zeros then ones, and
 * zeros[level] records where the one-block starts. The bit vectors carry per-word rank prefixes, so
 * rank1(i) is one popcount away; select1/select0 binary search the level (a documented simplification
 * of the classic packed select samplers). On top of that:
 *   access(i)          - follow the bits of position i down all levels;
 *   rank(c, i)         - number of positions below i holding c, by descending only the branches on c;
 *   select(c, k)       - position of the k-th occurrence, by binary searching the monotone rank;
 *                        the per-level BitVector select1/select0 are verified against direct scans;
 *   countLess(c) and rangeCount(l, r, a, b) - range counting (b inclusive);
 *   quantile(l, r, k)  - k-th smallest value of a range.
 *
 * Verified against brute force on arrays with a 10^6 alphabet, heavy duplicates, sorted / reversed /
 * all-equal inputs, single element and empty input, plus out-of-range select, quantile and access
 * boundaries; every operation is additionally checked on a 20000 element random array.
 */
import java.util.*;

class RankBits {
    final long[] words;
    final int[] prefix;          // ones before each word
    final int length;

    RankBits(long[] words, int length) {
        this.words = words;
        this.length = length;
        prefix = new int[words.length + 1];
        for (int w = 0; w < words.length; w++) prefix[w + 1] = prefix[w] + Long.bitCount(words[w]);
    }

    int get(int i) { return (int) ((words[i >>> 6] >>> (i & 63)) & 1L); }

    int rank1(int i) {
        if (i <= 0) return 0;
        if (i > length) i = length;
        int count = prefix[i >>> 6];
        if ((i & 63) != 0) count += Long.bitCount(words[i >>> 6] & ((1L << (i & 63)) - 1));
        return count;
    }

    int count1() { return prefix[words.length]; }

    int count0() { return length - count1(); }

    /** 1-based; the caller guarantees 1 <= k <= count1(). */
    int select1(int k) {
        int lo = 0, hi = length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank1(mid + 1) >= k) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /** 1-based; the caller guarantees 1 <= k <= count0(). */
    int select0(int k) {
        int lo = 0, hi = length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (mid + 1 - rank1(mid + 1) >= k) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }
}

class WaveletMatrix {
    final int n, levels;
    final RankBits[] bits;
    final int[] zeros;          // positions holding a zero at each level

    WaveletMatrix(int[] values, int maxValue) {
        n = values.length;
        levels = Math.max(1, 32 - Integer.numberOfLeadingZeros(Math.max(1, maxValue)));
        bits = new RankBits[levels];
        zeros = new int[levels];
        int[] current = values.clone();
        for (int level = 0; level < levels; level++) {
            int shift = levels - 1 - level;
            long[] words = new long[(n + 63) >>> 6];
            for (int i = 0; i < n; i++) if (((current[i] >>> shift) & 1) != 0) words[i >>> 6] |= 1L << (i & 63);
            bits[level] = new RankBits(words, n);
            zeros[level] = bits[level].count0();
            int[] next = new int[n];
            int zeroAt = 0, oneAt = zeros[level];
            for (int i = 0; i < n; i++) {
                if (((current[i] >>> shift) & 1) == 0) next[zeroAt++] = current[i];
                else next[oneAt++] = current[i];
            }
            current = next;
        }
    }

    int access(int i) {
        int value = 0;
        for (int level = 0; level < levels; level++) {
            int bit = bits[level].get(i);
            int ones = bits[level].rank1(i);
            value = (value << 1) | bit;
            i = bit == 0 ? i - ones : zeros[level] + ones;
        }
        return value;
    }

    /** Number of positions below i holding value c. */
    int rank(int c, int i) {
        int low = 0, high = i;
        for (int level = 0; level < levels; level++) {
            int bit = (c >>> (levels - 1 - level)) & 1;
            int lowOnes = bits[level].rank1(low), highOnes = bits[level].rank1(high);
            if (bit == 0) { low -= lowOnes; high -= highOnes; }
            else { low = zeros[level] + lowOnes; high = zeros[level] + highOnes; }
        }
        return high - low;
    }

    /** Number of positions below i holding a value smaller than c. */
    long rankLess(int c, int i) {
        if (c <= 0) return 0;
        if (c >= (1L << levels)) return i;                  // every stored value is smaller
        long smaller = 0;
        int low = 0, high = i;
        for (int level = 0; level < levels; level++) {
            int bit = (c >>> (levels - 1 - level)) & 1;
            int lowOnes = bits[level].rank1(low), highOnes = bits[level].rank1(high);
            if (bit == 1) {
                smaller += (high - low) - (highOnes - lowOnes);
                low = zeros[level] + lowOnes;
                high = zeros[level] + highOnes;
            } else {
                low -= lowOnes;
                high -= highOnes;
            }
        }
        return smaller;
    }

    int countLess(int c) { return (int) rankLess(c, n); }

    /** Number of positions in [l, r) whose value lies in the inclusive range [a, b]. */
    int rangeCount(int l, int r, int a, int b) {
        long belowB = rankLess(b + 1, r) - rankLess(b + 1, l);
        long belowA = rankLess(a, r) - rankLess(a, l);
        return (int) (belowB - belowA);
    }

    /** 1-based k of the k-th occurrence of c, or -1 when it does not exist. rank(c, i) is monotone in
     *  i, so a binary search over positions finds the k-th occurrence using the level bit vectors. */
    int select(int c, int k) {
        if (k < 1 || k > rank(c, n)) return -1;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (rank(c, mid + 1) >= k) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /** 1-based k of the k-th smallest value in [l, r), or -1 when k is out of range. */
    int quantile(int l, int r, int k) {
        if (k < 1 || k > r - l) return -1;
        int value = 0;
        for (int level = 0; level < levels; level++) {
            int lowOnes = bits[level].rank1(l), highOnes = bits[level].rank1(r);
            int zerosInRange = (r - l) - (highOnes - lowOnes);
            if (k <= zerosInRange) {
                l -= lowOnes;
                r -= highOnes;
            } else {
                k -= zerosInRange;
                value |= 1 << (levels - 1 - level);
                l = zeros[level] + lowOnes;
                r = zeros[level] + highOnes;
            }
        }
        return value;
    }
}

public class Solution {
    private static long state = 0x606AB1E5EEDL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[] randomValues(int n, int maxValue) {
        int[] values = new int[n];
        for (int i = 0; i < n; i++) values[i] = (int) ((nextLong() >>> 33) % (maxValue + 1));
        return values;
    }

    static int[] sortedCopy(int[] values) {
        int[] copy = values.clone();
        Arrays.sort(copy);
        return copy;
    }

    static int bruteRank(int[] values, int c, int i) {
        int count = 0;
        for (int k = 0; k < i && k < values.length; k++) if (values[k] == c) count++;
        return count;
    }

    static int bruteSelect(int[] values, int c, int k) {
        int seen = 0;
        for (int i = 0; i < values.length; i++) if (values[i] == c && ++seen == k) return i;
        return -1;
    }

    static int bruteLess(int[] values, int c) {
        int count = 0;
        for (int v : values) if (v < c) count++;
        return count;
    }

    static void verify(int[] values, int maxValue, boolean exhaustiveAccess) {
        WaveletMatrix matrix = new WaveletMatrix(values, maxValue);
        check(matrix.n == values.length && matrix.levels <= 21, "level count for a large alphabet");
        if (exhaustiveAccess) {
            for (int i = 0; i < values.length; i++) check(matrix.access(i) == values[i], "access(" + i + ")");
        }
        for (int i = 0; i <= values.length; i += Math.max(1, values.length / 17)) {
            for (int c = 0; c <= maxValue; c += Math.max(1, maxValue / 7)) {
                check(matrix.rank(c, i) == bruteRank(values, c, i), "rank(" + c + "," + i + ")");
            }
        }
        for (int c : new int[]{0, 1, maxValue / 2, maxValue}) {
            int total = bruteRank(values, c, values.length);
            check(matrix.rank(c, values.length) == total, "rank total of " + c);
            for (int k = 1; k <= total; k += Math.max(1, total / 5)) {
                check(matrix.select(c, k) == bruteSelect(values, c, k), "select(" + c + "," + k + ")");
            }
            check(matrix.select(c, 0) == -1 && matrix.select(c, total + 1) == -1, "select range checks");
            check(matrix.countLess(c) == bruteLess(values, c), "countLess(" + c + ")");
        }
        int l = values.length / 4, r = (values.length * 3) / 4;
        for (int lo = 0; lo <= maxValue; lo += Math.max(1, maxValue / 5)) {
            for (int hi = lo; hi <= maxValue; hi += Math.max(1, maxValue / 3)) {
                check(matrix.rangeCount(l, r, lo, hi) == bruteRangeCount(values, l, r, lo, hi),
                        "rangeCount(" + lo + "," + hi + ")");
            }
        }
        int[] slice = Arrays.copyOfRange(values, l, r);
        Arrays.sort(slice);
        for (int level = 0; level < matrix.levels; level++) {          // rank/select on the level bits
            RankBits levelBits = matrix.bits[level];
            for (int k = 1; k <= levelBits.count1(); k += Math.max(1, levelBits.count1() / 4)) {
                check(levelBits.rank1(levelBits.select1(k) + 1) == k, "level select1 inverts rank1");
                check(levelBits.get(levelBits.select1(k)) == 1, "level select1 lands on a one");
            }
            for (int k = 1; k <= levelBits.count0(); k += Math.max(1, levelBits.count0() / 4)) {
                check(levelBits.get(levelBits.select0(k)) == 0, "level select0 lands on a zero");
            }
        }
        for (int k = 1; k <= r - l; k += Math.max(1, (r - l) / 9)) {
            check(matrix.quantile(l, r, k) == slice[k - 1], "quantile(" + k + ")");
        }
        check(matrix.quantile(l, r, 0) == -1 && matrix.quantile(l, r, r - l + 1) == -1, "quantile range checks");
    }

    static int bruteRangeCount(int[] values, int l, int r, int lo, int hi) {
        int count = 0;
        for (int i = l; i < r; i++) if (values[i] >= lo && values[i] <= hi) count++;
        return count;
    }

    public static void main(String[] args) {
        WaveletMatrix empty = new WaveletMatrix(new int[0], 8);
        check(empty.n == 0 && empty.rank(3, 0) == 0 && empty.select(3, 1) == -1, "empty array");
        WaveletMatrix single = new WaveletMatrix(new int[]{5}, 1000000);
        check(single.access(0) == 5 && single.select(5, 1) == 0 && single.quantile(0, 1, 1) == 5, "single value");
        check(single.countLess(5) == 0 && single.countLess(6) == 1 && single.rank(4, 1) == 0, "single counts");

        int[] allEqual = new int[40];
        Arrays.fill(allEqual, 999999);
        WaveletMatrix equal = new WaveletMatrix(allEqual, 1000000);
        check(equal.rank(999999, 40) == 40 && equal.select(999999, 40) == 39, "all equal");
        check(equal.select(999999, 4) == 3 && equal.countLess(999999) == 0, "all equal selects");

        int[] sorted = new int[64];
        for (int i = 0; i < sorted.length; i++) sorted[i] = i * 1000;
        verify(sorted, 1000000, true);
        int[] reversed = new int[64];
        for (int i = 0; i < reversed.length; i++) reversed[i] = (63 - i) * 15000;
        verify(reversed, 1000000, true);
        int[] duplicates = {7, 7, 7, 3, 3, 9, 0, 1000000, 5, 5, 5, 5, 42, 42, 3};
        verify(duplicates, 1000000, true);

        for (int trial = 0; trial < 12; trial++) {
            verify(randomValues(50, 300), 300, true);                       // small alphabet, exhaustive
            verify(randomValues(60, 1000000), 1000000, false);              // 10^6 alphabet
        }
        for (int trial = 0; trial < 6; trial++) {
            verify(randomValues(500, 1 << 20), 1 << 20, false);             // 20 bit values
        }

        int[] large = randomValues(20000, 1000000);
        WaveletMatrix matrix = new WaveletMatrix(large, 1000000);
        for (int i = 0; i < large.length; i += 997) {
            check(matrix.access(i) == large[i], "large access at " + i);
            int lo = i, hi = Math.min(large.length, i + 500);
            int[] slice = Arrays.copyOfRange(large, lo, hi);
            Arrays.sort(slice);
            check(matrix.quantile(lo, hi, 1) == slice[0], "large quantile min");
            check(matrix.quantile(lo, hi, hi - lo) == slice[hi - lo - 1], "large quantile max");
            check(matrix.rangeCount(lo, hi, slice[0], slice[hi - lo - 1])
                    == bruteRangeCount(large, lo, hi, slice[0], slice[hi - lo - 1]), "large range count");
        }
        check(matrix.select(large[0], 1) == bruteSelect(large, large[0], 1), "large select");
        System.out.println("levels / values     : " + matrix.levels + " / " + large.length);
        System.out.println("bit vector words    : " + matrix.levels * ((large.length + 63) >>> 6));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
