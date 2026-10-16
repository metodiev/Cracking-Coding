/** Task 578 - Implement dynamic orthogonal range counting with wavelet tree (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A working range-counting structure, simplified relative to a fully succinct dynamic wavelet tree but
 * correct and honest about the simplification. Three layers are implemented: (a) a classic static
 * wavelet tree over a value sequence, built on bit vectors with prefix rank, supporting "how many
 * values in [low, high] lie in positions [from, to)" in O(log sigma); (b) a block decomposition that
 * splits the sequence into blocks of about sqrt(n) and keeps one wavelet tree per block, so a point
 * update rebuilds exactly one wavelet tree of size O(sqrt(n) log sigma) and a range query touches
 * O(sqrt(n)) wavelet trees; and (c) a brute-force array used as the oracle, rebuilt on every update.
 * A real dynamic wavelet tree stores each level as a bit vector with a Fenwick-like dynamic rank
 * structure, giving O(log sigma log n) per operation; that machinery is *not* implemented here, and
 * the header says so — the block-rebuild version is the documented simplification.
 *
 * Documented checks. (1) The static wavelet tree is verified against brute force for every (from, to)
 * pair and every rank threshold on a 64-element sequence, plus range counting of values in [low, high]
 * on 200 random queries over a 256-element sequence; the recursive node count is asserted to be below
 * 2 n log2(sigma) and measured at 31 nodes for 64 elements over an alphabet of 16 (the bound is 512). (2) The dynamic structure is verified against the brute-force oracle after
 * *every* one of 400 point updates (writing a new value at a random position) and on 400 random range
 * queries, with the count, an exact-match comparison of 20 sampled queries against a full scan, and
 * assertions on the structure's own accounting: exactly one block rebuild per update and at most
 * 2 ceil(sqrt(n)) + 2 wavelet trees touched per query (the 256-element sequence is split into 16
 * blocks of size 16, and the 400 updates produce exactly 400 block rebuilds). (3) Edge cases: an empty range counts 0, the
 * full range counts the length, a single-element range counts 0 or 1, and values at the extremes of
 * the alphabet are counted correctly (the wavelet tree's bit vectors must handle the first and last
 * value). (4) Updates never change the multiset: after every update the sorted content equals the
 * oracle's sorted content (asserted), so a lost or duplicated element would be caught immediately.
 * (5) The blocked structure answers the same queries as a freshly rebuilt static wavelet tree over the
 * same sequence on 100 random queries, which is an independent cross-check of the block boundaries
 * (in particular the partial first and last blocks).
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * values and positions), so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static long nodes;

    /** Static wavelet tree over a value sequence with a bounded alphabet. */
    static final class WaveletTree {
        final int[] data;
        final int lowValue;
        final int highValue;
        final int[] prefix; // prefix[i] = number of zeros in data[0..i)
        final WaveletTree left;
        final WaveletTree right;

        WaveletTree(int[] values, int from, int to, int lowValue, int highValue) {
            this.lowValue = lowValue;
            this.highValue = highValue;
            data = new int[to - from];
            System.arraycopy(values, from, data, 0, to - from);
            prefix = new int[data.length + 1];
            nodes++;
            if (lowValue == highValue) {
                left = null;
                right = null;
                for (int i = 0; i < data.length; i++) prefix[i + 1] = prefix[i] + 1;
                return;
            }
            int middle = (lowValue + highValue) >>> 1;
            int[] zeros = new int[data.length];
            int[] ones = new int[data.length];
            int zerosCount = 0;
            int onesCount = 0;
            for (int i = 0; i < data.length; i++) {
                if (data[i] <= middle) {
                    prefix[i + 1] = prefix[i] + 1;
                    zeros[zerosCount++] = data[i];
                } else {
                    prefix[i + 1] = prefix[i];
                    ones[onesCount++] = data[i];
                }
            }
            left = new WaveletTree(zeros, 0, zerosCount, lowValue, middle);
            right = new WaveletTree(ones, 0, onesCount, middle + 1, highValue);
        }

        /** Number of values equal to {@code rankValue} in positions [from, to). */
        int rank(int rankValue, int from, int to) {
            return rankLess(rankValue, from, to) - rankLess(rankValue - 1, from, to);
        }

        /** Number of values <= threshold in positions [from, to). */
        int rankLess(int threshold, int from, int to) {
            if (from >= to) return 0;
            if (threshold < lowValue) return 0;
            if (threshold >= highValue) return to - from;
            if (lowValue == highValue) return to - from;
            int zerosBefore = prefix[from];
            int zerosInner = prefix[to] - prefix[from];
            int middle = (lowValue + highValue) >>> 1;
            if (threshold <= middle) {
                return left.rankLess(threshold, zerosBefore, zerosBefore + zerosInner);
            }
            int onesBefore = from - zerosBefore;
            int onesInner = (to - from) - zerosInner;
            return zerosInner + right.rankLess(threshold, onesBefore, onesBefore + onesInner);
        }

        /** Number of values in [low, high] inside positions [from, to). */
        int countInRange(int low, int high, int from, int to) {
            return rankLess(high, from, to) - rankLess(low - 1, from, to);
        }

        int length() { return data.length; }
    }

    /** Block decomposition: one wavelet tree per block, rebuilt on update. */
    static final class BlockedCounter {
        final int[] values;
        final int blockSize;
        final WaveletTree[] blocks;
        final int alphabetLow;
        final int alphabetHigh;
        long rebuilds;

        BlockedCounter(int[] values, int alphabetLow, int alphabetHigh) {
            this.values = values.clone();
            this.alphabetLow = alphabetLow;
            this.alphabetHigh = alphabetHigh;
            this.blockSize = Math.max(1, (int) Math.ceil(Math.sqrt(this.values.length)));
            int blocks = (this.values.length + blockSize - 1) / blockSize;
            this.blocks = new WaveletTree[blocks];
            for (int b = 0; b < blocks; b++) rebuild(b);
        }

        private void rebuild(int block) {
            int from = block * blockSize;
            int to = Math.min(values.length, from + blockSize);
            blocks[block] = new WaveletTree(values, from, to, alphabetLow, alphabetHigh);
            rebuilds++;
        }

        void update(int position, int value) {
            values[position] = value;
            rebuild(position / blockSize);
        }

        /** Count values in [low, high] among positions [from, to); returns the blocks touched. */
        int countInRange(int low, int high, int from, int to, int[] touched) {
            if (from >= to) {
                touched[0] = 0;
                return 0;
            }
            int firstBlock = from / blockSize;
            int lastBlock = (to - 1) / blockSize;
            int total = 0;
            int count = 0;
            for (int block = firstBlock; block <= lastBlock; block++) {
                int blockFrom = block * blockSize;
                int localFrom = Math.max(from, blockFrom) - blockFrom;
                int localTo = Math.min(to, blockFrom + blocks[block].length()) - blockFrom;
                total += blocks[block].countInRange(low, high, localFrom, localTo);
                count++;
            }
            touched[0] = count;
            return total;
        }
    }

    static int bruteCount(int[] values, int low, int high, int from, int to) {
        int total = 0;
        for (int i = from; i < to; i++) if (values[i] >= low && values[i] <= high) total++;
        return total;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(578_2025L);

        // ---- (1) the static wavelet tree ---------------------------------------------------------
        int[] small = new int[64];
        for (int i = 0; i < small.length; i++) small[i] = rng.nextInt(16);
        nodes = 0;
        WaveletTree smallTree = new WaveletTree(small, 0, small.length, 0, 15);
        for (int from = 0; from <= small.length; from++) {
            for (int to = from; to <= small.length; to++) {
                for (int threshold = 0; threshold < 16; threshold++) {
                    check(smallTree.rankLess(threshold, from, to) == bruteCount(small, 0, threshold, from, to),
                            "rankLess matches brute force at [" + from + "," + to + ") threshold " + threshold);
                    check(smallTree.countInRange(threshold, threshold, from, to)
                                    == bruteCount(small, threshold, threshold, from, to),
                            "single-value counting matches brute force");
                }
            }
        }
        check(smallTree.rank(15, 0, small.length) == bruteCount(small, 15, 15, 0, small.length),
                "the largest alphabet value is counted correctly");
        check(smallTree.rank(0, 0, small.length) == bruteCount(small, 0, 0, 0, small.length),
                "the smallest alphabet value is counted correctly");
        System.out.println("static wavelet tree: 64 elements, " + nodes + " nodes (bound 2 n log2 sigma = 512)");
        check(nodes <= 2L * small.length * 4, "the node count respects 2 n log2(sigma)");

        int[] medium = new int[256];
        for (int i = 0; i < medium.length; i++) medium[i] = rng.nextInt(64);
        WaveletTree mediumTree = new WaveletTree(medium, 0, medium.length, 0, 63);
        for (int trial = 0; trial < 200; trial++) {
            int from = rng.nextInt(medium.length + 1);
            int to = from + rng.nextInt(medium.length - from + 1);
            int low = rng.nextInt(64);
            int high = low + rng.nextInt(64 - low);
            check(mediumTree.countInRange(low, high, from, to) == bruteCount(medium, low, high, from, to),
                    "medium wavelet tree range count matches brute force");
        }

        // ---- (2) updates and queries on the dynamic structure -------------------------------------
        int alphabet = 64;
        int[] values = new int[256];
        for (int i = 0; i < values.length; i++) values[i] = rng.nextInt(alphabet);
        int[] oracle = values.clone();
        BlockedCounter counter = new BlockedCounter(values, 0, alphabet - 1);
        int updates = 400;
        int queries = 400;
        int[] touched = new int[1];
        long rebuildBaseline = counter.rebuilds;
        for (int step = 0; step < updates + queries; step++) {
            if (step % 2 == 0) {
                int position = rng.nextInt(values.length);
                int value = rng.nextInt(alphabet);
                long before = counter.rebuilds;
                counter.update(position, value);
                oracle[position] = value;
                check(counter.rebuilds == before + 1, "each update rebuilds exactly one block");
                int[] a = counter.values.clone();
                int[] b = oracle.clone();
                Arrays.sort(a);
                Arrays.sort(b);
                check(Arrays.equals(a, b), "an update never changes the multiset");
            } else {
                int from = rng.nextInt(values.length + 1);
                int to = from + rng.nextInt(values.length - from + 1);
                int low = rng.nextInt(alphabet);
                int high = low + rng.nextInt(alphabet - low);
                int found = counter.countInRange(low, high, from, to, touched);
                int expected = bruteCount(oracle, low, high, from, to);
                check(found == expected, "dynamic range count matches brute force after the updates");
                check(touched[0] <= 2 * (int) Math.ceil(Math.sqrt(values.length)) + 2,
                        "a query touches at most 2 sqrt(n) + 2 blocks");
            }
        }
        // a full scan comparison on sampled queries
        for (int trial = 0; trial < 20; trial++) {
            int from = rng.nextInt(values.length + 1);
            int to = from + rng.nextInt(values.length - from + 1);
            int low = rng.nextInt(alphabet);
            int high = low + rng.nextInt(alphabet - low);
            List<Integer> expected = new ArrayList<>();
            for (int i = from; i < to; i++) if (oracle[i] >= low && oracle[i] <= high) expected.add(oracle[i]);
            int found = counter.countInRange(low, high, from, to, new int[1]);
            check(found == expected.size(), "sampled query equals the full scan");
        }
        System.out.println("dynamic structure: " + (counter.rebuilds - rebuildBaseline) + " block rebuilds for "
                + updates + " updates over " + values.length + " elements in "
                + counter.blocks.length + " blocks of size " + counter.blockSize);

        // ---- (3) edge cases ----------------------------------------------------------------------
        check(counter.countInRange(0, alphabet - 1, 10, 10, new int[1]) == 0, "an empty range counts 0");
        check(counter.countInRange(0, alphabet - 1, 0, values.length, new int[1]) == values.length,
                "the full range counts the length");
        for (int position : new int[]{0, values.length - 1}) {
            int value = oracle[position];
            int single = counter.countInRange(value, value, position, position + 1, new int[1]);
            check(single == 1, "a single-element range counts 1");
            check(counter.countInRange(0, alphabet - 1, position, position + 1, new int[1]) == 1,
                    "a single element is inside the full alphabet");
        }
        check(counter.countInRange(0, 0, 0, values.length, new int[1])
                == bruteCount(oracle, 0, 0, 0, values.length), "extreme low value");
        check(counter.countInRange(alphabet - 1, alphabet - 1, 0, values.length, new int[1])
                == bruteCount(oracle, alphabet - 1, alphabet - 1, 0, values.length), "extreme high value");

        // ---- (5) blocked structure versus a freshly built static tree -----------------------------
        WaveletTree refreshed = new WaveletTree(oracle, 0, oracle.length, 0, alphabet - 1);
        for (int trial = 0; trial < 100; trial++) {
            int from = rng.nextInt(oracle.length + 1);
            int to = from + rng.nextInt(oracle.length - from + 1);
            int low = rng.nextInt(alphabet);
            int high = low + rng.nextInt(alphabet - low);
            check(counter.countInRange(low, high, from, to, new int[1])
                            == refreshed.countInRange(low, high, from, to),
                    "the blocked structure agrees with a freshly built wavelet tree, including partial blocks");
        }

        System.out.println("PASS");
    }
}
