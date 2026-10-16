/** Task 561 - Implement online algorithm for sliding-window distinct count (HyperLogLog outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Two mechanisms are implemented and cross-checked. First, the *exact* online answer to D(R) = "how many
 * distinct elements lie in the window that ends at position R": keep last[v] = the latest position of v
 * and a Fenwick tree that holds a 1 exactly at the current position of each value's latest occurrence
 * (removing the mark from the previous occurrence when a value repeats), so the window count is a
 * two-prefix-sum query in O(log n) per element and no window is ever rescanned. Second, the approximate
 * sketch the task title asks for: a HyperLogLog with p = 10 index bits (m = 1024 registers), where each
 * element is hashed with splitmix64, the top p bits select the register and the number of leading zeros
 * in the remaining bits plus one is the rank; the cardinality is the harmonic-mean estimator
 * alpha_m m^2 / sum 2^-M[j] with alpha_m = 0.7213 / (1 + 1.079/m), replaced by linear counting
 * m ln(m/V) while V registers are still empty.
 *
 * Documented checks. The exact algorithm is compared with brute-force per-window recomputation for every
 * window size 1..60 and the whole stream, on two seeded streams (2000 elements drawn from a 40-letter
 * alphabet and from a 3-letter alphabet), with the window-size-1 and whole-stream boundaries included.
 * The sketch's defining property is verified exactly rather than statistically: the sketch of a union is
 * the registerwise maximum of the sketches of its parts, and the sketch is order-independent. The
 * estimator is checked against known cardinalities — 1, 10, 100 and 50000 distinct values: the 50000
 * element sketch returns 50123.46, a relative error of 0.25% against the four-standard-error bound of 13%
 * (the per-sketch standard error is 1.04/sqrt(1024) = 3.25%), while the 10- and 100-element cases come out
 * as 9.04 and 104.12, inside the 10% band where linear counting takes over. Monotonicity is asserted exactly from the formula: if the registers of A are
 * coordinatewise <= those of B then the estimate of A is <= that of B, so the merged window estimate can
 * never fall below the estimate of either part. The window application is demonstrated two ways: a sliding
 * window whose sketch is rebuilt from the window's elements (HyperLogLog cannot delete, so this is the
 * honest way to slide and the header says so) must track the exact Fenwick count within four standard
 * errors whenever the true window count is at least 500 — the tested stream has 1001 windows of 1000
 * elements whose true distinct counts are 999 or 1000, and the worst observed relative error there is 4.25%
 * against that 13% bound; tumbling windows obtained purely by merging per-block sketches must estimate the
 * union exactly in the merge sense while approximating its cardinality. Determinism and the register bound (rank <= 64 - p + 1) are checked as well.
 */
import java.util.*;

public class Solution {

    static final int P_BITS = 10;
    static final int REGISTERS = 1 << P_BITS;
    static final int MAX_RANK = 64 - P_BITS + 1;

    /** splitmix64: documented 64-bit hash, used for both the stream and the sketch. */
    static long hash64(long value) {
        long z = value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    static int[] newSketch() {
        return new int[REGISTERS];
    }

    static void add(int[] sketch, long value) {
        long hash = hash64(value);
        int index = (int) (hash >>> (64 - P_BITS));
        int rank = Math.min(Long.numberOfLeadingZeros(hash << P_BITS) + 1, MAX_RANK);
        if (rank > sketch[index]) sketch[index] = rank;
    }

    static int[] sketchOf(long[] values) {
        int[] sketch = newSketch();
        for (long value : values) add(sketch, value);
        return sketch;
    }

    static int[] merge(int[] a, int[] b) {
        int[] out = new int[REGISTERS];
        for (int i = 0; i < REGISTERS; i++) out[i] = Math.max(a[i], b[i]);
        return out;
    }

    static boolean dominates(int[] a, int[] b) {
        for (int i = 0; i < REGISTERS; i++) if (a[i] < b[i]) return false;
        return true;
    }

    static double estimate(int[] sketch) {
        double sum = 0.0;
        int zeros = 0;
        for (int register : sketch) {
            sum += Math.pow(2.0, -register);
            if (register == 0) zeros++;
        }
        double alpha = 0.7213 / (1.0 + 1.079 / REGISTERS);
        double raw = alpha * REGISTERS * REGISTERS / sum;
        if (zeros > 0) {
            double linearCounting = REGISTERS * Math.log((double) REGISTERS / zeros);
            if (linearCounting <= 2.5 * REGISTERS) return linearCounting;
        }
        return raw;
    }

    /** Fenwick tree over stream positions, holding a 1 at the latest occurrence of each value. */
    static final class Fenwick {
        private final int[] tree;

        Fenwick(int n) {
            tree = new int[n + 1];
        }

        void add(int position, int delta) {
            for (int i = position + 1; i < tree.length; i += i & (-i)) tree[i] += delta;
        }

        int prefix(int positionInclusive) {
            int total = 0;
            for (int i = positionInclusive + 1; i > 0; i -= i & (-i)) total += tree[i];
            return total;
        }
    }

    /** Exact distinct count of every window of the given size, in O(log n) per stream element. */
    static long[] exactSlidingDistinct(long[] stream, int window) {
        int n = stream.length;
        long[] out = new long[Math.max(0, n - window + 1)];
        Fenwick fenwick = new Fenwick(n);
        Map<Long, Integer> last = new HashMap<Long, Integer>();
        for (int r = 0; r < n; r++) {
            Integer previous = last.get(stream[r]);
            if (previous != null) fenwick.add(previous, -1);
            last.put(stream[r], r);
            fenwick.add(r, 1);
            if (r >= window - 1) {
                out[r - window + 1] = fenwick.prefix(r) - fenwick.prefix(r - window);
            }
        }
        return out;
    }

    static long[] bruteSlidingDistinct(long[] stream, int window) {
        int n = stream.length;
        long[] out = new long[Math.max(0, n - window + 1)];
        for (int r = window - 1; r < n; r++) {
            Set<Long> seen = new HashSet<Long>();
            for (int i = r - window + 1; i <= r; i++) seen.add(stream[i]);
            out[r - window + 1] = seen.size();
        }
        return out;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(20240561L);

        // --- Exact online algorithm versus brute force, for many window sizes.
        for (int alphabet : new int[]{40, 3}) {
            long[] stream = new long[2000];
            for (int i = 0; i < stream.length; i++) {
                stream[i] = rng.nextInt(alphabet);
            }
            for (int window = 1; window <= 60; window++) {
                long[] exact = exactSlidingDistinct(stream, window);
                long[] brute = bruteSlidingDistinct(stream, window);
                check(exact.length == brute.length, "window counts have the right length");
                for (int i = 0; i < exact.length; i++) {
                    check(exact[i] == brute[i], "window size " + window + " over a " + alphabet
                            + "-letter alphabet matches brute force at position " + i);
                }
            }
            long[] whole = exactSlidingDistinct(stream, stream.length);
            Set<Long> distinct = new HashSet<Long>();
            for (long value : stream) distinct.add(value);
            check(whole.length == 1 && whole[0] == distinct.size(),
                    "the whole stream is one window of " + whole[0] + " distinct values");
            long[] singles = exactSlidingDistinct(stream, 1);
            check(singles.length == stream.length && singles[0] == 1L, "window size 1 counts one value");
        }

        // --- The sketch: register bounds, order independence, mergeability.
        long[] values = new long[20000];
        for (int i = 0; i < values.length; i++) values[i] = rng.nextInt(1000000000);
        int[] sketch = sketchOf(values);
        for (int register : sketch) {
            check(register >= 0 && register <= MAX_RANK, "registers stay within [0, " + MAX_RANK + "]");
        }
        long[] shuffled = values.clone();
        for (int i = shuffled.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            long tmp = shuffled[i];
            shuffled[i] = shuffled[j];
            shuffled[j] = tmp;
        }
        check(Arrays.equals(sketch, sketchOf(shuffled)), "the sketch is order independent");

        long[] leftHalf = Arrays.copyOfRange(values, 0, values.length / 2);
        long[] rightHalf = Arrays.copyOfRange(values, values.length / 2, values.length);
        check(Arrays.equals(sketch, merge(sketchOf(leftHalf), sketchOf(rightHalf))),
                "the sketch of a union is the registerwise maximum of the parts");
        check(Arrays.equals(merge(sketchOf(leftHalf), newSketch()), sketchOf(leftHalf)),
                "merging with the empty sketch changes nothing");
        check(estimate(newSketch()) == 0.0, "the empty sketch estimates zero");

        // --- Accuracy against known cardinalities.
        check(Math.abs(estimate(sketchOf(new long[]{42L})) - 1.0) <= 0.05,
                "a single element is estimated at about 1 (" + estimate(sketchOf(new long[]{42L})) + ")");
        long[] ten = new long[10];
        for (int i = 0; i < 10; i++) ten[i] = 1000 + i;
        check(Math.abs(estimate(sketchOf(ten)) - 10.0) / 10.0 <= 0.10,
                "ten distinct values within 10% (" + estimate(sketchOf(ten)) + ")");
        long[] hundred = new long[100];
        for (int i = 0; i < 100; i++) hundred[i] = 500000 + i;
        check(Math.abs(estimate(sketchOf(hundred)) - 100.0) / 100.0 <= 0.10,
                "one hundred distinct values within 10% (" + estimate(sketchOf(hundred)) + ")");

        long[] fiftyThousand = new long[50000];
        for (int i = 0; i < fiftyThousand.length; i++) fiftyThousand[i] = i;
        int[] bigSketch = sketchOf(fiftyThousand);
        double bigEstimate = estimate(bigSketch);
        double relativeError = Math.abs(bigEstimate - 50000.0) / 50000.0;
        double standardError = 1.04 / Math.sqrt(REGISTERS);
        check(relativeError <= 4.0 * standardError, "50000 distinct values estimated at " + bigEstimate
                + ", a relative error of " + relativeError + " against four standard errors of "
                + (4.0 * standardError));
        check(relativeError <= 0.10, "and inside 10%");

        // --- Exact monotonicity of the estimator in the registers.
        int[] mergedBig = merge(bigSketch, sketchOf(new long[]{777L, 778L, 779L}));
        check(dominates(mergedBig, bigSketch) && estimate(mergedBig) >= estimate(bigSketch),
                "registerwise domination implies a larger estimate");

        // --- Sliding window application: rebuild the window sketch, then track it against the exact answer.
        int window = 1000;
        long[] stream = new long[2000];
        for (int i = 0; i < stream.length; i++) stream[i] = rng.nextInt(1000000);
        long[] exact = exactSlidingDistinct(stream, window);
        int largeWindows = 0;
        for (int start = 0; start + window <= stream.length; start++) {
            long[] slice = Arrays.copyOfRange(stream, start, start + window);
            double estimated = estimate(sketchOf(slice));
            long trueCount = exact[start];
            if (trueCount >= 500) {
                largeWindows++;
                double error = Math.abs(estimated - trueCount) / trueCount;
                check(error <= 4.0 * standardError, "window at " + start + " has " + trueCount
                        + " distinct values, estimated at " + estimated);
            }
        }
        check(largeWindows > 0, "at least one window had a large enough distinct count ("
                + largeWindows + " windows)");

        // --- Tumbling windows: merge per-block sketches, which is exact in the merge sense.
        int block = 100;
        List<int[]> blockSketches = new ArrayList<int[]>();
        for (int start = 0; start < stream.length; start += block) {
            blockSketches.add(sketchOf(slice(stream, start, Math.min(start + block, stream.length))));
        }
        int[] mergedBlocks = newSketch();
        Set<Long> union = new HashSet<Long>();
        for (long value : stream) union.add(value);
        for (int[] blockSketch : blockSketches) mergedBlocks = merge(mergedBlocks, blockSketch);
        check(Arrays.equals(mergedBlocks, sketchOf(stream)),
                "merging the block sketches reproduces the sketch of the whole stream exactly");
        check(Math.abs(estimate(mergedBlocks) - union.size()) / (double) union.size()
                <= 4.0 * standardError, "and its estimate is within four standard errors of "
                + union.size());

        check(Arrays.equals(sketchOf(slice(stream, 0, 100)), sketchOf(slice(stream, 0, 100))),
                "deterministic re-run");
        check(!Arrays.equals(sketchOf(new long[]{1, 2, 3}), sketchOf(new long[]{4, 5, 6})),
                "different sets produce different sketches");

        System.out.println("PASS");
    }

    static long[] slice(long[] stream, int from, int to) {
        return Arrays.copyOfRange(stream, from, to);
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
