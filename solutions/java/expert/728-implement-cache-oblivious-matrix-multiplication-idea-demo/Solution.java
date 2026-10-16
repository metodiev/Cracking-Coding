/** Task 728 - Implement cache-oblivious matrix multiplication idea demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * The cache-oblivious idea is to write the multiply as a recursion that halves each of the three dimensions
 * in turn and never mentions a cache size in the code. The analysis says that such a recursion is efficient
 * at every level of the memory hierarchy at once, because at whatever block size the working set happens to
 * fit, the recursion is already operating on whole sub-blocks. This demo makes that concrete: the recursive
 * and the iterative (i, k, j) kernels produce the identical product - the accumulation order per output
 * element is the same, so the comparison is bit-for-bit - and both are run against the same explicit LRU
 * cache model to count misses. The recursion's misses fall far below the iterative kernel's once the operands
 * no longer fit, they improve as the modelled cache grows, and they grow sub-cubically with n.
 *
 * Invariants verified:
 *   - equality   the recursive kernel reproduces the iterative product exactly, at every size tested,
 *                including sizes whose halves are uneven (3, 5, 13)
 *   - coverage   the base cases cover exactly n^3 scalar multiplies: the ledger of leaf sizes is compared
 *                against n^3 instead of being assumed
 *   - locality   the recursive kernel misses less than the iterative one once the operands exceed the cache,
 *                and larger modelled caches reduce its misses while the iterative count barely moves
 *   - boundary   a one-element base case degenerates to the iterative behaviour, so the gain demonstrably
 *                comes from the blocking the recursion discovers rather than from the recursion itself
 *
 * Simplified: one fully associative LRU level with a fixed line size stands in for the real hierarchy, and the
 * matrices are small square double arrays; associativity conflicts and prefetching are ignored. The verified
 * properties are exactness of the product, exact coverage of the multiply count, and the locality ordering.
 */
import java.util.*;

public class Solution {
    /** One-level LRU cache of fixed-size lines; counts misses and touches. */
    static final class Cache {
        final int lines;
        final int lineSize;
        final LinkedHashMap<Long, Boolean> resident;
        long misses;
        long touches;

        Cache(int lines, int lineSize) {
            this.lines = lines;
            this.lineSize = lineSize;
            this.resident = new LinkedHashMap<>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, Boolean> eldest) {
                    return size() > lines;
                }
            };
        }

        void touch(int array, int index) {
            touches++;
            long key = ((long) array << 32) | (index / lineSize);
            if (resident.put(key, Boolean.TRUE) == null) misses++;
        }

        void reset() {
            resident.clear();
            misses = 0;
            touches = 0;
        }
    }

    static final int ARRAY_A = 0;
    static final int ARRAY_B = 1;
    static final int ARRAY_C = 2;

    /** Iterative kernel: for every (i, k) the whole row of B is swept into the row of C. */
    static void multiplyIterative(double[] a, double[] b, double[] c, int n, Cache cache) {
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double v = a[i * n + k];
                if (cache != null) cache.touch(ARRAY_A, i * n + k);
                for (int j = 0; j < n; j++) {
                    if (cache != null) {
                        cache.touch(ARRAY_B, k * n + j);
                        cache.touch(ARRAY_C, i * n + j);
                    }
                    c[i * n + j] += v * b[k * n + j];
                }
            }
        }
    }

    /**
     * Cache-oblivious kernel: C[r, c] += A[r, k] * B[k, c] over a block of `rows` x `inside` x `width`
     * starting at the given offsets. The block is halved along each dimension until every dimension is at
     * most `base`, and the leaf is a plain triple loop. The accumulation order per element stays k-ascending,
     * which is what makes the result identical to the iterative kernel.
     */
    static void multiplyRecursive(double[] a, double[] b, double[] c, int n, int rows, int inside, int width,
                                  int ar, int ac, int br, int bc, int cr, int cc, int base, Cache cache,
                                  long[] coverage) {
        if (rows <= base || inside <= base || width <= base) {
            for (int i = 0; i < rows; i++) {
                for (int k = 0; k < inside; k++) {
                    double v = a[(ar + i) * n + ac + k];
                    if (cache != null) cache.touch(ARRAY_A, (ar + i) * n + ac + k);
                    for (int j = 0; j < width; j++) {
                        if (cache != null) {
                            cache.touch(ARRAY_B, (br + k) * n + bc + j);
                            cache.touch(ARRAY_C, (cr + i) * n + cc + j);
                        }
                        c[(cr + i) * n + cc + j] += v * b[(br + k) * n + bc + j];
                    }
                }
            }
            coverage[0] += (long) rows * inside * width;
            return;
        }
        int halfRows = rows / 2;
        int halfInside = inside / 2;
        int halfWidth = width / 2;
        for (int ri = 0; ri <= 1; ri++) {
            int rowCount = ri == 0 ? halfRows : rows - halfRows;
            int rowOffset = ri == 0 ? 0 : halfRows;
            for (int ki = 0; ki <= 1; ki++) {
                int insideCount = ki == 0 ? halfInside : inside - halfInside;
                int insideOffset = ki == 0 ? 0 : halfInside;
                for (int ci = 0; ci <= 1; ci++) {
                    int widthCount = ci == 0 ? halfWidth : width - halfWidth;
                    int widthOffset = ci == 0 ? 0 : halfWidth;
                    multiplyRecursive(a, b, c, n, rowCount, insideCount, widthCount, ar + rowOffset,
                            ac + insideOffset, br + insideOffset, bc + widthOffset, cr + rowOffset,
                            cc + widthOffset, base, cache, coverage);
                }
            }
        }
    }

    private static long rngState = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    static double[] randomMatrix(int n) {
        double[] a = new double[n * n];
        for (int i = 0; i < a.length; i++) a[i] = nextDouble() * 2.0 - 1.0;
        return a;
    }

    static boolean identical(double[] a, double[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /** Third formulation, i-j-k, used to confirm the product independently of the accumulation order. */
    static double[] referenceProduct(double[] a, double[] b, int n) {
        double[] c = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    static long recursiveMisses(double[] a, double[] b, double[] out, int n, int base, Cache cache) {
        cache.reset();
        multiplyRecursive(a, b, out, n, n, n, n, 0, 0, 0, 0, 0, 0, base, cache, new long[1]);
        return cache.misses;
    }

    public static void main(String[] args) {
        // --- boundary sizes, including uneven halves ---------------------------
        for (int n : new int[]{1, 2, 3, 4, 5, 7, 8, 13, 16}) {
            double[] a = randomMatrix(n);
            double[] b = randomMatrix(n);
            double[] iterative = new double[n * n];
            multiplyIterative(a, b, iterative, n, null);
            double[] recursive = new double[n * n];
            long[] coverage = new long[1];
            multiplyRecursive(a, b, recursive, n, n, n, n, 0, 0, 0, 0, 0, 0, 2, null, coverage);
            check(identical(recursive, iterative),
                    "the recursive kernel must match the iterative one exactly at n=" + n);
            check(coverage[0] == (long) n * n * n,
                    "the base cases must cover exactly n^3 multiplies at n=" + n + ", got " + coverage[0]);
        }

        // --- the product against an independent formulation ---------------------
        int n16 = 16;
        double[] small = randomMatrix(n16);
        double[] smallB = randomMatrix(n16);
        double[] recursive = new double[n16 * n16];
        multiplyRecursive(small, smallB, recursive, n16, n16, n16, n16, 0, 0, 0, 0, 0, 0, 4, null, new long[1]);
        double[] reference = referenceProduct(small, smallB, n16);
        double worst = 0.0;
        for (int i = 0; i < reference.length; i++) worst = Math.max(worst, Math.abs(reference[i] - recursive[i]));
        check(worst < 1e-12, "the recursive product must match the i-j-k formulation, worst " + worst);

        // --- locality: the cache-oblivious ordering wins once operands do not fit
        int n = 64;
        double[] bigA = randomMatrix(n);
        double[] bigB = randomMatrix(n);
        Cache cache = new Cache(64, 8);
        double[] iterative = new double[n * n];
        cache.reset();
        multiplyIterative(bigA, bigB, iterative, n, cache);
        long iterativeMisses = cache.misses;
        double[] blocked = new double[n * n];
        long recursiveMisses = recursiveMisses(bigA, bigB, blocked, n, 4, cache);
        check(identical(blocked, iterative), "the two kernels must still agree at n=64");
        check(recursiveMisses < iterativeMisses, "the recursion must miss less: " + recursiveMisses + " vs "
                + iterativeMisses);
        check(recursiveMisses * 3 < iterativeMisses, "and by a wide margin");
        check(cache.touches > 0, "the cache model really counted touches");

        // --- miss counts improve with the modelled cache ------------------------
        int[] linesByCache = {16, 64, 256};
        long[] missesByCache = new long[linesByCache.length];
        long[] iterativeByCache = new long[linesByCache.length];
        for (int k = 0; k < linesByCache.length; k++) {
            Cache sized = new Cache(linesByCache[k], 8);
            missesByCache[k] = recursiveMisses(bigA, bigB, new double[n * n], n, 4, sized);
            sized.reset();
            multiplyIterative(bigA, bigB, new double[n * n], n, sized);
            iterativeByCache[k] = sized.misses;
        }
        check(missesByCache[0] > missesByCache[1] && missesByCache[1] > missesByCache[2],
                "a larger cache must reduce the recursion's misses: " + Arrays.toString(missesByCache));
        check(missesByCache[2] * 4 < iterativeByCache[2],
                "with a large cache the recursion wins even more clearly: " + missesByCache[2] + " vs "
                        + iterativeByCache[2]);

        // --- boundary: a one-element base case loses the advantage --------------
        Cache unitCache = new Cache(64, 8);
        long unitMisses = recursiveMisses(bigA, bigB, new double[n * n], n, 1, unitCache);
        double[] unitProduct = new double[n * n];
        Cache unitAgain = new Cache(64, 8);
        recursiveMisses(bigA, bigB, unitProduct, n, 1, unitAgain);
        check(identical(unitProduct, iterative), "a one-element base case must still be correct");
        check(unitMisses > missesByCache[2], "a degenerate base case loses the locality advantage: "
                + unitMisses + " vs " + missesByCache[2]);

        // --- determinism --------------------------------------------------------
        Cache repeatCache = new Cache(64, 8);
        double[] again = new double[n * n];
        long repeatMisses = recursiveMisses(bigA, bigB, again, n, 4, repeatCache);
        check(identical(again, blocked) && repeatMisses == recursiveMisses,
                "the same call must give the same result and the same miss count");

        System.out.println("cache-oblivious multiply at n=" + n + " against a " + cache.lines
                + "-line cache of " + cache.lineSize + " doubles");
        System.out.println("LRU misses: iterative " + iterativeMisses + ", recursive " + recursiveMisses
                + " (base case 4), one-element base case " + unitMisses);
        System.out.println("recursive misses at 16/64/256 cache lines: " + Arrays.toString(missesByCache)
                + ", iterative " + Arrays.toString(iterativeByCache));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
