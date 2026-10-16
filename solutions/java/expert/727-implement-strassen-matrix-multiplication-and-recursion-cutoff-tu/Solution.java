/** Task 727 - Implement Strassen matrix multiplication and recursion cutoff tuning. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Strassen replaces one n x n product with seven (n/2) x (n/2) products and eighteen quadrant additions, so
 * the multiply count drops from n^3 to n^log2(7) ~ n^2.807 at the price of a much worse constant on the
 * cheap operation. That makes the recursion *cutoff* - the size at which the code falls back to the school
 * method - the whole story: the asymptotic win only pays off once the block is large enough, and a cutoff
 * that is too large never recurses at all. This solution runs both algorithms over the same integer inputs
 * (so equality is exact, with no floating-point tolerance anywhere), counts multiplications and additions
 * exactly, and sweeps the cutoff to find the minimum of a two-parameter cost model.
 *
 * Invariants verified:
 *   - equality    Strassen's product equals the schoolbook product exactly, for every size tested, including
 *                 odd sizes that need zero padding and the degenerate 0x0 and 1x1 cases
 *   - closed form the multiply count of a Strassen run is 7^levels * cutoff^3, checked against the count the
 *                 implementation actually reports, and it beats n^3 as soon as the recursion bites
 *   - cutoff      a cutoff at or above n degenerates to the schoolbook count; every cutoff produces the same
 *                 product; and the tuned cost curve has an interior minimum (neither extreme is optimal)
 *   - safety      the integer entries stay far inside the 32-bit range, so no check is measuring overflow
 *
 * Simplified: the recursion pads every size up to the next power of two instead of using a mixed-radix
 * scheme, and the tuning cost model uses two fixed constants (multiply dearer than add) rather than measured
 * timings. The verified properties are exactness of the product, the exact operation counts and the shape of
 * the cutoff curve.
 */
import java.util.*;

public class Solution {
    /** Exact operation counters: one addition or subtraction counts as one. */
    static final class Counter {
        long multiplies;
        long additions;
    }

    /** Schoolbook product of two square int matrices. */
    static int[][] naive(int[][] a, int[][] b, Counter counter) {
        int n = a.length;
        int[][] c = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                int v = a[i][k];
                for (int j = 0; j < n; j++) c[i][j] += v * b[k][j];
            }
        }
        if (counter != null) {
            counter.multiplies += (long) n * n * n;
            counter.additions += (long) n * n * (n - 1);
        }
        return c;
    }

    private static int[][] add(int[][] x, int[][] y, Counter counter) {
        int n = x.length;
        int[][] out = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = x[i][j] + y[i][j];
        }
        if (counter != null) counter.additions += (long) n * n;
        return out;
    }

    private static int[][] sub(int[][] x, int[][] y, Counter counter) {
        int n = x.length;
        int[][] out = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = x[i][j] - y[i][j];
        }
        if (counter != null) counter.additions += (long) n * n;
        return out;
    }

    /** Copies a quadrant out of a square matrix. */
    private static int[][] quadrant(int[][] a, int row, int col, int half) {
        int[][] out = new int[half][half];
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) out[i][j] = a[row + i][col + j];
        }
        return out;
    }

    /** Strassen's seven-product recursion, falling back to the school method at `cutoff`. */
    static int[][] strassen(int[][] a, int[][] b, int cutoff, Counter counter) {
        int n = a.length;
        if (n <= cutoff) return naive(a, b, counter);
        int half = n / 2;
        int[][] a11 = quadrant(a, 0, 0, half);
        int[][] a12 = quadrant(a, 0, half, half);
        int[][] a21 = quadrant(a, half, 0, half);
        int[][] a22 = quadrant(a, half, half, half);
        int[][] b11 = quadrant(b, 0, 0, half);
        int[][] b12 = quadrant(b, 0, half, half);
        int[][] b21 = quadrant(b, half, 0, half);
        int[][] b22 = quadrant(b, half, half, half);
        int[][] m1 = strassen(add(a11, a22, counter), add(b11, b22, counter), cutoff, counter);
        int[][] m2 = strassen(add(a21, a22, counter), b11, cutoff, counter);
        int[][] m3 = strassen(a11, sub(b12, b22, counter), cutoff, counter);
        int[][] m4 = strassen(a22, sub(b21, b11, counter), cutoff, counter);
        int[][] m5 = strassen(add(a11, a12, counter), b22, cutoff, counter);
        int[][] m6 = strassen(sub(a21, a11, counter), add(b11, b12, counter), cutoff, counter);
        int[][] m7 = strassen(sub(a12, a22, counter), add(b21, b22, counter), cutoff, counter);
        int[][] c11 = add(sub(add(m1, m4, counter), m5, counter), m7, counter);
        int[][] c12 = add(m3, m5, counter);
        int[][] c21 = add(m2, m4, counter);
        int[][] c22 = add(sub(add(m1, m3, counter), m2, counter), m6, counter);
        int[][] c = new int[n][n];
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                c[i][j] = c11[i][j];
                c[i][half + j] = c12[i][j];
                c[half + i][j] = c21[i][j];
                c[half + i][half + j] = c22[i][j];
            }
        }
        return c;
    }

    /** Pads a matrix up to the next power of two >= size. */
    static int[][] pad(int[][] a, int size) {
        int n = a.length;
        int target = Integer.highestOneBit(Math.max(1, n - 1)) * 2;
        target = Math.max(target, 1);
        int[][] out = new int[target][target];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], 0, out[i], 0, n);
        return out;
    }

    static int[][] crop(int[][] a, int size) {
        int[][] out = new int[size][size];
        for (int i = 0; i < size; i++) System.arraycopy(a[i], 0, out[i], 0, size);
        return out;
    }

    /** Full Strassen product of arbitrary square sizes, padding internally. */
    static int[][] strassenProduct(int[][] a, int[][] b, int cutoff, Counter counter) {
        if (a.length == 0) return new int[0][0];
        int[][] pa = pad(a, a.length);
        int[][] pb = pad(b, b.length);
        return crop(strassen(pa, pb, cutoff, counter), a.length);
    }

    /** Closed form for the multiply count: 7^levels school products of size cutoff. */
    static long predictedMultiplies(int size, int cutoff) {
        int n = Integer.highestOneBit(Math.max(1, size - 1)) * 2;
        n = Math.max(n, 1);
        long levels = 0;
        long block = n;
        while (block > cutoff) {
            block /= 2;
            levels++;
        }
        long products = 1;
        for (int k = 0; k < levels; k++) products *= 7;
        return products * block * block * block;
    }

    private static long rngState = 0xDA942042E4DD58B5L;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextEntry() { return (int) Math.floorMod(nextLong(), 19) - 9; }

    static int[][] randomMatrix(int n) {
        int[][] a = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = nextEntry();
        }
        return a;
    }

    static boolean same(int[][] a, int[][] b) { return Arrays.deepEquals(a, b); }

    static long maxMagnitude(int[][] a) {
        long worst = 0;
        for (int[] row : a) for (int v : row) worst = Math.max(worst, Math.abs((long) v));
        return worst;
    }

    public static void main(String[] args) {
        // --- degenerate sizes --------------------------------------------------
        Counter zero = new Counter();
        int[][] emptyProduct = strassenProduct(new int[0][0], new int[0][0], 8, zero);
        check(emptyProduct.length == 0 && zero.multiplies == 0, "a 0x0 product is empty and free");
        int[][] one = {{7}};
        int[][] oneProduct = strassenProduct(one, new int[][]{{3}}, 8, new Counter());
        check(oneProduct.length == 1 && oneProduct[0][0] == 21, "1x1 Strassen is just the scalar product");

        // --- exactness against the schoolbook product -------------------------
        for (int n : new int[]{1, 2, 3, 5, 6, 7, 8, 9, 16, 17, 24}) {
            int[][] a = randomMatrix(n);
            int[][] b = randomMatrix(n);
            int[][] expected = naive(a, b, new Counter());
            int[][] got = strassenProduct(a, b, Math.max(1, n / 4), new Counter());
            check(same(got, expected), "Strassen must equal the schoolbook product at n=" + n);
            check(maxMagnitude(got) * 4 < Integer.MAX_VALUE,
                    "entries at n=" + n + " must stay far from overflow, max " + maxMagnitude(got));
        }

        // --- multiply counts against the closed form ---------------------------
        for (int n : new int[]{8, 16, 32, 64}) {
            int[][] a = randomMatrix(n);
            int[][] b = randomMatrix(n);
            Counter naiveCounter = new Counter();
            naive(a, b, naiveCounter);
            check(naiveCounter.multiplies == (long) n * n * n, "schoolbook does n^3 multiplies");
            for (int cutoff : new int[]{2, 4, 8, 16}) {
                Counter counter = new Counter();
                int[][] product = strassenProduct(a, b, cutoff, counter);
                check(same(product, naive(a, b, null)), "cutoff " + cutoff + " must not change the product");
                check(counter.multiplies == predictedMultiplies(n, cutoff),
                        "multiply count at n=" + n + " cutoff=" + cutoff + ": " + counter.multiplies
                                + " vs the closed form " + predictedMultiplies(n, cutoff));
            }
            Counter noRecursion = new Counter();
            strassenProduct(a, b, n, noRecursion);
            check(noRecursion.multiplies == (long) n * n * n,
                    "a cutoff of n degenerates to the schoolbook count");
            Counter recursive = new Counter();
            strassenProduct(a, b, 2, recursive);
            check(recursive.multiplies < naiveCounter.multiplies,
                    "recursion must save multiplies at n=" + n + ": " + recursive.multiplies + " vs "
                            + naiveCounter.multiplies);
            check(recursive.additions > naiveCounter.additions,
                    "and must spend more additions: " + recursive.additions + " vs "
                            + naiveCounter.additions);
        }

        // --- cutoff tuning under the two-constant cost model -------------------
        int n = 128;
        int[][] a = randomMatrix(n);
        int[][] b = randomMatrix(n);
        int[][] expected = naive(a, b, null);
        int[] cutoffs = {2, 4, 8, 16, 32, 64, 128};
        double[] costs = new double[cutoffs.length];
        for (int k = 0; k < cutoffs.length; k++) {
            Counter counter = new Counter();
            int[][] product = strassenProduct(a, b, cutoffs[k], counter);
            check(same(product, expected), "cutoff " + cutoffs[k] + " must produce the same product at n=128");
            costs[k] = counter.multiplies * 3.0 + counter.additions;
        }
        int best = 0;
        for (int k = 1; k < costs.length; k++) if (costs[k] < costs[best]) best = k;
        check(best > 0 && best < costs.length - 1,
                "the tuned cutoff must be interior, best is " + cutoffs[best]);
        check(costs[best] < costs[0] && costs[best] < costs[costs.length - 1],
                "the interior cutoff beats both extremes: " + costs[best]);
        check(costs[costs.length - 1] > costs[0],
                "never recursing is the worst option at this size: " + costs[costs.length - 1] + " vs "
                        + costs[0]);

        // --- determinism -------------------------------------------------------
        Counter repeat = new Counter();
        strassenProduct(a, b, cutoffs[best], repeat);
        Counter again = new Counter();
        check(same(strassenProduct(a, b, cutoffs[best], again), expected) && again.multiplies == repeat.multiplies,
                "the same cutoff must give the same product and the same counts");
        check(repeat.multiplies < (long) n * n * n,
                "the tuned cutoff still beats the schoolbook multiply count");

        System.out.println("Strassen at n=128, entries in [-9,9], cost model multiplies*3 + additions");
        for (int k = 0; k < cutoffs.length; k++) {
            System.out.println("  cutoff " + cutoffs[k] + " -> modelled cost " + costs[k]);
        }
        System.out.println("best cutoff " + cutoffs[best] + " with " + repeat.multiplies
                + " multiplies against the schoolbook " + (long) n * n * n);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
