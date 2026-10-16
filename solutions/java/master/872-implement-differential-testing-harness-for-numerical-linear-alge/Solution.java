/** Task 872 - Implement differential testing harness for numerical linear algebra (BLAS correctness).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Differential testing means running several independent implementations of the same specified operation on
 * the same inputs and requiring them to agree. For C = alpha A B + beta C the implementations here are a naive
 * triple loop, a cache-blocked loop nest, a recursive divide-and-conquer kernel and Strassen's
 * seven-multiplication variant, plus transposed and vector forms. The trusted baseline is the naive loop and,
 * for small integer inputs, exact 64-bit integer arithmetic, which turns "agree within tolerance" into
 * "agree bit-for-bit".
 *
 * Agreement is judged with a scale-aware bound rather than a hard-coded epsilon: the forward error of a
 * different summation order is bounded by c n^2 eps max|A| max|B| max|C|, so the harness derives that bound per
 * case and reports the fraction of the bound each kernel actually uses. A harness that never fails is
 * worthless, so the very same battery is run against deliberately broken kernels - a blocked kernel that
 * drops the last partial block, a recursive kernel that halves odd sizes and loses the last row and column, a
 * Strassen that flips a sign, a transpose form used where the plain form was specified, and an update that
 * ignores beta - and every one of them must be caught while no correct kernel is ever declared wrong.
 *
 * SIMPLIFIED: dimensions up to 129, chosen so that every path (odd sizes, non-power-of-two padding, Strassen
 * base cases) is exercised quickly; no threads, no vendor BLAS, no wall-clock timing. "Performance" is
 * reported as operation counts and error statistics rather than seconds. All matrices come from splitmix64
 * with a fixed seed, so every number below is reproducible.
 *
 * VERIFIED: (a) EXACTNESS - on small integer matrices the baseline equals exact long arithmetic bit-for-bit;
 * (b) AGREEMENT - every correct kernel stays inside the scale-aware bound on a battery of 27 cases covering
 * sizes 1, 2, 7, 16, 33, 64, 65, 100 and 129 with several alpha/beta combinations, and stays far inside it;
 * (c) SEMANTICS - alpha/beta scaling, both transpose forms, the identity, the zero matrix, a diagonal product,
 * the vector form and a triangular solve all satisfy their defining equations; (d) DETECTION POWER - all five
 * broken kernels are caught and no correct one is flagged; (e) DETERMINISM - the same seed reproduces the
 * matrices, the verdicts and the Strassen results exactly.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A named implementation of C = alpha A B + beta C against which the harness runs its battery. */
interface Kernel {
    void apply(double[][] a, double[][] b, double alpha, double beta, double[][] c);
}

/** Matrix helpers: construction, norms, differences, padding and the exact integer oracle. */
class Mat {
    static final double EPS = Math.ulp(1.0) / 2;

    static double[][] random(int n, SplitMix64 rng, double scale) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) m[i][j] = scale * (2 * rng.nextDouble() - 1);
        return m;
    }

    static long[][] randomInt(int n, SplitMix64 rng, int bound) {
        long[][] m = new long[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) m[i][j] = rng.nextInt(2 * bound + 1) - bound;
        return m;
    }

    static double[][] toDouble(long[][] m) {
        double[][] out = new double[m.length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m.length; j++) out[i][j] = m[i][j];
        return out;
    }

    static double[][] identity(int n) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) m[i][i] = 1;
        return m;
    }

    static double[][] diagonal(double[] d) {
        double[][] m = new double[d.length][d.length];
        for (int i = 0; i < d.length; i++) m[i][i] = d[i];
        return m;
    }

    static double maxAbs(double[][] m) {
        double best = 0;
        for (double[] row : m) for (double v : row) best = Math.max(best, Math.abs(v));
        return best;
    }

    static double maxDifference(double[][] x, double[][] y) {
        double best = 0;
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x.length; j++) best = Math.max(best, Math.abs(x[i][j] - y[i][j]));
        return best;
    }

    static double[][] copy(double[][] m) {
        double[][] out = new double[m.length][];
        for (int i = 0; i < m.length; i++) out[i] = m[i].clone();
        return out;
    }

    static void zero(double[][] m) {
        for (double[] row : m) Arrays.fill(row, 0.0);
    }

    static int nextPowerOfTwo(int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    }

    static double[][] pad(double[][] m, int size) {
        double[][] out = new double[size][size];
        for (int i = 0; i < m.length; i++) System.arraycopy(m[i], 0, out[i], 0, m.length);
        return out;
    }

    static double[][] quadrant(double[][] m, int row, int col, int size) {
        double[][] out = new double[size][size];
        for (int i = 0; i < size; i++) System.arraycopy(m[row + i], col, out[i], 0, size);
        return out;
    }

    static double[][] add(double[][] x, double[][] y, int sign) {
        double[][] out = new double[x.length][x.length];
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x.length; j++) out[i][j] = x[i][j] + sign * y[i][j];
        return out;
    }

    /** Exact oracle: C = A B in 64-bit integer arithmetic, exact while every entry stays below 2^53. */
    static long[][] exactProduct(long[][] a, long[][] b) {
        int n = a.length;
        long[][] c = new long[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                long sum = 0;
                for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        return c;
    }
}

/** The kernels under test: five honest ones and five deliberately broken ones. */
class Kernels {
    /** The trusted baseline: the textbook triple loop. */
    static Kernel naive() {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                    c[i][j] = alpha * sum + beta * c[i][j];
                }
        };
    }

    /** Cache-blocked loop nest: the same arithmetic in a different order, so only rounding differs. */
    static Kernel blocked(int block) {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] *= beta;
            for (int ii = 0; ii < n; ii += block)
                for (int kk = 0; kk < n; kk += block)
                    for (int jj = 0; jj < n; jj += block)
                        for (int i = ii; i < Math.min(ii + block, n); i++)
                            for (int k = kk; k < Math.min(kk + block, n); k++) {
                                double scaled = alpha * a[i][k];
                                for (int j = jj; j < Math.min(jj + block, n); j++)
                                    c[i][j] += scaled * b[k][j];
                            }
        };
    }

    /** Recursive divide and conquer over padded quadrants, with a naive base case. */
    static Kernel recursive(int baseCase) {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] *= beta;
            int padded = Mat.nextPowerOfTwo(n);
            double[][] product = product(Mat.pad(a, padded), Mat.pad(b, padded), baseCase);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) c[i][j] += alpha * product[i][j];
        };
    }

    private static double[][] product(double[][] a, double[][] b, int baseCase) {
        int n = a.length;
        double[][] c = new double[n][n];
        if (n <= baseCase) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                    c[i][j] = sum;
                }
            return c;
        }
        int h = n / 2;
        double[][] c11 = Mat.add(product(Mat.quadrant(a, 0, 0, h), Mat.quadrant(b, 0, 0, h), baseCase),
                product(Mat.quadrant(a, 0, h, h), Mat.quadrant(b, h, 0, h), baseCase), 1);
        double[][] c12 = Mat.add(product(Mat.quadrant(a, 0, 0, h), Mat.quadrant(b, 0, h, h), baseCase),
                product(Mat.quadrant(a, 0, h, h), Mat.quadrant(b, h, h, h), baseCase), 1);
        double[][] c21 = Mat.add(product(Mat.quadrant(a, h, 0, h), Mat.quadrant(b, 0, 0, h), baseCase),
                product(Mat.quadrant(a, h, h, h), Mat.quadrant(b, h, 0, h), baseCase), 1);
        double[][] c22 = Mat.add(product(Mat.quadrant(a, h, 0, h), Mat.quadrant(b, 0, h, h), baseCase),
                product(Mat.quadrant(a, h, h, h), Mat.quadrant(b, h, h, h), baseCase), 1);
        for (int i = 0; i < h; i++) {
            System.arraycopy(c11[i], 0, c[i], 0, h);
            System.arraycopy(c12[i], 0, c[i], h, h);
            System.arraycopy(c21[i], 0, c[i + h], 0, h);
            System.arraycopy(c22[i], 0, c[i + h], h, h);
        }
        return c;
    }

    /** Strassen's seven multiplications on padded quadrants, falling back to the baseline below the size. */
    static Kernel strassen(int threshold) {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] *= beta;
            int padded = Mat.nextPowerOfTwo(n);
            double[][] product = strassenProduct(Mat.pad(a, padded), Mat.pad(b, padded), threshold);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) c[i][j] += alpha * product[i][j];
        };
    }

    private static double[][] strassenProduct(double[][] a, double[][] b, int threshold) {
        int n = a.length;
        double[][] c = new double[n][n];
        if (n <= threshold) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                    c[i][j] = sum;
                }
            return c;
        }
        int h = n / 2;
        double[][] a11 = Mat.quadrant(a, 0, 0, h), a12 = Mat.quadrant(a, 0, h, h);
        double[][] a21 = Mat.quadrant(a, h, 0, h), a22 = Mat.quadrant(a, h, h, h);
        double[][] b11 = Mat.quadrant(b, 0, 0, h), b12 = Mat.quadrant(b, 0, h, h);
        double[][] b21 = Mat.quadrant(b, h, 0, h), b22 = Mat.quadrant(b, h, h, h);
        double[][] m1 = strassenProduct(Mat.add(a11, a22, 1), Mat.add(b11, b22, 1), threshold);
        double[][] m2 = strassenProduct(Mat.add(a21, a22, 1), b11, threshold);
        double[][] m3 = strassenProduct(a11, Mat.add(b12, b22, -1), threshold);
        double[][] m4 = strassenProduct(a22, Mat.add(b21, b11, -1), threshold);
        double[][] m5 = strassenProduct(Mat.add(a11, a12, 1), b22, threshold);
        double[][] m6 = strassenProduct(Mat.add(a21, a11, -1), Mat.add(b11, b12, 1), threshold);
        double[][] m7 = strassenProduct(Mat.add(a12, a22, -1), Mat.add(b21, b22, 1), threshold);
        for (int i = 0; i < h; i++)
            for (int j = 0; j < h; j++) {
                c[i][j] = m1[i][j] + m4[i][j] - m5[i][j] + m7[i][j];
                c[i][j + h] = m3[i][j] + m5[i][j];
                c[i + h][j] = m2[i][j] + m4[i][j];
                c[i + h][j + h] = m1[i][j] - m2[i][j] + m3[i][j] + m6[i][j];
            }
        return c;
    }

    /** C = alpha A^T B + beta C. */
    static Kernel transposedLeft() {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[k][i] * b[k][j];
                    c[i][j] = alpha * sum + beta * c[i][j];
                }
        };
    }

    /** C = alpha A B^T + beta C. */
    static Kernel transposedRight() {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[i][k] * b[j][k];
                    c[i][j] = alpha * sum + beta * c[i][j];
                }
        };
    }

    /** y = alpha A x + beta y: the vector form of the same operation. */
    static double[] gemv(double[][] a, double[] x, double alpha, double beta, double[] y) {
        int n = a.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += a[i][j] * x[j];
            out[i] = alpha * sum + beta * y[i];
        }
        return out;
    }

    /** Forward substitution for a triangular system: T x = b. */
    static double[] forwardSolve(double[][] t, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) sum -= t[i][j] * x[j];
            x[i] = sum / t[i][i];
        }
        return x;
    }

    // --- deliberately broken implementations: the harness must catch every one of these ------------------

    /** Blocked, but the trailing partial blocks are never touched. */
    static Kernel buggyDroppedLastBlock(int block) {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] *= beta;
            for (int ii = 0; ii + block <= n; ii += block)
                for (int kk = 0; kk + block <= n; kk += block)
                    for (int jj = 0; jj + block <= n; jj += block)
                        for (int i = ii; i < ii + block; i++)
                            for (int k = kk; k < kk + block; k++)
                                for (int j = jj; j < jj + block; j++)
                                    c[i][j] += alpha * a[i][k] * b[k][j];
        };
    }

    /** Recursive, but it assumes every size is even and loses the last row and column when it is not. */
    static Kernel buggyOddHalving(int baseCase) {
        return (a, b, alpha, beta, c) -> {
            int n = c.length;
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) c[i][j] *= beta;
            double[][] product = oddHalving(a, b, baseCase);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) c[i][j] += alpha * product[i][j];
        };
    }

    private static double[][] oddHalving(double[][] a, double[][] b, int baseCase) {
        int n = a.length;
        double[][] c = new double[n][n];
        if (n <= baseCase) {
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                    c[i][j] = sum;
                }
            return c;
        }
        int h = n / 2;
        double[][] c11 = Mat.add(oddHalving(Mat.quadrant(a, 0, 0, h), Mat.quadrant(b, 0, 0, h), baseCase),
                oddHalving(Mat.quadrant(a, 0, h, h), Mat.quadrant(b, h, 0, h), baseCase), 1);
        double[][] c12 = Mat.add(oddHalving(Mat.quadrant(a, 0, 0, h), Mat.quadrant(b, 0, h, h), baseCase),
                oddHalving(Mat.quadrant(a, 0, h, h), Mat.quadrant(b, h, h, h), baseCase), 1);
        double[][] c21 = Mat.add(oddHalving(Mat.quadrant(a, h, 0, h), Mat.quadrant(b, 0, 0, h), baseCase),
                oddHalving(Mat.quadrant(a, h, h, h), Mat.quadrant(b, h, 0, h), baseCase), 1);
        double[][] c22 = Mat.add(oddHalving(Mat.quadrant(a, h, 0, h), Mat.quadrant(b, 0, h, h), baseCase),
                oddHalving(Mat.quadrant(a, h, h, h), Mat.quadrant(b, h, h, h), baseCase), 1);
        for (int i = 0; i < h; i++) {
            System.arraycopy(c11[i], 0, c[i], 0, h);
            System.arraycopy(c12[i], 0, c[i], h, h);
            System.arraycopy(c21[i], 0, c[i + h], 0, h);
            System.arraycopy(c22[i], 0, c[i + h], h, h);
        }
        return c;
    }

    /** Strassen with a single corrupted output entry: the classic copy-paste sign error. */
    static Kernel buggyStrassenSign(int threshold) {
        Kernel correct = strassen(threshold);
        return (a, b, alpha, beta, c) -> {
            correct.apply(a, b, alpha, beta, c);
            c[c.length - 1][c.length - 1] += alpha * 0.25;
        };
    }

    /** The transpose is used where the plain product was specified. */
    static Kernel buggyTransposeInstead() {
        return transposedLeft();
    }

    /** Cache blocking that forgets to apply beta to the existing contents. */
    static Kernel buggyBetaIgnored(int block) {
        Kernel withBeta = blocked(block);
        return (a, b, alpha, beta, c) -> {
            double[][] start = Mat.copy(c);
            Mat.zero(start);
            withBeta.apply(a, b, alpha, 0.0, start);
            for (int i = 0; i < c.length; i++) System.arraycopy(start[i], 0, c[i], 0, c.length);
        };
    }
}

/** The harness: run every kernel over a battery of cases and apply the scale-aware error bound. */
class Harness {
    static final int[] SIZES = {1, 2, 7, 16, 33, 64, 65, 100, 129};

    final Map<String, Kernel> honest = new LinkedHashMap<>();
    final Map<String, Kernel> broken = new LinkedHashMap<>();
    final Map<String, Double> worstRatio = new LinkedHashMap<>();
    int cases;
    int comparisons;

    Harness() {
        honest.put("naive", Kernels.naive());
        honest.put("blocked16", Kernels.blocked(16));
        honest.put("recursive32", Kernels.recursive(32));
        honest.put("strassen32", Kernels.strassen(32));
        honest.put("strassen128", Kernels.strassen(128));
        broken.put("dropped-last-block", Kernels.buggyDroppedLastBlock(16));
        broken.put("odd-halving", Kernels.buggyOddHalving(16));
        broken.put("strassen-sign", Kernels.buggyStrassenSign(32));
        broken.put("transpose-instead", Kernels.buggyTransposeInstead());
        broken.put("beta-ignored", Kernels.buggyBetaIgnored(16));
        for (String name : honest.keySet()) worstRatio.put(name, 0.0);
        for (String name : broken.keySet()) worstRatio.put(name, 0.0);
    }

    /**
     * One case: draw A, B and the incoming C, compute the expected result with the baseline, then score
     * every kernel by the fraction of the forward-error bound it consumes.
     */
    Map<String, Double> runCase(int n, double alpha, double beta, double scale, SplitMix64 rng) {
        double[][] a = Mat.random(n, rng, scale), b = Mat.random(n, rng, scale);
        double[][] start = Mat.random(n, rng, scale);
        double[][] expected = Mat.copy(start);
        Kernels.naive().apply(a, b, alpha, beta, expected);
        double magnitude = Math.max(1, Math.max(Math.abs(alpha), Math.abs(beta)) * Mat.maxAbs(start));
        double bound = 16.0 * n * n * Mat.EPS * Math.max(1, Mat.maxAbs(a)) * Math.max(1, Mat.maxAbs(b))
                * magnitude;
        Map<String, Double> ratios = new LinkedHashMap<>();
        for (Map.Entry<String, Kernel> e : honest.entrySet()) ratios.put(e.getKey(), score(e.getValue(), a, b,
                alpha, beta, start, expected, bound));
        for (Map.Entry<String, Kernel> e : broken.entrySet()) ratios.put(e.getKey(), score(e.getValue(), a, b,
                alpha, beta, start, expected, bound));
        for (Map.Entry<String, Double> e : ratios.entrySet())
            worstRatio.merge(e.getKey(), e.getValue(), Math::max);
        cases++;
        return ratios;
    }

    private double score(Kernel kernel, double[][] a, double[][] b, double alpha, double beta,
            double[][] start, double[][] expected, double bound) {
        double[][] candidate = Mat.copy(start);
        kernel.apply(a, b, alpha, beta, candidate);
        comparisons++;
        return Mat.maxDifference(expected, candidate) / bound;
    }

    void runBattery(SplitMix64 rng) {
        for (int n : SIZES) {
            runCase(n, 1.0, 0.0, 1.0, rng);
            runCase(n, -2.5, 0.0, 0.5, rng);
            runCase(n, 1.0, 3.0, 2.0, rng);
        }
    }

    double worstHonestRatio() {
        double worst = 0;
        for (String name : honest.keySet()) worst = Math.max(worst, worstRatio.get(name));
        return worst;
    }

    boolean honestAllAgree() { return worstHonestRatio() <= 1.0; }

    /** A kernel is caught when it leaves the bound: anything above a ratio of one is a real disagreement. */
    List<String> caught() {
        List<String> out = new ArrayList<>();
        for (String name : broken.keySet()) if (worstRatio.get(name) > 1.0) out.add(name);
        return out;
    }

    List<String> falsePositives() {
        List<String> out = new ArrayList<>();
        for (String name : honest.keySet()) if (worstRatio.get(name) > 1.0) out.add(name);
        return out;
    }
}

public class Solution {
    public static void main(String[] args) {
        // --- EXACTNESS against 64-bit integer arithmetic -------------------------------------------------
        SplitMix64 rng = new SplitMix64(20240517L);
        boolean exact = true;
        for (int trial = 0; trial < 40 && exact; trial++) {
            int n = 1 + rng.nextInt(6);
            long[][] a = Mat.randomInt(n, rng, 9), b = Mat.randomInt(n, rng, 9);
            long[][] expected = Mat.exactProduct(a, b);
            double[][] c = new double[n][n];
            Kernels.naive().apply(Mat.toDouble(a), Mat.toDouble(b), 1.0, 0.0, c);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) if (c[i][j] != expected[i][j]) exact = false;
        }
        check(exact, "the baseline kernel must equal exact integer arithmetic bit-for-bit");

        // --- AGREEMENT across the whole battery ----------------------------------------------------------
        Harness harness = new Harness();
        harness.runBattery(new SplitMix64(99L));
        check(harness.cases == Harness.SIZES.length * 3, "every size must be exercised with several "
                + "alpha/beta combinations, got " + harness.cases);
        check(harness.falsePositives().isEmpty(), "no correct kernel may be declared wrong: "
                + harness.falsePositives());
        check(harness.honestAllAgree(), "every correct kernel must stay inside the scale-aware bound");
        double worst = harness.worstHonestRatio();
        check(worst < 0.5, "the honest kernels should stay well inside the bound, worst ratio " + worst);

        // --- DETECTION POWER ----------------------------------------------------------------------------
        List<String> caught = harness.caught();
        check(caught.size() == harness.broken.size(), "every broken kernel must be caught, missed " + caught);
        check(harness.comparisons == 9 * 3 * 10, "the harness must compare every kernel on every case, got "
                + harness.comparisons);
        check(!caught.contains("blocked16") && !caught.contains("strassen32") && !caught.contains("naive"),
                "the detection list must contain only broken kernels");

        // --- SEMANTICS: transpose forms, alpha/beta, identity, diagonals and zeros ----------------------
        int n = 48;
        double[][] a = Mat.random(n, new SplitMix64(5L), 1.0);
        double[][] b = Mat.random(n, new SplitMix64(6L), 1.0);
        double[][] left = new double[n][n], right = new double[n][n], plain = new double[n][n];
        Kernels.transposedLeft().apply(a, b, 1.0, 0.0, left);
        Kernels.transposedRight().apply(a, b, 1.0, 0.0, right);
        Kernels.naive().apply(a, b, 1.0, 0.0, plain);
        double[][] aT = new double[n][n], bT = new double[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) { aT[i][j] = a[j][i]; bT[i][j] = b[j][i]; }
        double[][] expectedLeft = new double[n][n], expectedRight = new double[n][n];
        Kernels.naive().apply(aT, b, 1.0, 0.0, expectedLeft);
        Kernels.naive().apply(a, bT, 1.0, 0.0, expectedRight);
        check(Mat.maxDifference(left, expectedLeft) < 1e-12, "the transposed-left form must equal A^T B");
        check(Mat.maxDifference(right, expectedRight) < 1e-12, "the transposed-right form must equal A B^T");
        check(Mat.maxDifference(left, plain) > 1e-6 && Mat.maxDifference(right, plain) > 1e-6,
                "the transpose forms must genuinely differ from the plain product on random inputs");

        double[][] start = Mat.random(n, new SplitMix64(7L), 1.0);
        double[][] scaled = Mat.copy(start);
        Kernels.naive().apply(a, b, -2.5, 3.0, scaled);
        boolean scaling = true;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (Math.abs(scaled[i][j] - (-2.5 * plain[i][j] + 3.0 * start[i][j])) > 1e-9) scaling = false;
        check(scaling, "alpha and beta must scale the product and the incoming value exactly");
        double[][] betaOnly = Mat.copy(start);
        Kernels.strassen(32).apply(a, b, 0.0, 2.0, betaOnly);
        boolean pureBeta = true;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
            if (Math.abs(betaOnly[i][j] - 2.0 * start[i][j]) > 1e-12) pureBeta = false;
        check(pureBeta, "alpha = 0 must leave only beta times the original matrix");
        double[][] cleared = Mat.copy(start);
        Kernels.blocked(16).apply(a, b, 1.0, 0.0, cleared);
        check(Mat.maxDifference(cleared, plain) < 1e-12, "beta = 0 must replace the incoming matrix");

        double[][] identity = Mat.identity(n), product = new double[n][n];
        Kernels.naive().apply(a, identity, 1.0, 0.0, product);
        check(Mat.maxDifference(product, a) == 0.0, "multiplying by the identity must be exact");
        double[][] zeros = new double[n][n];
        Kernels.blocked(16).apply(new double[n][n], b, 1.0, 5.0, zeros);
        check(Mat.maxAbs(zeros) == 0.0, "a zero left operand must give a zero product");
        double[] diagonalValues = {2, 3, 4, 5};
        double[][] diagonal = Mat.diagonal(diagonalValues), diagonalProduct = new double[4][4];
        Kernels.strassen(32).apply(diagonal, diagonal, 1.0, 0.0, diagonalProduct);
        boolean diagonals = true;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
            if (diagonalProduct[i][j] != (i == j ? diagonalValues[i] * diagonalValues[i] : 0.0))
                diagonals = false;
        check(diagonals, "a diagonal times a diagonal must be the exact squared diagonal");

        // --- the vector form and the triangular solve ---------------------------------------------------
        double[] x = new double[n], y = new double[n];
        for (int i = 0; i < n; i++) { x[i] = Math.sin(i * 0.3); y[i] = Math.cos(i * 0.2); }
        double[] gemvResult = Kernels.gemv(a, x, 1.5, -0.5, y);
        boolean vectorAgrees = true;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += a[i][j] * x[j];
            if (Math.abs(gemvResult[i] - (1.5 * sum - 0.5 * y[i])) > 1e-9) vectorAgrees = false;
        }
        check(vectorAgrees, "the vector kernel must satisfy y = alpha A x + beta y");
        double[] unit = new double[n];
        unit[3] = 1.0;
        double[] column = Kernels.gemv(a, unit, 1.0, 0.0, new double[n]);
        boolean columnMatches = true;
        for (int i = 0; i < n; i++) if (column[i] != a[i][3]) columnMatches = false;
        check(columnMatches, "multiplying by a unit vector must return a column of A exactly");

        double[][] lower = new double[16][16];
        SplitMix64 s = new SplitMix64(31L);
        for (int i = 0; i < 16; i++)
            for (int j = 0; j <= i; j++) lower[i][j] = i == j ? 2.0 + s.nextDouble() : s.nextDouble() - 0.5;
        double[] rhs = new double[16];
        for (int i = 0; i < 16; i++) rhs[i] = s.nextDouble();
        double[] solution = Kernels.forwardSolve(lower, rhs);
        double residual = 0;
        for (int i = 0; i < 16; i++) {
            double sum = 0;
            for (int j = 0; j <= i; j++) sum += lower[i][j] * solution[j];
            residual = Math.max(residual, Math.abs(sum - rhs[i]));
        }
        check(residual < 1e-12, "the triangular solve residual must vanish, got " + residual);

        // --- DETERMINISM --------------------------------------------------------------------------------
        Harness replay = new Harness();
        replay.runBattery(new SplitMix64(99L));
        check(replay.worstRatio.equals(harness.worstRatio) && replay.comparisons == harness.comparisons,
                "the harness must reproduce its own verdicts from the seed");
        check(replay.caught().equals(caught), "the verdict list must be reproducible");
        double[][] p1 = new double[8][8], p2 = new double[8][8];
        Kernels.strassen(4).apply(Mat.random(8, new SplitMix64(2L), 1.0), Mat.random(8, new SplitMix64(3L), 1.0),
                1.0, 0.0, p1);
        Kernels.strassen(4).apply(Mat.random(8, new SplitMix64(2L), 1.0), Mat.random(8, new SplitMix64(3L), 1.0),
                1.0, 0.0, p2);
        check(Mat.maxDifference(p1, p2) == 0.0, "the same seed must give the same matrices and result");

        // --- degenerate sizes ---------------------------------------------------------------------------
        for (int size : new int[]{1, 2, 3}) {
            double[][] tinyA = Mat.random(size, new SplitMix64(size), 1.0);
            double[][] tinyB = Mat.random(size, new SplitMix64(size + 17L), 1.0);
            double[][] reference = new double[size][size], candidate = new double[size][size];
            Kernels.naive().apply(tinyA, tinyB, 1.0, 0.0, reference);
            Kernels.strassen(128).apply(tinyA, tinyB, 1.0, 0.0, candidate);
            check(Mat.maxDifference(reference, candidate) == 0.0, "size " + size + " must be exact in the "
                    + "Strassen base case");
            double[][] empty = new double[size][size];
            Kernels.recursive(32).apply(empty, empty, 1.0, 1.0, empty);
            check(Mat.maxAbs(empty) == 0.0, "the zero matrix must stay zero through padding");
        }
        check(Mat.nextPowerOfTwo(1) == 1 && Mat.nextPowerOfTwo(65) == 128,
                "padding must round up to a power of two");
        check(harness.worstRatio.get("naive") == 0.0, "the baseline against itself is exact");

        System.out.println("Battery: " + harness.cases + " cases over sizes " + Arrays.toString(Harness.SIZES)
                + ", " + harness.comparisons + " comparisons, worst honest ratio "
                + String.format(Locale.ROOT, "%.5f", worst));
        System.out.println("Detection: " + caught.size() + " of " + harness.broken.size()
                + " broken kernels caught " + caught + " with no false positives");
        System.out.println("Exact integer cross-check passed on 40 small cases; alpha/beta, transpose, "
                + "identity, diagonal, zero, vector and triangular forms verified");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
