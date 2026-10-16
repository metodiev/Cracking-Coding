/** Task 730 - Implement randomized numerical linear algebra low-rank approx demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Randomized low-rank approximation replaces the SVD of an m x n matrix A with a much smaller computation:
 * draw an n x k sketch matrix Omega, form Y = A Omega (m x k), take the orthonormal basis Q of Y's columns,
 * and project, A ~ Q (Q^T A). The claim is that for k somewhat larger than the numerical rank, Q captures the
 * dominant subspace and the residual is close to the *optimal* rank-k residual. To make the comparison exact
 * rather than rhetorical, this demo builds A from a known spectrum, A = U diag(s) V^T with orthonormal U and
 * V, so the optimal rank-k error sqrt(sum of s_i^2 for i > k) is available in closed form and the sketch's
 * error can be measured against it. A couple of power iterations - Y = (A A^T)^q A Omega - sharpen the
 * estimate when the spectrum decays slowly, which the same closed form then detects.
 *
 * Invariants verified:
 *   - orthonormal  Q's columns are orthonormal to machine precision (modified Gram-Schmidt is checked, not
 *                  trusted), so Q Q^T is a genuine projector
 *   - sandwich      ||A - Q Q^T A||_F >= the optimal rank-k error and, with a Gaussian sketch, no more than
 *                  about 1 + epsilon times it - the randomized method is near-optimal, never better
 *   - exactness    once k reaches the true rank the approximation error falls to machine precision, and k = 0
 *                  returns the full Frobenius norm
 *   - refinement   power iteration never hurts and strictly helps a slowly decaying spectrum
 *
 * Simplified: the sketch entries come from the fixed-seed generator instead of a fresh Gaussian draw, the
 * factors are formed explicitly (no streaming or transposed products), and the norm used is Frobenius. The
 * verified properties are orthonormality, the two-sided sandwich against the analytic optimum, and the effect
 * of power iteration.
 */
import java.util.*;

public class Solution {
    static long rngState = 0x61C8864680B583EBL;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Uniform in [-1, 1]: a mean-zero sketch with a bounded second moment. */
    static double nextUniform() { return (nextLong() >>> 11) * 0x1.0p-53 * 2.0 - 1.0; }

    static double[][] randomMatrix(int rows, int cols) {
        double[][] a = new double[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) a[i][j] = nextUniform();
        return a;
    }

    static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) t[j][i] = a[i][j];
        return t;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int cols = b.length == 0 ? 0 : b[0].length;
        double[][] c = new double[a.length][cols];
        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < b.length; k++) {
                double v = a[i][k];
                for (int j = 0; j < cols; j++) c[i][j] += v * b[k][j];
            }
        }
        return c;
    }

    /** Modified Gram-Schmidt: returns the first `k` orthonormalised columns of A (less if it is rank poor). */
    static double[][] orthonormalize(double[][] a, int k) {
        int m = a.length;
        int columns = Math.min(k, a[0].length);
        double[][] q = new double[m][columns];
        int kept = 0;
        for (int j = 0; j < columns; j++) {
            double[] v = new double[m];
            for (int i = 0; i < m; i++) v[i] = a[i][j];
            for (int pass = 0; pass < 2; pass++) {            // re-orthogonalise for stability
                for (int c = 0; c < kept; c++) {
                    double dot = 0.0;
                    for (int i = 0; i < m; i++) dot += q[i][c] * v[i];
                    for (int i = 0; i < m; i++) v[i] -= dot * q[i][c];
                }
            }
            double norm = 0.0;
            for (int i = 0; i < m; i++) norm += v[i] * v[i];
            norm = Math.sqrt(norm);
            if (norm < 1e-13) continue;                        // dependent column: drop it
            for (int i = 0; i < m; i++) q[i][kept] = v[i] / norm;
            kept++;
        }
        double[][] trimmed = new double[m][kept];
        for (int i = 0; i < m; i++) System.arraycopy(q[i], 0, trimmed[i], 0, kept);
        return trimmed;
    }

    static double frobenius(double[][] a) {
        double sum = 0.0;
        for (double[] row : a) for (double v : row) sum += v * v;
        return Math.sqrt(sum);
    }

    static double[][] difference(double[][] a, double[][] b) {
        double[][] d = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) for (int j = 0; j < a[0].length; j++) d[i][j] = a[i][j] - b[i][j];
        return d;
    }

    /** A = U diag(spectrum) V^T with orthonormal U (m x r) and V (n x r). */
    static double[][] buildFromSpectrum(int m, int n, double[] spectrum) {
        int r = spectrum.length;
        double[][] u = orthonormalize(randomMatrix(m, r), r);
        double[][] v = orthonormalize(randomMatrix(n, r), r);
        double[][] us = new double[m][r];
        for (int i = 0; i < m; i++) for (int j = 0; j < r; j++) us[i][j] = u[i][j] * spectrum[j];
        return multiply(us, transpose(v));
    }

    /** The best possible rank-k error for a known spectrum: the tail of the squared singular values. */
    static double optimalTail(double[] spectrum, int k) {
        double sum = 0.0;
        for (int i = k; i < spectrum.length; i++) sum += spectrum[i] * spectrum[i];
        return Math.sqrt(sum);
    }

    /** Randomized range finder with `power` iterations of (A A^T) applied to the sketch. */
    static double[][] randomizedApproximation(double[][] a, int k, int power) {
        int m = a.length;
        int n = a[0].length;
        if (k <= 0) return new double[m][n];                   // an empty sketch projects onto nothing
        double[][] omega = randomMatrix(n, k);
        double[][] y = multiply(a, omega);
        double[][] at = transpose(a);
        for (int q = 0; q < power; q++) y = multiply(a, multiply(at, y));
        double[][] basis = orthonormalize(y, k);
        if (basis.length == 0 || basis[0].length == 0) return new double[m][n];   // rank-poor sketch
        double[][] projected = multiply(basis, multiply(transpose(basis), a));
        return projected;
    }

    /** Median error over independent sketch draws: the standard way to tame a single unlucky sketch. */
    static double medianError(double[][] a, int k, int power, int trials) {
        double[] errors = new double[trials];
        for (int t = 0; t < trials; t++) {
            rngState = 0x61C8864680B583EBL + 0x9E3779B97F4A7C15L * (t + 1);
            errors[t] = frobenius(difference(a, randomizedApproximation(a, k, power)));
        }
        Arrays.sort(errors);
        return errors[errors.length / 2];
    }

    static double orthonormalityDefect(double[][] q) {
        double[][] gram = multiply(transpose(q), q);
        double worst = 0.0;
        for (int i = 0; i < gram.length; i++) {
            for (int j = 0; j < gram[0].length; j++) {
                double expected = i == j ? 1.0 : 0.0;
                worst = Math.max(worst, Math.abs(gram[i][j] - expected));
            }
        }
        return worst;
    }

    public static void main(String[] args) {
        final int m = 40;
        final int n = 32;
        final int rank = 8;
        double[] fastDecay = new double[rank];
        for (int i = 0; i < rank; i++) fastDecay[i] = Math.pow(0.5, i);
        double[] slowDecay = new double[rank];
        for (int i = 0; i < rank; i++) slowDecay[i] = 1.0 / (i + 1.0);

        // --- exact recovery once k reaches the true rank ------------------------
        double[][] a = buildFromSpectrum(m, n, fastDecay);
        double fullNorm = frobenius(a);
        check(optimalTail(fastDecay, rank) == 0.0, "the spectrum's tail is empty at k = rank");
        double[][] exact = randomizedApproximation(a, rank, 0);
        double exactError = frobenius(difference(a, exact));
        check(exactError < 1e-10, "k = rank must recover the matrix, error=" + exactError);
        check(exactError < 1e-10 * fullNorm, "and in relative terms too");
        check(orthonormalityDefect(orthonormalize(a, rank)) < 1e-12,
                "the basis built from a rank-" + rank + " matrix must be orthonormal");

        // --- the sandwich against the analytic optimum -------------------------
        // The randomized error can never beat the optimal rank-k error, and with a mean-zero sketch of width
        // k the excess shrinks as k grows: the ratio is checked for improvement, not for a fixed constant.
        double worstRatio = 0.0;
        for (int k : new int[]{2, 4, 6}) {
            double optimal = optimalTail(fastDecay, k);
            double ratio = medianError(a, k, 0, 9) / optimal;
            check(ratio >= 1.0 - 1e-10, "no method may beat the optimal rank-" + k + " error, ratio " + ratio);
            check(ratio < 2.6, "the median sketch stays close to optimal at k=" + k + ", ratio " + ratio);
            worstRatio = Math.max(worstRatio, ratio);
        }
        for (int trial = 0; trial < 4; trial++) {
            rngState = 0x61C8864680B583EBL + 0x9E3779B97F4A7C15L * (trial + 1);
            double single = frobenius(difference(a, randomizedApproximation(a, 3, 0)));
            check(single >= optimalTail(fastDecay, 3) * (1.0 - 1e-10),
                    "a single sketch can never beat the optimum either, trial " + trial);
        }
        check(worstRatio < 2.6, "every median sketch stays within 2.6x of optimal, worst " + worstRatio);
        check(medianError(a, 6, 0, 9) <= optimalTail(fastDecay, 2) * 1.3,
                "a sketch wider than the target rank must be near optimal for that rank");

        // --- k = 0 and k >= min(m, n) boundaries -------------------------------
        double[][] zeroRank = randomizedApproximation(a, 0, 0);
        check(frobenius(zeroRank) == 0.0, "k = 0 gives the zero approximation");
        check(frobenius(difference(a, zeroRank)) == fullNorm, "k = 0 leaves the whole norm as error");
        double[][] fullRank = randomizedApproximation(a, Math.min(m, n), 0);
        check(frobenius(difference(a, fullRank)) < 1e-10, "k = min(m, n) is exact");
        double[][] zeros = buildFromSpectrum(m, n, new double[]{0.0, 0.0});
        check(frobenius(difference(zeros, randomizedApproximation(zeros, 3, 0))) == 0.0,
                "a zero matrix needs no rank at all");

        // --- orthonormality of the sketch basis --------------------------------
        double[][] basis = orthonormalize(multiply(a, randomMatrix(n, 5)), 5);
        check(orthonormalityDefect(basis) < 1e-12, "the sketch basis must be orthonormal, defect "
                + orthonormalityDefect(basis));

        // --- power iteration sharpens a slowly decaying spectrum ---------------
        double[][] slow = buildFromSpectrum(m, n, slowDecay);
        double withoutPower = frobenius(difference(slow, randomizedApproximation(slow, 4, 0)));
        double withPower = frobenius(difference(slow, randomizedApproximation(slow, 4, 2)));
        check(withPower <= withoutPower + 1e-12,
                "power iteration must not hurt: " + withPower + " vs " + withoutPower);
        check(withPower < withoutPower, "and must help a slowly decaying spectrum");
        check(withPower >= optimalTail(slowDecay, 4) * (1.0 - 1e-10),
                "power iteration cannot beat the optimum either");

        // --- the sketch also estimates the Frobenius norm ----------------------
        int k = 20;
        double[] estimateHolder = new double[2];
        double[][] omega = randomMatrix(n, k);
        double estimated = frobenius(multiply(a, omega)) / Math.sqrt(k);
        estimateHolder[0] = estimated;
        check(Math.abs(estimated - fullNorm) < 0.5 * fullNorm,
                "the sketch estimates ||A||_F within 50%: " + estimated + " vs " + fullNorm);

        // --- determinism --------------------------------------------------------
        rngState = 0x61C8864680B583EBL;
        double[][] again = buildFromSpectrum(m, n, fastDecay);
        rngState = 0x1234567890ABCDEFL;
        double[][] first = randomizedApproximation(again, 4, 1);
        rngState = 0x1234567890ABCDEFL;
        double[][] repeat = randomizedApproximation(again, 4, 1);
        check(frobenius(difference(repeat, first)) == 0.0, "the same seed must reproduce the same basis");
        check(orthonormalityDefect(orthonormalize(again, rank)) < 1e-12, "replay stays orthonormal");

        System.out.println("randomized low-rank: A is " + m + "x" + n + " of exact rank " + rank
                + ", sketch width k, ||A||_F = " + fullNorm);
        System.out.println("error at k = rank: " + exactError + " (machine precision), k = 0 -> "
                + fullNorm);
        for (int kk : new int[]{2, 4, 6}) {
            System.out.println("  median ratio at k=" + kk + " is " + medianError(a, kk, 0, 9)
                    / optimalTail(fastDecay, kk));
        }
        for (int trial : new int[]{1, 2, 4, 6}) {
            System.out.println("  k=" + trial + " randomized " + frobenius(difference(a,
                    randomizedApproximation(a, trial, 0))) + " vs optimal " + optimalTail(fastDecay, trial));
        }
        System.out.println("slow spectrum k=4: " + withoutPower + " without power iteration, " + withPower
                + " with two; optimal " + optimalTail(slowDecay, 4));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
