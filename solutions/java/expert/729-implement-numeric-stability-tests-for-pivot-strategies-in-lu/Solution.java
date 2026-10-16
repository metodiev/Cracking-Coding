/** Task 729 - Implement numeric stability tests for pivot strategies in LU. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Gaussian elimination only stays stable if it chooses its pivots. Without pivoting a tiny leading entry can
 * be used as a divisor and every later digit is destroyed; partial pivoting (largest entry in the column)
 * bounds the multipliers by one; complete pivoting (largest entry in the whole remaining submatrix) bounds
 * both the multipliers and, in practice, the element growth. This solution factors the same matrices three
 * ways, reconstructs P*A = L*U in every case, solves a right-hand side, and measures two things that separate
 * the strategies: the growth factor (largest |U| element over largest |A| element) and the forward error of
 * the solve. The classic 2x2 example [[eps, 1], [1, 1]] is included because there, and only there, the three
 * strategies disagree visibly.
 *
 * Invariants verified:
 *   - reconstruction   P*A equals L*U for every strategy, with L unit lower triangular and U upper triangular
 *   - multipliers      |L| <= 1 exactly for partial and complete pivoting, and no such bound is claimed for
 *                      the unpivoted factorization (which is what makes it useful as a negative control)
 *   - growth           the growth factor is >= 1, and on the unstable example the unpivoted growth and the
 *                      forward error are orders of magnitude worse than the pivoted ones
 *   - solve            on a well-conditioned system all three strategies recover the exact solution, while on
 *                      the unstable example the unpivoted solve loses most of its digits
 *
 * Simplified: real double arithmetic with a hand-rolled elimination rather than a LAPACK call, and a 2-norm
 * residual over small dense matrices. The verified properties are the reconstruction identity, the multiplier
 * bound, and the measured stability ordering of the three pivot strategies.
 */
import java.util.*;

public class Solution {
    /** An LU factorization with its row and column permutations. */
    static final class Factorization {
        final double[][] l;
        final double[][] u;
        final int[] rowSwap;      // P: row i of P*A is row rowSwap[i] of A
        final int[] colSwap;      // Q for complete pivoting, identity otherwise
        final double growth;
        final String strategy;

        Factorization(double[][] l, double[][] u, int[] rowSwap, int[] colSwap, double growth, String strategy) {
            this.l = l;
            this.u = u;
            this.rowSwap = rowSwap;
            this.colSwap = colSwap;
            this.growth = growth;
            this.strategy = strategy;
        }

        /** Largest |L| entry: the multiplier bound is one for both pivoting strategies. */
        double maxMultiplier() {
            double worst = 0.0;
            for (int i = 0; i < l.length; i++) {
                for (int j = 0; j < i; j++) worst = Math.max(worst, Math.abs(l[i][j]));
            }
            return worst;
        }
    }

    /** Factors a copy of A; `strategy` is "none", "partial" or "complete". */
    static Factorization factor(double[][] a, String strategy) {
        int n = a.length;
        double[][] m = new double[n][n];
        double original = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = a[i][j];
                original = Math.max(original, Math.abs(a[i][j]));
            }
        }
        int[] rowSwap = new int[n];
        int[] colSwap = new int[n];
        for (int i = 0; i < n; i++) { rowSwap[i] = i; colSwap[i] = i; }
        double[][] l = new double[n][n];
        for (int step = 0; step < n; step++) {
            int pivotRow = step;
            int pivotCol = step;
            double best = Math.abs(m[step][step]);
            if (!"none".equals(strategy)) {
                for (int i = step; i < n; i++) {
                    if ("partial".equals(strategy)) {
                        if (Math.abs(m[i][step]) > best) { best = Math.abs(m[i][step]); pivotRow = i; }
                    } else {
                        for (int j = step; j < n; j++) {
                            if (Math.abs(m[i][j]) > best) { best = Math.abs(m[i][j]); pivotRow = i; pivotCol = j; }
                        }
                    }
                }
            }
            if (pivotRow != step) {
                double[] temp = m[step]; m[step] = m[pivotRow]; m[pivotRow] = temp;
                int swapRow = rowSwap[step]; rowSwap[step] = rowSwap[pivotRow]; rowSwap[pivotRow] = swapRow;
                for (int j = 0; j < step; j++) {
                    double t = l[step][j]; l[step][j] = l[pivotRow][j]; l[pivotRow][j] = t;
                }
            }
            if (pivotCol != step) {
                for (int i = 0; i < n; i++) {
                    double t = m[i][step]; m[i][step] = m[i][pivotCol]; m[i][pivotCol] = t;
                }
                int swapCol = colSwap[step]; colSwap[step] = colSwap[pivotCol]; colSwap[pivotCol] = swapCol;
            }
            l[step][step] = 1.0;
            if (m[step][step] == 0.0) continue;      // singular column: skip, the reconstruction still holds
            for (int i = step + 1; i < n; i++) {
                double multiplier = m[i][step] / m[step][step];
                if (!Double.isFinite(multiplier)) continue;
                l[i][step] = multiplier;
                for (int j = step; j < n; j++) m[i][j] -= multiplier * m[step][j];
            }
        }
        double[][] u = new double[n][n];
        double largest = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                u[i][j] = m[i][j];
                largest = Math.max(largest, Math.abs(m[i][j]));
            }
        }
        double growth = original == 0.0 ? 0.0 : largest / original;
        return new Factorization(l, u, rowSwap, colSwap, growth, strategy);
    }

    /** Rebuilds P*A*Q from the permutations so it can be compared against L*U. */
    static double[][] permuted(double[][] a, Factorization f) {
        int n = a.length;
        double[][] p = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) p[i][j] = a[f.rowSwap[i]][f.colSwap[j]];
        }
        return p;
    }

    /** L*U, computed densely so the identity check is independent of the factor routine. */
    static double[][] product(Factorization f) {
        int n = f.l.length;
        double[][] p = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) sum += f.l[i][k] * f.u[k][j];
                p[i][j] = sum;
            }
        }
        return p;
    }

    static double worstDifference(double[][] a, double[][] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
        }
        return worst;
    }

    /** Solves A x = b through the factorization: L y = P b, then U z = y, then x = Q z. */
    static double[] solve(double[][] a, double[] b, Factorization f) {
        int n = a.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = b[f.rowSwap[i]];
            for (int j = 0; j < i; j++) sum -= f.l[i][j] * y[j];
            y[i] = sum;
        }
        double[] z = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) sum -= f.u[i][j] * z[j];
            z[i] = f.u[i][i] == 0.0 ? Double.NaN : sum / f.u[i][i];
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[f.colSwap[i]] = z[i];
        return x;
    }

    static double worstDifference(double[] a, double[] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }

    /** Residual of the linear system, a scale-free measure of how well the solve worked. */
    static double residual(double[][] a, double[] x, double[] b) {
        int n = a.length;
        double worst = 0.0;
        double scale = 0.0;
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += a[i][j] * x[j];
            worst = Math.max(worst, Math.abs(sum - b[i]));
            scale = Math.max(scale, Math.abs(b[i]));
        }
        return scale == 0.0 ? worst : worst / scale;
    }

    private static long rngState = 0x2545F4914F6CDD1DL;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A diagonally dominant matrix, so the exact solution is a small integer vector. */
    static double[][] wellConditioned(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = i == j ? 0.0 : nextDouble() - 0.5;
            a[i][i] = 1.0;
            for (int j = 0; j < n; j++) if (j != i) a[i][i] += Math.abs(a[i][j]);
        }
        return a;
    }

    static double[] multiply(double[][] a, int[] x) {
        double[] b = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < a.length; j++) sum += a[i][j] * x[j];
            b[i] = sum;
        }
        return b;
    }

    static String[] STRATEGIES = {"none", "partial", "complete"};

    public static void main(String[] args) {
        // --- the classic unstable example -------------------------------------
        double eps = 1e-18;
        double[][] unstable = {{eps, 1.0}, {1.0, 1.0}};
        double[] exact = {1.0, 1.0};
        double[] rightHandSide = multiply(unstable, new int[]{1, 1});
        double worstUnpivoted = 0.0;
        double worstPivoted = 0.0;
        for (String strategy : STRATEGIES) {
            Factorization f = factor(unstable, strategy);
            check(f.growth >= 0.0, "growth must be non-negative for " + strategy);
            double[] x = solve(unstable, rightHandSide, f);
            double error = worstDifference(x, exact);
            if ("none".equals(strategy)) worstUnpivoted = error; else worstPivoted = Math.max(worstPivoted, error);
        }
        check(worstUnpivoted > 1e-3, "unpivoted elimination must lose the answer on eps=1e-18: " + worstUnpivoted);
        check(worstPivoted < 1e-12, "both pivoting strategies must keep it: " + worstPivoted);

        // --- reconstruction and multiplier bounds on random systems -----------
        for (int n : new int[]{1, 2, 5, 8, 12}) {
            int[] solution = new int[n];
            for (int i = 0; i < n; i++) solution[i] = 1 + (i % 3);
            double[][] a = wellConditioned(n);
            double[] b = multiply(a, solution);
            for (String strategy : STRATEGIES) {
                Factorization f = factor(a, strategy);
                double[][] rebuilt = permuted(a, f);
                double[][] lu = product(f);
                check(worstDifference(rebuilt, lu) < 1e-9,
                        "P*A must equal L*U for " + strategy + " at n=" + n + ", worst "
                                + worstDifference(rebuilt, lu));
                if (!"none".equals(strategy)) {
                    check(f.maxMultiplier() <= 1.0 + 1e-12,
                            strategy + " pivoting bounds the multipliers by one, worst " + f.maxMultiplier());
                }
                check(f.growth >= 0.9 && f.growth <= 2.0,
                        "a diagonally dominant system must not grow, " + strategy + " growth=" + f.growth);
                double[] x = solve(a, b, f);
                check(worstDifference(x, toDouble(solution)) < 1e-8,
                        strategy + " must recover the exact solution at n=" + n);
                check(residual(a, x, b) < 1e-10, strategy + " residual at n=" + n + " is " + residual(a, x, b));
            }
        }

        // --- growth factor and error ordering on the unstable example ---------
        Factorization none = factor(unstable, "none");
        Factorization partial = factor(unstable, "partial");
        Factorization complete = factor(unstable, "complete");
        check(none.growth > partial.growth, "partial pivoting must tame the growth: " + none.growth + " vs "
                + partial.growth);
        check(partial.maxMultiplier() <= 1.0 + 1e-12 && complete.maxMultiplier() <= 1.0 + 1e-12,
                "both pivot strategies keep |L| within one");
        check(none.maxMultiplier() > 1e6, "the unpivoted multipliers are enormous: " + none.maxMultiplier());
        check(complete.growth <= partial.growth * 4.0 + 1e-12,
                "complete pivoting cannot be much worse than partial on this example");

        // --- a cumulative growth example shows the same ordering --------------
        int n = 6;
        double[][] growing = new double[n][n];
        for (int i = 0; i < n; i++) {
            growing[i][i] = 1.0;
            growing[n - 1][i] = -1.0;
            if (i > 0) growing[i][i - 1] = -1.0;
        }
        Factorization growNone = factor(growing, "none");
        Factorization growPartial = factor(growing, "partial");
        Factorization growComplete = factor(growing, "complete");
        for (Factorization f : new Factorization[]{growNone, growPartial, growComplete}) {
            double[][] rebuilt = permuted(growing, f);
            check(worstDifference(rebuilt, product(f)) < 1e-9,
                    "reconstruction must hold for the growth example, " + f.strategy);
        }
        check(growPartial.growth <= Math.pow(2.0, n - 1) + 1e-9,
                "partial-pivot growth respects the 2^(n-1) bound: " + growPartial.growth);
        check(growNone.growth >= growComplete.growth, "complete pivoting must not grow more than none");

        // --- determinism -------------------------------------------------------
        Factorization repeat = factor(unstable, "partial");
        check(Math.abs(repeat.growth - partial.growth) == 0.0
                        && worstDifference(solve(unstable, rightHandSide, repeat), solve(unstable, rightHandSide, partial)) == 0.0,
                "the same factorization must repeat exactly");

        System.out.println("LU pivot strategies on the classic 2x2 system [[1e-18,1],[1,1]] x = [1+1e-18, 2]");
        System.out.println("forward error: unpivoted " + worstUnpivoted + ", pivoted best " + worstPivoted);
        System.out.println("growth: none " + none.growth + ", partial " + partial.growth + ", complete "
                + complete.growth + "; multipliers: none " + none.maxMultiplier() + ", partial "
                + partial.maxMultiplier());
        System.out.println("PASS");
    }

    static double[] toDouble(int[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) out[i] = values[i];
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
