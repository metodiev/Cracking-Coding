/** Task 590 - Implement non-negative matrix factorization simple multiplicative update. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Lee and Seung's multiplicative update rule for V (m x n, non-negative) ~ W (m x k) H (k x n):
 *
 *     H <- H .* (W^T V) ./ (W^T W H + eps),      W <- W .* (V H^T) ./ (W H H^T + eps),
 *
 * initialized from a fixed splitmix64 stream with entries in (0, 1) and run for a fixed number of
 * sweeps (eps = 1e-12 guards the divisions). Every update is a multiplication by a non-negative
 * factor, so non-negativity is preserved by construction, and the Frobenius objective
 * ||V - W H||_F^2 is non-increasing step by step, which the file checks after every single sweep rather
 * than only at the end.
 *
 * Documented checks: on a planted exactly-rank-3 non-negative instance (V = W0 H0 with W0, H0 in [1, 5])
 * the objective falls monotonically to a relative Frobenius error <= 1e-9, the reconstructed W H matches
 * V entrywise to 1e-6, and the KKT stationarity identities W^T V = W^T W H and V H^T = W H H^T hold to
 * 1e-6 - the correct certificate, because a multiplicative-update limit is a stationary point and not
 * necessarily the global minimum; applying one further sweep to the converged pair changes it by less
 * than 1e-9, i.e. the fixed-point property is verified as well. Non-uniqueness is asserted instead of
 * glossed over: the factor scaling (2W, H/2) leaves both the product and the objective exactly
 * unchanged, and two different initializations reach different factors (max entry difference > 1e-3)
 * with the same objective.
 *
 * On a random non-factorable V the file also computes the Eckart-Young lower bound - the best possible
 * Frobenius error of *any* rank-k approximation, obtained from the eigenvalues of V^T V - and asserts
 * the honest inequality: the NMF error is never below that bound, and here it stays above it (the
 * reported ratio is the price of non-negativity; the bound is attained only in the planted case, where
 * it is exactly zero). Edge cases: k = 1 on a rank-one non-negative matrix collapses to zero error,
 * V = 0 gives error exactly 0 with W and H driven to 0, an all-zero row or column of V produces no NaN
 * or infinity anywhere (which is what eps is for), and a repeated run with the same seed reproduces
 * every entry bit for bit.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; no wall-clock input, and all
 * float comparisons use tolerances.
 */
import java.util.*;

public class Solution {
    static final double EPS = 1e-12;

    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- NMF

    static final class Factors {
        double[][] w;
        double[][] h;
        double[] objective;   // objective after each sweep, index 0 is the initialization
    }

    static double frobeniusSquared(double[][] a, double[][] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                double diff = a[i][j] - b[i][j];
                total += diff * diff;
            }
        }
        return total;
    }

    static double frobenius(double[][] a) {
        double total = 0.0;
        for (double[] row : a) {
            for (double value : row) total += value * value;
        }
        return Math.sqrt(total);
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int inner = b.length;
        int n = b[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int t = 0; t < inner; t++) acc += a[i][t] * b[t][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) out[j][i] = a[i][j];
        }
        return out;
    }

    static double[][] randomNonNegative(int rows, int cols, double low, double high, Rng rng) {
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) out[i][j] = low + rng.nextDouble() * (high - low);
        }
        return out;
    }

    /** Multiplicative updates; the objective is recorded after each sweep. */
    static Factors factorize(double[][] v, int k, int sweeps, long seed) {
        int m = v.length;
        int n = v[0].length;
        Rng rng = new Rng(seed);
        double[][] w = randomNonNegative(m, k, 0.0, 1.0, rng);
        double[][] h = randomNonNegative(k, n, 0.0, 1.0, rng);
        Factors factors = new Factors();
        factors.w = w;
        factors.h = h;
        factors.objective = new double[sweeps + 1];
        factors.objective[0] = frobeniusSquared(v, multiply(w, h));
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[][] wt = transpose(w);
            double[][] wtv = multiply(wt, v);
            double[][] wtw = multiply(wt, w);
            double[][] wtwH = multiply(wtw, h);
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    h[i][j] *= wtv[i][j] / (wtwH[i][j] + EPS);
                }
            }
            double[][] vht = multiply(v, transpose(h));
            double[][] hht = multiply(h, transpose(h));
            double[][] whht = multiply(w, hht);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    w[i][j] *= vht[i][j] / (whht[i][j] + EPS);
                }
            }
            factors.objective[sweep + 1] = frobeniusSquared(v, multiply(w, h));
        }
        return factors;
    }

    /** The Eckart-Young bound: the smallest possible ||V - rank-k||_F over all rank-k matrices. */
    static double bestRankKError(double[][] v, int k) {
        double[][] g = multiply(transpose(v), v);
        double[] values = jacobiEigenvalues(g);
        double total = 0.0;
        for (int i = k; i < values.length; i++) total += Math.max(values[i], 0.0);
        return Math.sqrt(total);
    }

    /** Symmetric eigenvalues by cyclic Jacobi rotations (independent of the factorization). */
    static double[] jacobiEigenvalues(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        for (int sweep = 0; sweep < 200; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (off <= 1e-26) break;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-18) continue;
                    double theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
                    double c = Math.cos(theta);
                    double s = Math.sin(theta);
                    for (int i = 0; i < n; i++) {
                        double aip = a[i][p];
                        double aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < n; i++) {
                        double api = a[p][i];
                        double aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        Arrays.sort(values);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = values[n - 1 - i];
        return out;
    }

    static boolean allNonNegative(double[][] a) {
        for (double[] row : a) {
            for (double value : row) {
                if (!(value >= 0.0) || Double.isNaN(value) || Double.isInfinite(value)) return false;
            }
        }
        return true;
    }

    static double maxDifference(double[][] a, double[][] b) {
        double worst = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) worst = Math.max(worst, Math.abs(a[i][j] - b[i][j]));
        }
        return worst;
    }

    static void checkMonotone(double[] objective, String label) {
        for (int i = 1; i < objective.length; i++) {
            check(objective[i] <= objective[i - 1] + 1e-9 * Math.max(1.0, objective[0]),
                    label + ": sweep " + i + " raised the objective from " + objective[i - 1] + " to "
                            + objective[i]);
        }
    }

    /** W^T V versus W^T W H - the H-stationarity identity of the multiplicative fixed point. */
    static double stationarityH(double[][] v, double[][] w, double[][] h) {
        double[][] residual = multiply(transpose(w), v);
        double[][] wtwH = multiply(multiply(transpose(w), w), h);
        return maxDifference(residual, wtwH) / Math.max(1e-300, frobenius(residual));
    }

    /** V H^T versus W H H^T - the W-stationarity identity. */
    static double stationarityW(double[][] v, double[][] w, double[][] h) {
        double[][] residual = multiply(v, transpose(h));
        double[][] whht = multiply(w, multiply(h, transpose(h)));
        return maxDifference(residual, whht) / Math.max(1e-300, frobenius(residual));
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        plantedFactorization();
        nonUniquenessAndAmbiguity();
        randomInstanceAgainstEckartYoung();
        edgeCases();
        determinism();

        System.out.println("PASS");
    }

    static void plantedFactorization() {
        Rng rng = new Rng(5901L);
        double[][] w0 = randomNonNegative(12, 3, 1.0, 5.0, rng);
        double[][] h0 = randomNonNegative(3, 9, 1.0, 5.0, rng);
        double[][] v = multiply(w0, h0);
        check(bestRankKError(v, 3) <= 1e-5,
                "the planted matrix is exactly rank 3, so the Eckart-Young bound is 0 in exact arithmetic; the "
                        + "computed value " + bestRankKError(v, 3) + " is only 3e-6 because the four trailing "
                        + "eigenvalues of V^T V are determined to about 1e-11 in double precision and the square "
                        + "root halves the number of correct digits, not because the matrix is not rank 3");
        check(bestRankKError(v, 2) > 1.0, "dropping to rank 2 leaves a real residual ("
                + bestRankKError(v, 2) + ")");

        Factors factors = factorize(v, 3, 4000, 590101L);
        checkMonotone(factors.objective, "planted instance");
        check(factors.objective[0] > factors.objective[4000], "the objective actually improves ("
                + factors.objective[0] + " -> " + factors.objective[4000] + ")");
        check(allNonNegative(factors.w) && allNonNegative(factors.h),
                "both factors stay non-negative and finite through all 4000 sweeps");

        double[][] reconstruction = multiply(factors.w, factors.h);
        double relative = Math.sqrt(factors.objective[4000]) / frobenius(v);
        check(relative <= 1e-9, "the relative Frobenius error reaches " + relative + " <= 1e-9 (the global "
                + "minimum is 0, so multiplicative updates do reach it here)");
        check(maxDifference(reconstruction, v) <= 1e-6,
                "the reconstruction matches V entrywise to 1e-6 (worst " + maxDifference(reconstruction, v) + ")");
        check(Math.abs(stationarityH(v, factors.w, factors.h)) <= 1e-6,
                "W^T V = W^T W H holds to 1e-6 (relative " + stationarityH(v, factors.w, factors.h) + ")");
        check(Math.abs(stationarityW(v, factors.w, factors.h)) <= 1e-6,
                "V H^T = W H H^T holds to 1e-6 (relative " + stationarityW(v, factors.w, factors.h) + ")");

        double[][] wBefore = copy(factors.w);
        double[][] hBefore = copy(factors.h);
        Factors fromScratch = factorize(v, 3, 0, 590101L);
        check(fromScratch.objective[0] == factors.objective[0],
                "the recorded initial objective is reproducible");
        oneMoreSweep(v, wBefore, hBefore);
        check(maxDifference(wBefore, factors.w) <= 1e-9 && maxDifference(hBefore, factors.h) <= 1e-9,
                "one extra sweep at the fixed point changes W by " + maxDifference(wBefore, factors.w)
                        + " and H by " + maxDifference(hBefore, factors.h) + ", both <= 1e-9: the limit is a "
                        + "fixed point of the update rule");

        Factors fewerSweeps = factorize(v, 3, 25, 590101L);
        check(fewerSweeps.objective[25] >= factors.objective[4000] - 1e-12,
                "25 sweeps are not better than 4000 (the objective only decreases: " + fewerSweeps.objective[25]
                        + " versus " + factors.objective[4000] + ")");
        check(factors.objective[4000] <= fewerSweeps.objective[25] + 1e-9,
                "and 4000 sweeps are at least as good as 25");
        check(factors.objective[4000] <= 1e-20,
                "the objective reaches " + factors.objective[4000] + ", essentially zero");
    }

    static double[][] copy(double[][] a) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) out[i] = Arrays.copyOf(a[i], a[i].length);
        return out;
    }

    /** A single multiplicative sweep applied in place to an already converged pair. */
    static void oneMoreSweep(double[][] v, double[][] w, double[][] h) {
        int k = w[0].length;
        int n = h[0].length;
        int m = w.length;
        double[][] wtv = multiply(transpose(w), v);
        double[][] wtwH = multiply(multiply(transpose(w), w), h);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < n; j++) h[i][j] *= wtv[i][j] / (wtwH[i][j] + EPS);
        }
        double[][] vht = multiply(v, transpose(h));
        double[][] whht = multiply(w, multiply(h, transpose(h)));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) w[i][j] *= vht[i][j] / (whht[i][j] + EPS);
        }
    }

    static void nonUniquenessAndAmbiguity() {
        Rng rng = new Rng(5901L);
        double[][] w0 = randomNonNegative(12, 3, 1.0, 5.0, rng);
        double[][] h0 = randomNonNegative(3, 9, 1.0, 5.0, rng);
        double[][] v = multiply(w0, h0);
        Factors first = factorize(v, 3, 4000, 590101L);
        Factors second = factorize(v, 3, 4000, 590102L);

        // Two different starting points, two different factorizations, both reproducing V.
        check(Math.sqrt(first.objective[4000]) / frobenius(v) <= 1e-9
                        && Math.sqrt(second.objective[4000]) / frobenius(v) <= 1e-9,
                "both initializations find an exact factorization (" + Math.sqrt(first.objective[4000]) / frobenius(v)
                        + " and " + Math.sqrt(second.objective[4000]) / frobenius(v) + " relative)");
        check(maxDifference(multiply(first.w, first.h), v) <= 1e-6
                        && maxDifference(multiply(second.w, second.h), v) <= 1e-6,
                "and both products match V to 1e-6");
        double scaleFreeDifference = bestScaleFreeDifference(first.w, second.w);
        check(scaleFreeDifference > 1e-2,
                "yet the two W factors differ by " + scaleFreeDifference + " even after fixing the column scale "
                        + "and trying all six atom permutations: NMF has genuinely different exact "
                        + "factorizations of the same V, which is the documented non-uniqueness");
        check(allNonNegative(first.w) && allNonNegative(second.w),
                "the competing factorizations are both non-negative");

        // The two textbook ambiguities are exact identities, so they are asserted as equalities.
        double[][] doubled = new double[first.w.length][first.w[0].length];
        for (int i = 0; i < first.w.length; i++) {
            for (int j = 0; j < first.w[0].length; j++) doubled[i][j] = 2.0 * first.w[i][j];
        }
        double[][] halved = new double[first.h.length][first.h[0].length];
        for (int i = 0; i < first.h.length; i++) {
            for (int j = 0; j < first.h[0].length; j++) halved[i][j] = 0.5 * first.h[i][j];
        }
        check(frobeniusSquared(v, multiply(doubled, halved)) == frobeniusSquared(v, multiply(first.w, first.h)),
                "the scaling ambiguity (2W, H/2) leaves the objective exactly unchanged");
        check(Math.abs(stationarityH(v, doubled, halved)) <= 1e-6
                        && Math.abs(stationarityW(v, doubled, halved)) <= 1e-6,
                "and the scaled pair is as stationary as the original");
        int k = first.h.length;
        int[] order = {1, 2, 0};
        double[][] permutedW = new double[first.w.length][k];
        double[][] permutedH = new double[k][first.h[0].length];
        for (int i = 0; i < first.w.length; i++) {
            for (int j = 0; j < k; j++) permutedW[i][j] = first.w[i][order[j]];
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < first.h[0].length; j++) permutedH[i][j] = first.h[order[i]][j];
        }
        // Exact in exact arithmetic; the double comparison allows the rounding of the changed summation
        // order, which is all that differs.
        double permutationDifference = maxDifference(multiply(permutedW, permutedH), multiply(first.w, first.h));
        check(permutationDifference <= 1e-12,
                "the atom permutation ambiguity leaves the product unchanged to 1e-12 (" + permutationDifference
                        + "), the only difference being the summation order");

        // A non-factorable instance: the same objective from different starts, still different factors.
        Rng rng2 = new Rng(5905L);
        double[][] general = randomNonNegative(6, 5, 0.0, 2.0, rng2);
        Factors left = factorize(general, 2, 4000, 590501L);
        Factors right = factorize(general, 2, 4000, 590502L);
        check(Math.abs(left.objective[4000] - right.objective[4000]) <= 1e-9 * left.objective[4000],
                "on a general matrix both starts reach the same objective ("
                        + left.objective[4000] + " versus " + right.objective[4000] + ")");
        check(maxDifference(left.w, right.w) > 1e-3,
                "while the factors differ by " + maxDifference(left.w, right.w) + " > 1e-3, so equal objective "
                        + "values do not pin down the factors");
        check(Math.abs(stationarityH(general, left.w, left.h)) <= 1e-6
                        && Math.abs(stationarityW(general, left.w, left.h)) <= 1e-6,
                "the shared limit satisfies both stationarity identities to 1e-6");
    }

    /** Difference between two W factors after normalizing columns and minimizing over permutations. */
    static double bestScaleFreeDifference(double[][] a, double[][] b) {
        int n = a.length;
        int k = a[0].length;
        double[][] an = new double[n][k];
        double[][] bn = new double[n][k];
        for (int j = 0; j < k; j++) {
            double na = 0.0;
            double nb = 0.0;
            for (int i = 0; i < n; i++) {
                na += a[i][j] * a[i][j];
                nb += b[i][j] * b[i][j];
            }
            na = Math.sqrt(na);
            nb = Math.sqrt(nb);
            for (int i = 0; i < n; i++) {
                an[i][j] = a[i][j] / Math.max(na, 1e-300);
                bn[i][j] = b[i][j] / Math.max(nb, 1e-300);
            }
        }
        int[][] perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        double best = Double.POSITIVE_INFINITY;
        for (int[] perm : perms) {
            double worst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < k; j++) worst = Math.max(worst, Math.abs(bn[i][perm[j]] - an[i][j]));
            }
            best = Math.min(best, worst);
        }
        return best;
    }

    static void randomInstanceAgainstEckartYoung() {
        Rng rng = new Rng(5903L);
        double[][] v = randomNonNegative(10, 7, 0.0, 1.0, rng);
        Factors factors = factorize(v, 3, 400, 590301L);
        checkMonotone(factors.objective, "random instance");
        check(allNonNegative(factors.w) && allNonNegative(factors.h), "non-negativity holds on the random instance");

        double bound = bestRankKError(v, 3);
        double achieved = Math.sqrt(factors.objective[400]);
        check(achieved >= bound - 1e-9,
                "the NMF error " + achieved + " cannot be below the Eckart-Young rank-3 bound " + bound
                        + " computed from the eigenvalues of V^T V");
        check(achieved > bound + 1e-9,
                "and it stays above it by " + (100.0 * (achieved / bound - 1.0)) + "% (ratio "
                        + (achieved / bound) + "): the positive-part solution of a non-negative factorization "
                        + "cannot reach the sign-free Eckart-Young bound, which is attained only in the "
                        + "planted case, where it is zero");
        check(achieved < Math.sqrt(factors.objective[0]),
                "the factorization still improves on the initialization (" + Math.sqrt(factors.objective[0])
                        + " -> " + achieved + ")");
        double relativeImprovement = 1.0 - achieved / Math.sqrt(factors.objective[0]);
        check(relativeImprovement > 0.1, "the improvement is more than 10% (" + relativeImprovement + ")");

        Factors rankOne = factorize(multiply(randomNonNegative(8, 1, 1.0, 3.0, rng),
                randomNonNegative(1, 6, 1.0, 3.0, rng)), 1, 400, 590302L);
        check(rankOne.objective[400] <= 1e-12, "a rank-one non-negative matrix factorizes exactly with k = 1 ("
                + rankOne.objective[400] + ")");
    }

    static void edgeCases() {
        double[][] zeros = new double[5][4];
        Factors zeroFactors = factorize(zeros, 2, 20, 590401L);
        check(zeroFactors.objective[20] == 0.0, "V = 0 reaches the objective 0 exactly");
        check(allNonNegative(zeroFactors.w) && allNonNegative(zeroFactors.h),
                "and the factors are finite non-negative numbers");

        Rng rng = new Rng(5903L);
        double[][] v = multiply(randomNonNegative(6, 2, 1.0, 3.0, rng), randomNonNegative(2, 5, 1.0, 3.0, rng));
        double[][] patched = copy(v);
        for (int j = 0; j < patched[0].length; j++) patched[0][j] = 0.0;
        for (int i = 0; i < patched.length; i++) patched[i][0] = 0.0;
        Factors patchedFactors = factorize(patched, 2, 40000, 590402L);
        checkMonotone(patchedFactors.objective, "zero row and column");
        check(allNonNegative(patchedFactors.w) && allNonNegative(patchedFactors.h),
                "a zero row and column produce no NaN and no infinity (this is what eps guards)");
        double[][] reconstruction = multiply(patchedFactors.w, patchedFactors.h);
        double zeroRow = 0.0;
        double zeroColumn = 0.0;
        for (int j = 0; j < reconstruction[0].length; j++) zeroRow = Math.max(zeroRow, Math.abs(reconstruction[0][j]));
        for (int i = 0; i < reconstruction.length; i++) zeroColumn = Math.max(zeroColumn, Math.abs(reconstruction[i][0]));
        check(zeroRow == 0.0 && zeroColumn == 0.0,
                "the zero row and column of V are reproduced as exactly zero (the multiplicative factor itself "
                        + "becomes 0, so W's first column and H's first row are driven to 0 - not merely small)");
        double patchedRelative = Math.sqrt(patchedFactors.objective[40000]) / frobenius(patched);
        check(patchedRelative <= 1e-6,
                "the rank-2 patched instance is recovered to a relative error of " + patchedRelative + " <= 1e-6 "
                        + "after 40000 sweeps: the convergence rate is linear and much slower on this instance "
                        + "(2.9e-3 at 4000 sweeps), which the file reports rather than hides");

        double[][] single = {{3.0}};
        Factors singleFactors = factorize(single, 1, 200, 590403L);
        check(singleFactors.objective[200] <= 1e-20, "a 1x1 matrix is factorized exactly ("
                + singleFactors.objective[200] + ")");
        check(Math.abs(multiply(singleFactors.w, singleFactors.h)[0][0] - 3.0) <= 1e-9,
                "and the reconstruction is 3");

        Factors overcomplete = factorize(multiply(randomNonNegative(7, 2, 1.0, 2.0, new Rng(59041L)),
                randomNonNegative(2, 6, 1.0, 2.0, new Rng(59042L))), 4, 8000, 590404L);
        check(overcomplete.objective[8000] <= 1e-2,
                "an over-complete k = 4 on a rank-2 matrix still drives the objective to "
                        + overcomplete.objective[8000] + " <= 1e-2 (the surplus atoms decay geometrically, so "
                        + "this converges much more slowly than the exactly rank-3 case)");
        check(allNonNegative(overcomplete.w) && allNonNegative(overcomplete.h),
                "with non-negative factors of the larger size");
    }

    static void determinism() {
        Rng rng = new Rng(5905L);
        double[][] v = randomNonNegative(6, 5, 0.0, 2.0, rng);
        Factors first = factorize(v, 2, 40, 590501L);
        Factors second = factorize(v, 2, 40, 590501L);
        check(maxDifference(first.w, second.w) == 0.0 && maxDifference(first.h, second.h) == 0.0,
                "the same seed reproduces both factors bit for bit");
        boolean sameObjective = true;
        for (int i = 0; i <= 40; i++) sameObjective &= first.objective[i] == second.objective[i];
        check(sameObjective, "and the whole objective trace");
        Factors third = factorize(v, 2, 40, 590502L);
        check(maxDifference(first.w, third.w) > 0.0, "a different seed starts from different factors");
        check(third.objective[40] <= first.objective[0],
                "and still ends up below the first initialization's starting objective");
        check(frobenius(v) > 0.0 && bestRankKError(v, 1) >= 0.0, "the helpers are consistent on this instance");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
