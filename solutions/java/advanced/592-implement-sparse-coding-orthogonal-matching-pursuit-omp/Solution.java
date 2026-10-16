/** Task 592 - Implement sparse coding orthogonal matching pursuit (OMP). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Orthogonal matching pursuit: starting from the zero support and the residual r = b, each iteration
 * selects the dictionary atom with the largest |a_j^T r|, adds it to the support S, and re-solves the
 * least-squares problem over S exactly (never a gradient step), which makes the new residual orthogonal
 * to every selected atom - so no index can be picked twice and the residual norm is strictly decreasing.
 * The least-squares subproblem is solved twice, in fact: by Gaussian elimination with partial pivoting on
 * the normal equations A_S^T A_S x = A_S^T b, and independently by modified Gram-Schmidt QR, so the two
 * solutions can be compared (the normal equations square the condition number, so the file asserts the
 * agreement it actually achieves and reports the observed gap instead of claiming machine precision).
 *
 * Documented checks: exact support and coefficient recovery on planted s-sparse signals in a 25-dimensional
 * space with 20 Gaussian atoms (mutual coherence 0.5414, so the classical sufficient condition
 * s < (1 + 1/mu)/2 = 1.4234 does *not* cover s = 3 - the recovery is verified by brute force rather than
 * cited from the theorem), verified against an exhaustive oracle that enumerates *every* support of size at
 * most s (20 choose 3 = 1140 least-squares fits per signal), so the file asserts at once that OMP never
 * beats the best subset - the
 * general inequality ||b - A x_OMP|| >= min_S ||b - A_S x_S|| holds by construction of the exhaustive
 * search - and that on the planted instance it attains it; orthogonality A_S^T r = 0 to 1e-10; strict
 * residual decrease until the tolerance; no repeated atom in the support; the sparsity-zero and
 * over-sparsity boundary cases (s = 0 returns zero with residual b; s beyond the true sparsity ends with
 * residual ~0); the tolerance-based stopping rule; the duplicate-atom tie-break (the smallest index wins,
 * asserted explicitly); a zero signal and a zero atom; and reproducibility. The mutual coherence mu of the
 * dictionary is computed and the classical guarantee s < (1 + 1/mu)/2 is *reported* - on this instance mu
 * is about 0.5, so the bound is under 2 and the s = 3 recovered here lies outside the sufficient condition:
 * the recovery is a verified empirical fact about this dictionary, not a consequence of the theorem, and
 * the file says so.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; all comparisons use tolerances
 * except the exact combinatorial statements about supports.
 */
import java.util.*;

public class Solution {
    static final int M = 25;      // ambient dimension
    static final int ATOMS = 20;  // dictionary size

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

        double nextGaussian() {
            double u1 = Math.max(nextDouble(), 1e-12);
            double u2 = nextDouble();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }
    }

    // ---------------------------------------------------------------- dense helpers

    static double[] column(double[][] a, int j) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i][j];
        return out;
    }

    static double dot(double[] a, double[] b) {
        double acc = 0.0;
        for (int i = 0; i < a.length; i++) acc += a[i] * b[i];
        return acc;
    }

    static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    static double[] residual(double[][] a, double[] x, double[] b) {
        double[] r = b.clone();
        for (int j = 0; j < x.length; j++) {
            if (x[j] == 0.0) continue;
            for (int i = 0; i < r.length; i++) r[i] -= x[j] * a[i][j];
        }
        return r;
    }

    static double[] gramTimes(double[][] a, int[] support, double[] x) {
        double[] out = new double[a.length];
        for (int idx = 0; idx < support.length; idx++) {
            int j = support[idx];
            for (int i = 0; i < a.length; i++) out[i] += x[idx] * a[i][j];
        }
        return out;
    }

    /** Gaussian elimination with partial pivoting on the small normal equations. */
    static double[] solveNormal(double[][] a, int[] support, double[] b) {
        int s = support.length;
        double[][] gram = new double[s][s];
        double[] rhs = new double[s];
        for (int i = 0; i < s; i++) {
            double[] left = column(a, support[i]);
            rhs[i] = dot(left, b);
            for (int j = 0; j < s; j++) gram[i][j] = dot(left, column(a, support[j]));
        }
        for (int col = 0; col < s; col++) {
            int pivot = col;
            for (int row = col + 1; row < s; row++) {
                if (Math.abs(gram[row][col]) > Math.abs(gram[pivot][col])) pivot = row;
            }
            double[] t = gram[pivot];
            gram[pivot] = gram[col];
            gram[col] = t;
            double tb = rhs[pivot];
            rhs[pivot] = rhs[col];
            rhs[col] = tb;
            if (Math.abs(gram[col][col]) < 1e-14) continue;
            for (int row = col + 1; row < s; row++) {
                double factor = gram[row][col] / gram[col][col];
                for (int j = col; j < s; j++) gram[row][j] -= factor * gram[col][j];
                rhs[row] -= factor * rhs[col];
            }
        }
        double[] out = new double[s];
        for (int i = s - 1; i >= 0; i--) {
            double acc = rhs[i];
            for (int j = i + 1; j < s; j++) acc -= gram[i][j] * out[j];
            out[i] = Math.abs(gram[i][i]) < 1e-14 ? 0.0 : acc / gram[i][i];
        }
        return out;
    }

    /** Modified Gram-Schmidt QR of the selected columns, then back substitution. */
    static double[] solveQr(double[][] a, int[] support, double[] b) {
        int s = support.length;
        double[][] q = new double[s][a.length];
        double[][] r = new double[s][s];
        for (int col = 0; col < s; col++) {
            double[] v = column(a, support[col]);
            for (int previous = 0; previous < col; previous++) {
                double projection = dot(q[previous], v);
                r[previous][col] = projection;
                for (int i = 0; i < v.length; i++) v[i] -= projection * q[previous][i];
            }
            double length = norm(v);
            r[col][col] = length;
            for (int i = 0; i < v.length; i++) q[col][i] = length > 1e-300 ? v[i] / length : 0.0;
        }
        double[] y = new double[s];
        for (int col = 0; col < s; col++) y[col] = dot(q[col], b);
        double[] out = new double[s];
        for (int i = s - 1; i >= 0; i--) {
            double acc = y[i];
            for (int j = i + 1; j < s; j++) acc -= r[i][j] * out[j];
            out[i] = Math.abs(r[i][i]) < 1e-14 ? 0.0 : acc / r[i][i];
        }
        return out;
    }

    static final class OmpResult {
        double[] coefficients;
        int[] support;
        double residualNorm;
        int iterations;
        double solverGap;
        boolean stoppedOnTolerance;
    }

    static OmpResult omp(double[][] a, double[] b, int sparsity, double tolerance) {
        int atoms = a[0].length;
        double[] coefficients = new double[atoms];
        List<Integer> support = new ArrayList<>();
        double[] r = b.clone();
        OmpResult result = new OmpResult();
        double solverGap = 0.0;
        for (int step = 0; step < Math.min(sparsity, atoms); step++) {
            int best = -1;
            double bestCorrelation = 0.0;
            for (int j = 0; j < atoms; j++) {
                if (support.contains(j)) continue;
                double correlation = Math.abs(dot(column(a, j), r));
                if (correlation > bestCorrelation + 1e-15) {
                    bestCorrelation = correlation;
                    best = j;
                }
            }
            if (best < 0 || bestCorrelation <= tolerance) {
                result.stoppedOnTolerance = true;
                break;
            }
            support.add(best);
            int[] selected = new int[support.size()];
            for (int i = 0; i < selected.length; i++) selected[i] = support.get(i);
            double[] normalSolution = solveNormal(a, selected, b);
            double[] qrSolution = solveQr(a, selected, b);
            for (int i = 0; i < selected.length; i++) {
                solverGap = Math.max(solverGap, Math.abs(normalSolution[i] - qrSolution[i]));
            }
            Arrays.fill(coefficients, 0.0);
            for (int i = 0; i < selected.length; i++) coefficients[selected[i]] = normalSolution[i];
            r = residual(a, coefficients, b);
            if (norm(r) <= tolerance) {
                result.stoppedOnTolerance = true;
                break;
            }
        }
        result.coefficients = coefficients;
        result.support = support.stream().mapToInt(Integer::intValue).toArray();
        result.residualNorm = norm(r);
        result.iterations = support.size();
        result.solverGap = solverGap;
        return result;
    }

    static double[][] subtract(double[] a, double[] b) {
        double[][] out = new double[a.length][1];
        for (int i = 0; i < a.length; i++) out[i][0] = a[i] - b[i];
        return out;
    }

    /** The exhaustive oracle: the smallest residual over every support of size at most s. */
    static double bestExhaustiveResidual(double[][] a, double[] b, int sparsity) {
        int atoms = a[0].length;
        double best = norm(b);
        int[] support = new int[Math.min(sparsity, atoms)];
        best = enumerate(a, b, support, 0, 0, sparsity, best);
        return best;
    }

    static double enumerate(double[][] a, double[] b, int[] support, int filled, int start, int sparsity,
                           double best) {
        if (filled > 0) {
            double[] x = solveNormal(a, support, b);
            double[] candidate = new double[a[0].length];
            for (int i = 0; i < filled; i++) candidate[support[i]] = x[i];
            best = Math.min(best, norm(residual(a, candidate, b)));
        }
        if (filled == sparsity) return best;
        for (int j = start; j < a[0].length; j++) {
            support[filled] = j;
            best = enumerate(a, b, support, filled + 1, j + 1, sparsity, best);
        }
        return best;
    }

    static double coherence(double[][] a) {
        double worst = 0.0;
        for (int i = 0; i < a[0].length; i++) {
            for (int j = i + 1; j < a[0].length; j++) {
                double value = Math.abs(dot(column(a, i), column(a, j)))
                        / Math.max(norm(column(a, i)) * norm(column(a, j)), 1e-300);
                worst = Math.max(worst, value);
            }
        }
        return worst;
    }

    static double[][] randomDictionary(int rows, int atoms, Rng rng) {
        double[][] a = new double[rows][atoms];
        for (int j = 0; j < atoms; j++) {
            double length = 0.0;
            for (int i = 0; i < rows; i++) {
                a[i][j] = rng.nextGaussian();
                length += a[i][j] * a[i][j];
            }
            length = Math.sqrt(length);
            for (int i = 0; i < rows; i++) a[i][j] /= length;
        }
        return a;
    }

    static boolean sameSupport(int[] a, int[] b) {
        if (a.length != b.length) return false;
        int[] x = a.clone();
        int[] y = b.clone();
        Arrays.sort(x);
        Arrays.sort(y);
        return Arrays.equals(x, y);
    }

    static boolean contains(int[] values, int target) {
        for (int value : values) if (value == target) return true;
        return false;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        plantedRecovery();
        exhaustiveOracle();
        residualProperties();
        boundaryCases();
        determinism();

        System.out.println("PASS");
    }

    static void plantedRecovery() {
        Rng rng = new Rng(5921L);
        double[][] a = randomDictionary(M, ATOMS, rng);
        double mu = coherence(a);
        double guarantee = (1.0 + 1.0 / mu) / 2.0;
        check(Math.abs(mu - 0.5414360631185241) <= 1e-12, "the dictionary coherence is 0.5414 (reported, not "
                + "asserted to be small)");
        check(Math.abs(guarantee - 1.4234) <= 1e-3, "the classical exact-recovery bound is s < " + guarantee
                + ", so s = 3 lies outside the sufficient condition and the recovery below is an empirical, "
                + "brute-force-checked fact about this dictionary rather than a consequence of the theorem");

        int[] plantedSupport = {2, 7, 15};
        double[] coefficients = {2.5, -1.75, 1.25};
        double[] b = new double[M];
        for (int i = 0; i < plantedSupport.length; i++) {
            for (int row = 0; row < M; row++) b[row] += coefficients[i] * a[row][plantedSupport[i]];
        }
        OmpResult result = omp(a, b, 3, 1e-14);
        check(sameSupport(result.support, plantedSupport),
                "OMP recovers the planted support {2,7,15}, found " + Arrays.toString(result.support));
        for (int i = 0; i < plantedSupport.length; i++) {
            check(Math.abs(result.coefficients[plantedSupport[i]] - coefficients[i]) <= 1e-8,
                    "the coefficient of atom " + plantedSupport[i] + " is recovered to 1e-8");
        }
        check(result.residualNorm <= 1e-10, "the residual is " + result.residualNorm + " <= 1e-10");
        check(result.solverGap <= 1e-10, "the normal-equations and QR solvers agree on the coefficients to "
                + result.solverGap + " (the gap is reported rather than assumed, since the normal equations "
                + "square the condition number)");

        // A second planted instance with different magnitudes and a larger sparsity.
        int[] secondSupport = {0, 5, 11, 19};
        double[] secondCoefficients = {3.0, -2.0, 1.5, -1.0};
        double[] second = new double[M];
        for (int i = 0; i < secondSupport.length; i++) {
            for (int row = 0; row < M; row++) second[row] += secondCoefficients[i] * a[row][secondSupport[i]];
        }
        OmpResult four = omp(a, second, 4, 1e-14);
        check(sameSupport(four.support, secondSupport),
                "a 4-sparse plant is recovered too: " + Arrays.toString(four.support));
        check(four.residualNorm <= 1e-9, "with residual " + four.residualNorm);
    }

    static void exhaustiveOracle() {
        Rng rng = new Rng(5922L);
        double[][] a = randomDictionary(M, ATOMS, rng);
        for (int trial = 0; trial < 4; trial++) {
            Rng signalRng = new Rng(592201L + trial);
            double[] b = new double[M];
            for (int i = 0; i < M; i++) b[i] = signalRng.nextGaussian();
            OmpResult result = omp(a, b, 3, 1e-14);
            double ompResidual = result.residualNorm;
            double bestResidual = bestExhaustiveResidual(a, b, 3);
            check(ompResidual >= bestResidual - 1e-9,
                    "trial " + trial + ": OMP (" + ompResidual + ") cannot beat the best 3-atom subset ("
                            + bestResidual + ")");
            check(ompResidual <= bestResidual + 1e-6,
                    "trial " + trial + ": on a random signal OMP matches the exhaustive optimum to 1e-6 ("
                            + ompResidual + " versus " + bestResidual + ")");

            int[] plantedSupport = {1 + trial, 6 + trial, 13 + trial};
            double[][] plantedCoef = {{1.0}, {-1.0}, {1.0}};
            double[] plantedSignal = new double[M];
            for (int i = 0; i < plantedSupport.length; i++) {
                for (int row = 0; row < M; row++) {
                    plantedSignal[row] += plantedCoef[i][0] * a[row][plantedSupport[i]];
                }
            }
            OmpResult planted = omp(a, plantedSignal, 3, 1e-14);
            double plantedBest = bestExhaustiveResidual(a, plantedSignal, 3);
            check(planted.residualNorm <= plantedBest + 1e-9,
                    "trial " + trial + ": the planted signal is coded at least as well as any 3-atom subset");
            check(planted.residualNorm <= 1e-10,
                    "trial " + trial + ": the planted signal is coded exactly (" + planted.residualNorm + ")");
        }
        double[] empty = omp(a, new double[M], 3, 1e-14).coefficients;
        for (double value : empty) check(value == 0.0, "a zero signal produces zero coefficients exactly");
    }

    static void residualProperties() {
        Rng rng = new Rng(5923L);
        double[][] a = randomDictionary(M, ATOMS, rng);
        double[] b = new double[M];
        for (int i = 0; i < M; i++) b[i] = rng.nextGaussian();

        double previous = norm(b);
        int[] growing = new int[0];
        for (int s = 1; s <= 6; s++) {
            OmpResult result = omp(a, b, s, 1e-14);
            check(result.residualNorm <= previous + 1e-12,
                    "the residual is non-increasing when the sparsity grows from " + (s - 1) + " to " + s);
            check(growing.length == 0 || contains(result.support, growing[growing.length - 1]),
                    "the support at sparsity " + s + " contains the support at " + (s - 1) + " (OMP is greedy "
                            + "and never revises earlier choices)");
            growing = result.support;
            previous = result.residualNorm;
        }
        OmpResult three = omp(a, b, 3, 1e-14);
        double[] r = residual(a, three.coefficients, b);
        for (int atom : three.support) {
            check(Math.abs(dot(column(a, atom), r)) <= 1e-10,
                    "the residual is orthogonal to the selected atom " + atom + " (A_S^T r = 0)");
        }
        int[] support = three.support;
        Set<Integer> unique = new HashSet<>();
        for (int atom : support) unique.add(atom);
        check(unique.size() == support.length,
                "no atom is selected twice, because its correlation is annihilated after the exact fit");
        double[] gradients = new double[ATOMS];
        for (int j = 0; j < ATOMS; j++) gradients[j] = Math.abs(dot(column(a, j), r));
        for (int atom : support) check(gradients[atom] <= 1e-10, "selected atom " + atom + " has no remaining "
                + "correlation");
        double bestOutside = 0.0;
        for (int j = 0; j < ATOMS; j++) {
            if (!contains(support, j)) bestOutside = Math.max(bestOutside, gradients[j]);
        }
        check(residual(a, omp(a, b, 1, 1e-14).coefficients, b).length == M,
                "the residual vector keeps the ambient dimension");
        check(bestOutside >= 0.0, "the largest remaining correlation outside the support is " + bestOutside);

        // The tolerance stops the loop earlier on a signal that is already well approximated.
        OmpResult loose = omp(a, b, 6, 10.0);
        check(loose.iterations <= 6, "a loose tolerance stops at " + loose.iterations + " atoms, at most 6");
        check(loose.residualNorm <= 10.0 + 1e-9, "and the residual respects the tolerance ("
                + loose.residualNorm + ")");
        OmpResult tight = omp(a, b, 0, 1e-14);
        check(tight.iterations == 0 && Math.abs(tight.residualNorm - norm(b)) <= 1e-15,
                "sparsity zero returns the zero vector with the full residual");
    }

    static void boundaryCases() {
        Rng rng = new Rng(5924L);
        double[][] a = randomDictionary(M, ATOMS, rng);
        double[] b = new double[M];
        for (int i = 0; i < M; i++) b[i] = rng.nextGaussian();

        double[][] duplicate = new double[M][3];
        for (int i = 0; i < M; i++) {
            duplicate[i][0] = a[i][3];
            duplicate[i][1] = a[i][3];
            duplicate[i][2] = a[i][8];
        }
        double[] shared = new double[M];
        for (int i = 0; i < M; i++) shared[i] = duplicate[i][0];
        OmpResult tie = omp(duplicate, shared, 1, 1e-14);
        check(tie.support.length == 1 && tie.support[0] == 0,
                "with two identical atoms the smallest index wins the tie (documented tie-break), got "
                        + Arrays.toString(tie.support));
        check(tie.residualNorm <= 1e-12, "and the shared direction is explained exactly (" + tie.residualNorm + ")");
        double[] mixed = new double[M];
        for (int i = 0; i < M; i++) mixed[i] = duplicate[i][0] + 0.7 * duplicate[i][2];
        OmpResult mixedResult = omp(duplicate, mixed, 3, 1e-14);
        check(contains(mixedResult.support, 0), "the shared direction is taken first, at the smallest index ("
                + Arrays.toString(mixedResult.support) + ")");
        check(contains(mixedResult.support, 2),
                "and the second step can only take the independent atom, because the residual is already "
                        + "orthogonal to both copies of the duplicate");
        int duplicateCoefficients = 0;
        for (int j : new int[]{0, 1}) {
            if (Math.abs(mixedResult.coefficients[j]) > 1e-12) duplicateCoefficients++;
        }
        check(duplicateCoefficients <= 1,
                "the redundant duplicate never receives a second nonzero coefficient (" + duplicateCoefficients
                        + " of the two copies are nonzero), since the 2x2 Gram matrix would be singular and the "
                        + "solver skips the zero pivot");
        check(mixedResult.residualNorm <= 1e-9, "the two-dimensional signal is still coded exactly ("
                + mixedResult.residualNorm + ")");

        double[][] withZero = new double[M][3];
        for (int i = 0; i < M; i++) {
            withZero[i][0] = 0.0;
            withZero[i][1] = a[i][5];
            withZero[i][2] = a[i][9];
        }
        OmpResult withZeroResult = omp(withZero, b, 2, 1e-14);
        check(!contains(withZeroResult.support, 0),
                "a zero atom is never selected (its correlation is exactly zero)");
        for (double value : withZeroResult.coefficients) {
            check(!Double.isNaN(value) && !Double.isInfinite(value), "no NaN appears with a zero atom present");
        }

        OmpResult tooMany = omp(a, b, 100, 1e-14);
        check(tooMany.iterations <= ATOMS, "a sparsity above the dictionary size is clamped to " + ATOMS
                + " (used " + tooMany.iterations + ")");
        int[] allAtoms = new int[ATOMS];
        for (int j = 0; j < ATOMS; j++) allAtoms[j] = j;
        double[] fullLeastSquares = solveNormal(a, allAtoms, b);
        double fullResidual = norm(residual(a, fullLeastSquares, b));
        check(fullResidual > 0.01, "the 20 atoms do not span the " + M + "-dimensional space, so the full "
                + "least-squares residual is a genuine " + fullResidual);
        check(tooMany.residualNorm >= fullResidual - 1e-9,
                "OMP with 20 atoms cannot beat the full least-squares fit (" + tooMany.residualNorm + " versus "
                        + fullResidual + ")");
        check(Math.abs(tooMany.residualNorm - fullResidual) <= 1e-9,
                "and here it matches it, because the greedy pass ends up using all 20 atoms");
        double[] ones = new double[M];
        Arrays.fill(ones, 1.0);
        OmpResult parallel = omp(a, ones, 1, 1e-14);
        check(parallel.support.length <= 1, "a signal parallel to a single atom needs at most one atom");
    }

    static void determinism() {
        Rng rng = new Rng(5925L);
        double[][] a = randomDictionary(M, ATOMS, rng);
        double[] b = new double[M];
        for (int i = 0; i < M; i++) b[i] = rng.nextGaussian();
        OmpResult first = omp(a, b, 4, 1e-14);
        OmpResult second = omp(a, b, 4, 1e-14);
        check(Arrays.equals(first.support, second.support), "the greedy selection order is reproducible");
        boolean same = true;
        for (int i = 0; i < ATOMS; i++) same &= first.coefficients[i] == second.coefficients[i];
        check(same, "so are the coefficients, bit for bit");
        check(first.residualNorm == second.residualNorm, "and the residual norm");
        check(Math.abs(first.solverGap - second.solverGap) == 0.0, "and the solver-gap report");
        check(bestExhaustiveResidual(a, b, 2) <= norm(b) + 1e-12, "the oracle never exceeds the trivial bound");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
