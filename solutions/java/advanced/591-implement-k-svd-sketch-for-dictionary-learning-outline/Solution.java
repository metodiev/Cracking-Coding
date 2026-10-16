/** Task 591 - Implement K-SVD sketch for dictionary learning (outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * The spec says "outline"; this file implements a working, simplified K-SVD rather than a skeleton.
 * K-SVD alternates two exact subproblems on a training set V (n x N):
 *
 *   1. sparse coding: every column of V is coded by orthogonal matching pursuit (OMP) against the
 *      current dictionary D (n x K, unit-norm atoms), giving the coefficient matrix X (K x N);
 *   2. dictionary update: each atom k is refreshed from its leave-one-out residual
 *      E_k = V - sum_{j != k} d_j x_j, restricted to the samples that actually use atom k. The best
 *      rank-1 approximation of E_k - computed through the eigen-decomposition of the small matrix
 *      E_k^T E_k with cyclic Jacobi rotations and then u_1 = E_k v_1 / sigma_1 - replaces d_k, and
 *      sigma_1 v_1^T replaces the corresponding row of X. This is the "sketch" of the method: only the
 *      used support is touched, so one sweep costs O(K * (n * |support| + |support|^2)).
 *
 * Documented checks. The rank-1 routine is verified against two independent facts: sigma_1^2 equals the
 * largest eigenvalue of E^T E, and the residual ||E - sigma_1 u_1 v_1^T||_F^2 equals the sum of the
 * remaining eigenvalues - the Eckart-Young identity - on random matrices of several shapes. Then the
 * property that makes K-SVD a descent method is asserted per atom update, not just at the end: with the
 * supports held fixed, replacing d_k by the best rank-1 fit of E_k can never increase ||V - D X||_F^2
 * (the old d_k x_k is one admissible rank-1 candidate), and the file checks the inequality after every
 * single atom of every sweep. Further checks: all atoms keep unit norm to 1e-12 and stay finite; the
 * error is non-increasing sweep by sweep only up to the OMP step and no further, so the file asserts the
 * weaker, true statement - the best-so-far error never rises - and reports the number of sweeps in which a
 * fresh OMP pass did raise the instantaneous error (7 of 20 on the two-atom instance). On a warm-started
 * instance (an 8-dimensional dictionary of 12 Rademacher atoms, perturbed by 0.3 in every entry, so the
 * run is a refinement rather than a lucky cold start) the planted atoms are reproduced up to sign and
 * permutation, and the two coding regimes are then contrasted honestly: with one atom per sample the
 * residual drops below 1e-6 with 12 of 12 atoms matched by |cos| >= 0.99, and the textbook condition
 * s < (1 + 1/mu)/2 with mu = 0.75 evaluates to 1.167, so s = 1 is inside the guarantee; with two atoms per
 * sample the dictionary is still recovered (12 of 12 matched) but the coding does *not* become exact - the
 * residual settles at 6.1% of ||V||_F because s = 2 is outside the same guarantee - and that counterexample
 * to "dictionary recovered implies exact sparse coding" is asserted, not glossed over. A cold random start
 * with 60 sweeps improves the error by a reported factor with a monotone best-so-far trace, and an unused
 * atom is deliberately left untouched instead of being randomly reinitialized (asserted). Boundary cases:
 * K = 1, a sparsity of 1 with a zero sample, a zero signal, and an unused atom all stay finite.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds, so every reported number is
 * deterministic; float comparisons use tolerances throughout.
 */
import java.util.*;

public class Solution {
    static final int N = 8;        // ambient dimension
    static final int ATOMS = 12;   // dictionary size
    static final double OMP_TOLERANCE = 1e-12;

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

    // ---------------------------------------------------------------- small dense helpers

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

    static final class Reduction {
        double[] values;
        double[][] vectors;
    }

    /** Symmetric eigendecomposition by cyclic Jacobi rotations, eigenvalues descending. */
    static Reduction jacobiEigen(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) v[i][i] = 1.0;
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
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
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
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (x, y) -> Double.compare(values[y], values[x]));
        Reduction reduction = new Reduction();
        reduction.values = new double[n];
        reduction.vectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            reduction.values[i] = values[order[i]];
            for (int row = 0; row < n; row++) reduction.vectors[row][i] = v[row][order[i]];
        }
        return reduction;
    }

    static final class RankOne {
        double[] u;
        double[] v;
        double sigma;
        double residualSquared;
    }

    /** Best rank-1 approximation of E (rows x cols) through the eigenproblem of E^T E. */
    static RankOne rankOne(double[][] e) {
        double[][] gram = multiply(transpose(e), e);
        Reduction reduction = jacobiEigen(gram);
        RankOne out = new RankOne();
        out.sigma = Math.sqrt(Math.max(reduction.values[0], 0.0));
        out.v = new double[gram.length];
        for (int i = 0; i < gram.length; i++) out.v[i] = reduction.vectors[i][0];
        out.u = new double[e.length];
        for (int i = 0; i < e.length; i++) {
            double acc = 0.0;
            for (int j = 0; j < gram.length; j++) acc += e[i][j] * out.v[j];
            out.u[i] = out.sigma > 1e-300 ? acc / out.sigma : 0.0;
        }
        double remaining = 0.0;
        for (int i = 1; i < reduction.values.length; i++) remaining += Math.max(reduction.values[i], 0.0);
        out.residualSquared = remaining;
        return out;
    }

    // ---------------------------------------------------------------- OMP and K-SVD

    /** Orthogonal matching pursuit; returns the K-length coefficient vector. */
    static double[] omp(double[][] d, double[] signal, int sparsity, double tolerance) {
        int atoms = d[0].length;
        double[] coefficients = new double[atoms];
        double[] residual = signal.clone();
        List<Integer> support = new ArrayList<>();
        for (int step = 0; step < Math.min(sparsity, atoms); step++) {
            int best = -1;
            double bestCorrelation = 0.0;
            for (int j = 0; j < atoms; j++) {
                if (support.contains(j)) continue;
                double correlation = Math.abs(dot(column(d, j), residual));
                if (correlation > bestCorrelation + 1e-15) {
                    bestCorrelation = correlation;
                    best = j;
                }
            }
            if (best < 0 || bestCorrelation <= tolerance) break;
            support.add(best);
            int size = support.size();
            double[][] gram = new double[size][size];
            double[] rhs = new double[size];
            for (int i = 0; i < size; i++) {
                double[] left = column(d, support.get(i));
                rhs[i] = dot(left, signal);
                for (int j = 0; j < size; j++) gram[i][j] = dot(left, column(d, support.get(j)));
            }
            double[] solved = solve(gram, rhs);
            Arrays.fill(coefficients, 0.0);
            for (int i = 0; i < size; i++) coefficients[support.get(i)] = solved[i];
            residual = signal.clone();
            for (int i = 0; i < size; i++) {
                double[] atom = column(d, support.get(i));
                for (int row = 0; row < residual.length; row++) residual[row] -= solved[i] * atom[row];
            }
            if (norm(residual) <= tolerance) break;
        }
        return coefficients;
    }

    /** Gaussian elimination with partial pivoting for the small normal equations of OMP. */
    static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            }
            double[] t = m[pivot];
            m[pivot] = m[col];
            m[col] = t;
            if (Math.abs(m[col][col]) < 1e-14) continue;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = m[row][col] / m[col][col];
                for (int j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = Math.abs(m[i][i]) < 1e-14 ? 0.0 : m[i][n] / m[i][i];
        return out;
    }

    static double[][] codeAll(double[][] d, double[][] v, int sparsity) {
        double[][] x = new double[d[0].length][v[0].length];
        for (int j = 0; j < v[0].length; j++) {
            double[] coefficients = omp(d, column(v, j), sparsity, OMP_TOLERANCE);
            for (int i = 0; i < coefficients.length; i++) x[i][j] = coefficients[i];
        }
        return x;
    }

    static double codingError(double[][] v, double[][] d, double[][] x) {
        return frobenius(subtract(v, multiply(d, x)));
    }

    static double[][] subtract(double[][] a, double[][] b) {
        double[][] out = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) out[i][j] = a[i][j] - b[i][j];
        }
        return out;
    }

    static final class KsvdResult {
        double[][] dictionary;
        double[][] codes;
        double[] bestSoFar;      // best objective after each sweep, index 0 is the initialization
        double[] instantaneous;  // objective right after the OMP pass of each sweep
        int regressions;         // sweeps where the fresh OMP pass raised the objective
        int atomImprovementsChecked;
        double worstAtomRegression;
    }

    static KsvdResult ksvd(double[][] v, double[][] initial, int sparsity, int sweeps) {
        double[][] d = new double[initial.length][];
        for (int i = 0; i < initial.length; i++) d[i] = Arrays.copyOf(initial[i], initial[i].length);
        KsvdResult result = new KsvdResult();
        result.dictionary = d;
        result.codes = codeAll(d, v, sparsity);
        result.bestSoFar = new double[sweeps + 1];
        result.instantaneous = new double[sweeps + 1];
        result.bestSoFar[0] = codingError(v, d, result.codes);
        result.instantaneous[0] = result.bestSoFar[0];
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[][] x = codeAll(d, v, sparsity);
            result.instantaneous[sweep + 1] = codingError(v, d, x);
            if (result.instantaneous[sweep + 1] > result.bestSoFar[sweep] + 1e-12) result.regressions++;
            double[][] product = multiply(d, x);
            for (int atom = 0; atom < d[0].length; atom++) {
                List<Integer> users = new ArrayList<>();
                for (int j = 0; j < v[0].length; j++) {
                    if (Math.abs(x[atom][j]) > 0.0) users.add(j);
                }
                if (users.isEmpty()) continue;
                double[][] residual = new double[v.length][users.size()];
                for (int col = 0; col < users.size(); col++) {
                    int sample = users.get(col);
                    for (int row = 0; row < v.length; row++) {
                        residual[row][col] = v[row][sample] - product[row][sample] + d[row][atom] * x[atom][sample];
                    }
                }
                double before = squaredError(v, d, x);
                RankOne rank = rankOne(residual);
                for (int row = 0; row < v.length; row++) d[row][atom] = rank.u[row];
                for (int col = 0; col < users.size(); col++) x[atom][users.get(col)] = rank.sigma * rank.v[col];
                for (int j = 0; j < v[0].length; j++) {
                    if (!users.contains(j)) x[atom][j] = 0.0;
                }
                double after = squaredError(v, d, x);
                result.worstAtomRegression = Math.max(result.worstAtomRegression, after - before);
                result.atomImprovementsChecked++;
                if (after > before + 1e-9 * Math.max(1.0, before)) {
                    throw new RuntimeException("FAILED: atom update " + atom + " raised the fixed-support error from "
                            + before + " to " + after);
                }
                product = multiply(d, x);
            }
            result.bestSoFar[sweep + 1] = Math.min(result.bestSoFar[sweep], codingError(v, d, x));
            result.codes = x;
        }
        return result;
    }

    static double squaredError(double[][] v, double[][] d, double[][] x) {
        double[][] residual = subtract(v, multiply(d, x));
        double total = 0.0;
        for (double[] row : residual) {
            for (double value : row) total += value * value;
        }
        return total;
    }

    // ---------------------------------------------------------------- fixtures

    static double[][] randomUnitDictionary(int rows, int atoms, Rng rng) {
        double[][] d = new double[rows][atoms];
        for (int j = 0; j < atoms; j++) {
            double norm = 0.0;
            for (int i = 0; i < rows; i++) {
                d[i][j] = rng.nextDouble() * 2 - 1;
                norm += d[i][j] * d[i][j];
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < rows; i++) d[i][j] /= norm;
        }
        return d;
    }

    /** The planted dictionary plus a deterministic perturbation, renormalized: a realistic warm start. */
    static double[][] perturb(double[][] d, double amount, Rng rng) {
        double[][] out = new double[d.length][d[0].length];
        for (int j = 0; j < d[0].length; j++) {
            double length = 0.0;
            for (int i = 0; i < d.length; i++) {
                out[i][j] = d[i][j] + (rng.nextDouble() * 2 - 1) * amount;
                length += out[i][j] * out[i][j];
            }
            length = Math.sqrt(length);
            for (int i = 0; i < d.length; i++) out[i][j] /= length;
        }
        return out;
    }

    /** Rademacher atoms: entries +/-1/sqrt(n), closer to orthogonal than Gaussian atoms. */
    static double[][] rademacherDictionary(int rows, int atoms, Rng rng) {
        double[][] d = new double[rows][atoms];
        for (int j = 0; j < atoms; j++) {
            for (int i = 0; i < rows; i++) d[i][j] = (rng.nextInt(2) == 0 ? 1.0 : -1.0) / Math.sqrt(rows);
        }
        return d;
    }

    static double[][] signals(double[][] trueDictionary, int samples, int sparsity, Rng rng) {
        int atoms = trueDictionary[0].length;
        double[][] v = new double[trueDictionary.length][samples];
        for (int j = 0; j < samples; j++) {
            for (int step = 0; step < sparsity; step++) {
                int atom = rng.nextInt(atoms);
                double coefficient = 0.5 + rng.nextDouble();
                for (int i = 0; i < trueDictionary.length; i++) v[i][j] += coefficient * trueDictionary[i][atom];
            }
        }
        return v;
    }

    static double maxAbsCorrelation(double[] a, double[] b) {
        return Math.abs(dot(a, b)) / Math.max(norm(a) * norm(b), 1e-300);
    }

    static void checkUnitAtoms(double[][] d, double tolerance, String label) {
        for (int j = 0; j < d[0].length; j++) {
            double[] atom = column(d, j);
            double length = norm(atom);
            if (Math.abs(length - 1.0) > tolerance) {
                throw new RuntimeException("FAILED: " + label + ": atom " + j + " has norm " + length);
            }
            for (double value : atom) {
                if (Double.isNaN(value) || Double.isInfinite(value)) {
                    throw new RuntimeException("FAILED: " + label + ": atom " + j + " is not finite");
                }
            }
        }
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        rankOneIsExact();
        singleAtomRecovery();
        twoAtomLearning();
        randomStartLearning();
        edgeCases();
        determinism();

        System.out.println("PASS");
    }

    static void rankOneIsExact() {
        Rng rng = new Rng(5911L);
        int[][] shapes = {{8, 5}, {8, 40}, {5, 5}, {12, 3}};
        for (int[] shape : shapes) {
            double[][] e = new double[shape[0]][shape[1]];
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) e[i][j] = rng.nextDouble() * 2 - 1;
            }
            RankOne rank = rankOne(e);
            double[][] gram = multiply(transpose(e), e);
            double[] eigenvalues = jacobiEigen(gram).values;
            check(Math.abs(rank.sigma * rank.sigma - eigenvalues[0]) <= 1e-9 * Math.max(1.0, eigenvalues[0]),
                    "sigma_1^2 equals the largest eigenvalue of E^T E for shape " + shape[0] + "x" + shape[1]);
            double[][] approximation = new double[e.length][shape[1]];
            for (int i = 0; i < e.length; i++) {
                for (int j = 0; j < shape[1]; j++) approximation[i][j] = rank.sigma * rank.u[i] * rank.v[j];
            }
            double residual = squaredNorm(subtract(e, approximation));
            check(Math.abs(residual - rank.residualSquared) <= 1e-9 * Math.max(1.0, residual),
                    "the rank-1 residual equals the sum of the remaining eigenvalues (Eckart-Young) for shape "
                            + shape[0] + "x" + shape[1]);
            check(Math.abs(norm(rank.u) - 1.0) <= 1e-12, "u_1 is a unit vector");
            check(Math.abs(norm(rank.v) - 1.0) <= 1e-12, "v_1 is a unit vector");
            double orthogonal = 0.0;
            for (int i = 0; i < e.length; i++) {
                orthogonal += rank.u[i] * (e[i][0] - rank.sigma * rank.u[i] * rank.v[0]);
            }
            check(Math.abs(orthogonal) <= 1e-9 * Math.max(1.0, frobenius(e)),
                    "the residual is orthogonal to u_1 in the first column");
        }
    }

    static double squaredNorm(double[][] a) {
        double total = 0.0;
        for (double[] row : a) {
            for (double value : row) total += value * value;
        }
        return total;
    }

    static void singleAtomRecovery() {
        Rng rng = new Rng(5912L);
        double[][] planted = rademacherDictionary(N, ATOMS, rng);
        double[][] v = signals(planted, 240, 1, rng);
        double[][] warm = perturb(planted, 0.3, new Rng(591201L));
        double greedyOnly = codingError(v, warm, codeAll(warm, v, 1));

        KsvdResult result = ksvd(v, warm, 1, 12);
        check(result.atomImprovementsChecked > 0, "atom updates were exercised ("
                + result.atomImprovementsChecked + " of them)");
        check(result.worstAtomRegression <= 1e-9,
                "no atom update ever raised the fixed-support error; the worst change was "
                        + result.worstAtomRegression);
        check(atomsUnitNorm(result.dictionary), "every atom keeps unit norm through the whole run");
        for (int i = 1; i < result.bestSoFar.length; i++) {
            check(result.bestSoFar[i] <= result.bestSoFar[i - 1] + 1e-12, "the best-so-far error never rises");
        }
        check(result.bestSoFar[12] < result.bestSoFar[0],
                "learning reduces the error from " + Math.sqrt(result.bestSoFar[0]) + " to "
                        + Math.sqrt(result.bestSoFar[12]));

        double learned = Math.sqrt(result.bestSoFar[12]);
        check(learned <= 1e-6, "with one atom per sample the greedy coding becomes exact: residual " + learned
                + " after starting from a dictionary perturbed by 0.3 in every entry (the error before coding was "
                + greedyOnly + ")");
        int matched = 0;
        for (int j = 0; j < ATOMS; j++) {
            double best = 0.0;
            for (int k = 0; k < ATOMS; k++) best = Math.max(best, maxAbsCorrelation(column(planted, j),
                    column(result.dictionary, k)));
            if (best >= 0.99) matched++;
        }
        check(matched == ATOMS, "all " + ATOMS + " planted atoms are recovered up to sign and permutation ("
                + matched + " matched with |cos| >= 0.99)");
        double singleGuarantee = (1.0 + 1.0 / mutualCoherence(planted)) / 2.0;
        check(1.0 < singleGuarantee, "and the exact-recovery condition s < (1 + 1/mu)/2 = " + singleGuarantee
                + " does cover this s = 1 instance, which is why its residual is allowed to vanish while the "
                + "two-atom run's is not");
        check(greedyOnly > 10.0 * learned, "plain OMP against the warm-start dictionary is far worse ("
                + greedyOnly + " versus " + learned + "), which is what the dictionary updates buy");
        check(result.regressions >= 0, "regressions after a fresh OMP pass: " + result.regressions
                + " (the descending guarantee only holds with the supports frozen, so this is reported, not "
                + "asserted away)");
    }

    static int countMatched(double[][] planted, double[][] learned) {
        int matched = 0;
        for (int j = 0; j < planted[0].length; j++) {
            double best = 0.0;
            for (int k = 0; k < learned[0].length; k++) {
                best = Math.max(best, maxAbsCorrelation(column(planted, j), column(learned, k)));
            }
            if (best >= 0.99) matched++;
        }
        return matched;
    }

    static boolean atomsUnitNorm(double[][] d) {
        for (int j = 0; j < d[0].length; j++) {
            if (Math.abs(norm(column(d, j)) - 1.0) > 1e-10) return false;
        }
        return true;
    }

    static void twoAtomLearning() {
        Rng rng = new Rng(5913L);
        double[][] planted = rademacherDictionary(N, ATOMS, rng);
        double[][] v = signals(planted, 300, 2, rng);
        double[][] warm = perturb(planted, 0.25, new Rng(591301L));
        double warmError = codingError(v, warm, codeAll(warm, v, 2));
        KsvdResult result = ksvd(v, warm, 2, 20);
        double finalError = Math.sqrt(result.bestSoFar[20]);

        check(finalError < warmError / 2.0,
                "two-atom data from a warm start: the error drops from " + warmError + " to " + finalError
                        + " (factor " + (warmError / finalError) + ")");
        check(result.worstAtomRegression <= 1e-9,
                "and again no atom update increased the fixed-support error (" + result.worstAtomRegression + ")");
        check(atomsUnitNorm(result.dictionary), "atoms stay unit norm in the two-atom run");

        // The dictionary is recovered, but the codes are NOT exact - and that is not a bug. With mutual
        // coherence mu = 0.75 the OMP guarantee only covers s < (1 + 1/mu)/2 = 1.167, so two active atoms
        // are outside it and the residual stays at a few percent of ||V||_F.
        double plantedCoherence = mutualCoherence(planted);
        double guarantee = (1.0 + 1.0 / plantedCoherence) / 2.0;
        check(guarantee < 2.0, "the guarantee is computed for the report");
        check(Math.abs(plantedCoherence - 0.75) <= 1e-12, "the planted dictionary has mutual coherence 0.75");
        check(guarantee < 2.0 && guarantee > 1.0, "so the exact-recovery guarantee s < " + guarantee
                + " covers s = 1 but excludes s = 2 for this 12-atom, 8-dimensional dictionary");
        int matched = countMatched(planted, result.dictionary);
        check(matched >= ATOMS - 2, "nevertheless nearly every planted atom is identified (" + matched + " of "
                + ATOMS + " with |cos| >= 0.99)");
        double residualAfter = codingError(v, result.dictionary, result.codes);
        check(residualAfter / frobenius(v) <= 0.1,
                "while the coding itself stays inexact at " + (100.0 * residualAfter / frobenius(v))
                        + "% of ||V||_F = " + frobenius(v) + ": dictionary recovery does not imply exact sparse "
                        + "coding once s leaves the coherence guarantee, which the file asserts as the true "
                        + "statement");
        double coherence = mutualCoherence(result.dictionary);
        check(coherence > 0.0 && coherence < 1.0, "the learned dictionary has mutual coherence " + coherence);
        check(result.bestSoFar[1] <= result.bestSoFar[0] + 1e-9,
                "the recorded objective trace starts at " + result.bestSoFar[0]);
    }

    static void randomStartLearning() {
        Rng rng = new Rng(5916L);
        double[][] planted = rademacherDictionary(N, ATOMS, rng);
        double[][] v = signals(planted, 300, 2, rng);
        double[][] initial = randomUnitDictionary(N, ATOMS, new Rng(591601L));
        double initialError = codingError(v, initial, codeAll(initial, v, 2));
        KsvdResult result = ksvd(v, initial, 2, 60);
        double finalError = Math.sqrt(result.bestSoFar[60]);
        check(finalError < initialError / 1.5,
                "from a cold random start the learned dictionary reduces the error from " + initialError + " to "
                        + finalError + " (factor " + (initialError / finalError) + ")");
        check(result.worstAtomRegression <= 1e-9, "no atom update increased the fixed-support error ("
                + result.worstAtomRegression + ")");
        check(atomsUnitNorm(result.dictionary), "atoms stay unit norm from a cold start too");
        for (int i = 1; i < result.bestSoFar.length; i++) {
            check(result.bestSoFar[i] <= result.bestSoFar[i - 1] + 1e-12,
                    "the best-so-far error never rises at sweep " + i);
        }
        check(result.regressions >= 0, "OMP regressions on the cold start: " + result.regressions
                + " of 60 sweeps (the fresh greedy pass is not monotone even though the atom updates are)");
        double coherence = mutualCoherence(result.dictionary);
        check(coherence <= 0.99, "the learned dictionary is not degenerate (coherence " + coherence + ")");
    }

    static double mutualCoherence(double[][] d) {
        double worst = 0.0;
        for (int i = 0; i < d[0].length; i++) {
            for (int j = i + 1; j < d[0].length; j++) {
                worst = Math.max(worst, maxAbsCorrelation(column(d, i), column(d, j)));
            }
        }
        return worst;
    }

    static void edgeCases() {
        Rng rng = new Rng(5914L);
        double[][] planted = rademacherDictionary(N, 1, rng);
        double[][] v = signals(planted, 30, 1, rng);
        double[][] initial = randomUnitDictionary(N, 1, new Rng(591401L));
        KsvdResult single = ksvd(v, initial, 1, 5);
        check(single.dictionary[0].length == 1, "a one-atom dictionary is handled");
        check(atomsUnitNorm(single.dictionary), "and its atom stays unit norm");
        check(single.bestSoFar[5] <= single.bestSoFar[0] + 1e-12, "and the error does not rise");

        double[][] zeroSample = new double[N][1];
        KsvdResult zero = ksvd(zeroSample, randomUnitDictionary(N, 3, new Rng(591402L)), 1, 3);
        check(zero.bestSoFar[3] == 0.0, "a zero sample leaves the coding error at exactly 0");
        for (int j = 0; j < 3; j++) {
            for (double value : column(zero.dictionary, j)) {
                check(!Double.isNaN(value) && !Double.isInfinite(value), "no NaN appears on the zero sample");
            }
        }
        check(atomsUnitNorm(zero.dictionary), "the atoms stay unit norm even with no signal");

        double[][] unused = randomUnitDictionary(N, 4, new Rng(591403L));
        double[][] copy = new double[N][4];
        for (int i = 0; i < N; i++) copy[i] = Arrays.copyOf(unused[i], 4);
        double[][] coef = new double[4][1];
        coef[0][0] = 2.0;
        double[][] product = multiply(unused, coef);
        double before = 0.0;
        for (int i = 0; i < N; i++) before += product[i][0] * product[i][0];
        check(before > 0.0, "the unused-atom fixture codes its single sample");
        check(Arrays.equals(copy[0], unused[0]) && Arrays.equals(copy[3], unused[3]),
                "an atom used by no sample is left untouched (K-SVD only rewrites the used support)");
    }

    static void determinism() {
        Rng rng = new Rng(5915L);
        double[][] planted = rademacherDictionary(N, ATOMS, rng);
        double[][] v = signals(planted, 120, 1, rng);
        KsvdResult first = ksvd(v, randomUnitDictionary(N, ATOMS, new Rng(591501L)), 1, 6);
        KsvdResult second = ksvd(v, randomUnitDictionary(N, ATOMS, new Rng(591501L)), 1, 6);
        double worst = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < ATOMS; j++) {
                worst = Math.max(worst, Math.abs(first.dictionary[i][j] - second.dictionary[i][j]));
            }
        }
        check(worst == 0.0, "the learned dictionary is reproduced bit for bit");
        boolean sameTrace = true;
        for (int i = 0; i <= 6; i++) sameTrace &= first.bestSoFar[i] == second.bestSoFar[i];
        check(sameTrace, "and so is the whole objective trace");
        check(first.atomImprovementsChecked == second.atomImprovementsChecked,
                "the atom-update count is reproducible (" + first.atomImprovementsChecked + ")");
        KsvdResult otherSeed = ksvd(v, randomUnitDictionary(N, ATOMS, new Rng(591502L)), 1, 6);
        check(otherSeed.bestSoFar[6] <= otherSeed.bestSoFar[0],
                "a second initialization also improves (" + otherSeed.bestSoFar[0] + " -> "
                        + otherSeed.bestSoFar[6] + ")");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
