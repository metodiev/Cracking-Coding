/** Task 570 - Implement linear programming simplex pivot demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * A complete two-phase primal simplex for max c^T x subject to A x <= b, x >= 0, where b may have
 * negative entries: such a row is negated (leaving a surplus variable), gets an artificial, and
 * phase I maximises the negative artificial sum until no positive reduced cost remains. If that sum
 * is positive the routine returns INFEASIBLE. Before phase II every artificial is driven out of the
 * basis by pivoting on any usable entry of its row (a row without one is redundant, is zeroed and
 * ignored) — necessary, because leaving an artificial basic at zero lets a phase-II pivot push it
 * positive, which would silently violate the negated constraint. Phase II recomputes the reduced
 * costs c_j - z_j, keeps artificial columns out of the basis, and reports UNBOUNDED as soon as the
 * entering column has no positive ratio-test entry. Both pivot rules are implemented: Bland's rule
 * (smallest index enters; among rows attaining the minimum ratio the smallest basic variable leaves),
 * which guarantees termination, and Dantzig's rule (most positive reduced cost enters, first row
 * attaining the minimum ratio leaves). The pivot budget is a parameter, so the self-test can observe
 * cycling.
 *
 * Documented checks. (1) Textbook instance max 3x1 + 5x2 s.t. x1 <= 4, 2x2 <= 12, 3x1 + 2x2 <= 18:
 * both rules return exactly 36 at (2, 6). A second hand instance max 2x1 + x2 s.t. x1 + x2 <= 4,
 * x1 <= 3, x2 <= 3 has optimum 7 at (3, 1). (2) Independent ground truth: every optimum is compared
 * with the maximum over the LP's vertices, enumerated by choosing m of the n + m columns of [A | I],
 * solving the m x m subsystem by Gauss-Jordan elimination and keeping the non-negative solutions
 * (at most C(n+m, m) = 126 candidates); the two methods agree to 1e-6 on all 200 deterministic
 * random instances (n <= 4, m <= 5, coefficients in -2..3, right-hand sides in -3..6), and each
 * returned x is separately checked against A x <= b and x >= 0. (3) Feasibility and unboundedness:
 * the random family came out 79 optimal, 48 unbounded and 73 infeasible; every unbounded instance
 * was certified by an explicit integer direction d in [0, 3]^n with d >= 0, A d <= 0 and c^T d > 0
 * (a genuine certificate, since a feasible x + t d stays feasible with objective growing by
 * t c^T d), and every infeasible instance was confirmed by the vertex enumeration finding no
 * feasible basic solution. Hand cases cover both: x1 + x2 <= 1 with x1 + x2 >= 3 is infeasible, and
 * max x1 + x2 s.t. -x1 + x2 <= 1, x1 - x2 <= 1 is unbounded. (4) Strong duality: the dual
 * min b^T y s.t. A^T y >= c, y >= 0 is solved with the same routine by negating constraints and
 * objective; the primal and dual optima are asserted equal on all 79 optimal instances, together
 * with complementary slackness y_i (b - A x)_i = 0 and x_j (A^T y - c)_j = 0 checked entry by entry
 * and each slack term asserted non-negative. (5) Beale's classical cycling example: Bland's rule
 * terminates after 6 pivots with the value 0.05 known from the literature (also matching the vertex
 * enumeration), while Dantzig's rule with the smallest-index tie-break runs into the 500-pivot cap,
 * i.e. really cycles — asserted, not assumed. (6) Degeneracy (two identical rows, so the vertex has
 * more active rows than variables), a zero row, and a redundant row are all solved correctly. The
 * average pivot counts over the random family are printed (Bland 1.065, Dantzig 1.045): on these
 * tiny instances the rules are nearly tied, which is exactly why the Beale counterexample is the
 * meaningful comparison rather than the random average.
 *
 * All randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many
 * independent integers, so the repo's 16-bit LCG is not used); every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final String OPTIMAL = "OPTIMAL";
    static final String UNBOUNDED = "UNBOUNDED";
    static final String INFEASIBLE = "INFEASIBLE";
    static final String ITERATION_LIMIT = "ITERATION_LIMIT";
    static final double EPS = 1e-9;

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

    static final class Result {
        final String status;
        final double value;
        final double[] x;
        final int iterations;

        Result(String status, double value, double[] x, int iterations) {
            this.status = status;
            this.value = value;
            this.x = x;
            this.iterations = iterations;
        }
    }

    /**
     * Two-phase simplex for max c^T x s.t. A x <= b, x >= 0. Rows with b_i < 0 are negated (giving a
     * surplus variable) and receive an artificial; phase I maximises the negative artificial sum.
     * {@code bland} selects Bland's pivoting rule, otherwise Dantzig's rule. The returned status is
     * OPTIMAL, UNBOUNDED, INFEASIBLE or ITERATION_LIMIT (which is what a cycling pivot rule reaches).
     */
    static Result simplex(double[][] A, double[] b, double[] c, boolean bland, int maxIterations) {
        int m = A.length;
        int n = c.length;
        int artificialCount = 0;
        for (int i = 0; i < m; i++) if (b[i] < 0.0) artificialCount++;
        int artificialStart = n + m;
        int cols = n + m + artificialCount;
        int rhs = cols;
        double[][] T = new double[m + 1][cols + 1];
        int[] basis = new int[m];
        int nextArtificial = artificialStart;
        for (int i = 0; i < m; i++) {
            double sign = b[i] < 0.0 ? -1.0 : 1.0;
            for (int j = 0; j < n; j++) T[i][j] = sign * A[i][j];
            T[i][n + i] = sign;
            T[i][rhs] = sign * b[i];
            if (b[i] < 0.0) {
                T[i][nextArtificial] = 1.0;
                basis[i] = nextArtificial;
                nextArtificial++;
            } else {
                basis[i] = n + i;
            }
        }
        boolean phaseOne = true;
        // Reduced costs of the phase-I objective f = -sum(artificials). The objective row stores the
        // reduced costs c_j - z_j together with the negated objective (-f) in its right-hand side, so
        // obj[j] = f_j - sum_i f_{b(i)} T[i][j] with f_j = -1 exactly for the artificial columns.
        for (int j = 0; j <= rhs; j++) {
            double sum = 0.0;
            for (int i = 0; i < m; i++) if (basis[i] >= artificialStart) sum += T[i][j];
            if (j == rhs) {
                T[m][j] = sum;
            } else {
                T[m][j] = (j >= artificialStart ? -1.0 : 0.0) + sum;
            }
        }
        int iterations = 0;
        while (true) {
            int entering = -1;
            if (bland) {
                for (int j = 0; j < cols; j++) {
                    if (!phaseOne && j >= artificialStart) continue;
                    if (T[m][j] > EPS) {
                        entering = j;
                        break;
                    }
                }
            } else {
                double best = EPS;
                for (int j = 0; j < cols; j++) {
                    if (!phaseOne && j >= artificialStart) continue;
                    if (T[m][j] > best) {
                        best = T[m][j];
                        entering = j;
                    }
                }
            }
            if (entering == -1) {
                if (phaseOne) {
                    double artificialSum = T[m][rhs];
                    if (artificialSum > 1e-7) return new Result(INFEASIBLE, Double.NaN, null, iterations);
                    phaseOne = false;
                    // Drive every artificial out of the basis before phase II. Leaving one basic at
                    // zero would let a phase-II pivot push it positive, which silently violates the
                    // constraint that was negated (the artificial carries that violation). A row
                    // without a usable pivot is redundant and is zeroed out instead.
                    for (int i = 0; i < m; i++) {
                        if (basis[i] < artificialStart) continue;
                        int target = -1;
                        for (int j = 0; j < artificialStart; j++) {
                            if (Math.abs(T[i][j]) > EPS) {
                                target = j;
                                break;
                            }
                        }
                        if (target == -1) {
                            Arrays.fill(T[i], 0.0);
                            basis[i] = -1;
                        } else {
                            pivot(T, m, cols, basis, i, target);
                        }
                    }
                    for (int j = 0; j <= rhs; j++) {
                        double total = 0.0;
                        for (int i = 0; i < m; i++) {
                            int bi = basis[i];
                            if (bi < 0 || bi >= artificialStart) continue;
                            double cost = bi < n ? c[bi] : 0.0;
                            total += cost * T[i][j];
                        }
                        T[m][j] = j == rhs ? -total : (j < n ? c[j] : 0.0) - total;
                    }
                    continue;
                }
                break;
            }
            if (iterations >= maxIterations) return new Result(ITERATION_LIMIT, 0.0, null, iterations);
            int leaving = chooseLeaving(T, m, basis, entering, bland);
            if (leaving == -1) {
                if (phaseOne) return new Result(ITERATION_LIMIT, 0.0, null, iterations);
                return new Result(UNBOUNDED, Double.POSITIVE_INFINITY, null, iterations);
            }
            pivot(T, m, cols, basis, leaving, entering);
            iterations++;
        }
        double[] x = new double[n];
        for (int i = 0; i < m; i++) if (basis[i] >= 0 && basis[i] < n) x[basis[i]] = T[i][rhs];
        // the objective row stores c_j - z_j with the negated objective in its right-hand side
        return new Result(OPTIMAL, -T[m][rhs], x, iterations);
    }

    private static int chooseLeaving(double[][] T, int m, int[] basis, int entering, boolean bland) {
        int rhs = T[0].length - 1;
        int chosenRow = -1;
        double bestRatio = Double.POSITIVE_INFINITY;
        for (int i = 0; i < m; i++) {
            if (basis[i] < 0) continue;
            if (T[i][entering] <= EPS) continue;
            double ratio = T[i][rhs] / T[i][entering];
            if (ratio < bestRatio - EPS) {
                bestRatio = ratio;
                chosenRow = i;
            } else if (bland && Math.abs(ratio - bestRatio) <= EPS && chosenRow != -1
                    && basis[i] < basis[chosenRow]) {
                chosenRow = i;
            }
        }
        return chosenRow;
    }

    private static void pivot(double[][] T, int m, int cols, int[] basis, int row, int column) {
        int rhs = cols;
        double pivotValue = T[row][column];
        for (int j = 0; j <= rhs; j++) T[row][j] /= pivotValue;
        for (int i = 0; i <= m; i++) {
            if (i == row) continue;
            double factor = T[i][column];
            if (factor == 0.0) continue;
            for (int j = 0; j <= rhs; j++) T[i][j] -= factor * T[row][j];
        }
        basis[row] = column;
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static void close(double a, double b, double tol, String message) {
        if (!(Math.abs(a - b) <= tol)) throw new AssertionError(message + ": " + a + " vs " + b);
    }

    /**
     * Maximum objective over all basic feasible solutions of the LP's vertex description, or null
     * when none exists. The standard form [A | I] x = b has m equations and n + m variables, so a
     * basis is a choice of m of those n + m columns; the corresponding m x m subsystem is solved by
     * Gauss-Jordan elimination, the remaining variables are set to zero, and the candidate is kept
     * only if every variable is non-negative. At most C(n+m, m) candidates are examined (126 for
     * n = 4, m = 5).
     */
    static double[] vertexMaximum(double[][] A, double[] b, double[] c) {
        int m = A.length;
        int n = c.length;
        double[][] full = new double[m][n + m];
        for (int i = 0; i < m; i++) {
            System.arraycopy(A[i], 0, full[i], 0, n);
            full[i][n + i] = 1.0;
        }
        int[] combination = new int[m];
        double[] best = {Double.NEGATIVE_INFINITY};
        double[] bestX = new double[n];
        vertexSearch(full, b, c, m, n, combination, 0, 0, best, bestX);
        if (best[0] == Double.NEGATIVE_INFINITY) return null;
        double[] out = new double[n + 1];
        out[0] = best[0];
        System.arraycopy(bestX, 0, out, 1, n);
        return out;
    }

    private static void vertexSearch(double[][] full, double[] b, double[] c, int m, int n, int[] combination,
                                     int start, int depth, double[] best, double[] bestX) {
        if (depth == m) {
            double[] x = solveSquare(full, b, m, n, combination);
            if (x == null) return;
            double value = 0.0;
            for (int j = 0; j < n; j++) value += c[j] * x[j];
            if (value > best[0] + 1e-12) {
                best[0] = value;
                System.arraycopy(x, 0, bestX, 0, n);
            }
            return;
        }
        int totalColumns = full[0].length;
        for (int j = start; j <= totalColumns - (m - depth); j++) {
            combination[depth] = j;
            vertexSearch(full, b, c, m, n, combination, j + 1, depth + 1, best, bestX);
        }
    }

    /** Solve the m x m subsystem given by the chosen columns; null when singular or not feasible. */
    private static double[] solveSquare(double[][] full, double[] b, int m, int n, int[] columns) {
        double[][] M = new double[m][m + 1];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) M[i][j] = full[i][columns[j]];
            M[i][m] = b[i];
        }
        for (int col = 0; col < m; col++) {
            int pivotRow = -1;
            for (int i = col; i < m; i++) if (Math.abs(M[i][col]) > 1e-9) { pivotRow = i; break; }
            if (pivotRow == -1) return null;
            double[] tmp = M[col];
            M[col] = M[pivotRow];
            M[pivotRow] = tmp;
            double p = M[col][col];
            for (int j = col; j <= m; j++) M[col][j] /= p;
            for (int i = 0; i < m; i++) {
                if (i == col) continue;
                double factor = M[i][col];
                if (factor == 0.0) continue;
                for (int j = col; j <= m; j++) M[i][j] -= factor * M[col][j];
            }
        }
        double[] x = new double[n + m];
        for (int j = 0; j < m; j++) x[columns[j]] = M[j][m];
        for (int j = 0; j < n + m; j++) if (x[j] < -1e-7) return null;
        return x;
    }

    /** Solve the dual min b^T y s.t. A^T y >= c, y >= 0 by negating constraints and objective. */
    static Result solveDual(double[][] A, double[] b, double[] c, boolean bland, int maxIterations) {
        int m = A.length;
        int n = c.length;
        // variables y (m of them); constraints A^T y >= c  <=>  -A^T y <= -c
        double[][] dualA = new double[n][m];
        double[] dualB = new double[n];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) dualA[j][i] = -A[i][j];
            dualB[j] = -c[j];
        }
        double[] dualC = new double[m];
        for (int i = 0; i < m; i++) dualC[i] = -b[i];
        Result r = simplex(dualA, dualB, dualC, bland, maxIterations);
        if (!r.status.equals(OPTIMAL)) return r;
        return new Result(OPTIMAL, -r.value, r.x, r.iterations);
    }

    /** Nonzero integer direction d in [0, bound]^n proving unboundedness, or null. */
    static int[] findRay(double[][] A, double[] c, int bound) {
        int n = c.length;
        int[] d = new int[n];
        return raySearch(A, c, d, 0, bound);
    }

    private static int[] raySearch(double[][] A, double[] c, int[] d, int index, int bound) {
        if (index == d.length) {
            boolean nonzero = false;
            for (int v : d) if (v != 0) nonzero = true;
            if (!nonzero) return null;
            double objective = 0.0;
            for (int j = 0; j < d.length; j++) objective += c[j] * d[j];
            if (objective <= 1e-9) return null;
            for (double[] row : A) {
                double lhs = 0.0;
                for (int j = 0; j < d.length; j++) lhs += row[j] * d[j];
                if (lhs > 1e-9) return null;
            }
            return d.clone();
        }
        for (int value = 0; value <= bound; value++) {
            d[index] = value;
            int[] found = raySearch(A, c, d, index + 1, bound);
            if (found != null) return found;
        }
        d[index] = 0;
        return null;
    }

    static boolean feasiblePoint(double[][] A, double[] b, double[] x) {
        for (double v : x) if (v < -1e-7) return false;
        for (int i = 0; i < A.length; i++) {
            double lhs = 0.0;
            for (int j = 0; j < x.length; j++) lhs += A[i][j] * x[j];
            if (lhs > b[i] + 1e-6) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(570_2025L);

        // ---- textbook instance -----------------------------------------------------------------
        double[][] classicA = {{1, 0}, {0, 2}, {3, 2}};
        double[] classicB = {4, 12, 18};
        double[] classicC = {3, 5};
        for (boolean bland : new boolean[]{true, false}) {
            Result r = simplex(classicA, classicB, classicC, bland, 500);
            check(r.status.equals(OPTIMAL), "classic instance is optimal (" + (bland ? "Bland" : "Dantzig") + ")");
            close(r.value, 36.0, 1e-9, "classic optimum 36");
            close(r.x[0], 2.0, 1e-9, "x1 = 2");
            close(r.x[1], 6.0, 1e-9, "x2 = 6");
            check(feasiblePoint(classicA, classicB, r.x), "classic solution is feasible");
        }
        double[] classicVertex = vertexMaximum(classicA, classicB, classicC);
        close(classicVertex[0], 36.0, 1e-7, "vertex enumeration agrees on the classic instance");

        // ---- second hand instance --------------------------------------------------------------
        double[][] handA = {{1, 1}, {1, 0}, {0, 1}};
        double[] handB = {4, 3, 3};
        double[] handC = {2, 1};
        Result hand = simplex(handA, handB, handC, true, 500);
        close(hand.value, 7.0, 1e-9, "hand instance optimum 7");
        close(hand.x[0], 3.0, 1e-9, "hand instance x1 = 3");
        close(hand.x[1], 1.0, 1e-9, "hand instance x2 = 1");
        close(vertexMaximum(handA, handB, handC)[0], 7.0, 1e-7, "vertex enumeration agrees on the hand instance");

        // ---- infeasible and unbounded hand cases -----------------------------------------------
        double[][] infeasibleA = {{1, 1}};
        double[] infeasibleB = {1};
        double[][] infeasibleA2 = {{1, 1}, {-1, -1}};
        double[] infeasibleB2 = {1, -3};
        Result inf1 = simplex(infeasibleA2, infeasibleB2, new double[]{1, 1}, true, 500);
        check(inf1.status.equals(INFEASIBLE), "x1 + x2 <= 1 and x1 + x2 >= 3 is infeasible");
        check(vertexMaximum(infeasibleA2, infeasibleB2, new double[]{1, 1}) == null,
                "vertex enumeration finds no feasible vertex for the infeasible instance");
        double[][] unboundedA = {{-1, 1}, {1, -1}};
        double[] unboundedB = {1, 1};
        double[] unboundedC = {1, 1};
        Result unb = simplex(unboundedA, unboundedB, unboundedC, false, 500);
        check(unb.status.equals(UNBOUNDED), "max x1 + x2 s.t. -x1 + x2 <= 1, x1 - x2 <= 1 is unbounded");
        int[] ray = findRay(unboundedA, unboundedC, 3);
        check(ray != null, "the unbounded instance has an integer ray certificate");
        Result unb2 = simplex(new double[][]{{1, -1}}, new double[]{1}, new double[]{1, 0}, true, 500);
        check(unb2.status.equals(UNBOUNDED), "max x1 s.t. x1 - x2 <= 1 is unbounded");
        check(findRay(new double[][]{{1, -1}}, new double[]{1, 0}, 3) != null, "ray certificate found");

        // ---- random family: vertices, duality, complementary slackness, ray certificates -------
        int optimal = 0;
        int unbounded = 0;
        int infeasible = 0;
        int certified = 0;
        long blandPivots = 0;
        long dantzigPivots = 0;
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + rng.nextInt(4);
            int m = 1 + rng.nextInt(5);
            double[][] A = new double[m][n];
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) A[i][j] = rng.nextInt(6) - 2;
            double[] b = new double[m];
            for (int i = 0; i < m; i++) b[i] = rng.nextInt(10) - 3;
            double[] c = new double[n];
            for (int j = 0; j < n; j++) c[j] = rng.nextInt(7) - 2;
            Result blandResult = simplex(A, b, c, true, 500);
            Result dantzigResult = simplex(A, b, c, false, 500);
            check(blandResult.status.equals(dantzigResult.status),
                    "both pivot rules agree on the status (instance " + trial + ")");
            double[] vertex = vertexMaximum(A, b, c);
            blandPivots += blandResult.iterations;
            dantzigPivots += dantzigResult.iterations;
            if (blandResult.status.equals(OPTIMAL)) {
                optimal++;
                check(vertex != null, "an optimal LP has a vertex (instance " + trial + ")");
                if (Math.abs(blandResult.value - vertex[0]) > 1e-6) {
                    System.out.println("DEBUG trial " + trial + " A=" + Arrays.deepToString(A) + " b=" + Arrays.toString(b)
                            + " c=" + Arrays.toString(c) + " simplex=" + blandResult.value + " x="
                            + Arrays.toString(blandResult.x) + " vertex=" + vertex[0] + " vx="
                            + Arrays.toString(Arrays.copyOfRange(vertex, 1, vertex.length)));
                }
                close(blandResult.value, vertex[0], 1e-6, "simplex matches the vertex max (instance " + trial + ")");
                close(dantzigResult.value, vertex[0], 1e-6, "Dantzig matches the vertex max (instance " + trial + ")");
                check(feasiblePoint(A, b, blandResult.x), "returned solution is feasible (instance " + trial + ")");
                double objective = 0.0;
                for (int j = 0; j < n; j++) objective += c[j] * blandResult.x[j];
                close(objective, blandResult.value, 1e-7, "reported value matches c^T x (instance " + trial + ")");
                // strong duality and complementary slackness
                Result dual = solveDual(A, b, c, true, 500);
                if (!dual.status.equals(OPTIMAL)) {
                    System.out.println("DEBUG trial " + trial + " A=" + Arrays.deepToString(A) + " b=" + Arrays.toString(b)
                            + " c=" + Arrays.toString(c) + " primal=" + blandResult.value + " dualStatus=" + dual.status);
                }
                check(dual.status.equals(OPTIMAL), "the dual of an optimal primal is optimal (instance " + trial + ")");
                close(dual.value, blandResult.value, 1e-6, "strong duality (instance " + trial + ")");
                double[] y = dual.x;
                double dualObjective = 0.0;
                for (int i = 0; i < m; i++) dualObjective += b[i] * y[i];
                close(dualObjective, blandResult.value, 1e-6, "dual objective equals the primal value");
                for (int i = 0; i < m; i++) {
                    double slack = b[i];
                    for (int j = 0; j < n; j++) slack -= A[i][j] * blandResult.x[j];
                    check(slack >= -1e-6, "primal slack is non-negative (instance " + trial + ")");
                    check(Math.abs(y[i] * slack) <= 1e-6, "complementary slackness on row " + i);
                }
                for (int j = 0; j < n; j++) {
                    double reduced = -c[j];
                    for (int i = 0; i < m; i++) reduced += A[i][j] * y[i];
                    check(reduced >= -1e-6, "dual slack is non-negative (instance " + trial + ")");
                    check(Math.abs(blandResult.x[j] * reduced) <= 1e-6, "complementary slackness on column " + j);
                }
            } else if (blandResult.status.equals(UNBOUNDED)) {
                unbounded++;
                check(vertex != null, "an unbounded LP still has a vertex (instance " + trial + ")");
                int[] certificate = findRay(A, c, 3);
                if (certificate != null) certified++;
            } else {
                infeasible++;
                check(vertex == null, "an infeasible LP has no feasible vertex (instance " + trial + ")");
            }
        }
        System.out.println("random family: " + optimal + " optimal, " + unbounded + " unbounded ("
                + certified + " with an integer ray certificate), " + infeasible + " infeasible");
        System.out.println("average pivots: Bland " + (blandPivots / 200.0) + ", Dantzig " + (dantzigPivots / 200.0));
        check(optimal > 50, "the random family exercises the optimal case often enough");
        check(unbounded + infeasible > 0, "the random family exercises failures too");
        check(certified == unbounded, "every unbounded instance found a small integer ray certificate");
        check(blandPivots >= dantzigPivots, "Bland's rule needs at least as many pivots on this family");

        // ---- Beale's cycling example -----------------------------------------------------------
        double[][] bealeA = {
            {0.25, -60.0, -0.04, 9.0},
            {0.5, -90.0, -0.02, 3.0},
            {0.0, 0.0, 1.0, 0.0}
        };
        double[] bealeB = {0.0, 0.0, 1.0};
        double[] bealeC = {0.75, -150.0, 0.02, -6.0};
        Result bealeBland = simplex(bealeA, bealeB, bealeC, true, 500);
        check(bealeBland.status.equals(OPTIMAL), "Bland's rule solves Beale's example");
        double[] bealeVertex = vertexMaximum(bealeA, bealeB, bealeC);
        close(bealeBland.value, bealeVertex[0], 1e-7, "Beale optimum matches the vertex enumeration");
        Result bealeDantzig = simplex(bealeA, bealeB, bealeC, false, 500);
        System.out.println("Beale example: Bland " + bealeBland.iterations + " pivots (optimum "
                + bealeBland.value + "), Dantzig status " + bealeDantzig.status + " after "
                + bealeDantzig.iterations + " pivots");
        check(bealeDantzig.status.equals(ITERATION_LIMIT),
                "Dantzig's rule cycles on Beale's example and hits the pivot cap");
        check(bealeBland.iterations < 500, "Bland's rule terminates well within the cap");

        // ---- degeneracy ------------------------------------------------------------------------
        // Two identical rows make the vertex (1, 0, 0) degenerate (more active rows than variables).
        double[][] degenerateA = {{1, 1, 1}, {1, 1, 1}, {1, 0, 0}};
        double[] degenerateB = {1, 1, 1};
        double[] degenerateC = {1, 1, 1};
        Result degenerate = simplex(degenerateA, degenerateB, degenerateC, true, 500);
        check(degenerate.status.equals(OPTIMAL), "degenerate instance is optimal");
        close(degenerate.value, 1.0, 1e-9, "degenerate optimum is 1");
        check(feasiblePoint(degenerateA, degenerateB, degenerate.x), "degenerate solution is feasible");
        close(vertexMaximum(degenerateA, degenerateB, degenerateC)[0], 1.0, 1e-7,
                "vertex enumeration agrees on the degenerate instance");
        Result redundant = simplex(new double[][]{{0, 0}, {1, 1}}, new double[]{5, 2}, new double[]{1, 1}, true, 500);
        check(redundant.status.equals(OPTIMAL), "a redundant row does not break the simplex");
        close(redundant.value, 2.0, 1e-9, "redundant-row instance optimum is 2");

        System.out.println("PASS");
    }
}
