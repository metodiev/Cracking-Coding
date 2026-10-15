/** Task 377 - Simplex algorithm for small LPs (outline). (Tier: Intermediate, CTCI Ch 6) */
import java.util.*;

/**
 * Standard form: maximise c.x subject to A x <= b, x >= 0.  The tableau simplex
 * starts from the slack basis (x = 0) when b >= 0.  Rows with b_i < 0 are negated
 * (their slack coefficient becomes -1) and get an artificial variable; a phase-1
 * objective (-1 on the artificial columns, i.e. minimising their sum) drives them
 * out, and a positive phase-1 optimum means the LP is infeasible.  Phase 2 then runs
 * the real objective with the artificial columns forbidden from entering.
 *
 * Bland's rule is used for both iterations: the entering variable is the smallest
 * index with a positive reduced cost and the leaving variable breaks ratio ties by
 * the smallest basic-variable index.  This guarantees termination (no cycling); an
 * iteration cap is kept as a failure sentinel and is never hit by the test problems.
 *
 * Documented sentinels (the API returns a status string; x is null, value NaN unless
 * "optimal"):
 *   * "optimal"    -> x is the optimal vertex,
 *   * "unbounded"  -> some entering column has no positive tableau coefficient,
 *   * "infeasible" -> a positive phase-1 optimum (the constraints contradict).
 *
 * Re-derivation note: the task brief suggests "max 3x + 2y s.t. x+y<=4, x+3y<=6 ->
 * x=3, y=1, value 11".  That point is the intersection of the two constraint lines,
 * but the vertex (4, 0) satisfies both constraints (4+0=4, 4+0<=6) and gives
 * 3*4 + 2*0 = 12 > 11, so the true optimum is 12 at (4, 0) - the code asserts 12.
 *
 * The objective values are small exact integers here; comparisons use the documented
 * tolerance 1e-9 because the tableau arithmetic is in doubles.  A Cramer-rule vertex
 * enumeration is used as an independent oracle for the LCG-generated bounded LPs.
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/** status in {"optimal", "unbounded", "infeasible"}; x = null and value = NaN unless optimal. */
class LpResult {
    final String status;
    final double[] x;
    final double value;

    LpResult(String status, double[] x, double value) {
        this.status = status;
        this.x = x;
        this.value = value;
    }
}

public class Solution {
    static final double EPS = 1e-9;
    static final int MAX_ITERATIONS = 10000;

    /** Primal simplex core (Bland's rule). Mutates the tableau; returns {status, value}. */
    static Object[] runSimplex(double[][] tableau, double[] rhs, int[] basis,
                               double[] cost, boolean[] canEnter) {
        int rows = tableau.length;
        int columns = cost.length;

        double[] reduced = cost.clone();
        double value = 0.0;
        for (int i = 0; i < rows; i++) {
            double basicCost = cost[basis[i]];
            if (basicCost != 0.0) {
                value += basicCost * rhs[i];
                for (int j = 0; j < columns; j++) reduced[j] -= basicCost * tableau[i][j];
            }
        }

        int iterations = 0;
        while (true) {
            if (++iterations > MAX_ITERATIONS) throw new AssertionError("simplex iteration cap");
            int entering = -1;
            for (int j = 0; j < columns; j++) {                 // Bland: smallest improving index
                if (canEnter[j] && reduced[j] > EPS) {
                    entering = j;
                    break;
                }
            }
            if (entering == -1) return new Object[]{"optimal", value};

            int leaving = -1;
            double bestRatio = 0.0;
            for (int i = 0; i < rows; i++) {
                if (tableau[i][entering] > EPS) {
                    double ratio = rhs[i] / tableau[i][entering];
                    if (leaving == -1 || ratio < bestRatio - 1e-12) {
                        leaving = i;
                        bestRatio = ratio;
                    } else if (Math.abs(ratio - bestRatio) <= 1e-12 && basis[i] < basis[leaving]) {
                        leaving = i;                                // Bland ratio tie-break
                        bestRatio = ratio;
                    }
                }
            }
            if (leaving == -1) return new Object[]{"unbounded", Double.NaN};

            double pivot = tableau[leaving][entering];
            for (int j = 0; j < columns; j++) tableau[leaving][j] /= pivot;
            rhs[leaving] /= pivot;
            for (int i = 0; i < rows; i++) {
                if (i != leaving && tableau[i][entering] != 0.0) {
                    double factor = tableau[i][entering];
                    for (int j = 0; j < columns; j++) tableau[i][j] -= factor * tableau[leaving][j];
                    rhs[i] -= factor * rhs[leaving];
                }
            }
            double factor = reduced[entering];
            for (int j = 0; j < columns; j++) reduced[j] -= factor * tableau[leaving][j];
            value += factor * rhs[leaving];
            basis[leaving] = entering;
        }
    }

    /** max c.x s.t. A x <= b, x >= 0.  Returns an LpResult (see sentinel documentation). */
    static LpResult simplex(int[] objective, int[][] constraints, int[] bounds) {
        int n = objective.length;
        int m = constraints.length;
        int artificialCount = 0;
        for (int i = 0; i < m; i++) if (bounds[i] < 0) artificialCount++;
        int columns = n + m + artificialCount;

        double[][] tableau = new double[m][columns];
        double[] rhs = new double[m];
        int[] basis = new int[m];
        int[] artificialOfRow = new int[m];
        Arrays.fill(artificialOfRow, -1);
        int nextArtificial = n + m;
        for (int i = 0; i < m; i++) {
            if (bounds[i] >= 0) {
                for (int j = 0; j < n; j++) tableau[i][j] = constraints[i][j];
                tableau[i][n + i] = 1.0;                          // slack basis
                basis[i] = n + i;
                rhs[i] = bounds[i];
            } else {
                for (int j = 0; j < n; j++) tableau[i][j] = -constraints[i][j];
                tableau[i][n + i] = -1.0;
                tableau[i][nextArtificial] = 1.0;                 // artificial basis
                artificialOfRow[i] = nextArtificial;
                basis[i] = nextArtificial;
                nextArtificial++;
                rhs[i] = -bounds[i];
            }
        }

        if (artificialCount > 0) {
            double[] phaseOneCost = new double[columns];
            for (int i = 0; i < m; i++) {
                if (artificialOfRow[i] != -1) phaseOneCost[artificialOfRow[i]] = -1.0;
            }
            boolean[] allColumns = new boolean[columns];
            Arrays.fill(allColumns, true);
            Object[] outcome = runSimplex(tableau, rhs, basis, phaseOneCost, allColumns);
            String statusOne = (String) outcome[0];
            double valueOne = (Double) outcome[1];
            assert statusOne.equals("optimal");                   // -sum(a) is bounded by 0
            if (valueOne < -EPS) return new LpResult("infeasible", null, Double.NaN);
        }

        double[] phaseTwoCost = new double[columns];
        for (int j = 0; j < n; j++) phaseTwoCost[j] = objective[j];
        boolean[] canEnter = new boolean[columns];
        for (int j = 0; j < n + m; j++) canEnter[j] = true;        // artificials stay out
        Object[] outcome = runSimplex(tableau, rhs, basis, phaseTwoCost, canEnter);
        String status = (String) outcome[0];
        if (!status.equals("optimal")) return new LpResult(status, null, Double.NaN);

        double value = (Double) outcome[1];
        double[] x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) x[basis[i]] = rhs[i];
        }
        return new LpResult("optimal", x, value);
    }

    /** Exact-ish recursive determinant for tiny matrices (n <= 3 in this task). */
    static double determinant(double[][] matrix) {
        int size = matrix.length;
        if (size == 1) return matrix[0][0];
        double total = 0.0;
        for (int j = 0; j < size; j++) {
            double[][] minor = new double[size - 1][size - 1];
            for (int r = 1; r < size; r++) {
                int at = 0;
                for (int c = 0; c < size; c++) {
                    if (c != j) minor[r - 1][at++] = matrix[r][c];
                }
            }
            total += (j % 2 == 0 ? 1 : -1) * matrix[0][j] * determinant(minor);
        }
        return total;
    }

    static void collectCombinations(int n, int k, int start, int[] combo, int at, List<int[]> out) {
        if (at == k) {
            out.add(combo.clone());
            return;
        }
        for (int i = start; i <= n - (k - at); i++) {
            combo[at] = i;
            collectCombinations(n, k, i + 1, combo, at + 1, out);
        }
    }

    /** Cramer-rule vertex enumeration for bounded feasible LPs (unique optimum value). */
    static Double bruteForceLp(int[] objective, int[][] constraints, int[] bounds) {
        int n = objective.length;
        int m = constraints.length;
        List<double[]> hyperplanes = new ArrayList<>();
        List<Double> hyperRhs = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            double[] row = new double[n];
            for (int j = 0; j < n; j++) row[j] = constraints[i][j];
            hyperplanes.add(row);
            hyperRhs.add((double) bounds[i]);
        }
        for (int j = 0; j < n; j++) {
            double[] unit = new double[n];
            unit[j] = 1.0;
            hyperplanes.add(unit);
            hyperRhs.add(0.0);
        }

        List<int[]> combos = new ArrayList<>();
        collectCombinations(hyperplanes.size(), n, 0, new int[n], 0, combos);

        double tolerance = 1e-7;
        Double best = null;
        for (int[] combo : combos) {
            double[][] matrix = new double[n][n];
            double[] rhs = new double[n];
            for (int i = 0; i < n; i++) {
                matrix[i] = hyperplanes.get(combo[i]);
                rhs[i] = hyperRhs.get(combo[i]);
            }
            double det = determinant(matrix);
            if (Math.abs(det) < 1e-12) continue;
            double[] x = new double[n];
            for (int col = 0; col < n; col++) {
                double[][] replaced = new double[n][n];
                for (int r = 0; r < n; r++) {
                    for (int c = 0; c < n; c++) replaced[r][c] = (c == col) ? rhs[r] : matrix[r][c];
                }
                x[col] = determinant(replaced) / det;
            }
            boolean feasible = true;
            for (int j = 0; j < n && feasible; j++) if (x[j] < -tolerance) feasible = false;
            for (int i = 0; i < m && feasible; i++) {
                double lhs = 0.0;
                for (int j = 0; j < n; j++) lhs += constraints[i][j] * x[j];
                if (lhs > bounds[i] + tolerance) feasible = false;
            }
            if (!feasible) continue;
            double value = 0.0;
            for (int j = 0; j < n; j++) value += objective[j] * x[j];
            if (best == null || value > best) best = value;
        }
        return best;
    }

    static Object[] lcgBoundedLp(Lcg lcg) {
        int n = 1 + lcg.next(2);
        int m = 2 + lcg.next(2);
        int[][] constraints = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) constraints[i][j] = lcg.next(9);
        }
        for (int j = 0; j < n; j++) {                             // every column positive
            boolean positive = false;
            for (int i = 0; i < m; i++) if (constraints[i][j] > 0) positive = true;
            if (!positive) constraints[lcg.next(m)][j] = 1 + lcg.next(3);
        }
        int[] bounds = new int[m];
        for (int i = 0; i < m; i++) bounds[i] = 1 + lcg.next(20);
        int[] objective = new int[n];
        for (int j = 0; j < n; j++) objective[j] = lcg.next(10);
        return new Object[]{objective, constraints, bounds};
    }

    public static void main(String[] args) {
        // the brief's LP, re-derived: optimum (4, 0) with value 12, NOT (3, 1)/11
        LpResult result = simplex(new int[]{3, 2}, new int[][]{{1, 1}, {1, 3}}, new int[]{4, 6});
        check(result.status.equals("optimal"), "brief LP optimal");
        check(Math.abs(result.value - 12.0) < EPS, "brief LP value 12, got " + result.value);
        check(Math.abs(result.x[0] - 4.0) < EPS && Math.abs(result.x[1]) < EPS, "brief LP vertex (4, 0)");
        check(bruteForceLp(new int[]{3, 2}, new int[][]{{1, 1}, {1, 3}}, new int[]{4, 6}) == 12.0,
                "brief LP oracle 12");

        // hand-solved: vertices (0,0)=0, (2.5,0)=5, (0,4)=12, (1,3)=11 -> 12 at (0,4)
        result = simplex(new int[]{2, 3}, new int[][]{{1, 1}, {2, 1}}, new int[]{4, 5});
        check(result.status.equals("optimal") && Math.abs(result.value - 12.0) < EPS, "second LP 12");
        check(Math.abs(result.x[0]) < EPS && Math.abs(result.x[1] - 4.0) < EPS, "second LP vertex");
        check(bruteForceLp(new int[]{2, 3}, new int[][]{{1, 1}, {2, 1}}, new int[]{4, 5}) == 12.0,
                "second LP oracle");

        // 3 variables: optima (4,1,0), (2,2,1), (0,3,2) all give 14; Bland picks one
        result = simplex(new int[]{3, 2, 4}, new int[][]{{1, 1, 1}, {2, 1, 3}}, new int[]{5, 9});
        check(result.status.equals("optimal") && Math.abs(result.value - 14.0) < EPS, "3-var LP 14");
        check(Math.abs(result.x[0] + result.x[1] + result.x[2] - 5.0) < EPS, "3-var sum is 5");
        check(result.x[0] > -EPS && result.x[1] > -EPS && result.x[2] > -EPS, "3-var non-negative");
        check(2 * result.x[0] + result.x[1] + 3 * result.x[2] <= 9 + 1e-7, "3-var second constraint");
        check(bruteForceLp(new int[]{3, 2, 4}, new int[][]{{1, 1, 1}, {2, 1, 3}}, new int[]{5, 9}) == 14.0,
                "3-var LP oracle");

        // tie along x + y <= 1 (degenerate objective direction)
        result = simplex(new int[]{1, 1}, new int[][]{{1, 1}}, new int[]{1});
        check(result.status.equals("optimal") && Math.abs(result.value - 1.0) < EPS, "tie LP");
        check(Math.abs(result.x[0] + result.x[1] - 1.0) < EPS, "tie LP sum");

        // duplicate constraints (degenerate basis)
        result = simplex(new int[]{1}, new int[][]{{1}, {1}}, new int[]{3, 3});
        check(result.status.equals("optimal") && Math.abs(result.value - 3.0) < EPS, "duplicate LP");

        // --- sentinels ---------------------------------------------------------
        result = simplex(new int[]{1}, new int[0][], new int[0]);
        check(result.status.equals("unbounded") && result.x == null, "no constraints -> unbounded");
        result = simplex(new int[]{1}, new int[][]{{-1}}, new int[]{-1});     // x >= 1
        check(result.status.equals("unbounded") && result.x == null, "x >= 1 -> unbounded");
        result = simplex(new int[]{1}, new int[][]{{1}, {-1}}, new int[]{1, -3});  // x<=1, x>=3
        check(result.status.equals("infeasible") && result.x == null, "x <= 1 and x >= 3 infeasible");

        // feasible LP with a negative bound (phase 1 needed, artificial leaves the basis)
        result = simplex(new int[]{1}, new int[][]{{1}, {-1}}, new int[]{5, -2});  // 2 <= x <= 5
        check(result.status.equals("optimal") && Math.abs(result.x[0] - 5.0) < EPS, "negative bound max x");
        result = simplex(new int[]{-1}, new int[][]{{1}, {-1}}, new int[]{5, -2});
        check(result.status.equals("optimal") && Math.abs(result.value + 2.0) < EPS, "negative bound min x");

        // --- LCG cross-checks against vertex enumeration -----------------------
        Lcg lcg = new Lcg(37713);
        for (int trial = 0; trial < 40; trial++) {
            Object[] lp = lcgBoundedLp(lcg);
            int[] objective = (int[]) lp[0];
            int[][] constraints = (int[][]) lp[1];
            int[] bounds = (int[]) lp[2];
            result = simplex(objective, constraints, bounds);
            check(result.status.equals("optimal"), "random LP optimal " + trial);
            Double oracle = bruteForceLp(objective, constraints, bounds);
            check(oracle != null, "random LP has an oracle vertex " + trial);
            check(Math.abs(result.value - oracle) < 1e-6, "random LP value " + trial);
            for (int i = 0; i < constraints.length; i++) {
                double lhs = 0.0;
                for (int j = 0; j < objective.length; j++) lhs += constraints[i][j] * result.x[j];
                check(lhs <= bounds[i] + 1e-7, "random LP feasibility " + trial);
            }
            for (double component : result.x) check(component > -1e-7, "random LP non-negative " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
