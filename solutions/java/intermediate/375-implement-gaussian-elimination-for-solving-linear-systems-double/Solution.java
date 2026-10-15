/** Task 375 - Gaussian elimination with partial pivoting (doubles). (Tier: Intermediate, CTCI Ch 6) */
import java.util.*;

/**
 * Solve A x = b for a small dense double matrix.  Forward elimination with partial
 * pivoting (swap in the row with the largest |pivot|), then back substitution.
 * Documented conventions:
 *   * a pivot column whose remaining entries are all <= EPS in absolute value is
 *     treated as a free column - the system then has no unique solution;
 *   * after elimination, a row with an all-zero coefficient part but |rhs| > EPS is
 *     reported "inconsistent";
 *   * otherwise a rank-deficient consistent system is reported "infinite";
 *   * results are computed in doubles, so verification by substitution uses the
 *     documented tolerance TOL = 1e-9 (absolute) - the small hand-checked systems
 *     below have exact integer solutions that the elimination reproduces to well
 *     within that tolerance.
 * Sentinel: the status string ("unique", "infinite", "inconsistent") plus x = null for
 * anything that is not "unique".
 *
 * Includes systems that require a pivot swap (leading zero in the first column) and a
 * 4x4 diagonally dominant system, plus LCG cross-checks: a random integer matrix A and
 * integer x* give b = A x*; the solver must recover x* whenever the elimination reports
 * a unique solution.
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

/** status in {"unique", "infinite", "inconsistent"}; x is null unless unique. */
class SolveResult {
    final String status;
    final double[] x;

    SolveResult(String status, double[] x) {
        this.status = status;
        this.x = x;
    }
}

public class Solution {
    static final double EPS = 1e-12;      // pivot/inconsistency threshold for the elimination
    static final double TOL = 1e-9;       // tolerance for verifying a solution by substitution

    static SolveResult solve(double[][] matrix, double[] rhs) {
        int n = matrix.length;
        double[][] augmented = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) augmented[i][j] = matrix[i][j];
            augmented[i][n] = rhs[i];
        }

        int rank = 0;
        for (int col = 0; col < n; col++) {
            int pivotRow = -1;
            double pivotSize = 0.0;
            for (int row = rank; row < n; row++) {
                if (Math.abs(augmented[row][col]) > pivotSize) {
                    pivotSize = Math.abs(augmented[row][col]);
                    pivotRow = row;
                }
            }
            if (pivotRow == -1 || pivotSize <= EPS) continue;              // free column
            double[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++) {
                double factor = augmented[row][col] / augmented[rank][col];
                if (factor != 0.0) {
                    for (int k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
                }
            }
            rank++;
        }

        for (int row = rank; row < n; row++) {
            if (Math.abs(augmented[row][n]) > EPS) return new SolveResult("inconsistent", null);
        }
        if (rank < n) return new SolveResult("infinite", null);

        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double total = augmented[row][n];
            for (int col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
            x[row] = total / augmented[row][row];
        }
        return new SolveResult("unique", x);
    }

    /** max |A x - b| - the independent substitution check. */
    static double residual(double[][] matrix, double[] x, double[] rhs) {
        double worst = 0.0;
        for (int i = 0; i < matrix.length; i++) {
            double total = 0.0;
            for (int j = 0; j < matrix[i].length; j++) total += matrix[i][j] * x[j];
            worst = Math.max(worst, Math.abs(total - rhs[i]));
        }
        return worst;
    }

    /** Returns {matrix, rhs, planted solution, new seed packing} - state carried via the Lcg. */
    static Object[] lcgSystem(Lcg lcg) {
        int size = 2 + lcg.next(3);
        double[][] matrix = new double[size][size];
        int[][] intMatrix = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                intMatrix[i][j] = lcg.next(11) - 5;
                matrix[i][j] = intMatrix[i][j];
            }
        }
        int[] solution = new int[size];
        for (int i = 0; i < size; i++) solution[i] = lcg.next(11) - 5;
        double[] rhs = new double[size];
        for (int i = 0; i < size; i++) {
            int total = 0;
            for (int j = 0; j < size; j++) total += intMatrix[i][j] * solution[j];
            rhs[i] = total;
        }
        return new Object[]{matrix, rhs, solution};
    }

    public static void main(String[] args) {
        // 2x2, unique integer solution (2, 1)
        SolveResult result = solve(new double[][]{{1, 1}, {1, -1}}, new double[]{3, 1});
        check(result.status.equals("unique"), "2x2 unique");
        check(Math.abs(result.x[0] - 2) < TOL && Math.abs(result.x[1] - 1) < TOL, "2x2 solution");

        // needs a pivot swap: first pivot is zero
        result = solve(new double[][]{{0, 1}, {1, 0}}, new double[]{2, 3});
        check(result.status.equals("unique"), "pivot swap unique");
        check(Math.abs(result.x[0] - 3) < TOL && Math.abs(result.x[1] - 2) < TOL, "pivot swap solution");

        // 3x3 permutation system: x = (1, 2, 3)
        result = solve(new double[][]{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}}, new double[]{3, 2, 1});
        check(result.status.equals("unique"), "permutation unique");
        for (int i = 0; i < 3; i++) check(Math.abs(result.x[i] - (i + 1)) < TOL, "permutation " + i);

        // classic 3x3 with solution (5, 3, -2): checked by hand
        double[][] matrix3 = {{1, 1, 1}, {0, 2, 5}, {2, 5, -1}};
        double[] rhs3 = {6, -4, 27};
        check(5 + 3 - 2 == 6 && 2 * 3 + 5 * -2 == -4 && 2 * 5 + 5 * 3 + 2 == 27, "hand check 3x3");
        result = solve(matrix3, rhs3);
        check(result.status.equals("unique"), "3x3 unique");
        check(Math.abs(result.x[0] - 5) < TOL && Math.abs(result.x[1] - 3) < TOL
                && Math.abs(result.x[2] + 2) < TOL, "3x3 solution");
        check(residual(matrix3, result.x, rhs3) < TOL, "3x3 residual");

        // 4x4 diagonally dominant with solution (1, 2, 3, 4)
        double[][] matrix4 = {{2, 1, 1, 1}, {1, 2, 1, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}};
        double[] rhs4 = {11, 12, 13, 14};
        result = solve(matrix4, rhs4);
        check(result.status.equals("unique"), "4x4 unique");
        for (int i = 0; i < 4; i++) check(Math.abs(result.x[i] - (i + 1)) < TOL, "4x4 " + i);
        check(residual(matrix4, result.x, rhs4) < TOL, "4x4 residual");

        // rank-deficient consistent -> infinite
        result = solve(new double[][]{{1, 1}, {2, 2}}, new double[]{1, 2});
        check(result.status.equals("infinite") && result.x == null, "infinite sentinel");
        // rank-deficient inconsistent -> inconsistent
        result = solve(new double[][]{{1, 1}, {2, 2}}, new double[]{1, 2.5});
        check(result.status.equals("inconsistent") && result.x == null, "inconsistent sentinel");
        // three-way dependent 3x3: row3 = row1 + row2
        result = solve(new double[][]{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, new double[]{1, 2, 3});
        check(result.status.equals("infinite") && result.x == null, "3x3 dependent infinite");
        result = solve(new double[][]{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, new double[]{1, 2, 4});
        check(result.status.equals("inconsistent") && result.x == null, "3x3 dependent inconsistent");

        // --- LCG cross-checks: b = A x* must recover x* -----------------------
        Lcg lcg = new Lcg(37513);
        int uniqueCount = 0;
        for (int trial = 0; trial < 80; trial++) {
            Object[] system = lcgSystem(lcg);
            double[][] matrix = (double[][]) system[0];
            double[] rhs = (double[]) system[1];
            int[] planted = (int[]) system[2];
            result = solve(matrix, rhs);
            if (!result.status.equals("unique")) continue;   // singular random matrix; skipped
            uniqueCount++;
            for (int i = 0; i < planted.length; i++) {
                check(Math.abs(result.x[i] - planted[i]) < 1e-6, "recover planted " + trial);
            }
            check(residual(matrix, result.x, rhs) < TOL, "random residual " + trial);
        }
        check(uniqueCount >= 40, "the suite really exercises unique systems, got " + uniqueCount);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
