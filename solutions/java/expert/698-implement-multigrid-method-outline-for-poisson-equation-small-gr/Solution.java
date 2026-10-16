/** Task 698 - Implement multigrid method outline for Poisson equation small grid. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * A real 1D multigrid V-cycle for -u'' = f on the unit interval with Dirichlet ends. The grid
 * has n = 2^p - 1 interior points, the operator is A = tridiag(-1, 2, -1), and the pieces are the
 * textbook ones:
 *
 *   smoother      weighted Jacobi with omega = 2/3 (damps the oscillatory error, useless on the
 *                 smooth part - which is exactly why multigrid also needs a coarse grid);
 *   restriction   full weighting, R[i] = (1/4, 1/2, 1/4) over the fine neighbours of a coarse
 *                 point;
 *   prolongation  linear interpolation, P = 2 R^T (checked numerically);
 *   coarse operator  the Galerkin product A_c = R A P, which for this discretisation comes out
 *                 proportional to the coarse tridiagonal operator;
 *   recursion      V(2,2): two pre-smooths, recurse on the coarse residual equation, correct,
 *                 two post-smooths, and solve the coarsest level (n <= 3) directly.
 *
 * The course of the demo is the multigrid textbook result: a residual reduction factor around
 * 0.1 per cycle that barely depends on the grid size, versus an O(n^2) iteration count for plain
 * Jacobi. Verified: P = 2R^T, the Galerkin coarse operator's symmetry and scaling, the V-cycle
 * factor and its grid independence, the exact nodal solution for a constant right-hand side,
 * agreement with a direct solve, and the coarse correction leaving smooth modes untouched.
 *
 * Simplifications: one dimension and a fixed V(2,2) schedule; a 2D or 3D version would add more
 * transfer operators but no new idea.
 */
import java.util.*;

public class Solution {
    /** Weighted Jacobi smoother: x <- x + omega * D^-1 (b - A x). */
    static double[] smooth(double[] x, double[] b, int sweeps, double omega) {
        int n = x.length;
        double[] current = x.clone();
        for (int sweep = 0; sweep < sweeps; sweep++) {
            double[] next = current.clone();
            for (int i = 0; i < n; i++) {
                double residual = b[i] - (2 * current[i]
                        - (i > 0 ? current[i - 1] : 0)
                        - (i + 1 < n ? current[i + 1] : 0));
                next[i] = current[i] + omega * residual / 2;
            }
            current = next;
        }
        return current;
    }

    /** Fine residual b - A x for the Poisson operator. */
    static double[] residual(double[] x, double[] b) {
        int n = x.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = b[i] - (2 * x[i] - (i > 0 ? x[i - 1] : 0) - (i + 1 < n ? x[i + 1] : 0));
        }
        return out;
    }

    /** Full weighting restriction: coarse[j] = (f[2j] + 2 f[2j+1] + f[2j+2]) / 4. */
    static double[] restrict(double[] fine) {
        int coarseSize = (fine.length - 1) / 2;
        double[] coarse = new double[coarseSize];
        for (int j = 0; j < coarseSize; j++) {
            coarse[j] = 0.25 * fine[2 * j] + 0.5 * fine[2 * j + 1] + 0.25 * fine[2 * j + 2];
        }
        return coarse;
    }

    /** Linear interpolation prolongation; equals 2 R^T for this pair. */
    static double[] prolongate(double[] coarse) {
        int fineSize = 2 * coarse.length + 1;
        double[] fine = new double[fineSize];
        for (int i = 0; i < fineSize; i++) {
            if (i % 2 == 1) {
                fine[i] = coarse[(i - 1) / 2];
            } else {
                double left = (i - 2) / 2 >= 0 ? coarse[(i - 2) / 2] : 0;
                double right = (i / 2) < coarse.length ? coarse[i / 2] : 0;
                fine[i] = 0.5 * (left + right);
            }
        }
        return fine;
    }

    /** Explicit restriction matrix for the invariant checks. */
    static double[][] restrictionMatrix(int fine, int coarse) {
        double[][] r = new double[coarse][fine];
        for (int j = 0; j < coarse; j++) {
            r[j][2 * j] = 0.25;
            r[j][2 * j + 1] = 0.5;
            r[j][2 * j + 2] = 0.25;
        }
        return r;
    }

    /** Explicit prolongation matrix. */
    static double[][] prolongationMatrix(int fine, int coarse) {
        double[][] p = new double[fine][coarse];
        for (int j = 0; j < coarse; j++) p[2 * j + 1][j] = 1;
        for (int i = 0; i < fine; i += 2) {
            if ((i - 2) / 2 >= 0) p[i][(i - 2) / 2] += 0.5;
            if (i / 2 < coarse) p[i][i / 2] += 0.5;
        }
        return p;
    }

    static double[][] poissonOperator(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i][i] = 2;
            if (i > 0) a[i][i - 1] = -1;
            if (i + 1 < n) a[i][i + 1] = -1;
        }
        return a;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < b.length; k++) {
                if (a[i][k] == 0) continue;
                for (int j = 0; j < m; j++) out[i][j] += a[i][k] * b[k][j];
            }
        }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] out = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) out[j][i] = a[i][j];
        }
        return out;
    }

    static double[] apply(double[][] a, double[] x) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double sum = 0;
            for (int j = 0; j < x.length; j++) sum += a[i][j] * x[j];
            out[i] = sum;
        }
        return out;
    }

    static double norm(double[] values) {
        double sum = 0;
        for (double value : values) sum += value * value;
        return Math.sqrt(sum);
    }

    static double[] directSolve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int column = 0; column < n; column++) {
            int pivot = column;
            for (int row = column + 1; row < n; row++) {
                if (Math.abs(m[row][column]) > Math.abs(m[pivot][column])) pivot = row;
            }
            double[] swap = m[column];
            m[column] = m[pivot];
            m[pivot] = swap;
            for (int row = column + 1; row < n; row++) {
                double factor = m[row][column] / m[column][column];
                for (int j = column; j <= n; j++) m[row][j] -= factor * m[column][j];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = m[i][n];
            for (int j = i + 1; j < n; j++) sum -= m[i][j] * x[j];
            x[i] = sum / m[i][i];
        }
        return x;
    }

    /** One V(2,2) cycle. */
    static double[] vCycle(double[] x, double[] b, int preSweeps, int postSweeps) {
        int n = b.length;
        if (n <= 3) return directSolve(poissonOperator(n), b);
        double[] current = smooth(x, b, preSweeps, 2.0 / 3.0);
        double[] residualFine = residual(current, b);
        double[] coarseResidual = restrict(residualFine);
        // Scale change: A_h = h^-2 T and A_2h = (2h)^-2 T, so the coarse problem for the same
        // tridiagonal form is A_2h e = 4 R r (the Galerkin product R A_h P equals (1/4) A_2h).
        for (int i = 0; i < coarseResidual.length; i++) coarseResidual[i] *= 4;
        double[] coarseCorrection = vCycle(new double[coarseResidual.length], coarseResidual,
                preSweeps, postSweeps);
        double[] correction = prolongate(coarseCorrection);
        for (int i = 0; i < n; i++) current[i] += correction[i];
        return smooth(current, b, postSweeps, 2.0 / 3.0);
    }

    /** V-cycles needed to reach the tolerance, plus the residual history. */
    static List<Double> solve(double[] b, double tolerance, int maxCycles, List<Double> history) {
        int n = b.length;
        double[] x = new double[n];
        double initial = norm(b);
        for (int cycle = 0; cycle < maxCycles; cycle++) {
            x = vCycle(x, b, 2, 2);
            double relative = norm(residual(x, b)) / initial;
            history.add(relative);
            if (relative < tolerance) break;
        }
        List<Double> result = new ArrayList<>();
        for (double value : x) result.add(value);
        return result;
    }

    public static void main(String[] args) {
        // 1. The transfer operators: P = 2 R^T, and the Galerkin coarse operator's structure.
        int fine = 31;
        int coarse = 15;
        double[][] r = restrictionMatrix(fine, coarse);
        double[][] p = prolongationMatrix(fine, coarse);
        double[][] twoRt = transpose(r);
        boolean adjoint = true;
        for (int i = 0; i < fine; i++) {
            for (int j = 0; j < coarse; j++) {
                if (Math.abs(p[i][j] - 2 * twoRt[i][j]) > 1e-12) adjoint = false;
            }
        }
        check(adjoint, "prolongation is twice the transpose of restriction");
        double[] smoothVector = new double[coarse];
        for (int j = 0; j < coarse; j++) smoothVector[j] = Math.sin((j + 1) * Math.PI / (coarse + 1));
        double[] roundTrip = restrict(prolongate(smoothVector));
        double worst = 0;
        for (int j = 0; j < coarse; j++) {
            worst = Math.max(worst, Math.abs(roundTrip[j] - smoothVector[j]));
        }
        check(worst < 0.02, "smooth modes survive the interpolation-restriction round trip: " + worst);

        double[][] coarseOperator = multiply(r, multiply(poissonOperator(fine), p));
        boolean symmetric = true;
        for (int i = 0; i < coarse; i++) {
            for (int j = 0; j < coarse; j++) {
                if (Math.abs(coarseOperator[i][j] - coarseOperator[j][i]) > 1e-12) symmetric = false;
            }
        }
        check(symmetric, "the Galerkin coarse operator is symmetric");
        // Every row comes out as (1/4) * tridiag(-1, 2, -1): the full-weighting / linear-
        // interpolation pair makes the Galerkin product consistent with the rediscretised
        // operator, boundary rows included.
        boolean proportional = true;
        for (int i = 0; i < coarse; i++) {
            if (Math.abs(coarseOperator[i][i] - coarseOperator[0][0]) > 1e-12) proportional = false;
            if (i + 1 < coarse && Math.abs(coarseOperator[i][i + 1] * 2 + coarseOperator[i][i]) > 1e-12) {
                proportional = false;
            }
        }
        check(proportional, "R A P is the coarse Poisson operator up to scale: "
                + coarseOperator[0][0] + " / " + coarseOperator[0][1]);
        double[] probe = new double[coarse];
        for (int j = 0; j < coarse; j++) probe[j] = Math.cos(j * 0.9) + 0.1;
        check(inner(probe, apply(coarseOperator, probe)) > 0, "the coarse operator is positive definite");

        // 2. A V-cycle on a constant right-hand side has a known exact solution.
        int n = 31;
        double[] constantB = new double[n];
        Arrays.fill(constantB, 1.0);
        double h = 1.0 / (n + 1);
        double[] analytic = new double[n];
        for (int i = 0; i < n; i++) {
            double x = (i + 1) * h;
            // The operator is h^2 * (-d2/dx2), so with a unit right-hand side the exact nodal
            // values are u(x)/h^2 = x(1-x)/(2h^2). Equivalently u_j = j(n+1-j)/2.
            analytic[i] = x * (1 - x) / (2 * h * h);
        }
        List<Double> history = new ArrayList<>();
        List<Double> solution = solve(constantB, 1e-12, 50, history);
        double[] numeric = toArray(solution);
        double analyticError = 0;
        for (int i = 0; i < n; i++) {
            analyticError = Math.max(analyticError, Math.abs(numeric[i] - analytic[i]));
        }
        check(analyticError < 1e-10,
                "the discrete solution equals x(1-x)/2 at the nodes: " + analyticError);
        check(history.get(history.size() - 1) < 1e-12, "the solve reaches its tolerance");
        check(history.size() <= 15, "few V-cycles are needed: " + history.size());
        double firstFactor = history.get(0);
        check(firstFactor < 0.5, "one V-cycle already cuts the residual below half: " + firstFactor);
        if (history.size() >= 2) {
            double lastFactor = history.get(history.size() - 1) / history.get(history.size() - 2);
            check(Math.abs(lastFactor - firstFactor) < 0.1,
                    "the reduction factor is roughly constant: " + firstFactor + " vs " + lastFactor);
        }

        // 3. Agreement with a direct solve and with plain Jacobi's slow crawl.
        double[] rhs = new double[n];
        for (int i = 0; i < n; i++) rhs[i] = Math.sin((i + 1) * Math.PI / (n + 1));
        List<Double> history2 = new ArrayList<>();
        double[] multigrid = toArray(solve(rhs, 1e-10, 50, history2));
        double[] direct = directSolve(poissonOperator(n), rhs);
        double difference = 0;
        for (int i = 0; i < n; i++) difference = Math.max(difference, Math.abs(multigrid[i] - direct[i]));
        check(difference < 1e-9, "multigrid agrees with the direct solve: " + difference);

        double[] jacobi = new double[n];
        double initialResidual = norm(rhs);
        double[] jacobiResidual = new double[n];
        int jacobiIterations = 0;
        for (int sweep = 0; sweep < 500; sweep++) {
            jacobi = smooth(jacobi, rhs, 1, 2.0 / 3.0);
            jacobiIterations++;
            jacobiResidual = residual(jacobi, rhs);
            if (norm(jacobiResidual) / initialResidual < 1e-10) break;
        }
        check(jacobiIterations > 100, "plain Jacobi needs many more sweeps: " + jacobiIterations);
        check(history2.size() * 4 < jacobiIterations,
                "multigrid is an order of magnitude cheaper: " + history2.size() + " cycles");

        // 4. Grid independence: the reduction factor barely moves with the grid size.
        double factor31 = firstReductionFactor(31);
        double factor63 = firstReductionFactor(63);
        double factor127 = firstReductionFactor(127);
        check(factor31 < 0.25 && factor63 < 0.25 && factor127 < 0.25,
                "all grids reduce fast: " + factor31 + " / " + factor63 + " / " + factor127);
        check(Math.abs(factor31 - factor127) < 0.15,
                "the factor is nearly grid independent: " + factor31 + " vs " + factor127);
        for (int size : new int[]{1, 3, 7}) {
            double[] smallB = new double[size];
            for (int i = 0; i < size; i++) smallB[i] = 1 + i;
            List<Double> smallHistory = new ArrayList<>();
            double[] smallSolution = toArray(solve(smallB, 1e-12, 30, smallHistory));
            double[] smallDirect = directSolve(poissonOperator(size), smallB);
            double error = 0;
            for (int i = 0; i < size; i++) {
                error = Math.max(error, Math.abs(smallSolution[i] - smallDirect[i]));
            }
            check(error < 1e-10, "size " + size + " matches the direct solve");
        }

        // 5. Determinism.
        List<Double> firstRun = new ArrayList<>();
        List<Double> secondRun = new ArrayList<>();
        double[] a = toArray(solve(rhs, 1e-10, 50, firstRun));
        double[] b2 = toArray(solve(rhs, 1e-10, 50, secondRun));
        boolean identical = firstRun.size() == secondRun.size();
        for (int i = 0; identical && i < firstRun.size(); i++) {
            if (!firstRun.get(i).equals(secondRun.get(i))) identical = false;
        }
        check(identical, "the V-cycle history is reproducible: " + firstRun.size());
        check(Arrays.equals(a, b2), "the solution is reproducible");
        check(norm(residual(multigrid, rhs)) / initialResidual < 1e-10, "final residual is tiny");

        System.out.println("PASS");
    }

    /** Residual reduction factor after the first V-cycle for a given grid size. */
    static double firstReductionFactor(int n) {
        double[] b = new double[n];
        for (int i = 0; i < n; i++) b[i] = Math.sin((i + 1) * Math.PI / (n + 1));
        List<Double> history = new ArrayList<>();
        solve(b, 1e-3, 1, history);
        return history.get(0);
    }

    static double inner(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    static double[] toArray(List<Double> values) {
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
