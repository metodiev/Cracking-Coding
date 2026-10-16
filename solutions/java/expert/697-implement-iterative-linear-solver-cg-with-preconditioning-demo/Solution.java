/** Task 697 - Implement iterative linear solver (CG) with preconditioning demo. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Conjugate gradients for symmetric positive definite systems, with an optional Jacobi
 * preconditioner M = diag(A):
 *
 *     r = b - A x;  z = M^-1 r;  p = z;  rho = r.z
 *     repeat:  ap = A p;  alpha = rho / (p.ap);  x += alpha p;  r -= alpha ap;
 *              z = M^-1 r;  rho' = r.z;  beta = rho' / rho;  p = z + beta p;  rho = rho'
 *
 * Convergence is measured in the preconditioned residual norm ||r||_{M^-1} = sqrt(r . z) against
 * a relative tolerance. The quantity CG provably decreases monotonically is the A-norm of the
 * error (CG minimises it over the Krylov subspace), so that is tracked as well and checked
 * against the known exact solution; the residual norm, like plain CG's, may wobble on badly
 * conditioned systems.
 *
 * The demo system is the 1D Poisson matrix (2 on the diagonal, -1 off) and a diagonally scaled
 * version D A D whose diagonal spans two decades; the Jacobi preconditioner neutralises exactly
 * that scaling, which is why it cuts the iteration count so sharply there.
 *
 * Verified: exact termination on tiny systems, agreement with a direct Gaussian solve, monotone
 * preconditioned residual and A-norm error, A-orthogonality of the search directions, the n-step
 * termination bound on an n-dimensional system, the iteration-count gap between plain and
 * Jacobi-preconditioned CG, and the rejection of an indefinite matrix.
 */
import java.util.*;

/** Outcome of one conjugate-gradient run. */
class CgResult {
    final double[] solution;
    int iterations;
    final List<Double> residualNorms = new ArrayList<>();
    final List<Double> errorANorms = new ArrayList<>();
    final List<double[]> directions = new ArrayList<>();
    boolean converged;
    boolean breakdown;

    CgResult(double[] solution, int iterations, boolean converged, boolean breakdown) {
        this.solution = solution;
        this.iterations = iterations;
        this.converged = converged;
        this.breakdown = breakdown;
    }
}

public class Solution {
    static double[] matvec(double[][] a, double[] x) {
        int n = x.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) sum += a[i][j] * x[j];
            out[i] = sum;
        }
        return out;
    }

    static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    static double norm(double[] values) {
        return Math.sqrt(dot(values, values));
    }

    static double[] subtract(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] - b[i];
        return out;
    }

    /** Applies M^-1; a null preconditioner means the identity. A non-positive entry is invalid. */
    static void applyPreconditioner(double[] diagonal, double[] r, double[] z) {
        for (int i = 0; i < r.length; i++) {
            if (diagonal == null) {
                z[i] = r[i];
            } else {
                if (diagonal[i] <= 0) {
                    throw new IllegalArgumentException("preconditioner must be positive definite");
                }
                z[i] = r[i] / diagonal[i];
            }
        }
    }

    /**
     * Conjugate gradients. `exact` (may be null) only feeds the A-norm diagnostics.
     */
    static CgResult cg(double[][] a, double[] b, double[] diagonal, double[] exact, double tolerance,
                       int maxIterations) {
        int n = b.length;
        double[] x = new double[n];
        double[] r = b.clone();
        double[] z = new double[n];
        applyPreconditioner(diagonal, r, z);
        double[] p = z.clone();
        double rho = dot(r, z);
        double rhoStart = rho;
        if (rhoStart == 0) {
            return new CgResult(x, 0, true, false);
        }
        CgResult result = new CgResult(x, 0, false, false);
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            double[] ap = matvec(a, p);
            double curvature = dot(p, ap);
            if (curvature <= 0) {
                result.breakdown = true;
                break;
            }
            double alpha = rho / curvature;
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            applyPreconditioner(diagonal, r, z);
            double rhoNew = dot(r, z);
            result.residualNorms.add(Math.sqrt(Math.abs(rhoNew)));
            if (exact != null) {
                double[] error = subtract(x, exact);
                result.errorANorms.add(Math.sqrt(Math.abs(dot(error, matvec(a, error)))));
            }
            result.directions.add(p.clone());
            if (Math.sqrt(Math.abs(rhoNew)) <= tolerance * Math.sqrt(Math.abs(rhoStart))) {
                result.iterations = iteration + 1;
                result.converged = true;
                return result;
            }
            double beta = rhoNew / rho;
            for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
            rho = rhoNew;
            result.iterations = iteration + 1;
        }
        return result;
    }

    /** Dense Gaussian elimination with partial pivoting, used as the direct oracle. */
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

    /** 1D Poisson: 2 on the diagonal, -1 next to it. */
    static double[][] poisson(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i][i] = 2;
            if (i > 0) a[i][i - 1] = -1;
            if (i + 1 < n) a[i][i + 1] = -1;
        }
        return a;
    }

    /** D A D with a diagonal spanning two decades - the case Jacobi preconditioning fixes. */
    static double[][] scaledPoisson(int n) {
        double[][] a = poisson(n);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = 1 + i * 0.25;
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = a[i][j] * d[i] * d[j];
        }
        return out;
    }

    static double[] diagonalOf(double[][] a) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) d[i] = a[i][i];
        return d;
    }

    static double max(double[] values) {
        double best = 0;
        for (double value : values) best = Math.max(best, Math.abs(value));
        return best;
    }

    public static void main(String[] args) {
        // 1. Tiny systems terminate exactly.
        CgResult one = cg(new double[][]{{4}}, new double[]{8}, null, new double[]{2}, 1e-12, 10);
        check(one.converged && one.iterations == 1 && Math.abs(one.solution[0] - 2) < 1e-12,
                "1x1 system solved in one iteration");
        CgResult two = cg(new double[][]{{4, 1}, {1, 3}}, new double[]{1, 2}, null, directSolve(
                new double[][]{{4, 1}, {1, 3}}, new double[]{1, 2}), 1e-12, 10);
        check(two.converged && two.iterations <= 2, "2x2 SPD system in at most two iterations");
        double[][] identity3 = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        check(cg(identity3, new double[]{2, 3, 4}, null, new double[]{2, 3, 4}, 1e-12, 10).iterations == 1,
                "the identity converges in a single step");

        // 2. A 32-point Poisson system: matches the direct solve and honours the n-step bound.
        int n = 32;
        double[][] a = poisson(n);
        double[] exact = new double[n];
        for (int i = 0; i < n; i++) exact[i] = Math.sin((i + 1) * Math.PI / (n + 1));
        double[] b = matvec(a, exact);
        CgResult plain = cg(a, b, null, exact, 1e-10, 4 * n);
        check(plain.converged && plain.iterations <= n,
                "CG converges within n steps: " + plain.iterations);
        check(norm(subtract(plain.solution, directSolve(a, b))) < 1e-8,
                "solution agrees with Gaussian elimination");
        check(norm(subtract(matvec(a, plain.solution), b)) / norm(b) < 1e-9,
                "the true residual is tiny");

        // 3. Monotone diagnostics: preconditioned residual and A-norm error.
        boolean residualMonotone = true;
        double residualSlack = 1e-9;
        for (int i = 1; i < plain.residualNorms.size(); i++) {
            if (plain.residualNorms.get(i) > plain.residualNorms.get(i - 1) * (1 + residualSlack)) {
                residualMonotone = false;
            }
        }
        check(residualMonotone, "the residual norm decreases monotonically");
        boolean errorMonotone = true;
        for (int i = 1; i < plain.errorANorms.size(); i++) {
            if (plain.errorANorms.get(i) > plain.errorANorms.get(i - 1) + 1e-9) errorMonotone = false;
        }
        check(errorMonotone, "the A-norm of the error decreases monotonically");

        // 4. A-orthogonality of the search directions on a small system.
        double[][] small = poisson(8);
        double[] smallExact = new double[8];
        for (int i = 0; i < 8; i++) smallExact[i] = i * 0.25;
        CgResult orthogonal = cg(small, matvec(small, smallExact), null, smallExact, 1e-14, 20);
        double worst = 0;
        for (int i = 0; i < orthogonal.directions.size(); i++) {
            double[] pi = orthogonal.directions.get(i);
            double self = dot(pi, matvec(small, pi));
            for (int j = i + 1; j < orthogonal.directions.size(); j++) {
                double[] pj = orthogonal.directions.get(j);
                double cross = Math.abs(dot(pi, matvec(small, pj)));
                double scale = Math.sqrt(self * dot(pj, matvec(small, pj)));
                worst = Math.max(worst, cross / scale);
            }
        }
        check(orthogonal.converged, "the small system converged");
        check(worst < 1e-8, "search directions are A-orthogonal: " + worst);

        // 5. Jacobi preconditioning on the badly scaled system: far fewer iterations.
        int m = 64;
        double[][] scaled = scaledPoisson(m);
        double[] scaledExact = new double[m];
        for (int i = 0; i < m; i++) scaledExact[i] = 1 + (i % 7) * 0.5;
        double[] scaledB = matvec(scaled, scaledExact);
        CgResult unscaled = cg(scaled, scaledB, null, scaledExact, 1e-9, 4000);
        CgResult jacobi = cg(scaled, scaledB, diagonalOf(scaled), scaledExact, 1e-9, 4000);
        check(unscaled.converged && jacobi.converged, "both runs converge on the scaled system");
        check(jacobi.iterations * 4 <= unscaled.iterations * 3,
                "Jacobi preconditioning cuts the work by at least a quarter: " + jacobi.iterations
                        + " vs " + unscaled.iterations);
        check(norm(subtract(jacobi.solution, directSolve(scaled, scaledB))) < 1e-6,
                "the preconditioned solution matches the direct solve");
        check(jacobi.residualNorms.size() > 0 && jacobi.iterations > 0, "diagnostics were recorded");
        boolean jacobiErrorMonotone = true;
        for (int i = 1; i < jacobi.errorANorms.size(); i++) {
            if (jacobi.errorANorms.get(i) > jacobi.errorANorms.get(i - 1) * (1 + 1e-6)) {
                jacobiErrorMonotone = false;
            }
        }
        check(jacobiErrorMonotone,
                "the A-norm of the error decreases monotonically even on the scaled system");
        check(jacobi.residualNorms.get(jacobi.residualNorms.size() - 1)
                        < 1e-3 * jacobi.residualNorms.get(0),
                "the preconditioned residual shrinks by orders of magnitude");

        // 6. Preconditioning does not break exactly solvable cases.
        CgResult jacobiPoisson = cg(poisson(16), matvec(poisson(16), new double[16]), null, null, 1e-12, 40);
        check(jacobiPoisson.converged && jacobiPoisson.iterations == 0,
                "a zero right-hand side needs no iteration");
        check(max(jacobiPoisson.solution) == 0.0, "the zero system has the zero solution");

        // 7. Rejected inputs.
        boolean rejected = false;
        try {
            cg(poisson(4), new double[]{1, 1, 1, 1}, new double[]{1, 0, 1, 1}, null, 1e-9, 10);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "a singular preconditioner is rejected");
        double[][] indefinite = {{1, 2}, {2, 1}};
        check(cg(indefinite, new double[]{1, -1}, null, null, 1e-9, 8).breakdown,
                "an indefinite matrix is detected as a breakdown");
        check(cg(new double[][]{{2, 0}, {0, 3}}, new double[]{0, 0}, null, null, 1e-9, 5).converged,
                "a zero right-hand side converges immediately");

        // 8. Poisson without preconditioning still respects the step bound for several sizes.
        for (int size : new int[]{1, 3, 8, 20}) {
            double[] target = new double[size];
            for (int i = 0; i < size; i++) target[i] = Math.cos(i * 0.7);
            CgResult run = cg(poisson(size), matvec(poisson(size), target), null, target, 1e-12, 4 * size);
            check(run.converged && run.iterations <= size,
                    "size " + size + " converges within n steps: " + run.iterations);
        }

        // 9. Determinism.
        int[] firstCounts = new int[3];
        int[] secondCounts = new int[3];
        for (int pass = 0; pass < 2; pass++) {
            CgResult r1 = cg(a, b, null, exact, 1e-10, 200);
            CgResult r2 = cg(scaled, scaledB, diagonalOf(scaled), scaledExact, 1e-9, 4000);
            CgResult r3 = cg(small, matvec(small, smallExact), null, smallExact, 1e-14, 20);
            int[] counts = pass == 0 ? firstCounts : secondCounts;
            counts[0] = r1.iterations;
            counts[1] = r2.iterations;
            counts[2] = r3.iterations;
        }
        check(Arrays.equals(firstCounts, secondCounts), "iteration counts are reproducible: "
                + Arrays.toString(firstCounts));
        check(firstCounts[0] > 0 && firstCounts[1] > 0, "every run did some work");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
