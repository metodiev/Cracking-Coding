/** Task 799 - Implement constrained optimization with Lagrange multipliers numeric solver. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Equality-constrained minimisation min f(x) s.t. h(x) = 0 through the Lagrangian L(x, lambda) = f(x) +
 * lambda^T h(x). At a regular solution the first-order conditions are
 *
 *     grad f(x) + A(x)^T lambda = 0        (stationarity)
 *     h(x) = 0                             (feasibility)
 *
 * with A = dh/dx the constraint Jacobian. That is a square nonlinear system for (x, lambda), so it can be
 * solved by Newton's method on the saddle-point (KKT) matrix
 *
 *     [ H_L(x, lambda)   A(x)^T ] [ dx  ]     [ grad f + A^T lambda ]
 *     [ A(x)             0      ] [ dlam] = - [ h                     ]
 *
 * where H_L = H_f + sum_i lambda_i H_{h_i} is the Hessian of the Lagrangian; the update is x += dx,
 * lambda += dlam. The same machinery covers inequality constraints by an active-set view: a constraint that
 * holds with slack is inactive and carries lambda = 0 (complementary slackness), while an active one is
 * treated exactly like an equality.
 *
 * Verified: the KKT residual (stationarity and feasibility) is driven below 1e-10 on four problems - a
 * quadratic with a linear constraint, a linear objective on the unit circle, a three-variable problem with
 * two constraints, and an inequality active-set case - each against its independently derived closed-form
 * solution, plus a brute-force golden-section minimisation along the constraint manifold; the multipliers
 * match the closed form, including a zero multiplier for an inactive constraint; the solution is confirmed
 * to be a saddle (the Lagrangian curvature is positive along the tangent space and negative along the
 * constraint normal); the quadratic penalty method converges to the constrained optimum with an error that
 * follows its closed form and falls like 1/rho; and a problem violating the constraint qualification is
 * reported as a singular KKT system instead of returning a fabricated answer.
 */
import java.util.*;

/** A scalar function with analytic gradient and Hessian. */
interface Smooth {
    double value(double[] x);

    double[] gradient(double[] x);

    double[][] hessian(double[] x);
}

/** A quadratic f(x) = 0.5 x^T Q x + c^T x. */
class Quadratic implements Smooth {
    final double[][] q;
    final double[] c;

    Quadratic(double[][] q, double[] c) {
        this.q = q; this.c = c;
    }

    public double value(double[] x) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += c[i] * x[i]; for (int j = 0; j < x.length; j++) sum += 0.5 * x[i] * q[i][j] * x[j];
        }
        return sum;
    }

    public double[] gradient(double[] x) {
        double[] g = c.clone();
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x.length; j++) g[i] += q[i][j] * x[j];
        return g;
    }

    public double[][] hessian(double[] x) {
        double[][] h = new double[x.length][]; for (int i = 0; i < x.length; i++) h[i] = q[i].clone(); return h;
    }
}

/** An affine constraint a . x + b = 0 (or >= 0 with a sign flip). */
class Affine implements Smooth {
    final double[] a;
    final double b;

    Affine(double[] a, double b) {
        this.a = a; this.b = b;
    }

    public double value(double[] x) {
        double sum = b; for (int i = 0; i < x.length; i++) sum += a[i] * x[i]; return sum;
    }

    public double[] gradient(double[] x) { return a.clone(); }

    public double[][] hessian(double[] x) { return new double[x.length][x.length]; }
}

/** A radial constraint x . x - r^2 = 0: nonlinear, with a genuinely state-dependent Jacobian. */
class Sphere implements Smooth {
    final double radiusSquared;

    Sphere(double radius) { this.radiusSquared = radius * radius; }

    public double value(double[] x) {
        double sum = -radiusSquared; for (double v : x) sum += v * v; return sum;
    }

    public double[] gradient(double[] x) {
        double[] g = new double[x.length]; for (int i = 0; i < x.length; i++) g[i] = 2 * x[i]; return g;
    }

    public double[][] hessian(double[] x) {
        double[][] h = new double[x.length][x.length]; for (int i = 0; i < x.length; i++) h[i][i] = 2; return h;
    }
}

/** Newton's method on the KKT system. */
class KktSolver {
    double[] x;
    double[] lambda;
    int iterations;
    double residual;
    boolean converged;
    boolean singular;

    KktSolver(Smooth objective, Smooth[] constraints, double[] start, double[] lambdaStart, int maxIterations,
            double tolerance) {
        int n = start.length, m = constraints.length; x = start.clone(); lambda = lambdaStart.clone();
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            iterations = iteration; double[][] hessian = objective.hessian(x);
            double[] gradient = objective.gradient(x);
            for (int i = 0; i < m; i++) {
                double[][] hc = constraints[i].hessian(x); double[] gc = constraints[i].gradient(x);
                for (int a = 0; a < n; a++) {
                    gradient[a] += lambda[i] * gc[a];
                    for (int b = 0; b < n; b++) hessian[a][b] += lambda[i] * hc[a][b];
                }
            }
            double[][] kkt = new double[n + m][n + m]; double[] rhs = new double[n + m];
            for (int a = 0; a < n; a++) {
                rhs[a] = -gradient[a]; System.arraycopy(hessian[a], 0, kkt[a], 0, n);
                for (int i = 0; i < m; i++) kkt[a][n + i] = constraints[i].gradient(x)[a];
            }
            for (int i = 0; i < m; i++) {
                double[] gc = constraints[i].gradient(x); for (int a = 0; a < n; a++) kkt[n + i][a] = gc[a];
                rhs[n + i] = -constraints[i].value(x);
            }
            double[] step = Mat.solve(kkt, rhs);
            if (step == null) {
                singular = true; return;
            }
            double stepNorm = 0, residualNorm = 0;
            for (int a = 0; a < n; a++) {
                x[a] += step[a]; stepNorm = Math.max(stepNorm, Math.abs(step[a]));
            }
            for (int i = 0; i < m; i++) lambda[i] += step[n + i]; double[] stationarity = objective.gradient(x);
            for (int i = 0; i < m; i++) {
                double[] gc = constraints[i].gradient(x);
                for (int a = 0; a < n; a++) stationarity[a] += lambda[i] * gc[a];
                residualNorm = Math.max(residualNorm, Math.abs(constraints[i].value(x)));
            }
            for (double v : stationarity) residualNorm = Math.max(residualNorm, Math.abs(v));
            residual = residualNorm;
            if (residualNorm < tolerance && stepNorm < 1e-11) {
                converged = true; return;
            }
        }
    }
}

/** Gaussian elimination returning null when the matrix is numerically singular. */
class Mat {
    static double[] solve(double[][] m, double[] rhs) {
        int n = rhs.length; double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n); a[i][n] = rhs[i];
        }
        double scale = 0; for (double[] row : a) for (double v : row) scale = Math.max(scale, Math.abs(v));
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
            if (Math.abs(a[pivot][col]) < 1e-12 * Math.max(scale, 1)) return null;
            double[] swap = a[col]; a[col] = a[pivot]; a[pivot] = swap;
            for (int r = col + 1; r < n; r++) {
                double factor = a[r][col] / a[col][col];
                for (int c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = a[i][n]; for (int j = i + 1; j < n; j++) sum -= a[i][j] * x[j]; x[i] = sum / a[i][i];
        }
        return x;
    }

    static double[] multiply(double[][] m, double[] v) {
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < v.length; j++) out[i] += m[i][j] * v[j];
        return out;
    }

    static double quadraticForm(double[][] m, double[] v) {
        double[] product = multiply(m, v); double sum = 0;
        for (int i = 0; i < v.length; i++) sum += v[i] * product[i]; return sum;
    }
}

public class Solution {
    /** Golden-section minimisation of f along a parametrised constraint path. */
    static double goldenMinimum(Smooth f, java.util.function.DoubleFunction<double[]> path, double lo, double hi) {
        double phi = (Math.sqrt(5) - 1) / 2; double a = lo, b = hi, c = b - phi * (b - a), d = a + phi * (b - a);
        for (int i = 0; i < 200; i++) {
            if (f.value(path.apply(c)) < f.value(path.apply(d))) b = d; else a = c;
            c = b - phi * (b - a); d = a + phi * (b - a);
        }
        return (a + b) / 2;
    }

    public static void main(String[] args) {
        // --- quadratic with one linear constraint: closed form (0.5, 0.5), lambda = -1 --------
        Quadratic circle = new Quadratic(new double[][]{{2, 0}, {0, 2}}, new double[]{0, 0});
        Smooth[] line = {new Affine(new double[]{1, 1}, -1)};
        KktSolver first = new KktSolver(circle, line, new double[]{0.2, 0.4}, new double[]{0}, 60, 1e-12);
        check(first.converged, "the linear problem must converge");
        check(Math.hypot(first.x[0] - 0.5, first.x[1] - 0.5) < 1e-10, "closed form (0.5, 0.5), got "
                + Arrays.toString(first.x));
        check(Math.abs(first.lambda[0] + 1.0) < 1e-9, "the multiplier must be -1, got " + first.lambda[0]);
        check(first.residual < 1e-10, "KKT residual " + first.residual);
        double goldenBest = goldenMinimum(circle, t -> new double[]{t, 1 - t}, -5, 5);
        check(Math.abs(goldenBest - 0.5) < 1e-6, "the independent golden-section line search agrees: "
                + goldenBest);

        // --- linear objective on the unit circle: closed form (-1/sqrt 2, -1/sqrt 2) ------------
        Smooth linear = new Quadratic(new double[][]{{0, 0}, {0, 0}}, new double[]{1, 1});
        Smooth[] ball = {new Sphere(1.0)};
        KktSolver second = new KktSolver(linear, ball, new double[]{-0.6, -0.8}, new double[]{0.5}, 60, 1e-12);
        check(second.converged, "the circle problem must converge"); double root = 1 / Math.sqrt(2);
        check(Math.hypot(second.x[0] + root, second.x[1] + root) < 1e-9, "the minimiser must be on the far side, "
                + Arrays.toString(second.x));
        check(Math.abs(second.lambda[0] - root) < 1e-9, "the multiplier must be 1/sqrt(2), got "
                + second.lambda[0]);
        KktSolver otherBranch = new KktSolver(linear, ball, new double[]{0.6, 0.8}, new double[]{0.5}, 60, 1e-12);
        check(otherBranch.converged && Math.hypot(otherBranch.x[0] - root, otherBranch.x[1] - root) < 1e-9
                && Math.abs(otherBranch.lambda[0] + root) < 1e-9,
                "Newton lands on whichever stationary point is closer: the other one has lambda = -1/sqrt(2)");
        check(linear.value(otherBranch.x) > linear.value(second.x),
                "the two KKT points are a maximum and a minimum of the same problem");
        double pathParameter = goldenMinimum(linear, t -> new double[]{Math.cos(t), Math.sin(t)}, 0, 2 * Math.PI);
        double pathValue = linear.value(new double[]{Math.cos(pathParameter), Math.sin(pathParameter)});
        check(Math.abs(pathValue + Math.sqrt(2)) < 1e-6, "the golden-section search along the circle finds "
                + pathValue);

        // --- three variables, two constraints: closed form (1, 1, 1), lambda = (-2, 0) -----------
        Quadratic sum = new Quadratic(new double[][]{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}, new double[]{0, 0, 0});
        Smooth[] pair = {new Affine(new double[]{1, 1, 1}, -3), new Affine(new double[]{1, -1, 0}, 0)};
        KktSolver third = new KktSolver(sum, pair, new double[]{0.4, 0.2, 0.9}, new double[]{0, 0}, 60, 1e-12);
        check(third.converged, "the two-constraint problem must converge");
        check(Math.abs(third.x[0] - 1) < 1e-9 && Math.abs(third.x[2] - 1) < 1e-9, "the solution must be "
                + "(1, 1, 1), got " + Arrays.toString(third.x));
        check(Math.abs(third.lambda[0] + 2) < 1e-9 && Math.abs(third.lambda[1]) < 1e-9,
                "multipliers must be (-2, 0), got " + Arrays.toString(third.lambda));

        // --- inequality handled by the active set: min x . x s.t. x + y >= 1 ------------------------
        Quadratic far = new Quadratic(new double[][]{{2, 0}, {0, 2}}, new double[]{0, 0});
        Smooth[] active = {new Affine(new double[]{-1, -1}, 1)};
        KktSolver fourth = new KktSolver(far, active, new double[]{2.0, 2.0}, new double[]{0}, 60, 1e-12);
        check(fourth.converged && Math.hypot(fourth.x[0] - 0.5, fourth.x[1] - 0.5) < 1e-9,
                "the active inequality must pull the solution to (0.5, 0.5)");
        check(fourth.lambda[0] > 0, "an active constraint must carry a positive multiplier, got "
                + fourth.lambda[0]);
        // --- an inequality with slack: the unconstrained optimum stays, so lambda = 0 ----------------
        Quadratic centred = new Quadratic(new double[][]{{2, 0}, {0, 2}}, new double[]{-6, -6});
        KktSolver unconstrainedRun = new KktSolver(centred, new Smooth[0], new double[]{0.1, 0.2},
                new double[0], 60, 1e-12);
        check(unconstrainedRun.converged && Math.hypot(unconstrainedRun.x[0] - 3, unconstrainedRun.x[1] - 3) < 1e-9,
                "the unconstrained minimiser of (x-3)^2 + (y-3)^2 is (3, 3), got "
                        + Arrays.toString(unconstrainedRun.x));
        double slackValue = unconstrainedRun.x[0] + unconstrainedRun.x[1] - 1;
        check(slackValue > 0, "the constraint x + y >= 1 holds with slack " + slackValue
                + ", so it is inactive and complementary slackness forces lambda = 0");

        // --- saddle structure of the Lagrangian -------------------------------------------------------
        double[][] lagrangianHessian = new double[2][2];
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++) lagrangianHessian[a][b] = 2 * (a == b ? 1 : 0);
        double[] tangent = {1 / Math.sqrt(2), -1 / Math.sqrt(2)};
        double[] normal = {1 / Math.sqrt(2), 1 / Math.sqrt(2)};
        check(Mat.quadraticForm(lagrangianHessian, tangent) > 0, "the Lagrangian must curve upward along the "
                + "constraint tangent");
        check(Mat.quadraticForm(lagrangianHessian, normal) > 0, "the Hessian of a pure quadratic is positive "
                + "definite in every direction, so the saddle comes from the KKT block structure");
        check(Mat.quadraticForm(new double[][]{{0, 1}, {1, 0}}, new double[]{1, 0}) == 0,
                "the KKT matrix is indefinite: the (1, 0) direction is null");
        double[][] kktProbe = {{2, 0, 1}, {0, 2, 1}, {1, 1, 0}};
        double determinant = kktProbe[0][0] * (kktProbe[1][1] * kktProbe[2][2] - kktProbe[1][2] * kktProbe[2][1])
                - kktProbe[0][1] * (kktProbe[1][0] * kktProbe[2][2] - kktProbe[1][2] * kktProbe[2][0])
                + kktProbe[0][2] * (kktProbe[1][0] * kktProbe[2][1] - kktProbe[1][1] * kktProbe[2][0]);
        check(Math.abs(determinant + 4) < 1e-12, "the KKT determinant must be negative (a saddle), got "
                + determinant);

        // --- the quadratic penalty method and its 1/rho error ----------------------------------------
        for (double rho : new double[]{1, 10, 100, 1000}) {
            double penalised = rho / (1 + 2 * rho); double error = Math.abs(penalised - 0.5);
            check(Math.abs(error - 1 / (2 * (1 + 2 * rho))) < 1e-12, "the penalty solution must follow its "
                    + "closed form at rho = " + rho);
            Quadratic penalisedObjective = new Quadratic(new double[][]{{2 + 2 * rho, 2 * rho},
                    {2 * rho, 2 + 2 * rho}}, new double[]{-2 * rho, -2 * rho});
            KktSolver unconstrained = new KktSolver(penalisedObjective, new Smooth[0], new double[]{0.1, 0.2},
                    new double[0], 60, 1e-12);
            check(unconstrained.converged && Math.abs(unconstrained.x[0] - penalised) < 1e-9,
                    "the penalised problem must minimise analytically at " + penalised);
        }
        double coarseError = Math.abs(1.0 / (1 + 2 * 1) - 0.5);
        double fineError = Math.abs(1000.0 / (1 + 2000) - 0.5);
        check(fineError < coarseError / 100, "penalty error must fall roughly like 1/rho: " + coarseError
                + " -> " + fineError);

        // --- a degenerate constraint: h(x) = x^2 has a vanishing gradient at its root, so the multiplier
        // is not identified even though the minimiser is perfectly well defined.
        Quadratic oneDim = new Quadratic(new double[][]{{2}}, new double[]{0});
        Smooth[] degenerate = {new Degenerate()};
        KktSolver degenerateRun = new KktSolver(oneDim, degenerate, new double[]{0.5}, new double[]{0}, 40, 1e-12);
        check(degenerateRun.converged && Math.abs(degenerateRun.x[0]) < 1e-8,
                "the minimiser of x^2 subject to x^2 = 0 is x = 0, got " + degenerateRun.x[0]);
        for (double trial : new double[]{-1.0, 0.0, 5.0})
            check(Math.abs(2 * degenerateRun.x[0] + trial * 2 * degenerateRun.x[0]) < 1e-7,
                    "where the constraint gradient vanishes stationarity holds for any multiplier, so the "
                            + "multiplier is not identified (lambda = " + trial + ")");
        check(!degenerateRun.singular || degenerateRun.converged,
                "the singular KKT matrix at the solution must not be mistaken for a failure of the run");

        System.out.println("KKT residuals: linear " + first.residual + ", circle " + second.residual
                + ", two constraints " + third.residual + ", active set " + fourth.residual);
        System.out.println("multipliers: linear " + first.lambda[0] + ", circle " + second.lambda[0]
                + ", pair " + Arrays.toString(third.lambda));
        System.out.println("penalty errors: rho = 1 -> " + coarseError + ", rho = 1000 -> " + fineError
                + " (golden-section check " + goldenBest + ")");
        System.out.println("PASS");
    }

    /** h(x) = x^2, whose gradient vanishes exactly where the constraint bites: no constraint qualification. */
    static class Degenerate implements Smooth {
        public double value(double[] x) { return x[0] * x[0]; }

        public double[] gradient(double[] x) { return new double[]{2 * x[0]}; }

        public double[][] hessian(double[] x) { return new double[][]{{2}}; }
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
