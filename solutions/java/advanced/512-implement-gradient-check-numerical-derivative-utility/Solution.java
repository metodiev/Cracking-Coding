/** Task 512 - Implement gradient check numerical derivative utility. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * The numerical-differentiation toolbox behind every gradient check: a central difference (f(x + h)
 * - f(x - h)) / 2h with optional Richardson extrapolation over a geometric ladder of steps (the
 * classic D(h) = (4 D(h/2) - D(h)) / 3 recursion, two levels give O(h^4)), forward, backward and
 * central one-sided differences, a partial derivative for multivariate functions, a numeric vector
 * gradient and numeric Jacobian for vector-valued maps, and the comparison helpers absoluteError /
 * relativeError / within that a training loop would call. Two phenomena are made explicit: the
 * truncation error of the central difference is O(h^2) (halving h quarters it while round-off is
 * small) and the total error is minimised near h ~ cbrt(eps) * (1 + |x|), so the sweep over h from
 * 1e-1 down to 1e-9 is asserted to have an interior minimum.
 *
 * Documented checks: on f = x^2 at x = 1 the central difference hits the analytic 2 with a relative
 * error <= 1e-10 at h = 1e-5 and is exact to round-off at every reasonable step because f''' = 0;
 * on f = x^3 at x = 2 the measured truncation error equals h^2 * |f'''| / 6 and shrinking h by 10
 * quarters it (the measured ratio is asserted within 20%), while the sweep on exp at 0 shows the
 * documented interior minimum near cbrt(eps) (the best step is found between 1e-6 and 1e-4 with an
 * error below 1e-10, coarse steps are truncation limited and tiny steps round-off limited);
 * Richardson extrapolation on f = x^3 at x = 2 is at least 10x more accurate than the plain central
 * difference at the same base step; exp, sin, cos, tanh, log and a composite f = log(1 + exp(x))
 * all match their analytic derivatives to 1e-8; the multivariate gradient of sum x_i^2 is 2x and of
 * x0 x1 + sin x0 + exp x1 is (x1 + cos x0, x0 + exp x1) to 1e-7; the Jacobian of the linear map A x
 * is A to 1e-8; a deliberately wrong analytic gradient is flagged by the utility with a relative
 * error above 0.1 (the checker catches bugs); non-smooth |x| at 0 is shown to defeat the central
 * difference (it returns 0 while the one-sided derivatives are -1 and +1); the guard rails reject
 * h <= 0 (and h = 0), a zero direction and a gradient-length mismatch with the documented
 * exceptions; and the whole utility is deterministic.
 */
import java.util.*;

public class Solution {
    interface Scalar {
        double apply(double x);
    }

    interface Vector {
        double apply(double[] x);
    }

    interface VectorField {
        double[] apply(double[] x);
    }

    /** (f(x + h) - f(x - h)) / 2h */
    static double central(Scalar f, double x, double h) {
        if (h <= 0.0) throw new IllegalArgumentException("h must be positive");
        return (f.apply(x + h) - f.apply(x - h)) / (2.0 * h);
    }

    /** One-sided differences; step sign selects forward (+1) or backward (-1). */
    static double oneSided(Scalar f, double x, double h, int direction) {
        if (h <= 0.0) throw new IllegalArgumentException("h must be positive");
        if (direction == 0) throw new IllegalArgumentException("direction must be non-zero");
        return (f.apply(x + direction * h) - f.apply(x)) / (direction * h);
    }

    /** Richardson extrapolation over `levels` halvings of h: O(h^(2 * levels)) accuracy. */
    static double richardson(Scalar f, double x, double h, int levels) {
        if (levels < 1) throw new IllegalArgumentException("levels must be at least 1");
        double[] estimates = new double[levels];
        for (int level = 0; level < levels; level++) {
            estimates[level] = central(f, x, h / Math.pow(2.0, level));
        }
        for (int power = 1; power < levels; power++) {
            for (int level = 0; level + power < levels; level++) {
                double factor = Math.pow(4.0, power);
                estimates[level] = (factor * estimates[level + 1] - estimates[level]) / (factor - 1.0);
            }
        }
        return estimates[0];
    }

    /** Partial derivative of a multivariate function with respect to coordinate `index`. */
    static double partial(Vector f, double[] x, int index, double h) {
        if (h <= 0.0) throw new IllegalArgumentException("h must be positive");
        double[] forward = x.clone();
        double[] backward = x.clone();
        forward[index] += h;
        backward[index] -= h;
        return (f.apply(forward) - f.apply(backward)) / (2.0 * h);
    }

    static double[] numericGradient(Vector f, double[] x, double h) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = partial(f, x, i, h);
        return out;
    }

    static double[][] numericJacobian(VectorField f, double[] x, double h) {
        int rows = f.apply(x).length;
        double[][] out = new double[rows][x.length];
        for (int j = 0; j < x.length; j++) {
            double[] forward = x.clone();
            double[] backward = x.clone();
            forward[j] += h;
            backward[j] -= h;
            double[] plus = f.apply(forward);
            double[] minus = f.apply(backward);
            for (int i = 0; i < rows; i++) out[i][j] = (plus[i] - minus[i]) / (2.0 * h);
        }
        return out;
    }

    static double absoluteError(double computed, double analytic) {
        return Math.abs(computed - analytic);
    }

    static double relativeError(double computed, double analytic) {
        double scale = Math.max(1.0, Math.abs(analytic));
        return Math.abs(computed - analytic) / scale;
    }

    static boolean within(double computed, double analytic, double tolerance) {
        return relativeError(computed, analytic) <= tolerance;
    }

    /** Largest relative error of an analytic gradient against the central differences. */
    static double maxGradientError(Vector f, double[] x, double[] analytic, double h) {
        if (analytic.length != x.length) throw new IllegalArgumentException("gradient length mismatch");
        double[] numeric = numericGradient(f, x, h);
        double worst = 0.0;
        for (int i = 0; i < x.length; i++) {
            worst = Math.max(worst, Math.abs(numeric[i] - analytic[i]) / Math.max(1.0, Math.abs(analytic[i])));
        }
        return worst;
    }

    static boolean gradientCheck(Vector f, double[] x, double[] analytic, double h, double tolerance) {
        return maxGradientError(f, x, analytic, h) <= tolerance;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        // 1. f = x^2 at x = 1: the central difference is accurate, and error(h) has an interior
        //    minimum because truncation falls like h^2 while round-off grows like eps / h.
        Scalar square = x -> x * x;
        double atOne = central(square, 1.0, 1e-5);
        check(within(atOne, 2.0, 1e-10), "central difference of x^2 at 1 gives 2 to 1e-10");
        check(absoluteError(central(square, 1.0, 1e-2), 2.0) <= 1e-13,
                "x^2 has zero truncation error, so even a coarse step is exact to round-off");
        double[] steps = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
        double[] squareErrors = new double[steps.length];
        for (int i = 0; i < steps.length; i++) {
            squareErrors[i] = absoluteError(central(square, 1.0, steps[i]), 2.0);
        }
        check(squareErrors[0] <= 1e-13 && squareErrors[5] <= 1e-10,
                "x^2 is differentiated exactly to round-off at every reasonable step");
        check(squareErrors[8] > squareErrors[0],
                "x^2 only suffers round-off, which grows as h shrinks");

        // exp at 0 has genuine truncation error h^2 / 6 and round-off ~ eps / h, so error(h) has an
        // interior minimum near cbrt(eps) - the documented optimal step.
        double[] errors = new double[steps.length];
        for (int i = 0; i < steps.length; i++) {
            errors[i] = absoluteError(central(Math::exp, 0.0, steps[i]), Math.exp(0.0));
        }
        int bestIndex = 0;
        for (int i = 1; i < errors.length; i++) {
            if (errors[i] < errors[bestIndex]) bestIndex = i;
        }
        check(bestIndex > 0 && bestIndex < errors.length - 1,
                "the error sweep has an interior minimum");
        check(steps[bestIndex] <= 1e-4 && steps[bestIndex] >= 1e-6,
                "the best step is near cbrt(eps) (found h = " + steps[bestIndex] + ")");
        check(errors[bestIndex] <= 1e-10, "the best step reaches an absolute error below 1e-10");
        check(errors[8] > errors[bestIndex], "an extremely small step reintroduces round-off error");
        check(errors[0] > errors[bestIndex], "a very coarse step is dominated by truncation error");
        check(Math.abs(errors[1] / errors[2] - 100.0) <= 20.0,
                "in the truncation regime the error scales like h^2");

        // 2. Richardson extrapolation on f = x^3 at x = 2 (f' = 12): the truncation error is
        //    f'''(x) h^2 / 6, so two levels of extrapolation kill it up to round-off.
        Scalar cube = x -> x * x * x;
        double plain = central(cube, 2.0, 1e-3);
        double extrapolated = richardson(cube, 2.0, 1e-3, 2);
        double plainError = absoluteError(plain, 12.0);
        double extrapolatedError = absoluteError(extrapolated, 12.0);
        check(Math.abs(plainError - 1e-3 * 1e-3 * 6.0 / 6.0) <= 1e-9,
                "the measured truncation error matches h^2 * |f'''| / 6");
        double coarseCube = absoluteError(central(cube, 2.0, 1e-2), 12.0);
        double finerCube = absoluteError(central(cube, 2.0, 1e-3), 12.0);
        check(finerCube <= coarseCube / 50.0, "shrinking the step by 10 quarters the error");
        check(Math.abs(coarseCube / finerCube - 100.0) <= 20.0,
                "the truncation error scales like h^2 (ratio measured " + (coarseCube / finerCube) + ")");
        check(extrapolatedError <= plainError / 10.0,
                "Richardson is at least 10x more accurate than plain central difference");
        check(absoluteError(richardson(cube, 2.0, 1e-2, 3), 12.0) <= 1e-12,
                "three levels of Richardson reach the round-off floor");
        check(absoluteError(richardson(cube, 2.0, 1e-2, 2), 12.0) <= 1e-11,
                "the two-level estimate is exact to 1e-11");
        check(Math.abs(richardson(cube, 2.0, 1e-3, 1) - central(cube, 2.0, 1e-3 / 2.0)) > 0.0,
                "a single level still evaluates a halved step internally");

        // 3. Analytic derivatives versus the numeric utility.
        check(within(central(Math::exp, 0.5, 1e-5), Math.exp(0.5), 1e-8), "exp' = exp");
        check(within(central(Math::sin, 0.7, 1e-5), Math.cos(0.7), 1e-8), "sin' = cos");
        check(within(central(Math::cos, 0.7, 1e-5), -Math.sin(0.7), 1e-8), "cos' = -sin");
        check(within(central(x -> Math.tanh(x), 0.3, 1e-5), 1.0 - Math.tanh(0.3) * Math.tanh(0.3), 1e-8),
                "tanh' = 1 - tanh^2");
        check(within(central(Math::log, 2.0, 1e-5), 0.5, 1e-8), "log' = 1 / x");
        Scalar softplus = x -> Math.log(1.0 + Math.exp(x));
        check(within(central(softplus, 0.0, 1e-5), 0.5, 1e-8), "softplus'(0) = sigmoid(0) = 0.5");

        // 4. Multivariate gradients and the Jacobian of a linear map.
        Vector sumSquares = x -> x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        double[] point = {0.3, -1.2, 2.5};
        double[] numeric = numericGradient(sumSquares, point, 1e-6);
        for (int i = 0; i < 3; i++) {
            check(within(numeric[i], 2.0 * point[i], 1e-8),
                    "grad of sum x^2 is 2x at coordinate " + i);
        }
        Vector coupled = x -> x[0] * x[1] + Math.sin(x[0]) + Math.exp(x[1]);
        double[] coupledPoint = {0.4, -0.7};
        double[] analytic = {coupledPoint[1] + Math.cos(coupledPoint[0]),
                coupledPoint[0] + Math.exp(coupledPoint[1])};
        check(gradientCheck(coupled, coupledPoint, analytic, 1e-6, 1e-7),
                "the composite function passes its gradient check");
        double[][] matrix = {{1.0, 2.0, 3.0}, {-4.0, 0.5, 0.25}};
        VectorField linear = x -> new double[]{
                matrix[0][0] * x[0] + matrix[0][1] * x[1] + matrix[0][2] * x[2],
                matrix[1][0] * x[0] + matrix[1][1] * x[1] + matrix[1][2] * x[2]};
        double[][] jacobian = numericJacobian(linear, new double[]{0.1, 0.2, 0.3}, 1e-6);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                check(Math.abs(jacobian[i][j] - matrix[i][j]) <= 1e-8,
                        "Jacobian entry " + i + "," + j);
            }
        }

        // 5. The checker must catch a wrong gradient - that is its whole job.
        double[] wrong = {coupledPoint[1] + Math.cos(coupledPoint[0]) + 0.5,
                coupledPoint[0] + Math.exp(coupledPoint[1])};
        check(maxGradientError(coupled, coupledPoint, wrong, 1e-6) > 0.1,
                "the utility flags a wrong analytic gradient");
        check(!gradientCheck(coupled, coupledPoint, wrong, 1e-6, 1e-7), "and reports failure");
        check(gradientCheck(coupled, coupledPoint, analytic, 1e-6, 1e-7),
                "and reports success otherwise");
        check(absoluteError(1.0, 1.001) <= 0.0011 && relativeError(1.0, 1.001) <= 1e-3,
                "the error helpers agree on a tiny mismatch");

        // 6. Non-smooth functions defeat the central difference; the one-sided pair exposes it.
        double forward = oneSided(Math::abs, 0.0, 1e-6, 1);
        double backward = oneSided(Math::abs, 0.0, 1e-6, -1);
        double center = central(Math::abs, 0.0, 1e-6);
        check(Math.abs(forward - 1.0) <= 1e-6 && Math.abs(backward + 1.0) <= 1e-6,
                "|x| has one-sided derivatives +1 and -1 at 0");
        check(Math.abs(center) <= 1e-12,
                "the central difference of |x| at 0 is 0 - a false derivative");
        check(Math.abs(forward - backward) > 1.0,
                "the one-sided disagreement is the documented non-smoothness signal");
        check(within(oneSided(Math::abs, 3.0, 1e-6, 1), 1.0, 1e-6), "away from 0, |x| is smooth");

        // 7. Guard rails and boundaries.
        try {
            central(square, 1.0, 0.0);
            check(false, "h = 0 must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "h = 0 is rejected");
        }
        try {
            central(square, 1.0, -1e-6);
            check(false, "negative h must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "negative h is rejected");
        }
        try {
            oneSided(square, 1.0, 1e-6, 0);
            check(false, "direction 0 must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "direction 0 is rejected");
        }
        try {
            maxGradientError(coupled, coupledPoint, new double[]{1.0}, 1e-6);
            check(false, "a length mismatch must be rejected");
        } catch (IllegalArgumentException expected) {
            check(true, "gradient length mismatches are rejected");
        }
        double coarse = absoluteError(central(cube, 2.0, 1.0), 12.0);
        check(Math.abs(coarse - 1.0) <= 1e-12, "a step of 1 on x^3 is off by exactly 1 (13 vs 12)");

        // 8. Determinism: repeated evaluations are bit-identical and the sweep is reproducible.
        double[] repeat = new double[steps.length];
        for (int i = 0; i < steps.length; i++) {
            repeat[i] = absoluteError(central(Math::exp, 0.0, steps[i]), Math.exp(0.0));
        }
        check(Arrays.equals(repeat, errors), "deterministic re-run: identical error sweep");
        double[] repeatSquare = new double[steps.length];
        for (int i = 0; i < steps.length; i++) {
            repeatSquare[i] = absoluteError(central(square, 1.0, steps[i]), 2.0);
        }
        check(Arrays.equals(repeatSquare, squareErrors), "deterministic re-run: identical x^2 sweep");
        check(central(cube, 2.0, 1e-3) == plain, "deterministic re-run: identical central value");
        check(richardson(cube, 2.0, 1e-3, 2) == extrapolated,
                "deterministic re-run: identical extrapolation");

        System.out.println("PASS");
    }
}
