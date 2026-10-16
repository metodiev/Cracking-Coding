// Task 512 - Implement gradient check numerical derivative utility. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// A small numerical-differentiation toolbox and the gradient checker that machine-learning code
// uses to validate analytic derivatives. Provided schemes: forward difference (O(h)), central
// difference (O(h^2)), the five-point stencil (O(h^4)), Richardson extrapolation of two central
// differences, a central second difference, a numerical Jacobian for vector-valued functions and a
// directional derivative along a unit vector. The checker compares an analytic gradient with the
// numerical one component by component using the documented relative error
// |numeric - analytic| / max(1, |numeric|, |analytic|) (the max(1, ...) denominator keeps the check
// meaningful for near-zero components) and reports the worst component.
// Documented checks, all against f(x) = exp(x) at x = 1 where the true derivative is e:
//   * the central error shrinks by the predicted factor 1/16 when h is divided by four
//     (measured ratio 0.0625) while the forward error shrinks by about 1/4 - the measured ratios are
//     asserted to lie inside those bands;
//   * the five-point stencil and Richardson extrapolation at h = 1e-2 are at least 10x (resp. 20x)
//     more accurate than the plain central difference at the same h;
//   * the optimal step found by scanning h in [1e-1, 1e-14] lands in [1e-8, 1e-4] with an error below
//     1e-9, and h = 1e-12 is catastrophic: at least 10x worse than h = 1e-6 (cancellation);
//   * analytic gradients of six functions (squared norm, sin*exp + cube, logistic, log-sum-exp,
//     quadratic form, tanh(x0 x1) log(1 + x2^2)) match the central difference at 1e-5 to 1e-6
//     relative, the quadratic form's Hessian is symmetric under mixed second differences, the
//     directional derivative equals grad . d, and the checker flags a deliberately perturbed
//     gradient at 1e-3 while accepting the exact one.
using System;

internal static class Numeric
{
    /** Forward difference (f(x + h e_i) - f(x)) / h. */
    internal static double Forward(Func<double[], double> f, double[] x, int index, double h)
    {
        var shifted = (double[])x.Clone();
        shifted[index] += h;
        return (f(shifted) - f(x)) / h;
    }

    /** Central difference (f(x + h e_i) - f(x - h e_i)) / (2h). */
    internal static double Central(Func<double[], double> f, double[] x, int index, double h)
    {
        var plus = (double[])x.Clone();
        var minus = (double[])x.Clone();
        plus[index] += h;
        minus[index] -= h;
        return (f(plus) - f(minus)) / (2.0 * h);
    }

    /** Five-point stencil (-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)) / (12h). */
    internal static double FivePoint(Func<double[], double> f, double[] x, int index, double h)
    {
        var a = (double[])x.Clone();
        var b = (double[])x.Clone();
        var c = (double[])x.Clone();
        var d = (double[])x.Clone();
        a[index] += 2.0 * h;
        b[index] += h;
        c[index] -= h;
        d[index] -= 2.0 * h;
        return (-f(a) + 8.0 * f(b) - 8.0 * f(c) + f(d)) / (12.0 * h);
    }

    /** Richardson extrapolation: (4 * central(h/2) - central(h)) / 3, which cancels the h^2 term. */
    internal static double Richardson(Func<double[], double> f, double[] x, int index, double h)
    {
        return (4.0 * Central(f, x, index, h / 2.0) - Central(f, x, index, h)) / 3.0;
    }

    /** Central second difference (f(x+h) - 2 f(x) + f(x-h)) / h^2. */
    internal static double Second(Func<double[], double> f, double[] x, int index, double h)
    {
        var plus = (double[])x.Clone();
        var minus = (double[])x.Clone();
        plus[index] += h;
        minus[index] -= h;
        return (f(plus) - 2.0 * f(x) + f(minus)) / (h * h);
    }

    /** Mixed second difference d^2 f / (d x_i d x_j). */
    internal static double Mixed(Func<double[], double> f, double[] x, int i, int j, double h)
    {
        var pp = (double[])x.Clone();
        var pm = (double[])x.Clone();
        var mp = (double[])x.Clone();
        var mm = (double[])x.Clone();
        pp[i] += h;
        pp[j] += h;
        pm[i] += h;
        pm[j] -= h;
        mp[i] -= h;
        mp[j] += h;
        mm[i] -= h;
        mm[j] -= h;
        return (f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h * h);
    }

    internal static double Directional(Func<double[], double> f, double[] x, double[] direction, double h)
    {
        var plus = new double[x.Length];
        var minus = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            plus[i] = x[i] + h * direction[i];
            minus[i] = x[i] - h * direction[i];
        }
        return (f(plus) - f(minus)) / (2.0 * h);
    }

    internal static double[] CentralGradient(Func<double[], double> f, double[] x, double h)
    {
        var gradient = new double[x.Length];
        for (int i = 0; i < x.Length; i++) gradient[i] = Central(f, x, i, h);
        return gradient;
    }

    internal static double[][] Jacobian(Func<double[], double[]> g, double[] x, double h)
    {
        double[] baseline = g(x);
        var jacobian = new double[baseline.Length][];
        for (int row = 0; row < baseline.Length; row++) jacobian[row] = new double[x.Length];
        for (int column = 0; column < x.Length; column++)
        {
            var plus = (double[])x.Clone();
            var minus = (double[])x.Clone();
            plus[column] += h;
            minus[column] -= h;
            double[] forwardValues = g(plus);
            double[] backwardValues = g(minus);
            for (int row = 0; row < baseline.Length; row++)
            {
                jacobian[row][column] = (forwardValues[row] - backwardValues[row]) / (2.0 * h);
            }
        }
        return jacobian;
    }
}

internal class GradientReport
{
    internal double WorstRelativeError;
    internal int WorstComponent;
    internal double[] Numeric = Array.Empty<double>();
    internal double[] Analytic = Array.Empty<double>();
    internal bool Passed;
}

internal static class GradientChecker
{
    internal static double RelativeError(double numeric, double analytic)
    {
        return Math.Abs(numeric - analytic) / Math.Max(1.0, Math.Max(Math.Abs(numeric), Math.Abs(analytic)));
    }

    internal static GradientReport Check(Func<double[], double> f, Func<double[], double[]> analytic,
            double[] x, double h, double tolerance)
    {
        var report = new GradientReport
        {
            Numeric = Numeric.CentralGradient(f, x, h),
            Analytic = analytic(x),
        };
        report.WorstComponent = -1;
        for (int i = 0; i < report.Numeric.Length; i++)
        {
            double error = RelativeError(report.Numeric[i], report.Analytic[i]);
            if (report.WorstComponent < 0 || error > report.WorstRelativeError)
            {
                report.WorstRelativeError = error;
                report.WorstComponent = i;
            }
        }
        report.Passed = report.WorstRelativeError <= tolerance;
        return report;
    }
}

internal class TestFunctions
{
    internal static readonly double[][] A = {
        new[] { 4.0, 1.0, 0.0 },
        new[] { 1.0, 3.0, 1.0 },
        new[] { 0.0, 1.0, 2.0 },
    };

    internal static string[] Names =
            { "squared-norm", "sin*exp+cube", "logistic", "log-sum-exp", "quadratic-form", "tanh-product" };

    internal static Func<double[], double>[] Functions()
    {
        return new Func<double[], double>[]
        {
            x => x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
            x => Math.Sin(x[0]) * Math.Exp(x[1]) + x[2] * x[2] * x[2],
            x => 1.0 / (1.0 + Math.Exp(-x[0])) + 1.0 / (1.0 + Math.Exp(-x[1]))
                    + 1.0 / (1.0 + Math.Exp(-x[2])),
            x => LogSumExp(x),
            x => Quadratic(x),
            x => Math.Tanh(x[0] * x[1]) * Math.Log(1.0 + x[2] * x[2]),
        };
    }

    internal static Func<double[], double[]>[] Gradients()
    {
        return new Func<double[], double[]>[]
        {
            x => new[] { 2.0 * x[0], 2.0 * x[1], 2.0 * x[2] },
            x => new[]
            {
                Math.Cos(x[0]) * Math.Exp(x[1]),
                Math.Sin(x[0]) * Math.Exp(x[1]),
                3.0 * x[2] * x[2],
            },
            x => new[]
            {
                Sigmoid(x[0]) * (1.0 - Sigmoid(x[0])),
                Sigmoid(x[1]) * (1.0 - Sigmoid(x[1])),
                Sigmoid(x[2]) * (1.0 - Sigmoid(x[2])),
            },
            x =>
            {
                double total = Math.Exp(x[0]) + Math.Exp(x[1]) + Math.Exp(x[2]);
                return new[]
                {
                    Math.Exp(x[0]) / total,
                    Math.Exp(x[1]) / total,
                    Math.Exp(x[2]) / total,
                };
            },
            x =>
            {
                var gradient = new double[3];
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        gradient[j] += (A[i][j] + A[j][i]) * x[i];
                    }
                }
                return gradient;
            },
            x =>
            {
                double first = Math.Tanh(x[0] * x[1]);
                double logTerm = Math.Log(1.0 + x[2] * x[2]);
                double sech2 = 1.0 - first * first;
                return new[]
                {
                    sech2 * x[1] * logTerm,
                    sech2 * x[0] * logTerm,
                    first * (2.0 * x[2] / (1.0 + x[2] * x[2])),
                };
            },
        };
    }

    internal static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

    internal static double LogSumExp(double[] x)
    {
        double maximum = Math.Max(x[0], Math.Max(x[1], x[2]));
        double total = Math.Exp(x[0] - maximum) + Math.Exp(x[1] - maximum) + Math.Exp(x[2] - maximum);
        return maximum + Math.Log(total);
    }

    internal static double Quadratic(double[] x)
    {
        double total = 0.0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++) total += x[i] * A[i][j] * x[j];
        }
        return total;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- error order of the schemes ---------------------------------------------------------------
        Func<double[], double> exp = x => Math.Exp(x[0]);
        var point = new[] { 1.0 };
        double exact = Math.E;
        double centralCoarse = Math.Abs(Numeric.Central(exp, point, 0, 1e-2) - exact);
        double centralFine = Math.Abs(Numeric.Central(exp, point, 0, 2.5e-3) - exact);
        double forwardCoarse = Math.Abs(Numeric.Forward(exp, point, 0, 1e-2) - exact);
        double forwardFine = Math.Abs(Numeric.Forward(exp, point, 0, 2.5e-3) - exact);
        Check(centralCoarse > 0.0 && forwardCoarse > 0.0, "the coarse errors are non-zero");
        double centralRatio = centralFine / centralCoarse;
        double forwardRatio = forwardFine / forwardCoarse;
        Check(centralRatio >= 0.03 && centralRatio <= 0.12,
                "the central error falls like h^2 (ratio " + centralRatio + ")");
        Check(forwardRatio >= 0.15 && forwardRatio <= 0.4,
                "the forward error falls like h (ratio " + forwardRatio + ")");
        Check(forwardRatio > centralRatio, "the forward scheme is the less accurate one");

        double fivePointError = Math.Abs(Numeric.FivePoint(exp, point, 0, 1e-2) - exact);
        Check(fivePointError <= 0.1 * centralCoarse, "the five-point stencil beats the central difference");
        double richardsonError = Math.Abs(Numeric.Richardson(exp, point, 0, 1e-2) - exact);
        Check(richardsonError <= 0.05 * centralCoarse, "Richardson beats the central difference");
        Check(fivePointError <= 1e-8, "the five-point stencil is accurate at h = 1e-2");

        // --- the optimal step and the rounding floor ----------------------------------------------------
        double bestError = double.PositiveInfinity;
        double bestH = 0.0;
        for (int exponent = 1; exponent <= 14; exponent++)
        {
            double h = Math.Pow(10.0, -exponent);
            double error = Math.Abs(Numeric.Central(exp, point, 0, h) - exact);
            if (error < bestError)
            {
                bestError = error;
                bestH = h;
            }
        }
        Check(bestH >= 1e-8 && bestH <= 1e-4, "the optimal central step is " + bestH);
        Check(bestError <= 1e-9, "the optimal step reaches an error below 1e-9");
        double tiny = Math.Abs(Numeric.Central(exp, point, 0, 1e-12) - exact);
        double moderate = Math.Abs(Numeric.Central(exp, point, 0, 1e-6) - exact);
        Check(tiny >= 10.0 * moderate, "catastrophic cancellation at h = 1e-12 (" + tiny + " vs " + moderate + ")");

        // --- second derivatives -----------------------------------------------------------------------------
        Func<double[], double> square = x => x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1];
        Check(Math.Abs(Numeric.Second(square, new[] { 0.4, -0.7 }, 0, 1e-4) - 2.0) <= 1e-6,
                "the second difference recovers d2/dx0^2 = 2");
        Check(Math.Abs(Numeric.Second(square, new[] { 0.4, -0.7 }, 1, 1e-4) - 4.0) <= 1e-6,
                "the second difference recovers d2/dx1^2 = 4");
        double mixedA = Numeric.Mixed(square, new[] { 0.4, -0.7 }, 0, 1, 1e-4);
        double mixedB = Numeric.Mixed(square, new[] { 0.4, -0.7 }, 1, 0, 1e-4);
        Check(Math.Abs(mixedA - 3.0) <= 1e-6, "the mixed second difference is 3");
        Check(Math.Abs(mixedA - mixedB) <= 1e-9, "mixed partials commute numerically");

        // --- analytic gradients against the checker ------------------------------------------------------------
        Func<double[], double>[] functions = TestFunctions.Functions();
        Func<double[], double[]>[] gradients = TestFunctions.Gradients();
        Check(functions.Length == TestFunctions.Names.Length, "every function is named");
        var points = new[]
        {
            new[] { 0.3, -0.4, 0.9 },
            new[] { -1.2, 0.7, 0.25 },
            new[] { 0.0, 0.0, 0.0 },
            new[] { 1.5, 1.5, -2.0 },
        };
        for (int f = 0; f < functions.Length; f++)
        {
            foreach (double[] x in points)
            {
                GradientReport report = GradientChecker.Check(functions[f], gradients[f], x, 1e-5, 1e-6);
                Check(report.Passed, "gradient check of " + TestFunctions.Names[f] + " at (" + x[0] + ","
                        + x[1] + "," + x[2] + ") worst " + report.WorstRelativeError);
                Check(report.WorstComponent >= 0, "the checker reports a worst component");
                Check(report.Numeric.Length == 3 && report.Analytic.Length == 3, "gradient lengths");
            }
        }

        // --- the checker rejects a wrong gradient -----------------------------------------------------------------
        Func<double[], double[]> wrong = x => new[]
        {
            2.0 * x[0], 2.0 * x[1], 2.0 * x[2] + 1e-3,
        };
        GradientReport wrongReport = GradientChecker.Check(functions[0], wrong, points[0], 1e-5, 1e-6);
        Check(!wrongReport.Passed, "a perturbed gradient is rejected");
        Check(wrongReport.WorstComponent == 2, "the worst component is the perturbed one");
        Check(wrongReport.WorstRelativeError > 1e-4 && wrongReport.WorstRelativeError <= 1e-3,
                "the reported error matches the perturbation (" + wrongReport.WorstRelativeError + ")");

        // --- directional derivative and Jacobian --------------------------------------------------------------------
        double[] direction = { 1.0, -2.0, 0.5 };
        double norm = Math.Sqrt(direction[0] * direction[0] + direction[1] * direction[1]
                + direction[2] * direction[2]);
        var unit = new[] { direction[0] / norm, direction[1] / norm, direction[2] / norm };
        for (int f = 0; f < functions.Length; f++)
        {
            double directional = Numeric.Directional(functions[f], points[0], unit, 1e-5);
            double projected = 0.0;
            double[] analytic = gradients[f](points[0]);
            for (int i = 0; i < 3; i++) projected += analytic[i] * unit[i];
            Check(Math.Abs(directional - projected) <= 1e-6,
                    "directional derivative of " + TestFunctions.Names[f]);
        }

        Func<double[], double[]> vectorFunction = x => new[]
        {
            x[0] * x[1],
            Math.Sin(x[0]),
            Math.Exp(x[1]),
        };
        double[][] jacobian = Numeric.Jacobian(vectorFunction, new[] { 0.5, -0.25 }, 1e-6);
        Check(Math.Abs(jacobian[0][0] - (-0.25)) <= 1e-6 && Math.Abs(jacobian[0][1] - 0.5) <= 1e-6,
                "Jacobian row 0");
        Check(Math.Abs(jacobian[1][0] - Math.Cos(0.5)) <= 1e-6 && Math.Abs(jacobian[1][1]) <= 1e-9,
                "Jacobian row 1");
        Check(Math.Abs(jacobian[2][0]) <= 1e-9 && Math.Abs(jacobian[2][1] - Math.Exp(-0.25)) <= 1e-6,
                "Jacobian row 2");

        // --- edge cases ---------------------------------------------------------------------------------------------
        Func<double[], double> constant = x => 7.5;
        GradientReport constantReport = GradientChecker.Check(constant, x => new[] { 0.0, 0.0, 0.0 },
                new[] { 1.0, 2.0, 3.0 }, 1e-5, 1e-9);
        Check(constantReport.Passed, "a constant function has a zero gradient");
        Check(constantReport.Numeric[0] == 0.0, "the numerical gradient of a constant is exactly zero");
        Func<double[], double> linear = x => 3.0 * x[0] - 2.0 * x[1];
        GradientReport linearReport = GradientChecker.Check(linear, x => new[] { 3.0, -2.0, 0.0 },
                new[] { 0.25, 0.75, -1.0 }, 1e-3, 1e-9);
        Check(linearReport.Passed, "a linear function is differentiated exactly");
        Func<double[], double> single = x => Math.Log(x[0]);
        GradientReport singleReport = GradientChecker.Check(single, x => new[] { 1.0 / x[0] },
                new[] { 2.5 }, 1e-6, 1e-6);
        Check(singleReport.Passed && singleReport.Numeric.Length == 1, "the one-dimensional case");

        // --- determinism -----------------------------------------------------------------------------------------------
        GradientReport again = GradientChecker.Check(functions[4], gradients[4], points[3], 1e-5, 1e-6);
        GradientReport once = GradientChecker.Check(functions[4], gradients[4], points[3], 1e-5, 1e-6);
        Check(again.WorstComponent == once.WorstComponent
                && again.WorstRelativeError == once.WorstRelativeError,
                "the checker is deterministic");
        Check(Math.Abs(Numeric.Central(exp, point, 0, 1e-5) - Numeric.Central(exp, point, 0, 1e-5)) == 0.0,
                "repeated evaluations are bit-identical");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
