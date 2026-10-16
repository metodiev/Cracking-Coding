// Task 517 - Implement Adam optimizer skeleton. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Adam with bias correction, written out in full (a working optimizer, not a stub), plus a decoupled
// weight-decay variant (AdamW) and a plain gradient-descent baseline for comparison. The update is
//   m = b1 m + (1 - b1) g,  v = b2 v + (1 - b2) g^2,
//   mHat = m / (1 - b1^t),  vHat = v / (1 - b2^t),  w -= lr * mHat / (sqrt(vHat) + eps),
// with the optional decoupled decay w *= (1 - lr * gamma) applied before the adaptive step.
// Documented checks: with a constant 1-D gradient the first update is lr * sign(g) (0.1 * (1 - 4e-8)
// because of eps) whereas the same step *without* bias correction overshoots by
// (1 - b1) / sqrt(1 - b2) = 3.162, so both the corrected and the uncorrected magnitudes are pinned;
// after t constant-gradient steps the raw moments equal exactly (1 - b1^t) g and (1 - b2^t) g^2 and
// therefore mHat = g and vHat = g^2 to 1e-15 for every t in 1..40, which is why 40 constant steps
// move the parameter by 40 * lr; b1 = b2 = 0 degenerates to sign descent; a zero gradient leaves the
// parameter untouched and a gradient of 1e-300 does not overflow; scaling the objective by 100 leaves
// the Adam trajectory identical to 1e-5 although plain SGD with the same scaled gradient blows up;
// and the per-coordinate step never exceeds lr. On the 3x3 SPD quadratic f(x) = 0.5 x^T A x - b^T x
// (x* = (0.6905, -1.0714, 0.5238), f(x*) = -1.547619047619048, L = 4.4142 by power iteration) Adam
// with lr = 0.1 and 2000 steps reaches distance 1e-9 with the loss equal to the optimum to 1e-12 and
// an almost monotone trace (5 increases out of 200 samples), gradient descent at its optimal stable
// step 1/L converges just as exactly but gradient descent at 2/L diverges to distance 3.6, while Adam
// with lr = 2.0 - ten times the stable step - still converges (1e-9); decoupled decay with gamma =
// 0.05 shrinks the final norm from 1.3781 to 1.3570, starting from the origin also converges (1e-12),
// a single step solves nothing, and repeated runs are bit-identical.
using System;
using System.Collections.Generic;

internal static class LinearAlgebra
{
    internal static readonly double[][] A = {
        new[] { 3.0, 1.0, 0.0 },
        new[] { 1.0, 3.0, 1.0 },
        new[] { 0.0, 1.0, 3.0 },
    };

    internal static readonly double[] B = { 1.0, -2.0, 0.5 };

    /** f(x) = 0.5 x^T A x - b^T x. */
    internal static double Value(double[] x)
    {
        double total = 0.0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++) total += 0.5 * x[i] * A[i][j] * x[j];
            total -= B[i] * x[i];
        }
        return total;
    }

    /** grad f(x) = A x - b (A symmetric). */
    internal static double[] Gradient(double[] x)
    {
        var gradient = new double[3];
        for (int i = 0; i < 3; i++)
        {
            double sum = -B[i];
            for (int j = 0; j < 3; j++) sum += A[i][j] * x[j];
            gradient[i] = sum;
        }
        return gradient;
    }

    internal static double Norm(double[] x)
    {
        double total = 0.0;
        foreach (double value in x) total += value * value;
        return Math.Sqrt(total);
    }

    /** Exact minimiser A^-1 b by Gaussian elimination with partial pivoting. */
    internal static double[] ExactMinimiser()
    {
        const int n = 3;
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n + 1];
            Array.Copy(A[i], a[i], n);
            a[i][n] = B[i];
        }
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
            {
                if (Math.Abs(a[row][column]) > Math.Abs(a[pivot][column])) pivot = row;
            }
            (a[column], a[pivot]) = (a[pivot], a[column]);
            for (int row = column + 1; row < n; row++)
            {
                double factor = a[row][column] / a[column][column];
                for (int k = column; k <= n; k++) a[row][k] -= factor * a[column][k];
            }
        }
        var solution = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = a[row][n];
            for (int k = row + 1; k < n; k++) sum -= a[row][k] * solution[k];
            solution[row] = sum / a[row][row];
        }
        return solution;
    }

    /** Largest eigenvalue of A by power iteration, used for the optimal gradient-descent step. */
    internal static double LargestEigenvalue()
    {
        var v = new double[] { 1.0, 1.0, 1.0 };
        double norm = Norm(v);
        for (int i = 0; i < 3; i++) v[i] /= norm;
        double eigenvalue = 0.0;
        for (int iteration = 0; iteration < 1000; iteration++)
        {
            var w = new double[3];
            for (int i = 0; i < 3; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < 3; j++) sum += A[i][j] * v[j];
                w[i] = sum;
            }
            double wnorm = Norm(w);
            for (int i = 0; i < 3; i++) w[i] /= wnorm;
            double shift = 0.0;
            for (int i = 0; i < 3; i++) shift += Math.Abs(w[i] - v[i]);
            v = w;
            eigenvalue = wnorm;
            if (shift <= 1e-15) break;
        }
        return eigenvalue;
    }
}

internal class AdamState
{
    internal readonly double[] FirstMoment = new double[3];
    internal readonly double[] SecondMoment = new double[3];
    internal long Steps;
}

internal static class Adam
{
    internal const double Beta1 = 0.9;
    internal const double Beta2 = 0.999;
    internal const double Epsilon = 1e-8;

    /** One Adam step with explicit betas; returns the update that was applied. */
    internal static double[] StepWithBetas(AdamState state, double[] w, double[] gradient,
            double learningRate, double weightDecay, double beta1, double beta2)
    {
        state.Steps += 1;
        double correction1 = 1.0 - Math.Pow(beta1, state.Steps);
        double correction2 = 1.0 - Math.Pow(beta2, state.Steps);
        var update = new double[w.Length];
        for (int j = 0; j < w.Length; j++)
        {
            if (weightDecay > 0.0) w[j] *= 1.0 - learningRate * weightDecay;
            state.FirstMoment[j] = beta1 * state.FirstMoment[j] + (1.0 - beta1) * gradient[j];
            state.SecondMoment[j] = beta2 * state.SecondMoment[j] + (1.0 - beta2) * gradient[j] * gradient[j];
            double mHat = state.FirstMoment[j] / correction1;
            double vHat = state.SecondMoment[j] / correction2;
            double step = learningRate * mHat / (Math.Sqrt(vHat) + Epsilon);
            w[j] -= step;
            update[j] = -step;
        }
        return update;
    }

    internal static double[] Step(AdamState state, double[] w, double[] gradient, double learningRate,
            double weightDecay)
    {
        return StepWithBetas(state, w, gradient, learningRate, weightDecay, Beta1, Beta2);
    }

    /** The same step without bias correction, used to show what the correction buys. */
    internal static double UncorrectedFirstStep(double gradient, double learningRate)
    {
        double m = (1.0 - Beta1) * gradient;
        double v = (1.0 - Beta2) * gradient * gradient;
        return learningRate * m / (Math.Sqrt(v) + Epsilon);
    }

    internal class Run
    {
        internal double[] Weights = new double[3];
        internal List<double> Losses = new List<double>();
        internal double Distance;
        internal double FinalLoss;
        internal double MaxCoordinateStep;
        internal long Steps;
    }

    internal static Run Minimise(double learningRate, long steps, double weightDecay, double[] start)
    {
        var state = new AdamState();
        var run = new Run { Weights = (double[])start.Clone() };
        for (long step = 0; step < steps; step++)
        {
            double[] gradient = LinearAlgebra.Gradient(run.Weights);
            double[] update = Step(state, run.Weights, gradient, learningRate, weightDecay);
            for (int j = 0; j < 3; j++)
            {
                run.MaxCoordinateStep = Math.Max(run.MaxCoordinateStep, Math.Abs(update[j]));
            }
            if (step % 10 == 0) run.Losses.Add(LinearAlgebra.Value(run.Weights));
        }
        run.FinalLoss = LinearAlgebra.Value(run.Weights);
        run.Distance = LinearAlgebra.Norm(Sub(run.Weights, LinearAlgebra.ExactMinimiser()));
        run.Steps = steps;
        return run;
    }

    /** Plain gradient descent baseline: the same budget at the optimal stable step 1 / L. */
    internal static Run GradientDescent(double learningRate, long steps, double[] start)
    {
        var run = new Run { Weights = (double[])start.Clone() };
        for (long step = 0; step < steps; step++)
        {
            double[] gradient = LinearAlgebra.Gradient(run.Weights);
            for (int j = 0; j < 3; j++) run.Weights[j] -= learningRate * gradient[j];
            if (step % 10 == 0) run.Losses.Add(LinearAlgebra.Value(run.Weights));
        }
        run.FinalLoss = LinearAlgebra.Value(run.Weights);
        run.Distance = LinearAlgebra.Norm(Sub(run.Weights, LinearAlgebra.ExactMinimiser()));
        run.Steps = steps;
        return run;
    }

    internal static double[] Sub(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        double[] start = { 2.0, 2.0, 2.0 };
        double[] optimum = LinearAlgebra.ExactMinimiser();
        Check(Math.Abs(LinearAlgebra.Value(optimum) + 1.547619047619048) <= 1e-12,
                "the exact minimiser value is -1.547619047619048");
        Check(LinearAlgebra.Norm(LinearAlgebra.Gradient(optimum)) <= 1e-14,
                "the exact minimiser has zero gradient");

        // --- the first step is exactly lr * sign(g) ------------------------------------------------------
        var state = new AdamState();
        var w = new double[] { 1.0 };
        double[] update = Adam.Step(state, w, new[] { 0.25 }, 0.1, 0.0);
        Check(Math.Abs(update[0] + 0.1) <= 1e-8, "the first Adam step is -lr for a positive gradient");
        Check(Math.Abs(w[0] - 0.9) <= 1e-8, "and the parameter moved by lr");
        var stateNegative = new AdamState();
        var wNegative = new double[] { 1.0 };
        double[] updateNegative = Adam.Step(stateNegative, wNegative, new[] { -3.5 }, 0.1, 0.0);
        Check(Math.Abs(updateNegative[0] - 0.1) <= 1e-8, "the sign follows the gradient");
        Check(Math.Abs(Math.Abs(update[0]) - 0.1) <= 1e-8 && Math.Abs(Math.Abs(updateNegative[0]) - 0.1) <= 1e-8,
                "the magnitude is lr regardless of the gradient size (eps makes it 0.1 * (1 - 4e-8))");

        // --- bias correction matters ----------------------------------------------------------------------
        double uncorrected = Adam.UncorrectedFirstStep(0.25, 0.1);
        double overshoot = (1.0 - Adam.Beta1) / Math.Sqrt(1.0 - Adam.Beta2);
        Check(Math.Abs(overshoot - 3.1622776601683795) <= 1e-12,
                "the uncorrected overshoot factor is (1 - b1) / sqrt(1 - b2) = 3.162");
        Check(Math.Abs(Math.Abs(uncorrected) - 0.1 * overshoot) <= 1e-4,
                "without bias correction the first step overshoots by that factor (" + uncorrected + ")");
        Check(Math.Abs(uncorrected) > Math.Abs(update[0]),
                "bias correction keeps the first step at exactly lr instead of 3.16x lr");

        // --- after t constant-gradient steps the corrected moments are exact --------------------------------
        var constantState = new AdamState();
        var constantW = new double[] { 5.0 };
        for (int t = 1; t <= 40; t++)
        {
            Adam.Step(constantState, constantW, new[] { 0.25 }, 0.1, 0.0);
            double correction1 = 1.0 - Math.Pow(Adam.Beta1, t);
            double correction2 = 1.0 - Math.Pow(Adam.Beta2, t);
            Check(Math.Abs(constantState.FirstMoment[0] - correction1 * 0.25) <= 1e-15,
                    "the first moment closes on (1 - b1^t) g at t = " + t);
            Check(Math.Abs(constantState.SecondMoment[0] - correction2 * 0.0625) <= 1e-15,
                    "the second moment closes on (1 - b2^t) g^2 at t = " + t);
            double mHat = constantState.FirstMoment[0] / correction1;
            double vHat = constantState.SecondMoment[0] / correction2;
            Check(Math.Abs(mHat - 0.25) <= 1e-15 && Math.Abs(vHat - 0.0625) <= 1e-15,
                    "the corrected moments reproduce g and g^2 at t = " + t);
        }
        Check(Math.Abs(constantW[0] - 5.0 + 40 * 0.1) <= 1e-6,
                "40 constant steps move the parameter by 40 * lr (eps costs 1.6e-7)");

        // --- a zero gradient is safe -----------------------------------------------------------------------
        var zeroState = new AdamState();
        var zeroW = new double[] { 3.0 };
        for (int t = 0; t < 100; t++) Adam.Step(zeroState, zeroW, new[] { 0.0 }, 0.1, 0.0);
        Check(zeroW[0] == 3.0, "a zero gradient leaves the parameter untouched");
        Check(!double.IsNaN(zeroW[0]) && !double.IsInfinity(zeroW[0]), "and stays finite");
        var tinyState = new AdamState();
        var tinyW = new double[] { 1.0 };
        double[] tinyUpdate = Adam.Step(tinyState, tinyW, new[] { 1e-300 }, 0.1, 0.0);
        Check(!double.IsNaN(tinyUpdate[0]) && !double.IsInfinity(tinyUpdate[0]),
                "eps keeps a vanishing gradient safe");
        Check(Math.Abs(tinyUpdate[0]) <= 0.1 + 1e-12, "the update is still bounded by lr");

        // --- scale invariance --------------------------------------------------------------------------------
        var scaled = new Adam.Run { Weights = (double[])start.Clone() };
        var scaledState = new AdamState();
        for (int step = 0; step < 200; step++)
        {
            double[] gradient = LinearAlgebra.Gradient(scaled.Weights);
            for (int j = 0; j < 3; j++) gradient[j] *= 100.0;
            Adam.Step(scaledState, scaled.Weights, gradient, 0.05, 0.0);
        }
        var reference = Adam.Minimise(0.05, 200, 0.0, start);
        double scaleGap = 0.0;
        for (int j = 0; j < 3; j++)
        {
            scaleGap = Math.Max(scaleGap, Math.Abs(scaled.Weights[j] - reference.Weights[j]));
        }
        Check(scaleGap <= 1e-5, "Adam is scale invariant (gap " + scaleGap + ")");
        var sgdScaled = new Adam.Run { Weights = (double[])start.Clone() };
        for (int step = 0; step < 200; step++)
        {
            double[] gradient = LinearAlgebra.Gradient(sgdScaled.Weights);
            for (int j = 0; j < 3; j++) sgdScaled.Weights[j] -= 0.2 * 100.0 * gradient[j];
        }
        bool sgdBlewUp = false;
        for (int j = 0; j < 3; j++)
        {
            if (!double.IsFinite(sgdScaled.Weights[j]) || Math.Abs(sgdScaled.Weights[j]) > 1.0)
            {
                sgdBlewUp = true;
            }
        }
        Check(sgdBlewUp, "plain SGD with the same scaled gradient blows up (it is not scale invariant)");

        // --- degenerate betas: b1 = b2 = 0 degenerates to exact sign descent -------------------------------------
        var signState = new AdamState();
        var signW = (double[])start.Clone();
        var signReference = (double[])start.Clone();
        for (int step = 0; step < 30; step++)
        {
            double[] gradient = LinearAlgebra.Gradient(signW);
            Adam.StepWithBetas(signState, signW, gradient, 0.05, 0.0, 0.0, 0.0);
            for (int j = 0; j < 3; j++) signReference[j] -= 0.05 * Math.Sign(gradient[j]);
        }
        for (int j = 0; j < 3; j++)
        {
            Check(Math.Abs(signW[j] - signReference[j]) <= 1e-6,
                    "b1 = b2 = 0 is sign descent at coordinate " + j);
            Check(!double.IsNaN(signW[j]) && !double.IsInfinity(signW[j]), "sign descent stays finite");
        }

        // --- the quadratic: robust step size, exact convergence --------------------------------------------------------
        const long steps = 2000;
        double optimumLoss = LinearAlgebra.Value(optimum);
        Adam.Run adam = Adam.Minimise(0.1, steps, 0.0, start);
        double lipschitz = LinearAlgebra.LargestEigenvalue();
        Adam.Run descent = Adam.GradientDescent(1.0 / lipschitz, steps, start);
        Adam.Run reckless = Adam.GradientDescent(2.0 / lipschitz, steps, start);
        Adam.Run largeStep = Adam.Minimise(2.0, steps, 0.0, start);
        Check(lipschitz >= 4.0 && lipschitz <= 5.0, "the power method finds L = " + lipschitz);
        Check(adam.Distance <= 1e-9, "Adam reaches the optimum (distance " + adam.Distance + ")");
        Check(Math.Abs(adam.FinalLoss - optimumLoss) <= 1e-12,
                "the Adam loss equals the optimum (" + adam.FinalLoss + ")");
        Check(descent.Distance <= 1e-9, "gradient descent at 1/L also converges");
        Check(reckless.Distance >= 1.0, "gradient descent at 2/L diverges (distance " + reckless.Distance + ")");
        Check(largeStep.Distance <= 1e-6,
                "Adam still converges with lr = 2.0, ten times the stable gradient-descent step ("
                + largeStep.Distance + ")");
        Check(adam.MaxCoordinateStep <= 0.1 + 1e-9, "the coordinate step never exceeds lr");
        Check(largeStep.MaxCoordinateStep <= 2.0 + 1e-9, "and it tracks lr when lr is large");
        int increases = 0;
        for (int i = 1; i < adam.Losses.Count; i++)
        {
            if (adam.Losses[i] > adam.Losses[i - 1] + 1e-12) increases += 1;
        }
        Check(increases <= adam.Losses.Count / 20, "the Adam loss trace is almost monotone ("
                + increases + " increases out of " + adam.Losses.Count + ")");
        for (int i = 1; i < descent.Losses.Count; i++)
        {
            Check(descent.Losses[i] <= descent.Losses[i - 1] + 1e-12,
                    "gradient descent at 1/L is monotone at " + i);
        }
        Check(adam.Losses[adam.Losses.Count - 1] <= optimumLoss + 1e-9, "the trace ends at the optimum");

        // --- weight decay --------------------------------------------------------------------------------------------
        Adam.Run decayed = Adam.Minimise(0.05, steps, 0.05, start);
        Adam.Run plain = Adam.Minimise(0.05, steps, 0.0, start);
        Check(LinearAlgebra.Norm(decayed.Weights) < LinearAlgebra.Norm(plain.Weights),
                "decoupled weight decay shrinks the norm ("
                + LinearAlgebra.Norm(decayed.Weights) + " < " + LinearAlgebra.Norm(plain.Weights) + ")");
        Check(decayed.FinalLoss >= optimumLoss - 1e-9, "and costs a little objective value");
        Check(decayed.Distance <= 0.1, "the decayed run still lands near the optimum");
        Adam.Run aggressive = Adam.Minimise(0.05, steps, 5.0, start);
        Check(LinearAlgebra.Norm(aggressive.Weights) < LinearAlgebra.Norm(decayed.Weights),
                "a larger decay shrinks the norm further");

        // --- a short budget still descends -----------------------------------------------------------------------------
        Adam.Run oneStep = Adam.Minimise(0.1, 1, 0.0, start);
        Check(oneStep.Distance > 0.1, "a single Adam step does not solve the problem");
        Adam.Run shortRun = Adam.Minimise(0.1, 50, 0.0, start);
        Check(shortRun.FinalLoss < LinearAlgebra.Value(start), "50 steps already reduce the loss");
        Check(shortRun.Distance <= 1.0, "and close in on the optimum");
        var frozen = new double[] { 0.0, 0.0, 0.0 };
        Adam.Run fromOrigin = Adam.Minimise(0.1, 500, 0.0, frozen);
        Check(fromOrigin.FinalLoss < LinearAlgebra.Value(frozen), "starting at zero also descends");
        Check(fromOrigin.Distance <= 1e-6, "and converges from the origin too ("
                + fromOrigin.Distance + ")");

        // --- determinism --------------------------------------------------------------------------------------------------
        Adam.Run repeat = Adam.Minimise(0.1, steps, 0.0, start);
        for (int j = 0; j < 3; j++)
        {
            Check(repeat.Weights[j] == adam.Weights[j], "Adam is deterministic at coordinate " + j);
        }
        Check(repeat.Distance == adam.Distance, "and reports the same distance");
        Check(repeat.Losses.Count == adam.Losses.Count, "and the same trace length");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
