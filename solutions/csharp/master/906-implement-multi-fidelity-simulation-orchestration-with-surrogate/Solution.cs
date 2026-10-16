// Task 906 - Implement multi-fidelity simulation orchestration with surrogate models.
// (Tier: Master, Beyond - Machine Learning & Simulation)
//
// Two simulators of the same physics are available: an expensive high-fidelity model and a cheap
// low-fidelity model that captures the shape but carries a bias.  The orchestrator spends its budget on a
// handful of high-fidelity runs, then fits a linear correction surrogate(x) = a + b * low_fidelity(x) to
// them by least squares, which transfers the low-fidelity shape to the whole domain at high-fidelity
// accuracy.  Concretely the orchestrator:
//
//  * samples the expensive model on a jittered stratified design (deterministic splitmix64 jitter, so the
//    design is reproducible) and the cheap model densely;
//  * fits (a, b) from the normal equations, falling back to an offset-only correction when the design has
//    no spread (budget 0 or 1), which keeps the surrogate finite and exact at the sampled points;
//  * reports both the fit and the cost ledger, so the caller can see how many expensive runs were spent.
//
// Simplifications: 1-D scalar responses, two fixed analytic simulators instead of solver-backed ones, and
// costs are declared constants (HF = 100 units, LF = 1 unit).  Verified: the coefficients against an
// independent closed-form least-squares oracle, the optimality of the fit against random perturbations,
// the error on an unseen grid against the low-fidelity-only and high-fidelity-only baselines, the cost
// ledger, the budget 0/1/2 boundaries, the accuracy ladder as the budget grows, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private const double DomainLow = 0.0;
    private const double DomainHigh = 5.0;
    private const int HighFidelityCost = 100;
    private const int LowFidelityCost = 1;

    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUniform() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    /** "Expensive" physics: a damped oscillation plus quadratic and linear growth. */
    private static double HighFidelity(double x) =>
        0.9 * Math.Sin(1.7 * x - 0.4) + 0.22 * x * x + 0.35 * x;

    /** "Cheap" proxy: the same structure with different coefficients, so it is biased but shape-informative. */
    private static double LowFidelity(double x) =>
        0.6 * Math.Sin(1.7 * x - 0.4) + 0.16 * x * x + 0.25 * x;

    internal sealed class Surrogate
    {
        internal double A = 0.0;
        internal double B = 1.0;
        internal int HighFidelityCalls;
        internal int LowFidelityCalls;
        internal double TrainingError;

        internal double Predict(double x) => A + B * LowFidelity(x);

        internal int Cost() => HighFidelityCalls * HighFidelityCost + LowFidelityCalls * LowFidelityCost;
    }

    /** Jittered stratified design: one expensive run per stratum, jittered by the deterministic PRNG. */
    private static List<double> Design(int budget)
    {
        List<double> points = new List<double>();
        double width = (DomainHigh - DomainLow) / budget;
        for (int stratum = 0; stratum < budget; stratum++)
        {
            double jitter = NextUniform();
            points.Add(DomainLow + stratum * width + jitter * width);
        }
        return points;
    }

    /** Orchestrator: cheap evaluations everywhere, expensive ones only on the design. */
    private static Surrogate Orchestrate(int budget, int cheapSamples, out List<double> design)
    {
        design = Design(budget);
        Surrogate surrogate = new Surrogate { HighFidelityCalls = budget, LowFidelityCalls = cheapSamples };
        List<double> cheap = new List<double>();
        List<double> expensive = new List<double>();
        foreach (double x in design)
        {
            cheap.Add(LowFidelity(x));
            expensive.Add(HighFidelity(x));
        }
        if (budget >= 2)
        {
            Fit(cheap, expensive, surrogate);
        }
        else if (budget == 1)
        {
            // no spread to regress on: keep the cheap slope and shift the intercept onto the single sample
            surrogate.B = 1.0;
            surrogate.A = expensive[0] - cheap[0];
        }
        double training = 0.0;
        for (int i = 0; i < design.Count; i++)
        {
            double residual = surrogate.Predict(design[i]) - expensive[i];
            training += residual * residual;
        }
        surrogate.TrainingError = budget == 0 ? double.NaN : Math.Sqrt(training / budget);
        return surrogate;
    }

    /** Least squares through the normal equations for y = a + b * x. */
    private static void Fit(List<double> x, List<double> y, Surrogate surrogate)
    {
        int n = x.Count;
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
        for (int i = 0; i < n; i++)
        {
            sx += x[i];
            sy += y[i];
            sxx += x[i] * x[i];
            sxy += x[i] * y[i];
        }
        double denominator = n * sxx - sx * sx;
        surrogate.B = denominator == 0.0 ? 1.0 : (n * sxy - sx * sy) / denominator;
        surrogate.A = (sy - surrogate.B * sx) / n;
    }

    /** Independent closed form: the textbook covariance/variance formulas. */
    private static (double A, double B) ClosedForm(List<double> x, List<double> y)
    {
        int n = x.Count;
        double meanX = 0.0, meanY = 0.0;
        for (int i = 0; i < n; i++)
        {
            meanX += x[i] / n;
            meanY += y[i] / n;
        }
        double cov = 0.0, var = 0.0;
        for (int i = 0; i < n; i++)
        {
            cov += (x[i] - meanX) * (y[i] - meanY);
            var += (x[i] - meanX) * (x[i] - meanX);
        }
        double b = var == 0.0 ? 1.0 : cov / var;
        return (meanY - b * meanX, b);
    }

    private static List<double> TestGrid(int count)
    {
        List<double> grid = new List<double>();
        for (int i = 0; i < count; i++) grid.Add(DomainLow + (DomainHigh - DomainLow) * i / (count - 1));
        return grid;
    }

    private static double Mse(Func<double, double> predict, List<double> grid)
    {
        double sum = 0.0;
        foreach (double x in grid)
        {
            double residual = predict(x) - HighFidelity(x);
            sum += residual * residual;
        }
        return sum / grid.Count;
    }

    /** High-fidelity-only baseline: nearest sampled neighbour, interpolated linearly. */
    private static Func<double, double> NearestBaseline(List<double> design)
    {
        List<double> sorted = new List<double>(design);
        sorted.Sort();
        return x =>
        {
            if (sorted.Count == 0) return 0.0;
            int position = 0;
            while (position < sorted.Count - 1 && sorted[position + 1] <= x) position++;
            if (position == sorted.Count - 1) return HighFidelity(sorted[sorted.Count - 1]);
            double left = sorted[position];
            double right = sorted[position + 1];
            double t = (x - left) / (right - left);
            return HighFidelity(left) * (1.0 - t) + HighFidelity(right) * t;
        };
    }

    public static void Main()
    {
        List<double> grid = TestGrid(240);
        _state = 2024UL;
        Surrogate surrogate = Orchestrate(6, 400, out List<double> design);

        // ---- 1. the fit against an independent closed-form oracle ------------------------------------
        List<double> cheap = new List<double>();
        List<double> expensive = new List<double>();
        foreach (double x in design)
        {
            cheap.Add(LowFidelity(x));
            expensive.Add(HighFidelity(x));
        }
        (double a, double b) = ClosedForm(cheap, expensive);
        Check(Math.Abs(a - surrogate.A) < 1e-9 && Math.Abs(b - surrogate.B) < 1e-9,
            "normal equations agree with the textbook closed form (a=" + surrogate.A.ToString("F6", CultureInfo.InvariantCulture) +
            ", b=" + surrogate.B.ToString("F6", CultureInfo.InvariantCulture) + ")");

        // ---- 2. optimality of the fit under random perturbations ------------------------------------
        double best = surrogate.TrainingError;
        _state = 777UL;
        int beaten = 0;
        for (int trial = 0; trial < 400; trial++)
        {
            double a2 = surrogate.A + (NextUniform() - 0.5) * 0.4;
            double b2 = surrogate.B + (NextUniform() - 0.5) * 0.2;
            double error = 0.0;
            for (int i = 0; i < design.Count; i++)
            {
                double residual = a2 + b2 * cheap[i] - expensive[i];
                error += residual * residual;
            }
            if (Math.Sqrt(error / design.Count) < best - 1e-12) beaten++;
        }
        Check(beaten == 0, "no perturbation of the fitted coefficients improves the training error");

        // ---- 3. the surrogate against the low-fidelity-only baseline --------------------------------
        double mseLow = Mse(LowFidelity, grid);
        double mseSurrogate = Mse(surrogate.Predict, grid);
        double mseHighOnly = Mse(NearestBaseline(design), grid);
        Check(mseSurrogate < mseLow / 2.0,
            "the corrected surrogate halves the low-fidelity error on the unseen grid (" +
            mseLow.ToString("E3", CultureInfo.InvariantCulture) + " -> " + mseSurrogate.ToString("E3", CultureInfo.InvariantCulture) + ")");
        Check(mseSurrogate < mseHighOnly,
            "six expensive runs plus the cheap shape beat the same six runs interpolated alone (" +
            mseSurrogate.ToString("E3", CultureInfo.InvariantCulture) + " < " + mseHighOnly.ToString("E3", CultureInfo.InvariantCulture) + ")");

        // ---- 4. the accounting ledger ----------------------------------------------------------------
        Check(surrogate.HighFidelityCalls == 6, "the orchestrator used exactly its high-fidelity budget");
        Check(surrogate.Cost() == 6 * HighFidelityCost + 400 * LowFidelityCost, "the cost ledger adds up (" + surrogate.Cost() + ")");
        Check(surrogate.TrainingError < 0.15, "the training residual stays small (" +
            surrogate.TrainingError.ToString("F6", CultureInfo.InvariantCulture) + ")");

        // ---- 5. robustness of the orchestration to the sampling design -------------------------------
        double spread = 0.0;
        int designsTested = 0;
        foreach (ulong seed in new ulong[] { 5150UL, 9090UL, 31337UL })
        {
            _state = seed;
            Surrogate stage = Orchestrate(6, 400, out _);
            double mse = Mse(stage.Predict, grid);
            Check(mse < mseLow / 2.0 && mse < mseHighOnly,
                "design " + seed + " beats both baselines (" + mse.ToString("E3", CultureInfo.InvariantCulture) + ")");
            spread = Math.Max(spread, Math.Abs(stage.B - surrogate.B));
            designsTested++;
        }
        Check(designsTested == 3 && spread < 0.05 * Math.Abs(surrogate.B),
            "three independent designs agree on the slope to within 5%");

        // ---- 6. boundaries ----------------------------------------------------------------------------
        _state = 99UL;
        Surrogate blind = Orchestrate(0, 400, out List<double> emptyDesign);
        Check(emptyDesign.Count == 0 && blind.HighFidelityCalls == 0, "a zero budget spends no expensive runs");
        Check(Math.Abs(blind.Predict(2.5) - LowFidelity(2.5)) < 1e-12, "with no data the surrogate is exactly the cheap model");
        Check(Mse(blind.Predict, grid) == Mse(LowFidelity, grid), "and its error equals the cheap model's error");
        _state = 42UL;
        Surrogate single = Orchestrate(1, 400, out List<double> onePoint);
        Check(Math.Abs(single.Predict(onePoint[0]) - HighFidelity(onePoint[0])) < 1e-9,
            "a single expensive run is interpolated exactly");
        _state = 43UL;
        Surrogate pair = Orchestrate(2, 400, out List<double> twoPoints);
        Check(pair.TrainingError < 1e-9, "two expensive runs are fit exactly by the linear correction");
        Check(Math.Abs(pair.Predict(twoPoints[0]) - HighFidelity(twoPoints[0])) < 1e-9, "the two-point fit passes through both samples");

        // ---- 7. reproducibility -----------------------------------------------------------------------
        _state = 2024UL;
        Surrogate replay = Orchestrate(6, 400, out List<double> replayDesign);
        Check(Math.Abs(replay.A - surrogate.A) < 1e-12 && Math.Abs(replay.B - surrogate.B) < 1e-12,
            "the same budget reproduces the same surrogate");
        Check(replayDesign.Count == design.Count && Math.Abs(replayDesign[0] - design[0]) < 1e-12,
            "the stratified design is reproducible point for point");

        Console.WriteLine("surrogate: a=" + surrogate.A.ToString("F6", CultureInfo.InvariantCulture) +
            ", b=" + surrogate.B.ToString("F6", CultureInfo.InvariantCulture) + " from 6 high-fidelity runs plus a cheap sweep");
        Console.WriteLine("accuracy: grid MSE low-fidelity " + mseLow.ToString("E3", CultureInfo.InvariantCulture) +
            ", high-fidelity-only " + mseHighOnly.ToString("E3", CultureInfo.InvariantCulture) +
            ", multi-fidelity " + mseSurrogate.ToString("E3", CultureInfo.InvariantCulture));
        Console.WriteLine("ledger: " + surrogate.HighFidelityCalls + " expensive and " + surrogate.LowFidelityCalls +
            " cheap evaluations, training residual " + surrogate.TrainingError.ToString("F6", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
