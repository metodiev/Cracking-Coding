// Task 866 - Implement a feature attribution runtime (SHAP-like) scalable for tabular models.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// Three ways to attribute a tabular model's prediction to its features, all built from the Shapley value:
//
//   exact        enumerate every coalition, weight it with the Shapley kernel and average the marginal
//                contributions: exact but exponential in the feature count
//   permutation  sample permutations and average the marginal contribution of each feature along them: the
//                standard Monte-Carlo approximation, whose error falls as 1/sqrt(samples)
//   KernelSHAP   fit the attribution vector by weighted least squares against the coalition values, with the
//                Shapley kernel as the weight, and the efficiency constraint applied as a post-correction
//
// The model is deliberately structured - an additive term per feature plus one pairwise interaction - so an
// *analytic* Shapley decomposition exists and can be compared with all three numerical estimators: for an
// additive-plus-one-interaction model each feature receives its own marginal gain and the interaction
// contribution is split evenly between the two interacting features. That analytic value is the primary
// oracle; the axioms (efficiency, symmetry, dummy, linearity) are checked on top of it.
//
// What is simplified (documented, not behavioural): the feature count is small enough to enumerate the
// 2^d coalitions, the background distribution is a fixed sample of rows rather than a real data digest, and
// the model is a hand-written additive-plus-interaction function rather than a boosted ensemble. The
// coalition value function, the three estimators, the least-squares solve and the axiom suite are the real
// content; the feature count and the sample budget are typed parameters.
//
// Invariants verified: the analytic decomposition, the exhaustive Shapley values and KernelSHAP agree to
// 1e-6; the efficiency axiom holds for every estimator (the attributions sum to prediction minus baseline);
// symmetric features receive equal attributions and a feature the model ignores receives exactly zero; the
// attribution of a linear combination of models is the same combination of attributions; the permutation
// estimator's error decreases as the sample budget grows and its mean error at the full budget is small; the
// background sample, the coalition values and the whole report are reproducible; degenerate inputs (a
// constant model, a one-feature model, a single background row) are handled.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng866
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

// An additive-plus-interaction model: f(x) = sum_i g_i(x_i) + interaction * x_0 * x_1. The additive part
// makes the exact Shapley values available in closed form.
internal sealed class Model866
{
    internal readonly int Features;
    internal readonly double Interaction;
    internal readonly double[] Scales;

    internal Model866(int features, double interaction, double[] scales)
    {
        if (features < 1) throw new ArgumentException("a model needs at least one feature");
        Features = features;
        Interaction = interaction;
        Scales = scales;
    }

    internal double Predict(double[] x)
    {
        if (x.Length != Features) throw new ArgumentException("the feature count does not match the model");
        double sum = 0.0;
        for (int i = 0; i < Features; i++) sum += G(i, x[i]);
        sum += Interaction * x[0] * x[1 % Features];
        return sum;
    }

    // g_i(v) = scale_i * v + 0.5 * scale_i * v^2: monotone and non-linear, so attributions are not just
    // proportional to the raw feature values.
    internal double G(int feature, double value) => Scales[feature] * (value + 0.5 * value * value);
}

internal sealed class Explainer866
{
    private readonly Model866 model;
    private readonly double[][] background;

    internal Explainer866(Model866 model, double[][] background)
    {
        if (background.Length == 0) throw new ArgumentException("the background sample is empty");
        this.model = model;
        this.background = background;
        foreach (double[] row in background)
            if (row.Length != model.Features) throw new ArgumentException("a background row has the wrong width");
    }

    // The coalition value: features outside the coalition are drawn from the background distribution, which
    // is averaged exactly (all background rows, no sampling) so the value function is deterministic.
    internal double Value(double[] x, bool[] coalition)
    {
        double total = 0.0;
        foreach (double[] row in background)
        {
            var sample = new double[model.Features];
            for (int i = 0; i < model.Features; i++) sample[i] = coalition[i] ? x[i] : row[i];
            total += model.Predict(sample);
        }
        return total / background.Length;
    }

    internal double Baseline
    {
        get
        {
            var empty = new bool[model.Features];
            var mean = new double[model.Features];
            foreach (double[] row in background)
                for (int i = 0; i < model.Features; i++) mean[i] += row[i] / background.Length;
            return Value(mean, empty);
        }
    }

    internal static double Factorial(int n)
    {
        double result = 1.0;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }

    // Exact Shapley values by enumerating all 2^d coalitions with the Shapley kernel weight
    // |S|! (d - |S| - 1)! / d!.
    internal double[] Exact(double[] x)
    {
        int d = model.Features;
        double[] phi = new double[d];
        int total = 1 << d;
        double denominator = Factorial(d);
        for (int mask = 0; mask < total; mask++)
        {
            var coalition = new bool[d];
            int size = 0;
            for (int i = 0; i < d; i++)
            {
                coalition[i] = (mask & (1 << i)) != 0;
                if (coalition[i]) size++;
            }
            double valueWith = Value(x, coalition);
            for (int i = 0; i < d; i++)
            {
                if (coalition[i]) continue;
                var extended = (bool[])coalition.Clone();
                extended[i] = true;
                double weight = Factorial(size) * Factorial(d - size - 1) / denominator;
                phi[i] += weight * (Value(x, extended) - valueWith);
            }
        }
        return phi;
    }

    // Monte-Carlo permutation sampling of the same quantity.
    internal double[] Permutation(double[] x, int samples, ulong seed)
    {
        int d = model.Features;
        double[] phi = new double[d];
        var order = new int[d];
        for (int sample = 0; sample < samples; sample++)
        {
            for (int i = 0; i < d; i++) order[i] = i;
            for (int i = d - 1; i > 0; i--)
            {
                int j = Rng866.NextInt(ref seed, i + 1);
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
            var coalition = new bool[d];
            double previous = Value(x, coalition);
            for (int step = 0; step < d; step++)
            {
                int feature = order[step];
                coalition[feature] = true;
                double current = Value(x, coalition);
                phi[feature] += (current - previous) / samples;
                previous = current;
            }
        }
        return phi;
    }

    // KernelSHAP: weighted least squares against the coalition values with the Shapley kernel as the weight.
    // The empty and the exhaustive coalition carry the efficiency constraint (they pin the intercept), and
    // the residual after the solve is spread evenly over the features, the usual efficiency correction.
    internal double[] KernelShap(double[] x)
    {
        int d = model.Features;
        var rows = new List<double[]>();
        var targets = new List<double>();
        var weights = new List<double>();
        double denominator = Factorial(d);
        int total = 1 << d;
        for (int mask = 0; mask < total; mask++)
        {
            var coalition = new bool[d];
            int size = 0;
            for (int i = 0; i < d; i++)
            {
                coalition[i] = (mask & (1 << i)) != 0;
                if (coalition[i]) size++;
            }
            var row = new double[d + 1];
            row[0] = 1.0;
            for (int i = 0; i < d; i++) row[i + 1] = coalition[i] ? 1.0 : 0.0;
            rows.Add(row);
            targets.Add(Value(x, coalition));
            weights.Add(size == 0 || size == d
                ? 1e6
                : (d - 1) / (denominator * size * (d - size)));
        }
        double[] solution = WeightedLeastSquares866.Solve(rows, targets, weights, d + 1);
        var phi = new double[d];
        for (int i = 0; i < d; i++) phi[i] = solution[i + 1];
        double prediction = Value(x, AllTrue(d));
        double baseline = solution[0];
        double sum = 0.0;
        foreach (double value in phi) sum += value;
        double residual = (prediction - baseline - sum) / d;
        for (int i = 0; i < d; i++) phi[i] += residual;
        return phi;
    }

    private static bool[] AllTrue(int d)
    {
        var all = new bool[d];
        for (int i = 0; i < d; i++) all[i] = true;
        return all;
    }

    // Analytic Shapley values, derived from the model structure rather than enumerated. For an additive
    // model each feature receives E[g_i(x_i) - g_i(b_i)] over the background; the single pairwise term
    // c * x_a * x_b splits as phi_a = (c/2) E[(x_a - b_a)(x_b + b_b)] and symmetrically for b, which sums to
    // c (x_a x_b - E[b_a b_b]) as efficiency requires.
    internal double[] Analytic(double[] x)
    {
        int d = model.Features;
        var phi = new double[d];
        for (int i = 0; i < d; i++)
        {
            double total = 0.0;
            foreach (double[] row in background) total += model.G(i, x[i]) - model.G(i, row[i]);
            phi[i] = total / background.Length;
        }
        int a = 0, b = 1 % d;
        if (model.Interaction != 0.0 && d > 1)
        {
            double forA = 0.0, forB = 0.0;
            foreach (double[] row in background)
            {
                forA += (x[a] - row[a]) * (x[b] + row[b]);
                forB += (x[b] - row[b]) * (x[a] + row[a]);
            }
            phi[a] += model.Interaction * forA / (2.0 * background.Length);
            phi[b] += model.Interaction * forB / (2.0 * background.Length);
        }
        return phi;
    }
}

internal static class WeightedLeastSquares866
{
    // Solves (X^T W X) beta = X^T W y with Gaussian elimination: the normal equations are small (d x d).
    internal static double[] Solve(List<double[]> rows, List<double> targets, List<double> weights, int d)
    {
        var a = new double[d, d + 1];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0.0;
                for (int r = 0; r < rows.Count; r++) sum += weights[r] * rows[r][i] * rows[r][j];
                a[i, j] = sum;
            }
            double rhs = 0.0;
            for (int r = 0; r < rows.Count; r++) rhs += weights[r] * rows[r][i] * targets[r];
            a[i, d] = rhs;
        }
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            if (Math.Abs(a[pivot, col]) < 1e-12) throw new Exception("the KernelSHAP system is singular");
            for (int c = 0; c <= d; c++)
            {
                double tmp = a[col, c];
                a[col, c] = a[pivot, c];
                a[pivot, c] = tmp;
            }
            double scale = a[col, col];
            for (int c = col; c <= d; c++) a[col, c] /= scale;
            for (int r = 0; r < d; r++)
            {
                if (r == col) continue;
                double factor = a[r, col];
                for (int c = col; c <= d; c++) a[r, c] -= factor * a[col, c];
            }
        }
        var solution = new double[d];
        for (int i = 0; i < d; i++) solution[i] = a[i, d];
        return solution;
    }
}

public static class Solution
{
    private const int FeatureCount = 6;
    private const int BackgroundRows = 40;
    private const int Instances = 12;

    public static void Main()
    {
        var scales = new double[FeatureCount];
        for (int i = 0; i < FeatureCount; i++) scales[i] = 0.5 + 0.25 * i;
        var model = new Model866(FeatureCount, 1.5, scales);
        double[][] background = BuildBackground();
        var explainer = new Explainer866(model, background);
        List<double[]> instances = BuildInstances();

        // 1-3. The analytic decomposition, the exhaustive Shapley values and KernelSHAP must agree.
        bool exactMatchesAnalytic = true, kernelMatchesAnalytic = true;
        double worstExact = 0.0, worstKernel = 0.0;
        foreach (double[] x in instances)
        {
            double[] analytic = explainer.Analytic(x);
            double[] exact = explainer.Exact(x);
            double[] kernel = explainer.KernelShap(x);
            for (int i = 0; i < FeatureCount; i++)
            {
                worstExact = Math.Max(worstExact, Math.Abs(exact[i] - analytic[i]));
                worstKernel = Math.Max(worstKernel, Math.Abs(kernel[i] - analytic[i]));
            }
            if (worstExact > 1e-6) exactMatchesAnalytic = false;
            if (worstKernel > 1e-6) kernelMatchesAnalytic = false;
        }
        Console.WriteLine("max |exact - analytic| = " + worstExact.ToString("E2")
            + ", max |kernelshap - analytic| = " + worstKernel.ToString("E2"));
        Check(exactMatchesAnalytic, "the exhaustive Shapley enumeration reproduces the analytic decomposition");
        Check(kernelMatchesAnalytic, "KernelSHAP reproduces the analytic decomposition to 1e-6");

        // 4-6. Efficiency, symmetry/dummy and the background baseline.
        bool efficiency = true, dummyZero = true, symmetric = true;
        foreach (double[] x in instances)
        {
            double prediction = model.Predict(x);
            double baseline = explainer.Value(x, new bool[FeatureCount]);
            foreach (double[] phi in new[] { explainer.Exact(x), explainer.Analytic(x), explainer.KernelShap(x) })
            {
                double sum = 0.0;
                foreach (double value in phi) sum += value;
                if (Math.Abs(sum - (prediction - baseline)) > 1e-6) efficiency = false;
            }
        }
        Check(efficiency, "efficiency holds for all three estimators: the attributions sum to prediction minus baseline");
        var constant = new Model866(FeatureCount, 0.0, new double[FeatureCount]);
        var constantExplainer = new Explainer866(constant, background);
        double[] constantPhi = constantExplainer.Exact(instances[0]);
        foreach (double value in constantPhi) if (Math.Abs(value) > 1e-12) dummyZero = false;
        Check(dummyZero, "every feature of a constant model receives exactly zero attribution");
        var symmetricModel = new Model866(2, 0.0, new[] { 2.0, 2.0 });
        var symmetricBackground = new[] { new[] { 1.0, 1.0 }, new[] { 3.0, 3.0 }, new[] { 5.0, 5.0 } };
        var symmetricExplainer = new Explainer866(symmetricModel, symmetricBackground);
        double[] symmetricPhi = symmetricExplainer.Exact(new[] { 2.0, 2.0 });
        if (Math.Abs(symmetricPhi[0] - symmetricPhi[1]) > 1e-9) symmetric = false;
        Check(symmetric, "features with identical marginal effects receive identical attributions");

        // 7-8. The permutation estimator converges to the exact values as the budget grows.
        ulong seed = 0x866AUL;
        double errorSmall = 0.0, errorLarge = 0.0;
        foreach (double[] x in instances)
        {
            double[] exact = explainer.Exact(x);
            double[] small = explainer.Permutation(x, 32, seed + 1);
            double[] large = explainer.Permutation(x, 2048, seed + 2);
            for (int i = 0; i < FeatureCount; i++)
            {
                errorSmall += Math.Abs(small[i] - exact[i]) / (instances.Count * FeatureCount);
                errorLarge += Math.Abs(large[i] - exact[i]) / (instances.Count * FeatureCount);
            }
        }
        Console.WriteLine("permutation mean absolute error: 32 samples=" + errorSmall.ToString("F4")
            + ", 2048 samples=" + errorLarge.ToString("F4"));
        Check(errorLarge < errorSmall, "the permutation estimator's mean error falls as the sample budget grows");
        Check(errorLarge < 0.05 * (1.0 + Math.Abs(explainer.Baseline)),
            "with 2048 permutations the mean error is small (" + errorLarge.ToString("F4") + ")");

        // 9-11. Linearity of the explainer, interaction split and a one-feature model.
        var doubled = new Model866(FeatureCount, 3.0, new double[FeatureCount]);
        for (int i = 0; i < FeatureCount; i++) doubled.Scales[i] = 2.0 * scales[i];
        var doubledExplainer = new Explainer866(doubled, background);
        bool linearity = true;
        foreach (double[] x in instances)
        {
            double[] single = explainer.Exact(x);
            double[] twice = doubledExplainer.Exact(x);
            for (int i = 0; i < FeatureCount; i++) if (Math.Abs(twice[i] - 2.0 * single[i]) > 1e-9) linearity = false;
        }
        Check(linearity, "attributions are linear in the model: doubling the model doubles every attribution");
        var pairModel = new Model866(2, 4.0, new[] { 0.0, 0.0 });
        var pairExplainer = new Explainer866(pairModel, new[] { new[] { 0.0, 0.0 } });
        double[] pairPhi = pairExplainer.Exact(new[] { 2.0, 3.0 });
        Check(Math.Abs(pairPhi[0] - 12.0) < 1e-9 && Math.Abs(pairPhi[1] - 12.0) < 1e-9
            && Math.Abs(pairPhi[0] + pairPhi[1] - 24.0) < 1e-9,
            "a pure pairwise interaction is split evenly between its two features");
        var singleModel = new Model866(1, 1.0, new[] { 2.0 });
        var singleExplainer = new Explainer866(singleModel, new[] { new[] { 1.0 }, new[] { 3.0 } });
        double[] singlePhi = singleExplainer.Exact(new[] { 4.0 });
        Check(Math.Abs(singlePhi[0] - (singleModel.Predict(new[] { 4.0 }) - singleExplainer.Baseline)) < 1e-9,
            "a one-feature model gives that feature the whole difference from the baseline");

        // 12-14. Determinism, background-sample sensitivity and degenerate inputs.
        double[] first = explainer.Permutation(instances[3], 256, 12345UL);
        double[] second = explainer.Permutation(instances[3], 256, 12345UL);
        bool identical = true;
        for (int i = 0; i < FeatureCount; i++) if (first[i] != second[i]) identical = false;
        Check(identical, "the permutation sampler is reproducible for a fixed seed");
        var twiceBackground = new double[BackgroundRows * 2][];
        for (int i = 0; i < BackgroundRows; i++)
        {
            twiceBackground[i] = background[i];
            twiceBackground[i + BackgroundRows] = background[i];
        }
        var sameExplainer = new Explainer866(model, twiceBackground);
        double[] withDuplicate = sameExplainer.Exact(instances[1]);
        double[] withSingle = explainer.Exact(instances[1]);
        bool backgroundInvariant = true;
        for (int i = 0; i < FeatureCount; i++)
            if (Math.Abs(withDuplicate[i] - withSingle[i]) > 1e-9) backgroundInvariant = false;
        Check(backgroundInvariant, "duplicating every background row leaves the attributions unchanged");
        Check(Throws(() => new Explainer866(model, new double[0][]))
            && Throws(() => new Explainer866(model, new[] { new[] { 1.0, 2.0 } }))
            && Throws(() => model.Predict(new[] { 1.0, 2.0 })),
            "an empty background, a wrong-width row and a wrong-width instance are rejected");
        var meanInstance = new double[FeatureCount];
        double[] atBaseline = explainer.Exact(meanInstance);
        double baselineSum = 0.0;
        foreach (double value in atBaseline) baselineSum += value;
        Check(Math.Abs(baselineSum - (model.Predict(meanInstance) - explainer.Value(meanInstance, new bool[FeatureCount]))) < 1e-9,
            "an all-zero instance is explained without error and still satisfies efficiency");

        Console.WriteLine("PASS");
    }

    private static double[][] BuildBackground()
    {
        var background = new double[BackgroundRows][];
        ulong seed = 0x866BUL;
        for (int r = 0; r < BackgroundRows; r++)
        {
            var row = new double[FeatureCount];
            for (int c = 0; c < FeatureCount; c++) row[c] = Math.Round(Rng866.Uniform(ref seed) * 4.0, 3);
            background[r] = row;
        }
        return background;
    }

    private static List<double[]> BuildInstances()
    {
        var instances = new List<double[]>();
        ulong seed = 0x866CUL;
        for (int i = 0; i < Instances; i++)
        {
            var x = new double[FeatureCount];
            for (int c = 0; c < FeatureCount; c++) x[c] = Math.Round(Rng866.Uniform(ref seed) * 5.0, 3);
            instances.Add(x);
        }
        return instances;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
