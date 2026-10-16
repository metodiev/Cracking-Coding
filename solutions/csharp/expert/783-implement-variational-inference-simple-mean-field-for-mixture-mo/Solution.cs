// Task 783 - Implement variational inference simple mean-field for mixture models. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
//
// A conjugate Bayesian mixture of Gaussians with a known variance is the smallest model in which
// mean-field inference can be compared against the truth:
//
//   p(mu_k) = N(0, tau^2),  p(z_i = k) = 1/K,  p(x_i | z_i = k, mu) = N(mu_k, sigma^2)
//
// The mean-field family factorises as q(mu) q(z) with q(mu_k) = N(m_k, s_k^2) and q(z_i) a categorical
// r_i, and the coordinate ascent updates follow from the standard conjugacy: with the responsibilities
// fixed, s_k^2 = 1/(1/tau^2 + (1/sigma^2) sum_i r_ik) and m_k = s_k^2 (1/sigma^2) sum_i r_ik x_i, and
// with the means fixed, r_ik is proportional to exp(-(x_i - m_k)^2 / (2 sigma^2) - s_k^2 / (2 sigma^2)).
//
// Two things make this a real check rather than a self-fulfilling loop. First, the ELBO is assembled
// twice - once from the closed-form expected log joint minus the entropy, and once from the
// energy/entropy terms written independently - and the two values have to agree to 1e-9. Second, the
// exact log evidence is obtained by numerically integrating the K-dimensional evidence integral on a
// grid, which gives the variational inequality ELBO <= log p(x) something to be measured against: the
// gap is reported for well separated and for overlapping clusters, and shrinks with the number of
// coordinate ascent sweeps.
//
// Verified: the ELBO never decreases across sweeps (the defining property of coordinate ascent), it
// stays below the exact log evidence, both ELBO formulas agree, responsibilities sum to one for every
// point, the posterior variances shrink below the prior (s_k^2 < tau^2), well separated clusters are
// recovered with high accuracy against the labels used to generate the data, K = 1 reproduces the
// exact conjugate posterior and its evidence, two different initialisations reach different local
// optima whose ELBOs are both below the evidence, and the degenerate cases (one data point, repeated
// points, invalid hyperparameters) behave sanely.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    /// <summary>Box-Muller from two uniforms, so the whole file stays deterministic.</summary>
    internal static double NextNormal(ref ulong state)
    {
        double u1 = Math.Max(NextUnit(ref state), 1e-12);
        double u2 = NextUnit(ref state);
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }
}

internal sealed class MeanFieldMixture
{
    internal readonly int Components;
    internal readonly double[] Data;
    internal readonly int[] Labels;                 // ground truth, only used for the recovery check
    internal readonly double Sigma;
    internal readonly double Tau;
    internal readonly double[] Mean;
    internal readonly double[] Variance;
    internal readonly double[][] Responsibility;
    internal readonly List<double> ElboHistory = new List<double>();
    internal int Sweeps;

    internal MeanFieldMixture(double[] data, int components, double sigma, double tau, int[] labels, ulong seed)
    {
        if (components < 1) throw new ArgumentException("a mixture needs at least one component");
        if (!(sigma > 0) || !(tau > 0)) throw new ArgumentException("sigma and tau must be positive");
        Components = components;
        Data = data;
        Labels = labels ?? new int[data.Length];
        Sigma = sigma;
        Tau = tau;
        Mean = new double[components];
        Variance = new double[components];
        Responsibility = new double[data.Length][];
        for (int i = 0; i < data.Length; i++) Responsibility[i] = new double[components];
        ulong state = seed;
        for (int k = 0; k < components; k++)
        {
            Mean[k] = (SplitMix64.NextUnit(ref state) - 0.5) * 2 * tau;      // deterministic spread out initialisation
            Variance[k] = tau * tau;
        }
        for (int i = 0; i < data.Length; i++)
        {
            for (int k = 0; k < components; k++) Responsibility[i][k] = 1.0 / components;
        }
    }

    /// <summary>One coordinate ascent sweep: responsibilities first, then the Gaussian posteriors.</summary>
    internal double Sweep()
    {
        int n = Data.Length;
        for (int i = 0; i < n; i++)
        {
            var logWeights = new double[Components];
            double max = double.NegativeInfinity;
            for (int k = 0; k < Components; k++)
            {
                double diff = Data[i] - Mean[k];
                logWeights[k] = -(diff * diff) / (2 * Sigma * Sigma) - Variance[k] / (2 * Sigma * Sigma);
                if (logWeights[k] > max) max = logWeights[k];
            }
            double total = 0;
            for (int k = 0; k < Components; k++)
            {
                logWeights[k] = Math.Exp(logWeights[k] - max);
                total += logWeights[k];
            }
            for (int k = 0; k < Components; k++) Responsibility[i][k] = total <= 0 ? 1.0 / Components : logWeights[k] / total;
        }

        for (int k = 0; k < Components; k++)
        {
            double sum = 0, weighted = 0;
            for (int i = 0; i < n; i++)
            {
                sum += Responsibility[i][k];
                weighted += Responsibility[i][k] * Data[i];
            }
            double precision = 1.0 / (Tau * Tau) + sum / (Sigma * Sigma);
            Variance[k] = 1.0 / precision;
            Mean[k] = Variance[k] * weighted / (Sigma * Sigma);
        }

        Sweeps++;
        double elbo = Elbo();
        ElboHistory.Add(elbo);
        return elbo;
    }

    /// <summary>
    /// ELBO, first formula: the expected log joint of the truncated model minus the entropy of q. The
    /// truncated model drops the log(1/K) prior term of the assignments (it is a constant), so the
    /// value is a lower bound on log p(x) - log(1/K)^n, which is exactly what the evidence comparison
    /// below accounts for.
    /// </summary>
    internal double Elbo()
    {
        int n = Data.Length;
        double expectedLogLikelihood = 0, expectedLogPrior = 0, entropyAssignments = 0, entropyMeans = 0;
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < Components; k++)
            {
                double r = Responsibility[i][k];
                if (r <= 0) continue;
                double diff = Data[i] - Mean[k];
                expectedLogLikelihood -= r * (diff * diff + Variance[k]) / (2 * Sigma * Sigma);
                entropyAssignments -= r * Math.Log(r);
            }
        }
        for (int k = 0; k < Components; k++)
        {
            expectedLogPrior -= (Mean[k] * Mean[k] + Variance[k]) / (2 * Tau * Tau);
            expectedLogPrior -= 0.5 * Math.Log(2 * Math.PI * Tau * Tau);
            entropyMeans += 0.5 * Math.Log(2 * Math.PI * Math.E * Variance[k]);
        }
        double constant = -0.5 * n * Math.Log(2 * Math.PI * Sigma * Sigma);
        return constant + expectedLogLikelihood + expectedLogPrior + entropyAssignments + entropyMeans;
    }

    /// <summary>
    /// ELBO, second formula: the same quantity assembled from the energy terms and entropies in a
    /// different order, with the Gaussian cross-entropies written out longhand. The two must agree.
    /// </summary>
    internal double ElboEnergyEntropy()
    {
        int n = Data.Length;
        double energyData = 0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < Components; k++)
            {
                double r = Responsibility[i][k];
                if (r <= 0) continue;
                double diff = Data[i] - Mean[k];
                energyData += -0.5 * r * (diff * diff + Variance[k]) / (Sigma * Sigma) - 0.5 * r * Math.Log(2 * Math.PI * Sigma * Sigma);
            }
        double energyPrior = 0, entropyMean = 0, entropyAssign = 0;
        for (int k = 0; k < Components; k++)
        {
            energyPrior += -0.5 * (Mean[k] * Mean[k] + Variance[k]) / (Tau * Tau) - 0.5 * Math.Log(2 * Math.PI * Tau * Tau);
            entropyMean += 0.5 * Math.Log(2 * Math.PI * Variance[k]) + 0.5;                // -E_q[log q(mu)] in longhand
        }
        for (int i = 0; i < n; i++)
            for (int k = 0; k < Components; k++)
            {
                double r = Responsibility[i][k];
                if (r > 0) entropyAssign += -r * Math.Log(r);
            }
        return energyData + energyPrior + entropyMean + entropyAssign;
    }

    internal int[] HardAssignment()
    {
        var labels = new int[Data.Length];
        for (int i = 0; i < Data.Length; i++)
        {
            int best = 0;
            for (int k = 1; k < Components; k++) if (Responsibility[i][k] > Responsibility[i][best]) best = k;
            labels[i] = best;
        }
        return labels;
    }

    internal string InvariantViolation()
    {
        for (int i = 0; i < Data.Length; i++)
        {
            double sum = 0;
            for (int k = 0; k < Components; k++)
            {
                if (Responsibility[i][k] < 0 || Responsibility[i][k] > 1) return "responsibility out of [0,1]";
                sum += Responsibility[i][k];
            }
            if (Math.Abs(sum - 1) > 1e-9) return "responsibilities of point " + i + " sum to " + sum;
        }
        for (int k = 0; k < Components; k++)
        {
            if (!(Variance[k] > 0)) return "variance of component " + k + " is not positive";
            if (Variance[k] > Tau * Tau + 1e-12) return "the posterior variance of component " + k + " exceeds the prior";
        }
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Exact log evidence of the truncated mixture model by grid integration over the means.</summary>
    private static double ExactLogEvidence(double[] data, int components, double sigma, double tau, int grid)
    {
        double[][] axis = new double[components][];
        for (int k = 0; k < components; k++)
        {
            axis[k] = new double[grid];
            for (int g = 0; g < grid; g++) axis[k][g] = -6 * tau + 12 * tau * g / (grid - 1);
        }
        double step = axis[0][1] - axis[0][0];
        var means = new double[components];
        double total = 0;
        var indices = new int[components];
        while (true)
        {
            for (int k = 0; k < components; k++) means[k] = axis[k][indices[k]];
            double weight = 1;
            for (int k = 0; k < components; k++)
            {
                weight *= Math.Exp(-means[k] * means[k] / (2 * tau * tau)) / Math.Sqrt(2 * Math.PI * tau * tau);
            }
            for (int i = 0; i < data.Length; i++)
            {
                double mixture = 0;
                for (int k = 0; k < components; k++)
                {
                    double diff = data[i] - means[k];
                    mixture += Math.Exp(-diff * diff / (2 * sigma * sigma)) / Math.Sqrt(2 * Math.PI * sigma * sigma);
                }
                weight *= mixture / components;                 // the 1/K assignment prior
            }
            total += weight * Math.Pow(step, components);

            int position = components - 1;
            while (position >= 0)
            {
                indices[position]++;
                if (indices[position] < grid) break;
                indices[position] = 0;
                position--;
            }
            if (position < 0) break;
        }
        return Math.Log(total);
    }

    private static double[] GeneratedData(ulong seed, int perCluster, double[] centers, double sigma, int[] labels)
    {
        ulong state = seed;
        var values = new List<double>();
        for (int c = 0; c < centers.Length; c++)
            for (int i = 0; i < perCluster; i++)
            {
                values.Add(centers[c] + sigma * SplitMix64.NextNormal(ref state));
                labels[values.Count - 1] = c;
            }
        return values.ToArray();
    }

    private static double Accuracy(int[] inferred, int[] truth, int components)
    {
        // best permutation of the inferred labels against the truth
        var permutation = new int[components];
        var used = new bool[components];
        double best = 0;
        void Search(int index, double score)
        {
            if (index == components)
            {
                if (score > best) best = score;
                return;
            }
            for (int c = 0; c < components; c++)
            {
                if (used[c]) continue;
                used[c] = true;
                permutation[index] = c;
                double gained = 0;
                for (int i = 0; i < truth.Length; i++) if (truth[i] == index && inferred[i] == c) gained++;
                Search(index + 1, score + gained);
                used[c] = false;
            }
        }
        Search(0, 0);
        return best / truth.Length;
    }

    public static void Main()
    {
        // ---- well separated clusters: coordinate ascent and the variational bound -------------------
        {
            const int Components = 3, PerCluster = 8;
            double sigma = 0.5, tau = 3.0;
            var labels = new int[Components * PerCluster];
            double[] data = GeneratedData(0x5EED783000000001UL, PerCluster, new[] { -6.0, 0.0, 6.0 }, sigma, labels);
            var model = new MeanFieldMixture(data, Components, sigma, tau, labels, 0x5EED783000000002UL);
            double previous = double.NegativeInfinity;
            bool monotone = true;
            double firstElbo = 0, lastElbo = 0;
            for (int sweep = 0; sweep < 60; sweep++)
            {
                double elbo = model.Sweep();
                if (sweep == 0) firstElbo = elbo;
                lastElbo = elbo;
                if (elbo < previous - 1e-9) monotone = false;
                previous = elbo;
                Check(model.InvariantViolation() == null, "invariants hold after sweep " + sweep + ": " + model.InvariantViolation());
            }
            Check(monotone, "the ELBO never decreases across the 60 sweeps");
            Check(lastElbo >= firstElbo, "and it ends no lower than it started (" + firstElbo.ToString("F4") + " -> " + lastElbo.ToString("F4") + ")");
            Check(Math.Abs(model.Elbo() - model.ElboEnergyEntropy()) < 1e-9,
                "the two ELBO formulas agree (" + model.Elbo().ToString("F10") + " vs " + model.ElboEnergyEntropy().ToString("F10") + ")");

            double exact = ExactLogEvidence(data, Components, sigma, tau, 80) + data.Length * Math.Log(Components);
            Check(lastElbo <= exact + 1e-3, "the ELBO stays below the exact log evidence of the truncated model ("
                + lastElbo.ToString("F6") + " <= " + exact.ToString("F6") + ")");
            double gap = exact - lastElbo;
            Check(gap >= 0, "and the variational gap is non-negative (" + gap.ToString("F4") + ")");
            Check(gap < 5.0, "and bounded for separated clusters (" + gap.ToString("F4") + ")");

            double accuracy = Accuracy(model.HardAssignment(), labels, Components);
            Check(accuracy > 0.9, "the clusters are recovered (" + accuracy.ToString("F4") + " of the points)");
            bool variancesShrunk = true;
            for (int k = 0; k < Components; k++) if (!(model.Variance[k] < tau * tau)) variancesShrunk = false;
            Check(variancesShrunk, "every posterior variance is below the prior variance");
            Console.WriteLine("separated: elbo=" + lastElbo.ToString("F4") + " exact=" + exact.ToString("F4")
                + " gap=" + gap.ToString("F4") + " accuracy=" + accuracy.ToString("F4") + " sweeps=" + model.Sweeps);
        }

        // ---- overlapping clusters: the gap grows, the bound still holds -----------------------------
        {
            const int Components = 2, PerCluster = 10;
            double sigma = 1.2, tau = 3.0;
            var labels = new int[Components * PerCluster];
            double[] data = GeneratedData(0x5EED783000000003UL, PerCluster, new[] { -1.0, 1.0 }, sigma, labels);
            var model = new MeanFieldMixture(data, Components, sigma, tau, labels, 0x5EED783000000004UL);
            double elbo = 0;
            for (int sweep = 0; sweep < 80; sweep++) elbo = model.Sweep();
            double exact = ExactLogEvidence(data, Components, sigma, tau, 120) + data.Length * Math.Log(Components);
            Check(elbo <= exact + 1e-3, "the bound holds for overlapping clusters too (" + elbo.ToString("F6")
                + " <= " + exact.ToString("F6") + ")");
            Check(model.InvariantViolation() == null, "the overlapping model keeps its invariants");
            double gap = exact - elbo;
            Check(gap >= 0, "a positive gap remains because the family cannot represent the coupling (" + gap.ToString("F4") + ")");
            Console.WriteLine("overlapping: elbo=" + elbo.ToString("F4") + " exact=" + exact.ToString("F4")
                + " gap=" + gap.ToString("F4") + " accuracy=" + Accuracy(model.HardAssignment(), labels, Components).ToString("F4"));
        }

        // ---- K = 1 is exactly conjugate, so mean field is exact ------------------------------------
        {
            const int Components = 1, PerCluster = 12;
            double sigma = 0.8, tau = 2.0;
            var labels = new int[PerCluster];
            double[] data = GeneratedData(0x5EED783000000005UL, PerCluster, new[] { 1.5 }, sigma, labels);
            var model = new MeanFieldMixture(data, Components, sigma, tau, labels, 0x5EED783000000006UL);
            double previous = double.NegativeInfinity;
            bool monotone = true;
            for (int sweep = 0; sweep < 100; sweep++)
            {
                double elbo = model.Sweep();
                if (elbo < previous - 1e-9) monotone = false;
                previous = elbo;
            }
            double sum = 0;
            foreach (double x in data) sum += x;
            double expectedMean = sum / (data.Length + sigma * sigma / (tau * tau));
            double expectedVariance = 1.0 / (1.0 / (tau * tau) + data.Length / (sigma * sigma));
            Check(monotone, "the single-component ELBO is monotone");
            Check(Math.Abs(model.Mean[0] - expectedMean) < 1e-9, "the posterior mean is the conjugate mean ("
                + model.Mean[0].ToString("F8") + " vs " + expectedMean.ToString("F8") + ")");
            Check(Math.Abs(model.Variance[0] - expectedVariance) < 1e-12, "and the posterior variance is conjugate");
            double exact = ExactLogEvidence(data, Components, sigma, tau, 400) + data.Length * Math.Log(Components);
            Check(Math.Abs(model.Elbo() - exact) < 0.05, "mean field is exact for one component (ELBO "
                + model.Elbo().ToString("F6") + " vs evidence " + exact.ToString("F6") + ")");
            Console.WriteLine("single component: mean=" + model.Mean[0].ToString("F6") + " variance=" + model.Variance[0].ToString("F6")
                + " elbo=" + model.Elbo().ToString("F6") + " exact=" + exact.ToString("F6"));
        }

        // ---- local optima: two initialisations, two answers, both bounds hold -----------------------
        {
            const int Components = 3, PerCluster = 6;
            double sigma = 1.0, tau = 2.0;
            var labels = new int[Components * PerCluster];
            double[] data = GeneratedData(0x5EED783000000007UL, PerCluster, new[] { -3.0, 0.0, 3.0 }, sigma, labels);
            var first = new MeanFieldMixture(data, Components, sigma, tau, labels, 0x5EED783000000008UL);
            var collapsed = new MeanFieldMixture(data, Components, sigma, tau, labels, 0x5EED783000000008UL);
            double average = 0;
            foreach (double x in data) average += x;
            average /= data.Length;
            for (int k = 0; k < Components; k++) collapsed.Mean[k] = average;   // a degenerate initialisation
            for (int sweep = 0; sweep < 60; sweep++) { first.Sweep(); collapsed.Sweep(); }
            double exact = ExactLogEvidence(data, Components, sigma, tau, 100) + data.Length * Math.Log(Components);
            Check(first.Elbo() <= exact + 1e-3 && collapsed.Elbo() <= exact + 1e-3,
                "both fixed points respect the bound (" + first.Elbo().ToString("F4") + ", " + collapsed.Elbo().ToString("F4")
                + " <= " + exact.ToString("F4") + ")");
            Check(collapsed.Elbo() < first.Elbo(), "the collapsed initialisation sticks in a worse local optimum ("
                + collapsed.Elbo().ToString("F4") + " < " + first.Elbo().ToString("F4") + ")");
            bool collapsedComponentsEqual = true;
            for (int k = 1; k < Components; k++) if (Math.Abs(collapsed.Mean[k] - collapsed.Mean[0]) > 1e-9) collapsedComponentsEqual = false;
            Check(collapsedComponentsEqual, "identical initial means stay identical, which is why the optimum is worse");
            Check(first.InvariantViolation() == null && collapsed.InvariantViolation() == null, "both solutions are consistent");
            Console.WriteLine("local optima: separated=" + first.Elbo().ToString("F4") + " collapsed=" + collapsed.Elbo().ToString("F4")
                + " exact=" + exact.ToString("F4"));
        }

        // ---- degenerate cases ------------------------------------------------------------------------
        {
            var onePoint = new MeanFieldMixture(new[] { 2.0 }, 2, 1.0, 2.0, new[] { 0 }, 0x5EED78300000000AUL);
            onePoint.Sweep();
            Check(onePoint.InvariantViolation() == null, "a single data point keeps the invariants");
            Check(Math.Abs(onePoint.Mean[0]) < 2.0 + 2.0 && Math.Abs(onePoint.Mean[1]) < 4.0,
                "the posterior means remain bounded by the data and the prior scale");

            var repeats = new double[10];
            for (int i = 0; i < repeats.Length; i++) repeats[i] = 1.0;
            var repeated = new MeanFieldMixture(repeats, 2, 0.5, 3.0, new int[10], 0x5EED78300000000BUL);
            double previous = double.NegativeInfinity;
            bool monotone = true;
            for (int sweep = 0; sweep < 40; sweep++)
            {
                double elbo = repeated.Sweep();
                if (elbo < previous - 1e-9) monotone = false;
                previous = elbo;
            }
            Check(monotone, "identical points keep the ELBO monotone");
            Check(repeated.InvariantViolation() == null, "identical points keep the invariants");
            double mean = 0;
            foreach (double x in repeats) mean += x;
            mean /= repeats.Length;
            bool capturesData = Math.Abs(repeated.Mean[0] - mean) < 0.5 || Math.Abs(repeated.Mean[1] - mean) < 0.5;
            Check(capturesData, "at least one component captures the repeated value (" + repeated.Mean[0].ToString("F4")
                + ", " + repeated.Mean[1].ToString("F4") + ")");
            Check(Math.Abs(repeated.Mean[0]) < 3 && Math.Abs(repeated.Mean[1]) < 3, "and both stay bounded by the prior");

            bool threwComponents = false;
            try { new MeanFieldMixture(new[] { 1.0 }, 0, 1, 1, null, 1UL); } catch (ArgumentException) { threwComponents = true; }
            Check(threwComponents, "zero components is rejected");
            bool threwSigma = false;
            try { new MeanFieldMixture(new[] { 1.0 }, 2, 0.0, 1, null, 1UL); } catch (ArgumentException) { threwSigma = true; }
            Check(threwSigma, "a zero observation noise is rejected");
            bool threwTau = false;
            try { new MeanFieldMixture(new[] { 1.0 }, 2, 1, -1, null, 1UL); } catch (ArgumentException) { threwTau = true; }
            Check(threwTau, "a negative prior scale is rejected");
            Console.WriteLine("degenerate: singlePointMeans=" + onePoint.Mean[0].ToString("F4") + "/" + onePoint.Mean[1].ToString("F4")
                + " repeatedMeans=" + repeated.Mean[0].ToString("F4") + "/" + repeated.Mean[1].ToString("F4"));
        }

        // ---- determinism -----------------------------------------------------------------------------
        {
            double[] data = GeneratedData(0x5EED78300000000CUL, 5, new[] { -2.0, 2.0 }, 0.9, new int[10]);
            var a = new MeanFieldMixture(data, 2, 0.9, 2.5, new int[10], 0x5EED78300000000DUL);
            var b = new MeanFieldMixture(data, 2, 0.9, 2.5, new int[10], 0x5EED78300000000DUL);
            for (int sweep = 0; sweep < 30; sweep++) { a.Sweep(); b.Sweep(); }
            Check(Math.Abs(a.Elbo() - b.Elbo()) < 1e-12, "two identical runs reach the same ELBO");
            Check(Math.Abs(a.Mean[0] - b.Mean[0]) < 1e-12 && Math.Abs(a.Mean[1] - b.Mean[1]) < 1e-12, "and the same means");
            Console.WriteLine("determinism: elbo=" + a.Elbo().ToString("F6") + " means=" + a.Mean[0].ToString("F4") + "/" + a.Mean[1].ToString("F4"));
        }

        Console.WriteLine("PASS");
    }
}
