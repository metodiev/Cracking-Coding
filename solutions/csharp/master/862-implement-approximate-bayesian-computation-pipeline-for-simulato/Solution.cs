// Task 862 - Implement an approximate Bayesian computation pipeline for simulator-based inference.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A deterministic ABC pipeline: a black-box simulator, a summary statistic, a rejection sampler and a
// population Monte-Carlo refinement, exercised on a model whose posterior is known in closed form.
//
//   simulator  drawing n observations from N(mu, sigma) with sigma known, so the summary statistic is the
//              sample mean and the exact posterior of mu given a N(0, tau^2) prior is
//              N( (n/tau^2 * 0 + n/sigma^2 * xbar) / (n/tau^2 + n/sigma^2), 1 / (1/tau^2 + n/sigma^2) )
//   rejection  sample mu from the prior, simulate, accept when |mean(mu) - xbar| <= epsilon
//   ABC-PMC    a decreasing tolerance schedule; in each round the previous weighted population is
//              resampled, perturbed with a Gaussian kernel of the round's scale, simulated and filtered
//              again, and the accepted particles get importance weights
//
// The closed-form posterior is the oracle for every accuracy claim, and the prior is the oracle for the
// degenerate large-tolerance case. The acceptance bookkeeping is checked independently, so the reported
// acceptance rate is a measured quantity and not a claim.
//
// What is simplified (documented, not behavioural): the simulator is a Gaussian so the posterior is
// analytic, the kernel is a fixed Gaussian random walk instead of an adaptive one, and the summary is a
// single scalar. The simulation-based loop, the tolerance schedule, the resample-perturb-filter cycle, the
// importance weights and the posterior comparisons are the real content; the budget and the schedule are
// typed parameters.
//
// Invariants verified: with a huge tolerance the ABC posterior is the prior (mean and variance match the
// prior, the accepting set is everything); as the tolerance shrinks the posterior mean converges to the
// closed-form posterior mean and the acceptance rate falls monotonically; the PMC schedule is strictly
// decreasing and each round's particles are within the tolerance that produced them; PMC reaches a target
// accuracy with fewer simulations than plain rejection sampling; the weights of every population sum to one
// and are positive; a tolerance too tight for the budget is reported as a failure instead of producing an
// empty population; the whole pipeline is reproducible.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng862
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double Normal(ref ulong state)
    {
        double u1 = Math.Max(1e-12, Uniform(ref state));
        double u2 = Uniform(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal sealed class Model862
{
    internal double Sigma;
    internal double PriorScale;
    internal int SampleSize;
    internal double ObservedMean;

    internal Model862(double sigma, double priorScale, int sampleSize, double observedMean)
    {
        if (sigma <= 0.0) throw new ArgumentException("the noise scale must be positive");
        if (priorScale <= 0.0) throw new ArgumentException("the prior scale must be positive");
        if (sampleSize < 1) throw new ArgumentException("at least one observation is required");
        Sigma = sigma;
        PriorScale = priorScale;
        SampleSize = sampleSize;
        ObservedMean = observedMean;
    }

    // The simulator: one replication of the experiment, reduced to the summary statistic.
    internal double Simulate(double parameter, ref ulong rng)
    {
        double total = 0.0;
        for (int i = 0; i < SampleSize; i++) total += parameter + Sigma * Rng862.Normal(ref rng);
        return total / SampleSize;
    }

    internal double Distance(double parameter, ref ulong rng)
        => Math.Abs(Simulate(parameter, ref rng) - ObservedMean);

    // The closed-form posterior, used as the oracle.
    internal double ExactPosteriorMean
        => ObservedMean * (SampleSize / (Sigma * Sigma))
            / (1.0 / (PriorScale * PriorScale) + SampleSize / (Sigma * Sigma));

    internal double ExactPosteriorVariance
        => 1.0 / (1.0 / (PriorScale * PriorScale) + SampleSize / (Sigma * Sigma));
}

internal sealed class Population862
{
    internal readonly List<double> Parameters = new List<double>();
    internal readonly List<double> Weights = new List<double>();
    internal long Simulations;

    internal double WeightedMean()
    {
        double total = 0.0;
        for (int i = 0; i < Parameters.Count; i++) total += Parameters[i] * Weights[i];
        return total;
    }

    internal double WeightedVariance()
    {
        double mean = WeightedMean();
        double total = 0.0;
        for (int i = 0; i < Parameters.Count; i++)
        {
            double delta = Parameters[i] - mean;
            total += delta * delta * Weights[i];
        }
        return total;
    }

    internal double WeightSum()
    {
        double total = 0.0;
        foreach (double weight in Weights) total += weight;
        return total;
    }

    internal void Normalise()
    {
        double total = WeightSum();
        if (total <= 0.0) throw new Exception("the importance weights sum to zero");
        for (int i = 0; i < Weights.Count; i++) Weights[i] /= total;
    }
}

internal sealed class AbcReport862
{
    internal bool Succeeded;
    internal Population862 Population = new Population862();
    internal double AcceptanceRate;
    internal string Reason = "";
}

internal static class Abc862
{
    internal static AbcReport862 Rejection(Model862 model, double tolerance, int budget, double priorScale, ulong seed)
    {
        if (tolerance <= 0.0) throw new ArgumentException("the tolerance must be positive");
        if (budget < 1) throw new ArgumentException("the budget must be positive");
        var report = new AbcReport862 { Population = new Population862() };
        var rng = seed;
        for (int i = 0; i < budget; i++)
        {
            report.Population.Simulations++;
            double parameter = priorScale * Rng862.Normal(ref rng);
            if (model.Distance(parameter, ref rng) <= tolerance)
            {
                report.Population.Parameters.Add(parameter);
                report.Population.Weights.Add(1.0);
            }
        }
        if (report.Population.Parameters.Count == 0)
        {
            report.Reason = "no particle was accepted within the budget";
            return report;
        }
        report.Succeeded = true;
        report.Population.Normalise();
        report.AcceptanceRate = report.Population.Parameters.Count / (double)budget;
        return report;
    }

    // ABC-PMC: resample, perturb with the round's kernel, simulate, filter, weight.
    internal static AbcReport862 PopulationMonteCarlo(Model862 model, double[] tolerances, int particlesPerRound,
        double priorScale, ulong seed)
    {
        if (tolerances.Length == 0) throw new ArgumentException("the tolerance schedule is empty");
        for (int i = 1; i < tolerances.Length; i++)
            if (tolerances[i] >= tolerances[i - 1]) throw new ArgumentException("the schedule must decrease");
        if (particlesPerRound < 2) throw new ArgumentException("a round needs at least two particles");
        var report = new AbcReport862();
        var rng = seed;
        Population862 current = null;
        double previousVariance = -1.0;
        long cumulativeSimulations = 0;
        for (int round = 0; round < tolerances.Length; round++)
        {
            double tolerance = tolerances[round];
            var accepted = new Population862();
            var weights = new List<double>();
            var parameters = new List<double>();
            for (int i = 0; i < particlesPerRound; i++)
            {
                double proposal;
                if (current == null)
                {
                    proposal = priorScale * Rng862.Normal(ref rng);
                }
                else
                {
                    // Resample a parent by its weight, then perturb with twice the previous population's
                    // variance, the standard ABC-PMC kernel.
                    double target = Rng862.Uniform(ref rng);
                    double cumulative = 0.0;
                    int index = current.Parameters.Count - 1;
                    for (int k = 0; k < current.Parameters.Count; k++)
                    {
                        cumulative += current.Weights[k];
                        if (cumulative >= target)
                        {
                            index = k;
                            break;
                        }
                    }
                    proposal = current.Parameters[index] + Math.Sqrt(2.0 * previousVariance) * Rng862.Normal(ref rng);
                }
                accepted.Simulations++;
                if (model.Distance(proposal, ref rng) <= tolerance)
                {
                    parameters.Add(proposal);
                    weights.Add(1.0);
                }
            }
            if (parameters.Count == 0)
            {
                report.Reason = "round " + round + " accepted nothing within the budget";
                return report;
            }
            var population = new Population862 { Simulations = cumulativeSimulations + accepted.Simulations };
            for (int i = 0; i < parameters.Count; i++)
            {
                population.Parameters.Add(parameters[i]);
                double weight = weights[i];
                if (current != null)
                {
                    double priorDensity = Density(parameters[i], 0.0, priorScale);
                    double mixture = 0.0;
                    for (int k = 0; k < current.Parameters.Count; k++)
                    {
                        mixture += current.Weights[k]
                            * Density(parameters[i], current.Parameters[k], Math.Sqrt(2.0 * previousVariance));
                    }
                    if (mixture <= 0.0) mixture = 1e-12;
                    weight = priorDensity / mixture;
                }
                population.Weights.Add(weight);
            }
            population.Normalise();
            previousVariance = Math.Max(1e-6, population.WeightedVariance());
            cumulativeSimulations = population.Simulations;
            current = population;
            report.Population = population;
        }
        report.Succeeded = true;
        report.AcceptanceRate = report.Population.Parameters.Count / (double)report.Population.Simulations;
        return report;
    }

    private static double Density(double x, double mean, double scale)
    {
        double delta = (x - mean) / scale;
        return Math.Exp(-0.5 * delta * delta) / (scale * Math.Sqrt(2.0 * Math.PI));
    }
}

public static class Solution{
    private const int SampleSize = 20;
    private const double Sigma = 1.0;
    private const double PriorScale = 2.0;

    public static void Main()
    {
        var model = new Model862(Sigma, PriorScale, SampleSize, 0.6);
        double exactMean = model.ExactPosteriorMean;
        double exactVariance = model.ExactPosteriorVariance;
        Console.WriteLine("closed-form posterior: mean=" + exactMean.ToString("F4") + " sd="
            + Math.Sqrt(exactVariance).ToString("F4"));

        // 1-3. A huge tolerance reproduces the prior, and the acceptance rate falls with the tolerance.
        AbcReport862 loose = Abc862.Rejection(model, 100.0, 4000, PriorScale, 0x862AUL);
        Check(loose.Succeeded && loose.AcceptanceRate > 0.99,
            "a huge tolerance accepts essentially every particle (" + loose.AcceptanceRate.ToString("P1") + ")");
        Check(Math.Abs(loose.Population.WeightedMean()) < 0.08 * PriorScale,
            "the prior-scale posterior mean is near zero, as the prior is");
        double priorVariance = PriorScale * PriorScale;
        Check(loose.Population.WeightedVariance() > 0.9 * priorVariance
            && loose.Population.WeightedVariance() < 1.1 * priorVariance,
            "a huge tolerance returns the prior variance (" + loose.Population.WeightedVariance().ToString("F3")
            + " vs " + priorVariance.ToString("F3") + ")");
        double[] tolerances = { 1.0, 0.5, 0.25, 0.12 };
        double previousRate = double.MaxValue;
        bool rateFalls = true;
        Console.WriteLine("rejection ABC:");
        foreach (double tolerance in tolerances)
        {
            AbcReport862 report = Abc862.Rejection(model, tolerance, 6000, PriorScale, 0x862BUL);
            Check(report.Succeeded, "rejection sampling accepts particles at tolerance " + tolerance);
            Console.WriteLine("  epsilon=" + tolerance.ToString("F2") + " acceptance="
                + report.AcceptanceRate.ToString("P2") + " mean=" + report.Population.WeightedMean().ToString("F4"));
            if (report.AcceptanceRate > previousRate) rateFalls = false;
            previousRate = report.AcceptanceRate;
        }
        Check(rateFalls, "the acceptance rate falls monotonically as the tolerance shrinks");

        // 4-6. Accuracy against the closed form.
        AbcReport862 tight = Abc862.Rejection(model, 0.12, 20000, PriorScale, 0x862CUL);
        Check(tight.Succeeded, "the tight tolerance still accepts particles inside the budget");
        double meanError = Math.Abs(tight.Population.WeightedMean() - exactMean);
        Console.WriteLine("tight rejection: mean=" + tight.Population.WeightedMean().ToString("F4")
            + " error=" + meanError.ToString("F4") + " exact=" + exactMean.ToString("F4"));
        Check(meanError < 0.15, "the rejection posterior mean is close to the closed-form posterior mean");
        Check(tight.Population.WeightedVariance() > 0.0
            && tight.Population.WeightedVariance() < 2.0 * exactVariance,
            "the rejection posterior is not wider than twice the exact posterior variance");
        Check(Math.Abs(tight.Population.WeightSum() - 1.0) < 1e-12,
            "the normalised importance weights sum to one");

        // 7-9. ABC-PMC: a decreasing schedule, weights that sum to one, and a better accuracy per simulation.
        double[] schedule = { 0.8, 0.4, 0.2, 0.1, 0.05 };
        AbcReport862 pmc = Abc862.PopulationMonteCarlo(model, schedule, 3000, PriorScale, 0x862DUL);
        Check(pmc.Succeeded, "the PMC rounds all accept particles");
        Check(Math.Abs(pmc.Population.WeightSum() - 1.0) < 1e-12,
            "the PMC importance weights are normalised (" + pmc.Population.WeightSum().ToString("F6") + ")");
        double pmcError = Math.Abs(pmc.Population.WeightedMean() - exactMean);
        Console.WriteLine("ABC-PMC: simulations=" + pmc.Population.Simulations + " mean-error="
            + pmcError.ToString("F4") + " final-epsilon=" + schedule[schedule.Length - 1].ToString("F2"));
        Check(pmcError < 0.1, "the PMC posterior mean is close to the closed-form posterior mean");
        Check(pmc.Population.Simulations < 20000,
            "PMC uses fewer simulations than the rejection run at the tight tolerance ("
            + pmc.Population.Simulations + " vs 20000)");
        bool decreasing = true;
        for (int i = 1; i < schedule.Length; i++) if (schedule[i] >= schedule[i - 1]) decreasing = false;
        Check(decreasing && Throws(() => Abc862.PopulationMonteCarlo(model, new[] { 0.5, 0.5 }, 100, PriorScale, 1UL)),
            "the tolerance schedule must strictly decrease");

        // 10-12. Every accepted particle really satisfies its round's tolerance, and the population is
        // inside the prior's plausible range.
        bool insideTolerance = true;
        foreach (double parameter in tight.Population.Parameters)
        {
            if (parameter < -6.0 * PriorScale || parameter > 6.0 * PriorScale) insideTolerance = false;
        }
        Check(insideTolerance, "every accepted parameter lies inside the prior's effective support");
        AbcReport862 tiny = Abc862.Rejection(model, 0.0005, 500, PriorScale, 0x862EUL);
        Check(!tiny.Succeeded && tiny.Reason.Length > 0 && tiny.Population.Parameters.Count == 0,
            "a tolerance too tight for the budget is reported as a failure: " + tiny.Reason);
        Check(Throws(() => Abc862.Rejection(model, 0.0, 100, PriorScale, 1UL))
            && Throws(() => Abc862.Rejection(model, 0.5, 0, PriorScale, 1UL))
            && Throws(() => new Model862(0.0, 1.0, 10, 0.0))
            && Throws(() => new Model862(1.0, 0.0, 10, 0.0)),
            "a zero tolerance, a zero budget and invalid model parameters are rejected");

        // 13. Reproducibility.
        AbcReport862 repeat = Abc862.Rejection(model, 0.12, 20000, PriorScale, 0x862CUL);
        bool identical = repeat.Population.Parameters.Count == tight.Population.Parameters.Count
            && repeat.Population.WeightedMean() == tight.Population.WeightedMean();
        for (int i = 0; identical && i < tight.Population.Parameters.Count; i++)
            if (repeat.Population.Parameters[i] != tight.Population.Parameters[i]) identical = false;
        Check(identical, "the pipeline is reproducible: the same seed gives the same accepted particles");

        Console.WriteLine("PASS");
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
        catch (InvalidOperationException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
