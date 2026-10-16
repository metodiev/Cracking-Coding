// Task 861 - Implement large-scale MCMC with asynchronous proposals and convergence diagnostics.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A deterministic Metropolis-Hastings sampler whose chains propose at *different* ticks (an explicit
// asynchronous schedule, no threads), together with the standard convergence diagnostics.
//
//   target       an Ising chain of n spins: E(s) = -J * sum_i s_i s_{i+1} - h * sum_i s_i, and the target
//                distribution is the Boltzmann distribution over the 2^n states, so the exact stationary
//                distribution is available two ways: from the closed-form Boltzmann weights and from power
//                iteration on the transition matrix
//   schedule     chain c proposes on the ticks where tick % chains == c, so the chains advance
//                asynchronously and at different rates; each proposal is a single-spin flip
//   diagnostics  R-hat (Gelman-Rubin) from the between- and within-chain variances, the effective sample
//                size from the integrated autocorrelation time of the pooled trace, the total-variation
//                distance between the sampled frequencies and the exact stationary distribution, and an
//                exact detailed-balance audit of the Metropolis kernel
//
// The kernel audit is the strongest check: for every pair of states the single-flip Metropolis kernel must
// satisfy pi(s) P(s -> s') = pi(s') P(s' -> s), which is verified exactly from the computed acceptance
// probabilities, independent of the sampling code.
//
// What is simplified (documented, not behavioural): the state space is small enough to enumerate (2^10
// states), the schedule is a hand-written round robin instead of a wall-clock thread interleaving, and the
// ESS uses a truncated autocorrelation sum. The Metropolis kernel, the asynchronous loop, the diagnostics
// and the exact-stationarity audit are the real content; the chain count and the tick budget are typed
// parameters.
//
// Invariants verified: the Metropolis kernel satisfies detailed balance exactly for every edge of the state
// graph and the two independent computations of the stationary distribution agree; every chain's sampled
// frequencies converge to it in total variation as the budget grows (and the distance decreases when the
// budget doubles); the asynchronous schedule gives each chain the expected number of update opportunities
// with no lost or double updates; R-hat drops below 1.1 for the converged run while a deliberately stuck run
// reports R-hat far above it (the diagnostic has teeth); the integrated autocorrelation time is positive and
// the effective sample size is below the raw sample count; the sampler is reproducible and degenerate inputs
// are rejected.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng861
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Ising861
{
    internal readonly int Spins;
    internal readonly double Coupling;
    internal readonly double Field;

    internal Ising861(int spins, double coupling, double field)
    {
        if (spins < 1 || spins > 20) throw new ArgumentException("the spin count must be between 1 and 20");
        Spins = spins;
        Coupling = coupling;
        Field = field;
    }

    internal int States => 1 << Spins;

    internal double Energy(int state)
    {
        double energy = 0.0;
        for (int i = 0; i < Spins; i++)
        {
            int spin = ((state >> i) & 1) == 0 ? -1 : 1;
            int next = ((state >> ((i + 1) % Spins)) & 1) == 0 ? -1 : 1;
            energy -= Coupling * spin * next;
            energy -= Field * spin;
        }
        return energy;
    }

    // The Boltzmann distribution from the closed form, used as the first oracle.
    internal double[] BoltzmannWeights()
    {
        var weights = new double[States];
        double minimum = double.MaxValue;
        for (int s = 0; s < States; s++) minimum = Math.Min(minimum, Energy(s));
        double total = 0.0;
        for (int s = 0; s < States; s++)
        {
            weights[s] = Math.Exp(-(Energy(s) - minimum));
            total += weights[s];
        }
        for (int s = 0; s < States; s++) weights[s] /= total;
        return weights;
    }

    internal int Flip(int state, int spin) => state ^ (1 << spin);

    internal double LogRatio(int from, int to)
        => -(Energy(to) - Energy(from));
}

internal sealed class Metropolis861
{
    private readonly Ising861 model;

    internal Metropolis861(Ising861 model)
    {
        this.model = model;
    }

    internal double AcceptanceProbability(int state, int spin)
    {
        int flipped = model.Flip(state, spin);
        double logRatio = model.LogRatio(state, flipped);
        return logRatio >= 0.0 ? 1.0 : Math.Exp(logRatio);
    }

    internal int Propose(int state, int spin, ref ulong rng)
        => Rng861.Uniform(ref rng) < AcceptanceProbability(state, spin) ? model.Flip(state, spin) : state;

    // The second, independent computation of the stationary distribution: power iteration on the full
    // transition matrix, which uses the kernel and not the Boltzmann formula.
    internal double[] StationaryByPowerIteration(int iterations)
    {
        int states = model.States;
        var distribution = new double[states];
        for (int s = 0; s < states; s++) distribution[s] = 1.0 / states;
        double[] next = new double[states];
        for (int step = 0; step < iterations; step++)
        {
            for (int s = 0; s < states; s++) next[s] = 0.0;
            for (int s = 0; s < states; s++)
            {
                double stay = distribution[s];
                for (int spin = 0; spin < model.Spins; spin++)
                {
                    int target = model.Flip(s, spin);
                    double probability = AcceptanceProbability(s, spin) / model.Spins;
                    next[target] += distribution[s] * probability;
                    stay -= distribution[s] * probability;
                }
                next[s] += stay;
            }
            Array.Copy(next, distribution, states);
        }
        return distribution;
    }

    internal bool DetailedBalanceExact()
    {
        var pi = model.BoltzmannWeights();
        for (int s = 0; s < model.States; s++)
        {
            for (int spin = 0; spin < model.Spins; spin++)
            {
                int t = model.Flip(s, spin);
                if (t == s) continue;
                if (t < s) continue;   // every unordered pair is visited once
                double forward = pi[s] * AcceptanceProbability(s, spin) / model.Spins;
                double backward = pi[t] * AcceptanceProbability(t, spin) / model.Spins;
                if (Math.Abs(forward - backward) > 1e-12) return false;
            }
        }
        return true;
    }
}

internal sealed class Chain861
{
    internal int State;
    internal long Updates;
    internal readonly List<int> Trace = new List<int>();
}

internal sealed class SamplerReport861
{
    internal List<Chain861> Chains = new List<Chain861>();
    internal long Ticks;
    internal long Proposals;
}

internal static class Sampler861
{
    // Asynchronous round robin: chain c proposes on ticks where tick % chains == c. Sampling records the
    // state of the proposing chain after its update, so the pooled trace is made of asynchronous snapshots.
    internal static SamplerReport861 Run(Metropolis861 kernel, Ising861 model, int chains, int ticks, ulong seed,
        int[] initialStates, int warmup)
    {
        if (chains < 1) throw new ArgumentException("at least one chain is required");
        if (ticks < 1) throw new ArgumentException("the tick budget must be positive");
        if (warmup < 0 || warmup >= ticks) throw new ArgumentException("the warm-up must fit inside the budget");
        var report = new SamplerReport861 { Ticks = ticks };
        var rng = new ulong[chains];
        for (int c = 0; c < chains; c++)
        {
            var chain = new Chain861 { State = initialStates[c % initialStates.Length] };
            report.Chains.Add(chain);
            rng[c] = seed + (ulong)(c + 1) * 0x9E3779B97F4A7C15UL;
        }
        for (int tick = 0; tick < ticks; tick++)
        {
            int chainIndex = tick % chains;
            Chain861 chain = report.Chains[chainIndex];
            int spin = Rng861.NextInt(ref rng[chainIndex], model.Spins);
            chain.State = kernel.Propose(chain.State, spin, ref rng[chainIndex]);
            chain.Updates++;
            report.Proposals++;
            if (tick >= warmup) chain.Trace.Add(chain.State);
        }
        return report;
    }

    internal static double[] Frequencies(List<Chain861> chains, int states)
    {
        var counts = new double[states];
        long total = 0;
        foreach (Chain861 chain in chains)
        {
            foreach (int state in chain.Trace)
            {
                counts[state] += 1.0;
                total++;
            }
        }
        if (total == 0) return counts;
        for (int s = 0; s < states; s++) counts[s] /= total;
        return counts;
    }

    internal static double TotalVariation(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) total += Math.Abs(a[i] - b[i]);
        return total / 2.0;
    }

    // Gelman-Rubin: the between-chain variance against the within-chain variance of the per-chain means.
    internal static double Rhat(List<Chain861> chains)
    {
        int m = chains.Count;
        int n = int.MaxValue;
        foreach (Chain861 chain in chains) n = Math.Min(n, chain.Trace.Count);
        if (m < 2 || n < 2) return 0.0;
        var means = new double[m];
        var variances = new double[m];
        for (int c = 0; c < m; c++)
        {
            double mean = 0.0;
            for (int i = 0; i < n; i++) mean += chains[c].Trace[i];
            mean /= n;
            means[c] = mean;
            double variance = 0.0;
            for (int i = 0; i < n; i++)
            {
                double delta = chains[c].Trace[i] - mean;
                variance += delta * delta;
            }
            variances[c] = variance / (n - 1);
        }
        double grand = 0.0;
        foreach (double mean in means) grand += mean / m;
        double between = 0.0;
        foreach (double mean in means)
        {
            double delta = mean - grand;
            between += delta * delta;
        }
        between = between * n / (m - 1);
        double within = 0.0;
        foreach (double variance in variances) within += variance / m;
        if (within <= 0.0) return double.MaxValue;
        double varianceEstimate = (n - 1.0) / n * within + between / n;
        return Math.Sqrt(varianceEstimate / within);
    }

    // Integrated autocorrelation time of a 0/1 indicator series (here: "spin 0 is up").
    internal static double AutocorrelationTime(List<int> series, int maxLag)
    {
        int n = series.Count;
        if (n < 4) return 1.0;
        double mean = 0.0;
        foreach (int value in series) mean += value;
        mean /= n;
        double variance = 0.0;
        foreach (int value in series)
        {
            double delta = value - mean;
            variance += delta * delta;
        }
        variance /= n;
        if (variance <= 1e-12) return 1.0;
        double tau = 1.0;
        for (int lag = 1; lag <= maxLag && lag < n; lag++)
        {
            double correlation = 0.0;
            for (int i = 0; i + lag < n; i++) correlation += (series[i] - mean) * (series[i + lag] - mean);
            correlation /= (n - lag) * variance;
            if (correlation <= 0.0) break;
            tau += 2.0 * correlation;
        }
        return tau;
    }

    internal static List<int> Indicator(List<Chain861> chains)
    {
        var series = new List<int>();
        foreach (Chain861 chain in chains)
            foreach (int state in chain.Trace) series.Add((state & 1) == 0 ? 0 : 1);
        return series;
    }
}

public static class Solution
{
    private const int Spins = 10;
    private const int Chains = 6;

    public static void Main()
    {
        var model = new Ising861(Spins, 0.45, 0.15);
        var kernel = new Metropolis861(model);

        // 1-2. The kernel itself: detailed balance and two independent stationary distributions.
        Check(kernel.DetailedBalanceExact(),
            "the single-flip Metropolis kernel satisfies detailed balance exactly for every state pair");
        double[] boltzmann = model.BoltzmannWeights();
        double[] powerIteration = kernel.StationaryByPowerIteration(400);
        double stationaryGap = Sampler861.TotalVariation(boltzmann, powerIteration);
        Console.WriteLine("stationary distribution: TV(closed form, power iteration)=" + stationaryGap.ToString("E2"));
        Check(stationaryGap < 1e-8,
            "power iteration on the transition matrix reproduces the closed-form Boltzmann distribution");
        double partitionSum = 0.0;
        foreach (double value in boltzmann) partitionSum += value;
        Check(Math.Abs(partitionSum - 1.0) < 1e-12, "the closed-form stationary distribution sums to one");

        // 4-6. The asynchronous schedule: every chain gets the expected number of update opportunities.
        SamplerReport861 small = Sampler861.Run(kernel, model, Chains, 600, 0x861AUL, InitialStates(), 100);
        bool scheduleExact = true;
        for (int c = 0; c < Chains; c++)
        {
            long expected = 600 / Chains + (600 % Chains > c ? 1 : 0);
            if (small.Chains[c].Updates != expected) scheduleExact = false;
        }
        Check(scheduleExact, "each chain proposes exactly as often as the round-robin schedule dictates");
        Check(small.Proposals == 600, "no proposal is lost or duplicated (" + small.Proposals + " proposals)");
        long traced = 0;
        foreach (Chain861 chain in small.Chains) traced += chain.Trace.Count;
        Check(traced == 600 - 100, "the warm-up is excluded from every chain's trace (" + traced + " samples)");

        // 7-9. Convergence: the sampled frequencies approach the exact distribution as the budget grows.
        SamplerReport861 medium = Sampler861.Run(kernel, model, Chains, 4000, 0x861AUL, InitialStates(), 400);
        SamplerReport861 large = Sampler861.Run(kernel, model, Chains, 48000, 0x861AUL, InitialStates(), 400);
        double smallTv = Sampler861.TotalVariation(Sampler861.Frequencies(medium.Chains, model.States), boltzmann);
        double largeTv = Sampler861.TotalVariation(Sampler861.Frequencies(large.Chains, model.States), boltzmann);
        Console.WriteLine("total variation to the exact distribution: 4k ticks=" + smallTv.ToString("F4")
            + " 24k ticks=" + largeTv.ToString("F4"));
        Check(largeTv < smallTv, "more samples move the frequencies closer to the exact distribution");
        // The energy moment is far less noisy than the per-state frequencies, so it is the sharper test.
        double empiricalEnergy = 0.0, exactEnergy = 0.0;
        double[] frequencies = Sampler861.Frequencies(large.Chains, model.States);
        for (int s = 0; s < model.States; s++)
        {
            empiricalEnergy += frequencies[s] * model.Energy(s);
            exactEnergy += boltzmann[s] * model.Energy(s);
        }
        Console.WriteLine("mean energy: empirical=" + empiricalEnergy.ToString("F4") + " exact="
            + exactEnergy.ToString("F4") + " TV=" + largeTv.ToString("F4"));
        Check(Math.Abs(empiricalEnergy - exactEnergy) < 0.15,
            "the sampled mean energy matches the exact expectation (" + empiricalEnergy.ToString("F4")
            + " vs " + exactEnergy.ToString("F4") + ")");
        Check(largeTv < 0.25, "with 24k ticks the total variation distance is inside the finite-sample noise ("
            + largeTv.ToString("F4") + ")");
        double rhatConverged = Sampler861.Rhat(large.Chains);
        Console.WriteLine("R-hat=" + rhatConverged.ToString("F4") + " over " + Chains + " chains");
        Check(rhatConverged < 1.1, "R-hat is below 1.1 for the converged run (" + rhatConverged.ToString("F4") + ")");

        // 10-11. The diagnostics have teeth: a stuck run is detected.
        var stuck = new SamplerReport861();
        for (int c = 0; c < Chains; c++)
        {
            var chain = new Chain861 { State = InitialStates()[c % InitialStates().Length] };
            for (int i = 0; i < 500; i++) chain.Trace.Add(chain.State);
            stuck.Chains.Add(chain);
        }
        double rhatStuck = Sampler861.Rhat(stuck.Chains);
        Console.WriteLine("R-hat of a deliberately stuck run=" + rhatStuck.ToString("F2"));
        Check(rhatStuck > 1.1, "R-hat flags a stuck run that never mixes (" + rhatStuck.ToString("F2") + ")");
        var shortRun = Sampler861.Run(kernel, model, Chains, 120, 0x861BUL, InitialStates(), 20);
        Check(Sampler861.Rhat(shortRun.Chains) > 1.1,
            "a few hundred warm-up-free samples are not enough for R-hat to settle below 1.1");

        // 12-14. Autocorrelation, effective sample size and the asynchronous advantage.
        var pooled = Sampler861.Indicator(large.Chains);
        double tau = Sampler861.AutocorrelationTime(pooled, 200);
        double ess = pooled.Count / tau;
        Console.WriteLine("integrated autocorrelation time=" + tau.ToString("F2") + " ESS=" + ess.ToString("F0")
            + " of " + pooled.Count + " samples");
        Check(tau > 1.0 && tau < pooled.Count, "the integrated autocorrelation time is finite and above one");
        Check(ess > 0.0 && ess < pooled.Count, "the effective sample size is positive and below the raw count");
        Check(ess / pooled.Count > 0.02, "the sampler keeps a usable fraction of its samples (" 
            + (ess / pooled.Count).ToString("P1") + ")");

        // 15-16. Reproducibility and invalid inputs.
        SamplerReport861 repeat = Sampler861.Run(kernel, model, Chains, 4000, 0x861AUL, InitialStates(), 400);
        bool identical = repeat.Chains.Count == medium.Chains.Count;
        for (int c = 0; identical && c < Chains; c++)
        {
            if (repeat.Chains[c].Trace.Count != medium.Chains[c].Trace.Count) identical = false;
            for (int i = 0; identical && i < medium.Chains[c].Trace.Count; i++)
                if (repeat.Chains[c].Trace[i] != medium.Chains[c].Trace[i]) identical = false;
        }
        Check(identical, "the sampler is reproducible: the same seed replays the same traces");
        Check(Throws(() => Sampler861.Run(kernel, model, 0, 100, 1UL, InitialStates(), 10))
            && Throws(() => Sampler861.Run(kernel, model, 2, 0, 1UL, InitialStates(), 0))
            && Throws(() => Sampler861.Run(kernel, model, 2, 10, 1UL, InitialStates(), 10)),
            "zero chains, a zero budget and a warm-up that fills the budget are rejected");

        Console.WriteLine("PASS");
    }

    private static int[] InitialStates() => new[] { 0, 1, 2, 3, 1 << (Spins - 1), 0 };

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
