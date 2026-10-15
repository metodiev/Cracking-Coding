// Task 494 - Implement Thompson sampling simulator for bandit problems. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Beta-Bernoulli Thompson sampling on a planted three-armed bandit: arm i pays 1 with the known
// probability arms[i], and every trial samples each arm's success rate from its Beta posterior
// (prior Beta(1,1)), plays the sampled argmax (ties to the lowest index), observes a Bernoulli
// reward drawn from the fixed LCG and updates the chosen arm's posterior. Beta(a,b) with integer
// parameters is exact: two Gamma(a,1)/Gamma(b,1) draws built as sums of a (resp. b) unit
// exponentials -ln(u), u uniform; that only needs the fixed LCG, no random source. Every draw
// steps seed = (seed*75 + 74) % 65537 and returns (seed + 0.5)/65537 in (0,1). Documented
// expectations for arms [0.2, 0.5, 0.8] over 1200 fixed trials: the best arm is picked at least
// 85% of the time, it is picked more often than each other arm, and the average regret
// R = sum(max(mu) - mu[a_t]) stays under 100; the posterior mean of the best arm sits within
// 0.05 of 0.8. Re-running with the same seed reproduces the pull vector exactly.
using System;

internal class ThompsonSampler
{
    private readonly double[] arms;
    private int seed;
    private readonly int[] alpha;
    private readonly int[] beta;
    internal readonly int[] Pulls;

    internal ThompsonSampler(double[] arms, int seed)
    {
        this.arms = (double[])arms.Clone();
        this.seed = seed;
        alpha = new int[arms.Length];
        beta = new int[arms.Length];
        Pulls = new int[arms.Length];
        for (int i = 0; i < arms.Length; i++)
        {
            alpha[i] = 1;                       // integer parameters: exact Beta draws
            beta[i] = 1;
        }
    }

    private double Uniform()
    {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 0.5) / 65537.0;
    }

    /** Gamma(shape, 1) for integer shape >= 1: a sum of shape unit exponentials. */
    private double GammaUnit(int shape)
    {
        double total = 0.0;
        for (int i = 0; i < shape; i++) total += -Math.Log(Uniform());
        return total;
    }

    private double BetaSample(int a, int b)
    {
        double x = GammaUnit(a);
        double y = GammaUnit(b);
        return x / (x + y);
    }

    internal void Run(int trials)
    {
        double best = arms[0];
        foreach (double value in arms) best = Math.Max(best, value);
        for (int t = 0; t < trials; t++)
        {
            var samples = new double[arms.Length];
            for (int i = 0; i < arms.Length; i++) samples[i] = BetaSample(alpha[i], beta[i]);
            int choice = 0;
            for (int i = 1; i < samples.Length; i++)
            {
                if (samples[i] > samples[choice]) choice = i;
            }
            int reward = Uniform() < arms[choice] ? 1 : 0;
            alpha[choice] += reward;
            beta[choice] += 1 - reward;
            Pulls[choice] += 1;
        }
    }

    internal double PosteriorMean(int arm) => (double)alpha[arm] / (alpha[arm] + beta[arm]);
}

public static class Solution
{
    private static readonly double[] Arms = { 0.2, 0.5, 0.8 };
    private const int Trials = 1200;
    private const int Seed = 42;

    public static void Main()
    {
        var sampler = new ThompsonSampler(Arms, Seed);
        sampler.Run(Trials);

        int sum = 0;
        foreach (int pull in sampler.Pulls) sum += pull;
        Check(sum == Trials, "pull total");
        double bestFrequency = (double)sampler.Pulls[2] / Trials;
        Check(bestFrequency >= 0.85, "documented: best arm picked >= 85% of the time");
        Check(sampler.Pulls[2] > sampler.Pulls[1] && sampler.Pulls[1] > sampler.Pulls[0],
                "arm ordering");
        double posterior = sampler.PosteriorMean(2);
        Check(posterior >= 0.75 && posterior <= 0.85, "posterior mean of the best arm");
        Check(Math.Abs(posterior - Arms[2]) <= 0.05, "posterior tracks the true rate");

        var repeat = new ThompsonSampler(Arms, Seed);
        repeat.Run(Trials);
        for (int i = 0; i < Arms.Length; i++)
        {
            Check(repeat.Pulls[i] == sampler.Pulls[i], "fixed seed reproduces the pulls");
        }
        Check(repeat.PosteriorMean(0) == sampler.PosteriorMean(0), "deterministic posterior");

        var trivial = new ThompsonSampler(new[] { 0.9 }, Seed + 1);
        trivial.Run(200);
        Check(trivial.Pulls[0] == 200, "single arm always pulled");

        var duet = new ThompsonSampler(new[] { 0.3, 0.7 }, Seed + 2);
        duet.Run(600);
        Check(duet.Pulls[1] >= 0.8 * 600, "larger gap concentrates faster");

        var tied = new ThompsonSampler(new[] { 0.5, 0.5 }, Seed + 3);
        tied.Run(600);
        Check(tied.Pulls[0] + tied.Pulls[1] == 600, "tied total");
        Check(Math.Min(tied.Pulls[0], tied.Pulls[1]) >= 100, "equal arms both explored");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
