// Task 495 - Implement UCB (upper confidence bound) algorithm simulation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// UCB1 on a planted three-armed bandit: each arm is pulled once to initialise its averages
// (arms in index order), then every trial pulls argmax(mu_hat[i] + sqrt(2 ln t / n_i)), ties to
// the lowest index, t counting trials so far. Rewards are Bernoulli draws from the fixed LCG
// (seed = (seed*75 + 74) % 65537, reward when (seed + 0.5)/65537 < p_arm), so the whole run is
// deterministic. The pseudo-regret uses the known true means, R(T) = sum over trials of
// (max(mu) - mu[a_t]) with mu = [0.2, 0.5, 0.8] as planted. Documented expectations over 2000
// fixed trials: the best arm is pulled at least 80% of the time, pulls are strictly ordered
// best > middle > worst, total regret stays under 100, and the mean per-trial regret of the
// second half is strictly below that of the first half (the regret rate decreases as UCB1
// learns). A uniform-random baseline using the same reward stream is used to show UCB1's
// advantage.
using System;

internal class UCB1
{
    private readonly double[] arms;
    private int seed;
    private readonly int[] counts;
    private readonly double[] sums;
    internal readonly int[] Pulls;

    internal UCB1(double[] arms, int seed)
    {
        this.arms = (double[])arms.Clone();
        this.seed = seed;
        counts = new int[arms.Length];
        sums = new double[arms.Length];
        Pulls = new int[arms.Length];
    }

    private int Reward(double p)
    {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 0.5) / 65537.0 < p ? 1 : 0;
    }

    private double Index(int arm, int t)
    {
        if (counts[arm] == 0) return double.PositiveInfinity;   // untried arms come first
        double mean = sums[arm] / counts[arm];
        return mean + Math.Sqrt(2.0 * Math.Log(t) / counts[arm]);
    }

    /** Runs `trials` pulls and returns the cumulative pseudo-regret trace. */
    internal double[] Run(int trials)
    {
        double best = arms[0];
        foreach (double value in arms) best = Math.Max(best, value);
        var trace = new double[trials];
        double regret = 0.0;
        for (int t = 1; t <= trials; t++)
        {
            int choice = 0;
            for (int i = 1; i < arms.Length; i++)
            {
                if (Index(i, t) > Index(choice, t)) choice = i;
            }
            int r = Reward(arms[choice]);
            counts[choice] += 1;
            sums[choice] += r;
            Pulls[choice] += 1;
            regret += best - arms[choice];
            trace[t - 1] = regret;
        }
        return trace;
    }
}

/** Uniform-random arm choice over the same LCG reward stream (control experiment). */
internal class RandomBaseline
{
    private readonly double[] arms;
    private int seed;

    internal RandomBaseline(double[] arms, int seed)
    {
        this.arms = (double[])arms.Clone();
        this.seed = seed;
    }

    internal double Run(int trials)
    {
        double best = arms[0];
        foreach (double value in arms) best = Math.Max(best, value);
        double regret = 0.0;
        for (int t = 0; t < trials; t++)
        {
            seed = (seed * 75 + 74) % 65537;
            int choice = seed % arms.Length;
            seed = (seed * 75 + 74) % 65537;
            regret += best - arms[choice];
        }
        return regret;
    }
}

public static class Solution
{
    private static readonly double[] Arms = { 0.2, 0.5, 0.8 };
    private const int Trials = 2000;
    private const int Seed = 99;

    public static void Main()
    {
        var ucb = new UCB1(Arms, Seed);
        double[] trace = ucb.Run(Trials);

        int sum = 0;
        foreach (int pull in ucb.Pulls) sum += pull;
        Check(sum == Trials, "pull total");

        double bestFrequency = (double)ucb.Pulls[2] / Trials;
        Check(bestFrequency >= 0.80, "documented: best arm picked >= 80% of the time");
        Check(ucb.Pulls[2] > ucb.Pulls[1] && ucb.Pulls[1] > ucb.Pulls[0], "strict ordering");
        double regret = trace[Trials - 1];
        Check(regret <= 100.0, "documented: total pseudo-regret bound");

        for (int i = 1; i < trace.Length; i++)
        {
            Check(trace[i] >= trace[i - 1], "regret is a cumulative sum of gaps");
        }
        int half = Trials / 2;
        double firstRate = trace[half - 1] / half;
        double secondRate = (trace[Trials - 1] - trace[half - 1]) / (Trials - half);
        Check(secondRate < firstRate, "documented: regret rate decreases");

        var baseline = new RandomBaseline(Arms, Seed);
        double baselineRegret = baseline.Run(Trials);
        Check(baselineRegret > 2.0 * regret, "UCB1 must beat uniform sampling");

        var repeat = new UCB1(Arms, Seed);
        double[] repeatTrace = repeat.Run(Trials);
        for (int i = 0; i < Arms.Length; i++)
        {
            Check(repeat.Pulls[i] == ucb.Pulls[i], "deterministic pulls");
        }
        Check(repeatTrace[Trials - 1] == regret, "deterministic regret");

        var warm = new UCB1(Arms, Seed);
        warm.Run(3);
        Check(warm.Pulls[0] == 1 && warm.Pulls[1] == 1 && warm.Pulls[2] == 1,
                "warm-up visits each arm exactly once");
        warm.Run(7);
        Check(Math.Min(warm.Pulls[0], Math.Min(warm.Pulls[1], warm.Pulls[2])) >= 1,
                "no arm is skipped during warm-up");
        Check(warm.Pulls[0] + warm.Pulls[1] + warm.Pulls[2] == 10, "warm total");

        var duet = new UCB1(new[] { 0.3, 0.7 }, Seed + 1);
        duet.Run(1000);
        Check(duet.Pulls[1] >= 0.9 * 1000, "large-gap concentration");
        Check(duet.Pulls[1] > duet.Pulls[0], "large-gap ordering");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
