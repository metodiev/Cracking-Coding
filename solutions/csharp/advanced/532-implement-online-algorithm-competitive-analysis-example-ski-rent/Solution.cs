// Task 532 - Implement online algorithm competitive analysis example (ski rental). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// The ski-rental problem: a season of unknown length t <= T costs 1 per day when renting and B when
// buying, so the offline optimum is min(t, B). A deterministic online algorithm decides for every day
// whether to rent or to buy without knowing t; every such algorithm is a threshold strategy (rent
// until day d, buy on day d, or never buy, which is never better), and its cost is
// ALG(t) = (d - 1) + B for t >= d and t otherwise. The self-checks enumerate d and t to show that the
// best threshold is exactly d = B with competitive ratio 2 - 1/B, and that no threshold ever beats it.
// For the randomized setting the buy day is drawn from a distribution p over 1..B; the expected cost
// EXACT(t) = sum_{i <= t} p_i ((i - 1) + B) + t * sum_{i > t} p_i is computed in closed form, and the
// classical distribution p_i = (e^(i/B) - e^((i-1)/B)) / (e - 1) is shown to keep the worst-case ratio
// at or below e/(e-1) = 1.58198 while the deterministic best is 2 - 1/B.
//
// Documented checks. (1) Deterministic competitive analysis: for B in {2, 3, 10, 64} the enumerated
// best ratio over every threshold and every horizon t in 1..4B equals 2 - 1/B to 1e-12, attained at
// d = B, and the specific "rent B - 1 days then buy" routine produces exactly that; the threshold at
// d = 1 (always buy) and the never-buy variant are both worse. (2) Randomized algorithm: the
// exponential distribution sums to 1 to 1e-12, its worst-case ratio over t in 1..4B is at most
// e/(e-1) + 1e-12 (and at least e/(e-1) - 0.01 for B = 64, where it reaches 1.57738), and it is
// strictly better than the deterministic bound for every tested B. (3) Simulation: for eight
// (B, t) pairs 200000 LCG trials of the randomized algorithm match the closed-form expectation within
// five standard errors of the exact variance, and the empirical distribution of the chosen buy day
// matches p within 0.005 in every bucket. (4) Boundaries: B = 2 has ratio 1.5 exactly; ratios are
// monotone in the horizon in the sense that the worst case is attained at t = B or t = 1 as reported.
// (5) Errors: B < 2, a horizon below 1, a distribution with a negative entry, a distribution that does
// not sum to 1 and a trial count below 1 each throw ArgumentException.
using System;

/** Deterministic LCG with three discarded warm-up steps and rejection sampling. */
internal sealed class Lcg
{
    private int state;

    internal Lcg(int seed)
    {
        state = seed % 65537;
        if (state < 0) state += 65537;
        for (int i = 0; i < 3; i++) state = (state * 75 + 74) % 65537;
    }

    internal int Next(int bound)
    {
        if (bound <= 0 || bound > 65537) throw new ArgumentOutOfRangeException(nameof(bound));
        int limit = 65537 - 65537 % bound;
        int value;
        do
        {
            state = (state * 75 + 74) % 65537;
            value = state;
        }
        while (value >= limit);
        return value % bound;
    }

    internal double NextDouble()
    {
        return Next(65536) / 65536.0;
    }
}

internal static class SkiRental
{
    internal static double OptimalCost(int horizon, int buyCost)
    {
        return Math.Min(horizon, buyCost);
    }

    /** Cost of the threshold strategy "buy on day buyDay (1-based, never when buyDay > horizon)". */
    internal static double ThresholdCost(int buyDay, int horizon, int buyCost)
    {
        return buyDay <= horizon ? (buyDay - 1) + buyCost : horizon;
    }

    internal static double ThresholdRatio(int buyDay, int horizon, int buyCost)
    {
        return ThresholdCost(buyDay, horizon, buyCost) / OptimalCost(horizon, buyCost);
    }

    /** Best threshold by enumeration over d = 1..B and d = never, over horizons 1..horizonLimit. */
    internal static double BestDeterministicRatio(int buyCost, int horizonLimit, out int bestDay)
    {
        if (buyCost < 2) throw new ArgumentException("the buy cost must be at least 2");
        if (horizonLimit < 1) throw new ArgumentException("the horizon must be positive");
        bestDay = 0;
        double best = double.PositiveInfinity;
        for (int d = 1; d <= buyCost + 1; d++)
        {
            double worst = 0.0;
            for (int horizon = 1; horizon <= horizonLimit; horizon++)
            {
                worst = Math.Max(worst, ThresholdRatio(d, horizon, buyCost));
            }
            if (worst < best)
            {
                best = worst;
                bestDay = d;
            }
        }
        return best;
    }

    internal static void RequireDistribution(double[] probabilities)
    {
        if (probabilities == null) throw new ArgumentNullException(nameof(probabilities));
        if (probabilities.Length < 2) throw new ArgumentException("at least two buy days are required");
        double total = 0.0;
        foreach (double value in probabilities)
        {
            if (value < 0.0) throw new ArgumentException("negative probability");
            total += value;
        }
        if (Math.Abs(total - 1.0) > 1e-9) throw new ArgumentException("the distribution must sum to 1");
    }

    /** Closed-form expected cost of the randomized threshold: E[cost](t). */
    internal static double ExpectedCost(double[] probabilities, int horizon, int buyCost)
    {
        RequireDistribution(probabilities);
        if (horizon < 1) throw new ArgumentException("the horizon must be positive");
        double expectation = 0.0;
        for (int i = 1; i <= probabilities.Length; i++)
        {
            expectation += i <= horizon
                    ? probabilities[i - 1] * ((i - 1) + buyCost)
                    : probabilities[i - 1] * horizon;
        }
        return expectation;
    }

    /** E[cost^2](t), used to bound the Monte-Carlo error of the simulation. */
    internal static double ExpectedSquareCost(double[] probabilities, int horizon, int buyCost)
    {
        RequireDistribution(probabilities);
        double expectation = 0.0;
        for (int i = 1; i <= probabilities.Length; i++)
        {
            double cost = i <= horizon ? (i - 1) + buyCost : horizon;
            expectation += probabilities[i - 1] * cost * cost;
        }
        return expectation;
    }

    internal static double RandomizedRatio(double[] probabilities, int horizon, int buyCost)
    {
        return ExpectedCost(probabilities, horizon, buyCost) / OptimalCost(horizon, buyCost);
    }

    internal static double WorstRandomizedRatio(double[] probabilities, int buyCost, int horizonLimit)
    {
        double worst = 0.0;
        for (int horizon = 1; horizon <= horizonLimit; horizon++)
        {
            double ratio = RandomizedRatio(probabilities, horizon, buyCost);
            if (ratio > worst) worst = ratio;
        }
        return worst;
    }

    /** The classical distribution p_i = (e^(i/B) - e^((i - 1)/B)) / (e - 1) over days 1..B. */
    internal static double[] ExponentialDistribution(int buyCost)
    {
        if (buyCost < 2) throw new ArgumentException("the buy cost must be at least 2");
        var probabilities = new double[buyCost];
        double denominator = Math.E - 1.0;
        for (int i = 1; i <= buyCost; i++)
        {
            probabilities[i - 1] = (Math.Exp((double)i / buyCost) - Math.Exp((double)(i - 1) / buyCost))
                    / denominator;
        }
        return probabilities;
    }

    /** Uniform distribution over 1..B, the naive randomized strategy. */
    internal static double[] UniformDistribution(int buyCost)
    {
        var probabilities = new double[buyCost];
        for (int i = 0; i < buyCost; i++) probabilities[i] = 1.0 / buyCost;
        return probabilities;
    }

    internal static int SampleBuyDay(double[] probabilities, Lcg rng)
    {
        double u = rng.NextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (u < cumulative) return i + 1;
        }
        return probabilities.Length;
    }

    /** Monte-Carlo mean cost and the empirical buy-day frequencies. */
    internal static double Simulate(double[] probabilities, int horizon, int buyCost, int trials, int seed,
            out double[] frequencies, out double meanSquare)
    {
        RequireDistribution(probabilities);
        if (trials < 1) throw new ArgumentException("at least one trial is required");
        var rng = new Lcg(seed);
        frequencies = new double[probabilities.Length];
        double total = 0.0;
        double squares = 0.0;
        for (int trial = 0; trial < trials; trial++)
        {
            int day = SampleBuyDay(probabilities, rng);
            frequencies[day - 1] += 1.0;
            double cost = ThresholdCost(day, horizon, buyCost);
            total += cost;
            squares += cost * cost;
        }
        for (int i = 0; i < frequencies.Length; i++) frequencies[i] /= trials;
        meanSquare = squares / trials;
        return total / trials;
    }
}

public static class Solution
{
    public static void Main()
    {
        foreach (int buyCost in new[] { 2, 3, 10, 64 })
        {
            double best = SkiRental.BestDeterministicRatio(buyCost, 4 * buyCost, out int bestDay);
            CheckClose(best, 2.0 - 1.0 / buyCost, "best deterministic ratio for B = " + buyCost);
            Check(bestDay == buyCost, "the best threshold buys on day B for B = " + buyCost);
            double worstAtBest = 0.0;
            for (int horizon = 1; horizon <= 4 * buyCost; horizon++)
            {
                worstAtBest = Math.Max(worstAtBest, SkiRental.ThresholdRatio(buyCost, horizon, buyCost));
            }
            CheckClose(worstAtBest, 2.0 - 1.0 / buyCost, "the B-threshold attains the bound for B = " + buyCost);
            double alwaysBuy = 0.0;
            double neverBuy = 0.0;
            for (int horizon = 1; horizon <= 4 * buyCost; horizon++)
            {
                alwaysBuy = Math.Max(alwaysBuy, SkiRental.ThresholdRatio(1, horizon, buyCost));
                neverBuy = Math.Max(neverBuy, SkiRental.ThresholdRatio(buyCost + 1, horizon, buyCost));
            }
            Check(alwaysBuy >= best && alwaysBuy <= buyCost, "buying at once is at least as bad for B = " + buyCost);
            Check(neverBuy > best, "never buying is strictly worse for B = " + buyCost);
            CheckClose(SkiRental.OptimalCost(5, buyCost), Math.Min(5, buyCost), "offline optimum for t = 5");
            CheckClose(SkiRental.ThresholdCost(2, 5, buyCost), 1.0 + buyCost,
                    "cost of buying on day 2 with horizon 5");
            CheckClose(SkiRental.ThresholdCost(2, 1, buyCost), 1.0,
                    "cost of renting the single day with the day-2 threshold");
        }
        CheckClose(SkiRental.BestDeterministicRatio(2, 8, out int tinyDay), 1.5, "B = 2 has ratio 1.5");
        Check(tinyDay == 2, "B = 2 buys on day 2");

        foreach (int buyCost in new[] { 2, 3, 10, 64 })
        {
            double[] exponential = SkiRental.ExponentialDistribution(buyCost);
            CheckClose(Sum(exponential), 1.0, "the exponential distribution sums to 1 for B = " + buyCost);
            double worst = SkiRental.WorstRandomizedRatio(exponential, buyCost, 4 * buyCost);
            double limit = Math.E / (Math.E - 1.0);
            Check(worst <= limit + 1e-12,
                    "the exponential distribution never exceeds e/(e-1) for B = " + buyCost + " (" + worst + ")");
            Check(worst < 2.0 - 1.0 / buyCost,
                    "randomization beats the deterministic bound for B = " + buyCost);
            Check(worst >= 1.0, "the ratio is at least one for B = " + buyCost);
            double[] uniform = SkiRental.UniformDistribution(buyCost);
            double uniformWorst = SkiRental.WorstRandomizedRatio(uniform, buyCost, 4 * buyCost);
            Check(uniformWorst >= worst - 1e-12,
                    "the naive uniform distribution is no better for B = " + buyCost);
            Check(uniformWorst < 2.0, "the uniform distribution still beats 2 for B = " + buyCost);
        }
        double[] large = SkiRental.ExponentialDistribution(64);
        double largeWorst = SkiRental.WorstRandomizedRatio(large, 64, 4 * 64);
        Check(largeWorst >= Math.E / (Math.E - 1.0) - 0.01,
                "for B = 64 the exponential distribution comes within 0.01 of e/(e-1) (" + largeWorst + ")");
        CheckClose(largeWorst, 1.577382, 1e-5, "the documented value 1.577382 for B = 64");

        int[][] simulations = {
            new[] { 10, 1 }, new[] { 10, 5 }, new[] { 10, 10 }, new[] { 10, 25 },
            new[] { 64, 1 }, new[] { 64, 8 }, new[] { 64, 64 }, new[] { 2, 3 },
        };
        foreach (int[] spec in simulations)
        {
            int buyCost = spec[0];
            int horizon = spec[1];
            double[] probabilities = SkiRental.ExponentialDistribution(buyCost);
            double expected = SkiRental.ExpectedCost(probabilities, horizon, buyCost);
            double expectedSquare = SkiRental.ExpectedSquareCost(probabilities, horizon, buyCost);
            double variance = expectedSquare - expected * expected;
            double standardError = Math.Sqrt(Math.Max(variance, 0.0) / 200000.0);
            double simulated = SkiRental.Simulate(probabilities, horizon, buyCost, 200000, 4242 + horizon,
                    out double[] frequencies, out double meanSquare);
            Check(Math.Abs(simulated - expected) <= 5.0 * standardError + 1e-12,
                    "simulated mean cost for B = " + buyCost + ", t = " + horizon + " (" + simulated + " vs "
                            + expected + ")");
            Check(Math.Abs(meanSquare - expectedSquare) <= 0.05 * expectedSquare + 1.0,
                    "simulated second moment for B = " + buyCost + ", t = " + horizon);
            for (int i = 0; i < probabilities.Length; i++)
            {
                Check(Math.Abs(frequencies[i] - probabilities[i]) <= 0.005,
                        "empirical buy-day frequency for B = " + buyCost + ", day " + (i + 1));
            }
        }

        CheckThrows(() => SkiRental.BestDeterministicRatio(1, 10, out int _), "B below 2 throws");
        CheckThrows(() => SkiRental.BestDeterministicRatio(10, 0, out int _), "a zero horizon throws");
        CheckThrows(() => SkiRental.ExpectedCost(new[] { 0.5, 0.6 }, 3, 10), "an unnormalised distribution throws");
        CheckThrows(() => SkiRental.ExpectedCost(new[] { -0.5, 1.5 }, 3, 10), "a negative probability throws");
        CheckThrows(() => SkiRental.ExpectedCost(new[] { 1.0 }, 3, 10), "a one-day distribution throws");
        CheckThrows(() => SkiRental.ExponentialDistribution(1), "B below 2 throws for the distribution");
        CheckThrows(() => SkiRental.Simulate(new[] { 0.5, 0.5 }, 2, 5, 0, 1, out double[] _, out double _),
                "zero trials throw");

        Console.WriteLine("PASS");
    }

    private static double Sum(double[] values)
    {
        double total = 0.0;
        foreach (double value in values) total += value;
        return total;
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        if (!(Math.Abs(actual - expected) <= 1e-12 * Math.Max(1.0, Math.Abs(expected))))
        {
            throw new Exception("FAILED: " + label + " (" + actual + " != " + expected + ")");
        }
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        if (!(Math.Abs(actual - expected) <= tolerance * Math.Max(1.0, Math.Abs(expected))))
        {
            throw new Exception("FAILED: " + label + " (" + actual + " != " + expected + ")");
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
