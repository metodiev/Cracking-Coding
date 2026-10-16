// Task 523 - Implement simple Markov chain Monte Carlo (Metropolis-Hastings). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Two Metropolis-Hastings samplers sharing one acceptance rule, min(1, pi(x')/pi(x)):
//   * A one-dimensional random-walk sampler for a density given only up to a constant. It works in
//     log density (the standard normal target is -x^2/2 - log sqrt(2 pi), and a bounded uniform
//     target is the indicator of [-1, 1]), proposes x' = x + u*w with u uniform on (-1, 1) from the
//     deterministic LCG (three discarded warm-up steps, and the uniform is taken on the open interval
//     so that log(u) is always finite), and accepts when log u < log pi(x') - log pi(x). The class
//     reports the acceptance rate and the empirical mean, standard deviation and CDF of the chain.
//   * A finite-state sampler on pi = (0.1, 0.2, 0.3, 0.4) whose proposal picks one of the k-1 other
//     states uniformly (a symmetric proposal), so its transition matrix is an exact k x k matrix that
//     can be inspected rather than simulated.
//
// Documented checks. (1) Exact theory, no simulation: the constructed transition matrix satisfies
// pi_i P_ij = pi_j P_ji for every pair to 1e-12, pi P = pi to 1e-12, every row sums to 1, and the
// stationary distribution found by 200 power iterations from a uniform start reproduces pi to 1e-9.
// (2) Simulation: 200k steps of the discrete chain reproduce pi within 0.01 per state (the last
// recorded state is the chain state, so consecutive visits are dependent as they must be).
// (3) The standard-normal random walk with step width 1.0 and 200k steps has |mean| <= 0.03,
// |sd - 1| <= 0.03, an acceptance rate in (0.5, 0.9), and its empirical CDF at -1 / 0 / 1 is within
// 0.02 of 0.158655 / 0.5 / 0.841345; step widths 0.25 and 4.0 both give |mean| <= 0.05 while the
// narrow proposal is accepted strictly more often than the wide one. (4) A chain started at x = 20
// mixes into the target (|mean of the second half| <= 0.05, acceptance > 0.3). (5) The bounded
// uniform target on [-1, 1] never leaves [-1, 1] over 100k steps, with |mean| <= 0.02,
// |E|x| - 0.5| <= 0.02 and |sd - 0.57735| <= 0.03. (6) Errors: a non-positive step count or step
// width, a target with fewer than two states, a target with a non-positive entry and an unnormalised
// target each throw ArgumentException.
using System;

/** Deterministic LCG; Next(bound) rejects to give an exactly uniform value in [0, bound). */
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

    /** Uniform on (0, 1) so that the logarithm is always finite. */
    internal double NextOpenDouble()
    {
        return (Next(65536) + 1) / 65537.0;
    }
}

internal sealed class ChainSummary
{
    internal readonly double[] Values;
    internal readonly int Accepted;

    internal ChainSummary(double[] values, int accepted)
    {
        Values = values;
        Accepted = accepted;
    }

    internal double AcceptanceRate => (double)Accepted / Values.Length;

    internal double Mean(int from)
    {
        double sum = 0.0;
        for (int i = from; i < Values.Length; i++) sum += Values[i];
        return sum / (Values.Length - from);
    }

    internal double MeanAbsolute(int from)
    {
        double sum = 0.0;
        for (int i = from; i < Values.Length; i++) sum += Math.Abs(Values[i]);
        return sum / (Values.Length - from);
    }

    internal double StandardDeviation(int from)
    {
        double mean = Mean(from);
        double sum = 0.0;
        for (int i = from; i < Values.Length; i++) sum += (Values[i] - mean) * (Values[i] - mean);
        return Math.Sqrt(sum / (Values.Length - from));
    }

    internal double Cdf(double x, int from)
    {
        int count = 0;
        for (int i = from; i < Values.Length; i++)
        {
            if (Values[i] <= x) count++;
        }
        return (double)count / (Values.Length - from);
    }
}

internal static class Metropolis
{
    internal const double LogSqrtTwoPi = 0.91893853320467274178032973640562;
    internal const double NormalCdfAtMinusOne = 0.15865525393145705;

    /** log pi(x) for the standard normal target; pi is only needed up to a constant. */
    internal static double LogStandardNormal(double x)
    {
        return -0.5 * x * x - LogSqrtTwoPi;
    }

    /** log pi(x) for the uniform target on [-1, 1]. */
    internal static double LogBoundedUniform(double x)
    {
        return x >= -1.0 && x <= 1.0 ? 0.0 : double.NegativeInfinity;
    }

    internal static ChainSummary Run(Func<double, double> logDensity, int steps, double start,
            double stepWidth, int seed)
    {
        if (steps < 1) throw new ArgumentException("at least one step is required");
        if (!(stepWidth > 0.0)) throw new ArgumentException("the step width must be positive");
        var rng = new Lcg(seed);
        var values = new double[steps];
        double current = start;
        double currentLog = logDensity(current);
        int accepted = 0;
        for (int i = 0; i < steps; i++)
        {
            double proposal = current + (rng.NextDouble() * 2.0 - 1.0) * stepWidth;
            double proposalLog = logDensity(proposal);
            if (Math.Log(rng.NextOpenDouble()) < proposalLog - currentLog)
            {
                current = proposal;
                currentLog = proposalLog;
                accepted++;
            }
            values[i] = current;
        }
        return new ChainSummary(values, accepted);
    }

    /** Symmetric random-scan proposal over a finite state space: the exact transition matrix. */
    internal static double[][] TransitionMatrix(double[] pi)
    {
        RequireDistribution(pi);
        int k = pi.Length;
        var matrix = new double[k][];
        for (int i = 0; i < k; i++) matrix[i] = new double[k];
        double proposal = 1.0 / (k - 1);
        for (int i = 0; i < k; i++)
        {
            double stay = 0.0;
            for (int j = 0; j < k; j++)
            {
                if (i == j) continue;
                matrix[i][j] = proposal * Math.Min(1.0, pi[j] / pi[i]);
                stay += matrix[i][j];
            }
            matrix[i][i] = 1.0 - stay;
        }
        return matrix;
    }

    internal static double[] Simulate(double[] pi, int steps, int seed)
    {
        RequireDistribution(pi);
        if (steps < 1) throw new ArgumentException("at least one step is required");
        var rng = new Lcg(seed);
        int k = pi.Length;
        var visits = new double[k];
        int state = 0;
        for (int i = 0; i < steps; i++)
        {
            int candidate = rng.Next(k - 1);
            if (candidate >= state) candidate++;
            if (rng.NextOpenDouble() < Math.Min(1.0, pi[candidate] / pi[state])) state = candidate;
            visits[state]++;
        }
        for (int j = 0; j < k; j++) visits[j] /= steps;
        return visits;
    }

    /** Stationary distribution by power iteration on the transposed transition matrix. */
    internal static double[] StationaryByPowerIteration(double[][] matrix, int iterations)
    {
        int k = matrix.Length;
        var vector = new double[k];
        for (int i = 0; i < k; i++) vector[i] = 1.0 / k;
        for (int step = 0; step < iterations; step++)
        {
            var next = new double[k];
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++) next[j] += vector[i] * matrix[i][j];
            }
            vector = next;
        }
        return vector;
    }

    internal static void RequireDistribution(double[] pi)
    {
        if (pi.Length < 2) throw new ArgumentException("at least two states are required");
        double total = 0.0;
        foreach (double value in pi)
        {
            if (!(value > 0.0)) throw new ArgumentException("every probability must be positive");
            total += value;
        }
        if (Math.Abs(total - 1.0) > 1e-9) throw new ArgumentException("the distribution must sum to 1");
    }
}

public static class Solution
{
    public static void Main()
    {
        double[] pi = { 0.1, 0.2, 0.3, 0.4 };
        double[][] matrix = Metropolis.TransitionMatrix(pi);
        for (int i = 0; i < pi.Length; i++)
        {
            double rowSum = 0.0;
            for (int j = 0; j < pi.Length; j++) rowSum += matrix[i][j];
            CheckClose(rowSum, 1.0, "row " + i + " sums to 1");
        }
        for (int i = 0; i < pi.Length; i++)
        {
            for (int j = 0; j < pi.Length; j++)
            {
                CheckClose(pi[i] * matrix[i][j], pi[j] * matrix[j][i], "detailed balance at (" + i + ", " + j + ")");
            }
        }
        var product = new double[pi.Length];
        for (int j = 0; j < pi.Length; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < pi.Length; i++) sum += pi[i] * matrix[i][j];
            product[j] = sum;
            CheckClose(product[j], pi[j], "pi P = pi at column " + j);
        }
        double[] stationary = Metropolis.StationaryByPowerIteration(matrix, 200);
        for (int j = 0; j < pi.Length; j++)
        {
            Check(Math.Abs(stationary[j] - pi[j]) <= 1e-9,
                    "power iteration reaches pi at state " + j + " (" + stationary[j] + ")");
        }

        double[] frequencies = Metropolis.Simulate(pi, 200000, 4242);
        for (int j = 0; j < pi.Length; j++)
        {
            Check(Math.Abs(frequencies[j] - pi[j]) <= 0.01,
                    "the simulated chain visits state " + j + " with frequency " + frequencies[j]);
        }

        ChainSummary standard = Metropolis.Run(Metropolis.LogStandardNormal, 200000, 0.0, 1.0, 202513);
        Check(Math.Abs(standard.Mean(0)) <= 0.03, "standard normal mean (" + standard.Mean(0) + ")");
        Check(Math.Abs(standard.StandardDeviation(0) - 1.0) <= 0.03,
                "standard normal sd (" + standard.StandardDeviation(0) + ")");
        Check(standard.AcceptanceRate > 0.5 && standard.AcceptanceRate < 0.9,
                "acceptance rate for step width 1 (" + standard.AcceptanceRate + ")");
        Check(Math.Abs(standard.Cdf(-1.0, 0) - Metropolis.NormalCdfAtMinusOne) <= 0.02, "empirical CDF at -1");
        Check(Math.Abs(standard.Cdf(0.0, 0) - 0.5) <= 0.02, "empirical CDF at 0");
        Check(Math.Abs(standard.Cdf(1.0, 0) - (1.0 - Metropolis.NormalCdfAtMinusOne)) <= 0.02,
                "empirical CDF at 1");
        Check(Math.Abs(standard.Cdf(1.0, 0) + standard.Cdf(-1.0, 0) - 1.0) <= 0.03,
                "the two tails carry roughly one unit of probability together");

        ChainSummary narrow = Metropolis.Run(Metropolis.LogStandardNormal, 200000, 0.0, 0.25, 777);
        ChainSummary wide = Metropolis.Run(Metropolis.LogStandardNormal, 200000, 0.0, 4.0, 777);
        Check(Math.Abs(narrow.Mean(0)) <= 0.05 && Math.Abs(wide.Mean(0)) <= 0.05,
                "both step widths still target the standard normal mean");
        Check(narrow.AcceptanceRate > wide.AcceptanceRate,
                "the narrow proposal is accepted more often (" + narrow.AcceptanceRate + " vs "
                        + wide.AcceptanceRate + ")");

        ChainSummary far = Metropolis.Run(Metropolis.LogStandardNormal, 200000, 20.0, 1.0, 99);
        Check(Math.Abs(far.Mean(100000)) <= 0.05, "a chain started at 20 mixes into the target");
        Check(far.AcceptanceRate > 0.3, "the mixing chain still accepts proposals");
        Check(Math.Abs(far.Values[0] - 20.0) <= 1.0,
                "the first recorded state stays within one proposal of the start");

        ChainSummary bounded = Metropolis.Run(Metropolis.LogBoundedUniform, 100000, 0.5, 0.75, 5150);
        bool inside = true;
        foreach (double value in bounded.Values)
        {
            if (value < -1.0 || value > 1.0) inside = false;
        }
        Check(inside, "the bounded target never leaves [-1, 1]");
        Check(Math.Abs(bounded.Mean(0)) <= 0.02, "bounded uniform mean (" + bounded.Mean(0) + ")");
        Check(Math.Abs(bounded.MeanAbsolute(0) - 0.5) <= 0.02, "E|x| of the bounded uniform target");
        Check(Math.Abs(bounded.StandardDeviation(0) - 1.0 / Math.Sqrt(3.0)) <= 0.03,
                "bounded uniform sd (" + bounded.StandardDeviation(0) + ")");

        double[] repeat = Metropolis.Simulate(pi, 200000, 4242);
        for (int j = 0; j < pi.Length; j++)
        {
            CheckClose(repeat[j], frequencies[j], "the discrete simulation is deterministic at state " + j);
        }

        CheckThrows(() => Metropolis.Run(Metropolis.LogStandardNormal, 0, 0.0, 1.0, 1), "zero steps throw");
        CheckThrows(() => Metropolis.Run(Metropolis.LogStandardNormal, 10, 0.0, 0.0, 1), "zero width throws");
        CheckThrows(() => Metropolis.TransitionMatrix(new[] { 1.0 }), "a one-state target throws");
        CheckThrows(() => Metropolis.TransitionMatrix(new[] { 0.0, 1.0 }), "a zero entry throws");
        CheckThrows(() => Metropolis.TransitionMatrix(new[] { 0.3, 0.3 }), "an unnormalised target throws");
        CheckThrows(() => Metropolis.Simulate(pi, -5, 1), "a negative step count throws");

        Console.WriteLine("PASS");
    }

    private static void CheckClose(double actual, double expected, string label)
    {
        if (!(Math.Abs(actual - expected) <= 1e-12))
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
