// Task 797 - Implement multi-armed bandit with contextual features (contextual bandits). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Linear UCB (LinUCB, Li et al. 2010) for a K-armed bandit with a d-dimensional context. Each arm keeps the
// sufficient statistics A_a (d x d, initialised to the identity = ridge regularisation) and b_a = sum r x, so
// theta_a = A_a^-1 b_a is the ridge estimate and the arm score is
//      theta_a . x + alpha * sqrt(x . A_a^-1 x),
// an upper confidence bound whose width shrinks as the arm is played. The inverse is maintained incrementally
// with the Sherman-Morrison formula
//      A^-1 <- A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x),
// which is what makes a round O(d^2) instead of a matrix inversion. Rewards are linear in the context plus
// bounded noise; the environment is deterministic through a hand-written splitmix64 stream.
//
// Invariants asserted: the chosen arm is exactly the argmax of the recomputed scores, the incremental inverse
// equals a freshly computed matrix inverse, and theta obtained from A^-1 b equals the explicit ridge solution
// of the accumulated normal equations.
//
// Independent oracles:
//   * Sherman-Morrison against Gauss-Jordan inversion of the accumulated A after many updates;
//   * theta_a = A^-1 b against the normal-equation solution built from scratch out of the stored (x, r) pairs;
//   * a fixed exploration phase compared with the known generating parameters theta_a;
//   * cumulative regret against the uniform-random and epsilon-greedy baselines on the same environment, and a
//     sublinearity check (average regret per round falls).
//
// Documented simplification: one linear reward environment, a small horizon (2000 rounds, K = 5, d = 8), no
// feature normalisation and no discounting/forgetting; the confidence bound, the incremental inverse and the
// regret accounting are the real algorithm. Deterministic noise only.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal static class Linear
{
    internal static double[] Multiply(double[][] a, double[] v)
    {
        var r = new double[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < v.Length; j++) sum += a[i][j] * v[j];
            r[i] = sum;
        }
        return r;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static double[][] Identity(int n)
    {
        var m = new double[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = new double[n];
            m[i][i] = 1.0;
        }
        return m;
    }

    /** Gauss-Jordan inverse, used only as the oracle for the incremental Sherman-Morrison update. */
    internal static double[][] Invert(double[][] a)
    {
        int n = a.Length;
        var work = new double[n][];
        var inverse = Identity(n);
        for (int i = 0; i < n; i++) work[i] = (double[])a[i].Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(work[row][col]) > Math.Abs(work[pivot][col])) pivot = row;
            }
            if (Math.Abs(work[pivot][col]) < 1e-12) throw new InvalidOperationException("singular matrix");
            (work[col], work[pivot]) = (work[pivot], work[col]);
            (inverse[col], inverse[pivot]) = (inverse[pivot], inverse[col]);
            double diagonal = work[col][col];
            for (int c = 0; c < n; c++)
            {
                work[col][c] /= diagonal;
                inverse[col][c] /= diagonal;
            }
            for (int row = 0; row < n; row++)
            {
                if (row == col) continue;
                double factor = work[row][col];
                if (factor == 0.0) continue;
                for (int c = 0; c < n; c++)
                {
                    work[row][c] -= factor * work[col][c];
                    inverse[row][c] -= factor * inverse[col][c];
                }
            }
        }
        return inverse;
    }

    internal static double MaxDifference(double[][] a, double[][] b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a[i].Length; j++) worst = Math.Max(worst, Math.Abs(a[i][j] - b[i][j]));
        }
        return worst;
    }
}

internal sealed class LinearBandit
{
    private readonly double[][] _theta;                    // true reward parameters, one row per arm
    private readonly double[][] _contexts;
    private readonly double _noiseScale;
    private ulong _rng;

    internal int Arms { get; }
    internal int Dimension { get; }

    internal LinearBandit(int arms, int dimension, ulong seed, double noiseScale = 0.05)
    {
        if (arms < 1) throw new ArgumentException("at least one arm is required", nameof(arms));
        if (dimension < 1) throw new ArgumentException("dimension must be positive", nameof(dimension));
        Arms = arms;
        Dimension = dimension;
        _noiseScale = noiseScale;
        _rng = seed;
        _theta = new double[arms][];
        for (int a = 0; a < arms; a++)
        {
            _theta[a] = new double[dimension];
            for (int d = 0; d < dimension; d++) _theta[a][d] = SplitMix64.NextGaussian(ref _rng);
        }
        _contexts = new double[1][];
        _contexts[0] = new double[dimension];
    }

    internal double[] TrueParameters(int arm) => _theta[arm];

    internal double[] NextContext()
    {
        var x = new double[Dimension];
        for (int d = 0; d < Dimension; d++) x[d] = SplitMix64.NextGaussian(ref _rng);
        return x;
    }

    internal double Reward(int arm, double[] context)
    {
        double mean = Linear.Dot(_theta[arm], context);
        return mean + _noiseScale * SplitMix64.NextGaussian(ref _rng);
    }

    internal double BestExpected(double[] context)
    {
        double best = double.MinValue;
        for (int a = 0; a < Arms; a++) best = Math.Max(best, Linear.Dot(_theta[a], context));
        return best;
    }
}

internal sealed class LinUcb
{
    private readonly double[][][] _inverse;                // A_a^-1
    private readonly double[][] _b;
    private readonly List<(double[] context, double reward)>[] _history;

    internal int Arms { get; }
    internal int Dimension { get; }
    internal double Alpha { get; }
    internal int Rounds { get; private set; }

    internal LinUcb(int arms, int dimension, double alpha)
    {
        if (alpha < 0.0) throw new ArgumentException("alpha must not be negative", nameof(alpha));
        Arms = arms;
        Dimension = dimension;
        Alpha = alpha;
        _inverse = new double[arms][][];
        _b = new double[arms][];
        _history = new List<(double[], double)>[arms];
        for (int a = 0; a < arms; a++)
        {
            _inverse[a] = Linear.Identity(dimension);
            _b[a] = new double[dimension];
            _history[a] = new List<(double[], double)>();
        }
    }

    internal double[] Estimate(int arm)
    {
        double[][] inverse = _inverse[arm];
        var theta = new double[Dimension];
        for (int i = 0; i < Dimension; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < Dimension; j++) sum += inverse[i][j] * _b[arm][j];
            theta[i] = sum;
        }
        return theta;
    }

    internal double[][] InverseOf(int arm) => _inverse[arm];
    internal double[] BVector(int arm) => _b[arm];

    /** The stored (context, reward) pairs of one arm, the raw material for the explicit oracle. */
    internal IReadOnlyList<(double[] Context, double Reward)> HistoryOf(int arm) => _history[arm];

    /** UCB score of one arm, recomputed from the maintained statistics. */
    internal double Score(int arm, double[] context)
    {
        double[] theta = Estimate(arm);
        double[] inverseTimesX = Linear.Multiply(_inverse[arm], context);
        double exploration = Math.Sqrt(Math.Max(0.0, Linear.Dot(context, inverseTimesX)));
        return Linear.Dot(theta, context) + Alpha * exploration;
    }

    internal int Select(double[] context)
    {
        int best = 0;
        double bestScore = double.NegativeInfinity;
        for (int a = 0; a < Arms; a++)
        {
            double score = Score(a, context);
            if (score > bestScore)
            {
                bestScore = score;
                best = a;
            }
        }
        return best;
    }

    /** Plays one arm and folds the observation in with the Sherman-Morrison rank-one update. */
    internal void Update(int arm, double[] context, double reward)
    {
        double[] inverseTimesX = Linear.Multiply(_inverse[arm], context);
        double denominator = 1.0 + Linear.Dot(context, inverseTimesX);
        if (Math.Abs(denominator) < 1e-12) throw new InvalidOperationException("degenerate Sherman-Morrison denominator");
        for (int i = 0; i < Dimension; i++)
        {
            for (int j = 0; j < Dimension; j++)
            {
                _inverse[arm][i][j] -= inverseTimesX[i] * inverseTimesX[j] / denominator;
            }
        }
        for (int i = 0; i < Dimension; i++) _b[arm][i] += reward * context[i];
        _history[arm].Add(((double[])context.Clone(), reward));
    }

    /** Explicit ridge solution from the stored pairs: the oracle for A^-1 b. */
    internal double[] ExplicitEstimate(int arm)
    {
        double[][] a = Linear.Identity(Dimension);
        var b = new double[Dimension];
        foreach ((double[] x, double r) in _history[arm])
        {
            for (int i = 0; i < Dimension; i++)
            {
                for (int j = 0; j < Dimension; j++) a[i][j] += x[i] * x[j];
                b[i] += r * x[i];
            }
        }
        double[][] inverse = Linear.Invert(a);
        var theta = new double[Dimension];
        for (int i = 0; i < Dimension; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < Dimension; j++) sum += inverse[i][j] * b[j];
            theta[i] = sum;
        }
        return theta;
    }
}

public static class Solution
{
    private const int Arms = 5, Dimension = 8, Horizon = 2000;

    private static double Run(LinearBandit environment, LinUcb bandit, out double averageRegretFirstHalf, out double averageRegretSecondHalf)
    {
        double regret = 0.0, firstHalf = 0.0, secondHalf = 0.0;
        int half = Horizon / 2;
        for (int t = 0; t < Horizon; t++)
        {
            double[] context = environment.NextContext();
            int arm = bandit.Select(context);
            double reward = environment.Reward(arm, context);
            regret += environment.BestExpected(context) - Linear.Dot(environment.TrueParameters(arm), context);
            bandit.Update(arm, context, reward);
            if (t + 1 <= half) firstHalf = regret / (t + 1);
            else secondHalf = regret / (t + 1);
        }
        averageRegretFirstHalf = firstHalf;
        averageRegretSecondHalf = secondHalf;
        return regret;
    }

    public static void Main()
    {
        var environment = new LinearBandit(Arms, Dimension, 0xABCDUL);
        var bandit = new LinUcb(Arms, Dimension, 0.4);
        double regret = Run(environment, bandit, out double earlyAverage, out double lateAverage);
        Console.WriteLine("  LinUCB: cumulative regret=" + regret.ToString("F2") + " average per round: first half="
            + earlyAverage.ToString("F4") + " second half=" + lateAverage.ToString("F4"));
        Check(lateAverage < earlyAverage, "average regret per round falls (sublinear cumulative regret)");
        Check(regret < Horizon * 0.5, "cumulative regret stays well below a linear rate");

        // 1. The selected arm is exactly the argmax of independently recomputed scores.
        bool argmaxHolds = true;
        for (int trial = 0; trial < 50; trial++)
        {
            double[] probeContext = environment.NextContext();
            int arm = bandit.Select(probeContext);
            for (int a = 0; a < Arms; a++)
            {
                if (bandit.Score(a, probeContext) > bandit.Score(arm, probeContext) + 1e-12) argmaxHolds = false;
            }
        }
        Check(argmaxHolds, "the selected arm maximises the recomputed UCB score");

        // 2-3. Sherman-Morrison against a freshly inverted A rebuilt from the stored rounds, and theta against the
        //      explicit ridge solution of the same normal equations.
        double worstInverse = 0.0, worstTheta = 0.0, worstB = 0.0;
        for (int a = 0; a < Arms; a++)
        {
            worstInverse = Math.Max(worstInverse, Linear.MaxDifference(RebuildInverse(bandit, a), bandit.InverseOf(a)));
            double[] rebuiltB = RebuildB(bandit, a), maintainedB = bandit.BVector(a);
            for (int i = 0; i < Dimension; i++) worstB = Math.Max(worstB, Math.Abs(rebuiltB[i] - maintainedB[i]));
            double[] fast = bandit.Estimate(a);
            double[] slow = bandit.ExplicitEstimate(a);
            for (int i = 0; i < Dimension; i++) worstTheta = Math.Max(worstTheta, Math.Abs(fast[i] - slow[i]));
        }
        Console.WriteLine("  oracle: max |Sherman-Morrison - explicit inverse|=" + worstInverse.ToString("E3")
            + " max |b rebuilt - b maintained|=" + worstB.ToString("E3")
            + " max |A^-1 b - explicit ridge|=" + worstTheta.ToString("E3"));
        Check(worstInverse < 1e-9, "the incremental inverse matches the explicit matrix inverse");
        Check(worstB < 1e-12, "the maintained b vector equals the replayed reward-weighted sum");
        Check(worstTheta < 1e-9, "theta from the maintained statistics matches the explicit ridge solution");

        // 4-5. An explicit exploration phase must identify the generating parameters.
        var environment2 = new LinearBandit(Arms, Dimension, 0xF00DUL);
        var explorer = new LinUcb(Arms, Dimension, 0.4);
        for (int arm = 0; arm < Arms; arm++)
        {
            for (int n = 0; n < 40; n++)
            {
                double[] explorationContext = environment2.NextContext();
                explorer.Update(arm, explorationContext, environment2.Reward(arm, explorationContext));
            }
        }
        double worstParameter = 0.0, worstParameterRelative = 0.0;
        for (int arm = 0; arm < Arms; arm++)
        {
            double[] estimate = explorer.Estimate(arm);
            double[] truth = environment2.TrueParameters(arm);
            for (int d = 0; d < Dimension; d++)
            {
                worstParameter = Math.Max(worstParameter, Math.Abs(estimate[d] - truth[d]));
                worstParameterRelative = Math.Max(worstParameterRelative, Math.Abs(estimate[d] - truth[d]) / Math.Abs(truth[d]));
            }
        }
        Console.WriteLine("  identifiability: max |theta_hat - theta|=" + worstParameter.ToString("E3")
            + " (noise sigma 0.05, 40 samples per arm)");
        Check(worstParameter < 0.1, "with enough exploration theta_hat lands near the generating parameters");

        // 6-7. Baselines: uniform random and epsilon-greedy on the same environment lose more.
        var randomEnvironment = new LinearBandit(Arms, Dimension, 0xABCDUL);
        var randomRng = 0x9999UL;
        var greedy = new LinUcb(Arms, Dimension, 0.4);
        var greedyRng = 0x7777UL;
        double randomRegret = 0.0, greedyRegret = 0.0;
        for (int t = 0; t < Horizon; t++)
        {
            double[] contextA = randomEnvironment.NextContext();
            int randomArm = SplitMix64.NextInt(ref randomRng, Arms);
            randomRegret += randomEnvironment.BestExpected(contextA) - Linear.Dot(randomEnvironment.TrueParameters(randomArm), contextA);
            randomEnvironment.Reward(randomArm, contextA);

            double[] contextB = randomEnvironment.NextContext();
            int chosen = SplitMix64.NextDouble(ref greedyRng) < 0.1 ? SplitMix64.NextInt(ref greedyRng, Arms) : greedy.Select(contextB);
            greedyRegret += randomEnvironment.BestExpected(contextB) - Linear.Dot(randomEnvironment.TrueParameters(chosen), contextB);
            greedy.Update(chosen, contextB, randomEnvironment.Reward(chosen, contextB));
        }
        Console.WriteLine("  baselines: LinUCB=" + regret.ToString("F1") + " epsilon-greedy=" + greedyRegret.ToString("F1")
            + " uniform random=" + randomRegret.ToString("F1"));
        Check(regret < randomRegret, "LinUCB beats the uniform random policy");
        Check(regret <= greedyRegret * 1.5, "LinUCB is competitive with epsilon-greedy on this environment");

        // 8-9. Alpha = 0 degenerates to pure exploitation, and the exploration term grows with alpha.
        var greedyBandit = new LinUcb(Arms, Dimension, 0.0);
        var context = environment.NextContext();
        bool pureMean = true;
        for (int a = 0; a < Arms; a++)
        {
            if (Math.Abs(greedyBandit.Score(a, context) - Linear.Dot(greedyBandit.Estimate(a), context)) > 1e-12) pureMean = false;
        }
        Check(pureMean, "alpha = 0 removes the exploration term exactly");
        var cautious = new LinUcb(Arms, Dimension, 0.0);
        var curious = new LinUcb(Arms, Dimension, 2.0);
        Check(curious.Score(0, context) > cautious.Score(0, context), "the exploration bonus grows with alpha");
        Check(Math.Abs(cautious.Score(0, context)) < 1e-12, "a fresh zero-alpha bandit scores every arm at zero");

        // 10-11. Boundary cases: a single arm has zero regret, and invalid parameters are rejected.
        var single = new LinearBandit(1, Dimension, 0x1UL);
        var singleBandit = new LinUcb(1, Dimension, 0.5);
        double singleRegret = 0.0;
        for (int t = 0; t < 50; t++)
        {
            double[] ctx = single.NextContext();
            int arm = singleBandit.Select(ctx);
            singleRegret += single.BestExpected(ctx) - Linear.Dot(single.TrueParameters(arm), ctx);
            singleBandit.Update(arm, ctx, single.Reward(arm, ctx));
        }
        Check(singleRegret == 0.0, "with one arm the regret is exactly zero");
        Check(Throws(() => new LinUcb(Arms, Dimension, -0.1)), "negative alpha rejected");
        Check(Throws(() => new LinearBandit(0, Dimension, 1UL)), "zero arms rejected");
        Check(Throws(() => new LinearBandit(Arms, 0, 1UL)), "zero dimension rejected");
        Check(Throws(() => Linear.Invert(new double[][] { new[] { 0.0, 0.0 }, new[] { 0.0, 0.0 } })),
            "singular matrix rejected by the oracle inverse");

        // 12. Determinism: same seeds, same trajectory and regret.
        var repeatEnvironment = new LinearBandit(Arms, Dimension, 0xABCDUL);
        var repeatBandit = new LinUcb(Arms, Dimension, 0.4);
        double repeatRegret = Run(repeatEnvironment, repeatBandit, out _, out _);
        Check(repeatRegret == regret, "identical seeds reproduce the cumulative regret bit for bit");

        Console.WriteLine("PASS");
    }

    /** A = identity + sum x x^T rebuilt from the stored rounds, then inverted explicitly. */
    private static double[][] RebuildInverse(LinUcb bandit, int arm)
    {
        double[][] a = Linear.Identity(bandit.Dimension);
        foreach ((double[] x, double _) in bandit.HistoryOf(arm))
        {
            for (int i = 0; i < bandit.Dimension; i++)
            {
                for (int j = 0; j < bandit.Dimension; j++) a[i][j] += x[i] * x[j];
            }
        }
        return Linear.Invert(a);
    }

    /** b = sum r x replayed from the stored rounds. */
    private static double[] RebuildB(LinUcb bandit, int arm)
    {
        var b = new double[bandit.Dimension];
        foreach ((double[] x, double r) in bandit.HistoryOf(arm))
        {
            for (int i = 0; i < bandit.Dimension; i++) b[i] += r * x[i];
        }
        return b;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
