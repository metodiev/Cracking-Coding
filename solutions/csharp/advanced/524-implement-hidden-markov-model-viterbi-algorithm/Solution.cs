// Task 524 - Implement Hidden Markov Model Viterbi algorithm. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Viterbi decoding of a discrete HMM (start distribution, row-stochastic transition matrix, row-
// stochastic emission matrix). The dynamic program keeps delta_t(j) = the best joint log probability
// of an observation prefix that ends in state j, together with a backpointer to the best predecessor;
// state indices at each step are probed with a strict ">" comparison so that ties are broken towards
// the smaller state index, which makes the result agree with the lexicographically smallest optimum
// of an exhaustive enumeration. Everything runs in log space: a length-2000 sequence has a joint
// probability far below the smallest positive double, so the linear-domain version would underflow to
// zero, while the log version stays finite (checked explicitly). Three decoders are implemented and
// compared: the log-space Viterbi, the same recurrence in the linear domain, and an exhaustive
// enumeration of all s^T state paths.
//
// Documented checks. (1) The classical weather HMM (Rainy/Sunny with start 0.6/0.4, transitions
// 0.7/0.3 and 0.4/0.6, emissions Walk/Shop/Clean 0.1/0.4/0.5 and 0.6/0.3/0.1) on the observation
// Walk, Shop, Clean: the hand-checked optimum is Sunny, Rainy, Rainy with joint probability 0.01344
// (unique - the runner counts the optimal paths), beating the hand-picked Sunny/Sunny/Rainy (0.00864)
// and Rainy/Rainy/Rainy (0.00588) alternative paths; the exhaustive enumeration of the eight paths
// reproduces exactly that path and value. (2) The reported probability equals the product of start, transition
// and emission factors of the returned path computed independently. (3) On four LCG-generated
// three-state HMMs with T = 5, 6, 7 and eight symbols the log-space decoder, the linear-domain
// decoder and the brute-force enumeration all return the same probability (relative 1e-9), and the
// paths are compared wherever the enumeration shows the optimum to be unique (the greedy backpointer
// tie-break and the lexicographic enumeration tie-break may legitimately differ on ties, so the
// comparison is guarded by an explicit count of the optimal paths). (4) Boundaries: a single
// observation decodes to the arg-max of start*emission with the smaller state index breaking ties; a
// deterministic (identity) emission matrix forces the state sequence that matches the symbols; a
// length-2000 decode stays finite and is below -1000 in log space, i.e. below the smallest positive
// double in the linear domain. (5) Errors: an out-of-range observation symbol, an empty observation
// and a non-stochastic or negative-probability model each throw ArgumentException.
using System;
using System.Collections.Generic;

internal sealed class Hmm
{
    internal readonly double[] Start;
    internal readonly double[][] Trans;
    internal readonly double[][] Emit;

    internal Hmm(double[] start, double[][] trans, double[][] emit)
    {
        if (trans.Length != start.Length || emit.Length != start.Length)
        {
            throw new ArgumentException("start, transition and emission sizes disagree");
        }
        Start = (double[])start.Clone();
        Trans = Clone(trans, "transition");
        Emit = Clone(emit, "emission");
        double startSum = 0.0;
        foreach (double value in Start)
        {
            if (value < 0.0) throw new ArgumentException("negative start probability");
            startSum += value;
        }
        if (Math.Abs(startSum - 1.0) > 1e-9) throw new ArgumentException("the start distribution must sum to 1");
    }

    private static double[][] Clone(double[][] rows, string role)
    {
        var copy = new double[rows.Length][];
        for (int i = 0; i < rows.Length; i++)
        {
            copy[i] = (double[])rows[i].Clone();
            double sum = 0.0;
            foreach (double value in copy[i])
            {
                if (value < 0.0) throw new ArgumentException("negative " + role + " probability");
                sum += value;
            }
            if (Math.Abs(sum - 1.0) > 1e-9)
            {
                throw new ArgumentException("every " + role + " row must sum to 1");
            }
        }
        return copy;
    }

    internal int States => Start.Length;
    internal int Symbols => Emit[0].Length;

    /** Rainy = 0, Sunny = 1; Walk = 0, Shop = 1, Clean = 2. */
    internal static Hmm Weather()
    {
        return new Hmm(
                new[] { 0.6, 0.4 },
                new[] { new[] { 0.7, 0.3 }, new[] { 0.4, 0.6 } },
                new[] { new[] { 0.1, 0.4, 0.5 }, new[] { 0.6, 0.3, 0.1 } });
    }

    internal static Hmm Lcg(int states, int symbols, int seed)
    {
        var start = new double[states];
        var trans = new double[states][];
        var emit = new double[states][];
        double startSum = 0.0;
        for (int i = 0; i < states; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            start[i] = 1.0 + seed % 100;
            startSum += start[i];
        }
        for (int i = 0; i < states; i++) start[i] /= startSum;
        for (int i = 0; i < states; i++)
        {
            trans[i] = new double[states];
            emit[i] = new double[symbols];
            double rowSum = 0.0;
            for (int j = 0; j < states; j++)
            {
                seed = (seed * 75 + 74) % 65537;
                trans[i][j] = 1.0 + seed % 100;
                rowSum += trans[i][j];
            }
            for (int j = 0; j < states; j++) trans[i][j] /= rowSum;
            rowSum = 0.0;
            for (int k = 0; k < symbols; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                emit[i][k] = 1.0 + seed % 100;
                rowSum += emit[i][k];
            }
            for (int k = 0; k < symbols; k++) emit[i][k] /= rowSum;
        }
        return new Hmm(start, trans, emit);
    }
}

internal sealed class Decoding
{
    internal readonly int[] Path;
    internal readonly double LogProbability;

    internal Decoding(int[] path, double logProbability)
    {
        Path = path;
        LogProbability = logProbability;
    }
}

internal static class Viterbi
{
    internal static void Validate(Hmm hmm, int[] observation)
    {
        if (observation.Length == 0) throw new ArgumentException("empty observation");
        foreach (int symbol in observation)
        {
            if (symbol < 0 || symbol >= hmm.Symbols) throw new ArgumentException("unknown symbol " + symbol);
        }
    }

    /** Log-space Viterbi; strict comparisons break ties towards the smaller state index. */
    internal static Decoding Decode(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var delta = new double[states];
        var back = new int[steps][];
        for (int k = 0; k < states; k++)
        {
            delta[k] = Log(hmm.Start[k]) + Log(hmm.Emit[k][observation[0]]);
        }
        for (int t = 1; t < steps; t++)
        {
            var next = new double[states];
            back[t] = new int[states];
            for (int k = 0; k < states; k++)
            {
                double bestValue = double.NegativeInfinity;
                int bestPrevious = 0;
                for (int j = 0; j < states; j++)
                {
                    double value = delta[j] + Log(hmm.Trans[j][k]);
                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestPrevious = j;
                    }
                }
                next[k] = bestValue + Log(hmm.Emit[k][observation[t]]);
                back[t][k] = bestPrevious;
            }
            delta = next;
        }
        int best = 0;
        for (int k = 1; k < states; k++)
        {
            if (delta[k] > delta[best]) best = k;
        }
        var path = new int[steps];
        int state = best;
        for (int t = steps - 1; t >= 0; t--)
        {
            path[t] = state;
            if (t > 0) state = back[t][state];
        }
        return new Decoding(path, delta[best]);
    }

    /** The identical recurrence carried out in the linear domain (fine for short sequences). */
    internal static Decoding DecodeLinear(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var delta = new double[states];
        var back = new int[steps][];
        for (int k = 0; k < states; k++) delta[k] = hmm.Start[k] * hmm.Emit[k][observation[0]];
        for (int t = 1; t < steps; t++)
        {
            var next = new double[states];
            back[t] = new int[states];
            for (int k = 0; k < states; k++)
            {
                double bestValue = double.NegativeInfinity;
                int bestPrevious = 0;
                for (int j = 0; j < states; j++)
                {
                    double value = delta[j] * hmm.Trans[j][k];
                    if (value > bestValue)
                    {
                        bestValue = value;
                        bestPrevious = j;
                    }
                }
                next[k] = bestValue * hmm.Emit[k][observation[t]];
                back[t][k] = bestPrevious;
            }
            delta = next;
        }
        int best = 0;
        for (int k = 1; k < states; k++)
        {
            if (delta[k] > delta[best]) best = k;
        }
        var path = new int[steps];
        int state = best;
        for (int t = steps - 1; t >= 0; t--)
        {
            path[t] = state;
            if (t > 0) state = back[t][state];
        }
        return new Decoding(path, Math.Log(delta[best]));
    }

    /** Exhaustive enumeration over all s^T paths, in lexicographic order so ties keep the first. */
    internal static Decoding BruteForce(Hmm hmm, int[] observation)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var path = new int[steps];
        var best = new int[steps];
        double bestLog = double.NegativeInfinity;
        while (true)
        {
            double log = Math.Log(hmm.Start[path[0]]) + Math.Log(hmm.Emit[path[0]][observation[0]]);
            for (int t = 1; t < steps; t++)
            {
                log += Math.Log(hmm.Trans[path[t - 1]][path[t]]) + Math.Log(hmm.Emit[path[t]][observation[t]]);
            }
            if (log > bestLog)
            {
                bestLog = log;
                Array.Copy(path, best, steps);
            }
            int position = steps - 1;
            while (position >= 0)
            {
                path[position]++;
                if (path[position] < states) break;
                path[position] = 0;
                position--;
            }
            if (position < 0) break;
        }
        return new Decoding(best, bestLog);
    }

    /** Counts how many state sequences attain the given optimum, allowing an absolute tolerance. */
    internal static int CountOptimal(Hmm hmm, int[] observation, double logProbability, double tolerance)
    {
        Validate(hmm, observation);
        int steps = observation.Length;
        int states = hmm.States;
        var path = new int[steps];
        int count = 0;
        while (true)
        {
            double log = Math.Log(hmm.Start[path[0]]) + Math.Log(hmm.Emit[path[0]][observation[0]]);
            for (int t = 1; t < steps; t++)
            {
                log += Math.Log(hmm.Trans[path[t - 1]][path[t]]) + Math.Log(hmm.Emit[path[t]][observation[t]]);
            }
            if (Math.Abs(log - logProbability) <= tolerance * Math.Max(1.0, Math.Abs(logProbability))) count++;
            int position = steps - 1;
            while (position >= 0)
            {
                path[position]++;
                if (path[position] < states) break;
                path[position] = 0;
                position--;
            }
            if (position < 0) break;
        }
        return count;
    }

    /** Joint probability of one explicit state sequence, computed without the DP. */
    internal static double JointProbability(Hmm hmm, int[] path, int[] observation)
    {
        double probability = hmm.Start[path[0]] * hmm.Emit[path[0]][observation[0]];
        for (int t = 1; t < path.Length; t++)
        {
            probability *= hmm.Trans[path[t - 1]][path[t]] * hmm.Emit[path[t]][observation[t]];
        }
        return probability;
    }

    internal static double Log(double value)
    {
        return value > 0.0 ? Math.Log(value) : double.NegativeInfinity;
    }
}

public static class Solution
{
    public static void Main()
    {
        Hmm weather = Hmm.Weather();
        int[] observation = { 0, 1, 2 };
        Decoding decoded = Viterbi.Decode(weather, observation);
        Check(decoded.Path[0] == 1 && decoded.Path[1] == 0 && decoded.Path[2] == 0,
                "hand-checked weather path Sunny, Rainy, Rainy");
        CheckClose(Math.Exp(decoded.LogProbability), 0.01344, 1e-15, "hand-checked weather probability");
        CheckClose(Viterbi.JointProbability(weather, decoded.Path, observation), Math.Exp(decoded.LogProbability),
                1e-15, "the probability equals the product over the returned path");
        CheckClose(Viterbi.JointProbability(weather, new[] { 0, 0, 0 }, observation), 0.00588, 1e-15,
                "hand-checked Rainy/Rainy/Rainy probability");
        CheckClose(Viterbi.JointProbability(weather, new[] { 1, 1, 0 }, observation), 0.00864, 1e-15,
                "hand-checked Sunny/Sunny/Rainy probability");
        Check(Math.Exp(decoded.LogProbability) > 0.00864 && Math.Exp(decoded.LogProbability) > 0.00588,
                "the decoded path beats the two hand-picked competitors");

        Decoding brute = Viterbi.BruteForce(weather, observation);
        CheckClose(decoded.LogProbability, brute.LogProbability, 1e-12, "brute force agrees on the probability");
        Check(Viterbi.CountOptimal(weather, observation, brute.LogProbability, 1e-12) == 1,
                "the weather optimum is unique");
        CheckSamePath(decoded.Path, brute.Path, "brute force agrees on the weather path");
        Decoding linear = Viterbi.DecodeLinear(weather, observation);
        CheckSamePath(decoded.Path, linear.Path, "the linear domain agrees on the weather path");
        CheckClose(linear.LogProbability, decoded.LogProbability, 1e-12, "linear and log probabilities");

        int[] seeds = { 101, 202, 303, 404 };
        int[] lengths = { 5, 6, 7, 5 };
        for (int s = 0; s < seeds.Length; s++)
        {
            Hmm model = Hmm.Lcg(3, 8, seeds[s]);
            var sequence = new int[lengths[s]];
            int seed = seeds[s] + 1;
            for (int t = 0; t < sequence.Length; t++)
            {
                seed = (seed * 75 + 74) % 65537;
                sequence[t] = seed % 8;
            }
            string tag = "seed " + seeds[s];
            Decoding fast = Viterbi.Decode(model, sequence);
            Decoding slow = Viterbi.BruteForce(model, sequence);
            CheckClose(fast.LogProbability, slow.LogProbability, 1e-9, tag + ": brute force probability");
            int optimal = Viterbi.CountOptimal(model, sequence, slow.LogProbability, 1e-9);
            Check(optimal >= 1, tag + ": at least one optimal path exists");
            if (optimal == 1) CheckSamePath(fast.Path, slow.Path, tag + ": unique optimum path");
            Decoding inLinear = Viterbi.DecodeLinear(model, sequence);
            CheckSamePath(fast.Path, inLinear.Path, tag + ": linear path");
            CheckClose(fast.LogProbability, inLinear.LogProbability, 1e-9, tag + ": linear probability");
            CheckClose(Viterbi.JointProbability(model, fast.Path, sequence), Math.Exp(fast.LogProbability),
                    1e-9, tag + ": reported probability matches the explicit product");
            Check(fast.Path.Length == sequence.Length, tag + ": the path has one state per observation");
        }

        var single = new[] { 1 };
        Decoding singleDecoded = Viterbi.Decode(weather, single);
        Check(singleDecoded.Path[0] == 0, "a single Shop observation favours Rainy (0.6*0.4 vs 0.4*0.3)");
        CheckClose(Math.Exp(singleDecoded.LogProbability), 0.6 * 0.4, 1e-15, "single observation probability");
        Check(Viterbi.CountOptimal(weather, single, Viterbi.BruteForce(weather, single).LogProbability, 1e-12) == 1,
                "the single-observation optimum is unique");

        int[] tieSequence = { 0 };
        Hmm tieModel = new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } },
                new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } });
        Check(Viterbi.Decode(tieModel, tieSequence).Path[0] == 0, "ties break towards the smaller state index");

        Hmm deterministic = new Hmm(new[] { 0.1, 0.1, 0.8 },
                new[] { new[] { 0.6, 0.3, 0.1 }, new[] { 0.2, 0.7, 0.1 }, new[] { 0.1, 0.1, 0.8 } },
                new[] { new[] { 1.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 0.0, 0.0, 1.0 } });
        int[] symbols = { 0, 1, 0 };
        Decoding forced = Viterbi.Decode(deterministic, symbols);
        Check(forced.Path[0] == 0 && forced.Path[1] == 1 && forced.Path[2] == 0,
                "a deterministic emission matrix forces the matching states");
        CheckClose(forced.LogProbability, Math.Log(0.1 * 0.3 * 0.2),
                1e-12, "deterministic emission probability");
        Check(Viterbi.CountOptimal(deterministic, symbols,
                Viterbi.BruteForce(deterministic, symbols).LogProbability, 1e-12) == 1,
                "the forced path is the only positive-probability path");
        CheckSamePath(forced.Path, Viterbi.BruteForce(deterministic, symbols).Path, "forced path brute force");

        var longSequence = new int[2000];
        int state = 7;
        for (int t = 0; t < longSequence.Length; t++)
        {
            state = (state * 75 + 74) % 65537;
            longSequence[t] = state % 8;
        }
        Decoding longDecoded = Viterbi.Decode(Hmm.Lcg(3, 8, 55), longSequence);
        Check(longDecoded.Path.Length == 2000, "the long sequence decodes to 2000 states");
        Check(!double.IsNaN(longDecoded.LogProbability) && !double.IsInfinity(longDecoded.LogProbability),
                "the log probability of a 2000-symbol sequence stays finite");
        Check(longDecoded.LogProbability < -1000.0,
                "the corresponding joint probability underflows the linear domain");

        CheckThrows(() => Viterbi.Decode(weather, new[] { 0, 5 }), "an unknown symbol throws");
        CheckThrows(() => Viterbi.Decode(weather, new int[0]), "an empty observation throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.5 } }, new[] { new[] { 1.0 } }),
                "a mismatched model throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.4 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } },
                new[] { new[] { 1.0 }, new[] { 1.0 } }), "a non-stochastic start throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.6 }, new[] { 0.5, 0.5 } },
                new[] { new[] { 1.0 }, new[] { 1.0 } }), "a non-stochastic transition row throws");
        CheckThrows(() => new Hmm(new[] { 0.5, 0.5 }, new[] { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } },
                new[] { new[] { -0.5, 1.5 }, new[] { 1.0, 0.0 } }), "a negative emission throws");

        Console.WriteLine("PASS");
    }

    private static void CheckSamePath(int[] a, int[] b, string label)
    {
        Check(a.Length == b.Length, label + ": length");
        for (int i = 0; i < a.Length; i++)
        {
            Check(a[i] == b[i], label + ": state " + i + " (" + a[i] + " != " + b[i] + ")");
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
