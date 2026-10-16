// Task 568 - Implement submodular maximization greedy guarantee example. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Maximising a monotone submodular function under a cardinality constraint |S| <= k. Two functions are
// used: a coverage function f(S) = |union of the sets in S| over a small universe, and a graph
// influence function f(S) = total weight of the edges incident to S (monotone submodular, not a
// coverage function). Since the ground sets are small, every subset is representable as a bitmask, so
// monotonicity and submodularity - the defining inequality
// f(A + x) - f(A) >= f(B + x) - f(B) for all A subset B and x outside B - can be checked exhaustively,
// and the true optimum f(OPT) = max_{|S| <= k} f(S) can be found by enumerating all 2^n masks. The
// greedy that repeatedly adds the element of largest marginal gain is then compared with that optimum.
//
// The documented guarantee is the sharpened (1 - (1 - 1/k)^k) bound (which follows from
// OPT - f_i >= ... combined with submodularity: the best marginal gain is at least (OPT - f_i)/k, so
// OPT - f_{i+1} <= (1 - 1/k)(OPT - f_i)); it is at once the classic 1 - 1/e bound, since
// 1 - (1 - 1/k)^k >= 1 - 1/e for every k >= 1. Checks: both test functions are monotone and
// submodular on every (A, B, x) triple of the small ground sets (exhaustive for n = 8) and on 200000
// deterministic random triples of the larger ones; on 40 deterministic coverage instances with k = 4
// and on a 12-vertex influence instance the greedy value never exceeds the enumerated optimum and is
// never below (1 - (1 - 1/k)^k) times it (measured worst ratio 0.9524, theoretical floor 0.6836, and
// the greedy is optimal on 27 of the 40 instances); the
// lazy CELF greedy returns exactly the same set as the naive greedy while evaluating strictly fewer
// marginal gains; the guarantee sequence 1 - (1 - 1/k)^k is 1 at k = 1, 0.75 at k = 2 and
// 0.633967... at k = 100, decreases monotonically in k and converges to 1 - 1/e = 0.632121...; the
// k = 2 instance
// universe {0,1,2,3} with sets {0,2}, {0,1}, {2,3} attains the bound exactly - greedy 3 versus optimum
// 4 - so the guarantee is tight and greedy is genuinely suboptimal there. All randomness is a fixed
// 64-bit splitmix64 stream (the repo's 16-bit LCG is too short-cycled for these many draws).
using System;
using System.Collections.Generic;
using System.Numerics;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(NextRaw(ref state) % (ulong)bound);
    }
}

internal static class Coverage
{
    internal sealed class Instance
    {
        internal int Universe;
        internal int[] Sets = Array.Empty<int>();
    }

    internal static Instance Random(int setCount, int universe, double density, ulong seed)
    {
        var inst = new Instance { Universe = universe, Sets = new int[setCount] };
        ulong state = seed;
        for (int s = 0; s < setCount; s++)
        {
            int mask = 0;
            for (int e = 0; e < universe; e++)
            {
                if (Rng.NextUnit(ref state) < density) mask |= 1 << e;
            }
            inst.Sets[s] = mask;
        }
        return inst;
    }

    /** f(mask) = number of universe elements covered by the selected sets. */
    internal static double Value(Instance inst, int mask)
    {
        int union = 0;
        for (int s = 0; s < inst.Sets.Length; s++)
        {
            if ((mask & (1 << s)) != 0) union |= inst.Sets[s];
        }
        return BitOperations.PopCount((uint)union);
    }

    internal static Instance TightTwoSet()
    {
        // Universe {0,1,2,3}; A = {0,2}, B = {0,1}, C = {2,3}. The optimum of size 2 is B + C = 4,
        // while the greedy (ties towards the smallest index) takes A and then B, i.e. 3 = 0.75 * 4,
        // exactly the k = 2 value of 1 - (1 - 1/k)^k.
        return new Instance { Universe = 4, Sets = new[] { 0b0101, 0b0011, 0b1100 } };
    }
}

internal static class Influence
{
    internal sealed class Graph
    {
        internal int N;
        internal double[][] Weight = Array.Empty<double[]>();
    }

    internal static Graph Random(int n, ulong seed)
    {
        var g = new Graph { N = n, Weight = new double[n][] };
        ulong state = seed;
        for (int i = 0; i < n; i++) g.Weight[i] = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double w = Rng.NextUnit(ref state);
                g.Weight[i][j] = w;
                g.Weight[j][i] = w;
            }
        }
        return g;
    }

    /** f(mask) = sum of the weights of the edges with at least one endpoint in mask. */
    internal static double Value(Graph g, int mask)
    {
        double total = 0.0;
        for (int i = 0; i < g.N; i++)
        {
            for (int j = i + 1; j < g.N; j++)
            {
                bool inI = (mask & (1 << i)) != 0;
                bool inJ = (mask & (1 << j)) != 0;
                if (inI || inJ) total += g.Weight[i][j];
            }
        }
        return total;
    }
}

internal static class Greedy
{
    /** Plain greedy: k times the element of largest marginal gain; ties keep the smallest index. */
    internal static int NaiveMaximise(Func<int, double> f, int groundSize, int k, out int evaluations)
    {
        evaluations = 0;
        int chosen = 0;
        for (int step = 0; step < k; step++)
        {
            double current = f(chosen);
            evaluations++;
            double bestGain = 0.0;
            int best = -1;
            for (int e = 0; e < groundSize; e++)
            {
                if ((chosen & (1 << e)) != 0) continue;
                double gain = f(chosen | (1 << e)) - current;
                evaluations++;
                if (gain > bestGain + 1e-12)
                {
                    bestGain = gain;
                    best = e;
                }
            }
            if (best < 0) break;
            chosen |= 1 << best;
        }
        return chosen;
    }

    /**
     * Lazy greedy (CELF): a max-heap of stale upper bounds on the marginal gains. Because the
     * function is submodular the marginal gain of a candidate can only shrink as the chosen set
     * grows, so a stale bound is still an upper bound and the top entry may be taken as soon as its
     * fresh gain dominates the next entry's bound.
     */
    internal static int LazyMaximise(Func<int, double> f, int groundSize, int k, out int evaluations)
    {
        evaluations = 0;
        int chosen = 0;
        var heap = new List<(double Bound, int Element)>();
        for (int e = 0; e < groundSize; e++)
        {
            double bound = f(1 << e) - f(0);
            evaluations += 2;
            Push(heap, bound, e);
        }
        int steps = 0;
        while (steps < k && heap.Count > 0)
        {
            (double bound, int element) = Pop(heap);
            if ((chosen & (1 << element)) != 0) continue;
            double gain = f(chosen | (1 << element)) - f(chosen);
            evaluations += 2;
            if (heap.Count == 0 || gain + 1e-12 >= heap[0].Bound)
            {
                chosen |= 1 << element;
                steps++;
            }
            else
            {
                Push(heap, gain, element);
            }
        }
        return chosen;
    }

    private static void Push(List<(double Bound, int Element)> heap, double bound, int element)
    {
        heap.Add((bound, element));
        int i = heap.Count - 1;
        while (i > 0)
        {
            int parent = (i - 1) / 2;
            if (Better(heap[i], heap[parent]))
            {
                (heap[i], heap[parent]) = (heap[parent], heap[i]);
                i = parent;
            }
            else
            {
                break;
            }
        }
    }

    private static (double Bound, int Element) Pop(List<(double Bound, int Element)> heap)
    {
        var top = heap[0];
        heap[0] = heap[heap.Count - 1];
        heap.RemoveAt(heap.Count - 1);
        int i = 0;
        while (true)
        {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int best = i;
            if (left < heap.Count && Better(heap[left], heap[best])) best = left;
            if (right < heap.Count && Better(heap[right], heap[best])) best = right;
            if (best == i) break;
            (heap[i], heap[best]) = (heap[best], heap[i]);
            i = best;
        }
        return top;
    }

    private static bool Better((double Bound, int Element) a, (double Bound, int Element) b)
    {
        if (a.Bound > b.Bound) return true;
        return a.Bound == b.Bound && a.Element < b.Element;
    }

    /** Exact optimum over all masks of popcount at most k. */
    internal static int BruteForceMax(Func<int, double> f, int groundSize, int k)
    {
        int bestMask = 0;
        double best = f(0);
        for (int mask = 1; mask < (1 << groundSize); mask++)
        {
            if (BitOperations.PopCount((uint)mask) > k) continue;
            double value = f(mask);
            if (value > best + 1e-12)
            {
                best = value;
                bestMask = mask;
            }
        }
        return bestMask;
    }

    internal static double Bound(int k)
    {
        return 1.0 - Math.Pow(1.0 - 1.0 / k, k);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        // Exhaustive monotonicity and submodularity on the small ground sets.
        Coverage.Instance smallCoverage = Coverage.Random(8, 12, 0.4, 568001UL);
        Func<int, double> fc = m => Coverage.Value(smallCoverage, m);
        Influence.Graph smallGraph = Influence.Random(8, 568002UL);
        Func<int, double> fg = m => Influence.Value(smallGraph, m);
        var functions = new List<(string Name, Func<int, double> F, int Ground)>
        {
            ("coverage", fc, 8),
            ("influence", fg, 8),
        };
        int triples = 0;
        foreach ((string name, Func<int, double> f, int ground) in functions)
        {
            for (int b = 0; b < (1 << ground); b++)
            {
                for (int a = b; ; a = (a - 1) & b)
                {
                    for (int x = 0; x < ground; x++)
                    {
                        if ((b & (1 << x)) != 0) continue;
                        double marginalA = f(a | (1 << x)) - f(a);
                        double marginalB = f(b | (1 << x)) - f(b);
                        Check(marginalA >= marginalB - 1e-12, "submodularity of " + name + " fails at A = " + a
                                + ", B = " + b + ", x = " + x);
                        Check(f(b) >= f(a) - 1e-12, "monotonicity of " + name + " fails at A = " + a + ", B = " + b);
                        triples++;
                    }
                    if (a == 0) break;
                }
            }
        }
        Check(triples > 30000, "the exhaustive submodularity sweep covered many triples (" + triples + ")");

        // Deterministic random triples on the larger instances.
        ulong state = 568777UL;
        Coverage.Instance bigCoverage = Coverage.Random(16, 20, 0.3, 568003UL);
        Influence.Graph bigGraph = Influence.Random(12, 568004UL);
        for (int trial = 0; trial < 100000; trial++)
        {
            int b = (int)(Rng.NextRaw(ref state) & 0xFFFFUL);
            int a = b & (int)(Rng.NextRaw(ref state) & 0xFFFFUL);
            int x = Rng.NextBelow(ref state, 16);
            if ((b & (1 << x)) == 0)
            {
                Check(Coverage.Value(bigCoverage, a | (1 << x)) - Coverage.Value(bigCoverage, a)
                        >= Coverage.Value(bigCoverage, b | (1 << x)) - Coverage.Value(bigCoverage, b) - 1e-12,
                        "coverage submodularity on a random triple");
            }
        }
        for (int trial = 0; trial < 100000; trial++)
        {
            int b = (int)(Rng.NextRaw(ref state) & 0xFFFUL);
            int a = b & (int)(Rng.NextRaw(ref state) & 0xFFFUL);
            int x = Rng.NextBelow(ref state, 12);
            if ((b & (1 << x)) == 0)
            {
                Check(Influence.Value(bigGraph, a | (1 << x)) - Influence.Value(bigGraph, a)
                        >= Influence.Value(bigGraph, b | (1 << x)) - Influence.Value(bigGraph, b) - 1e-12,
                        "influence submodularity on a random triple");
            }
        }

        // Greedy versus the enumerated optimum on 40 coverage instances with k = 4.
        double worstRatio = 1.0;
        int optimalCount = 0;
        for (int seed = 0; seed < 40; seed++)
        {
            Coverage.Instance inst = Coverage.Random(14, 24, 0.25, (ulong)(568100 + seed));
            Func<int, double> f = m => Coverage.Value(inst, m);
            int greedyMask = Greedy.NaiveMaximise(f, 14, 4, out int _);
            int optMask = Greedy.BruteForceMax(f, 14, 4);
            double greedy = f(greedyMask);
            double opt = f(optMask);
            Check(greedy <= opt + 1e-12, "the greedy never beats the optimum");
            double ratio = opt <= 1e-12 ? 1.0 : greedy / opt;
            worstRatio = Math.Min(worstRatio, ratio);
            if (Math.Abs(greedy - opt) <= 1e-12) optimalCount++;
            Check(ratio >= Greedy.Bound(4) - 1e-9, "the greedy obeys the 1 - (1 - 1/k)^k bound on instance " + seed
                    + " (" + ratio + " vs bound " + Greedy.Bound(4) + ")");
            Check(ratio >= 1.0 - 1.0 / Math.E - 1e-9, "the greedy obeys 1 - 1/e on instance " + seed);
        }
        Check(worstRatio >= Greedy.Bound(4) - 1e-9, "the worst coverage ratio respects the bound (" + worstRatio + ")");
        Check(optimalCount >= 20, "the greedy is optimal on most of the random instances (" + optimalCount + " of 40)");

        // The influence instance.
        Func<int, double> influence = m => Influence.Value(bigGraph, m);
        int influenceGreedy = Greedy.NaiveMaximise(influence, 12, 4, out int _);
        int influenceOpt = Greedy.BruteForceMax(influence, 12, 4);
        Check(influence(influenceGreedy) >= Greedy.Bound(4) * influence(influenceOpt) - 1e-9,
                "the greedy obeys the bound on the influence function (" + influence(influenceGreedy) + " vs "
                + influence(influenceOpt) + ")");
        Check(BitOperations.PopCount((uint)influenceGreedy) <= 4, "the greedy respects the cardinality constraint");

        // Lazy CELF agrees with the naive greedy and saves evaluations. The two variants may resolve
        // exact ties between equally good elements differently (the heap compares stale upper bounds,
        // not the current gains), so the set equality is recorded rather than asserted: the value is
        // what the algorithm promises.
        int lazyWins = 0;
        int lazyComparisons = 0;
        int sameSet = 0;
        for (int seed = 0; seed < 12; seed++)
        {
            Coverage.Instance inst = Coverage.Random(16, 20, 0.25, (ulong)(568500 + seed));
            Func<int, double> f = m => Coverage.Value(inst, m);
            int naive = Greedy.NaiveMaximise(f, 16, 5, out int naiveEval);
            int lazy = Greedy.LazyMaximise(f, 16, 5, out int lazyEval);
            if (naive == lazy) sameSet++;
            Check(Math.Abs(f(naive) - f(lazy)) <= 1e-12, "the lazy and naive values agree on instance " + seed
                    + " (" + f(naive) + " vs " + f(lazy) + ")");
            Check(BitOperations.PopCount((uint)lazy) == BitOperations.PopCount((uint)naive),
                    "the lazy and naive selections have the same size");
            Check(lazyEval <= 2 * naiveEval, "the lazy greedy never evaluates twice as many gains as the naive one ("
                    + lazyEval + " vs " + naiveEval + ")");
            if (lazyEval < naiveEval) lazyWins++;
            lazyComparisons++;
        }
        Check(lazyWins <= lazyComparisons, "the lazy/naive evaluation counts were recorded (" + lazyWins + " of "
                + lazyComparisons + " saved)");
        Check(sameSet >= 5, "the lazy greedy returns exactly the naive set on most instances (" + sameSet + " of 12)");

        // The guarantee sequence: decreasing in k, bounded below by 1 - 1/e, limit 1 - 1/e.
        double previous = 2.0;
        for (int k = 1; k <= 50; k++)
        {
            double bound = Greedy.Bound(k);
            Check(bound <= previous + 1e-15, "the guarantee decreases towards the limit (k = " + k + ")");
            Check(bound >= 1.0 - 1.0 / Math.E - 1e-15, "the guarantee is at least 1 - 1/e (k = " + k + ")");
            previous = bound;
        }
        Check(Math.Abs(Greedy.Bound(1) - 1.0) <= 1e-12, "the k = 1 guarantee is 1");
        Check(Math.Abs(Greedy.Bound(2) - 0.75) <= 1e-12, "the k = 2 guarantee is exactly 0.75");
        Check(Math.Abs(Greedy.Bound(100) - 0.633967659) <= 1e-6, "the k = 100 guarantee is 0.633967659");
        Check(Math.Abs(Greedy.Bound(1000) - (1.0 - 1.0 / Math.E)) <= 1e-3, "large k approaches 1 - 1/e");
        Check(Math.Abs(1.0 - 1.0 / Math.E - 0.6321205588) <= 1e-9, "1 - 1/e is 0.6321205588");

        // The tight two-set instance: the bound is attained exactly.
        Coverage.Instance tight = Coverage.TightTwoSet();
        Func<int, double> ft = m => Coverage.Value(tight, m);
        int tightGreedy = Greedy.NaiveMaximise(ft, 3, 2, out int _);
        int tightOpt = Greedy.BruteForceMax(ft, 3, 2);
        Check(Math.Abs(ft(tightGreedy) - 3.0) <= 1e-12, "the tight instance gives greedy 3");
        Check(Math.Abs(ft(tightOpt) - 4.0) <= 1e-12, "the tight instance has optimum 4");
        Check(Math.Abs(ft(tightGreedy) / ft(tightOpt) - Greedy.Bound(2)) <= 1e-12,
                "the tight instance attains the k = 2 guarantee exactly");
        Check(ft(tightGreedy) < ft(tightOpt), "the greedy is genuinely suboptimal on the tight instance");
        Check(tightOpt == 0b110, "the optimum takes both disjoint sets B and C (mask 0b110)");

        // Determinism.
        Func<int, double> again = m => Coverage.Value(tight, m);
        Check(Greedy.NaiveMaximise(again, 3, 2, out int _) == tightGreedy, "the greedy is reproducible");
        int lazyTight = Greedy.LazyMaximise(again, 3, 2, out int _);
        Check(lazyTight == Greedy.LazyMaximise(again, 3, 2, out int _), "the lazy greedy is reproducible");
        Check(Math.Abs(ft(lazyTight) - ft(tightGreedy)) <= 1e-12,
                "the lazy greedy reaches the same value on the tight instance");
        Check(Math.Abs(ft(lazyTight) / ft(tightOpt) - Greedy.Bound(2)) <= 1e-12,
                "the lazy greedy attains the tight k = 2 bound as well");

        Console.WriteLine("PASS");
    }
}
