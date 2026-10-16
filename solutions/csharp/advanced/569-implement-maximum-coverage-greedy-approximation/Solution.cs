// Task 569 - Implement maximum coverage greedy approximation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Maximum coverage: given m subsets S_1..S_m of a small universe and a budget k, pick k sets whose
// union has maximum cardinality (and a weighted variant where each element carries a weight). Sets and
// selections are int bitmasks, so the union - and therefore the covered count - is an OR plus a
// popcount, and the true optimum is obtained by enumerating all C(m, k) selections. The greedy picks,
// k times in a row, the set with the largest number of not-yet-covered elements; ties are broken
// towards the smallest index, which is what makes every run reproducible.
//
// Guarantee. With Δ_i the i-th greedy marginal gain, submodularity of the coverage function gives
// OPT <= f(S_i) + k Δ_i, hence OPT - f(S_{i+1}) <= (1 - 1/k)(OPT - f(S_i)) and finally
// f(greedy) >= (1 - (1 - 1/k)^k) OPT >= (1 - 1/e) OPT. Both bounds are theorems; the checks verify
// them numerically on 400 deterministic instances (m = 12 or 14 sets, k = 2..5) against the
// enumerated optimum, and record the slack. Attainment is a different matter: the k = 2 instance with
// universe {0,1,2,3} and sets {0,2}, {0,1}, {2,3} attains the bound exactly (greedy 3 against optimum
// 4, ratio 0.75 = 1 - (1/2)^2), while for k >= 3 the bound is only approached: on the explicit
// three-row/"geometric" instance below greedy covers 21 of the optimal 27 elements, a ratio of 0.7778
// against the bound 0.7037, so no finite-k tightness is claimed - only the k = 2 case is tight.
//
// Further documented checks: coverage computed by the popcount of the OR equals an independent
// HashSet-based union for every mask of the small instances; the greedy is exact for k = 1 (largest
// set), for k = m (the whole union) and whenever the optimal sets are pairwise disjoint; the weighted
// variant obeys the same bounds on one weighted instance for k = 2..4 (brute-force optimum, measured
// slack 0.25 above the k = 2 bound); the greedy never
// exceeds the optimum; and every greedy run is reproducible. Randomness is a fixed-seed 64-bit
// splitmix64 stream, documented because the 16-bit LCG of the earlier waves cycles too fast here.
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
}

internal static class Coverage
{
    internal sealed class Instance
    {
        internal int Universe;
        internal int[] Sets = Array.Empty<int>();
        internal double[] Weights = Array.Empty<double>();
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

    internal static double[] RandomWeights(int universe, ulong seed)
    {
        var weights = new double[universe];
        ulong state = seed;
        for (int e = 0; e < universe; e++) weights[e] = 0.5 + Rng.NextUnit(ref state);
        return weights;
    }

    /** Unweighted objective: number of covered universe elements. */
    internal static double Value(Instance inst, int selection)
    {
        return BitOperations.PopCount((uint)Union(inst, selection));
    }

    internal static int Union(Instance inst, int selection)
    {
        int union = 0;
        for (int s = 0; s < inst.Sets.Length; s++)
        {
            if ((selection & (1 << s)) != 0) union |= inst.Sets[s];
        }
        return union;
    }

    /** Weighted objective: total weight of the covered elements. */
    internal static double WeightedValue(Instance inst, int selection)
    {
        int union = Union(inst, selection);
        double total = 0.0;
        for (int e = 0; e < inst.Universe; e++)
        {
            if ((union & (1 << e)) != 0) total += inst.Weights[e];
        }
        return total;
    }

    /** Independent union computation through HashSet, used to validate the bitmask version. */
    internal static int UnionBySets(Instance inst, int selection)
    {
        var covered = new HashSet<int>();
        for (int s = 0; s < inst.Sets.Length; s++)
        {
            if ((selection & (1 << s)) == 0) continue;
            for (int e = 0; e < inst.Universe; e++)
            {
                if ((inst.Sets[s] & (1 << e)) != 0) covered.Add(e);
            }
        }
        int mask = 0;
        foreach (int e in covered) mask |= 1 << e;
        return mask;
    }

    internal static Instance TightTwoSet()
    {
        return new Instance { Universe = 4, Sets = new[] { 0b0101, 0b0011, 0b1100 } };
    }

    /**
     * Three "rows" of nine universe elements; greedy's best first choice is the diagonal set D1 (it
     * ties with a whole row at nine elements and has the smaller index), after which a row adds six
     * and then another row adds six more, giving 21 covered elements against the optimum 27 of the
     * three disjoint rows - a ratio of 0.7778, above the finite-k bound 0.7037, so the instance is
     * bad for greedy but not tight.
     */
    internal static Instance Geometric()
    {
        int[] sets = new int[6];
        int d1 = 0;
        for (int row = 0; row < 3; row++)
        {
            for (int j = 0; j < 3; j++) d1 |= 1 << (row * 9 + j);
        }
        sets[0] = d1;
        for (int row = 0; row < 3; row++)
        {
            int full = 0;
            for (int j = 0; j < 9; j++) full |= 1 << (row * 9 + j);
            sets[row + 1] = full;
        }
        sets[4] = (1 << 3) | (1 << 12) | (1 << 21);
        sets[5] = (1 << 4) | (1 << 13) | (1 << 22);
        return new Instance { Universe = 27, Sets = sets };
    }
}

internal static class Greedy
{
    /** k times the largest uncovered-elements gain; ties keep the smallest index. */
    internal static int Maximise(Func<int, double> f, int setCount, int k)
    {
        int chosen = 0;
        for (int step = 0; step < k; step++)
        {
            double current = f(chosen);
            double bestGain = 0.0;
            int best = -1;
            for (int s = 0; s < setCount; s++)
            {
                if ((chosen & (1 << s)) != 0) continue;
                double gain = f(chosen | (1 << s)) - current;
                if (gain > bestGain + 1e-12)
                {
                    bestGain = gain;
                    best = s;
                }
            }
            if (best < 0) break;
            chosen |= 1 << best;
        }
        return chosen;
    }

    /** Exact optimum over all C(setCount, k) selections. */
    internal static int BruteForce(Func<int, double> f, int setCount, int k, out double value)
    {
        int bestMask = 0;
        value = f(0);
        for (int mask = 1; mask < (1 << setCount); mask++)
        {
            if (BitOperations.PopCount((uint)mask) > k) continue;
            double v = f(mask);
            if (v > value + 1e-12)
            {
                value = v;
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
        // The bitmask union agrees with an independent HashSet union on every mask.
        Coverage.Instance small = Coverage.Random(8, 16, 0.35, 569001UL);
        for (int mask = 0; mask < (1 << 8); mask++)
        {
            Check(Coverage.Union(small, mask) == Coverage.UnionBySets(small, mask),
                    "the bitmask union matches the HashSet union at mask " + mask);
        }

        // Greedy versus the enumerated optimum over 400 instances and k = 2..5.
        double worstSlack = double.PositiveInfinity;
        int instances = 0;
        int tightCount = 0;
        for (int seed = 0; seed < 100; seed++)
        {
            Coverage.Instance inst = Coverage.Random(seed % 2 == 0 ? 12 : 14, 26, 0.22, (ulong)(569100 + seed));
            Func<int, double> f = m => Coverage.Value(inst, m);
            for (int k = 2; k <= 5; k++)
            {
                int greedyMask = Greedy.Maximise(f, inst.Sets.Length, k);
                int optMask = Greedy.BruteForce(f, inst.Sets.Length, k, out double opt);
                double greedy = f(greedyMask);
                Check(greedy <= opt + 1e-12, "the greedy never beats the optimum (seed " + seed + ", k = " + k + ")");
                Check(BitOperations.PopCount((uint)greedyMask) <= k, "the greedy respects the budget");
                double ratio = opt <= 1e-12 ? 1.0 : greedy / opt;
                Check(ratio >= Greedy.Bound(k) - 1e-9, "the (1 - (1 - 1/k)^k) bound holds (seed " + seed
                        + ", k = " + k + ", ratio " + ratio + ", bound " + Greedy.Bound(k) + ")");
                Check(ratio >= 1.0 - 1.0 / Math.E - 1e-9, "the 1 - 1/e bound holds (seed " + seed + ", k = " + k + ")");
                worstSlack = Math.Min(worstSlack, ratio - Greedy.Bound(k));
                if (Math.Abs(greedy - opt) <= 1e-12) tightCount++;
                instances++;
            }
        }
        Check(instances == 400, "four hundred (instance, k) pairs were compared");
        Check(worstSlack >= -1e-9, "no instance came close to violating the bound (" + worstSlack + ")");
        Check(tightCount > 200, "the greedy is optimal on most random instances (" + tightCount + " of 400)");

        // Boundary cases: k = 1, k = m, and disjoint optimal sets.
        Coverage.Instance boundary = Coverage.Random(10, 20, 0.3, 569777UL);
        Func<int, double> fb = m => Coverage.Value(boundary, m);
        int largest = -1;
        double largestSize = -1.0;
        for (int s = 0; s < boundary.Sets.Length; s++)
        {
            double size = BitOperations.PopCount((uint)boundary.Sets[s]);
            if (size > largestSize + 1e-12)
            {
                largestSize = size;
                largest = s;
            }
        }
        Check(Greedy.Maximise(fb, boundary.Sets.Length, 1) == (1 << largest), "the greedy picks the largest set for k = 1");
        int allSets = (1 << boundary.Sets.Length) - 1;
        int unionAll = Coverage.Union(boundary, allSets);
        int fullSelection = Greedy.Maximise(fb, boundary.Sets.Length, boundary.Sets.Length);
        Check(Coverage.Union(boundary, fullSelection) == unionAll,
                "with k = m the greedy covers the whole union (it stops once no set adds a new element)");
        Check(Math.Abs(fb(fullSelection) - BitOperations.PopCount((uint)unionAll)) <= 1e-12,
                "the full-budget greedy value equals the union size");
        Check(Math.Abs(fb(allSets) - BitOperations.PopCount((uint)unionAll)) <= 1e-12,
                "the full selection covers the whole union");
        var disjoint = new Coverage.Instance
        {
            Universe = 12,
            Sets = new[] { 0b000000001111, 0b000011110000, 0b111100000000 },
        };
        Func<int, double> fd = m => Coverage.Value(disjoint, m);
        int disjointGreedy = Greedy.Maximise(fd, 3, 3);
        Greedy.BruteForce(fd, 3, 3, out double disjointOpt);
        Check(Math.Abs(fd(disjointGreedy) - disjointOpt) <= 1e-12,
                "three pairwise disjoint sets make the greedy optimal (" + fd(disjointGreedy) + " vs " + disjointOpt + ")");
        Check(disjointOpt == 12.0, "three disjoint sets cover all twelve elements");

        // Weighted coverage.
        Coverage.Instance weighted = Coverage.Random(12, 20, 0.25, 569555UL);
        weighted.Weights = Coverage.RandomWeights(20, 569556UL);
        Func<int, double> fw = m => Coverage.WeightedValue(weighted, m);
        double worstWeightedSlack = double.PositiveInfinity;
        for (int k = 2; k <= 4; k++)
        {
            int greedyMask = Greedy.Maximise(fw, weighted.Sets.Length, k);
            Greedy.BruteForce(fw, weighted.Sets.Length, k, out double opt);
            double greedy = fw(greedyMask);
            Check(greedy <= opt + 1e-12, "the weighted greedy never beats the optimum");
            Check(greedy >= Greedy.Bound(k) * opt - 1e-9, "the weighted greedy obeys the bound (k = " + k + ")");
            worstWeightedSlack = Math.Min(worstWeightedSlack, greedy / opt - Greedy.Bound(k));
        }
        Check(worstWeightedSlack >= -1e-9, "the weighted slack is non-negative (" + worstWeightedSlack + ")");
        Check(Coverage.WeightedValue(weighted, 0) == 0.0, "an empty selection covers no weight");

        // Exact attainment at k = 2.
        Coverage.Instance tight = Coverage.TightTwoSet();
        Func<int, double> ft = m => Coverage.Value(tight, m);
        int tightGreedy = Greedy.Maximise(ft, 3, 2);
        Greedy.BruteForce(ft, 3, 2, out double tightOpt);
        Check(Math.Abs(ft(tightGreedy) - 3.0) <= 1e-12, "the tight instance gives greedy 3");
        Check(Math.Abs(tightOpt - 4.0) <= 1e-12, "the tight instance has optimum 4");
        Check(Math.Abs(ft(tightGreedy) / tightOpt - Greedy.Bound(2)) <= 1e-12,
                "the k = 2 bound is attained exactly");

        // No finite-k tightness for k = 3: the geometric instance beats the bound.
        Coverage.Instance geometric = Coverage.Geometric();
        Func<int, double> fgeo = m => Coverage.Value(geometric, m);
        int geometricGreedy = Greedy.Maximise(fgeo, geometric.Sets.Length, 3);
        Greedy.BruteForce(fgeo, geometric.Sets.Length, 3, out double geometricOpt);
        Check(Math.Abs(fgeo(geometricGreedy) - 21.0) <= 1e-12, "the geometric instance gives greedy 21");
        Check(Math.Abs(geometricOpt - 27.0) <= 1e-12, "the geometric instance has optimum 27");
        Check(fgeo(geometricGreedy) / geometricOpt > Greedy.Bound(3) + 0.05,
                "the geometric instance beats the k = 3 bound by a wide margin, so the bound is not attained");
        Check(Math.Abs(Greedy.Bound(3) - 0.7037037037) <= 1e-9, "the k = 3 bound is 0.7037037037");
        Check(Greedy.Bound(1) == 1.0, "the k = 1 bound is 1");
        Check(Math.Abs(Greedy.Bound(100) - 0.6339676590) <= 1e-9, "the k = 100 bound is 0.6339676590");
        Check(Math.Abs(Greedy.Bound(100) - (1.0 - 1.0 / Math.E)) <= 0.002, "large k approaches 1 - 1/e");
        double previous = 2.0;
        for (int k = 1; k <= 60; k++)
        {
            double bound = Greedy.Bound(k);
            Check(bound <= previous + 1e-15, "the bound decreases towards the limit (k = " + k + ")");
            Check(bound >= 1.0 - 1.0 / Math.E - 1e-15, "the bound stays at or above 1 - 1/e (k = " + k + ")");
            previous = bound;
        }

        // Determinism.
        Check(Greedy.Maximise(ft, 3, 2) == tightGreedy, "the greedy is reproducible");
        Check(Greedy.Maximise(fd, 3, 3) == disjointGreedy, "the disjoint selection is reproducible");
        Check(Greedy.Maximise(fgeo, geometric.Sets.Length, 3) == geometricGreedy,
                "the geometric-instance selection is reproducible");

        Console.WriteLine("PASS");
    }
}
