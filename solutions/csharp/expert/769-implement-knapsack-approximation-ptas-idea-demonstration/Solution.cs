// Task 769 - Implement knapsack approximation PTAS idea demonstration. (Tier: Expert, CTCI Ch 8 - Recursion and Dynamic Programming)
//
// 0/1 knapsack: choose a subset of items with total weight <= capacity that maximises total profit.
// The file builds the exact dynamic program as the oracle and the PTAS (Sahni / Vazirani style) on
// top of it:
//
//   * ExactDp          - O(n * capacity) table over weights, with reconstruction of the chosen set.
//   * SubsetBruteForce - 2^n enumeration, the independent oracle for tiny instances.
//   * DensityGreedy    - sort by profit/weight, take greedily; kept only to show that pure greedy is
//                        NOT optimal (the file proves that with a hand-built counterexample).
//   * Ptas(value, eps) - profit scaling: m = floor(eps * pMax / n), p_bar_i = p_i / m, and then an
//                        exact DP over the rounded profits (state = rounded profit, value = minimum
//                        weight), whose table size is sum(p_bar_i) <= n^2/eps. Rounding every profit
//                        down costs at most n*m = eps*pMax <= eps*OPT, because the most profitable
//                        item always fits in the instances built here, so the reconstructed set is
//                        feasible and worth at least (1 - eps) * OPT. The result is combined with
//                        the most profitable single fitting item, the standard guard for instances
//                        in which the rounded DP would return nothing.
//
// Verified: the exact DP equals subset enumeration on tiny instances; every PTAS set is feasible and
// never beats OPT; the documented (1 - eps) * OPT bound and the tighter OPT - eps*pMax accounting
// bound hold for eps in {1.0, 0.5, 0.25} over a randomized suite; a rounding step of m = 1 collapses
// the PTAS back to the exact optimum; greedy-plus-best-single-item is a 2-approximation; and greedy
// by density is strictly suboptimal on a hand-built counterexample, so the approximation claims are
// not vacuous.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Instance
{
    internal readonly int[] Profit;
    internal readonly int[] Weight;
    internal readonly int Capacity;

    internal Instance(int[] profit, int[] weight, int capacity)
    {
        Profit = profit;
        Weight = weight;
        Capacity = capacity;
    }

    internal int N => Profit.Length;

    internal int TotalWeight(IReadOnlyList<int> take)
    {
        int total = 0;
        foreach (int i in take) total += Weight[i];
        return total;
    }

    internal int TotalProfit(IReadOnlyList<int> take)
    {
        int total = 0;
        foreach (int i in take) total += Profit[i];
        return total;
    }

    internal int MaxProfit()
    {
        int best = 0;
        foreach (int p in Profit) if (p > best) best = p;
        return best;
    }
}

internal static class Knapsack
{
    internal const int Unreachable = int.MaxValue;

    /// <summary>Exact 0/1 DP over capacity; returns the optimal profit and one optimal item set.</summary>
    internal static int ExactDp(Instance inst, out List<int> take)
    {
        int n = inst.N, cap = inst.Capacity;
        take = new List<int>();
        if (n == 0 || cap <= 0) return 0;
        var best = new int[cap + 1];
        var choice = new bool[n, cap + 1];
        for (int i = 0; i < n; i++)
        {
            for (int c = cap; c >= inst.Weight[i]; c--)
            {
                int candidate = best[c - inst.Weight[i]] + inst.Profit[i];
                if (candidate > best[c]) { best[c] = candidate; choice[i, c] = true; }
            }
        }
        int widest = 0;
        for (int c = 1; c <= cap; c++) if (best[c] > best[widest]) widest = c;
        int value = best[widest];
        for (int i = n - 1; i >= 0; i--)
        {
            if (widest == 0) break;
            if (choice[i, widest]) { take.Add(i); widest -= inst.Weight[i]; }
        }
        take.Reverse();
        return value;
    }

    /// <summary>Oracle: every subset is enumerated explicitly (n &lt;= 20 only).</summary>
    internal static int SubsetBruteForce(Instance inst, out int subsetCount)
    {
        int n = inst.N, cap = inst.Capacity;
        if (n > 20) throw new ArgumentException("enumeration is only the oracle for n <= 20");
        int best = 0;
        subsetCount = 1 << n;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int weight = 0, profit = 0;
            for (int i = 0; i < n; i++)
            {
                if ((mask & (1 << i)) == 0) continue;
                weight += inst.Weight[i];
                profit += inst.Profit[i];
            }
            if (weight <= cap && profit > best) best = profit;
        }
        return best;
    }

    /// <summary>Greedy by profit/weight ratio; not optimal, used as a contrast.</summary>
    internal static int DensityGreedy(Instance inst, out List<int> take)
    {
        int n = inst.N;
        var order = new List<int>();
        for (int i = 0; i < n; i++) if (inst.Weight[i] <= inst.Capacity) order.Add(i);
        order.Sort((a, b) =>
        {
            long left = (long)inst.Profit[a] * inst.Weight[b];
            long right = (long)inst.Profit[b] * inst.Weight[a];
            int c = right.CompareTo(left);                       // larger ratio first
            return c != 0 ? c : a.CompareTo(b);
        });
        take = new List<int>();
        int remaining = inst.Capacity, value = 0;
        foreach (int i in order)
        {
            if (inst.Weight[i] > remaining) continue;
            remaining -= inst.Weight[i];
            value += inst.Profit[i];
            take.Add(i);
        }
        return value;
    }

    /// <summary>Single most profitable item that fits: the guard the PTAS is combined with.</summary>
    internal static int BestSingleItem(Instance inst, out List<int> take)
    {
        take = new List<int>();
        int best = 0, bestIndex = -1;
        for (int i = 0; i < inst.N; i++)
        {
            if (inst.Weight[i] <= inst.Capacity && inst.Profit[i] > best) { best = inst.Profit[i]; bestIndex = i; }
        }
        if (bestIndex >= 0) take.Add(bestIndex);
        return best;
    }

    internal sealed class PtasResult
    {
        internal int Value;
        internal List<int> Take = new List<int>();
        internal int RoundingStep;      // m
        internal int ScaledObjective;   // optimum of the rounded DP, in rounded profit units
        internal int ScaledProfit;      // total rounded profit, i.e. the DP table bound n*pMax/m
        internal int DpOnlyValue;       // value of the rounded DP set without the single-item guard
        internal bool UsedGuard;
    }

    /// <summary>Profit-scaling PTAS: round profits down to multiples of m = eps*pMax/n, DP on them.</summary>
    internal static PtasResult Ptas(Instance inst, double eps)
    {
        if (eps <= 0 || eps > 1) throw new ArgumentException("eps must lie in (0, 1]");
        var result = new PtasResult();
        int n = inst.N;
        int pMax = inst.MaxProfit();
        double mReal = n == 0 ? 1.0 : eps * pMax / n;
        int m = Math.Max(1, (int)Math.Floor(mReal));             // integer arithmetic from here on
        result.RoundingStep = m;

        List<int> guardTake;
        result.Value = BestSingleItem(inst, out guardTake);
        result.UsedGuard = guardTake.Count > 0;
        result.Take = guardTake;
        if (n == 0 || inst.Capacity <= 0) return result;

        var scaled = new int[n];
        int scaledTotal = 0;
        for (int i = 0; i < n; i++)
        {
            scaled[i] = inst.Profit[i] / m;                      // floor division, profits are non-negative
            scaledTotal += scaled[i];
        }
        result.ScaledProfit = scaledTotal;

        // DP: lightest way to reach each exact rounded profit (0/1, one item per row).
        var lightest = new int[scaledTotal + 1];
        var choice = new bool[n, scaledTotal + 1];
        for (int s = 1; s <= scaledTotal; s++) lightest[s] = Unreachable;
        for (int i = 0; i < n; i++)
        {
            for (int s = scaledTotal; s >= scaled[i]; s--)
            {
                if (scaled[i] == 0 || lightest[s - scaled[i]] == Unreachable) continue;
                int candidate = lightest[s - scaled[i]] + inst.Weight[i];
                if (candidate < lightest[s]) { lightest[s] = candidate; choice[i, s] = true; }
            }
        }

        int bestObjective = 0;
        for (int s = scaledTotal; s >= 0; s--)
        {
            if (lightest[s] != Unreachable && lightest[s] <= inst.Capacity) { bestObjective = s; break; }
        }
        result.ScaledObjective = bestObjective;
        if (bestObjective > 0)
        {
            var picked = new List<int>();
            int left = bestObjective;
            for (int i = n - 1; i >= 0 && left > 0; i--)
            {
                if (choice[i, left]) { picked.Add(i); left -= scaled[i]; }
            }
            picked.Reverse();
            if (left != 0) throw new Exception("internal error: reconstruction did not reach the rounded optimum");
            result.DpOnlyValue = inst.TotalProfit(picked);
            if (inst.TotalWeight(picked) > inst.Capacity) throw new Exception("internal error: reconstructed set exceeds capacity");
            if (result.DpOnlyValue > result.Value) { result.Take = picked; result.Value = result.DpOnlyValue; }
        }
        return result;
    }

    internal static Instance RandomInstance(ref ulong state, int n, int maxWeight, int maxProfit, double tightness)
    {
        var profit = new int[n];
        var weight = new int[n];
        for (int i = 0; i < n; i++)
        {
            weight[i] = 1 + SplitMix64.NextInt(ref state, maxWeight);
            profit[i] = 1 + SplitMix64.NextInt(ref state, maxProfit);
        }
        int total = 0;
        foreach (int w in weight) total += w;
        int capacity = Math.Max(1, (int)(tightness * total));    // capacity as a fraction of the total weight
        // Every instance built here keeps the most profitable item fitting (capacity >= its weight),
        // which is exactly the assumption that turns eps*pMax into eps*OPT in the PTAS bound.
        int richest = 0;
        for (int i = 1; i < n; i++) if (profit[i] > profit[richest]) richest = i;
        if (weight[richest] > capacity) capacity = weight[richest];
        return new Instance(profit, weight, capacity);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // ---- exact DP versus subset enumeration on tiny instances -------------------------------
        {
            ulong state = 0x5EED999900001234UL;
            bool allAgree = true, feasible = true;
            string detail = "ok";
            for (int trial = 0; trial < 40; trial++)
            {
                int n = 1 + SplitMix64.NextInt(ref state, 10);
                Instance inst = Knapsack.RandomInstance(ref state, n, 12, 40, 0.6);
                int oracle = Knapsack.SubsetBruteForce(inst, out _);
                List<int> take;
                int dp = Knapsack.ExactDp(inst, out take);
                if (dp != oracle) { allAgree = false; detail = "trial " + trial + ": dp=" + dp + " brute=" + oracle; }
                if (inst.TotalWeight(take) > inst.Capacity) feasible = false;
            }
            Check(allAgree, "the weight DP equals subset enumeration on 40 tiny instances: " + detail);
            Check(feasible, "every reconstructed DP set respects the capacity");
        }

        // ---- boundary inputs ---------------------------------------------------------------------
        {
            List<int> take;
            Check(Knapsack.ExactDp(new Instance(new int[0], new int[0], 10), out take) == 0, "no items means no profit");
            Check(take.Count == 0, "the empty instance reconstructs an empty set");
            var zeroCap = new Instance(new[] { 5, 6 }, new[] { 1, 2 }, 0);
            Check(Knapsack.ExactDp(zeroCap, out take) == 0, "capacity zero admits nothing");
            Check(Knapsack.Ptas(zeroCap, 0.5).Value == 0, "the PTAS reports 0 when the capacity is zero");
            var tooHeavy = new Instance(new[] { 100, 200 }, new[] { 11, 12 }, 10);
            Check(Knapsack.ExactDp(tooHeavy, out take) == 0, "items heavier than the capacity are skipped");
            Check(Knapsack.Ptas(tooHeavy, 1.0).Value == 0, "the PTAS also refuses to exceed the capacity");
            var single = new Instance(new[] { 7 }, new[] { 3 }, 3);
            Check(Knapsack.ExactDp(single, out take) == 7, "a single fitting item is taken");
            Check(Knapsack.BestSingleItem(single, out take) == 7 && take.Count == 1, "the guard finds the single item");
            Check(Knapsack.Ptas(single, 1.0).Value == 7, "the PTAS takes the only item that fits");
            var singleTooHeavy = new Instance(new[] { 7 }, new[] { 4 }, 3);
            Check(Knapsack.BestSingleItem(singleTooHeavy, out take) == 0 && take.Count == 0, "the guard refuses an oversized item");
            Check(Knapsack.Ptas(singleTooHeavy, 0.25).Value == 0, "the PTAS does not force an oversized item in");

            bool threw = false;
            try { Knapsack.Ptas(single, 0.0); } catch (ArgumentException) { threw = true; }
            Check(threw, "eps = 0 is rejected instead of silently degrading");
            bool threwEps = false;
            try { Knapsack.Ptas(single, 1.5); } catch (ArgumentException) { threwEps = true; }
            Check(threwEps, "eps > 1 is rejected");
        }

        // ---- greedy by density is strictly suboptimal (negative check) ---------------------------
        {
            // capacity 10: the two 4/4 items beat the 5/6 item, but the 5/6 item has the better ratio
            var inst = new Instance(new[] { 5, 4, 4 }, new[] { 6, 4, 4 }, 10);
            List<int> take;
            int greedy = Knapsack.DensityGreedy(inst, out take);
            int oracle = Knapsack.SubsetBruteForce(inst, out _);
            int exact = Knapsack.ExactDp(inst, out take);
            Check(oracle == 9, "the counterexample optimum is 9 (the two ratio-worse items), got " + oracle);
            Check(greedy == 8, "greedy by density only collects 8, got " + greedy);
            Check(greedy < exact, "greedy by density is strictly suboptimal (" + greedy + " < " + exact + ")");
            Check(Knapsack.Ptas(inst, 0.5).Value == exact, "the PTAS still finds the optimum on the counterexample");
        }

        // ---- the approximation guarantees on a randomized suite ---------------------------------
        {
            const int Trials = 30;
            var epsilons = new double[] { 1.0, 0.5, 0.25 };
            ulong state = 0xC0FFEE5EED000001UL;
            var instances = new Instance[Trials];
            var optima = new int[Trials];
            for (int t = 0; t < Trials; t++)
            {
                instances[t] = Knapsack.RandomInstance(ref state, 40, 400, 900, 0.45);
                List<int> optimalSet;
                optima[t] = Knapsack.ExactDp(instances[t], out optimalSet);
            }

            foreach (double eps in epsilons)
            {
                double worstRatio = double.MaxValue, worstAccounting = 0;
                bool feasible = true, withinOptimum = true, guaranteed = true, accounting = true;
                for (int t = 0; t < Trials; t++)
                {
                    Instance inst = instances[t];
                    int largest = inst.MaxProfit();
                    Knapsack.PtasResult r = Knapsack.Ptas(inst, eps);
                    if (inst.TotalWeight(r.Take) > inst.Capacity) feasible = false;
                    if (r.Value > optima[t]) withinOptimum = false;
                    double ratio = optima[t] == 0 ? 1.0 : (double)r.Value / optima[t];
                    if (ratio < worstRatio) worstRatio = ratio;
                    if (ratio < 1.0 - eps) guaranteed = false;
                    double loss = optima[t] - r.Value;
                    if (loss > eps * largest) accounting = false;
                    double scaledLoss = eps * largest == 0 ? 0 : loss / (eps * largest);
                    if (scaledLoss > worstAccounting) worstAccounting = scaledLoss;
                }
                Check(feasible, "eps=" + eps + ": every PTAS set respects the capacity");
                Check(withinOptimum, "eps=" + eps + ": the PTAS never reports more than the exact optimum");
                Check(guaranteed, "eps=" + eps + ": value >= (1-eps)*OPT on all " + Trials + " instances (worst ratio " + worstRatio.ToString("F4") + ")");
                Check(accounting, "eps=" + eps + ": the rounding loss stays within eps*pMax on all instances");
                Console.WriteLine("eps=" + eps.ToString("F2") + " worstValue/OPT=" + worstRatio.ToString("F4")
                    + " worst (OPT-value)/(eps*pMax)=" + worstAccounting.ToString("F4"));
            }

            // eps small enough that the rounding step collapses to m = 1: the PTAS must be exact
            double tinyEps = 20.0 / instances[0].MaxProfit() / 2.0;
            bool exactWhenFine = true;
            int collapsed = 0;
            for (int t = 0; t < Trials; t++)
            {
                Knapsack.PtasResult r = Knapsack.Ptas(instances[t], tinyEps);
                if (r.RoundingStep == 1) collapsed++;
                if (r.Value != optima[t]) exactWhenFine = false;
            }
            Check(collapsed == Trials, "with eps=" + tinyEps.ToString("F5") + " the rounding step collapses to m = 1");
            Check(exactWhenFine, "a rounding step of m = 1 makes the profit-scaled PTAS exact on all instances");
        }

        // ---- greedy-or-best-single-item is a 2-approximation -------------------------------------
        {
            const int Trials = 60;
            ulong state = 0xBADC0DE5EED00002UL;
            double worst = double.MaxValue;
            bool twoApprox = true;
            int greedyWorse = 0;
            for (int t = 0; t < Trials; t++)
            {
                Instance inst = Knapsack.RandomInstance(ref state, 40, 400, 900, 0.45);
                List<int> take;
                int oracle = Knapsack.ExactDp(inst, out take);
                List<int> greedySet, guardSet;
                int greedy = Knapsack.DensityGreedy(inst, out greedySet);
                int guard = Knapsack.BestSingleItem(inst, out guardSet);
                int combined = Math.Max(greedy, guard);
                if (oracle > 0)
                {
                    double ratio = (double)combined / oracle;
                    if (ratio < worst) worst = ratio;
                    if (ratio < 0.5) twoApprox = false;
                }
                if (greedy < oracle) greedyWorse++;
            }
            Check(twoApprox, "greedy-or-best-single-item is a 2-approximation on all 60 instances (worst ratio " + worst.ToString("F4") + ")");
            Check(greedyWorse > 0, "density greedy alone is suboptimal on " + greedyWorse + " of the 60 random instances");
            Console.WriteLine("greedy+bestSingle worst value/OPT=" + worst.ToString("F4") + " over 60 instances, greedy suboptimal " + greedyWorse + " times");
        }

        // ---- skewed profits: coarse rounding really does lose optimum value ----------------------
        {
            // one 1000/1 item and 39 items worth 1 each in a capacity of 39: rounding by
            // m = floor(0.5 * 1000 / 40) = 12 wipes the small items out of the DP table
            var profit = new int[40];
            var weight = new int[40];
            profit[0] = 1000;
            weight[0] = 1;
            for (int i = 1; i < 40; i++) { profit[i] = 1; weight[i] = 1; }
            var skewed = new Instance(profit, weight, 39);
            List<int> take;
            int opt = Knapsack.ExactDp(skewed, out take);
            Knapsack.PtasResult r = Knapsack.Ptas(skewed, 0.5);
            Check(opt == 1038, "the skewed instance optimum collects the big item plus 38 small ones, got " + opt);
            Check(r.RoundingStep == 12, "the skewed instance rounds profits to multiples of 12, got " + r.RoundingStep);
            Check(r.Value < opt, "coarse rounding loses value on the skewed instance (" + r.Value + " < " + opt + ")");
            Check(r.Value >= (1 - 0.5) * opt, "the skewed instance still meets the (1-eps)*OPT bound");
            Check(opt - r.Value <= 0.5 * skewed.MaxProfit(), "the skewed losses stay inside eps*pMax");
            Console.WriteLine("skewed instance: m=" + r.RoundingStep + " ptas=" + r.Value + " opt=" + opt
                + " dpOnly=" + r.DpOnlyValue + " guard=" + Knapsack.BestSingleItem(skewed, out take));

            // the same story on 20 fresh skewed instances
            int nonExact = 0;
            bool boundHolds = true;
            for (int t = 0; t < 20; t++)
            {
                var p = new int[40];
                var w = new int[40];
                p[0] = 1000 + t * 37;
                w[0] = 1;
                for (int i = 1; i < 40; i++) { p[i] = 1 + (i % 3); w[i] = 1; }
                var inst = new Instance(p, w, 1 + t * 2);
                List<int> chosen;
                int exactOpt = Knapsack.ExactDp(inst, out chosen);
                Knapsack.PtasResult approx = Knapsack.Ptas(inst, 0.5);
                if (approx.Value < exactOpt) nonExact++;
                if (approx.Value < 0.5 * exactOpt) boundHolds = false;
            }
            Check(boundHolds, "eps=0.5 never violates the (1-eps)*OPT bound on the 20 skewed instances");
            Check(nonExact > 0, "coarse rounding really does lose optimum value on " + nonExact + " of the 20 skewed instances");
            Console.WriteLine("eps=0.5 on 20 skewed instances: suboptimal=" + nonExact);
        }

        Console.WriteLine("PASS");
    }
}
