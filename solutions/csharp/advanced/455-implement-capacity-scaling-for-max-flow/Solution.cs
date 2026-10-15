// Task 455 - Implement capacity scaling for max flow. (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

/// <summary>Arc-list flow network: to/cap arrays, reverse arc index is idx ^ 1.</summary>
internal class MaxFlow
{
    private readonly int n;
    private readonly List<int> to = new List<int>();
    private readonly List<int> initial = new List<int>();
    private readonly List<int> cap = new List<int>();
    private readonly List<List<int>> head;

    public MaxFlow(int n)
    {
        this.n = n;
        head = new List<List<int>>();
        for (int i = 0; i < n; i++) head.Add(new List<int>());
    }

    public int AddEdge(int u, int v, int capacity)
    {
        int idx = to.Count;
        to.Add(v);
        initial.Add(capacity);
        cap.Add(capacity);
        head[u].Add(idx);
        to.Add(u);
        initial.Add(0);
        cap.Add(0);
        head[v].Add(idx + 1);
        return idx;
    }

    private int Augment(int[] parentEdge, int s, int t)
    {
        int aug = int.MaxValue / 4;
        for (int v = t; v != s; v = to[parentEdge[v] ^ 1]) aug = Math.Min(aug, cap[parentEdge[v]]);
        for (int v = t; v != s; v = to[parentEdge[v] ^ 1])
        {
            int idx = parentEdge[v];
            cap[idx] -= aug;
            cap[idx ^ 1] += aug;
        }
        return aug;
    }

    private int[] FindPath(int s, int t, int minCapacity)
    {
        var parentEdge = new int[n];
        for (int i = 0; i < n; i++) parentEdge[i] = -1;
        parentEdge[s] = -2;
        var q = new Queue<int>();
        q.Enqueue(s);
        while (q.Count > 0)
        {
            int u = q.Dequeue();
            foreach (int idx in head[u])
            {
                int v = to[idx];
                if (parentEdge[v] == -1 && cap[idx] >= minCapacity)
                {
                    parentEdge[v] = idx;
                    q.Enqueue(v);
                }
            }
        }
        return parentEdge[t] != -1 ? parentEdge : null;
    }

    public int EdmondsKarp(int s, int t)
    {
        int flow = 0;
        while (true)
        {
            var parentEdge = FindPath(s, t, 1);
            if (parentEdge == null) return flow;
            flow += Augment(parentEdge, s, t);
        }
    }

    /// <summary>Returns (value, deltas, perPhase, amounts).</summary>
    public (int value, int[] deltas, int[] perPhase, int[] amounts) CapacityScaling(int s, int t)
    {
        int maxCapacity = 0;
        for (int idx = 0; idx < to.Count; idx += 2) maxCapacity = Math.Max(maxCapacity, initial[idx]);
        int delta = 0;
        if (maxCapacity >= 1)
        {
            delta = 1;
            while (delta * 2 <= maxCapacity) delta *= 2;
        }
        var deltas = new List<int>();
        var perPhase = new List<int>();
        var amounts = new List<int>();
        int flow = 0;
        while (delta >= 1)
        {
            deltas.Add(delta);
            int count = 0;
            while (true)
            {
                var parentEdge = FindPath(s, t, delta);
                if (parentEdge == null) break;
                int amount = Augment(parentEdge, s, t);
                if (amount < delta) throw new Exception("FAILED: augmentation at least delta");
                amounts.Add(amount);
                flow += amount;
                count++;
            }
            perPhase.Add(count);
            delta /= 2;
        }
        return (flow, deltas.ToArray(), perPhase.ToArray(), amounts.ToArray());
    }
}

internal static class ScalingSupport
{
    public static MaxFlow BuildNetwork(int n, List<int[]> edges)
    {
        var network = new MaxFlow(n);
        foreach (var e in edges) network.AddEdge(e[0], e[1], e[2]);
        return network;
    }

    public static List<(int n, List<int[]> edges)> LcgFlowGraphs(int count, int seed)
    {
        var graphs = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 4 + seed % 4;
            var edges = new List<int[]>();
            for (int node = 0; node < n - 1; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                edges.Add(new[] { node, node + 1, 1 + seed % 500 });
            }
            for (int k = 0; k < n + 3; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.Add(new[] { u, v, 1 + seed % 500 });
            }
            graphs.Add((n, edges));
        }
        return graphs;
    }

    public static int PhaseCountFor(int maxCapacity)
    {
        int delta = 1;
        int count = 0;
        while (delta <= maxCapacity)
        {
            count++;
            delta *= 2;
        }
        return count;
    }
}

public static class Solution
{
    public static void Main()
    {
        var scaling = ScalingSupport.BuildNetwork(2, new List<int[]> { new[] { 0, 1, 6 } }).CapacityScaling(0, 1);
        Check(scaling.value == 6, "single value");
        Check(ArrayEquals(scaling.deltas, new[] { 4, 2, 1 }), "single deltas");
        Check(ArrayEquals(scaling.perPhase, new[] { 1, 0, 0 }), "single phases");
        Check(ArrayEquals(scaling.amounts, new[] { 6 }), "single amounts");

        scaling = ScalingSupport.BuildNetwork(2, new List<int[]> { new[] { 0, 1, 3 }, new[] { 0, 1, 4 } }).CapacityScaling(0, 1);
        Check(scaling.value == 7, "parallel value");
        Check(ArrayEquals(scaling.deltas, new[] { 4, 2, 1 }), "parallel deltas");
        Check(ArrayEquals(scaling.perPhase, new[] { 1, 1, 0 }), "parallel phases");
        Check(ArrayEquals(scaling.amounts, new[] { 4, 3 }), "parallel amounts");

        var chainEdges = new List<int[]> { new[] { 0, 1, 1000 }, new[] { 1, 2, 999 } };
        scaling = ScalingSupport.BuildNetwork(3, chainEdges).CapacityScaling(0, 2);
        Check(scaling.value == 999, "chain value");
        Check(scaling.deltas.Length == 10 && scaling.deltas[0] == 512 && scaling.deltas[9] == 1, "chain deltas");
        Check(Sum(scaling.amounts) == scaling.value, "chain amounts sum");
        Check(ScalingSupport.BuildNetwork(3, chainEdges).EdmondsKarp(0, 2) == 999, "chain oracle");

        scaling = ScalingSupport.BuildNetwork(2, new List<int[]> { new[] { 0, 1, 1 } }).CapacityScaling(0, 1);
        Check(scaling.value == 1, "tiny value");
        Check(ArrayEquals(scaling.deltas, new[] { 1 }) && ArrayEquals(scaling.amounts, new[] { 1 }), "tiny phases");

        scaling = ScalingSupport.BuildNetwork(2, new List<int[]>()).CapacityScaling(0, 1);
        Check(scaling.value == 0 && scaling.deltas.Length == 0 && scaling.amounts.Length == 0, "empty network");

        var clrs = new List<int[]>
        {
            new[] { 0, 1, 16 }, new[] { 0, 2, 13 }, new[] { 1, 2, 10 }, new[] { 2, 1, 4 },
            new[] { 1, 3, 12 }, new[] { 3, 2, 9 }, new[] { 2, 4, 14 }, new[] { 4, 3, 7 },
            new[] { 3, 5, 20 }, new[] { 4, 5, 4 },
        };
        scaling = ScalingSupport.BuildNetwork(6, clrs).CapacityScaling(0, 5);
        Check(scaling.value == 23, "CLRS value");
        Check(ArrayEquals(scaling.deltas, new[] { 16, 8, 4, 2, 1 }), "CLRS deltas");
        Check(Sum(scaling.amounts) == scaling.value, "CLRS amounts sum");
        Check(ScalingSupport.BuildNetwork(6, clrs).EdmondsKarp(0, 5) == 23, "CLRS oracle");

        int checkedCount = 0, latePhaseWork = 0, totalAugmentations = 0;
        foreach (int seed in new[] { 4551, 4552, 4553 })
        {
            foreach (var (n, edges) in ScalingSupport.LcgFlowGraphs(8, seed))
            {
                scaling = ScalingSupport.BuildNetwork(n, edges).CapacityScaling(0, n - 1);
                int maxCapacity = 0;
                foreach (var e in edges) maxCapacity = Math.Max(maxCapacity, e[2]);
                Check(scaling.deltas.Length == ScalingSupport.PhaseCountFor(maxCapacity), "phase count");
                Check(scaling.deltas[0] * 2 > maxCapacity && scaling.deltas[0] <= maxCapacity, "top delta");
                Check(Sum(scaling.amounts) == scaling.value, "amount sum");
                Check(Sum(scaling.perPhase) == scaling.amounts.Length, "phase total");
                totalAugmentations += scaling.amounts.Length;
                bool late = false;
                for (int i = 1; i < scaling.perPhase.Length; i++) late |= scaling.perPhase[i] > 0;
                if (late) latePhaseWork++;
                int expected = ScalingSupport.BuildNetwork(n, edges).EdmondsKarp(0, n - 1);
                Check(scaling.value == expected, "LCG value " + scaling.value + " vs " + expected);
                checkedCount++;
            }
        }
        Check(checkedCount == 24, "graph count");
        Check(totalAugmentations > 0, "augmentations happened");
        Check(latePhaseWork >= 1, "later phases did work");
        Console.WriteLine("PASS");
    }

    private static int Sum(int[] values)
    {
        int total = 0;
        foreach (int v in values) total += v;
        return total;
    }

    private static bool ArrayEquals(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
