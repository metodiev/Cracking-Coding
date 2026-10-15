// Task 454 - Implement push-relabel max-flow algorithm. (Tier: Advanced, CTCI Ch 4)
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

    public int FlowOn(int idx) => initial[idx] - cap[idx];

    /// <summary>FIFO preflow-push; returns (max flow, push count, relabel count).</summary>
    public (int value, int pushes, int relabels) PushRelabel(int s, int t)
    {
        var height = new int[n];
        var excess = new int[n];
        height[s] = n;
        var inQueue = new bool[n];
        var queue = new Queue<int>();
        foreach (int idx in head[s])
        {
            int amount = cap[idx];
            if (amount > 0)
            {
                int v = to[idx];
                cap[idx] -= amount;
                cap[idx ^ 1] += amount;
                excess[v] += amount;
                excess[s] -= amount;
            }
        }
        int pushes = 0, relabels = 0;
        for (int v = 0; v < n; v++)
        {
            if (v != s && v != t && excess[v] > 0)
            {
                queue.Enqueue(v);
                inQueue[v] = true;
            }
        }
        while (queue.Count > 0)
        {
            int u = queue.Dequeue();
            inQueue[u] = false;
            while (excess[u] > 0)
            {
                bool pushed = false;
                foreach (int idx in head[u])
                {
                    int v = to[idx];
                    if (cap[idx] > 0 && height[u] == height[v] + 1)
                    {
                        int amount = Math.Min(excess[u], cap[idx]);
                        cap[idx] -= amount;
                        cap[idx ^ 1] += amount;
                        excess[u] -= amount;
                        excess[v] += amount;
                        pushes++;
                        if (v != s && v != t && !inQueue[v] && excess[v] > 0)
                        {
                            queue.Enqueue(v);
                            inQueue[v] = true;
                        }
                        pushed = true;
                        break;
                    }
                }
                if (!pushed)
                {
                    int best = int.MaxValue / 4;
                    foreach (int idx in head[u])
                    {
                        if (cap[idx] > 0 && height[to[idx]] < best) best = height[to[idx]];
                    }
                    height[u] = best + 1;
                    relabels++;
                }
            }
        }
        return (excess[t], pushes, relabels);
    }

    public int EdmondsKarp(int s, int t)
    {
        int flow = 0;
        while (true)
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
                    if (parentEdge[v] == -1 && cap[idx] > 0)
                    {
                        parentEdge[v] = idx;
                        q.Enqueue(v);
                    }
                }
            }
            if (parentEdge[t] == -1) return flow;
            int aug = int.MaxValue / 4;
            for (int v = t; v != s; v = to[parentEdge[v] ^ 1]) aug = Math.Min(aug, cap[parentEdge[v]]);
            for (int v = t; v != s; v = to[parentEdge[v] ^ 1])
            {
                int idx = parentEdge[v];
                cap[idx] -= aug;
                cap[idx ^ 1] += aug;
            }
            flow += aug;
        }
    }
}

internal static class FlowOracle
{
    public static (MaxFlow network, int[] ids) BuildNetwork(int n, List<int[]> edges)
    {
        var network = new MaxFlow(n);
        var ids = new int[edges.Count];
        for (int i = 0; i < edges.Count; i++) ids[i] = network.AddEdge(edges[i][0], edges[i][1], edges[i][2]);
        return (network, ids);
    }

    public static void CheckFlowValid(int n, List<int[]> edges, int[] ids, MaxFlow network, int s, int t, int value)
    {
        var balance = new int[n];
        for (int i = 0; i < edges.Count; i++)
        {
            var e = edges[i];
            int f = network.FlowOn(ids[i]);
            if (!(f >= 0 && f <= e[2])) throw new Exception("FAILED: capacity bound");
            balance[e[0]] -= f;
            balance[e[1]] += f;
        }
        if (balance[t] != value) throw new Exception("FAILED: sink balance");
        if (balance[s] != -value) throw new Exception("FAILED: source balance");
        for (int node = 0; node < n; node++)
        {
            if (node != s && node != t && balance[node] != 0)
                throw new Exception("FAILED: conservation at " + node);
        }
    }

    public static List<(int n, List<int[]> edges)> LcgFlowGraphs(int count, int seed)
    {
        var graphs = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 4 + seed % 5;
            var edges = new List<int[]>();
            for (int node = 0; node < n - 1; node++)
            {
                seed = (seed * 75 + 74) % 65537;
                edges.Add(new[] { node, node + 1, 1 + seed % 20 });
            }
            for (int k = 0; k < n + 3; k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.Add(new[] { u, v, 1 + seed % 20 });
            }
            graphs.Add((n, edges));
        }
        return graphs;
    }
}

public static class Solution
{
    public static void Main()
    {
        var clrs = new List<int[]>
        {
            new[] { 0, 1, 16 }, new[] { 0, 2, 13 }, new[] { 1, 2, 10 }, new[] { 2, 1, 4 },
            new[] { 1, 3, 12 }, new[] { 3, 2, 9 }, new[] { 2, 4, 14 }, new[] { 4, 3, 7 },
            new[] { 3, 5, 20 }, new[] { 4, 5, 4 },
        };
        var (network, ids) = FlowOracle.BuildNetwork(6, clrs);
        var result = network.PushRelabel(0, 5);
        Check(result.value == 23, "CLRS 26.1 value");
        Check(result.pushes > 0 && result.relabels > 0, "CLRS pushes and relabels");
        FlowOracle.CheckFlowValid(6, clrs, ids, network, 0, 5, 23);
        Check(FlowOracle.BuildNetwork(6, clrs).network.EdmondsKarp(0, 5) == 23, "CLRS oracle");

        var single = new List<int[]> { new[] { 0, 1, 5 } };
        (network, ids) = FlowOracle.BuildNetwork(2, single);
        Check(network.PushRelabel(0, 1).value == 5, "single edge");
        FlowOracle.CheckFlowValid(2, single, ids, network, 0, 1, 5);

        var parallel = new List<int[]> { new[] { 0, 1, 3 }, new[] { 0, 1, 4 }, new[] { 1, 2, 2 } };
        Check(FlowOracle.BuildNetwork(3, parallel).network.PushRelabel(0, 2).value == 2, "parallel edges");
        Check(FlowOracle.BuildNetwork(3, parallel).network.EdmondsKarp(0, 2) == 2, "parallel oracle");

        var noPath = new List<int[]> { new[] { 0, 1, 9 }, new[] { 2, 3, 9 } };
        Check(FlowOracle.BuildNetwork(4, noPath).network.PushRelabel(0, 3).value == 0, "no path");
        var zeroCap = new List<int[]> { new[] { 0, 1, 0 }, new[] { 1, 2, 5 } };
        Check(FlowOracle.BuildNetwork(3, zeroCap).network.PushRelabel(0, 2).value == 0, "zero capacity");

        int checkedCount = 0, totalPushes = 0, totalRelabels = 0;
        foreach (int seed in new[] { 4541, 4542, 4543, 4544 })
        {
            foreach (var (n, edges) in FlowOracle.LcgFlowGraphs(8, seed))
            {
                (network, ids) = FlowOracle.BuildNetwork(n, edges);
                result = network.PushRelabel(0, n - 1);
                totalPushes += result.pushes;
                totalRelabels += result.relabels;
                FlowOracle.CheckFlowValid(n, edges, ids, network, 0, n - 1, result.value);
                int expected = FlowOracle.BuildNetwork(n, edges).network.EdmondsKarp(0, n - 1);
                Check(result.value == expected, "LCG value " + result.value + " vs " + expected);
                checkedCount++;
            }
        }
        Check(checkedCount == 32, "graph count");
        Check(totalPushes > 0 && totalRelabels > 0, "algorithm exercised");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
