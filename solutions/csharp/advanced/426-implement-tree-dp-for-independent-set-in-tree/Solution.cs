// Task 426 - Implement tree DP for independent set in tree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal static class IndependentSet
{
    public static (long Total, bool[] Picked) Solve(int n, int[][] edges, int[] weights)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = -1;
        parent[0] = 0;
        var order = new List<int> { 0 };
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    parent[v] = u;
                    order.Add(v);
                }
            }
        }
        var children = new List<List<int>>();
        for (int i = 0; i < n; i++) children.Add(new List<int>());
        for (int i = 1; i < order.Count; i++) children[parent[order[i]]].Add(order[i]);

        var take = new long[n];
        var skip = new long[n];
        for (int index = order.Count - 1; index >= 0; index--)
        {
            int u = order[index];
            long chosen = weights[u];
            long notChosen = 0;
            foreach (int v in children[u])
            {
                chosen += skip[v];
                notChosen += Math.Max(take[v], skip[v]);
            }
            take[u] = chosen;
            skip[u] = notChosen;
        }

        var picked = new bool[n];
        var stack = new Stack<(int Node, bool Taking)>();
        stack.Push((0, take[0] > skip[0]));
        while (stack.Count > 0)
        {
            var (u, taking) = stack.Pop();
            picked[u] = taking;
            foreach (int v in children[u])
                stack.Push((v, taking ? false : take[v] > skip[v]));
        }
        long total = 0;
        for (int u = 0; u < n; u++)
            if (picked[u]) total += weights[u];
        return (total, picked);
    }

    public static bool IsValid(int[][] edges, bool[] picked)
    {
        foreach (var edge in edges)
            if (picked[edge[0]] && picked[edge[1]]) return false;
        return true;
    }

    public static long WeightOf(int n, int[] weights, bool[] picked)
    {
        long total = 0;
        for (int u = 0; u < n; u++)
            if (picked[u]) total += weights[u];
        return total;
    }

    public static long BruteForce(int n, int[][] edges, int[] weights)
    {
        long best = 0;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            long weight = 0;
            for (int i = 0; i < n; i++)
                if ((mask >> i & 1) != 0) weight += weights[i];
            if (weight <= best) continue;
            bool ok = true;
            foreach (var edge in edges)
            {
                if ((mask >> edge[0] & 1) != 0 && (mask >> edge[1] & 1) != 0)
                {
                    ok = false;
                    break;
                }
            }
            if (ok) best = weight;
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        var result = IndependentSet.Solve(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }, new[] { 10, 1, 10 });
        Check(result.Total == 20 && Same(result.Picked, new[] { true, false, true }), "path3");

        result = IndependentSet.Solve(3, new[] { new[] { 0, 1 }, new[] { 0, 2 } }, new[] { 5, 4, 4 });
        Check(result.Total == 8 && Same(result.Picked, new[] { false, true, true }), "star");

        result = IndependentSet.Solve(2, new[] { new[] { 0, 1 } }, new[] { -1, -2 });
        Check(result.Total == 0 && Same(result.Picked, new[] { false, false }), "negative pair");

        result = IndependentSet.Solve(1, Array.Empty<int[]>(), new[] { 7 });
        Check(result.Total == 7 && Same(result.Picked, new[] { true }), "single positive");
        result = IndependentSet.Solve(1, Array.Empty<int[]>(), new[] { -7 });
        Check(result.Total == 0 && Same(result.Picked, new[] { false }), "single negative");

        result = IndependentSet.Solve(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } }, new[] { 3, 10, 3, 10 });
        Check(result.Total == 20, "path4 weight");
        Check(Same(result.Picked, new[] { false, true, false, true }), "path4 choice");

        int local = 426426;
        for (int trial = 0; trial < 15; trial++)
        {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 15;
            var edges = new List<int[]>();
            var weights = new int[n];
            for (int node = 0; node < n; node++)
            {
                local = (local * 75 + 74) % 65537;
                weights[node] = local % 21 - 10;
                if (node > 0)
                {
                    local = (local * 75 + 74) % 65537;
                    edges.Add(new[] { local % node, node });
                }
            }
            var edgeArray = edges.ToArray();
            var got = IndependentSet.Solve(n, edgeArray, weights);
            long expected = IndependentSet.BruteForce(n, edgeArray, weights);
            Check(got.Total == expected, "small brute trial " + trial);
            Check(IndependentSet.IsValid(edgeArray, got.Picked), "small validity " + trial);
            Check(IndependentSet.WeightOf(n, weights, got.Picked) == got.Total, "small reconstruction " + trial);
        }

        local = 99999;
        for (int trial = 0; trial < 10; trial++)
        {
            local = (local * 75 + 74) % 65537;
            int n = 20 + local % 41;
            var edges = new List<int[]>();
            var weights = new int[n];
            for (int node = 0; node < n; node++)
            {
                local = (local * 75 + 74) % 65537;
                weights[node] = local % 41 - 20;
                if (node > 0)
                {
                    local = (local * 75 + 74) % 65537;
                    edges.Add(new[] { local % node, node });
                }
            }
            var edgeArray = edges.ToArray();
            var got = IndependentSet.Solve(n, edgeArray, weights);
            Check(IndependentSet.IsValid(edgeArray, got.Picked), "large validity " + trial);
            Check(IndependentSet.WeightOf(n, weights, got.Picked) == got.Total, "large reconstruction " + trial);
            Check(got.Total >= 0, "large non-negative " + trial);
        }

        Console.WriteLine("PASS");
    }

    private static bool Same(bool[] a, bool[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
