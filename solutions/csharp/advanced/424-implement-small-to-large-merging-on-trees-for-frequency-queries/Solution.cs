// Task 424 - Implement small-to-large merging on trees for frequency queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal static class FrequencySummary
{
    private static List<List<int>> ChildrenOf(int n, int[][] edges, out int[] parent, out List<int> order)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = -1;
        parent[0] = 0;
        order = new List<int> { 0 };
        var children = new List<List<int>>();
        for (int i = 0; i < n; i++) children.Add(new List<int>());
        for (int i = 0; i < order.Count; i++)
        {
            int u = order[i];
            foreach (int v in adjacency[u])
            {
                if (v != parent[u])
                {
                    parent[v] = u;
                    children[u].Add(v);
                    order.Add(v);
                }
            }
        }
        return children;
    }

    public static int[][] Compute(int n, int[][] edges, int[] colors)
    {
        var children = ChildrenOf(n, edges, out _, out var order);
        var counters = new Dictionary<int, int>[n];
        var distinct = new int[n];
        var top = new int[n];
        for (int index = order.Count - 1; index >= 0; index--)
        {
            int u = order[index];
            int biggest = -1;
            foreach (int v in children[u])
            {
                if (biggest == -1 || counters[v].Count > counters[biggest].Count) biggest = v;
            }
            Dictionary<int, int> table;
            int best;
            if (biggest == -1)
            {
                table = new Dictionary<int, int> { [colors[u]] = 1 };
                best = 1;
            }
            else
            {
                table = counters[biggest];               // reuse the largest counter
                best = top[biggest];
                table.TryGetValue(colors[u], out int own);
                table[colors[u]] = own + 1;
                if (own + 1 > best) best = own + 1;
                foreach (int v in children[u])
                {
                    if (v == biggest) continue;
                    foreach (var entry in counters[v])
                    {
                        table.TryGetValue(entry.Key, out int current);
                        int merged = current + entry.Value;
                        table[entry.Key] = merged;
                        if (merged > best) best = merged;
                    }
                }
            }
            counters[u] = table;
            distinct[u] = table.Count;
            top[u] = best;
        }
        return new[] { distinct, top };
    }

    public static int[][] Naive(int n, int[][] edges, int[] colors)
    {
        var children = ChildrenOf(n, edges, out _, out _);
        var distinct = new int[n];
        var top = new int[n];
        for (int u = 0; u < n; u++)
        {
            var counts = new Dictionary<int, int>();
            var stack = new Stack<int>();
            stack.Push(u);
            while (stack.Count > 0)
            {
                int node = stack.Pop();
                counts.TryGetValue(colors[node], out int current);
                counts[colors[node]] = current + 1;
                foreach (int v in children[node]) stack.Push(v);
            }
            distinct[u] = counts.Count;
            int best = 0;
            foreach (int amount in counts.Values) if (amount > best) best = amount;
            top[u] = best;
        }
        return new[] { distinct, top };
    }
}

public static class Solution
{
    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        var tree = new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 } };
        var colors = new[] { 0, 1, 0, 1 };
        var summary = FrequencySummary.Compute(4, tree, colors);
        Check(Same(summary[0], new[] { 2, 1, 1, 1 }), "distinct counts");
        Check(Same(summary[1], new[] { 2, 2, 1, 1 }), "top frequencies");
        var expected = FrequencySummary.Naive(4, tree, colors);
        Check(Same(summary[0], expected[0]) && Same(summary[1], expected[1]), "small tree vs naive");

        var path = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 } };
        summary = FrequencySummary.Compute(5, path, new[] { 1, 2, 2, 3, 3 });
        Check(Same(summary[0], new[] { 3, 2, 2, 1, 1 }), "path distinct");
        Check(Same(summary[1], new[] { 2, 2, 2, 2, 1 }), "path top");
        expected = FrequencySummary.Naive(5, path, new[] { 1, 2, 2, 3, 3 });
        Check(Same(summary[0], expected[0]) && Same(summary[1], expected[1]), "path vs naive");

        summary = FrequencySummary.Compute(1, Array.Empty<int[]>(), new[] { 7 });
        Check(Same(summary[0], new[] { 1 }) && Same(summary[1], new[] { 1 }), "single node");

        var star = new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 } };
        summary = FrequencySummary.Compute(5, star, new[] { 3, 3, 3, 3, 3 });
        Check(Same(summary[0], new[] { 1, 1, 1, 1, 1 }), "star distinct");
        Check(Same(summary[1], new[] { 5, 1, 1, 1, 1 }), "star top");

        foreach (int seed in new[] { 424, 2024, 77777 })
        {
            int local = seed;
            for (int trial = 0; trial < 20; trial++)
            {
                local = (local * 75 + 74) % 65537;
                int n = 2 + local % 59;
                var edges = new List<int[]>();
                for (int node = 1; node < n; node++)
                {
                    local = (local * 75 + 74) % 65537;
                    edges.Add(new[] { local % node, node });
                }
                var palette = new int[n];
                for (int node = 0; node < n; node++)
                {
                    local = (local * 75 + 74) % 65537;
                    palette[node] = local % (2 + seed % 5);
                }
                int[][] got = FrequencySummary.Compute(n, edges.ToArray(), palette);
                int[][] want = FrequencySummary.Naive(n, edges.ToArray(), palette);
                Check(Same(got[0], want[0]) && Same(got[1], want[1]), "lcg trial " + trial + " n=" + n);
            }
        }

        int wideLocal = 909090;
        for (int trial = 0; trial < 10; trial++)
        {
            wideLocal = (wideLocal * 75 + 74) % 65537;
            int n = 2 + wideLocal % 59;
            var edges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                wideLocal = (wideLocal * 75 + 74) % 65537;
                edges.Add(new[] { wideLocal % node, node });
            }
            var palette = new int[n];
            for (int node = 0; node < n; node++)
            {
                wideLocal = (wideLocal * 75 + 74) % 65537;
                palette[node] = wideLocal % 1000;
            }
            int[][] got = FrequencySummary.Compute(n, edges.ToArray(), palette);
            int[][] want = FrequencySummary.Naive(n, edges.ToArray(), palette);
            Check(Same(got[0], want[0]) && Same(got[1], want[1]), "wide palette " + n);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
