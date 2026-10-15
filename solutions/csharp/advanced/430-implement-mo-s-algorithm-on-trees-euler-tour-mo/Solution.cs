// Task 430 - Implement Mo's algorithm on trees (Euler tour + Mo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class EulerTour
{
    public readonly int[] Tin;
    public readonly int[] Tout;
    public readonly int[] Euler;
    public readonly List<List<int>> Children;

    public EulerTour(int n, int[][] edges)
    {
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }
        Children = new List<List<int>>();
        for (int i = 0; i < n; i++) Children.Add(new List<int>());
        Tin = new int[n];
        Tout = new int[n];
        Euler = new int[n];
        var timer = new int[1];
        Dfs(0, -1, adjacency, timer);
    }

    private void Dfs(int u, int parent, List<List<int>> adjacency, int[] timer)
    {
        Tin[u] = timer[0];
        Euler[timer[0]] = u;
        timer[0]++;
        foreach (int v in adjacency[u])
        {
            if (v != parent)
            {
                Children[u].Add(v);
                Dfs(v, u, adjacency, timer);
            }
        }
        Tout[u] = timer[0] - 1;
    }
}

internal static class MoSubtreeQueries
{
    public static int[] SubtreeDistinctCounts(int n, int[][] edges, int[] colors, int[] queries)
    {
        var tour = new EulerTour(n, edges);
        int[] tin = tour.Tin;
        int[] tout = tour.Tout;
        int[] euler = tour.Euler;
        int block = Math.Max(1, (int)Math.Sqrt(n));
        var order = new int[queries.Length];
        for (int i = 0; i < queries.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int slotA = tin[queries[a]] / block;
            int slotB = tin[queries[b]] / block;
            if (slotA != slotB) return slotA - slotB;
            int rightA = slotA % 2 == 0 ? tout[queries[a]] : -tout[queries[a]];
            int rightB = slotB % 2 == 0 ? tout[queries[b]] : -tout[queries[b]];
            return rightA - rightB;
        });
        var counts = new Dictionary<int, int>();
        int distinct = 0;
        int currentLeft = 0;
        int currentRight = -1;
        var answers = new int[queries.Length];
        foreach (int index in order)
        {
            int left = tin[queries[index]];
            int right = tout[queries[index]];
            while (currentLeft > left)
            {
                currentLeft--;
                int color = colors[euler[currentLeft]];
                counts.TryGetValue(color, out int amount);
                counts[color] = amount + 1;
                if (amount + 1 == 1) distinct++;
            }
            while (currentRight < right)
            {
                currentRight++;
                int color = colors[euler[currentRight]];
                counts.TryGetValue(color, out int amount);
                counts[color] = amount + 1;
                if (amount + 1 == 1) distinct++;
            }
            while (currentLeft < left)
            {
                int color = colors[euler[currentLeft]];
                counts.TryGetValue(color, out int amount);
                counts[color] = amount - 1;
                if (amount - 1 == 0) distinct--;
                currentLeft++;
            }
            while (currentRight > right)
            {
                int color = colors[euler[currentRight]];
                counts.TryGetValue(color, out int amount);
                counts[color] = amount - 1;
                if (amount - 1 == 0) distinct--;
                currentRight--;
            }
            answers[index] = distinct;
        }
        return answers;
    }

    public static int[] NaiveSubtreeDistinct(int n, int[][] edges, int[] colors, int[] queries)
    {
        var tour = new EulerTour(n, edges);
        var answers = new int[queries.Length];
        for (int i = 0; i < queries.Length; i++)
        {
            var seen = new HashSet<int>();
            var stack = new Stack<int>();
            stack.Push(queries[i]);
            while (stack.Count > 0)
            {
                int node = stack.Pop();
                seen.Add(colors[node]);
                foreach (int v in tour.Children[node]) stack.Push(v);
            }
            answers[i] = seen.Count;
        }
        return answers;
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
        var edges = new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 } };
        var colors = new[] { 0, 1, 0, 1 };
        Check(Same(MoSubtreeQueries.SubtreeDistinctCounts(4, edges, colors, new[] { 0, 1, 2, 3 }),
                   new[] { 2, 1, 1, 1 }), "ascending queries");
        Check(Same(MoSubtreeQueries.SubtreeDistinctCounts(4, edges, colors, new[] { 3, 2, 1, 0 }),
                   new[] { 1, 1, 1, 2 }), "descending queries");
        Check(Same(MoSubtreeQueries.SubtreeDistinctCounts(4, edges, colors, new[] { 2, 0, 2, 0 }),
                   new[] { 1, 2, 1, 2 }), "repeated queries");
        Check(Same(MoSubtreeQueries.SubtreeDistinctCounts(4, edges, new[] { 5, 5, 5, 5 }, new[] { 0, 1, 2, 3 }),
                   new[] { 1, 1, 1, 1 }), "uniform colours");
        Check(Same(MoSubtreeQueries.SubtreeDistinctCounts(1, Array.Empty<int[]>(), new[] { 9 }, new[] { 0 }),
                   new[] { 1 }), "single node");

        foreach (int seed in new[] { 430, 24680, 13579 })
        {
            int local = seed;
            for (int trial = 0; trial < 15; trial++)
            {
                local = (local * 75 + 74) % 65537;
                int n = 2 + local % 49;
                var treeEdges = new List<int[]>();
                for (int node = 1; node < n; node++)
                {
                    local = (local * 75 + 74) % 65537;
                    treeEdges.Add(new[] { local % node, node });
                }
                var palette = new int[n];
                for (int node = 0; node < n; node++)
                {
                    local = (local * 75 + 74) % 65537;
                    palette[node] = local % (2 + seed % 6);
                }
                var queries = new int[n + 12];
                for (int i = 0; i < n; i++) queries[i] = i;
                for (int i = n; i < queries.Length; i++)
                {
                    local = (local * 75 + 74) % 65537;
                    queries[i] = local % n;
                }
                var edgeArray = treeEdges.ToArray();
                int[] got = MoSubtreeQueries.SubtreeDistinctCounts(n, edgeArray, palette, queries);
                int[] want = MoSubtreeQueries.NaiveSubtreeDistinct(n, edgeArray, palette, queries);
                Check(Same(got, want), "lcg trial " + trial + " n=" + n);
            }
        }

        int wideLocal = 123123;
        for (int trial = 0; trial < 10; trial++)
        {
            wideLocal = (wideLocal * 75 + 74) % 65537;
            int n = 2 + wideLocal % 49;
            var treeEdges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                wideLocal = (wideLocal * 75 + 74) % 65537;
                treeEdges.Add(new[] { wideLocal % node, node });
            }
            var palette = new int[n];
            for (int node = 0; node < n; node++)
            {
                wideLocal = (wideLocal * 75 + 74) % 65537;
                palette[node] = wideLocal % 1000;
            }
            var queries = new int[n];
            for (int i = 0; i < n; i++) queries[i] = i;
            var edgeArray = treeEdges.ToArray();
            int[] got = MoSubtreeQueries.SubtreeDistinctCounts(n, edgeArray, palette, queries);
            int[] want = MoSubtreeQueries.NaiveSubtreeDistinct(n, edgeArray, palette, queries);
            Check(Same(got, want), "wide palette " + n);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
