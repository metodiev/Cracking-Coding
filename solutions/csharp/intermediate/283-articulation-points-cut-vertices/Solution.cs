// Task 283 - Articulation points (cut vertices). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static List<List<int>> adjacency = new List<List<int>>();
    private static int[] disc = Array.Empty<int>();
    private static int[] low = Array.Empty<int>();
    private static int timer;
    private static SortedSet<int> points = new SortedSet<int>();

    /// <summary>Returns cut vertices in increasing order.</summary>
    public static int[] ArticulationPoints(int n, int[][] edges)
    {
        adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            adjacency[edge[1]].Add(edge[0]);
        }

        disc = Enumerable.Repeat(-1, n).ToArray();
        low = new int[n];
        timer = 0;
        points = new SortedSet<int>();
        for (int v = 0; v < n; v++) if (disc[v] == -1) Dfs(v, -1);

        return points.ToArray();
    }

    private static void Dfs(int u, int parent)
    {
        disc[u] = low[u] = timer++;
        int children = 0;
        foreach (int v in adjacency[u])
        {
            if (disc[v] == -1)                       // tree edge
            {
                children++;
                Dfs(v, u);
                low[u] = Math.Min(low[u], low[v]);
                if (parent != -1 && low[v] >= disc[u]) points.Add(u); // v's subtree cannot escape
            }
            else if (v != parent)                    // back edge to an ancestor
            {
                low[u] = Math.Min(low[u], disc[v]);
            }
        }
        if (parent == -1 && children > 1) points.Add(u); // root special case
    }

    public static int CountComponents(int n, int[][] edges, int skipped)
    {
        var graph = new List<List<int>>();
        for (int i = 0; i < n; i++) graph.Add(new List<int>());
        foreach (var edge in edges)
        {
            if (edge[0] == skipped || edge[1] == skipped) continue;
            graph[edge[0]].Add(edge[1]);
            graph[edge[1]].Add(edge[0]);
        }
        var seen = new bool[n];
        int total = 0;
        for (int s = 0; s < n; s++)
        {
            if (s == skipped || seen[s]) continue;
            total++;
            var stack = new Stack<int>();
            stack.Push(s);
            seen[s] = true;
            while (stack.Count > 0)
            {
                int x = stack.Pop();
                foreach (int y in graph[x]) if (!seen[y]) { seen[y] = true; stack.Push(y); }
            }
        }
        return total;
    }

    /// <summary>Oracle: remove each vertex and count connected components.</summary>
    public static int[] BruteForceArticulationPoints(int n, int[][] edges)
    {
        int baseCount = CountComponents(n, edges, -1);
        var result = new List<int>();
        for (int v = 0; v < n; v++) if (CountComponents(n, edges, v) > baseCount) result.Add(v);
        return result.ToArray();
    }

    /// <summary>Each unordered pair {u, v} appears at most once; edge when seed % divisor == 0.</summary>
    public static int[][] RandomSimpleGraph(int n, int divisor, int[] seed)
    {
        var edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % divisor == 0) edges.Add(new[] { u, v });
            }
        }
        return edges.ToArray();
    }

    private static string Key(int[] values)
    {
        var parts = values.Select(v => v.ToString()).ToList();
        return string.Join(",", parts);
    }

    public static void Main()
    {
        Check(Key(ArticulationPoints(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 } })) == "1,2", "path middles");
        Check(Key(ArticulationPoints(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 } })) == "", "cycle has none");
        Check(Key(ArticulationPoints(5, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 } })) == "0", "star center");
        Check(Key(ArticulationPoints(3, new[] { new[] { 0, 1 }, new[] { 0, 2 } })) == "0", "root with 2 children");
        Check(Key(ArticulationPoints(6, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 3 } })) == "", "two triangles");
        Check(Key(ArticulationPoints(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 3 } })) == "2", "triangle plus tail");
        Check(Key(ArticulationPoints(2, new[] { new[] { 0, 1 } })) == "", "single edge");
        Check(Key(ArticulationPoints(3, Array.Empty<int[]>())) == "", "isolated vertices");
        Check(Key(ArticulationPoints(3, new[] { new[] { 0, 1 }, new[] { 0, 1 }, new[] { 1, 2 } })) == "1", "parallel edges");
        Check(Key(ArticulationPoints(7, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 3 }, new[] { 2, 6 } })) == "2", "two cycles linked");

        foreach (int divisor in new[] { 3, 5 })
        {
            var seed = new[] { 13579 };
            bool seenCut = false, seenNone = false;
            for (int trial = 0; trial < 120; trial++)
            {
                var edges = RandomSimpleGraph(6, divisor, seed);
                var found = ArticulationPoints(6, edges);
                Check(Key(found) == Key(BruteForceArticulationPoints(6, edges)), "random " + divisor + " " + trial);
                if (found.Length > 0) seenCut = true; else seenNone = true;
            }
            Check(seenCut && seenNone, "coverage " + divisor);
        }

        var pathEdges = new int[9][];
        for (int i = 0; i < 9; i++) pathEdges[i] = new[] { i, i + 1 };
        Check(Key(ArticulationPoints(10, pathEdges)) == "1,2,3,4,5,6,7,8", "long path interior");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
