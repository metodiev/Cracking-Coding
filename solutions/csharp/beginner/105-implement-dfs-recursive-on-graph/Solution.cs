// Task 105 - Implement DFS (recursive) on graph. (Tier: Beginner, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static List<int> DfsOrder(Dictionary<int, List<int>> graph, int source)
    {
        var order = new List<int>();
        if (!graph.ContainsKey(source)) return order;
        var visited = new HashSet<int>();
        Dfs(graph, source, visited, order);
        return order;
    }

    public static List<int> DfsAll(Dictionary<int, List<int>> graph)
    {
        var visited = new HashSet<int>();
        var order = new List<int>();
        var keys = new List<int>(graph.Keys);
        keys.Sort();
        foreach (int v in keys)
        {
            if (!visited.Contains(v)) Dfs(graph, v, visited, order);
        }
        return order;
    }

    private static void Dfs(Dictionary<int, List<int>> graph, int v, HashSet<int> visited, List<int> order)
    {
        visited.Add(v);
        order.Add(v);
        if (graph.TryGetValue(v, out var neighbors))
        {
            foreach (int w in neighbors)
            {
                if (!visited.Contains(w)) Dfs(graph, w, visited, order);
            }
        }
    }

    public static void Main()
    {
        var cyclic = MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 2, 3 }, new[] { 3, 3 } });
        Check(SameList(DfsOrder(cyclic, 0), 0, 1, 2, 3), "cycle guard");

        var twoCycles = MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 2, 3 }, new[] { 3, 2 } });
        Check(SameList(DfsOrder(twoCycles, 0), 0, 1), "single component only");
        Check(SameList(DfsAll(twoCycles), 0, 1, 2, 3), "all components");

        var single = new Dictionary<int, List<int>> { [5] = new List<int>() };
        Check(SameList(DfsOrder(single, 5), 5), "single vertex");
        Check(DfsOrder(new Dictionary<int, List<int>>(), 0).Count == 0, "empty graph");
        Check(DfsOrder(cyclic, 99).Count == 0, "source not in graph");
        Check(DfsAll(new Dictionary<int, List<int>>()).Count == 0, "dfs_all empty");
        var lone = new Dictionary<int, List<int>> { [7] = new List<int>() };
        Check(SameList(DfsAll(lone), 7), "dfs_all single");

        var chain = new Dictionary<int, List<int>>();
        for (int i = 0; i <= 50; i++)
            chain[i] = i < 50 ? new List<int> { i + 1 } : new List<int>();
        Check(DfsOrder(chain, 0).Count == 51, "deep chain");
        Console.WriteLine("PASS");
    }

    private static Dictionary<int, List<int>> MakeGraph(int[][] edges)
    {
        var g = new Dictionary<int, List<int>>();
        foreach (var e in edges)
        {
            if (!g.TryGetValue(e[0], out var list))
            {
                list = new List<int>();
                g[e[0]] = list;
            }
            list.Add(e[1]);
        }
        return g;
    }

    private static bool SameList(List<int> actual, params int[] expected)
    {
        if (actual.Count != expected.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (actual[i] != expected[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
