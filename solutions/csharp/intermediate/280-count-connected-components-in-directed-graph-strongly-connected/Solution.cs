// Task 280 - Count strongly connected components of a directed graph. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static List<List<int>> adjacency = new List<List<int>>();
    private static List<List<int>> reversedAdjacency = new List<List<int>>();
    private static bool[] visited = Array.Empty<bool>();
    private static List<int> order = new List<int>();

    /// <summary>Kosaraju: returns components as sorted vertex lists.</summary>
    public static List<List<int>> StronglyConnectedComponents(int n, int[][] edges)
    {
        adjacency = new List<List<int>>();
        reversedAdjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) { adjacency.Add(new List<int>()); reversedAdjacency.Add(new List<int>()); }
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            reversedAdjacency[edge[1]].Add(edge[0]);
        }

        visited = new bool[n];
        order = new List<int>();
        for (int v = 0; v < n; v++) if (!visited[v]) DfsFinish(v);

        var component = Enumerable.Repeat(-1, n).ToArray();
        var components = new List<List<int>>();
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int u = order[i];
            if (component[u] != -1) continue;
            int index = components.Count;
            component[u] = index;
            var group = new List<int> { u };
            var stack = new Stack<int>();
            stack.Push(u);
            while (stack.Count > 0)
            {
                int x = stack.Pop();
                foreach (int y in reversedAdjacency[x])
                {
                    if (component[y] == -1)
                    {
                        component[y] = index;
                        group.Add(y);
                        stack.Push(y);
                    }
                }
            }
            group.Sort();
            components.Add(group);
        }
        return components;
    }

    private static void DfsFinish(int u)
    {
        visited[u] = true;
        foreach (int v in adjacency[u]) if (!visited[v]) DfsFinish(v);
        order.Add(u);
    }

    public static int CountComponents(int n, int[][] edges) => StronglyConnectedComponents(n, edges).Count;

    /// <summary>Transitive closure oracle: u ~ v iff each reaches the other.</summary>
    public static List<List<int>> BruteForceComponents(int n, int[][] edges)
    {
        var reach = new bool[n, n];
        for (int i = 0; i < n; i++) reach[i, i] = true;
        foreach (var edge in edges) reach[edge[0], edge[1]] = true;
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i, k])
                    for (int j = 0; j < n; j++)
                        if (reach[k, j]) reach[i, j] = true;

        var groups = new SortedDictionary<string, List<int>>();
        for (int v = 0; v < n; v++)
        {
            var key = new List<string>();
            for (int i = 0; i < n; i++) if (reach[v, i] && reach[i, v]) key.Add(i.ToString());
            string k = string.Join(",", key);
            if (!groups.TryGetValue(k, out var list)) { list = new List<int>(); groups[k] = list; }
            list.Add(v);
        }
        var result = groups.Values.ToList();
        foreach (var group in result) group.Sort();
        return result;
    }

    private static int seed = 13579;

    private static int[][] RandomEdges(int n)
    {
        var edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 3 == 0) edges.Add(new[] { u, v });
            }
        }
        return edges.ToArray();
    }

    private static string Join(List<List<int>> components)
    {
        var parts = components.Select(g => string.Join(",", g)).ToList();
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }

    public static void Main()
    {
        Check(CountComponents(3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } }) == 1, "cycle");
        Check(CountComponents(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 3, 3 } }) == 2, "cycle plus self loop");
        Check(CountComponents(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }) == 3, "DAG");
        Check(CountComponents(3, Array.Empty<int[]>()) == 3, "no edges");
        Check(CountComponents(3, new[] { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 2 } }) == 2, "mutual pair plus tail");
        Check(CountComponents(2, new[] { new[] { 0, 1 } }) == 2, "one-way edge");

        var classic = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 3, 4 }, new[] { 4, 3 }, new[] { 4, 5 } };
        Check(Join(StronglyConnectedComponents(6, classic)) == "0,1,2|3,4|5", "classic components");
        Check(CountComponents(6, classic) == 3, "classic count");

        for (int trial = 0; trial < 120; trial++)
        {
            var edges = RandomEdges(6);
            Check(Join(StronglyConnectedComponents(6, edges)) == Join(BruteForceComponents(6, edges)), "random " + trial);
            Check(CountComponents(6, edges) <= 6, "bounded " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
