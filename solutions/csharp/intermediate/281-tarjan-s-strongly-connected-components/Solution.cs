// Task 281 - Tarjan's strongly connected components in one DFS pass. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static List<List<int>> adjacency = new List<List<int>>();
    private static int[] index = Array.Empty<int>();
    private static int[] low = Array.Empty<int>();
    private static bool[] onStack = Array.Empty<bool>();
    private static Stack<int> stack = new Stack<int>();
    private static List<List<int>> components = new List<List<int>>();
    private static int counter;

    /// <summary>Single-pass Tarjan; returns components sorted by smallest vertex.</summary>
    public static List<List<int>> TarjanScc(int n, int[][] edges)
    {
        adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges) adjacency[edge[0]].Add(edge[1]);

        index = Enumerable.Repeat(-1, n).ToArray();
        low = new int[n];
        onStack = new bool[n];
        stack = new Stack<int>();
        components = new List<List<int>>();
        counter = 0;

        for (int v = 0; v < n; v++) if (index[v] == -1) Strongconnect(v);
        components.Sort((a, b) => a[0].CompareTo(b[0]));
        return components;
    }

    private static void Strongconnect(int u)
    {
        index[u] = low[u] = counter++;
        stack.Push(u);
        onStack[u] = true;
        foreach (int v in adjacency[u])
        {
            if (index[v] == -1)                 // tree edge: descend
            {
                Strongconnect(v);
                low[u] = Math.Min(low[u], low[v]);
            }
            else if (onStack[v])                // back/cross edge to an open SCC
            {
                low[u] = Math.Min(low[u], index[v]);
            }
        }
        if (low[u] == index[u])                 // u closes an SCC: pop until u
        {
            var group = new List<int>();
            while (true)
            {
                int w = stack.Pop();
                onStack[w] = false;
                group.Add(w);
                if (w == u) break;
            }
            group.Sort();
            components.Add(group);
        }
    }

    private static List<List<int>> dfsAdjacency = new List<List<int>>();
    private static bool[] dfsVisited = Array.Empty<bool>();
    private static List<int> dfsOrder = new List<int>();

    /// <summary>Task 280's two-pass algorithm, used here only as an independent oracle.</summary>
    public static List<List<int>> KosarajuScc(int n, int[][] edges)
    {
        dfsAdjacency = new List<List<int>>();
        var backward = new List<List<int>>();
        for (int i = 0; i < n; i++) { dfsAdjacency.Add(new List<int>()); backward.Add(new List<int>()); }
        foreach (var edge in edges)
        {
            dfsAdjacency[edge[0]].Add(edge[1]);
            backward[edge[1]].Add(edge[0]);
        }

        dfsVisited = new bool[n];
        dfsOrder = new List<int>();
        for (int v = 0; v < n; v++) if (!dfsVisited[v]) DfsFinish(v);

        var component = Enumerable.Repeat(-1, n).ToArray();
        var result = new List<List<int>>();
        for (int i = dfsOrder.Count - 1; i >= 0; i--)
        {
            int u = dfsOrder[i];
            if (component[u] != -1) continue;
            int groupIndex = result.Count;
            component[u] = groupIndex;
            var group = new List<int> { u };
            var pending = new Stack<int>();
            pending.Push(u);
            while (pending.Count > 0)
            {
                int x = pending.Pop();
                foreach (int y in backward[x])
                {
                    if (component[y] == -1)
                    {
                        component[y] = groupIndex;
                        group.Add(y);
                        pending.Push(y);
                    }
                }
            }
            group.Sort();
            result.Add(group);
        }
        return result;
    }

    private static void DfsFinish(int u)
    {
        dfsVisited[u] = true;
        foreach (int v in dfsAdjacency[u]) if (!dfsVisited[v]) DfsFinish(v);
        dfsOrder.Add(u);
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

    private static string Join(List<List<int>> groups)
    {
        var parts = groups.Select(g => string.Join(",", g)).ToList();
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }

    public static void Main()
    {
        var classic = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 3, 4 }, new[] { 4, 3 }, new[] { 4, 5 } };
        Check(Join(TarjanScc(6, classic)) == "0,1,2|3,4|5", "classic");

        Check(Join(TarjanScc(3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } })) == "0,1,2", "cycle");
        Check(Join(TarjanScc(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } })) == "0|1|2", "DAG");
        Check(Join(TarjanScc(3, Array.Empty<int[]>())) == "0|1|2", "no edges");
        Check(Join(TarjanScc(3, new[] { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 2 } })) == "0,1|2", "mutual pair plus tail");
        Check(Join(TarjanScc(2, new[] { new[] { 0, 1 } })) == "0|1", "one-way edge");
        Check(Join(TarjanScc(1, new[] { new[] { 0, 0 } })) == "0", "self loop");

        bool nontrivialSeen = false;
        for (int trial = 0; trial < 120; trial++)
        {
            var edges = RandomEdges(6);
            var components = TarjanScc(6, edges);
            Check(Join(components) == Join(KosarajuScc(6, edges)), "kosaraju match " + trial);
            var flat = components.SelectMany(g => g).OrderBy(v => v).ToArray();
            Check(flat.SequenceEqual(new[] { 0, 1, 2, 3, 4, 5 }), "partition " + trial);
            if (components.Any(g => g.Count > 1)) nontrivialSeen = true;
        }
        Check(nontrivialSeen, "LCG graphs should contain real SCCs");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
