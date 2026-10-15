// Task 282 - Bridge edges via Tarjan's disc/low values. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static List<List<int[]>> adjacency = new List<List<int[]>>();
    private static int[] disc = Array.Empty<int>();
    private static int[] low = Array.Empty<int>();
    private static int timer;
    private static List<int[]> bridges = new List<int[]>();

    /// <summary>Returns bridges as canonical sorted (min, max) pairs.</summary>
    public static int[][] FindBridges(int n, int[][] edges)
    {
        adjacency = new List<List<int[]>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int[]>());
        for (int eid = 0; eid < edges.Length; eid++)
        {
            int a = edges[eid][0], b = edges[eid][1];
            if (a == b) continue;                              // self loops are never bridges
            adjacency[a].Add(new[] { b, eid });
            adjacency[b].Add(new[] { a, eid });
        }

        disc = Enumerable.Repeat(-1, n).ToArray();
        low = new int[n];
        timer = 0;
        bridges = new List<int[]>();
        for (int v = 0; v < n; v++) if (disc[v] == -1) Dfs(v, -1);

        bridges.Sort((p, q) => p[0] != q[0] ? p[0].CompareTo(q[0]) : p[1].CompareTo(q[1]));
        return bridges.ToArray();
    }

    private static void Dfs(int u, int parentEdge)
    {
        disc[u] = low[u] = timer++;
        foreach (var step in adjacency[u])
        {
            int v = step[0], eid = step[1];
            if (eid == parentEdge) continue;                   // skip only the arrival edge itself
            if (disc[v] != -1)
            {
                low[u] = Math.Min(low[u], disc[v]);            // back edge to an ancestor
            }
            else
            {
                Dfs(v, eid);
                low[u] = Math.Min(low[u], low[v]);
                if (low[v] > disc[u])                          // v's subtree cannot escape above u
                {
                    bridges.Add(new[] { Math.Min(u, v), Math.Max(u, v) });
                }
            }
        }
    }

    public static int CountComponents(int n, int[][] edges)
    {
        var graph = new List<List<int>>();
        for (int i = 0; i < n; i++) graph.Add(new List<int>());
        foreach (var edge in edges)
        {
            if (edge[0] == edge[1]) continue;
            graph[edge[0]].Add(edge[1]);
            graph[edge[1]].Add(edge[0]);
        }
        var seen = new bool[n];
        int total = 0;
        for (int s = 0; s < n; s++)
        {
            if (seen[s]) continue;
            total++;
            var stack = new Stack<int>();
            stack.Push(s);
            seen[s] = true;
            while (stack.Count > 0)
            {
                int x = stack.Pop();
                foreach (int y in graph[x])
                {
                    if (!seen[y]) { seen[y] = true; stack.Push(y); }
                }
            }
        }
        return total;
    }

    /// <summary>Oracle: deleting a bridge increases the number of connected components.</summary>
    public static int[][] BruteForceBridges(int n, int[][] edges)
    {
        int baseCount = CountComponents(n, edges);
        var found = new SortedSet<string>();
        for (int i = 0; i < edges.Length; i++)
        {
            var without = new List<int[]>();
            for (int j = 0; j < edges.Length; j++) if (j != i) without.Add(edges[j]);
            if (CountComponents(n, without.ToArray()) > baseCount)
            {
                found.Add(Math.Min(edges[i][0], edges[i][1]) + "-" + Math.Max(edges[i][0], edges[i][1]));
            }
        }
        var result = new List<int[]>();
        foreach (string key in found)
        {
            var parts = key.Split('-');
            result.Add(new[] { int.Parse(parts[0]), int.Parse(parts[1]) });
        }
        return result.ToArray();
    }

    /// <summary>Simple graphs: each unordered pair {u, v} considered once.</summary>
    public static int[][] RandomUndirectedGraph(int n, int[] seed)
    {
        var edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % 3 == 0) edges.Add(new[] { u, v });
            }
        }
        return edges.ToArray();
    }

    /// <summary>Multigraphs: (u, v) and (v, u) may both appear as parallel edges.</summary>
    public static int[][] RandomMultigraph(int n, int[] seed)
    {
        var edges = new List<int[]>();
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                if (u == v) continue;
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % 3 == 0) edges.Add(new[] { u, v });
            }
        }
        return edges.ToArray();
    }

    private static string Key(int[][] pairs)
    {
        var parts = pairs.Select(p => p[0] + "-" + p[1]).ToList();
        parts.Sort(StringComparer.Ordinal);
        return string.Join("|", parts);
    }

    public static void Main()
    {
        Check(Key(FindBridges(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } })) == "0-1|1-2", "path has 2 bridges");
        Check(Key(FindBridges(3, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } })) == "", "triangle has none");
        Check(Key(FindBridges(4, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 } })) == "", "square has none");
        Check(Key(FindBridges(5, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 2, 4 } })) == "2-4", "square plus tail");
        Check(Key(FindBridges(5, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 3, 4 } })) == "0-1|1-2|3-4", "disconnected");
        Check(Key(FindBridges(3, new[] { new[] { 0, 1 }, new[] { 0, 1 }, new[] { 1, 2 } })) == "1-2", "parallel edges");
        Check(Key(FindBridges(2, new[] { new[] { 0, 0 }, new[] { 0, 1 } })) == "0-1", "self loop ignored");
        Check(Key(FindBridges(3, new[] { new[] { 0, 1 } })) == "0-1", "isolated vertex");

        var simpleSeed = new[] { 13579 };
        var multiSeed = new[] { 13579 };
        bool simpleWithBridge = false, simpleWithoutBridge = false;
        for (int trial = 0; trial < 120; trial++)
        {
            var edges = RandomUndirectedGraph(6, simpleSeed);
            var bridgesFound = FindBridges(6, edges);
            Check(Key(bridgesFound) == Key(BruteForceBridges(6, edges)), "simple " + trial);
            if (bridgesFound.Length > 0) simpleWithBridge = true; else simpleWithoutBridge = true;
        }
        Check(simpleWithBridge && simpleWithoutBridge, "simple coverage");

        bool multiWithBridge = false, multiWithoutBridge = false;
        for (int trial = 0; trial < 120; trial++)
        {
            var edges = RandomMultigraph(6, multiSeed);
            var bridgesFound = FindBridges(6, edges);
            Check(Key(bridgesFound) == Key(BruteForceBridges(6, edges)), "multi " + trial);
            if (bridgesFound.Length > 0) multiWithBridge = true; else multiWithoutBridge = true;
        }
        Check(multiWithBridge && multiWithoutBridge, "multigraph coverage");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
