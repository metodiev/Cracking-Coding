// Task 273 - Find an Eulerian trail in a directed graph (Hierholzer). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns the vertex sequence, an empty list for no edges, or null if no trail exists.</summary>
    public static List<int>? FindEulerianTrail(int n, int[][] edges, int? start)
    {
        if (edges.Length == 0) return new List<int>();
        var outDegree = new int[n];
        var inDegree = new int[n];
        var adjacency = new List<List<int>>();
        for (int i = 0; i < n; i++) adjacency.Add(new List<int>());
        foreach (var edge in edges)
        {
            adjacency[edge[0]].Add(edge[1]);
            outDegree[edge[0]]++;
            inDegree[edge[1]]++;
        }
        foreach (var neighbours in adjacency) neighbours.Sort();

        var starts = new List<int>();
        var ends = new List<int>();
        for (int v = 0; v < n; v++)
        {
            if (outDegree[v] - inDegree[v] == 1) starts.Add(v);
            if (inDegree[v] - outDegree[v] == 1) ends.Add(v);
        }
        if (starts.Count > 1 || ends.Count > 1) return null;
        for (int v = 0; v < n; v++)
        {
            if (outDegree[v] != inDegree[v] && !starts.Contains(v) && !ends.Contains(v)) return null;
        }

        int from;
        if (start == null)
        {
            if (starts.Count > 0) from = starts[0];
            else
            {
                from = -1;
                for (int v = 0; v < n; v++) if (outDegree[v] > 0) { from = v; break; }
            }
        }
        else
        {
            from = start.Value;
            if (starts.Count > 0 && from != starts[0]) return null;
            int diff = outDegree[from] - inDegree[from];
            if (diff != 0 && diff != 1) return null;
        }

        var active = new HashSet<int>();
        for (int v = 0; v < n; v++) if (outDegree[v] > 0 || inDegree[v] > 0) active.Add(v);
        if (active.Count > 0)
        {
            var undirected = new List<List<int>>();
            for (int i = 0; i < n; i++) undirected.Add(new List<int>());
            foreach (var edge in edges)
            {
                undirected[edge[0]].Add(edge[1]);
                undirected[edge[1]].Add(edge[0]);
            }
            var seen = new HashSet<int> { from };
            var stack = new Stack<int>();
            stack.Push(from);
            while (stack.Count > 0)
            {
                int u = stack.Pop();
                foreach (int v in undirected[u]) if (seen.Add(v)) stack.Push(v);
            }
            if (active.Any(v => !seen.Contains(v))) return null;
        }

        var pointer = new int[n];
        var trail = new List<int>();
        var walk = new Stack<int>();
        walk.Push(from);
        while (walk.Count > 0)
        {
            int u = walk.Peek();
            if (pointer[u] < adjacency[u].Count) walk.Push(adjacency[u][pointer[u]++]);
            else trail.Add(walk.Pop());
        }
        trail.Reverse();
        return trail.Count == edges.Length + 1 ? trail : null;
    }

    public static bool UsesAllEdges(List<int>? trail, int[][] edges)
    {
        if (trail == null) return false;
        var used = new SortedDictionary<string, int>();
        for (int i = 0; i + 1 < trail.Count; i++)
        {
            string key = trail[i] + "->" + trail[i + 1];
            used[key] = used.TryGetValue(key, out int c) ? c + 1 : 1;
        }
        var wanted = new SortedDictionary<string, int>();
        foreach (var edge in edges)
        {
            string key = edge[0] + "->" + edge[1];
            wanted[key] = wanted.TryGetValue(key, out int c) ? c + 1 : 1;
        }
        return used.Count == wanted.Count && wanted.All(kv => used.TryGetValue(kv.Key, out int v) && v == kv.Value);
    }

    private static string Join(List<int>? values) => values == null ? "null" : string.Join(",", values);

    public static void Main()
    {
        var cycle = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } };
        Check(Join(FindEulerianTrail(3, cycle, null)) == "0,1,2,0", "cycle");
        Check(UsesAllEdges(FindEulerianTrail(3, cycle, null), cycle), "cycle uses every edge");
        Check(Join(FindEulerianTrail(3, cycle, 1)) == "1,2,0,1", "balanced graph: any start");

        Check(Join(FindEulerianTrail(3, new[] { new[] { 0, 1 }, new[] { 1, 2 } }, null)) == "0,1,2", "open path");
        var mixed = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 }, new[] { 0, 3 } };
        var trail = FindEulerianTrail(4, mixed, null);
        Check(Join(trail) == "0,1,2,0,3", "circuit plus spur");
        Check(UsesAllEdges(trail, mixed), "spur uses every edge");

        Check(FindEulerianTrail(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 2, 0 }, new[] { 0, 3 } }, null) == null, "out-in == 2");
        Check(FindEulerianTrail(4, new[] { new[] { 0, 1 }, new[] { 2, 3 } }, null) == null, "disconnected");
        Check(Join(FindEulerianTrail(3, Array.Empty<int[]>(), null)) == "", "no edges");
        Check(Join(FindEulerianTrail(3, cycle, 2)) == "2,0,1,2", "balanced graph: other start");
        Check(Join(FindEulerianTrail(3, new[] { new[] { 1, 2 }, new[] { 2, 0 } }, 1)) == "1,2,0", "explicit valid start");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
