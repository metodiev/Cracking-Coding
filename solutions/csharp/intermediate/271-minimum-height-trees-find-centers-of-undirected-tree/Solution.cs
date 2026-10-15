// Task 271 - Minimum height trees (tree centers). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    public static List<int> MinHeightTrees(int n, int[][] edges)
    {
        if (n == 1) return new List<int> { 0 };
        var adj = new List<List<int>>();
        for (int i = 0; i < n; i++) adj.Add(new List<int>());
        var degree = new int[n];
        foreach (var edge in edges)
        {
            adj[edge[0]].Add(edge[1]);
            adj[edge[1]].Add(edge[0]);
            degree[edge[0]]++;
            degree[edge[1]]++;
        }

        var leaves = new Queue<int>();
        for (int i = 0; i < n; i++) if (degree[i] == 1) leaves.Enqueue(i);
        int remaining = n;
        while (remaining > 2)
        {
            int size = leaves.Count;
            remaining -= size;
            for (int i = 0; i < size; i++)
            {
                int leaf = leaves.Dequeue();
                foreach (int neighbour in adj[leaf])
                {
                    if (--degree[neighbour] == 1) leaves.Enqueue(neighbour);
                }
            }
        }
        var outList = leaves.ToList();
        outList.Sort();
        return outList;
    }

    /// <summary>Independent oracle: BFS height from every root, keep the minimizers.</summary>
    public static (List<int> Centers, int Best) BruteForceCenters(int n, int[][] edges)
    {
        var adj = new List<List<int>>();
        for (int i = 0; i < n; i++) adj.Add(new List<int>());
        foreach (var edge in edges)
        {
            adj[edge[0]].Add(edge[1]);
            adj[edge[1]].Add(edge[0]);
        }
        var heights = new int[n];
        for (int root = 0; root < n; root++)
        {
            var seen = new bool[n];
            seen[root] = true;
            var frontier = new List<int> { root };
            int levels = -1;
            while (frontier.Count > 0)
            {
                levels++;
                var next = new List<int>();
                foreach (int u in frontier)
                {
                    foreach (int v in adj[u])
                    {
                        if (!seen[v]) { seen[v] = true; next.Add(v); }
                    }
                }
                frontier = next;
            }
            heights[root] = levels;
        }
        int best = heights.Min();
        var centers = new List<int>();
        for (int i = 0; i < n; i++) if (heights[i] == best) centers.Add(i);
        return (centers, best);
    }

    public static void Main()
    {
        Check(string.Join(",", MinHeightTrees(6, new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 } })) == "2,3", "path of 6");
        Check(string.Join(",", MinHeightTrees(4, new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 } })) == "0", "star");
        Check(string.Join(",", MinHeightTrees(1, Array.Empty<int[]>())) == "0", "single node");
        Check(string.Join(",", MinHeightTrees(2, new[] { new[] { 0, 1 } })) == "0,1", "two nodes");

        var sample = new[] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 1, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 6 } };
        var oracle = BruteForceCenters(7, sample);
        Check(string.Join(",", MinHeightTrees(7, sample)) == string.Join(",", oracle.Centers), "matches brute force centres");
        Check(oracle.Best == 3, "minimum height is 3");
        Check(MinHeightTrees(7, sample).Count <= 2, "at most two centres");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
