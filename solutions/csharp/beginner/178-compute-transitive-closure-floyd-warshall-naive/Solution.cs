// Task 178 - Compute transitive closure (Floyd-Warshall naive). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool[][] TransitiveClosure(int[][] adj)
    {
        int n = adj.Length;
        var reach = new bool[n][];
        for (int i = 0; i < n; i++)
        {
            reach[i] = new bool[n];
            for (int j = 0; j < n; j++) reach[i][j] = adj[i][j] != 0 || i == j;
        }
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i][k])
                    for (int j = 0; j < n; j++)
                        if (reach[k][j]) reach[i][j] = true;
        return reach;
    }

    public static void Main()
    {
        var chain = new[] { new[] { 0, 1, 0 }, new[] { 0, 0, 1 }, new[] { 0, 0, 0 } };
        var reach = TransitiveClosure(chain);
        Check(reach[0][0] && reach[0][1] && reach[0][2], "0 reaches all");
        Check(!reach[1][0] && reach[1][1] && reach[1][2], "1 reaches 1,2");
        Check(!reach[2][0] && !reach[2][1] && reach[2][2], "2 only itself");

        var cycle = new[] { new[] { 0, 1, 0 }, new[] { 1, 0, 0 }, new[] { 0, 0, 0 } };
        var reach2 = TransitiveClosure(cycle);
        Check(reach2[0][1] && reach2[1][0] && !reach2[0][2], "cycle 0<->1");

        Check(TransitiveClosure(new int[0][]).Length == 0, "empty graph");
        Check(TransitiveClosure(new[] { new[] { 0 } })[0][0], "single node");

        // cross-check against DFS reachability
        var rng = new Random(3);
        int n = 8;
        var adj = new int[n][];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new int[n];
            for (int j = 0; j < n; j++)
                if (i != j && rng.NextDouble() < 0.2) adj[i][j] = 1;
        }
        var reach3 = TransitiveClosure(adj);
        for (int src = 0; src < n; src++)
        {
            var seen = new bool[n];
            var stack = new Stack<int>();
            seen[src] = true;
            stack.Push(src);
            while (stack.Count > 0)
            {
                int u = stack.Pop();
                for (int v = 0; v < n; v++)
                    if (adj[u][v] == 1 && !seen[v])
                    {
                        seen[v] = true;
                        stack.Push(v);
                    }
            }
            for (int v = 0; v < n; v++) Check(reach3[src][v] == seen[v], $"dfs agrees src={src} v={v}");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
