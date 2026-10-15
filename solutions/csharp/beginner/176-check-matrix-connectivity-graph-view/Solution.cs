// Task 176 - Check matrix connectivity (graph view). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static bool IsConnected(int[][] adj)
    {
        int n = adj.Length;
        if (n <= 1) return true;
        var seen = new bool[n];
        seen[0] = true;
        var q = new Queue<int>();
        q.Enqueue(0);
        int reached = 1;
        while (q.Count > 0)
        {
            int u = q.Dequeue();
            for (int v = 0; v < n; v++)
                if (adj[u][v] == 1 && !seen[v])
                {
                    seen[v] = true;
                    reached++;
                    q.Enqueue(v);
                }
        }
        return reached == n;
    }

    public static int ConnectedComponents(int[][] adj)
    {
        int n = adj.Length;
        var seen = new bool[n];
        int components = 0;
        for (int start = 0; start < n; start++)
        {
            if (seen[start]) continue;
            components++;
            seen[start] = true;
            var q = new Queue<int>();
            q.Enqueue(start);
            while (q.Count > 0)
            {
                int u = q.Dequeue();
                for (int v = 0; v < n; v++)
                    if (adj[u][v] == 1 && !seen[v])
                    {
                        seen[v] = true;
                        q.Enqueue(v);
                    }
            }
        }
        return components;
    }

    public static void Main()
    {
        var path = new[] { new[] { 0, 1, 0, 0 }, new[] { 1, 0, 1, 0 }, new[] { 0, 1, 0, 1 }, new[] { 0, 0, 1, 0 } };
        Check(IsConnected(path) && ConnectedComponents(path) == 1, "path graph");

        var split = new[] { new[] { 0, 1, 0, 0 }, new[] { 1, 0, 0, 0 }, new[] { 0, 0, 0, 1 }, new[] { 0, 0, 1, 0 } };
        Check(!IsConnected(split) && ConnectedComponents(split) == 2, "two pairs");

        var lonely = new[] { new[] { 0, 1, 1, 0 }, new[] { 1, 0, 0, 0 }, new[] { 1, 0, 0, 0 }, new[] { 0, 0, 0, 0 } };
        Check(!IsConnected(lonely) && ConnectedComponents(lonely) == 2, "isolated vertex");

        Check(IsConnected(new int[0][]) && IsConnected(new[] { new[] { 0 } }), "trivial graphs");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
