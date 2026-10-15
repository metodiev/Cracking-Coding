// Task 453 - Implement circulation with lower bounds (flow transform). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

/// <summary>Edmonds-Karp on an explicit arc list; each arc keeps its own reverse residual.</summary>
internal class MaxFlow
{
    private readonly int n;
    private readonly List<int> to = new List<int>();
    private readonly List<int> initial = new List<int>();
    private readonly List<int> cap = new List<int>();
    private readonly List<List<int>> head;

    public MaxFlow(int n)
    {
        this.n = n;
        head = new List<List<int>>();
        for (int i = 0; i < n; i++) head.Add(new List<int>());
    }

    public int AddEdge(int u, int v, int capacity)
    {
        int idx = to.Count;
        to.Add(v);
        initial.Add(capacity);
        cap.Add(capacity);
        head[u].Add(idx);
        to.Add(u);
        initial.Add(0);
        cap.Add(0);
        head[v].Add(idx + 1);
        return idx;
    }

    public int Used(int idx) => initial[idx] - cap[idx];

    public int MaxFlowValue(int s, int t)
    {
        int flow = 0;
        while (true)
        {
            var parentEdge = new int[n];
            for (int i = 0; i < n; i++) parentEdge[i] = -1;
            parentEdge[s] = -2;
            var q = new Queue<int>();
            q.Enqueue(s);
            while (q.Count > 0)
            {
                int u = q.Dequeue();
                foreach (int idx in head[u])
                {
                    int v = to[idx];
                    if (parentEdge[v] == -1 && cap[idx] > 0)
                    {
                        parentEdge[v] = idx;
                        q.Enqueue(v);
                    }
                }
            }
            if (parentEdge[t] == -1) return flow;
            int aug = int.MaxValue / 4;
            for (int v = t; v != s; v = to[parentEdge[v] ^ 1]) aug = Math.Min(aug, cap[parentEdge[v]]);
            for (int v = t; v != s; v = to[parentEdge[v] ^ 1])
            {
                int idx = parentEdge[v];
                cap[idx] -= aug;
                cap[idx ^ 1] += aug;
            }
            flow += aug;
        }
    }
}

internal static class Circulation
{
    /// <summary>Edge flows for edges = {u, v, lower, upper}, or null when infeasible.</summary>
    public static int[] Solve(int n, List<int[]> edges)
    {
        int ss = n, tt = n + 1;
        var network = new MaxFlow(n + 2);
        var balance = new int[n];
        var arcIds = new List<int>();
        foreach (var e in edges)
        {
            int u = e[0], v = e[1], lower = e[2], upper = e[3];
            balance[v] += lower;
            balance[u] -= lower;
            arcIds.Add(network.AddEdge(u, v, upper - lower));
        }
        int required = 0;
        for (int v = 0; v < n; v++)
        {
            if (balance[v] > 0)
            {
                network.AddEdge(ss, v, balance[v]);
                required += balance[v];
            }
            else if (balance[v] < 0)
            {
                network.AddEdge(v, tt, -balance[v]);
            }
        }
        if (network.MaxFlowValue(ss, tt) != required) return null;
        var flows = new int[edges.Count];
        for (int i = 0; i < edges.Count; i++) flows[i] = edges[i][2] + network.Used(arcIds[i]);
        return flows;
    }

    /// <summary>s-t flow of exactly `requiredValue` via the t->s [F, F] circulation transform.</summary>
    public static int[] SolveWithRequiredValue(int n, List<int[]> edges, int s, int t, int requiredValue)
    {
        var extended = new List<int[]>(edges) { new[] { t, s, requiredValue, requiredValue } };
        return Solve(n, extended);
    }

    /// <summary>Enumerate every integer flow assignment in [lower, upper] and test conservation.</summary>
    public static bool BruteFeasible(int n, List<int[]> edges)
    {
        int m = edges.Count;
        var values = new int[m];
        while (true)
        {
            var balance = new int[n];
            for (int i = 0; i < m; i++)
            {
                int f = edges[i][2] + values[i];
                balance[edges[i][0]] -= f;
                balance[edges[i][1]] += f;
            }
            bool ok = true;
            foreach (int b in balance) ok &= b == 0;
            if (ok) return true;
            int j = 0;
            while (j < m)
            {
                if (values[j] + 1 < edges[j][3] - edges[j][2] + 1)
                {
                    values[j]++;
                    break;
                }
                values[j] = 0;
                j++;
            }
            if (j == m) return false;
        }
    }

    public static List<(int n, List<int[]> edges)> LcgInstances(int count, int seed)
    {
        var instances = new List<(int, List<int[]>)>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 3 + seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int m = 3 + seed % 3;
            var pool = new List<int[]>();
            for (int u = 0; u < n; u++)
                for (int v = 0; v < n; v++)
                    if (u != v) pool.Add(new[] { u, v });
            var edges = new List<int[]>();
            for (int k = 0; k < Math.Min(m, pool.Count); k++)
            {
                seed = (seed * 75 + 74) % 65537;
                int idx = seed % pool.Count;
                var uv = pool[idx];
                pool[idx] = pool[pool.Count - 1];
                pool.RemoveAt(pool.Count - 1);
                seed = (seed * 75 + 74) % 65537;
                int lower = seed % 3;
                seed = (seed * 75 + 74) % 65537;
                int upper = lower + seed % 3;
                edges.Add(new[] { uv[0], uv[1], lower, upper });
            }
            instances.Add((n, edges));
        }
        return instances;
    }
}

public static class Solution
{
    public static void Main()
    {
        var triangle = new List<int[]> { new[] { 0, 1, 1, 3 }, new[] { 1, 2, 1, 3 }, new[] { 2, 0, 1, 3 } };
        int[] flows = Circulation.Solve(3, triangle);
        Check(flows != null && flows[0] == 1 && flows[1] == 1 && flows[2] == 1, "triangle flows");
        CheckFlowValid(3, triangle, flows);
        Check(Circulation.BruteFeasible(3, triangle), "triangle brute");

        var tightOut = new List<int[]> { new[] { 0, 1, 2, 3 }, new[] { 1, 2, 0, 1 }, new[] { 2, 0, 1, 3 } };
        Check(Circulation.Solve(3, tightOut) == null, "tight out infeasible");
        Check(!Circulation.BruteFeasible(3, tightOut), "tight out brute");

        var twoWay = new List<int[]> { new[] { 0, 1, 3, 5 }, new[] { 1, 0, 0, 5 } };
        flows = Circulation.Solve(2, twoWay);
        Check(flows != null && flows[0] == 3 && flows[1] == 3, "two way flows");
        CheckFlowValid(2, twoWay, flows);
        Check(Circulation.BruteFeasible(2, twoWay), "two way brute");

        var tooMuchIn = new List<int[]> { new[] { 0, 1, 4, 5 }, new[] { 1, 0, 0, 2 } };
        Check(Circulation.Solve(2, tooMuchIn) == null, "too much in infeasible");
        Check(!Circulation.BruteFeasible(2, tooMuchIn), "too much in brute");

        var disjoint = new List<int[]> { new[] { 0, 1, 1, 2 }, new[] { 1, 0, 1, 2 }, new[] { 2, 3, 0, 1 }, new[] { 3, 2, 0, 1 } };
        flows = Circulation.Solve(4, disjoint);
        Check(flows != null, "disjoint feasible");
        CheckFlowValid(4, disjoint, flows);
        Check(Circulation.BruteFeasible(4, disjoint), "disjoint brute");

        var deadEnd = new List<int[]> { new[] { 0, 1, 1, 1 } };
        Check(Circulation.Solve(2, deadEnd) == null, "dead end infeasible");
        Check(!Circulation.BruteFeasible(2, deadEnd), "dead end brute");

        var withIsolated = new List<int[]> { new[] { 0, 1, 1, 1 }, new[] { 1, 0, 1, 1 } };
        flows = Circulation.Solve(3, withIsolated);
        Check(flows != null && flows[0] == 1 && flows[1] == 1, "isolated flows");
        CheckFlowValid(3, withIsolated, flows);
        Check(Circulation.BruteFeasible(3, withIsolated), "isolated brute");

        var stEdges = new List<int[]> { new[] { 0, 1, 0, 2 }, new[] { 1, 3, 0, 2 }, new[] { 0, 2, 0, 1 }, new[] { 2, 3, 0, 1 } };
        flows = Circulation.SolveWithRequiredValue(4, stEdges, 0, 3, 3);
        Check(flows != null && flows[flows.Length - 1] == 3, "s-t value 3 feasible");
        var extended = new List<int[]>(stEdges) { new[] { 3, 0, 3, 3 } };
        CheckFlowValid(4, extended, flows);
        Check(Circulation.SolveWithRequiredValue(4, stEdges, 0, 3, 4) == null, "s-t value 4 infeasible");

        int feasibleCount = 0, infeasibleCount = 0;
        foreach (int seed in new[] { 4531, 4532, 4533 })
        {
            foreach (var (n, edges) in Circulation.LcgInstances(8, seed))
            {
                flows = Circulation.Solve(n, edges);
                bool expected = Circulation.BruteFeasible(n, edges);
                Check((flows != null) == expected, "LCG feasibility");
                if (flows != null)
                {
                    CheckFlowValid(n, edges, flows);
                    feasibleCount++;
                }
                else
                {
                    infeasibleCount++;
                }
            }
        }
        Check(feasibleCount >= 1 && infeasibleCount >= 1, "both outcomes present");
        Check(feasibleCount + infeasibleCount == 24, "instance count");
        Console.WriteLine("PASS");
    }

    private static void CheckFlowValid(int n, List<int[]> edges, int[] flows)
    {
        var balance = new int[n];
        for (int i = 0; i < edges.Count; i++)
        {
            int lower = edges[i][2], upper = edges[i][3];
            Check(lower <= flows[i] && flows[i] <= upper, "flow within bounds");
            balance[edges[i][0]] -= flows[i];
            balance[edges[i][1]] += flows[i];
        }
        foreach (int b in balance) Check(b == 0, "conservation");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
