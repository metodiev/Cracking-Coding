// Task 448 - Implement parametric search for optimizing monotone predicates. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal delegate bool IntPred(int value);

internal class WeightedGraph
{
    public readonly int N;
    public readonly int[][] Edges;

    public WeightedGraph(int n, int[][] edges)
    {
        N = n;
        Edges = edges;
    }
}

internal static class Parametric
{
    public static int? ParametricMin(int[] candidates, IntPred predicate)
    {
        int lo = 0, hi = candidates.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (predicate(candidates[mid])) hi = mid;
            else lo = mid + 1;
        }
        return lo < candidates.Length ? candidates[lo] : (int?)null;
    }

    public static int? LinearScan(int[] candidates, IntPred predicate)
    {
        foreach (int candidate in candidates)
        {
            if (predicate(candidate)) return candidate;
        }
        return null;
    }

    public static bool ReachableAll(int n, int[][] edges, int d, int source)
    {
        var adjacency = new List<int>[n];
        for (int i = 0; i < n; i++) adjacency[i] = new List<int>();
        foreach (int[] e in edges)
        {
            if (e[2] <= d)
            {
                adjacency[e[0]].Add(e[1]);
                adjacency[e[1]].Add(e[0]);
            }
        }
        var seen = new bool[n];
        seen[source] = true;
        var stack = new Stack<int>();
        stack.Push(source);
        while (stack.Count > 0)
        {
            int x = stack.Pop();
            foreach (int y in adjacency[x])
            {
                if (!seen[y])
                {
                    seen[y] = true;
                    stack.Push(y);
                }
            }
        }
        foreach (bool flag in seen)
        {
            if (!flag) return false;
        }
        return true;
    }

    public static bool ComponentsLeq(int n, int[][] edges, int d, int k)
    {
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int components = n;
        foreach (int[] e in edges)
        {
            if (e[2] > d) continue;
            int ru = Find(parent, e[0]);
            int rv = Find(parent, e[1]);
            if (ru != rv)
            {
                parent[ru] = rv;
                components--;
            }
        }
        return components <= k;
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public static void AssertMonotone(int[] candidates, IntPred predicate)
    {
        bool seenTrue = false;
        foreach (int candidate in candidates)
        {
            bool value = predicate(candidate);
            if (value) seenTrue = true;
            else if (seenTrue) throw new Exception("FAILED: predicate not monotone at " + candidate);
        }
    }
}

public static class Solution
{
    private static int seed;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<WeightedGraph> LcgGraphs(int count, int maxN, int baseSeed)
    {
        seed = baseSeed;
        var graphs = new List<WeightedGraph>();
        for (int g = 0; g < count; g++)
        {
            int n = 3 + seed % (maxN - 2);
            var edges = new List<int[]>();
            for (int node = 1; node < n; node++)
            {
                int parent = NextSeed() % node;
                int weight = 1 + NextSeed() % 30;
                edges.Add(new[] { parent, node, weight });
            }
            for (int extra = 0; extra < n; extra++)
            {
                int u = NextSeed() % n;
                int v = NextSeed() % n;
                int weight = 1 + NextSeed() % 30;
                if (u != v) edges.Add(new[] { u, v, weight });
            }
            graphs.Add(new WeightedGraph(n, edges.ToArray()));
        }
        return graphs;
    }

    private static int[] CandidatesOf(int[][] edges)
    {
        var distinct = new SortedSet<int>();
        foreach (int[] e in edges) distinct.Add(e[2]);
        var candidates = new int[distinct.Count];
        int idx = 0;
        foreach (int value in distinct) candidates[idx++] = value;
        return candidates;
    }

    public static void Main()
    {
        int[][] path = { new[] { 0, 1, 4 }, new[] { 1, 2, 1 }, new[] { 2, 3, 7 } };
        int[] candidates = { 1, 4, 7 };
        Check(Parametric.ParametricMin(candidates, d => Parametric.ReachableAll(4, path, d, 0)) == 7, "bottleneck path");
        Check(Parametric.ParametricMin(candidates, d => Parametric.ComponentsLeq(4, path, d, 2)) == 4, "two components");
        Check(Parametric.ParametricMin(candidates, d => Parametric.ComponentsLeq(4, path, d, 1)) == 7, "single component");
        Parametric.AssertMonotone(candidates, d => Parametric.ReachableAll(4, path, d, 0));
        Parametric.AssertMonotone(candidates, d => Parametric.ComponentsLeq(4, path, d, 2));

        int[][] edgeCase = { new[] { 0, 1, 5 } };
        Check(Parametric.ParametricMin(new[] { 5 }, d => Parametric.ReachableAll(2, edgeCase, d, 0)) == 5, "single edge");
        Check(Parametric.ParametricMin(new[] { 1, 2, 3 }, d => false) == null, "never true");
        Check(Parametric.ParametricMin(new[] { 1, 2, 3 }, d => true) == 1, "always true");

        int checkedCount = 0;
        foreach (WeightedGraph graph in LcgGraphs(20, 12, 448))
        {
            int[] graphCandidates = CandidatesOf(graph.Edges);
            foreach (int source in new[] { 0, graph.N - 1 })
            {
                IntPred predicate = d => Parametric.ReachableAll(graph.N, graph.Edges, d, source);
                Parametric.AssertMonotone(graphCandidates, predicate);
                Check(Parametric.ParametricMin(graphCandidates, predicate) == Parametric.LinearScan(graphCandidates, predicate),
                        "reachable source " + source);
                checkedCount++;
            }
            foreach (int k in new[] { 1, 2, 4 })
            {
                IntPred predicate = d => Parametric.ComponentsLeq(graph.N, graph.Edges, d, k);
                Parametric.AssertMonotone(graphCandidates, predicate);
                Check(Parametric.ParametricMin(graphCandidates, predicate) == Parametric.LinearScan(graphCandidates, predicate),
                        "components <= " + k);
                checkedCount++;
            }
        }
        Check(checkedCount == 100, "coverage count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
