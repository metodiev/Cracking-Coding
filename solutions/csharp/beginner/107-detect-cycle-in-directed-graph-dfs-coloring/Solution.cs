// Task 107 - Detect cycle in directed graph (DFS coloring). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const int White = 0, Gray = 1, Black = 2;

    public static bool HasCycle(Dictionary<int, List<int>> graph)
    {
        var color = new Dictionary<int, int>();
        foreach (var v in graph.Keys) color[v] = White;
        foreach (var v in graph.Keys)
        {
            if (color[v] == White && Visit(v, graph, color)) return true;
        }
        return false;
    }

    private static bool Visit(int v, Dictionary<int, List<int>> graph, Dictionary<int, int> color)
    {
        color[v] = Gray;
        if (graph.TryGetValue(v, out var neighbors))
        {
            foreach (int w in neighbors)
            {
                if (!color.TryGetValue(w, out int c))
                {
                    c = White;
                    color[w] = White;
                }
                if (c == Gray) return true;
                if (c == White && Visit(w, graph, color)) return true;
            }
        }
        color[v] = Black;
        return false;
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

    public static void Main()
    {
        Check(!HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2, 3 } })), "DAG");
        Check(!HasCycle(new Dictionary<int, List<int>>()), "empty graph");
        Check(!HasCycle(new Dictionary<int, List<int>> { [7] = new List<int>() }), "single vertex");
        Check(HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 0 } })), "simple cycle");
        Check(HasCycle(new Dictionary<int, List<int>> { [0] = new List<int> { 0 } }), "self-loop");
        Check(HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 2, 3 }, new[] { 3, 2 } })), "cycle in one component");
        Check(!HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 2, 3 } })), "disconnected acyclic");
        Check(HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 1 } })), "back edge into gray");
        Check(!HasCycle(MakeGraph(new int[][] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2, 3 } })), "diamond is acyclic");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
