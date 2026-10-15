// Task 106 - Topological sort using DFS (DAG). (Tier: Beginner, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const int White = 0, Gray = 1, Black = 2;

    public static List<int>? TopoSort(Dictionary<int, List<int>> graph)
    {
        var color = new Dictionary<int, int>();
        foreach (var v in graph.Keys) color[v] = White;
        var order = new List<int>();
        foreach (var v in graph.Keys)
        {
            if (color[v] == White && !Visit(v, graph, color, order)) return null;
        }
        order.Reverse();
        return order;
    }

    private static bool Visit(int v, Dictionary<int, List<int>> graph, Dictionary<int, int> color, List<int> order)
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
                if (c == Gray) return false;
                if (c == White && !Visit(w, graph, color, order)) return false;
            }
        }
        color[v] = Black;
        order.Add(v);
        return true;
    }

    private static bool IsValidTopo(Dictionary<int, List<int>> graph, List<int>? order)
    {
        if (order == null || order.Count != graph.Count) return false;
        var index = new Dictionary<int, int>();
        for (int i = 0; i < order.Count; i++) index[order[i]] = i;
        foreach (var kv in graph)
        {
            if (!index.TryGetValue(kv.Key, out int ui)) return false;
            foreach (int w in kv.Value)
            {
                if (!index.TryGetValue(w, out int wi) || ui >= wi) return false;
            }
        }
        return true;
    }

    public static void Main()
    {
        var courses = new Dictionary<int, List<int>>
        {
            [0] = new List<int> { 1, 2 },
            [1] = new List<int> { 3 },
            [2] = new List<int> { 3 },
            [3] = new List<int>(),
        };
        var order = TopoSort(courses);
        Check(order != null && SameList(order, 0, 2, 1, 3), "course prerequisites");
        Check(IsValidTopo(courses, order), "order respects every edge");

        var cycle = new Dictionary<int, List<int>>
        {
            [0] = new List<int> { 1 },
            [1] = new List<int> { 2 },
            [2] = new List<int> { 0 },
        };
        Check(TopoSort(cycle) == null, "cycle detected");

        var selfLoop = new Dictionary<int, List<int>> { [0] = new List<int> { 0 } };
        Check(TopoSort(selfLoop) == null, "self-loop detected");

        Check(TopoSort(new Dictionary<int, List<int>>())?.Count == 0, "empty graph");
        var single = new Dictionary<int, List<int>> { [9] = new List<int>() };
        Check(SameList(TopoSort(single), 9), "single vertex");

        var forest = new Dictionary<int, List<int>>
        {
            [0] = new List<int> { 1 },
            [1] = new List<int>(),
            [2] = new List<int> { 3 },
            [3] = new List<int>(),
        };
        var forestOrder = TopoSort(forest);
        Check(forestOrder != null && forestOrder.Count == 4 && IsValidTopo(forest, forestOrder), "disconnected DAG");

        var chained = new Dictionary<int, List<int>>
        {
            [0] = new List<int>(),
            [1] = new List<int> { 0 },
            [2] = new List<int> { 1 },
            [3] = new List<int> { 2 },
        };
        var chainOrder = TopoSort(chained);
        Check(SameList(chainOrder, 3, 2, 1, 0) && IsValidTopo(chained, chainOrder), "chain");
        Console.WriteLine("PASS");
    }

    private static bool SameList(List<int>? actual, params int[] expected)
    {
        if (actual == null || actual.Count != expected.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (actual[i] != expected[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
