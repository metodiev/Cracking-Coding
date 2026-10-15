// Task 459 - Implement virtual tree building from set of nodes (for queries). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal static class VirtualTree
{
    public static (List<List<int>> children, int[] tin, int[] depth, int[] dist) BuildTreeData(
        int n, int[] parent, int[] edgeWeight)
    {
        var children = new List<List<int>>();
        for (int i = 0; i < n; i++) children.Add(new List<int>());
        for (int v = 1; v < n; v++) children[parent[v]].Add(v);
        var tin = new int[n];
        var depth = new int[n];
        var dist = new int[n];
        int order = 0;
        var stack = new Stack<int>();
        stack.Push(0);
        while (stack.Count > 0)
        {
            int u = stack.Pop();
            tin[u] = order++;
            for (int i = children[u].Count - 1; i >= 0; i--)
            {
                int child = children[u][i];
                depth[child] = depth[u] + 1;
                dist[child] = dist[u] + edgeWeight[child];
                stack.Push(child);
            }
        }
        return (children, tin, depth, dist);
    }

    public static int Lca(int[] parent, int[] depth, int u, int v)
    {
        while (depth[u] > depth[v]) u = parent[u];
        while (depth[v] > depth[u]) v = parent[v];
        while (u != v)
        {
            u = parent[u];
            v = parent[v];
        }
        return u;
    }

    public static bool IsAncestor(int[] parent, int candidate, int node)
    {
        while (node != -1)
        {
            if (node == candidate) return true;
            node = parent[node];
        }
        return false;
    }

    public static int NaiveDistance(int[] parent, int[] depth, int[] dist, int u, int v)
        => dist[u] + dist[v] - 2 * dist[Lca(parent, depth, u, v)];

    /// <summary>Returns (nodes sorted by index, edges as {vtParent, vtChild, weight}).</summary>
    public static (int[] nodes, List<int[]> edges) Build(int[] parent, int[] depth, int[] dist, int[] tin, int[] selected)
    {
        var order = new List<int>(selected);
        order.Sort((a, b) => tin[a].CompareTo(tin[b]));
        var edges = new List<int[]>();
        if (order.Count == 0) return (new int[0], edges);
        var stack = new List<int> { order[0] };
        for (int idx = 1; idx < order.Count; idx++)
        {
            int v = order[idx];
            int l = Lca(parent, depth, v, stack[stack.Count - 1]);
            if (l == stack[stack.Count - 1])
            {
                stack.Add(v);
                continue;
            }
            while (stack.Count >= 2 && depth[stack[stack.Count - 2]] >= depth[l])
            {
                int child = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                int par = stack[stack.Count - 1];
                edges.Add(new[] { par, child, dist[child] - dist[par] });
            }
            if (stack[stack.Count - 1] != l)
            {
                int child = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                edges.Add(new[] { l, child, dist[child] - dist[l] });
                stack.Add(l);
            }
            stack.Add(v);
        }
        while (stack.Count >= 2)
        {
            int child = stack[stack.Count - 1];
            stack.RemoveAt(stack.Count - 1);
            int par = stack[stack.Count - 1];
            edges.Add(new[] { par, child, dist[child] - dist[par] });
        }
        var nodeSet = new SortedSet<int>();
        foreach (int v in selected) nodeSet.Add(v);
        foreach (var e in edges)
        {
            nodeSet.Add(e[0]);
            nodeSet.Add(e[1]);
        }
        var nodes = new int[nodeSet.Count];
        int i = 0;
        foreach (int v in nodeSet) nodes[i++] = v;
        return (nodes, edges);
    }

    public static int VirtualTreePathWeight(int[] nodes, List<int[]> edges, int source, int target)
    {
        var adj = new Dictionary<int, List<int[]>>();
        foreach (int v in nodes) adj[v] = new List<int[]>();
        foreach (var e in edges)
        {
            adj[e[0]].Add(new[] { e[1], e[2] });
            adj[e[1]].Add(new[] { e[0], e[2] });
        }
        var seen = new HashSet<int> { source };
        var stack = new Stack<int[]>();
        stack.Push(new[] { source, 0 });
        while (stack.Count > 0)
        {
            var cur = stack.Pop();
            if (cur[0] == target) return cur[1];
            foreach (var nxt in adj[cur[0]])
            {
                if (!seen.Contains(nxt[0]))
                {
                    seen.Add(nxt[0]);
                    stack.Push(new[] { nxt[0], cur[1] + nxt[1] });
                }
            }
        }
        return -1;
    }

    public static void CheckVirtualTree(int n, int[] parent, int[] selected,
        int[] tin, int[] depth, int[] dist)
    {
        var (nodes, edges) = Build(parent, depth, dist, tin, selected);
        var nodeSet = new HashSet<int>(nodes);
        foreach (int s in selected)
        {
            if (!nodeSet.Contains(s)) throw new Exception("FAILED: selected present");
        }
        foreach (int v in nodes)
        {
            bool ok = false;
            foreach (int s in selected) ok |= IsAncestor(parent, v, s);
            if (!ok) throw new Exception("FAILED: ancestor-or-selected");
        }
        if (edges.Count != nodes.Length - 1) throw new Exception("FAILED: tree edge count");
        if (nodes.Length > 2 * selected.Length - 1) throw new Exception("FAILED: node bound");
        foreach (var e in edges)
        {
            if (e[2] != dist[e[1]] - dist[e[0]] || e[2] <= 0) throw new Exception("FAILED: edge weight");
        }
        for (int i = 0; i < selected.Length; i++)
        {
            for (int j = i + 1; j < selected.Length; j++)
            {
                int got = VirtualTreePathWeight(nodes, edges, selected[i], selected[j]);
                int want = NaiveDistance(parent, depth, dist, selected[i], selected[j]);
                if (got != want) throw new Exception("FAILED: pair distance " + selected[i] + "," + selected[j]);
            }
        }
    }

    public static List<(int n, int[] parent, int[] edgeWeight)> LcgTrees(int count, int seed)
    {
        var trees = new List<(int, int[], int[])>();
        for (int g = 0; g < count; g++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 8 + seed % 5;
            var parent = new int[n];
            var edgeWeight = new int[n];
            parent[0] = -1;
            for (int v = 1; v < n; v++)
            {
                seed = (seed * 75 + 74) % 65537;
                parent[v] = seed % v;
                seed = (seed * 75 + 74) % 65537;
                edgeWeight[v] = 1 + seed % 10;
            }
            trees.Add((n, parent, edgeWeight));
        }
        return trees;
    }

    public static int[] LcgSelection(int n, int seed, int size)
    {
        seed = (seed * 75 + 74) % 65537;
        var pool = new List<int>();
        for (int i = 0; i < n; i++) pool.Add(i);
        int count = Math.Min(size, n);
        var selected = new List<int>();
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            int idx = seed % pool.Count;
            selected.Add(pool[idx]);
            pool[idx] = pool[pool.Count - 1];
            pool.RemoveAt(pool.Count - 1);
            seed = (seed * 75 + 74) % 65537;
        }
        return selected.ToArray();
    }
}

public static class Solution
{
    public static void Main()
    {
        var chainParent = new[] { -1, 0, 1, 2, 3 };
        var chainWeight = new[] { 0, 1, 1, 1, 1 };
        var (_, chainTin, chainDepth, chainDist) = VirtualTree.BuildTreeData(5, chainParent, chainWeight);
        VirtualTree.CheckVirtualTree(5, chainParent, new[] { 2, 4 }, chainTin, chainDepth, chainDist);
        var built = VirtualTree.Build(chainParent, chainDepth, chainDist, chainTin, new[] { 2, 4 });
        Check(built.nodes.Length == 2 && built.nodes[0] == 2 && built.nodes[1] == 4, "chain nodes");
        Check(built.edges.Count == 1 && built.edges[0][2] == 2, "chain edge weight 2");
        VirtualTree.CheckVirtualTree(5, chainParent, new[] { 1, 3 }, chainTin, chainDepth, chainDist);
        VirtualTree.CheckVirtualTree(5, chainParent, new[] { 0, 3 }, chainTin, chainDepth, chainDist);
        VirtualTree.CheckVirtualTree(5, chainParent, new[] { 1 }, chainTin, chainDepth, chainDist);
        built = VirtualTree.Build(chainParent, chainDepth, chainDist, chainTin, new[] { 1 });
        Check(built.nodes.Length == 1 && built.edges.Count == 0, "single node no edges");

        var starParent = new[] { -1, 0, 0, 0 };
        var starWeight = new[] { 0, 1, 1, 1 };
        var (_, starTin, starDepth, starDist) = VirtualTree.BuildTreeData(4, starParent, starWeight);
        VirtualTree.CheckVirtualTree(4, starParent, new[] { 1, 2 }, starTin, starDepth, starDist);
        built = VirtualTree.Build(starParent, starDepth, starDist, starTin, new[] { 1, 2 });
        Check(built.nodes.Length == 3 && built.nodes[0] == 0, "star adds LCA root");
        VirtualTree.CheckVirtualTree(4, starParent, new[] { 1, 2, 3 }, starTin, starDepth, starDist);

        var balParent = new[] { -1, 0, 0, 1, 1, 2, 2 };
        var balWeight = new[] { 0, 3, 4, 5, 6, 7, 8 };
        var (_, balTin, balDepth, balDist) = VirtualTree.BuildTreeData(7, balParent, balWeight);
        VirtualTree.CheckVirtualTree(7, balParent, new[] { 3, 4, 5, 6 }, balTin, balDepth, balDist);

        int checkedCount = 0, trees = 0;
        foreach (int seed in new[] { 4591, 4592, 4593, 4594 })
        {
            foreach (var (n, parent, edgeWeight) in VirtualTree.LcgTrees(4, seed))
            {
                var (_, tin, depth, dist) = VirtualTree.BuildTreeData(n, parent, edgeWeight);
                int seed2 = seed + n;
                for (int extra = 0; extra < 2; extra++)
                {
                    int size = 2 + seed2 % 4;
                    var selected = VirtualTree.LcgSelection(n, seed2 + extra * 3, size);
                    VirtualTree.CheckVirtualTree(n, parent, selected, tin, depth, dist);
                    checkedCount++;
                }
                trees++;
            }
        }
        Check(trees == 16, "tree count");
        Check(checkedCount == 32, "selection count");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
