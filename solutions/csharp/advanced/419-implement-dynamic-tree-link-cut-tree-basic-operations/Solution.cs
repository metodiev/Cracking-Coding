// Task 419 - Implement dynamic tree (link-cut tree) basic operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A link-cut tree represents a forest of rooted trees with splay trees over preferred
// paths. Nodes are 1-based here, 0 is the null child. Arrays: left/right/parent (the
// parent entry doubles as the splay parent and the path-parent), rev (lazy subtree flip),
// val (node value) and total (splay-subtree aggregate = path sum of the preferred path).
//
// Splay details:
//   * IsRoot(x): parent[x] == 0 or x is neither the left nor the right child of parent[x]
//     — that is where a preferred path ends and a path-parent edge starts,
//   * every splay first pushes the rev flags from the splay root down to x, so rotations
//     never see stale orientations,
//   * rotations fix three families of links (b-subtree, p, g) and pull p then x.
// Operations: access (make the x-root path preferred), makeroot (access + rev toggle),
// findroot (access + descend leftmost + splay), link/cut (both guarded), connected, and
// PathSum(u, v) = makeroot(u) + access(v) -> total[v]. Ops in the tests stay in the
// hundreds; every result is validated against an explicit adjacency oracle.
using System;
using System.Collections.Generic;
using System.Linq;

internal class LinkCutTree
{
    private readonly int[] left, right, parent;
    private readonly bool[] rev;
    private readonly long[] val, total;

    public LinkCutTree(long[] values)
    {
        int n = values.Length;
        left = new int[n + 1];
        right = new int[n + 1];
        parent = new int[n + 1];
        rev = new bool[n + 1];
        val = new long[n + 1];
        total = new long[n + 1];
        for (int i = 1; i <= n; i++)
        {
            val[i] = values[i - 1];
            total[i] = values[i - 1];
        }
    }

    private bool IsRoot(int x)
    {
        int p = parent[x];
        return p == 0 || (left[p] != x && right[p] != x);
    }

    private void Push(int x)
    {
        if (rev[x])
        {
            int t = left[x];
            left[x] = right[x];
            right[x] = t;
            if (left[x] != 0) rev[left[x]] = !rev[left[x]];
            if (right[x] != 0) rev[right[x]] = !rev[right[x]];
            rev[x] = false;
        }
    }

    private void Pull(int x)
    {
        total[x] = total[left[x]] + total[right[x]] + val[x];
    }

    private void Rotate(int x)
    {
        int p = parent[x], g = parent[p];
        if (left[p] == x)
        {
            int b = right[x];
            right[x] = p;
            left[p] = b;
            if (b != 0) parent[b] = p;
        }
        else
        {
            int b = left[x];
            left[x] = p;
            right[p] = b;
            if (b != 0) parent[b] = p;
        }
        parent[p] = x;
        parent[x] = g;
        if (g != 0)
        {
            if (left[g] == p) left[g] = x;
            else if (right[g] == p) right[g] = x;
        }
        Pull(p);
        Pull(x);
    }

    private void Splay(int x)
    {
        var stack = new Stack<int>();
        stack.Push(x);
        int y = x;
        while (!IsRoot(y))
        {
            y = parent[y];
            stack.Push(y);
        }
        while (stack.Count > 0) Push(stack.Pop());
        while (!IsRoot(x))
        {
            int p = parent[x], g = parent[p];
            if (!IsRoot(p))
            {
                if ((left[p] == x) == (left[g] == p)) Rotate(p);
                else Rotate(x);
            }
            Rotate(x);
        }
    }

    private void Access(int x)
    {
        int last = 0;
        int y = x;
        while (y != 0)
        {
            Splay(y);
            right[y] = last;
            if (last != 0) parent[last] = y;
            Pull(y);
            last = y;
            y = parent[y];
        }
        Splay(x);
    }

    public void Makeroot(int x)
    {
        Access(x);
        rev[x] = !rev[x];
    }

    public int Findroot(int x)
    {
        Access(x);
        while (true)
        {
            Push(x);
            if (left[x] == 0) break;
            x = left[x];
        }
        Splay(x);
        return x;
    }

    public bool Connected(int u, int v)
    {
        if (u == v) return true;
        return Findroot(u) == Findroot(v);
    }

    public void Link(int u, int v)
    {
        Makeroot(u);
        if (Findroot(v) != u) parent[u] = v;
    }

    public void Cut(int u, int v)
    {
        Makeroot(u);
        Access(v);
        if (left[v] == u && right[u] == 0)
        {
            left[v] = 0;
            parent[u] = 0;
            Pull(v);
        }
    }

    public long PathSum(int u, int v)
    {
        Makeroot(u);
        Access(v);
        return total[v];
    }
}

/// <summary>Oracle: explicit adjacency plus BFS for connectivity and path sums.</summary>
internal class Forest
{
    private readonly List<SortedSet<int>> adj;

    public Forest(int n)
    {
        adj = new List<SortedSet<int>>();
        for (int i = 0; i <= n; i++) adj.Add(new SortedSet<int>());
    }

    public void Link(int u, int v)
    {
        adj[u].Add(v);
        adj[v].Add(u);
    }

    public void Cut(int u, int v)
    {
        adj[u].Remove(v);
        adj[v].Remove(u);
    }

    public bool HasEdge(int u, int v) => adj[u].Contains(v);

    /// <summary>Unique forest path u -> v as a vertex list, or null when disconnected.</summary>
    public int[] Path(int u, int v)
    {
        int n = adj.Count - 1;
        var prev = new int[n + 1];
        for (int i = 0; i <= n; i++) prev[i] = -1;
        prev[u] = 0;
        var queue = new Queue<int>();
        queue.Enqueue(u);
        while (queue.Count > 0)
        {
            int cur = queue.Dequeue();
            if (cur == v)
            {
                var path = new List<int>();
                int node = v;
                while (node != 0)
                {
                    path.Add(node);
                    node = prev[node];
                }
                path.Reverse();
                return path.ToArray();
            }
            foreach (int nxt in adj[cur])
            {
                if (prev[nxt] == -1)
                {
                    prev[nxt] = cur;
                    queue.Enqueue(nxt);
                }
            }
        }
        return null;
    }

    public bool Connected(int u, int v) => Path(u, v) != null;
}

public static class Solution
{
    private static int seed;

    private static int NextRand()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static long PathSumOracle(Forest forest, long[] values, int u, int v)
    {
        var path = forest.Path(u, v);
        long sum = 0;
        foreach (int node in path) sum += values[node - 1];
        return sum;
    }

    public static void Main()
    {
        var values = new long[] { 3, 1, 4, 1, 5, 9 };
        var lct = new LinkCutTree(values);

        // isolated vertices: every node is its own root, no paths connect them
        for (int v = 1; v <= 6; v++) Check(lct.Findroot(v) == v, "isolated root " + v);
        Check(!lct.Connected(1, 2), "isolated 1 2");
        Check(!lct.Connected(2, 3) && !lct.Connected(4, 5), "isolated pairs");

        // link two singletons and read the path
        lct.Link(1, 2);
        Check(lct.Connected(1, 2) && lct.Connected(2, 1), "link pair");
        Check(lct.PathSum(1, 2) == 4 && lct.PathSum(2, 1) == 4, "pair sum");   // 3 + 1
        Check(lct.Findroot(2) == lct.Findroot(1), "pair root");
        Check(lct.PathSum(1, 1) == 3, "single value");

        // extend to a chain 1 - 2 - 3 and query across it
        lct.Link(2, 3);
        Check(lct.PathSum(1, 3) == 8, "chain sum");                            // 3 + 1 + 4
        Check(lct.PathSum(3, 1) == 8, "chain sum reversed");
        Check(lct.PathSum(2, 2) == 1, "middle value");
        // PathSum calls makeroot internally; an explicit Makeroot(3) makes 3 the root
        lct.Makeroot(3);
        Check(lct.Findroot(1) == 3 && lct.Findroot(3) == 3, "makeroot 3");

        // cutting an existing edge splits the tree; missing edges are left untouched
        lct.Cut(2, 3);
        Check(!lct.Connected(1, 3), "cut 2 3");
        Check(lct.Connected(1, 2), "still linked 1 2");
        Check(lct.Findroot(3) == 3, "3 isolated after cut");
        lct.Cut(1, 3);                    // no such edge: guarded no-op
        Check(lct.Connected(1, 2) && !lct.Connected(1, 3), "cut guard");

        // build two trees, then join them through single edges
        lct.Link(3, 4);
        lct.Link(5, 6);
        Check(lct.Connected(3, 4) && lct.Connected(5, 6), "two trees");
        Check(!lct.Connected(1, 4), "trees apart");
        lct.Link(2, 3);                    // re-join {1,2} with {3,4}
        Check(lct.Connected(1, 4), "joined 1 4");
        Check(lct.PathSum(1, 4) == 9, "sum 1 4");                              // 3+1+4+1
        lct.Link(4, 5);
        Check(lct.Connected(1, 6), "joined 1 6");
        Check(lct.PathSum(1, 6) == 23, "sum 1 6");                             // 3+1+4+1+5+9
        Check(lct.PathSum(6, 1) == 23, "sum 6 1");
        Check(lct.PathSum(3, 5) == 10, "sum 3 5");                             // 4+1+5

        // cutting the bridge splits the forest again
        lct.Cut(4, 5);
        Check(lct.Connected(1, 4) && lct.Connected(5, 6), "post bridge cut");
        Check(!lct.Connected(1, 5), "bridge gone");

        // LCG-driven script against the adjacency oracle (hundreds of operations)
        int n = 30;
        seed = 4190;
        var vals = new long[n];
        for (int i = 0; i < n; i++) vals[i] = 1 + NextRand() % 100;
        lct = new LinkCutTree(vals);
        var forest = new Forest(n);

        for (int step = 0; step < 350; step++)
        {
            int kind = NextRand() % 4;
            if (kind == 0)
            {
                int u = 1 + NextRand() % n;
                int v = 1 + NextRand() % n;
                if (u != v && forest.Connected(u, v) && forest.HasEdge(u, v))
                {
                    lct.Cut(u, v);
                    forest.Cut(u, v);
                    Check(!lct.Connected(u, v), "script cut " + step);
                }
            }
            else if (kind == 1)
            {
                int u = 1 + NextRand() % n;
                int v = 1 + NextRand() % n;
                if (u != v && !forest.Connected(u, v))
                {
                    lct.Link(u, v);
                    forest.Link(u, v);
                    Check(lct.Connected(u, v), "script link " + step);
                }
            }
            else if (kind == 2)
            {
                int u = 1 + NextRand() % n;
                int v = 1 + NextRand() % n;
                Check(lct.Connected(u, v) == forest.Connected(u, v), "script conn " + step);
                if (forest.Connected(u, v))
                {
                    Check(lct.PathSum(u, v) == PathSumOracle(forest, vals, u, v), "script sum " + step);
                    Check(lct.PathSum(v, u) == lct.PathSum(u, v), "script sum sym " + step);
                }
            }
            else
            {
                // full pairwise connectivity audit
                for (int u = 1; u <= n; u++)
                {
                    for (int v = u + 1; v <= n; v++)
                    {
                        Check(lct.Connected(u, v) == forest.Connected(u, v), "audit " + step + " " + u + " " + v);
                    }
                }
            }
        }

        // final audit: every path sum across the whole forest matches the oracle
        int checks = 0;
        for (int u = 1; u <= n; u++)
        {
            for (int v = u; v <= n; v++)
            {
                Check(lct.Connected(u, v) == forest.Connected(u, v), "final conn");
                if (forest.Connected(u, v))
                {
                    Check(lct.PathSum(u, v) == PathSumOracle(forest, vals, u, v), "final sum " + u + " " + v);
                    checks++;
                }
            }
        }
        Check(checks > n, "final coverage");

        // a star built by links from one hub answers all leaf-to-leaf paths through it
        seed = 777;
        var hubValues = new long[9];
        for (int i = 0; i < 9; i++) hubValues[i] = 1 + NextRand() % 9;
        lct = new LinkCutTree(hubValues);
        for (int leaf = 2; leaf <= 9; leaf++) lct.Link(1, leaf);
        for (int leaf = 2; leaf <= 9; leaf++)
        {
            Check(lct.PathSum(1, leaf) == hubValues[0] + hubValues[leaf - 1], "star hub " + leaf);
        }
        for (int a = 2; a <= 9; a++)
        {
            for (int b = a + 1; b <= 9; b++)
            {
                long want = hubValues[0] + hubValues[a - 1] + hubValues[b - 1];
                Check(lct.PathSum(a, b) == want, "star leaf " + a + " " + b);
            }
        }
        // each path query re-roots through makeroot; after Makeroot(9), 9 is the tree root
        lct.Makeroot(9);
        Check(lct.Findroot(1) == 9 && lct.Findroot(9) == 9, "star root");

        // cutting every leaf leaves a single node
        for (int leaf = 2; leaf <= 9; leaf++) lct.Cut(1, leaf);
        for (int leaf = 2; leaf <= 9; leaf++)
        {
            Check(!lct.Connected(1, leaf), "leaf cut " + leaf);
            Check(lct.Findroot(leaf) == leaf, "leaf root " + leaf);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
