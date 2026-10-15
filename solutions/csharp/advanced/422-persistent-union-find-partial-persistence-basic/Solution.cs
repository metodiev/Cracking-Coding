// Task 422 - Persistent union-find (partial persistence) basic. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class PersistentUnionFind
{
    public readonly int N;
    private readonly List<List<int[]>> parentHistory = new List<List<int[]>>();
    private readonly List<List<int[]>> sizeHistory = new List<List<int[]>>();

    public PersistentUnionFind(int n)
    {
        N = n;
        for (int i = 0; i < n; i++)
        {
            parentHistory.Add(new List<int[]>());
            sizeHistory.Add(new List<int[]>());
        }
    }

    private static int? ValueAt(List<int[]> history, int time)
    {
        int lo = 0, hi = history.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (history[mid][0] <= time) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? (int?)null : history[lo - 1][1];
    }

    public int Find(int node, int time)
    {
        while (true)
        {
            int? parent = ValueAt(parentHistory[node], time);
            if (parent == null) return node;
            node = parent.Value;
        }
    }

    public int Size(int node, int time)
    {
        int? recorded = ValueAt(sizeHistory[node], time);
        return recorded ?? 1;
    }

    public bool Union(int a, int b, int time)
    {
        int rootA = Find(a, time - 1);
        int rootB = Find(b, time - 1);
        if (rootA == rootB) return false;
        int sizeA = Size(rootA, time - 1);
        int sizeB = Size(rootB, time - 1);
        if (sizeA < sizeB)
        {
            int tmp = rootA; rootA = rootB; rootB = tmp;
            tmp = sizeA; sizeA = sizeB; sizeB = tmp;
        }
        parentHistory[rootB].Add(new[] { time, rootA });
        sizeHistory[rootA].Add(new[] { time, sizeA + sizeB });
        return true;
    }

    public int ComponentCount(int time)
    {
        var roots = new HashSet<int>();
        for (int node = 0; node < N; node++) roots.Add(Find(node, time));
        return roots.Count;
    }
}

public static class Solution
{
    private static int RootOf(int[] parent, int node)
    {
        while (parent[node] != node) node = parent[node];
        return node;
    }

    private static int[] Replay(int n, int[][] operations, int time)
    {
        var parent = new int[n];
        var size = new int[n];
        for (int i = 0; i < n; i++) { parent[i] = i; size[i] = 1; }
        for (int step = 1; step <= time; step++)
        {
            int rootA = RootOf(parent, operations[step - 1][0]);
            int rootB = RootOf(parent, operations[step - 1][1]);
            if (rootA != rootB)
            {
                if (size[rootA] < size[rootB]) { int tmp = rootA; rootA = rootB; rootB = tmp; }
                parent[rootB] = rootA;
                size[rootA] += size[rootB];
            }
        }
        return parent;
    }

    private static int[] OracleRoots(int n, int[][] operations, int time)
    {
        int[] parent = Replay(n, operations, time);
        var roots = new int[n];
        for (int node = 0; node < n; node++) roots[node] = RootOf(parent, node);
        return roots;
    }

    public static void Main()
    {
        var operations = new[] { new[] { 0, 1 }, new[] { 2, 3 }, new[] { 1, 3 }, new[] { 0, 2 } };
        var uf = new PersistentUnionFind(4);
        var merges = new bool[4];
        for (int time = 0; time < 4; time++)
            merges[time] = uf.Union(operations[time][0], operations[time][1], time + 1);
        Check(merges[0] && merges[1] && merges[2] && !merges[3], "merge flags");

        Check(uf.ComponentCount(0) == 4, "time 0 count");
        Check(uf.ComponentCount(1) == 3, "time 1 count");
        Check(uf.ComponentCount(2) == 2, "time 2 count");
        Check(uf.ComponentCount(3) == 1, "time 3 count");
        Check(uf.ComponentCount(4) == 1, "time 4 count");

        Check(uf.Find(1, 1) == 0 && uf.Find(0, 1) == 0, "first union representative");
        Check(uf.Find(3, 1) == 3, "second union not yet applied");
        Check(uf.Find(3, 2) == 2, "after second union");
        Check(uf.Find(3, 3) == 0, "after third union");
        Check(uf.Find(3, 4) == 0, "after fourth union");
        Check(uf.Find(2, 2) == 2, "own root early");
        Check(uf.Find(1, 0) == 1 && uf.Find(2, 0) == 2, "pre-union representatives");

        for (int time = 0; time <= 4; time++)
        {
            int[] expected = OracleRoots(4, operations, time);
            var distinct = new HashSet<int>(expected);
            Check(uf.ComponentCount(time) == distinct.Count, "oracle count at " + time);
            for (int node = 0; node < 4; node++)
                Check(uf.Find(node, time) == expected[node], "oracle representative " + time + "/" + node);
        }

        foreach (int seed in new[] { 422, 31415, 271828 })
        {
            int n = 2 + seed % 9;
            var structure = new PersistentUnionFind(n);
            var ops = new int[40][];
            int local = seed;
            for (int i = 0; i < 40; i++)
            {
                local = (local * 75 + 74) % 65537;
                int a = local % n;
                local = (local * 75 + 74) % 65537;
                ops[i] = new[] { a, local % n };
            }
            int applied = 0;
            for (int time = 1; time <= 40; time++)
            {
                if (structure.Union(ops[time - 1][0], ops[time - 1][1], time)) applied++;
                Check(structure.ComponentCount(time) == n - applied, "lcg count");
            }
            for (int step = 0; step < 60; step++)
            {
                local = (local * 75 + 74) % 65537;
                int time = local % 41;
                int[] expected = OracleRoots(n, ops, time);
                var distinct = new HashSet<int>(expected);
                Check(structure.ComponentCount(time) == distinct.Count, "lcg oracle count");
                for (int node = 0; node < n; node++)
                    Check(structure.Find(node, time) == expected[node], "lcg oracle representative");
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
