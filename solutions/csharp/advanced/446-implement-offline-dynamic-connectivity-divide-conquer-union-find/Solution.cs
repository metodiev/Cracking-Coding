// Task 446 - Implement offline dynamic connectivity (divide & conquer + union find). (Tier: Advanced, CTCI Ch 4)
using System;
using System.Collections.Generic;

internal class RollbackDSU
{
    private readonly int[] parent;
    private readonly int[] size;
    private readonly int[] stackA;
    private readonly int[] stackB;
    private int top;

    public RollbackDSU(int n)
    {
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++)
        {
            parent[i] = i;
            size[i] = 1;
        }
        stackA = new int[n];
        stackB = new int[n];
    }

    public int Find(int x)
    {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    public bool Union(int a, int b)
    {
        int ra = Find(a);
        int rb = Find(b);
        if (ra == rb) return false;
        if (size[ra] > size[rb]) (ra, rb) = (rb, ra);
        parent[ra] = rb;
        size[rb] += size[ra];
        stackA[top] = ra;
        stackB[top] = rb;
        top++;
        return true;
    }

    public bool Same(int a, int b) => Find(a) == Find(b);

    public int Snapshot() => top;

    public void Rollback(int snapshot)
    {
        while (top > snapshot)
        {
            top--;
            int ra = stackA[top];
            int rb = stackB[top];
            parent[ra] = ra;
            size[rb] -= size[ra];
        }
    }
}

internal class OfflineConnectivity
{
    private readonly int n;
    private readonly int[][] ops;
    private readonly RollbackDSU dsu;
    private readonly bool?[] answers;

    public OfflineConnectivity(int n, int[][] ops)
    {
        this.n = n;
        this.ops = ops;
        dsu = new RollbackDSU(n);
        answers = new bool?[ops.Length];
    }

    public bool?[] Run()
    {
        int q = ops.Length;
        if (q == 0) return answers;
        var counts = new Dictionary<long, int>();
        var open = new Dictionary<long, int[]>();
        var intervals = new List<int[]>();
        for (int t = 0; t < q; t++)
        {
            int kind = ops[t][0], u = ops[t][1], v = ops[t][2];
            long key = (long)Math.Min(u, v) * 1000 + Math.Max(u, v);
            if (kind == 0)
            {
                counts.TryGetValue(key, out int c);
                counts[key] = c + 1;
                if (c + 1 == 1) open[key] = new[] { Math.Min(u, v), Math.Max(u, v), t };
            }
            else if (kind == 1)
            {
                counts.TryGetValue(key, out int c);
                if (c > 0)
                {
                    counts[key] = c - 1;
                    if (c - 1 == 0)
                    {
                        int[] info = open[key];
                        open.Remove(key);
                        if (info[2] <= t - 1) intervals.Add(new[] { info[0], info[1], info[2], t - 1 });
                    }
                }
            }
        }
        foreach (int[] info in open.Values)
        {
            intervals.Add(new[] { info[0], info[1], info[2], q - 1 });
        }
        Solve(0, q - 1, intervals);
        return answers;
    }

    private void Solve(int lo, int hi, List<int[]> active)
    {
        int snapshot = dsu.Snapshot();
        var rest = new List<int[]>();
        foreach (int[] e in active)
        {
            if (e[2] <= lo && hi <= e[3]) dsu.Union(e[0], e[1]);
            else rest.Add(e);
        }
        if (lo == hi)
        {
            if (ops[lo][0] == 2) answers[lo] = dsu.Same(ops[lo][1], ops[lo][2]);
        }
        else
        {
            int mid = (lo + hi) >> 1;
            var left = new List<int[]>();
            var right = new List<int[]>();
            foreach (int[] e in rest)
            {
                if (e[2] <= mid && e[3] >= lo) left.Add(e);
                if (e[2] <= hi && e[3] >= mid + 1) right.Add(e);
            }
            Solve(lo, mid, left);
            Solve(mid + 1, hi, right);
        }
        dsu.Rollback(snapshot);
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

    private static List<int[][]> LcgOpScripts(int trials, int n, int q, int baseSeed)
    {
        var scripts = new List<int[][]>();
        seed = baseSeed;
        for (int t = 0; t < trials; t++)
        {
            var counts = new Dictionary<long, int>();
            var ops = new int[q][];
            for (int step = 0; step < q; step++)
            {
                int kind = NextSeed() % 3;
                int u = NextSeed() % n;
                int v = NextSeed() % n;
                if (u == v) v = (v + 1) % n;
                long key = (long)Math.Min(u, v) * 1000 + Math.Max(u, v);
                if (kind == 1 && counts.Count == 0) kind = 0;
                if (kind == 0)
                {
                    ops[step] = new[] { 0, u, v };
                    counts.TryGetValue(key, out int c);
                    counts[key] = c + 1;
                }
                else if (kind == 1)
                {
                    var keys = new List<long>(counts.Keys);
                    keys.Sort();
                    long pick = keys[NextSeed() % keys.Count];
                    ops[step] = new[] { 1, (int)(pick / 1000), (int)(pick % 1000) };
                    int c = counts[pick] - 1;
                    if (c == 0) counts.Remove(pick);
                    else counts[pick] = c;
                }
                else
                {
                    ops[step] = new[] { 2, u, v };
                }
            }
            scripts.Add(ops);
        }
        return scripts;
    }

    private static bool?[] BruteAnswers(int n, int[][] ops)
    {
        var counts = new Dictionary<long, int>();
        var answers = new bool?[ops.Length];
        for (int t = 0; t < ops.Length; t++)
        {
            int kind = ops[t][0], u = ops[t][1], v = ops[t][2];
            long key = (long)Math.Min(u, v) * 1000 + Math.Max(u, v);
            if (kind == 0)
            {
                counts.TryGetValue(key, out int c);
                counts[key] = c + 1;
            }
            else if (kind == 1)
            {
                int c = counts[key] - 1;
                if (c == 0) counts.Remove(key);
                else counts[key] = c;
            }
            else
            {
                var adjacency = new List<int>[n];
                for (int i = 0; i < n; i++) adjacency[i] = new List<int>();
                foreach (long edge in counts.Keys)
                {
                    int a = (int)(edge / 1000);
                    int b = (int)(edge % 1000);
                    adjacency[a].Add(b);
                    adjacency[b].Add(a);
                }
                var seen = new bool[n];
                seen[u] = true;
                var stack = new Stack<int>();
                stack.Push(u);
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
                answers[t] = seen[v];
            }
        }
        return answers;
    }

    private static bool Same(bool?[] a, bool?[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        int[][] ops = { new[] { 0, 0, 1 }, new[] { 2, 0, 1 }, new[] { 0, 1, 2 }, new[] { 2, 0, 2 }, new[] { 1, 0, 1 }, new[] { 2, 0, 2 }, new[] { 2, 0, 0 } };
        Check(Same(new OfflineConnectivity(3, ops).Run(), new bool?[] { null, true, null, true, null, false, true }), "hand script 1");
        int[][] ops2 = { new[] { 2, 0, 1 }, new[] { 0, 0, 1 }, new[] { 2, 0, 1 }, new[] { 1, 0, 1 }, new[] { 2, 0, 1 }, new[] { 2, 1, 1 } };
        Check(Same(new OfflineConnectivity(2, ops2).Run(), new bool?[] { false, null, true, null, false, true }), "hand script 2");

        int queries = 0;
        foreach (int[][] script in LcgOpScripts(30, 8, 40, 446))
        {
            bool?[] brute = BruteAnswers(8, script);
            foreach (bool? answer in brute)
            {
                if (answer != null) queries++;
            }
            Check(Same(new OfflineConnectivity(8, script).Run(), brute), "random script n=8");
        }
        Check(queries > 100, "query coverage");

        foreach (int[][] script in LcgOpScripts(10, 12, 60, 777))
        {
            Check(Same(new OfflineConnectivity(12, script).Run(), BruteAnswers(12, script)), "random script n=12");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
