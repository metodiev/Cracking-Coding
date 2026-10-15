// Task 343 - Graph isomorphism: simple checks with degree-sequence invariants. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Necessary conditions only - never proves isomorphism, but rules it out.</summary>
    public static bool CheckIsomorphic(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        if (!ArraysEqual(Degrees(a), Degrees(b))) return false;
        if (TriangleCount(a) != TriangleCount(b)) return false;
        if (!ArraysEqual(ComponentSizes(a), ComponentSizes(b))) return false;
        return true;
    }

    public static int[] Degrees(int[][] adj)
    {
        var outDegrees = new int[adj.Length];
        for (int i = 0; i < adj.Length; i++)
        {
            var unique = new HashSet<int>(adj[i]);
            outDegrees[i] = unique.Count;
        }
        Array.Sort(outDegrees);
        return outDegrees;
    }

    public static int TriangleCount(int[][] adj)
    {
        var sets = new HashSet<int>[adj.Length];
        for (int i = 0; i < adj.Length; i++) sets[i] = new HashSet<int>(adj[i]);
        int count = 0;
        for (int u = 0; u < adj.Length; u++)
        {
            foreach (int v in sets[u])
            {
                if (v <= u) continue;
                foreach (int w in sets[u])
                {
                    if (w > v && sets[v].Contains(w)) count++;
                }
            }
        }
        return count;
    }

    public static int[] ComponentSizes(int[][] adj)
    {
        int n = adj.Length;
        var seen = new bool[n];
        var sizes = new List<int>();
        for (int start = 0; start < n; start++)
        {
            if (seen[start]) continue;
            seen[start] = true;
            var stack = new Stack<int>();
            stack.Push(start);
            int size = 0;
            while (stack.Count > 0)
            {
                int u = stack.Pop();
                size++;
                foreach (int v in adj[u])
                {
                    if (!seen[v])
                    {
                        seen[v] = true;
                        stack.Push(v);
                    }
                }
            }
            sizes.Add(size);
        }
        var outSizes = sizes.ToArray();
        Array.Sort(outSizes);
        return outSizes;
    }

    /// <summary>Independent oracle for the component multiset.</summary>
    public static int[] ComponentSizesUnionFind(int[][] adj)
    {
        int n = adj.Length;
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int u = 0; u < n; u++)
        {
            foreach (int v in adj[u])
            {
                int rootU = Find(parent, u);
                int rootV = Find(parent, v);
                if (rootU != rootV) parent[rootU] = rootV;
            }
        }
        var counts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int root = Find(parent, i);
            counts.TryGetValue(root, out int current);
            counts[root] = current + 1;
        }
        var outSizes = new int[counts.Count];
        int k = 0;
        foreach (int size in counts.Values) outSizes[k++] = size;
        Array.Sort(outSizes);
        return outSizes;
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

    public static int Diameter(int[][] adj)
    {
        int n = adj.Length;
        int best = 0;
        for (int start = 0; start < n; start++)
        {
            var dist = new int[n];
            for (int i = 0; i < n; i++) dist[i] = -1;
            dist[start] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                foreach (int v in adj[u])
                {
                    if (dist[v] < 0)
                    {
                        dist[v] = dist[u] + 1;
                        queue.Enqueue(v);
                    }
                }
            }
            foreach (int d in dist)
            {
                if (d < 0) return -1;
                best = Math.Max(best, d);
            }
        }
        return best;
    }

    /// <summary>perm maps old vertex -> new vertex.</summary>
    public static int[][] Relabel(int[][] adj, int[] perm)
    {
        var outAdj = new int[adj.Length][];
        for (int u = 0; u < adj.Length; u++)
        {
            var mapped = new SortedSet<int>();
            foreach (int v in adj[u]) mapped.Add(perm[v]);
            var row = new int[mapped.Count];
            int i = 0;
            foreach (int value in mapped) row[i++] = value;
            outAdj[perm[u]] = row;
        }
        return outAdj;
    }

    public static int[] Inverse(int[] perm)
    {
        var inverse = new int[perm.Length];
        for (int old = 0; old < perm.Length; old++) inverse[perm[old]] = old;
        return inverse;
    }

    public static bool IsBipartite(int[][] adj)
    {
        var color = new int[adj.Length];
        for (int i = 0; i < color.Length; i++) color[i] = -1;
        for (int start = 0; start < adj.Length; start++)
        {
            if (color[start] >= 0) continue;
            color[start] = 0;
            var queue = new Queue<int>();
            queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int u = queue.Dequeue();
                foreach (int v in adj[u])
                {
                    if (color[v] < 0)
                    {
                        color[v] = 1 - color[u];
                        queue.Enqueue(v);
                    }
                    else if (color[v] == color[u])
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public static int[][] Normalize(int[][] adj)
    {
        var outAdj = new int[adj.Length][];
        for (int i = 0; i < adj.Length; i++)
        {
            var unique = new SortedSet<int>(adj[i]);
            var row = new int[unique.Count];
            int k = 0;
            foreach (int value in unique) row[k++] = value;
            outAdj[i] = row;
        }
        return outAdj;
    }

    public static int[][] CycleGraph(int n)
    {
        var adj = new int[n][];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new[] { (i - 1 + n) % n, (i + 1) % n };
            Array.Sort(adj[i]);
        }
        return adj;
    }

    public static int[][] CubeQ3()
    {
        var adj = new int[8][];
        for (int i = 0; i < 8; i++)
        {
            adj[i] = new[] { i ^ 1, i ^ 2, i ^ 4 };
            Array.Sort(adj[i]);
        }
        return adj;
    }

    public static int[][] MobiusV8()
    {
        var sets = new SortedSet<int>[8];
        for (int i = 0; i < 8; i++) sets[i] = new SortedSet<int>();
        for (int i = 0; i < 8; i++)
        {
            sets[i].Add((i + 1) % 8);
            sets[(i + 1) % 8].Add(i);
        }
        for (int i = 0; i < 4; i++)
        {
            sets[i].Add(i + 4);
            sets[i + 4].Add(i);
        }
        var adj = new int[8][];
        for (int i = 0; i < 8; i++)
        {
            adj[i] = new int[sets[i].Count];
            int k = 0;
            foreach (int value in sets[i]) adj[i][k++] = value;
        }
        return adj;
    }

    private static int seed = 20261016;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static int[][] LcgGraph(int n)
    {
        var sets = new SortedSet<int>[n];
        for (int i = 0; i < n; i++) sets[i] = new SortedSet<int>();
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (NextSeed() % 2 == 0)
                {
                    sets[u].Add(v);
                    sets[v].Add(u);
                }
            }
        }
        var adj = new int[n][];
        for (int i = 0; i < n; i++)
        {
            adj[i] = new int[sets[i].Count];
            int k = 0;
            foreach (int value in sets[i]) adj[i][k++] = value;
        }
        return adj;
    }

    public static int[] LcgPermutation(int n)
    {
        var perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = NextSeed() % (i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        return perm;
    }

    private static bool ArraysEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static bool SameAdjacency(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (!ArraysEqual(a[i], b[i])) return false;
        }
        return true;
    }

    public static void Main()
    {
        var c5 = CycleGraph(5);
        var c6 = CycleGraph(6);
        var path4 = Normalize(new[] { new[] { 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2 } });
        var star4 = Normalize(new[] { new[] { 1, 2, 3, 4 }, new[] { 0 }, new[] { 0 }, new[] { 0 }, new[] { 0 } });
        var bowtie = Normalize(new[] { new[] { 1, 2, 3, 4 }, new[] { 0, 2 }, new[] { 0, 1 }, new[] { 0, 4 }, new[] { 0, 3 } });

        Check(CheckIsomorphic(c5, c5), "C5 with itself");
        var pentagram = Relabel(c5, new[] { 0, 2, 4, 1, 3 });
        Check(SameAdjacency(pentagram, new[] { new[] { 2, 3 }, new[] { 3, 4 }, new[] { 0, 4 }, new[] { 0, 1 }, new[] { 1, 2 } }),
              "pentagram adjacency");
        Check(!SameAdjacency(pentagram, c5), "pentagram is a different labelling");
        Check(CheckIsomorphic(c5, pentagram), "C5 vs pentagram");
        Check(CheckIsomorphic(path4, Relabel(path4, new[] { 2, 0, 3, 1 })), "P4 relabel");
        Check(CheckIsomorphic(star4, Relabel(star4, new[] { 4, 0, 1, 2, 3 })), "star relabel");
        Check(CheckIsomorphic(bowtie, Relabel(bowtie, new[] { 3, 1, 4, 0, 2 })), "bowtie relabel");
        Check(CheckIsomorphic(Array.Empty<int[]>(), Array.Empty<int[]>()), "empty graphs");
        Check(CheckIsomorphic(new[] { Array.Empty<int>() }, new[] { Array.Empty<int>() }), "single vertex");

        var star3 = Normalize(new[] { new[] { 1, 2, 3 }, new[] { 0 }, new[] { 0 }, new[] { 0 } });
        Check(!CheckIsomorphic(star3, path4), "star K1,3 vs P4");
        Check(!CheckIsomorphic(bowtie, c5), "bowtie vs C5");

        var twoTriangles = Normalize(new[] { new[] { 1, 2 }, new[] { 0, 2 }, new[] { 0, 1 }, new[] { 4, 5 }, new[] { 3, 5 }, new[] { 3, 4 } });
        Check(ArraysEqual(Degrees(c6), Degrees(twoTriangles)), "C6 vs 2 triangles same degrees");
        Check(ArraysEqual(Degrees(c6), new[] { 2, 2, 2, 2, 2, 2 }), "degrees [2]*6");
        Check(TriangleCount(c6) == 0 && TriangleCount(twoTriangles) == 2, "triangles 0 vs 2");
        Check(ArraysEqual(ComponentSizes(c6), new[] { 6 }), "comps [6]");
        Check(ArraysEqual(ComponentSizes(twoTriangles), new[] { 3, 3 }), "comps [3,3]");
        Check(!CheckIsomorphic(c6, twoTriangles), "C6 not isomorphic to 2 triangles");

        var k33 = Normalize(new[] { new[] { 3, 4, 5 }, new[] { 3, 4, 5 }, new[] { 3, 4, 5 }, new[] { 0, 1, 2 }, new[] { 0, 1, 2 }, new[] { 0, 1, 2 } });
        var prism = Normalize(new[] { new[] { 1, 2, 3 }, new[] { 0, 2, 4 }, new[] { 0, 1, 5 }, new[] { 4, 5, 0 }, new[] { 3, 5, 1 }, new[] { 3, 4, 2 } });
        Check(ArraysEqual(Degrees(k33), Degrees(prism)), "K33 vs prism same degrees");
        Check(ArraysEqual(Degrees(k33), new[] { 3, 3, 3, 3, 3, 3 }), "degrees [3]*6");
        Check(ArraysEqual(ComponentSizes(k33), ComponentSizes(prism)), "K33 vs prism same components");
        Check(TriangleCount(k33) == 0 && TriangleCount(prism) == 2, "K33 vs prism triangles");
        Check(!CheckIsomorphic(k33, prism), "K33 not isomorphic to prism");

        var cube = CubeQ3();
        var v8 = MobiusV8();
        Check(CheckIsomorphic(cube, v8), "cube vs V8 passes all three checks");
        Check(ArraysEqual(Degrees(cube), Degrees(v8)), "cube vs V8 same degrees");
        Check(TriangleCount(cube) == 0 && TriangleCount(v8) == 0, "cube vs V8 triangles");
        Check(ArraysEqual(ComponentSizes(cube), ComponentSizes(v8)), "cube vs V8 same components");
        Check(IsBipartite(cube) && !IsBipartite(v8), "bipartiteness separates the counterexample");
        Check(Diameter(cube) == 3 && Diameter(v8) == 2, "diameters 3 vs 2");

        for (int trial = 0; trial < 40; trial++)
        {
            var graph = LcgGraph(7);
            var perm = LcgPermutation(7);
            var relabelled = Relabel(graph, perm);
            Check(CheckIsomorphic(graph, relabelled), "random relabel " + trial);
            Check(ArraysEqual(Degrees(graph), Degrees(relabelled)), "random degrees " + trial);
            Check(TriangleCount(graph) == TriangleCount(relabelled), "random triangles " + trial);
            Check(ArraysEqual(ComponentSizes(graph), ComponentSizesUnionFind(graph)), "union-find oracle " + trial);
            Check(ArraysEqual(ComponentSizes(relabelled), ComponentSizesUnionFind(relabelled)), "union-find relabelled " + trial);
            Check(SameAdjacency(Relabel(relabelled, Inverse(perm)), graph), "double relabel " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
