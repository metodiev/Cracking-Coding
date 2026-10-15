// Task 329 - Range intersection queries (interval trees basics). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537.</summary>
internal sealed class Lcg
{
    private int seed;

    public Lcg(int seed) => this.seed = seed;

    public int Next(int modulus)
    {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/// <summary>Augmented BST over interval starts with subtree max-end values.</summary>
internal sealed class IntervalTree
{
    private sealed class Node
    {
        public int Start, End, MaxEnd;
        public Node? Left, Right;

        public Node(int start, int end)
        {
            Start = start;
            End = end;
            MaxEnd = end;
        }
    }

    private Node? root;

    public IntervalTree(int[][] intervals)
    {
        var ordered = intervals.Select(iv => (int[])iv.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        root = Build(ordered, 0, ordered.Length);
    }

    private static Node? Build(int[][] ordered, int from, int to)
    {
        if (from >= to) return null;
        int mid = (from + to) / 2;
        var node = new Node(ordered[mid][0], ordered[mid][1]);
        node.Left = Build(ordered, from, mid);
        node.Right = Build(ordered, mid + 1, to);
        if (node.Left != null) node.MaxEnd = Math.Max(node.MaxEnd, node.Left.MaxEnd);
        if (node.Right != null) node.MaxEnd = Math.Max(node.MaxEnd, node.Right.MaxEnd);
        return node;
    }

    /// <summary>All stored intervals intersecting the closed range [lo, hi], in start order.</summary>
    public List<int[]> Query(int lo, int hi)
    {
        var found = new List<int[]>();
        Walk(root, lo, hi, found);
        return found;
    }

    private static void Walk(Node? node, int lo, int hi, List<int[]> found)
    {
        if (node == null) return;
        if (node.Left != null && node.Left.MaxEnd >= lo) Walk(node.Left, lo, hi, found);
        if (node.Start <= hi && node.End >= lo) found.Add(new[] { node.Start, node.End });
        if (node.Start <= hi) Walk(node.Right, lo, hi, found);
    }
}

public static class Solution
{
    /// <summary>Independent oracle: filter every interval.</summary>
    private static List<int[]> NaiveQuery(int[][] intervals, int lo, int hi) =>
        SortedPoints(intervals.Where(iv => iv[0] <= hi && iv[1] >= lo).Select(iv => (int[])iv.Clone()).ToArray());

    private static List<int[]> SortedPoints(int[][] points)
    {
        var outPoints = points.Select(p => (int[])p.Clone()).ToArray();
        Array.Sort(outPoints, (u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        return outPoints.ToList();
    }

    private static int[][] LcgIntervals(int count, Lcg lcg)
    {
        var intervals = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int start = lcg.Next(60);
            intervals[i] = new[] { start, start + 1 + lcg.Next(12) };
        }
        return intervals;
    }

    private static bool SamePoints(List<int[]> a, List<int[]> b) =>
        a.Count == b.Count && a.Zip(b).All(pair => pair.First.SequenceEqual(pair.Second));

    private static readonly int[][] Fixed =
    {
        new[] { 15, 20 }, new[] { 10, 30 }, new[] { 17, 19 }, new[] { 5, 20 }, new[] { 12, 15 }, new[] { 30, 40 },
    };

    public static void Main()
    {
        var tree = new IntervalTree(Fixed);
        Check(SamePoints(tree.Query(6, 7), new List<int[]> { new[] { 5, 20 } }), "query [6,7]");
        Check(SamePoints(tree.Query(14, 16), new List<int[]>
        {
            new[] { 5, 20 }, new[] { 10, 30 }, new[] { 12, 15 }, new[] { 15, 20 },
        }), "query [14,16]");
        Check(SamePoints(tree.Query(30, 35), new List<int[]> { new[] { 10, 30 }, new[] { 30, 40 } }), "query [30,35]");
        Check(SamePoints(tree.Query(19, 19), new List<int[]>
        {
            new[] { 5, 20 }, new[] { 10, 30 }, new[] { 15, 20 }, new[] { 17, 19 },
        }), "query [19,19]");
        Check(tree.Query(0, 4).Count == 0, "query below everything");
        Check(tree.Query(41, 50).Count == 0, "query above everything");

        Check(new IntervalTree(Array.Empty<int[]>()).Query(0, 10).Count == 0, "empty tree");
        Check(SamePoints(new IntervalTree(new[] { new[] { 4, 4 } }).Query(4, 4),
            new List<int[]> { new[] { 4, 4 } }), "point interval");

        var lcg = new Lcg(7071);
        for (int trial = 0; trial < 40; trial++)
        {
            int[][] intervals = LcgIntervals(30, lcg);
            var fromBuild = new IntervalTree(intervals);
            for (int q = 0; q < 30; q++)
            {
                int lo = lcg.Next(70);
                int hi = lo + lcg.Next(25);
                Check(SamePoints(SortedPoints(fromBuild.Query(lo, hi).ToArray()), NaiveQuery(intervals, lo, hi)),
                    "random query cross-check " + trial + "/" + q);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
