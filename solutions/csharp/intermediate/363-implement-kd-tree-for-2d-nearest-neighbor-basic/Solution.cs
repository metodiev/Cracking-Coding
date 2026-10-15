// Task 363 - Implement kd-tree for 2D nearest neighbor (basic). (Tier: Intermediate, CTCI Ch 4)
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

internal sealed class KdNode
{
    public readonly int[] Point;
    public readonly int Index;
    public readonly int Axis;
    public readonly KdNode Left;
    public readonly KdNode Right;

    public KdNode(int[] point, int index, int axis, KdNode left, KdNode right)
    {
        Point = point;
        Index = index;
        Axis = axis;
        Left = left;
        Right = right;
    }
}

/// <summary>
/// Axis alternates with depth and each node takes the median of its points sorted by that
/// axis (input index breaks ties), so the tree is balanced and fully deterministic.
/// </summary>
internal sealed class KdTree
{
    private readonly int[][] points;
    private readonly KdNode root;

    public KdTree(int[][] rawPoints)
    {
        points = (int[][])rawPoints.Clone();
        root = Build(Enumerable.Range(0, points.Length).ToList(), 0);
    }

    private KdNode Build(List<int> ids, int depth)
    {
        if (ids.Count == 0) return null;
        int axis = depth % 2;
        ids.Sort((i, j) => points[i][axis] != points[j][axis]
            ? points[i][axis].CompareTo(points[j][axis])
            : i.CompareTo(j));
        int mid = ids.Count / 2;
        int index = ids[mid];
        KdNode left = Build(ids.GetRange(0, mid), depth + 1);
        KdNode right = Build(ids.GetRange(mid + 1, ids.Count - mid - 1), depth + 1);
        return new KdNode((int[])points[index].Clone(), index, axis, left, right);
    }

    /// <summary>Returns (squared distance, input index) of the nearest stored point.</summary>
    public (long Distance, int Index) Nearest(int[] query)
    {
        if (root == null) throw new Exception("empty kd-tree");
        return Search(root, query, null);
    }

    private static (long Distance, int Index) Search(KdNode node, int[] query,
            (long Distance, int Index)? best)
    {
        if (node == null) return best.Value;
        long dx = query[0] - node.Point[0];
        long dy = query[1] - node.Point[1];
        long distance = dx * dx + dy * dy;
        if (best == null || distance < best.Value.Distance ||
                (distance == best.Value.Distance && node.Index < best.Value.Index))
        {
            best = (distance, node.Index);
        }
        long offset = query[node.Axis] - node.Point[node.Axis];
        KdNode near = offset <= 0 ? node.Left : node.Right;
        KdNode far = offset <= 0 ? node.Right : node.Left;
        best = Search(near, query, best);
        // `<=` also explores the far side on ties so an equal-distance point with a
        // smaller input index can still win, exactly matching brute force.
        if (offset * offset <= best.Value.Distance) best = Search(far, query, best);
        return best.Value;
    }
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(60601);

    private static (long Distance, int Index) BruteNearest(int[][] points, int[] query)
    {
        (long Distance, int Index)? best = null;
        for (int index = 0; index < points.Length; index++)
        {
            long dx = query[0] - points[index][0];
            long dy = query[1] - points[index][1];
            long distance = dx * dx + dy * dy;
            if (best == null || distance < best.Value.Distance ||
                    (distance == best.Value.Distance && index < best.Value.Index))
            {
                best = (distance, index);
            }
        }
        return best.Value;
    }

    private static void SameAsBrute(KdTree tree, int[][] points, int[] query, string label)
    {
        var got = tree.Nearest(query);
        var want = BruteNearest(points, query);
        Check(got == want, label + " got=(" + got.Distance + "," + got.Index + ")" +
            " want=(" + want.Distance + "," + want.Index + ")");
    }

    public static void Main()
    {
        var hand = new[] { new[] { 0, 0 }, new[] { 3, 4 }, new[] { 1, 1 }, new[] { -5, -5 } };
        var tree = new KdTree(hand);
        Check(tree.Nearest(new[] { 3, 3 }) == (1L, 1), "hand (3,3) -> (3,4)");
        Check(tree.Nearest(new[] { 1, 1 }) == (0L, 2), "query is a stored point");
        Check(tree.Nearest(new[] { 0, 0 }) == (0L, 0), "query is the first point");
        Check(tree.Nearest(new[] { -4, -6 }) == (2L, 3), "hand (-4,-6) -> (-5,-5)");
        Check(tree.Nearest(new[] { 7, 7 }) == (25L, 1), "hand (7,7) -> (3,4)");
        foreach (var query in new[] { new[] { 3, 3 }, new[] { 1, 1 }, new[] { -4, -6 },
                new[] { 7, 7 }, new[] { -9, 4 } })
        {
            SameAsBrute(tree, hand, query, "hand");
        }

        // Deterministic tie-break: four points at squared distance 2 from (1,1); index 0 wins.
        var tie = new[] { new[] { 0, 0 }, new[] { 2, 0 }, new[] { 0, 2 }, new[] { 2, 2 } };
        var tieTree = new KdTree(tie);
        Check(tieTree.Nearest(new[] { 1, 1 }) == (2L, 0), "tie-break picks smallest index");
        SameAsBrute(tieTree, tie, new[] { 1, 1 }, "tie set");

        // Duplicate points: the earliest input index among equal points must win.
        var dup = new[] { new[] { 5, 5 }, new[] { 5, 5 }, new[] { 1, 1 } };
        var dupTree = new KdTree(dup);
        Check(dupTree.Nearest(new[] { 4, 4 }) == (2L, 0), "duplicate tie-break");
        Check(dupTree.Nearest(new[] { 1, 1 }) == (0L, 2), "duplicate query");

        var single = new KdTree(new[] { new[] { -2, 7 } });
        Check(single.Nearest(new[] { 0, 0 }) == (53L, 0), "single point 4 + 49");

        // LCG cross-check: many sets, queries both random and drawn from the set itself.
        for (int trial = 0; trial < 40; trial++)
        {
            int size = 1 + Random.Next(40);
            var points = new int[size][];
            for (int i = 0; i < size; i++)
            {
                int x = Random.Next(101) - 50;
                int y = Random.Next(101) - 50;
                points[i] = new[] { x, y };
            }
            var randomTree = new KdTree(points);
            foreach (var point in points)
            {
                SameAsBrute(randomTree, points, point, "in-set query trial " + trial);
            }
            for (int q = 0; q < 12; q++)
            {
                int x = Random.Next(121) - 60;
                int y = Random.Next(121) - 60;
                SameAsBrute(randomTree, points, new[] { x, y }, "random query trial " + trial);
            }
        }

        // Duplicate-heavy sets (coordinates in a tiny window) agree with brute force too.
        for (int trial = 0; trial < 10; trial++)
        {
            int size = 2 + Random.Next(30);
            var points = new int[size][];
            for (int i = 0; i < size; i++)
            {
                points[i] = new[] { Random.Next(4), Random.Next(4) };
            }
            var randomTree = new KdTree(points);
            var queries = new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 3, 3 },
                new[] { 2, 1 }, new[] { 1, 2 } };
            foreach (var query in queries)
            {
                SameAsBrute(randomTree, points, query, "dense trial " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
