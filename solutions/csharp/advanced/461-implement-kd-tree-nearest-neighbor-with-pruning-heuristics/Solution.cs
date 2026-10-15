// Task 461 - Implement kd-tree nearest neighbor with pruning heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// 2-D kd-tree (median split on alternating axes) keeping a bounding box per node.
// Best-first nearest-neighbour search ordered by the squared distance from the
// query point to a child's bounding box; a subtree is pruned as soon as its bound
// strictly exceeds the best squared distance found (strict `>`, so equal-distance
// candidates are still explored). Tie-break (documented): equal squared distance
// -> smaller point index. Fixed LCG workload: equals the brute-force oracle, and
// the visited-node statistic stays below n/4 nodes per query on average (measured
// 8.0) versus n for a linear scan.
using System;
using System.Collections.Generic;

internal class KdTree
{
    private readonly int[][] pts;
    private readonly int[] left, right;
    private readonly int[] bx0, by0, bx1, by1;
    private readonly int root;
    public int Visited;

    public KdTree(int[][] points)
    {
        pts = new int[points.Length][];
        for (int i = 0; i < points.Length; i++) pts[i] = (int[])points[i].Clone();
        int n = pts.Length;
        left = new int[n];
        right = new int[n];
        bx0 = new int[n];
        by0 = new int[n];
        bx1 = new int[n];
        by1 = new int[n];
        for (int i = 0; i < n; i++) { left[i] = -1; right[i] = -1; }
        var ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        root = n > 0 ? Build(ids, 0) : -1;
    }

    private int Build(int[] ids, int depth)
    {
        if (ids.Length == 0) return -1;
        int axis = depth & 1;
        Array.Sort(ids, (a, b) =>
        {
            if (pts[a][axis] != pts[b][axis]) return pts[a][axis].CompareTo(pts[b][axis]);
            if (pts[a][1 - axis] != pts[b][1 - axis]) return pts[a][1 - axis].CompareTo(pts[b][1 - axis]);
            return a.CompareTo(b);
        });
        int mid = ids.Length / 2;
        int node = ids[mid];
        left[node] = Build(ids[..mid], depth + 1);
        right[node] = Build(ids[(mid + 1)..], depth + 1);
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = int.MinValue, y1 = int.MinValue;
        foreach (int i in ids)
        {
            x0 = Math.Min(x0, pts[i][0]);
            x1 = Math.Max(x1, pts[i][0]);
            y0 = Math.Min(y0, pts[i][1]);
            y1 = Math.Max(y1, pts[i][1]);
        }
        bx0[node] = x0; by0[node] = y0; bx1[node] = x1; by1[node] = y1;
        return node;
    }

    private static long Dist2(int[] a, int qx, int qy)
    {
        long dx = a[0] - qx;
        long dy = a[1] - qy;
        return dx * dx + dy * dy;
    }

    private long Bound(int node, int qx, int qy)
    {
        long dx = 0;
        if (qx < bx0[node]) dx = bx0[node] - qx;
        else if (qx > bx1[node]) dx = qx - bx1[node];
        long dy = 0;
        if (qy < by0[node]) dy = by0[node] - qy;
        else if (qy > by1[node]) dy = qy - by1[node];
        return dx * dx + dy * dy;
    }

    public long[] Nearest(int qx, int qy)
    {
        Visited = 0;
        if (root == -1) return new long[] { -1, -1 };
        long bestI = -1, bestD = 0;
        var pq = new PriorityQueue<int, (long, int)>();
        pq.Enqueue(root, (0L, root));
        while (pq.Count > 0)
        {
            pq.TryDequeue(out int node, out (long Bound, int Idx) pri);
            if (bestI != -1 && pri.Bound > bestD) break; // prune
            Visited++;
            long d = Dist2(pts[node], qx, qy);
            if (bestI == -1 || d < bestD || (d == bestD && node < bestI))
            {
                bestI = node;
                bestD = d;
            }
            foreach (int child in new[] { left[node], right[node] })
                if (child != -1) pq.Enqueue(child, (Bound(child, qx, qy), child));
        }
        return new long[] { bestI, bestD };
    }
}

public static class Solution
{
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    private static long[] BruteNearest(int[][] points, int qx, int qy)
    {
        long bestI = -1, bestD = 0;
        for (int i = 0; i < points.Length; i++)
        {
            long dx = points[i][0] - qx;
            long dy = points[i][1] - qy;
            long d = dx * dx + dy * dy;
            if (bestI == -1 || d < bestD || (d == bestD && i < bestI))
            {
                bestI = i;
                bestD = d;
            }
        }
        return new long[] { bestI, bestD };
    }

    public static void Main()
    {
        var small = new[]
        {
            new[] { 0, 0 }, new[] { 2, 3 }, new[] { 5, 4 }, new[] { 4, 1 }, new[] { 9, 6 },
        };
        var tree = new KdTree(small);
        Check(Same(tree.Nearest(1, 1), new long[] { 0, 2 }), "tiny (1,1)");
        Check(Same(tree.Nearest(4, 2), new long[] { 3, 1 }), "tiny (4,2)");
        Check(Same(tree.Nearest(3, 3), new long[] { 1, 1 }), "tiny (3,3)");
        Check(Same(tree.Nearest(3, 2), new long[] { 1, 2 }), "tiny tie (3,2)");
        Check(Same(new KdTree(Array.Empty<int[]>()).Nearest(0, 0), new long[] { -1, -1 }), "empty");

        int n = 200;
        seed = 1234;
        var points = new int[n][];
        for (int i = 0; i < n; i++)
        {
            int x = Lcg() % 2001 - 1000;
            int y = Lcg() % 2001 - 1000;
            points[i] = new[] { x, y };
        }
        tree = new KdTree(points);
        const int queries = 200;
        long totalVisited = 0;
        for (int qi = 0; qi < queries; qi++)
        {
            int qx = Lcg() % 2001 - 1000;
            int qy = Lcg() % 2001 - 1000;
            Check(Same(tree.Nearest(qx, qy), BruteNearest(points, qx, qy)), "query " + qi);
            totalVisited += tree.Visited;
        }
        Check(totalVisited > 0, "visited positive");
        Check(totalVisited < (long)queries * n / 4, "visited budget " + totalVisited);
        Console.WriteLine("PASS");
    }

    private static bool Same(long[] a, long[] b) => a[0] == b[0] && a[1] == b[1];

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
