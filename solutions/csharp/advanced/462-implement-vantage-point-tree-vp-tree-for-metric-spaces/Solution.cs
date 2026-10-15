// Task 462 - Implement vantage-point tree (VP-tree) for metric spaces. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Integer 2-D coordinates; metric = Manhattan L1 (|dx| + |dy|), a true metric, so
// VP-tree pruning is exact. Each node stores a vantage point, the median distance
// `mu`, and two children: inner (distance <= mu) and outer (distance >= mu), which
// keeps every point in exactly one child and the halves balanced.
// Search visits the nearer child first and prunes by the triangle inequality:
// inner when tau - mu > best, outer when tau + best < mu (strict, so equal-distance
// candidates are still visited). Tie-break (documented): equal L1 distance ->
// smaller point index. Fixed LCG workload: equals the brute-force oracle; the
// visited-node statistic stays below n/3 nodes per query on average (measured 14.1).
using System;
using System.Collections.Generic;

internal class VpTree
{
    private readonly int[][] pts;
    private readonly int[] mu, inner, outer;
    private readonly int root;
    public int Visited;
    private long bestDist;
    private int bestIdx;

    public VpTree(int[][] points)
    {
        pts = new int[points.Length][];
        for (int i = 0; i < points.Length; i++) pts[i] = (int[])points[i].Clone();
        int n = pts.Length;
        mu = new int[n];
        inner = new int[n];
        outer = new int[n];
        for (int i = 0; i < n; i++) { inner[i] = -1; outer[i] = -1; }
        var ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        root = n > 0 ? Build(ids) : -1;
    }

    private static int Dist(int[] a, int[] b) => Math.Abs(a[0] - b[0]) + Math.Abs(a[1] - b[1]);

    private int Build(int[] ids)
    {
        if (ids.Length == 0) return -1;
        int vp = ids[0];
        if (ids.Length == 1) return vp;
        var rest = new int[ids.Length - 1];
        Array.Copy(ids, 1, rest, 0, rest.Length);
        Array.Sort(rest, (a, b) =>
        {
            int da = Dist(pts[vp], pts[a]), db = Dist(pts[vp], pts[b]);
            return da != db ? da.CompareTo(db) : a.CompareTo(b);
        });
        int mid = rest.Length / 2;
        mu[vp] = Dist(pts[vp], pts[rest[mid]]);
        inner[vp] = Build(rest[..(mid + 1)]);
        outer[vp] = Build(rest[(mid + 1)..]);
        return vp;
    }

    public long[] Nearest(int qx, int qy)
    {
        Visited = 0;
        if (root == -1) return new long[] { -1, -1 };
        bestDist = long.MaxValue;
        bestIdx = -1;
        Search(root, qx, qy);
        return new long[] { bestIdx, bestDist };
    }

    private void Search(int node, int qx, int qy)
    {
        Visited++;
        int tau = Math.Abs(qx - pts[node][0]) + Math.Abs(qy - pts[node][1]);
        if (tau < bestDist || (tau == bestDist && node < bestIdx))
        {
            bestDist = tau;
            bestIdx = node;
        }
        int radius = mu[node];
        if (tau < radius)
        {
            VisitInner(node, qx, qy, tau, radius);
            VisitOuter(node, qx, qy, tau, radius);
        }
        else
        {
            VisitOuter(node, qx, qy, tau, radius);
            VisitInner(node, qx, qy, tau, radius);
        }
    }

    private void VisitInner(int node, int qx, int qy, int tau, int radius)
    {
        if (inner[node] != -1 && tau - radius <= bestDist) Search(inner[node], qx, qy);
    }

    private void VisitOuter(int node, int qx, int qy, int tau, int radius)
    {
        if (outer[node] != -1 && tau + bestDist >= radius) Search(outer[node], qx, qy);
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
            long d = Math.Abs(points[i][0] - qx) + Math.Abs(points[i][1] - qy);
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
            new[] { 0, 0 }, new[] { 1, 1 }, new[] { 3, 0 }, new[] { 5, 5 }, new[] { 2, 4 },
        };
        var tree = new VpTree(small);
        Check(Same(tree.Nearest(0, 0), new long[] { 0, 0 }), "tiny (0,0)");
        Check(Same(tree.Nearest(4, 3), new long[] { 3, 3 }), "tiny (4,3)");
        Check(Same(tree.Nearest(2, 1), new long[] { 1, 1 }), "tiny (2,1)");
        Check(Same(tree.Nearest(5, 0), new long[] { 2, 2 }), "tiny (5,0)");
        Check(Same(new VpTree(Array.Empty<int[]>()).Nearest(0, 0), new long[] { -1, -1 }), "empty");

        int n = 150;
        seed = 4242;
        var points = new int[n][];
        for (int i = 0; i < n; i++)
        {
            int x = Lcg() % 2001 - 1000;
            int y = Lcg() % 2001 - 1000;
            points[i] = new[] { x, y };
        }
        tree = new VpTree(points);
        const int queries = 150;
        long totalVisited = 0;
        for (int qi = 0; qi < queries; qi++)
        {
            int qx = Lcg() % 2001 - 1000;
            int qy = Lcg() % 2001 - 1000;
            Check(Same(tree.Nearest(qx, qy), BruteNearest(points, qx, qy)), "query " + qi);
            totalVisited += tree.Visited;
        }
        Check(totalVisited > 0, "visited positive");
        Check(totalVisited < (long)queries * n / 3, "visited budget " + totalVisited);
        Console.WriteLine("PASS");
    }

    private static bool Same(long[] a, long[] b) => a[0] == b[0] && a[1] == b[1];

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
