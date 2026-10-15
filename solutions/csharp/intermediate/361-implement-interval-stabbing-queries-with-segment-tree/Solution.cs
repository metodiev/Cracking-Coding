// Task 361 - Implement interval stabbing queries with segment tree. (Tier: Intermediate, CTCI Ch 4)
using System;

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

/// <summary>
/// All query points are known up front, so they are coordinate-compressed to indices and
/// the segment tree stores cover-count adds; a point query walks leaf -> root.
/// </summary>
internal sealed class StabbingSegmentTree
{
    private readonly int[] points;
    private readonly int n;
    private readonly int[] tree;

    public StabbingSegmentTree(int[] rawPoints)
    {
        var sorted = (int[])rawPoints.Clone();
        Array.Sort(sorted);
        int unique = 0;
        for (int i = 0; i < sorted.Length; i++)
        {
            if (i == 0 || sorted[i] != sorted[i - 1]) sorted[unique++] = sorted[i];
        }
        points = new int[unique];
        Array.Copy(sorted, points, unique);
        n = unique;
        tree = new int[2 * Math.Max(n, 1)];
    }

    public void AddInterval(int left, int right)
    {
        if (n == 0) return;
        int l = LowerBound(left) + n;
        int r = UpperBound(right) + n;
        while (l < r)
        {
            if ((l & 1) == 1) tree[l++]++;
            if ((r & 1) == 1) tree[--r]++;
            l >>= 1;
            r >>= 1;
        }
    }

    public int CoverCount(int point)
    {
        int index = LowerBound(point);
        if (index >= n || points[index] != point) throw new Exception("point was not registered");
        int total = 0;
        for (int i = index + n; i >= 1; i >>= 1) total += tree[i];
        return total;
    }

    private int LowerBound(int target)
    {
        int lo = 0, hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (points[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private int UpperBound(int target)
    {
        int lo = 0, hi = n;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (points[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(20261);

    private static int NaiveCoverCount(int[][] intervals, int point)
    {
        int count = 0;
        foreach (var interval in intervals)
        {
            if (interval[0] <= point && point <= interval[1]) count++;
        }
        return count;
    }

    public static void Main()
    {
        // Hand-built example from the task description: intervals [1,4], [3,6], [5,8].
        int[][] intervals = { new[] { 1, 4 }, new[] { 3, 6 }, new[] { 5, 8 } };
        int[] points = { 0, 2, 4, 6, 7, 9 };
        int[] expected = { 0, 1, 2, 2, 1, 0 };
        var tree = new StabbingSegmentTree(points);
        foreach (var interval in intervals) tree.AddInterval(interval[0], interval[1]);
        for (int k = 0; k < points.Length; k++)
        {
            Check(tree.CoverCount(points[k]) == expected[k], "hand example point " + points[k]);
            Check(NaiveCoverCount(intervals, points[k]) == expected[k], "naive at " + points[k]);
        }

        // Adding the same interval twice doubles the counts.
        var doubled = new StabbingSegmentTree(points);
        foreach (var interval in intervals)
        {
            doubled.AddInterval(interval[0], interval[1]);
            doubled.AddInterval(interval[0], interval[1]);
        }
        for (int k = 0; k < points.Length; k++)
        {
            Check(doubled.CoverCount(points[k]) == 2 * expected[k], "doubled at " + points[k]);
        }

        // Duplicate query points collapse; single point works; no points is a no-op.
        var dup = new StabbingSegmentTree(new[] { 3, 3, 5, 3 });
        dup.AddInterval(1, 4);
        Check(dup.CoverCount(3) == 1 && dup.CoverCount(5) == 0, "duplicate query points");
        var solo = new StabbingSegmentTree(new[] { 7 });
        solo.AddInterval(7, 7);
        solo.AddInterval(8, 9);
        Check(solo.CoverCount(7) == 1, "single registered point");
        new StabbingSegmentTree(Array.Empty<int>()).AddInterval(-3, 12);   // must not fail

        // LCG cross-check against the naive cover count.
        int[] probe = new int[48];
        for (int i = 0; i < probe.Length; i++) probe[i] = i - 2;
        for (int trial = 0; trial < 25; trial++)
        {
            int count = 1 + Random.Next(40);
            var randomIntervals = new int[count][];
            for (int i = 0; i < count; i++)
            {
                int left = Random.Next(30);
                int width = 1 + Random.Next(10);
                randomIntervals[i] = new[] { left, left + width };
            }
            var randomTree = new StabbingSegmentTree(probe);
            foreach (var interval in randomIntervals)
            {
                randomTree.AddInterval(interval[0], interval[1]);
            }
            foreach (int p in probe)
            {
                Check(randomTree.CoverCount(p) == NaiveCoverCount(randomIntervals, p),
                    "random trial " + trial + " point " + p);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
