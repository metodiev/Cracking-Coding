// Task 335 - Implement data structure for skyline queries (segment tree variant). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public class SkylineTree
{
    // Each node keeps `tag` (a height pending for its whole range) and `best` (tallest point
    // inside it). Updates only raise heights, so applying max is monotone and no push-down is
    // needed for point or range-max queries.
    private readonly int _n;
    private readonly int _size;
    private readonly int[] _tag;
    private readonly int[] _best;

    public SkylineTree(int n)
    {
        if (n < 1) throw new Exception("n must be >= 1");
        int power = 1;
        while (power < n) power <<= 1;
        _n = n;
        _size = power;
        _tag = new int[2 * power];
        _best = new int[2 * power];
    }

    /// <summary>Raises the skyline to at least `height` on the half-open range [left, right).</summary>
    public void Update(int left, int right, int height)
    {
        if (left < 0 || left >= right || right > _n) throw new Exception("invalid range");
        UpdateNode(1, 0, _size, left, right, height);
    }

    /// <summary>Skyline height at `index`.</summary>
    public int Point(int index)
    {
        if (index < 0 || index >= _n) throw new Exception("index out of range");
        int node = 1, lo = 0, hi = _size, height = 0;
        while (true)
        {
            height = Math.Max(height, _tag[node]);
            if (hi - lo == 1) return height;
            int mid = (lo + hi) / 2;
            if (index < mid)
            {
                node = 2 * node;
                hi = mid;
            }
            else
            {
                node = 2 * node + 1;
                lo = mid;
            }
        }
    }

    /// <summary>Tallest point in [left, right); 0 when the range is empty.</summary>
    public int RangeMax(int left, int right)
    {
        if (left < 0 || left > right || right > _n) throw new Exception("invalid range");
        if (left == right) return 0;
        return RangeMaxNode(1, 0, _size, left, right);
    }

    public int[] Skyline()
    {
        var heights = new int[_n];
        for (int i = 0; i < _n; i++) heights[i] = Point(i);
        return heights;
    }

    private void UpdateNode(int node, int lo, int hi, int left, int right, int height)
    {
        if (right <= lo || hi <= left) return;
        if (left <= lo && hi <= right)
        {
            _tag[node] = Math.Max(_tag[node], height);
            _best[node] = Math.Max(_best[node], height);
            return;
        }
        int mid = (lo + hi) / 2;
        UpdateNode(2 * node, lo, mid, left, right, height);
        UpdateNode(2 * node + 1, mid, hi, left, right, height);
        _best[node] = Math.Max(_tag[node], Math.Max(_best[2 * node], _best[2 * node + 1]));
    }

    private int RangeMaxNode(int node, int lo, int hi, int left, int right)
    {
        if (left >= right || right <= lo || hi <= left) return 0;
        if (left <= lo && hi <= right) return _best[node];
        int mid = (lo + hi) / 2;
        return Math.Max(_tag[node], Math.Max(RangeMaxNode(2 * node, lo, mid, left, right),
            RangeMaxNode(2 * node + 1, mid, hi, left, right)));
    }
}

public static class Solution
{
    /// <summary>Oracle: keep a height per position and apply each update directly.</summary>
    public static int[] BruteSchedule(int n, List<int[]> updates)
    {
        var heights = new int[n];
        foreach (int[] update in updates)
        {
            for (int i = update[0]; i < update[1]; i++) heights[i] = Math.Max(heights[i], update[2]);
        }
        return heights;
    }

    private static int seed = 7788;

    private static List<int[]> LcgUpdates(int count, int n)
    {
        var updates = new List<int[]>();
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            int left = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int right = left + 1 + seed % (n - left);
            seed = (seed * 75 + 74) % 65537;
            updates.Add(new[] { left, right, seed % 100 });
        }
        return updates;
    }

    public static void Main()
    {
        // Hand-checked: [1,4)->5, [2,3)->7, [4,6)->2 over 7 positions.
        var tree = new SkylineTree(7);
        foreach (int[] update in new[] { new[] { 1, 4, 5 }, new[] { 2, 3, 7 }, new[] { 4, 6, 2 } })
        {
            tree.Update(update[0], update[1], update[2]);
        }
        Check(string.Join(",", tree.Skyline()) == "0,5,7,5,2,2,0", "hand skyline");
        Check(tree.Point(2) == 7 && tree.Point(6) == 0 && tree.Point(0) == 0, "hand points");
        Check(tree.RangeMax(0, 3) == 7, "range 0..3");
        Check(tree.RangeMax(2, 5) == 7, "range 2..5");
        Check(tree.RangeMax(3, 6) == 5, "range 3..6");
        Check(tree.RangeMax(0, 7) == 7, "range 0..7");
        Check(tree.RangeMax(5, 5) == 0, "empty range");
        Check(tree.RangeMax(6, 7) == 0, "untouched tail");

        tree.Update(2, 3, 4);
        Check(tree.Point(2) == 7, "smaller update is a no-op");
        tree.Update(2, 3, 9);
        Check(tree.Point(2) == 9 && tree.RangeMax(1, 3) == 9, "bigger update raises");

        var tiny = new SkylineTree(1);
        Check(tiny.Point(0) == 0 && tiny.RangeMax(0, 1) == 0, "tiny empty");
        tiny.Update(0, 1, 42);
        Check(tiny.Point(0) == 42 && tiny.RangeMax(0, 1) == 42, "tiny updated");

        var boundary = new SkylineTree(4);
        foreach (int[] update in new[] { new[] { 0, 4, 1 }, new[] { 0, 1, 2 }, new[] { 3, 4, 3 }, new[] { 1, 3, 4 } })
        {
            boundary.Update(update[0], update[1], update[2]);
        }
        Check(string.Join(",", boundary.Skyline()) == "2,4,4,3", "boundary skyline");
        Check(boundary.RangeMax(0, 4) == 4, "boundary full");
        Check(boundary.RangeMax(0, 2) == 4, "boundary left");
        Check(boundary.RangeMax(2, 4) == 4, "boundary right");
        Check(boundary.RangeMax(3, 4) == 3, "boundary tail");

        for (int trial = 0; trial < 20; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 40;
            var updates = LcgUpdates(1 + seed % 60, n);
            var randomTree = new SkylineTree(n);
            foreach (int[] update in updates) randomTree.Update(update[0], update[1], update[2]);
            var expected = BruteSchedule(n, updates);
            Check(string.Join(",", randomTree.Skyline()) == string.Join(",", expected),
                "random skyline " + trial);
            for (int left = 0; left < n; left++)
            {
                int bestSoFar = 0;
                for (int right = left + 1; right <= n; right++)
                {
                    bestSoFar = Math.Max(bestSoFar, expected[right - 1]);
                    Check(randomTree.RangeMax(left, right) == bestSoFar, "random range " + trial);
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
