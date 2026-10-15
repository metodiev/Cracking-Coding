// Task 444 - Implement 2D segment tree for range sum rectangle queries. (Tier: Advanced, CTCI Ch 4)
using System;

internal class SegmentTree2D
{
    private readonly int rows;
    private readonly int cols;
    private readonly long[][] grid;
    private readonly long[][] tree;

    public SegmentTree2D(long[][] grid)
    {
        rows = grid.Length;
        cols = rows > 0 ? grid[0].Length : 0;
        this.grid = new long[rows][];
        for (int r = 0; r < rows; r++) this.grid[r] = (long[])grid[r].Clone();
        tree = new long[4 * Math.Max(1, rows)][];
        for (int i = 0; i < tree.Length; i++) tree[i] = new long[4 * Math.Max(1, cols)];
        if (rows > 0) Build(1, 0, rows - 1, grid);
    }

    private void InnerBuild(long[] arr, int node, int lo, int hi, long[] row)
    {
        if (lo == hi)
        {
            arr[node] = row[lo];
            return;
        }
        int mid = (lo + hi) >> 1;
        InnerBuild(arr, 2 * node, lo, mid, row);
        InnerBuild(arr, 2 * node + 1, mid + 1, hi, row);
        arr[node] = arr[2 * node] + arr[2 * node + 1];
    }

    private void Build(int node, int lo, int hi, long[][] values)
    {
        if (lo == hi)
        {
            InnerBuild(tree[node], 1, 0, cols - 1, values[lo]);
            return;
        }
        int mid = (lo + hi) >> 1;
        Build(2 * node, lo, mid, values);
        Build(2 * node + 1, mid + 1, hi, values);
        long[] current = tree[node];
        long[] left = tree[2 * node];
        long[] right = tree[2 * node + 1];
        for (int k = 1; k < current.Length; k++) current[k] = left[k] + right[k];
    }

    public void SetValue(int row, int col, long value)
    {
        long delta = value - grid[row][col];
        grid[row][col] = value;
        OuterUpdate(1, 0, rows - 1, row, col, delta);
    }

    private void OuterUpdate(int node, int lo, int hi, int row, int col, long delta)
    {
        InnerUpdate(tree[node], 1, 0, cols - 1, col, delta);
        if (lo == hi) return;
        int mid = (lo + hi) >> 1;
        if (row <= mid) OuterUpdate(2 * node, lo, mid, row, col, delta);
        else OuterUpdate(2 * node + 1, mid + 1, hi, row, col, delta);
    }

    private void InnerUpdate(long[] arr, int node, int lo, int hi, int pos, long delta)
    {
        arr[node] += delta;
        if (lo == hi) return;
        int mid = (lo + hi) >> 1;
        if (pos <= mid) InnerUpdate(arr, 2 * node, lo, mid, pos, delta);
        else InnerUpdate(arr, 2 * node + 1, mid + 1, hi, pos, delta);
    }

    private long InnerQuery(long[] arr, int node, int lo, int hi, int left, int right)
    {
        if (left <= lo && hi <= right) return arr[node];
        int mid = (lo + hi) >> 1;
        long total = 0;
        if (left <= mid) total += InnerQuery(arr, 2 * node, lo, mid, left, right);
        if (right > mid) total += InnerQuery(arr, 2 * node + 1, mid + 1, hi, left, right);
        return total;
    }

    public long RectSum(int r1, int c1, int r2, int c2)
    {
        return Rect(1, 0, rows - 1, r1, c1, r2, c2);
    }

    private long Rect(int node, int lo, int hi, int r1, int c1, int r2, int c2)
    {
        if (r2 < lo || hi < r1) return 0;
        if (r1 <= lo && hi <= r2) return InnerQuery(tree[node], 1, 0, cols - 1, c1, c2);
        int mid = (lo + hi) >> 1;
        return Rect(2 * node, lo, mid, r1, c1, r2, c2)
             + Rect(2 * node + 1, mid + 1, hi, r1, c1, r2, c2);
    }
}

public static class Solution
{
    private static int seed;

    private static int[][] LcgGridOps(int count, int n)
    {
        var ops = new int[count][];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            int r1 = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int c1 = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int r2 = r1 + seed % (n - r1);
            seed = (seed * 75 + 74) % 65537;
            int c2 = c1 + seed % (n - c1);
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 41 - 20;
            ops[k] = new[] { kind, r1, c1, r2, c2, value };
        }
        return ops;
    }

    public static void Main()
    {
        var tree = new SegmentTree2D(new long[][] { new long[] { 1, 2, 3 }, new long[] { 4, 5, 6 } });
        Check(tree.RectSum(0, 0, 0, 1) == 3, "top row prefix");
        Check(tree.RectSum(0, 0, 1, 2) == 21, "whole grid");
        Check(tree.RectSum(1, 0, 1, 1) == 9, "bottom row prefix");
        tree.SetValue(1, 1, 50);
        Check(tree.RectSum(1, 1, 1, 1) == 50, "updated cell");
        Check(tree.RectSum(0, 0, 1, 2) == 66, "whole grid after update");
        Check(tree.RectSum(1, 0, 1, 1) == 54, "bottom row after update");
        tree.SetValue(0, 2, 1);
        Check(tree.RectSum(0, 2, 0, 2) == 1, "second update");
        Check(tree.RectSum(0, 0, 1, 2) == 64, "whole grid after second update");

        int n = 16;
        int[][] ops = LcgGridOps(300, n);
        seed = 99;
        var grid = new long[n][];
        for (int r = 0; r < n; r++) grid[r] = new long[n];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 50 - 25;
            }
        }
        tree = new SegmentTree2D(grid);
        var brute = new long[n][];
        for (int r = 0; r < n; r++) brute[r] = (long[])grid[r].Clone();
        foreach (var op in ops)
        {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0)
            {
                tree.SetValue(r1, c1, value);
                brute[r1][c1] = value;
            }
            else
            {
                long expected = 0;
                for (int r = r1; r <= r2; r++)
                {
                    for (int c = c1; c <= c2; c++) expected += brute[r][c];
                }
                Check(tree.RectSum(r1, c1, r2, c2) == expected, "random rectangle");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
