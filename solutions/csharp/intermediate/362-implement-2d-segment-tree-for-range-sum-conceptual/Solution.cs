// Task 362 - Implement 2D segment tree for range sum (conceptual). (Tier: Intermediate, CTCI Ch 4)
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
/// A segment tree over rows whose every node stores an inner segment tree over columns;
/// internal outer nodes are the element-wise sum of their two children.
/// </summary>
internal sealed class SegmentTree2D
{
    private readonly int sizeR;
    private readonly int sizeC;
    private readonly int[][] trees;

    public SegmentTree2D(int[][] grid)
    {
        int rows = grid.Length;
        int cols = rows == 0 ? 0 : grid[0].Length;
        sizeR = NextPowerOfTwo(Math.Max(rows, 1));
        sizeC = NextPowerOfTwo(Math.Max(cols, 1));
        trees = new int[2 * sizeR][];
        for (int i = 0; i < trees.Length; i++) trees[i] = new int[2 * sizeC];
        for (int r = 0; r < rows; r++)
        {
            int[] rowTree = trees[sizeR + r];
            for (int c = 0; c < cols; c++) rowTree[sizeC + c] = grid[r][c];
            for (int node = sizeC - 1; node >= 1; node--)
            {
                rowTree[node] = rowTree[2 * node] + rowTree[2 * node + 1];
            }
        }
        for (int node = sizeR - 1; node >= 1; node--)
        {
            int[] left = trees[2 * node];
            int[] right = trees[2 * node + 1];
            int[] merged = trees[node];
            for (int j = 1; j < 2 * sizeC; j++) merged[j] = left[j] + right[j];
        }
    }

    private static int NextPowerOfTwo(int value)
    {
        int size = 1;
        while (size < value) size <<= 1;
        return size;
    }

    /// <summary>Sum over the inclusive rectangle [rowLo..rowHi] x [colLo..colHi].</summary>
    public int RangeSum(int rowLo, int rowHi, int colLo, int colHi)
    {
        if (rowLo > rowHi || colLo > colHi) return 0;
        int total = 0;
        int l = rowLo + sizeR;
        int r = rowHi + 1 + sizeR;
        while (l < r)
        {
            if ((l & 1) == 1) total += InnerSum(trees[l++], colLo, colHi + 1);
            if ((r & 1) == 1) total += InnerSum(trees[--r], colLo, colHi + 1);
            l >>= 1;
            r >>= 1;
        }
        return total;
    }

    private int InnerSum(int[] tree, int colLo, int colHi)
    {
        int total = 0;
        int l = colLo + sizeC;
        int r = colHi + sizeC;
        while (l < r)
        {
            if ((l & 1) == 1) total += tree[l++];
            if ((r & 1) == 1) total += tree[--r];
            l >>= 1;
            r >>= 1;
        }
        return total;
    }
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(31337);

    private static int NaiveRectSum(int[][] grid, int rowLo, int rowHi, int colLo, int colHi)
    {
        int total = 0;
        for (int r = rowLo; r <= rowHi; r++)
        {
            for (int c = colLo; c <= colHi; c++) total += grid[r][c];
        }
        return total;
    }

    private static int[][] LcgGrid(int rows, int cols)
    {
        var grid = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            grid[r] = new int[cols];
            for (int c = 0; c < cols; c++) grid[r][c] = Random.Next(20) - 5;
        }
        return grid;
    }

    public static void Main()
    {
        // Hand-checked sums on the 4x4 grid 1..16.
        var hand = new[]
        {
            new[] { 1, 2, 3, 4 },
            new[] { 5, 6, 7, 8 },
            new[] { 9, 10, 11, 12 },
            new[] { 13, 14, 15, 16 },
        };
        var tree = new SegmentTree2D(hand);
        Check(tree.RangeSum(0, 3, 0, 3) == 136, "full grid 1+...+16");
        Check(tree.RangeSum(0, 1, 0, 1) == 14, "top-left 2x2");
        Check(tree.RangeSum(2, 3, 1, 2) == 50, "bottom-right 2x2 offset");
        Check(tree.RangeSum(1, 1, 0, 3) == 26, "row 1");
        Check(tree.RangeSum(0, 3, 0, 0) == 28, "column 0");
        Check(tree.RangeSum(2, 2, 3, 3) == 12, "single cell");
        Check(tree.RangeSum(3, 0, 0, 3) == 0, "empty row range");
        Check(tree.RangeSum(0, 3, 3, 2) == 0, "empty column range");

        // Degenerate shapes: one cell, one row, one column.
        var single = new SegmentTree2D(new[] { new[] { 7 } });
        Check(single.RangeSum(0, 0, 0, 0) == 7, "single cell grid");
        var row = new SegmentTree2D(new[] { new[] { 1, -2, 3, 4, 5 } });
        Check(row.RangeSum(0, 0, 0, 4) == 11 && row.RangeSum(0, 0, 1, 3) == 5, "single row grid");
        var col = new SegmentTree2D(new[] { new[] { 1 }, new[] { 2 }, new[] { 3 }, new[] { 4 }, new[] { 5 } });
        Check(col.RangeSum(0, 4, 0, 0) == 15 && col.RangeSum(2, 4, 0, 0) == 12, "single column grid");

        // 8x8 LCG grids: every rectangle cross-checked against the naive double loop.
        for (int trial = 0; trial < 6; trial++)
        {
            int[][] grid = LcgGrid(8, 8);
            var randomTree = new SegmentTree2D(grid);
            for (int rowLo = 0; rowLo < 8; rowLo++)
            {
                for (int rowHi = rowLo; rowHi < 8; rowHi++)
                {
                    for (int colLo = 0; colLo < 8; colLo++)
                    {
                        for (int colHi = colLo; colHi < 8; colHi++)
                        {
                            Check(randomTree.RangeSum(rowLo, rowHi, colLo, colHi) ==
                                    NaiveRectSum(grid, rowLo, rowHi, colLo, colHi),
                                "random trial " + trial + " rectangle");
                        }
                    }
                }
            }
        }

        // Non power-of-two shapes (3x5, 5x3, 7x1) still work.
        var shapes = new[] { new[] { 3, 5 }, new[] { 5, 3 }, new[] { 7, 1 } };
        foreach (var shape in shapes)
        {
            int rows = shape[0];
            int cols = shape[1];
            int[][] grid = LcgGrid(rows, cols);
            var shapeTree = new SegmentTree2D(grid);
            for (int rowLo = 0; rowLo < rows; rowLo++)
            {
                for (int rowHi = rowLo; rowHi < rows; rowHi++)
                {
                    for (int colLo = 0; colLo < cols; colLo++)
                    {
                        for (int colHi = colLo; colHi < cols; colHi++)
                        {
                            Check(shapeTree.RangeSum(rowLo, rowHi, colLo, colHi) ==
                                    NaiveRectSum(grid, rowLo, rowHi, colLo, colHi),
                                "shape " + rows + "x" + cols);
                        }
                    }
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
