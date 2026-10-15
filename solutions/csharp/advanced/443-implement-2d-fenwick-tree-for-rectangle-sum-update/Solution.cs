// Task 443 - Implement 2D Fenwick tree for rectangle sum/update. (Tier: Advanced, CTCI Ch 4)
using System;

internal class Fenwick2D
{
    private readonly int rows;
    private readonly int cols;
    private readonly long[][] tree;

    public Fenwick2D(int rows, int cols)
    {
        this.rows = rows;
        this.cols = cols;
        tree = new long[rows + 1][];
        for (int i = 0; i <= rows; i++) tree[i] = new long[cols + 1];
    }

    public void Add(int row, int col, long delta)
    {
        for (int i = row + 1; i <= rows; i += i & -i)
        {
            for (int j = col + 1; j <= cols; j += j & -j)
            {
                tree[i][j] += delta;
            }
        }
    }

    public long Prefix(int row, int col)
    {
        long total = 0;
        for (int i = row + 1; i > 0; i -= i & -i)
        {
            for (int j = col + 1; j > 0; j -= j & -j)
            {
                total += tree[i][j];
            }
        }
        return total;
    }

    public long RectSum(int r1, int c1, int r2, int c2)
    {
        return Prefix(r2, c2) - Prefix(r1 - 1, c2) - Prefix(r2, c1 - 1) + Prefix(r1 - 1, c1 - 1);
    }
}

internal class DiffFenwick2D
{
    private readonly int rows;
    private readonly int cols;
    private readonly Fenwick2D inner;

    public DiffFenwick2D(int rows, int cols)
    {
        this.rows = rows;
        this.cols = cols;
        inner = new Fenwick2D(rows, cols);
    }

    public void RectAdd(int r1, int c1, int r2, int c2, long delta)
    {
        inner.Add(r1, c1, delta);
        if (r2 + 1 < rows) inner.Add(r2 + 1, c1, -delta);
        if (c2 + 1 < cols) inner.Add(r1, c2 + 1, -delta);
        if (r2 + 1 < rows && c2 + 1 < cols) inner.Add(r2 + 1, c2 + 1, delta);
    }

    public long Point(int row, int col)
    {
        return inner.Prefix(row, col);
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
            int value = seed % 21 - 10;
            ops[k] = new[] { kind, r1, c1, r2, c2, value };
        }
        return ops;
    }

    public static void Main()
    {
        var bit = new Fenwick2D(4, 4);
        bit.Add(0, 0, 1);
        bit.Add(1, 2, 5);
        bit.Add(3, 3, 2);
        Check(bit.RectSum(0, 0, 3, 3) == 8, "full rectangle");
        Check(bit.RectSum(1, 2, 1, 2) == 5, "single cell");
        Check(bit.RectSum(0, 1, 2, 1) == 0, "empty column strip");
        Check(bit.RectSum(0, 0, 0, 0) == 1, "corner cell");
        Check(bit.RectSum(1, 2, 3, 3) == 7, "offset block");

        var diff = new DiffFenwick2D(4, 4);
        diff.RectAdd(0, 0, 1, 1, 3);
        Check(diff.Point(0, 0) == 3, "covered point");
        Check(diff.Point(1, 1) == 3, "covered corner");
        Check(diff.Point(1, 2) == 0, "outside point");
        Check(diff.Point(2, 2) == 0, "far point");
        diff.RectAdd(1, 1, 2, 2, 10);
        Check(diff.Point(1, 1) == 13, "overlap sum");
        Check(diff.Point(2, 2) == 10, "second block");
        Check(diff.Point(3, 3) == 0, "still outside");
        Check(diff.Point(0, 0) == 3, "first block intact");

        int n = 16;
        int[][] ops = LcgGridOps(300, n);

        bit = new Fenwick2D(n, n);
        var grid = new long[n][];
        for (int r = 0; r < n; r++) grid[r] = new long[n];
        foreach (var op in ops)
        {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0)
            {
                bit.Add(r1, c1, value);
                grid[r1][c1] += value;
            }
            else
            {
                long expected = 0;
                for (int r = r1; r <= r2; r++)
                {
                    for (int c = c1; c <= c2; c++) expected += grid[r][c];
                }
                Check(bit.RectSum(r1, c1, r2, c2) == expected, "random rectangle");
            }
        }

        diff = new DiffFenwick2D(n, n);
        for (int r = 0; r < n; r++) grid[r] = new long[n];
        foreach (var op in ops)
        {
            int kind = op[0], r1 = op[1], c1 = op[2], r2 = op[3], c2 = op[4], value = op[5];
            if (kind == 0)
            {
                diff.RectAdd(r1, c1, r2, c2, value);
                for (int r = r1; r <= r2; r++)
                {
                    for (int c = c1; c <= c2; c++) grid[r][c] += value;
                }
            }
            else
            {
                Check(diff.Point(r1, c1) == grid[r1][c1], "random diff point");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
