// Task 364 - Implement quadtree for 2D region queries. (Tier: Intermediate, CTCI Ch 4)
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

internal sealed class QuadNode
{
    public readonly int Row;
    public readonly int Col;
    public readonly int Size;
    public readonly int Count;
    public readonly int Value;          // uniform leaf value; -1 when the node was split
    public readonly QuadNode[] Children; // null for leaves

    public QuadNode(int row, int col, int size, int count, int value, QuadNode[] children)
    {
        Row = row;
        Col = col;
        Size = size;
        Count = count;
        Value = value;
        Children = children;
    }
}

/// <summary>
/// Split rule: a node is a leaf when size == 1 or all of its cells are equal; otherwise
/// it splits into four quadrants of size / 2 and stores the total count of its region.
/// </summary>
internal sealed class QuadTree
{
    private readonly QuadNode root;

    public QuadTree(int[][] grid)
    {
        int size = grid.Length;
        if (size == 0 || (size & (size - 1)) != 0) throw new Exception("grid side must be a power of two");
        foreach (var row in grid)
        {
            if (row.Length != size) throw new Exception("grid must be square");
        }
        root = Build(grid, 0, 0, size);
    }

    private static QuadNode Build(int[][] grid, int row, int col, int size)
    {
        int first = grid[row][col];
        int total = 0;
        bool uniform = true;
        for (int r = row; r < row + size; r++)
        {
            for (int c = col; c < col + size; c++)
            {
                int cell = grid[r][c];
                total += cell;
                if (cell != first) uniform = false;
            }
        }
        if (uniform || size == 1) return new QuadNode(row, col, size, total, first, null);
        int half = size / 2;
        var children = new[]
        {
            Build(grid, row, col, half),
            Build(grid, row, col + half, half),
            Build(grid, row + half, col, half),
            Build(grid, row + half, col + half, half),
        };
        return new QuadNode(row, col, size, total, -1, children);
    }

    /// <summary>Number of ones in the inclusive rectangle [rowLo..rowHi] x [colLo..colHi].</summary>
    public int CountRect(int rowLo, int rowHi, int colLo, int colHi) =>
        Count(root, rowLo, rowHi, colLo, colHi);

    private static int Count(QuadNode node, int rowLo, int rowHi, int colLo, int colHi)
    {
        if (node.Row > rowHi || node.Row + node.Size - 1 < rowLo ||
                node.Col > colHi || node.Col + node.Size - 1 < colLo)
        {
            return 0;                                        // disjoint
        }
        if (rowLo <= node.Row && node.Row + node.Size - 1 <= rowHi &&
                colLo <= node.Col && node.Col + node.Size - 1 <= colHi)
        {
            return node.Count;                               // fully contained
        }
        if (node.Children == null) {                         // uniform leaf, partial overlap
            int overlapRows = Math.Min(node.Row + node.Size - 1, rowHi) - Math.Max(node.Row, rowLo) + 1;
            int overlapCols = Math.Min(node.Col + node.Size - 1, colHi) - Math.Max(node.Col, colLo) + 1;
            return node.Value * overlapRows * overlapCols;
        }
        int total = 0;
        foreach (var child in node.Children) total += Count(child, rowLo, rowHi, colLo, colHi);
        return total;
    }
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(4919);

    private static int NaiveRectCount(int[][] grid, int rowLo, int rowHi, int colLo, int colHi)
    {
        int total = 0;
        for (int r = rowLo; r <= rowHi; r++)
        {
            for (int c = colLo; c <= colHi; c++) total += grid[r][c];
        }
        return total;
    }

    private static int[][] DiagonalGrid(int size)
    {
        var grid = new int[size][];
        for (int r = 0; r < size; r++)
        {
            grid[r] = new int[size];
            for (int c = 0; c < size; c++) grid[r][c] = r == c ? 1 : 0;
        }
        return grid;
    }

    private static int[][] LcgBinaryGrid(int size)
    {
        var grid = new int[size][];
        for (int r = 0; r < size; r++)
        {
            grid[r] = new int[size];
            for (int c = 0; c < size; c++) grid[r][c] = Random.Next(2);
        }
        return grid;
    }

    public static void Main()
    {
        // Hand-checked diagonal grid: ones only on the main diagonal.
        var tree = new QuadTree(DiagonalGrid(8));
        Check(tree.CountRect(0, 7, 0, 7) == 8, "diagonal full grid");
        Check(tree.CountRect(0, 3, 0, 3) == 4, "top-left quadrant");
        Check(tree.CountRect(2, 5, 2, 5) == 4, "middle 4x4");
        Check(tree.CountRect(1, 6, 0, 7) == 6, "interior rows");
        Check(tree.CountRect(0, 7, 7, 7) == 1, "right column");
        Check(tree.CountRect(0, 0, 1, 7) == 0, "top row minus corner");
        Check(tree.CountRect(3, 4, 3, 4) == 2, "two diagonal cells");
        Check(tree.CountRect(2, 2, 0, 1) == 0, "row 2 left of the diagonal");

        // Uniform grids: all ones and all zeros.
        var onesGrid = new int[8][];
        for (int r = 0; r < 8; r++)
        {
            onesGrid[r] = new int[8];
            for (int c = 0; c < 8; c++) onesGrid[r][c] = 1;
        }
        var ones = new QuadTree(onesGrid);
        Check(ones.CountRect(0, 7, 0, 7) == 64 && ones.CountRect(2, 4, 1, 3) == 9, "all ones");
        var zeros = new QuadTree(new int[8][] { new int[8], new int[8], new int[8], new int[8],
            new int[8], new int[8], new int[8], new int[8] });
        Check(zeros.CountRect(0, 7, 0, 7) == 0, "all zeros");

        var cell = new QuadTree(new[] { new[] { 1 } });
        Check(cell.CountRect(0, 0, 0, 0) == 1, "single cell");

        // 8x8 LCG grids: every possible rectangle against the naive loop.
        for (int trial = 0; trial < 8; trial++)
        {
            int[][] grid = LcgBinaryGrid(8);
            var randomTree = new QuadTree(grid);
            for (int rowLo = 0; rowLo < 8; rowLo++)
            {
                for (int rowHi = rowLo; rowHi < 8; rowHi++)
                {
                    for (int colLo = 0; colLo < 8; colLo++)
                    {
                        for (int colHi = colLo; colHi < 8; colHi++)
                        {
                            Check(randomTree.CountRect(rowLo, rowHi, colLo, colHi) ==
                                    NaiveRectCount(grid, rowLo, rowHi, colLo, colHi),
                                "random trial " + trial + " rectangle");
                        }
                    }
                }
            }
        }

        // A 16x16 grid (deeper splits) with random rectangles.
        for (int trial = 0; trial < 3; trial++)
        {
            int[][] grid = LcgBinaryGrid(16);
            var deepTree = new QuadTree(grid);
            for (int q = 0; q < 60; q++)
            {
                int rowLo = Random.Next(16);
                int rowHi = rowLo + Random.Next(16 - rowLo);
                int colLo = Random.Next(16);
                int colHi = colLo + Random.Next(16 - colLo);
                Check(deepTree.CountRect(rowLo, rowHi, colLo, colHi) ==
                        NaiveRectCount(grid, rowLo, rowHi, colLo, colHi),
                    "16x16 trial " + trial + " rectangle");
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
