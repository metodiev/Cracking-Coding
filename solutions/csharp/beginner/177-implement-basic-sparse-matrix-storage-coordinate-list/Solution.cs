// Task 177 - Implement basic sparse matrix storage (coordinate list). (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class SparseMatrix
    {
        private readonly int rows, cols;
        private readonly Dictionary<(int, int), int> data = new Dictionary<(int, int), int>();

        public SparseMatrix(int rows, int cols)
        {
            if (rows < 0 || cols < 0) throw new ArgumentException("dimensions must be non-negative");
            this.rows = rows;
            this.cols = cols;
        }

        private void CheckIndex(int row, int col)
        {
            if (row < 0 || row >= rows || col < 0 || col >= cols)
                throw new IndexOutOfRangeException($"index out of range: ({row}, {col})");
        }

        public void Set(int row, int col, int value)
        {
            CheckIndex(row, col);
            if (value == 0) data.Remove((row, col));
            else data[(row, col)] = value;
        }

        public int Get(int row, int col)
        {
            CheckIndex(row, col);
            return data.TryGetValue((row, col), out var v) ? v : 0;
        }

        public int Nnz => data.Count;

        public int[][] ToDense()
        {
            var dense = new int[rows][];
            for (int r = 0; r < rows; r++) dense[r] = new int[cols];
            foreach (var kv in data) dense[kv.Key.Item1][kv.Key.Item2] = kv.Value;
            return dense;
        }
    }

    public static void Main()
    {
        var m = new SparseMatrix(3, 4);
        m.Set(0, 0, 5);
        m.Set(1, 2, 7);
        m.Set(2, 3, -1);
        Check(m.Nnz == 3, "three stored");
        Check(m.Get(0, 0) == 5 && m.Get(0, 1) == 0 && m.Get(2, 3) == -1, "get");
        var dense = m.ToDense();
        Check(dense[0][0] == 5 && dense[1][2] == 7 && dense[2][3] == -1, "dense");

        m.Set(1, 2, 0);
        Check(m.Nnz == 2 && m.Get(1, 2) == 0, "zero removes");

        bool threw = false;
        try { m.Get(5, 0); } catch (IndexOutOfRangeException) { threw = true; }
        Check(threw, "bounds checked");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
