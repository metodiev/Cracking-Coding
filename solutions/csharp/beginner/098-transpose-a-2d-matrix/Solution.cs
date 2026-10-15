// Task 098 - Transpose a 2D matrix. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[][] Transpose(int[][] matrix)
    {
        if (matrix.Length == 0) return new int[0][];
        int rows = matrix.Length;
        int cols = matrix[0].Length;
        var result = new int[cols][];
        for (int c = 0; c < cols; c++)
        {
            result[c] = new int[rows];
            for (int r = 0; r < rows; r++)
            {
                result[c][r] = matrix[r][c];
            }
        }
        return result;
    }

    public static void Main()
    {
        Check(DeepEquals(Transpose(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } }), new[] { new[] { 1, 4 }, new[] { 2, 5 }, new[] { 3, 6 } }), "2x3");
        Check(DeepEquals(Transpose(new[] { new[] { 1, 2 }, new[] { 3, 4 } }), new[] { new[] { 1, 3 }, new[] { 2, 4 } }), "2x2");
        Check(DeepEquals(Transpose(new[] { new[] { 1, 2, 3 } }), new[] { new[] { 1 }, new[] { 2 }, new[] { 3 } }), "1x3");
        Check(DeepEquals(Transpose(new[] { new[] { 1 }, new[] { 2 }, new[] { 3 } }), new[] { new[] { 1, 2, 3 } }), "3x1");
        Check(DeepEquals(Transpose(new[] { new[] { 1 } }), new[] { new[] { 1 } }), "1x1");
        Check(DeepEquals(Transpose(new int[0][]), new int[0][]), "empty");
        Check(DeepEquals(Transpose(new[] { new int[0] }), new int[0][]), "one empty row");
        Check(DeepEquals(Transpose(new[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5, 6 } }), new[] { new[] { 1, 3, 5 }, new[] { 2, 4, 6 } }), "3x2");

        var m = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        int[][] t = Transpose(m);
        Check(!ReferenceEquals(t, m), "returns a new matrix");
        Check(DeepEquals(m, new[] { new[] { 1, 2 }, new[] { 3, 4 } }), "input unchanged");
        Console.WriteLine("PASS");
    }

    private static bool DeepEquals(int[][] a, int[][] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].Length != b[i].Length) return false;
            for (int j = 0; j < a[i].Length; j++)
            {
                if (a[i][j] != b[i][j]) return false;
            }
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
