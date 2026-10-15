// Task 100 - Check if matrix is symmetric. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static bool IsSymmetric(int[][] m)
    {
        int n = m.Length;
        foreach (int[] row in m)
        {
            if (row.Length != n) return false;
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (m[i][j] != m[j][i]) return false;
            }
        }
        return true;
    }

    public static void Main()
    {
        Check(IsSymmetric(new[] { new[] { 1, 2 }, new[] { 2, 1 } }), "2x2 symmetric");
        Check(!IsSymmetric(new[] { new[] { 1, 2 }, new[] { 3, 4 } }), "2x2 asymmetric");
        Check(IsSymmetric(new[] { new[] { 5 } }), "1x1");
        Check(IsSymmetric(new int[0][]), "empty");
        Check(!IsSymmetric(new[] { new int[0] }), "one empty row");
        Check(IsSymmetric(new[] { new[] { 1, 2, 3 }, new[] { 2, 4, 5 }, new[] { 3, 5, 6 } }), "3x3 symmetric");
        Check(!IsSymmetric(new[] { new[] { 1, 2, 3 }, new[] { 2, 4, 5 }, new[] { 3, 6, 6 } }), "3x3 asymmetric");
        Check(!IsSymmetric(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } }), "non-square");
        Check(!IsSymmetric(new[] { new[] { 1, 2 }, new[] { 2 } }), "ragged");
        Check(IsSymmetric(new[] { new[] { 0, -1 }, new[] { -1, 2 } }), "negatives symmetric");
        Check(!IsSymmetric(new[] { new[] { 0, -1 }, new[] { 1, 2 } }), "negatives asymmetric");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
