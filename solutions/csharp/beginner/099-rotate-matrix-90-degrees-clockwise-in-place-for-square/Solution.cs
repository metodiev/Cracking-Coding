// Task 099 - Rotate matrix 90 degrees clockwise (in-place for square). (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static void RotateMatrix90Clockwise(int[][] m)
    {
        int n = m.Length;
        foreach (int[] row in m)
        {
            if (row.Length != n) throw new ArgumentException("matrix must be square");
        }
        // Transpose across the main diagonal.
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                int tmp = m[i][j];
                m[i][j] = m[j][i];
                m[j][i] = tmp;
            }
        }
        // Reverse every row.
        foreach (int[] row in m)
        {
            for (int lo = 0, hi = row.Length - 1; lo < hi; lo++, hi--)
            {
                int tmp = row[lo];
                row[lo] = row[hi];
                row[hi] = tmp;
            }
        }
    }

    public static void Main()
    {
        var m1 = new[] { new[] { 1 } };
        RotateMatrix90Clockwise(m1);
        Check(DeepEquals(m1, new[] { new[] { 1 } }), "1x1");

        var m2 = new[] { new[] { 1, 2 }, new[] { 3, 4 } };
        RotateMatrix90Clockwise(m2);
        Check(DeepEquals(m2, new[] { new[] { 3, 1 }, new[] { 4, 2 } }), "2x2");

        var m3 = new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 }, new[] { 7, 8, 9 } };
        RotateMatrix90Clockwise(m3);
        Check(DeepEquals(m3, new[] { new[] { 7, 4, 1 }, new[] { 8, 5, 2 }, new[] { 9, 6, 3 } }), "3x3");

        var m4 = new[] { new[] { 1, 2, 3, 4 }, new[] { 5, 6, 7, 8 }, new[] { 9, 10, 11, 12 }, new[] { 13, 14, 15, 16 } };
        RotateMatrix90Clockwise(m4);
        Check(DeepEquals(m4, new[]
        {
            new[] { 13, 9, 5, 1 },
            new[] { 14, 10, 6, 2 },
            new[] { 15, 11, 7, 3 },
            new[] { 16, 12, 8, 4 },
        }), "4x4");

        bool threw = false;
        try
        {
            RotateMatrix90Clockwise(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } });
        }
        catch (ArgumentException)
        {
            threw = true;
        }
        Check(threw, "non-square throws");
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
