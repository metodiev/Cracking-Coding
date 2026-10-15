// Task 097 - Flatten a 2D array (row-major). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] Flatten2D(int[][] matrix)
    {
        var result = new List<int>();
        foreach (int[] row in matrix)
        {
            foreach (int value in row)
            {
                result.Add(value);
            }
        }
        return result.ToArray();
    }

    public static void Main()
    {
        Check(Same(Flatten2D(new[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5 } }), new[] { 1, 2, 3, 4, 5 }), "ragged");
        Check(Same(Flatten2D(new[] { new[] { 1 }, new[] { 2, 3 }, new[] { 4, 5, 6 } }), new[] { 1, 2, 3, 4, 5, 6 }), "growing rows");
        Check(Same(Flatten2D(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } }), new[] { 1, 2, 3, 4, 5, 6 }), "rectangular");
        Check(Same(Flatten2D(new[] { new[] { 1, 2 }, new[] { 3, 4 } }), new[] { 1, 2, 3, 4 }), "2x2");
        Check(Same(Flatten2D(new[] { new int[0], new[] { 1, 2 }, new int[0] }), new[] { 1, 2 }), "empty rows");
        Check(Same(Flatten2D(new[] { new[] { 1 } }), new[] { 1 }), "single");
        Check(Same(Flatten2D(new int[0][]), new int[0]), "empty matrix");
        Check(Same(Flatten2D(new[] { new int[0] }), new int[0]), "one empty row");
        Check(Same(Flatten2D(new[] { new[] { -1, 0 }, new[] { 0, -1 } }), new[] { -1, 0, 0, -1 }), "negatives");
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
