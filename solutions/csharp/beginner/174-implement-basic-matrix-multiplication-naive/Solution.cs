// Task 174 - Implement basic matrix multiplication (naive). (Tier: Beginner, CTCI Ch 6)
using System;

public static class Solution
{
    public static int[][] Matmul(int[][] a, int[][] b)
    {
        if (a.Length == 0 || b.Length == 0) return new int[0][];
        int n = a.Length, m = a[0].Length;
        int m2 = b.Length, p = b[0].Length;
        if (m != m2)
            throw new ArgumentException($"dimension mismatch: {n}x{m} vs {m2}x{p}");
        var c = new int[n][];
        for (int i = 0; i < n; i++)
        {
            c[i] = new int[p];
            for (int j = 0; j < p; j++)
            {
                int total = 0;
                for (int k = 0; k < m; k++) total += a[i][k] * b[k][j];
                c[i][j] = total;
            }
        }
        return c;
    }

    public static void Main()
    {
        var product = Matmul(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 } },
                             new[] { new[] { 7, 8 }, new[] { 9, 10 }, new[] { 11, 12 } });
        Check(product[0][0] == 58 && product[0][1] == 64 && product[1][0] == 139 && product[1][1] == 154, "2x3 * 3x2");

        var identity = new[] { new[] { 1, 0 }, new[] { 0, 1 } };
        var m = new[] { new[] { 5, 6 }, new[] { 7, 8 } };
        var withId = Matmul(m, identity);
        Check(withId[0][0] == 5 && withId[1][1] == 8 && withId[0][1] == 6, "identity");

        Check(Matmul(new[] { new[] { 1, 2 } }, new[] { new[] { 3 }, new[] { 4 } })[0][0] == 11, "1x2 * 2x1");

        bool threw = false;
        try { Matmul(new[] { new[] { 1, 2 } }, new[] { new[] { 1, 2 } }); }
        catch (ArgumentException) { threw = true; }
        Check(threw, "mismatch rejected");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
