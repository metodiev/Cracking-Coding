// Task 276 - Implement the Hungarian algorithm for assignment (basic). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns (total cost, assignment) where assignment[row] is the column.</summary>
    public static (long Total, int[] Assignment) Hungarian(int[][] cost)
    {
        int n = cost.Length;
        if (n == 0) return (0, Array.Empty<int>());
        long infinity = long.MaxValue / 4;
        var u = new long[n + 1];
        var v = new long[n + 1];
        var p = new int[n + 1];
        var way = new int[n + 1];

        for (int i = 1; i <= n; i++)
        {
            p[0] = i;
            int j0 = 0;
            var minv = Enumerable.Repeat(infinity, n + 1).ToArray();
            var used = new bool[n + 1];
            while (true)
            {
                used[j0] = true;
                int i0 = p[j0];
                long delta = infinity;
                int j1 = -1;
                for (int j = 1; j <= n; j++)
                {
                    if (!used[j])
                    {
                        long cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                }
                for (int j = 0; j <= n; j++)
                {
                    if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                    else minv[j] -= delta;
                }
                j0 = j1;
                if (p[j0] == 0) break;
            }
            while (j0 != 0)
            {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            }
        }

        var assignment = Enumerable.Repeat(-1, n).ToArray();
        for (int j = 1; j <= n; j++) if (p[j] != 0) assignment[p[j] - 1] = j - 1;
        long total = 0;
        for (int i = 0; i < n; i++) total += cost[i][assignment[i]];
        return (total, assignment);
    }

    public static long BruteForce(int[][] cost)
    {
        int n = cost.Length;
        var perm = Enumerable.Range(0, n).ToArray();
        return Permute(cost, perm, 0);
    }

    private static long Permute(int[][] cost, int[] perm, int start)
    {
        int n = perm.Length;
        if (start == n)
        {
            long total = 0;
            for (int i = 0; i < n; i++) total += cost[i][perm[i]];
            return total;
        }
        long best = long.MaxValue;
        for (int i = start; i < n; i++)
        {
            (perm[start], perm[i]) = (perm[i], perm[start]);
            best = Math.Min(best, Permute(cost, perm, start + 1));
            (perm[start], perm[i]) = (perm[i], perm[start]);
        }
        return best;
    }

    private static int seed = 4242;

    private static int[][] LcgMatrix(int n)
    {
        var matrix = new int[n][];
        for (int i = 0; i < n; i++)
        {
            matrix[i] = new int[n];
            for (int j = 0; j < n; j++)
            {
                seed = (seed * 75 + 74) % 65537;
                matrix[i][j] = seed % 20;
            }
        }
        return matrix;
    }

    public static void Main()
    {
        var result = Hungarian(new[] { new[] { 1, 2 }, new[] { 3, 4 } });
        Check(result.Total == 5 && result.Assignment[0] == 0 && result.Assignment[1] == 1, "2x2 tie");
        Check(Hungarian(new[] { new[] { 10, 19 }, new[] { 20, 11 } }).Total == 21, "2x2 classic");
        Check(Hungarian(new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6 }, new[] { 7, 8, 9 } }).Total == 15, "3x3");
        var single = Hungarian(new[] { new[] { 7 } });
        Check(single.Total == 7 && single.Assignment[0] == 0, "1x1");
        Check(Hungarian(Array.Empty<int[]>()).Total == 0, "empty");
        Check(Hungarian(new[] { new[] { 9, 1, 1 }, new[] { 1, 9, 1 }, new[] { 1, 1, 9 } }).Total == 3, "anti-diagonal");

        for (int size = 2; size <= 5; size++)
        {
            for (int trial = 0; trial < 15; trial++)
            {
                var matrix = LcgMatrix(size);
                var (total, assignment) = Hungarian(matrix);
                Check(total == BruteForce(matrix), $"random size {size} trial {trial}");
                var seen = new bool[size];
                for (int i = 0; i < size; i++)
                {
                    int column = assignment[i];
                    Check(column >= 0 && column < size && !seen[column], "permutation " + trial);
                    seen[column] = true;
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
