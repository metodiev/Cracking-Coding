// Task 423 - Implement offline queries with divide & conquer on value (CDQ). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal class CdqCounter
{
    private readonly int[] xs;
    private readonly int[] ys;
    private readonly int[] ranks;
    private readonly int[] bit;
    private readonly int rankCount;
    private long total;

    public CdqCounter(int[][] points)
    {
        int n = points.Length;
        xs = new int[n];
        ys = new int[n];
        for (int i = 0; i < n; i++)
        {
            xs[i] = points[i][0];
            ys[i] = points[i][1];
        }
        var sortedY = (int[])ys.Clone();
        Array.Sort(sortedY);
        var rankOf = new Dictionary<int, int>();
        foreach (int value in sortedY)
        {
            if (!rankOf.ContainsKey(value)) rankOf[value] = rankOf.Count;
        }
        ranks = new int[n];
        for (int i = 0; i < n; i++) ranks[i] = rankOf[ys[i]];
        rankCount = rankOf.Count;
        bit = new int[rankCount + 1];
    }

    public long Count()
    {
        if (xs.Length < 2) return 0;
        Cdq(0, xs.Length);
        return total;
    }

    private void BitAdd(int rank, int delta)
    {
        int i = rank + 1;
        while (i <= rankCount)
        {
            bit[i] += delta;
            i += i & -i;
        }
    }

    private int BitPrefix(int count)          // active values with rank < count
    {
        int result = 0;
        int i = count;
        while (i > 0)
        {
            result += bit[i];
            i -= i & -i;
        }
        return result;
    }

    private int[] Cdq(int lo, int hi)
    {
        // returns indices [lo, hi) sorted by x descending; adds cross pairs to total
        if (hi - lo <= 1) return new[] { lo };
        int mid = (lo + hi) / 2;
        int[] left = Cdq(lo, mid);
        int[] right = Cdq(mid, hi);
        var touched = new int[left.Length];
        int touchedCount = 0;
        int pointer = 0;
        foreach (int j in right)              // right is sorted by x descending too
        {
            while (pointer < left.Length && xs[left[pointer]] > xs[j])
            {
                int rank = ranks[left[pointer]];
                BitAdd(rank, 1);
                touched[touchedCount++] = rank;
                pointer++;
            }
            total += pointer - BitPrefix(ranks[j] + 1);
        }
        for (int i = 0; i < touchedCount; i++) BitAdd(touched[i], -1);
        var merged = new int[left.Length + right.Length];
        int a = 0, b = 0, k = 0;
        while (a < left.Length && b < right.Length)
        {
            if (xs[left[a]] >= xs[right[b]]) merged[k++] = left[a++];
            else merged[k++] = right[b++];
        }
        while (a < left.Length) merged[k++] = left[a++];
        while (b < right.Length) merged[k++] = right[b++];
        return merged;
    }
}

public static class Solution
{
    private static long BruteCount(int[][] points)
    {
        long total = 0;
        for (int i = 0; i < points.Length; i++)
        {
            for (int j = i + 1; j < points.Length; j++)
            {
                if (points[i][0] > points[j][0] && points[i][1] > points[j][1]) total++;
            }
        }
        return total;
    }

    public static void Main()
    {
        Check(new CdqCounter(Array.Empty<int[]>()).Count() == 0, "empty");
        Check(new CdqCounter(new[] { new[] { 1, 1 } }).Count() == 0, "single");
        Check(new CdqCounter(new[] { new[] { 3, 1 }, new[] { 2, 2 } }).Count() == 0, "first fails y");
        Check(new CdqCounter(new[] { new[] { 2, 2 }, new[] { 3, 1 } }).Count() == 0, "x order");
        Check(new CdqCounter(new[] { new[] { 5, 5 }, new[] { 3, 4 }, new[] { 1, 2 } }).Count() == 3, "decreasing chain");
        Check(new CdqCounter(new[] { new[] { 1, 1 }, new[] { 2, 2 }, new[] { 3, 3 } }).Count() == 0, "increasing chain");
        Check(new CdqCounter(new[] { new[] { 3, 3 }, new[] { 3, 2 } }).Count() == 0, "equal x not counted");
        Check(new CdqCounter(new[] { new[] { 4, 4 }, new[] { 3, 4 } }).Count() == 0, "equal y not counted");
        var decreasing = new int[5][];
        for (int i = 0; i < 5; i++) decreasing[i] = new[] { 7 - i, 9 - i };
        Check(new CdqCounter(decreasing).Count() == 10, "all pairs of a decreasing chain");
        Check(new CdqCounter(new[] { new[] { 5, 1 }, new[] { 4, 4 }, new[] { 6, 2 }, new[] { 3, 3 } }).Count() == 1, "mixed points");

        int local = 424242;
        for (int trial = 0; trial < 40; trial++)
        {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 59;
            var points = new int[n][];
            for (int i = 0; i < n; i++)
            {
                local = (local * 75 + 74) % 65537;
                int x = local % 11 - 5;
                local = (local * 75 + 74) % 65537;
                points[i] = new[] { x, local % 11 - 5 };
            }
            long got = new CdqCounter(points).Count();
            long expected = BruteCount(points);
            Check(got == expected, "lcg trial " + trial + " got=" + got + " want=" + expected);
        }

        var equal = new int[12][];
        for (int i = 0; i < 12; i++) equal[i] = new[] { 7, 7 };
        Check(new CdqCounter(equal).Count() == 0, "all-equal points");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
