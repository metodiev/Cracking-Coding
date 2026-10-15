// Task 369 - Closest pair of points (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;
using System.Collections.Generic;
using System.Linq;

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

/// <summary>
/// O(n log n) divide &amp; conquer.  Each recursive call returns its slice sorted by y
/// (merged from the children), so the strip is already y-sorted; strip pairs with
/// dy^2 >= best_so_far cannot improve the answer and are skipped.  Distances are
/// squared exact integers -- no epsilon anywhere.
/// </summary>
internal sealed class ClosestPairSolver
{
    public long Best;
    public int[] BestA;
    public int[] BestB;

    public static long Dist2(int[] a, int[] b)
    {
        long dx = a[0] - b[0];
        long dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    public void Solve(int[][] pointsByX)
    {
        Best = -1;
        BestA = BestB = null;
        Rec(pointsByX);
    }

    private int[][] Rec(int[][] ptsByX)
    {
        int n = ptsByX.Length;
        if (n <= 3)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    long d = Dist2(ptsByX[i], ptsByX[j]);
                    if (Best < 0 || d < Best)
                    {
                        Best = d;
                        BestA = ptsByX[i];
                        BestB = ptsByX[j];
                    }
                }
            }
            return SortByY(ptsByX);
        }
        int mid = n / 2;
        int[][] leftY = Rec(CopyRange(ptsByX, 0, mid));
        int[][] rightY = Rec(CopyRange(ptsByX, mid, n));
        int[][] merged = MergeByY(leftY, rightY);
        int midX = ptsByX[mid][0];
        var strip = new List<int[]>();
        foreach (var p in merged)
        {
            long dx = p[0] - midX;
            if (dx * dx < Best) strip.Add(p);
        }
        for (int i = 0; i < strip.Count; i++)
        {
            for (int j = i + 1; j < strip.Count; j++)
            {
                long dy = strip[j][1] - strip[i][1];
                if (dy * dy >= Best) break;
                long d = Dist2(strip[i], strip[j]);
                if (d < Best)
                {
                    Best = d;
                    BestA = strip[i];
                    BestB = strip[j];
                }
            }
        }
        return merged;
    }

    private static int[][] CopyRange(int[][] source, int from, int to)
    {
        var result = new int[to - from][];
        Array.Copy(source, from, result, 0, to - from);
        return result;
    }

    private static int[][] SortByY(int[][] pts)
    {
        var copy = (int[][])pts.Clone();
        Array.Sort(copy, (a, b) => a[1] != b[1] ? a[1].CompareTo(b[1]) : a[0].CompareTo(b[0]));
        return copy;
    }

    private static int[][] MergeByY(int[][] a, int[][] b)
    {
        var merged = new int[a.Length + b.Length][];
        int i = 0, j = 0, k = 0;
        while (i < a.Length || j < b.Length)
        {
            if (j >= b.Length || (i < a.Length && (a[i][1] < b[j][1] ||
                    (a[i][1] == b[j][1] && a[i][0] <= b[j][0]))))
            {
                merged[k++] = a[i++];
            }
            else
            {
                merged[k++] = b[j++];
            }
        }
        return merged;
    }
}

public static class Solution
{
    private static readonly Lcg Random = new Lcg(271828);

    private static int[][] SortedPoints(int[][] points)
    {
        var copy = (int[][])points.Clone();
        Array.Sort(copy, (a, b) => a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
        return copy;
    }

    /// <summary>Independent O(n^2) oracle: same (squared distance) semantics.</summary>
    public static long BruteForce(int[][] points)
    {
        long best = -1;
        for (int i = 0; i < points.Length; i++)
        {
            for (int j = i + 1; j < points.Length; j++)
            {
                long d = ClosestPairSolver.Dist2(points[i], points[j]);
                if (best < 0 || d < best) best = d;
            }
        }
        return best;
    }

    public static long ClosestPairSq(int[][] points)
    {
        if (points.Length < 2) throw new Exception("need at least two points");
        var solver = new ClosestPairSolver();
        solver.Solve(SortedPoints(points));
        return solver.Best;
    }

    public static void Main()
    {
        // Hand case: (0,0)-(3,4) and (3,4)-(7,1) both have squared distance 25.
        var hand = new[] { new[] { 0, 0 }, new[] { 3, 4 }, new[] { 7, 1 } };
        Check(ClosestPairSq(hand) == 25, "hand case 25");
        Check(BruteForce(hand) == 25, "hand case brute force 25");

        // Duplicate point: distance 0.
        var dup = new[] { new[] { 5, 5 }, new[] { 5, 5 }, new[] { 9, 9 } };
        Check(ClosestPairSq(dup) == 0, "duplicate distance 0");
        Check(BruteForce(dup) == 0, "duplicate brute force");

        Check(ClosestPairSq(new[] { new[] { 0, 0 }, new[] { -3, -4 } }) == 25, "two points 3-4-5");

        var grid = new List<int[]>();
        for (int x = 0; x < 4; x++)
        {
            for (int y = 0; y < 4; y++) grid.Add(new[] { x, y });
        }
        int[][] gridArray = grid.ToArray();
        Check(ClosestPairSq(gridArray) == 1 && BruteForce(gridArray) == 1, "unit grid");

        // LCG cross-check: sizes up to ~60.
        for (int trial = 0; trial < 30; trial++)
        {
            int size = 2 + Random.Next(59);
            int modulus = 10 + Random.Next(90);
            var points = new int[size][];
            for (int i = 0; i < size; i++)
            {
                points[i] = new[] { Random.Next(modulus), Random.Next(modulus) };
            }
            Check(ClosestPairSq(points) == BruteForce(points),
                "random trial " + trial + " size " + size);
        }

        // Concentrated points: many duplicates and very small distances.
        for (int trial = 0; trial < 10; trial++)
        {
            int size = 2 + Random.Next(30);
            var points = new int[size][];
            for (int i = 0; i < size; i++)
            {
                points[i] = new[] { Random.Next(5), Random.Next(5) };
            }
            Check(ClosestPairSq(points) == BruteForce(points), "dense trial " + trial);
        }

        // Worst-case strip shape: a vertical line.
        var line = new int[60][];
        for (int y = 0; y < 60; y++) line[y] = new[] { 7, y };
        Check(ClosestPairSq(line) == 1, "vertical line");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
