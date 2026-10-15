// Task 322 - Implement algorithm to find skyline (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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

public static class Solution
{
    /// <summary>Buildings are {left, right, height}; returns critical points {x, height}.</summary>
    public static List<int[]> SkylineDivideConquer(int[][] buildings) => Divide(buildings, 0, buildings.Length);

    private static List<int[]> Divide(int[][] buildings, int from, int to)
    {
        if (from >= to) return new List<int[]>();
        if (to - from == 1)
        {
            return new List<int[]>
            {
                new[] { buildings[from][0], buildings[from][2] },
                new[] { buildings[from][1], 0 },
            };
        }
        int mid = (from + to) / 2;
        return MergeSkylines(Divide(buildings, from, mid), Divide(buildings, mid, to));
    }

    /// <summary>Linear merge of two x-sorted skylines of {x, height} critical points.</summary>
    public static List<int[]> MergeSkylines(List<int[]> a, List<int[]> b)
    {
        var merged = new List<int[]>();
        int i = 0, j = 0, heightA = 0, heightB = 0;
        while (i < a.Count || j < b.Count)
        {
            int x;
            if (j >= b.Count || (i < a.Count && a[i][0] < b[j][0]))
            {
                x = a[i][0];
                heightA = a[i][1];
                i++;
            }
            else if (i >= a.Count || b[j][0] < a[i][0])
            {
                x = b[j][0];
                heightB = b[j][1];
                j++;
            }
            else
            {
                x = a[i][0];
                heightA = a[i][1];
                heightB = b[j][1];
                i++;
                j++;
            }
            int height = Math.Max(heightA, heightB);
            if (merged.Count == 0 || merged[^1][1] != height) merged.Add(new[] { x, height });
        }
        return merged;
    }

    /// <summary>Independent oracle: sweep all boundaries keeping a count of active heights.</summary>
    public static List<int[]> SkylineSweep(int[][] buildings)
    {
        if (buildings.Length == 0) return new List<int[]>();
        int maxHeight = 0;
        var starts = new SortedDictionary<int, List<int>>();
        var ends = new SortedDictionary<int, List<int>>();
        var xs = new SortedSet<int>();
        foreach (int[] building in buildings)
        {
            if (!starts.TryGetValue(building[0], out var startList)) starts[building[0]] = startList = new List<int>();
            startList.Add(building[2]);
            if (!ends.TryGetValue(building[1], out var endList)) ends[building[1]] = endList = new List<int>();
            endList.Add(building[2]);
            maxHeight = Math.Max(maxHeight, building[2]);
            xs.Add(building[0]);
            xs.Add(building[1]);
        }
        var counts = new int[maxHeight + 1];
        int current = 0;
        var result = new List<int[]>();
        foreach (int x in xs)
        {
            if (ends.TryGetValue(x, out var endList)) foreach (int height in endList) counts[height]--; // active on [left, right)
            if (starts.TryGetValue(x, out var startList))
            {
                foreach (int height in startList)
                {
                    counts[height]++;
                    current = Math.Max(current, height);
                }
            }
            while (current > 0 && counts[current] == 0) current--;
            if (result.Count == 0 || result[^1][1] != current) result.Add(new[] { x, current });
        }
        return result;
    }

    private static int[][] LcgBuildings(int count, Lcg lcg)
    {
        var buildings = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int left = lcg.Next(40);
            buildings[i] = new[] { left, left + 1 + lcg.Next(10), 1 + lcg.Next(20) };
        }
        return buildings;
    }

    private static bool SamePoints(List<int[]> a, List<int[]> b) =>
        a.Count == b.Count && a.Zip(b).All(pair => pair.First.SequenceEqual(pair.Second));

    private static readonly int[][] Canonical =
    {
        new[] { 2, 9, 10 }, new[] { 3, 7, 15 }, new[] { 5, 12, 12 }, new[] { 15, 20, 10 }, new[] { 19, 24, 8 },
    };

    private static readonly int[][] CanonicalSkyline =
    {
        new[] { 2, 10 }, new[] { 3, 15 }, new[] { 7, 12 }, new[] { 12, 0 }, new[] { 15, 10 }, new[] { 20, 8 }, new[] { 24, 0 },
    };

    public static void Main()
    {
        Check(SamePoints(SkylineDivideConquer(Canonical), CanonicalSkyline.ToList()), "canonical divide and conquer");
        Check(SamePoints(SkylineSweep(Canonical), CanonicalSkyline.ToList()), "canonical sweep");

        Check(SkylineDivideConquer(new int[0][]).Count == 0, "empty divide and conquer");
        Check(SkylineSweep(new int[0][]).Count == 0, "empty sweep");
        Check(SkylineDivideConquer(new[] { new[] { 1, 3, 4 } }).Select(p => string.Join(",", p)).SequenceEqual(new[] { "1,4", "3,0" }),
            "single building");
        Check(SkylineDivideConquer(new[] { new[] { 0, 2, 5 }, new[] { 2, 4, 7 } })
            .Select(p => string.Join(",", p)).SequenceEqual(new[] { "0,5", "2,7", "4,0" }), "touching buildings");
        Check(SkylineDivideConquer(new[] { new[] { 0, 2, 5 }, new[] { 5, 8, 3 } })
            .Select(p => string.Join(",", p)).SequenceEqual(new[] { "0,5", "2,0", "5,3", "8,0" }), "gap produces zero stretch");

        var lcg = new Lcg(1123);
        for (int trial = 0; trial < 300; trial++)
        {
            int count = 1 + lcg.Next(8);
            int[][] buildings = LcgBuildings(count, lcg);
            Check(SamePoints(SkylineDivideConquer(buildings), SkylineSweep(buildings)),
                "random sweep cross-check " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
