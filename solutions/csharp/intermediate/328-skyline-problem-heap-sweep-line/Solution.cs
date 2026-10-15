// Task 328 - Skyline problem (heap + sweep line). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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
    /// <summary>Sweep all critical x values; a max-heap keyed by (-height, right) gives the current height.</summary>
    public static List<int[]> SkylineHeap(int[][] buildings)
    {
        if (buildings.Length == 0) return new List<int[]>();
        var starts = new SortedDictionary<int, List<int[]>>(); // left -> list of {right, height}
        var xs = new SortedSet<int>();
        foreach (int[] building in buildings)
        {
            if (!starts.TryGetValue(building[0], out var list)) starts[building[0]] = list = new List<int[]>();
            list.Add(new[] { building[1], building[2] });
            xs.Add(building[0]);
            xs.Add(building[1]);
        }
        var heap = new PriorityQueue<int[], (int, int)>(); // element {right, height}, priority (-height, right)
        var result = new List<int[]>();
        foreach (int x in xs)
        {
            if (starts.TryGetValue(x, out var list))
            {
                foreach (int[] start in list) heap.Enqueue(start, (-start[1], start[0]));
            }
            while (heap.TryPeek(out var top, out _) && top[0] <= x) heap.Dequeue(); // lazy deletion
            int height = heap.Count == 0 ? 0 : heap.Peek()[1];
            if (result.Count == 0 || result[^1][1] != height) result.Add(new[] { x, height });
        }
        return result;
    }

    /// <summary>Cross-check implementation: recursive split with a linear skyline merge.</summary>
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
        Check(SamePoints(SkylineHeap(Canonical), CanonicalSkyline.ToList()), "canonical heap skyline");
        Check(SamePoints(SkylineDivideConquer(Canonical), CanonicalSkyline.ToList()), "canonical divide and conquer skyline");

        Check(SkylineHeap(Array.Empty<int[]>()).Count == 0 && SkylineDivideConquer(Array.Empty<int[]>()).Count == 0, "empty input");
        Check(SkylineHeap(new[] { new[] { 0, 2, 5 }, new[] { 2, 4, 7 } })
            .Select(p => string.Join(",", p)).SequenceEqual(new[] { "0,5", "2,7", "4,0" }), "touching buildings");
        Check(SkylineHeap(new[] { new[] { 0, 2, 5 }, new[] { 5, 8, 3 } })
            .Select(p => string.Join(",", p)).SequenceEqual(new[] { "0,5", "2,0", "5,3", "8,0" }), "gap produces zero stretch");

        var lcg = new Lcg(24601);
        for (int trial = 0; trial < 300; trial++)
        {
            int count = 1 + lcg.Next(8);
            int[][] buildings = LcgBuildings(count, lcg);
            Check(SamePoints(SkylineHeap(buildings), SkylineDivideConquer(buildings)),
                "heap agrees with divide and conquer " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
