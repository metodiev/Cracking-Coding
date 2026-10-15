// Task 365 - Skyline of buildings via divide & conquer (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
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
    private static readonly Lcg Random = new Lcg(8113);

    /// <summary>Returns critical points [x, height]; degenerate buildings (zero width/height) ignored.</summary>
    public static int[][] SkylineDivideConquer(int[][] buildings)
    {
        var ordered = buildings.Where(b => b[1] > b[0] && b[2] > 0)
            .Select(b => (int[])b.Clone())
            .OrderBy(b => b[0]).ThenBy(b => b[1]).ThenBy(b => b[2])
            .ToArray();
        return Solve(ordered, 0, ordered.Length);
    }

    private static int[][] Solve(int[][] ordered, int from, int to)
    {
        if (from >= to) return Array.Empty<int[]>();
        if (to - from == 1)
        {
            return new[] { new[] { ordered[from][0], ordered[from][2] },
                new[] { ordered[from][1], 0 } };
        }
        int mid = from + (to - from) / 2;
        return Merge(Solve(ordered, from, mid), Solve(ordered, mid, to));
    }

    /// <summary>Linear two-pointer merge of two x-sorted skylines of [x, height] critical points.</summary>
    private static int[][] Merge(int[][] skyA, int[][] skyB)
    {
        var merged = new List<int[]>();
        int i = 0, j = 0;
        int heightA = 0, heightB = 0;
        while (i < skyA.Length || j < skyB.Length)
        {
            int x;
            if (j >= skyB.Length || (i < skyA.Length && skyA[i][0] < skyB[j][0]))
            {
                x = skyA[i][0];
                heightA = skyA[i][1];
                i++;
            }
            else if (i >= skyA.Length || skyB[j][0] < skyA[i][0])
            {
                x = skyB[j][0];
                heightB = skyB[j][1];
                j++;
            }
            else
            {
                x = skyA[i][0];
                heightA = skyA[i][1];
                heightB = skyB[j][1];
                i++;
                j++;
            }
            int height = Math.Max(heightA, heightB);
            if (merged.Count == 0 || merged[merged.Count - 1][1] != height)
            {
                merged.Add(new[] { x, height });
            }
        }
        return merged.ToArray();
    }

    /// <summary>Independent oracle: sweep every boundary x keeping counts of active heights on [l, r).</summary>
    public static int[][] SkylineSweep(int[][] buildings)
    {
        var events = new SortedDictionary<int, List<int[]>>();
        void AddEvent(int x, int height, int delta)
        {
            if (!events.TryGetValue(x, out var list))
            {
                list = new List<int[]>();
                events[x] = list;
            }
            list.Add(new[] { height, delta });
        }
        foreach (var b in buildings)
        {
            if (b[1] <= b[0] || b[2] <= 0) continue;
            AddEvent(b[0], b[2], 1);
            AddEvent(b[1], b[2], -1);
        }
        var counts = new Dictionary<int, int>();
        var result = new List<int[]>();
        foreach (var entry in events)
        {
            foreach (var evt in entry.Value)
            {
                counts.TryGetValue(evt[0], out int currentCount);
                counts[evt[0]] = currentCount + evt[1];
            }
            int current = 0;
            foreach (var pair in counts)
            {
                if (pair.Value > 0) current = Math.Max(current, pair.Key);
            }
            if (result.Count == 0 || result[result.Count - 1][1] != current)
            {
                result.Add(new[] { entry.Key, current });
            }
        }
        return result.ToArray();
    }

    private static bool SameSkyline(int[][] a, int[][] b) =>
        a.Length == b.Length && a.Zip(b).All(pair =>
            pair.First[0] == pair.Second[0] && pair.First[1] == pair.Second[1]);

    private static string Show(int[][] skyline) =>
        string.Join("", skyline.Select(p => "[" + p[0] + "," + p[1] + "]"));

    public static void Main()
    {
        // Canonical example: at x=7 the 15-block ends (12 remains); at x=9 the 10-block
        // ends but 12 still stands, so no point is emitted there.
        var canonical = new[] { new[] { 2, 9, 10 }, new[] { 3, 7, 15 }, new[] { 5, 12, 12 },
            new[] { 15, 20, 10 }, new[] { 19, 24, 8 } };
        var canonicalSkyline = new[] { new[] { 2, 10 }, new[] { 3, 15 }, new[] { 7, 12 },
            new[] { 12, 0 }, new[] { 15, 10 }, new[] { 20, 8 }, new[] { 24, 0 } };
        Check(SameSkyline(SkylineDivideConquer(canonical), canonicalSkyline), "canonical D&C");
        Check(SameSkyline(SkylineSweep(canonical), canonicalSkyline), "canonical sweep");

        Check(SkylineDivideConquer(Array.Empty<int[]>()).Length == 0, "empty D&C");
        Check(SkylineSweep(Array.Empty<int[]>()).Length == 0, "empty sweep");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 1, 3, 4 } }),
            new[] { new[] { 1, 4 }, new[] { 3, 0 } }), "single building");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 2, 5 }, new[] { 2, 4, 7 } }),
            new[] { new[] { 0, 5 }, new[] { 2, 7 }, new[] { 4, 0 } }), "touching buildings");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 2, 5 }, new[] { 5, 8, 3 } }),
            new[] { new[] { 0, 5 }, new[] { 2, 0 }, new[] { 5, 3 }, new[] { 8, 0 } }),
            "gap between buildings");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 10, 5 }, new[] { 2, 4, 3 } }),
            new[] { new[] { 0, 5 }, new[] { 10, 0 } }), "nested shorter building");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 10, 5 }, new[] { 2, 4, 9 } }),
            new[] { new[] { 0, 5 }, new[] { 2, 9 }, new[] { 4, 5 }, new[] { 10, 0 } }),
            "nested taller building");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 4, 3 }, new[] { 0, 4, 5 } }),
            new[] { new[] { 0, 5 }, new[] { 4, 0 } }), "same span, taller wins");
        Check(SameSkyline(SkylineDivideConquer(new[]
            { new[] { 0, 4, 5 }, new[] { 0, 4, 5 }, new[] { 1, 2, 1 } }),
            new[] { new[] { 0, 5 }, new[] { 4, 0 } }), "duplicate buildings");

        // Degenerate buildings are ignored by both implementations.
        Check(SkylineDivideConquer(new[] { new[] { 3, 3, 9 } }).Length == 0, "zero width ignored");
        Check(SkylineDivideConquer(new[] { new[] { 0, 5, 0 } }).Length == 0, "zero height ignored");
        Check(SameSkyline(SkylineDivideConquer(new[] { new[] { 0, 5, 4 }, new[] { 2, 2, 9 } }),
            new[] { new[] { 0, 4 }, new[] { 5, 0 } }), "degenerate among valid");
        Check(SameSkyline(SkylineSweep(new[] { new[] { 0, 5, 4 }, new[] { 2, 2, 9 } }),
            new[] { new[] { 0, 4 }, new[] { 5, 0 } }), "degenerate among valid sweep");

        // LCG cross-check with many building sets; both algorithms must agree exactly.
        for (int trial = 0; trial < 400; trial++)
        {
            int count = 1 + Random.Next(10);
            var buildings = new int[count][];
            for (int i = 0; i < count; i++)
            {
                int left = Random.Next(50);
                int width = 1 + Random.Next(12);
                int height = 1 + Random.Next(15);
                buildings[i] = new[] { left, left + width, height };
            }
            int[][] expected = SkylineSweep(buildings);
            Check(SameSkyline(SkylineDivideConquer(buildings), expected),
                "random trial " + trial + " expected " + Show(expected));
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
