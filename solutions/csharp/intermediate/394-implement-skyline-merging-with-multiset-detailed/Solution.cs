// Task 394 - Implement skyline merging with a multiset (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>events: {x, delta}; returns skyline points {x, h}.</summary>
    private static List<int[]> Sweep(List<int[]> events)
    {
        var deltas = new SortedDictionary<int, List<int>>();
        foreach (var e in events)
        {
            if (!deltas.TryGetValue(e[0], out var list)) { list = new List<int>(); deltas[e[0]] = list; }
            list.Add(e[1]);
        }
        var counts = new SortedDictionary<int, int>();
        var points = new List<int[]>();
        int current = 0;
        foreach (var entry in deltas)
        {
            foreach (int delta in entry.Value)
            {
                if (delta >= 0)
                {
                    counts[delta] = counts.TryGetValue(delta, out int c) ? c + 1 : 1;
                }
                else
                {
                    int height = -delta;
                    if (counts[height] == 1) counts.Remove(height);
                    else counts[height]--;
                }
            }
            int top = counts.Count > 0 ? counts.Keys.Max() : 0;
            if (top != current)
            {
                points.Add(new[] { entry.Key, top });
                current = top;
            }
        }
        return points;
    }

    public static List<int[]> SkylineOfRectangles(int[][] rectangles)
    {
        var events = new List<int[]>();
        foreach (var rectangle in rectangles)
        {
            int x1 = rectangle[0], x2 = rectangle[1], height = rectangle[2];
            if (x1 < x2 && height > 0)
            {
                events.Add(new[] { x1, height });
                events.Add(new[] { x2, -height });
            }
        }
        return Sweep(events);
    }

    public static List<int[]> MergeSkylines(List<List<int[]>> skylines)
    {
        var events = new List<int[]>();
        foreach (var skyline in skylines)
        {
            for (int i = 0; i + 1 < skyline.Count; i++)
            {
                int x1 = skyline[i][0], height = skyline[i][1];
                int x2 = skyline[i + 1][0];
                if (height > 0 && x1 < x2)
                {
                    events.Add(new[] { x1, height });
                    events.Add(new[] { x2, -height });
                }
            }
        }
        return Sweep(events);
    }

    private static int seed = 777;

    private static int[][] LcgRectangles(int count, int span)
    {
        var rectangles = new int[count][];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            int x1 = seed % span;
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % 6;
            seed = (seed * 75 + 74) % 65537;
            int height = 1 + seed % 10;
            rectangles[k] = new[] { x1, x1 + length, height };
        }
        return rectangles;
    }

    private static string Join(List<int[]> points) =>
        string.Join(";", points.Select(p => p[0] + "," + p[1]));

    public static void Main()
    {
        var canonical = new[] { new[] { 2, 9, 10 }, new[] { 3, 7, 15 }, new[] { 5, 12, 12 }, new[] { 15, 20, 10 }, new[] { 19, 24, 8 } };
        const string expected = "2,10;3,15;7,12;12,0;15,10;20,8;24,0";
        Check(Join(SkylineOfRectangles(canonical)) == expected, "canonical skyline");

        var left = new List<int[]> { new[] { 0, 3 }, new[] { 4, 0 } };
        var right = new List<int[]> { new[] { 1, 1 }, new[] { 2, 4 }, new[] { 5, 0 } };
        Check(Join(MergeSkylines(new List<List<int[]>> { left, right })) == "0,3;2,4;5,0", "merge two skylines");

        var expectedPoints = SkylineOfRectangles(canonical);
        Check(Join(MergeSkylines(new List<List<int[]>> { expectedPoints, expectedPoints })) == expected, "idempotent merge");

        Check(Join(SkylineOfRectangles(new[] { new[] { 0, 2, 3 }, new[] { 2, 4, 3 } })) == "0,3;4,0", "touching rectangles");
        Check(Join(SkylineOfRectangles(new[] { new[] { 0, 10, 5 }, new[] { 2, 5, 3 } })) == "0,5;10,0", "nested rectangles");
        Check(SkylineOfRectangles(Array.Empty<int[]>()).Count == 0, "no rectangles");

        for (int trial = 0; trial < 40; trial++)
        {
            var rectangles = LcgRectangles(9, 20);
            seed = (seed * 75 + 74) % 65537;
            int cut = seed % (rectangles.Length + 1);
            var first = rectangles.Take(cut).ToArray();
            var second = rectangles.Skip(cut).ToArray();
            string combined = Join(SkylineOfRectangles(rectangles));
            string merged = Join(MergeSkylines(new List<List<int[]>> { SkylineOfRectangles(first), SkylineOfRectangles(second) }));
            Check(merged == combined, "split merge " + trial);
        }

        Check(SkylineOfRectangles(new[] { new[] { 3, 1, 5 } }).Count == 0, "ignored backwards rectangle");
        Check(SkylineOfRectangles(new[] { new[] { 0, 5, 0 } }).Count == 0, "ignored zero height");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
