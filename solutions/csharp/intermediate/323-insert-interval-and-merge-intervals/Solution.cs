// Task 323 - Insert interval and merge intervals. (Tier: Intermediate, CTCI Ch 1)
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
    /// <summary>Sort by start and merge overlapping or touching intervals (start &lt;= current end).</summary>
    public static int[][] MergeIntervals(int[][] intervals)
    {
        if (intervals.Length == 0) return Array.Empty<int[]>();
        var ordered = intervals.Select(iv => (int[])iv.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        var merged = new List<int[]> { ordered[0] };
        for (int i = 1; i < ordered.Length; i++)
        {
            int[] last = merged[^1];
            if (ordered[i][0] <= last[1]) last[1] = Math.Max(last[1], ordered[i][1]); // touching counts
            else merged.Add(ordered[i]);
        }
        return merged.ToArray();
    }

    /// <summary>Insert into a sorted non-overlapping list, merging overlapping/touching parts.</summary>
    public static int[][] InsertInterval(int[][] intervals, int[] newInterval)
    {
        int start = newInterval[0], end = newInterval[1];
        var result = new List<int[]>();
        int i = 0;
        while (i < intervals.Length && intervals[i][1] < start)
        {
            result.Add((int[])intervals[i].Clone());
            i++;
        }
        while (i < intervals.Length && intervals[i][0] <= end)
        {
            start = Math.Min(start, intervals[i][0]);
            end = Math.Max(end, intervals[i][1]);
            i++;
        }
        result.Add(new[] { start, end });
        while (i < intervals.Length)
        {
            result.Add((int[])intervals[i].Clone());
            i++;
        }
        return result.ToArray();
    }

    /// <summary>Independent oracle: endpoint events; starts sort before ends at the same x.</summary>
    public static int[][] MergeIntervalsSweep(int[][] intervals)
    {
        var events = new List<int[]>();
        foreach (int[] interval in intervals)
        {
            events.Add(new[] { interval[0], 0 });
            events.Add(new[] { interval[1], 1 });
        }
        events.Sort((u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        var merged = new List<int[]>();
        int depth = 0;
        foreach (int[] e in events)
        {
            if (e[1] == 0)
            {
                if (depth == 0) merged.Add(new[] { e[0], 0 });
                depth++;
            }
            else
            {
                depth--;
                if (depth == 0) merged[^1][1] = e[0];
            }
        }
        return merged.ToArray();
    }

    private static int[][] LcgIntervals(int count, Lcg lcg)
    {
        var intervals = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int start = lcg.Next(25);
            intervals[i] = new[] { start, start + 1 + lcg.Next(8) };
        }
        return intervals;
    }

    private static int[][] LcgDisjointSorted(int count, Lcg lcg)
    {
        var intervals = new int[count][];
        int cursor = 0;
        for (int i = 0; i < count; i++)
        {
            cursor += 1 + lcg.Next(4);
            int end = cursor + 1 + lcg.Next(5);
            intervals[i] = new[] { cursor, end };
            cursor = end + 1;
        }
        return intervals;
    }

    private static bool SameIntervals(int[][] a, int[][] b) =>
        a.Length == b.Length && a.Zip(b).All(pair => pair.First.SequenceEqual(pair.Second));

    public static void Main()
    {
        Check(SameIntervals(MergeIntervals(new[] { new[] { 1, 3 }, new[] { 2, 6 }, new[] { 8, 10 }, new[] { 15, 18 } }),
            new[] { new[] { 1, 6 }, new[] { 8, 10 }, new[] { 15, 18 } }), "classic merge");
        Check(SameIntervals(MergeIntervals(new[] { new[] { 1, 2 }, new[] { 2, 3 } }),
            new[] { new[] { 1, 3 } }), "touching merge policy");
        Check(SameIntervals(MergeIntervals(new[] { new[] { 1, 4 }, new[] { 4, 5 }, new[] { 7, 9 } }),
            new[] { new[] { 1, 5 }, new[] { 7, 9 } }), "chain merge");
        Check(SameIntervals(MergeIntervals(new[] { new[] { 1, 10 }, new[] { 2, 3 }, new[] { 4, 5 } }),
            new[] { new[] { 1, 10 } }), "swallowed intervals");
        Check(SameIntervals(MergeIntervals(new[] { new[] { 5, 5 } }), new[] { new[] { 5, 5 } }), "point interval");
        Check(MergeIntervals(Array.Empty<int[]>()).Length == 0, "empty merge");
        Check(SameIntervals(MergeIntervalsSweep(new[] { new[] { 1, 2 }, new[] { 2, 3 } }),
            new[] { new[] { 1, 3 } }), "sweep touching merge");

        var baseIntervals = new[] { new[] { 1, 2 }, new[] { 3, 5 }, new[] { 6, 7 }, new[] { 8, 10 }, new[] { 12, 16 } };
        Check(SameIntervals(InsertInterval(baseIntervals, new[] { 9, 10 }),
            new[] { new[] { 1, 2 }, new[] { 3, 5 }, new[] { 6, 7 }, new[] { 8, 10 }, new[] { 12, 16 } }), "insert inside");
        Check(SameIntervals(InsertInterval(baseIntervals, new[] { 0, 0 }),
            new[] { new[] { 0, 0 }, new[] { 1, 2 }, new[] { 3, 5 }, new[] { 6, 7 }, new[] { 8, 10 }, new[] { 12, 16 } }), "insert before");
        Check(SameIntervals(InsertInterval(baseIntervals, new[] { 4, 4 }),
            new[] { new[] { 1, 2 }, new[] { 3, 5 }, new[] { 6, 7 }, new[] { 8, 10 }, new[] { 12, 16 } }), "insert swallowed");
        Check(SameIntervals(InsertInterval(new[] { new[] { 1, 2 }, new[] { 5, 7 } }, new[] { 2, 5 }),
            new[] { new[] { 1, 7 } }), "insert touches both sides");
        Check(SameIntervals(InsertInterval(new[] { new[] { 3, 4 } }, new[] { 1, 2 }),
            new[] { new[] { 1, 2 }, new[] { 3, 4 } }), "insert with gap");
        Check(SameIntervals(InsertInterval(Array.Empty<int[]>(), new[] { 4, 6 }),
            new[] { new[] { 4, 6 } }), "insert into empty");

        var lcg = new Lcg(20077);
        for (int trial = 0; trial < 200; trial++)
        {
            int[][] intervals = LcgIntervals(6, lcg);
            Check(SameIntervals(MergeIntervals(intervals), MergeIntervalsSweep(intervals)), "random sweep cross-check " + trial);
        }

        var insertLcg = new Lcg(4242);
        for (int trial = 0; trial < 200; trial++)
        {
            int[][] baseSorted = LcgDisjointSorted(5, insertLcg);
            int newStart = insertLcg.Next(30);
            var newInterval = new[] { newStart, newStart + 1 + insertLcg.Next(8) };
            int[][] withNew = baseSorted.Append(newInterval).ToArray();
            Check(SameIntervals(InsertInterval(baseSorted, newInterval), MergeIntervals(withNew)),
                "insert equals merge of the union " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
