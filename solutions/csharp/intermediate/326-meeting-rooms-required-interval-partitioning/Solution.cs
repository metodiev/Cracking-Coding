// Task 326 - Meeting rooms required (interval partitioning). (Tier: Intermediate, CTCI Ch 1)
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
    /// <summary>Maximum overlap via endpoint events; an end (-1) sorts before a start (+1).</summary>
    public static int MinMeetingRooms(int[][] intervals)
    {
        var events = new List<int[]>();
        foreach (int[] interval in intervals)
        {
            events.Add(new[] { interval[0], 1 });
            events.Add(new[] { interval[1], -1 });
        }
        events.Sort((u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        int rooms = 0, best = 0;
        foreach (int[] e in events)
        {
            rooms += e[1];
            best = Math.Max(best, rooms);
        }
        return best;
    }

    /// <summary>Independent sweep with a min-heap of finish times.</summary>
    public static int MinRoomsHeap(int[][] intervals)
    {
        var ordered = intervals.Select(iv => (int[])iv.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
        var heap = new PriorityQueue<int, int>();
        foreach (int[] interval in ordered)
        {
            if (heap.Count > 0 && heap.Peek() <= interval[0]) heap.Dequeue();
            heap.Enqueue(interval[1], interval[1]);
        }
        return heap.Count;
    }

    /// <summary>Independent oracle for small coordinates: count active meetings at each minute.</summary>
    public static int MinRoomsPerMinute(int[][] intervals)
    {
        if (intervals.Length == 0) return 0;
        int lo = intervals.Min(iv => iv[0]);
        int hi = intervals.Max(iv => iv[1]);
        int best = 0;
        for (int minute = lo; minute <= hi; minute++)
        {
            int active = 0;
            foreach (int[] interval in intervals)
            {
                if (interval[0] <= minute && minute < interval[1]) active++;
            }
            best = Math.Max(best, active);
        }
        return best;
    }

    private static int[][] LcgIntervals(int count, Lcg lcg)
    {
        var intervals = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int start = lcg.Next(30);
            intervals[i] = new[] { start, start + 1 + lcg.Next(10) };
        }
        return intervals;
    }

    public static void Main()
    {
        Check(MinMeetingRooms(new[] { new[] { 0, 30 }, new[] { 5, 10 }, new[] { 15, 20 } }) == 2, "classic rooms 2");
        Check(MinMeetingRooms(new[] { new[] { 7, 10 }, new[] { 2, 4 } }) == 1, "classic rooms 1");
        Check(MinRoomsHeap(new[] { new[] { 0, 30 }, new[] { 5, 10 }, new[] { 15, 20 } }) == 2, "heap rooms 2");
        Check(MinRoomsHeap(new[] { new[] { 7, 10 }, new[] { 2, 4 } }) == 1, "heap rooms 1");

        Check(MinMeetingRooms(new[] { new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 } }) == 1, "back-to-back shares a room");
        Check(MinMeetingRooms(new[] { new[] { 1, 5 }, new[] { 2, 6 }, new[] { 3, 7 } }) == 3, "triple overlap");
        Check(MinMeetingRooms(Array.Empty<int[]>()) == 0, "no meetings");
        Check(MinRoomsHeap(Array.Empty<int[]>()) == 0, "no meetings heap");

        var lcg = new Lcg(2718);
        for (int trial = 0; trial < 300; trial++)
        {
            int count = 1 + lcg.Next(10);
            int[][] intervals = LcgIntervals(count, lcg);
            int expected = MinRoomsPerMinute(intervals);
            Check(MinMeetingRooms(intervals) == expected, "random event cross-check " + trial);
            Check(MinRoomsHeap(intervals) == expected, "random heap cross-check " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
