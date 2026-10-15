// Task 332 - Implement calendar booking (MyCalendar I/II/III style). (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class MyCalendar
{
    public readonly List<int[]> Accepted = new List<int[]>();

    /// <summary>Accepts a booking iff it overlaps no accepted booking.</summary>
    public bool Book(int start, int end)
    {
        foreach (int[] other in Accepted)
        {
            if (MyCalendar.Overlaps(start, end, other)) return false;
        }
        Accepted.Add(new[] { start, end });
        return true;
    }

    public static bool Overlaps(int start, int end, int[] other)
    {
        return start < other[1] && other[0] < end;
    }

    public static bool Overlaps(int[] a, int[] b)
    {
        return a[0] < b[1] && b[0] < a[1];
    }
}

public class MyCalendarTwo
{
    private readonly List<int[]> _singles = new List<int[]>();
    private readonly List<int[]> _pairs = new List<int[]>();

    /// <summary>Allows double booking: rejects a booking that would create a triple overlap.</summary>
    public bool Book(int start, int end)
    {
        foreach (int[] other in _pairs)
        {
            if (MyCalendar.Overlaps(start, end, other)) return false;
        }
        foreach (int[] other in _singles)
        {
            if (MyCalendar.Overlaps(start, end, other))
            {
                _pairs.Add(new[] { Math.Max(start, other[0]), Math.Min(end, other[1]) });
            }
        }
        _singles.Add(new[] { start, end });
        return true;
    }

    public int MaxOverlap()
    {
        return Solution.SweepPeak(_singles);
    }
}

public class MyCalendarK
{
    private readonly int _k;
    private readonly SortedDictionary<int, int> _delta = new SortedDictionary<int, int>();

    public MyCalendarK(int k)
    {
        if (k < 1) throw new Exception("k must be >= 1");
        _k = k;
    }

    /// <summary>Accepts a booking iff no instant ends up covered by more than k bookings.</summary>
    public bool Book(int start, int end)
    {
        Add(start, 1);
        Add(end, -1);
        if (Peak() > _k)
        {
            Add(start, -1);
            Add(end, 1);
            return false;
        }
        return true;
    }

    public int MaxOverlap()
    {
        return Peak();
    }

    private void Add(int time, int change)
    {
        if (_delta.TryGetValue(time, out int current)) _delta[time] = current + change;
        else _delta[time] = change;
    }

    private int Peak()
    {
        int current = 0, peak = 0;
        foreach (int change in _delta.Values)
        {
            current += change;
            peak = Math.Max(peak, current);
        }
        return peak;
    }
}

public static class Solution
{
    /// <summary>Maximum number of intervals covering any single instant (independent oracle).</summary>
    public static int SweepPeak(List<int[]> intervals)
    {
        var events = new SortedDictionary<int, int>();
        foreach (int[] interval in intervals)
        {
            if (events.TryGetValue(interval[0], out int atStart)) events[interval[0]] = atStart + 1;
            else events[interval[0]] = 1;
            if (events.TryGetValue(interval[1], out int atEnd)) events[interval[1]] = atEnd - 1;
            else events[interval[1]] = -1;
        }
        int current = 0, peak = 0;
        foreach (int change in events.Values)
        {
            current += change;
            peak = Math.Max(peak, current);
        }
        return peak;
    }

    private static int seed = 1717;

    private static List<int[]> LcgBookings(int count, int span)
    {
        var bookings = new List<int[]>();
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            int start = seed % span;
            seed = (seed * 75 + 74) % 65537;
            bookings.Add(new[] { start, start + 1 + seed % 9 });
        }
        return bookings;
    }

    private static string Join(List<int[]> intervals)
    {
        var parts = new List<string>();
        foreach (int[] interval in intervals) parts.Add("[" + interval[0] + "," + interval[1] + "]");
        return string.Join("", parts);
    }

    public static void Main()
    {
        var cal = new MyCalendar();
        Check(cal.Book(10, 20), "I accept (10,20)");
        Check(!cal.Book(15, 25), "I reject overlap");
        Check(cal.Book(20, 30), "I accept touching (20,30)");
        Check(cal.Book(5, 10), "I accept touching (5,10)");
        Check(!cal.Book(19, 20), "I reject (19,20)");
        Check(!cal.Book(10, 15), "I reject (10,15)");
        Check(Join(cal.Accepted) == "[10,20][20,30][5,10]", "I accepted list");

        var two = new MyCalendarTwo();
        var classic = new[]
        {
            new[] { 10, 20 }, new[] { 50, 60 }, new[] { 10, 40 },
            new[] { 5, 15 }, new[] { 5, 10 }, new[] { 25, 55 },
        };
        var expectedTwo = new[] { true, true, true, false, true, true };
        for (int i = 0; i < classic.Length; i++)
        {
            Check(two.Book(classic[i][0], classic[i][1]) == expectedTwo[i], "II step " + i);
        }
        Check(two.MaxOverlap() == 2, "II peak two");
        Check(!two.Book(30, 45), "II reject triple at (30,45)");
        Check(two.Book(60, 70), "II accept (60,70)");

        var three = new MyCalendarK(3);
        for (int i = 0; i < classic.Length; i++)
        {
            Check(three.Book(classic[i][0], classic[i][1]), "III k=3 step " + i);
        }
        Check(three.MaxOverlap() == 3, "III peak three");
        Check(!three.Book(12, 13), "III k=3 rejects fourth cover");

        var twoK = new MyCalendarK(2);
        for (int i = 0; i < classic.Length; i++)
        {
            Check(twoK.Book(classic[i][0], classic[i][1]) == expectedTwo[i], "III k=2 step " + i);
        }
        Check(twoK.MaxOverlap() == 2, "III k=2 peak");
        Check(!twoK.Book(12, 13), "III k=2 rejects (12,13)");

        Check(SweepPeak(new List<int[]>()) == 0, "oracle empty");
        Check(SweepPeak(new List<int[]> { new[] { 1, 2 } }) == 1, "oracle one");
        Check(SweepPeak(new List<int[]> { new[] { 1, 5 }, new[] { 2, 6 }, new[] { 3, 7 } }) == 3,
            "oracle triple");
        Check(SweepPeak(new List<int[]> { new[] { 1, 2 }, new[] { 2, 3 } }) == 1, "oracle touching");

        for (int k = 1; k <= 3; k++)
        {
            seed = 1717;
            var bookings = LcgBookings(240, 40);
            var calendarK = new MyCalendarK(k);
            var twoForComparison = k == 2 ? new MyCalendarTwo() : null;
            var oracle = new List<int[]>();
            foreach (int[] booking in bookings)
            {
                var candidate = new List<int[]>(oracle) { booking };
                bool expected = SweepPeak(candidate) <= k;
                bool actual = calendarK.Book(booking[0], booking[1]);
                Check(actual == expected, "random k=" + k + " [" + booking[0] + "," + booking[1] + "]");
                if (actual) oracle.Add(booking);
                if (twoForComparison != null)
                {
                    Check(twoForComparison.Book(booking[0], booking[1]) == actual,
                        "II vs k=2 [" + booking[0] + "," + booking[1] + "]");
                }
                Check(calendarK.MaxOverlap() == SweepPeak(oracle) && calendarK.MaxOverlap() <= k,
                    "peak invariant");
            }
        }

        seed = 2626;
        var lastBookings = LcgBookings(200, 40);
        var one = new MyCalendar();
        var oneK = new MyCalendarK(1);
        var lastOracle = new List<int[]>();
        foreach (int[] booking in lastBookings)
        {
            var candidate = new List<int[]>(lastOracle) { booking };
            bool expected = SweepPeak(candidate) <= 1;
            Check(one.Book(booking[0], booking[1]) == expected,
                "I random [" + booking[0] + "," + booking[1] + "]");
            Check(oneK.Book(booking[0], booking[1]) == expected,
                "I via k=1 [" + booking[0] + "," + booking[1] + "]");
            if (expected) lastOracle.Add(booking);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
