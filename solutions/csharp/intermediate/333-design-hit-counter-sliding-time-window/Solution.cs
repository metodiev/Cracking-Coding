// Task 333 - Design hit counter (sliding time window). (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class HitCounter
{
    public const int Window = 300;
    private readonly List<int> _times = new List<int>();

    /// <summary>Records a hit; timestamps must be non-decreasing.</summary>
    public void Hit(int timestamp)
    {
        if (_times.Count > 0 && timestamp < _times[_times.Count - 1])
        {
            throw new Exception("timestamps must not go backwards");
        }
        _times.Add(timestamp);
    }

    /// <summary>Number of hits h with timestamp - 299 &lt;= h &lt;= timestamp.</summary>
    public int GetHits(int timestamp)
    {
        return FirstGreater(timestamp) - FirstAtLeast(timestamp - (Window - 1));
    }

    private int FirstAtLeast(int timestamp)
    {
        int lo = 0, hi = _times.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (_times[mid] < timestamp) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private int FirstGreater(int timestamp)
    {
        int lo = 0, hi = _times.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (_times[mid] <= timestamp) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public static class Solution
{
    /// <summary>Oracle: linear scan of every recorded timestamp.</summary>
    public static int NaiveHits(List<int> times, int timestamp)
    {
        int count = 0;
        foreach (int time in times)
        {
            if (timestamp - (HitCounter.Window - 1) <= time && time <= timestamp) count++;
        }
        return count;
    }

    private static int seed = 2027;

    public static void Main()
    {
        // Hand-checked: hits at 1, 2, 3 -> get(4) = 3; windows slide away one at a time.
        var counter = new HitCounter();
        foreach (int t in new[] { 1, 2, 3 }) counter.Hit(t);
        Check(counter.GetHits(1) == 1, "get(1)");
        Check(counter.GetHits(2) == 2, "get(2)");
        Check(counter.GetHits(3) == 3, "get(3)");
        Check(counter.GetHits(4) == 3, "get(4) sees all three");
        Check(counter.GetHits(299) == 3, "get(299)");
        Check(counter.GetHits(300) == 3, "get(300): 300-299 = 1 still counts hit 1");
        Check(counter.GetHits(301) == 2, "get(301): hit 1 left the window");
        Check(counter.GetHits(302) == 1, "get(302)");
        Check(counter.GetHits(303) == 0, "get(303)");

        // A later hit at 300 joins the window: get(300) = 4, get(301) = 3, get(302) = 2.
        counter.Hit(300);
        Check(counter.GetHits(300) == 4, "get(300) after hit(300)");
        Check(counter.GetHits(301) == 3, "get(301) after hit(300)");
        Check(counter.GetHits(302) == 2, "get(302) after hit(300)");
        Check(counter.GetHits(599) == 1, "get(599)");
        Check(counter.GetHits(600) == 0, "get(600)");

        // LeetCode-flavoured sequence: hits at 1, 2, 3 and (again) at 300.
        var leet = new HitCounter();
        foreach (int t in new[] { 1, 2, 3, 300 }) leet.Hit(t);
        Check(leet.GetHits(3) == 3, "leet get(3)");
        Check(leet.GetHits(300) == 4, "leet get(300)");
        Check(leet.GetHits(301) == 3, "leet get(301)");
        Check(leet.GetHits(2) == 2, "query before newest hit");
        Check(leet.GetHits(0) == 0, "query before everything");

        // Same-timestamp bursts.
        var burst = new HitCounter();
        for (int i = 0; i < 5; i++) burst.Hit(7);
        Check(burst.GetHits(7) == 5, "burst get(7)");
        Check(burst.GetHits(306) == 5, "burst get(306)");
        Check(burst.GetHits(307) == 0, "burst get(307)");

        // LCG event stream against the naive scan.
        var stream = new HitCounter();
        var history = new List<int>();
        int now = 0;
        for (int step = 0; step < 4000; step++)
        {
            seed = (seed * 75 + 74) % 65537;
            now += seed % 5;
            seed = (seed * 75 + 74) % 65537;
            if (seed % 4 == 0)
            {
                seed = (seed * 75 + 74) % 65537;
                int timestamp = now + seed % 3;
                stream.Hit(timestamp);
                history.Add(timestamp);
                now = timestamp;
            }
            else
            {
                Check(stream.GetHits(now) == NaiveHits(history, now), "random get " + step);
            }
            if (step % 97 == 0)
            {
                Check(stream.GetHits(now) == NaiveHits(history, now), "periodic get " + step);
            }
        }
        Check(stream.GetHits(now) == NaiveHits(history, now), "final get");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
