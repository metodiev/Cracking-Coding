// Task 325 - Minimum number of platforms for trains (sweep-line). (Tier: Intermediate, CTCI Beyond — Software Engineering & Tooling)
using System;
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
    /// <summary>Two-pointer sweep; a train leaving at t frees its platform before any arrival at t.</summary>
    public static int MinPlatforms(int[] arrivals, int[] departures)
    {
        int[] arr = arrivals.OrderBy(v => v).ToArray();
        int[] dep = departures.OrderBy(v => v).ToArray();
        int i = 0, j = 0, needed = 0, best = 0;
        while (i < arr.Length)
        {
            if (arr[i] < dep[j])
            {
                needed++;
                best = Math.Max(best, needed);
                i++;
            }
            else
            {
                needed--;
                j++;
            }
        }
        return best;
    }

    /// <summary>Independent oracle: count active trains at every minute of the schedule.</summary>
    public static int MinPlatformsPerMinute(int[] arrivals, int[] departures)
    {
        if (arrivals.Length == 0) return 0;
        int lo = arrivals.Min();
        int hi = departures.Max();
        int best = 0;
        for (int minute = lo; minute <= hi; minute++)
        {
            int active = 0;
            for (int k = 0; k < arrivals.Length; k++)
            {
                if (arrivals[k] <= minute && minute < departures[k]) active++;
            }
            best = Math.Max(best, active);
        }
        return best;
    }

    private static (int[] Arrivals, int[] Departures) LcgSchedules(int count, Lcg lcg)
    {
        var arrivals = new int[count];
        var departures = new int[count];
        for (int i = 0; i < count; i++)
        {
            int arrival = lcg.Next(60);
            arrivals[i] = arrival;
            departures[i] = arrival + 1 + lcg.Next(20);
        }
        return (arrivals, departures);
    }

    public static void Main()
    {
        int[] arrivals = { 900, 940, 950, 1100, 1500, 1800 };
        int[] departures = { 910, 1200, 1120, 1130, 1900, 2000 };
        Check(MinPlatforms(arrivals, departures) == 3, "classic trains need 3 platforms");
        Check(MinPlatformsPerMinute(arrivals, departures) == 3, "classic per-minute oracle");

        Check(MinPlatforms(new[] { 1, 2 }, new[] { 2, 3 }) == 1, "release before acquire");
        Check(MinPlatformsPerMinute(new[] { 1, 2 }, new[] { 2, 3 }) == 1, "release per-minute");
        Check(MinPlatforms(new[] { 1, 1, 1 }, new[] { 2, 2, 2 }) == 3, "three simultaneous trains");
        Check(MinPlatforms(new[] { 5 }, new[] { 9 }) == 1, "single train");
        Check(MinPlatforms(Array.Empty<int>(), Array.Empty<int>()) == 0, "no trains");

        var lcg = new Lcg(31415);
        for (int trial = 0; trial < 300; trial++)
        {
            int count = 1 + lcg.Next(12);
            (int[] arr, int[] dep) = LcgSchedules(count, lcg);
            int expected = MinPlatformsPerMinute(arr, dep);
            Check(MinPlatforms(arr, dep) == expected, "random per-minute cross-check " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
