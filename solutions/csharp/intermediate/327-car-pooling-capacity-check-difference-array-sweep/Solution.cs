// Task 327 - Car pooling capacity check (difference array / sweep). (Tier: Intermediate, CTCI Ch 1)
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
    /// <summary>Difference-array sweep; true when the load never exceeds the capacity.</summary>
    public static bool CarPooling(int[][] trips, int capacity)
    {
        var delta = new SortedDictionary<int, int>();
        foreach (int[] trip in trips)
        {
            delta.TryGetValue(trip[1], out int pickUp);
            delta[trip[1]] = pickUp + trip[0];          // pick up at start
            delta.TryGetValue(trip[2], out int dropOff);
            delta[trip[2]] = dropOff - trip[0];         // drop off at end
        }
        int load = 0;
        foreach (int change in delta.Values)
        {
            load += change;
            if (load > capacity) return false;
        }
        return true;
    }

    /// <summary>Independent oracle for small coordinates: materialise the load at every stop.</summary>
    public static bool CarPoolingBruteforce(int[][] trips, int capacity)
    {
        if (trips.Length == 0) return capacity >= 0;
        int lo = trips.Min(trip => trip[1]);
        int hi = trips.Max(trip => trip[2]);
        var loads = new int[hi - lo];
        foreach (int[] trip in trips)
        {
            for (int stop = trip[1]; stop < trip[2]; stop++) loads[stop - lo] += trip[0];
        }
        return loads.DefaultIfEmpty(0).Max() <= capacity;
    }

    private static int[][] LcgTrips(int count, Lcg lcg)
    {
        var trips = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int passengers = 1 + lcg.Next(6);
            int start = lcg.Next(25);
            trips[i] = new[] { passengers, start, start + 1 + lcg.Next(8) };
        }
        return trips;
    }

    public static void Main()
    {
        Check(!CarPooling(new[] { new[] { 2, 1, 5 }, new[] { 3, 3, 7 } }, 4), "capacity 4 is too small");
        Check(CarPooling(new[] { new[] { 2, 1, 5 }, new[] { 3, 3, 7 } }, 5), "capacity 5 fits");
        Check(CarPooling(new[] { new[] { 2, 1, 5 }, new[] { 3, 5, 7 } }, 3), "drop before pick-up at stop 5");
        Check(!CarPooling(new[] { new[] { 2, 1, 5 }, new[] { 3, 5, 7 } }, 2), "capacity 2 is too small");
        Check(CarPooling(Array.Empty<int[]>(), 0), "no trips fit zero capacity");
        Check(!CarPooling(new[] { new[] { 1, 0, 1 } }, 0), "single passenger over capacity");

        Check(!CarPoolingBruteforce(new[] { new[] { 2, 1, 5 }, new[] { 3, 3, 7 } }, 4), "bruteforce false");
        Check(CarPoolingBruteforce(new[] { new[] { 2, 1, 5 }, new[] { 3, 5, 7 } }, 3), "bruteforce true");

        var lcg = new Lcg(1618);
        for (int trial = 0; trial < 300; trial++)
        {
            int count = 1 + lcg.Next(8);
            int[][] trips = LcgTrips(count, lcg);
            int capacity = lcg.Next(21);
            bool expected = CarPoolingBruteforce(trips, capacity);
            Check(CarPooling(trips, capacity) == expected, "random bruteforce cross-check " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
