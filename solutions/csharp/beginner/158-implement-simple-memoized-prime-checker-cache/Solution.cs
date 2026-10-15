// Task 158 - Implement simple memoized prime checker cache. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Collections.Generic;

/// <summary>
/// Trial division up to sqrt(n) with a verdict cache; hits/misses counters prove
/// that repeated queries for the same number are served from the cache.
/// </summary>
public class PrimeChecker
{
    private readonly Dictionary<int, bool> cache = new Dictionary<int, bool>();

    public int Hits;
    public int Misses;

    /// <summary>Memoized primality test: n &lt; 2 is never prime.</summary>
    public bool IsPrime(int n)
    {
        if (cache.TryGetValue(n, out bool cached))
        {
            Hits++;
            return cached;
        }
        Misses++;
        bool verdict = Check(n);
        cache[n] = verdict;
        return verdict;
    }

    public int Size => cache.Count;

    /// <summary>Trial division up to sqrt(n): O(sqrt(n)) time, O(1) space.</summary>
    private static bool Check(int n)
    {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int d = 3; (long)d * d <= n; d += 2)
        {
            if (n % d == 0) return false;
        }
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        var checker = new PrimeChecker();
        Check(checker.IsPrime(97), "97 is prime");
        Check(checker.Hits == 0 && checker.Misses == 1 && checker.Size == 1, "one miss first");

        // repeated queries for the same number hit the cache
        Check(checker.IsPrime(97) && checker.IsPrime(97), "repeat hits");
        Check(checker.Hits == 2 && checker.Misses == 1 && checker.Size == 1, "two hits, one miss");

        var cases = new (int n, bool expected)[]
        {
            (0, false), (1, false), (2, true), (100, false), (7919, true), (7917, false),
        };
        foreach (var (n, expected) in cases) Check(checker.IsPrime(n) == expected, "case " + n);
        Check(checker.Hits == 2 && checker.Misses == 7 && checker.Size == 7, "counters after first sweep");

        // repeat the whole set: every answer now comes from the cache
        var again = new (int n, bool expected)[]
        {
            (0, false), (1, false), (2, true), (97, true), (100, false), (7919, true), (7917, false),
        };
        foreach (var (n, expected) in again) Check(checker.IsPrime(n) == expected, "again " + n);
        Check(checker.Hits == 9 && checker.Misses == 7 && checker.Size == 7, "counters after cached sweep");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
