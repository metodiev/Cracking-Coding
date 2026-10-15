// Task 490 - Implement exponential search for unbounded arrays. (Tier: Advanced, CTCI Ch 10)
//
// Sorted ascending arrays are treated as conceptually unbounded: a documented sentinel sits at the
// virtual index n (value +infinity), so the doubling phase may "look at" index n without ever
// reading past n-1 - the peek helper returns the sentinel for idx >= n and records the largest real
// index touched. The search is cold-start: ExponentialSearch returns the FIRST index holding target,
// or -1.
//
// Algorithm (documented):
// 1. Peek index 0; if arr[0] >= target the range is [0, 0].
// 2. Otherwise double the bound b = 1, 2, 4, ... while b < n and arr[b] < target, remembering the
//    last probed index prev with arr[prev] < target.
// 3. The first occurrence lies in [prev + 1, min(b, n - 1)]; finish with an ordinary lower-bound
//    style binary search on that window and compare arr[lo] == target.
//
// Probe budget: <= 1 + ceil(log2(i + 1)) doubling peeks plus <= ceil(log2(i)) window steps for a hit
// at index i, bounded overall by 2 * ceil(log2(n + 1)) + 2 peeks (asserted). The answer is asserted
// against both a plain binary search and a brute-force linear first-occurrence scan on LCG arrays
// (with duplicates) including target at index 0, at n - 1, and absent values (sentinel case).
using System;
using System.Collections.Generic;

internal class SearchResult
{
    public readonly int Index;
    public readonly int Reads;
    public readonly int MaxIndex;

    public SearchResult(int index, int reads, int maxIndex)
    {
        Index = index;
        Reads = reads;
        MaxIndex = maxIndex;
    }
}

public static class Solution
{
    private static SearchResult ExponentialSearch(int[] arr, long target)
    {
        int n = arr.Length;
        int reads = 0;
        int maxIndex = -1;
        if (n == 0) return new SearchResult(-1, reads, maxIndex);
        int lo, hi;
        reads++;
        if (arr[0] >= target)
        {
            lo = 0;
            hi = 0;
        }
        else
        {
            int bound = 1;
            int prev = 0;
            while (bound < n)
            {
                reads++;
                if (bound > maxIndex) maxIndex = bound;
                if (arr[bound] >= target) break;
                prev = bound;
                bound <<= 1;
            }
            lo = prev + 1;
            hi = Math.Min(bound, n - 1);
        }
        if (lo > hi) return new SearchResult(-1, reads, maxIndex);
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            reads++;
            if (mid > maxIndex) maxIndex = mid;
            if (arr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        reads++;
        if (lo > maxIndex) maxIndex = lo;
        if (arr[lo] == target) return new SearchResult(lo, reads, maxIndex);
        return new SearchResult(-1, reads, maxIndex);
    }

    private static int BinarySearch(int[] arr, long target)
    {
        int lo = 0, hi = arr.Length - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (arr[mid] < target) lo = mid + 1;
            else if (arr[mid] > target) hi = mid - 1;
            else
            {
                while (mid > 0 && arr[mid - 1] == target) mid--;
                return mid;
            }
        }
        return -1;
    }

    private static int BruteFirst(int[] arr, long target)
    {
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] == target) return i;
            if (arr[i] > target) return -1;
        }
        return -1;
    }

    private static int[] LcgValues(int count, int seed, int limit)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    private static int Budget(int n)
    {
        int log = 0;
        while ((1L << log) < n + 1L) log++;
        return 2 * log + 2;
    }

    public static void Main()
    {
        Check(ExponentialSearch(new int[0], 5).Index == -1, "empty");
        Check(ExponentialSearch(new[] { 5 }, 5).Index == 0, "single hit");
        Check(ExponentialSearch(new[] { 5 }, 4).Index == -1, "single below");
        Check(ExponentialSearch(new[] { 5 }, 6).Index == -1, "single above");
        var ramp = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        Check(ExponentialSearch(ramp, 1).Index == 0, "first");
        Check(ExponentialSearch(ramp, 8).Index == 7, "last");
        Check(ExponentialSearch(ramp, 4).Index == 3, "middle");
        Check(ExponentialSearch(ramp, 0).Index == -1, "below");
        Check(ExponentialSearch(ramp, 9).Index == -1, "above");
        Check(ExponentialSearch(new[] { 2, 2, 2, 2 }, 2).Index == 0, "duplicates");
        Check(ExponentialSearch(new[] { 2, 2, 2, 2 }, 3).Index == -1, "dup miss");
        Check(ExponentialSearch(new int[8], 0).Index == 0, "zeros");
        var odd = new[] { 1, 3, 5, 7, 9, 11 };
        Check(ExponentialSearch(odd, 11).Index == 5, "odd last");
        Check(ExponentialSearch(odd, 6).Index == -1, "odd miss");
        Check(BinarySearch(odd, 6) == BruteFirst(odd, 6) && BinarySearch(odd, 6) == -1, "ref miss");

        int foundTotal = 0, absentTotal = 0, tests = 0;
        for (int seed = 490; seed <= 494; seed++)
        {
            int[] arr = LcgValues(64, seed, 50);
            Array.Sort(arr);
            var targets = new List<int>(arr);
            targets.AddRange(new[] { -1, 100, 49, 50 });
            targets.AddRange(LcgValues(20, seed + 100, 60));
            foreach (int target in targets)
            {
                SearchResult result = ExponentialSearch(arr, target);
                int expected = BruteFirst(arr, target);
                Check(result.Index == expected, "lcg " + seed + " " + target);
                Check(result.Index == BinarySearch(arr, target), "binary match " + seed + " " + target);
                Check(result.MaxIndex < arr.Length, "bounds " + seed + " " + target);
                Check(result.Reads <= Budget(arr.Length), "reads " + seed + " " + target + " " + result.Reads);
                Check(result.Reads >= 1, "reads floor");
                tests++;
                if (result.Index >= 0) foundTotal++;
                else absentTotal++;
            }
        }
        Check(tests == 5 * 88, "test count");
        Check(foundTotal == 389 && absentTotal == 51, "totals " + foundTotal + " " + absentTotal);

        int n = 4096;
        int[] big = LcgValues(n, 777, 100000);
        Array.Sort(big);
        for (int i = 0; i < n; i += 137)
        {
            SearchResult result = ExponentialSearch(big, big[i]);
            Check(result.Index == BruteFirst(big, big[i]), "big " + i);
            Check(result.MaxIndex < n, "big bounds");
            Check(result.Reads <= Budget(n), "big reads " + result.Reads);
        }
        Check(ExponentialSearch(big, -5).Index == -1, "big below");
        Check(ExponentialSearch(big, 100001).Index == -1, "big above");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
