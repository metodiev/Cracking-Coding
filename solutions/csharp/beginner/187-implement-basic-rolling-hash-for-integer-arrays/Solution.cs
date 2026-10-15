// Task 187 - Implement basic rolling hash for integer arrays. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    private const long DefaultBase = 257;
    private const long DefaultMod = 1000000007L;

    private static long PowMod(long baseValue, long exponent, long mod)
    {
        long result = 1;
        for (long i = 0; i < exponent; i++) result = result * baseValue % mod;
        return result;
    }

    private static long Normalize(long value, long mod)
    {
        return ((value % mod) + mod) % mod;
    }

    /// <summary>Comma-joined window contents; used as an exact key when verifying hash matches.</summary>
    private static string WindowKey(int[] values, int start, int k)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < k; i++) sb.Append(values[start + i]).Append(',');
        return sb.ToString();
    }

    /// <summary>
    /// Map each repeated window's first start index to its later occurrences. Each length-k
    /// window hash slides in O(1); equal hashes are only candidates, so every bucket is
    /// confirmed by comparing the windows themselves (a collision cannot fake a duplicate).
    /// </summary>
    private static SortedDictionary<int, List<int>> FindDuplicateWindows(int[] values, int k)
    {
        return FindDuplicateWindows(values, k, DefaultBase, DefaultMod);
    }

    private static SortedDictionary<int, List<int>> FindDuplicateWindows(int[] values, int k, long baseValue, long mod)
    {
        if (k <= 0) throw new ArgumentException("window length must be positive");
        var duplicates = new SortedDictionary<int, List<int>>();
        int n = values.Length;
        if (k > n) return duplicates;

        long highest = PowMod(baseValue, k - 1, mod);
        long windowHash = 0;
        for (int i = 0; i < k; i++) windowHash = (windowHash * baseValue + Normalize(values[i], mod)) % mod;

        var buckets = new Dictionary<long, List<int>>();
        AddToBucket(buckets, windowHash, 0);
        for (int start = 1; start + k <= n; start++)
        {
            windowHash = (windowHash - Normalize(values[start - 1], mod) * highest) % mod;
            if (windowHash < 0) windowHash += mod;
            windowHash = (windowHash * baseValue + Normalize(values[start + k - 1], mod)) % mod;
            AddToBucket(buckets, windowHash, start);
        }

        foreach (var indices in buckets.Values)
        {
            if (indices.Count < 2) continue;
            var groups = new Dictionary<string, List<int>>();
            foreach (int index in indices)
            {
                string key = WindowKey(values, index, k);
                if (!groups.ContainsKey(key)) groups[key] = new List<int>();
                groups[key].Add(index);
            }
            foreach (var positions in groups.Values)
            {
                if (positions.Count > 1)
                {
                    duplicates[positions[0]] = positions.GetRange(1, positions.Count - 1);
                }
            }
        }
        return duplicates;
    }

    /// <summary>Reference implementation: group windows directly, no hashing.</summary>
    private static SortedDictionary<int, List<int>> FindDuplicateWindowsNaive(int[] values, int k)
    {
        var duplicates = new SortedDictionary<int, List<int>>();
        if (k <= 0 || k > values.Length) return duplicates;
        var groups = new Dictionary<string, List<int>>();
        for (int start = 0; start + k <= values.Length; start++)
        {
            string key = WindowKey(values, start, k);
            if (!groups.ContainsKey(key)) groups[key] = new List<int>();
            groups[key].Add(start);
        }
        foreach (var positions in groups.Values)
        {
            if (positions.Count > 1)
            {
                duplicates[positions[0]] = positions.GetRange(1, positions.Count - 1);
            }
        }
        return duplicates;
    }

    private static void AddToBucket(Dictionary<long, List<int>> buckets, long hash, int start)
    {
        if (!buckets.ContainsKey(hash)) buckets[hash] = new List<int>();
        buckets[hash].Add(start);
    }

    /// <summary>Rendering such as "0:[3, 6];1:[4];2:[5]" so tests read the whole answer.</summary>
    private static string Describe(SortedDictionary<int, List<int>> duplicates)
    {
        var parts = new List<string>();
        foreach (var pair in duplicates)
        {
            parts.Add(pair.Key + ":[" + string.Join(", ", pair.Value) + "]");
        }
        return string.Join(";", parts);
    }

    public static void Main()
    {
        Check(Describe(FindDuplicateWindows(new[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 }, 3)) == "0:[3, 6];1:[4];2:[5]",
                "triple repeat");
        Check(Describe(FindDuplicateWindows(new[] { 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 }, 3)) == "0:[3, 6];1:[4, 7];2:[5, 8]",
                "three interleaved windows");
        Check(Describe(FindDuplicateWindows(new[] { 5, 5, 5, 5 }, 2)) == "0:[1, 2]", "all equal");
        Check(Describe(FindDuplicateWindows(new[] { 4, 7, 4, 7, 4 }, 2)) == "0:[2];1:[3]", "overlaps count");
        Check(Describe(FindDuplicateWindows(new[] { 7, 7, 8, 7, 7 }, 1)) == "0:[1, 3, 4]", "k = 1");
        Check(Describe(FindDuplicateWindows(new[] { 1, 2, 3, 1, 2, 4 }, 3)) == "", "near-duplicate only");
        Check(Describe(FindDuplicateWindows(new[] { 1, 2, 3, 4 }, 4)) == "", "single window");
        Check(Describe(FindDuplicateWindows(new[] { 1, 2, 3, 4 }, 5)) == "", "k larger than array");
        Check(Describe(FindDuplicateWindows(new int[0], 2)) == "", "empty array");
        Check(Describe(FindDuplicateWindows(new[] { -5, -5, 3, -5, -5 }, 2)) == "0:[3]", "negative values");

        bool threw = false;
        try
        {
            FindDuplicateWindows(new[] { 1, 2 }, 0);
        }
        catch (ArgumentException)
        {
            threw = true;
        }
        Check(threw, "k = 0 rejected");

        // agreement with the direct grouping on pseudo-random arrays
        var rng = new Random(23);
        for (int t = 0; t < 200; t++)
        {
            var values = new int[rng.Next(31)];
            for (int i = 0; i < values.Length; i++) values[i] = rng.Next(4);
            int k = 1 + rng.Next(4);
            Check(Describe(FindDuplicateWindows(values, k)) == Describe(FindDuplicateWindowsNaive(values, k)),
                    "cross-check k = " + k);
        }

        // a tiny modulus makes collisions common; results must still be exact
        for (int t = 0; t < 200; t++)
        {
            var values = new int[rng.Next(31)];
            for (int i = 0; i < values.Length; i++) values[i] = rng.Next(200);
            int k = 1 + rng.Next(3);
            Check(Describe(FindDuplicateWindows(values, k, 257, 97)) == Describe(FindDuplicateWindowsNaive(values, k)),
                    "tiny modulus cross-check k = " + k);
        }

        // concrete collision: the windows (5, 63) and (6, 0) share one hash under mod 97
        Check((5L * 257 + 63) % 97 == (6L * 257 + 0) % 97, "collision premise");
        Check(Describe(FindDuplicateWindows(new[] { 5, 63, 4, 6, 0 }, 2, 257, 97)) == "", "collision rejected");
        Check(Describe(FindDuplicateWindows(new[] { 5, 63, 4, 6, 0, 5, 63 }, 2, 257, 97)) == "0:[5]",
                "collision plus real duplicate");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
