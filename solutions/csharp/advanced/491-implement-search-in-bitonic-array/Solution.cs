// Task 491 - Implement search in bitonic array. (Tier: Advanced, CTCI Ch 10)
//
// A bitonic (mountain) array strictly rises to a single peak and then strictly falls; the peak
// may sit at either end, so purely increasing and purely decreasing arrays are the degenerate
// cases. FindPeak is the standard O(log n) binary search that compares values[mid] with its
// neighbour; Search then binary searches the ascending side [0, peak] and the descending side
// (peak, n-1] separately, for O(log n) total. An absent target returns the documented sentinel
// -1 (indices are non-negative, so -1 is unambiguous). Verified against a linear-scan oracle
// on every element and on absent probes over LCG-built bitonic arrays of many sizes, including
// single-element arrays and arrays whose peak is at a boundary.
using System;

public static class Solution
{
    private const int Sentinel = -1;

    /** Index of the unique maximum (the peak) of a strictly bitonic array. */
    private static int FindPeak(int[] values)
    {
        int lo = 0;
        int hi = values.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (values[mid] < values[mid + 1]) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /** Index of target in a strictly bitonic array, or Sentinel (-1) when absent. */
    private static int Search(int[] values, int target)
    {
        int n = values.Length;
        if (n == 0) return Sentinel;
        int peak = FindPeak(values);

        int lo = 0;
        int hi = peak;                              // ascending side
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (values[mid] == target) return mid;
            if (values[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }

        lo = peak + 1;
        hi = n - 1;                                 // descending side
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (values[mid] == target) return mid;
            if (values[mid] > target) lo = mid + 1;
            else hi = mid - 1;
        }

        return Sentinel;
    }

    /** Simple oracle: first index holding target, or Sentinel. */
    private static int LinearSearch(int[] values, int target)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] == target) return i;
        }
        return Sentinel;
    }

    /** Strictly bitonic array of n distinct LCG values with the peak at `peak`. */
    private static int[] LcgBitonic(int n, int seed, int peak)
    {
        var distinct = new int[n];
        var seen = new bool[1000];
        int count = 0;
        while (count < n)
        {
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 1000;
            if (!seen[value])
            {
                seen[value] = true;
                distinct[count++] = value;
            }
        }
        Array.Sort(distinct);
        int tailLen = n - 1 - peak;                 // values below the peak, descending
        var result = new int[n];
        for (int i = 0; i < n - tailLen; i++) result[i] = distinct[tailLen + i];
        for (int i = 0; i < tailLen; i++) result[n - tailLen + i] = distinct[tailLen - 1 - i];
        return result;
    }

    public static void Main()
    {
        int[] sample = { 1, 3, 5, 9, 7, 4, 2 };
        Check(FindPeak(sample) == 3, "hand peak");
        for (int i = 0; i < sample.Length; i++) Check(Search(sample, sample[i]) == i, "hand " + i);
        foreach (int absent in new[] { 0, 6, 8, 10, -5 })
        {
            Check(Search(sample, absent) == Sentinel, "hand absent " + absent);
        }

        Check(FindPeak(new[] { 5 }) == 0, "single peak");
        Check(Search(new[] { 5 }, 5) == 0, "single hit");
        Check(Search(new[] { 5 }, 4) == Sentinel, "single miss");
        Check(FindPeak(new[] { 1, 2, 3, 4 }) == 3, "increasing peak");
        Check(Search(new[] { 1, 2, 3, 4 }, 4) == 3, "increasing hit");
        Check(Search(new[] { 1, 2, 3, 4 }, 0) == Sentinel, "increasing miss");
        Check(FindPeak(new[] { 4, 3, 2, 1 }) == 0, "decreasing peak");
        Check(Search(new[] { 4, 3, 2, 1 }, 1) == 3, "decreasing hit");
        Check(Search(new[] { 4, 3, 2, 1 }, 5) == Sentinel, "decreasing miss");

        int checkedArrays = 0;
        for (int n = 1; n <= 17; n++)
        {
            for (int peak = 0; peak < n; peak++)
            {
                int[] values = LcgBitonic(n, 1000 + 13 * n + peak, peak);
                checkedArrays++;
                Check(FindPeak(values) == peak, "peak " + n + "/" + peak);
                for (int i = 0; i < n; i++)
                {
                    Check(Search(values, values[i]) == i, "element " + n + "/" + peak + "/" + i);
                    Check(Search(values, values[i]) == LinearSearch(values, values[i]), "oracle");
                }
                foreach (int absent in new[] { values[0] - 1, values[peak] + 1, 1000, -1, 65537 })
                {
                    Check(Search(values, absent) == Sentinel, "absent " + n + "/" + peak);
                    Check(LinearSearch(values, absent) == Sentinel, "oracle absent");
                }
            }
        }
        Check(checkedArrays == 17 * 18 / 2, "array count " + checkedArrays);

        int[][] larger = { new[] { 256, 4242, 100 }, new[] { 257, 777, 0 },
                           new[] { 255, 31337, 254 }, new[] { 300, 5, 150 } };
        foreach (int[] spec in larger)
        {
            int n = spec[0];
            int[] values = LcgBitonic(n, spec[1], spec[2]);
            Check(FindPeak(values) == spec[2], "large peak");
            foreach (int i in new[] { 0, 1, n / 2, spec[2] - 1, spec[2], spec[2] + 1, n - 2, n - 1 })
            {
                if (i >= 0 && i < n) Check(Search(values, values[i]) == i, "large element " + i);
            }
            Check(Search(values, values[spec[2]] + 7) == Sentinel, "large absent");
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
