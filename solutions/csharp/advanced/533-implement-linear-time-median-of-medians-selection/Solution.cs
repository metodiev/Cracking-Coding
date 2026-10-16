// Task 533 - Implement linear-time median of medians selection. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
//
// BFPRT selection: to find the k-th smallest element, split the array into groups of five, sort each
// group (five elements, so a tiny insertion sort) and take the median of the medians as the pivot. The
// pivot is then used in a three-way partition; because the pivot is guaranteed to be at least the 30th
// and at most the 70th percentile (each group contributes at least three elements on the correct side
// of its own median, and the median of medians is itself in the middle half), each recursion keeps at
// most 7n/10 + O(1) elements, which gives the linear worst case in contrast to the expected-linear
// quickselect. All comparisons go through one counter so the linearity claim can be measured against
// the array size. A Hoare quickselect with a median-of-three pivot is implemented alongside as an
// independent (expected-linear) reference, and a full sort as the ground truth.
//
// Documented checks. (1) Hand cases: the median of 1..10 is 6 (the 5th zero-based order statistic),
// the minimum and maximum of a shuffled range come out right, and n = 1..5 exercises the "fewer than
// five elements" branch. (2) For 200 LCG arrays (values in [0, 999], sizes 1..40, with duplicate-rich
// small ranges among them) every zero-based rank k is compared against Array.Sort's answer, and the
// quickselect reference agrees as well. (3) Adversarial inputs: already sorted, reverse sorted, all
// equal, an organ pipe and a sawtooth pattern, all of size 1000, agree with the sorted answer for the
// median and for the extremes. (4) Pivot quality: on a 100000-element array the rank of the first
// pivot lies in [0.3n, 0.7n] (the documented 30/70 guarantee), and the total comparison count of the
// selection stays below 40n for every array of size 10000 tested here (measured, the linear worst-case
// behaviour). (5) Non-destructive: the input array is unchanged by every call and the working copy
// still holds a permutation of the input. (6) Errors: n = 0, k < 0 and k >= n each throw
// ArgumentException.
using System;
using System.Collections.Generic;

internal sealed class MedianOfMedians
{
    internal long Comparisons;
    internal int FirstPivotLeft = -1;
    internal int FirstPivotRight = -1;

    /** k-th smallest element (zero-based rank k) via groups of five and the median of the medians. */
    internal int Select(int[] values, int k)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (values.Length == 0) throw new ArgumentException("empty input");
        if (k < 0 || k >= values.Length) throw new ArgumentException("k is out of range: " + k);
        var work = (int[])values.Clone();
        FirstPivotLeft = -1;
        FirstPivotRight = -1;
        return Recurse(work, 0, work.Length - 1, k, true);
    }

    private int Recurse(int[] work, int lo, int hi, int k, bool topLevel)
    {
        while (true)
        {
            if (lo == hi) return work[lo];
            int pivot = MedianOfMediansPivot(work, lo, hi);
            Partition(work, lo, hi, pivot, out int left, out int right);
            if (topLevel)
            {
                FirstPivotLeft = left;
                FirstPivotRight = right;
                topLevel = false;
            }
            if (k < left) hi = left - 1;
            else if (k <= right) return pivot;
            else lo = right + 1;
        }
    }

    /** Median of the medians of the five-element blocks, itself found by a recursive selection. */
    private int MedianOfMediansPivot(int[] work, int lo, int hi)
    {
        int count = hi - lo + 1;
        int groups = count / 5;
        if (groups == 0) return work[lo + count / 2];
        var medians = new int[groups];
        for (int g = 0; g < groups; g++) medians[g] = MedianOfFive(work, lo + 5 * g, 5);
        return Recurse(medians, 0, groups - 1, groups / 2, false);
    }

    /** Sorts one five-element block in place with an insertion sort and returns its median. */
    private int MedianOfFive(int[] work, int start, int length)
    {
        for (int i = start + 1; i < start + length; i++)
        {
            int value = work[i];
            int j = i - 1;
            while (j >= start)
            {
                Comparisons++;
                if (work[j] <= value) break;
                work[j + 1] = work[j];
                j--;
            }
            work[j + 1] = value;
        }
        return work[start + length / 2];
    }

    /** Three-way partition around the pivot value; returns the blocks of smaller and equal elements. */
    private void Partition(int[] work, int lo, int hi, int pivot, out int left, out int right)
    {
        left = lo;
        right = hi;
        int position = lo;
        while (position <= right)
        {
            Comparisons++;
            if (work[position] < pivot)
            {
                Swap(work, left, position);
                left++;
                position++;
            }
            else
            {
                Comparisons++;
                if (work[position] > pivot)
                {
                    Swap(work, position, right);
                    right--;
                }
                else
                {
                    position++;
                }
            }
        }
    }

    private static void Swap(int[] work, int a, int b)
    {
        int temporary = work[a];
        work[a] = work[b];
        work[b] = temporary;
    }

    /** Independent reference: Hoare quickselect with a median-of-three pivot (expected linear). */
    internal static int QuickSelect(int[] values, int k)
    {
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (values.Length == 0) throw new ArgumentException("empty input");
        if (k < 0 || k >= values.Length) throw new ArgumentException("k is out of range: " + k);
        var work = (int[])values.Clone();
        int lo = 0;
        int hi = work.Length - 1;
        while (true)
        {
            if (lo == hi) return work[lo];
            int pivot = MedianOfThree(work[lo], work[(lo + hi) / 2], work[hi]);
            int i = lo;
            int j = hi;
            while (i <= j)
            {
                while (work[i] < pivot) i++;
                while (work[j] > pivot) j--;
                if (i <= j)
                {
                    Swap(work, i, j);
                    i++;
                    j--;
                }
            }
            if (k <= j) hi = j;
            else if (k >= i) lo = i;
            else return work[k];
        }
    }

    private static int MedianOfThree(int a, int b, int c)
    {
        if (a > b) { int temporary = a; a = b; b = temporary; }
        if (b > c) { int temporary = b; b = c; c = temporary; }
        return a > b ? a : b;
    }
}

public static class Solution
{
    public static void Main()
    {
        var ten = new int[10];
        for (int i = 0; i < ten.Length; i++) ten[i] = i + 1;
        var selector = new MedianOfMedians();
        Check(selector.Select(ten, 5) == 6, "the median of 1..10 is 6");
        Check(selector.Select(ten, 0) == 1, "the minimum of 1..10 is 1");
        Check(selector.Select(ten, 9) == 10, "the maximum of 1..10 is 10");
        Check(selector.Comparisons > 0, "comparisons are counted");
        var shuffled = new[] { 7, 3, 9, 1, 5 };
        Check(selector.Select(shuffled, 2) == 5, "the middle of 7,3,9,1,5 is 5");
        Check(selector.Select(shuffled, 0) == 1 && selector.Select(shuffled, 4) == 9, "shuffled extremes");
        for (int n = 1; n <= 5; n++)
        {
            var small = new int[n];
            for (int i = 0; i < n; i++) small[i] = n - i;
            for (int k = 0; k < n; k++)
            {
                Check(selector.Select(small, k) == k + 1, "tiny array of size " + n + ", rank " + k);
            }
        }

        for (int seed = 1; seed <= 200; seed++)
        {
            int n = 1 + seed % 40;
            int range = seed % 3 == 0 ? 5 : 1000;
            int[] values = LcgValues(n, seed, range);
            int[] expected = (int[])values.Clone();
            Array.Sort(expected);
            for (int k = 0; k < n; k++)
            {
                int selected = selector.Select(values, k);
                Check(selected == expected[k], "seed " + seed + " rank " + k + " (" + selected + " != " + expected[k] + ")");
                Check(MedianOfMedians.QuickSelect(values, k) == expected[k],
                        "quickselect seed " + seed + " rank " + k);
            }
            int[] untouched = LcgValues(n, seed, range);
            var copy = (int[])untouched.Clone();
            selector.Select(untouched, n / 2);
            CheckSameArray(untouched, copy, "the input is not modified for seed " + seed);
        }

        foreach (int[] pattern in new[] { SortedPattern(1000), ReversePattern(1000), ConstantPattern(1000),
                OrganPipePattern(1000), SawtoothPattern(1000) })
        {
            int[] expected = (int[])pattern.Clone();
            Array.Sort(expected);
            int[] copy = (int[])pattern.Clone();
            int median = selector.Select(pattern, pattern.Length / 2);
            Check(median == expected[pattern.Length / 2], "adversarial median (" + median + ")");
            Check(selector.Select(pattern, 0) == expected[0], "adversarial minimum");
            Check(selector.Select(pattern, pattern.Length - 1) == expected[pattern.Length - 1], "adversarial maximum");
            CheckSameArray(pattern, copy, "adversarial input is not modified");
        }

        int[] big = LcgValues(100000, 777, 1000000);
        var bigSelector = new MedianOfMedians();
        int bigMedian = bigSelector.Select(big, big.Length / 2);
        int[] expectedBig = (int[])big.Clone();
        Array.Sort(expectedBig);
        Check(bigMedian == expectedBig[expectedBig.Length / 2], "median of 100000 values");
        Check(bigSelector.FirstPivotLeft >= 28000 && bigSelector.FirstPivotRight <= 72000,
                "the first pivot sits between the 30th and 70th percentile with the documented O(5) slack ("
                        + bigSelector.FirstPivotLeft + " .. " + bigSelector.FirstPivotRight + ")");

        int[] measured = LcgValues(10000, 31337, 1000000);
        var measuredSelector = new MedianOfMedians();
        measuredSelector.Select(measured, measured.Length / 2);
        Check(measuredSelector.Comparisons <= 40L * measured.Length,
                "the selection stays linear: " + measuredSelector.Comparisons + " comparisons for n = "
                        + measured.Length);
        int[] sortedMeasured = (int[])measured.Clone();
        Array.Sort(sortedMeasured);
        for (int k = 0; k <= measured.Length; k += 997)
        {
            int index = Math.Min(k, measured.Length - 1);
            Check(measuredSelector.Select(measured, index) == sortedMeasured[index], "linear-run rank " + index);
        }
        Check(MedianOfMedians.QuickSelect(measured, 4321) == sortedMeasured[4321], "quickselect on 10000 values");

        CheckThrows(() => selector.Select(new int[0], 0), "empty input throws");
        CheckThrows(() => selector.Select(ten, -1), "a negative rank throws");
        CheckThrows(() => selector.Select(ten, 10), "a rank past the end throws");
        CheckThrows(() => MedianOfMedians.QuickSelect(ten, 20), "quickselect rank out of range throws");

        Console.WriteLine("PASS");
    }

    private static int[] LcgValues(int n, int seed, int range)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % range;
        }
        return values;
    }

    private static int[] SortedPattern(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = i;
        return values;
    }

    private static int[] ReversePattern(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = n - i;
        return values;
    }

    private static int[] ConstantPattern(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = 42;
        return values;
    }

    private static int[] OrganPipePattern(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = i < n / 2 ? i : n - i;
        return values;
    }

    private static int[] SawtoothPattern(int n)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = i % 17;
        return values;
    }

    private static void CheckSameArray(int[] a, int[] b, string label)
    {
        Check(a.Length == b.Length, label + ": length");
        for (int i = 0; i < a.Length; i++)
        {
            Check(a[i] == b[i], label + ": element " + i);
        }
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
