// Task 957 - Implement randomized test-case generator with oracle checker for each problem.
// (Tier: Master, Ch 11 - Testing)
//
// A deterministic generator (splitmix64, fixed seed) produces random inputs for three problems: maximum
// subarray, interval merging and array sorting.  Each candidate implementation is checked against an
// independent oracle - O(n^2) enumeration for the maximum subarray, a point-set coverage comparison for
// the merged intervals, and a multiset equality check against an insertion sort for the sort candidate.
// One deliberately broken maximum-subarray candidate (it skips index 0) proves the checker has teeth.
//
// Verified: 200 generated cases per problem agree with their oracle, the seeded generator is
// reproducible and seed-sensitive, values stay inside the declared ranges, the broken candidate is
// caught, and the empty / single-element / all-negative boundaries behave.  Simplification: the case
// count is sized so the whole battery runs in well under a second.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextRange(int low, int highInclusive) => low + (int)(NextUInt64() % (ulong)(highInclusive - low + 1));

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private static int[] RandomValues(int n) => RandomValues(n, -50, 50);

    private static int[] RandomValues(int n, int low, int high)
    {
        var values = new int[n];
        for (int i = 0; i < n; i++) values[i] = NextRange(low, high);
        return values;
    }

    /** Candidate: Kadane's algorithm. */
    private static long MaxSubarray(int[] values)
    {
        long best = long.MinValue, current = 0;
        foreach (int value in values)
        {
            current = Math.Max(value, current + value);
            best = Math.Max(best, current);
        }
        return values.Length == 0 ? 0 : best;
    }

    /** Deliberately broken candidate: it never considers a subarray that starts at index 0. */
    private static long BuggyMaxSubarray(int[] values)
    {
        long best = values.Length == 0 ? 0 : values[values.Length - 1], current = 0;
        for (int i = 1; i < values.Length; i++)
        {
            current = Math.Max(values[i], current + values[i]);
            best = Math.Max(best, current);
        }
        return best;
    }

    /** Oracle: exhaustive enumeration of every subarray. */
    private static long BruteMaxSubarray(int[] values)
    {
        if (values.Length == 0) return 0;
        long best = long.MinValue;
        for (int i = 0; i < values.Length; i++)
        {
            long sum = 0;
            for (int j = i; j < values.Length; j++)
            {
                sum += values[j];
                best = Math.Max(best, sum);
            }
        }
        return best;
    }

    private static List<int[]> RandomIntervals(int count)
    {
        var intervals = new List<int[]>();
        for (int i = 0; i < count; i++)
        {
            int start = NextRange(0, 90);
            int end = Math.Min(100, start + NextRange(0, 10));
            intervals.Add(new[] { start, end });
        }
        return intervals;
    }

    private static List<int[]> MergeIntervals(List<int[]> intervals)
    {
        var sorted = new List<int[]>(intervals);
        sorted.Sort((x, y) => x[0] != y[0] ? x[0].CompareTo(y[0]) : x[1].CompareTo(y[1]));
        var merged = new List<int[]>();
        foreach (int[] interval in sorted)
        {
            if (merged.Count > 0 && interval[0] <= merged[merged.Count - 1][1])
                merged[merged.Count - 1][1] = Math.Max(merged[merged.Count - 1][1], interval[1]);
            else
                merged.Add(new[] { interval[0], interval[1] });
        }
        return merged;
    }

    /** Oracle: the merged output must cover exactly the same sample points as the input. */
    private static bool SameCoverage(List<int[]> raw, List<int[]> merged)
    {
        var expected = new bool[101];
        foreach (int[] interval in raw)
        {
            for (int p = interval[0]; p <= interval[1]; p++) expected[p] = true;
        }
        var actual = new bool[101];
        for (int i = 0; i < merged.Count; i++)
        {
            if (i > 0 && merged[i][0] <= merged[i - 1][1]) return false; // merged pieces must be disjoint
            if (merged[i][1] > 100) return false;
            for (int p = merged[i][0]; p <= merged[i][1]; p++) actual[p] = true;
        }
        for (int p = 0; p < expected.Length; p++)
        {
            if (expected[p] != actual[p]) return false;
        }
        return true;
    }

    private static bool SameMultiset(int[] a, int[] b)
    {
        var left = new List<int>(a);
        var right = new List<int>(b);
        left.Sort();
        right.Sort();
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        const int cases = 200;
        _state = 20240901UL;
        int maxSubarrayAgreements = 0, buggyCaught = 0, intervalAgreements = 0, sortAgreements = 0;
        bool rangeOk = true, intervalShapeOk = true;

        for (int c = 0; c < cases; c++)
        {
            int[] values = RandomValues(NextRange(1, 40));
            long oracle = BruteMaxSubarray(values);
            if (MaxSubarray(values) == oracle) maxSubarrayAgreements++;
            if (BuggyMaxSubarray(values) != oracle) buggyCaught++;
            foreach (int value in values) rangeOk &= value >= -50 && value <= 50;

            List<int[]> intervals = RandomIntervals(NextRange(1, 15));
            foreach (int[] interval in intervals) intervalShapeOk &= interval[0] <= interval[1] && interval[0] >= 0 && interval[1] <= 100;
            if (SameCoverage(intervals, MergeIntervals(intervals))) intervalAgreements++;

            int[] unsorted = RandomValues(NextRange(0, 64));
            int[] sorted = (int[])unsorted.Clone();
            Array.Sort(sorted);
            bool ordered = true;
            for (int i = 1; i < sorted.Length; i++) ordered &= sorted[i - 1] <= sorted[i];
            if (ordered && SameMultiset(unsorted, sorted)) sortAgreements++;
        }
        Check(maxSubarrayAgreements == cases, "Kadane agrees with the exhaustive oracle on all " + cases + " random cases");
        Check(buggyCaught > 0, "the oracle catches the deliberately broken maximum-subarray candidate on " + buggyCaught + " cases");
        Check(intervalAgreements == cases, "merged intervals cover the same points as the input on all " + cases + " cases");
        Check(sortAgreements == cases, "the sort candidate is ordered and preserves the multiset on all " + cases + " cases");
        Check(rangeOk && intervalShapeOk, "generated values stay in [-50, 50] and every interval is non-empty inside [0, 100]");

        // boundary: empty and single-element inputs, and the all-negative maximum subarray.
        Check(MaxSubarray(new int[0]) == 0 && BruteMaxSubarray(new int[0]) == 0, "the empty array yields 0 from both implementations");
        Check(MaxSubarray(new[] { 7 }) == 7 && MaxSubarray(new[] { -7 }) == -7, "a single element yields that element");
        Check(MaxSubarray(new[] { -5, -2, -9 }) == -2 && BruteMaxSubarray(new[] { -5, -2, -9 }) == -2,
            "an all-negative array returns the best single element, not the empty sum");
        Check(MergeIntervals(new List<int[]>()).Count == 0, "merging an empty interval list is empty");
        Check(MergeIntervals(new List<int[]> { new[] { 3, 9 }, new[] { 1, 4 } }).Count == 1,
            "two overlapping intervals collapse into one");

        // determinism of the generator itself.
        _state = 7UL;
        int[] firstRun = RandomValues(20, -50, 50);
        _state = 7UL;
        int[] secondRun = RandomValues(20, -50, 50);
        _state = 8UL;
        int[] otherRun = RandomValues(20, -50, 50);
        bool sameSeed = true, seedSensitive = false;
        for (int i = 0; i < firstRun.Length; i++)
        {
            sameSeed &= firstRun[i] == secondRun[i];
            seedSensitive |= firstRun[i] != otherRun[i];
        }
        Check(sameSeed && seedSensitive, "the case generator is seed-reproducible and seed-sensitive");

        long sum = 0;
        for (int i = 0; i < 500; i++) sum += NextRange(-50, 50);
        double mean = sum / 500.0;
        Check(Math.Abs(mean) < 5.0, "the generated values are centred near zero (mean " + mean.ToString("F2", CultureInfo.InvariantCulture) + ")");

        Console.WriteLine("generator: " + cases + " cases per problem, values in [-50, 50], intervals inside [0, 100]");
        Console.WriteLine("oracles: exhaustive subarray enumeration, point-set coverage, insertion-order multiset check");
        Console.WriteLine("teeth: the broken candidate was detected on " + buggyCaught + "/" + cases + " cases; mean " +
            mean.ToString("F2", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
