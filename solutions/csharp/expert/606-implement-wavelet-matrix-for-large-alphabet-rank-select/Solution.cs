// Task 606 - Implement wavelet matrix for large alphabet rank/select. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A wavelet matrix stores an array of non-negative integers level by level, one bitvector per bit of
// the value range (levels = bits needed for maxValue). Level d holds bit levels-1-d of the elements
// in the order left by the first d stable partitions; each level is ulong words with cumulative
// popcounts per word, so rank1 (hence rank0) is one array read plus one popcount. Zeros keep the
// front of a level and ones the back, which makes the position mapping a single subtraction:
//     zeros side: rank0(position)          ones side: zeroCount[d] + rank1(position)
// On that mapping the class answers the standard query set:
//   Rank(value, position)  - occurrences of value in [0, position): descend one level per bit and
//                            return the width of the surviving range;
//   Select(value, k)       - original index of the k-th occurrence: descend to the value's range at
//                            the last level, then walk back up with select0/select1 per level (select
//                            is a binary search over word popcounts plus one word scan);
//   Quantile(lo, hi, k)    - k-th smallest value in a range;
//   RangeCount(lo, hi, ..) - how many values fall into a value interval, via "count less than".
// Nothing in the structure depends on the number of distinct symbols, so 2^20 values cost the same
// as two: only the bit width (levels) matters.
//
// Simplifications (documented, not behavioural): per-level select is the binary-search-over-words
// variant rather than a precomputed select index; the matrix itself is the real structure.
//
// Verified: rank, select, quantile and range counts against brute force on every query type for ten
// fixed and random arrays with alphabets of 2, 8, 16, 1024, 2^16 and 2^20 values; the select/rank
// round trip (the k-th occurrence has exactly k predecessors equal to the value); quantile against
// sorting the queried window; bitvector invariants (ones per level, zero counts, clear padding bits);
// and boundary cases (empty array, single element, all values equal, values 0 and maxValue, empty
// ranges, out-of-range select).
using System;
using System.Linq;
using System.Numerics;

internal class WaveletMatrix
{
    private readonly int levels, length;
    private readonly ulong[][] bits;
    private readonly int[][] prefOnes;
    private readonly int[] zeroCount;

    public WaveletMatrix(int[] values, int maxValue)
    {
        length = values.Length;
        levels = 1;
        while (maxValue >= 1 << levels) levels++;              // bits needed to represent maxValue
        bits = new ulong[levels][];
        prefOnes = new int[levels][];
        zeroCount = new int[levels];
        var current = (int[])values.Clone();
        var next = new int[length];
        for (int level = 0; level < levels; level++)
        {
            int shift = levels - 1 - level;
            int words = (length + 63) / 64;
            bits[level] = new ulong[words];
            prefOnes[level] = new int[words + 1];
            for (int i = 0; i < length; i++)
            {
                if (((current[i] >> shift) & 1) != 0) bits[level][i >> 6] |= 1UL << (i & 63);
                else zeroCount[level]++;
            }
            for (int w = 0; w < words; w++) prefOnes[level][w + 1] = prefOnes[level][w] + BitOperations.PopCount(bits[level][w]);
            int zeros = 0, ones = zeroCount[level];
            for (int i = 0; i < length; i++)                       // stable partition, zeros first
            {
                if (((current[i] >> shift) & 1) == 0) next[zeros++] = current[i];
                else next[ones++] = current[i];
            }
            (current, next) = (next, current);
        }
    }

    public int Length => length;
    public int Levels => levels;
    public int WordCount => levels * ((length + 63) / 64); public int ZerosAt(int level) => zeroCount[level];

    private int Rank1(int level, int position)
    {
        int word = position >> 6, offset = position & 63;
        int ones = prefOnes[level][word];
        if (offset != 0) ones += BitOperations.PopCount(bits[level][word] & ((1UL << offset) - 1));
        return ones;
    }

    private int Rank0(int level, int position) => position - Rank1(level, position);

    private int ZerosBefore(int level, int words) => (words << 6) - prefOnes[level][words];

    private int Select1(int level, int k)
    {
        int lo = 0, hi = bits[level].Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (prefOnes[level][mid + 1] <= k) lo = mid + 1;
            else hi = mid;
        }
        int remaining = k - prefOnes[level][lo];
        for (int b = 0; b < 64; b++)
        {
            if (((bits[level][lo] >> b) & 1UL) != 0 && remaining-- == 0) return (lo << 6) + b;
        }
        return -1;
    }

    private int Select0(int level, int k)
    {
        int lo = 0, hi = bits[level].Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (ZerosBefore(level, mid + 1) <= k) lo = mid + 1;
            else hi = mid;
        }
        int remaining = k - ZerosBefore(level, lo);
        ulong flipped = ~bits[level][lo];
        for (int b = 0; b < 64; b++)
        {
            if (((flipped >> b) & 1UL) != 0 && remaining-- == 0) return (lo << 6) + b;
        }
        return -1;
    }

    // Occurrences of value in [0, position): descend one level per bit, the range width is the count.
    public int Rank(int value, int position) => RankPrefix(value, Math.Min(position, length));

    private int RankPrefix(int value, int position)
    {
        int lo = 0, hi = position;
        for (int level = 0; level < levels; level++)
        {
            if (((value >> (levels - 1 - level)) & 1) == 0)
            {
                lo = Rank0(level, lo);
                hi = Rank0(level, hi);
            }
            else
            {
                lo = zeroCount[level] + Rank1(level, lo);
                hi = zeroCount[level] + Rank1(level, hi);
            }
        }
        return hi - lo;
    }

    public int CountInRange(int lo, int hi, int value) => RankPrefix(value, hi) - RankPrefix(value, lo);

    // Original index of the k-th (0-based) occurrence of value, -1 when there is none.
    public int Select(int value, int k)
    {
        if (k < 0 || k >= RankPrefix(value, length)) return -1;
        int lo = 0, hi = length;
        for (int level = 0; level < levels; level++)
        {
            if (((value >> (levels - 1 - level)) & 1) == 0)
            {
                lo = Rank0(level, lo);
                hi = Rank0(level, hi);
            }
            else
            {
                lo = zeroCount[level] + Rank1(level, lo);
                hi = zeroCount[level] + Rank1(level, hi);
            }
        }
        int position = lo + k;                                 // k-th occurrence in the last level order
        for (int level = levels - 1; level >= 0; level--)
        {
            int bit = (value >> (levels - 1 - level)) & 1;
            int offset = position - (bit == 0 ? 0 : zeroCount[level]);
            position = bit == 0 ? Select0(level, offset) : Select1(level, offset);
        }
        return position;
    }

    public int Quantile(int lo, int hi, int k)
    {
        int value = 0;
        for (int level = 0; level < levels; level++)
        {
            int zeros = Rank0(level, hi) - Rank0(level, lo);
            if (k < zeros)
            {
                lo = Rank0(level, lo);
                hi = Rank0(level, hi);
            }
            else
            {
                k -= zeros;
                value |= 1 << (levels - 1 - level);
                lo = zeroCount[level] + Rank1(level, lo);
                hi = zeroCount[level] + Rank1(level, hi);
            }
        }
        return value;
    }

    private int CountLessThan(int lo, int hi, int x)
    {
        if (x >= 1 << levels) return hi - lo;
        if (x <= 0) return 0;
        int count = 0;
        for (int level = 0; level < levels; level++)
        {
            int lo0 = Rank0(level, lo), hi0 = Rank0(level, hi);
            if (((x >> (levels - 1 - level)) & 1) != 0)
            {
                count += hi0 - lo0;                            // the whole zero side is smaller than x
                lo = zeroCount[level] + Rank1(level, lo);
                hi = zeroCount[level] + Rank1(level, hi);
            }
            else
            {
                lo = lo0;
                hi = hi0;
            }
        }
        return count;
    }

    public int RangeCount(int lo, int hi, int minValue, int maxValue) =>
        CountLessThan(lo, hi, maxValue + 1) - CountLessThan(lo, hi, minValue);

    public bool BitvectorsAreConsistent(int[] values)
    {
        for (int level = 0; level < levels; level++)
        {
            int ones = values.Count(v => ((v >> (levels - 1 - level)) & 1) != 0);
            if (ones != prefOnes[level][bits[level].Length] || zeroCount[level] != length - ones) return false;
            if (length % 64 != 0 && bits[level].Length > 0 && (bits[level][^1] >> (length % 64)) != 0) return false;
        }
        return true;
    }
}

public static class Solution
{
    private static ulong seed = 0x510E527FADE682D1UL;

    private static ulong NextRandom()
    {
        seed += 0x9E3779B97F4A7C15UL;
        ulong z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static int[] RandomArray(int length, int maxValue)
    {
        var values = new int[length];
        for (int i = 0; i < length; i++) values[i] = (int)(NextRandom() % (ulong)(maxValue + 1));
        return values;
    }

    // Every query type against brute force on one array.
    private static void Battery(int[] values, int maxValue)
    {
        var matrix = new WaveletMatrix(values, maxValue);
        Check(matrix.BitvectorsAreConsistent(values), "bitvector invariants");
        var distinct = values.Distinct().ToArray();
        foreach (int value in distinct)
        {
            int total = values.Count(v => v == value);
            for (int position = 0; position <= values.Length; position += Math.Max(1, values.Length / 40))
            {
                Check(matrix.Rank(value, position) == values.Take(position).Count(v => v == value),
                    "rank " + value + " at " + position);
            }
            for (int k = 0; k < total; k++)
            {
                int index = matrix.Select(value, k);
                Check(index >= 0 && values[index] == value, "select value " + value);
                Check(values.Take(index).Count(v => v == value) == k, "select order " + value + " " + k);
            }
            Check(matrix.Select(value, total) == -1 && matrix.Select(value, -1) == -1, "select out of range");
        }
        for (int lo = 0; lo < values.Length; lo += 7)
        {
            for (int hi = lo; hi <= values.Length; hi += 11)
            {
                var slice = values.Skip(lo).Take(hi - lo).OrderBy(v => v).ToArray();
                if (slice.Length == 0) continue;
                for (int k = 0; k < slice.Length; k += Math.Max(1, slice.Length / 4))
                    Check(matrix.Quantile(lo, hi, k) == slice[k], "quantile " + lo + "," + hi + "," + k);
                for (int bound = 0; bound <= maxValue + 1; bound += Math.Max(1, (maxValue + 1) / 5))
                {
                    Check(matrix.RangeCount(lo, hi, 0, bound - 1) == slice.Count(v => v < bound),
                        "range count " + lo + "," + hi + "," + bound);
                }
            }
        }
    }

    public static void Main()
    {
        var empty = new WaveletMatrix(Array.Empty<int>(), 0);
        Check(empty.Length == 0 && empty.Rank(0, 0) == 0 && empty.Select(0, 0) == -1, "empty array");

        var single = new WaveletMatrix(new[] { 7 }, 7);
        Check(single.Rank(7, 1) == 1 && single.Select(7, 0) == 0 && single.Quantile(0, 1, 0) == 7, "single element");

        var equal = new WaveletMatrix(new[] { 5, 5, 5, 5, 5 }, 5);
        Check(equal.Rank(5, 5) == 5 && equal.Select(5, 4) == 4 && equal.RangeCount(0, 5, 5, 5) == 5 &&
            equal.RangeCount(0, 5, 0, 4) == 0 && equal.RangeCount(0, 5, 6, 9) == 0, "all equal");

        var extremes = new WaveletMatrix(new[] { 0, 1 << 20, 5, 1 << 20, 0 }, 1 << 20);
        Check(extremes.Levels == 21 && extremes.Rank(0, 5) == 2 && extremes.Select(1 << 20, 1) == 3 &&
            extremes.Quantile(0, 5, 2) == 5 && extremes.Quantile(0, 5, 4) == 1 << 20, "extreme values");

        Battery(new[] { 0 }, 0);
        Battery(new[] { 1, 0, 1, 0 }, 1);
        Battery(new[] { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 }, 15);
        Battery(new[] { 0, 1023, 512, 0, 1023, 1, 1022 }, 1023);
        Battery(RandomArray(200, 1), 1);
        Battery(RandomArray(200, 15), 15);
        Battery(RandomArray(300, 1023), 1023);
        Battery(RandomArray(300, (1 << 20) - 1), (1 << 20) - 1);
        Battery(RandomArray(500, 65535), 65535);
        Battery(RandomArray(700, 7), 7);

        // Large alphabet, long array: rank/select on every distinct value, quantiles over windows.
        int maxLarge = (1 << 20) - 1;
        var large = RandomArray(2000, maxLarge);
        var largeMatrix = new WaveletMatrix(large, maxLarge);
        Check(largeMatrix.Levels == 20 && largeMatrix.BitvectorsAreConsistent(large), "large invariants");
        foreach (int value in large.Distinct())
        {
            int total = large.Count(v => v == value);
            for (int k = 0; k < total; k++)
            {
                int index = largeMatrix.Select(value, k);
                Check(large[index] == value && large.Take(index).Count(v => v == value) == k, "large select");
            }
            Check(largeMatrix.Rank(value, 1234) == large.Take(1234).Count(v => v == value), "large rank");
        }
        var window = large.Skip(100).Take(500).OrderBy(v => v).ToArray();
        for (int k = 0; k < 500; k += 97) Check(largeMatrix.Quantile(100, 600, k) == window[k], "large quantile");
        Check(largeMatrix.RangeCount(100, 600, 0, maxLarge / 2) == window.Count(v => v <= maxLarge / 2), "large range count");
        Check(largeMatrix.RangeCount(300, 300, 0, maxLarge) == 0, "empty range");

        Console.WriteLine("2000 values in [0, 2^20-1] -> " + largeMatrix.Levels + " levels, " + largeMatrix.WordCount +
            " words for " + large.Distinct().Count() + " distinct values");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
