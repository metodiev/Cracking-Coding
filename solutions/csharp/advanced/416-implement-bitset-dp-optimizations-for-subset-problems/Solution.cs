// Task 416 - Implement bitset DP optimizations for subset problems. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Subset-sum reachability with a bitset: bit s of `mask` is set iff some subset of the
// processed values sums to s. Adding value v shifts the whole reachable set by v:
//     mask |= mask << v        (masked to the [0, limit] window after every step)
// C#'s int/long cannot address bit 4000 or 20000, so the bitset is a
// System.Numerics.BigInteger; bit counting is done per byte with BitOperations.PopCount.
// The classic boolean DP is the oracle and the reachable sets must match exactly.
using System;
using System.Numerics;

public static class Solution
{
    public static BigInteger ReachableMask(int[] values, int limit)
    {
        BigInteger mask = BigInteger.One;               // the empty subset sums to 0
        BigInteger window = (BigInteger.One << (limit + 1)) - BigInteger.One;
        foreach (int v in values)
        {
            if (v <= limit) mask = (mask | (mask << v)) & window;
        }
        return mask & window;
    }

    public static bool CanMake(int[] values, int target, int limit)
    {
        if (target < 0 || target > limit) return false;
        return !((ReachableMask(values, limit) >> target) & BigInteger.One).IsZero;
    }

    public static bool CanMake(int[] values, int target) => CanMake(values, target, target);

    public static bool CanPartition(int[] values)
    {
        long total = 0;
        foreach (int v in values) total += v;
        if (total % 2 != 0) return false;
        return CanMake(values, (int)(total / 2));
    }

    /// <summary>Oracle: plain boolean DP over sums 0..limit.</summary>
    public static bool[] ClassicDp(int[] values, int limit)
    {
        var reach = new bool[limit + 1];
        reach[0] = true;
        foreach (int v in values)
        {
            if (v > limit) continue;
            for (int s = limit; s >= v; s--)
            {
                if (reach[s - v]) reach[s] = true;
            }
        }
        return reach;
    }

    private static int MaskCount(BigInteger mask)
    {
        int count = 0;
        foreach (byte b in mask.ToByteArray()) count += BitOperations.PopCount((uint)b);
        return count;
    }

    private static bool HasBit(BigInteger mask, int s) => !((mask >> s) & BigInteger.One).IsZero;

    private static int seed = 416;

    private static int[] LcgValues(int count, int lo, int hi)
    {
        var values = new int[count];
        for (int k = 0; k < count; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[k] = lo + seed % (hi - lo + 1);
        }
        return values;
    }

    public static void Main()
    {
        int[] baseSet = { 3, 34, 4, 12, 5, 2 };
        // classic subset-sum hand cases
        Check(CanMake(baseSet, 9), "base 9");            // 3 + 4 + 2 (or 5 + 4)
        Check(!CanMake(baseSet, 30), "base 30");         // sums jump from 26 to 34+
        Check(CanMake(baseSet, 34), "base 34");          // the single element 34
        Check(CanMake(baseSet, 0), "base 0");            // the empty subset
        Check(CanMake(baseSet, 60), "base 60");          // everything
        Check(!CanMake(baseSet, 59), "base 59");
        Check(CanMake(baseSet, 50), "base 50");          // 34 + 12 + 4

        // hand-derived reachable set of {2, 3, 4, 5, 12, 34}: {2,3,4,5} reaches
        // {0, 2..12, 14} (1 and 13 stay empty), adding 12 fills {0, 2..12, 14..24, 26}
        // (13 and 25 stay empty), and adding 34 mirrors that set 34 higher:
        // {0, 2..12, 14..24, 26, 34, 36..46, 48..58, 60} — 48 values.
        BigInteger mask = ReachableMask(baseSet, 60);
        Check(MaskCount(mask) == 48, "hand count 48");
        foreach (int s in new[] { 0, 2, 12, 14, 24, 26, 34, 36, 46, 48, 58, 60 })
        {
            Check(HasBit(mask, s), "hand present " + s);
        }
        foreach (int s in new[] { 1, 13, 25, 35, 47, 59 })
        {
            Check(!HasBit(mask, s), "hand absent " + s);
        }
        Check(!CanMake(baseSet, 25), "base 25");

        // equal-partition hand cases
        Check(CanPartition(new[] { 1, 5, 11, 5 }), "partition 1 5 11 5");   // 11 vs 1 + 5 + 5
        Check(!CanPartition(new[] { 1, 5, 3 }), "partition odd total");
        Check(CanPartition(new[] { 2, 2, 2, 2 }), "partition four 2s");

        // oracle agreement on hand and LCG sets
        int[][] handSets = { new[] { 3, 34, 4, 12, 5, 2 }, new[] { 1, 5, 11, 5 }, new[] { 2, 2, 2, 2 }, new[] { 7 }, Array.Empty<int>() };
        int[] handLimits = { 60, 22, 8, 7, 5 };
        for (int h = 0; h < handSets.Length; h++)
        {
            BigInteger m = ReachableMask(handSets[h], handLimits[h]);
            var oracle = ClassicDp(handSets[h], handLimits[h]);
            for (int s = 0; s <= handLimits[h]; s++)
            {
                Check(HasBit(m, s) == oracle[s], "hand oracle set " + h + " sum " + s);
            }
        }

        for (int round = 0; round < 8; round++)
        {
            int n = 10 + seed % 31;
            var values = LcgValues(n, 1, 60);
            int sum = 0;
            foreach (int v in values) sum += v;
            int limit = Math.Min(sum, 1500);
            BigInteger m = ReachableMask(values, limit);
            var oracle = ClassicDp(values, limit);
            for (int s = 0; s <= limit; s++)
            {
                Check(HasBit(m, s) == oracle[s], "lcg " + round + " sum " + s);
            }
            Check(m.GetBitLength() <= limit + 1, "lcg mask length " + round);
            int running = 0;
            foreach (int v in values)
            {
                running += v;
                if (running <= limit) Check(HasBit(m, running), "prefix sum " + round);
            }
            Check(!CanMake(values, limit + 1, limit), "truncated target " + round);
        }

        // limits beyond 63 bits: a single 4000-value jump needs bit 4000
        BigInteger big = ReachableMask(new[] { 4000 }, 5000);
        Check(HasBit(big, 4000) && !HasBit(big, 4001), "bit 4000");
        Check(big.GetBitLength() == 4001, "bit length 4001");
        Check(CanMake(new[] { 4000 }, 4000) && !CanMake(new[] { 4000 }, 3999), "single jump");

        // n = 40 stress: no 32/64-bit integer could hold these masks
        seed = 65537 / 2;
        for (int round = 0; round < 3; round++)
        {
            var values = LcgValues(40, 1, 1000);
            int limit = 20000;
            BigInteger m = ReachableMask(values, limit);
            var oracle = ClassicDp(values, limit);
            int reached = 0;
            for (int s = 0; s <= limit; s++)
            {
                bool bit = HasBit(m, s);
                Check(bit == oracle[s], "stress " + round + " sum " + s);
                if (bit) reached++;
            }
            int subsetSum = 0;
            for (int k = 0; k < 20; k++) subsetSum += values[k];
            if (subsetSum <= limit) Check(HasBit(m, subsetSum), "stress subset sum");
            Check(m.GetBitLength() <= limit + 1, "stress length");
            int running = 0;
            for (int k = 0; k < 5; k++)
            {
                running += values[k];
                Check(HasBit(m, running), "stress prefix");
            }
            Check(reached >= 1, "stress reached");
        }

        // partition on n = 40 inputs (both parities exercised)
        var values40 = new int[40];
        for (int i = 0; i < 40; i++) values40[i] = i + 1;      // total 820, even
        Check(CanPartition(values40), "partition 1..40");      // 1..n reaches every sum
        var values41 = new int[41];
        for (int i = 0; i < 41; i++) values41[i] = i + 1;      // total 861, odd
        Check(!CanPartition(values41), "partition odd 1..41");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
