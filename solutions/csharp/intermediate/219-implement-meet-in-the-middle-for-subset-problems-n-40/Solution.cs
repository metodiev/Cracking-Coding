// Task 219 - Implement meet-in-the-middle for subset problems (n ~ 40). (Tier: Intermediate, CTCI Ch 8)
using System;

public static class Solution
{
    private static long[] SubsetSums(int[] values)
    {
        var sums = new long[1 << values.Length];
        int size = 1;
        foreach (var x in values)
        {
            for (int i = 0; i < size; i++) sums[size + i] = sums[i] + x;
            size <<= 1;
        }
        return sums;
    }

    private class MeetInTheMiddle
    {
        private readonly long[] left, right;

        public MeetInTheMiddle(int[] nums)
        {
            int mid = nums.Length / 2;
            var leftPart = new int[mid];
            Array.Copy(nums, 0, leftPart, 0, mid);
            var rightPart = new int[nums.Length - mid];
            Array.Copy(nums, mid, rightPart, 0, rightPart.Length);
            left = SubsetSums(leftPart);
            right = SubsetSums(rightPart);
            Array.Sort(right);
        }

        private static int LowerBound(long[] a, long value)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (a[mid] < value) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }

        public bool Exists(long target)
        {
            foreach (var s in left)
            {
                int i = LowerBound(right, target - s);
                if (i < right.Length && right[i] == target - s) return true;
            }
            return false;
        }
    }

    private static bool BruteExists(int[] nums, long target)
    {
        foreach (var s in SubsetSums(nums)) if (s == target) return true;
        return false;
    }

    /// <summary>Independent oracle: subset-sum via a BigInteger bitset (brute force is 2^40 here).</summary>
    private static bool SubsetExistsBitset(int[] nums, long target)
    {
        if (target < 0) return false;
        var bits = System.Numerics.BigInteger.One;
        foreach (var x in nums) bits = bits | (bits << x);
        return ((bits >> (int)target) & System.Numerics.BigInteger.One) == System.Numerics.BigInteger.One;
    }

    public static void Main()
    {
        var rng = new Random(4);

        var small = new int[16];
        for (int i = 0; i < small.Length; i++) small[i] = 1 + rng.Next(99);
        var mitmSmall = new MeetInTheMiddle(small);
        for (int t = 0; t < 200; t++)
        {
            long target = rng.Next(600);
            Check(mitmSmall.Exists(target) == BruteExists(small, target), "small cross-check " + target);
        }

        var big = new int[40];
        long total = 0;
        for (int i = 0; i < big.Length; i++) { big[i] = 1 + rng.Next(9); total += big[i]; }
        var mitmBig = new MeetInTheMiddle(big);
        Check(mitmBig.Exists(total) == SubsetExistsBitset(big, total), "full set sums to total");
        Check(!mitmBig.Exists(total + 1), "impossible target");
        Check(mitmBig.Exists(0) == SubsetExistsBitset(big, 0), "empty subset");
        for (int t = 0; t < 3; t++)
        {
            long target = rng.Next((int)total + 1);
            Check(mitmBig.Exists(target) == SubsetExistsBitset(big, target), "big cross-check " + target);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
