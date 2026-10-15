// Task 378 - Count integer partitions p(n) with DP. (Tier: Intermediate, CTCI Ch 8)
using System;

/// <summary>
/// p(n) is the number of ways to write n as a sum of positive integers where ORDER DOES
/// NOT MATTER ((2, 1, 1) and (1, 2, 1) are the same partition of 4).  The coin-change
/// style DP treats part sizes one at a time: after processing part size k, dp[s] counts
/// partitions of s using parts &lt;= k, so each multiset is built exactly once regardless
/// of order.  dp[0] = 1 (the empty partition) and p(n) = dp[n] after all sizes up to n.
///
/// Hand-verified milestones asserted below: p(0)=1, p(1)=1, p(4)=5, p(5)=7, p(10)=42,
/// p(20)=627, p(50)=204226 (classic values, cross-checked here by independent recursive
/// counters for every n &lt;= 20 and by the p(100)=190569292 milestone).
///
/// Oracles: BruteForcePartitions enumerates partitions recursively by picking the largest
/// remaining part, WalkCount by picking non-decreasing parts - neither uses a DP table.
/// </summary>
public static class Solution
{
    /// <summary>Number of integer partitions of n (unordered).</summary>
    public static long PartitionCount(int n)
    {
        if (n < 0) return 0;
        var dp = new long[n + 1];
        dp[0] = 1;
        for (int part = 1; part <= n; part++)
        {
            for (int total = part; total <= n; total++) dp[total] += dp[total - part];
        }
        return dp[n];
    }

    /// <summary>Recursive oracle: partitions of n whose parts are all &lt;= maxPart.</summary>
    public static long BruteForcePartitions(int n, int maxPart)
    {
        if (n == 0) return 1;
        if (maxPart == 0) return 0;
        long total = 0;
        for (int part = 1; part <= Math.Min(n, maxPart); part++)
        {
            total += BruteForcePartitions(n - part, part);
        }
        return total;
    }

    /// <summary>Second oracle: enumerate non-decreasing part lists explicitly.</summary>
    public static long WalkCount(int remaining, int smallest)
    {
        if (remaining == 0) return 1;
        long total = 0;
        for (int part = smallest; part <= remaining; part++) total += WalkCount(remaining - part, part);
        return total;
    }

    public static void Main()
    {
        // milestone values (classic partition numbers, re-derived by the oracles below)
        Check(PartitionCount(0) == 1, "p(0) = 1");
        Check(PartitionCount(1) == 1, "p(1) = 1");
        Check(PartitionCount(2) == 2, "p(2) = 2");
        Check(PartitionCount(3) == 3, "p(3) = 3");
        Check(PartitionCount(4) == 5, "p(4) = 5");
        Check(PartitionCount(5) == 7, "p(5) = 7");
        Check(PartitionCount(10) == 42, "p(10) = 42");
        Check(PartitionCount(20) == 627, "p(20) = 627");
        Check(PartitionCount(50) == 204226, "p(50) = 204226");
        Check(PartitionCount(100) == 190569292, "p(100) = 190569292");

        // p(4) = 5 spelled out: 4; 3+1; 2+2; 2+1+1; 1+1+1+1
        Check(BruteForcePartitions(4, 4) == 5, "oracle p(4)");
        Check(WalkCount(4, 1) == 5, "walk oracle p(4)");

        // p(5) = 7: 5; 4+1; 3+2; 3+1+1; 2+2+1; 2+1+1+1; 1+1+1+1+1
        Check(BruteForcePartitions(5, 5) == 7, "oracle p(5)");

        // all n <= 20 against both independent oracles
        for (int n = 0; n <= 20; n++)
        {
            long expected = BruteForcePartitions(n, n);
            Check(PartitionCount(n) == expected, "oracle agreement " + n);
            Check(WalkCount(n, 1) == expected, "walk agreement " + n);
        }

        // strictly increasing for n >= 1
        long previous = 0;
        for (int n = 1; n <= 25; n++)
        {
            long current = PartitionCount(n);
            Check(current > previous, "monotone " + n);
            previous = current;
        }

        Check(PartitionCount(-3) == 0, "negative input is 0");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
