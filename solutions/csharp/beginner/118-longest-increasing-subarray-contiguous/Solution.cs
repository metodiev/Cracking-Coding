// Task 118 - Longest increasing subarray (contiguous). (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Returns {length, start}; start = -1 for an empty array.</summary>
    public static int[] LongestIncreasingSubarray(int[] a)
    {
        if (a.Length == 0) return new[] { 0, -1 };
        int bestLen = 1, bestStart = 0, curLen = 1, curStart = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] > a[i - 1])
            {
                curLen++;
            }
            else
            {
                curLen = 1;
                curStart = i;
            }
            if (curLen > bestLen)
            {
                bestLen = curLen;
                bestStart = curStart;
            }
        }
        return new[] { bestLen, bestStart };
    }

    public static void Main()
    {
        Check(LongestIncreasingSubarray(new[] { 1, 2, 3, 1, 2 })[0] == 3, "run at start");
        Check(LongestIncreasingSubarray(new[] { 5, 4, 3 })[0] == 1, "decreasing");
        var r = LongestIncreasingSubarray(new[] { 1, 3, 2, 4, 5, 6 });
        Check(r[0] == 4 && r[1] == 2, "later run");
        Check(LongestIncreasingSubarray(new int[0])[1] == -1, "empty");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
