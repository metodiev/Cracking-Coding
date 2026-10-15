// Task 092 - Max over sliding window using deque. (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static int[] MaxSlidingWindowNaive(int[] nums, int k)
    {
        int n = nums.Length;
        if (k <= 0 || k > n) return new int[0];
        var result = new int[n - k + 1];
        for (int start = 0; start + k <= n; start++)
        {
            int best = nums[start];
            for (int i = start + 1; i < start + k; i++)
            {
                if (nums[i] > best) best = nums[i];
            }
            result[start] = best;
        }
        return result;
    }

    public static int[] MaxSlidingWindowDeque(int[] nums, int k)
    {
        int n = nums.Length;
        if (k <= 0 || k > n) return new int[0];
        // Monotonic deque as an index ring: indices dq[head..tail) with decreasing values.
        var dq = new int[n];
        int head = 0, tail = 0;
        var result = new List<int>();
        for (int i = 0; i < n; i++)
        {
            while (tail > head && nums[dq[tail - 1]] <= nums[i]) tail--;
            dq[tail++] = i;
            if (dq[head] <= i - k) head++;
            if (i >= k - 1) result.Add(nums[dq[head]]);
        }
        return result.ToArray();
    }

    public static void Main()
    {
        var nums = new[] { 1, 3, -1, -3, 5, 3, 6, 7 };
        Check(Same(MaxSlidingWindowDeque(nums, 3), new[] { 3, 3, 5, 5, 6, 7 }), "k=3");
        Check(Same(MaxSlidingWindowDeque(nums, 1), nums), "k=1");
        Check(Same(MaxSlidingWindowDeque(nums, 8), new[] { 7 }), "k=n");
        Check(Same(MaxSlidingWindowDeque(new[] { 9, 9, 9 }, 2), new[] { 9, 9 }), "duplicates");
        Check(Same(MaxSlidingWindowDeque(new[] { -5, -2, -9 }, 2), new[] { -2, -2 }), "negatives");
        Check(Same(MaxSlidingWindowDeque(new[] { 5 }, 1), new[] { 5 }), "single");
        Check(Same(MaxSlidingWindowDeque(new[] { 1, 2, 3 }, 0), new int[0]), "k=0");
        Check(Same(MaxSlidingWindowDeque(new[] { 1, 2, 3 }, 4), new int[0]), "k>n");
        Check(Same(MaxSlidingWindowDeque(new int[0], 1), new int[0]), "empty");

        var arr = new[] { 9, -4, 7, 7, 2, -1, 8, 0, 5, 3 };
        for (int k = 0; k <= arr.Length + 1; k++)
        {
            Check(Same(MaxSlidingWindowDeque(arr, k), MaxSlidingWindowNaive(arr, k)), "cross-check k=" + k);
        }

        var samples = new[]
        {
            new[] { 4, 2, 12, 3, 8, 7, 5, 9, 1, 6 },
            new[] { -1, -1, -2, -3 },
            new[] { 1, 1, 1, 1 },
            new[] { 10, -10, 5, -5, 0 },
            new[] { 7, 2 },
        };
        var ks = new[] { 4, 2, 3, 5, 2 };
        for (int s = 0; s < samples.Length; s++)
        {
            Check(Same(MaxSlidingWindowDeque(samples[s], ks[s]), MaxSlidingWindowNaive(samples[s], ks[s])), "cross-check sample " + s);
        }
        Console.WriteLine("PASS");
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
