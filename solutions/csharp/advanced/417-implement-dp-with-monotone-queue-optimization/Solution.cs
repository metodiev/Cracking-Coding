// Task 417 - Implement DP with monotone queue optimization. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Sliding-window minimum DP: jump from j to i at cost a[i], landing windows of size k:
//     dp[0] = a[0],   dp[i] = a[i] + min(dp[j] for j in [max(0, i - k), i - 1]).
// The naive transition scans k predecessors (O(n*k) overall). A monotone deque keeps the
// candidate indices with STRICTLY increasing dp values: expired indices (j < i - k) leave
// from the front, and while adding i every index with dp >= dp[i] is popped from the back,
// because it is both older and worse. Every step is amortized O(1).
// Tie-break: newer index wins (the pop condition keeps equal-valued indices newest-first),
// which the tests only use through value equality.
using System;

public static class Solution
{
    /// <summary>Returns (dp, pred); pred[i] is the j achieving the window minimum.</summary>
    public static (int[] Dp, int[] Pred) SlidingMinDp(int[] a, int k)
    {
        int n = a.Length;
        var dp = new int[n];
        var pred = new int[n];
        if (n == 0) return (dp, pred);
        dp[0] = a[0];
        pred[0] = -1;
        var dq = new int[n + 1];
        int head = 0, tail = 0;
        dq[tail++] = 0;
        for (int i = 1; i < n; i++)
        {
            while (dq[head] < i - k) head++;
            int j = dq[head];
            dp[i] = a[i] + dp[j];
            pred[i] = j;
            while (tail > head && dp[dq[tail - 1]] >= dp[i]) tail--;
            dq[tail++] = i;
        }
        return (dp, pred);
    }

    /// <summary>Oracle: O(n*k) direct evaluation of the recurrence.</summary>
    public static int[] SlidingMinDpNaive(int[] a, int k)
    {
        int n = a.Length;
        if (n == 0) return Array.Empty<int>();
        var dp = new int[n];
        dp[0] = a[0];
        for (int i = 1; i < n; i++)
        {
            int best = int.MaxValue;
            for (int j = Math.Max(0, i - k); j < i; j++) best = Math.Min(best, dp[j]);
            dp[i] = a[i] + best;
        }
        return dp;
    }

    private static int seed = 417;

    private static int[] LcgArray(int length, int lo, int hi)
    {
        var outp = new int[length];
        for (int k = 0; k < length; k++)
        {
            seed = (seed * 75 + 74) % 65537;
            outp[k] = lo + seed % (hi - lo + 1);
        }
        return outp;
    }

    private static bool Same(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static int WindowMin(int[] dp, int from, int to)
    {
        int best = int.MaxValue;
        for (int j = from; j < to; j++) best = Math.Min(best, dp[j]);
        return best;
    }

    public static void Main()
    {
        // hand-derived values, a = [10, 1, 5, 7, 3, 2]
        int[] a = { 10, 1, 5, 7, 3, 2 };
        var r2 = SlidingMinDp(a, 2);
        Check(Same(r2.Dp, new[] { 10, 11, 15, 18, 18, 20 }), "dp k2");
        // at i = 4 the equal-valued index 3 is dropped (newer ties win), so pred[5] = 4
        Check(Same(r2.Pred, new[] { -1, 0, 0, 1, 2, 4 }), "pred k2");
        Check(r2.Dp[4] == a[4] + Math.Min(r2.Dp[2], r2.Dp[3]), "dp4 window min");
        Check(r2.Dp[5] == a[5] + Math.Min(r2.Dp[3], r2.Dp[4]), "dp5 window min");

        var r3 = SlidingMinDp(a, 3);
        Check(Same(r3.Dp, new[] { 10, 11, 15, 17, 14, 16 }), "dp k3");
        var r1 = SlidingMinDp(a, 1);
        Check(Same(r1.Dp, new[] { 10, 11, 16, 23, 26, 28 }), "dp k1");
        Check(Same(r1.Pred, new[] { -1, 0, 1, 2, 3, 4 }), "pred k1");

        // k = 1 makes dp a running prefix sum of a
        int running = 0;
        for (int i = 0; i < a.Length; i++)
        {
            running += a[i];
            Check(r1.Dp[i] == running, "prefix sum " + i);
        }
        // k >= n opens the whole history
        var big = SlidingMinDp(a, 100);
        Check(big.Dp[0] == 10 && big.Dp[1] == 11 && big.Dp[2] == 15, "big window front");
        Check(big.Dp[3] == 17 && big.Dp[4] == 13 && big.Dp[5] == 12, "big window back");

        // every hand value also matches the naive oracle and the predecessor optimality
        for (int k = 1; k <= 7; k++)
        {
            var res = SlidingMinDp(a, k);
            Check(Same(res.Dp, SlidingMinDpNaive(a, k)), "oracle " + k);
            for (int i = 1; i < a.Length; i++)
            {
                Check(res.Pred[i] >= Math.Max(0, i - k) && res.Pred[i] < i, "pred range " + k + " " + i);
                Check(res.Dp[res.Pred[i]] == WindowMin(res.Dp, Math.Max(0, i - k), i), "pred optimal " + k + " " + i);
            }
        }

        // degenerate sizes
        Check(SlidingMinDp(Array.Empty<int>(), 5).Dp.Length == 0, "empty");
        var single = SlidingMinDp(new[] { 7 }, 3);
        Check(single.Dp.Length == 1 && single.Dp[0] == 7 && single.Pred[0] == -1, "single");
        Check(SlidingMinDpNaive(Array.Empty<int>(), 5).Length == 0, "empty naive");
        Check(SlidingMinDp(new[] { -4, -8 }, 2).Dp[1] == -12, "negative");

        for (int round = 0; round < 8; round++)
        {
            int n = 120 + seed % 180;
            var values = LcgArray(n, -50, 50);
            int k = 1 + seed % 15;
            var res = SlidingMinDp(values, k);
            var dp = res.Dp;
            var pred = res.Pred;
            Check(Same(dp, SlidingMinDpNaive(values, k)), "lcg oracle " + round);
            for (int i = 1; i < n; i++)
            {
                int j = pred[i];
                Check(j >= Math.Max(0, i - k) && j < i, "lcg pred range " + round);
                Check(dp[i] == values[i] + dp[j], "lcg recurrence " + round);
                Check(dp[j] == WindowMin(dp, Math.Max(0, i - k), i), "lcg pred optimal " + round);
            }
            // monotone deque invariant: candidate values strictly increase front to back
            var dq = new int[n + 1];
            int head = 0, tail = 0;
            dq[tail++] = 0;
            for (int i = 1; i < n; i++)
            {
                while (dq[head] < i - k) head++;
                Check(dp[dq[head]] == WindowMin(dp, Math.Max(0, i - k), i), "deque front " + round + " " + i);
                while (tail > head && dp[dq[tail - 1]] >= dp[i]) tail--;
                dq[tail++] = i;
            }
        }

        // window bound: widening k never increases any dp value (only relaxes the minimum)
        var seq = LcgArray(150, 0, 30);
        var previous = SlidingMinDpNaive(seq, 1);
        for (int k = 2; k <= 10; k++)
        {
            var current = SlidingMinDpNaive(seq, k);
            for (int i = 0; i < seq.Length; i++)
            {
                Check(current[i] <= previous[i], "monotone k " + k + " i " + i);
            }
            previous = current;
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
