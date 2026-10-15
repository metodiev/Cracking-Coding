// Task 331 - Weighted job scheduling with reconstruction. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Returns (best total weight, chosen indices) for (start, end, weight) jobs.</summary>
    public static (int Value, int[] Chosen) WeightedSchedule(int[][] jobs)
    {
        int n = jobs.Length;
        if (n == 0) return (0, Array.Empty<int>());
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) => jobs[a][1] != jobs[b][1]
            ? jobs[a][1].CompareTo(jobs[b][1]) : jobs[a][0].CompareTo(jobs[b][0]));
        var ends = new int[n];
        for (int i = 0; i < n; i++) ends[i] = jobs[order[i]][1];

        // pred[i] = last job (in sorted order) finishing at or before job i starts.
        var pred = new int[n];
        for (int i = 0; i < n; i++)
        {
            int lo = 0, hi = i;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (ends[mid] <= jobs[order[i]][0]) lo = mid + 1;
                else hi = mid;
            }
            pred[i] = lo - 1;
        }

        var dp = new int[n + 1];
        for (int i = 1; i <= n; i++)
        {
            int take = jobs[order[i - 1]][2] + dp[pred[i - 1] + 1];
            dp[i] = Math.Max(dp[i - 1], take);
        }

        var chosen = new List<int>();
        int index = n;
        while (index > 0)
        {
            if (dp[index] > dp[index - 1])
            {
                chosen.Add(order[index - 1]);
                index = pred[index - 1] + 1;
            }
            else
            {
                index--;
            }
        }
        chosen.Reverse();
        return (dp[n], chosen.ToArray());
    }

    /// <summary>Exhaustive best weight over all subsets (n &lt;= 12).</summary>
    public static int BruteSchedule(int[][] jobs)
    {
        int n = jobs.Length;
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++)
        {
            int weight = 0;
            var picked = new List<int>();
            bool valid = true;
            for (int i = 0; i < n && valid; i++)
            {
                if (((mask >> i) & 1) == 0) continue;
                foreach (int j in picked)
                {
                    if (Overlaps(jobs[i], jobs[j])) { valid = false; break; }
                }
                if (valid) { picked.Add(i); weight += jobs[i][2]; }
            }
            if (valid) best = Math.Max(best, weight);
        }
        return best;
    }

    public static bool Overlaps(int[] a, int[] b)
    {
        return a[0] < b[1] && b[0] < a[1];
    }

    private static int seed = 9091;

    private static int[][] LcgJobs(int count)
    {
        var jobs = new int[count][];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            int start = seed % 30;
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % 8;
            seed = (seed * 75 + 74) % 65537;
            jobs[i] = new[] { start, start + length, 1 + seed % 20 };
        }
        return jobs;
    }

    private static string Join(int[] values)
    {
        return string.Join(",", values);
    }

    private static bool IsNonOverlapping(int[][] jobs, int[] indices)
    {
        for (int i = 0; i < indices.Length; i++)
            for (int j = i + 1; j < indices.Length; j++)
                if (Overlaps(jobs[indices[i]], jobs[indices[j]])) return false;
        return true;
    }

    public static void Main()
    {
        // Hand-checked: (1,3,w5) + (4,6,w5) + (6,7,w4) = 14.
        var hand = new[]
        {
            new[] { 1, 3, 5 }, new[] { 2, 5, 6 }, new[] { 4, 6, 5 }, new[] { 6, 7, 4 },
        };
        var result = WeightedSchedule(hand);
        Check(result.Value == 14 && Join(result.Chosen) == "0,2,3", "hand value and set");
        int sum = 0;
        foreach (int i in result.Chosen) sum += hand[i][2];
        Check(sum == result.Value && IsNonOverlapping(hand, result.Chosen), "hand feasibility");
        Check(result.Value == BruteSchedule(hand), "hand matches brute force");

        Check(WeightedSchedule(Array.Empty<int[]>()).Value == 0, "empty value");
        Check(Join(WeightedSchedule(new[] { new[] { 5, 9, 7 } }).Chosen) == "0", "single job");

        var touch = new[] { new[] { 0, 5, 3 }, new[] { 5, 10, 4 } };
        Check(Join(WeightedSchedule(touch).Chosen) == "0,1", "touching intervals chain");

        var heavy = new[] { new[] { 0, 10, 100 }, new[] { 0, 4, 3 }, new[] { 6, 10, 3 } };
        Check(Join(WeightedSchedule(heavy).Chosen) == "0", "one heavy job wins");

        for (int trial = 0; trial < 200; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 12;
            var jobs = LcgJobs(n);
            var schedule = WeightedSchedule(jobs);
            Check(schedule.Value == BruteSchedule(jobs), "random value " + trial);
            Check(IsNonOverlapping(jobs, schedule.Chosen), "random feasibility " + trial);
            int total = 0;
            foreach (int i in schedule.Chosen) total += jobs[i][2];
            Check(total == schedule.Value, "random weight sum " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
