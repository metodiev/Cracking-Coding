// Task 330 - Weighted interval scheduling (DP). (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537.</summary>
internal sealed class Lcg
{
    private int seed;

    public Lcg(int seed) => this.seed = seed;

    public int Next(int modulus)
    {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

public static class Solution
{
    /// <summary>
    /// Jobs are {start, finish, weight}; compatible when one finishes at or before the
    /// next starts (finish &lt;= start). Returns (best total weight, chosen jobs).
    /// </summary>
    public static (int Total, List<int[]> Chosen) WeightedIntervalScheduling(int[][] jobs)
    {
        if (jobs.Length == 0) return (0, new List<int[]>());
        var ordered = jobs.Select(job => (int[])job.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[1] != v[1] ? u[1].CompareTo(v[1]) : u[0].CompareTo(v[0]));
        int[] finishes = ordered.Select(job => job[1]).ToArray();

        int LastCompatible(int start, int i)
        {
            int lo = 0, hi = i - 1, best = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if (finishes[mid] <= start)
                {
                    best = mid;
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
            return best;
        }

        var best = new int[ordered.Length + 1];
        var choose = new bool[ordered.Length + 1];
        for (int i = 1; i <= ordered.Length; i++)
        {
            int without = best[i - 1];
            int withJob = ordered[i - 1][2] + best[LastCompatible(ordered[i - 1][0], i - 1) + 1];
            if (withJob > without)
            {
                best[i] = withJob;
                choose[i] = true;
            }
            else
            {
                best[i] = without;
            }
        }

        var chosen = new List<int[]>();
        int index = ordered.Length;
        while (index > 0)
        {
            if (choose[index])
            {
                chosen.Add(ordered[index - 1]);
                index = LastCompatible(ordered[index - 1][0], index - 1) + 1;
            }
            else
            {
                index--;
            }
        }
        chosen.Reverse();
        return (best[ordered.Length], chosen);
    }

    /// <summary>Exhaustive subset enumeration; the independent oracle for small inputs.</summary>
    public static int BruteForce(int[][] jobs)
    {
        int n = jobs.Length;
        int best = 0;
        for (int mask = 0; mask < 1 << n; mask++)
        {
            var chosen = new List<int[]>();
            int total = 0;
            for (int i = 0; i < n; i++)
            {
                if ((mask >> i & 1) == 1)
                {
                    chosen.Add(jobs[i]);
                    total += jobs[i][2];
                }
            }
            if (total <= best) continue;
            chosen.Sort((u, v) => u[0] != v[0] ? u[0].CompareTo(v[0]) : u[1].CompareTo(v[1]));
            bool compatible = true;
            for (int k = 0; k + 1 < chosen.Count; k++)
            {
                if (chosen[k][1] > chosen[k + 1][0]) compatible = false;
            }
            if (compatible) best = total;
        }
        return best;
    }

    private static int[][] LcgJobs(int count, Lcg lcg)
    {
        var jobs = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int start = lcg.Next(30);
            jobs[i] = new[] { start, start + 1 + lcg.Next(8), 1 + lcg.Next(20) };
        }
        return jobs;
    }

    private static bool SameJobs(List<int[]> a, int[][] b) =>
        a.Count == b.Length && a.Zip(b).All(pair => pair.First.SequenceEqual(pair.Second));

    public static void Main()
    {
        // the job set from the task description: (1,3)+(4,6)+(6,7) = 5+5+4 = 14, not 12
        var briefJobs = new[] { new[] { 1, 3, 5 }, new[] { 2, 5, 6 }, new[] { 4, 6, 5 }, new[] { 6, 7, 4 } };
        (int total, List<int[]> chosen) = WeightedIntervalScheduling(briefJobs);
        Check(total == 14, "brief job set is 14, not 12: 5+5+4");
        Check(SameJobs(chosen, new[] { new[] { 1, 3, 5 }, new[] { 4, 6, 5 }, new[] { 6, 7, 4 } }), "brief job set chosen jobs");
        Check(BruteForce(briefJobs) == 14, "brief job set brute force");

        // hand-solved example: (1,3,5)+(3,5,4)+(5,7,3) = 12; the heavy decoys lose
        var hand = new[] { new[] { 1, 3, 5 }, new[] { 3, 5, 4 }, new[] { 5, 7, 3 }, new[] { 1, 7, 10 }, new[] { 2, 3, 2 } };
        (total, chosen) = WeightedIntervalScheduling(hand);
        Check(total == 12, "hand example totals 12");
        Check(SameJobs(chosen, new[] { new[] { 1, 3, 5 }, new[] { 3, 5, 4 }, new[] { 5, 7, 3 } }), "hand example chosen jobs");
        Check(BruteForce(hand) == 12, "hand example brute force");

        (total, chosen) = WeightedIntervalScheduling(Array.Empty<int[]>());
        Check(total == 0 && chosen.Count == 0, "no jobs");
        (total, chosen) = WeightedIntervalScheduling(new[] { new[] { 4, 9, 7 } });
        Check(total == 7 && SameJobs(chosen, new[] { new[] { 4, 9, 7 } }), "single job");

        var lcg = new Lcg(57721);
        for (int trial = 0; trial < 60; trial++)
        {
            int count = 4 + lcg.Next(6);
            int[][] jobs = LcgJobs(count, lcg);
            (total, chosen) = WeightedIntervalScheduling(jobs);
            Check(total == BruteForce(jobs), "random brute force cross-check " + trial);
            Check(chosen.Sum(job => job[2]) == total, "chosen weights match total " + trial);
            for (int k = 0; k + 1 < chosen.Count; k++)
            {
                Check(chosen[k][1] <= chosen[k + 1][0], "chosen jobs are compatible " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
