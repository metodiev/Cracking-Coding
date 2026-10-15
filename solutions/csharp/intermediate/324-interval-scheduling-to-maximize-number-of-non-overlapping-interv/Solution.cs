// Task 324 - Interval scheduling to maximize number of non-overlapping intervals (greedy). (Tier: Intermediate, CTCI Ch 1)
using System;
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
    /// <summary>Greedy earliest finish time; touching endpoints conflict (next start must be strictly after the last end).</summary>
    public static int MaxNonOverlapping(int[][] intervals)
    {
        var ordered = intervals.Select(iv => (int[])iv.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[1] != v[1] ? u[1].CompareTo(v[1]) : u[0].CompareTo(v[0]));
        int count = 0;
        int lastEnd = int.MinValue;
        bool hasLast = false;
        foreach (int[] interval in ordered)
        {
            if (!hasLast || interval[0] > lastEnd)
            {
                count++;
                lastEnd = interval[1];
                hasLast = true;
            }
        }
        return count;
    }

    /// <summary>Independent DP: best[i] = max(best[i-1], 1 + best[p(i)]) with p by binary search.</summary>
    public static int MaxNonOverlappingDp(int[][] intervals)
    {
        if (intervals.Length == 0) return 0;
        var ordered = intervals.Select(iv => (int[])iv.Clone()).ToArray();
        Array.Sort(ordered, (u, v) => u[1] != v[1] ? u[1].CompareTo(v[1]) : u[0].CompareTo(v[0]));
        int[] finishes = ordered.Select(iv => iv[1]).ToArray();
        var best = new int[ordered.Length + 1];
        for (int i = 1; i <= ordered.Length; i++)
        {
            int start = ordered[i - 1][0];
            int lo = 0, hi = i - 1; // last j < i with finishes[j-1] < start
            while (lo < hi)
            {
                int mid = (lo + hi + 1) / 2;
                if (finishes[mid - 1] < start) lo = mid;
                else hi = mid - 1;
            }
            int j = lo > 0 && finishes[lo - 1] < start ? lo : 0;
            best[i] = Math.Max(best[i - 1], 1 + best[j]);
        }
        return best[ordered.Length];
    }

    private static int[][] LcgIntervals(int count, Lcg lcg)
    {
        var intervals = new int[count][];
        for (int i = 0; i < count; i++)
        {
            int start = lcg.Next(20);
            intervals[i] = new[] { start, start + 1 + lcg.Next(6) };
        }
        return intervals;
    }

    public static void Main()
    {
        // classic example: [1,2] and [2,3] touch, so they conflict; best is 2
        Check(MaxNonOverlapping(new[] { new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 1, 3 } }) == 2, "classic greedy");
        Check(MaxNonOverlappingDp(new[] { new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 1, 3 } }) == 2, "classic dp");

        // chain with gaps: all four fit; the wide decoy is not worth taking
        var chain = new[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5, 6 }, new[] { 7, 8 }, new[] { 1, 8 } };
        Check(MaxNonOverlapping(chain) == 4, "chain greedy");
        Check(MaxNonOverlappingDp(chain) == 4, "chain dp");
        Check(MaxNonOverlapping(new[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 5, 6 }, new[] { 7, 8 } }) == 4, "plain chain");

        Check(MaxNonOverlapping(Array.Empty<int[]>()) == 0, "empty");
        Check(MaxNonOverlapping(new[] { new[] { 1, 5 } }) == 1, "single");
        Check(MaxNonOverlapping(new[] { new[] { 1, 3 }, new[] { 2, 4 }, new[] { 3, 5 }, new[] { 4, 6 } }) == 2, "overlap chain");
        Check(MaxNonOverlapping(new[] { new[] { 1, 2 }, new[] { 1, 2 }, new[] { 1, 2 } }) == 1, "duplicates");
        Check(MaxNonOverlapping(new[] { new[] { 1, 4 }, new[] { 4, 4 } }) == 1, "touching conflicts");

        var lcg = new Lcg(909);
        for (int trial = 0; trial < 400; trial++)
        {
            int count = 1 + lcg.Next(9);
            int[][] intervals = LcgIntervals(count, lcg);
            Check(MaxNonOverlapping(intervals) == MaxNonOverlappingDp(intervals), "random dp cross-check " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
