// Task 352 - Implement fork-join style parallel sum (threads). (Tier: Intermediate, CTCI Ch 15)
using System;
using System.Collections.Generic;
using System.Threading;

public static class Solution
{
    /// <summary>Leaf ranges hold at most THRESHOLD elements; larger ranges split in half.</summary>
    private const int Threshold = 256;

    private class Stats
    {
        private readonly object gate = new object();
        private int leaves;
        private int spawned;

        public void CountLeaf() { lock (gate) { leaves++; } }
        public void CountSpawn() { lock (gate) { spawned++; } }
        public int Leaves { get { lock (gate) { return leaves; } } }
        public int Spawned { get { lock (gate) { return spawned; } } }
    }

    /// <summary>SumRange returns the sum and records leaf/spawn accounting in stats.</summary>
    private static long SumRange(long[] values, int lo, int hi, Stats stats, int timeoutMillis)
    {
        if (hi - lo <= Threshold)                     // leaf task: sum without splitting
        {
            stats.CountLeaf();
            long sum = 0;
            for (int i = lo; i < hi; i++) sum += values[i];
            return sum;
        }

        int mid = (lo + hi) / 2;
        long[] forked = new long[1];
        stats.CountSpawn();
        var thread = new Thread(() => { forked[0] = SumRange(values, lo, mid, stats, timeoutMillis); });
        thread.Start();
        long right = SumRange(values, mid, hi, stats, timeoutMillis);   // one half inline
        thread.Join(timeoutMillis);                                     // join the fork
        Check(!thread.IsAlive, "forked worker did not finish in time");
        return forked[0] + right;
    }

    /// <summary>Independent (iterative) oracle for the number of leaves the split policy creates.</summary>
    private static int CountLeaves(int size)
    {
        int leaves = 0;
        var pending = new Stack<int[]>();
        pending.Push(new[] { 0, size });
        while (pending.Count > 0)
        {
            var range = pending.Pop();
            if (range[1] - range[0] <= Threshold)
            {
                leaves++;
            }
            else
            {
                int mid = (range[0] + range[1]) / 2;
                pending.Push(new[] { range[0], mid });
                pending.Push(new[] { mid, range[1] });
            }
        }
        return leaves;
    }

    private static long[] LcgArray(int size)
    {
        var values = new long[size];
        long seed = 24680;
        for (int i = 0; i < size; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed;
        }
        return values;
    }

    private static long SequentialSum(long[] values)
    {
        long total = 0;
        foreach (long value in values) total += value;
        return total;
    }

    public static void Main()
    {
        var emptyStats = new Stats();
        Check(SumRange(new long[0], 0, 0, emptyStats, 10000) == 0, "empty array sums to 0");
        Check(emptyStats.Leaves == 1 && emptyStats.Spawned == 0, "empty range is a single leaf");

        int[] sizes = { 1, 7, Threshold, Threshold + 1, 2 * Threshold, 3 * Threshold,
                        1024, 2048, 4096, 8192 };
        foreach (int size in sizes)
        {
            var values = LcgArray(size);
            var stats = new Stats();
            long total = SumRange(values, 0, size, stats, 10000);
            Check(total == SequentialSum(values), "parallel sum mismatch at size " + size);
            Check(stats.Leaves == CountLeaves(size), "leaf count at size " + size);
            Check(stats.Spawned == stats.Leaves - 1, "binary fork tree at size " + size);
        }

        // Documented threshold accounting for power-of-two sizes: leaves == n / THRESHOLD.
        int[] powers = { 512, 1024, 2048, 4096, 8192 };
        foreach (int size in powers)
        {
            var values = LcgArray(size);
            var stats = new Stats();
            Check(SumRange(values, 0, size, stats, 10000) == SequentialSum(values),
                  "power-of-two sum at " + size);
            Check(stats.Leaves == size / Threshold, "power-of-two leaves at " + size);
            Check(stats.Spawned == size / Threshold - 1, "power-of-two threads at " + size);
        }

        // Splitting really happens above the threshold, and 8192 splits into exactly 31 forks.
        var atThreshold = new Stats();
        SumRange(LcgArray(Threshold), 0, Threshold, atThreshold, 10000);
        Check(atThreshold.Spawned == 0, "at threshold: no fork");
        var justAbove = new Stats();
        SumRange(LcgArray(Threshold + 1), 0, Threshold + 1, justAbove, 10000);
        Check(justAbove.Spawned == 1, "just above: one fork");
        var big = new Stats();
        SumRange(LcgArray(8192), 0, 8192, big, 10000);
        Check(big.Spawned == 31, "8192 -> 32 leaves, 31 forks");

        // LCG-driven sizes across the whole range, each cross-checked against the sequential sum.
        long seed = 9753;
        for (int trial = 0; trial < 20; trial++)
        {
            seed = (seed * 75 + 74) % 65537;
            int size = (int)(seed % 3000);
            var values = LcgArray(size);
            var stats = new Stats();
            long total = SumRange(values, 0, size, stats, 10000);
            Check(total == SequentialSum(values), "LCG size " + size);
            Check(stats.Leaves == CountLeaves(size), "LCG leaves at " + size);
            Check(stats.Spawned == stats.Leaves - 1, "LCG fork tree at " + size);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
