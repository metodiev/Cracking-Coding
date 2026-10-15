// Task 481 - Implement optimistic checkpointing/restart simulation. (Tier: Advanced, CTCI Ch 14)
//
// Model (documented semantics):
// - A job applies one effect per step, step ids 0..n-1. Effects are idempotent: replaying a step
//   never duplicates effect id `step` (Apply is a set insert).
// - Optimistic checkpointing: after every `interval` completed steps (progress % interval == 0)
//   the job snapshots (progress, effects) into a single checkpoint slot, overwriting the old one.
// - A crash when "about to start step k" means steps 0..k-1 executed and the checkpoint slot is
//   the last one taken among them. Recovery rolls back to that slot: effects of steps that were
//   never checkpointed are lost (optimistic, uncommitted), while everything at or before the last
//   checkpoint survives: no committed effect is ever lost.
// - Recovered progress is exactly (k / interval) * interval, so redo work is < interval steps.
using System;
using System.Collections.Generic;

internal class OptimisticJob
{
    private readonly int n;
    private readonly int interval;
    public int Progress;
    public readonly List<int> Effects = new List<int>();
    public int CheckpointProgress;
    public List<int> CheckpointEffects = new List<int>();
    public int CheckpointCount;
    public int Executions;

    public OptimisticJob(int n, int interval)
    {
        this.n = n;
        this.interval = interval;
    }

    public void Apply(int step)
    {
        Executions++;
        if (!Effects.Contains(step)) Effects.Add(step);
    }

    // Execute steps 0..k-1, then crash while trying to start step k.
    public void RunUntil(int k)
    {
        while (Progress < k)
        {
            int step = Progress;
            Apply(step);
            Progress = step + 1;
            if (Progress % interval == 0)
            {
                CheckpointProgress = Progress;
                CheckpointEffects = new List<int>(Effects);
                CheckpointCount++;
            }
        }
    }

    public void Recover()
    {
        Progress = CheckpointProgress;
        Effects.Clear();
        Effects.AddRange(CheckpointEffects);
    }

    public void Finish() => RunUntil(n);

    public int Redone() => Executions - Effects.Count;
}

public static class Solution
{
    private static int seed = 481;

    private static int[] CrashPoints(int count, int limit)
    {
        var points = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            points[i] = seed % limit;
        }
        return points;
    }

    private static List<int> EffectsSorted(OptimisticJob job)
    {
        var sorted = new List<int>(job.Effects);
        sorted.Sort();
        return sorted;
    }

    private static List<int> RangeList(int n)
    {
        var list = new List<int>();
        for (int i = 0; i < n; i++) list.Add(i);
        return list;
    }

    private static OptimisticJob VerifyCrash(int n, int interval, int k)
    {
        var job = new OptimisticJob(n, interval);
        job.RunUntil(k);
        Check(job.Progress == k, "progress at crash " + k);
        job.Recover();
        int expected = (k / interval) * interval;
        Check(job.Progress == expected, "recovered progress " + k);
        Check(job.Progress <= k, "recovered <= crash " + k);
        Check(k - job.Progress < interval, "redo bound " + k);
        Check(SameList(EffectsSorted(job), RangeList(expected)), "committed effects survive " + k);
        job.Finish();
        Check(job.Progress == n, "finish progress");
        Check(SameList(EffectsSorted(job), RangeList(n)), "final effect set");
        Check(job.Redone() == k - expected, "redone count " + k);
        return job;
    }

    private static bool SameList(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        var job = new OptimisticJob(25, 4);
        job.RunUntil(7);
        Check(job.Progress == 7 && SameList(EffectsSorted(job), RangeList(7)), "crash state");
        job.Recover();
        Check(job.Progress == 4, "first recovery");
        Check(SameList(EffectsSorted(job), RangeList(4)), "first recovered effects");
        job.RunUntil(16);
        job.Recover();
        Check(job.Progress == 16, "second recovery");
        Check(SameList(EffectsSorted(job), RangeList(16)), "second recovered effects");
        job.RunUntil(23);
        job.Recover();
        Check(job.Progress == 20, "third recovery");
        Check(SameList(EffectsSorted(job), RangeList(20)), "third recovered effects");
        job.Finish();
        Check(job.Progress == 25, "final progress");
        Check(SameList(EffectsSorted(job), RangeList(25)), "final effects");
        Check(job.Executions == 7 + 12 + 7 + 5, "execution count");
        Check(job.Redone() == (7 - 4) + (23 - 20), "total redone");
        Check(job.CheckpointCount == 6, "checkpoint count");
        Check(job.CheckpointProgress == 24 && job.CheckpointEffects.Count == 24, "last checkpoint");

        foreach (int k in new[] { 0, 3, 7, 8, 16, 23 })
        {
            Check(VerifyCrash(25, 4, k).Progress == 25, "scripted finish " + k);
        }
        var expectations = new[] { new[] { 3, 0 }, new[] { 7, 4 }, new[] { 8, 8 }, new[] { 16, 16 }, new[] { 23, 20 } };
        foreach (var pair in expectations)
        {
            Check((pair[0] / 4) * 4 == pair[1], "boundary expectation " + pair[0]);
        }

        int checkedCount = 0;
        foreach (int k in CrashPoints(20, 26))
        {
            int expected = (k / 4) * 4;
            Check(VerifyCrash(25, 4, k).Progress == 25, "lcg finish " + k);
            Check(expected == (k / 4) * 4, "lcg expectation " + k);
            checkedCount++;
        }
        Check(checkedCount == 20, "lcg crash count");

        foreach (int interval in new[] { 1, 3, 5 })
        {
            for (int k = 0; k <= 20; k++)
            {
                var verified = VerifyCrash(20, interval, k);
                Check(verified.Redone() == k - (k / interval) * interval, "redo " + interval + " " + k);
                Check(verified.Redone() < interval, "redo bound " + interval);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
