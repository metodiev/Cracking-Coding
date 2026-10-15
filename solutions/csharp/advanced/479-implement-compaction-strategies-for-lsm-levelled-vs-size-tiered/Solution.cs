// Task 479 - Implement compaction strategies for LSM (levelled vs size-tiered). (Tier: Advanced, CTCI Ch 14 — Databases)
using System;
using System.Collections.Generic;

internal class SizeTieredEngine
{
    private const int MergeFanout = 4;
    private const int Factor = 2;

    public List<Dictionary<int, int>> Runs = new List<Dictionary<int, int>>(); // chronological
    public int TotalWritten;
    public int MergeCount;
    public int PeakRuns;
    public int PeakEntries;

    public void Flush(Dictionary<int, int> memtable)
    {
        Runs.Add(new Dictionary<int, int>(memtable));
        TotalWritten += memtable.Count;
        Compact();
        PeakRuns = Math.Max(PeakRuns, Runs.Count);
        int entries = 0;
        foreach (var run in Runs) entries += run.Count;
        PeakEntries = Math.Max(PeakEntries, entries);
    }

    private void Compact()
    {
        while (true)
        {
            var sizes = new List<int[]>();
            for (int i = 0; i < Runs.Count; i++) sizes.Add(new[] { Runs[i].Count, i });
            sizes.Sort((a, b) => a[0] != b[0] ? a[0].CompareTo(b[0]) : a[1].CompareTo(b[1]));
            List<int> target = null;
            for (int i = 0; i + 3 < sizes.Count; i++)
            {
                if (sizes[i + 3][0] <= Factor * sizes[i][0])
                {
                    target = new List<int>();
                    for (int j = i; j < i + MergeFanout; j++) target.Add(sizes[j][1]);
                    target.Sort();
                    break;
                }
            }
            if (target == null) return;
            var merged = new Dictionary<int, int>();
            foreach (int idx in target)                  // chronological: newer wins
            {
                foreach (var pair in Runs[idx]) merged[pair.Key] = pair.Value;
            }
            TotalWritten += merged.Count;
            MergeCount++;
            var newRuns = new List<Dictionary<int, int>>();
            bool inserted = false;
            for (int i = 0; i < Runs.Count; i++)
            {
                if (target.Contains(i))
                {
                    if (!inserted)
                    {
                        newRuns.Add(merged);
                        inserted = true;
                    }
                }
                else
                {
                    newRuns.Add(Runs[i]);
                }
            }
            Runs = newRuns;
        }
    }

    public Dictionary<int, int> Materialize()
    {
        var output = new Dictionary<int, int>();
        foreach (var run in Runs)
        {
            foreach (var pair in run) output[pair.Key] = pair.Value;
        }
        return output;
    }
}

internal class LevelledEngine
{
    public List<List<Dictionary<int, int>>> Levels = new List<List<Dictionary<int, int>>>();
    public int TotalWritten;
    public int MergeCount;
    public int PeakRuns;
    public int PeakEntries;

    private void EnsureLevel(int i)
    {
        while (Levels.Count <= i) Levels.Add(new List<Dictionary<int, int>>());
    }

    public void Flush(Dictionary<int, int> memtable)
    {
        EnsureLevel(0);
        Levels[0].Add(new Dictionary<int, int>(memtable));
        TotalWritten += memtable.Count;
        Compact();
        int runs = 0;
        int entries = 0;
        foreach (var level in Levels)
        {
            runs += level.Count;
            foreach (var run in level) entries += run.Count;
        }
        PeakRuns = Math.Max(PeakRuns, runs);
        PeakEntries = Math.Max(PeakEntries, entries);
    }

    private void Compact()
    {
        while (true)
        {
            bool merged = false;
            for (int i = 0; i < Levels.Count; i++)
            {
                if (Levels[i].Count >= 2)
                {
                    var combined = new Dictionary<int, int>(Levels[i][0]); // older
                    foreach (var pair in Levels[i][1]) combined[pair.Key] = pair.Value; // newer wins
                    TotalWritten += combined.Count;
                    MergeCount++;
                    Levels[i].RemoveRange(0, 2);
                    EnsureLevel(i + 1);
                    Levels[i + 1].Add(combined);
                    merged = true;
                    break;
                }
            }
            if (!merged) return;
        }
    }

    public Dictionary<int, int> Materialize()
    {
        var output = new Dictionary<int, int>();
        for (int i = Levels.Count - 1; i >= 0; i--) // higher levels hold older data
        {
            foreach (var run in Levels[i])
            {
                foreach (var pair in run) output[pair.Key] = pair.Value;
            }
        }
        return output;
    }
}

public static class Solution
{
    private const int Keys = 256;
    private const int Ops = 128;
    private const int FlushEvery = 8;

    private static int Lcg(int seed) => (seed * 75 + 74) % 65537;

    private static int[][] GenWrites(int count, int seed, int keyspace)
    {
        var ops = new int[count][];
        for (int i = 0; i < count; i++)
        {
            seed = Lcg(seed);
            int key = seed % keyspace;
            seed = Lcg(seed);
            int value = seed % 1000;
            ops[i] = new[] { key, value };
        }
        return ops;
    }

    private static bool SameMap(Dictionary<int, int> a, Dictionary<int, int> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var pair in a)
        {
            if (!b.TryGetValue(pair.Key, out int other) || other != pair.Value) return false;
        }
        return true;
    }

    public static void Main()
    {
        int[][] ops = GenWrites(Ops, 24680, Keys);
        var oracle = new Dictionary<int, int>();
        var st = new SizeTieredEngine();
        var lv = new LevelledEngine();
        for (int start = 0; start < Ops; start += FlushEvery)
        {
            var memtable = new Dictionary<int, int>();
            for (int i = start; i < start + FlushEvery; i++)
            {
                memtable[ops[i][0]] = ops[i][1];
                oracle[ops[i][0]] = ops[i][1];
            }
            st.Flush(memtable);
            lv.Flush(memtable);
            Check(SameMap(st.Materialize(), oracle), "size-tiered lost a key at " + start);
            Check(SameMap(lv.Materialize(), oracle), "levelled lost a key at " + start);
        }
        Check(oracle.Count == 107, "live keys hand-counted to 107");
        Check(SameMap(st.Materialize(), oracle) && SameMap(lv.Materialize(), oracle), "final materialized states");

        Check(st.TotalWritten == 357, "size-tiered entry writes");
        Check(st.MergeCount == 5, "size-tiered compactions");
        Check(st.PeakRuns == 6, "size-tiered peak runs");
        Check(st.PeakEntries == 115, "size-tiered peak entries");
        Check(st.Runs.Count == 1 && st.Runs[0].Count == 107, "size-tiered final run");

        Check(lv.TotalWritten == 600, "levelled entry writes");
        Check(lv.MergeCount == 15, "levelled compactions (8 + 4 + 2 + 1)");
        Check(lv.PeakRuns == 4, "levelled peak runs");
        Check(lv.PeakEntries == 112, "levelled peak entries");
        Check(lv.Levels.Count == 5, "levelled depth");
        for (int i = 0; i < lv.Levels.Count - 1; i++) Check(lv.Levels[i].Count == 0, "lower levels drained");
        Check(lv.Levels[4].Count == 1 && lv.Levels[4][0].Count == 107, "levelled final run");

        // Documented qualitative outcomes for this parameterisation.
        Check(lv.TotalWritten > st.TotalWritten, "levelled rewrites more entries (write amplification)");
        Check(lv.MergeCount > st.MergeCount, "levelled compacts more often, in smaller steps");
        Check(st.PeakRuns > lv.PeakRuns, "size-tiered keeps more runs alive (read amplification)");
        Check(st.PeakEntries >= lv.PeakEntries, "size-tiered temporarily stores more copies");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
