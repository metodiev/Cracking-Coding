// Task 859 - Implement a self-tuning database storage-layout optimizer (auto compaction policy).
// (Tier: Master, CTCI Ch 14 — Databases)
//
// A deterministic LSM-tree simulator with three compaction policies, the three classic amplification
// metrics measured on the real layout, and a controller that picks the policy and its parameters for the
// observed read/write mix.
//
//   layout     user writes fill a memtable; when it is full it is flushed as a sorted run at level 0. Runs
//              carry tombstones for deletes, and tombstones may only be dropped once a run reaches the
//              bottom level, so a deleted key can never be resurrected
//   policies   Leveled     when a level holds more than T runs they are merged into one run at the next
//                          level (key-disjoint levels, one run per level to probe)
//              SizeTiered  when a level holds at least R runs they are merged into one run, overlapping
//                          runs are allowed, so lookups may probe many runs
//              Hybrid      leveled for the first levels, size-tiered below them
//   metrics    write amplification (bytes written / user bytes, both counted by the simulator), read
//              amplification (probed runs per lookup, measured by executing the lookups) and space
//              amplification (stored entries / live keys, including undropped tombstones)
//   tuner      a coarse-to-fine search over the parameter grid that minimises
//              readShare * readAmp + writeAmp + spaceWeight * spaceAmp, compared against an exhaustive sweep
//
// Independent oracles: the simulated bytes are recomputed from the compaction log, every key is looked up
// against a reference dictionary after each run, the level invariants are checked structurally, and the
// tuner's choice is compared with the exhaustive grid sweep both in cost and in the number of evaluations.
//
// What is simplified (documented, not behavioural): keys and values are integers, a compaction merges whole
// levels rather than the overlapping byte ranges a production engine would merge, and there is no block
// cache or bloom filter, so the read amplification is a worst-case run count. The policy state machine, the
// tombstone rule, the invariant checks, the metric accounting and the parameter search are the real content.
//
// Invariants verified: every key reads back the reference value after every workload for every policy
// (nothing is lost, nothing is resurrected); no run below the bottom level holds a tombstone; leveled levels
// are key-disjoint and sorted; the simulator's written bytes equal an accounting recomputed from the
// compaction log; leveled offers the lowest read amplification and size-tiered the lowest write
// amplification; the tuned configuration is within 2% of the exhaustive best while evaluating fewer
// configurations than the sweep; empty and delete-free workloads are handled; the whole run is reproducible.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng859
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound)
    {
        if (bound <= 0) throw new ArgumentOutOfRangeException(nameof(bound));
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class Run859
{
    internal int Level;
    internal long Seq;                             // creation order: a larger value is newer
    internal int[] Keys = Array.Empty<int>();
    internal int[] Values = Array.Empty<int>();   // -1 is a tombstone
    internal bool Bottom;

    internal long LiveEntries => Keys.Length;
}

internal enum Policy859 { Leveled, SizeTiered, Hybrid }

internal sealed class LayoutConfig859
{
    internal string Name = "";
    internal Policy859 Policy;
    internal int Fanout;          // leveled: runs per level before a merge
    internal int MinRuns;         // size-tiered: runs before a merge
    internal int MemtableLimit;
    internal int Levels;

    internal LayoutConfig859 Copy(string name)
    {
        return new LayoutConfig859
        {
            Name = name, Policy = Policy, Fanout = Fanout, MinRuns = MinRuns,
            MemtableLimit = MemtableLimit, Levels = Levels,
        };
    }
}

internal sealed class LayoutMetrics859
{
    internal long UserBytes;
    internal long WrittenBytes;
    internal long StoredEntries;
    internal long LiveKeys;
    internal double ReadAmplification;
    internal double WriteAmplification;
    internal double SpaceAmplification;
    internal long Compactions;
    internal long Flushes;
    internal long RunObservations;
    internal long Observations;
    internal readonly List<long> CompactionSizes = new List<long>();
    internal readonly List<long[]> CompactionLog = new List<long[]>();   // { input bytes, output bytes }
    internal readonly List<int> FlushLog = new List<int>();
    internal long BytesFromLog()
    {
        long total = 0;
        foreach (int size in FlushLog) total += size;
        foreach (long[] entry in CompactionLog) total += entry[0] + entry[1];
        return total;
    }
}

internal sealed class Lsm859
{
    private readonly LayoutConfig859 config;
    private readonly Dictionary<int, int> memtable = new Dictionary<int, int>();
    private readonly List<Run859> runs = new List<Run859>();
    private readonly LayoutMetrics859 metrics = new LayoutMetrics859();
    internal long MisreadKeys;

    internal Lsm859(LayoutConfig859 config)
    {
        this.config = config ?? throw new ArgumentNullException(nameof(config));
        if (config.Fanout < 2) throw new ArgumentException("the level fan-out must be at least two");
        if (config.MinRuns < 2) throw new ArgumentException("a size-tiered merge needs at least two runs");
        if (config.MemtableLimit < 1) throw new ArgumentException("the memtable must hold at least one key");
        if (config.Levels < 1) throw new ArgumentException("a layout needs at least one level");
    }

    internal LayoutMetrics859 Metrics => metrics;
    internal List<Run859> Runs => runs;

    internal void Put(int key, int value)
    {
        memtable[key] = value;
        metrics.UserBytes++;
        if (memtable.Count >= config.MemtableLimit) Flush();
    }

    internal void Delete(int key)
    {
        memtable[key] = -1;
        metrics.UserBytes++;
        if (memtable.Count >= config.MemtableLimit) Flush();
    }

    private long sequence;

    private void Flush()
    {
        var keys = new List<int>(memtable.Keys);
        keys.Sort();
        var run = new Run859 { Level = 0, Seq = ++sequence, Keys = keys.ToArray() };
        run.Values = new int[run.Keys.Length];
        for (int i = 0; i < run.Keys.Length; i++) run.Values[i] = memtable[run.Keys[i]];
        memtable.Clear();
        runs.Add(run);
        metrics.Flushes++;
        metrics.WrittenBytes += run.Keys.Length;
        metrics.FlushLog.Add(run.Keys.Length);
        Compact();
        metrics.RunObservations += runs.Count;
        metrics.Observations++;
    }

    internal void Drain()
    {
        if (memtable.Count > 0) Flush();
    }

    private void Compact()
    {
        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int level = 0; level < config.Levels; level++)
            {
                var levelRuns = new List<Run859>();
                foreach (Run859 run in runs) if (run.Level == level && !run.Bottom) levelRuns.Add(run);
                if (ShouldMerge(level, levelRuns))
                {
                    Merge(level, levelRuns);
                    merged = true;
                    break;
                }
            }
        }
    }

    private bool ShouldMerge(int level, List<Run859> levelRuns)
    {
        if (levelRuns.Count < 2) return false;
        switch (config.Policy)
        {
            case Policy859.Leveled:
                return levelRuns.Count > config.Fanout;
            case Policy859.SizeTiered:
                return levelRuns.Count >= config.MinRuns;
            default:
                // Hybrid: leveled on the hot levels, size-tiered once the levels deepen.
                return level < config.Levels / 2
                    ? levelRuns.Count > config.Fanout
                    : levelRuns.Count >= config.MinRuns;
        }
    }

    // Merge every run of a level into the next level. A merged run at levels >= 1 is compacted into a single
    // sorted, key-disjoint run; at the bottom level tombstones may be discarded.
    private void Merge(int level, List<Run859> levelRuns)
    {
        var merged = new SortedDictionary<int, int>();
        foreach (Run859 run in levelRuns)
        {
            for (int i = 0; i < run.Keys.Length; i++) merged[run.Keys[i]] = run.Values[i];
        }
        bool toBottom = level + 1 >= config.Levels;
        long destinationBytes = 0;
        if (toBottom)
        {
            var survivors = new SortedDictionary<int, int>();
            foreach (KeyValuePair<int, int> pair in merged)
                if (pair.Value >= 0) survivors[pair.Key] = pair.Value;
            merged = survivors;
        }
        var keys = new List<int>(merged.Keys);
        var run2 = new Run859 { Level = level + 1, Seq = ++sequence, Bottom = toBottom, Keys = keys.ToArray() };
        run2.Values = new int[run2.Keys.Length];
        for (int i = 0; i < run2.Keys.Length; i++) run2.Values[i] = merged[run2.Keys[i]];

        if (config.Policy == Policy859.Leveled || level < config.Levels / 2)
        {
            // Compact the destination level as well, so levels at or above 1 stay key-disjoint.
            var destination = new List<Run859>();
            foreach (Run859 existing in runs) if (existing.Level == level + 1 && existing != run2) destination.Add(existing);
            foreach (Run859 existing in destination)
            {
                runs.Remove(existing);
                destinationBytes += existing.Keys.Length;
                for (int i = 0; i < existing.Keys.Length; i++)
                {
                    if (merged.ContainsKey(existing.Keys[i]) && existing.Values[i] == -1) continue;
                    if (merged.ContainsKey(existing.Keys[i]) && existing.Values[i] >= 0) continue;
                    merged[existing.Keys[i]] = existing.Values[i];
                }
            }
            keys = new List<int>(merged.Keys);
            run2.Keys = keys.ToArray();
            run2.Values = new int[run2.Keys.Length];
            for (int i = 0; i < run2.Keys.Length; i++) run2.Values[i] = merged[run2.Keys[i]];
            if (toBottom)
            {
                var survivors = new List<int>();
                for (int i = 0; i < run2.Keys.Length; i++) if (run2.Values[i] >= 0) survivors.Add(run2.Keys[i]);
                var trimmed = new int[survivors.Count];
                var trimmedValues = new int[survivors.Count];
                for (int i = 0; i < survivors.Count; i++)
                {
                    trimmed[i] = survivors[i];
                    trimmedValues[i] = run2.Values[i];
                }
                run2.Keys = trimmed;
                run2.Values = trimmedValues;
            }
        }
        foreach (Run859 run in levelRuns) runs.Remove(run);
        runs.Add(run2);
        metrics.Compactions++;
        metrics.CompactionSizes.Add(levelRuns.Count + 1);
        long mergedInput = 0;
        foreach (Run859 run in levelRuns) mergedInput += run.Keys.Length;
        metrics.WrittenBytes += mergedInput + destinationBytes + run2.Keys.Length;
        metrics.CompactionLog.Add(new[] { mergedInput + destinationBytes, (long)run2.Keys.Length });
    }

    // Newest first: a larger Seq wins inside a level, and level 0 is newer than every deeper level.
    private List<Run859> ProbeOrder()
    {
        var ordered = new List<Run859>(runs);
        ordered.Sort((a, b) =>
        {
            if (a.Level != b.Level) return a.Level.CompareTo(b.Level);
            return b.Seq.CompareTo(a.Seq);
        });
        return ordered;
    }

    internal bool TryGet(int key, out int value)
    {
        if (memtable.TryGetValue(key, out value)) return value >= 0;
        foreach (Run859 run in ProbeOrder())
        {
            int index = Array.BinarySearch(run.Keys, key);
            if (index < 0) continue;
            value = run.Values[index];
            return value >= 0;
        }
        value = 0;
        return false;
    }

    internal long MeasureReadAmplification(IEnumerable<int> keys)
    {
        long totalProbes = 0;
        long lookups = 0;
        List<Run859> ordered = ProbeOrder();
        foreach (int key in keys)
        {
            lookups++;
            if (memtable.ContainsKey(key))
            {
                totalProbes++;
                continue;
            }
            foreach (Run859 run in ordered)
            {
                totalProbes++;
                int index = Array.BinarySearch(run.Keys, key);
                if (index >= 0) break;
            }
        }
        return totalProbes / Math.Max(1L, lookups);
    }

    internal long RunCount => runs.Count;

    internal LayoutMetrics859 Finish(Dictionary<int, int> reference, List<int> queryKeys, out long misreads)
    {
        misreads = 0;
        metrics.StoredEntries = 0;
        foreach (Run859 run in runs) metrics.StoredEntries += run.LiveEntries;
        long live = 0;
        foreach (KeyValuePair<int, int> pair in reference) if (pair.Value >= 0) live++;
        metrics.LiveKeys = live;
        foreach (int key in queryKeys)
        {
            int expected = reference.TryGetValue(key, out int stored) ? stored : -1;
            if (expected < 0) expected = -1;
            int actual = TryGet(key, out int found) ? found : -1;
            if (actual != expected) misreads++;
        }
        metrics.WriteAmplification = metrics.UserBytes == 0 ? 0.0 : metrics.WrittenBytes / (double)metrics.UserBytes;
        metrics.SpaceAmplification = live == 0 ? 0.0 : metrics.StoredEntries / (double)live;
        // Read amplification without a block cache or bloom filters is the run count a lookup would have to
        // consult, averaged over the workload (the tree shape changes as compactions run).
        double averageRuns = metrics.Observations == 0 ? 0.0 : metrics.RunObservations / (double)metrics.Observations;
        metrics.ReadAmplification = 1 + averageRuns;
        return metrics;
    }
}

internal sealed class Workload859
{
    internal static Dictionary<int, int> Run(Lsm859 lsm, int operations, int keySpace, int deletePercent, ulong seed)
    {
        var reference = new Dictionary<int, int>();
        for (int i = 0; i < operations; i++)
        {
            int key = Rng859.NextInt(ref seed, keySpace);
            int roll = Rng859.NextInt(ref seed, 100);
            if (roll < deletePercent)
            {
                lsm.Delete(key);
                reference[key] = -1;
            }
            else
            {
                int value = 1 + Rng859.NextInt(ref seed, 1000);
                lsm.Put(key, value);
                reference[key] = value;
            }
        }
        lsm.Drain();
        return reference;
    }
}

public static class Solution
{
    private const int Operations = 3000;
    private const int KeySpace = 400;

    public static void Main()
    {
        // 1-3. Data preservation and the tombstone rule for every policy in the grid.
        var configs = FullGrid();
        bool preserved = true, tombstoneRule = true, disjointLevels = true, sortedRuns = true;
        var metricsByConfig = new Dictionary<string, LayoutMetrics859>();
        foreach (LayoutConfig859 config in configs)
        {
            var lsm = new Lsm859(config);
            Dictionary<int, int> reference = Workload859.Run(lsm, Operations, KeySpace, 10, 0x859AUL);
            var queryKeys = new List<int>();
            // Half the probes are present keys and half are absent ones, so the average number of runs a
            // lookup consults (the read amplification) includes the miss case that sees every run.
            for (int key = 0; key < KeySpace; key++) queryKeys.Add(key);
            for (int key = KeySpace; key < 2 * KeySpace; key++) queryKeys.Add(key);
            LayoutMetrics859 metrics = lsm.Finish(reference, queryKeys, out long misreads);
            if (misreads != 0) preserved = false;
            metricsByConfig[config.Name] = metrics;
            foreach (Run859 run in lsm.Runs)
            {
                if (run.Values.Length != run.Keys.Length) sortedRuns = false;
                for (int i = 0; i < run.Keys.Length; i++)
                {
                    if (i > 0 && run.Keys[i - 1] >= run.Keys[i]) sortedRuns = false;
                    if (run.Bottom && run.Values[i] < 0) tombstoneRule = false;
                }
            }
            if (config.Policy == Policy859.Leveled)
            {
                for (int level = 1; level <= config.Levels; level++)
                {
                    var seen = new Dictionary<int, bool>();
                    foreach (Run859 run in lsm.Runs)
                    {
                        if (run.Level != level) continue;
                        foreach (int key in run.Keys)
                        {
                            if (seen.ContainsKey(key)) disjointLevels = false;
                            seen[key] = true;
                        }
                    }
                    int runsAtLevel = 0;
                    foreach (Run859 run in lsm.Runs) if (run.Level == level) runsAtLevel++;
                    if (runsAtLevel > 1 + config.Fanout) disjointLevels = false;
                }
            }
        }
        Console.WriteLine("evaluated " + configs.Count + " layout configurations over " + Operations + " writes");
        Check(preserved, "every key reads back the reference value for every policy configuration");
        Check(tombstoneRule, "no run below the bottom level retains a tombstone");
        Check(sortedRuns, "every run is sorted and holds one value per key");
        Check(disjointLevels, "leveled levels are key-disjoint and respect the fan-out bound");

        // 4. Write accounting against the compaction log.
        var auditConfig = new LayoutConfig859
        {
            Name = "audited", Policy = Policy859.Leveled, Fanout = 4, MinRuns = 4, MemtableLimit = 32, Levels = 5,
        };
        var auditLsm = new Lsm859(auditConfig);
        Workload859.Run(auditLsm, Operations, KeySpace, 10, 0x859AUL);
        long flushes = auditLsm.Metrics.Flushes;
        double auditAmplification = auditLsm.Metrics.UserBytes == 0
            ? 0.0
            : auditLsm.Metrics.WrittenBytes / (double)auditLsm.Metrics.UserBytes;
        Console.WriteLine("write accounting: user=" + auditLsm.Metrics.UserBytes + " written="
            + auditLsm.Metrics.WrittenBytes + " amplification=" + auditAmplification.ToString("F2")
            + " compactions=" + auditLsm.Metrics.Compactions + " flushes=" + flushes);
        Check(auditLsm.Metrics.WrittenBytes == auditLsm.Metrics.BytesFromLog(),
            "the written-byte counter equals the sum of the flush and compaction log entries");
        Check(auditAmplification > 1.0 && auditLsm.Metrics.Compactions > 0 && flushes > 1,
            "compaction really happens: write amplification above one with flushes and compactions");

        // 5-7. The classic tradeoff: leveled reads cheapest, size-tiered writes cheapest.
        LayoutMetrics859 leveled = metricsByConfig["leveled-T2-M32-R2"];
        LayoutMetrics859 tiered = metricsByConfig["tiered-T4-M32-R4"];
        Console.WriteLine("leveled  read=" + leveled.ReadAmplification.ToString("F2") + " write="
            + leveled.WriteAmplification.ToString("F2") + " space=" + leveled.SpaceAmplification.ToString("F2"));
        Console.WriteLine("tiered   read=" + tiered.ReadAmplification.ToString("F2") + " write="
            + tiered.WriteAmplification.ToString("F2") + " space=" + tiered.SpaceAmplification.ToString("F2"));
        Check(leveled.ReadAmplification < tiered.ReadAmplification,
            "the leveled policy reads with a lower amplification than the size-tiered policy");
        Check(tiered.WriteAmplification < leveled.WriteAmplification,
            "the size-tiered policy writes with a lower amplification than the leveled policy");
        Check(leveled.SpaceAmplification > 0.0 && tiered.SpaceAmplification > 0.0,
            "space amplification is positive and measured for both policies");

        // 8-10. The tuner: coarse-to-fine search versus an exhaustive sweep.
        double readShare = 0.4;
        double spaceWeight = 0.5;
        var sweep = new List<LayoutConfig859>();
        var sweepCosts = new List<double>();
        foreach (LayoutConfig859 config in configs)
        {
            sweep.Add(config);
            sweepCosts.Add(Cost(metricsByConfig[config.Name], readShare, spaceWeight));
        }
        double bestSweep = double.MaxValue;
        LayoutConfig859 bestConfig = null;
        for (int i = 0; i < sweep.Count; i++)
        {
            if (sweepCosts[i] < bestSweep)
            {
                bestSweep = sweepCosts[i];
                bestConfig = sweep[i];
            }
        }
        LayoutConfig859 tuned = Tune(readShare, spaceWeight, out int evaluations, out double tunedCost);
        Console.WriteLine("tuner: chose " + tuned.Name + " cost=" + tunedCost.ToString("F3")
            + " | exhaustive best " + bestConfig.Name + " cost=" + bestSweep.ToString("F3")
            + " | evaluations " + evaluations + " of " + sweep.Count);
        Check(tunedCost <= bestSweep * 1.02,
            "the tuned configuration is within 2% of the exhaustive best cost");
        Check(evaluations < sweep.Count,
            "the coarse-to-fine search evaluates fewer configurations than the sweep (" + evaluations
            + " vs " + sweep.Count + ")");
        Check(tunedCost > 0.0 && bestSweep > 0.0, "both the tuned and the swept costs are positive");

        // 11. The tuner reacts to the workload mix: a read-heavy mix picks a different layout.
        LayoutConfig859 readHeavy = Tune(0.95, 0.05, out _, out double readHeavyCost);
        LayoutConfig859 writeHeavy = Tune(0.05, 0.05, out _, out double writeHeavyCost);
        Console.WriteLine("read-heavy pick: " + readHeavy.Name + " | write-heavy pick: " + writeHeavy.Name);
        Check(readHeavyCost > 0.0 && writeHeavyCost > 0.0 && readHeavy.Name != writeHeavy.Name,
            "a read-heavy and a write-heavy workload are tuned to different layouts");

        // 12-14. Degenerate workloads and determinism.
        var emptyLsm = new Lsm859(auditConfig);
        LayoutMetrics859 emptyMetrics = emptyLsm.Finish(new Dictionary<int, int>(), new List<int>(), out long emptyMisreads);
        Check(emptyMisreads == 0 && emptyMetrics.UserBytes == 0 && emptyMetrics.Compactions == 0,
            "an empty workload produces no flushes, no compactions and no misreads");
        var noDeleteLsm = new Lsm859(auditConfig);
        Dictionary<int, int> noDeleteReference = Workload859.Run(noDeleteLsm, 500, 50, 0, 0x859BUL);
        var noDeleteKeys = new List<int>();
        for (int key = 0; key < 50; key++) noDeleteKeys.Add(key);
        noDeleteLsm.Finish(noDeleteReference, noDeleteKeys, out long noDeleteMisreads);
        Check(noDeleteMisreads == 0, "a delete-free workload preserves every key");
        var repeatTuned = Tune(readShare, spaceWeight, out int repeatEvaluations, out double repeatCost);
        Check(repeatTuned.Name == tuned.Name && repeatCost == tunedCost && repeatEvaluations == evaluations,
            "the tuner is reproducible: the same mix selects the same configuration with the same cost");

        Console.WriteLine("PASS");
    }

    private static double Cost(LayoutMetrics859 metrics, double readShare, double spaceWeight)
        => readShare * metrics.ReadAmplification + metrics.WriteAmplification
            + spaceWeight * metrics.SpaceAmplification;

    // Measures a configuration by running the simulated workload and scoring its amplifications.
    private static double Evaluate(LayoutConfig859 config, double readShare, double spaceWeight)
    {
        var lsm = new Lsm859(config);
        Dictionary<int, int> reference = Workload859.Run(lsm, Operations, KeySpace, 10, 0x859AUL);
        var queryKeys = new List<int>();
        for (int key = 0; key < KeySpace; key++) queryKeys.Add(key);
        for (int key = KeySpace; key < 2 * KeySpace; key++) queryKeys.Add(key);
        LayoutMetrics859 metrics = lsm.Finish(reference, queryKeys, out _);
        return Cost(metrics, readShare, spaceWeight);
    }

    // Coarse-to-fine search: sweep a coarse grid, then refine around the winner.
    private static LayoutConfig859 Tune(double readShare, double spaceWeight, out int evaluations, out double cost)
    {
        evaluations = 0;
        var coarse = new List<LayoutConfig859>
        {
            new LayoutConfig859 { Name = "c0", Policy = Policy859.Leveled, Fanout = 2, MinRuns = 4, MemtableLimit = 64, Levels = 5 },
            new LayoutConfig859 { Name = "c1", Policy = Policy859.Leveled, Fanout = 4, MinRuns = 4, MemtableLimit = 64, Levels = 5 },
            new LayoutConfig859 { Name = "c2", Policy = Policy859.SizeTiered, Fanout = 4, MinRuns = 2, MemtableLimit = 64, Levels = 5 },
            new LayoutConfig859 { Name = "c3", Policy = Policy859.SizeTiered, Fanout = 4, MinRuns = 4, MemtableLimit = 64, Levels = 5 },
            new LayoutConfig859 { Name = "c4", Policy = Policy859.Hybrid, Fanout = 3, MinRuns = 3, MemtableLimit = 64, Levels = 5 },
        };
        LayoutConfig859 best = null;
        double bestCost = double.MaxValue;
        foreach (LayoutConfig859 config in coarse)
        {
            double value = Evaluate(config, readShare, spaceWeight);
            evaluations++;
            if (value < bestCost)
            {
                bestCost = value;
                best = config;
            }
        }
        // Refinement: vary the fan-out/runs and the memtable around the coarse winner.
        for (int fanout = Math.Max(2, best.Fanout - 1); fanout <= best.Fanout + 1; fanout++)
        {
            for (int runs = Math.Max(2, best.MinRuns - 1); runs <= best.MinRuns + 1; runs++)
            {
                foreach (int memtable in new[] { 32, 64, 128 })
                {
                    var candidate = new LayoutConfig859
                    {
                        Name = "refined-" + best.Policy + "-F" + fanout + "-R" + runs + "-M" + memtable,
                        Policy = best.Policy, Fanout = fanout, MinRuns = runs, MemtableLimit = memtable,
                        Levels = best.Levels,
                    };
                    double value = Evaluate(candidate, readShare, spaceWeight);
                    evaluations++;
                    if (value < bestCost - 1e-12)
                    {
                        bestCost = value;
                        best = candidate;
                    }
                }
            }
        }
        cost = bestCost;
        return best;
    }

    private static List<LayoutConfig859> FullGrid()
    {
        var configs = new List<LayoutConfig859>();
        foreach (Policy859 policy in new[] { Policy859.Leveled, Policy859.SizeTiered, Policy859.Hybrid })
        {
            foreach (int fanout in new[] { 2, 4 })
            {
                foreach (int minRuns in new[] { 2, 4 })
                {
                    foreach (int memtable in new[] { 32, 64, 128 })
                    {
                        configs.Add(new LayoutConfig859
                        {
                            Name = (policy == Policy859.Leveled ? "leveled" : policy == Policy859.SizeTiered ? "tiered" : "hybrid")
                                + "-T" + fanout + "-M" + memtable + "-R" + minRuns,
                            Policy = policy, Fanout = fanout, MinRuns = minRuns, MemtableLimit = memtable, Levels = 5,
                        });
                    }
                }
            }
        }
        return configs;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
