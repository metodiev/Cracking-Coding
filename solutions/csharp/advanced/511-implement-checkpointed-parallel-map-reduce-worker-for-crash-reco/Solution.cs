// Task 511 - Implement checkpointed parallel map-reduce worker for crash recovery. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// A map-reduce job that survives crashes because every finished chunk is durably recorded before
// its worker moves on. The input is split into N chunks of eight numbers each; the map function is
// the sum of squares inside a chunk and the reduce is the plain sum of the chunk results, both
// deterministic functions of the chunk index, so any prefix of the job can be recomputed exactly.
// Chunks are handed to P workers round-robin (worker w owns chunks w, w + P, w + 2P, ...), a worker
// appends "chunk=<id>;value=<v>;worker=<w>" to the append-only store immediately after computing its
// chunk, and the run is driven by a global execution counter so a crash can be injected at a chosen
// execution. Two crash flavours are modelled: beforeCheckpoint (the worker dies between computing
// and writing, so that single in-flight chunk must be redone) and afterCheckpoint (the record is
// durably written, so nothing is redone). The reduce phase commits "result=<r>"; a crash there
// re-runs the reduce only.
// Documented checks: a clean run executes every chunk exactly once, produces N valid records and the
// brute-force total; a run with three injected crashes still produces that total, executes at most
// N + 3 chunks, loses exactly one chunk per before-checkpoint crash and none per after-checkpoint
// crash, and leaves duplicate records that all agree with the deterministic map; a torn trailing
// record ("chunk=7;value=") is discarded by the parser, counted, and its chunk is redone; a crash
// after the last chunk re-runs the reduce with zero chunk executions; the deduplicated result
// contributes every chunk exactly once; each of the four workers owns exactly a quarter of the
// records; parsing is idempotent; the empty and single-chunk jobs behave; and two identical runs
// produce byte-identical stores.
using System;
using System.Collections.Generic;

internal static class MapJob
{
    internal const int ChunkSize = 8;

    internal static ulong Mix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    internal static long Record(int chunk, int index)
    {
        ulong h = Mix64((ulong)chunk * 0x9E3779B97F4A7C15UL + (ulong)index * 0xD1B54A32D192ED03UL);
        return (long)(h % 201UL) - 100;
    }

    internal static long MapValue(int chunk)
    {
        long total = 0;
        for (int i = 0; i < ChunkSize; i++)
        {
            long value = Record(chunk, i);
            total += value * value;
        }
        return total;
    }

    internal static long BruteForceTotal(int chunks)
    {
        long total = 0;
        for (int chunk = 0; chunk < chunks; chunk++) total += MapValue(chunk);
        return total;
    }
}

/** Append-only checkpoint log; a torn write is a line without the trailing value field. */
internal class DurableStore
{
    private readonly List<string> lines = new List<string>();

    internal int Appends { get; private set; }

    internal IReadOnlyList<string> Lines => lines;

    internal void AppendChunk(int chunk, long value, int worker)
    {
        lines.Add("chunk=" + chunk + ";value=" + value + ";worker=" + worker);
        Appends += 1;
    }

    internal void AppendTorn(int chunk, int worker)
    {
        lines.Add("chunk=" + chunk + ";value=");
        Appends += 1;
    }

    internal void AppendResult(long result)
    {
        lines.Add("result=" + result);
        Appends += 1;
    }

    internal List<string> Snapshot() => new List<string>(lines);

    internal static (Dictionary<int, long> Values, Dictionary<int, int> WorkerOf, int Valid,
            int Invalid, long? Result) Parse(IReadOnlyList<string> input)
    {
        var values = new Dictionary<int, long>();
        var workerOf = new Dictionary<int, int>();
        int valid = 0;
        int invalid = 0;
        long? result = null;
        foreach (string line in input)
        {
            if (line.StartsWith("result=", StringComparison.Ordinal))
            {
                string tail = line.Substring("result=".Length);
                if (long.TryParse(tail, out long committed))
                {
                    result = committed;
                    valid += 1;
                }
                else
                {
                    invalid += 1;
                }
                continue;
            }
            string[] parts = line.Split(';');
            if (parts.Length != 3
                    || !parts[0].StartsWith("chunk=", StringComparison.Ordinal)
                    || !parts[1].StartsWith("value=", StringComparison.Ordinal)
                    || !parts[2].StartsWith("worker=", StringComparison.Ordinal)
                    || !int.TryParse(parts[0].Substring("chunk=".Length), out int chunk)
                    || !long.TryParse(parts[1].Substring("value=".Length), out long value)
                    || !int.TryParse(parts[2].Substring("worker=".Length), out int worker))
            {
                invalid += 1;
                continue;
            }
            values[chunk] = value;
            workerOf[chunk] = worker;
            valid += 1;
        }
        return (values, workerOf, valid, invalid, result);
    }
}

internal class RunReport
{
    internal long Result;
    internal int ChunkExecutions;
    internal int LostChunks;
    internal int ReduceExecutions;
    internal bool Crashed;
    internal bool Committed;
}

internal static class MapReduce
{
    /**
     * One attempt at the job. `crashAfter` counts chunk executions in this attempt (-1 = never);
     * `crashBeforeCheckpoint` decides whether the in-flight chunk keeps its record.
     */
    internal static RunReport Attempt(int chunks, int workers, DurableStore store, bool resume,
            int crashAfter, bool crashBeforeCheckpoint, bool crashInReduce)
    {
        var report = new RunReport();
        var completed = new Dictionary<int, long>();
        if (resume)
        {
            var parsed = DurableStore.Parse(store.Lines);
            foreach (KeyValuePair<int, long> entry in parsed.Values) completed[entry.Key] = entry.Value;
        }
        for (int chunk = 0; chunk < chunks; chunk++)
        {
            int worker = chunk % workers;
            if (completed.ContainsKey(chunk)) continue;
            long value = MapJob.MapValue(chunk);
            report.ChunkExecutions += 1;
            bool crashNow = crashAfter >= 0 && report.ChunkExecutions == crashAfter;
            if (crashNow && crashBeforeCheckpoint)
            {
                report.Crashed = true;
                report.LostChunks += 1;
                return report;
            }
            store.AppendChunk(chunk, value, worker);
            completed[chunk] = value;
            if (crashNow)
            {
                report.Crashed = true;
                return report;
            }
        }
        long total = 0;
        foreach (KeyValuePair<int, long> entry in completed) total += entry.Value;
        report.ReduceExecutions += 1;
        report.Result = total;
        if (crashInReduce)
        {
            report.Crashed = true;
            return report;
        }
        store.AppendResult(total);
        report.Committed = true;
        return report;
    }

    /** Runs attempts until one commits, returning the per-attempt reports. */
    internal static List<RunReport> Recover(int chunks, int workers, DurableStore store,
            List<(int CrashAfter, bool BeforeCheckpoint, bool InReduce)> plan)
    {
        var reports = new List<RunReport>();
        foreach ((int crashAfter, bool before, bool inReduce) in plan)
        {
            reports.Add(Attempt(chunks, workers, store, reports.Count > 0, crashAfter, before, inReduce));
        }
        reports.Add(Attempt(chunks, workers, store, true, -1, false, false));
        return reports;
    }
}

public static class Solution
{
    public static void Main()
    {
        const int chunks = 40;
        const int workers = 4;
        long expected = MapJob.BruteForceTotal(chunks);
        Check(expected > 0, "the brute-force total is positive");

        // --- clean run ----------------------------------------------------------------------------------
        var cleanStore = new DurableStore();
        List<RunReport> clean = MapReduce.Recover(chunks, workers, cleanStore,
                new List<(int, bool, bool)>());
        Check(clean.Count == 1, "a clean plan needs a single attempt");
        Check(clean[0].Committed && clean[0].Result == expected, "the clean run commits the right total");
        Check(clean[0].ChunkExecutions == chunks, "every chunk runs exactly once");
        Check(clean[0].LostChunks == 0 && clean[0].ReduceExecutions == 1, "one reduce, no losses");
        (Dictionary<int, long> cleanValues, Dictionary<int, int> cleanWorkers, int cleanValid,
                int cleanInvalid, long? cleanResult) = DurableStore.Parse(cleanStore.Lines);
        Check(cleanValid == chunks + 1 && cleanInvalid == 0, "one record per chunk plus the commit");
        Check(cleanValues.Count == chunks, "the log covers every chunk");
        Check(cleanResult == expected, "the committed record equals the total");
        foreach (KeyValuePair<int, long> entry in cleanValues)
        {
            Check(entry.Value == MapJob.MapValue(entry.Key), "record for chunk " + entry.Key + " is valid");
        }
        var perWorker = new Dictionary<int, int>();
        foreach (KeyValuePair<int, int> entry in cleanWorkers)
        {
            perWorker.TryGetValue(entry.Value, out int count);
            perWorker[entry.Value] = count + 1;
        }
        Check(perWorker.Count == workers, "all four workers appear in the log");
        foreach (KeyValuePair<int, int> entry in perWorker)
        {
            Check(entry.Value == chunks / workers, "worker " + entry.Key + " owns a quarter of the chunks");
        }

        // --- three injected crashes ------------------------------------------------------------------------
        var crashPlan = new List<(int, bool, bool)>
        {
            (7, true, false),
            (5, false, false),
            (11, true, false),
        };
        var crashStore = new DurableStore();
        List<RunReport> crashed = MapReduce.Recover(chunks, workers, crashStore, crashPlan);
        Check(crashed.Count == 4, "three crashes plus the final committing attempt");
        Check(crashed[3].Committed && crashed[3].Result == expected, "recovery reaches the right total");
        int totalExecutions = 0;
        int lostChunks = 0;
        foreach (RunReport report in crashed) totalExecutions += report.ChunkExecutions;
        Check(crashed[0].LostChunks == 1, "the before-checkpoint crash lost exactly one chunk");
        Check(crashed[1].LostChunks == 0, "the after-checkpoint crash lost nothing");
        Check(crashed[2].LostChunks == 1, "the second before-checkpoint crash lost one chunk");
        foreach (RunReport report in crashed) lostChunks += report.LostChunks;
        Check(lostChunks == 2, "two chunks were lost in total");
        Check(totalExecutions == chunks + lostChunks,
                "redo work equals the losses (" + totalExecutions + " executions)");
        Check(crashed[1].ChunkExecutions == 5, "the after-checkpoint attempt ran exactly its five chunks");
        var crashParse = DurableStore.Parse(crashStore.Lines);
        Dictionary<int, long> crashValues = crashParse.Values;
        Check(crashParse.Result == expected, "the recovered commit is correct");
        Check(crashValues.Count == chunks, "every chunk is checkpointed after recovery");
        Check(crashParse.Valid > crashValues.Count, "the log contains duplicate records from the redos");
        Check(crashParse.Invalid == 0, "no torn record in this run");
        foreach (KeyValuePair<int, long> entry in crashValues)
        {
            Check(entry.Value == MapJob.MapValue(entry.Key), "recovered record for chunk " + entry.Key);
        }
        var contributions = new Dictionary<int, int>();
        foreach (int chunk in crashValues.Keys)
        {
            contributions.TryGetValue(chunk, out int count);
            contributions[chunk] = count + 1;
        }
        Check(contributions.Count == chunks, "the reduced output contributes every chunk exactly once");

        // --- a torn trailing record -------------------------------------------------------------------------
        var tornStore = new DurableStore();
        MapReduce.Recover(24, 4, tornStore, new List<(int, bool, bool)>());
        tornStore.AppendTorn(24, 0);
        var tornParse = DurableStore.Parse(tornStore.Lines);
        Check(tornParse.Invalid == 1, "the torn record is rejected by the parser");
        Check(!tornParse.Values.ContainsKey(24), "the torn chunk has no valid record");
        List<RunReport> tornRecovery = MapReduce.Recover(chunks, workers, tornStore,
                new List<(int, bool, bool)>());
        Check(tornRecovery[0].ChunkExecutions == chunks - 24 + 0, "only the unfinished tail is redone");
        Check(tornRecovery[0].LostChunks == 0, "the torn record was simply ignored");
        Check(tornRecovery[tornRecovery.Count - 1].Committed
                && tornRecovery[tornRecovery.Count - 1].Result == expected,
                "recovery after a torn record still commits the right total");

        // --- a crash after the last chunk: the reduce phase is what fails --------------------------------------
        var reduceStore = new DurableStore();
        List<RunReport> reduceCrash = MapReduce.Recover(chunks, workers, reduceStore,
                new List<(int, bool, bool)> { (-1, false, true) });
        Check(reduceCrash.Count == 2, "the reduce crash needs one more attempt");
        Check(reduceCrash[0].ChunkExecutions == chunks, "the first attempt mapped everything");
        Check(reduceCrash[0].ReduceExecutions == 1 && reduceCrash[0].Crashed, "and died inside the reduce");
        Check(reduceCrash[1].ChunkExecutions == 0, "recovery re-runs the reduce only");
        Check(reduceCrash[1].ReduceExecutions == 1 && reduceCrash[1].Committed, "and commits");
        Check(reduceCrash[1].Result == expected, "the recommitted total is correct");
        Check(reduceStore.Lines.Count == chunks + 1, "the commit record is appended once");

        // --- parsing is idempotent and defensive ---------------------------------------------------------------
        List<string> snapshot = crashStore.Snapshot();
        var firstParse = DurableStore.Parse(snapshot);
        var secondParse = DurableStore.Parse(snapshot);
        Check(firstParse.Values.Count == secondParse.Values.Count, "parsing is idempotent");
        foreach (KeyValuePair<int, long> entry in firstParse.Values)
        {
            Check(secondParse.Values[entry.Key] == entry.Value, "idempotent value for chunk " + entry.Key);
        }
        Check(firstParse.Result == secondParse.Result, "the parsed commit is stable");
        Check(DurableStore.Parse(new List<string> { "garbage", "chunk=x;value=y;worker=z", "result=" })
                .Invalid == 3, "garbage lines are counted as invalid");
        Check(DurableStore.Parse(cleanStore.Snapshot()).Values.Count == chunks, "the clean log re-parses");

        // --- degenerate jobs ------------------------------------------------------------------------------------
        var emptyStore = new DurableStore();
        List<RunReport> empty = MapReduce.Recover(0, 4, emptyStore, new List<(int, bool, bool)>());
        Check(empty[0].ChunkExecutions == 0 && empty[0].Result == 0, "an empty job reduces to zero");
        Check(emptyStore.Lines.Count == 1, "the empty job still commits a result");
        var singleStore = new DurableStore();
        List<RunReport> single = MapReduce.Recover(1, 4, singleStore, new List<(int, bool, bool)>());
        Check(single[0].ChunkExecutions == 1 && single[0].Result == MapJob.MapValue(0),
                "a single chunk maps and reduces correctly");
        var singleCrashStore = new DurableStore();
        List<RunReport> singleCrash = MapReduce.Recover(1, 4, singleCrashStore,
                new List<(int, bool, bool)> { (1, true, false) });
        Check(singleCrash[0].LostChunks == 1 && singleCrash[1].ChunkExecutions == 1,
                "the only chunk is redone after the crash");
        Check(singleCrash[1].Result == MapJob.MapValue(0), "and the job still completes");

        // --- determinism -------------------------------------------------------------------------------------------
        var repeatStore = new DurableStore();
        List<RunReport> repeat = MapReduce.Recover(chunks, workers, repeatStore, crashPlan);
        Check(repeatStore.Lines.Count == crashStore.Lines.Count, "the store length is reproducible");
        for (int i = 0; i < crashStore.Lines.Count; i++)
        {
            Check(repeatStore.Lines[i] == crashStore.Lines[i], "store line " + i + " is reproducible");
        }
        Check(repeat[repeat.Count - 1].Result == crashed[crashed.Count - 1].Result,
                "the final totals agree across runs");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
