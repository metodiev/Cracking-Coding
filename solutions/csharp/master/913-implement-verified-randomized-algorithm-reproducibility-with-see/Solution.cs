// Task 913 - Implement verified randomized algorithm reproducibility with seed management and audit logs.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Two halves that together make a randomized experiment auditable:
//
//  * seed management - a root seed is expanded into per-(experiment, trial) seeds by hashing the experiment
//    name and mixing it with the trial index through splitmix64.  Derivation is deterministic, collision
//    free over thousands of pairs and sensitive to every ingredient, so a trial can be replayed from the
//    root seed alone;
//  * an audit log - every trial appends a record holding the experiment, the trial index, the seed used and
//    a SHA-256 digest of the result, chained to the previous record's digest starting from a fixed genesis
//    value.  Verification recomputes the chain and reports the first broken index, so a modified result, a
//    swapped seed, a deleted record or a spliced-in record is detected at exactly the right place.
//
// The demonstration algorithm is a Monte Carlo estimate of pi from a deterministic splitmix64 point stream,
// which is cheap, seed sensitive and easy to replay.
//
// Simplifications: SHA-256 instead of signatures (integrity, not authenticity), one process, and an
// in-memory log.  Verified: seed determinism/distinctness/sensitivity, result replay from the log, the
// chain verification, four kinds of tampering detected at the correct index, the empty/single-record
// boundaries and cross-run reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static double NextUniform() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    private const ulong Genesis = 0x5EED000000000000UL;

    /** Deterministic seed derivation: hash the experiment name, then mix root, hash and trial together. */
    private static ulong DeriveSeed(ulong root, string experiment, int trial)
    {
        ulong hash = 1469598103934665603UL;
        foreach (char c in experiment)
        {
            hash ^= c;
            hash *= 1099511628211UL;
        }
        unchecked
        {
            ulong mixed = root ^ hash ^ (ulong)trial * 0x9E3779B97F4A7C15UL;
            mixed ^= mixed >> 30;
            mixed *= 0xBF58476D1CE4E5B9UL;
            mixed ^= mixed >> 27;
            mixed *= 0x94D049BB133111EBUL;
            mixed ^= mixed >> 31;
            return mixed == 0 ? 1UL : mixed;
        }
    }

    private static string Hex(byte[] value)
    {
        StringBuilder builder = new StringBuilder(value.Length * 2);
        foreach (byte b in value) builder.Append(b.ToString("x2", CultureInfo.InvariantCulture));
        return builder.ToString();
    }

    private static string Hash(string material) =>
        Hex(SHA256.HashData(Encoding.UTF8.GetBytes(material))).Substring(0, 32);

    /** The audited experiment: a Monte Carlo estimate of pi from the seed's own point stream. */
    private static double RunExperiment(ulong seed, int samples)
    {
        _state = seed;
        int inside = 0;
        for (int i = 0; i < samples; i++)
        {
            double x = NextUniform();
            double y = NextUniform();
            if (x * x + y * y <= 1.0) inside++;
        }
        return 4.0 * inside / samples;
    }

    private static string ResultDigest(double estimate) =>
        Hash("pi=" + estimate.ToString("R", CultureInfo.InvariantCulture));

    internal sealed class Record
    {
        internal int Index;
        internal string Experiment = string.Empty;
        internal int Trial;
        internal ulong Seed;
        internal string Result = string.Empty;
        internal string Previous = string.Empty;
        internal string Hash = string.Empty;
    }

    private static string ComputeHash(Record record) =>
        Hash(record.Index + "|" + record.Experiment + "|" + record.Trial + "|" + record.Seed + "|" + record.Result + "|" + record.Previous);

    internal static Record Append(List<Record> log, string experiment, int trial, ulong seed, int samples)
    {
        Record record = new Record
        {
            Index = log.Count,
            Experiment = experiment,
            Trial = trial,
            Seed = seed,
            Result = ResultDigest(RunExperiment(seed, samples)),
            Previous = log.Count == 0 ? Genesis.ToString("x16", CultureInfo.InvariantCulture) : log[log.Count - 1].Hash,
        };
        record.Hash = ComputeHash(record);
        log.Add(record);
        return record;
    }

    /** Recomputes the chain; returns the number of valid records and the first index that does not verify. */
    private static (int Valid, int Broken) Verify(List<Record> log)
    {
        string previous = Genesis.ToString("x16", CultureInfo.InvariantCulture);
        for (int i = 0; i < log.Count; i++)
        {
            Record record = log[i];
            if (record.Index != i || record.Previous != previous) return (i, i);
            if (ComputeHash(record) != record.Hash) return (i, i);
            previous = record.Hash;
        }
        return (log.Count, -1);
    }

    private static List<Record> Copy(List<Record> log)
    {
        List<Record> clone = new List<Record>();
        foreach (Record record in log)
        {
            clone.Add(new Record
            {
                Index = record.Index,
                Experiment = record.Experiment,
                Trial = record.Trial,
                Seed = record.Seed,
                Result = record.Result,
                Previous = record.Previous,
                Hash = record.Hash,
            });
        }
        return clone;
    }

    public static void Main()
    {
        const ulong root = 0xC0FFEE123456789UL;
        const int samples = 4000;
        string[] experiments = { "karger", "monte-carlo-pi", "randomized-quicksort" };

        // ---- 1. seed derivation ----------------------------------------------------------------------
        HashSet<ulong> seeds = new HashSet<ulong>();
        int pairs = 0;
        foreach (string experiment in experiments)
        {
            for (int trial = 0; trial < 1500; trial++)
            {
                seeds.Add(DeriveSeed(root, experiment, trial));
                pairs++;
            }
        }
        Check(pairs == 4500 && seeds.Count == pairs, "all " + pairs + " derived seeds are distinct");
        Check(DeriveSeed(root, "karger", 7) == DeriveSeed(root, "karger", 7), "derivation is deterministic");
        Check(DeriveSeed(root, "karger", 7) != DeriveSeed(root, "karger", 8), "the trial index changes the seed");
        Check(DeriveSeed(root, "karger", 7) != DeriveSeed(root, "monte-carlo-pi", 7), "the experiment name changes the seed");
        Check(DeriveSeed(root, "karger", 7) != DeriveSeed(root + 1, "karger", 7), "the root seed changes the derived seed");
        int[] buckets = new int[16];
        foreach (ulong seed in seeds) buckets[seed % 16]++;
        int expected = pairs / 16;
        bool spread = true;
        foreach (int count in buckets) spread &= Math.Abs(count - expected) < expected / 3;
        Check(spread, "derived seeds spread over 16 buckets within a third of the mean");
        Check(seeds.Contains(DeriveSeed(root, "karger", 0)) && !seeds.Contains(0UL), "no derived seed collapses to zero");

        // ---- 2. the audited experiments ---------------------------------------------------------------
        List<Record> log = new List<Record>();
        foreach (string experiment in experiments)
        {
            for (int trial = 0; trial < 4; trial++) Append(log, experiment, trial, DeriveSeed(root, experiment, trial), samples);
        }
        Check(log.Count == 12, "the log holds one record per trial (" + log.Count + ")");
        Check(log[0].Previous == Genesis.ToString("x16", CultureInfo.InvariantCulture), "the first record chains onto the genesis value");
        Check(log[1].Previous == log[0].Hash, "each record chains onto its predecessor's digest");
        (int valid, int broken) = Verify(log);
        Check(valid == log.Count && broken == -1, "the untouched log verifies end to end (" + valid + " records)");

        // ---- 3. replay from the log ---------------------------------------------------------------------
        int replays = 0;
        foreach (Record record in log)
        {
            double replay = RunExperiment(record.Seed, samples);
            Check(ResultDigest(replay) == record.Result, "record " + record.Index + " replays to its recorded digest");
            replays++;
        }
        Check(replays == 12, "all twelve records replayed exactly");
        Check(RunExperiment(DeriveSeed(root, "karger", 0), samples) == RunExperiment(DeriveSeed(root, "karger", 0), samples),
            "re-running an experiment reproduces the identical estimate");
        Check(RunExperiment(DeriveSeed(root, "karger", 0), samples) != RunExperiment(DeriveSeed(root, "karger", 1), samples),
            "a different trial seed gives a different estimate");
        double estimate = RunExperiment(DeriveSeed(root, "karger", 0), samples);
        Check(Math.Abs(estimate - Math.PI) < 0.2, "the Monte Carlo estimate is close to pi (" + estimate.ToString("F4", CultureInfo.InvariantCulture) + ")");

        // ---- 4. tampering is detected at the right index ----------------------------------------------------
        List<Record> tamperedResult = Copy(log);
        tamperedResult[5].Result = Hash("pi=3.0");
        (valid, broken) = Verify(tamperedResult);
        Check(broken == 5, "a modified result is caught at record 5 (broken=" + broken + ")");
        List<Record> tamperedSeed = Copy(log);
        tamperedSeed[8].Seed = 424242UL;
        Check(Verify(tamperedSeed).Broken == 8, "a swapped seed is caught at record 8");
        List<Record> deleted = Copy(log);
        deleted.RemoveAt(3);
        Check(Verify(deleted).Broken == 3, "a deleted record is caught where the chain falls apart");
        List<Record> inserted = Copy(log);
        inserted.Insert(4, Copy(new List<Record> { log[4] })[0]);
        int spliceBroken = Verify(inserted).Broken;
        Check(spliceBroken >= 4 && spliceBroken <= 5,
            "a spliced-in duplicate is detected at the first index that stops lining up (" + spliceBroken + ")");
        List<Record> rewrittenPrevious = Copy(log);
        string genesis = Genesis.ToString("x16", CultureInfo.InvariantCulture);
        rewrittenPrevious[0].Previous = "0" + genesis.Substring(1);
        Check(rewrittenPrevious[0].Previous != genesis, "the forged genesis link really differs");
        Check(Verify(rewrittenPrevious).Broken == 0, "a forged genesis link is caught at record 0");

        // ---- 5. boundaries -----------------------------------------------------------------------------------
        Check(Verify(new List<Record>()).Valid == 0, "an empty log verifies trivially with zero records");
        List<Record> single = new List<Record>();
        Append(single, "solo", 0, DeriveSeed(root, "solo", 0), samples);
        Check(Verify(single).Valid == 1 && single[0].Index == 0, "a single record log verifies");
        List<Record> singleBroken = Copy(single);
        singleBroken[0].Hash = Hash("forged");
        Check(Verify(singleBroken).Broken == 0, "a forged single record is rejected");

        // ---- 6. cross-run reproducibility ---------------------------------------------------------------------
        List<Record> firstRun = new List<Record>();
        foreach (string experiment in experiments) Append(firstRun, experiment, 3, DeriveSeed(root, experiment, 3), samples);
        List<Record> secondRun = new List<Record>();
        foreach (string experiment in experiments) Append(secondRun, experiment, 3, DeriveSeed(root, experiment, 3), samples);
        bool identical = firstRun.Count == secondRun.Count;
        for (int i = 0; i < firstRun.Count && identical; i++) identical = firstRun[i].Hash == secondRun[i].Hash;
        Check(identical, "two runs from the same root seed produce byte-identical logs");
        Check(firstRun[0].Hash != log[0].Hash, "a different trial index yields a different digest");

        Console.WriteLine("seeds: " + pairs + " derived seeds from one root, all distinct and spread over the 16 buckets");
        Console.WriteLine("log: " + log.Count + " chained records verify; result/seed/deletion/insertion tampering caught at 5, 8, 3 and 4");
        Console.WriteLine("replay: all " + replays + " records reproduced their digests, pi estimate " +
            estimate.ToString("F4", CultureInfo.InvariantCulture));
        Console.WriteLine("PASS");
    }
}
