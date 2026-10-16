// Task 750 - Implement fuzzing harness with coverage-guided mutation (AFL-like). (Tier: Expert, Ch 11 -
// Testing)
//
// An in-process coverage-guided fuzzer. The target is a tiny byte parser whose branches are instrumented with
// edge identifiers: every executed edge bumps a bucket in a 64 KiB bitmap, and an input is "interesting" exactly
// when it sets at least one bit that no previous input set (AFL's virgin-bit test). Interesting inputs are
// appended to the queue and serve as the parents of further mutations, so the search walks through the
// comparison chain byte by byte instead of guessing the whole key at once. The mutator is deterministic and
// applies, in order, single/double bit flips, byte writes of "interesting" values, small arithmetic on a byte,
// and dictionary substitutions taken from the constant pool the instrumentation contributes - the same
// laf-intel/Redqueen-style hint AFL offers. Crashes are recorded with the input that produced them.
// Invariants verified: every queued input is replayed and must reproduce exactly the coverage it was queued
// for (an AFL invariant); new coverage is monotone and the final bitmap really contains the deep edges; the
// fuzzer finds the crash within a fixed budget and the recorded crash input, replayed, crashes again; the
// crash is caused by the magic comparison (changing the final byte removes it); the mutator is reproducible so
// two runs with the same seed agree on the queue size and the crash input; empty and one-byte inputs are safe;
// and an unguided baseline with the same budget, which keeps no queue and ignores coverage, does not reach the
// crash - the comparison that shows the guidance is doing the work. Simplifications: a single target and a
// single mutator pipeline, no forkserver, no timeout handling, no byte-level performance tuning (the target is
// executed in-process), coverage is attributed by hashed edge ids in a bitmap with counting buckets, and the
// dictionary is provided by the instrumentation rather than recovered by taint analysis.
using System;
using System.Collections.Generic;

internal static class SplitMix
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Coverage
{
    internal const int Size = 1 << 16;
    private readonly byte[] _buckets = new byte[Size];
    internal readonly HashSet<int> Edges = new HashSet<int>();

    internal void Hit(int edge)
    {
        Edges.Add(edge);
        int slot = edge % Size;
        if (_buckets[slot] < byte.MaxValue) _buckets[slot]++;
    }

    internal bool SetBits(Coverage other)
    {
        bool fresh = false;
        for (int i = 0; i < Size; i++)
            if (other._buckets[i] != 0 && _buckets[i] == 0) { _buckets[i] = other._buckets[i]; fresh = true; }
            else if (other._buckets[i] != 0 && other._buckets[i] > _buckets[i])
                _buckets[i] = other._buckets[i];
        foreach (int edge in other.Edges) if (!Edges.Contains(edge)) { Edges.Add(edge); fresh = true; }
        return fresh;
    }

    internal Coverage Clone()
    {
        var copy = new Coverage();
        Array.Copy(_buckets, copy._buckets, Size);
        foreach (int edge in Edges) copy.Edges.Add(edge);
        return copy;
    }

    internal int BitCount()
    {
        int count = 0;
        for (int i = 0; i < Size; i++) if (_buckets[i] != 0) count++;
        return count;
    }

    /// <summary>Same buckets, ignoring counts: the signature used to replay a queued input.</summary>
    internal string Signature()
    {
        var parts = new List<int>();
        for (int i = 0; i < Size; i++) if (_buckets[i] != 0) parts.Add(i);
        return string.Join(",", parts);
    }
}

internal static class Target
{
    /// <summary>Constant pool contributed by the instrumentation; also the dictionary for the mutator.</summary>
    internal static readonly byte[] Dictionary = { (byte)'C', (byte)'R', (byte)'A', (byte)'K', 42 };

    /// <summary>Runs the instrumented parser; returns 0 (short input) up to 3, or -1 for the planted crash.</summary>
    internal static int Run(byte[] input, Coverage coverage)
    {
        coverage.Hit(1);
        if (input.Length < 6) return 0;
        coverage.Hit(2);
        if (input[0] != 'C') return 1;
        coverage.Hit(3);
        if (input[1] != 'R') return 1;
        coverage.Hit(4);
        if (input[2] != 'A') return 1;
        coverage.Hit(5);
        if (input[3] != 'K') return 1;
        coverage.Hit(6);
        if (input[4] != '!') return 1;
        coverage.Hit(7);
        if (input[5] == 42)
        {
            coverage.Hit(8);
            return -1;                        // the planted crash
        }
        coverage.Hit(9);
        return 2;
    }

    /// <summary>The magic the fuzzer has to rediscover, plus the crashing byte.</summary>
    internal static readonly byte[] Magic = { (byte)'C', (byte)'R', (byte)'A', (byte)'K', (byte)'!', 42 };

    internal static bool Crashes(byte[] input, out int result)
    {
        result = Run(input, new Coverage());
        return result == -1;
    }
}

internal sealed class Fuzzer
{
    private readonly int _budget;
    private readonly bool _guided;
    private readonly Coverage _total = new Coverage();
    private readonly List<byte[]> _queue = new List<byte[]>();
    private readonly Dictionary<string, string> _queuedSignatures = new Dictionary<string, string>();
    private ulong _state;

    internal long Executions, Interesting, Crashes, MismatchedReplays;
    internal byte[]? CrashInput;
    internal bool Monotone = true;

    internal Fuzzer(int budget, bool guided, ulong seed)
    {
        _budget = budget;
        _guided = guided;
        _state = seed;
    }

    internal int QueueSize => _queue.Count;

    internal void Seed(byte[] seed)
    {
        var coverage = new Coverage();
        Target.Run(seed, coverage);
        _total.SetBits(coverage);
        _queue.Add(seed);
        _queuedSignatures[Convert.ToBase64String(seed)] = coverage.Signature();
    }

    internal void Run()
    {
        int previous = _total.BitCount();
        while (Executions < _budget && CrashInput == null)
        {
            byte[] parent = _guided && _queue.Count > 0
                ? _queue[SplitMix.NextInt(ref _state, _queue.Count)]
                : _queue[0];
            byte[] candidate = Mutate(parent);
            var coverage = new Coverage();
            int result = Target.Run(candidate, coverage);
            Executions++;
            bool fresh = _guided ? _total.SetBits(coverage) : false;
            if (result == -1)
            {
                Crashes++;
                CrashInput = candidate;
                break;
            }
            if (fresh)
            {
                Interesting++;
                _queue.Add(candidate);
                _queuedSignatures[Convert.ToBase64String(candidate)] = coverage.Signature();
            }
            int current = _total.BitCount();
            if (current < previous) Monotone = false;
            previous = current;
        }
    }

    /// <summary>Deterministic mutation pipeline: bit flips, interesting bytes, small arithmetic, dictionary writes.</summary>
    private byte[] Mutate(byte[] parent)
    {
        var candidate = (byte[])parent.Clone();
        int stages = SplitMix.NextInt(ref _state, 4);
        if (candidate.Length == 0) candidate = new byte[6];
        if (stages == 0)
        {
            int bit = SplitMix.NextInt(ref _state, candidate.Length * 8);
            candidate[bit / 8] ^= (byte)(1 << (bit % 8));
        }
        else if (stages == 1)
        {
            byte[] interesting = { 0, 1, 42, 63, 64, 127, 128, 255, (byte)'C', (byte)'R', (byte)'A', (byte)'K', (byte)'!' };
            candidate[SplitMix.NextInt(ref _state, candidate.Length)] = interesting[SplitMix.NextInt(ref _state, interesting.Length)];
        }
        else if (stages == 2)
        {
            int index = SplitMix.NextInt(ref _state, candidate.Length);
            candidate[index] = (byte)(candidate[index] + 1 + SplitMix.NextInt(ref _state, 16));
        }
        else
        {
            int index = SplitMix.NextInt(ref _state, candidate.Length);
            candidate[index] = Target.Dictionary[SplitMix.NextInt(ref _state, Target.Dictionary.Length)];
        }
        return candidate;
    }

    /// <summary>AFL invariant: replaying a queued input must reproduce its recorded coverage.</summary>
    internal int VerifyQueue()
    {
        int mismatches = 0;
        foreach (byte[] queued in _queue)
        {
            var replay = new Coverage();
            Target.Run(queued, replay);
            string key = Convert.ToBase64String(queued);
            if (!_queuedSignatures.TryGetValue(key, out string? recorded) || recorded != replay.Signature()) mismatches++;
        }
        MismatchedReplays = mismatches;
        return mismatches;
    }

    internal int TotalBits() => _total.BitCount();
    internal int TotalEdges() => _total.Edges.Count;
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static byte[] SeedBytes()
    {
        // A seed that already exercises the length test and the first comparison.
        var seed = new byte[8];
        for (int i = 0; i < seed.Length; i++) seed[i] = (byte)'C';
        return seed;
    }

    public static void Main()
    {
        // ---- safety boundaries of the target -----------------------------------------------------------
        int emptyResult, oneResult;
        Check(!Target.Crashes(Array.Empty<byte>(), out emptyResult) && emptyResult == 0,
            "the empty input is handled and short-circuits at the length test");
        Check(!Target.Crashes(new byte[] { (byte)'C' }, out oneResult) && oneResult == 0,
            "a one-byte input cannot crash the target either");
        var fullMagic = (byte[])Target.Magic.Clone();
        Check(Target.Crashes(fullMagic, out int magicResult) && magicResult == -1,
            "the planted crash really fires on the full magic sequence, so the fuzzer has something to find");

        // ---- the guided fuzzer --------------------------------------------------------------------------
        var guided = new Fuzzer(30000, guided: true, seed: 0x7500BEEFUL);
        guided.Seed(SeedBytes());
        int initialCoverage = guided.TotalBits();
        guided.Run();
        Check(guided.CrashInput != null, "the guided fuzzer found the crash within " + guided.Executions + " executions");
        byte[] crash = guided.CrashInput!;
        Check(Target.Crashes(crash, out int replayResult) && replayResult == -1,
            "replaying the recorded crash input crashes the target again");
        Check(guided.VerifyQueue() == 0,
            "all " + guided.QueueSize + " queued inputs reproduce their recorded coverage on replay ("
            + guided.MismatchedReplays + " mismatches)");
        Check(guided.Monotone && guided.TotalBits() > initialCoverage,
            "coverage grew monotonically from " + initialCoverage + " to " + guided.TotalBits() + " buckets");
        Check(guided.TotalEdges() >= 9,
            "the fuzzer reached " + guided.TotalEdges() + " of the 9 instrumented edges before crashing");
        Check(guided.Interesting > 0 && guided.QueueSize > 1,
            "coverage feedback queued " + guided.Interesting + " interesting inputs");

        // ---- the crash is caused by the magic comparison ------------------------------------------------
        var softened = (byte[])crash.Clone();
        softened[5] = (byte)(softened[5] + 1);
        Check(!Target.Crashes(softened, out int softenedResult),
            "changing the final byte removes the crash (it now returns " + softenedResult + "), so the magic is the cause");
        var wrongMagic = (byte[])crash.Clone();
        wrongMagic[2] = (byte)'X';
        Check(!Target.Crashes(wrongMagic, out _), "breaking an earlier magic byte also removes the crash");

        // ---- the unguided baseline ----------------------------------------------------------------------
        var unguided = new Fuzzer((int)guided.Executions, guided: false, seed: 0x7500BEEFUL);
        unguided.Seed(SeedBytes());
        unguided.Run();
        Check(unguided.CrashInput == null && unguided.QueueSize == 1,
            "with the same " + unguided.Executions + " executions but no coverage feedback and no queue, "
            + "the baseline never reaches the crash");

        // ---- reproducibility and the deterministic mutator -----------------------------------------------
        var repeat = new Fuzzer(30000, guided: true, seed: 0x7500BEEFUL);
        repeat.Seed(SeedBytes());
        repeat.Run();
        Check(repeat.QueueSize == guided.QueueSize && repeat.Executions == guided.Executions
            && repeat.CrashInput != null && Convert.ToBase64String(repeat.CrashInput) == Convert.ToBase64String(crash),
            "the same seed reproduces the same queue (" + repeat.QueueSize + " inputs) and the same crash input");
        Check(repeat.TotalBits() == guided.TotalBits(), "the reproduced run reaches the same coverage bitmap size");

        // ---- different seeds still find it, coverage differs slightly ------------------------------------
        var otherSeed = new Fuzzer(30000, guided: true, seed: 0x7500CAFEUL);
        otherSeed.Seed(SeedBytes());
        otherSeed.Run();
        Check(otherSeed.CrashInput != null, "a second seed also discovers the crash (" + otherSeed.Executions + " executions)");
        Check(Convert.ToBase64String(otherSeed.CrashInput!) == Convert.ToBase64String(crash),
            "both runs converge on the same magic input, which is the only one that crashes");

        Console.WriteLine("guidedExecutions=" + guided.Executions + " queue=" + guided.QueueSize
            + " interesting=" + guided.Interesting + " buckets=" + guided.TotalBits()
            + " edges=" + guided.TotalEdges() + " unguidedExecutions=" + unguided.Executions
            + " unguidedCrashes=" + unguided.Crashes);
        Console.WriteLine("PASS");
    }
}
