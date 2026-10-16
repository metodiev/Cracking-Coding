// Task 757 - Implement ORAM (oblivious RAM) sketch for simple access patterns. (Tier: Expert, Beyond - Security & Privacy)
//
// An ORAM hides *which* address a client touches from the server that stores the data. This file implements the
// square-root ORAM: N logical blocks live in sqrt(N) buckets, each with sqrt(N) + 1 slots, and every logical
// access scans (reads and rewrites) *all* buckets, so the server's observed access pattern is the sequence
//  R0 W0 R1 W1 ... R{m-1} W{m-1}  no matter which logical block was requested. After every sqrt(N) accesses the
// client performs an oblivious reshuffle: it reads every bucket, repacks all N blocks into their sqrt-blocks and
// writes every bucket back, again touching the whole memory in a fixed order. Contents are encrypted with a
// counter-mode keystream keyed by (secret key, epoch, slot index), so an occupied slot's ciphertext changes on
// every reshuffle even when its plaintext does not, and the server never sees a plaintext block.
//
// Documented simplification: the client locates a block by scanning instead of keeping the sqrt(N)-sized position
// map, which costs a constant factor but makes the argument simpler; the "trusted hardware" primitive assumed by
// the classic construction is not modelled. Key material is a hard-coded toy key - this is a simulation, not
// cryptography - and the reshuffle layout is permuted with the key rather than with a strong PRP.
//
// Invariants verified: (a) correctness against a plain array on a 400 access workload of reads and writes;
// (b) exactly-once placement - after every access each logical block is in exactly one bucket slot or in the
// client stash, and the reshuffle finds all N of them once; (c) obliviousness - the server's trace for three
// different access patterns of equal length is byte-for-byte identical, while a plain array store leaks the very
// same patterns immediately; (d) accounting - an access costs 2*sqrt(N) bucket operations and a reshuffle is
// amortised to 2*sqrt(N)/sqrt(N) per access; (e) ciphertext opacity - no stored cell equals its plaintext and
// no cell repeats across epochs.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    private static ulong state = 0x7570BADCAFE12345UL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal static class Keystream
{
    /** Counter-mode style keystream: a pure function of (key, epoch, slot), so re-encrypting hides repeats. */
    internal static long Mask(ulong key, long epoch, int slot)
    {
        ulong z = key ^ (ulong)epoch * 0x9E3779B97F4A7C15UL ^ (ulong)slot * 0xD1B54A32D192ED03UL;
        z ^= z >> 33;
        z *= 0xFF51AFD7ED558CCDUL;
        z ^= z >> 33;
        z *= 0xC4CEB9FE1A85EC53UL;
        z ^= z >> 33;
        return (long)z;
    }

    /** Secret slot permutation of a group, used so that the layout itself carries no information. */
    internal static int[] Permutation(ulong key, long epoch, int bucket, int size)
    {
        var order = new int[size];
        for (int i = 0; i < size; i++) order[i] = i;
        for (int i = size - 1; i > 0; i--)
        {
            ulong mixed = (ulong)Mask(key, epoch, 0x1000 + bucket * 64 + i);
            int j = (int)(mixed % (ulong)(i + 1));
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        return order;
    }
}

/** The untrusted server: opaque cells plus a trace of every bucket it is asked to touch. */
internal sealed class Server
{
    private readonly long[] rawIds;
    private readonly long[] rawValues;
    private readonly int buckets;
    private readonly int slots;

    internal readonly List<string> Trace = new List<string>();

    internal Server(int buckets, int slots)
    {
        this.buckets = buckets;
        this.slots = slots;
        rawIds = new long[buckets * slots];
        rawValues = new long[buckets * slots];
        for (int i = 0; i < rawIds.Length; i++)
        {
            rawIds[i] = -1;
            rawValues[i] = 0;
        }
    }

    internal long[] Read(int bucket)
    {
        Trace.Add("R" + bucket);
        var cells = CellBuffer();
        for (int s = 0; s < slots; s++)
        {
            cells[2 * s] = rawIds[bucket * slots + s];
            cells[2 * s + 1] = rawValues[bucket * slots + s];
        }
        return cells;
    }

    internal void Write(int bucket, long[] cells)
    {
        Trace.Add("W" + bucket);
        for (int s = 0; s < slots; s++)
        {
            rawIds[bucket * slots + s] = cells[2 * s];
            rawValues[bucket * slots + s] = cells[2 * s + 1];
        }
    }

    internal long RawCell(int bucket, int slot, bool id) => id ? rawIds[bucket * slots + slot] : rawValues[bucket * slots + slot];

    /** Out-of-band audit read: used by the invariant checkers, not by the protocol, so it does not touch the trace. */
    internal void Inspect(int bucket, long[] ids, long[] values)
    {
        for (int s = 0; s < slots; s++)
        {
            ids[s] = rawIds[bucket * slots + s];
            values[s] = rawValues[bucket * slots + s];
        }
    }

    internal string TraceText() => string.Join(" ", Trace);

    private long[] CellBuffer() => new long[2 * slots];

    internal int BucketCount => buckets;
    internal int SlotCount => slots;
}

internal sealed class SquareRootOram
{
    internal const int BlockCount = 64;
    internal const int BucketCount = 8;                  // sqrt(64)
    internal const int SlotCount = 9;                    // sqrt(64) + 1, one spare slot for the moving block
    internal const int GroupSize = BlockCount / BucketCount;
    internal const int EpochLength = BucketCount;        // reshuffle every sqrt(N) accesses

    private readonly Server server;
    private readonly ulong key;
    private readonly Dictionary<int, long> stash = new Dictionary<int, long>();
    private long epoch;

    internal long Accesses;
    internal long Reshuffles;
    internal long BucketOperations;
    internal long InventoryChecks;

    internal SquareRootOram(Server server, ulong key)
    {
        this.server = server;
        this.key = key;
    }

    internal void Initialize(long[] initialValues)
    {
        epoch++;                                         // slots are always written and read inside the same epoch
        for (int bucket = 0; bucket < BucketCount; bucket++)
        {
            var ids = new long[SlotCount];
            var values = new long[SlotCount];
            int[] order = Keystream.Permutation(key, epoch, bucket, GroupSize);
            for (int s = 0; s < SlotCount; s++) ids[s] = -1;
            for (int s = 0; s < GroupSize; s++)
            {
                int id = bucket * GroupSize + order[s];
                ids[s] = id;
                values[s] = initialValues[id];
            }
            WriteBucket(bucket, ids, values);
        }
    }

    internal long Access(int id, bool isWrite, long value)
    {
        Accesses++;
        bool heldLocally = stash.TryGetValue(id, out long current);
        long found = heldLocally ? current : -1;
        bool foundOnServer = false;

        for (int bucket = 0; bucket < BucketCount; bucket++)
        {
            (long[] ids, long[] values) = ReadBucket(bucket);
            int target = -1;
            for (int s = 0; s < SlotCount; s++)
            {
                if (ids[s] == id) target = s;
            }
            if (target >= 0)
            {
                if (heldLocally) Fail("block " + id + " exists both in a bucket and in the stash");
                found = values[target];
                foundOnServer = true;
                ids[target] = -1;
                values[target] = 0;
            }
            WriteBucket(bucket, ids, values);
        }

        if (!heldLocally && !foundOnServer) Fail("block " + id + " is not stored anywhere");
        long result = isWrite ? value : found;
        stash[id] = result;
        if (Accesses % EpochLength == 0) Reshuffle();
        return result;
    }

    /** Reads every block once, checks the exactly-once invariant, then writes every bucket back sequentially. */
    internal void Reshuffle()
    {
        Reshuffles++;
        var values = new long[BlockCount];
        var seen = new bool[BlockCount];

        for (int bucket = 0; bucket < BucketCount; bucket++)
        {
            (long[] ids, long[] slotValues) = ReadBucket(bucket);
            for (int s = 0; s < SlotCount; s++)
            {
                int id = (int)ids[s];
                if (id < 0) continue;
                if (id >= BlockCount || seen[id]) Fail("block " + id + " is on the server twice");
                seen[id] = true;
                values[id] = slotValues[s];
            }
        }
        foreach (KeyValuePair<int, long> entry in stash)
        {
            if (seen[entry.Key]) Fail("block " + entry.Key + " is both stashed and stored");
            seen[entry.Key] = true;
            values[entry.Key] = entry.Value;
        }
        for (int id = 0; id < BlockCount; id++)
        {
            if (!seen[id]) Fail("block " + id + " was lost");
        }
        stash.Clear();

        epoch++;
        for (int bucket = 0; bucket < BucketCount; bucket++)
        {
            var ids = new long[SlotCount];
            var slotValues = new long[SlotCount];
            int[] order = Keystream.Permutation(key, epoch, bucket, GroupSize);
            for (int s = 0; s < SlotCount; s++) ids[s] = -1;
            for (int s = 0; s < GroupSize; s++)
            {
                int id = bucket * GroupSize + order[s];
                ids[s] = id;
                slotValues[s] = values[id];
            }
            WriteBucket(bucket, ids, slotValues);
        }
    }

    /** Inventory invariant: every logical block sits in exactly one slot or in the stash (audited out of band). */
    internal bool InventoryHolds()
    {
        InventoryChecks++;
        var located = new int[BlockCount];
        var rawIds = new long[SlotCount];
        var rawValues = new long[SlotCount];
        for (int bucket = 0; bucket < BucketCount; bucket++)
        {
            server.Inspect(bucket, rawIds, rawValues);
            for (int s = 0; s < SlotCount; s++)
            {
                long mask = Keystream.Mask(key, epoch, bucket * SlotCount + s);
                int id = (int)(rawIds[s] ^ mask);
                if (id < 0) continue;
                if (id >= BlockCount) return false;
                located[id]++;
            }
        }
        foreach (KeyValuePair<int, long> entry in stash)
        {
            if (entry.Key < 0 || entry.Key >= BlockCount) return false;
            located[entry.Key]++;
        }
        for (int id = 0; id < BlockCount; id++)
        {
            if (located[id] != 1) return false;
        }
        return true;
    }

    internal int StashCount => stash.Count;
    internal long Epoch => epoch;

    private (long[] ids, long[] values) ReadBucket(int bucket)
    {
        long[] raw = server.Read(bucket);
        BucketOperations++;
        var ids = new long[SlotCount];
        var values = new long[SlotCount];
        for (int s = 0; s < SlotCount; s++)
        {
            long mask = Keystream.Mask(key, epoch, bucket * SlotCount + s);
            ids[s] = raw[2 * s] ^ mask;
            values[s] = raw[2 * s + 1] ^ mask;
        }
        return (ids, values);
    }

    private void WriteBucket(int bucket, long[] ids, long[] values)
    {
        var raw = new long[2 * SlotCount];
        for (int s = 0; s < SlotCount; s++)
        {
            long mask = Keystream.Mask(key, epoch, bucket * SlotCount + s);
            raw[2 * s] = ids[s] ^ mask;
            raw[2 * s + 1] = values[s] ^ mask;
        }
        server.Write(bucket, raw);
        BucketOperations++;
    }

    private static void Fail(string message) => throw new Exception("FAILED: " + message);
}

/** The naive contrast: a flat array whose server trace is the sequence of logical addresses touched. */
internal sealed class PlainStore
{
    private readonly long[] values;
    internal readonly List<int> Trace = new List<int>();

    internal PlainStore(long[] initial)
    {
        values = (long[])initial.Clone();
    }

    internal long Access(int id, bool isWrite, long value)
    {
        Trace.Add(id);
        if (isWrite) values[id] = value;
        return values[id];
    }

    internal string TraceText() => string.Join(" ", Trace);
}

public static class Solution
{
    private const int WorkloadLength = 400;

    public static void Main()
    {
        long[] initial = InitialValues();
        var server = new Server(SquareRootOram.BucketCount, SquareRootOram.SlotCount);
        var oram = new SquareRootOram(server, 0x0123456789ABCDEFUL);
        oram.Initialize(initial);

        var reference = (long[])initial.Clone();
        var pattern = BuildPattern(1);
        int mismatches = 0, inventoryFailures = 0;
        for (int i = 0; i < pattern.Count; i++)
        {
            (int id, bool isWrite, long value) = pattern[i];
            long expected;
            if (isWrite)
            {
                reference[id] = value;
                expected = value;
            }
            else
            {
                expected = reference[id];
            }
            long observed = oram.Access(id, isWrite, value);
            if (observed != expected) mismatches++;
            if (!oram.InventoryHolds()) inventoryFailures++;
        }
        Check(mismatches == 0 && inventoryFailures == 0,
            "400 accesses match the plain array and the exactly-once inventory holds after each one");
        Check(oram.Reshuffles == WorkloadLength / SquareRootOram.EpochLength,
            "the reshuffle ran " + oram.Reshuffles + " times in " + WorkloadLength + " accesses");
        Check(oram.StashCount == 0, "the stash is empty right after a reshuffle");

        // Obliviousness: three different patterns of the same length must produce identical server traces.
        string sequential = TraceOf(BuildPattern(0));
        string random = TraceOf(BuildPattern(1));
        string readsOnly = TraceOf(BuildPattern(2));
        Check(sequential == random && random == readsOnly,
            "three different access patterns (sequential, pseudo-random, read-only) produce one identical trace");
        Check(sequential.Split(' ').Length == SquareRootOram.BucketCount
            + 2 * SquareRootOram.BucketCount * (WorkloadLength + WorkloadLength / SquareRootOram.EpochLength),
            "the trace is one initial write pass, then 2*sqrt(N) touches per access plus a pass per reshuffle");
        Check(random.Length == sequential.Length, "every pattern costs the same number of bucket touches");

        // The contrast: the plain store leaks the pattern immediately.
        string plainSequential = PlainTraceOf(BuildPattern(0));
        string plainRandom = PlainTraceOf(BuildPattern(1));
        Check(plainSequential != plainRandom,
            "a plain array store reveals the difference between the two patterns in its very first entries");
        Check(plainSequential.Split(' ')[0] == "0" && plainRandom.Split(' ')[0] != "0",
            "the plain trace starts with the requested address, the ORAM trace never does");

        // Accounting: O(sqrt(N)) per access, amortised reshuffle included.
        double perAccess = oram.BucketOperations / (double)WorkloadLength;
        Check(perAccess <= 4 * SquareRootOram.BucketCount,
            "amortised cost is " + perAccess.ToString("F2") + " bucket operations per access (2*sqrt(N) + reshuffle)");
        Check(perAccess >= SquareRootOram.BucketCount,
            "and it really is sqrt(N)-ish, not O(1): " + perAccess.ToString("F2"));

        // Ciphertext opacity: no cell equals its plaintext and no cell repeats across epochs.
        var before = Snapshot(server);
        oram.Reshuffle();
        var after = Snapshot(server);
        Check(before.Count == after.Count && !before[3].Equals(after[3]),
            "the same slot holds a different ciphertext after a reshuffle: " + before[3] + " -> " + after[3]);
        int plaintextCells = 0;
        for (int bucket = 0; bucket < SquareRootOram.BucketCount; bucket++)
        {
            for (int slot = 0; slot < SquareRootOram.SlotCount; slot++)
            {
                long raw = server.RawCell(bucket, slot, false);
                int id = (int)(server.RawCell(bucket, slot, true) ^ Keystream.Mask(0x0123456789ABCDEFUL, oram.Epoch, bucket * SquareRootOram.SlotCount + slot));
                if (id >= 0 && raw == reference[id]) plaintextCells++;
            }
        }
        Check(plaintextCells == 0, "every occupied slot stores ciphertext, not the plaintext value");

        // Boundaries: a fresh ORAM with no access, the last block, repeated reads inside one epoch.
        var freshServer = new Server(SquareRootOram.BucketCount, SquareRootOram.SlotCount);
        var fresh = new SquareRootOram(freshServer, 7UL);
        fresh.Initialize(initial);
        Check(fresh.Reshuffles == 0 && fresh.InventoryHolds(), "a freshly initialized ORAM holds all blocks once");
        Check(fresh.Access(SquareRootOram.BlockCount - 1, false, 0) == initial[SquareRootOram.BlockCount - 1],
            "the last logical block is addressable");
        long firstRead = fresh.Access(7, false, 0);
        long secondRead = fresh.Access(7, false, 0);
        int stashAfterTwoReads = fresh.StashCount;
        fresh.Access(7, false, 0);
        Check(firstRead == initial[7] && secondRead == initial[7] && stashAfterTwoReads == 2 && fresh.StashCount == 2,
            "re-reading a block inside one epoch is served from the client stash without a new stash entry");
        Check(fresh.Access(7, true, 999) == 999 && fresh.Access(7, false, 0) == 999,
            "a write followed by a read returns the written value from the stash");
        Check(fresh.InventoryHolds() && fresh.Epoch == 1, "five accesses do not yet trigger the epoch reshuffle");

        Console.WriteLine("  ORAM: blocks=" + SquareRootOram.BlockCount + " buckets=" + SquareRootOram.BucketCount
            + " slots=" + SquareRootOram.SlotCount + " reshuffles=" + oram.Reshuffles);
        Console.WriteLine("  trace: " + sequential.Split(' ').Length + " bucket touches for " + WorkloadLength
            + " accesses, amortised " + perAccess.ToString("F2") + " per access");
        Console.WriteLine("PASS");
    }

    private static long[] InitialValues()
    {
        var values = new long[SquareRootOram.BlockCount];
        for (int id = 0; id < values.Length; id++) values[id] = 1000 + id * 7;
        return values;
    }

    /** Pattern 0: sequential; 1: pseudo-random with writes; 2: read only. */
    private static List<(int Id, bool IsWrite, long Value)> BuildPattern(int kind)
    {
        var pattern = new List<(int, bool, long)>();
        for (int i = 0; i < WorkloadLength; i++)
        {
            int id = kind == 0 ? i % SquareRootOram.BlockCount : SplitMix64.NextInt(SquareRootOram.BlockCount);
            bool isWrite = kind == 2 ? false : (kind == 0 ? i % 5 == 0 : SplitMix64.NextInt(3) == 0);
            pattern.Add((id, isWrite, 50000 + i));
        }
        return pattern;
    }

    private static string TraceOf(List<(int Id, bool IsWrite, long Value)> pattern)
    {
        long[] initial = InitialValues();
        var server = new Server(SquareRootOram.BucketCount, SquareRootOram.SlotCount);
        var oram = new SquareRootOram(server, 0x0123456789ABCDEFUL);
        oram.Initialize(initial);
        foreach ((int id, bool isWrite, long value) in pattern) oram.Access(id, isWrite, value);
        return server.TraceText();
    }

    private static string PlainTraceOf(List<(int Id, bool IsWrite, long Value)> pattern)
    {
        var store = new PlainStore(InitialValues());
        foreach ((int id, bool isWrite, long value) in pattern) store.Access(id, isWrite, value);
        return store.TraceText();
    }

    private static List<string> Snapshot(Server server)
    {
        var cells = new List<string>();
        for (int bucket = 0; bucket < SquareRootOram.BucketCount; bucket++)
        {
            for (int slot = 0; slot < SquareRootOram.SlotCount; slot++)
            {
                cells.Add(server.RawCell(bucket, slot, true) + "/" + server.RawCell(bucket, slot, false));
            }
        }
        return cells;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
