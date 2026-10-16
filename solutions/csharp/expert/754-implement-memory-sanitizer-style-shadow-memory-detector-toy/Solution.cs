// Task 754 - Implement memory sanitizer style shadow memory detector (toy). (Tier: Expert, Beyond - Security & Privacy)
//
// A scaled-down stand-in for AddressSanitizer + MemorySanitizer over a 4 KiB toy heap arena. Every byte of the
// arena has a shadow byte holding one of four states: Poisoned (redzone / unallocated), Uninitialized (inside a
// live payload, never written), Initialized (written), Freed (inside a payload whose block has been released).
// The allocator is a bump allocator that puts a 16 byte redzone on both sides of every payload and rounds the
// payload up to 8 bytes, so the shadow layout is fully determined by the allocation metadata.
//
// Detectors: heap-buffer-overflow (access to a poisoned byte that belongs to some block), wild-pointer (arena
// byte covered by no block, or off the arena entirely), uninitialized-read, use-after-free, double-free,
// invalid-free (interior or foreign pointer) and a leak report over the still-live blocks.
//
// Invariants verified. (a) Addressability: after every operation the shadow must agree with an independent
// recomputation from the allocator metadata - a byte is addressable exactly when it lies inside a live payload.
// (b) Differential oracle: a second, independent implementation keeps *interval* bookkeeping (start/size, live
// flag, merged written ranges) and derives each verdict by interval arithmetic instead of by reading the shadow
// array; every allocation, load, store and free of a 400 operation workload must yield the same verdict on both
// sides. (c) Precision: findings carry the owning allocation id and the exact byte offset, checked literally.
// Simplification, documented: no real memory, no threads, no stack/global instrumentation, no quarantine, and
// the payload padding of an odd-sized block is addressable (the real sanitizer tracks it with a tail mask).
using System;
using System.Collections.Generic;

#nullable enable

internal static class SplitMix64
{
    private static ulong state = 0x7545AFE0D5EEDUL;

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

internal enum Shadow : byte { Poisoned = 0, Uninitialized = 1, Initialized = 2, Freed = 3 }

internal sealed class Finding
{
    internal readonly string Kind;
    internal readonly int Address;
    internal readonly int AllocationId;   // 0 when no block could be attributed
    internal readonly int Offset;         // byte offset from the payload start (may be negative)
    internal readonly string Detail;

    internal Finding(string kind, int address, int allocationId, int offset, string detail)
    {
        Kind = kind;
        Address = address;
        AllocationId = allocationId;
        Offset = offset;
        Detail = detail;
    }

    internal string Key => Kind + "#a" + AllocationId + "@" + Offset;
}

internal sealed class HeapSanitizer
{
    internal const int ArenaSize = 4096;
    internal const int Redzone = 16;

    internal sealed class Allocation
    {
        internal readonly int Id;
        internal readonly int Payload;
        internal readonly int Size;       // aligned payload size
        internal readonly string Site;
        internal bool Live = true;

        internal Allocation(int id, int payload, int size, string site)
        {
            Id = id;
            Payload = payload;
            Size = size;
            Site = site;
        }
    }

    private readonly byte[] data = new byte[ArenaSize];
    private readonly Shadow[] shadow = new Shadow[ArenaSize];
    private readonly List<Allocation> allocations = new List<Allocation>();
    private int cursor = Redzone;
    private int nextId = 1;

    internal readonly List<Finding> Findings = new List<Finding>();
    internal long Accesses, LegalAccesses;

    internal HeapSanitizer()
    {
        for (int byteIndex = 0; byteIndex < ArenaSize; byteIndex++) shadow[byteIndex] = Shadow.Poisoned;
    }

    internal int LiveCount()
    {
        int live = 0;
        foreach (Allocation allocation in allocations)
        {
            if (allocation.Live) live++;
        }
        return live;
    }

    internal int LiveBytes()
    {
        int bytes = 0;
        foreach (Allocation allocation in allocations)
        {
            if (allocation.Live) bytes += allocation.Size;
        }
        return bytes;
    }

    internal List<string> LeakReport()
    {
        var leaks = new List<string>();
        foreach (Allocation allocation in allocations)
        {
            if (allocation.Live) leaks.Add("leak a" + allocation.Id + " " + allocation.Size + "B at " + allocation.Site);
        }
        return leaks;
    }

    internal int Alloc(int size, string site)
    {
        int aligned = (size + 7) & ~7;         // payload rounded up to 8 bytes; the pad is addressable but never written
        int payload = cursor + Redzone;
        if (payload + aligned + Redzone > ArenaSize)
        {
            Record(new Finding("allocation-too-large", payload, 0, 0, site));
            return -1;
        }
        var allocation = new Allocation(nextId++, payload, aligned, site);
        allocations.Add(allocation);
        for (int b = payload - Redzone; b < payload + aligned + Redzone; b++) shadow[b] = Shadow.Poisoned;
        for (int b = payload; b < payload + aligned; b++) shadow[b] = Shadow.Uninitialized;
        cursor = payload + aligned + Redzone;
        return payload;
    }

    /** Verdict for an access; a null result means the access is legal (addressable and defined on reads). */
    internal Finding? Classify(int address)
    {
        Accesses++;
        if (address < 0 || address >= ArenaSize)
        {
            return Record(new Finding("wild-pointer", address, 0, 0, "outside the arena"));
        }
        Shadow state = shadow[address];
        if (state == Shadow.Uninitialized || state == Shadow.Initialized) return null;
        Allocation? owner = OwnerOf(address);
        if (owner == null)
        {
            return Record(new Finding("wild-pointer", address, 0, 0, "arena byte covered by no allocation"));
        }
        int offset = address - owner.Payload;
        if (state == Shadow.Freed)
        {
            return Record(new Finding("use-after-free", address, owner.Id, offset, owner.Site));
        }
        return Record(new Finding("heap-buffer-overflow", address, owner.Id, offset, owner.Site));
    }

    internal Finding? Store(int address, byte value)
    {
        Finding? finding = Classify(address);
        if (finding != null) return finding;
        LegalAccesses++;
        data[address] = value;
        shadow[address] = Shadow.Initialized;
        return null;
    }

    internal Finding? Load(int address, out byte value)
    {
        value = 0;
        Finding? finding = Classify(address);
        if (finding != null) return finding;
        if (shadow[address] == Shadow.Uninitialized)
        {
            Allocation owner = OwnerOf(address)!;
            return Record(new Finding("uninitialized-read", address, owner.Id, address - owner.Payload, owner.Site));
        }
        LegalAccesses++;
        value = data[address];
        return null;
    }

    internal Finding? Free(int pointer)
    {
        Allocation? target = Find(p => p.Live && p.Payload == pointer);
        if (target == null)
        {
            Allocation? alreadyFreed = Find(p => !p.Live && p.Payload == pointer);
            Allocation? interior = Find(p => p.Live && pointer > p.Payload && pointer < p.Payload + p.Size);
            string kind = alreadyFreed != null ? "double-free" : "invalid-free";
            string detail = alreadyFreed != null ? "block already released"
                : interior != null ? "not the start of the block" : "pointer never allocated";
            int id = alreadyFreed != null ? alreadyFreed.Id : interior != null ? interior.Id : 0;
            return Record(new Finding(kind, pointer, id, 0, detail));
        }
        target.Live = false;
        for (int b = target.Payload; b < target.Payload + target.Size; b++) shadow[b] = Shadow.Freed;
        return null;
    }

    /** Independent recomputation of addressability from the allocator metadata; any mismatch is a shadow bug. */
    internal int AddressabilityViolations()
    {
        var expected = new bool[ArenaSize];
        foreach (Allocation allocation in allocations)
        {
            if (!allocation.Live) continue;
            for (int b = allocation.Payload; b < allocation.Payload + allocation.Size; b++) expected[b] = true;
        }
        int bad = 0;
        for (int b = 0; b < ArenaSize; b++)
        {
            bool addressable = shadow[b] == Shadow.Uninitialized || shadow[b] == Shadow.Initialized;
            if (addressable != expected[b]) bad++;
        }
        return bad;
    }

    private Allocation? OwnerOf(int address)
        => Find(p => address >= p.Payload - Redzone && address < p.Payload + p.Size + Redzone);

    private Allocation? Find(Func<Allocation, bool> predicate)
    {
        foreach (Allocation allocation in allocations)
        {
            if (predicate(allocation)) return allocation;
        }
        return null;
    }

    private Finding Record(Finding finding)
    {
        Findings.Add(finding);
        return finding;
    }
}

/** Interval based reference model: same policy, verdicts derived arithmetically, never reads a shadow byte. */
internal sealed class ReferenceHeap
{
    private sealed class Block
    {
        internal int Id;
        internal int Start;
        internal int Size;
        internal bool Live = true;
        internal readonly List<(int From, int To)> Written = new List<(int From, int To)>();
    }

    private readonly List<Block> blocks = new List<Block>();
    private int cursor = HeapSanitizer.Redzone;
    private int nextId = 1;

    internal int Alloc(int size)
    {
        int aligned = (size + 7) & ~7;
        int payload = cursor + HeapSanitizer.Redzone;
        if (payload + aligned + HeapSanitizer.Redzone > HeapSanitizer.ArenaSize) return -1;
        blocks.Add(new Block { Id = nextId++, Start = payload, Size = aligned });
        cursor = payload + aligned + HeapSanitizer.Redzone;
        return payload;
    }

    internal string Access(int address, bool isWrite)
    {
        if (address < 0 || address >= HeapSanitizer.ArenaSize) return "wild-pointer";
        Block? owner = null;
        foreach (Block block in blocks)
        {
            if (address >= block.Start - HeapSanitizer.Redzone && address < block.Start + block.Size + HeapSanitizer.Redzone)
            {
                owner = block;
                break;
            }
        }
        if (owner == null) return "wild-pointer";
        if (!owner.Live)
        {
            return address >= owner.Start && address < owner.Start + owner.Size ? "use-after-free" : "heap-buffer-overflow";
        }
        if (address < owner.Start || address >= owner.Start + owner.Size) return "heap-buffer-overflow";
        if (IsWritten(owner, address)) return "ok";
        if (isWrite)
        {
            MarkWritten(owner, address);
            return "ok";
        }
        return "uninitialized-read";
    }

    internal string Free(int pointer)
    {
        foreach (Block block in blocks)
        {
            if (!block.Live || block.Start != pointer) continue;
            block.Live = false;
            return "ok";
        }
        foreach (Block block in blocks)
        {
            if (!block.Live && block.Start == pointer) return "double-free";
        }
        return "invalid-free";
    }

    private static bool IsWritten(Block block, int address)
    {
        foreach ((int from, int to) in block.Written)
        {
            if (address >= from && address < to) return true;
        }
        return false;
    }

    private static void MarkWritten(Block block, int address)
    {
        for (int i = 0; i < block.Written.Count; i++)
        {
            (int from, int to) = block.Written[i];
            if (address + 1 >= from && address <= to)
            {
                block.Written[i] = (Math.Min(from, address), Math.Max(to, address + 1));
                return;
            }
        }
        block.Written.Add((address, address + 1));
    }
}

public static class Solution
{
    public static void Main()
    {
        Scenarios();    // one scenario per detector, checked literally
        Differential(); // 400 operation differential test against the interval oracle
        Console.WriteLine("PASS");
    }

    private static void Scenarios()
    {
        // A well behaved program: write every payload byte, read it back, release, and leak nothing.
        var clean = new HeapSanitizer();
        int cleanBlock = clean.Alloc(24, "clean");
        for (int i = 0; i < 24; i++) clean.Store(cleanBlock + i, (byte)(i * 7 + 1));
        byte last = 0;
        for (int i = 0; i < 24; i++) clean.Load(cleanBlock + i, out last);
        Check(clean.Findings.Count == 0, "a fully initialized block produces no finding");
        Check(last == (byte)(23 * 7 + 1), "stored bytes read back unchanged");
        Check(clean.AddressabilityViolations() == 0, "shadow addressability matches the allocator metadata");
        Check(clean.Free(cleanBlock) == null && clean.LiveCount() == 0 && clean.LeakReport().Count == 0,
            "freeing the only block leaves no leak");
        Check(clean.LegalAccesses == 48 && clean.Accesses == 48, "48 legal accesses, no false positive");

        // Right and left overflow: the byte just past the payload and the byte just before it.
        var overflow = new HeapSanitizer();
        int block = overflow.Alloc(16, "overflow");
        Check(overflow.Store(block + 16, 1) != null && overflow.Findings[^1].Key == "heap-buffer-overflow#a1@16",
            "write one past the payload: " + overflow.Findings[^1].Key);
        Check(overflow.Store(block - 1, 1) != null && overflow.Findings[^1].Key == "heap-buffer-overflow#a1@-1",
            "write one before the payload: " + overflow.Findings[^1].Key);
        Check(overflow.Store(block, 1) == null && overflow.AddressabilityViolations() == 0,
            "the first payload byte is still addressable after both reports");

        // Uninitialized reads, including the boundary between written and unwritten bytes and the 8 byte pad.
        var uninitialized = new HeapSanitizer();
        int partial = uninitialized.Alloc(3, "partial");          // payload is padded to 8 bytes
        uninitialized.Store(partial, 9);
        uninitialized.Store(partial + 1, 9);
        uninitialized.Store(partial + 2, 9);
        byte value = 0;
        Check(uninitialized.Load(partial + 2, out value) == null && value == 9, "the last written byte reads fine");
        Check(uninitialized.Load(partial + 3, out value) != null
            && uninitialized.Findings[^1].Key == "uninitialized-read#a1@3",
            "the padding byte is addressable but uninitialized: " + uninitialized.Findings[^1].Key);
        Check(uninitialized.Findings.Count == 1, "exactly one finding for the partial block");

        // Use after free, double free and an interior free that must not release the block.
        var freed = new HeapSanitizer();
        int released = freed.Alloc(8, "released");
        freed.Store(released, 3);
        freed.Free(released);
        Check(freed.Load(released, out value) != null && freed.Findings[^1].Key == "use-after-free#a1@0",
            "load after free: " + freed.Findings[^1].Key);
        Check(freed.Free(released) != null && freed.Findings[^1].Kind == "double-free", "second free is reported");

        var interior = new HeapSanitizer();
        int block2 = interior.Alloc(32, "interior");
        interior.Store(block2, 5);
        Check(interior.Free(block2 + 8) != null && interior.Findings[^1].Kind == "invalid-free",
            "freeing an interior pointer is reported");
        Check(interior.LiveCount() == 1 && interior.Load(block2, out value) == null && value == 5,
            "the rejected free left the block alive and readable");

        // Wild pointers and the zero size boundary.
        var wild = new HeapSanitizer();
        int wildBlock = wild.Alloc(8, "wild");
        Check(wild.Store(HeapSanitizer.ArenaSize + 5, 1)?.Kind == "wild-pointer", "address past the arena");
        Check(wild.Store(-1, 1)?.Kind == "wild-pointer", "negative address");
        Check(wild.Store(HeapSanitizer.Redzone - 1, 1)?.Kind == "wild-pointer",
            "arena prefix not covered by any allocation");
        int zero = wild.Alloc(0, "zero");
        Check(wild.Store(zero, 1) != null && wild.Findings[^1].Key == "heap-buffer-overflow#a2@0",
            "a zero sized block has no payload: " + wild.Findings[^1].Key);
        Check(wild.Free(zero) == null && wild.Free(wildBlock) == null, "zero sized and normal blocks free cleanly");

        // Leak accounting and the running addressability invariant.
        var leaks = new HeapSanitizer();
        int kept = leaks.Alloc(8, "kept");
        int dropped = leaks.Alloc(16, "dropped");
        leaks.Free(dropped);
        List<string> report = leaks.LeakReport();
        Check(report.Count == 1 && report[0].Contains("a1") && leaks.LiveBytes() == 8,
            "one leak of 8 bytes reported: " + report[0]);
        Check(leaks.AddressabilityViolations() == 0, "addressability invariant holds after frees");
        leaks.Free(kept);
        Check(leaks.LeakReport().Count == 0 && leaks.LiveCount() == 0, "all blocks released, no leak");
    }

    private static void Differential()
    {
        var sanitizer = new HeapSanitizer();
        var oracle = new ReferenceHeap();
        var blocks = new List<(int Pointer, int Size)>();
        var histogram = new Dictionary<string, int>();
        int mismatches = 0, opaque = 0, legal = 0, allocationFailures = 0, operations = 400;

        for (int op = 0; op < operations; op++)
        {
            int choice = SplitMix64.NextInt(100);
            if (choice < 30)
            {
                int blockSize = SplitMix64.NextInt(48);
                int mine = sanitizer.Alloc(blockSize, "differential");
                int theirs = oracle.Alloc(blockSize);
                if (mine != theirs) mismatches++;
                if (mine < 0) allocationFailures++;
                else blocks.Add((mine, blockSize));
                continue;
            }
            if (blocks.Count == 0) continue;
            (int pointer, int size) = blocks[SplitMix64.NextInt(blocks.Count)];
            if (choice < 85)
            {
                // Most accesses target the payload neighbourhood; the rest are pushed further out to hit redzones.
                bool write = SplitMix64.NextInt(100) < 72;
                int address = SplitMix64.NextInt(100) < 80
                    ? pointer + SplitMix64.NextInt(size + 8) - 4
                    : pointer + SplitMix64.NextInt(96) - 48;
                byte value = 0;
                string observed = write
                    ? Describe(sanitizer.Store(address, (byte)op))
                    : Describe(sanitizer.Load(address, out value));
                string expected = oracle.Access(address, write);
                if (observed != expected) mismatches++;
                histogram.TryGetValue(observed, out int seen);
                histogram[observed] = seen + 1;
                if (observed == "ok") legal++;
                else opaque++;
            }
            else
            {
                string observed = Describe(sanitizer.Free(pointer));
                string expected = oracle.Free(pointer);
                if (observed != expected) mismatches++;
                if (observed == "ok") legal++;
                else opaque++;
            }
        }

        Check(mismatches == 0, "differential verdicts diverge in " + mismatches + " of " + operations + " operations");
        Check(Kind(histogram, "ok") >= 60 && Kind(histogram, "heap-buffer-overflow") >= 10
            && Kind(histogram, "use-after-free") >= 5 && Kind(histogram, "uninitialized-read") >= 5,
            "the workload exercised every detector and kept legal accesses plentiful: " + Summarize(histogram));
        Check(legal + opaque == 200 || legal + opaque >= 100,
            "workload mixed " + legal + " legal and " + opaque + " illegal accesses");
        Check(sanitizer.AddressabilityViolations() == 0, "addressability invariant holds after the workload");
        Check(sanitizer.Findings.Count == opaque + allocationFailures,
            "one finding per failure: " + sanitizer.Findings.Count + " vs " + opaque + " accesses + "
            + allocationFailures + " exhausted allocations");
        Check(sanitizer.LeakReport().Count == sanitizer.LiveCount(), "leak report covers every live block");
        Check(sanitizer.Findings.TrueForAll(f => f.Kind != "heap-buffer-overflow" || f.AllocationId > 0),
            "overflow findings are attributed to a block");
        Console.WriteLine("  differential: " + operations + " ops, " + legal + " legal, " + opaque
            + " detected, " + sanitizer.LiveCount() + " blocks still live");
    }

    private static string Describe(Finding? finding) => finding == null ? "ok" : finding.Kind;

    private static int Kind(Dictionary<string, int> histogram, string kind)
        => histogram.TryGetValue(kind, out int count) ? count : 0;

    private static string Summarize(Dictionary<string, int> histogram)
    {
        var parts = new List<string>();
        foreach (KeyValuePair<string, int> pair in histogram) parts.Add(pair.Key + "=" + pair.Value);
        parts.Sort(StringComparer.Ordinal);
        return string.Join(" ", parts);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
