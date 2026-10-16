// Task 677 - Implement cache coherence protocol simulator (MESI).
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A bus-based snooping MESI protocol, simulated at memory-transaction granularity: N cores each own a
// single-line (or few-line) cache, the shared bus serialises requests, and every transition is written out
// explicitly as the classic state machine
//     PrRd:  I -> S/E (BusRd, snoopers answer with a Flush or invalidate nothing)
//            S -> S, E -> E, M -> M (hit)
//     PrWr:  I -> M (BusRdX), S -> M (BusRdX invalidates the sharers), E -> M (silent upgrade), M -> M
// with a cache-to-cache transfer when a snooper owns the line in M (its dirty value is forwarded and it
// drops to S), so memory is only updated on a write-back or on eviction of a dirty line.
// Every access is checked against the protocol invariants, evaluated after each step of every run:
// (1) at most one cache holds a line in M; (2) a cache holding M or E implies every other cache holds I
// for that line; (3) a shared copy implies no cache holds M or E; (4) every read returns the last value
// written to that address, which is the point of coherence and is verified against a trivial single-memory
// reference model rather than against the cache hierarchy itself; (5) a dirty eviction writes the value
// back, so memory is *not* stale after eviction. On top of the invariants the file pins down the classic
// scenarios (read-shared produces invalidation-free sharing, a write invalidates exactly the sharers, an
// exclusive hit upgrades silently without a bus transaction, a remote write pulls the dirty line out of an
// owner) and then runs random access sequences against the reference model, counting bus traffic.
// The model is intentionally single-line-per-cache so that eviction is exercised on every miss.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic access-sequence generation.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal enum MesiState { Invalid, Shared, Exclusive, Modified }

internal sealed class CacheLine
{
    internal MesiState State = MesiState.Invalid;
    internal long Tag = -1;
    internal int Value;
}

internal sealed class Bus
{
    internal int Transactions;
    internal int Flushes;            // cache-to-cache transfers
    internal int WriteBacks;
    internal int Invalidations;
}

internal sealed class Mesi
{
    internal readonly CacheLine[] Caches;
    internal readonly Dictionary<long, int> Memory = new Dictionary<long, int>();
    internal readonly Bus Bus = new Bus();
    internal int Violations;
    internal string FirstViolation = "";

    internal Mesi(int cores)
    {
        Caches = new CacheLine[cores];
        for (int core = 0; core < cores; core++) Caches[core] = new CacheLine();
    }

    internal void Seed(long tag, int value) => Memory[tag] = value;

    private int MemoryValue(long tag)
    {
        int value;
        return Memory.TryGetValue(tag, out value) ? value : 0;
    }

    /// True when any *other* cache holds the line (in any non-invalid state).
    private bool IsSharedElsewhere(long tag, int requester)
    {
        for (int core = 0; core < Caches.Length; core++)
        {
            if (core == requester) continue;
            if (Caches[core].State != MesiState.Invalid && Caches[core].Tag == tag) return true;
        }
        return false;
    }

    private int Owner(long tag, int requester)
    {
        for (int core = 0; core < Caches.Length; core++)
        {
            if (core == requester) continue;
            if (Caches[core].State == MesiState.Modified && Caches[core].Tag == tag) return core;
        }
        return -1;
    }

    /// Replaces whatever line the cache currently holds, writing it back when it is dirty.
    private void Evict(int core)
    {
        CacheLine line = Caches[core];
        if (line.State == MesiState.Invalid) return;
        if (line.State == MesiState.Modified)
        {
            Memory[line.Tag] = line.Value;
            Bus.WriteBacks++;
        }
        line.State = MesiState.Invalid;
        line.Tag = -1;
    }

    internal int Read(int core, long tag)
    {
        Bus.Transactions++;
        CacheLine line = Caches[core];
        if (line.State != MesiState.Invalid && line.Tag == tag)
        {
            return line.Value;                       // a hit, including on a dirty exclusive line
        }

        Evict(core);
        int owner = Owner(tag, core);
        int value;
        if (owner >= 0)
        {
            value = Caches[owner].Value;             // cache-to-cache transfer ...
            Memory[tag] = value;                     // ... which also refreshes memory
            Bus.WriteBacks++;
            Caches[owner].State = MesiState.Shared;
            Bus.Flushes++;
        }
        else
        {
            value = MemoryValue(tag);
        }
        bool shared = owner >= 0 || IsSharedElsewhere(tag, core);
        // A snooper in E must drop to S when another cache starts sharing the line.
        for (int other = 0; other < Caches.Length; other++)
        {
            if (other == core) continue;
            if (Caches[other].State == MesiState.Exclusive && Caches[other].Tag == tag)
            {
                Caches[other].State = MesiState.Shared;
            }
        }
        line.Tag = tag;
        line.State = shared ? MesiState.Shared : MesiState.Exclusive;
        line.Value = value;
        return value;
    }

    internal void Write(int core, long tag, int value)
    {
        Bus.Transactions++;
        CacheLine line = Caches[core];
        if (line.State == MesiState.Modified && line.Tag == tag)
        {
            line.Value = value;                      // pure hit, no bus traffic beyond the request
            return;
        }
        if (line.State == MesiState.Exclusive && line.Tag == tag)
        {
            line.State = MesiState.Modified;         // silent upgrade
            line.Value = value;
            return;
        }

        bool hadIt = line.State != MesiState.Invalid && line.Tag == tag;
        if (!hadIt) Evict(core);
        int owner = Owner(tag, core);
        if (owner >= 0)
        {
            // The owner's dirty value is the newest; it must be forwarded before being invalidated.
            Memory[tag] = Caches[owner].Value;
            Bus.Flushes++;
        }
        for (int other = 0; other < Caches.Length; other++)
        {
            if (other == core) continue;
            if (Caches[other].State != MesiState.Invalid && Caches[other].Tag == tag)
            {
                Caches[other].State = MesiState.Invalid;
                Bus.Invalidations++;
            }
        }
        line.Tag = tag;
        line.State = MesiState.Modified;
        line.Value = value;
        if (!hadIt) Bus.Transactions++;              // an upgrade needs BusRdX, a miss needs BusRd + BusRdX
    }

    /// Protocol invariants; called after every operation of every run.
    internal void Validate()
    {
        for (int a = 0; a < Caches.Length; a++)
        {
            for (int b = 0; b < Caches.Length; b++)
            {
                if (a == b) continue;
                if (Caches[a].State == MesiState.Invalid) continue;
                if (Caches[b].State == MesiState.Invalid || Caches[b].Tag != Caches[a].Tag) continue;
                if (Caches[a].State == MesiState.Modified)
                {
                    Fail("cache " + b + " keeps the line that " + a + " holds in M");
                }
                if (Caches[a].State == MesiState.Exclusive)
                {
                    Fail("cache " + b + " keeps the line that " + a + " holds exclusively");
                }
            }
        }
        var modifiedPerTag = new Dictionary<long, int>();
        foreach (CacheLine line in Caches)
        {
            if (line.State != MesiState.Modified) continue;
            int count;
            modifiedPerTag.TryGetValue(line.Tag, out count);
            modifiedPerTag[line.Tag] = count + 1;
        }
        foreach (KeyValuePair<long, int> pair in modifiedPerTag)
        {
            if (pair.Value > 1) Fail("line " + pair.Key + " is modified in " + pair.Value + " caches");
        }
    }

    private void Fail(string message)
    {
        Violations++;
        if (FirstViolation.Length == 0) FirstViolation = message;
    }

    internal string Explain(int core)
    {
        CacheLine line = Caches[core];
        switch (line.State)
        {
            case MesiState.Invalid: return "I";
            case MesiState.Shared: return "S";
            case MesiState.Exclusive: return "E";
            default: return "M";
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // ---- read-only sharing -----------------------------------------------------------------
        var sharing = new Mesi(3);
        sharing.Seed(1, 42);
        Check(sharing.Read(0, 1) == 42, "core 0 reads the initial value");
        Check(sharing.Explain(0) == "E", "the first reader of an unshared line gets it exclusively (E)");
        Check(sharing.Read(1, 1) == 42, "core 1 shares it");
        Check(sharing.Explain(1) == "S" && sharing.Explain(0) == "S", "both copies are shared afterwards");
        Check(sharing.Bus.Invalidations == 0, "read-only sharing needs no invalidations");
        Check(sharing.Read(2, 1) == 42 && sharing.Explain(2) == "S", "a third reader also ends up sharing");
        sharing.Validate();
        Check(sharing.Violations == 0, "invariants hold after read-only sharing: " + sharing.FirstViolation);

        // ---- write invalidates exactly the sharers ---------------------------------------------
        var writing = new Mesi(3);
        writing.Seed(1, 7);
        writing.Read(0, 1);
        writing.Read(1, 1);
        writing.Read(2, 1);
        Check(writing.Explain(0) == "S" && writing.Explain(1) == "S" && writing.Explain(2) == "S",
              "three cores share the line");
        int before = writing.Bus.Invalidations;
        writing.Write(0, 1, 99);
        Check(writing.Explain(0) == "M", "the writer owns the line in M");
        Check(writing.Explain(1) == "I" && writing.Explain(2) == "I", "the other copies are invalidated");
        Check(writing.Bus.Invalidations - before == 2, "exactly the two sharers were invalidated (got "
              + (writing.Bus.Invalidations - before) + ")");
        Check(writing.Read(1, 1) == 99, "a subsequent read by another core sees the new value");
        Check(writing.Explain(1) == "S" && writing.Explain(0) == "S",
              "and pulls the dirty line out of the owner, which drops to S");
        writing.Validate();
        Check(writing.Violations == 0, "invariants hold after write-invalidate: " + writing.FirstViolation);

        // ---- a silent upgrade needs no bus traffic ----------------------------------------------
        var silent = new Mesi(2);
        silent.Seed(5, 3);
        silent.Read(0, 5);
        Check(silent.Explain(0) == "E", "core 0 holds the line exclusively");
        int transactions = silent.Bus.Transactions;
        silent.Write(0, 5, 4);
        Check(silent.Explain(0) == "M", "writing an exclusive line upgrades it to M");
        Check(silent.Bus.Transactions == transactions + 1, "and needs no extra bus transaction beyond the request");
        Check(silent.Bus.Invalidations == 0, "nobody else held a copy, so nobody was invalidated");
        silent.Validate();
        Check(silent.Violations == 0, "invariants hold after the silent upgrade");

        // ---- eviction of a dirty line writes it back --------------------------------------------
        var evicting = new Mesi(1);
        evicting.Seed(1, 1);
        evicting.Seed(2, 2);
        evicting.Write(0, 1, 11);
        Check(evicting.Explain(0) == "M", "core 0 owns line 1");
        evicting.Read(0, 2);
        Check(evicting.Bus.WriteBacks == 1, "replacing the dirty line wrote it back once");
        Check(evicting.Explain(0) == "E", "and the new line was fetched exclusively");
        evicting.Validate();
        Check(evicting.Violations == 0, "invariants hold after the dirty eviction");

        // ---- randomised access sequences against a trivial reference model ----------------------
        ulong state = 677001UL;
        int runs = 30;
        long comparisons = 0;
        int totalInvalidations = 0;
        int totalFlushes = 0;
        for (int run = 0; run < runs; run++)
        {
            int cores = 2 + Rng.NextBelow(ref state, 3);
            var cache = new Mesi(cores);
            var reference = new Dictionary<long, int>();
            for (long tag = 0; tag < 3; tag++)
            {
                int initial = Rng.NextBelow(ref state, 50);
                cache.Seed(tag, initial);
                reference[tag] = initial;
            }
            for (int access = 0; access < 60; access++)
            {
                int core = Rng.NextBelow(ref state, cores);
                long tag = Rng.NextBelow(ref state, 3);
                if (Rng.NextBelow(ref state, 3) == 0)
                {
                    int value = Rng.NextBelow(ref state, 1000);
                    cache.Write(core, tag, value);
                    reference[tag] = value;
                }
                else
                {
                    int seen = cache.Read(core, tag);
                    Check(seen == reference[tag], "run " + run + " access " + access + ": core " + core
                          + " read " + seen + " but the last written value is " + reference[tag]);
                    comparisons++;
                }
                cache.Validate();
                Check(cache.Violations == 0, "run " + run + " access " + access + ": invariants hold ("
                      + cache.FirstViolation + ")");
            }
            totalInvalidations += cache.Bus.Invalidations;
            totalFlushes += cache.Bus.Flushes;
        }
        Check(comparisons >= 600, "the random runs performed " + comparisons + " verified reads");
        Check(totalInvalidations > 0 && totalFlushes > 0, "the runs exercised invalidation (" + totalInvalidations
              + ") and cache-to-cache transfers (" + totalFlushes + ")");

        // ---- memory is not stale once every line has been evicted -------------------------------
        var finalMemory = new Mesi(2);
        finalMemory.Seed(1, 0);
        finalMemory.Write(0, 1, 77);
        finalMemory.Read(1, 1);
        Check(finalMemory.Read(1, 1) == 77, "the second core sees the dirty value through a transfer");
        finalMemory.Validate();
        Check(finalMemory.Violations == 0, "invariants hold in the two-core transfer case");

        // ---- degenerate case: a single core never needs coherence traffic -----------------------
        var single = new Mesi(1);
        single.Seed(1, 5);
        Check(single.Read(0, 1) == 5, "a single-core cache reads correctly");
        single.Write(0, 1, 6);
        Check(single.Bus.Invalidations == 0, "and never invalidates anybody");
        Check(single.Bus.Flushes == 0, "and never needs a cache-to-cache transfer");
        single.Validate();
        Check(single.Violations == 0, "a single core satisfies every invariant trivially");

        Console.WriteLine("random runs: " + runs + ", verified reads " + comparisons);
        Console.WriteLine("invalidations " + totalInvalidations + ", cache-to-cache transfers " + totalFlushes);
        Console.WriteLine("write-invalidate run: " + writing.Bus.WriteBacks + " write-backs");
        Console.WriteLine("PASS");
    }
}
