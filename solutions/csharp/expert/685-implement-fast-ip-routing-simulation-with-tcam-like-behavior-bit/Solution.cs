// Task 685 - Implement fast IP routing simulation with TCAM-like behavior (bitwise). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A software model of a ternary CAM (TCAM) forwarding table. Every entry is a (value, mask,
// port-range) triple plus a next hop, and the match semantics are exactly the hardware ones:
//
//   * the whole key is compared against an entry with ONE bitwise step:
//         mismatch = (key ^ value) & mask        (0 means "match line asserted")
//     which is how a TCAM spends one word per entry instead of one comparator per bit;
//   * wildcard bits are mask bits set to 0, so a prefix /L is just a mask with L ones;
//   * the source-port window is tested with the unsigned-subtraction range trick
//         inRange(x, lo, hi)  ==  ((x - lo) <= (hi - lo))   in 16-bit arithmetic,
//     which is the standard way to fold a range field into a masked comparison;
//   * the hardware priority encoder is modelled by collecting the match lines of 64 entries
//     into one 64-bit word and taking the lowest set bit (TrailingZeroCount), so the result is
//     the lowest-index match - TCAM semantics, not longest-prefix semantics. Ordering the table
//     by decreasing prefix length turns the same table into an LPM table (checked below).
//
// A chunk is additionally given a genuinely word-parallel fast path: OR-ing every entry's
// masked mismatch tells us at once whether the whole chunk matches, in which case the priority
// encoder can stop after a single reduction. Documented simplifications: a 32-bit IPv4 address
// plus a 16-bit port is packed into one 64-bit key word (real TCAMs are 144-576 bits wide, but
// the per-entry semantics are identical) and the "parallel" comparison is issued sequentially
// by the CPU - the parallel match lines, the priority encoder and the cost model are what is
// simulated. All results are cross-checked against a literal per-entry predicate scan.
using System;
using System.Collections.Generic;
using System.Numerics;

internal sealed class TcamTable
{
    private const int MaxEntries = 4096;

    private readonly uint[] _addr = new uint[MaxEntries];
    private readonly int[] _length = new int[MaxEntries];
    private readonly int[] _portLo = new int[MaxEntries];
    private readonly int[] _portHi = new int[MaxEntries];
    private readonly bool[] _usesPort = new bool[MaxEntries];
    private readonly int[] _nextHop = new int[MaxEntries];
    private readonly bool[] _valid = new bool[MaxEntries];
    private readonly ulong[] _value = new ulong[MaxEntries];   // value << 32 (address field only)
    private readonly ulong[] _mask = new ulong[MaxEntries];

    private int _count;

    public int Count => _count;
    public int Capacity => MaxEntries;

    public static uint MaskFor(int length)
    {
        if (length < 0 || length > 32) throw new ArgumentOutOfRangeException(nameof(length));
        return length == 0 ? 0u : 0xFFFFFFFFu << (32 - length);
    }

    /// <summary>Packs the IPv4 address in bits 63..32 and the port in bits 31..16.</summary>
    public static ulong Pack(uint address, int port)
    {
        if (port < 0 || port > 65535) throw new ArgumentOutOfRangeException(nameof(port));
        return ((ulong)address << 32) | ((ulong)(uint)port << 16);
    }

    /// <summary>The single bitwise step a TCAM performs per entry; 0 means the match line fires.</summary>
    public static ulong Mismatch(ulong key, ulong value, ulong mask) => (key ^ value) & mask;

    /// <summary>Unsigned wrap-around range test, the classic TCAM range encoding.</summary>
    public static bool InRange(int value, int low, int high)
    {
        if (low < 0 || high > 65535 || low > high) throw new ArgumentOutOfRangeException(nameof(low));
        return (uint)(value - low) <= (uint)(high - low);
    }

    public int Add(uint prefix, int length, int nextHop) => Add(prefix, length, nextHop, 0, 65535);

    /// <summary>Appends an entry at the next priority slot (index order == priority order).</summary>
    public int Add(uint prefix, int length, int nextHop, int portLow, int portHigh)
    {
        if (_count == MaxEntries) throw new InvalidOperationException("table full");
        uint mask = MaskFor(length);
        if ((prefix & ~mask) != 0) throw new ArgumentException("prefix has bits outside its length", nameof(prefix));
        if (portLow < 0 || portHigh > 65535 || portLow > portHigh)
            throw new ArgumentException("port window must lie inside 0..65535", nameof(portHigh));
        int slot = _count++;
        _addr[slot] = prefix;
        _length[slot] = length;
        _nextHop[slot] = nextHop;
        _portLo[slot] = portLow;
        _portHi[slot] = portHigh;
        _usesPort[slot] = !(portLow == 0 && portHigh == 65535);
        _mask[slot] = (ulong)mask << 32;
        _value[slot] = (ulong)(prefix & mask) << 32;
        _valid[slot] = true;
        return slot;
    }

    public void Invalidate(int slot) => _valid[slot] = false;

    public void Revalidate(int slot) => _valid[slot] = true;

    /// <summary>True when every entry of the 64-entry chunk beginning at <paramref name="start"/> matches.</summary>
    public bool AllEntriesMatch(ulong key, int start)
    {
        // Word-parallel reduction: OR of the masked mismatches. Zero => every entry matched.
        ulong accumulated = 0;
        int end = Math.Min(start + 64, _count);
        for (int i = start; i < end; i++)
        {
            if (!_valid[i]) continue;
            accumulated |= Mismatch(key, _value[i], _mask[i]);
            if (_usesPort[i] && !InRange((int)((key >> 16) & 0xFFFFu), _portLo[i], _portHi[i])) return false;
        }
        return accumulated == 0;
    }

    /// <summary>TCAM lookup: the lowest-index matching entry wins, or -1.</summary>
    public int Lookup(uint address, int port) => Lookup(address, port, out _, out _);

    public int Lookup(uint address, int port, out int entriesExamined, out int wordsTouched)
    {
        ulong key = Pack(address, port);
        entriesExamined = 0;
        wordsTouched = 0;
        for (int start = 0; start < _count; start += 64)
        {
            wordsTouched++;
            int end = Math.Min(start + 64, _count);
            if (AllEntriesMatch(key, start))
            {
                // Fast path: one reduction proves the whole chunk matches, so the encoder simply
                // returns the first live entry of the chunk.
                for (int i = start; i < end; i++)
                {
                    if (_valid[i]) return _nextHop[i];
                }
            }
            ulong hits = 0;
            for (int i = start; i < end; i++)
            {
                entriesExamined++;
                if (!_valid[i]) continue;
                if (Mismatch(key, _value[i], _mask[i]) != 0) continue;
                if (_usesPort[i] && !InRange(port, _portLo[i], _portHi[i])) continue;
                hits |= 1UL << (i - start);
            }
            if (hits != 0) return _nextHop[start + BitOperations.TrailingZeroCount(hits)];
        }
        return -1;
    }

    /// <summary>Bitwise longest-prefix match: leading-zero count against the candidate prefix.</summary>
    public int LookupByLeadingZeros(uint address)
    {
        int bestLength = -1, bestHop = -1;
        for (int i = 0; i < _count; i++)
        {
            if (!_valid[i] || _length[i] <= bestLength) continue;
            int shared = BitOperations.LeadingZeroCount(address ^ _addr[i]);
            if (shared >= _length[i])
            {
                bestLength = _length[i];
                bestHop = _nextHop[i];
            }
        }
        return bestHop;
    }

    /// <summary>Reference semantics spelled out with plain predicates, no bit tricks.</summary>
    public int LookupLiteral(uint address, int port)
    {
        for (int i = 0; i < _count; i++)
        {
            if (!_valid[i]) continue;
            bool addressOk = true;
            for (int b = 0; b < _length[i]; b++)
            {
                if (GetBit(address, b) != GetBit(_addr[i], b)) { addressOk = false; break; }
            }
            if (!addressOk) continue;
            if (_usesPort[i] && (port < _portLo[i] || port > _portHi[i])) continue;
            return _nextHop[i];
        }
        return -1;
    }

    private static int GetBit(uint value, int index) => (int)((value >> (31 - index)) & 1u);

    /// <summary>Reference longest-prefix match ignoring masks entirely.</summary>
    public int LookupLiteralLpm(uint address, int port)
    {
        int bestLength = -1, bestHop = -1;
        for (int i = 0; i < _count; i++)
        {
            if (!_valid[i]) continue;
            bool addressOk = true;
            for (int b = 0; b < _length[i]; b++)
            {
                if (GetBit(address, b) != GetBit(_addr[i], b)) { addressOk = false; break; }
            }
            if (!addressOk) continue;
            if (_usesPort[i] && (port < _portLo[i] || port > _portHi[i])) continue;
            if (_length[i] > bestLength) { bestLength = _length[i]; bestHop = _nextHop[i]; }
        }
        return bestHop;
    }

    /// <summary>Rebuilds the table with entries sorted by decreasing prefix length (LPM order).</summary>
    public TcamTable ToLongestPrefixFirst()
    {
        var order = new List<int>();
        for (int i = 0; i < _count; i++) if (_valid[i]) order.Add(i);
        order.Sort((a, b) =>
        {
            int byLength = _length[b].CompareTo(_length[a]);
            return byLength != 0 ? byLength : a.CompareTo(b);
        });
        var result = new TcamTable();
        foreach (int i in order) result.Add(_addr[i], _length[i], _nextHop[i], _portLo[i], _portHi[i]);
        return result;
    }
}

internal static class Rng685
{
    private static ulong _state = 0x243F6A8885A308D3UL;

    public static void Seed(ulong seed) => _state = seed;

    public static ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static uint NextUInt() => (uint)Next();
    public static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

public static class Solution
{
    public static void Main()
    {
        var empty = new TcamTable();
        Check(empty.Lookup(0x0A000001u, 80) == -1, "empty TCAM misses");
        Check(empty.AllEntriesMatch(TcamTable.Pack(1u, 1), 0), "vacuously all entries of an empty chunk match");

        // Exact host entry: one bitwise compare, no wildcard bits.
        var host = new TcamTable();
        host.Add(0x0A000001u, 32, 11);
        Check(host.Lookup(0x0A000001u, 22) == 11, "/32 hit");
        Check(host.Lookup(0x0A000002u, 22) == -1, "/32 miss on neighbour");
        Check(TcamTable.Mismatch(TcamTable.Pack(0x0A000001u, 22), (ulong)0x0A000001u << 32, (ulong)0xFFFFFFFFu << 32) == 0,
            "masked compare of a bit-exact entry is zero");

        // Default entry matches every key, whatever the port.
        var withDefault = new TcamTable();
        withDefault.Add(0u, 0, 1);
        Check(withDefault.Lookup(0xFFFFFFFFu, 65535) == 1, "/0 catches the maximum key");
        Check(withDefault.Lookup(0u, 0) == 1, "/0 catches the minimum key");
        Check(withDefault.AllEntriesMatch(TcamTable.Pack(0xDEADBEEFu, 7), 0), "chunk fast path agrees with the scan");

        // Priority, not specificity: the lowest index wins.
        var priority = new TcamTable();
        priority.Add(0x0A000000u, 8, 100);      // slot 0, broad
        priority.Add(0x0A000001u, 32, 200);     // slot 1, narrow
        Check(priority.Lookup(0x0A000001u, 9) == 100, "TCAM stops at the first match line");
        Check(priority.LookupLiteral(0x0A000001u, 9) == 100, "literal scan agrees on priority");
        var lpmOrdered = priority.ToLongestPrefixFirst();
        Check(lpmOrdered.Lookup(0x0A000001u, 9) == 200, "sorted table now yields the longest prefix");
        Check(lpmOrdered.LookupLiteralLpm(0x0A000001u, 9) == 200, "longest-prefix oracle agrees");

        // Invalidating an entry swallows the match line without disturbing the others.
        priority.Invalidate(0);
        Check(priority.Lookup(0x0A000001u, 9) == 200, "invalidated entry no longer matches");
        Check(priority.Lookup(0x0A000009u, 9) == -1, "invalidated entry leaves its prefix uncovered");
        priority.Revalidate(0);
        Check(priority.Lookup(0x0A000009u, 9) == 100, "revalidated entry matches again");

        // Port windows: the subtraction trick must equal a literal range test.
        var ranges = new (int Low, int High)[] { (0, 65535), (0, 0), (65535, 65535), (80, 80), (1024, 2048), (1, 3) };
        foreach ((int low, int high) in ranges)
        {
            foreach (int probe in new[] { 0, 1, 2, 3, 4, 80, 1023, 1024, 2048, 2049, 65534, 65535 })
            {
                bool trick = TcamTable.InRange(probe, low, high);
                bool literal = probe >= low && probe <= high;
                Check(trick == literal, $"range trick [{low},{high}] on {probe}");
            }
        }
        bool rejected = false;
        try { TcamTable.InRange(5, 10, 0); } catch (ArgumentOutOfRangeException) { rejected = true; }
        Check(rejected, "inverted range is rejected");

        var ports = new TcamTable();
        ports.Add(0x0A000000u, 8, 1);
        ports.Add(0x0A000000u, 8, 2, 80, 80);
        ports.Add(0x0A000000u, 8, 3, 443, 1024);
        Check(ports.Lookup(0x0A0000FFu, 80) == 1, "broad entry precedes the port-specific one");
        var portFirst = new TcamTable();
        portFirst.Add(0x0A000000u, 8, 3, 443, 1024);
        portFirst.Add(0x0A000000u, 8, 2, 80, 80);
        portFirst.Add(0x0A000000u, 8, 1);
        Check(portFirst.Lookup(0x0A0000FFu, 80) == 2, "port-specific entry wins for port 80");
        Check(portFirst.Lookup(0x0A0000FFu, 500) == 3, "port-specific entry wins inside 443..1024");
        Check(portFirst.Lookup(0x0A0000FFu, 1025) == 1, "outside the windows only the wildcard entry matches");

        // Random table: TCAM, literal scan and the leading-zero LPM must agree pairwise.
        Rng685.Seed(0x685C0FFEEUL);
        var random = new TcamTable();
        for (int i = 0; i < 120; i++)
        {
            int length = new[] { 0, 8, 16, 20, 24, 28, 30, 31, 32 }[Rng685.NextInt(9)];
            uint prefix = Rng685.NextUInt() & TcamTable.MaskFor(length);
            bool withPort = Rng685.NextInt(4) == 0;
            if (withPort)
            {
                int low = Rng685.NextInt(65536);
                int high = Math.Min(65535, low + Rng685.NextInt(4096));
                random.Add(prefix, length, 500 + i, low, high);
            }
            else
            {
                random.Add(prefix, length, 500 + i);
            }
        }
        for (int i = 0; i < 400; i++)
        {
            uint address = Rng685.NextUInt();
            int port = Rng685.NextInt(65536);
            int tcam = random.Lookup(address, port);
            Check(tcam == random.LookupLiteral(address, port), $"tcam==literal at i={i} addr={address:x8} port={port}");
        }
        var ordered = random.ToLongestPrefixFirst();
        for (int i = 0; i < 400; i++)
        {
            uint address = Rng685.NextUInt();
            int port = Rng685.NextInt(65536);
            Check(ordered.Lookup(address, port) == ordered.LookupLiteralLpm(address, port),
                $"sorted tcam==lpm at i={i} addr={address:x8} port={port}");
        }

        // Port-agnostic subset: leading-zero LPM must equal the mask mechanism.
        var maskOnly = new TcamTable();
        for (int i = 0; i < 120; i++)
        {
            int length = new[] { 0, 4, 8, 12, 16, 20, 24, 28, 32 }[Rng685.NextInt(9)];
            uint prefix = Rng685.NextUInt() & TcamTable.MaskFor(length);
            maskOnly.Add(prefix, length, 900 + i);
        }
        for (int i = 0; i < 300; i++)
        {
            uint address = Rng685.NextUInt();
            Check(maskOnly.LookupByLeadingZeros(address) == maskOnly.LookupLiteralLpm(address, 0),
                "leading-zero LPM == literal LPM");
        }

        // Cost model: a lookup touches ceil(entries/64) match-line words, not one word per entry.
        var big = new TcamTable();
        for (int i = 0; i < 1024; i++)
        {
            int length = i % 2 == 0 ? 24 : 16;
            uint prefix = Rng685.NextUInt() & TcamTable.MaskFor(length);
            big.Add(prefix, length, 5000 + i);
        }
        bool foundMiss = false;
        for (int attempt = 0; attempt < 4096 && !foundMiss; attempt++)
        {
            uint address = Rng685.NextUInt();
            int hop = big.Lookup(address, 80, out int examined, out int words);
            if (hop != -1) continue;
            foundMiss = true;
            Check(words == 16, "1024 entries cost 16 word-parallel chunks, saw " + words);
            Check(examined == big.Count, "a lookup that misses scans every entry, saw " + examined);
        }
        Check(foundMiss, "found an address covered by no entry");

        // 64 entries are exactly one chunk, so a first-chunk hit is resolved immediately.
        var oneChunk = new TcamTable();
        oneChunk.Add(0x0A0B0C00u, 24, 77);
        for (int i = 0; i < 63; i++) oneChunk.Add(Rng685.NextUInt(), 32, 900 + i);
        Check(oneChunk.Lookup(0x0A0B0C99u, 1, out int hitEntries, out int hitWords) == 77, "hit inside the first chunk");
        Check(hitWords == 1, "64 entries are one match-line word, saw " + hitWords);
        Check(hitEntries <= 64, "only the first chunk is examined, saw " + hitEntries);

        // Rejections: bits outside the prefix length and illegal ports are not silently accepted.
        bool badPrefix = false;
        try { big.Add(0x0A000001u, 8, 1); } catch (ArgumentException) { badPrefix = true; }
        Check(badPrefix, "prefix with bits outside its length is rejected");
        bool badPort = false;
        try { big.Add(0u, 0, 1, 0, 70000); } catch (ArgumentException) { badPort = true; }
        Check(badPort, "port outside 0..65535 is rejected");
        bool badLength = false;
        try { TcamTable.MaskFor(33); } catch (ArgumentOutOfRangeException) { badLength = true; }
        Check(badLength, "prefix length 33 is rejected");

        Console.WriteLine($"entries={big.Count} wordsPerLookup={hitWords} examined={hitEntries}");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
