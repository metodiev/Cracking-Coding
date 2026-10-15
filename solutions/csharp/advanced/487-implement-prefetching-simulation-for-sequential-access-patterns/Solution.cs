// Task 487 - Implement prefetching simulation for sequential access patterns. (Tier: Advanced, CTCI Ch 14)
//
// Simulated memory with documented semantics:
// - Address space is [0, bound); every access and every prefetched address must be inside it - the
//   prefetcher clips at bound-1 so it can never read past the array.
// - Sequential pattern detector: the first access of a run is only recorded; every following access
//   with addr == last + 1 continues the run and issues speculative prefetches for
//   addr + 1 .. min(addr + depth, bound - 1). Any non-consecutive access restarts the detector.
// - A prefetch "hit" is an access to an address sitting in the prefetched set (one entry per
//   address); a hit consumes the entry. Documented lower bound: a cold run of length L >= 2 yields
//   at least L - 2 prefetch hits (the run's first access and its immediate successor cannot be
//   covered), so a fresh run of length L scores (L-2)/L (0.9375 at L = 32).
//
// The test pins hand cases, then asserts on sequential traces that the per-segment prefetch hit
// ratio is >= 0.9 (documented threshold) and that no simulated read ever leaves [0, bound).
using System;
using System.Collections.Generic;

internal class Prefetcher
{
    private readonly int bound;
    private readonly int depth;
    private int last = -1;
    private bool started;
    public readonly HashSet<int> Pending = new HashSet<int>();
    public int Hits;
    public int Accesses;
    public int Issued;
    public int MaxPrefetch = -1;

    public Prefetcher(int bound, int depth)
    {
        this.bound = bound;
        this.depth = depth;
    }

    public bool Access(int addr)
    {
        if (addr < 0 || addr >= bound) throw new Exception("FAILED: access out of bounds " + addr);
        Accesses++;
        bool hit = Pending.Remove(addr);
        if (hit) Hits++;
        bool continues = started && addr == last + 1;
        last = addr;
        started = true;
        if (continues)
        {
            for (int nxt = addr + 1; nxt <= Math.Min(addr + depth, bound - 1); nxt++)
            {
                Pending.Add(nxt);
                Issued++;
                MaxPrefetch = Math.Max(MaxPrefetch, nxt);
            }
        }
        return hit;
    }
}

public static class Solution
{
    private static int seed = 487;

    private static int[] LcgValues(int count, int limit)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    private static bool SameSet(HashSet<int> set, params int[] expected)
    {
        if (set.Count != expected.Length) return false;
        foreach (int value in expected) if (!set.Contains(value)) return false;
        return true;
    }

    private static bool SameBools(bool[] a, bool[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        var prefetcher = new Prefetcher(10, 3);
        var observed = new bool[7];
        for (int addr = 0; addr < 7; addr++) observed[addr] = prefetcher.Access(addr);
        Check(SameBools(observed, new[] { false, false, true, true, true, true, true }), "hand vector");
        Check(prefetcher.Hits == 5, "hand hits");
        Check(SameSet(prefetcher.Pending, 7, 8, 9), "hand pending");
        Check(prefetcher.MaxPrefetch == 9, "hand max");
        Check(prefetcher.Issued == 18, "hand issued " + prefetcher.Issued);
        Check(prefetcher.Access(9), "stale hit");
        Check(prefetcher.Hits == 6, "stale hits");
        Check(SameSet(prefetcher.Pending, 7, 8), "pending after 9");
        Check(!prefetcher.Access(4), "backwards jump misses");
        Check(prefetcher.Hits == 6 && prefetcher.Issued == 18, "backwards counters");

        prefetcher = new Prefetcher(10, 3);
        observed = new bool[10];
        for (int addr = 0; addr < 10; addr++) observed[addr] = prefetcher.Access(addr);
        Check(SameBools(observed, new[] { false, false, true, true, true, true, true, true, true, true }), "clip vector");
        Check(prefetcher.Hits == 8, "clip hits");
        Check(prefetcher.MaxPrefetch == 9, "clip max");
        Check(prefetcher.Pending.Count == 0, "clip pending empty");

        prefetcher = new Prefetcher(64, 4);
        for (int addr = 0; addr < 40; addr++) prefetcher.Access(addr);
        Check(prefetcher.Hits == 38 && prefetcher.Accesses == 40, "fresh run " + prefetcher.Hits);
        Check(prefetcher.Hits * 10 >= 9 * prefetcher.Accesses, "fresh ratio");
        Check(prefetcher.MaxPrefetch == 43, "fresh max " + prefetcher.MaxPrefetch);

        prefetcher = new Prefetcher(128, 4);
        var segmentHits = new List<int>();
        int totalSegmentHits = 0;
        foreach (int jump in LcgValues(4, 80))
        {
            int before = prefetcher.Hits;
            for (int offset = 0; offset < 32; offset++) prefetcher.Access(jump + offset);
            int got = prefetcher.Hits - before;
            segmentHits.Add(got);
            Check(got >= 30, "segment hits " + jump + " " + got);
            Check(got * 10 >= 9 * 32, "segment ratio " + jump);
            totalSegmentHits += got;
        }
        Check(segmentHits.Count == 4 && segmentHits[0] == 30 && segmentHits[1] == 30
              && segmentHits[2] == 30 && segmentHits[3] == 30, "segment pins");
        Check(totalSegmentHits == 120, "segment total " + totalSegmentHits);
        Check(prefetcher.MaxPrefetch < 128, "mixed max");
        Check(prefetcher.Hits <= prefetcher.Accesses, "sane hits");
        Check(prefetcher.Hits * 10 >= 9 * prefetcher.Accesses, "mixed ratio");

        prefetcher = new Prefetcher(64, 5);
        seed = 488;
        var trace = new List<int>();
        foreach (int value in LcgValues(30, 64))
        {
            trace.Add(value);
            if (value + 2 < 64)
            {
                trace.Add(value + 1);
                trace.Add(value + 2);
            }
        }
        foreach (int addr in trace)
        {
            Check(addr >= 0 && addr < 64, "trace bounds");
            prefetcher.Access(addr);
        }
        Check(prefetcher.MaxPrefetch <= 63, "interleaved max");
        Check(prefetcher.Hits <= prefetcher.Accesses && prefetcher.Accesses == trace.Count, "interleaved counters");

        prefetcher = new Prefetcher(128, 4);
        seed = 487;
        foreach (int jump in LcgValues(4, 80))
        {
            for (int offset = 0; offset < 32; offset++) prefetcher.Access(jump + offset);
        }
        Check(prefetcher.Accesses == 128 && prefetcher.Issued == 496, "pins " + prefetcher.Accesses + " " + prefetcher.Issued);
        Check(prefetcher.Hits == 120, "pinned hits " + prefetcher.Hits);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
