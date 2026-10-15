// Task 493 - Implement probabilistic data structure composition (bloom + cmsketch). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A SketchPipeline composes two sketches over the same item stream: a Bloom filter (relaxed
// membership) and a Count-Min sketch (frequency estimation with d rows and w counters).
// Membership(item) = bloom.MaybeContains(item) AND cms.Estimate(item) >= 1 - so a positive
// answer always implies an estimated frequency of at least 1, and a Bloom false positive whose
// every CMS counter is still empty is rejected (the pipeline tightens membership without ever
// introducing a false negative). Hashing is the stable task-254 scheme: h1 = Fmix32(Fnv1a(item)),
// h2 = Fmix32(Fnv1a(item + '#')) | 1 for the Bloom probes, and per-row Fmix32(Fnv1a(item + '|r'))
// for the CMS; all arithmetic is 32-bit unsigned integer work, identical in all four languages.
// Documented guarantees: CMS estimates never underestimate the true count and overestimate by
// at most epsilon*N with epsilon = e/w and delta = e^(-d) (w = 256, d = 4, N = 754 updates); on
// 5000 absent probes the Bloom rate (m = 1024, k = 7, n = 250 items, theory ~25.6%) stays under
// the documented 28% tolerance, and the pipeline rejects most of those Bloom hits because their
// CMS counters are still empty - with the same Bloom filter and a width-1024 CMS almost all are
// rejected. The exact popcounts and false-positive counts are asserted as cross-language
// constants.
using System;
using System.Collections.Generic;

internal class BloomFilter
{
    private readonly int m;
    private readonly int k;
    private readonly byte[] words;

    internal BloomFilter(int bits, int hashes)
    {
        m = bits;
        k = hashes;
        words = new byte[bits];
    }

    internal int[] Positions(string item)
    {
        long h1 = Solution.Fmix32(Solution.Fnv1a(item)) & 0xFFFFFFFFL;
        long h2 = (Solution.Fmix32(Solution.Fnv1a(item + "#")) & 0xFFFFFFFFL) | 1L;
        var result = new int[k];
        for (int i = 0; i < k; i++) result[i] = (int)((h1 + i * h2) % m);
        return result;
    }

    internal void Add(string item)
    {
        foreach (int p in Positions(item)) words[p] = 1;
    }

    internal bool MaybeContains(string item)
    {
        foreach (int p in Positions(item))
        {
            if (words[p] == 0) return false;
        }
        return true;
    }

    internal int Popcount()
    {
        int total = 0;
        foreach (byte word in words) total += word;
        return total;
    }
}

internal class CountMinSketch
{
    private readonly int w;
    private readonly int d;
    private readonly int[][] counters;

    internal int Width => w;

    internal CountMinSketch(int width, int rows)
    {
        w = width;
        d = rows;
        counters = new int[rows][];
        for (int row = 0; row < rows; row++) counters[row] = new int[width];
    }

    internal int Column(int row, string item)
    {
        return (int)((Solution.Fmix32(Solution.Fnv1a(item + "|" + row)) & 0xFFFFFFFFL) % w);
    }

    internal void Add(string item, int delta)
    {
        for (int row = 0; row < d; row++) counters[row][Column(row, item)] += delta;
    }

    internal int Estimate(string item)
    {
        int best = int.MaxValue;
        for (int row = 0; row < d; row++) best = Math.Min(best, counters[row][Column(row, item)]);
        return best;
    }

    internal int Total()
    {
        int total = 0;
        foreach (int[] row in counters)
        {
            foreach (int value in row) total += value;
        }
        return total;
    }
}

internal class SketchPipeline
{
    internal readonly BloomFilter Bloom;
    internal readonly CountMinSketch Cms;

    internal SketchPipeline() : this(1024, 7, 256, 4) { }

    internal SketchPipeline(int bloomBits, int bloomHashes, int cmsWidth, int cmsRows)
    {
        Bloom = new BloomFilter(bloomBits, bloomHashes);
        Cms = new CountMinSketch(cmsWidth, cmsRows);
    }

    internal void Add(string item, int count)
    {
        Bloom.Add(item);
        Cms.Add(item, count);
    }

    internal int Estimate(string item) => Cms.Estimate(item);

    /** True only when the Bloom filter accepts AND the CMS sees a nonzero count. */
    internal bool Membership(string item) => Bloom.MaybeContains(item) && Cms.Estimate(item) >= 1;
}

public static class Solution
{
    internal static uint Fnv1a(string text)
    {
        uint h = 2166136261u;
        foreach (char ch in text)
        {
            h ^= ch;
            h *= 16777619u;
        }
        return h;
    }

    /** MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed. */
    internal static uint Fmix32(uint x)
    {
        x ^= x >> 16;
        x *= 0x85EBCA6Bu;
        x ^= x >> 13;
        x *= 0xC2B2AE35u;
        x ^= x >> 16;
        return x;
    }

    /** Deterministic insertion stream: item key-<j> gets 1 + seed % 5 copies. */
    private static int[] LcgCounts(int itemCount, int seed)
    {
        var counts = new int[itemCount];
        for (int j = 0; j < itemCount; j++)
        {
            seed = (seed * 75 + 74) % 65537;
            counts[j] = 1 + seed % 5;
        }
        return counts;
    }

    public static void Main()
    {
        int[] counts = LcgCounts(250, 123);
        int totalUpdates = 0;
        foreach (int count in counts) totalUpdates += count;
        var pipeline = new SketchPipeline();
        for (int j = 0; j < counts.Length; j++) pipeline.Add("key-" + j, counts[j]);

        double slack = ((double)Math.E / pipeline.Cms.Width) * totalUpdates;
        for (int j = 0; j < counts.Length; j++)
        {
            string item = "key-" + j;
            Check(pipeline.Membership(item), "no false negatives");
            int estimate = pipeline.Estimate(item);
            Check(estimate >= counts[j], "CMS must never underestimate");
            Check(estimate <= counts[j] + slack, "CMS additive error bound");
            Check(pipeline.Bloom.MaybeContains(item), "bloom member");
        }
        for (int j = 0; j < counts.Length; j++)
        {
            string item = "key-" + j;
            if (pipeline.Membership(item)) Check(pipeline.Estimate(item) >= 1, "membership >= 1");
        }

        Check(pipeline.Estimate("never-seen") == 0, "cold estimate");
        Check(!pipeline.Membership("never-seen"), "cold membership");

        int bloomFp = 0;
        int pipelineFp = 0;
        for (int j = 0; j < 5000; j++)
        {
            string item = "probe-" + j;
            if (pipeline.Bloom.MaybeContains(item)) bloomFp++;
            if (pipeline.Membership(item)) pipelineFp++;
        }
        Check(bloomFp <= 1400, "documented Bloom tolerance");
        Check(pipelineFp <= bloomFp - 1000, "the pipeline must reject most Bloom false hits");
        Check(pipelineFp <= 200, "documented pipeline tolerance");

        var tight = new SketchPipeline(1024, 7, 1024, 4);
        for (int j = 0; j < counts.Length; j++) tight.Add("key-" + j, counts[j]);
        int tightBloomFp = 0;
        int tightPipelineFp = 0;
        for (int j = 0; j < 5000; j++)
        {
            string item = "probe-" + j;
            if (tight.Bloom.MaybeContains(item)) tightBloomFp++;
            if (tight.Membership(item)) tightPipelineFp++;
        }
        Check(tightPipelineFp < tightBloomFp, "cold counters must reject some Bloom hits");
        Check(tightPipelineFp <= tightBloomFp - 1000, "tightening factor");
        for (int j = 0; j < counts.Length; j++)
        {
            string item = "key-" + j;
            Check(tight.Membership(item), "tight member");
            Check(tight.Estimate(item) >= counts[j], "tight lower bound");
            Check(tight.Estimate(item) <= counts[j] + 1, "wider sketch error");
        }

        Check(totalUpdates == 754, "total updates " + totalUpdates);
        Check(pipeline.Bloom.Popcount() == 842, "popcount " + pipeline.Bloom.Popcount());
        Check(tight.Bloom.Popcount() == 842, "tight popcount");
        Check(bloomFp == 1296, "bloom fps " + bloomFp);
        Check(pipelineFp == 194, "pipeline fps " + pipelineFp);
        Check(tightBloomFp == 1296, "tight bloom fps");
        Check(tightPipelineFp == 5, "tight pipeline fps " + tightPipelineFp);

        var solo = new SketchPipeline();
        solo.Add("solo", 3);
        Check(solo.Membership("solo"), "solo member");
        Check(solo.Estimate("solo") == 3, "solo estimate");
        foreach (string probe in new[] { "solo#", "solo1", "sol", "proxy-7" })
        {
            if (solo.Membership(probe)) Check(solo.Estimate(probe) >= 1, "solo consistency");
        }
        Check(solo.Cms.Total() == 12, "solo cms total");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
