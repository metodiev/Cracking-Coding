// Task 467 - Implement bloom filter with double hashing and test false positive. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Bloom filter with m = 2^16 bits and k = 7 probes via double hashing:
// h1 = mix32(key), h2 = mix32(key ^ 0x9E3779B9) | 1 (forced odd, hence coprime to
// the power-of-two m, so the 7 probes are pairwise distinct),
// probe i = (h1 + i*h2) mod m.
//
// Checks: a hand-verifiable small filter (single insert sets exactly 7 distinct bits
// and is idempotent), then insert 10000 distinct member keys (LCG values in
// [0, 65536)), verify all 10000 are reported present (zero false negatives), and
// test 10000 disjoint non-member keys (200000 + i) counting false positives. The
// documented bound is FP_MAX = 800; the theoretical rate (1 - e^(-k*n/m))^k ~ 0.0524
// gives ~524 expected and the measured count on this fixed key set is 488 (asserted
// <= 800 with a >= 444 sanity floor).
using System;
using System.Collections.Generic;

internal class BloomFilter
{
    public const int MBits = 1 << 16;
    public const int Probes = 7;

    private readonly byte[] data = new byte[MBits / 8];

    public static uint Mix32(uint x)
    {
        x ^= x >> 16;
        x *= 0x45D9F3Bu;
        x ^= x >> 16;
        x *= 0x45D9F3Bu;
        x ^= x >> 16;
        return x;
    }

    public int[] Positions(uint key)
    {
        long h1 = Mix32(key);
        long h2 = Mix32(key ^ 0x9E3779B9u) | 1L;
        var outPos = new int[Probes];
        for (int i = 0; i < Probes; i++) outPos[i] = (int)((h1 + (long)i * h2) % MBits);
        return outPos;
    }

    public void Add(uint key)
    {
        foreach (int pos in Positions(key)) data[pos >> 3] |= (byte)(1 << (pos & 7));
    }

    public bool Contains(uint key)
    {
        foreach (int pos in Positions(key))
            if (((data[pos >> 3] >> (pos & 7)) & 1) == 0) return false;
        return true;
    }

    public int Popcount()
    {
        int total = 0;
        foreach (byte b in data) total += System.Numerics.BitOperations.PopCount(b);
        return total;
    }
}

public static class Solution
{
    private const int FpMax = 800;
    private static int seed;

    private static int Lcg() => seed = (seed * 75 + 74) % 65537;

    public static void Main()
    {
        // Hand-verifiable small filter: 7 distinct probes, idempotent insert.
        var tiny = new BloomFilter();
        var distinct = new HashSet<int>(tiny.Positions(5));
        Check(distinct.Count == BloomFilter.Probes, "distinct probes");
        tiny.Add(5);
        Check(tiny.Popcount() == BloomFilter.Probes, "popcount after insert");
        Check(tiny.Contains(5), "member present");
        tiny.Add(5);
        Check(tiny.Popcount() == BloomFilter.Probes, "idempotent insert");
        Check(!tiny.Contains(6), "non-member absent here");

        // Workload: 10000 distinct members from the LCG, 10000 disjoint non-members.
        var bloom = new BloomFilter();
        seed = 3141592;
        var members = new uint[10000];
        int n = 0;
        var seen = new HashSet<int>();
        while (n < members.Length)
        {
            int value = Lcg();
            if (seen.Add(value)) members[n++] = (uint)value;
        }
        foreach (uint key in members) bloom.Add(key);
        foreach (uint key in members) Check(bloom.Contains(key), "zero false negatives");
        int falsePositives = 0;
        for (int i = 0; i < 10000; i++) if (bloom.Contains((uint)(200000 + i))) falsePositives++;
        Check(falsePositives <= FpMax, "fp bound " + falsePositives); // measured 488
        Check(falsePositives >= 444, "fp sanity floor " + falsePositives);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
