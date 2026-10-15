// Task 254 - Implement bloom filter insert/check (probabilistic). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;

public static class Solution
{
    private const int MBits = 1024;
    private const int KHashes = 7;
    // Deterministic values for the exact test set below (cross-language check).
    private const int ExpectedBits = 503;
    private const int ExpectedFp = 74;

    // FNV-1a, 32-bit; uint arithmetic wraps exactly like the other languages' 32-bit math.
    private static uint Fnv1a(string text)
    {
        uint h = 2166136261u;
        foreach (char ch in text)
        {
            h ^= ch;
            h *= 16777619u;
        }
        return h;
    }

    // MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed for double hashing.
    private static uint Fmix32(uint x)
    {
        x ^= x >> 16;
        x *= 0x85EBCA6Bu;
        x ^= x >> 13;
        x *= 0xC2B2AE35u;
        x ^= x >> 16;
        return x;
    }

    private static int[] Positions(string item, int bits, int hashes)
    {
        long h1 = Fmix32(Fnv1a(item));
        long h2 = Fmix32(Fnv1a(item + "#")) | 1L;
        var outArr = new int[hashes];
        for (int i = 0; i < hashes; i++) outArr[i] = (int)((h1 + i * h2) % bits);
        return outArr;
    }

    private sealed class BloomFilter
    {
        private readonly int _m;
        private readonly int _k;
        private readonly bool[] _bits;

        public BloomFilter(int m, int k)
        {
            _m = m;
            _k = k;
            _bits = new bool[m];
        }

        public void Add(string item)
        {
            foreach (int p in Positions(item, _m, _k)) _bits[p] = true;
        }

        public bool MaybeContains(string item)
        {
            foreach (int p in Positions(item, _m, _k)) if (!_bits[p]) return false;
            return true;
        }

        public int Popcount()
        {
            int count = 0;
            foreach (bool b in _bits) if (b) count++;
            return count;
        }
    }

    public static void Main()
    {
        var f = new BloomFilter(MBits, KHashes);
        Check(f.Popcount() == 0, "empty filter");
        foreach (string probe in new[] { "item-0", "x", "hello" })
        {
            Check(!f.MaybeContains(probe), "miss on empty: " + probe);
        }

        for (int i = 0; i < 100; i++) f.Add("item-" + i);
        for (int i = 0; i < 100; i++) Check(f.MaybeContains("item-" + i), "no false negative: item-" + i);
        int filled = f.Popcount();
        f.Add("item-0");
        Check(f.Popcount() == filled, "insert is idempotent");

        int falsePositives = 0;
        for (int i = 0; i < 10000; i++) if (f.MaybeContains("other-" + i)) falsePositives++;
        Check(f.Popcount() == ExpectedBits, "bits set agree across languages");
        Check(falsePositives == ExpectedFp, "false positives agree across languages");
        // Documented bound: theory gives (1 - e^(-k*n/m))^k ~ 0.7% for n=100, m=1024, k=7;
        // the measured rate must stay under 2% (200 out of 10000 queries).
        Check(falsePositives <= 200, "measured false-positive rate under the documented bound");

        var small = new BloomFilter(256, 4);
        foreach (string item in new[] { "a", "b", "c" }) small.Add(item);
        foreach (string item in new[] { "a", "b", "c" }) Check(small.MaybeContains(item), "small filter: " + item);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
