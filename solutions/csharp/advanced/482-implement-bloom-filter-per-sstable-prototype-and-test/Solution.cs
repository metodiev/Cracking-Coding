// Task 482 - Implement bloom filter per SSTable prototype and test. (Tier: Advanced, CTCI Ch 14)
//
// Prototype of an LSM-style lookup: each SSTable owns a bloom filter sized to 10 bits per key with
// k = 7 double-hashed probes, and a sorted entry list for the exact confirmation step. Hashing is
// FNV-1a 32-bit; probe j uses (h1 + j * h2) % m with h2 = fnv1a(key + "!") | 1 (odd, so the probe
// sequence never collapses). Lookup consults every table's filter first (false positives only cost
// an exact binary search, never a wrong answer).
//
// Documented false-positive estimate (bits m, keys n, probes k): p = (1 - e^(-k*n/m))^k ~ 0.0082
// per table; for the combined union the estimate is 1 - prod(1 - p_t) ~ 0.0244. The test pins the
// exact deterministic false-positive counts and asserts the measured rate stays within a
// documented 3x-theory bound, plus zero false negatives for every inserted key.
using System;
using System.Collections.Generic;

internal class BloomFilter
{
    public readonly int M;
    public readonly int K;
    public readonly byte[] Words;
    public int Count;

    public BloomFilter(int bits, int hashes)
    {
        M = bits;
        K = hashes;
        Words = new byte[(bits + 7) / 8];
    }

    public static uint Fnv1a(string text)
    {
        uint h = 2166136261u;
        foreach (char ch in text)
        {
            h ^= ch;
            h = unchecked(h * 16777619u);
        }
        return h;
    }

    public long[] Probes(string key)
    {
        long h1 = Fnv1a(key);
        long h2 = Fnv1a(key + "!") | 1L;
        var idx = new long[K];
        for (int j = 0; j < K; j++) idx[j] = (h1 + j * h2) % M;
        return idx;
    }

    public void Add(string key)
    {
        foreach (long i in Probes(key)) Words[(int)(i >> 3)] |= (byte)(1 << (int)(i & 7));
        Count++;
    }

    public bool MaybeContains(string key)
    {
        foreach (long i in Probes(key))
        {
            if ((Words[(int)(i >> 3)] & (1 << (int)(i & 7))) == 0) return false;
        }
        return true;
    }
}

internal class SSTable
{
    public readonly List<string> Entries;
    public readonly BloomFilter Bloom;

    public SSTable(List<string> entries, int bitsPerKey, int hashes)
    {
        Entries = new List<string>(entries);
        Entries.Sort(StringComparer.Ordinal);
        Bloom = new BloomFilter(bitsPerKey * Entries.Count, hashes);
    }

    public SSTable Build(List<string> keys)
    {
        foreach (string key in keys) Bloom.Add(key);
        return this;
    }

    public bool ContainsExact(string key)
    {
        int lo = 0, hi = Entries.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            int cmp = string.CompareOrdinal(Entries[mid], key);
            if (cmp == 0) return true;
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    public bool Contains(string key) => Bloom.MaybeContains(key) && ContainsExact(key);
}

internal class Store
{
    private readonly List<SSTable> tables;

    public Store(List<SSTable> tables)
    {
        this.tables = tables;
    }

    public bool Get(string key)
    {
        foreach (SSTable table in tables) if (table.Contains(key)) return true;
        return false;
    }

    public bool BloomAny(string key)
    {
        foreach (SSTable table in tables) if (table.Bloom.MaybeContains(key)) return true;
        return false;
    }
}

public static class Solution
{
    private static int seed = 482;

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

    private static double TheoreticalFpRate(int keys, int bits, int hashes)
    {
        return Math.Pow(1.0 - Math.Exp(-hashes * (double)keys / bits), hashes);
    }

    private static List<SSTable> BuildStore(int[] sizes, int startSeed)
    {
        var tables = new List<SSTable>();
        seed = startSeed;
        for (int i = 0; i < sizes.Length; i++)
        {
            int size = sizes[i];
            var one = new List<string>();
            for (int j = 0; j < size; j++) one.Add("t" + i + ":" + j);
            seed = (seed * 75 + 74) % 65537;
            int rotation = seed % size;
            var keys = new List<string>();
            keys.AddRange(one.GetRange(rotation, size - rotation));
            keys.AddRange(one.GetRange(0, rotation));
            tables.Add(new SSTable(keys, 10, 7).Build(keys));
        }
        return tables;
    }

    private static int Popcount(byte[] bytes)
    {
        int total = 0;
        foreach (byte b in bytes)
        {
            int n = b;
            while (n != 0)
            {
                n &= n - 1;
                total++;
            }
        }
        return total;
    }

    public static void Main()
    {
        var tiny = new BloomFilter(64, 4);
        tiny.Add("alpha");
        Check(tiny.MaybeContains("alpha"), "tiny contains");
        Check(!new BloomFilter(64, 4).MaybeContains("alpha"), "empty rejects");

        var sizes = new[] { 150, 180, 210 };
        var tables = BuildStore(sizes, 482);
        var store = new Store(tables);
        for (int i = 0; i < 3; i++)
        {
            Check(tables[i].Entries.Count == sizes[i], "table size " + i);
            Check(tables[i].Bloom.M == sizes[i] * 10, "bloom bits " + i);
            Check(tables[i].Bloom.K == 7, "bloom hashes " + i);
        }

        int presentQueries = 0;
        foreach (SSTable table in tables)
        {
            foreach (string key in table.Entries)
            {
                Check(table.Bloom.MaybeContains(key), "no false negative " + key);
                Check(store.Get(key), "store finds " + key);
                presentQueries++;
            }
        }
        Check(presentQueries == 540, "present query count");

        var unionSet = new SortedSet<string>(StringComparer.Ordinal);
        foreach (SSTable table in tables) unionSet.UnionWith(table.Entries);
        var union = new List<string>(unionSet);
        seed = 9001;
        foreach (int value in LcgValues(200, 540))
        {
            string key = union[value];
            Check(store.Get(key) && store.BloomAny(key), "present probe " + key);
        }
        var absent = new List<string>();
        seed = 9002;
        foreach (int value in LcgValues(240, 65537)) absent.Add("absent:" + value);
        foreach (string key in absent)
        {
            Check(!unionSet.Contains(key), "absent build " + key);
            Check(!store.Get(key), "absent lookup " + key);
        }

        var perTableFp = new int[3];
        for (int i = 0; i < 3; i++)
        {
            foreach (string key in absent) if (tables[i].Bloom.MaybeContains(key)) perTableFp[i]++;
        }
        int combinedFp = 0;
        foreach (string key in absent) if (store.BloomAny(key)) combinedFp++;
        // Pinned from the deterministic construction; identical in all four languages.
        Check(perTableFp[0] == 2 && perTableFp[1] == 1 && perTableFp[2] == 1, "per-table fp");
        Check(combinedFp == 4, "combined fp " + combinedFp);

        double pUnionComplement = 1.0;
        for (int i = 0; i < 3; i++)
        {
            SSTable t = tables[i];
            double p = TheoreticalFpRate(sizes[i], t.Bloom.M, t.Bloom.K);
            pUnionComplement *= 1.0 - p;
            Check(perTableFp[i] <= 3 * p * absent.Count + 3, "theory bound " + i);
            Check(perTableFp[i] / (double)absent.Count <= 0.05, "rate bound " + i);
            Check((int)Math.Round(p * 10000) == 82, "theoretical p " + i);
        }
        double pUnion = 1.0 - pUnionComplement;
        Check(combinedFp <= 3 * pUnion * absent.Count + 3, "union bound");
        Check(combinedFp / (double)absent.Count <= 0.05, "union rate bound");
        Check((int)Math.Round(pUnion * 10000) == 244, "theoretical union " + pUnion);

        var setBits = new int[3];
        for (int i = 0; i < 3; i++) setBits[i] = Popcount(tables[i].Bloom.Words);
        Check(setBits[0] == 768 && setBits[1] == 905 && setBits[2] == 1052, "set bits");
        for (int i = 0; i < 3; i++) Check(setBits[i] < tables[i].Bloom.M, "not saturated " + i);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
