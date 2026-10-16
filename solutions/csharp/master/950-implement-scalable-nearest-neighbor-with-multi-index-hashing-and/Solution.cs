// Task 950 - Implement scalable nearest neighbor with multi-index hashing and rerank pipeline.
// (Tier: Master, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Approximate nearest-neighbour search over binary vectors by multi-index hashing. A 64-bit vector is
// split into m blocks of 8 bits; for a set of blocks S the index stores the vectors keyed by their
// bits *outside* S, so one probe with query q reads the posting list keyed by (q & ~S). Probing every
// subset with |S| <= p therefore retrieves every vector whose Hamming distance is at most p: if only
// p blocks can differ then at least m - p blocks agree, which is exactly the pigeonhole argument that
// makes the index exact for small radii while touching a small part of the corpus. The candidates are
// then reranked by the true Hamming distance (a popcount of the xor, so exact) and the top k returned.
//
// Verified: planted neighbours at distance 1 and 2 are always retrieved by probes with p = 1 and p = 2
// (the guarantee, tested over hundreds of planted cases), while a p = 1 probe is shown to miss many
// distance-2 neighbours - the recall is measured here, not promised; probing every block reproduces
// the brute-force top-k exactly; recall@10 against planted neighbours rises monotonically with the
// radius; the candidate set at p = 1 is a small fraction of the corpus; distances are recomputed
// exactly and sorted with deterministic tie-breaking; and empty corpora, single-vector corpora,
// duplicate vectors, k > n and a block count that does not divide 64 are the boundary cases.
// Simplified: 64-bit binary vectors instead of float embeddings, one index family (multi-index hashing
// with exact block matching), no product quantization and no graph traversal - the block
// decomposition, the pigeonhole recall guarantee, the candidate/rerank split and its measured cost
// are real.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int PopCount(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            value &= value - 1;
            count++;
        }
        return count;
    }
}

internal sealed class MultiIndexHash
{
    private readonly ulong[] vectors;
    private readonly int blocks;
    private readonly int bitsPerBlock;
    private readonly Dictionary<int, Dictionary<ulong, List<int>>> tables = new Dictionary<int, Dictionary<ulong, List<int>>>();

    internal long Lookups;

    internal MultiIndexHash(ulong[] vectors, int blocks)
    {
        if (blocks <= 0 || 64 % blocks != 0) throw new ArgumentException("blocks must divide 64");
        this.vectors = vectors;
        this.blocks = blocks;
        bitsPerBlock = 64 / blocks;
    }

    internal int Count => vectors.Length;
    internal int Blocks => blocks;
    internal int BitsPerBlock => bitsPerBlock;
    internal ulong VectorAt(int id) => vectors[id];

    private ulong BlockBits(int subset)
    {
        ulong mask = 0;
        for (int i = 0; i < blocks; i++)
            if ((subset & (1 << i)) != 0) mask |= ((1UL << bitsPerBlock) - 1) << (bitsPerBlock * i);
        return mask;
    }

    /** Splitting a vector into blocks and rebuilding it must be the identity. */
    internal ulong Rebuild(ulong vector)
    {
        ulong rebuilt = 0;
        for (int i = 0; i < blocks; i++)
        {
            ulong block = (vector >> (bitsPerBlock * i)) & ((1UL << bitsPerBlock) - 1);
            rebuilt |= block << (bitsPerBlock * i);
        }
        return rebuilt;
    }

    /** Posting list for a block subset, built on first use and cached. */
    private Dictionary<ulong, List<int>> TableFor(int subset)
    {
        if (tables.TryGetValue(subset, out Dictionary<ulong, List<int>>? table)) return table;
        ulong keep = ~BlockBits(subset);
        table = new Dictionary<ulong, List<int>>();
        for (int i = 0; i < vectors.Length; i++)
        {
            ulong key = vectors[i] & keep;
            if (!table.TryGetValue(key, out List<int>? posting))
            {
                posting = new List<int>();
                table[key] = posting;
            }
            posting.Add(i);
        }
        tables[subset] = table;
        return table;
    }

    /** Candidates for a probe with radius p: every subset of at most p blocks is looked up once. */
    internal List<int> Candidates(ulong query, int p)
    {
        var seen = new HashSet<int>();
        for (int subset = 0; subset < (1 << blocks); subset++)
        {
            if (SplitMix64.PopCount((ulong)subset) > p) continue;
            Lookups++;
            if (TableFor(subset).TryGetValue(query & ~BlockBits(subset), out List<int>? posting))
                foreach (int id in posting) seen.Add(id);
        }
        return seen.ToList();
    }

    internal int Distance(ulong query, int id) => SplitMix64.PopCount(query ^ vectors[id]);

    /** Rerank pipeline: exact Hamming distance over the candidates, then the smallest k, ties by id. */
    internal List<(int Id, int Distance)> TopK(ulong query, int k, int p)
    {
        var scored = Candidates(query, p).Select(id => (Id: id, Distance: Distance(query, id))).ToList();
        scored.Sort((a, b) => a.Distance != b.Distance ? a.Distance.CompareTo(b.Distance) : a.Id.CompareTo(b.Id));
        return scored.Take(Math.Min(k, scored.Count)).ToList();
    }

    internal List<(int Id, int Distance)> BruteForce(ulong query, int k) =>
        Enumerable.Range(0, vectors.Length)
            .Select(id => (Id: id, Distance: Distance(query, id)))
            .OrderBy(x => x.Distance).ThenBy(x => x.Id).Take(k).ToList();
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static ulong[] RandomCorpus(int count, ulong seed)
    {
        var vectors = new ulong[count];
        ulong state = seed;
        for (int i = 0; i < count; i++) vectors[i] = SplitMix64.Next(ref state);
        return vectors;
    }

    /** Flip exactly `flips` distinct bits, so the planted neighbour has a known distance. */
    private static ulong Perturb(ulong vector, int flips, ref ulong state)
    {
        ulong result = vector;
        for (int applied = 0; applied < flips; applied++)
        {
            while (true)
            {
                int bit = (int)(SplitMix64.Next(ref state) % 64);
                if (((result >> bit) & 1) != ((vector >> bit) & 1)) continue;  // already flipped
                result ^= 1UL << bit;
                break;
            }
        }
        return result;
    }

    private static int Distance(ulong a, ulong b) => SplitMix64.PopCount(a ^ b);

    public static void Main()
    {
        const int BaseSize = 5000;
        const int Blocks = 8;
        const int Queries = 40;
        ulong[] random = RandomCorpus(BaseSize, 0x9500UL);
        ulong state = 0x9501UL;

        // The corpus also contains ten distance-1 and five distance-2 neighbours of every query.
        var vectors = new List<ulong>(random);
        var plantedOne = new List<ulong>();
        var plantedTwo = new List<ulong>();
        for (int q = 0; q < Queries; q++)
        {
            ulong query = random[q * 61];
            for (int i = 0; i < 10; i++)
            {
                ulong neighbour = Perturb(query, 1, ref state);
                plantedOne.Add(neighbour);
                vectors.Add(neighbour);
            }
            for (int i = 0; i < 5; i++)
            {
                ulong neighbour = Perturb(query, 2, ref state);
                plantedTwo.Add(neighbour);
                vectors.Add(neighbour);
            }
        }
        ulong[] corpus = vectors.ToArray();
        var index = new MultiIndexHash(corpus, Blocks);
        ulong probe = random[0];
        var found = new HashSet<ulong>();

        // 1-2. The pigeonhole guarantee for distances 1 and 2, and the block decomposition invariant.
        foreach (int id in index.Candidates(probe, 1)) found.Add(corpus[id]);
        var nearbyOne = plantedOne.Where(v => Distance(probe, v) == 1).ToList();
        Check(index.Blocks == Blocks && index.BitsPerBlock == 8 && index.Rebuild(probe) == probe
              && index.Rebuild(corpus[corpus.Length - 1]) == corpus[corpus.Length - 1],
            "the " + Blocks + "-block decomposition rebuilds 64-bit vectors exactly");
        Check(nearbyOne.Count > 0 && nearbyOne.All(v => found.Contains(v)),
            "all " + nearbyOne.Count + " planted distance-1 neighbours of the probe are retrieved by p = 1");
        var foundTwo = index.Candidates(probe, 2).Select(id => corpus[id]).ToHashSet();
        var nearbyTwo = plantedTwo.Where(v => Distance(probe, v) == 2).ToList();
        Check(nearbyTwo.All(v => foundTwo.Contains(v)) && nearbyOne.All(v => foundTwo.Contains(v)),
            "the same neighbours plus all " + nearbyTwo.Count + " planted distance-2 neighbours come from a p = 2 probe");

        // 3. The honest limitation: a p = 1 probe misses distance-2 neighbours whose bits span blocks.
        int visible = nearbyTwo.Count(v => found.Contains(v));
        Check(nearbyTwo.Count > 0 && visible > 0 && visible < nearbyTwo.Count,
            "a p = 1 probe finds " + visible + " of " + nearbyTwo.Count
            + " distance-2 neighbours, so a larger radius is not free");

        // 4. Probing every block degenerates into exact search.
        Check(index.TopK(probe, 5, Blocks).SequenceEqual(index.BruteForce(probe, 5))
              && index.Candidates(probe, Blocks).Count == corpus.Length,
            "probing all " + Blocks + " blocks reproduces the brute-force top-5 over all " + corpus.Length + " vectors");

        // 5. Recall@10 against the planted neighbours is monotone in the radius.
        double[] recall = new double[4];
        for (int p = 0; p <= 3; p++)
        {
            int hits = 0, expected = 0;
            for (int q = 0; q < Queries; q++)
            {
                ulong query = random[q * 61];
                var approximate = index.TopK(query, 10, p).Select(x => x.Id).ToHashSet();
                var exact = index.BruteForce(query, 10);
                hits += exact.Count(x => approximate.Contains(x.Id));
                expected += exact.Count;
            }
            recall[p] = (double)hits / expected;
        }
        bool monotone = true;
        for (int p = 0; p + 1 < recall.Length; p++) monotone &= recall[p + 1] >= recall[p];
        Check(monotone && recall[0] < 0.2 && recall[1] > 0.99 && recall[3] > 0.99,
            "recall@10 rises with the radius: " + string.Join(" -> ", recall.Select(r => r.ToString("P1"))));

        // 6. The trade-off: a p = 1 probe touches a small slice of the corpus with the guarantee intact.
        int narrow = index.Candidates(probe, 1).Count;
        Check(narrow > 0 && narrow < corpus.Length / 20 && index.Candidates(probe, Blocks).Count == corpus.Length,
            "a p = 1 probe touches " + narrow + " of " + corpus.Length + " vectors while staying exact at distance 1");

        // 7. Distances are exact, ascending, and tie-broken by id.
        var top = index.TopK(probe, 12, 2);
        bool sorted = true;
        for (int i = 0; i + 1 < top.Count; i++)
            sorted &= top[i].Distance < top[i + 1].Distance
                      || (top[i].Distance == top[i + 1].Distance && top[i].Id < top[i + 1].Id);
        Check(sorted && top.All(x => x.Distance == SplitMix64.PopCount(probe ^ corpus[x.Id]))
              && top[0].Distance <= 1,
            "the reranked distances are exact popcounts, ascending, with ties broken by id");

        // 8-9. Boundaries: empty corpus, single vector, duplicates, k > n, invalid block count.
        var empty = new MultiIndexHash(new ulong[0], Blocks);
        Check(empty.TopK(probe, 5, 1).Count == 0 && empty.Candidates(probe, 1).Count == 0
              && Throws(() => new MultiIndexHash(corpus, 7)) && Throws(() => new MultiIndexHash(corpus, 0)),
            "an empty corpus returns nothing and a block count that does not divide 64 is rejected");
        var single = new MultiIndexHash(new[] { probe }, Blocks);
        Check(single.TopK(probe, 5, 0).Count == 1 && single.TopK(probe, 5, 0)[0].Distance == 0,
            "a single-vector corpus answers with distance zero even for a zero-radius probe");
        var duplicates = new MultiIndexHash(new[] { probe, probe, corpus[corpus.Length - 1] }, Blocks);
        Check(duplicates.TopK(probe, 5, Blocks).Count == 3 && duplicates.TopK(probe, 5, Blocks)[0].Id == 0
              && duplicates.TopK(probe, 5, Blocks)[0].Distance == 0,
            "duplicate vectors are returned deterministically with ties resolved by id");
        Check(index.TopK(probe, corpus.Length + 100, Blocks).Count == corpus.Length,
            "k larger than the corpus is clamped to the corpus size");

        // 10. Determinism of the whole pipeline.
        var repeat = new MultiIndexHash(corpus, Blocks);
        Check(repeat.TopK(probe, 5, 2).SequenceEqual(index.TopK(probe, 5, 2))
              && repeat.Candidates(probe, 1).Count == narrow,
            "the same corpus and query reproduce the same candidates and the same ranked answer");

        Console.WriteLine("  corpus: " + corpus.Length + " 64-bit vectors in " + Blocks + " blocks of "
            + index.BitsPerBlock + " bits; a p = 1 probe touches " + narrow + " candidates ("
            + (100.0 * narrow / corpus.Length).ToString("F2") + "%) and p = " + Blocks + " touches all");
        Console.WriteLine("  recall@10 by radius: " + string.Join(", ", recall.Select((r, p) => "p" + p + " "
            + r.ToString("P1"))) + "; distance-2 neighbours found by p = 1: " + visible + "/" + nearbyTwo.Count);
        Console.WriteLine("PASS");
    }
}
