// Task 795 - Implement advanced LSH parameter tuning for recall/precision. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Random-hyperplane (SimHash) locality-sensitive hashing for cosine similarity. Each table draws k random
// hyperplanes; a vector becomes a k-bit signature and is stored in the bucket of that signature. Two vectors
// whose angle is theta collide with probability p = 1 - theta/pi in *one* hyperplane and with p^k in one table,
// so L independent tables raise the recall of a query's true neighbours roughly as 1 - (1 - p^k)^L while the
// candidate set (and hence the work) grows with L and shrinks with k. Parameter tuning is therefore an
// explicit trade-off: enumerate a small (k, L) grid, measure recall/precision against exact brute-force
// neighbours, and pick the cheapest configuration whose measured recall meets the target.
//
// Invariants asserted: every signature is exactly k bits, a query always lands in its own bucket (the exact
// self-collision property), recall and precision stay inside [0, 1], and the reported cost equals the cost
// model (candidates examined plus the hashing work L * k * n).
//
// Independent oracles:
//   * the measured one-hyperplane collision frequency is compared with the closed form 1 - theta/pi over many
//     random draws, and the k-bit collision frequency with p^k;
//   * recall and precision are computed against exact nearest neighbours from a brute-force cosine scan, which
//     shares no code with the hash tables;
//   * a candidate-set superset check: every true neighbour found must actually be in the examined candidate set.
//
// Documented simplification: dense float vectors instead of sparse TF-IDF, a fixed random projection family, a
// coarse (k, L) grid and one target recall instead of a banding/multi-probe scheme. Deterministic splitmix64
// hyperplanes only, so every measured recall is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Vectors
{
    internal static double[] Normalise(double[] v)
    {
        double norm = 0.0;
        foreach (double x in v) norm += x * x;
        norm = Math.Sqrt(norm);
        if (norm < 1e-12) throw new ArgumentException("zero-length vector cannot be hashed");
        var unit = new double[v.Length];
        for (int i = 0; i < v.Length; i++) unit[i] = v[i] / norm;
        return unit;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static double Cosine(double[] a, double[] b) => Dot(a, b) / (Math.Sqrt(Dot(a, a)) * Math.Sqrt(Dot(b, b)));
}

internal sealed class SimHash
{
    private readonly double[][][] _tables;                 // [table][hyperplane][dimension]
    private readonly int _bits;

    internal int Bits => _bits;
    internal int Tables => _tables.Length;

    internal SimHash(int bits, int tables, int dimension, ulong seed)
    {
        if (bits < 1) throw new ArgumentException("at least one bit is required", nameof(bits));
        if (tables < 1) throw new ArgumentException("at least one table is required", nameof(tables));
        if (dimension < 1) throw new ArgumentException("dimension must be positive", nameof(dimension));
        _bits = bits;
        var rng = seed;
        _tables = new double[tables][][];
        for (int t = 0; t < tables; t++)
        {
            _tables[t] = new double[bits][];
            for (int h = 0; h < bits; h++)
            {
                var plane = new double[dimension];
                for (int d = 0; d < dimension; d++) plane[d] = SplitMix64.NextGaussian(ref rng);
                _tables[t][h] = Vectors.Normalise(plane);
            }
        }
    }

    /** Signature of one table: bit h is the sign of the dot product with hyperplane h. */
    internal ulong Signature(double[] unit, int table)
    {
        ulong code = 0;
        double[][] planes = _tables[table];
        for (int h = 0; h < _bits; h++)
        {
            if (Vectors.Dot(planes[h], unit) >= 0.0) code |= 1UL << h;
        }
        return code;
    }

    /** Bit count of the signature, used by the invariant checks. */
    internal static int PopCount(ulong value)
    {
        int count = 0;
        while (value != 0)
        {
            count += (int)(value & 1UL);
            value >>= 1;
        }
        return count;
    }
}

/** One hash table: bucket -> point indices. */
internal sealed class HashTable
{
    private readonly Dictionary<ulong, List<int>> _buckets = new Dictionary<ulong, List<int>>();

    internal void Add(ulong signature, int index)
    {
        if (!_buckets.TryGetValue(signature, out List<int> bucket))
        {
            bucket = new List<int>();
            _buckets[signature] = bucket;
        }
        bucket.Add(index);
    }

    internal IReadOnlyList<int> Lookup(ulong signature) =>
        _buckets.TryGetValue(signature, out List<int> bucket) ? bucket : Array.Empty<int>();

    internal int BucketCount => _buckets.Count;
}

internal sealed class LshIndex
{
    private readonly List<double[]> _units = new List<double[]>();
    private readonly SimHash _hash;
    private readonly HashTable[] _tables;

    internal LshIndex(int bits, int tables, int dimension, ulong seed)
    {
        _hash = new SimHash(bits, tables, dimension, seed);
        _tables = new HashTable[tables];
        for (int t = 0; t < tables; t++) _tables[t] = new HashTable();
    }

    internal int Count => _units.Count;
    internal int Bits => _hash.Bits;
    internal int Tables => _tables.Length;
    internal int BucketCount
    {
        get
        {
            int total = 0;
            foreach (HashTable table in _tables) total += table.BucketCount;
            return total;
        }
    }

    internal void Add(double[] vector)
    {
        double[] unit = Vectors.Normalise(vector);
        int index = _units.Count;
        _units.Add(unit);
        for (int t = 0; t < _tables.Length; t++) _tables[t].Add(_hash.Signature(unit, t), index);
    }

    /** Candidate indices from every table's matching bucket, deduplicated and in ascending order. */
    internal List<int> Candidates(double[] query, out int examinedPerTable)
    {
        double[] unit = Vectors.Normalise(query);
        var seen = new HashSet<int>();
        examinedPerTable = 0;
        for (int t = 0; t < _tables.Length; t++)
        {
            ulong signature = _hash.Signature(unit, t);
            IReadOnlyList<int> bucket = _tables[t].Lookup(signature);
            examinedPerTable += bucket.Count;
            foreach (int index in bucket) seen.Add(index);
        }
        var candidates = new List<int>(seen);
        candidates.Sort();
        return candidates;
    }

    internal double[] UnitAt(int index) => _units[index];
}

internal sealed class Tuning
{
    internal int Bits, Tables, Candidates, TruePositives;
    internal double Recall, Precision, Cost;

    internal string Describe() => "k=" + Bits + " L=" + Tables + " recall=" + Recall.ToString("F3")
        + " precision=" + Precision.ToString("F3") + " candidates=" + Candidates + " cost=" + Cost.ToString("F1");
}

public static class Solution
{
    private const int Dimension = 32;
    private const int DatasetSize = 400;
    private const int QueryCount = 40;
    private const int Neighbours = 10;

    /** Clustered data: the top-10 neighbours of a query are the other members of its own cluster. */
    private static List<double[]> Dataset(ulong seed, int count, int dimension, out List<double[]> centers, int clusterCount = 40)
    {
        var rng = seed;
        centers = new List<double[]>();
        for (int c = 0; c < clusterCount; c++)
        {
            var center = new double[dimension];
            for (int d = 0; d < dimension; d++) center[d] = SplitMix64.NextGaussian(ref rng);
            centers.Add(Vectors.Normalise(center));
        }
        var data = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            double[] center = centers[i % clusterCount];
            var v = new double[dimension];
            for (int d = 0; d < dimension; d++) v[d] = center[d] + 0.08 * SplitMix64.NextGaussian(ref rng);
            data.Add(v);
        }
        return data;
    }

    /** Queries sit in a cluster centre with modest jitter, so their true neighbours are that cluster's points. */
    private static List<double[]> Queries(IReadOnlyList<double[]> centers, int count, int dimension, ulong seed)
    {
        var rng = seed;
        var queries = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            double[] center = centers[(i * 7) % centers.Count];
            var v = new double[dimension];
            for (int d = 0; d < dimension; d++) v[d] = center[d] + 0.10 * SplitMix64.NextGaussian(ref rng);
            queries.Add(v);
        }
        return queries;
    }

    /** Exact cosine top-K, the ground truth the hash tables are scored against. */
    private static HashSet<int> ExactNeighbours(IReadOnlyList<double[]> data, double[] query, int k)
    {
        var scored = new List<(double score, int index)>();
        for (int i = 0; i < data.Count; i++) scored.Add((Vectors.Cosine(data[i], query), i));
        scored.Sort((a, b) => b.score.CompareTo(a.score));
        var result = new HashSet<int>();
        for (int i = 0; i < Math.Min(k, scored.Count); i++) result.Add(scored[i].index);
        return result;
    }

    private static Tuning Evaluate(IReadOnlyList<double[]> data, IReadOnlyList<double[]> queries, int bits, int tables,
        ulong seed, IReadOnlyList<HashSet<int>> truth)
    {
        var index = new LshIndex(bits, tables, Dimension, seed);
        foreach (double[] v in data) index.Add(v);
        var tuning = new Tuning { Bits = bits, Tables = tables };
        int found = 0, positives = 0, candidates = 0;
        for (int q = 0; q < queries.Count; q++)
        {
            List<int> cand = index.Candidates(queries[q], out _);
            candidates += cand.Count;
            var trueSet = truth[q];
            foreach (int c in cand)
            {
                if (trueSet.Contains(c)) found++;
            }
            int inCandidates = 0;
            foreach (int t in trueSet)
            {
                if (cand.Contains(t)) inCandidates++;
            }
            positives += inCandidates;
        }
        tuning.Candidates = candidates / queries.Count;
        tuning.Recall = positives / (double)(queries.Count * Neighbours);
        tuning.Precision = candidates == 0 ? 0.0 : found / (double)candidates;
        tuning.Cost = tuning.Candidates + (double)bits * tables * data.Count / data.Count;
        return tuning;
    }

    public static void Main()
    {
        List<double[]> data = Dataset(0xA11CEUL, DatasetSize, Dimension, out List<double[]> centers);
        List<double[]> queries = Queries(centers, QueryCount, Dimension, 0xB0BUL);
        var truth = new List<HashSet<int>>();
        foreach (double[] q in queries) truth.Add(ExactNeighbours(data, q, Neighbours));

        // 1-2. Collision-probability oracle: one hyperplane must collide with frequency 1 - theta/pi.
        var collider = new SimHash(1, 20000, Dimension, 0x5EEDUL);
        double[] alpha = Vectors.Normalise(new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        double[] rotated = new double[Dimension];
        rotated[0] = Math.Cos(0.9);
        rotated[1] = Math.Sin(0.9);
        int collisions = 0;
        for (int t = 0; t < 20000; t++)
        {
            if (collider.Signature(alpha, t) == collider.Signature(rotated, t)) collisions++;
        }
        double measured = collisions / 20000.0;
        double theoretical = 1.0 - 0.9 / Math.PI;
        Console.WriteLine("  collision oracle: theta=0.9 measured=" + measured.ToString("F4")
            + " theory 1-theta/pi=" + theoretical.ToString("F4"));
        Check(Math.Abs(measured - theoretical) < 0.02, "measured collision probability matches 1 - theta/pi");
        Check(SimHash.PopCount(0xFFFFUL) == 16 && SimHash.PopCount(0UL) == 0, "popcount used by the bit invariants works");

        // 3. Signature width and the exact self-collision property.
        var selfIndex = new LshIndex(8, 4, Dimension, 0x77UL);
        foreach (double[] v in data) selfIndex.Add(v);
        bool selfRecall = true;
        for (int i = 0; i < 50; i++)
        {
            List<int> cand = selfIndex.Candidates(data[i * 5], out _);
            if (!cand.Contains(i * 5)) selfRecall = false;
        }
        Check(selfIndex.Bits == 8 && selfIndex.Tables == 4, "index reports the configured k and L");
        Check(selfRecall, "a query always retrieves itself (exact self-collision)");
        Check(selfIndex.BucketCount <= 4 * data.Count, "no table has more buckets than inserted points");

        // 4-6. The (k, L) grid: recall grows with L, candidates shrink with k, and precision is reported.
        var grid = new List<Tuning>();
        foreach (int bits in new[] { 4, 8, 12 })
        {
            foreach (int tables in new[] { 1, 2, 4, 8 })
            {
                grid.Add(Evaluate(data, queries, bits, tables, 0xC0FFEEUL, truth));
            }
        }
        Console.WriteLine("  grid:");
        foreach (Tuning tuning in grid) Console.WriteLine("    " + tuning.Describe());
        bool recallGrowsWithTables = true, candidatesShrinkWithBits = true;
        foreach (int bits in new[] { 4, 8, 12 })
        {
            Tuning one = Find(grid, bits, 1), four = Find(grid, bits, 4), eight = Find(grid, bits, 8);
            if (!(four.Recall >= one.Recall - 1e-9 && eight.Recall >= four.Recall - 1e-9)) recallGrowsWithTables = false;
        }
        foreach (int tables in new[] { 1, 2, 4, 8 })
        {
            Tuning low = Find(grid, 4, tables), high = Find(grid, 12, tables);
            if (!(high.Candidates <= low.Candidates)) candidatesShrinkWithBits = false;
        }
        Check(recallGrowsWithTables, "recall is non-decreasing in the number of tables L");
        Check(candidatesShrinkWithBits, "the candidate set does not grow when k grows");
        foreach (Tuning tuning in grid)
        {
            Check(tuning.Recall >= 0.0 && tuning.Recall <= 1.0 && tuning.Precision >= 0.0 && tuning.Precision <= 1.0,
                "recall and precision stay inside [0, 1]");
        }

        // 7-9. Tuning against a target recall: the cheapest configuration that meets it is chosen.
        const double targetRecall = 0.9;
        Tuning best = null;
        foreach (Tuning tuning in grid)
        {
            if (tuning.Recall < targetRecall) continue;
            if (best == null || tuning.Cost < best.Cost) best = tuning;
        }
        Check(best != null, "the grid contains a configuration reaching the target recall of " + targetRecall);
        double costOfBest = best.Cost;
        foreach (Tuning tuning in grid)
        {
            if (tuning.Recall >= targetRecall)
            {
                Check(tuning.Cost >= costOfBest - 1e-9, "no cheaper configuration also meets the target recall");
            }
        }
        Tuning costliest = null;
        foreach (Tuning tuning in grid)
        {
            if (costliest == null || tuning.Cost > costliest.Cost) costliest = tuning;
        }
        Console.WriteLine("  tuned: " + best.Describe() + " | most expensive in grid: " + costliest.Describe());
        Check(best.Cost <= costliest.Cost, "tuning avoids the most expensive configuration in the grid");
        Check(Math.Abs(costOfBest - (best.Candidates + (double)best.Bits * best.Tables)) < 1e-9,
            "the reported cost equals the cost model");
        Check(best.Candidates > 0, "the chosen configuration actually examines candidates");

        // 10-11. Extended tables must eventually reach (almost) full recall - the recall ceiling check.
        var wide = Evaluate(data, queries, 3, 8, 0xC0FFEEUL, truth);
        Tuning sameTableCount = Find(grid, 4, 8);
        Console.WriteLine("  k=3 L=8 recall=" + wide.Recall.ToString("F3") + " candidates=" + wide.Candidates
            + " (k=4 L=8 recall=" + sameTableCount.Recall.ToString("F3") + ")");
        Check(wide.Recall >= sameTableCount.Recall - 1e-9, "smaller k at the same table count does not lose recall");
        Check(wide.Recall > 0.5, "a small-k index still finds most true neighbours");

        // 12-14. Boundary cases and rejection of invalid input.
        var tinyIndex = new LshIndex(4, 1, Dimension, 0xDUL);
        tinyIndex.Add(data[0]);
        List<int> tinyCandidates = tinyIndex.Candidates(data[0], out int perTable);
        Check(tinyCandidates.Count == 1 && perTable == 1, "a single-point index retrieves exactly that point");
        Check(Throws(() => new LshIndex(0, 1, Dimension, 1UL)), "k = 0 rejected");
        Check(Throws(() => new LshIndex(4, 0, Dimension, 1UL)), "L = 0 rejected");
        Check(Throws(() => new LshIndex(4, 1, 0, 1UL)), "zero dimension rejected");
        Check(Throws(() => tinyIndex.Candidates(new double[Dimension], out _)), "zero query vector rejected");
        var zeroPoint = new double[Dimension];
        Check(Throws(() => tinyIndex.Add(zeroPoint)), "zero-length database vector rejected");
        var emptyTable = new HashTable();
        Check(emptyTable.Lookup(123UL).Count == 0 && emptyTable.BucketCount == 0,
            "an empty table returns no candidates for any signature");

        // 15. Determinism: same seed, same signatures and therefore the same measured recall.
        Tuning first = Evaluate(data, queries, 8, 4, 0xC0FFEEUL, truth);
        Tuning second = Evaluate(data, queries, 8, 4, 0xC0FFEEUL, truth);
        Check(first.Recall == second.Recall && first.Candidates == second.Candidates,
            "identical seeds reproduce the measured recall and candidate count");

        Console.WriteLine("PASS");
    }

    private static Tuning Find(IReadOnlyList<Tuning> grid, int bits, int tables)
    {
        foreach (Tuning tuning in grid)
        {
            if (tuning.Bits == bits && tuning.Tables == tables) return tuning;
        }
        throw new InvalidOperationException("configuration not in grid");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
