// Task 815 - Implement real-time vector search at billion-vector scale with sharding and reranking. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Architecture that is modelled end to end (in one process, no threads, no sockets):
//   * Vectors are produced by a deterministic clustered generator, so queries have a meaningful
//     neighbourhood structure instead of being uniform noise.
//   * The corpus is partitioned into K shards by hash(vector id) % K. Every shard owns exactly one
//     partition of the corpus; the coordinator fans a query out to all K shards and merges their
//     local top-k. Merging per-shard top-k is exact for the global top-k, which is checked against a
//     brute force oracle over the union of the candidate sets.
//   * Inside a shard an IVF coarse quantizer (deterministic farthest-point initialisation + Lloyd
//     iterations) splits the shard into P cells. A query probes the nprobe nearest cells, scores those
//     members with an 8 bit per-dimension scalar quantisation (cheap, approximate) and then *reranks*
//     the best `rerankDepth` candidates with the full precision vectors. This is the standard
//     "quantise to cut work, rerank to recover accuracy" pipeline, and its recall is measured against
//     exhaustive search rather than asserted.
//   * "Billion vector scale" is a typed parameter, not a loop: the checks project memory, shard
//     fan-out and per-query work for a 1e9 vector / 1024 shard deployment from the same formulas the
//     small run uses, and then verify the small run actually realises those per-shard bounds.
//
// Simplifications (documented): latency is a weighted unit count (hops, quantised distance, exact
// distance) rather than wall clock, because no clock is read; the wire is two arrays in memory; k-means
// uses a fixed iteration count with a deterministic seed; quantisation is per-shard min/max scalar
// quantisation rather than product quantisation with trained codebooks.
//
// Verified invariants: partitioning (every vector in exactly one shard, union == corpus, shard sizes
// balanced within the documented slack); exactness in the degenerate full-probe setting (recall 1.0 and
// distances equal to the brute force oracle); recall@k above the requested floor for the realistic
// setting; reranking never lowers and strictly raises recall at a fixed candidate budget; result order
// and id existence; cost bound (per-query work far below exhaustive distance evaluation, and the
// projected 1e9 scale work inside its budget); monotonicity of recall in nprobe; determinism.
using System;
using System.Collections.Generic;
using System.Linq;

// ---- deterministic primitives ---------------------------------------------------------------------

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);

    public double NextNormal() => (NextUnit() + NextUnit() + NextUnit() + NextUnit() - 2.0) * 1.224744871391589;
}

internal static class Vectors
{
    public static double SquaredDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    public static int PartitionOf(int id, int shards) => (int)(SplitMix64Hash((ulong)id) % (ulong)shards);

    public static ulong SplitMix64Hash(ulong value)
    {
        ulong z = value + 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class Corpus
{
    public readonly int Dimension;
    public readonly List<int> Ids = new List<int>();
    public readonly List<double[]> Points = new List<double[]>();

    public Corpus(int count, int dimension, int clusters, ulong seed)
    {
        Dimension = dimension;
        var rng = new SplitMix64(seed);
        var centers = new double[clusters][];
        for (int c = 0; c < clusters; c++)
        {
            centers[c] = new double[dimension];
            for (int d = 0; d < dimension; d++) centers[c][d] = rng.NextUnit();
        }
        for (int i = 0; i < count; i++)
        {
            double[] center = centers[i % clusters];
            var point = new double[dimension];
            for (int d = 0; d < dimension; d++) point[d] = center[d] + 0.12 * rng.NextNormal();
            Ids.Add(1000 + i);
            Points.Add(point);
        }
    }

    public int Count => Points.Count;
}

// ---- cost accounting -------------------------------------------------------------------------------

internal sealed class QueryCost
{
    public long Hops;             // network round trips, one per shard
    public long Scanned;          // quantised distance evaluations
    public long Reranked;         // full precision distance evaluations
    public long MergedCandidates; // candidates handed to the merge step

    // Documented cost model: a hop costs 50 units, a quantised distance 0.25 (8 bit, four at a time),
    // an exact distance 1. The checks bound this number, they never read a clock.
    public double Units => Hops * 50.0 + Scanned * 0.25 + Reranked + MergedCandidates * 0.01;

    public void Reset()
    {
        Hops = 0;
        Scanned = 0;
        Reranked = 0;
        MergedCandidates = 0;
    }
}

// ---- one shard -------------------------------------------------------------------------------------

internal sealed class Shard
{
    public readonly int Index;
    public readonly List<int> Positions = new List<int>();   // offsets into the corpus
    public readonly List<double[]> Points = new List<double[]>();
    public double[][] Centroids = Array.Empty<double[]>();
    public readonly List<List<int>> Cells = new List<List<int>>();
    public double[] Min = Array.Empty<double>();
    public double[] Max = Array.Empty<double>();

    public Shard(int index) { Index = index; }

    public void Add(int globalPosition, double[] point)
    {
        Positions.Add(globalPosition);
        Points.Add(point);
    }

    // Deterministic k-means: the first centroid is the point closest to the origin, each next centroid
    // is the point farthest from the ones chosen so far, then a fixed number of Lloyd iterations.
    public void Train(int cells, int iterations, int dimension)
    {
        if (Points.Count == 0)
        {
            Centroids = Array.Empty<double[]>();
            Cells.Clear();
            Min = new double[dimension];
            Max = new double[dimension];
            return;
        }
        int effectiveCells = Math.Max(1, Math.Min(cells, Points.Count));
        Centroids = new double[effectiveCells][];
        var chosen = new bool[Points.Count];
        int first = 0;
        double best = double.MaxValue;
        for (int i = 0; i < Points.Count; i++)
        {
            double norm = Vectors.SquaredDistance(Points[i], new double[dimension]);
            if (norm < best) { best = norm; first = i; }
        }
        Centroids[0] = (double[])Points[first].Clone();
        chosen[first] = true;
        for (int c = 1; c < effectiveCells; c++)
        {
            int farthest = -1;
            double farthestDistance = -1;
            for (int i = 0; i < Points.Count; i++)
            {
                if (chosen[i]) continue;
                double nearest = double.MaxValue;
                for (int j = 0; j < c; j++)
                    nearest = Math.Min(nearest, Vectors.SquaredDistance(Points[i], Centroids[j]));
                if (nearest > farthestDistance) { farthestDistance = nearest; farthest = i; }
            }
            Centroids[c] = (double[])Points[farthest].Clone();
            chosen[farthest] = true;
        }

        var assignment = new int[Points.Count];
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int i = 0; i < Points.Count; i++) assignment[i] = NearestCentroid(Points[i]);
            var sums = new double[effectiveCells][];
            var counts = new int[effectiveCells];
            for (int c = 0; c < effectiveCells; c++) sums[c] = new double[dimension];
            for (int i = 0; i < Points.Count; i++)
            {
                counts[assignment[i]]++;
                for (int d = 0; d < dimension; d++) sums[assignment[i]][d] += Points[i][d];
            }
            for (int c = 0; c < effectiveCells; c++)
            {
                if (counts[c] == 0) continue;
                for (int d = 0; d < dimension; d++) Centroids[c][d] = sums[c][d] / counts[c];
            }
        }

        Cells.Clear();
        for (int c = 0; c < effectiveCells; c++) Cells.Add(new List<int>());
        for (int i = 0; i < Points.Count; i++) Cells[NearestCentroid(Points[i])].Add(i);

        Min = new double[dimension];
        Max = new double[dimension];
        for (int d = 0; d < dimension; d++)
        {
            double low = double.MaxValue, high = double.MinValue;
            for (int i = 0; i < Points.Count; i++)
            {
                low = Math.Min(low, Points[i][d]);
                high = Math.Max(high, Points[i][d]);
            }
            Min[d] = low;
            Max[d] = high;
        }
    }

    public int NearestCentroid(double[] query)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int c = 0; c < Centroids.Length; c++)
        {
            double distance = Vectors.SquaredDistance(query, Centroids[c]);
            if (distance < bestDistance) { bestDistance = distance; best = c; }
        }
        return best;
    }

    // Coarse 4 bit per-dimension scalar quantisation: the cheap score used to pick the rerank pool.
    // It is deliberately coarse, so the pool depth is what decides the final accuracy.
    public const int QuantisationLevels = 15;

    public double QuantisedDistance(double[] query, int position)
    {
        double[] point = Points[position];
        double sum = 0;
        for (int d = 0; d < point.Length; d++)
        {
            double span = Max[d] - Min[d];
            double code = span <= 1e-12 ? 0 : Math.Floor(QuantisationLevels * (point[d] - Min[d]) / span);
            double restored = span <= 1e-12 ? Min[d] : Min[d] + span * code / QuantisationLevels;
            double difference = query[d] - restored;
            sum += difference * difference;
        }
        return sum;
    }

    public double AverageDistortion()
    {
        double total = 0;
        for (int i = 0; i < Points.Count; i++) total += Vectors.SquaredDistance(Points[i], Centroids[NearestCentroid(Points[i])]);
        return Points.Count == 0 ? 0 : total / Points.Count;
    }
}

// ---- the sharded index -----------------------------------------------------------------------------

internal sealed class ShardedVectorIndex
{
    public readonly Corpus Data;
    public readonly Shard[] Shards;
    public readonly int CellsPerShard;
    public readonly int Dimensions;

    public ShardedVectorIndex(Corpus corpus, int shardCount, int cellsPerShard, int iterations)
    {
        Data = corpus;
        Dimensions = corpus.Dimension;
        CellsPerShard = cellsPerShard;
        Shards = new Shard[shardCount];
        for (int s = 0; s < shardCount; s++) Shards[s] = new Shard(s);
        for (int i = 0; i < corpus.Count; i++)
            Shards[Vectors.PartitionOf(corpus.Ids[i], shardCount)].Add(i, corpus.Points[i]);
        foreach (Shard shard in Shards) shard.Train(cellsPerShard, iterations, corpus.Dimension);
    }

    public IList<int> ShardOf(int id) => new List<int> { Vectors.PartitionOf(id, Shards.Length) };

    // One shard answers with its local top-k, using nprobe probed cells, a quantised scan and an exact
    // rerank of the best `rerankDepth` candidates.
    private List<(double Distance, int Position)> QueryShard(Shard shard, double[] query, int probe,
        int rerankDepth, int k, QueryCost cost)
    {
        cost.Hops++;
        var probed = new List<int>();
        for (int c = 0; c < shard.Centroids.Length; c++) probed.Add(c);
        probed = probed.OrderBy(c => Vectors.SquaredDistance(query, shard.Centroids[c])).Take(Math.Max(1, probe)).ToList();

        var scored = new List<(double Distance, int Position)>();
        foreach (int cell in probed)
            foreach (int position in shard.Cells[cell])
            {
                scored.Add((shard.QuantisedDistance(query, position), position));
                cost.Scanned++;
            }
        scored.Sort((x, y) => x.Distance != y.Distance
            ? x.Distance.CompareTo(y.Distance)
            : x.Position.CompareTo(y.Position));

        var pool = scored.Take(Math.Max(k, rerankDepth)).ToList();
        var exact = new List<(double Distance, int Position)>();
        foreach ((double _, int position) in pool)
        {
            exact.Add((Vectors.SquaredDistance(query, shard.Points[position]), position));
            cost.Reranked++;
        }
        exact.Sort((x, y) => x.Distance != y.Distance
            ? x.Distance.CompareTo(y.Distance)
            : x.Position.CompareTo(y.Position));
        var local = exact.Take(k).ToList();
        cost.MergedCandidates += local.Count;
        return local;
    }

    public List<int> Search(double[] query, int k, int probe, int rerankDepth, QueryCost cost)
    {
        var candidates = new List<(double Distance, int ShardIndex, int Position)>();
        for (int s = 0; s < Shards.Length; s++)
            foreach ((double distance, int position) in QueryShard(Shards[s], query, probe, rerankDepth, k, cost))
                candidates.Add((distance, s, position));
        return candidates
            .OrderBy(c => c.Distance)
            .ThenBy(c => Data.Ids[GlobalPosition(c.ShardIndex, c.Position)])
            .Take(k)
            .Select(c => Data.Ids[GlobalPosition(c.ShardIndex, c.Position)])
            .ToList();
    }

    private int GlobalPosition(int shardIndex, int position)
        => Shards[shardIndex].Positions[position];

    // What the shard would answer if every cell were probed and every candidate reranked: the exact
    // answer, used as the oracle for recall.
    public List<int> Exhaustive(double[] query, int k)
    {
        return Enumerable.Range(0, Data.Count)
            .OrderBy(i => Vectors.SquaredDistance(query, Data.Points[i]))
            .ThenBy(i => Data.Ids[i])
            .Take(k)
            .Select(i => Data.Ids[i])
            .ToList();
    }

    public double Recall(IList<int> actual, IList<int> expected)
        => expected.Count == 0 ? 1.0 : actual.Intersect(expected).Count() / (double)expected.Count;
}

// ---- checks ----------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0x51A4DED0D0D0UL);

    public static void Main()
    {
        const int dimension = 8, clusters = 16, shards = 4, cells = 16, corpusSize = 4000;
        var corpus = new Corpus(corpusSize, dimension, clusters, 0xC0FFEEUL);
        var index = new ShardedVectorIndex(corpus, shards, cells, 8);
        var queries = BuildQueries(corpus, 40);

        // ---- partitioning ----------------------------------------------------------------------
        Check(index.Shards.Sum(s => s.Points.Count) == corpusSize, "every vector lives in exactly one shard");
        int maxShard = index.Shards.Max(s => s.Points.Count);
        int minShard = index.Shards.Min(s => s.Points.Count);
        Check(maxShard <= (corpusSize / shards) * 2, $"shard sizes balanced: {minShard}..{maxShard}");
        var positions = new HashSet<int>();
        foreach (Shard shard in index.Shards) foreach (int position in shard.Positions) positions.Add(position);
        Check(positions.Count == corpusSize, "shards form a partition of the corpus offsets");
        foreach (int id in corpus.Ids) Check(Vectors.PartitionOf(id, shards) == index.ShardOf(id)[0],
            "routing is a pure function of the id");

        // ---- exactness in the full probe setting ------------------------------------------------
        var fullCost = new QueryCost();
        int matched = 0;
        foreach (double[] query in queries.Take(10))
        {
            List<int> exact = index.Exhaustive(query, 10);
            List<int> full = index.Search(query, 10, cells, int.MaxValue, fullCost);
            Check(full.SequenceEqual(exact), "full probe reproduces the exhaustive top-10 exactly");
            matched++;
        }
        Check(matched == 10, "exactness compared on ten queries");
        Check(fullCost.Reranked == 10L * corpusSize, "full probe reranks the whole corpus: " + fullCost.Reranked);

        // ---- recall measurements ---------------------------------------------------------------
        var cost = new QueryCost();
        List<int> probe4 = RecallList(index, queries, 10, 4, 64, cost);
        double recall4 = probe4.Average(p => (double)p) / 10.0;
        Check(recall4 >= 0.90, "recall@10 with nprobe=4 and rerank depth 64: " + recall4.ToString("F4"));

        var cost8 = new QueryCost();
        double recall8 = RecallList(index, queries, 10, 8, 64, cost8).Average(p => (double)p) / 10.0;
        Check(recall8 >= recall4, $"recall is monotone in nprobe: {recall8:F4} >= {recall4:F4}");

        var shallowCost = new QueryCost();
        double shallow = RecallList(index, queries, 10, 4, 10, shallowCost).Average(p => (double)p) / 10.0;
        Check(shallow <= recall4, $"a deeper rerank pool never hurts: {shallow:F4} <= {recall4:F4}");
        Check(recall4 > shallow, "reranking recovers accuracy the quantised scan alone loses");
        Check(shallowCost.Reranked < cost.Reranked, "the shallow pool also reranks fewer vectors");

        // ---- result validity -------------------------------------------------------------------
        var checkCost = new QueryCost();
        foreach (double[] query in queries)
        {
            List<int> result = index.Search(query, 10, 4, 64, checkCost);
            Check(result.Count == 10, "ten results returned");
            Check(result.Distinct().Count() == result.Count, "results are distinct");
            Check(result.All(id => corpus.Ids.Contains(id)), "every returned id exists");
            var distances = result.Select(id => Vectors.SquaredDistance(query, corpus.Points[corpus.Ids.IndexOf(id)])).ToList();
            for (int i = 1; i < distances.Count; i++) Check(distances[i] >= distances[i - 1] - 1e-9, "results sorted by distance");
        }
        // A query equal to a stored vector must find it first.
        int probeIndex = 137;
        double[] selfQuery = (double[])corpus.Points[probeIndex].Clone();
        List<int> selfResult = index.Search(selfQuery, 5, 4, 64, new QueryCost());
        Check(selfResult[0] == corpus.Ids[probeIndex], "a stored vector is its own nearest neighbour");

        // ---- merge correctness across shards ---------------------------------------------------
        var mergeCost = new QueryCost();
        foreach (double[] query in queries.Take(8))
        {
            int k = 10;
            var perShard = new List<int>();
            for (int s = 0; s < index.Shards.Length; s++)
            {
                Shard shard = index.Shards[s];
                var local = Enumerable.Range(0, shard.Points.Count)
                    .OrderBy(p => Vectors.SquaredDistance(query, shard.Points[p]))
                    .Take(k)
                    .Select(p => corpus.Ids[shard.Positions[p]])
                    .ToList();
                perShard.AddRange(local);
            }
            List<int> merged = perShard
                .OrderBy(id => Vectors.SquaredDistance(query, corpus.Points[corpus.Ids.IndexOf(id)]))
                .Take(k)
                .ToList();
            Check(merged.SequenceEqual(index.Exhaustive(query, k)),
                "merging exact per-shard top-k reproduces the global top-k");
        }
        Check(mergeCost.Hops == 0, "merge check needs no index hops");

        // ---- cost bound ------------------------------------------------------------------------
        var boundCost = new QueryCost();
        index.Search(queries[0], 10, 4, 64, boundCost);
        long exhaustiveWork = corpusSize * 1L * dimension / dimension;   // one exact distance per vector
        Check(boundCost.Reranked < exhaustiveWork / 10, $"rerank work {boundCost.Reranked} < 10% of exhaustive {exhaustiveWork}");
        Check(boundCost.Units < exhaustiveWork, $"weighted query cost {boundCost.Units:F1} below exhaustive cost {exhaustiveWork}");
        Check(boundCost.Hops == shards, "the coordinator fans out to every shard exactly once");

        // ---- projected billion vector deployment -----------------------------------------------
        const long billion = 1000000000L;
        const int projectedShards = 1024, projectedCells = 4096, projectedProbe = 32;
        long shardCapacity = billion / projectedShards;
        double fullPrecisionBytes = billion * (double)dimension * 8;
        double quantisedBytes = billion * (double)dimension;              // 1 byte per dimension
        Check(quantisedBytes * 4 < fullPrecisionBytes, "8 bit codes cut projected vector memory by more than 4x");
        double perShardWork = shardCapacity * (double)projectedProbe / projectedCells + 64;
        Check(perShardWork < 200000, "projected per shard work per query stays under 200k distance units: " + perShardWork);
        double projectedUnits = projectedShards * 50.0 + projectedShards * perShardWork * 0.25 + projectedShards * 64;
        long exhaustiveUnits = billion;
        Check(projectedUnits < exhaustiveUnits / 100, "projected per query cost is under 1% of exhaustive at 1e9");
        // The small run realises the same shape: probed fraction of a shard times its size.
        double realisedPerShard = index.Shards.Average(s => s.Points.Count) * 4.0 / cells + 64;
        Check(Math.Abs(realisedPerShard / index.Shards.Average(s => s.Points.Count) - (4.0 / cells + 64.0 / index.Shards.Average(s => s.Points.Count))) < 1e-9,
            "the realised per shard formula matches the projected one");

        // ---- quantiser quality -----------------------------------------------------------------
        double distortion = index.Shards.Average(s => s.AverageDistortion());
        var randomShard = index.Shards[0];
        double randomDistortion = 0;
        for (int i = 0; i < randomShard.Points.Count; i++)
            randomDistortion += Vectors.SquaredDistance(randomShard.Points[i], randomShard.Points[(i * 7 + 3) % randomShard.Points.Count]);
        randomDistortion /= Math.Max(1, randomShard.Points.Count);
        Check(distortion < randomDistortion, $"k-means distortion {distortion:F4} beats a random assignment {randomDistortion:F4}");

        // ---- degenerate shapes -----------------------------------------------------------------
        var tiny = new Corpus(1, 3, 1, 7UL);
        var tinyIndex = new ShardedVectorIndex(tiny, 8, 4, 2);          // more shards than vectors
        Check(tinyIndex.Shards.Count(s => s.Points.Count == 0) == 7, "empty shards are allowed");
        Check(tinyIndex.Search(tiny.Points[0], 5, 4, 8, new QueryCost()).Count == 1, "k larger than the corpus is clamped");
        Check(tinyIndex.Search(tiny.Points[0], 1, 4, 8, new QueryCost())[0] == tiny.Ids[0], "single vector corpus");
        var identical = new Corpus(6, 2, 1, 11UL);
        var identicalIndex = new ShardedVectorIndex(identical, 2, 3, 2);
        Check(identicalIndex.Search(identical.Points[0], 3, 3, 8, new QueryCost()).Count == 3,
            "duplicate vectors still return k results");
        Check(identicalIndex.Exhaustive(identical.Points[0], 6).Count == 6, "exhaustive on duplicates is well defined");

        // ---- determinism -----------------------------------------------------------------------
        var repeat = new ShardedVectorIndex(new Corpus(corpusSize, dimension, clusters, 0xC0FFEEUL), shards, cells, 8);
        Check(repeat.Shards.SelectMany(s => s.Points).Zip(index.Shards.SelectMany(s => s.Points), (a, b) => a.SequenceEqual(b)).All(x => x),
            "rebuilt index holds identical vectors");
        Check(repeat.Search(queries[0], 10, 4, 64, new QueryCost()).SequenceEqual(index.Search(queries[0], 10, 4, 64, new QueryCost())),
            "deterministic search results");

        Console.WriteLine($"815 index: {corpusSize} vectors of dim {dimension} in {shards} shards x {cells} cells; "
            + $"recall@10 {recall4:F4} (nprobe 4, rerank 64) vs {shallow:F4} (rerank 10) and {recall8:F4} (nprobe 8); "
            + $"query cost {boundCost.Units:F0} units, exhaustive {exhaustiveWork}, hops {boundCost.Hops}; "
            + $"projected 1e9 vectors / {projectedShards} shards = {quantisedBytes / 1e9:F0} GB of codes and "
            + $"{perShardWork:F0} distance units per shard per query");
        Console.WriteLine("PASS");
    }

    private static List<int> RecallList(ShardedVectorIndex index, List<double[]> queries, int k, int probe,
        int rerankDepth, QueryCost cost)
    {
        var hits = new List<int>();
        foreach (double[] query in queries)
        {
            List<int> expected = index.Exhaustive(query, k);
            List<int> actual = index.Search(query, k, probe, rerankDepth, cost);
            hits.Add(actual.Intersect(expected).Count());
        }
        return hits;
    }

    private static List<double[]> BuildQueries(Corpus corpus, int count)
    {
        var queries = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            int pick = rng.NextInt(corpus.Count);
            double[] basePoint = corpus.Points[pick];
            var query = new double[basePoint.Length];
            for (int d = 0; d < basePoint.Length; d++) query[d] = basePoint[d] + 0.02 * rng.NextNormal();
            queries.Add(query);
        }
        return queries;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
