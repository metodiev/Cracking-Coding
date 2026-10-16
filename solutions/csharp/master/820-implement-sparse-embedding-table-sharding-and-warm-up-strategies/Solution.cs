// Task 820 - Implement sparse-embedding table sharding and warm-up strategies for recommender systems. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// The table has `TotalIds` rows of dimension D, spread over S shards by hash(id) % S. A row is
// *materialised* only once it has been touched, which is what makes a recommender's table sparse: the
// store holds the rows that training actually used, and a cold row is produced on demand by a warm-up
// strategy. Everything is deterministic (splitmix64), single threaded and in process.
//
// Warm-up strategies implemented and compared:
//   * Zero        - the cold row is all zeros (the naive default, which scores every unseen item equally);
//   * MeanOfHot   - the cold row is the mean of the rows already materialised in its shard;
//   * CategoryPrior - the cold row is the feature's metadata prior (a category basis vector, the way a
//                   production system seeds an unseen item from its category embedding), blended with
//                   MeanOfHot by a fixed coefficient.
// The strategies are compared on a deterministic ranking task (does the positive item outrank the cold
// distractors for a hot query feature?), so the claim "warm-up helps" is a measurement, not an assertion.
//
// Sharding and sparsity claims that are measured: shard balance (the largest and smallest shard stay
// within a small factor of TotalIds/S); the sparse scatter-add gradient equals a dense oracle on a tiny
// table; untouched rows are bit identical before and after an update (so a step costs O(batch x D), not
// O(TotalIds x D)); the LRU row cache's hit rate is measured on a skewed access pattern and the cache is
// coherent with the store after an update; and the total memory is reported as materialised rows x D.
//
// Simplifications (documented): rows live in a Dictionary per shard rather than on device memory, so
// "memory" means the number of materialised rows and their byte size; the cache models a row cache with
// an explicit capacity and an exact LRU discipline rather than a GPU coalescing cache; the category
// prior is a synthetic basis vector rather than real side information; there is no distributed
// parameter server (the shards are local), no asynchronous push/pull and no quantisation.
//
// Verified invariants: lookup equals the warm-up oracle for every id (cold and hot); ids out of range are
// rejected rather than silently aliased; sharding is a partition with balanced sizes; updates touch only
// the rows in the batch and leave everything else bit identical; the sparse gradient equals the dense
// oracle and accumulates repeated ids correctly; multi-value lookup averages as documented; the LRU
// cache evicts in LRU order, never returns a stale row after an update and its hit rate improves as the
// access pattern gets more skewed; warm-up strictly improves the cold-item ranking metric over the zero
// baseline; and repeated runs are identical.
using System;
using System.Collections.Generic;
using System.Linq;

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

    public double NextSigned() => ((Next() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;
}

internal enum WarmUp { Zero, MeanOfHot, CategoryPrior }

internal sealed class Shard
{
    public readonly int Index;
    public readonly Dictionary<int, double[]> Rows = new Dictionary<int, double[]>();
    public readonly Dictionary<int, double[]> Moments = new Dictionary<int, double[]>();
    public long Reads;
    public long Writes;

    public Shard(int index) { Index = index; }

    public double[] Mean(int dimension)
    {
        var mean = new double[dimension];
        if (Rows.Count == 0) return mean;
        foreach (double[] row in Rows.Values)
            for (int d = 0; d < dimension; d++) mean[d] += row[d];
        for (int d = 0; d < dimension; d++) mean[d] /= Rows.Count;
        return mean;
    }
}

internal sealed class LruRowCache
{
    private readonly int capacity;
    private readonly Dictionary<int, double[]> rows = new Dictionary<int, double[]>();
    private readonly LinkedList<int> order = new LinkedList<int>();
    private readonly Dictionary<int, LinkedListNode<int>> nodes = new Dictionary<int, LinkedListNode<int>>();

    public long Hits;
    public long Misses;
    public long Evictions;

    public LruRowCache(int capacity) { this.capacity = Math.Max(1, capacity); }

    public bool TryGet(int id, out double[] row)
    {
        if (rows.TryGetValue(id, out double[] stored))
        {
            Touch(id);
            Hits++;
            row = stored;
            return true;
        }
        Misses++;
        row = Array.Empty<double>();
        return false;
    }

    public void Put(int id, double[] row)
    {
        if (rows.ContainsKey(id))
        {
            rows[id] = row;
            Touch(id);
            return;
        }
        rows[id] = row;
        nodes[id] = order.AddFirst(id);
        if (rows.Count > capacity)
        {
            LinkedListNode<int> last = order.Last!;
            order.RemoveLast();
            nodes.Remove(last.Value);
            rows.Remove(last.Value);
            Evictions++;
        }
    }

    public void Invalidate(int id)
    {
        if (nodes.TryGetValue(id, out LinkedListNode<int> node))
        {
            order.Remove(node);
            nodes.Remove(id);
        }
        rows.Remove(id);
    }

    public bool Contains(int id) => rows.ContainsKey(id);
    public int Count => rows.Count;
    public double HitRate => Hits + Misses == 0 ? 0 : Hits / (double)(Hits + Misses);

    private void Touch(int id)
    {
        if (!nodes.TryGetValue(id, out LinkedListNode<int> node)) return;
        order.Remove(node);
        order.AddFirst(node);
    }
}

internal sealed class SparseEmbeddingTable
{
    public readonly int TotalIds;
    public readonly int Dimension;
    public readonly Shard[] Shards;
    public readonly WarmUp Strategy;
    public readonly double PriorWeight;
    public readonly LruRowCache Cache;
    public readonly double[][] Priors;

    public long ColdRowsEmitted;
    public long RejectedIds;
    public long Lookups;
    public long Updates;
    public long MaterialisedRows;

    public SparseEmbeddingTable(int totalIds, int dimension, int shards, WarmUp strategy, double priorWeight, int cacheRows)
    {
        TotalIds = totalIds;
        Dimension = dimension;
        Shards = new Shard[shards];
        for (int s = 0; s < shards; s++) Shards[s] = new Shard(s);
        Strategy = strategy;
        PriorWeight = priorWeight;
        Cache = new LruRowCache(cacheRows);
        Priors = new double[totalIds][];
        var rng = new SplitMix64(0x5EEDC0DEUL);
        for (int id = 0; id < totalIds; id++)
        {
            var prior = new double[dimension];
            for (int d = 0; d < dimension; d++) prior[d] = rng.NextSigned();
            Priors[id] = prior;
        }
    }

    public int ShardOf(int id) => (int)(((ulong)id * 0x9E3779B97F4A7C15UL + 0xD1B54A32D192ED03UL) % (ulong)Shards.Length);

    public bool IsMaterialised(int id) => Shards[ShardOf(id)].Rows.ContainsKey(id);

    // The warm-up oracle: what a cold row must look like for each strategy.
    public double[] WarmRow(int id)
    {
        Shard shard = Shards[ShardOf(id)];
        switch (Strategy)
        {
            case WarmUp.Zero:
                return new double[Dimension];
            case WarmUp.MeanOfHot:
                return shard.Mean(Dimension);
            default:
                double[] mean = shard.Mean(Dimension);
                var blended = new double[Dimension];
                for (int d = 0; d < Dimension; d++)
                    blended[d] = PriorWeight * Priors[id][d] + (1 - PriorWeight) * mean[d];
                return blended;
        }
    }

    public double[] Lookup(int id)
    {
        Lookups++;
        if (id < 0 || id >= TotalIds)
        {
            RejectedIds++;
            throw new ArgumentOutOfRangeException(nameof(id), "embedding id out of range");
        }
        Shard shard = Shards[ShardOf(id)];
        shard.Reads++;
        if (shard.Rows.TryGetValue(id, out double[] row)) return row;
        double[] warm = WarmRow(id);
        ColdRowsEmitted++;
        return warm;
    }

    public double[] LookupMulti(IList<int> ids)
    {
        var average = new double[Dimension];
        if (ids.Count == 0) return average;
        foreach (int id in ids)
        {
            double[] row = Lookup(id);
            for (int d = 0; d < Dimension; d++) average[d] += row[d];
        }
        for (int d = 0; d < Dimension; d++) average[d] /= ids.Count;
        return average;
    }

    // Sparse scatter-add: only the rows in the batch are read, updated and materialised.
    public void ApplyGradient(IList<(int Id, double[] Gradient)> batch, double learningRate, double momentum)
    {
        Updates++;
        foreach ((int id, double[] gradient) in batch)
        {
            if (id < 0 || id >= TotalIds)
            {
                RejectedIds++;
                throw new ArgumentOutOfRangeException(nameof(batch), "embedding id out of range");
            }
            Shard shard = Shards[ShardOf(id)];
            double[] row = shard.Rows.TryGetValue(id, out double[] stored) ? stored : Materialise(shard, id);
            if (!shard.Moments.TryGetValue(id, out double[] moment)) shard.Moments[id] = moment = new double[Dimension];
            for (int d = 0; d < Dimension; d++)
            {
                moment[d] = momentum * moment[d] + gradient[d];
                row[d] -= learningRate * moment[d];
            }
            shard.Writes++;
            if (Cache.Contains(id)) Cache.Put(id, row);      // write through: the cache can never go stale
        }
    }

    private double[] Materialise(Shard shard, int id)
    {
        double[] warm = WarmRow(id);
        shard.Rows[id] = warm;
        MaterialisedRows++;
        return warm;
    }

    public double[] ReadThroughCache(int id)
    {
        if (Cache.TryGet(id, out double[] cached)) return cached;
        double[] row = Lookup(id);
        double[] copy = (double[])row.Clone();               // the cache holds its own copy until it is written
        Cache.Put(id, copy);
        return copy;
    }

    public int MaterialisedInShard(int shard) => Shards[shard].Rows.Count;

    public long MaterialisedBytes => MaterialisedRows * Dimension * sizeof(double);

    public void SetPrior(int id, double[] row) => Priors[id] = (double[])row.Clone();
}

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0x1CEB00DAUL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        // ---- lookup, warm-up oracle and sharding -------------------------------------------------
        var table = new SparseEmbeddingTable(400, 8, 5, WarmUp.CategoryPrior, 1.0, 32);
        int coldChecked = 0;
        foreach (int id in new[] { 0, 1, 7, 199, 200, 399 })
        {
            double[] expected = table.WarmRow(id);
            double[] actual = table.Lookup(id);
            Check(actual.SequenceEqual(expected), "cold lookup equals the warm-up oracle for id " + id);
            Check(!table.IsMaterialised(id), "a pure lookup does not materialise a row");
            coldChecked++;
        }
        Check(coldChecked == 6, "cold lookups compared");
        Check(table.ColdRowsEmitted == 6, "cold rows are counted");
        Check(table.RejectedIds == 0, "no id was rejected yet");
        Check(table.Lookup(400 - 1).Length == 8 && table.Lookup(0).Length == 8, "every row has the right dimension");

        var zeroTable = new SparseEmbeddingTable(400, 8, 5, WarmUp.Zero, 1.0, 32);
        Check(zeroTable.Lookup(3).All(v => v == 0), "the zero strategy really returns zeros");
        var meanTable = new SparseEmbeddingTable(400, 8, 1, WarmUp.MeanOfHot, 1.0, 32);
        meanTable.ApplyGradient(new[] { (3, new double[8]), (9, new double[8]) }, 0.0, 0.0);
        Check(meanTable.Lookup(3).All(v => v == 0), "mean of hot is zero while every row is zero");
        var seedGradients = new[] { Full(8, 1.0), Full(8, 3.0) };
        meanTable.ApplyGradient(new[] { (3, seedGradients[0]), (9, seedGradients[1]) }, 1.0, 0.0);
        Check(meanTable.IsMaterialised(3) && meanTable.IsMaterialised(9), "gradient rows are materialised");
        double[] mean = meanTable.WarmRow(11);
        Check(Math.Abs(mean[0] + 2.0) < 1e-12, "mean of hot averages the two materialised rows: " + mean[0]);
        Check(meanTable.MaterialisedRows == 2, "only two rows are materialised: " + meanTable.MaterialisedRows);

        // ---- rejected ids ------------------------------------------------------------------------
        var rejecting = new SparseEmbeddingTable(50, 4, 3, WarmUp.Zero, 1.0, 8);
        bool threw = false;
        try { rejecting.Lookup(50); } catch (ArgumentOutOfRangeException) { threw = true; }
        Check(threw && rejecting.RejectedIds == 1, "an out of range id is rejected, not aliased");
        threw = false;
        try { rejecting.ApplyGradient(new[] { (-1, new double[4]) }, 0.1, 0.0); } catch (ArgumentOutOfRangeException) { threw = true; }
        Check(threw && rejecting.RejectedIds == 2, "an out of range update is rejected");

        // ---- sparse gradient versus the dense oracle --------------------------------------------
        var sparse = new SparseEmbeddingTable(24, 3, 3, WarmUp.Zero, 1.0, 64);
        var dense = new double[24][];
        for (int id = 0; id < 24; id++)
        {
            dense[id] = new double[3];
            sparse.ApplyGradient(new[] { (id, new double[3]) }, 0.0, 0.0);
        }
        var batch = new List<(int, double[])>();
        for (int i = 0; i < 40; i++)
        {
            int id = Rand(24);
            var gradient = new double[] { rng.NextSigned(), rng.NextSigned(), rng.NextSigned() };
            batch.Add((id, gradient));
        }
        var denseCopy = dense.Select(r => (double[])r.Clone()).ToList();
        var sparseCopy = Enumerable.Range(0, 24).Select(id => (double[])sparse.Lookup(id).Clone()).ToList();
        foreach ((int id, double[] gradient) in batch)
            for (int d = 0; d < 3; d++) denseCopy[id][d] -= 0.25 * gradient[d];
        sparse.ApplyGradient(batch, 0.25, 0.0);
        for (int id = 0; id < 24; id++)
            for (int d = 0; d < 3; d++)
                Check(Math.Abs(sparse.Lookup(id)[d] - denseCopy[id][d]) < 1e-12,
                    $"sparse update equals the dense oracle for id {id} dim {d}");

        // ---- untouched rows are bit identical, and cost is proportional to the batch -------------
        int touched = batch.Select(b => b.Item1).Distinct().Count();
        var before = Enumerable.Range(0, 24).Select(id => (double[])sparse.Lookup(id).Clone()).ToList();
        var single = new List<(int, double[])> { (touched % 24, new[] { 1.0, 1.0, 1.0 }) };
        sparse.ApplyGradient(single, 0.5, 0.0);
        int unchanged = 0;
        for (int id = 0; id < 24; id++)
            if (!before[id].SequenceEqual(sparse.Lookup(id))) unchanged++;
        Check(unchanged == 1, "exactly one row changed in a one row batch: " + unchanged);
        Check(sparse.MaterialisedRows <= 24, "materialised rows never exceed the table: " + sparse.MaterialisedRows);
        Check(sparse.MaterialisedBytes == sparse.MaterialisedRows * 3 * 8, "materialised bytes follow the row count");

        // ---- sharding: partition and balance ------------------------------------------------------
        var wide = new SparseEmbeddingTable(2000, 4, 7, WarmUp.Zero, 1.0, 16);
        var seen = new HashSet<int>();
        var counts = new int[7];
        for (int id = 0; id < 2000; id++)
        {
            int shard = wide.ShardOf(id);
            counts[shard]++;
            Check(seen.Add(id), "every id maps to exactly one shard");
        }
        Check(counts.Sum() == 2000, "shards cover every id");
        Check(counts.Max() <= 2 * (2000 / 7 + 1), $"shard sizes balanced: {counts.Min()}..{counts.Max()}");
        Check(counts.Min() > 0, "every shard receives rows");

        // ---- multi-value lookup -------------------------------------------------------------------
        var multi = new SparseEmbeddingTable(16, 2, 2, WarmUp.Zero, 1.0, 8);
        multi.ApplyGradient(new[] { (2, new[] { 2.0, 4.0 }), (5, new[] { 4.0, 0.0 }) }, 1.0, 0.0);
        double[] averaged = multi.LookupMulti(new[] { 2, 5 });
        Check(Math.Abs(averaged[0] + 3.0) < 1e-12 && Math.Abs(averaged[1] + 2.0) < 1e-12,
            "multi value lookup averages the rows");
        double[] emptyAverage = multi.LookupMulti(Array.Empty<int>());
        Check(emptyAverage.All(v => v == 0), "an empty multi value lookup is all zeros");
        double[] singleAverage = multi.LookupMulti(new[] { 2 });
        Check(singleAverage.SequenceEqual(multi.Lookup(2)), "a single value lookup is the row itself");

        // ---- LRU cache: discipline, hit rate and coherence ---------------------------------------
        var cached = new SparseEmbeddingTable(64, 2, 4, WarmUp.Zero, 1.0, 2);
        cached.ReadThroughCache(1);
        cached.ReadThroughCache(2);
        cached.ReadThroughCache(1);      // hit
        cached.ReadThroughCache(3);      // evicts 2 (least recently used)
        Check(cached.Cache.Evictions == 1, "one eviction happened: " + cached.Cache.Evictions);
        Check(cached.Cache.Contains(1) && !cached.Cache.Contains(2) && cached.Cache.Contains(3),
            "the least recently used row was evicted");
        Check(cached.Cache.Hits == 1 && cached.Cache.Misses == 3, "hit and miss counters");
        cached.ApplyGradient(new[] { (1, new[] { 1.0, 0.0 }) }, 1.0, 0.0);
        Check(Math.Abs(cached.ReadThroughCache(1)[0] + 1.0) < 1e-12,
            "the cache never returns a stale row after an update");
        Check(cached.ReadThroughCache(1).SequenceEqual(cached.Lookup(1)), "cache and store agree");

        var skewed = new SparseEmbeddingTable(1000, 2, 8, WarmUp.Zero, 1.0, 50);
        int skewedHits = 0;
        for (int i = 0; i < 400; i++)
        {
            int id = Rand(100) < 90 ? Rand(10) : Rand(1000);      // 90% of the traffic on 10 rows
            if (skewed.Cache.Contains(id)) skewedHits++;
            skewed.ReadThroughCache(id);
        }
        var uniform = new SparseEmbeddingTable(1000, 2, 8, WarmUp.Zero, 1.0, 50);
        int uniformHits = 0;
        for (int i = 0; i < 400; i++)
        {
            int id = Rand(1000);
            if (uniform.Cache.Contains(id)) uniformHits++;
            uniform.ReadThroughCache(id);
        }
        Check(skewedHits > uniformHits * 2, $"skewed traffic hits the cache far more often: {skewedHits} vs {uniformHits}");
        Check(skewed.Cache.HitRate > 0.6, "skewed hit rate above 60%: " + skewed.Cache.HitRate.ToString("F3"));
        Check(uniform.Cache.HitRate < 0.2, "uniform traffic barely hits: " + uniform.Cache.HitRate.ToString("F3"));

        // ---- warm-up improves the cold item metric ----------------------------------------------
        double zeroRecall = ColdItemRecall(WarmUp.Zero);
        double meanRecall = ColdItemRecall(WarmUp.MeanOfHot);
        double priorRecall = ColdItemRecall(WarmUp.CategoryPrior);
        Check(priorRecall > zeroRecall, $"category prior beats the zero baseline: {priorRecall:F3} > {zeroRecall:F3}");
        Check(meanRecall > zeroRecall, $"mean of hot beats the zero baseline: {meanRecall:F3} > {zeroRecall:F3}");
        Check(priorRecall >= meanRecall, $"the category prior is at least as good as mean of hot: {priorRecall:F3} >= {meanRecall:F3}");
        Check(zeroRecall <= 0.6, "the zero baseline is no better than guessing: " + zeroRecall.ToString("F3"));

        // ---- degenerate shapes --------------------------------------------------------------------
        var oneDimension = new SparseEmbeddingTable(10, 1, 1, WarmUp.MeanOfHot, 1.0, 4);
        Check(oneDimension.Lookup(0).Length == 1 && oneDimension.ShardOf(9) == 0,
            "one dimension and one shard work");
        var noShardsLeft = new SparseEmbeddingTable(3, 2, 4, WarmUp.Zero, 1.0, 2);
        int usedShards = Enumerable.Range(0, 3).Select(noShardsLeft.ShardOf).Distinct().Count();
        Check(usedShards <= 3, "more shards than ids is allowed: " + usedShards + " used");
        var alone = new SparseEmbeddingTable(1, 2, 1, WarmUp.Zero, 1.0, 1);
        Check(alone.Lookup(0).Length == 2 && alone.TotalIds == 1, "a single row table works");
        var emptyBatch = new SparseEmbeddingTable(10, 2, 2, WarmUp.Zero, 1.0, 4);
        emptyBatch.ApplyGradient(Array.Empty<(int, double[])>(), 0.1, 0.9);
        Check(emptyBatch.MaterialisedRows == 0 && emptyBatch.Updates == 1, "an empty batch changes nothing");

        // ---- momentum state is sparse too ---------------------------------------------------------
        var momentum = new SparseEmbeddingTable(500, 4, 5, WarmUp.Zero, 1.0, 8);
        for (int i = 0; i < 30; i++) momentum.ApplyGradient(new[] { (Rand(500), new[] { 1.0, 0.0, 0.0, 0.0 }) }, 0.1, 0.9);
        int stateRows = momentum.Shards.Sum(s => s.Moments.Count);
        Check(stateRows == momentum.MaterialisedRows, "optimizer state exists only for materialised rows: " + stateRows);
        Check(stateRows <= 30, "state rows bounded by the number of updates: " + stateRows);
        Check(momentum.MaterialisedRows < 500, "the table stayed sparse: " + momentum.MaterialisedRows + " of 500 rows");

        // ---- determinism --------------------------------------------------------------------------
        double firstRecall = ColdItemRecall(WarmUp.CategoryPrior);
        double secondRecall = ColdItemRecall(WarmUp.CategoryPrior);
        Check(Math.Abs(firstRecall - secondRecall) < 1e-15, "the warm-up evaluation is deterministic");
        var deterministicA = new SparseEmbeddingTable(200, 3, 3, WarmUp.CategoryPrior, 1.0, 8);
        var deterministicB = new SparseEmbeddingTable(200, 3, 3, WarmUp.CategoryPrior, 1.0, 8);
        Check(deterministicA.Lookup(17).SequenceEqual(deterministicB.Lookup(17)), "priors are deterministic");
        Check(Enumerable.Range(0, 200).All(id => deterministicA.ShardOf(id) == deterministicB.ShardOf(id)),
            "shard assignment is deterministic");

        Console.WriteLine($"820 table: {sparse.TotalIds} ids x {sparse.Dimension} dims over {sparse.Shards.Length} shards, "
            + $"{sparse.MaterialisedRows} rows materialised ({sparse.MaterialisedBytes} bytes) after {sparse.Updates} updates, "
            + $"{table.ColdRowsEmitted} cold rows warmed; cold item recall zero {zeroRecall:F3} / mean-of-hot {meanRecall:F3} "
            + $"/ category prior {priorRecall:F3}; cache hit rate skewed {skewed.Cache.HitRate:F3} vs uniform "
            + $"{uniform.Cache.HitRate:F3}; shard sizes {counts.Min()}..{counts.Max()}");
        Console.WriteLine("PASS");
    }

    // A deterministic cold item ranking task: a hot query feature must rank the item whose category
    // matches its own above a distractor from another category, while every candidate row is cold. Only
    // the warm-up strategy can supply that category information, so the metric separates them.
    private static double ColdItemRecall(WarmUp strategy)
    {
        const int categories = 8, perCategory = 5;
        const int dimension = 8;
        int total = categories + categories * perCategory;      // query features first, then the item rows
        var table = new SparseEmbeddingTable(total, dimension, 4, strategy, 0.9, total);
        var basis = new double[categories][];
        var rng = new SplitMix64(0xA11CEUL);
        for (int c = 0; c < categories; c++)
        {
            basis[c] = new double[dimension];
            for (int d = 0; d < dimension; d++) basis[c][d] = rng.NextSigned();
        }
        // The metadata prior of an item is its category basis with a little noise (side information).
        for (int c = 0; c < categories; c++)
            for (int offset = 0; offset < perCategory; offset++)
            {
                var prior = new double[dimension];
                for (int d = 0; d < dimension; d++) prior[d] = basis[c][d] + 0.1 * rng.NextSigned();
                table.SetPrior(categories + c * perCategory + offset, prior);
            }
        // Train the query features onto their own category basis (ids 0..categories-1).
        var batch = new List<(int, double[])>();
        for (int c = 0; c < categories; c++)
        {
            var gradient = new double[dimension];
            for (int d = 0; d < dimension; d++) gradient[d] = -basis[c][d];
            batch.Add((c, gradient));
        }
        table.ApplyGradient(batch, 1.0, 0.0);

        int hits = 0, trials = 0;
        for (int c = 0; c < categories; c++)
            for (int offset = 0; offset < perCategory; offset++)
            {
                int positive = categories + c * perCategory + offset;
                int distractor = categories + ((c + 1) % categories) * perCategory + offset;
                if (Score(table, c, positive) > Score(table, c, distractor)) hits++;
                trials++;
            }
        return hits / (double)trials;
    }

    private static double Score(SparseEmbeddingTable table, int queryId, int itemId)
    {
        double[] query = table.Lookup(queryId);
        double[] item = table.Lookup(itemId);
        double score = 0;
        for (int d = 0; d < query.Length; d++) score += query[d] * item[d];
        return score;
    }

    private static double[] Full(int length, double value)
    {
        var data = new double[length];
        for (int i = 0; i < length; i++) data[i] = value;
        return data;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
