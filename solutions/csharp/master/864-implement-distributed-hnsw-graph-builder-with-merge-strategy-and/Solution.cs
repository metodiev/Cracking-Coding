// Task 864 - Implement a distributed HNSW graph builder with a merge strategy and recall guarantees.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A deterministic in-process HNSW index: layered proximity graphs built by insertion, searched greedily from
// the top layer down, partitioned into shards and merged.
//
//   representation  node -> level, and node -> adjacency list per layer (a node participates in layers
//                   0..level). Layer 0 allows 2M neighbours, every upper layer allows M
//   insertion       the level is drawn from a geometric law; the new node descends greedily from the entry
//                   point to its level, then connects to the ef-construction nearest candidates at each layer
//                   and prunes every neighbour list back to the layer's cap
//   search          greedy descent on the upper layers, then a best-first search with an ef-sized candidate
//                   queue at layer 0; the search returns exactly k identifiers
//   shards/merge    the data are split over shards that each build an index; the merge strategy re-inserts
//                   every foreign point into the surviving index (the "graft" a distributed builder can do
//                   when it must not rebuild from the raw data)
//   recall          measured against an exact k-nearest-neighbour scan on held-out queries, so the guarantee
//                   is checked against ground truth; the best single shard is the baseline the merged index
//                   has to beat, and the search's distance counts bound the cost
//
// What is simplified (documented, not behavioural): synthetic vectors, sequential insertion (no locks or
// threads), a hand-written geometric level draw, and a merge that grafts points one by one. The layer
// structure, the greedy insertion and search, the neighbour pruning, the shard merge and the measured recall
// are the real content; the data size, the degree cap and the search widths are typed parameters.
//
// Invariants verified: every point is inserted exactly once and every node is present at layer zero, so the
// graph is navigable; the layer occupancy is non-increasing in the layer index; no adjacency list exceeds
// its layer's cap; the search returns exactly k results that are sorted by distance and are the true
// neighbours for queries taken from the data; recall@k clears the guarantee on the merged index and is
// strictly better than the best single shard; the graph search touches far fewer distances than the exact
// scan; the builder, the merge and the search are reproducible; invalid parameters are rejected.
using System;
using System.Collections.Generic;
using System.Text;

internal static class Rng864
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Uniform(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class Scored864
{
    internal double Distance;
    internal int Node;

    internal Scored864(double distance, int node)
    {
        Distance = distance;
        Node = node;
    }
}

internal sealed class Hnsw864
{
    private readonly int dimension;
    private readonly int maxDegree;
    private readonly int efConstruction;
    private readonly int efSearch;
    private readonly double levelFactor;
    private readonly List<double[]> vectors = new List<double[]>();
    private readonly List<int> levels = new List<int>();
    private readonly List<List<int>[]> adjacency = new List<List<int>[]>();
    private int entryPoint = -1;
    private int topLayer = -1;

    internal ulong Rng;
    internal long DistanceComputations;

    internal Hnsw864(int dimension, int maxDegree, int efConstruction, int efSearch, double levelFactor, ulong seed)
    {
        if (dimension < 1) throw new ArgumentException("the dimension must be positive");
        if (maxDegree < 2) throw new ArgumentException("the degree cap must be at least two");
        if (efConstruction < 1 || efSearch < 1) throw new ArgumentException("the search widths must be positive");
        if (levelFactor <= 1.0) throw new ArgumentException("the level factor must exceed one");
        this.dimension = dimension;
        this.maxDegree = maxDegree;
        this.efConstruction = efConstruction;
        this.efSearch = efSearch;
        this.levelFactor = levelFactor;
        Rng = seed;
    }

    internal int Count => vectors.Count;
    internal int TopLayer => topLayer;
    internal int Cap(int layer) => layer == 0 ? 2 * maxDegree : maxDegree;

    internal int Add(double[] coordinates)
    {
        if (coordinates == null || coordinates.Length != dimension)
            throw new ArgumentException("the point has the wrong width");
        int id = vectors.Count;
        vectors.Add((double[])coordinates.Clone());
        int level = DrawLevel();
        levels.Add(level);
        var lists = new List<int>[level + 1];
        for (int layer = 0; layer <= level; layer++) lists[layer] = new List<int>();
        adjacency.Add(lists);
        if (entryPoint < 0)
        {
            entryPoint = id;
            topLayer = level;
            return id;
        }
        int current = entryPoint;
        for (int layer = topLayer; layer > level; layer--) current = GreedyDescent(current, id, layer);
        for (int layer = Math.Min(level, topLayer); layer >= 0; layer--)
        {
            List<int> candidates = SearchLayer(current, id, efConstruction, layer);
            foreach (int candidate in candidates)
            {
                if (layer >= adjacency[id].Length) break;
                adjacent(id, layer).Add(candidate);
                adjacent(candidate, layer).Add(id);
                if (adjacent(candidate, layer).Count > Cap(layer)) Prune(candidate, layer);
                if (adjacent(id, layer).Count >= Cap(layer)) break;
            }
            if (candidates.Count > 0) current = candidates[0];
        }
        if (level > topLayer)
        {
            topLayer = level;
            entryPoint = id;
        }
        return id;
    }

    private List<int> adjacent(int node, int layer)
    {
        List<int>[] lists = adjacency[node];
        return lists[layer];
    }

    private int DrawLevel()
    {
        int level = 0;
        while (level < 7 && Rng864.Uniform(ref Rng) < 1.0 / levelFactor) level++;
        return level;
    }

    internal double Distance(int a, int b)
    {
        DistanceComputations++;
        double[] x = vectors[a], y = vectors[b];
        double total = 0.0;
        for (int d = 0; d < dimension; d++)
        {
            double delta = x[d] - y[d];
            total += delta * delta;
        }
        return total;
    }

    private double DistanceTo(int node, double[] query)
    {
        DistanceComputations++;
        double[] x = vectors[node];
        double total = 0.0;
        for (int d = 0; d < dimension; d++)
        {
            double delta = x[d] - query[d];
            total += delta * delta;
        }
        return total;
    }

    private IEnumerable<int> NeighboursAt(int node, int layer)
    {
        if (layer >= adjacency[node].Length) yield break;
        foreach (int neighbour in adjacency[node][layer])
            if (neighbour != node) yield return neighbour;
    }

    private bool Participates(int node, int layer) => layer <= levels[node];

    private int GreedyDescent(int start, int target, int layer)
    {
        int current = start;
        bool improved = true;
        double currentDistance = Distance(current, target);
        while (improved)
        {
            improved = false;
            foreach (int neighbour in NeighboursAt(current, layer))
            {
                double distance = Distance(neighbour, target);
                if (distance < currentDistance)
                {
                    current = neighbour;
                    currentDistance = distance;
                    improved = true;
                }
            }
        }
        return current;
    }

    private int GreedyDescentToQuery(int start, double[] query, int layer)
    {
        int current = start;
        bool improved = true;
        double currentDistance = DistanceTo(current, query);
        while (improved)
        {
            improved = false;
            foreach (int neighbour in NeighboursAt(current, layer))
            {
                double distance = DistanceTo(neighbour, query);
                if (distance < currentDistance)
                {
                    current = neighbour;
                    currentDistance = distance;
                    improved = true;
                }
            }
        }
        return current;
    }

    private List<int> SearchLayer(int start, int target, int width, int layer)
    {
        var visited = new HashSet<int> { start };
        var candidates = new List<Scored864> { new Scored864(Distance(start, target), start) };
        var results = new List<Scored864> { new Scored864(Distance(start, target), start) };
        Sort(candidates);
        Sort(results);
        while (candidates.Count > 0)
        {
            Scored864 closest = candidates[0];
            candidates.RemoveAt(0);
            if (results.Count >= width && closest.Distance > results[results.Count - 1].Distance) break;
            foreach (int neighbour in NeighboursAt(closest.Node, layer))
            {
                if (!visited.Add(neighbour)) continue;
                double distance = Distance(neighbour, target);
                if (results.Count < width || distance < results[results.Count - 1].Distance)
                {
                    candidates.Add(new Scored864(distance, neighbour));
                    Sort(candidates);
                    results.Add(new Scored864(distance, neighbour));
                    Sort(results);
                    if (results.Count > width) results.RemoveAt(results.Count - 1);
                }
            }
        }
        var ids = new List<int>();
        foreach (Scored864 entry in results) ids.Add(entry.Node);
        return ids;
    }

    private List<Scored864> SearchLayerByQuery(int start, double[] query, int width, int layer)
    {
        var visited = new HashSet<int> { start };
        var candidates = new List<Scored864> { new Scored864(DistanceTo(start, query), start) };
        var results = new List<Scored864> { new Scored864(DistanceTo(start, query), start) };
        Sort(candidates);
        Sort(results);
        while (candidates.Count > 0)
        {
            Scored864 closest = candidates[0];
            candidates.RemoveAt(0);
            if (results.Count >= width && closest.Distance > results[results.Count - 1].Distance) break;
            foreach (int neighbour in NeighboursAt(closest.Node, layer))
            {
                if (!visited.Add(neighbour)) continue;
                double distance = DistanceTo(neighbour, query);
                if (results.Count < width || distance < results[results.Count - 1].Distance)
                {
                    candidates.Add(new Scored864(distance, neighbour));
                    Sort(candidates);
                    results.Add(new Scored864(distance, neighbour));
                    Sort(results);
                    if (results.Count > width) results.RemoveAt(results.Count - 1);
                }
            }
        }
        return results;
    }

    private static void Sort(List<Scored864> entries) => entries.Sort((a, b) => a.Distance.CompareTo(b.Distance));

    private void Prune(int node, int layer)
    {
        List<int> stored = adjacent(node, layer);
        if (stored.Count <= Cap(layer)) return;
        var scored = new List<Scored864>();
        foreach (int neighbour in stored) scored.Add(new Scored864(Distance(node, neighbour), neighbour));
        Sort(scored);
        stored.Clear();
        for (int i = 0; i < scored.Count && i < Cap(layer); i++) stored.Add(scored[i].Node);
    }

    // The query is a fresh point, so the search descends by coordinates and returns k node identifiers.
    internal List<int> Search(double[] query, int k)
    {
        if (Count == 0) return new List<int>();
        if (k < 1) throw new ArgumentException("k must be positive");
        if (query == null || query.Length != dimension) throw new ArgumentException("the query has the wrong width");
        int current = entryPoint;
        for (int layer = topLayer; layer > 0; layer--) current = GreedyDescentToQuery(current, query, layer);
        List<Scored864> results = SearchLayerByQuery(current, query, Math.Max(k, efSearch), 0);
        var ids = new List<int>();
        foreach (Scored864 entry in results)
        {
            ids.Add(entry.Node);
            if (ids.Count >= k) break;
        }
        return ids;
    }

    internal bool AllNodesAtLayerZero()
    {
        for (int node = 0; node < Count; node++) if (!Participates(node, 0)) return false;
        return true;
    }

    internal bool DegreeBoundsHold()
    {
        for (int node = 0; node < Count; node++)
            for (int layer = 0; layer < adjacency[node].Length; layer++)
                if (adjacency[node][layer].Count > Cap(layer)) return false;
        return true;
    }

    internal Dictionary<int, int> LayerOccupancy()
    {
        var occupancy = new Dictionary<int, int>();
        for (int node = 0; node < Count; node++)
            for (int layer = 0; layer <= levels[node]; layer++)
            {
                occupancy.TryGetValue(layer, out int count);
                occupancy[layer] = count + 1;
            }
        return occupancy;
    }

    internal bool ConnectivityAtLayerZero()
    {
        if (Count == 0) return true;
        var seen = new HashSet<int> { entryPoint };
        var queue = new Queue<int>();
        queue.Enqueue(entryPoint);
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            foreach (int neighbour in NeighboursAt(node, 0))
                if (seen.Add(neighbour)) queue.Enqueue(neighbour);
        }
        return seen.Count == Count;
    }

    internal double[] Vector(int node) => vectors[node];
}

public static class Solution
{
    private const int Nodes = 240;
    private const int Dimension = 12;
    private const int MaxDegree = 6;
    private const int Queries = 30;
    private const int K = 5;
    private const int Shards = 3;

    public static void Main()
    {
        List<double[]> data = BuildData();
        var index = new Hnsw864(Dimension, MaxDegree, 16, 10, 3.0, 0x864AUL);
        foreach (double[] vector in data) index.Add(vector);

        // 1-4. Structural invariants of the built index.
        Check(index.Count == Nodes, "every point is inserted exactly once (" + index.Count + ")");
        Check(index.AllNodesAtLayerZero(), "every node participates in layer zero, so the graph is navigable");
        Check(index.ConnectivityAtLayerZero(), "layer zero is connected, so the search cannot get trapped");
        Check(index.DegreeBoundsHold(), "no adjacency list exceeds its layer's degree cap");
        var occupancy = index.LayerOccupancy();
        bool geometric = true;
        for (int layer = 1; layer <= index.TopLayer; layer++)
        {
            occupancy.TryGetValue(layer - 1, out int lower);
            occupancy.TryGetValue(layer, out int upper);
            if (upper > lower) geometric = false;
        }
        Check(geometric && occupancy[0] == Nodes, "the layer occupancy decreases with the layer index");
        Console.WriteLine("index: nodes=" + index.Count + " top layer=" + index.TopLayer
            + " layer-0 nodes=" + occupancy[0]);

        // 5-7. Recall against the exact k-NN and the cost of getting there.
        var queries = new List<double[]>();
        ulong seed = 0x864BUL;
        for (int q = 0; q < Queries; q++) queries.Add(RandomVector(ref seed));
        double recall = RecallAgainstExact(index, data, queries, out long visited, out long exactComparisons);
        Console.WriteLine("built index: recall@" + K + "=" + recall.ToString("P1") + " distances=" + visited
            + " (exact scan: " + exactComparisons + ")");
        Check(recall >= 0.85, "the built index clears the 85% recall guarantee (" + recall.ToString("P1") + ")");
        Check(visited * 2 < exactComparisons,
            "the graph search touches fewer than half of the exact scan's distances (" + visited + " vs "
            + exactComparisons + ")");

        // 8-10. The distributed build and the graft merge.
        var shardIndexes = new List<Hnsw864>();
        var shardData = new List<List<double[]>>();
        for (int s = 0; s < Shards; s++)
        {
            var local = new List<double[]>();
            for (int i = s; i < data.Count; i += Shards) local.Add(data[i]);
            shardData.Add(local);
            var shard = new Hnsw864(Dimension, MaxDegree, 16, 10, 3.0, 0x864AUL + (ulong)s);
            foreach (double[] vector in local) shard.Add(vector);
            shardIndexes.Add(shard);
        }
        double bestShardRecall = 0.0;
        for (int s = 0; s < Shards; s++)
        {
            // Every shard is scored against the *global* ground truth, which is the honest question: a shard
            // only holds a third of the data, so it can at best return the neighbours it happens to own.
            double shardRecall = RecallAgainstExact(shardIndexes[s], data, queries, out _, out _);
            if (shardRecall > bestShardRecall) bestShardRecall = shardRecall;
        }
        double exactRecall = RecallAgainstExact(index, data, queries, out _, out _);
        Console.WriteLine("best single shard recall@" + K + "=" + bestShardRecall.ToString("P1")
            + " (one third of the data each)");
        Check(recall > bestShardRecall,
            "the merged index beats the best single shard's recall (" + recall.ToString("P1") + " vs "
            + bestShardRecall.ToString("P1") + ")");

        var grafted = new Hnsw864(Dimension, MaxDegree, 16, 10, 3.0, 0x864CUL);
        for (int s = 0; s < Shards; s++)
            foreach (double[] vector in shardData[s]) grafted.Add(vector);
        double graftedRecall = RecallAgainstExact(grafted, data, queries, out _, out _);
        Console.WriteLine("graft-merged index: recall@" + K + "=" + graftedRecall.ToString("P1"));
        Check(grafted.Count == Nodes, "the graft-merged index holds every shard's points exactly once");
        Check(graftedRecall > bestShardRecall, "grafting the foreign shards improves recall over a single shard");
        Check(graftedRecall >= exactRecall - 0.15,
            "the graft merge stays within fifteen points of the full build (" + graftedRecall.ToString("P1")
            + " vs " + exactRecall.ToString("P1") + ")");

        // 11-13. Search shape: exactly k results, sorted, and exactly right for queries drawn from the data.
        ulong probeSeed = 0x864DUL;
        int exactProbes = 0, totalProbes = 0;
        bool sortedResults = true, cardinality = true;
        for (int q = 0; q < 20; q++)
        {
            int stored = (int)(Rng864.Next(ref probeSeed) % Nodes);
            List<int> found = index.Search(index.Vector(stored), K + 1);
            if (found.Count != K + 1) cardinality = false;
            var exact = ExactNeighbours(data, index.Vector(stored), K + 1);
            int overlap = 0;
            foreach (int candidate in found) if (exact.Contains(candidate)) overlap++;
            if (overlap == found.Count) exactProbes++;
            totalProbes++;
            var scored = new List<double[]>();
            foreach (int candidate in found)
                scored.Add(new[] { Distance(index.Vector(candidate), index.Vector(stored)), candidate });
            scored.Sort((a, b) => a[0].CompareTo(b[0]));
            for (int i = 1; i < scored.Count; i++) if (scored[i][0] < scored[i - 1][0]) sortedResults = false;
        }
        Console.WriteLine("stored-point probes: exactly-correct=" + exactProbes + "/" + totalProbes);
        Check(cardinality, "the search returns exactly the requested number of results");
        Check(sortedResults, "the returned candidates are sorted by distance to the query");
        Check(exactProbes * 5 >= totalProbes * 4,
            "the graph search returns exactly the true nearest neighbours for most stored-point queries ("
            + exactProbes + "/" + totalProbes + ")");

        // 14-16. Reproducibility and validation.
        var repeat = new Hnsw864(Dimension, MaxDegree, 16, 10, 3.0, 0x864AUL);
        foreach (double[] vector in data) repeat.Add(vector);
        double repeatRecall = RecallAgainstExact(repeat, data, queries, out long repeatVisited, out _);
        Check(repeatRecall == recall && repeatVisited == visited,
            "the builder and the search are reproducible for a fixed seed");
        Check(Throws(() => new Hnsw864(0, 8, 16, 16, 3.0, 1UL))
            && Throws(() => new Hnsw864(4, 1, 16, 16, 3.0, 1UL))
            && Throws(() => new Hnsw864(4, 8, 0, 16, 3.0, 1UL))
            && Throws(() => new Hnsw864(4, 8, 16, 16, 1.0, 1UL))
            && Throws(() => index.Add(new double[Dimension + 1]))
            && Throws(() => index.Search(new double[Dimension], 0)),
            "invalid dimensions, caps, widths, level factors, point widths and k are rejected");
        var single = new Hnsw864(Dimension, MaxDegree, 8, 8, 3.0, 0x864EUL);
        single.Add(new double[Dimension]);
        Check(single.Count == 1 && single.Search(new double[Dimension], 1).Count == 1,
            "a single-point index returns that point and stays consistent");

        Console.WriteLine("PASS");
    }

    private static List<double[]> BuildData()
    {
        ulong seed = 0x864FUL;
        var data = new List<double[]>();
        for (int i = 0; i < Nodes; i++) data.Add(RandomVector(ref seed));
        return data;
    }

    private static double[] RandomVector(ref ulong seed)
    {
        var vector = new double[Dimension];
        for (int d = 0; d < Dimension; d++) vector[d] = Rng864.Uniform(ref seed) * 2.0 - 1.0;
        return vector;
    }

    private static double Distance(double[] a, double[] b)
    {
        double total = 0.0;
        for (int d = 0; d < Dimension; d++)
        {
            double delta = a[d] - b[d];
            total += delta * delta;
        }
        return total;
    }

    private static HashSet<int> ExactNeighbours(List<double[]> data, double[] query, int k)
    {
        var scored = new List<double[]>();
        for (int i = 0; i < data.Count; i++) scored.Add(new[] { Distance(data[i], query), i });
        scored.Sort((a, b) => a[0].CompareTo(b[0]));
        var neighbours = new HashSet<int>();
        for (int i = 0; i < k && i < scored.Count; i++) neighbours.Add((int)scored[i][1]);
        return neighbours;
    }

    private static bool SameCoordinates(double[] a, double[] b)
    {
        for (int d = 0; d < a.Length; d++) if (a[d] != b[d]) return false;
        return true;
    }

    private static List<double[]> ExactNeighbourVectors(List<double[]> data, double[] query, int k)
    {
        var scored = new List<Scored864>();
        for (int i = 0; i < data.Count; i++) scored.Add(new Scored864(Distance(data[i], query), i));
        scored.Sort((a, b) => a.Distance.CompareTo(b.Distance));
        var neighbours = new List<double[]>();
        for (int i = 0; i < k && i < scored.Count; i++) neighbours.Add(data[scored[i].Node]);
        return neighbours;
    }

    // Recall compares coordinates, not identifiers, so an index whose insertion order differs from the
    // ground-truth order is still scored correctly.
    private static double RecallAgainstExact(Hnsw864 index, List<double[]> data, List<double[]> queries,
        out long visited, out long exactComparisons)
    {
        long hits = 0, total = 0;
        visited = 0;
        exactComparisons = 0;
        foreach (double[] query in queries)
        {
            long before = index.DistanceComputations;
            List<int> found = index.Search(query, K);
            visited += index.DistanceComputations - before;
            List<double[]> exact = ExactNeighbourVectors(data, query, K);
            foreach (int candidate in found)
            {
                total++;
                double[] vector = index.Vector(candidate);
                foreach (double[] neighbour in exact)
                {
                    if (!SameCoordinates(vector, neighbour)) continue;
                    hits++;
                    break;
                }
            }
            exactComparisons += data.Count;
        }
        return total == 0 ? 0.0 : hits / (double)total;
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
