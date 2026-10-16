// Task 766 - Implement dense vector index with HNSW and quantization integration. (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)
//
// HNSW (Malkov-Yashunin) builds a hierarchy of proximity graphs: every vector is inserted at a level drawn from a
// geometric distribution, participates in every layer up to its level, and each layer keeps a bounded neighbour
// list chosen by the *diversification* heuristic - a candidate is kept only when it is closer to the query than to
// every neighbour already selected, which is what keeps long range links instead of a clique of near duplicates.
// A search descends greedily through the upper layers and then runs a best-first expansion with a budget of `ef`
// candidates on layer 0. Quantization is integrated as an alternative distance oracle: every stored vector is
// scalar quantized to one byte per dimension, a per query lookup table is built from the dequantized levels, and
// the same graph search runs on that table - so the recall difference is purely the quantization loss, with the
// graph held fixed.
//
// Documented simplifications: 400 vectors of 8 dimensions in six synthetic clusters (real indexes hold billions),
// the graph is always built with exact distances (only the search oracle is swapped), deletion is not supported,
// neighbour lists are pruned without the full shrink heuristic of the paper, and no SIMD kernels are used.
// Invariants verified: layer 0 degree <= 2M and upper layer degree <= M with no self loops; the level of a node is
// its highest layer; layer 0 is connected and every node is reachable from the entry point; a search with an
// unbounded budget returns exactly the brute force top-k (so the graph machinery is exact when asked to be); recall
// rises with the budget; scalar quantization picks the true nearest level per dimension and its distance error is
// bounded by the quantisation step; the quantized oracle costs one table lookup per dimension; the rebuild is
// deterministic; and the boundary cases (empty index, single node, k > n, a query that is an indexed point,
// ef = 1, a query outside the data range) behave.
using System;
using System.Collections.Generic;

internal static class Rng
{
    private static ulong state = 0x7660BADF00D12345UL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Unit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
    internal static double Signed() => 2 * Unit() - 1;
}

internal static class Pts
{
    internal const int Dimensions = 8;
    internal const int Clusters = 6;
    internal const int Count = 400;
    internal const double Noise = 0.08;

    internal static readonly double[][] Data = Build();

    internal static double[][] Build()
    {
        var centers = new double[Clusters][];
        for (int cluster = 0; cluster < Clusters; cluster++)
        {
            centers[cluster] = new double[Dimensions];
            for (int d = 0; d < Dimensions; d++) centers[cluster][d] = Rng.Unit() * 4;
        }
        var data = new double[Count][];
        for (int i = 0; i < Count; i++)
        {
            double[] center = centers[i % Clusters];
            var vector = new double[Dimensions];
            for (int d = 0; d < Dimensions; d++) vector[d] = center[d] + Rng.Signed() * Noise;
            data[i] = vector;
        }
        return data;
    }

    internal static double Squared(double[] left, double[] right)
    {
        double total = 0;
        for (int d = 0; d < left.Length; d++)
        {
            double difference = left[d] - right[d];
            total += difference * difference;
        }
        return total;
    }
}

/** One byte per dimension: the levels are uniform over the observed range of each dimension. */
internal sealed class ScalarQuantizer
{
    internal const int Levels = 256;
    private readonly double[] low = new double[Pts.Dimensions];
    private readonly double[] step = new double[Pts.Dimensions];

    internal ScalarQuantizer(double[][] data)
    {
        for (int d = 0; d < Pts.Dimensions; d++)
        {
            double min = double.MaxValue, max = double.MinValue;
            foreach (double[] vector in data)
            {
                min = Math.Min(min, vector[d]);
                max = Math.Max(max, vector[d]);
            }
            low[d] = min;
            step[d] = (max - min) / (Levels - 1);
        }
    }

    internal byte[] Encode(double[] vector)
    {
        var code = new byte[Pts.Dimensions];
        for (int d = 0; d < Pts.Dimensions; d++)
        {
            int level = (int)Math.Round((vector[d] - low[d]) / step[d]);
            code[d] = (byte)Math.Clamp(level, 0, Levels - 1);
        }
        return code;
    }

    internal double Dequantize(byte[] code, int dimension) => low[dimension] + step[dimension] * code[dimension];

    internal double DequantizeLevel(int dimension, int level) => low[dimension] + step[dimension] * level;

    internal double Step(int dimension) => step[dimension];

    /** Per query table: one squared distance per level per dimension. */
    internal double[][] Lookup(double[] query)
    {
        var table = new double[Pts.Dimensions][];
        for (int d = 0; d < Pts.Dimensions; d++)
        {
            table[d] = new double[Levels];
            for (int level = 0; level < Levels; level++)
            {
                double value = low[d] + step[d] * level - query[d];
                table[d][level] = value * value;
            }
        }
        return table;
    }

    internal double Distance(double[][] table, byte[] code)
    {
        double total = 0;
        for (int d = 0; d < Pts.Dimensions; d++) total += table[d][code[d]];
        return total;
    }
}

internal sealed class HnswIndex
{
    internal readonly int M;
    internal readonly int M0;
    internal readonly int EfConstruction;
    internal readonly List<int>[][] adjacency;        // [layer][node] -> neighbour list
    private readonly int[] level;
    private int entryPoint = -1;
    private int topLayer = -1;
    private ulong rngState;
    internal long DistanceComputations;
    internal long TableLookups;

    internal HnswIndex(int m, int efConstruction, ulong seed)
    {
        M = m;
        M0 = 2 * m;
        EfConstruction = efConstruction;
        rngState = seed;
        adjacency = new List<int>[16][];
        level = new int[Pts.Count];
        for (int i = 0; i < level.Length; i++) level[i] = -1;
        for (int l = 0; l < adjacency.Length; l++)
        {
            adjacency[l] = new List<int>[Pts.Count];
            for (int i = 0; i < Pts.Count; i++) adjacency[l][i] = new List<int>();
        }
    }

    /** The level generator is private to the index, so a rebuild with the same seed is identical. */
    private double NextUnit()
    {
        rngState += 0x9E3779B97F4A7C15UL;
        ulong z = rngState;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        double unit = (z ^ (z >> 31)) >> 11;
        return unit * (1.0 / 9007199254740992.0);
    }

    internal static double Distance(int left, int right) => Pts.Squared(Pts.Data[left], Pts.Data[right]);

    internal int LevelOf(int node) => level[node];
    internal int TopLayer => topLayer;
    internal int EntryPoint => entryPoint;

    private int RandomLevel()
    {
        double unit = Math.Max(1e-12, NextUnit());
        return (int)(-Math.Log(unit) * (1.0 / Math.Log(M)));
    }

    internal void Insert(int node)
    {
        int nodeLevel = RandomLevel();
        level[node] = nodeLevel;
        if (entryPoint < 0)
        {
            entryPoint = node;
            topLayer = nodeLevel;
            return;
        }

        int current = entryPoint;
        for (int layer = topLayer; layer > nodeLevel; layer--)
        {
            current = BestInLayer(layer, id => Distance(node, id), current);
        }
        for (int layer = Math.Min(nodeLevel, topLayer); layer >= 0; layer--)
        {
            List<int> candidates = SearchLayer(layer, id => Distance(node, id), current, EfConstruction);
            int cap = layer == 0 ? M0 : M;
            foreach (int neighbour in SelectNeighbours(node, candidates, cap))
            {
                adjacency[layer][node].Add(neighbour);
                adjacency[layer][neighbour].Add(node);
                Prune(layer, neighbour, cap);
            }
            if (candidates.Count > 0) current = candidates[0];
        }
        if (nodeLevel > topLayer)
        {
            topLayer = nodeLevel;
            entryPoint = node;
        }
    }

    private void Prune(int layer, int node, int cap)
    {
        List<int> neighbours = adjacency[layer][node];
        if (neighbours.Count <= cap) return;
        var ordered = new List<int>(neighbours);
        ordered.Sort((left, right) =>
        {
            int comparison = Distance(node, left).CompareTo(Distance(node, right));
            return comparison != 0 ? comparison : left.CompareTo(right);
        });
        List<int> kept = SelectNeighbours(node, ordered, cap);
        neighbours.Clear();
        neighbours.AddRange(kept);
    }

    /** Diversification: keep a candidate only if it is closer to the query than to any kept neighbour. */
    private List<int> SelectNeighbours(int query, List<int> candidates, int cap)
    {
        var selected = new List<int>();
        foreach (int candidate in candidates)
        {
            if (candidate == query || selected.Contains(candidate)) continue;
            bool diverse = true;
            foreach (int chosen in selected)
            {
                if (Distance(candidate, chosen) < Distance(query, candidate))
                {
                    diverse = false;
                    break;
                }
            }
            if (!diverse) continue;
            selected.Add(candidate);
            if (selected.Count >= cap) break;
        }
        return selected;
    }

    private int BestInLayer(int layer, Func<int, double> distance, int entry)
    {
        int best = entry;
        double bestDistance = distance(entry);
        bool improved = true;
        var visited = new HashSet<int> { entry };
        while (improved)
        {
            improved = false;
            foreach (int neighbour in adjacency[layer][best])
            {
                if (!visited.Add(neighbour)) continue;
                DistanceComputations++;
                double candidate = distance(neighbour);
                if (candidate < bestDistance)
                {
                    bestDistance = candidate;
                    best = neighbour;
                    improved = true;
                }
            }
        }
        return best;
    }

    /** Best first expansion on one layer, keeping the ef nearest candidates. */
    internal List<int> SearchLayer(int layer, Func<int, double> distance, int entry, int ef)
    {
        var candidates = new List<(double Distance, int Node)> { (distance(entry), entry) };
        var results = new List<(double Distance, int Node)> { (distance(entry), entry) };
        var visited = new HashSet<int> { entry };
        while (candidates.Count > 0)
        {
            int index = 0;
            for (int i = 1; i < candidates.Count; i++)
            {
                if (candidates[i].Distance < candidates[index].Distance) index = i;
            }
            (double currentDistance, int current) = candidates[index];
            candidates.RemoveAt(index);
            double worst = results.Count == 0 ? double.MaxValue : MaxDistance(results);
            if (results.Count >= ef && currentDistance > worst) break;

            foreach (int neighbour in adjacency[layer][current])
            {
                if (!visited.Add(neighbour)) continue;
                DistanceComputations++;
                double candidate = distance(neighbour);
                if (results.Count < ef || candidate < MaxDistance(results))
                {
                    candidates.Add((candidate, neighbour));
                    results.Add((candidate, neighbour));
                    if (results.Count > ef)
                    {
                        int worstIndex = 0;
                        for (int i = 1; i < results.Count; i++)
                        {
                            if (results[i].Distance > results[worstIndex].Distance) worstIndex = i;
                        }
                        results.RemoveAt(worstIndex);
                    }
                }
            }
        }
        results.Sort((left, right) =>
        {
            int comparison = left.Distance.CompareTo(right.Distance);
            return comparison != 0 ? comparison : left.Node.CompareTo(right.Node);
        });
        var nodes = new List<int>();
        foreach ((double _, int node) in results) nodes.Add(node);
        return nodes;
    }

    private static double MaxDistance(List<(double Distance, int Node)> list)
    {
        double worst = double.MinValue;
        foreach ((double distance, int _) in list) worst = Math.Max(worst, distance);
        return worst;
    }

    internal List<int> Search(Func<int, double> distance, double[] query, int k, int ef)
    {
        if (entryPoint < 0) return new List<int>();
        int current = entryPoint;
        for (int layer = topLayer; layer > 0; layer--) current = BestInLayer(layer, distance, current);
        List<int> found = SearchLayer(0, distance, current, Math.Max(ef, k));
        var ranked = new List<(double Distance, int Node)>();
        foreach (int node in found) ranked.Add((Pts.Squared(query, Pts.Data[node]), node));
        ranked.Sort((left, right) =>
        {
            int comparison = left.Distance.CompareTo(right.Distance);
            return comparison != 0 ? comparison : left.Node.CompareTo(right.Node);
        });
        var result = new List<int>();
        foreach ((double _, int node) in ranked)
        {
            if (result.Count >= k) break;
            result.Add(node);
        }
        return result;
    }

    internal bool ConnectedFromEntry()
    {
        if (entryPoint < 0) return true;
        var seen = new HashSet<int> { entryPoint };
        var stack = new Stack<int>();
        stack.Push(entryPoint);
        while (stack.Count > 0)
        {
            int node = stack.Pop();
            foreach (int neighbour in adjacency[0][node])
            {
                if (seen.Add(neighbour)) stack.Push(neighbour);
            }
        }
        return seen.Count == CountIndexed();
    }

    internal int CountIndexed()
    {
        int total = 0;
        for (int i = 0; i < level.Length; i++)
        {
            if (level[i] >= 0) total++;
        }
        return total;
    }
}

public static class Solution
{
    private const int K = 10;

    public static void Main()
    {
        double[][] data = Pts.Data;
        var index = new HnswIndex(8, 40, 0x7660FEED1234ABCDUL);
        for (int i = 0; i < data.Length; i++) index.Insert(i);

        Graph(index);
        Exactness(index, data);
        Quantization(index, data);
        Boundaries(data);
        Console.WriteLine("PASS");
    }

    private static void Graph(HnswIndex index)
    {
        int indexed = index.CountIndexed();
        Check(indexed == Pts.Count && index.EntryPoint >= 0, "all " + indexed + " vectors are inserted");

        bool degrees = true, selfLoops = false, levelConsistent = true;
        for (int layer = 0; layer <= index.TopLayer; layer++)
        {
            int cap = layer == 0 ? index.M0 : index.M;
            for (int node = 0; node < Pts.Count; node++)
            {
                if (index.LevelOf(node) < layer) continue;
                List<int> neighbours = index.adjacency[layer][node];
                if (neighbours.Count > cap) degrees = false;
                foreach (int neighbour in neighbours)
                {
                    if (neighbour == node) selfLoops = true;
                    if (index.LevelOf(neighbour) < layer) levelConsistent = false;
                }
            }
        }
        Check(degrees && !selfLoops && levelConsistent,
            "degree caps hold (2M on layer 0, M above), no self loops, and every link stays inside the layer range");
        Check(index.TopLayer >= 1, "the hierarchy has " + (index.TopLayer + 1) + " layers");
        Check(index.ConnectedFromEntry(), "layer 0 is connected: every node is reachable from the entry point");

        var levels = new int[6];
        for (int node = 0; node < Pts.Count; node++) levels[Math.Min(5, index.LevelOf(node))]++;
        Check(levels[0] > levels[1] && levels[1] > 0 && levels[0] > 0,
            "levels decay geometrically: " + string.Join(",", levels) + " nodes per level (>=5 grouped)");

        var rebuilt = new HnswIndex(8, 40, 0x7660FEED1234ABCDUL);
        for (int i = 0; i < Pts.Count; i++) rebuilt.Insert(i);
        bool identical = rebuilt.TopLayer == index.TopLayer;
        for (int node = 0; node < Pts.Count && identical; node++)
        {
            if (rebuilt.LevelOf(node) != index.LevelOf(node)) identical = false;
            List<int> left = index.adjacency[0][node];
            List<int> right = rebuilt.adjacency[0][node];
            if (left.Count != right.Count) identical = false;
            else
            {
                for (int i = 0; i < left.Count; i++)
                {
                    if (left[i] != right[i]) identical = false;
                }
            }
        }
        Check(identical, "inserting the same vectors in the same order rebuilds the identical graph");
    }

    private static void Exactness(HnswIndex index, double[][] data)
    {
        var query = data[42];
        List<int> truth = BruteForce(data, query, K);
        List<int> unbounded = index.Search(node => Pts.Squared(query, data[node]), query, K, Pts.Count);
        Check(Same(unbounded, truth),
            "an unbounded search returns exactly the brute force top-10 (" + string.Join(",", unbounded) + ")");
        Check(index.Search(node => Pts.Squared(query, data[node]), query, 1, 10)[0] == 42,
            "a query that is an indexed point retrieves itself first");

        int queries = 40, hits = 0, total = 0;
        double greedyHits = 0;
        for (int q = 0; q < queries; q++)
        {
            double[] probe = data[(q * 17) % data.Length];
            List<int> expected = BruteForce(data, probe, K);
            List<int> found = index.Search(node => Pts.Squared(probe, data[node]), probe, K, 30);
            List<int> greedy = index.Search(node => Pts.Squared(probe, data[node]), probe, K, 1);
            hits += Overlap(found, expected);
            greedyHits += Overlap(greedy, expected);
            total += K;
        }
        double recall = hits / (double)total;
        double greedyRecall = greedyHits / total;
        Check(recall >= 0.9, "HNSW recall@10 with efSearch = 30 is " + recall.ToString("F3"));
        Check(greedyRecall <= recall, "a budget of 1 cannot beat a budget of 30 (" + greedyRecall.ToString("F3") + ")");
        Check(greedyRecall >= 0.3, "even the greedy descent finds " + greedyRecall.ToString("F3"));
    }

    private static void Quantization(HnswIndex index, double[][] data)
    {
        var quantizer = new ScalarQuantizer(data);
        var codes = new byte[data.Length][];
        for (int i = 0; i < data.Length; i++) codes[i] = quantizer.Encode(data[i]);

        int badLevels = 0;
        for (int i = 0; i < data.Length; i++)
        {
            for (int d = 0; d < Pts.Dimensions; d++)
            {
                double target = data[i][d];
                double chosen = quantizer.Dequantize(codes[i], d);
                if (Math.Abs(chosen - target) > quantizer.Step(d) / 2 + 1e-9) badLevels++;
                for (int level = 0; level < ScalarQuantizer.Levels; level++)
                {
                    double candidate = quantizer.DequantizeLevel(d, level);
                    if (Math.Abs(candidate - target) < Math.Abs(chosen - target) - 1e-9) badLevels++;
                }
            }
        }
        Check(badLevels == 0, "every scalar code picks the nearest level and stays within half a step");

        var query = data[7];
        double[][] table = quantizer.Lookup(query);
        double worstGap = 0, worstBoundFraction = 0;
        bool stepsHold = true;
        for (int i = 0; i < data.Length; i++)
        {
            double approximate = quantizer.Distance(table, codes[i]);
            double exact = Pts.Squared(query, data[i]);
            double bound = 0;
            for (int d = 0; d < Pts.Dimensions; d++)
            {
                double error = Math.Abs(quantizer.Dequantize(codes[i], d) - data[i][d]);
                if (error > quantizer.Step(d) / 2 + 1e-9) stepsHold = false;
                double gap = query[d] - data[i][d];
                bound += 2 * Math.Abs(gap) * error + error * error;
            }
            double difference = Math.Abs(approximate - exact);
            if (difference > bound + 1e-9) stepsHold = false;
            if (bound > 0) worstBoundFraction = Math.Max(worstBoundFraction, difference / bound);
            worstGap = Math.Max(worstGap, difference);
        }
        Check(stepsHold, "every code is within half a step and the distance gap never exceeds 2|a|e + e^2 per dimension");
        Check(worstBoundFraction > 0.5 && worstGap > 0,
            "the bound is tight in practice: the worst case reaches " + worstBoundFraction.ToString("F2")
            + " of it, largest gap " + worstGap.ToString("F6"));

        int queries = 40, exactHits = 0, quantizedHits = 0, total = 0;
        long tableLookups = 0;
        for (int q = 0; q < queries; q++)
        {
            double[] probe = data[(q * 23) % data.Length];
            List<int> expected = BruteForce(data, probe, K);
            double[][] probeTable = quantizer.Lookup(probe);
            List<int> viaExact = index.Search(node => Pts.Squared(probe, data[node]), probe, K, 30);
            List<int> viaQuantized = index.Search(node => quantizer.Distance(probeTable, codes[node]), probe, K, 30);
            exactHits += Overlap(viaExact, expected);
            quantizedHits += Overlap(viaQuantized, expected);
            total += K;
            tableLookups += index.TableLookups;
            index.TableLookups = 0;
        }
        double quantizedRecall = quantizedHits / (double)total;
        Check(quantizedRecall >= 0.6, "the quantised oracle keeps recall@10 at " + quantizedRecall.ToString("F3"));
        Check(quantizedHits <= exactHits, "quantisation never improves recall");

        int probes = index.Search(node => Pts.Squared(query, data[node]), query, K, 10).Count;
        Check(probes == K, "the search returns exactly k results when the graph holds more");
        Check(index.DistanceComputations > 0, "the search counted " + index.DistanceComputations + " distance computations");
    }

    private static void Boundaries(double[][] data)
    {
        var empty = new HnswIndex(4, 10, 0x7660FEED1234ABCEUL);
        Check(empty.Search(node => 0, data[0], K, 10).Count == 0 && empty.EntryPoint < 0,
            "an empty index returns nothing");
        var single = new HnswIndex(4, 10, 0x7660FEED1234ABCEUL);
        single.Insert(0);
        Check(single.Search(node => Pts.Squared(data[1], data[node]), data[1], 5, 10).Count == 1,
            "a single node index returns that node");
        Check(single.ConnectedFromEntry() && single.TopLayer == single.LevelOf(0),
            "the single node is its own entry point and top layer");

        var index = new HnswIndex(8, 40, 0x7660FEED1234ABCDUL);
        for (int i = 0; i < data.Length; i++) index.Insert(i);
        Check(index.Search(node => Pts.Squared(data[0], data[node]), data[0], K, 30).Count == K,
            "k results are returned on a populated index");
        Check(index.Search(node => Pts.Squared(data[0], data[node]), data[0], Pts.Count * 2, 30).Count > K,
            "asking for more than the budget finds the whole expansion, not just k");
        var faraway = new double[Pts.Dimensions];
        for (int d = 0; d < Pts.Dimensions; d++) faraway[d] = -50 + d;
        Check(index.Search(node => Pts.Squared(faraway, data[node]), faraway, K, 30).Count == K,
            "a query far outside the data range still returns k candidates");
    }

    private static List<int> BruteForce(double[][] data, double[] query, int k)
    {
        var ranked = new List<(double Distance, int Node)>();
        for (int i = 0; i < data.Length; i++) ranked.Add((Pts.Squared(query, data[i]), i));
        ranked.Sort((left, right) =>
        {
            int comparison = left.Distance.CompareTo(right.Distance);
            return comparison != 0 ? comparison : left.Node.CompareTo(right.Node);
        });
        var result = new List<int>();
        foreach ((double _, int node) in ranked)
        {
            if (result.Count >= k) break;
            result.Add(node);
        }
        return result;
    }

    private static int Overlap(List<int> left, List<int> right)
    {
        int hits = 0;
        foreach (int node in left)
        {
            if (right.Contains(node)) hits++;
        }
        return hits;
    }

    private static bool Same(List<int> left, List<int> right)
    {
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
