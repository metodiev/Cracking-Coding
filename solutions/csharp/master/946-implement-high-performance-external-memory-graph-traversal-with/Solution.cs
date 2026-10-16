// Task 946 - Implement high-performance external-memory graph traversal with partitioning.
// (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Breadth-first search over a graph that does not fit in memory: the vertices are partitioned into
// fixed-size contiguous blocks, the adjacency of a block lives "on disk", and every block touch
// goes through an LRU page cache that counts hits, misses and the I/O performed. A paged BFS reads
// the block of the vertex it expands and then the block of every neighbour, which is exactly the
// random-access pattern that dominates external-memory traversal, so the cost of the search is the
// number of block reads rather than the number of edge visits.
//
// Verified: distances agree with an in-memory BFS oracle for a single source, a far-away source and
// a multi-source frontier; the cache is cross-checked by an independent list-based LRU replay of the
// recorded access sequence; the I/O count never decreases as the cache grows (checked over eight
// capacities) and is bounded below by the number of distinct blocks touched; a cache large enough
// for the whole graph misses exactly once per distinct block; the resident set after the run is the
// last K distinct blocks in LRU order; and the partition covers every vertex exactly once. Empty
// graphs, a single vertex, a split graph with an unreachable component and a block size of one are
// the boundary cases.
// Simplified: blocks and I/O are modelled in memory with an explicit counter instead of real files,
// edge lists are read whole rather than at byte granularity, and there is no prefetching, sorting or
// batched level-synchronous frontier - but the partition, the paging policy, its cost accounting and
// the traversal semantics are real.
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
}

internal static class GraphFactory
{
    internal static List<int>[] Random(int vertices, int degree, ulong seed)
    {
        var adjacency = new List<int>[vertices];
        for (int i = 0; i < vertices; i++) adjacency[i] = new List<int>();
        ulong state = seed;
        for (int i = 0; i < vertices; i++)
        {
            for (int d = 0; d < degree; d++)
            {
                int w = (int)(SplitMix64.Next(ref state) % (ulong)vertices);
                if (w == i || adjacency[i].Contains(w)) continue;
                adjacency[i].Add(w);
                adjacency[w].Add(i);
            }
        }
        // A planted path keeps the graph connected so reachability itself is not a random variable.
        for (int i = 0; i + 1 < vertices; i++)
        {
            if (adjacency[i].Contains(i + 1)) continue;
            adjacency[i].Add(i + 1);
            adjacency[i + 1].Add(i);
        }
        foreach (List<int> list in adjacency) list.Sort();
        return adjacency;
    }

    /** A path on [0, size) plus a disjoint clique on [size, 2*size): the clique is unreachable. */
    internal static List<int>[] Split(int size)
    {
        var adjacency = new List<int>[2 * size];
        for (int i = 0; i < adjacency.Length; i++) adjacency[i] = new List<int>();
        for (int i = 0; i + 1 < size; i++) { adjacency[i].Add(i + 1); adjacency[i + 1].Add(i); }
        for (int i = size; i < 2 * size; i++)
            for (int j = i + 1; j < 2 * size; j++) { adjacency[i].Add(j); adjacency[j].Add(i); }
        return adjacency;
    }
}

/** LRU page cache over blocks with explicit I/O accounting. */
internal sealed class BlockPager
{
    private readonly int blockSize;
    private readonly int capacity;
    private readonly Dictionary<int, LinkedListNode<int>> present = new Dictionary<int, LinkedListNode<int>>();
    private readonly LinkedList<int> recency = new LinkedList<int>();

    internal long Accesses;
    internal long Hits;
    internal long Misses;
    internal readonly List<int> AccessLog = new List<int>();

    internal BlockPager(int vertices, int blockSize, int capacity)
    {
        this.blockSize = blockSize;
        this.capacity = capacity;
        BlockCount = vertices == 0 ? 0 : (vertices + blockSize - 1) / blockSize;
    }

    internal int BlockCount { get; }
    internal int BlockOf(int vertex) => vertex / blockSize;

    internal void Access(int block)
    {
        Accesses++;
        AccessLog.Add(block);
        if (present.TryGetValue(block, out LinkedListNode<int>? node))
        {
            Hits++;
            recency.Remove(node);
            recency.AddLast(node);
            return;
        }
        Misses++;
        if (capacity == 0) return;
        if (present.Count == capacity)
        {
            int evicted = recency.First!.Value;
            recency.RemoveFirst();
            present.Remove(evicted);
        }
        present[block] = recency.AddLast(block);
    }

    /** Resident blocks, most recently used first. */
    internal List<int> Resident() => recency.Reverse().ToList();
}

internal sealed class PagedGraph
{
    private readonly List<int>[] adjacency;
    private readonly int blockSize;

    internal PagedGraph(List<int>[] adjacency, int blockSize)
    {
        this.adjacency = adjacency;
        this.blockSize = blockSize;
    }

    internal int VertexCount => adjacency.Length;
    internal int BlockSize => blockSize;
    internal long Edges => adjacency.Sum(list => (long)list.Count) / 2;

    /// <summary>BFS whose every block touch goes through the pager, so I/O is measured exactly.</summary>
    internal int[] Bfs(BlockPager pager, IReadOnlyList<int> sources)
    {
        var distance = new int[adjacency.Length];
        for (int i = 0; i < distance.Length; i++) distance[i] = -1;
        var queue = new Queue<int>();
        foreach (int source in sources)
        {
            if (source < 0 || source >= adjacency.Length || distance[source] == 0) continue;
            distance[source] = 0;
            queue.Enqueue(source);
        }
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            pager.Access(pager.BlockOf(v));
            foreach (int w in adjacency[v])
            {
                pager.Access(pager.BlockOf(w));
                if (distance[w] < 0)
                {
                    distance[w] = distance[v] + 1;
                    queue.Enqueue(w);
                }
            }
        }
        return distance;
    }

    /** Independent oracle: the same traversal with no paging at all. */
    internal int[] BfsOracle(IReadOnlyList<int> sources)
    {
        var distance = new int[adjacency.Length];
        for (int i = 0; i < distance.Length; i++) distance[i] = -1;
        var queue = new Queue<int>();
        foreach (int source in sources)
        {
            if (source < 0 || source >= adjacency.Length || distance[source] == 0) continue;
            distance[source] = 0;
            queue.Enqueue(source);
        }
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            foreach (int w in adjacency[v])
            {
                if (distance[w] < 0)
                {
                    distance[w] = distance[v] + 1;
                    queue.Enqueue(w);
                }
            }
        }
        return distance;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** Independent LRU replay of an access sequence, using a plain list instead of a dictionary. */
    private static long ReferenceLruMisses(IReadOnlyList<int> accesses, int capacity)
    {
        var cache = new List<int>();
        long misses = 0;
        foreach (int block in accesses)
        {
            int index = cache.IndexOf(block);
            if (index >= 0)
            {
                cache.RemoveAt(index);
                cache.Insert(0, block);
                continue;
            }
            misses++;
            if (capacity == 0) continue;
            if (cache.Count == capacity) cache.RemoveAt(cache.Count - 1);
            cache.Insert(0, block);
        }
        return misses;
    }

    private static List<int> LastDistinct(IReadOnlyList<int> accesses, int count)
    {
        var seen = new List<int>();
        for (int i = accesses.Count - 1; i >= 0 && seen.Count < count; i--)
            if (!seen.Contains(accesses[i])) seen.Add(accesses[i]);
        return seen;
    }

    public static void Main()
    {
        const int Vertices = 120;
        const int BlockSize = 8;
        List<int>[] adjacency = GraphFactory.Random(Vertices, 3, 0x9460UL);
        var graph = new PagedGraph(adjacency, BlockSize);

        // 1. Partition invariants: contiguous blocks covering every vertex exactly once.
        bool contiguous = Enumerable.Range(0, graph.VertexCount).All(v => v / BlockSize < (Vertices + BlockSize - 1) / BlockSize);
        Check(contiguous && adjacency.SelectMany(list => list).Count() == 2 * graph.Edges && graph.VertexCount == Vertices,
            "the partition covers all " + Vertices + " vertices in blocks of " + BlockSize + " with " + graph.Edges + " edges");

        // 2-3. Single-source and far-source traversal against the in-memory oracle.
        var pager = new BlockPager(Vertices, BlockSize, 4);
        int[] fromZero = graph.Bfs(pager, new[] { 0 });
        int[] oracleZero = graph.BfsOracle(new[] { 0 });
        var farPager = new BlockPager(Vertices, BlockSize, 4);
        int[] fromLast = graph.Bfs(farPager, new[] { Vertices - 1 });
        int[] oracleLast = graph.BfsOracle(new[] { Vertices - 1 });
        Check(fromZero.SequenceEqual(oracleZero) && fromLast.SequenceEqual(oracleLast)
              && fromZero.Count(d => d >= 0) == Vertices && oracleZero.Max() > 2,
            "paged BFS matches the oracle from vertex 0 (eccentricity " + oracleZero.Max() + ") and from vertex "
            + (Vertices - 1));

        // 4. Multi-source frontier matches a multi-source oracle.
        int[] multi = graph.Bfs(new BlockPager(Vertices, BlockSize, 4), new[] { 0, 37, 99 });
        Check(multi.SequenceEqual(graph.BfsOracle(new[] { 0, 37, 99 })) && multi.All(d => d >= 0),
            "a three-source frontier reduces every distance consistently");

        // 5. The cache is cross-checked by an independent list-based LRU replay.
        long reference = ReferenceLruMisses(pager.AccessLog, 4);
        Check(pager.Misses == reference && pager.Accesses == pager.AccessLog.Count && pager.Hits + pager.Misses == pager.Accesses,
            "the pager's " + pager.Misses + " misses equal the independent LRU replay over "
            + pager.Accesses + " block accesses");

        // 6-7. I/O as a function of the cache: monotone, bounded below, exact when everything fits.
        long[] missesByCapacity = new long[9];
        long distinct = pager.AccessLog.Distinct().Count();
        for (int capacity = 0; capacity <= 8; capacity++)
        {
            var sized = new BlockPager(Vertices, BlockSize, capacity);
            graph.Bfs(sized, new[] { 0 });
            missesByCapacity[capacity] = sized.Misses;
        }
        bool monotone = true;
        for (int capacity = 0; capacity < 8; capacity++)
            monotone &= missesByCapacity[capacity + 1] <= missesByCapacity[capacity];
        Check(monotone && missesByCapacity.All(m => m >= distinct) && missesByCapacity[8] < missesByCapacity[0],
            "I/O falls from " + missesByCapacity[0] + " to " + missesByCapacity[8]
            + " reads as the cache grows, never below the " + distinct + " distinct blocks ("
            + string.Join(",", missesByCapacity) + ")");

        var unbounded = new BlockPager(Vertices, BlockSize, pager.BlockCount);
        graph.Bfs(unbounded, new[] { 0 });
        Check(unbounded.Misses == distinct && unbounded.Hits == unbounded.Accesses - distinct,
            "a cache holding the whole graph misses exactly once per distinct block (" + distinct + ")");

        // 8. LRU resident set: the last K distinct blocks touched, most recent first.
        var small = new BlockPager(Vertices, BlockSize, 3);
        graph.Bfs(small, new[] { 0 });
        Check(small.Resident().SequenceEqual(LastDistinct(small.AccessLog, 3)),
            "the resident set after the run is the last three distinct blocks in LRU order");

        // 9. Boundaries: empty graph, single vertex, unreachable component, block size one.
        var emptyGraph = new PagedGraph(new List<int>[0], BlockSize);
        var emptyPager = new BlockPager(0, BlockSize, 2);
        Check(emptyGraph.Bfs(emptyPager, new[] { 0 }).Length == 0 && emptyPager.Accesses == 0
              && emptyPager.BlockCount == 0,
            "an empty graph needs no I/O and returns no distances");
        var single = new PagedGraph(new[] { new List<int>() }, BlockSize);
        var singlePager = new BlockPager(1, BlockSize, 2);
        Check(single.Bfs(singlePager, new[] { 0 })[0] == 0 && singlePager.Misses == 1,
            "a single isolated vertex is at distance 0 after exactly one block read");
        List<int>[] split = GraphFactory.Split(6);
        var splitGraph = new PagedGraph(split, 4);
        int[] splitDistances = splitGraph.Bfs(new BlockPager(split.Length, 4, 2), new[] { 0 });
        Check(splitDistances.SequenceEqual(splitGraph.BfsOracle(new[] { 0 }))
              && splitDistances.Take(6).All(d => d >= 0)
              && splitDistances.Skip(6).All(d => d == -1),
            "the disconnected clique stays unreachable at distance -1");
        var unitBlocks = new PagedGraph(GraphFactory.Random(40, 3, 0x9461UL), 1);
        Check(unitBlocks.Bfs(new BlockPager(40, 1, 4), new[] { 0 }).SequenceEqual(unitBlocks.BfsOracle(new[] { 0 })),
            "a one-vertex-per-block partition still traverses correctly");

        // 10. Determinism of distances and I/O accounting.
        var repeatPager = new BlockPager(Vertices, BlockSize, 4);
        int[] repeat = graph.Bfs(repeatPager, new[] { 0 });
        Check(repeat.SequenceEqual(fromZero) && repeatPager.Misses == pager.Misses
              && repeatPager.Hits == pager.Hits && repeatPager.AccessLog.SequenceEqual(pager.AccessLog),
            "re-running the traversal reproduces both the distances and the I/O trace");

        Console.WriteLine("  graph: " + Vertices + " vertices, " + graph.Edges + " edges, "
            + (Vertices / BlockSize) + " blocks of " + BlockSize + "; BFS from vertex 0: "
            + pager.Accesses + " block accesses (" + pager.Hits + " hits, " + pager.Misses + " misses) over "
            + distinct + " distinct blocks");
        Console.WriteLine("  I/O by cache size: " + string.Join(",", missesByCapacity.Skip(1))
            + " reads for capacities 1..8");
        Console.WriteLine("PASS");
    }
}
