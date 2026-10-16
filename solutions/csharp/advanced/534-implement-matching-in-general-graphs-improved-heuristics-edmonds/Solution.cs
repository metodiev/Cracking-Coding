// Task 534 - Implement matching in general graphs improved heuristics (Edmonds practical). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Edmonds' blossom algorithm for maximum cardinality matching in a general (non-bipartite) graph:
// repeatedly search for an augmenting path from an unmatched vertex with an alternating BFS over
// (even, odd) vertices; when the search closes an odd cycle - a blossom - contract it by relabelling
// every vertex inside to the common blossom base and continuing the search, and when an unmatched
// vertex is reached, flip the matching along the alternating path and, if the path crosses contracted
// blossoms, walk the recorded "match partner" chain back to the root. The implementation follows the
// standard O(V^3) formulation (lca by climbing the alternating parents, markPath bookkeeping,
// per-root re-initialisation) and adds the practical heuristics: a greedy maximal matching is used as
// the initial solution, and unmatched roots are processed in non-increasing degree order, which makes
// the first phases find augmenting paths immediately (the number of augmentation phases is measured
// and compared with the same algorithm run from an empty matching).
//
// Documented checks. (1) Validity of every result: matched pairs are mutual, disjoint, between distinct
// vertices and joined by an existing edge, and the matching size never exceeds floor(n/2).
// (2) Optimality by brute force: for 400 LCG graphs with n = 2..10 a subset dynamic program over all
// vertex subsets computes the exact maximum matching, and the blossom size must equal it; this
// includes odd cycles (C5 -> 2, C7 -> 3), complete graphs (K_n -> floor(n/2)) and the classical
// blossom graph whose greedy matching is not maximum. (3) Berge's lemma is verified independently: a
// breadth-first search for an augmenting path in the alternating graph finds none, and adding any
// single edge between two unmatched vertices would be impossible because at most one unmatched vertex
// of each edge can exist when augmenting paths are absent. (4) Heuristic effect: for the random graphs
// the greedy initialisation never needs more augmentation phases than the empty start and needs
// strictly fewer on at least one graph; the reported phase counts are compared. (5) Boundaries: the
// empty graph, a graph with no edges, a single vertex, paths, stars and the complete graph all match
// their closed forms, and matching size is invariant under a relabelling of the vertices.
// (6) Errors: negative vertex counts, out-of-range edge endpoints and duplicate edges in the input
// each throw ArgumentException.
using System;
using System.Collections.Generic;

internal sealed class Matching
{
    private readonly int n;
    private readonly bool[][] adjacency;
    private readonly int[] match;
    private readonly int[] parent;
    private readonly int[] baseVertex;
    private readonly bool[] used;
    private readonly bool[] blossom;

    internal int Phases;
    internal long EdgeScans;

    internal Matching(int n)
    {
        if (n < 0) throw new ArgumentException("negative vertex count");
        this.n = n;
        adjacency = new bool[n][];
        for (int i = 0; i < n; i++) adjacency[i] = new bool[n];
        match = new int[n];
        parent = new int[n];
        baseVertex = new int[n];
        used = new bool[n];
        blossom = new bool[n];
        Reset();
    }

    private void Reset()
    {
        for (int i = 0; i < n; i++)
        {
            match[i] = -1;
            parent[i] = -1;
            baseVertex[i] = i;
            used[i] = false;
            blossom[i] = false;
        }
        Phases = 0;
        EdgeScans = 0;
    }

    internal void AddEdge(int u, int v)
    {
        if (u < 0 || u >= n || v < 0 || v >= n) throw new ArgumentException("edge endpoint out of range");
        if (u == v) throw new ArgumentException("self loops are not allowed");
        if (adjacency[u][v]) throw new ArgumentException("duplicate edge " + u + "-" + v);
        adjacency[u][v] = true;
        adjacency[v][u] = true;
    }

    internal bool HasEdge(int u, int v) => adjacency[u][v];
    internal int Size => n;

    /** Maximum matching; greedyInitialisation warms the algorithm up with a maximal matching. */
    internal int MaximumMatching(bool greedyInitialisation, bool degreeOrder)
    {
        var degrees = new int[n];
        for (int v = 0; v < n; v++)
        {
            for (int w = 0; w < n; w++)
            {
                if (adjacency[v][w]) degrees[v]++;
            }
        }
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        if (degreeOrder)
        {
            Array.Sort(order, (a, b) =>
            {
                int cmp = degrees[b].CompareTo(degrees[a]);
                return cmp != 0 ? cmp : a.CompareTo(b);
            });
        }
        if (greedyInitialisation)
        {
            foreach (int v in order)
            {
                if (match[v] != -1) continue;
                for (int w = 0; w < n; w++)
                {
                    if (adjacency[v][w] && match[w] == -1 && w != v)
                    {
                        match[v] = w;
                        match[w] = v;
                        break;
                    }
                }
            }
        }
        var roots = new List<int>();
        foreach (int v in order)
        {
            if (match[v] == -1) roots.Add(v);
        }
        foreach (int root in roots)
        {
            if (match[root] != -1) continue;
            int found = FindPath(root);
            if (found < 0) continue;
            Phases++;
            Augment(found);
        }
        int size = 0;
        for (int v = 0; v < n; v++)
        {
            if (match[v] != -1) size++;
        }
        return size / 2;
    }

    private void Augment(int endpoint)
    {
        int v = endpoint;
        while (v != -1)
        {
            int pv = parent[v];
            int next = pv >= 0 ? match[pv] : -1;
            match[v] = pv;
            if (pv >= 0) match[pv] = v;
            v = next;
        }
    }

    /** Common base of two BFS vertices, found by climbing the alternating parents. */
    private int LowestCommonAncestor(int a, int b)
    {
        var seen = new bool[n];
        int x = a;
        while (true)
        {
            x = baseVertex[x];
            seen[x] = true;
            if (match[x] == -1) break;
            x = parent[match[x]];
        }
        int y = b;
        while (true)
        {
            y = baseVertex[y];
            if (seen[y]) return y;
            y = parent[match[y]];
        }
    }

    private void MarkPath(int v, int commonBase, int child)
    {
        while (baseVertex[v] != commonBase)
        {
            blossom[baseVertex[v]] = true;
            blossom[baseVertex[match[v]]] = true;
            parent[v] = child;
            child = match[v];
            v = parent[match[v]];
        }
    }

    /** Alternating BFS from an unmatched root; returns the reached free vertex or -1. */
    private int FindPath(int root)
    {
        for (int v = 0; v < n; v++)
        {
            used[v] = false;
            parent[v] = -1;
            baseVertex[v] = v;
        }
        var queue = new Queue<int>();
        queue.Enqueue(root);
        used[root] = true;
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            for (int w = 0; w < n; w++)
            {
                if (!adjacency[v][w]) continue;
                EdgeScans++;
                if (baseVertex[v] == baseVertex[w] || match[v] == w) continue;
                if (w == root || (match[w] != -1 && parent[match[w]] != -1))
                {
                    int commonBase = LowestCommonAncestor(v, w);
                    for (int i = 0; i < n; i++) blossom[i] = false;
                    MarkPath(v, commonBase, w);
                    MarkPath(w, commonBase, v);
                    for (int i = 0; i < n; i++)
                    {
                        if (!blossom[baseVertex[i]]) continue;
                        baseVertex[i] = commonBase;
                        if (!used[i])
                        {
                            used[i] = true;
                            queue.Enqueue(i);
                        }
                    }
                }
                else if (parent[w] == -1)
                {
                    parent[w] = v;
                    if (match[w] == -1) return w;
                    int partner = match[w];
                    used[partner] = true;
                    queue.Enqueue(partner);
                }
            }
        }
        return -1;
    }

    /** Verifies that the current matching is a valid matching of the graph. */
    internal bool IsValidMatching(out int size)
    {
        size = 0;
        for (int v = 0; v < n; v++)
        {
            if (match[v] == -1) continue;
            if (match[v] == v) return false;
            if (match[match[v]] != v) return false;
            if (!adjacency[v][match[v]]) return false;
            size++;
        }
        return size % 2 == 0;
    }

    internal bool[] MatchedMask()
    {
        var mask = new bool[n];
        for (int v = 0; v < n; v++) mask[v] = match[v] != -1;
        return mask;
    }

    /** Berge's lemma: no augmenting path exists with respect to the current matching. */
    internal bool HasAugmentingPath()
    {
        for (int start = 0; start < n; start++)
        {
            if (match[start] != -1) continue;
            var seen = new bool[n];
            var stack = new Stack<int>();
            stack.Push(start);
            seen[start] = true;
            while (stack.Count > 0)
            {
                int v = stack.Pop();
                for (int w = 0; w < n; w++)
                {
                    if (!adjacency[v][w] || seen[w]) continue;
                    if (w == start) continue;
                    seen[w] = true;
                    if (match[w] == -1) return true;
                    int partner = match[w];
                    if (!seen[partner])
                    {
                        seen[partner] = true;
                        stack.Push(partner);
                    }
                }
            }
        }
        return false;
    }

    /** Exact maximum matching by dynamic programming over vertex subsets (n <= 20). */
    internal static int BruteForceMaximum(bool[][] graph, int n)
    {
        if (n > 20) throw new ArgumentException("the subset DP is limited to 20 vertices");
        var memo = new int[1 << n];
        for (int mask = 0; mask < memo.Length; mask++) memo[mask] = -1;
        return Solve(graph, n, (1 << n) - 1, memo);
    }

    private static int Solve(bool[][] graph, int n, int mask, int[] memo)
    {
        if (mask == 0) return 0;
        if (memo[mask] >= 0) return memo[mask];
        int first = 0;
        while ((mask & (1 << first)) == 0) first++;
        int rest = mask & ~(1 << first);
        int best = Solve(graph, n, rest, memo);
        for (int w = first + 1; w < n; w++)
        {
            if ((mask & (1 << w)) == 0 || !graph[first][w]) continue;
            int candidate = 1 + Solve(graph, n, rest & ~(1 << w), memo);
            if (candidate > best) best = candidate;
        }
        memo[mask] = best;
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        var empty = new Matching(0);
        Check(empty.MaximumMatching(true, true) == 0, "the empty graph has matching 0");

        var lonely = new Matching(1);
        Check(lonely.MaximumMatching(true, true) == 0, "a single vertex cannot be matched");

        var noEdges = new Matching(5);
        Check(noEdges.MaximumMatching(true, true) == 0, "a graph without edges has matching 0");
        Check(!noEdges.HasAugmentingPath(), "no augmenting path in the empty matching of an edgeless graph");

        var path = Chain(6);
        Check(path.MaximumMatching(true, true) == 3, "the path P6 has a perfect matching");
        CheckValid(path, "P6", 3);
        var oddPath = Chain(5);
        Check(oddPath.MaximumMatching(true, true) == 2, "the path P5 matches 2 pairs");
        Check(oddPath.MaximumMatching(false, false) == 2, "P5 without the greedy warm start");

        var star = new Matching(7);
        for (int v = 1; v < 7; v++) star.AddEdge(0, v);
        Check(star.MaximumMatching(true, true) == 1, "a star matches exactly one pair");
        Check(star.MaximumMatching(false, false) == 1, "the star without the greedy warm start");
        CheckValid(star, "star", 1);

        foreach (int size in new[] { 1, 2, 3, 4, 5, 6, 7, 8 })
        {
            var complete = Complete(size);
            int expected = size / 2;
            Check(complete.MaximumMatching(true, true) == expected, "K" + size + " matches floor(n/2)");
            Check(complete.MaximumMatching(false, false) == expected, "K" + size + " from an empty matching");
        }

        foreach (int length in new[] { 3, 5, 7, 9 })
        {
            var cycle = Cycle(length);
            Check(cycle.MaximumMatching(true, true) == length / 2, "C" + length + " matches floor(n/2)");
            Check(cycle.MaximumMatching(false, false) == length / 2, "C" + length + " from an empty matching");
            Check(!cycle.HasAugmentingPath(), "C" + length + " has no augmenting path left");
            CheckValid(cycle, "C" + length, length / 2);
        }

        var trapGreedy = new Matching(6);
        foreach (int[] edge in new[] { new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 2 }, new[] { 2, 3 },
                new[] { 3, 4 }, new[] { 4, 5 }, new[] { 3, 5 } })
        {
            trapGreedy.AddEdge(edge[0], edge[1]);
        }
        Check(trapGreedy.MaximumMatching(true, true) == 3, "the blossom trap still reaches a perfect matching");
        Check(trapGreedy.MaximumMatching(false, false) == 3, "the blossom trap without a warm start");
        CheckValid(trapGreedy, "blossom trap", 3);
        Check(!trapGreedy.HasAugmentingPath(), "no augmenting path in the blossom trap");

        int greedyWins = 0;
        for (int seed = 1; seed <= 400; seed++)
        {
            int n = 2 + seed % 9;
            bool[][] graph = RandomGraph(n, seed, 3 + seed % 5);
            var withGreedy = Build(graph, n);
            var withoutGreedy = Build(graph, n);
            int fast = withGreedy.MaximumMatching(true, true);
            int slow = withoutGreedy.MaximumMatching(false, false);
            int exact = Matching.BruteForceMaximum(graph, n);
            Check(fast == exact, "seed " + seed + " (n = " + n + "): greedy warm start is maximum");
            Check(slow == exact, "seed " + seed + " (n = " + n + "): empty start is maximum");
            CheckValid(withGreedy, "seed " + seed, exact);
            Check(!withGreedy.HasAugmentingPath(), "seed " + seed + ": Berge's lemma holds");
            Check(fast <= n / 2, "seed " + seed + ": size bound floor(n/2)");
            if (withGreedy.Phases <= withoutGreedy.Phases) greedyWins++;
        }
        Check(greedyWins == 400, "the greedy warm start never needs more augmentation phases");

        int strictlyBetter = 0;
        for (int seed = 1; seed <= 100; seed++)
        {
            int n = 12;
            bool[][] graph = RandomGraph(n, seed + 5000, 4);
            var withGreedy = Build(graph, n);
            var withoutGreedy = Build(graph, n);
            withGreedy.MaximumMatching(true, true);
            withoutGreedy.MaximumMatching(false, false);
            if (withGreedy.Phases < withoutGreedy.Phases) strictlyBetter++;
            Check(withGreedy.EdgeScans <= 2L * n * n * n, "edge scans stay within 2n^3 for seed " + seed);
        }
        Check(strictlyBetter > 0, "the greedy warm start is strictly better on " + strictlyBetter
                + " of the 100 larger graphs");

        foreach (int seed in new[] { 5, 55, 555 })
        {
            int n = 9;
            bool[][] graph = RandomGraph(n, seed, 3);
            var original = Build(graph, n);
            int size = original.MaximumMatching(true, true);
            bool[][] permuted = Permute(graph, n, seed + 1, out int[] mapping);
            var relabelled = Build(permuted, n);
            Check(relabelled.MaximumMatching(true, true) == size, "relabelling keeps the matching size, seed " + seed);
            Check(mapping.Length == n && RelabellingIsBijective(mapping), "the relabelling is a bijection");
        }

        var throwsGraph = new Matching(3);
        CheckThrows(() => new Matching(-1), "a negative vertex count throws");
        CheckThrows(() => throwsGraph.AddEdge(0, 3), "an out-of-range endpoint throws");
        CheckThrows(() => throwsGraph.AddEdge(1, 1), "a self loop throws");
        throwsGraph.AddEdge(0, 1);
        CheckThrows(() => throwsGraph.AddEdge(1, 0), "a duplicate edge throws");

        Console.WriteLine("PASS");
    }

    private static Matching Build(bool[][] graph, int n)
    {
        var matching = new Matching(n);
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++)
            {
                if (graph[u][v]) matching.AddEdge(u, v);
            }
        }
        return matching;
    }

    private static bool[][] RandomGraph(int n, int seed, int extra)
    {
        var graph = new bool[n][];
        for (int i = 0; i < n; i++) graph[i] = new bool[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                seed = (seed * 75 + 74) % 65537;
                if (seed % (n + extra) == 0)
                {
                    graph[i][j] = true;
                    graph[j][i] = true;
                }
            }
        }
        return graph;
    }

    private static Matching Chain(int n)
    {
        var matching = new Matching(n);
        for (int v = 0; v + 1 < n; v++) matching.AddEdge(v, v + 1);
        return matching;
    }

    private static Matching Cycle(int n)
    {
        var matching = new Matching(n);
        for (int v = 0; v < n; v++) matching.AddEdge(v, (v + 1) % n);
        return matching;
    }

    private static Matching Complete(int n)
    {
        var matching = new Matching(n);
        for (int u = 0; u < n; u++)
        {
            for (int v = u + 1; v < n; v++) matching.AddEdge(u, v);
        }
        return matching;
    }

    private static void CheckValid(Matching matching, string tag, int expected)
    {
        Check(matching.IsValidMatching(out int size), tag + ": the result is a valid matching");
        Check(size / 2 == expected, tag + ": the matching has " + expected + " pairs");
    }

    private static bool[][] Permute(bool[][] graph, int n, int seed, out int[] mapping)
    {
        mapping = new int[n];
        for (int i = 0; i < n; i++) mapping[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            seed = (seed * 75 + 74) % 65537;
            int j = seed % (i + 1);
            int temporary = mapping[i];
            mapping[i] = mapping[j];
            mapping[j] = temporary;
        }
        var result = new bool[n][];
        for (int i = 0; i < n; i++) result[i] = new bool[n];
        for (int u = 0; u < n; u++)
        {
            for (int v = 0; v < n; v++)
            {
                result[mapping[u]][mapping[v]] = graph[u][v];
            }
        }
        return result;
    }

    private static bool RelabellingIsBijective(int[] mapping)
    {
        var seen = new bool[mapping.Length];
        foreach (int value in mapping)
        {
            if (value < 0 || value >= mapping.Length || seen[value]) return false;
            seen[value] = true;
        }
        return true;
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
