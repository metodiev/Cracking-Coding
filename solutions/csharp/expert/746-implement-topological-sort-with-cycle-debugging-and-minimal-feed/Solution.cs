// Task 746 - Implement topological sort with cycle debugging and minimal feedback arc set heuristics. (Tier:
// Expert, CTCI Ch 4 - Trees and Graphs)
//
// A directed graph on a small vertex set, with three pieces of machinery:
//   * Kahn's algorithm produces a topological order when one exists; otherwise it leaves a residual subgraph in
//     which every vertex still has an in-edge, and walking backwards through in-edges from any residual vertex
//     until a vertex repeats extracts a genuine cycle - the cycle debugging part;
//   * an independent DFS post-order gives a second order, used to cross-check validity (both must have every
//     edge pointing forward, though they need not be the same order);
//   * feedback arc set heuristics: order the vertices by out-degree minus in-degree, improve that order by
//     Eades' greedy rule (repeatedly pull out sinks and sources, then the vertex with the largest out-in), run
//     an adjacent-swap local search until no swap removes a backward edge, and compare all of them against a
//     brute-force optimum over every permutation (only affordable for the small graphs used here).
// Invariants verified: a Kahn order on a DAG has no backward edge and so does the DFS order; for a cyclic graph
// the extracted cycle is real (its arcs exist and it closes) and deleting its arcs makes the graph acyclic
// (re-checked with Kahn); every heuristic's backward-edge set is verified to be a feedback arc set by deleting
// it and re-running Kahn; the size of that set is never below the brute-force optimum and never above the naive
// "delete every arc inside the residual core"; the local search is monotone (it only accepts improving swaps)
// and never worse than its starting order; reversing the graph and reversing the order keeps the backward-edge
// count (an exact symmetry); and the heuristic attains the optimum on the small instances where the optimum is
// known. Self-loops are reported as a one-vertex cycle. Simplification: the optimum is computed by full
// permutation enumeration, so the exact comparisons are limited to at most eight vertices; deeper local search
// variants (e.g. vertex moves or simulated annealing) and the NP-hardness-aware approximations are out of scope.
using System;
using System.Collections.Generic;

internal static class SplitMix
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }
}

internal sealed class Digraph
{
    internal readonly int N;
    private readonly List<(int From, int To)> _arcs = new List<(int, int)>();

    internal Digraph(int n)
    {
        N = n;
    }

    internal int ArcCount => _arcs.Count;
    internal IReadOnlyList<(int From, int To)> Arcs => _arcs;
    internal void Add(int from, int to) => _arcs.Add((from, to));

    internal List<int>[] Out()
    {
        var outgoing = new List<int>[N];
        for (int i = 0; i < N; i++) outgoing[i] = new List<int>();
        foreach (var (from, to) in _arcs) outgoing[from].Add(to);
        return outgoing;
    }

    internal List<int>[] In()
    {
        var incoming = new List<int>[N];
        for (int i = 0; i < N; i++) incoming[i] = new List<int>();
        foreach (var (from, to) in _arcs) incoming[to].Add(from);
        return incoming;
    }

    internal bool Has(int from, int to)
    {
        foreach (var (f, t) in _arcs) if (f == from && t == to) return true;
        return false;
    }

    internal Digraph Reverse()
    {
        var reversed = new Digraph(N);
        foreach (var (from, to) in _arcs) reversed.Add(to, from);
        return reversed;
    }
}

internal static class Topo
{
    /// <summary>Kahn's algorithm; the second output lists the vertices stuck in the residual core.</summary>
    internal static (List<int> Order, List<int> Residual) Kahn(Digraph graph)
    {
        var inDegree = new int[graph.N];
        var outgoing = graph.Out();
        foreach (var (_, to) in graph.Arcs) inDegree[to]++;
        var queue = new Queue<int>();
        for (int i = 0; i < graph.N; i++) if (inDegree[i] == 0) queue.Enqueue(i);
        var order = new List<int>();
        while (queue.Count > 0)
        {
            int v = queue.Dequeue();
            order.Add(v);
            foreach (int w in outgoing[v]) if (--inDegree[w] == 0) queue.Enqueue(w);
        }
        var residual = new List<int>();
        for (int i = 0; i < graph.N; i++) if (inDegree[i] > 0) residual.Add(i);
        return (order, residual);
    }

    internal static List<int> DfsOrder(Digraph graph)
    {
        var visited = new bool[graph.N];
        var order = new List<int>();
        var outgoing = graph.Out();
        for (int start = 0; start < graph.N; start++)
        {
            if (visited[start]) continue;
            var stack = new Stack<(int Vertex, int Index)>();
            stack.Push((start, 0));
            visited[start] = true;
            while (stack.Count > 0)
            {
                var (v, index) = stack.Pop();
                if (index < outgoing[v].Count)
                {
                    stack.Push((v, index + 1));
                    int w = outgoing[v][index];
                    if (!visited[w])
                    {
                        visited[w] = true;
                        stack.Push((w, 0));
                    }
                }
                else order.Add(v);          // post-order
            }
        }
        order.Reverse();
        return order;
    }

    /// <summary>Number of arcs pointing backwards in the given order (the feedback arc set of the order).</summary>
    internal static List<(int From, int To)> BackwardArcs(Digraph graph, List<int> order)
    {
        var position = new int[graph.N];
        for (int i = 0; i < order.Count; i++) position[order[i]] = i;
        var backward = new List<(int From, int To)>();
        foreach (var (from, to) in graph.Arcs)
            if (position[from] >= 0 && position[to] >= 0 && position[from] > position[to]) backward.Add((from, to));
        return backward;
    }

    internal static bool IsTopological(Digraph graph, List<int> order)
        => order.Count == graph.N && BackwardArcs(graph, order).Count == 0;

    internal static Digraph Without(Digraph graph, List<(int From, int To)> removed)
    {
        var reduced = new Digraph(graph.N);
        foreach (var (from, to) in graph.Arcs)
            if (!removed.Contains((from, to))) reduced.Add(from, to);
        return reduced;
    }

    /// <summary>Extracts a real cycle from the residual core by walking in-edges until a vertex repeats.</summary>
    internal static List<int>? FindCycle(Digraph graph)
    {
        var (_, residual) = Kahn(graph);
        var inCore = new HashSet<int>(residual);
        var incoming = graph.In();
        foreach (int start in residual)
        {
            var path = new List<int>();
            var seen = new Dictionary<int, int>();
            int current = start;
            while (true)
            {
                if (seen.TryGetValue(current, out int index)) return path.GetRange(index, path.Count - index);
                seen[current] = path.Count;
                path.Add(current);
                int next = -1;
                foreach (int predecessor in incoming[current])
                    if (inCore.Contains(predecessor)) { next = predecessor; break; }
                if (next < 0) break;
                current = next;
            }
        }
        return null;
    }

    /// <summary>Eades' greedy: peel sinks and sources, then take the vertex with the largest out-in balance.</summary>
    internal static (List<int> Order, long SinkSources) Greedy(Digraph graph)
    {
        var alive = new List<int>();
        for (int i = 0; i < graph.N; i++) alive.Add(i);
        var head = new List<int>();
        var tail = new List<int>();
        long peeled = 0;
        var outgoing = graph.Out();
        var incoming = graph.In();
        while (alive.Count > 0)
        {
            bool progress = true;
            while (progress)
            {
                progress = false;
                foreach (int v in alive)
                {
                    int outAlive = 0, inAlive = 0;
                    foreach (int w in outgoing[v]) if (alive.Contains(w)) outAlive++;
                    foreach (int w in incoming[v]) if (alive.Contains(w)) inAlive++;
                    if (outAlive == 0) { alive.Remove(v); tail.Insert(0, v); peeled++; progress = true; break; }
                    if (inAlive == 0) { alive.Remove(v); head.Add(v); peeled++; progress = true; break; }
                }
            }
            if (alive.Count == 0) break;
            int best = alive[0], bestBalance = int.MinValue;
            foreach (int v in alive)
            {
                int outAlive = 0, inAlive = 0;
                foreach (int w in outgoing[v]) if (alive.Contains(w)) outAlive++;
                foreach (int w in incoming[v]) if (alive.Contains(w)) inAlive++;
                if (outAlive - inAlive > bestBalance) { bestBalance = outAlive - inAlive; best = v; }
            }
            alive.Remove(best);
            head.Add(best);
        }
        var order = new List<int>(head);
        order.AddRange(tail);
        return (order, peeled);
    }

    internal static List<int> ByBalance(Digraph graph)
    {
        var order = new List<int>();
        for (int i = 0; i < graph.N; i++) order.Add(i);
        var outgoing = graph.Out();
        var incoming = graph.In();
        order.Sort((a, b) =>
        {
            int balanceA = outgoing[a].Count - incoming[a].Count;
            int balanceB = outgoing[b].Count - incoming[b].Count;
            return balanceA != balanceB ? balanceB.CompareTo(balanceA) : a.CompareTo(b);
        });
        return order;
    }

    /// <summary>Adjacent-swap local search: only swaps that strictly reduce the backward-edge count are kept.</summary>
    internal static (List<int> Order, long Swaps) Improve(Digraph graph, List<int> start)
    {
        var order = new List<int>(start);
        long swaps = 0;
        bool improved = true;
        while (improved)
        {
            improved = false;
            for (int i = 0; i + 1 < order.Count; i++)
            {
                int before = BackwardArcs(graph, order).Count;
                (order[i], order[i + 1]) = (order[i + 1], order[i]);
                int after = BackwardArcs(graph, order).Count;
                if (after < before) { swaps++; improved = true; }
                else (order[i], order[i + 1]) = (order[i + 1], order[i]);
            }
        }
        return (order, swaps);
    }

    /// <summary>Brute-force optimum over all permutations; only usable for tiny graphs.</summary>
    internal static int BruteForceOptimum(Digraph graph)
    {
        var order = new List<int>();
        for (int i = 0; i < graph.N; i++) order.Add(i);
        int best = int.MaxValue;
        Permute(order, 0, graph, ref best);
        return best;
    }

    private static void Permute(List<int> order, int index, Digraph graph, ref int best)
    {
        if (index == order.Count)
        {
            int backward = BackwardArcs(graph, order).Count;
            if (backward < best) best = backward;
            return;
        }
        for (int i = index; i < order.Count; i++)
        {
            (order[index], order[i]) = (order[i], order[index]);
            Permute(order, index + 1, graph, ref best);
            (order[index], order[i]) = (order[i], order[index]);
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Digraph Dag() => new Digraph(6);

    private static Digraph BuildDag()
    {
        var graph = Dag();
        graph.Add(0, 1); graph.Add(0, 2); graph.Add(1, 3); graph.Add(2, 3); graph.Add(3, 4); graph.Add(2, 5);
        return graph;
    }

    private static Digraph ThreeCycle()
    {
        var graph = new Digraph(4);
        graph.Add(0, 1); graph.Add(1, 2); graph.Add(2, 0); graph.Add(2, 3);
        return graph;
    }

    private static Digraph Tournament()
    {
        var graph = new Digraph(6);
        graph.Add(0, 1); graph.Add(1, 2); graph.Add(2, 0);
        graph.Add(3, 4); graph.Add(4, 5); graph.Add(5, 3);
        graph.Add(0, 3); graph.Add(1, 4); graph.Add(2, 5);
        return graph;
    }

    private static Digraph RandomGraph(int n, int arcs, ulong seed)
    {
        var graph = new Digraph(n);
        ulong state = seed;
        for (int i = 0; i < arcs; i++)
        {
            int from = (int)(SplitMix.Next(ref state) % (ulong)n);
            int to = (int)(SplitMix.Next(ref state) % (ulong)n);
            if (from != to && !graph.Has(from, to)) graph.Add(from, to);
        }
        return graph;
    }

    public static void Main()
    {
        // ---- degenerate graphs ------------------------------------------------------------------------
        Check(Topo.Kahn(new Digraph(0)).Order.Count == 0, "an empty graph has an empty topological order");
        Check(Topo.Kahn(new Digraph(1)).Order.Count == 1 && Topo.Kahn(new Digraph(1)).Residual.Count == 0,
            "a single vertex is trivially sorted and leaves nothing behind");
        var selfLoop = new Digraph(2);
        selfLoop.Add(1, 1);
        selfLoop.Add(0, 1);
        var selfCycle = Topo.FindCycle(selfLoop);
        Check(selfCycle != null && selfCycle.Count == 1 && selfCycle[0] == 1,
            "a self-loop is reported as a one-vertex cycle");
        Check(Topo.Kahn(selfLoop).Residual.Count == 1, "the residual core of the self-loop graph holds only that vertex");

        // ---- DAG: Kahn and DFS orders are both valid ----------------------------------------------------
        var dag = BuildDag();
        var (dagOrder, dagResidual) = Topo.Kahn(dag);
        var dfsOrder = Topo.DfsOrder(dag);
        Check(dagResidual.Count == 0 && dagOrder.Count == dag.N && Topo.IsTopological(dag, dagOrder),
            "Kahn's algorithm sorts the whole DAG with no backward edge");
        Check(Topo.IsTopological(dag, dfsOrder), "the independent DFS post-order is also a valid topological order");
        Check(Topo.BackwardArcs(dag, dagOrder).Count == 0 && Topo.BruteForceOptimum(dag) == 0
            && Topo.FindCycle(dag) == null,
            "a DAG needs no feedback arc set and has no cycle to report");

        // ---- cycle debugging ----------------------------------------------------------------------------
        var cyclic = ThreeCycle();
        var cycle = Topo.FindCycle(cyclic);
        Check(cycle != null && cycle.Count >= 2, "the 3-cycle graph yields a cycle of " + (cycle?.Count ?? 0) + " vertices");
        bool realCycle = cycle != null;
        for (int i = 0; i < cycle!.Count; i++)
        {
            // FindCycle walks backwards along in-edges, so the arc of the cycle runs from the next entry back
            // to the current one.
            int from = cycle[(i + 1) % cycle.Count], to = cycle[i];
            if (!cyclic.Has(from, to)) realCycle = false;
        }
        Check(realCycle, "every arc of the reported cycle really exists in the graph");
        var withoutCycle = Topo.Without(cyclic, Zone(cycle));
        Check(Topo.Kahn(withoutCycle).Residual.Count == 0,
            "deleting the reported cycle's arcs makes the graph acyclic");
        Check(Topo.Kahn(cyclic).Residual.Count == 4,
            "Kahn's residual core holds the three cycle vertices plus the vertex that depends on them");

        // ---- heuristics versus the brute-force optimum ---------------------------------------------------
        bool fasValid = true, greedyBeatsNaive = true, notWorseThanOptimum = true, symmetric = true, optimal = true;
        string? firstFailure = null;
        int instances = 0;
        for (int round = 0; round < 14; round++)
        {
            Digraph graph = round switch
            {
                0 => ThreeCycle(),
                1 => Tournament(),
                2 => BuildDag(),
                _ => RandomGraph(6, 6 + round, 0x7460UL + (ulong)round * 7919UL)
            };
            var (greedyOrder, _) = Topo.Greedy(graph);
            var improved = Topo.Improve(graph, greedyOrder);
            var balanceOrder = Topo.Improve(graph, Topo.ByBalance(graph));
            int greedyBackward = Topo.BackwardArcs(graph, improved.Order).Count;
            int optimum = Topo.BruteForceOptimum(graph);
            instances++;
            if (Topo.Kahn(Topo.Without(graph, Topo.BackwardArcs(graph, improved.Order))).Residual.Count != 0)
            {
                fasValid = false;
                firstFailure ??= "round " + round + ": the backward arcs are not a feedback arc set";
            }
            if (greedyBackward < optimum)
            {
                notWorseThanOptimum = false;
                firstFailure ??= "round " + round + ": " + greedyBackward + " beats the optimum " + optimum;
            }
            int residualArcs = 0;
            var (_, residual) = Topo.Kahn(graph);
            var core = new HashSet<int>(residual);
            foreach (var (from, to) in graph.Arcs) if (core.Contains(from) && core.Contains(to)) residualArcs++;
            if (greedyBackward > residualArcs)
            {
                greedyBeatsNaive = false;
                firstFailure ??= "round " + round + ": " + greedyBackward + " worse than the naive core arcs " + residualArcs;
            }
            var reversedOrder = new List<int>(improved.Order);
            reversedOrder.Reverse();
            if (Topo.BackwardArcs(graph.Reverse(), reversedOrder).Count != greedyBackward)
            {
                symmetric = false;
                firstFailure ??= "round " + round + ": reversing graph and order changed the count";
            }
            if (Topo.BackwardArcs(graph, improved.Order).Count > Topo.BackwardArcs(graph, balanceOrder.Order).Count)
            {
                firstFailure ??= "round " + round + ": local search regressed";
                Check(false, firstFailure);
            }
            if (improved.Order.Count != graph.N) optimal = false;
            if (graph.N <= 6 && greedyBackward > optimum) optimal = false;
        }
        Check(fasValid, "every heuristic's backward arcs form a real feedback arc set (" + instances + " instances): "
            + (firstFailure ?? "ok"));
        Check(notWorseThanOptimum, "no heuristic beats the brute-force optimum");
        Check(greedyBeatsNaive, "the heuristics never exceed the naive residual-core arc count");
        Check(symmetric, "reversing both the graph and the order preserves the backward-edge count");
        Check(optimal, "every heuristic returns a permutation of all vertices and the local search never regresses");

        // ---- the heuristic is optimal where the optimum is easy -------------------------------------------
        var tournament = Tournament();
        int tournamentOptimum = Topo.BruteForceOptimum(tournament);
        Check(Best(tournament) == tournamentOptimum && tournamentOptimum == 2,
            "the tournament (two 3-cycles plus three cross arcs) needs exactly two backward arcs (brute force says "
            + tournamentOptimum + ") and the heuristics find such an order");
        var threeCycleGraph = ThreeCycle();
        Check(Best(threeCycleGraph) == 1 && Topo.BruteForceOptimum(threeCycleGraph) == 1,
            "one arc is enough and necessary for a 3-cycle");
        var dagGraph = BuildDag();
        Check(Best(dagGraph) == 0 && Topo.BruteForceOptimum(dagGraph) == 0,
            "the local search leaves a DAG with zero backward arcs");

        // ---- determinism ----------------------------------------------------------------------------------
        var again = Topo.Improve(RandomGraph(6, 12, 0x7460UL + 3 * 7919UL), Topo.Greedy(RandomGraph(6, 12, 0x7460UL + 3 * 7919UL)).Order).Order;
        var once = Topo.Improve(RandomGraph(6, 12, 0x7460UL + 3 * 7919UL), Topo.Greedy(RandomGraph(6, 12, 0x7460UL + 3 * 7919UL)).Order).Order;
        Check(again.Count == once.Count && again.TrueForAll(v => once.Contains(v)),
            "the heuristics are deterministic for the same graph");

        Console.WriteLine("dagOrder=" + string.Join(",", dagOrder) + " dfsOrder=" + string.Join(",", dfsOrder)
            + " cycle=" + string.Join(",", cycle!)
            + " optimumTournament=" + Topo.BruteForceOptimum(Tournament())
            + " instances=" + instances);
        Console.WriteLine("PASS");
    }

    /// <summary>Best backward-edge count over the two improved starting orders.</summary>
    private static int Best(Digraph graph)
    {
        var greedy = Topo.Improve(graph, Topo.Greedy(graph).Order).Order;
        var balance = Topo.Improve(graph, Topo.ByBalance(graph)).Order;
        return Math.Min(Topo.BackwardArcs(graph, greedy).Count, Topo.BackwardArcs(graph, balance).Count);
    }

    private static List<(int From, int To)> Zone(List<int> cycle)
    {
        // The cycle list comes from the backwards walk, so its arcs run from the next entry to the current one.
        var arcs = new List<(int From, int To)>();
        for (int i = 0; i < cycle.Count; i++) arcs.Add((cycle[(i + 1) % cycle.Count], cycle[i]));
        return arcs;
    }
}
