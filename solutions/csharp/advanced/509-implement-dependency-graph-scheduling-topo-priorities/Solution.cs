// Task 509 - Implement dependency graph scheduling (topo + priorities). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
//
// A DAG (nodes with a duration and a priority, edges "u must finish before v starts") and the four
// classic derived views:
//   * TopologicalOrder - Kahn's algorithm over a binary min-heap keyed by (priority, node id), so
//     among the nodes whose dependencies are already satisfied the most urgent one goes first and
//     ties keep id order.
//   * Levels - longest path in edges from any source (a node's depth), by DP over the topological
//     order.
//   * EarliestSchedule - start(v) = max over dependencies of finish(u), finish = start + duration,
//     makespan = the longest weighted path (the unlimited-worker schedule), plus CriticalPath,
//     the node sequence that realises that makespan.
//   * FindCycle - Kahn's algorithm already proves acyclicity by ordering every node; when it cannot,
//     a DFS returns the concrete cycle path (a self-loop is a cycle of length one).
// Documented checks on the crafted 12-node DAG (14 edges, durations 5/3/4/7/2/6/1/9/2/4/3/5):
// every edge respects the produced order, which is exactly 0 2 1 3 5 4 6 7 8 9 10 11; the depths are
// 0..8 with node 11 at depth 8; start(v) == max dependency finish and finish == start + duration
// everywhere; the makespan is 44 and equals the independently brute-forced longest weighted path
// and the planted critical path 0-2-3-5-6-7-9-10-11 (nine nodes summing to 44). The greedy order is
// checked to be the lexicographically smallest valid order under the (priority, id) key by
// brute-force enumeration of all six topological orders of a six-node DAG; a 5-node cycle yields a
// valid closed cycle path and a self-loop a cycle of length one; empty graphs, isolated nodes,
// duplicate edges, the diamond join and a priority-heavy chain are all pinned; twelve seeded random
// DAGs cross-check depths, makespan and dependency safety against brute force.
using System;
using System.Collections.Generic;

internal class Dag
{
    internal Dag(int nodes)
    {
        N = nodes;
        Successors = new List<int>[nodes];
        Predecessors = new List<int>[nodes];
        for (int i = 0; i < nodes; i++)
        {
            Successors[i] = new List<int>();
            Predecessors[i] = new List<int>();
        }
        Duration = new long[nodes];
        Priority = new long[nodes];
        Indegree = new int[nodes];
        for (int i = 0; i < nodes; i++) Duration[i] = 1;
    }

    internal int N;
    internal List<int>[] Successors;
    internal List<int>[] Predecessors;
    internal long[] Duration;
    internal long[] Priority;
    internal int[] Indegree;

    /** Duplicate edges are ignored, so Indegree stays consistent with the adjacency lists. */
    internal void AddEdge(int from, int to)
    {
        foreach (int existing in Successors[from])
        {
            if (existing == to) return;
        }
        Successors[from].Add(to);
        Predecessors[to].Add(from);
        Indegree[to] += 1;
    }

    internal bool HasEdge(int from, int to)
    {
        foreach (int existing in Successors[from])
        {
            if (existing == to) return true;
        }
        return false;
    }

    internal long EdgeCount()
    {
        long total = 0;
        for (int i = 0; i < N; i++) total += Successors[i].Count;
        return total;
    }
}

internal class Seeded
{
    private ulong state;

    internal Seeded(ulong seed) => state = seed;

    internal int Next(int bound)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong h = state;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        h ^= h >> 31;
        return (int)(h % (ulong)bound);
    }
}

internal static class GraphScheduler
{
    /** Kahn with a (priority, id) min-heap: the greedy lexicographically smallest order. */
    internal static List<int> TopologicalOrder(Dag g)
    {
        var heap = new List<int>();
        var indegree = (int[])g.Indegree.Clone();
        for (int i = 0; i < g.N; i++)
        {
            if (indegree[i] == 0) HeapPush(heap, i, g);
        }
        var order = new List<int>();
        while (heap.Count > 0)
        {
            int node = HeapPop(heap, g);
            order.Add(node);
            foreach (int next in g.Successors[node])
            {
                indegree[next] -= 1;
                if (indegree[next] == 0) HeapPush(heap, next, g);
            }
        }
        return order;
    }

    private static int Compare(int a, int b, Dag g)
    {
        if (g.Priority[a] != g.Priority[b]) return g.Priority[a].CompareTo(g.Priority[b]);
        return a.CompareTo(b);
    }

    private static void HeapPush(List<int> heap, int node, Dag g)
    {
        heap.Add(node);
        int index = heap.Count - 1;
        while (index > 0)
        {
            int parent = (index - 1) / 2;
            if (Compare(heap[parent], heap[index], g) <= 0) break;
            (heap[parent], heap[index]) = (heap[index], heap[parent]);
            index = parent;
        }
    }

    private static int HeapPop(List<int> heap, Dag g)
    {
        int top = heap[0];
        heap[0] = heap[heap.Count - 1];
        heap.RemoveAt(heap.Count - 1);
        int index = 0;
        while (true)
        {
            int left = 2 * index + 1;
            int right = left + 1;
            int best = index;
            if (left < heap.Count && Compare(heap[left], heap[best], g) < 0) best = left;
            if (right < heap.Count && Compare(heap[right], heap[best], g) < 0) best = right;
            if (best == index) break;
            (heap[best], heap[index]) = (heap[index], heap[best]);
            index = best;
        }
        return top;
    }

    /** Depth: longest path in edges from any source. */
    internal static int[] Levels(Dag g)
    {
        var levels = new int[g.N];
        foreach (int node in TopologicalOrder(g))
        {
            foreach (int next in g.Successors[node])
            {
                levels[next] = Math.Max(levels[next], levels[node] + 1);
            }
        }
        return levels;
    }

    internal static (long[] Start, long[] Finish, long Makespan) EarliestSchedule(Dag g)
    {
        var start = new long[g.N];
        var finish = new long[g.N];
        long makespan = 0;
        foreach (int node in TopologicalOrder(g))
        {
            long begin = 0;
            foreach (int previous in g.Predecessors[node]) begin = Math.Max(begin, finish[previous]);
            start[node] = begin;
            finish[node] = begin + g.Duration[node];
            makespan = Math.Max(makespan, finish[node]);
        }
        return (start, finish, makespan);
    }

    /** Longest weighted path from a node to any sink (DP over the topological order). */
    private static long[] TailLengths(Dag g)
    {
        var tails = new long[g.N];
        List<int> order = TopologicalOrder(g);
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int node = order[i];
            long best = 0;
            foreach (int next in g.Successors[node]) best = Math.Max(best, tails[next]);
            tails[node] = g.Duration[node] + best;
        }
        return tails;
    }

    /** The node sequence that realises the longest weighted path (ties broken by node id). */
    internal static List<int> CriticalPath(Dag g)
    {
        long[] tails = TailLengths(g);
        int best = -1;
        for (int i = 0; i < g.N; i++)
        {
            if (g.Indegree[i] != 0) continue;
            if (best < 0 || tails[i] > tails[best]
                    || (tails[i] == tails[best] && i < best)) best = i;
        }
        var path = new List<int>();
        while (best >= 0)
        {
            path.Add(best);
            int next = -1;
            foreach (int candidate in g.Successors[best])
            {
                if (next < 0 || tails[candidate] > tails[next]
                        || (tails[candidate] == tails[next] && candidate < next)) next = candidate;
            }
            best = next;
        }
        return path;
    }

    internal static List<int> FindCycle(Dag g)
    {
        if (TopologicalOrder(g).Count == g.N) return new List<int>();
        var state = new int[g.N];     // 0 unseen, 1 on stack, 2 done
        var stack = new List<int>();
        var path = new List<int>();
        for (int i = 0; i < g.N; i++)
        {
            if (state[i] == 0 && DfsCycle(g, i, state, stack, path)) return path;
        }
        return path;
    }

    private static bool DfsCycle(Dag g, int node, int[] state, List<int> stack, List<int> path)
    {
        state[node] = 1;
        stack.Add(node);
        foreach (int next in g.Successors[node])
        {
            if (state[next] == 1)
            {
                int start = stack.IndexOf(next);
                for (int i = start; i < stack.Count; i++) path.Add(stack[i]);
                path.Add(next);
                return true;
            }
            if (state[next] == 0 && DfsCycle(g, next, state, stack, path)) return true;
        }
        stack.RemoveAt(stack.Count - 1);
        state[node] = 2;
        return false;
    }

    /** All topological orders of a small DAG, using a different search style than Kahn. */
    internal static List<List<int>> AllTopologicalOrders(Dag g)
    {
        var results = new List<List<int>>();
        var indegree = (int[])g.Indegree.Clone();
        var current = new List<int>();
        var used = new bool[g.N];
        Explore(g, indegree, current, used, results);
        return results;
    }

    private static void Explore(Dag g, int[] indegree, List<int> current, bool[] used,
            List<List<int>> results)
    {
        if (current.Count == g.N)
        {
            results.Add(new List<int>(current));
            return;
        }
        for (int node = 0; node < g.N; node++)
        {
            if (used[node] || indegree[node] != 0) continue;
            used[node] = true;
            current.Add(node);
            foreach (int next in g.Successors[node]) indegree[next] -= 1;
            Explore(g, indegree, current, used, results);
            foreach (int next in g.Successors[node]) indegree[next] += 1;
            current.RemoveAt(current.Count - 1);
            used[node] = false;
        }
    }

    /** Brute-force longest weighted path over all simple paths from every source. */
    internal static long BruteForceLongestPath(Dag g)
    {
        long best = 0;
        for (int start = 0; start < g.N; start++)
        {
            if (g.Indegree[start] != 0) continue;
            best = Math.Max(best, DfsTail(g, start));
        }
        return best;
    }

    private static long DfsTail(Dag g, int node)
    {
        long bestTail = 0;
        foreach (int next in g.Successors[node]) bestTail = Math.Max(bestTail, DfsTail(g, next));
        return g.Duration[node] + bestTail;
    }

    internal static int BruteForceLevel(Dag g, int node)
    {
        int best = 0;
        foreach (int previous in g.Predecessors[node])
        {
            best = Math.Max(best, 1 + BruteForceLevel(g, previous));
        }
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- the crafted 12-node DAG -------------------------------------------------------------------
        var dag = new Dag(12);
        long[] durations = { 5, 3, 4, 7, 2, 6, 1, 9, 2, 4, 3, 5 };
        long[] priorities = { 3, 1, 0, 2, 1, 0, 4, 1, 2, 3, 0, 1 };
        for (int i = 0; i < dag.N; i++)
        {
            dag.Duration[i] = durations[i];
            dag.Priority[i] = priorities[i];
        }
        int[][] edges = {
            new[] { 0, 1 }, new[] { 0, 2 }, new[] { 1, 3 }, new[] { 2, 3 }, new[] { 3, 4 },
            new[] { 3, 5 }, new[] { 4, 6 }, new[] { 5, 6 }, new[] { 6, 7 }, new[] { 7, 8 },
            new[] { 7, 9 }, new[] { 8, 10 }, new[] { 9, 10 }, new[] { 10, 11 },
        };
        foreach (int[] edge in edges) dag.AddEdge(edge[0], edge[1]);
        Check(dag.EdgeCount() == 14, "the crafted DAG has 14 edges");
        dag.AddEdge(0, 1);
        Check(dag.EdgeCount() == 14, "duplicate edges are ignored");

        List<int> order = GraphScheduler.TopologicalOrder(dag);
        Check(order.Count == dag.N, "the crafted DAG is acyclic");
        int[] expectedOrder = { 0, 2, 1, 3, 5, 4, 6, 7, 8, 9, 10, 11 };
        for (int i = 0; i < order.Count; i++)
        {
            Check(order[i] == expectedOrder[i], "order position " + i + " is " + expectedOrder[i]);
        }
        var position = new int[dag.N];
        for (int i = 0; i < order.Count; i++) position[order[i]] = i;
        for (int u = 0; u < dag.N; u++)
        {
            foreach (int v in dag.Successors[u])
            {
                Check(position[u] < position[v], "edge " + u + " -> " + v + " respects the order");
            }
        }

        // --- depths, schedule and critical path ------------------------------------------------------------
        int[] levels = GraphScheduler.Levels(dag);
        for (int i = 0; i < dag.N; i++)
        {
            Check(levels[i] == GraphScheduler.BruteForceLevel(dag, i), "depth of node " + i);
        }
        Check(levels[0] == 0 && levels[11] == 8, "depth 0 at the source and depth 8 at the sink");
        (long[] start, long[] finish, long makespan) = GraphScheduler.EarliestSchedule(dag);
        Check(makespan == 44, "makespan is 44 (got " + makespan + ")");
        Check(makespan == GraphScheduler.BruteForceLongestPath(dag), "makespan == longest weighted path");
        Check(makespan == 5 + 4 + 7 + 6 + 1 + 9 + 4 + 3 + 5, "makespan equals the planted critical path sum");
        for (int v = 0; v < dag.N; v++)
        {
            Check(finish[v] == start[v] + dag.Duration[v], "finish = start + duration at " + v);
            long bound = 0;
            foreach (int u in dag.Predecessors[v]) bound = Math.Max(bound, finish[u]);
            Check(start[v] == bound, "start = max dependency finish at " + v);
        }
        Check(start[0] == 0, "the source starts at zero");
        Check(finish[11] == makespan, "the sink finishes at the makespan");
        List<int> critical = GraphScheduler.CriticalPath(dag);
        int[] expectedCritical = { 0, 2, 3, 5, 6, 7, 9, 10, 11 };
        Check(critical.Count == expectedCritical.Length, "the critical path has nine nodes");
        for (int i = 0; i < expectedCritical.Length; i++)
        {
            Check(critical[i] == expectedCritical[i], "critical path node " + i);
        }
        for (int i = 0; i + 1 < critical.Count; i++)
        {
            Check(dag.HasEdge(critical[i], critical[i + 1]), "critical path edge " + i);
        }
        Check(dag.Predecessors[critical[0]].Count == 0, "the critical path starts at a source");
        Check(dag.Successors[critical[critical.Count - 1]].Count == 0, "and ends at a sink");
        long pathLength = 0;
        foreach (int node in critical) pathLength += dag.Duration[node];
        Check(pathLength == makespan, "the critical path accumulates the whole makespan");

        // --- the greedy order is the lexicographically smallest valid order ----------------------------------
        var small = new Dag(6);
        long[] smallDurations = { 1, 2, 3, 1, 1, 2 };
        long[] smallPriorities = { 2, 0, 1, 0, 3, 1 };
        for (int i = 0; i < small.N; i++)
        {
            small.Duration[i] = smallDurations[i];
            small.Priority[i] = smallPriorities[i];
        }
        small.AddEdge(0, 2);
        small.AddEdge(2, 3);
        small.AddEdge(1, 3);
        small.AddEdge(3, 4);
        small.AddEdge(3, 5);
        List<List<int>> allOrders = GraphScheduler.AllTopologicalOrders(small);
        Check(allOrders.Count == 6, "the six-node DAG has six topological orders");
        List<int>? bruteBest = null;
        foreach (List<int> candidate in allOrders)
        {
            if (bruteBest == null || KeyLess(candidate, bruteBest, small)) bruteBest = candidate;
        }
        List<int> greedy = GraphScheduler.TopologicalOrder(small);
        Check(greedy.Count == 6, "the greedy order covers every node");
        int[] expectedSmall = { 1, 0, 2, 3, 5, 4 };
        for (int i = 0; i < greedy.Count; i++)
        {
            Check(greedy[i] == expectedSmall[i], "small order position " + i);
            Check(greedy[i] == bruteBest![i], "greedy == lexicographically smallest at " + i);
        }
        Check(greedy[0] == 1, "the highest-priority source goes first");

        // --- cycles -------------------------------------------------------------------------------------------
        var cyclic = new Dag(5);
        cyclic.AddEdge(0, 1);
        cyclic.AddEdge(1, 2);
        cyclic.AddEdge(2, 3);
        cyclic.AddEdge(3, 1);
        cyclic.AddEdge(2, 4);
        Check(GraphScheduler.TopologicalOrder(cyclic).Count == 1, "only node 0 is orderable in the cycle");
        List<int> cycle = GraphScheduler.FindCycle(cyclic);
        Check(cycle.Count == 4, "the cycle visits four nodes");
        for (int i = 0; i + 1 < cycle.Count; i++)
        {
            Check(cyclic.HasEdge(cycle[i], cycle[i + 1]), "cycle edge " + i);
        }
        Check(cycle[0] == cycle[cycle.Count - 1], "the reported path closes on itself");
        Check(cycle.Contains(1) && cycle.Contains(2) && cycle.Contains(3), "the cycle contains 1, 2 and 3");
        var selfLoop = new Dag(3);
        selfLoop.AddEdge(0, 1);
        selfLoop.AddEdge(1, 2);
        selfLoop.AddEdge(2, 2);
        List<int> selfLoopCycle = GraphScheduler.FindCycle(selfLoop);
        Check(selfLoopCycle.Count == 2 && selfLoopCycle[0] == 2 && selfLoopCycle[1] == 2,
                "a self-loop is reported as a cycle of length one");
        Check(GraphScheduler.FindCycle(dag).Count == 0, "the crafted DAG has no cycle");

        // --- degenerate shapes ---------------------------------------------------------------------------------
        var empty = new Dag(0);
        Check(GraphScheduler.TopologicalOrder(empty).Count == 0, "the empty graph orders to nothing");
        Check(GraphScheduler.EarliestSchedule(empty).Makespan == 0, "the empty graph has no makespan");
        Check(GraphScheduler.CriticalPath(empty).Count == 0, "the empty graph has no critical path");
        var isolated = new Dag(4);
        for (int i = 0; i < 4; i++)
        {
            isolated.Duration[i] = i + 1;
            isolated.Priority[i] = 3 - i;
        }
        List<int> isolatedOrder = GraphScheduler.TopologicalOrder(isolated);
        Check(isolatedOrder[0] == 3 && isolatedOrder[3] == 0, "isolated nodes follow priority order");
        Check(GraphScheduler.EarliestSchedule(isolated).Makespan == 4,
                "isolated nodes run in parallel, so the makespan is the longest one");
        Check(GraphScheduler.CriticalPath(isolated).Count == 1, "an isolated node is its own critical path");
        var diamond = new Dag(4);
        long[] diamondDurations = { 4, 2, 6, 1 };
        for (int i = 0; i < 4; i++) diamond.Duration[i] = diamondDurations[i];
        diamond.AddEdge(0, 1);
        diamond.AddEdge(0, 2);
        diamond.AddEdge(1, 3);
        diamond.AddEdge(2, 3);
        var (diamondStart, diamondFinish, diamondMakespan) = GraphScheduler.EarliestSchedule(diamond);
        Check(diamondMakespan == 11, "the longer diamond branch sets the makespan");
        Check(diamondStart[1] == 4 && diamondStart[2] == 4, "both branches start together");
        Check(diamondStart[3] == 10, "the join waits for the slower branch");
        Check(diamondFinish[3] == 11, "the join finishes at the makespan");

        // --- priorities never break dependencies -------------------------------------------------------------------
        var priorityHeavy = new Dag(6);
        for (int i = 0; i < 6; i++) priorityHeavy.Priority[i] = 5 - i;
        priorityHeavy.AddEdge(5, 0);
        priorityHeavy.AddEdge(0, 1);
        priorityHeavy.AddEdge(1, 2);
        List<int> heavyOrder = GraphScheduler.TopologicalOrder(priorityHeavy);
        int[] expectedHeavy = { 5, 4, 3, 0, 1, 2 };
        for (int i = 0; i < expectedHeavy.Length; i++)
        {
            Check(heavyOrder[i] == expectedHeavy[i], "priority-heavy order position " + i);
        }
        Check(priorityHeavy.Priority[0] == 5 && heavyOrder.IndexOf(0) > heavyOrder.IndexOf(5),
                "the urgent node still waits for its blocker");

        // --- seeded random DAGs cross-checked against brute force -------------------------------------------------------
        var rng = new Seeded(20240609UL);
        for (int trial = 0; trial < 12; trial++)
        {
            const int n = 7;
            var randomDag = new Dag(n);
            for (int i = 0; i < n; i++)
            {
                randomDag.Duration[i] = rng.Next(5) + 1;
                randomDag.Priority[i] = rng.Next(4);
            }
            for (int u = 0; u < n; u++)
            {
                for (int v = u + 1; v < n; v++)
                {
                    if (rng.Next(3) == 0) randomDag.AddEdge(u, v);
                }
            }
            List<int> randomOrder = GraphScheduler.TopologicalOrder(randomDag);
            Check(randomOrder.Count == n, "seeded DAG " + trial + " is acyclic");
            int[] randomLevels = GraphScheduler.Levels(randomDag);
            var randomSchedule = GraphScheduler.EarliestSchedule(randomDag);
            for (int i = 0; i < n; i++)
            {
                Check(randomLevels[i] == GraphScheduler.BruteForceLevel(randomDag, i),
                        "seeded depth at trial " + trial + " node " + i);
            }
            Check(randomSchedule.Makespan == GraphScheduler.BruteForceLongestPath(randomDag),
                    "seeded makespan at trial " + trial);
            List<int> randomCritical = GraphScheduler.CriticalPath(randomDag);
            long randomCriticalLength = 0;
            foreach (int node in randomCritical) randomCriticalLength += randomDag.Duration[node];
            Check(randomCriticalLength == randomSchedule.Makespan,
                    "seeded critical path length at trial " + trial);
            var orderIndex = new int[n];
            for (int i = 0; i < n; i++) orderIndex[randomOrder[i]] = i;
            for (int v = 0; v < n; v++)
            {
                Check(randomSchedule.Start[v] >= 0, "seeded start is non-negative");
                foreach (int u in randomDag.Predecessors[v])
                {
                    Check(randomSchedule.Start[v] >= randomSchedule.Finish[u],
                            "seeded schedule respects dependency " + u + " -> " + v);
                    Check(orderIndex[u] < orderIndex[v], "seeded order respects dependency");
                }
            }
        }

        // --- determinism ------------------------------------------------------------------------------------------------
        List<int> repeatOrder = GraphScheduler.TopologicalOrder(dag);
        for (int i = 0; i < order.Count; i++) Check(repeatOrder[i] == order[i], "deterministic order at " + i);
        var repeatSchedule = GraphScheduler.EarliestSchedule(dag);
        Check(repeatSchedule.Makespan == makespan, "deterministic makespan");
        List<int> repeatCritical = GraphScheduler.CriticalPath(dag);
        Check(repeatCritical.Count == critical.Count, "deterministic critical path");

        Console.WriteLine("PASS");
    }

    private static bool KeyLess(List<int> a, List<int> b, Dag g)
    {
        for (int i = 0; i < a.Count; i++)
        {
            if (g.Priority[a[i]] != g.Priority[b[i]]) return g.Priority[a[i]] < g.Priority[b[i]];
            if (a[i] != b[i]) return a[i] < b[i];
        }
        return false;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
