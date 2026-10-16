// Task 536 - Implement constrained shortest path (resource constrained) DP. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
//
// The resource-constrained shortest path (RCSP) problem asks for a minimum-cost s -> t path whose
// total resource consumption stays within a budget B; unlike plain shortest paths it is weakly
// NP-hard, so the file solves the small instances exactly with two independent dynamic programs and
// cross-checks them against exhaustive enumeration of simple paths. Each edge carries (to, cost,
// use) with cost >= 1 and use >= 0 (zero-use edges are allowed on purpose). DP #1 (`LevelDp`)
// stores dp[r][v] = cheapest way to reach v using exactly the resource r and sweeps r = 0..B: at
// each level it first closes the level under zero-use edges (a mini Bellman-Ford, terminating
// because costs are positive) and only then pushes positive-use relaxations into higher levels
// (valid because exactly-B is exhausted in increasing order). DP #2 (`StateBellmanFord`) is a
// Bellman-Ford relaxation over the (resource, node) state graph; it is independent of the level
// ordering and copes with zero-use edges natively.
//
// Documented checks: on eight LCG-random graphs (n = 6, 14 edges, cost 1..9, use 0..5, seed 53601)
// the three methods agree for every budget in {0,1,2,3,5,8,13,20}; the hand-checked tight-budget
// example (0->1 cost 4 use 3, 0->2 cost 1 use 5, 1->3 cost 1 use 1, 2->3 cost 1 use 1) yields 5 at
// budgets 4 and 5 - the cheaper 0->2->3 path needs 6 - and 2 from budget 6 on, while budget 3 is
// infeasible because both routes consume 4 or 6 units; budget 0 cannot leave the source
// (sentinel long.MaxValue / 4); dp is monotone
// non-increasing in the budget, the finite-reachability set only grows with the budget; the
// resource-blind Dijkstra value is always a lower bound and is attained exactly once the budget
// covers the total use of the whole graph; and every table is reproduced byte-identically on a
// second run.
using System;
using System.Collections.Generic;

internal sealed class ResourceGraph
{
    internal const long Inf = long.MaxValue / 4;

    internal readonly int N;
    internal readonly List<int[]>[] Out;   // entries are {to, cost, use}

    internal ResourceGraph(int n, IReadOnlyList<int[]> edges)
    {
        N = n;
        Out = new List<int[]>[n];
        for (int v = 0; v < n; v++) Out[v] = new List<int[]>();
        for (int i = 0; i < edges.Count; i++)
        {
            Out[edges[i][0]].Add(new[] { edges[i][1], edges[i][2], edges[i][3] });
        }
    }

    internal long TotalUse()
    {
        long total = 0;
        for (int v = 0; v < N; v++)
        {
            foreach (int[] e in Out[v]) total += e[2];
        }
        return total;
    }

    /** dp[r][v]: cheapest arrival at v consuming exactly r resource. */
    internal long[,] LevelDp(int budget, int source)
    {
        var dp = new long[budget + 1, N];
        for (int r = 0; r <= budget; r++)
        {
            for (int v = 0; v < N; v++) dp[r, v] = Inf;
        }
        dp[0, source] = 0;
        for (int r = 0; r <= budget; r++)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int v = 0; v < N; v++)
                {
                    if (dp[r, v] >= Inf) continue;
                    foreach (int[] e in Out[v])
                    {
                        if (e[2] != 0) continue;
                        long cand = dp[r, v] + e[1];
                        if (cand < dp[r, e[0]])
                        {
                            dp[r, e[0]] = cand;
                            changed = true;
                        }
                    }
                }
            }
            for (int v = 0; v < N; v++)
            {
                if (dp[r, v] >= Inf) continue;
                foreach (int[] e in Out[v])
                {
                    if (e[2] == 0) continue;
                    int next = r + e[2];
                    if (next > budget) continue;
                    long cand = dp[r, v] + e[1];
                    if (cand < dp[next, e[0]]) dp[next, e[0]] = cand;
                }
            }
        }
        return dp;
    }

    /** Independent method: Bellman-Ford over the (resource, node) state graph. */
    internal long[,] StateBellmanFord(int budget, int source)
    {
        var dp = new long[budget + 1, N];
        for (int r = 0; r <= budget; r++)
        {
            for (int v = 0; v < N; v++) dp[r, v] = Inf;
        }
        dp[0, source] = 0;
        int maxPasses = (budget + 1) * N;
        for (int pass = 0; pass < maxPasses; pass++)
        {
            bool changed = false;
            for (int r = 0; r <= budget; r++)
            {
                for (int v = 0; v < N; v++)
                {
                    if (dp[r, v] >= Inf) continue;
                    foreach (int[] e in Out[v])
                    {
                        int next = r + e[2];
                        if (next > budget) continue;
                        long cand = dp[r, v] + e[1];
                        if (cand < dp[next, e[0]])
                        {
                            dp[next, e[0]] = cand;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) break;
        }
        return dp;
    }

    /** Best budget-feasible cost from source to target, exhaustive over simple paths. */
    internal long BruteForce(int budget, int source, int target)
    {
        var used = new bool[N];
        used[source] = true;
        long best = Inf;
        Explore(source, target, budget, 0, used, ref best);
        return best;
    }

    /** Resource-blind Dijkstra, used as a relaxation bound. */
    internal long[] BlindDijkstra(int source)
    {
        var dist = new long[N];
        var done = new bool[N];
        for (int v = 0; v < N; v++) dist[v] = Inf;
        dist[source] = 0;
        for (int step = 0; step < N; step++)
        {
            int best = -1;
            for (int v = 0; v < N; v++)
            {
                if (!done[v] && dist[v] < Inf && (best < 0 || dist[v] < dist[best])) best = v;
            }
            if (best < 0) break;
            done[best] = true;
            foreach (int[] e in Out[best])
            {
                long cand = dist[best] + e[1];
                if (cand < dist[e[0]]) dist[e[0]] = cand;
            }
        }
        return dist;
    }

    internal long BestFeasible(long[,] table, int budget, int target)
    {
        long best = Inf;
        for (int r = 0; r <= budget; r++)
        {
            if (table[r, target] < best) best = table[r, target];
        }
        return best;
    }

    private void Explore(int v, int target, int remaining, long cost, bool[] used, ref long best)
    {
        if (v == target && cost < best) best = cost;
        foreach (int[] e in Out[v])
        {
            if (used[e[0]] || e[2] > remaining) continue;
            used[e[0]] = true;
            Explore(e[0], target, remaining - e[2], cost + e[1], used, ref best);
            used[e[0]] = false;
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /** 14 edges on 6 nodes from a fixed LCG; costs are 1..9 and resource uses 0..5. */
    private static List<int[]> LcgEdges(int seed)
    {
        var edges = new List<int[]>();
        var used = new bool[6, 6];
        int state = seed;
        int attempts = 0;
        while (edges.Count < 14)
        {
            if (++attempts > 100000) throw new InvalidOperationException("LCG failed to produce 14 edges");
            state = (state * 75 + 74) % 65537;
            int u = state % 6;
            state = (state * 75 + 74) % 65537;
            int v = state % 6;
            if (u == v) continue;
            if (used[u, v]) continue;
            used[u, v] = true;
            state = (state * 75 + 74) % 65537;
            int cost = 1 + state % 9;
            state = (state * 75 + 74) % 65537;
            int use = state % 6;
            edges.Add(new[] { u, v, cost, use });
        }
        return edges;
    }

    private static ResourceGraph Examples()
    {
        // 0->1 (4,3), 0->2 (1,5), 1->3 (1,1), 2->3 (1,1)
        return new ResourceGraph(4, new List<int[]>
        {
            new[] { 0, 1, 4, 3 }, new[] { 0, 2, 1, 5 }, new[] { 1, 3, 1, 1 }, new[] { 2, 3, 1, 1 },
        });
    }

    internal static void Main()
    {
        int[] budgets = { 0, 1, 2, 3, 5, 8, 13, 20 };

        for (int seed = 53601; seed < 53609; seed++)
        {
            var graph = new ResourceGraph(6, LcgEdges(seed));
            foreach (int budget in budgets)
            {
                long[,] level = graph.LevelDp(budget, 0);
                long[,] states = graph.StateBellmanFord(budget, 0);
                long brute = graph.BruteForce(budget, 0, 5);
                long levelBest = graph.BestFeasible(level, budget, 5);
                long stateBest = graph.BestFeasible(states, budget, 5);
                Check(levelBest == brute, "level DP equals brute force (seed " + seed + ", B=" + budget + ")");
                Check(stateBest == brute, "state Bellman-Ford equals brute force (seed " + seed + ", B=" + budget + ")");
                for (int r = 0; r <= budget; r++)
                {
                    for (int v = 0; v < 6; v++)
                    {
                        Check(level[r, v] == states[r, v], "tables agree at r=" + r + ", v=" + v);
                    }
                }

                for (int smaller = 0; smaller < budget; smaller++)
                {
                    long[,] less = graph.LevelDp(smaller, 0);
                    long lessBest = graph.BestFeasible(less, smaller, 5);
                    Check(lessBest >= levelBest || levelBest >= ResourceGraph.Inf,
                            "cost is non-increasing in the budget (B=" + budget + " vs " + smaller + ")");
                    if (lessBest < ResourceGraph.Inf)
                    {
                        Check(levelBest < ResourceGraph.Inf, "feasibility only grows with the budget");
                    }
                }
            }

            var blind = graph.BlindDijkstra(0);
            foreach (int budget in budgets)
            {
                long constrained = graph.BestFeasible(graph.LevelDp(budget, 0), budget, 5);
                if (blind[5] < ResourceGraph.Inf)
                {
                    Check(constrained >= blind[5] || constrained >= ResourceGraph.Inf,
                            "resource-blind distance is a lower bound (B=" + budget + ")");
                }
            }

            int huge = (int)graph.TotalUse() + 1;
            long unconstrained = graph.BestFeasible(graph.LevelDp(huge, 0), huge, 5);
            Check(unconstrained == blind[5], "a budget covering every edge reduces to the blind shortest path");

            long[,] repeated = graph.LevelDp(20, 0);
            long[,] first = graph.LevelDp(20, 0);
            for (int r = 0; r <= 20; r++)
            {
                for (int v = 0; v < 6; v++) Check(repeated[r, v] == first[r, v], "deterministic re-run at r=" + r);
            }
        }

        // Hand-checked tight-budget example.
        var example = Examples();
        Check(example.BestFeasible(example.LevelDp(5, 0), 5, 3) == 5, "tight budget 5 forces the cost-5 route");
        Check(example.BestFeasible(example.LevelDp(6, 0), 6, 3) == 2, "budget 6 admits the cost-2 route");
        Check(example.BestFeasible(example.LevelDp(4, 0), 4, 3) == 5, "budget 4 admits only the cost-5 route");
        Check(example.BestFeasible(example.LevelDp(3, 0), 3, 3) >= ResourceGraph.Inf, "budget 3 is infeasible");
        Check(example.LevelDp(0, 0)[0, 0] == 0 && example.BestFeasible(example.LevelDp(0, 0), 0, 3) >= ResourceGraph.Inf,
                "budget 0 cannot leave the source");
        long[,] exampleBf = example.StateBellmanFord(6, 0);
        Check(exampleBf[6, 3] == 2, "zero-use-free example agrees between the two DPs");
        Check(example.BruteForce(6, 0, 3) == 2 && example.BruteForce(5, 0, 3) == 5, "brute force on the example");
        Check(example.BlindDijkstra(0)[3] == 2, "blind shortest path of the example is 2");

        // A zero-use edge must be handled by both DPs.
        var zeroUse = new ResourceGraph(3, new List<int[]>
        {
            new[] { 0, 1, 7, 0 }, new[] { 1, 2, 1, 1 }, new[] { 0, 2, 9, 1 }, new[] { 2, 1, 1, 0 },
        });
        long zeroLevel = zeroUse.BestFeasible(zeroUse.LevelDp(1, 0), 1, 2);
        long zeroStates = zeroUse.BestFeasible(zeroUse.StateBellmanFord(1, 0), 1, 2);
        Check(zeroLevel == 8 && zeroStates == 8, "zero-use edge chains 0->1->2 for cost 8");
        Check(zeroUse.BruteForce(1, 0, 2) == 8, "brute force agrees on the zero-use chain");
        Check(zeroUse.BestFeasible(zeroUse.LevelDp(0, 0), 0, 2) >= ResourceGraph.Inf, "zero-use edge has no budget");

        Console.WriteLine("PASS");
    }
}
