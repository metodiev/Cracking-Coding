// Task 670 - Implement resource allocation graph and Banker's algorithm simulator.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// Two views of the same deadlock-avoidance question, implemented independently so they can police
// each other:
//   * Banker's algorithm on the textbook matrices (Available / Allocation / Max, with
//     Need = Max - Allocation maintained as a live invariant). The safety test is the greedy
//     "find any process whose Need fits the work vector, let it finish, add its Allocation back" scan;
//     `TryGrant` applies the three-part test a request must pass — Request <= Need, Request <= Available,
//     and the state that *would* result is still safe — and only mutates the state when all three pass.
//   * A resource allocation graph (counted per-resource instances, process->resource request edges and
//     resource->process allocation edges) with a cycle detector (DFS over the bipartite graph) and a
//     reduction-based deadlock test (a process whose outstanding requests can be satisfied finishes and
//     hands its instances back; whatever is left over is deadlocked).
// The file demonstrates the classical facts rather than asserting them by fiat: a safe state with a
// granted request, a request refused because it is unsafe (the state must be left untouched — the
// negative check that matters), the single-instance equivalence cycle <=> deadlock <=> unsafe, and the
// multi-instance counter-example where a cycle exists but nothing is deadlocked (with >= 2 instances per
// resource type a cycle is necessary but not sufficient for deadlock).
// Cross-checks: the greedy safety scan is compared against a brute-force search over all completion
// permutations for small states (200 random states); the graph reduction is compared with the cycle
// detector and with the matrix safety test over randomised single-instance states (100 cases: all three
// must agree), and over randomised multi-instance states the implication deadlock => unsafe is asserted
// (100 cases). The graph's per-process requests count as "at least one more instance", which is exactly
// the weakening under which deadlock still implies unsafe.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic; no framework RNG and no wall-clock seeds.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Safety
{
    internal bool Safe;
    internal List<int> Order = new List<int>();
}

internal sealed class BankerState
{
    internal int N;
    internal int M;
    internal int[] Available = Array.Empty<int>();
    internal int[][] Allocation = Array.Empty<int[]>();
    internal int[][] Max = Array.Empty<int[]>();
    internal int[][] Need = Array.Empty<int[]>();

    internal static BankerState Create(int[] available, int[][] allocation, int[][] max)
    {
        var state = new BankerState
        {
            N = allocation.Length,
            M = available.Length,
            Available = (int[])available.Clone(),
            Allocation = Clone(allocation),
            Max = Clone(max)
        };
        state.Need = new int[state.N][];
        for (int i = 0; i < state.N; i++)
        {
            state.Need[i] = new int[state.M];
            for (int j = 0; j < state.M; j++) state.Need[i][j] = state.Max[i][j] - state.Allocation[i][j];
        }
        return state;
    }

    internal static int[][] Clone(int[][] matrix)
    {
        var copy = new int[matrix.Length][];
        for (int i = 0; i < matrix.Length; i++) copy[i] = (int[])matrix[i].Clone();
        return copy;
    }

    internal BankerState Clone()
    {
        var copy = Create(Available, Allocation, Max);
        return copy;
    }

    /// Invariant: Need = Max - Allocation and every entry is non-negative.
    internal bool NeedConsistent()
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                if (Need[i][j] != Max[i][j] - Allocation[i][j]) return false;
                if (Need[i][j] < 0 || Allocation[i][j] < 0 || Available[j] < 0) return false;
            }
        }
        return true;
    }

    /// Conservation: free instances plus everything allocated always adds up to the capacity vector.
    internal bool Conserved(int[] capacity)
    {
        for (int j = 0; j < M; j++)
        {
            int total = Available[j];
            for (int i = 0; i < N; i++) total += Allocation[i][j];
            if (total != capacity[j]) return false;
        }
        return true;
    }

    /// Greedy safety scan: O(N^2 * M), and the classic teaching form of the algorithm.
    internal Safety Greedy()
    {
        var result = new Safety();
        var work = (int[])Available.Clone();
        var finished = new bool[N];
        int remaining = N;
        while (remaining > 0)
        {
            int pick = -1;
            for (int i = 0; i < N; i++)
            {
                if (finished[i]) continue;
                bool fits = true;
                for (int j = 0; j < M; j++)
                {
                    if (Need[i][j] > work[j]) { fits = false; break; }
                }
                if (fits) { pick = i; break; }
            }
            if (pick < 0) break;
            for (int j = 0; j < M; j++) work[j] += Allocation[pick][j];
            finished[pick] = true;
            result.Order.Add(pick);
            remaining--;
        }
        result.Safe = remaining == 0;
        return result;
    }

    /// Brute force: is there *any* permutation of the processes that can be completed in order? Used only
    /// on small states to validate the greedy scan.
    internal bool BruteForceSafe()
    {
        var work = (int[])Available.Clone();
        var used = new bool[N];
        return Search(work, used, 0);
    }

    private bool Search(int[] work, bool[] used, int done)
    {
        if (done == N) return true;
        for (int i = 0; i < N; i++)
        {
            if (used[i]) continue;
            bool fits = true;
            for (int j = 0; j < M; j++)
            {
                if (Need[i][j] > work[j]) { fits = false; break; }
            }
            if (!fits) continue;
            var next = (int[])work.Clone();
            for (int j = 0; j < M; j++) next[j] += Allocation[i][j];
            used[i] = true;
            bool ok = Search(next, used, done + 1);
            used[i] = false;
            if (ok) return true;
        }
        return false;
    }

    /// Replays a claimed safe sequence against the state and reports whether it really is one.
    internal bool ValidateSequence(List<int> order)
    {
        if (order.Count != N) return false;
        var seen = new bool[N];
        var work = (int[])Available.Clone();
        foreach (int p in order)
        {
            if (p < 0 || p >= N || seen[p]) return false;
            seen[p] = true;
            for (int j = 0; j < M; j++) if (Need[p][j] > work[j]) return false;
            for (int j = 0; j < M; j++) work[j] += Allocation[p][j];
        }
        return true;
    }
}

internal static class Banker
{
    /// Full request test. The state is mutated only when the request is granted.
    internal static bool TryGrant(BankerState state, int process, int[] request, out string reason)
    {
        reason = "granted";
        for (int j = 0; j < state.M; j++)
        {
            if (request[j] > state.Need[process][j])
            {
                reason = "request exceeds the declared maximum (Need)";
                return false;
            }
            if (request[j] > state.Available[j])
            {
                reason = "not enough free instances right now";
                return false;
            }
        }
        var trial = state.Clone();
        for (int j = 0; j < trial.M; j++)
        {
            trial.Available[j] -= request[j];
            trial.Allocation[process][j] += request[j];
            trial.Need[process][j] -= request[j];
        }
        if (!trial.Greedy().Safe)
        {
            reason = "granting it leads to an unsafe state";
            return false;
        }
        state.Available = trial.Available;
        state.Allocation = trial.Allocation;
        state.Need = trial.Need;
        return true;
    }

    /// Budgeted request: the process asks for every instance it still needs (only used for cross-checks).
    internal static int[] FullNeed(BankerState state, int process)
    {
        var request = new int[state.M];
        for (int j = 0; j < state.M; j++) request[j] = state.Need[process][j];
        return request;
    }
}

internal sealed class Rag
{
    internal int Processes;
    internal int Resources;
    internal int[] Instances = Array.Empty<int>();
    internal int[][] Allocated = Array.Empty<int[]>();   // [process][resource] -> instances held
    internal bool[][] Request = Array.Empty<bool[]>();   // [process][resource] -> waiting for one more

    internal static Rag FromMatrices(int[] instances, int[][] allocation, int[][] need)
    {
        int n = allocation.Length;
        int m = instances.Length;
        var g = new Rag
        {
            Processes = n,
            Resources = m,
            Instances = (int[])instances.Clone(),
            Allocated = BankerState.Clone(allocation),
            Request = new bool[n][]
        };
        for (int i = 0; i < n; i++)
        {
            g.Request[i] = new bool[m];
            for (int j = 0; j < m; j++) g.Request[i][j] = need[i][j] > 0;
        }
        return g;
    }

    internal int Free(int resource)
    {
        int used = 0;
        for (int i = 0; i < Processes; i++) used += Allocated[i][resource];
        return Instances[resource] - used;
    }

    /// Bipartite DFS cycle detection over the allocation/request graph.
    internal bool HasCycle()
    {
        int nodes = Processes + Resources;
        var colour = new int[nodes];    // 0 = white, 1 = grey, 2 = black
        for (int start = 0; start < nodes; start++)
        {
            if (colour[start] == 0 && Visit(start, colour)) return true;
        }
        return false;
    }

    private bool Visit(int node, int[] colour)
    {
        colour[node] = 1;
        foreach (int next in Neighbours(node))
        {
            if (colour[next] == 1) return true;
            if (colour[next] == 0 && Visit(next, colour)) return true;
        }
        colour[node] = 2;
        return false;
    }

    private List<int> Neighbours(int node)
    {
        var list = new List<int>();
        if (node < Processes)
        {
            for (int j = 0; j < Resources; j++) if (Request[node][j]) list.Add(Processes + j);
        }
        else
        {
            int resource = node - Processes;
            for (int i = 0; i < Processes; i++) if (Allocated[i][resource] > 0) list.Add(i);
        }
        return list;
    }

    /// Reduction test: repeatedly let a process whose outstanding requests can be served run to
    /// completion and hand its instances back. Processes left over are deadlocked.
    internal bool Deadlocked()
    {
        int n = Processes;
        int m = Resources;
        var free = new int[m];
        for (int j = 0; j < m; j++) free[j] = Free(j);
        var done = new bool[n];
        int remaining = n;
        while (remaining > 0)
        {
            int pick = -1;
            for (int i = 0; i < n; i++)
            {
                if (done[i]) continue;
                bool fits = true;
                for (int j = 0; j < m; j++)
                {
                    if (Request[i][j] && free[j] < 1) { fits = false; break; }
                }
                if (fits) { pick = i; break; }
            }
            if (pick < 0) break;
            for (int j = 0; j < m; j++) free[j] += Allocated[pick][j];
            done[pick] = true;
            remaining--;
        }
        return remaining > 0;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// The textbook snapshot: Available = [3,3,2] with the five-process Allocation/Max matrices.
    private static BankerState Textbook() => BankerState.Create(
        new[] { 3, 3, 2 },
        new[]
        {
            new[] { 0, 1, 0 },
            new[] { 2, 0, 0 },
            new[] { 3, 0, 2 },
            new[] { 2, 1, 1 },
            new[] { 0, 0, 2 }
        },
        new[]
        {
            new[] { 7, 5, 3 },
            new[] { 3, 2, 2 },
            new[] { 9, 0, 2 },
            new[] { 2, 2, 2 },
            new[] { 4, 3, 3 }
        });

    private static readonly int[] Capacity = { 10, 5, 7 };

    public static void Main()
    {
        var state = Textbook();
        Check(state.NeedConsistent(), "Need = Max - Allocation is maintained on construction");
        Check(state.Conserved(Capacity), "Available + sum(Allocation) equals the capacity vector");
        Check(state.Need[1][0] == 1 && state.Need[0][1] == 4, "spot-check of the Need matrix (P1/R0 = 1, P0/R1 = 4)");

        // ---- the safe state and its witness --------------------------------------------------
        Safety safety = state.Greedy();
        Check(safety.Safe, "the textbook state is safe");
        Check(state.ValidateSequence(safety.Order), "the reported order really is a safe sequence");
        Check(state.BruteForceSafe(), "the brute-force permutation search agrees that it is safe");

        // ---- grant P1's request [1,0,2] -------------------------------------------------------
        var afterP1 = state.Clone();
        var request1 = new[] { 1, 0, 2 };
        string reason;
        Check(Banker.TryGrant(afterP1, 1, request1, out reason), "P1's request [1,0,2] is granted: " + reason);
        Check(afterP1.Available[0] == 2 && afterP1.Available[1] == 3 && afterP1.Available[2] == 0,
              "after the grant Available is [2,3,0]");
        Check(afterP1.Need[1][0] == 0 && afterP1.Need[1][2] == 0, "P1's remaining need is [0,2,0]");
        Check(afterP1.NeedConsistent() && afterP1.Conserved(Capacity), "the granted state still satisfies both invariants");
        Check(afterP1.Greedy().Safe && afterP1.ValidateSequence(afterP1.Greedy().Order),
              "the granted state is safe and the witness replays");

        // ---- the three ways a request can be refused -----------------------------------------
        var before = afterP1.Clone();
        var tooMany = new[] { 3, 3, 0 };
        Check(!Banker.TryGrant(afterP1, 4, tooMany, out reason), "P4's request [3,3,0] is refused");
        Check(reason.Contains("free instances"), "and it is refused for lack of free instances");
        var tooGreedy = new[] { 2, 0, 0 };
        Check(!Banker.TryGrant(afterP1, 1, tooGreedy, out reason), "P1 cannot ask for more than its declared maximum");
        Check(reason.Contains("Need"), "and it is refused on the Need test");
        var unsafeRequest = new[] { 0, 2, 0 };
        Check(!Banker.TryGrant(afterP1, 0, unsafeRequest, out reason), "P0's request [0,2,0] is refused");
        Check(reason.Contains("unsafe"), "and it is refused because the resulting state is unsafe");

        // A refused request must leave the state completely untouched.
        bool untouched = true;
        for (int i = 0; i < afterP1.N && untouched; i++)
        {
            for (int j = 0; j < afterP1.M; j++)
            {
                if (afterP1.Allocation[i][j] != before.Allocation[i][j] || afterP1.Need[i][j] != before.Need[i][j])
                {
                    untouched = false;
                    break;
                }
            }
        }
        for (int j = 0; j < afterP1.M; j++) if (afterP1.Available[j] != before.Available[j]) untouched = false;
        Check(untouched, "the refused request did not disturb the state at all");

        // ---- degenerate states --------------------------------------------------------------
        var empty = BankerState.Create(new[] { 1 }, Array.Empty<int[]>(), Array.Empty<int[]>());
        Check(empty.Greedy().Safe && empty.Greedy().Order.Count == 0, "a state with no processes is trivially safe");
        var idle = BankerState.Create(new[] { 0, 0 }, new[] { new[] { 0, 0 }, new[] { 0, 0 } },
                                                      new[] { new[] { 0, 0 }, new[] { 0, 0 } });
        Check(idle.Greedy().Safe && idle.BruteForceSafe(), "all-zero needs are safe even with nothing free");
        var stranded = BankerState.Create(new[] { 0, 1, 0 }, state.Allocation, state.Max);
        Check(!stranded.Greedy().Safe, "shrinking Available to [0,1,0] makes the same allocation unsafe");
        Check(!stranded.BruteForceSafe(), "and the brute force agrees on that state too");

        // ---- the graph view ------------------------------------------------------------------
        var deadlockGraph = Rag.FromMatrices(
            new[] { 1, 1 },
            new[] { new[] { 1, 0 }, new[] { 0, 1 } },        // P0 holds R0, P1 holds R1
            new[] { new[] { 0, 1 }, new[] { 1, 0 } });       // P0 wants R1, P1 wants R0
        Check(deadlockGraph.HasCycle(), "the classic two-process graph contains a cycle");
        Check(deadlockGraph.Deadlocked(), "and it is genuinely deadlocked (nothing can be reduced away)");
        var deadlockState = BankerState.Create(
            new[] { 0, 0 },
            new[] { new[] { 1, 0 }, new[] { 0, 1 } },
            new[] { new[] { 1, 1 }, new[] { 1, 1 } });
        Check(!deadlockState.Greedy().Safe, "the same situation is unsafe for Banker's algorithm");

        var cycleOnly = Rag.FromMatrices(
            new[] { 1, 3 },
            new[] { new[] { 1, 1 }, new[] { 0, 1 } },        // P0 holds R0 and one R1, P1 holds one R1
            new[] { new[] { 0, 1 }, new[] { 1, 0 } });       // P0 waits for R1, P1 waits for R0
        Check(cycleOnly.HasCycle(), "the multi-instance graph also contains a cycle (P0 -> R1 -> P1 -> R0 -> P0)");
        Check(!cycleOnly.Deadlocked(), "but nothing is deadlocked: one free R1 instance lets P1 finish");
        var cycleState = BankerState.Create(
            new[] { 0, 1 },
            new[] { new[] { 1, 1 }, new[] { 0, 1 } },
            new[] { new[] { 1, 1 }, new[] { 1, 1 } });
        Check(cycleState.Greedy().Safe, "and Banker's algorithm calls that state safe");

        // ---- randomised cross-checks ---------------------------------------------------------
        ulong rng = 670001UL;
        int permutationCases = 200;
        for (int s = 0; s < permutationCases; s++)
        {
            BankerState random = RandomState(ref rng, 2 + Rng.NextBelow(ref rng, 4), 2 + Rng.NextBelow(ref rng, 3), 2);
            Safety greedy = random.Greedy();
            Check(greedy.Safe == random.BruteForceSafe(),
                  "state " + s + ": greedy scan and permutation search agree (greedy=" + greedy.Safe + ")");
            if (greedy.Safe) Check(random.ValidateSequence(greedy.Order), "state " + s + ": the greedy witness replays");
            Check(random.NeedConsistent(), "state " + s + ": Need stays consistent");
        }

        int singleInstanceCases = 100;
        int cycleCases = 0;
        int deadlockCases = 0;
        for (int s = 0; s < singleInstanceCases; s++)
        {
            int n = 2 + Rng.NextBelow(ref rng, 3);
            int m = 2 + Rng.NextBelow(ref rng, 2);
            var instances = new int[m];
            for (int j = 0; j < m; j++) instances[j] = 1;
            var allocation = new int[n][];
            var max = new int[n][];
            for (int i = 0; i < n; i++)
            {
                allocation[i] = new int[m];
                max[i] = new int[m];
            }
            for (int j = 0; j < m; j++)
            {
                int holder = Rng.NextBelow(ref rng, n + 1) - 1;    // -1 means the instance is free
                if (holder >= 0) allocation[holder][j] = 1;
                for (int i = 0; i < n; i++) max[i][j] = allocation[i][j] == 1 ? 1 : (Rng.NextBelow(ref rng, 2));
            }
            var single = BankerState.Create(FreeInstances(instances, allocation), allocation, max);
            bool safe = single.Greedy().Safe;
            Check(safe == single.BruteForceSafe(), "single-instance state " + s + ": greedy and brute force agree");
            var graph = Rag.FromMatrices(instances, single.Allocation, single.Need);
            bool cycle = graph.HasCycle();
            bool deadlock = graph.Deadlocked();
            Check(cycle == deadlock,
                  "single-instance state " + s + ": a cycle and a deadlock coincide (cycle=" + cycle + ")");
            Check(deadlock == !safe, "single-instance state " + s + ": deadlock coincides with unsafety");
            if (cycle) cycleCases++;
            if (deadlock) deadlockCases++;
        }
        Check(deadlockCases >= 10 && cycleCases >= 10,
              "the single-instance sweep produced deadlocks in " + deadlockCases + " of " + singleInstanceCases + " states");
        Check(deadlockCases < singleInstanceCases, "and it also produced plenty of deadlock-free states");

        int multiCases = 100;
        int multiDeadlocks = 0;
        for (int s = 0; s < multiCases; s++)
        {
            int n = 2 + Rng.NextBelow(ref rng, 3);
            int m = 2 + Rng.NextBelow(ref rng, 2);
            var instances = new int[m];
            for (int j = 0; j < m; j++) instances[j] = 1 + Rng.NextBelow(ref rng, 3);
            var allocation = new int[n][];
            var max = new int[n][];
            for (int i = 0; i < n; i++)
            {
                allocation[i] = new int[m];
                max[i] = new int[m];
            }
            for (int j = 0; j < m; j++)
            {
                int left = instances[j];
                for (int i = 0; i < n && left > 0; i++)
                {
                    int give = Rng.NextBelow(ref rng, left + 1);
                    allocation[i][j] = give;
                    left -= give;
                }
                for (int i = 0; i < n; i++)
                {
                    max[i][j] = allocation[i][j] + Rng.NextBelow(ref rng, 2);
                    if (max[i][j] > instances[j]) max[i][j] = instances[j];
                    if (max[i][j] < allocation[i][j]) max[i][j] = allocation[i][j];
                }
            }
            var multi = BankerState.Create(FreeInstances(instances, allocation), allocation, max);
            Check(multi.Conserved(instances), "multi-instance state " + s + ": instances are conserved");
            bool safe = multi.Greedy().Safe;
            var graph = Rag.FromMatrices(instances, multi.Allocation, multi.Need);
            if (graph.Deadlocked())
            {
                multiDeadlocks++;
                Check(!safe, "multi-instance state " + s + ": a deadlocked graph implies an unsafe matrix state");
            }
            Check(safe == multi.BruteForceSafe(), "multi-instance state " + s + ": greedy and brute force agree");
        }
        Check(multiDeadlocks >= 5, "the multi-instance sweep found " + multiDeadlocks + " deadlocked graphs");
        Check(multiDeadlocks < multiCases, "the multi-instance sweep is not uniformly deadlocked");

        // ---- determinism ---------------------------------------------------------------------
        Safety first = state.Greedy();
        Safety second = state.Greedy();
        bool same = first.Safe == second.Safe && first.Order.Count == second.Order.Count;
        for (int i = 0; same && i < first.Order.Count; i++) if (first.Order[i] != second.Order[i]) same = false;
        Check(same, "the safety scan is deterministic");

        Console.WriteLine("textbook safe sequence   : " + string.Join(",", safety.Order));
        Console.WriteLine("single-instance sweep    : " + deadlockCases + " deadlocked of " + singleInstanceCases);
        Console.WriteLine("multi-instance deadlocks : " + multiDeadlocks + " of " + multiCases + " (all unsafe)");
        Console.WriteLine("PASS");
    }

    private static int[] FreeInstances(int[] instances, int[][] allocation)
    {
        var available = new int[instances.Length];
        for (int j = 0; j < instances.Length; j++)
        {
            int used = 0;
            for (int i = 0; i < allocation.Length; i++) used += allocation[i][j];
            available[j] = instances[j] - used;
        }
        return available;
    }

    /// Random but *valid* state: Allocation never exceeds Max, and everything allocated is accounted for
    /// in Available. `slack` bounds how much more each process may declare it needs.
    private static BankerState RandomState(ref ulong state, int n, int m, int slack)
    {
        var instances = new int[m];
        for (int j = 0; j < m; j++) instances[j] = 1 + Rng.NextBelow(ref state, 3);
        var allocation = new int[n][];
        var max = new int[n][];
        for (int i = 0; i < n; i++)
        {
            allocation[i] = new int[m];
            max[i] = new int[m];
        }
        for (int j = 0; j < m; j++)
        {
            int left = instances[j];
            for (int i = 0; i < n && left > 0; i++)
            {
                int give = Rng.NextBelow(ref state, left + 1);
                allocation[i][j] = give;
                left -= give;
            }
            for (int i = 0; i < n; i++)
            {
                int extra = Rng.NextBelow(ref state, slack + 1);
                max[i][j] = Math.Min(instances[j], allocation[i][j] + extra);
            }
        }
        return BankerState.Create(FreeInstances(instances, allocation), allocation, max);
    }
}
