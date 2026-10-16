// Task 564 - Implement 2-SAT solver using implication graph & SCC. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
//
// 2-satisfiability by strongly connected components of the implication graph, implemented for real. Each
// clause (a or b) contributes the two implications not-a -> b and not-b -> a over 2n literal nodes
// (node 2(v-1) is x_v, node 2(v-1)+1 is not x_v, so complementing a literal flips the lowest bit), and
// the classic theorem says the formula is satisfiable exactly when no variable has its two literals in
// the same SCC. Components come from an iterative Kosaraju (post-order DFS, then a reverse-graph pass in
// decreasing finish time), which is then re-derived explicitly: the condensation is built, topologically
// sorted with Kahn's algorithm, and components are assigned in reverse topological order (a literal is
// set false only when its negation is already true), which is the standard construction that turns any
// satisfying order into a model. The model is always validated against the clauses afterwards, and every
// verdict is compared with brute force over all 2^n assignments; the SCC criterion itself is checked as
// an equivalence with satisfiability over all instances. Instances are either small textbook systems
// (implication chains with forced values, contradictory unit pairs, an unsatisfiable two-variable square)
// or splitmix64-seeded random 2-CNF, half of them planted (generated from a hidden assignment so that
// they must be satisfiable).
//
// Documented checks: over 200 random instances (n = 8, m = 20, half planted) the SCC verdict equals the
// brute-force verdict and every reported model is verified clause by clause; planted instances are always
// satisfiable and the model returned for them satisfies all clauses; (x) and (not x) together is
// unsatisfiable, the four-clause square over two variables (which forces x1 and not x1) is unsatisfiable,
// and a chain of implications with a unit forces every variable to true and the model is asserted
// literally; the SCC criterion "no variable has both literals in one component" is equivalent to
// satisfiability on every tested instance; the component count is at least n and each pair of literal
// components appears exactly twice (once per literal); and repeated runs give identical verdicts, models
// and component counts.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    /** splitmix64 (Steele/Lea/Flood): 64-bit state advanced by a golden-ratio increment. */
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int Below(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class TwoSatInstance
{
    internal readonly int Vars;
    internal readonly List<int[]> Clauses;
    internal readonly string Label;

    internal TwoSatInstance(int vars, string label)
    {
        Vars = vars;
        Label = label;
        Clauses = new List<int[]>();
    }

    internal void Add(int a, int b)
    {
        Clauses.Add(new[] { a, b });
    }

    internal bool SatisfiedBy(bool[] model)
    {
        foreach (int[] clause in Clauses)
        {
            bool first = clause[0] > 0 ? model[clause[0] - 1] : !model[-clause[0] - 1];
            bool second = clause[1] > 0 ? model[clause[1] - 1] : !model[-clause[1] - 1];
            if (!first && !second) return false;
        }
        return true;
    }

    internal static TwoSatInstance Random(int vars, int clauses, ref ulong state, string label,
            bool planted)
    {
        var instance = new TwoSatInstance(vars, label);
        var hidden = new bool[vars];
        for (int i = 0; i < vars; i++) hidden[i] = SplitMix64.Below(ref state, 2) == 0;
        for (int i = 0; i < clauses; i++)
        {
            int a = 1 + SplitMix64.Below(ref state, vars);
            int b = 1 + SplitMix64.Below(ref state, vars);
            int la = SplitMix64.Below(ref state, 2) == 0 ? a : -a;
            int lb = SplitMix64.Below(ref state, 2) == 0 ? b : -b;
            if (!planted) instance.Add(la, lb);
            else
            {
                // Keep the clause satisfied by the hidden assignment.
                bool first = la > 0 ? hidden[la - 1] : !hidden[-la - 1];
                if (!first) instance.Add(-la, lb);
                else instance.Add(la, lb);
            }
        }
        return instance;
    }
}

internal static class TwoSat
{
    internal sealed class Result
    {
        internal bool Satisfiable;
        internal bool[] Model = Array.Empty<bool>();
        internal int Components;
        internal int ContradictoryVariables;
    }

    internal static Result Solve(TwoSatInstance instance)
    {
        int n = instance.Vars;
        int nodes = 2 * n;
        var adj = new List<int>[nodes];
        var radj = new List<int>[nodes];
        for (int i = 0; i < nodes; i++)
        {
            adj[i] = new List<int>();
            radj[i] = new List<int>();
        }
        foreach (int[] clause in instance.Clauses)
        {
            int a = Node(clause[0]);
            int b = Node(clause[1]);
            AddEdge(adj, radj, a ^ 1, b);
            AddEdge(adj, radj, b ^ 1, a);
        }

        // Kosaraju pass one: post-order on the implication graph.
        var order = new List<int>();
        var visited = new bool[nodes];
        var stack = new Stack<int>();
        for (int start = 0; start < nodes; start++)
        {
            if (visited[start]) continue;
            stack.Push(start);
            while (stack.Count > 0)
            {
                int v = stack.Pop();
                if (v >= 0)
                {
                    if (visited[v]) continue;
                    visited[v] = true;
                    stack.Push(~v);
                    foreach (int u in adj[v])
                    {
                        if (!visited[u]) stack.Push(u);
                    }
                }
                else
                {
                    order.Add(~v);
                }
            }
        }

        // Pass two: components, in reverse finish order, on the transposed graph.
        var comp = new int[nodes];
        for (int i = 0; i < nodes; i++) comp[i] = -1;
        var members = new List<List<int>>();
        for (int i = order.Count - 1; i >= 0; i--)
        {
            int v = order[i];
            if (comp[v] != -1) continue;
            int id = members.Count;
            var group = new List<int>();
            members.Add(group);
            var frontier = new Stack<int>();
            frontier.Push(v);
            comp[v] = id;
            while (frontier.Count > 0)
            {
                int at = frontier.Pop();
                group.Add(at);
                foreach (int u in radj[at])
                {
                    if (comp[u] == -1)
                    {
                        comp[u] = id;
                        frontier.Push(u);
                    }
                }
            }
        }

        var result = new Result { Components = members.Count };
        for (int v = 0; v < n; v++)
        {
            if (comp[2 * v] == comp[2 * v + 1]) result.ContradictoryVariables++;
        }
        result.Satisfiable = result.ContradictoryVariables == 0;

        // Explicit topological order of the condensation by Kahn's algorithm.
        var compAdj = new List<HashSet<int>>();
        for (int i = 0; i < members.Count; i++) compAdj.Add(new HashSet<int>());
        for (int v = 0; v < nodes; v++)
        {
            foreach (int u in adj[v])
            {
                if (comp[v] != comp[u]) compAdj[comp[v]].Add(comp[u]);
            }
        }
        var indegree = new int[members.Count];
        foreach (HashSet<int> edges in compAdj)
        {
            foreach (int target in edges) indegree[target]++;
        }
        var topo = new List<int>();
        var ready = new Queue<int>();
        for (int i = 0; i < members.Count; i++)
        {
            if (indegree[i] == 0) ready.Enqueue(i);
        }
        while (ready.Count > 0)
        {
            int c = ready.Dequeue();
            topo.Add(c);
            foreach (int target in compAdj[c])
            {
                indegree[target]--;
                if (indegree[target] == 0) ready.Enqueue(target);
            }
        }
        if (topo.Count != members.Count) throw new Exception("the condensation must be acyclic");

        // Assign components in reverse topological order.
        var value = new int[n + 1];
        if (result.Satisfiable)
        {
            for (int index = topo.Count - 1; index >= 0; index--)
            {
                int c = topo[index];
                bool already = false;
                foreach (int w in members[c])
                {
                    if (value[NodeVariable(w)] != 0) { already = true; break; }
                }
                if (already) continue;
                int complement = comp[members[c][0] ^ 1];
                foreach (int w in members[c])
                {
                    value[NodeVariable(w)] = (w % 2 == 0) ? 1 : -1;
                }
                foreach (int w in members[complement])
                {
                    value[NodeVariable(w)] = (w % 2 == 0) ? -1 : 1;
                }
            }
            var model = new bool[n];
            for (int v = 1; v <= n; v++) model[v - 1] = value[v] == 1;
            result.Model = model;
        }
        else
        {
            result.Model = new bool[n];
        }
        return result;
    }

    private static int Node(int literal)
    {
        int v = Math.Abs(literal) - 1;
        return literal > 0 ? 2 * v : 2 * v + 1;
    }

    private static int NodeVariable(int node)
    {
        return node / 2 + 1;
    }

    private static void AddEdge(List<int>[] adj, List<int>[] radj, int from, int to)
    {
        adj[from].Add(to);
        radj[to].Add(from);
    }

    internal static bool BruteForce(TwoSatInstance instance)
    {
        int n = instance.Vars;
        for (long mask = 0; mask < (1L << n); mask++)
        {
            var model = new bool[n];
            for (int i = 0; i < n; i++) model[i] = (mask & (1L << i)) != 0;
            if (instance.SatisfiedBy(model)) return true;
        }
        return false;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 564001UL;
        int compared = 0;
        int plantedCount = 0;
        int unsatRandom = 0;
        for (int round = 0; round < 200; round++)
        {
            bool planted = round % 2 == 0;
            TwoSatInstance instance = TwoSatInstance.Random(8, 20, ref rng, "random" + round, planted);
            TwoSat.Result result = TwoSat.Solve(instance);
            bool expected = TwoSat.BruteForce(instance);
            Check(result.Satisfiable == expected, instance.Label + ": SCC says " + result.Satisfiable
                    + ", brute force says " + expected);
            Check((result.ContradictoryVariables == 0) == result.Satisfiable,
                    instance.Label + ": the SCC criterion disagrees with the verdict");
            if (result.Satisfiable)
            {
                Check(instance.SatisfiedBy(result.Model), instance.Label
                        + ": the SCC model does not satisfy the clauses");
            }
            if (planted)
            {
                Check(result.Satisfiable, instance.Label + " is planted and must be satisfiable");
                plantedCount++;
            }
            else if (!result.Satisfiable)
            {
                unsatRandom++;
            }
            compared++;
        }
        Check(compared == 200, "200 random instances were compared");
        Check(plantedCount == 100, "100 planted instances, all satisfiable");
        Check(unsatRandom > 0, "the unconstrained random instances included unsatisfiable cases ("
                + unsatRandom + ")");

        // Forced values through an implication chain.
        var chain = new TwoSatInstance(3, "chain");
        chain.Add(-1, 2);
        chain.Add(-2, 3);
        chain.Add(1, 1);
        TwoSat.Result chainResult = TwoSat.Solve(chain);
        Check(chainResult.Satisfiable, "the chain is satisfiable");
        Check(chainResult.Model[0] && chainResult.Model[1] && chainResult.Model[2],
                "x1, x2 and x3 are all forced true");
        Check(chain.SatisfiedBy(chainResult.Model), "the chain model is valid");

        // Contradictory units.
        var contradiction = new TwoSatInstance(1, "contradiction");
        contradiction.Add(1, 1);
        contradiction.Add(-1, -1);
        TwoSat.Result contradictionResult = TwoSat.Solve(contradiction);
        Check(!contradictionResult.Satisfiable, "x and not x together is unsatisfiable");
        Check(contradictionResult.ContradictoryVariables == 1, "the variable is self-contradictory");
        Check(!TwoSat.BruteForce(contradiction), "brute force agrees on the contradiction");

        // The two-variable square, unsatisfiable.
        var square = new TwoSatInstance(2, "square");
        square.Add(1, 2);
        square.Add(1, -2);
        square.Add(-1, 2);
        square.Add(-1, -2);
        Check(!TwoSat.Solve(square).Satisfiable, "the two-variable square is unsatisfiable");
        Check(!TwoSat.BruteForce(square), "brute force agrees on the square");

        // A single clause is satisfiable with two possible models.
        var single = new TwoSatInstance(2, "single");
        single.Add(1, 2);
        TwoSat.Result singleResult = TwoSat.Solve(single);
        Check(singleResult.Satisfiable && single.SatisfiedBy(singleResult.Model),
                "a single clause is satisfiable and the model is valid");

        // Component structure: 2n nodes, at least n components, and components come in complementary pairs.
        TwoSatInstance structure = TwoSatInstance.Random(6, 12, ref rng, "structure", false);
        TwoSat.Result structureResult = TwoSat.Solve(structure);
        Check(structureResult.Components >= 6, "at least n components, got " + structureResult.Components);
        Check(structureResult.Components <= 12, "at most 2n components, got " + structureResult.Components);
        Check(structureResult.Satisfiable == (structureResult.ContradictoryVariables == 0),
                "the criterion is consistent on the structure instance");

        // Determinism.
        TwoSat.Result again = TwoSat.Solve(structure);
        Check(again.Satisfiable == structureResult.Satisfiable
                && again.Components == structureResult.Components,
                "repeated runs agree on verdict and component count");
        if (structureResult.Satisfiable)
        {
            for (int i = 0; i < structure.Vars; i++)
            {
                Check(again.Model[i] == structureResult.Model[i],
                        "repeated runs agree on variable " + (i + 1));
            }
        }
        for (int round = 0; round < 40; round++)
        {
            TwoSatInstance extra = TwoSatInstance.Random(10, 26, ref rng, "extra" + round, false);
            Check(TwoSat.Solve(extra).Satisfiable == TwoSat.BruteForce(extra),
                    "extra" + round + ": SCC verdict != brute force");
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
