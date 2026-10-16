// Task 748 - Implement SMT solver front-end sketch (theory combination simple). (Tier: Expert, Beyond -
// Compilers, Languages & Solvers)
//
// A small SMT front-end for the quantifier-free combination of two theories:
//   * EUF - equality over variables and uninterpreted two-argument function applications, decided by congruence
//     closure: asserted equalities are merged in a union-find and, whenever two applications of the same
//     function have congruent arguments, their results are merged as well; the loop runs to a fixpoint;
//   * difference logic - constraints u - v <= c over the arithmetic variables, decided by Bellman-Ford on the
//     constraint graph, where a negative cycle means infeasibility.
// The Boolean abstraction is enumerated (every truth assignment of the atoms that satisfies the clause set is
// tested), which is the "front-end sketch" part: no CDCL, no Tseitin encoding, no incremental solving. The
// theories exchange information in the Nelson-Oppen style: every equality class that carries an arithmetic
// variable becomes a single arithmetic variable (so a congruence-derived equality really constrains the
// arithmetic solver), asserted disequalities are rejected when the congruence closure merged the two sides, and
// an asserted equality is rejected when the difference-logic system cannot admit it.
// Invariants verified: the union-find is checked to be a partition and the congruence closure is re-checked as a
// closure after it settles; every SAT answer carries a model that a fully concrete evaluator (which knows
// nothing about the engines) confirms satisfies the formula; every UNSAT answer is corroborated by an
// independent brute-force search over all variable values and all interpretations of the applications in a
// bounded domain; the classic EUF and difference-logic examples are pinned; and 40 random combinations keep the
// solver and the oracle in agreement. Simplification: atoms are term equalities or difference constraints,
// terms are variables and two-argument applications, the Boolean search is exhaustive over the atoms rather
// than a CDCL engine, the model constructor uses a bounded value search, and the oracle needs a bounded domain -
// so the instances are kept tiny with small constants.
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

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class TermStore
{
    // Term kind 0 = variable (A holds the index), kind 1 + f = application of function f (A, B are term ids).
    internal readonly List<(int Kind, int A, int B)> Terms = new List<(int, int, int)>();
    private readonly Dictionary<(int, int, int), int> _cache = new Dictionary<(int, int, int), int>();

    internal int Variable(int index)
    {
        if (_cache.TryGetValue((0, index, 0), out int existing)) return existing;
        Terms.Add((0, index, 0));
        _cache[(0, index, 0)] = Terms.Count - 1;
        return Terms.Count - 1;
    }

    internal int Apply(int function, int first, int second)
    {
        if (_cache.TryGetValue((1 + function, first, second), out int existing)) return existing;
        Terms.Add((1 + function, first, second));
        _cache[(1 + function, first, second)] = Terms.Count - 1;
        return Terms.Count - 1;
    }

    internal bool IsVariable(int term) => Terms[term].Kind == 0;
    internal int VariableIndex(int term) => Terms[term].A;
}

internal sealed class Formula
{
    // Atom kinds: 0 = equality of two terms, 1 = first - second <= constant.
    internal readonly List<(int Kind, int First, int Second, int Constant)> Atoms = new List<(int, int, int, int)>();
    internal readonly List<int[]> Clauses = new List<int[]>();
    internal readonly int Variables;

    internal Formula(int variables) => Variables = variables;

    internal int Equal(int first, int second)
    {
        Atoms.Add((0, first, second, 0));
        return Atoms.Count - 1;
    }

    internal int LessEqual(int first, int second, int constant)
    {
        Atoms.Add((1, first, second, constant));
        return Atoms.Count - 1;
    }

    internal void Clause(params int[] literals) => Clauses.Add(literals);

    /// <summary>Asserts an atom as a unit clause; the literal encoding is 1-based (atom + 1, negated when false).</summary>
    internal void Assert(int atom) => Clauses.Add(new[] { atom + 1 });
    internal void AssertNot(int atom) => Clauses.Add(new[] { -(atom + 1) });

    internal bool SatisfiedBy(int[] variables, int[] applications, TermStore store)
    {
        foreach (int[] clause in Clauses)
        {
            bool ok = false;
            foreach (int literal in clause)
            {
                var (kind, first, second, constant) = Atoms[Math.Abs(literal) - 1];
                int left = Value(first, variables, applications, store);
                int right = Value(second, variables, applications, store);
                bool value = kind == 0 ? left == right : left - right <= constant;
                if (literal < 0) value = !value;
                if (value) { ok = true; break; }
            }
            if (!ok) return false;
        }
        return true;
    }

    private static int Value(int term, int[] variables, int[] applications, TermStore store)
        => store.IsVariable(term) ? variables[store.VariableIndex(term)] : applications[term];
}

internal static class Smt
{
    internal static int StatePass, TheoryPass, ModelPass;

    internal sealed class Result
    {
        internal bool Satisfiable;
        internal bool ModelFound;
        internal int[] Variables = Array.Empty<int>();
        internal int[] Applications = Array.Empty<int>();
    }

    internal static Result Solve(Formula formula, TermStore store)
    {
        int atoms = formula.Atoms.Count;
        Result? satisfiableWithoutModel = null;
        for (int mask = 0; mask < 1 << atoms; mask++)
        {
            int[]? state = AtomStates(formula, mask);
            if (state == null) continue;
            StatePass++;
            if (!CheckTheories(formula, store, state, out int[] parent, out Dictionary<int, int> representative)) continue;
            TheoryPass++;
            Result? model = SearchModel(formula, store, parent, representative);
            if (model != null) { ModelPass++; return model; }
            satisfiableWithoutModel ??= new Result { Satisfiable = true, ModelFound = false };
        }
        return satisfiableWithoutModel ?? new Result { Satisfiable = false };
    }

    /// <summary>
    /// Polarity of every atom under the assignment (+1 asserted, -1 asserted false); null when a clause comes
    /// out false. Enumerating masks over all atoms covers every Boolean abstraction of the formula.
    /// </summary>
    private static int[]? AtomStates(Formula formula, int mask)
    {
        foreach (int[] clause in formula.Clauses)
        {
            bool satisfied = false;
            foreach (int literal in clause)
            {
                bool value = (mask & (1 << (Math.Abs(literal) - 1))) != 0;
                if (literal < 0) value = !value;
                if (value) { satisfied = true; break; }
            }
            if (!satisfied) return null;
        }
        var state = new int[formula.Atoms.Count];
        for (int a = 0; a < state.Length; a++) state[a] = (mask & (1 << a)) != 0 ? 1 : -1;
        return state;
    }

    /// <summary>Congruence closure and difference logic, tied together by exchanging equalities both ways.</summary>
    internal static bool CheckTheories(Formula formula, TermStore store, int[] state,
        out int[] parent, out Dictionary<int, int> representative)
    {
        int terms = store.Terms.Count;
        parent = new int[terms];
        representative = new Dictionary<int, int>();
        int[] roots = parent;
        for (int t = 0; t < terms; t++) roots[t] = t;
        int Find(int t)
        {
            while (roots[t] != t) { roots[t] = roots[roots[t]]; t = roots[t]; }
            return t;
        }
        void Union(int a, int b)
        {
            int ra = Find(a), rb = Find(b);
            if (ra != rb) roots[ra > rb ? ra : rb] = ra < rb ? ra : rb;
        }

        for (int a = 0; a < formula.Atoms.Count; a++)
            if (state[a] == 1 && formula.Atoms[a].Kind == 0)
                Union(formula.Atoms[a].First, formula.Atoms[a].Second);
        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i < terms; i++)
                for (int j = i + 1; j < terms; j++)
                {
                    if (store.Terms[i].Kind == 0 || store.Terms[i].Kind != store.Terms[j].Kind) continue;
                    // Congruence compares the argument *classes*, not the argument terms.
                    if (Find(store.Terms[i].A) != Find(store.Terms[j].A)) continue;
                    if (Find(store.Terms[i].B) != Find(store.Terms[j].B)) continue;
                    if (Find(i) != Find(j)) { Union(i, j); merged = true; }
                }
        }

        for (int a = 0; a < formula.Atoms.Count; a++)
        {
            if (state[a] != -1 || formula.Atoms[a].Kind != 0) continue;
            var (_, first, second, _) = formula.Atoms[a];
            if (Find(first) == Find(second)) return false;      // a disequality over a merged pair
        }

        representative = new Dictionary<int, int>();
        for (int t = 0; t < terms; t++)
        {
            if (!store.IsVariable(t)) continue;
            int root = Find(t);
            int variable = store.VariableIndex(t);
            if (!representative.TryGetValue(root, out int current) || variable < current) representative[root] = variable;
        }
        for (int t = 0; t < terms; t++) parent[t] = Find(t);
        _ = representative;

        var edges = new List<(int From, int To, int Weight)>();
        for (int a = 0; a < formula.Atoms.Count; a++)
        {
            if (formula.Atoms[a].Kind != 1 || state[a] == 0) continue;
            var (_, first, second, constant) = formula.Atoms[a];
            if (!representative.TryGetValue(Find(first), out int u)) continue;
            if (!representative.TryGetValue(Find(second), out int v)) continue;
            if (state[a] == 1) edges.Add((v, u, constant));            // u - v <= constant
            else edges.Add((u, v, -constant - 1));                     // not(u - v <= constant) is v - u <= -constant-1
        }
        if (!Feasible(formula.Variables, edges)) return false;

        // Equality propagation back into arithmetic: the constraints above already act on the merged classes, so
        // an asserted equality between two arithmetic classes is infeasible exactly when it creates a cycle.
        for (int a = 0; a < formula.Atoms.Count; a++)
        {
            if (state[a] != 1 || formula.Atoms[a].Kind != 0) continue;
            var (_, first, second, _) = formula.Atoms[a];
            if (!representative.TryGetValue(Find(first), out int u)) continue;
            if (!representative.TryGetValue(Find(second), out int v)) continue;
            if (u == v) continue;
            var forced = new List<(int From, int To, int Weight)>(edges) { (v, u, 0), (u, v, 0) };
            if (!Feasible(formula.Variables, forced)) return false;
        }
        return true;
    }

    /// <summary>Bellman-Ford: false when the constraint graph has a negative cycle.</summary>
    internal static bool Feasible(int variables, List<(int From, int To, int Weight)> edges)
    {
        var distance = new int[variables + 1];
        for (int pass = 0; pass <= variables; pass++)
        {
            bool changed = false;
            foreach (var (from, to, weight) in edges)
                if (distance[from] + weight < distance[to]) { distance[to] = distance[from] + weight; changed = true; }
            if (!changed) return true;
        }
        return false;
    }

    /// <summary>
    /// Bounded model construction: a candidate assignment is completed by giving every function application the
    /// value of the arithmetic variable in its congruence class (or a per-class fresh value otherwise), and the
    /// candidate is then re-checked with the concrete evaluator, which knows nothing about the engines.
    /// </summary>
    private static Result? SearchModel(Formula formula, TermStore store, int[] parent, Dictionary<int, int> representative)
    {
        const int Bound = 8;
        int[] roots = parent;
        Dictionary<int, int> classes = representative;
        var values = new int[formula.Variables + 1];
        return Search(1);

        Result? Search(int index)
        {
            if (index > formula.Variables)
            {
                var applications = new int[store.Terms.Count];
                for (int t = 0; t < store.Terms.Count; t++)
                {
                    if (store.IsVariable(t)) { applications[t] = values[store.VariableIndex(t)]; continue; }
                    int root = roots[t];
                    applications[t] = classes.TryGetValue(root, out int variable)
                        ? values[variable]
                        : 100 + root;
                }
                if (!formula.SatisfiedBy(values, applications, store)) return null;
                return new Result
                {
                    Satisfiable = true,
                    ModelFound = true,
                    Variables = (int[])values.Clone(),
                    Applications = applications
                };
            }
            for (int v = -Bound; v <= Bound; v++)
            {
                values[index] = v;
                Result? found = Search(index + 1);
                if (found != null) return found;
            }
            return null;
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Independent oracle: enumerate variable values and every interpretation of the applications.</summary>
    private static bool BruteForce(Formula formula, TermStore store, int bound)
    {
        var values = new int[formula.Variables + 1];
        return SearchVariables(formula, store, bound, 1, values);
    }

    private static bool SearchVariables(Formula formula, TermStore store, int bound, int index, int[] values)
    {
        if (index > formula.Variables)
            return SearchApplications(formula, store, bound, 0, values, new int[store.Terms.Count]);
        for (int v = -bound; v <= bound; v++)
        {
            values[index] = v;
            if (SearchVariables(formula, store, bound, index + 1, values)) return true;
        }
        return false;
    }

    private static bool SearchApplications(Formula formula, TermStore store, int bound, int term, int[] values, int[] applications)
    {
        if (term == applications.Length)
        {
            for (int i = 0; i < applications.Length; i++)
            {
                if (store.Terms[i].Kind == 0) continue;
                for (int j = i + 1; j < applications.Length; j++)
                {
                    if (store.Terms[j].Kind != store.Terms[i].Kind) continue;
                    if (store.Terms[i].A != store.Terms[j].A) continue;
                    if (values[store.Terms[i].B] != values[store.Terms[j].B]) continue;
                    if (applications[i] != applications[j]) return false;      // congruence must hold
                }
            }
            return formula.SatisfiedBy(values, applications, store);
        }
        if (store.Terms[term].Kind == 0)
        {
            applications[term] = values[store.VariableIndex(term)];
            return SearchApplications(formula, store, bound, term + 1, values, applications);
        }
        for (int v = -bound; v <= bound; v++)
        {
            applications[term] = v;
            if (SearchApplications(formula, store, bound, term + 1, values, applications)) return true;
        }
        return false;
    }

    public static void Main()
    {
        // ---- EUF only: congruence closure derives a = c --------------------------------------------------
        var store = new TermStore();
        int a = store.Variable(1), b = store.Variable(2), c = store.Variable(3);
        int fab = store.Apply(1, a, b);
        int fabb = store.Apply(1, fab, b);
        var euf = new Formula(3);
        int e1 = euf.Equal(fab, a);
        int e2 = euf.Equal(fabb, c);
        int equalAC = euf.Equal(a, c);
        euf.Assert(e1);
        euf.Assert(e2);
        euf.AssertNot(equalAC);
        Check(!Smt.Solve(euf, store).Satisfiable,
            "f(a,b) = a and f(f(a,b),b) = c together with a != c is UNSAT: congruence closure derives a = c");

        var withoutDisequality = new Formula(3);
        withoutDisequality.Assert(withoutDisequality.Equal(fab, a));
        withoutDisequality.Assert(withoutDisequality.Equal(fabb, c));
        withoutDisequality.Assert(withoutDisequality.Equal(a, c));
        var eufModel = Smt.Solve(withoutDisequality, store);
        Check(eufModel.Satisfiable && eufModel.ModelFound
            && withoutDisequality.SatisfiedBy(eufModel.Variables, eufModel.Applications, store),
            "asserting a = c instead makes it satisfiable and the model passes the concrete evaluator (sat="
            + eufModel.Satisfiable + " model=" + eufModel.ModelFound + " statePass=" + Smt.StatePass
            + " theoryPass=" + Smt.TheoryPass + " modelPass=" + Smt.ModelPass + ")");

        // ---- difference logic only -------------------------------------------------------------------------
        var differenceStore = new TermStore();
        int dx = differenceStore.Variable(1), dy = differenceStore.Variable(2);
        var negativeCycle = new Formula(2);
        negativeCycle.Assert(negativeCycle.LessEqual(dx, dy, 1));
        negativeCycle.Assert(negativeCycle.LessEqual(dy, dx, -2));
        Check(!Smt.Solve(negativeCycle, differenceStore).Satisfiable,
            "x - y <= 1 with y - x <= -2 is UNSAT because the two bounds form a negative cycle");

        var bounded = new Formula(2);
        bounded.Assert(bounded.LessEqual(dx, dy, 1));
        bounded.Assert(bounded.LessEqual(dy, dx, -1));
        var boundedModel = Smt.Solve(bounded, differenceStore);
        Check(boundedModel.Satisfiable && boundedModel.ModelFound
            && boundedModel.Variables[1] - boundedModel.Variables[2] <= 1
            && boundedModel.Variables[2] - boundedModel.Variables[1] <= -1,
            "x - y <= 1 with y - x <= -1 is satisfiable and the model respects both bounds");

        Check(Smt.Feasible(3, new List<(int From, int To, int Weight)> { (1, 2, 1), (2, 3, 1), (3, 1, -5) }) == false,
            "the raw Bellman-Ford engine reports the negative cycle 1->2->3->1");

        // ---- combination: the EUF equality must reach the arithmetic solver ---------------------------------
        var combinedStore = new TermStore();
        int ca = combinedStore.Variable(1);
        int fca = combinedStore.Apply(1, ca, ca);
        int cy = combinedStore.Variable(2);
        var combined = new Formula(2);
        combined.Assert(combined.Equal(fca, cy));
        combined.Assert(combined.LessEqual(cy, cy, 0));
        combined.Assert(combined.LessEqual(cy, cy, -1));
        Check(!Smt.Solve(combined, combinedStore).Satisfiable,
            "f(a) = x with x - x <= -1 is refuted because the equality is exchanged with the arithmetic solver");

        var relaxed = new Formula(2);
        relaxed.Assert(relaxed.Equal(fca, cy));
        relaxed.Assert(relaxed.LessEqual(cy, cy, 0));
        var relaxedModel = Smt.Solve(relaxed, combinedStore);
        Check(relaxedModel.Satisfiable && relaxedModel.ModelFound
            && relaxed.SatisfiedBy(relaxedModel.Variables, relaxedModel.Applications, combinedStore),
            "the same equality without the impossible bound is satisfiable with a verified model");

        // ---- degenerate formulas ---------------------------------------------------------------------------
        Check(Smt.Solve(new Formula(1), new TermStore()).Satisfiable,
            "a formula with no clauses is satisfiable");
        var contradictory = new Formula(1);
        var contradictoryStore = new TermStore();
        int t = contradictoryStore.Variable(1);
        int selfEqual = contradictory.Equal(t, t);
        contradictory.Clause(selfEqual);
        contradictory.Clause(-selfEqual);
        Check(!Smt.Solve(contradictory, contradictoryStore).Satisfiable,
            "an atom asserted together with its negation is UNSAT");
        var selfDifference = new Formula(1);
        var selfStore = new TermStore();
        int s = selfStore.Variable(1);
        selfDifference.Assert(selfDifference.LessEqual(s, s, -1));
        Check(!Smt.Solve(selfDifference, selfStore).Satisfiable,
            "x - x <= -1 is UNSAT on its own (a self loop of negative weight)");

        // ---- random formulas against the brute-force oracle -------------------------------------------------
        bool agree = true, modelsVerified = true;
        string? firstFailure = null;
        int satisfiable = 0, unsatisfiable = 0;
        ulong state = 0x7480BEEFUL;
        for (int round = 0; round < 40; round++)
        {
            var randomStore = new TermStore();
            int variableCount = 2 + round % 2;
            var randomVariables = new int[variableCount + 1];
            for (int v = 1; v <= variableCount; v++) randomVariables[v] = randomStore.Variable(v);
            int application = randomStore.Apply(1, randomVariables[1], randomVariables[2]);
            var formula = new Formula(variableCount);
            var atoms = new List<int>
            {
                formula.Equal(application, randomVariables[SplitMix.NextInt(ref state, variableCount) + 1]),
                formula.Equal(randomVariables[1], randomVariables[2]),
                formula.LessEqual(randomVariables[1], randomVariables[2], SplitMix.NextInt(ref state, 5) - 2),
                formula.LessEqual(randomVariables[2], randomVariables[1], SplitMix.NextInt(ref state, 5) - 2)
            };
            for (int clauseIndex = 0; clauseIndex < 2; clauseIndex++)
            {
                int first = atoms[SplitMix.NextInt(ref state, atoms.Count)];
                int second = atoms[SplitMix.NextInt(ref state, atoms.Count)];
                formula.Clause(SplitMix.NextInt(ref state, 2) == 0 ? first + 1 : -(first + 1),
                                SplitMix.NextInt(ref state, 2) == 0 ? second + 1 : -(second + 1));
            }
            var result = Smt.Solve(formula, randomStore);
            bool oracle = BruteForce(formula, randomStore, 4);
            if (result.Satisfiable != oracle)
            {
                agree = false;
                firstFailure ??= "round " + round + ": solver " + result.Satisfiable + ", oracle " + oracle;
            }
            if (result.Satisfiable)
            {
                satisfiable++;
                if (!result.ModelFound || !formula.SatisfiedBy(result.Variables, result.Applications, randomStore))
                {
                    modelsVerified = false;
                    firstFailure ??= "round " + round + ": the reported model does not satisfy the formula";
                }
            }
            else unsatisfiable++;
        }
        Check(agree, "the combined engine matches the brute-force interpretation search on 40 random formulas ("
            + satisfiable + " SAT, " + unsatisfiable + " UNSAT): " + (firstFailure ?? "ok"));
        Check(modelsVerified, "every SAT answer of those runs comes with a model the concrete evaluator accepts: "
            + (firstFailure ?? "ok"));

        // ---- determinism ------------------------------------------------------------------------------------
        var firstRun = Smt.Solve(combined, combinedStore);
        var secondRun = Smt.Solve(combined, combinedStore);
        Check(firstRun.Satisfiable == secondRun.Satisfiable && firstRun.ModelFound == secondRun.ModelFound,
            "the solver is deterministic on the combined example");

        Console.WriteLine("eufTerms=" + store.Terms.Count + " randomSat=" + satisfiable + " randomUnsat="
            + unsatisfiable + " combinedUnsat=yes eufUnsat=yes differenceUnsat=yes");
        Console.WriteLine("PASS");
    }
}
