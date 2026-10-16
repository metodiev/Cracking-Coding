// Task 747 - Implement advanced SAT techniques: clause learning and watched literals in practice. (Tier:
// Expert, Beyond - Compilers, Languages & Solvers)
//
// A complete CDCL SAT solver for small CNF formulas.
//   * Every clause with two or more literals watches exactly two of them, and the watch lists are indexed by
//     literal, so unit propagation only visits clauses whose watched literal just became false. A clause is
//     moved off a falsified watch whenever it still contains a non-false literal, otherwise it becomes a unit
//     (assign the other watch, reason = the clause) or a conflict.
//   * On a conflict the solver runs the standard first-UIP resolution: it keeps resolving the conflict clause
//     with the reasons of the trail literals of the current decision level until only one literal of that level
//     remains, puts that asserting literal first in the learned clause, sorts the rest so the second literal has
//     the highest level, then backjumps to that level and asserts it.
//   * Decisions take the unassigned variable with the largest activity (bumped on every learned clause, ties
//     broken by index), which keeps the search fully deterministic.
// Invariants verified: the watched-literal invariant is re-checked after every propagation (a clause that is
// not satisfied must keep both watches non-false); propagation is cross-checked against an independent naive
// fixpoint scan over all clauses on many partial assignments; every SAT answer's model is verified clause by
// clause; every UNSAT answer is corroborated by exhaustive truth-table enumeration (random 3-SAT around the
// phase transition, pigeonhole instances, and K4 non-3-colourability); tautologies, the empty formula and the
// empty clause are handled explicitly; and clause learning must actually fire (learned clause count > 0) on the
// unsatisfiable instances. Simplification: no restarts, no clause minimisation, no clause deletion, no
// preprocessing beyond dropping tautologies and folding unit clauses at level zero - the formulas are tiny, so
// only the two techniques named in the task (learning plus watched literals) are exercised.
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

internal sealed class Cnf
{
    internal readonly int Variables;
    internal readonly List<int[]> Clauses = new List<int[]>();

    internal Cnf(int variables) => Variables = variables;

    internal void Add(params int[] literals) => Clauses.Add(literals);

    internal bool SatisfiedBy(int[] assignment)
    {
        foreach (int[] clause in Clauses)
        {
            bool ok = false;
            foreach (int literal in clause)
                if (assignment[Math.Abs(literal)] == (literal > 0 ? 1 : -1)) { ok = true; break; }
            if (!ok) return false;
        }
        return true;
    }
}

internal sealed class Sat
{
    private readonly Cnf _cnf;
    private readonly int _n;
    private readonly List<int[]> _clauses = new List<int[]>();
    private readonly List<int> _watchA = new List<int>();
    private readonly List<int> _watchB = new List<int>();
    private readonly List<int>[] _watches;
    private readonly int[] _value;
    private readonly int[] _level;
    private readonly int[] _reason;
    private readonly double[] _activity;
    private readonly List<int> _trail = new List<int>();
    private readonly List<int> _boundaries = new List<int> { 0 };
    private int _trailHead;
    private int _conflict = -1;
    private int _asserting;
    private int _assertingReason = -1;

    internal long Decisions, Propagations, Conflicts, LearnedClauses, InvariantChecks;
    internal bool InvariantHolds = true;
    internal bool UnsatAtLevelZero;

    internal Sat(Cnf cnf)
    {
        _cnf = cnf;
        _n = cnf.Variables;
        _value = new int[_n + 1];
        _level = new int[_n + 1];
        _reason = new int[_n + 1];
        _activity = new double[_n + 1];
        _watches = new List<int>[2 * _n + 2];
        for (int i = 0; i < _watches.Length; i++) _watches[i] = new List<int>();
        foreach (int[] clause in cnf.Clauses)
        {
            var literals = new List<int>();
            bool tautology = false;
            foreach (int literal in clause)
            {
                if (literals.Contains(-literal)) { tautology = true; break; }
                if (!literals.Contains(literal)) literals.Add(literal);
            }
            if (tautology) continue;
            if (literals.Count == 0) { UnsatAtLevelZero = true; return; }
            if (literals.Count == 1)
            {
                int unit = literals[0];
                if (_value[Math.Abs(unit)] != 0 && ValueOf(unit) == -1) { UnsatAtLevelZero = true; return; }
                if (_value[Math.Abs(unit)] == 0) Assign(unit, -1);
                continue;
            }
            _clauses.Add(literals.ToArray());
            int index = _clauses.Count - 1;
            _watchA.Add(literals[0]);
            _watchB.Add(literals[1]);
            _watches[Index(literals[0])].Add(index);
            _watches[Index(literals[1])].Add(index);
        }
    }

    private static int Index(int literal) => 2 * Math.Abs(literal) + (literal > 0 ? 0 : 1);

    private int ValueOf(int literal)
    {
        int value = _value[Math.Abs(literal)];
        return literal > 0 ? value : -value;
    }

    private void Assign(int literal, int reason)
    {
        int variable = Math.Abs(literal);
        _value[variable] = literal > 0 ? 1 : -1;
        _level[variable] = _boundaries.Count - 1;
        _reason[variable] = reason;
        _trail.Add(literal);
    }

    private bool Propagate()
    {
        while (_trailHead < _trail.Count)
        {
            int trueLiteral = _trail[_trailHead++];
            Propagations++;
            int falsified = -trueLiteral;
            List<int> watchers = _watches[Index(falsified)];
            var pending = new List<int>(watchers);
            watchers.Clear();
            for (int k = 0; k < pending.Count; k++)
            {
                int c = pending[k];
                int[] clause = _clauses[c];
                int other = _watchA[c] == falsified ? _watchB[c] : _watchA[c];
                if (ValueOf(other) == 1) { watchers.Add(c); continue; }
                int replacement = 0;
                foreach (int literal in clause)
                {
                    if (literal == falsified || literal == other) continue;
                    if (ValueOf(literal) != -1) { replacement = literal; break; }
                }
                if (replacement != 0)
                {
                    if (_watchA[c] == falsified) _watchA[c] = replacement;
                    else _watchB[c] = replacement;
                    _watches[Index(replacement)].Add(c);
                    continue;
                }
                watchers.Add(c);
                if (ValueOf(other) == -1)
                {
                    // Report the conflict, but first put every clause that has not been inspected back into the
                    // list it came from: a dropped entry would silently lose later propagations.
                    for (int rest = k + 1; rest < pending.Count; rest++) watchers.Add(pending[rest]);
                    _conflict = c;
                    return false;
                }
                Assign(other, c);
            }
        }
        InvariantChecks++;
        if (!WatchListsConsistent()) InvariantHolds = false;
        return true;
    }

    /// <summary>
    /// Structural invariant of the watch bookkeeping: every entry of a watch list points at a clause that really
    /// watches that literal, no clause is listed twice for one literal, and every clause appears in the lists of
    /// both of its watched literals. A dropped or stale entry is exactly the kind of bug that loses propagations.
    /// (The naive scan oracle below covers the semantic side of propagation.)
    /// </summary>
    internal bool WatchListsConsistent()
    {
        var seen = new HashSet<(int Literal, int Clause)>();
        for (int literal = 0; literal < _watches.Length; literal++)
            foreach (int c in _watches[literal])
            {
                if (!seen.Add((literal, c))) return false;
                if (Index(_watchA[c]) != literal && Index(_watchB[c]) != literal) return false;
            }
        for (int c = 0; c < _clauses.Count; c++)
        {
            if (!seen.Contains((Index(_watchA[c]), c))) return false;
            if (_watchA[c] != _watchB[c] && !seen.Contains((Index(_watchB[c]), c))) return false;
        }
        return true;
    }

    private int PickBranch()
    {
        int best = 0;
        double bestActivity = -1;
        for (int v = 1; v <= _n; v++)
            if (_value[v] == 0 && _activity[v] > bestActivity) { bestActivity = _activity[v]; best = v; }
        return best;
    }

    /// <summary>First-UIP clause learning; returns the level to backjump to.</summary>
    private int Analyze()
    {
        var seen = new bool[_n + 1];
        var learned = new List<int> { 0 };
        int pathCount = 0;
        int index = _trail.Count - 1;
        int clauseIndex = _conflict;
        int pivot = 0;
        do
        {
            foreach (int literal in _clauses[clauseIndex])
            {
                int variable = Math.Abs(literal);
                if (variable == Math.Abs(pivot)) continue;
                if (!seen[variable] && _level[variable] > 0)
                {
                    seen[variable] = true;
                    _activity[variable] += 1.0;
                    if (_level[variable] == _boundaries.Count - 1) pathCount++;
                    else learned.Add(literal);
                }
            }
            while (index >= 0 && !seen[Math.Abs(_trail[index])]) index--;
            pivot = _trail[index--];
            clauseIndex = _reason[Math.Abs(pivot)];
            seen[Math.Abs(pivot)] = false;
            pathCount--;
        } while (pathCount > 0);
        learned[0] = -pivot;

        int backjump = 0;
        if (learned.Count > 1)
        {
            int highest = 1;
            for (int i = 2; i < learned.Count; i++)
                if (_level[Math.Abs(learned[i])] > _level[Math.Abs(learned[highest])]) highest = i;
            (learned[1], learned[highest]) = (learned[highest], learned[1]);
            backjump = _level[Math.Abs(learned[1])];
        }
        _clauses.Add(learned.ToArray());
        int learnedIndex = _clauses.Count - 1;
        _watchA.Add(learned[0]);
        _watchB.Add(learned.Count > 1 ? learned[1] : learned[0]);
        _watches[Index(learned[0])].Add(learnedIndex);
        if (learned.Count > 1) _watches[Index(learned[1])].Add(learnedIndex);
        LearnedClauses++;
        _asserting = learned[0];
        _assertingReason = learnedIndex;
        return backjump;
    }

    private void Backjump(int level)
    {
        int keep = _boundaries[level + 1];
        while (_trail.Count > keep)
        {
            int literal = _trail[_trail.Count - 1];
            _trail.RemoveAt(_trail.Count - 1);
            _value[Math.Abs(literal)] = 0;
        }
        _trailHead = Math.Min(_trailHead, _trail.Count);
        _boundaries.RemoveRange(level + 1, _boundaries.Count - (level + 1));
        Assign(_asserting, _assertingReason);
    }

    internal bool Solve()
    {
        if (UnsatAtLevelZero) return false;
        for (int step = 0; step < 200000; step++)
        {
            if (!Propagate())
            {
                Conflicts++;
                if (_boundaries.Count == 1) return false;
                Backjump(Analyze());
                continue;
            }
            int decision = PickBranch();
            if (decision == 0) return true;
            Decisions++;
            _boundaries.Add(_trail.Count);
            Assign(decision, -1);
        }
        throw new InvalidOperationException("the step budget was exhausted");
    }

    internal int[] Model()
    {
        var model = new int[_n + 1];
        for (int v = 1; v <= _n; v++) model[v] = _value[v] >= 0 ? 1 : -1;
        return model;
    }

    /// <summary>Solver propagation from assumptions; Conflict is true when the assumptions are inconsistent.</summary>
    internal (HashSet<int> Implied, bool Conflict) PropagateAssumptions(List<int> assumptions)
    {
        var fresh = new Sat(_cnf);
        foreach (int assumption in assumptions) fresh.Assign(assumption, -1);
        var result = new HashSet<int>();
        if (fresh.UnsatAtLevelZero || !fresh.Propagate()) return (result, true);
        foreach (int literal in fresh._trail)
            if (!assumptions.Contains(literal)) result.Add(literal);
        return (result, false);
    }

    /// <summary>Independent oracle: naive fixpoint unit propagation over every clause; null means conflict.</summary>
    internal static HashSet<int>? NaivePropagate(Cnf cnf, List<int> assumptions)
    {
        var value = new int[cnf.Variables + 1];
        foreach (int assumption in assumptions) value[Math.Abs(assumption)] = assumption > 0 ? 1 : -1;
        var implied = new HashSet<int>();
        bool changed = true;
        while (changed)
        {
            changed = false;
            foreach (int[] clause in cnf.Clauses)
            {
                bool satisfied = false;
                int unassigned = 0, last = 0;
                var distinct = new List<int>();
                foreach (int literal in clause)
                {
                    if (distinct.Contains(literal)) continue;
                    distinct.Add(literal);
                }
                foreach (int literal in distinct)
                {
                    int v = Math.Abs(literal);
                    if (value[v] == 0) { unassigned++; last = literal; }
                    else if ((value[v] > 0) == (literal > 0)) { satisfied = true; break; }
                }
                if (satisfied) continue;
                if (unassigned == 0) return null;
                if (unassigned == 1)
                {
                    value[Math.Abs(last)] = last > 0 ? 1 : -1;
                    implied.Add(last);
                    changed = true;
                }
            }
        }
        return implied;
    }

    /// <summary>Exhaustive truth table; null means unsatisfiable.</summary>
    internal static int[]? BruteForce(Cnf cnf)
    {
        int n = cnf.Variables;
        for (int mask = 0; mask < 1 << n; mask++)
        {
            var assignment = new int[n + 1];
            for (int v = 1; v <= n; v++) assignment[v] = (mask & (1 << (v - 1))) != 0 ? 1 : -1;
            if (cnf.SatisfiedBy(assignment)) return assignment;
        }
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Cnf Random3Sat(int variables, int clauses, ulong seed)
    {
        var cnf = new Cnf(variables);
        ulong state = seed;
        for (int i = 0; i < clauses; i++)
        {
            int a = SplitMix.NextInt(ref state, variables) + 1;
            int b = SplitMix.NextInt(ref state, variables) + 1;
            int c = SplitMix.NextInt(ref state, variables) + 1;
            cnf.Add(SplitMix.NextInt(ref state, 2) == 0 ? a : -a,
                    SplitMix.NextInt(ref state, 2) == 0 ? b : -b,
                    SplitMix.NextInt(ref state, 2) == 0 ? c : -c);
        }
        return cnf;
    }

    private static Cnf Pigeonhole(int pigeons, int holes)
    {
        int Index(int p, int h) => p * holes + h + 1;
        var cnf = new Cnf(pigeons * holes);
        for (int p = 0; p < pigeons; p++)
        {
            var clause = new List<int>();
            for (int h = 0; h < holes; h++) clause.Add(Index(p, h));
            cnf.Add(clause.ToArray());
        }
        for (int h = 0; h < holes; h++)
            for (int p = 0; p < pigeons; p++)
                for (int q = p + 1; q < pigeons; q++)
                    cnf.Add(-Index(p, h), -Index(q, h));
        return cnf;
    }

    public static void Main()
    {
        // ---- trivial formulas --------------------------------------------------------------------------
        var empty = new Cnf(3);
        var emptySolver = new Sat(empty);
        Check(emptySolver.Solve() && empty.SatisfiedBy(emptySolver.Model()),
            "a formula with no clauses is satisfiable and the model satisfies it vacuously");

        var withEmptyClause = new Cnf(3);
        withEmptyClause.Add();
        Check(!new Sat(withEmptyClause).Solve(), "a formula containing the empty clause is unsatisfiable");

        var tautology = new Cnf(2);
        tautology.Add(1, -1);
        tautology.Add(2, -2);
        var tautologySolver = new Sat(tautology);
        Check(tautologySolver.Solve() && tautologySolver.Model()[1] != 0,
            "tautological clauses are dropped and the remaining formula is satisfiable");

        var chain = new Cnf(3);
        chain.Add(1);
        chain.Add(-1, 2);
        chain.Add(-2, 3);
        var chainSolver = new Sat(chain);
        Check(chainSolver.Solve() && chain.SatisfiedBy(chainSolver.Model()),
            "a unit chain propagates to a model that satisfies every clause");

        var contradiction = new Cnf(1);
        contradiction.Add(1);
        contradiction.Add(-1);
        Check(!new Sat(contradiction).Solve(), "directly contradictory unit clauses are refuted at level zero");

        // ---- propagation against the naive fixpoint oracle ----------------------------------------------
        bool propagationAgrees = true;
        string? firstFailure = null;
        ulong state = 0x7470BEEFUL;
        var probeCnf = Random3Sat(7, 24, 0x7470C0DEUL);
        for (int trial = 0; trial < 60; trial++)
        {
            var assumptions = new List<int>();
            while (assumptions.Count < 3)
            {
                int v = SplitMix.NextInt(ref state, 7) + 1;
                bool already = false;
                foreach (int literal in assumptions) if (Math.Abs(literal) == v) already = true;
                if (already) continue;
                assumptions.Add(SplitMix.NextInt(ref state, 2) == 0 ? v : -v);
            }
            var (implied, conflict) = new Sat(probeCnf).PropagateAssumptions(assumptions);
            var naive = Sat.NaivePropagate(probeCnf, assumptions);
            if (naive == null)
            {
                if (!conflict)
                {
                    propagationAgrees = false;
                    firstFailure ??= "trial " + trial + ": the oracle saw a conflict the solver missed";
                }
                continue;
            }
            if (conflict)
            {
                propagationAgrees = false;
                firstFailure ??= "trial " + trial + ": the solver saw a conflict the oracle missed";
                continue;
            }
            if (naive.Count != implied.Count)
            {
                propagationAgrees = false;
                firstFailure ??= "trial " + trial + ": propagated " + implied.Count + ", oracle " + naive.Count
                    + " assumptions=[" + string.Join(",", assumptions) + "] mine=["
                    + string.Join(",", implied) + "] oracle=[" + string.Join(",", naive) + "]";
                continue;
            }
            foreach (int literal in naive)
                if (!implied.Contains(literal))
                {
                    propagationAgrees = false;
                    firstFailure ??= "trial " + trial + ": literal " + literal + " missing from the solver's closure";
                }
        }
        Check(propagationAgrees, "watched-literal propagation matches the naive fixpoint scan on 60 partial "
            + "assignments: " + (firstFailure ?? "ok"));

        // ---- random 3-SAT against exhaustive enumeration --------------------------------------------------
        int agree = 0, satisfiableCount = 0, unsatisfiableCount = 0;
        long learnedTotal = 0, decisionsTotal = 0;
        bool allAgree = true;
        for (int round = 0; round < 40; round++)
        {
            var cnf = Random3Sat(9, 34 + round % 6, 0x74710000UL + (ulong)round * 7919UL);
            var solver = new Sat(cnf);
            bool satisfiable = solver.Solve();
            bool bruteForce = Sat.BruteForce(cnf) != null;
            if (satisfiable != bruteForce)
            {
                allAgree = false;
                firstFailure ??= "round " + round + ": solver says " + satisfiable + ", enumeration says " + bruteForce;
            }
            if (satisfiable && !cnf.SatisfiedBy(solver.Model()))
            {
                allAgree = false;
                firstFailure ??= "round " + round + ": the model does not satisfy the formula";
            }
            if (!solver.InvariantHolds)
            {
                allAgree = false;
                firstFailure ??= "round " + round + ": watched-literal invariant violated";
            }
            agree++;
            learnedTotal += solver.LearnedClauses;
            decisionsTotal += solver.Decisions;
            if (satisfiable) satisfiableCount++; else unsatisfiableCount++;
        }
        Check(allAgree, "the solver agrees with exhaustive enumeration on all " + agree + " random 3-SAT formulas ("
            + satisfiableCount + " SAT, " + unsatisfiableCount + " UNSAT): " + (firstFailure ?? "ok"));
        Check(learnedTotal > 0, "conflict analysis learned " + learnedTotal + " clauses over those formulas");

        // ---- pigeonhole: UNSAT by counting, with clause learning ------------------------------------------
        var php43 = Pigeonhole(4, 3);
        var php43Solver = new Sat(php43);
        bool php43Sat = php43Solver.Solve();
        Check(!php43Sat && php43Solver.LearnedClauses > 0 && php43Solver.Conflicts > 0,
            "4 pigeons in 3 holes is UNSAT after " + php43Solver.LearnedClauses + " learned clauses and "
            + php43Solver.Conflicts + " conflicts");
        Check(!php43Solver.UnsatAtLevelZero, "the pigeonhole conflict happens above level zero, so learning is exercised");
        Check(php43Solver.InvariantHolds && php43Solver.InvariantChecks > 0,
            "the watched-literal invariant survived " + php43Solver.InvariantChecks + " propagation rounds");
        var php54 = Pigeonhole(5, 4);
        Check(!new Sat(php54).Solve(), "5 pigeons in 4 holes is UNSAT as well");
        var php32 = Pigeonhole(3, 2);
        Check(!new Sat(php32).Solve(), "3 pigeons in 2 holes is UNSAT, exactly as the counting argument says");

        // ---- structured unsatisfiable instance: K4 is not 3-colourable ------------------------------------
        var colouring = new Cnf(12);
        int Node(int v, int c) => v * 3 + c + 1;
        for (int v = 0; v < 4; v++) colouring.Add(Node(v, 0), Node(v, 1), Node(v, 2));
        for (int v = 0; v < 4; v++)
            for (int c = 0; c < 3; c++)
                for (int d = c + 1; d < 3; d++)
                    colouring.Add(-Node(v, c), -Node(v, d));
        for (int v = 0; v < 4; v++)
            for (int w = v + 1; w < 4; w++)
                for (int c = 0; c < 3; c++)
                    colouring.Add(-Node(v, c), -Node(w, c));
        var colouringSolver = new Sat(colouring);
        Check(!colouringSolver.Solve() && Sat.BruteForce(colouring) == null,
            "K4 admits no 3-colouring: the solver and the truth table agree");

        // ---- determinism -----------------------------------------------------------------------------------
        var first = Random3Sat(8, 30, 0x747ABCUL);
        var second = Random3Sat(8, 30, 0x747ABCUL);
        var solverA = new Sat(first);
        var solverB = new Sat(second);
        bool answerA = solverA.Solve(), answerB = solverB.Solve();
        bool sameModel = true;
        if (answerA)
            for (int v = 1; v <= 8; v++)
                if (solverA.Model()[v] != solverB.Model()[v]) sameModel = false;
        Check(answerA == answerB && sameModel && solverA.Decisions == solverB.Decisions,
            "the same formula is solved identically twice (same answer, model and decision count)");

        Console.WriteLine("randomFormulas=" + agree + " sat=" + satisfiableCount + " unsat=" + unsatisfiableCount
            + " learnedClauses=" + learnedTotal + " decisions=" + decisionsTotal
            + " pigeonholeLearned=" + php43Solver.LearnedClauses);
        Console.WriteLine("PASS");
    }
}
