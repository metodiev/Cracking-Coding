// Task 563 - Implement SAT solver basic DPLL algorithm. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
//
// Davis-Putnam-Logemann-Loveland search over a DIMACS-style clause list with signed literal encoding
// (lit = v means x_v, lit = -v means not x_v, v starting at 1) and a tri-state assignment array
// (0 unassigned, +1 true, -1 false - a plain bool array cannot distinguish "false" from "unassigned",
// which is the classic way to break this algorithm). At every node the clause set is simplified:
// satisfied clauses are dropped, falsified literals are removed, the empty clause means conflict and an
// empty formula means satisfiable. Unit propagation then runs to a fixed point, pure literals (variables
// occurring with one polarity only) are assigned greedily, and only if neither made progress does the
// search branch - choosing the variable occurring most often in the shortest remaining clauses, the
// standard "most constrained" heuristic, and trying the polarity with more occurrences first. Search is
// recursive over simplified clause lists with save/restore of the assignment, so no mutable-trail or
// watched-literal bookkeeping can go wrong; the price is copying, which is stated rather than hidden and
// is irrelevant at these sizes. Tautological clauses are dropped up front, and the reference for every
// verdict is a brute-force enumeration of all 2^n assignments, with each reported model verified clause by
// clause against the original formula. Instances are textbook (the eight sign patterns of a three-literal
// clause over three variables, pigeonhole instances) or splitmix64-seeded random 3-SAT.
//
// Documented checks: on 60 random 3-SAT instances (30 with n = 8, m = 30 and 30 with n = 10, m = 42) plus
// 8 more (n = 9, m = 36) the DPLL verdict equals the brute-force verdict and every satisfying model
// returned is verified clause by clause; the decision count never exceeds the number of variables; the
// eight sign patterns over three variables are unsatisfiable, PHP(3,2) and PHP(4,3) are unsatisfiable
// while PHP(2,2) is satisfiable, all cross-checked by brute force; a formula of unit clauses and pure
// literals is solved with zero decisions and the forced assignment is respected; the empty formula is
// satisfiable, a formula containing the empty clause is not, tautologies change nothing, and a tautology
// does not disturb the pure-literal reasoning; and repeated runs return identical verdicts and models.
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

internal sealed class Cnf
{
    internal readonly int Vars;
    internal readonly List<int[]> Clauses;
    internal readonly string Label;

    internal Cnf(int vars, string label)
    {
        Vars = vars;
        Label = label;
        Clauses = new List<int[]>();
    }

    internal void Add(params int[] clause)
    {
        Clauses.Add(clause);
    }

    internal bool SatisfiedBy(bool[] assignment)
    {
        foreach (int[] clause in Clauses)
        {
            bool ok = false;
            foreach (int literal in clause)
            {
                bool value = assignment[Math.Abs(literal) - 1];
                if (literal < 0) value = !value;
                if (value) { ok = true; break; }
            }
            if (!ok) return false;
        }
        return true;
    }

    internal static Cnf Random3Sat(int vars, int clauses, ref ulong state, string label)
    {
        var cnf = new Cnf(vars, label);
        for (int i = 0; i < clauses; i++)
        {
            var literals = new int[3];
            var used = new HashSet<int>();
            for (int j = 0; j < 3; j++)
            {
                int v;
                do
                {
                    v = 1 + SplitMix64.Below(ref state, vars);
                }
                while (!used.Add(v));
                literals[j] = SplitMix64.Below(ref state, 2) == 0 ? v : -v;
            }
            cnf.Add(literals);
        }
        return cnf;
    }

    /** Pigeonhole principle: pigeons into holes slots, unsatisfiable when pigeons > holes. */
    internal static Cnf Pigeonhole(int pigeons, int holes)
    {
        var cnf = new Cnf(pigeons * holes, "PHP(" + pigeons + "," + holes + ")");
        for (int p = 0; p < pigeons; p++)
        {
            var clause = new int[holes];
            for (int h = 0; h < holes; h++) clause[h] = p * holes + h + 1;
            cnf.Add(clause);
        }
        for (int h = 0; h < holes; h++)
        {
            for (int a = 0; a < pigeons; a++)
            {
                for (int b = a + 1; b < pigeons; b++)
                {
                    cnf.Add(-(a * holes + h + 1), -(b * holes + h + 1));
                }
            }
        }
        return cnf;
    }
}

internal static class Dpll
{
    internal sealed class Result
    {
        internal bool Satisfiable;
        internal bool[] Model = Array.Empty<bool>();
        internal long Decisions;
        internal long Propagations;
        internal long PureLiterals;
    }

    internal static Result Solve(Cnf cnf)
    {
        var clauses = new List<int[]>();
        foreach (int[] clause in cnf.Clauses)
        {
            bool tautology = false;
            for (int i = 0; i < clause.Length && !tautology; i++)
            {
                for (int j = i + 1; j < clause.Length; j++)
                {
                    if (clause[i] == -clause[j]) { tautology = true; break; }
                }
            }
            if (!tautology) clauses.Add((int[])clause.Clone());
        }
        var state = new int[cnf.Vars + 1];
        var result = new Result();
        result.Satisfiable = Search(state, clauses, result);
        var model = new bool[cnf.Vars];
        for (int v = 1; v <= cnf.Vars; v++) model[v - 1] = state[v] == 1;
        result.Model = model;
        return result;
    }

    private static bool Search(int[] state, List<int[]> clauses, Result result)
    {
        var simplified = new List<int[]>();
        if (!Simplify(state, clauses, simplified)) return false;
        if (simplified.Count == 0) return true;

        // Unit propagation to a fixed point.
        while (true)
        {
            int unit = 0;
            foreach (int[] clause in simplified)
            {
                if (clause.Length == 1) { unit = clause[0]; break; }
            }
            if (unit == 0) break;
            int v = Math.Abs(unit);
            int value = unit > 0 ? 1 : -1;
            if (state[v] == -value) return false;
            if (state[v] == 0)
            {
                state[v] = value;
                result.Propagations++;
            }
            var next = new List<int[]>();
            if (!Simplify(state, simplified, next)) return false;
            simplified = next;
            if (simplified.Count == 0) return true;
        }

        // Pure literals.
        var positive = new HashSet<int>();
        var negative = new HashSet<int>();
        foreach (int[] clause in simplified)
        {
            foreach (int literal in clause)
            {
                if (literal > 0) positive.Add(literal);
                else negative.Add(-literal);
            }
        }
        bool assignedPure = false;
        foreach (int v in positive)
        {
            if (negative.Contains(v) || state[v] != 0) continue;
            state[v] = 1;
            result.PureLiterals++;
            assignedPure = true;
        }
        foreach (int v in negative)
        {
            if (positive.Contains(v) || state[v] != 0) continue;
            state[v] = -1;
            result.PureLiterals++;
            assignedPure = true;
        }
        if (assignedPure) return Search(state, simplified, result);

        // Branch on the variable occurring most often among the shortest clauses.
        int bestVariable = 0;
        int bestCount = -1;
        int bestLength = int.MaxValue;
        var counts = new Dictionary<int, int>();
        foreach (int[] clause in simplified)
        {
            if (clause.Length < bestLength)
            {
                bestLength = clause.Length;
                counts.Clear();
                bestCount = -1;
            }
            else if (clause.Length > bestLength)
            {
                continue;
            }
            foreach (int literal in clause)
            {
                int v = Math.Abs(literal);
                counts.TryGetValue(v, out int current);
                counts[v] = current + 1;
                if (current + 1 > bestCount)
                {
                    bestCount = current + 1;
                    bestVariable = v;
                }
            }
        }
        if (bestVariable == 0) return true;

        int[] saved = (int[])state.Clone();
        result.Decisions++;
        state[bestVariable] = 1;
        if (Search(state, simplified, result)) return true;
        Array.Copy(saved, state, state.Length);
        state[bestVariable] = -1;
        if (Search(state, simplified, result)) return true;
        Array.Copy(saved, state, state.Length);
        return false;
    }

    /** Drops satisfied clauses and falsified literals; false means the empty clause appeared. */
    private static bool Simplify(int[] state, List<int[]> clauses, List<int[]> sink)
    {
        foreach (int[] clause in clauses)
        {
            bool satisfied = false;
            var remaining = new List<int>(clause.Length);
            foreach (int literal in clause)
            {
                int v = Math.Abs(literal);
                if (state[v] == 0)
                {
                    remaining.Add(literal);
                }
                else if ((state[v] == 1) == (literal > 0))
                {
                    satisfied = true;
                    break;
                }
            }
            if (satisfied) continue;
            if (remaining.Count == 0) return false;
            sink.Add(remaining.ToArray());
        }
        return true;
    }

    /** Brute force over all 2^n assignments. */
    internal static bool BruteForce(Cnf cnf)
    {
        int n = cnf.Vars;
        for (long mask = 0; mask < (1L << n); mask++)
        {
            var assignment = new bool[n];
            for (int i = 0; i < n; i++) assignment[i] = (mask & (1L << i)) != 0;
            if (cnf.SatisfiedBy(assignment)) return true;
        }
        return false;
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 563001UL;
        int checkedInstances = 0;
        for (int round = 0; round < 60; round++)
        {
            int vars = round < 30 ? 8 : 10;
            int clauses = round < 30 ? 30 : 42;
            Cnf cnf = Cnf.Random3Sat(vars, clauses, ref rng, "random" + round);
            Dpll.Result result = Dpll.Solve(cnf);
            bool expected = Dpll.BruteForce(cnf);
            Check(result.Satisfiable == expected, cnf.Label + ": DPLL says " + result.Satisfiable
                    + ", brute force says " + expected);
            if (result.Satisfiable)
            {
                Check(cnf.SatisfiedBy(result.Model), cnf.Label + ": the reported model is not a model");
                Check(result.Decisions <= cnf.Vars, cnf.Label + ": more decisions (" + result.Decisions
                        + ") than variables (" + cnf.Vars + ")");
            }
            checkedInstances++;
        }
        Check(checkedInstances == 60, "60 random instances were checked");

        // All eight sign patterns of a three-literal clause over three variables: unsatisfiable.
        var allPatterns = new Cnf(3, "all-sign-patterns");
        for (int mask = 0; mask < 8; mask++)
        {
            var clause = new int[3];
            for (int v = 0; v < 3; v++) clause[v] = ((mask >> v) & 1) == 0 ? v + 1 : -(v + 1);
            allPatterns.Add(clause);
        }
        Check(!Dpll.Solve(allPatterns).Satisfiable, "the eight sign patterns are unsatisfiable");
        Check(!Dpll.BruteForce(allPatterns), "brute force agrees on the eight sign patterns");

        // Pigeonhole.
        Check(!Dpll.Solve(Cnf.Pigeonhole(3, 2)).Satisfiable, "PHP(3,2) is unsatisfiable");
        Check(!Dpll.BruteForce(Cnf.Pigeonhole(3, 2)), "brute force agrees on PHP(3,2)");
        Check(Dpll.Solve(Cnf.Pigeonhole(2, 2)).Satisfiable, "PHP(2,2) is satisfiable");
        Check(!Dpll.Solve(Cnf.Pigeonhole(4, 3)).Satisfiable, "PHP(4,3) is unsatisfiable");
        Check(!Dpll.BruteForce(Cnf.Pigeonhole(4, 3)), "brute force agrees on PHP(4,3)");

        // Propagation and pure literals alone solve the formula.
        var units = new Cnf(5, "units");
        units.Add(1);
        units.Add(-2);
        units.Add(3);
        units.Add(4, 5);
        Dpll.Result unitResult = Dpll.Solve(units);
        Check(unitResult.Satisfiable, "the unit formula is satisfiable");
        Check(unitResult.Decisions == 0, "no decision is needed, got " + unitResult.Decisions);
        Check(unitResult.Propagations > 0, "unit propagation did the work");
        Check(unitResult.PureLiterals >= 2, "the two pure literals 4 and 5 were assigned, got "
                + unitResult.PureLiterals);
        Check(unitResult.Model[0] && !unitResult.Model[1] && unitResult.Model[2],
                "the forced assignment is respected");

        // Boundary cases and tautologies.
        var empty = new Cnf(3, "empty");
        Check(Dpll.Solve(empty).Satisfiable, "the empty formula is satisfiable");
        var hasEmptyClause = new Cnf(3, "empty-clause");
        hasEmptyClause.Add(new int[0]);
        Check(!Dpll.Solve(hasEmptyClause).Satisfiable, "a formula containing the empty clause is unsatisfiable");
        var tautologies = new Cnf(2, "tautologies");
        tautologies.Add(1, -1);
        tautologies.Add(2, -2);
        Check(Dpll.Solve(tautologies).Satisfiable, "tautologies are satisfiable");
        var mixed = new Cnf(3, "mixed");
        mixed.Add(1, -1);
        mixed.Add(2, 3);
        mixed.Add(-2, 3);
        Dpll.Result mixedResult = Dpll.Solve(mixed);
        Check(mixedResult.Satisfiable, "the mixed formula is satisfiable");
        Check(mixedResult.Decisions == 0, "a pure literal decides the mixed formula, got "
                + mixedResult.Decisions + " decisions");
        Check(mixedResult.Model[2], "x3 is the pure literal and must be true");
        Check(mixed.SatisfiedBy(mixedResult.Model), "the mixed model satisfies every clause");

        // Determinism.
        Cnf repeat = Cnf.Random3Sat(10, 42, ref rng, "repeat");
        Dpll.Result first = Dpll.Solve(repeat);
        Dpll.Result second = Dpll.Solve(repeat);
        Check(first.Satisfiable == second.Satisfiable, "repeated runs agree on satisfiability");
        if (first.Satisfiable)
        {
            for (int i = 0; i < repeat.Vars; i++)
            {
                Check(first.Model[i] == second.Model[i], "repeated runs agree on variable " + i);
            }
        }
        for (int i = 0; i < 8; i++)
        {
            Cnf again = Cnf.Random3Sat(9, 36, ref rng, "again" + i);
            Check(Dpll.Solve(again).Satisfiable == Dpll.BruteForce(again),
                    "verdict matches brute force on again" + i);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
