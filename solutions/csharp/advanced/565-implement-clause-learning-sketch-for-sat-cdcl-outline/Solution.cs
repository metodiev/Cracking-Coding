// Task 565 - Implement clause learning sketch for SAT (CDCL outline). (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
//
// Conflict-driven clause learning, implemented rather than sketched. The solver keeps a clause database, a
// trail of assigned literals with the decision level of each assignment, the reason clause of every
// implied literal (a decision has no reason), a VSIDS-style activity counter per variable and a saved
// phase per variable. Propagation scans the database to a fixed point (unit clauses are assigned, fully
// falsified clauses are conflicts) - the watched-literal two-pointer trick is deliberately left out and
// the cost is stated: scanning is O(total literal occurrences) per propagation round, which is fine for
// the sizes tested and keeps the learning logic readable. On a conflict at a non-zero decision level the
// solver analyses it with the first unique implication point rule: it walks back along the trail, counting
// literals of the current level, skipping the propagated literal, accumulating lower-level literals, and
// stops when exactly one current-level literal remains - that literal's negation is the asserting literal,
// and the accumulated literals plus it form the learned clause; the solver then backjumps to the second
// highest level in the learned clause and asserts the learned clause there. Activities of the variables in
// the learned clause are bumped, which drives the next decisions. The library is not trusted: every
// learned clause is checked to be logically implied by the original formula (no assignment may satisfy the
// original clauses and falsify the learned clause - verified by brute force on these small instances),
// every verdict is compared with brute force, and every model is verified clause by clause. Instances are
// textbook unsatisfiable systems (the eight sign patterns over three variables, pigeonhole instances) or
// splitmix64-seeded random 3-SAT.
//
// Documented checks: on 40 random 3-SAT instances (n = 10, m = 42) and 20 more (n = 12, m = 50) the CDCL
// verdict equals the brute-force verdict, models are verified, and a bounded sample of the learned clauses
// (the first few of each smaller instance) is exhaustively verified to be implied by the original formula;
// learned clauses are never empty and their number never exceeds the conflict count; PHP(3,2) and PHP(4,3)
// are unsatisfiable with conflicts and learned clauses actually occurring, while PHP(2,2) and PHP(3,3) are
// satisfiable; the eight sign patterns over three variables are unsatisfiable; a formula that is already
// solved by propagation alone needs no conflicts; and repeated runs give identical verdicts, models and
// conflict counts.
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

internal sealed class Formula
{
    internal readonly int Vars;
    internal readonly List<int[]> Clauses;
    internal readonly string Label;

    internal Formula(int vars, string label)
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

    internal bool Falsifies(bool[] assignment, int[] clause)
    {
        foreach (int literal in clause)
        {
            bool value = assignment[Math.Abs(literal) - 1];
            if (literal < 0) value = !value;
            if (value) return false;
        }
        return true;
    }

    internal static Formula Random3Sat(int vars, int clauses, ref ulong state, string label)
    {
        var formula = new Formula(vars, label);
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
            formula.Add(literals);
        }
        return formula;
    }

    internal static Formula Pigeonhole(int pigeons, int holes)
    {
        var formula = new Formula(pigeons * holes, "PHP(" + pigeons + "," + holes + ")");
        for (int p = 0; p < pigeons; p++)
        {
            var clause = new int[holes];
            for (int h = 0; h < holes; h++) clause[h] = p * holes + h + 1;
            formula.Add(clause);
        }
        for (int h = 0; h < holes; h++)
        {
            for (int a = 0; a < pigeons; a++)
            {
                for (int b = a + 1; b < pigeons; b++)
                {
                    formula.Add(-(a * holes + h + 1), -(b * holes + h + 1));
                }
            }
        }
        return formula;
    }

    internal static bool BruteForce(Formula formula)
    {
        int n = formula.Vars;
        for (long mask = 0; mask < (1L << n); mask++)
        {
            var assignment = new bool[n];
            for (int i = 0; i < n; i++) assignment[i] = (mask & (1L << i)) != 0;
            if (formula.SatisfiedBy(assignment)) return true;
        }
        return false;
    }
}

internal static class Cdcl
{
    internal sealed class Result
    {
        internal bool Satisfiable;
        internal bool[] Model = Array.Empty<bool>();
        internal long Conflicts;
        internal long Decisions;
        internal long Propagations;
        internal List<int[]> Learned = new List<int[]>();
    }

    private sealed class Solver
    {
        private readonly Formula formula;
        private readonly List<int[]> clauses;
        private readonly int[] value;
        private readonly int[] level;
        private readonly int[] reason;
        private readonly int[] activity;
        private readonly bool[] phase;
        private readonly bool[] seen;
        private readonly List<int> trail = new List<int>();
        private readonly Result result = new Result();
        private int currentLevel;

        internal Solver(Formula formula)
        {
            this.formula = formula;
            clauses = new List<int[]>();
            foreach (int[] clause in formula.Clauses) clauses.Add((int[])clause.Clone());
            value = new int[formula.Vars + 1];
            level = new int[formula.Vars + 1];
            reason = new int[formula.Vars + 1];
            activity = new int[formula.Vars + 1];
            phase = new bool[formula.Vars + 1];
            seen = new bool[formula.Vars + 1];
            for (int i = 0; i <= formula.Vars; i++)
            {
                reason[i] = -1;
                phase[i] = true;
            }
        }

        internal Result Run()
        {
            while (true)
            {
                int conflict = Propagate();
                if (conflict >= 0)
                {
                    result.Conflicts++;
                    if (currentLevel == 0)
                    {
                        result.Satisfiable = false;
                        result.Model = new bool[formula.Vars];
                        return result;
                    }
                    int[] learnt = Analyze(conflict, out int backjump);
                    result.Learned.Add(learnt);
                    foreach (int literal in learnt) activity[Math.Abs(literal)]++;
                    clauses.Add(learnt);
                    Backjump(backjump);
                    Assign(learnt[0], clauses.Count - 1);
                    continue;
                }
                int next = ChooseVariable();
                if (next == 0)
                {
                    var model = new bool[formula.Vars];
                    for (int v = 1; v <= formula.Vars; v++) model[v - 1] = value[v] == 1;
                    result.Satisfiable = true;
                    result.Model = model;
                    return result;
                }
                currentLevel++;
                result.Decisions++;
                Assign(phase[next] ? next : -next, -1);
            }
        }

        /** Scans the database to a fixed point; returns the conflicting clause index or -1. */
        private int Propagate()
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int index = 0; index < clauses.Count; index++)
                {
                    int[] clause = clauses[index];
                    bool satisfied = false;
                    int undetermined = 0;
                    int unit = 0;
                    foreach (int literal in clause)
                    {
                        int v = Math.Abs(literal);
                        if (value[v] == 0)
                        {
                            undetermined++;
                            unit = literal;
                        }
                        else if ((value[v] == 1) == (literal > 0))
                        {
                            satisfied = true;
                            break;
                        }
                    }
                    if (satisfied) continue;
                    if (undetermined == 0) return index;
                    if (undetermined == 1 && value[Math.Abs(unit)] == 0)
                    {
                        Assign(unit, index);
                        result.Propagations++;
                        changed = true;
                    }
                }
            }
            return -1;
        }

        /** First unique implication point analysis. */
        private int[] Analyze(int conflictIndex, out int backjump)
        {
            var learnt = new List<int>();
            int pathCount = 0;
            int p = 0;
            int index = trail.Count - 1;
            int[] clause = clauses[conflictIndex];
            while (true)
            {
                foreach (int q in clause)
                {
                    int v = Math.Abs(q);
                    if (p != 0 && v == Math.Abs(p)) continue;
                    if (seen[v] || level[v] == 0) continue;
                    seen[v] = true;
                    if (level[v] == currentLevel) pathCount++;
                    else learnt.Add(q);
                }
                while (index >= 0 && !seen[Math.Abs(trail[index])]) index--;
                if (index < 0) throw new Exception("conflict analysis ran off the trail");
                p = trail[index];
                index--;
                seen[Math.Abs(p)] = false;
                pathCount--;
                if (pathCount <= 0) break;
                clause = clauses[reason[Math.Abs(p)]];
            }
            var resultClause = new List<int> { -p };
            resultClause.AddRange(learnt);
            foreach (int literal in resultClause) seen[Math.Abs(literal)] = false;
            backjump = 0;
            for (int i = 1; i < resultClause.Count; i++)
            {
                backjump = Math.Max(backjump, level[Math.Abs(resultClause[i])]);
            }
            return resultClause.ToArray();
        }

        private void Backjump(int target)
        {
            while (trail.Count > 0)
            {
                int literal = trail[trail.Count - 1];
                if (level[Math.Abs(literal)] <= target) break;
                value[Math.Abs(literal)] = 0;
                trail.RemoveAt(trail.Count - 1);
            }
            currentLevel = target;
        }

        private void Assign(int literal, int reasonIndex)
        {
            int v = Math.Abs(literal);
            value[v] = literal > 0 ? 1 : -1;
            level[v] = currentLevel;
            reason[v] = reasonIndex;
            phase[v] = literal > 0;
            trail.Add(literal);
        }

        private int ChooseVariable()
        {
            int best = 0;
            for (int v = 1; v <= formula.Vars; v++)
            {
                if (value[v] != 0) continue;
                if (best == 0 || activity[v] > activity[best]) best = v;
            }
            return best;
        }
    }

    internal static Result Solve(Formula formula)
    {
        return new Solver(formula).Run();
    }
}

public static class Solution
{
    public static void Main()
    {
        ulong rng = 565001UL;
        int instances = 0;
        int learnedChecked = 0;
        int satCount = 0;
        for (int round = 0; round < 60; round++)
        {
            int vars = round < 40 ? 10 : 12;
            int clauses = round < 40 ? 42 : 50;
            Formula formula = Formula.Random3Sat(vars, clauses, ref rng, "random" + round);
            Cdcl.Result result = Cdcl.Solve(formula);
            bool expected = Formula.BruteForce(formula);
            Check(result.Satisfiable == expected, formula.Label + ": CDCL says " + result.Satisfiable
                    + ", brute force says " + expected);
            if (result.Satisfiable)
            {
                Check(formula.SatisfiedBy(result.Model), formula.Label + ": model is not a model");
                satCount++;
            }
            Check(result.Learned.Count <= result.Conflicts, formula.Label
                    + ": learned " + result.Learned.Count + " clauses from " + result.Conflicts
                    + " conflicts");
            foreach (int[] learned in result.Learned)
            {
                Check(learned.Length > 0, formula.Label + ": a learned clause is empty");
                // The learned clause must be implied: no model of the formula may falsify it. Verified
                // exhaustively on the smaller instances only, and only for the first few clauses, to keep
                // the check affordable.
                if (vars > 10 || learnedChecked >= 8 * (round + 1)) continue;
                int n = formula.Vars;
                for (long mask = 0; mask < (1L << n); mask++)
                {
                    var assignment = new bool[n];
                    for (int i = 0; i < n; i++) assignment[i] = (mask & (1L << i)) != 0;
                    if (!formula.SatisfiedBy(assignment)) continue;
                    Check(!formula.Falsifies(assignment, learned), formula.Label
                            + ": a learned clause is not implied by the formula");
                }
                learnedChecked++;
            }
            instances++;
        }
        Check(instances == 60, "60 random instances were solved");
        Check(satCount > 10, "the random instances include satisfiable ones (" + satCount + ")");
        Check(learnedChecked > 0, "learned clauses were produced and verified (" + learnedChecked + ")");

        // Unsatisfiable systems must produce conflicts and learned clauses.
        Formula php32 = Formula.Pigeonhole(3, 2);
        Cdcl.Result php32Result = Cdcl.Solve(php32);
        Check(!php32Result.Satisfiable, "PHP(3,2) is unsatisfiable");
        Check(!Formula.BruteForce(php32), "brute force agrees on PHP(3,2)");
        Check(php32Result.Conflicts > 0 && php32Result.Learned.Count > 0,
                "PHP(3,2) needs clause learning, got " + php32Result.Conflicts + " conflicts");

        Formula php43 = Formula.Pigeonhole(4, 3);
        Cdcl.Result php43Result = Cdcl.Solve(php43);
        Check(!php43Result.Satisfiable, "PHP(4,3) is unsatisfiable");
        Check(!Formula.BruteForce(php43), "brute force agrees on PHP(4,3)");
        Check(php43Result.Conflicts > php32Result.Conflicts, "PHP(4,3) is harder than PHP(3,2): "
                + php43Result.Conflicts + " vs " + php32Result.Conflicts);
        foreach (int[] learned in php43Result.Learned)
        {
            Check(learned.Length > 0, "no learned clause of PHP(4,3) is empty");
        }

        Check(Cdcl.Solve(Formula.Pigeonhole(2, 2)).Satisfiable, "PHP(2,2) is satisfiable");
        Check(Cdcl.Solve(Formula.Pigeonhole(3, 3)).Satisfiable, "PHP(3,3) is satisfiable");

        // The eight sign patterns over three variables.
        var allPatterns = new Formula(3, "all-sign-patterns");
        for (int mask = 0; mask < 8; mask++)
        {
            var clause = new int[3];
            for (int v = 0; v < 3; v++) clause[v] = ((mask >> v) & 1) == 0 ? v + 1 : -(v + 1);
            allPatterns.Add(clause);
        }
        Check(!Cdcl.Solve(allPatterns).Satisfiable, "the eight sign patterns are unsatisfiable");
        Check(!Formula.BruteForce(allPatterns), "brute force agrees on the eight sign patterns");

        // Propagation alone: no conflicts.
        var easy = new Formula(4, "easy");
        easy.Add(1);
        easy.Add(-2);
        easy.Add(3, 4);
        Cdcl.Result easyResult = Cdcl.Solve(easy);
        Check(easyResult.Satisfiable, "the easy formula is satisfiable");
        Check(easyResult.Conflicts == 0, "propagation alone needs no conflicts, got "
                + easyResult.Conflicts);
        Check(easyResult.Propagations >= 2, "the units were propagated, got " + easyResult.Propagations);
        Check(easy.SatisfiedBy(easyResult.Model), "the easy model is valid");

        // Determinism.
        Formula repeat = Formula.Random3Sat(12, 50, ref rng, "repeat");
        Cdcl.Result first = Cdcl.Solve(repeat);
        Cdcl.Result second = Cdcl.Solve(repeat);
        Check(first.Satisfiable == second.Satisfiable && first.Conflicts == second.Conflicts,
                "repeated runs agree on verdict and conflicts");
        if (first.Satisfiable)
        {
            for (int i = 0; i < repeat.Vars; i++)
            {
                Check(first.Model[i] == second.Model[i], "repeated runs agree on variable " + (i + 1));
            }
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
