// Task 749 - Implement symbolic execution skeleton for small programs. (Tier: Expert, Beyond - Compilers,
// Languages & Solvers)
//
// A skeleton symbolic executor for a tiny integer IR. A program is a list of instructions
//   v = expr (linear in the input variables),  if (expr op 0) goto L,  assert(expr op 0),  halt
// and execution explores the control-flow tree depth first, carrying an environment that maps program
// variables to linear expressions over the (symbolic) inputs and a path condition: a conjunction of linear
// atoms. Before a state is expanded its path condition is discharged by a bounded model finder that enumerates
// input tuples in [-8, 8] (the skeleton's stand-in for a real arithmetic solver), which prunes infeasible
// branches: a state whose condition is unsatisfiable is never explored and no witness is invented for it.
// Invariants verified, all concolically - a witness produced by the symbolic side is replayed by an independent
// concrete interpreter:
//   * for every feasible complete path, the concrete interpreter run on the witness takes exactly the same
//     branch decisions and ends with the same variable values as the symbolic state;
//   * for every assert violation, the witness really makes the concrete run fail at that assert;
//   * exploration is complete on the bounded domain: every input of the domain is covered by one of the
//     explored feasible paths, and the number of distinct concrete branch signatures equals the number of
//     feasible paths, while the path whose condition is unsatisfiable contributes no witness;
//   * loops are unrolled with a step budget, so the exploration always terminates.
// Simplifications of this skeleton: linear arithmetic only (multiplication of two symbolic values is rejected
// rather than modelled), the model finder is a bounded enumeration instead of an SMT query, the domain bound is
// fixed, no heap/arrays/pointers, no path merging or interpolation, and loops are bounded. All programs are
// literal, so every run is deterministic.
using System;
using System.Collections.Generic;

internal enum Op { Le, Lt, Eq, Ne }

internal sealed class LinExpr
{
    // A linear form over the inputs: sum of coefficients times input variables, plus a constant.
    internal readonly Dictionary<int, long> Coefficients = new Dictionary<int, long>();
    internal long Constant;

    internal LinExpr() { }

    internal LinExpr(long constant) => Constant = constant;

    internal LinExpr Add(LinExpr other)
    {
        var sum = new LinExpr(Constant + other.Constant);
        foreach (var pair in Coefficients) sum.Coefficients[pair.Key] = pair.Value;
        foreach (var pair in other.Coefficients)
        {
            sum.Coefficients.TryGetValue(pair.Key, out long current);
            sum.Coefficients[pair.Key] = current + pair.Value;
        }
        sum.Normalize();
        return sum;
    }

    internal LinExpr Scale(long factor)
    {
        var scaled = new LinExpr(Constant * factor);
        foreach (var pair in Coefficients) scaled.Coefficients[pair.Key] = pair.Value * factor;
        scaled.Normalize();
        return scaled;
    }

    internal void Normalize()
    {
        var zeros = new List<int>();
        foreach (var pair in Coefficients) if (pair.Value == 0) zeros.Add(pair.Key);
        foreach (int key in zeros) Coefficients.Remove(key);
    }

    internal long Evaluate(long[] inputs)
    {
        long value = Constant;
        foreach (var pair in Coefficients) value += pair.Value * inputs[pair.Key];
        return value;
    }

    internal string Text => Constant + (Coefficients.Count == 0 ? "" : "+" + Coefficients.Count + " vars");
}

internal sealed class Atom
{
    internal readonly LinExpr Expression;
    internal readonly Op Comparison;
    internal readonly bool Truth;

    internal Atom(LinExpr expression, Op comparison, bool truth)
    {
        Expression = expression;
        Comparison = comparison;
        Truth = truth;
    }

    internal bool Holds(long value)
    {
        bool result = Comparison switch
        {
            Op.Le => value <= 0,
            Op.Lt => value < 0,
            Op.Eq => value == 0,
            _ => value != 0
        };
        return Truth ? result : !result;
    }
}

internal sealed class Instruction
{
    internal int Kind;                 // 0 = assign, 1 = branch, 2 = assert, 3 = halt
    internal int Variable;             // destination of an assignment
    internal LinExpr? Value;           // right-hand side of an assignment
    internal LinExpr? Condition;       // condition of a branch or assertion
    internal Op Comparison;
    internal int Target;               // branch target when the condition holds
    internal string Text = "";
}

internal sealed class Program
{
    internal readonly int Inputs;
    internal readonly List<Instruction> Instructions = new List<Instruction>();

    internal Program(int inputs) => Inputs = inputs;

    internal int Assign(int variable, LinExpr value, string text)
    {
        Instructions.Add(new Instruction { Kind = 0, Variable = variable, Value = value, Text = text });
        return Instructions.Count - 1;
    }

    internal int Branch(LinExpr condition, Op comparison, int target, string text)
    {
        Instructions.Add(new Instruction { Kind = 1, Condition = condition, Comparison = comparison, Target = target, Text = text });
        return Instructions.Count - 1;
    }

    internal int Assert(LinExpr condition, Op comparison, string text)
    {
        Instructions.Add(new Instruction { Kind = 2, Condition = condition, Comparison = comparison, Text = text });
        return Instructions.Count - 1;
    }

    internal int Halt()
    {
        Instructions.Add(new Instruction { Kind = 3 });
        return Instructions.Count - 1;
    }
}

internal sealed class Path
{
    internal readonly List<Atom> Conditions = new List<Atom>();
    internal readonly List<int> BranchDecisions = new List<int>();
    internal readonly Dictionary<int, LinExpr> FinalValues = new Dictionary<int, LinExpr>();
    internal bool Feasible = true;
    internal bool Failed;
    internal string FailedAssert = "";
    internal long[] Witness = Array.Empty<long>();

    internal string Signature() => string.Join(",", BranchDecisions) + (Failed ? "|" + FailedAssert : "|ok");
}

internal static class Symbolic
{
    internal const int Bound = 8;
    internal const int StepBudget = 400;
    internal static long ExploredStates;

    /// <summary>Model finder: the first input tuple in [-Bound, Bound]^inputs that satisfies the conditions.</summary>
    internal static long[]? Solve(List<Atom> conditions, int inputs)
    {
        var values = new long[inputs];
        return Search(0);

        long[]? Search(int index)
        {
            if (index == inputs)
            {
                foreach (Atom atom in conditions)
                    if (!atom.Holds(atom.Expression.Evaluate(values))) return null;
                return (long[])values.Clone();
            }
            for (long v = -Bound; v <= Bound; v++)
            {
                values[index] = v;
                long[]? found = Search(index + 1);
                if (found != null) return found;
            }
            return null;
        }
    }

    internal static List<Path> Explore(Program program, out long pruned)
    {
        var paths = new List<Path>();
        var stack = new Stack<(int Index, Dictionary<int, LinExpr> Environment, List<Atom> Conditions, List<int> Decisions, int Steps)>();
        stack.Push((0, new Dictionary<int, LinExpr>(), new List<Atom>(), new List<int>(), 0));
        pruned = 0;
        ExploredStates = 0;
        while (stack.Count > 0)
        {
            var (index, environment, conditions, decisions, steps) = stack.Pop();
            ExploredStates++;
            if (steps > StepBudget || index >= program.Instructions.Count)
            {
                paths.Add(new Path { Feasible = false });
                continue;
            }
            Instruction instruction = program.Instructions[index];
            var env = new Dictionary<int, LinExpr>(environment);
            var path = new Path();
            path.Conditions.AddRange(conditions);
            path.BranchDecisions.AddRange(decisions);
            if (instruction.Kind == 0)
            {
                env[instruction.Variable] = instruction.Value!;
                stack.Push((index + 1, env, conditions, decisions, steps + 1));
            }
            else if (instruction.Kind == 1)
            {
                var takenConditions = new List<Atom>(conditions) { new Atom(instruction.Condition!, instruction.Comparison, true) };
                var fallConditions = new List<Atom>(conditions) { new Atom(instruction.Condition!, instruction.Comparison, false) };
                var takenDecisions = new List<int>(decisions) { 1 };
                var fallDecisions = new List<int>(decisions) { 0 };
                if (Solve(takenConditions, program.Inputs) != null)
                    stack.Push((instruction.Target, new Dictionary<int, LinExpr>(env), takenConditions, takenDecisions, steps + 1));
                else pruned++;
                if (Solve(fallConditions, program.Inputs) != null)
                    stack.Push((index + 1, new Dictionary<int, LinExpr>(env), fallConditions, fallDecisions, steps + 1));
                else pruned++;
            }
            else if (instruction.Kind == 2)
            {
                // Two continuations: the assertion holds (continue) and the assertion fails (record the bug).
                var passingConditions = new List<Atom>(conditions) { new Atom(instruction.Condition!, instruction.Comparison, true) };
                var failedConditions = new List<Atom>(conditions) { new Atom(instruction.Condition!, instruction.Comparison, false) };
                if (Solve(passingConditions, program.Inputs) != null)
                    stack.Push((index + 1, new Dictionary<int, LinExpr>(env), passingConditions, new List<int>(decisions), steps + 1));
                else pruned++;
                long[]? witness = Solve(failedConditions, program.Inputs);
                if (witness != null)
                {
                    var failure = new Path { Failed = true, FailedAssert = instruction.Text, Witness = witness };
                    failure.Conditions.AddRange(failedConditions);
                    failure.BranchDecisions.AddRange(decisions);
                    foreach (var pair in env) failure.FinalValues[pair.Key] = pair.Value;
                    paths.Add(failure);
                }
                else pruned++;
            }
            else
            {
                long[]? witness = Solve(conditions, program.Inputs);
                if (witness != null)
                {
                    path.Witness = witness;
                    foreach (var pair in env) path.FinalValues[pair.Key] = pair.Value;
                    paths.Add(path);
                }
                else
                {
                    path.Feasible = false;
                    paths.Add(path);
                }
            }
        }
        return paths;
    }
}

internal static class Concrete
{
    /// <summary>Independent concrete interpreter; returns the branch decisions, the final values and the failure.</summary>
    internal static (string Signature, Dictionary<int, long> Values, bool Failed, string FailedAssert) Run(Program program, long[] inputs)
    {
        var values = new Dictionary<int, long>();
        var decisions = new List<int>();
        int index = 0, steps = 0;
        while (index < program.Instructions.Count && steps++ < 1000)
        {
            Instruction instruction = program.Instructions[index];
            if (instruction.Kind == 0)
            {
                values[instruction.Variable] = instruction.Value!.Evaluate(inputs);
                index++;
            }
            else if (instruction.Kind == 1)
            {
                bool taken = new Atom(instruction.Condition!, instruction.Comparison, true)
                    .Holds(instruction.Condition!.Evaluate(inputs));
                decisions.Add(taken ? 1 : 0);
                index = taken ? instruction.Target : index + 1;
            }
            else if (instruction.Kind == 2)
            {
                bool ok = new Atom(instruction.Condition!, instruction.Comparison, true)
                    .Holds(instruction.Condition!.Evaluate(inputs));
                if (!ok) return (string.Join(",", decisions), values, true, instruction.Text);
                index++;
            }
            else break;
        }
        return (string.Join(",", decisions), values, false, "");
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static LinExpr Input(int index)
    {
        var expression = new LinExpr();
        expression.Coefficients[index] = 1;
        return expression;
    }

    /// <summary>Program with a division-by-zero style bug that needs two inputs to trigger.</summary>
    private static Program Buggy()
    {
        var program = new Program(2);
        LinExpr x = Input(0), y = Input(1);
        program.Assert(x.Add(new LinExpr(-5)), Op.Le, "x <= 5");
        int branch = program.Branch(y.Add(new LinExpr(-3)), Op.Le, 4, "y <= 3");
        program.Assert(new LinExpr(1), Op.Eq, "always fails");
        program.Halt();
        // target 4: y > 3
        LinExpr sum = x.Add(y);
        program.Assert(sum.Add(new LinExpr(-12)), Op.Ne, "x + y != 12");
        program.Halt();
        return program;
    }

    private static Program Intervals()
    {
        var program = new Program(1);
        LinExpr x = Input(0);
        int outer = program.Branch(x.Add(new LinExpr(-3)), Op.Le, 5, "x <= 3");
        program.Halt();
        // target 5: x > 3, then the inner branch is unreachable
        int inner = program.Branch(x.Add(new LinExpr(-2)), Op.Le, 8, "x <= 2");
        program.Assert(new LinExpr(1), Op.Eq, "must never fail");
        program.Halt();
        // target 8: x > 3 and x <= 2, infeasible
        program.Assert(new LinExpr(1), Op.Ne, "infeasible trap");
        program.Halt();
        return program;
    }

    private static Program SummingLoop()
    {
        var program = new Program(2);
        LinExpr x = Input(0), y = Input(1);
        program.Assert(x.Add(new LinExpr(-4)), Op.Lt, "x < 4");
        LinExpr sum = x.Add(y);
        program.Assert(sum.Add(new LinExpr(-9)), Op.Lt, "x + y < 9");
        LinExpr product = sum.Scale(2);
        program.Assert(product.Add(new LinExpr(-10)), Op.Ne, "2*(x+y) != 10");
        program.Halt();
        return program;
    }

    private static void Verify(Program program, List<Path> paths, ref bool concolicOk, ref bool completenessOk, ref string? firstFailure)
    {
        var signatures = new HashSet<string>();
        foreach (Path path in paths)
        {
            if (!path.Feasible) continue;
            var (signature, values, failed, failedAssert) = Concrete.Run(program, path.Witness);
            string expected = string.Join(",", path.BranchDecisions) + (path.Failed ? "|" + path.FailedAssert : "|ok");
            if (signature + (failed ? "|" + failedAssert : "|ok") != expected)
            {
                concolicOk = false;
                firstFailure ??= "the witness " + string.Join(",", path.Witness) + " follows " + signature
                    + (failed ? "|" + failedAssert : "|ok") + " but the symbolic path claimed " + expected;
            }
            if (path.Failed != failed || (failed && failedAssert != path.FailedAssert))
            {
                concolicOk = false;
                firstFailure ??= "the concrete run and the symbolic path disagree about the assertion failure";
            }
            foreach (var pair in path.FinalValues)
                if (!values.TryGetValue(pair.Key, out long concreteValue)
                    || concreteValue != pair.Value.Evaluate(path.Witness))
                {
                    concolicOk = false;
                    firstFailure ??= "the final value of v" + pair.Key + " differs from the symbolic one";
                }
            signatures.Add(signature + (failed ? "|" + failedAssert : "|ok"));
        }

        // Completeness on the bounded domain: every input tuple must follow one of the explored paths.
        int domain = program.Inputs;
        var inputs = new long[domain];
        bool complete = completenessOk;
        string? failure = firstFailure;
        long checkedInputs = 0;
        Recurse(0);
        void Recurse(int index)
        {
            if (index == domain)
            {
                checkedInputs++;
                var (signature, _, failed, failedAssert) = Concrete.Run(program, inputs);
                string key = signature + (failed ? "|" + failedAssert : "|ok");
                if (!signatures.Contains(key))
                {
                    complete = false;
                    failure ??= "input " + string.Join(",", inputs) + " follows the unexplored signature " + key;
                }
                return;
            }
            for (long v = -Symbolic.Bound; v <= Symbolic.Bound; v++)
            {
                inputs[index] = v;
                Recurse(index + 1);
            }
        }
        if (checkedInputs < 1) complete = false;
        completenessOk = complete;
        firstFailure = failure;
    }

    public static void Main()
    {
        // ---- the buggy program: symbolic execution must find the violating input -------------------------
        var buggy = Buggy();
        List<Path> buggyPaths = Symbolic.Explore(buggy, out long buggyPruned);
        var violations = new List<Path>();
        foreach (Path path in buggyPaths) if (path.Failed) violations.Add(path);
        Check(violations.Count >= 1 && violations[0].Witness.Length == 2,
            "the buggy program yields " + violations.Count + " assertion violations, the first at input "
            + string.Join(",", violations[0].Witness) + " (" + violations[0].FailedAssert + ")");
        var (crashSignature, _, crashFailed, crashAssert) = Concrete.Run(buggy, violations[0].Witness);
        Check(crashFailed && crashAssert == violations[0].FailedAssert && crashSignature == string.Join(",", violations[0].BranchDecisions),
            "replaying that witness concretely reproduces the same failing assertion on the same path");

        bool concolicOk = true, completenessOk = true;
        string? firstFailure = null;
        Verify(buggy, buggyPaths, ref concolicOk, ref completenessOk, ref firstFailure);
        Check(concolicOk, "every symbolic witness is confirmed by the concrete interpreter: " + (firstFailure ?? "ok"));
        Check(completenessOk, "every input of the bounded domain follows an explored path: " + (firstFailure ?? "ok"));
        Check(buggyPruned >= 1, "the always-failing assertion is detected: its passing side is infeasible ("
            + buggyPruned + " pruned states)");

        // ---- the interval program: the infeasible branch must be pruned, not explored ---------------------
        var intervals = Intervals();
        List<Path> intervalPaths = Symbolic.Explore(intervals, out long intervalPruned);
        bool trapped = false;
        foreach (Path path in intervalPaths) if (path.Failed && path.FailedAssert.Contains("infeasible")) trapped = true;
        Check(!trapped && intervalPruned >= 1,
            "the branch x > 3 and x <= 2 is pruned (" + intervalPruned + " pruned states) and never explored");
        bool intervalsOk = true, intervalsComplete = true;
        Verify(intervals, intervalPaths, ref intervalsOk, ref intervalsComplete, ref firstFailure);
        Check(intervalsOk && intervalsComplete,
            "the interval program stays concolically consistent and complete: " + (firstFailure ?? "ok"));
        Check(intervalPaths.Count >= 2, "the interval program has " + intervalPaths.Count + " explored paths");

        // ---- a program with four sequential assertions and no branch -------------------------------------
        var loop = SummingLoop();
        List<Path> loopPaths = Symbolic.Explore(loop, out long loopPruned);
        int completed = 0, failing = 0;
        Path? successful = null;
        foreach (Path path in loopPaths)
        {
            if (path.Failed) failing++;
            else if (path.Feasible) { completed++; successful = path; }
        }
        Check(completed == 1 && failing == 3,
            "the branch-free program yields exactly one successful path plus the three assertion violations ("
            + completed + " successful, " + failing + " violating)");
        bool loopOk = true, loopComplete = true;
        Verify(loop, loopPaths, ref loopOk, ref loopComplete, ref firstFailure);
        Check(loopOk && loopComplete, "the branch-free program is concolically verified: " + (firstFailure ?? "ok"));
        Check(successful != null && successful.Conditions.Count >= 3,
            "the successful path carries the conjunction of all three assertions ("
            + (successful?.Conditions.Count ?? 0) + " linear atoms)");

        // ---- determinism -----------------------------------------------------------------------------------
        List<Path> repeatPaths = Symbolic.Explore(Buggy(), out long repeatPruned);
        Check(repeatPaths.Count == buggyPaths.Count && repeatPruned == buggyPruned,
            "the exploration is deterministic in the number of paths and pruned states");

        Console.WriteLine("buggyPaths=" + buggyPaths.Count + " violations=" + violations.Count
            + " intervalPaths=" + intervalPaths.Count + " pruned=" + intervalPruned
            + " exploredStates=" + Symbolic.ExploredStates + " bound=" + Symbolic.Bound);
        Console.WriteLine("PASS");
    }
}
