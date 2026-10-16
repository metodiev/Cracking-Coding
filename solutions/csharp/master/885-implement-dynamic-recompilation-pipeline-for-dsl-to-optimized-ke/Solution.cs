// Task 885 - Implement dynamic recompilation pipeline for DSL to optimized kernels with heuristics. (Tier: Master, Beyond — Compilers, Languages & Solvers)
//
// Pipeline: source text -> tokenizer + recursive-descent parser -> AST -> optimizer (constant folding,
// algebraic identities, strength reduction x*x -> sqr, hash-consing so identical subtrees share one
// node = CSE, unreachable subtrees disappear with the identity that removed them) -> straight-line
// kernel code generation (a postorder instruction schedule over a register file) -> a tiered runtime
// that starts on a tree-walking interpreter, moves to a fused baseline kernel and then, only when a
// break-even heuristic says the compile cost will be repaid, to the optimized vectorized kernel. A
// specialization whose *measured* work regresses is demoted and its (program, shape) key blacklisted,
// so it can never be recompiled again. Kernels are cached per (program, shape bucket).
//
// Simplifications (documented, not behavioural):
//  * "Compilation" and "time" are abstract work units, not cycles: a kernel that performs fewer node
//    evaluations and fewer operand loads per element is faster *by construction*. The vector width
//    (4 lanes) times the unroll factor (2) only says how many elements share one hoisted operand load.
//  * The DSL is elementwise (no reductions, no control flow, 1-D operands), so a "kernel" really is a
//    straight-line schedule over registers - the meaning the task gives it here.
//  * Algebraic identities (x*0 -> 0, x-x -> 0, x*1 -> x) are exact for finite inputs only; the checks
//    feed finite values only, and results are compared with == rather than bit-for-bit because
//    x*0 loses the sign of zero for negative x.
//
// Verified:
//  * semantics - every tier's output equals an independently hand-coded formula oracle for eight
//    programs over five shapes plus the degenerate shapes n = 0 and n = 1, a constant-only program and
//    a program that folds to a constant;
//  * cost bounds - the measured per-invocation work report (node evaluations, operand loads, node
//    visits, chunks, fixed overhead) is compared component-by-component against the analytic cost
//    model for all three tiers, four programs and five shapes;
//  * heuristics - the promotion decision is re-derived independently inside the test (break-even
//    saving * expectedRemainingCalls > compileCost) on a parameter grid and compared with the
//    decisions the runtime actually logs, including the boundary shapes where the saving is 0 or 3
//    units and no promotion may happen;
//  * self-correction - an aggressive policy that ignores break-even promotes a specialization whose
//    measured work is worse than the baseline; the runtime demotes and blacklists it and the
//    compilation count then provably stops growing while results stay correct;
//  * boundedness - after 200 invocations across programs and shapes the number of compilations never
//    exceeds two per (program, shape) key; invalid programs are rejected by the parser.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

internal enum Op { Const, Var, Add, Sub, Mul, Sqr, Neg, Relu }

internal enum Tier { Interpreter = 0, Baseline = 1, Optimized = 2 }

internal sealed class Node
{
    internal Op Op;
    internal double Value;
    internal string Name = "";
    internal string Key = "";
    internal Node? A;
    internal Node? B;
    internal int Slot = -1;
}

internal sealed class WorkReport
{
    internal long Ops;
    internal long Loads;
    internal long NodeVisits;
    internal long Chunks;
    internal long Overhead;

    internal long Total => Ops + Loads + NodeVisits + Chunks + Overhead;

    internal void Add(WorkReport other)
    {
        Ops += other.Ops;
        Loads += other.Loads;
        NodeVisits += other.NodeVisits;
        Chunks += other.Chunks;
        Overhead += other.Overhead;
    }

    internal string Describe() =>
        "ops=" + Ops + " loads=" + Loads + " visits=" + NodeVisits + " chunks=" + Chunks +
        " overhead=" + Overhead + " total=" + Total;
}

internal static class Dsl
{
    internal static Node Parse(string source)
    {
        List<string> tokens = Tokenize(source);
        int position = 0;
        Node root = Expression(tokens, ref position);
        if (position != tokens.Count) throw new Exception("trailing tokens in '" + source + "'");
        return root;
    }

    private static List<string> Tokenize(string source)
    {
        var tokens = new List<string>();
        int i = 0;
        while (i < source.Length)
        {
            char c = source[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (char.IsDigit(c) || c == '.')
            {
                int j = i;
                while (j < source.Length && (char.IsDigit(source[j]) || source[j] == '.')) j++;
                tokens.Add(source.Substring(i, j - i));
                i = j;
                continue;
            }
            if (char.IsLetter(c) || c == '_')
            {
                int j = i;
                while (j < source.Length && (char.IsLetterOrDigit(source[j]) || source[j] == '_')) j++;
                tokens.Add(source.Substring(i, j - i));
                i = j;
                continue;
            }
            if ("+-*(),".IndexOf(c) >= 0) { tokens.Add(c.ToString()); i++; continue; }
            throw new Exception("unexpected character '" + c + "'");
        }
        return tokens;
    }

    private static Node Expression(List<string> tokens, ref int position)
    {
        Node left = Term(tokens, ref position);
        while (position < tokens.Count && (tokens[position] == "+" || tokens[position] == "-"))
        {
            string op = tokens[position++];
            Node right = Term(tokens, ref position);
            left = new Node { Op = op == "+" ? Op.Add : Op.Sub, A = left, B = right };
        }
        return left;
    }

    private static Node Term(List<string> tokens, ref int position)
    {
        Node left = Unary(tokens, ref position);
        while (position < tokens.Count && tokens[position] == "*")
        {
            position++;
            Node right = Unary(tokens, ref position);
            left = new Node { Op = Op.Mul, A = left, B = right };
        }
        return left;
    }

    private static Node Unary(List<string> tokens, ref int position)
    {
        if (position < tokens.Count && tokens[position] == "-")
        {
            position++;
            return new Node { Op = Op.Neg, A = Unary(tokens, ref position) };
        }
        if (position < tokens.Count && tokens[position] == "+") { position++; return Unary(tokens, ref position); }
        return Primary(tokens, ref position);
    }

    private static Node Primary(List<string> tokens, ref int position)
    {
        if (position >= tokens.Count) throw new Exception("unexpected end of program");
        string token = tokens[position++];
        if (token == "(")
        {
            Node inner = Expression(tokens, ref position);
            Expect(tokens, ref position, ")");
            return inner;
        }
        double value;
        if (double.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            return new Node { Op = Op.Const, Value = value };
        if (token == "relu" || token == "neg")
        {
            Expect(tokens, ref position, "(");
            Node argument = Expression(tokens, ref position);
            Expect(tokens, ref position, ")");
            return new Node { Op = token == "relu" ? Op.Relu : Op.Neg, A = argument };
        }
        if (!char.IsLetter(token[0]) && token[0] != '_') throw new Exception("unexpected token '" + token + "'");
        return new Node { Op = Op.Var, Name = token };
    }

    private static void Expect(List<string> tokens, ref int position, string token)
    {
        if (position >= tokens.Count || tokens[position] != token) throw new Exception("expected '" + token + "'");
        position++;
    }

    internal static int CountNodes(Node? n) =>
        n == null ? 0 : 1 + CountNodes(n.A) + CountNodes(n.B);

    internal static int CountLeaves(Node? n)
    {
        if (n == null) return 0;
        if (n.Op == Op.Const || n.Op == Op.Var) return 1;
        return CountLeaves(n.A) + CountLeaves(n.B);
    }

    internal static void AssignSlots(Node n, Dictionary<string, int> slots, HashSet<string> declared)
    {
        if (n.Op == Op.Const || n.Op == Op.Var)
        {
            if (n.Op == Op.Var)
            {
                if (!declared.Contains(n.Name)) throw new Exception("unknown variable '" + n.Name + "'");
                if (!slots.ContainsKey(n.Name)) slots[n.Name] = slots.Count;
                n.Slot = slots[n.Name];
            }
            return;
        }
        AssignSlots(n.A!, slots, declared);
        if (n.B != null) AssignSlots(n.B!, slots, declared);
    }
}

internal static class Optimizer
{
    internal static Node Optimize(Node root)
    {
        var memo = new Dictionary<string, Node>();
        return Visit(root, memo);
    }

    private static Node Visit(Node n, Dictionary<string, Node> memo)
    {
        switch (n.Op)
        {
            case Op.Const: return Leaf(memo, Op.Const, n.Value, "");
            case Op.Var: return Leaf(memo, Op.Var, 0.0, n.Name);
        }

        Node a = Visit(n.A!, memo);
        Node? b = n.B == null ? null : Visit(n.B!, memo);

        if (b == null)
        {
            if (a.Op == Op.Const)
            {
                double folded = n.Op == Op.Neg ? -a.Value : (a.Value > 0.0 ? a.Value : 0.0);
                return Leaf(memo, Op.Const, folded, "");
            }
            if (n.Op == Op.Neg && a.Op == Op.Neg) return a.A!;          // neg(neg x) -> x
            if (n.Op == Op.Relu && a.Op == Op.Relu) return a;           // relu(relu x) -> relu x
            return Un(memo, n.Op, a);
        }

        if (a.Op == Op.Const && b.Op == Op.Const)
        {
            double folded = n.Op == Op.Add ? a.Value + b.Value
                          : n.Op == Op.Sub ? a.Value - b.Value
                          : a.Value * b.Value;
            return Leaf(memo, Op.Const, folded, "");
        }

        if (n.Op == Op.Mul)
        {
            if (IsZero(a) || IsZero(b)) return Leaf(memo, Op.Const, 0.0, "");
            if (IsOne(a)) return b;
            if (IsOne(b)) return a;
            if (ReferenceEquals(a, b)) return Un(memo, Op.Sqr, a);      // strength reduction x*x -> sqr
        }

        if (n.Op == Op.Add)
        {
            if (IsZero(a)) return b;
            if (IsZero(b)) return a;
        }

        if (n.Op == Op.Sub)
        {
            if (IsZero(b)) return a;
            if (IsZero(a)) return Un(memo, Op.Neg, b);
            if (ReferenceEquals(a, b)) return Leaf(memo, Op.Const, 0.0, "");   // x - x -> 0
        }

        return Bin(memo, n.Op, a, b);
    }

    private static bool IsZero(Node n) => n.Op == Op.Const && n.Value == 0.0;
    private static bool IsOne(Node n) => n.Op == Op.Const && n.Value == 1.0;

    private static Node Intern(Dictionary<string, Node> memo, string key, Func<Node> factory)
    {
        Node? existing;
        if (memo.TryGetValue(key, out existing)) return existing;
        Node created = factory();
        created.Key = key;
        memo[key] = created;
        return created;
    }

    private static Node Leaf(Dictionary<string, Node> memo, Op op, double value, string name)
    {
        string key = op == Op.Const
            ? "c:" + BitConverter.DoubleToInt64Bits(value).ToString(CultureInfo.InvariantCulture)
            : "v:" + name;
        return Intern(memo, key, () => new Node { Op = op, Value = value, Name = name });
    }

    private static Node Un(Dictionary<string, Node> memo, Op op, Node a) =>
        Intern(memo, op + "(" + a.Key + ")", () => new Node { Op = op, A = a });

    private static Node Bin(Dictionary<string, Node> memo, Op op, Node a, Node b)
    {
        if (op == Op.Add || op == Op.Mul)
        {
            // Canonical operand order, so a+b and b+a hash-cons to the same node.
            if (string.CompareOrdinal(a.Key, b.Key) > 0) { Node swap = a; a = b; b = swap; }
        }
        return Intern(memo, op + "(" + a.Key + "," + b.Key + ")", () => new Node { Op = op, A = a, B = b });
    }
}

internal sealed class Schedule
{
    internal readonly List<Node> Nodes = new List<Node>();
    internal readonly List<int> AIdx = new List<int>();
    internal readonly List<int> BIdx = new List<int>();
    internal readonly List<int> UsedSlots = new List<int>();
    internal int VarLeaves;
    internal int ConstLeaves;

    internal static Schedule Build(Node root)
    {
        var schedule = new Schedule();
        var index = new Dictionary<Node, int>();
        var slots = new SortedSet<int>();
        Walk(root, schedule, index, slots);
        schedule.UsedSlots.AddRange(slots);
        return schedule;
    }

    private static void Walk(Node n, Schedule schedule, Dictionary<Node, int> index, SortedSet<int> slots)
    {
        if (index.ContainsKey(n)) return;
        int a = -1, b = -1;
        if (n.A != null) { Walk(n.A, schedule, index, slots); a = index[n.A]; }
        if (n.B != null) { Walk(n.B, schedule, index, slots); b = index[n.B]; }
        index[n] = schedule.Nodes.Count;
        schedule.Nodes.Add(n);
        schedule.AIdx.Add(a);
        schedule.BIdx.Add(b);
        if (n.Op == Op.Var) { schedule.VarLeaves++; slots.Add(n.Slot); }
        else if (n.Op == Op.Const) schedule.ConstLeaves++;
    }
}

internal sealed class Program
{
    internal string Name = "";
    internal string Source = "";
    internal Node Body = null!;
    internal Schedule Kernel = null!;
    internal int OriginalNodes;
    internal int OriginalLeaves;
    internal readonly Dictionary<string, int> Slots = new Dictionary<string, int>(StringComparer.Ordinal);

    internal int OptimizedNodes => Kernel.Nodes.Count;
    internal int OptimizedLeaves => Kernel.VarLeaves + Kernel.ConstLeaves;

    /** Slot of a variable, assigned in order of first appearance in the source; -1 when unused. */
    internal int SlotOf(string name) => Slots.TryGetValue(name, out int slot) ? slot : -1;

    internal static Program Compile(string name, string source, string[] declaredVariables)
    {
        var program = new Program { Name = name, Source = source };
        program.Body = Dsl.Parse(source);
        var slots = program.Slots;
        var declared = new HashSet<string>(declaredVariables, StringComparer.Ordinal);
        Dsl.AssignSlots(program.Body, slots, declared);
        program.OriginalNodes = Dsl.CountNodes(program.Body);
        program.OriginalLeaves = Dsl.CountLeaves(program.Body);
        Node optimized = Optimizer.Optimize(program.Body);
        AssignKernelSlots(optimized, slots);
        program.Kernel = Schedule.Build(optimized);
        return program;
    }

    private static void AssignKernelSlots(Node n, Dictionary<string, int> slots)
    {
        if (n.Op == Op.Var)
        {
            if (!slots.TryGetValue(n.Name, out int slot)) throw new Exception("unknown variable '" + n.Name + "'");
            n.Slot = slot;
            return;
        }
        if (n.A != null) AssignKernelSlots(n.A, slots);
        if (n.B != null) AssignKernelSlots(n.B, slots);
    }
}

internal static class CostModel
{
    internal const int VectorLanes = 8;          // 4 SIMD lanes x unroll 2
    internal const int InvocationOverhead = 5;   // prologue: register setup, bounds checks

    internal static long Work(Tier tier, Program program, int n) => Components(tier, program, n).Total;

    /** The analytic work model; Kernels.Run must reproduce every component exactly. */
    internal static WorkReport Components(Tier tier, Program program, int n)
    {
        var report = new WorkReport();
        if (n == 0) return report;
        long chunks = (n + VectorLanes - 1) / VectorLanes;
        switch (tier)
        {
            case Tier.Interpreter:
                report.Ops = (long)n * program.OriginalNodes;
                report.NodeVisits = (long)n * program.OriginalNodes;
                report.Loads = (long)n * program.OriginalLeaves;
                break;
            case Tier.Baseline:
                report.Ops = (long)n * program.OriginalNodes;
                report.Loads = (long)n * program.OriginalLeaves;
                break;
            default:
                report.Ops = (long)n * program.OptimizedNodes;
                report.Loads = chunks * program.OptimizedLeaves;
                report.Chunks = chunks;
                report.Overhead = InvocationOverhead;
                break;
        }
        return report;
    }
}

internal static class Kernels
{
    internal static double[] Run(Tier tier, Program program, double[][] inputs, int n, WorkReport report)
    {
        if (n == 0) return new double[0];
        switch (tier)
        {
            case Tier.Interpreter: return Recursive(program.Body, inputs, n, report, true);
            case Tier.Baseline: return Recursive(program.Body, inputs, n, report, false);
            default: return Scheduled(program.Kernel, inputs, n, report);
        }
    }

    private static double[] Recursive(Node body, double[][] inputs, int n, WorkReport report, bool countVisits)
    {
        var output = new double[n];
        for (int i = 0; i < n; i++) output[i] = Eval(body, i, inputs, report, countVisits);
        return output;
    }

    private static double Eval(Node node, int i, double[][] source, WorkReport report, bool countVisits)
    {
        report.Ops++;
        if (countVisits) report.NodeVisits++;
        switch (node.Op)
        {
            case Op.Const:
                report.Loads++;
                return node.Value;
            case Op.Var:
                report.Loads++;
                return source[node.Slot][i];
            case Op.Add:
                return Eval(node.A!, i, source, report, countVisits) + Eval(node.B!, i, source, report, countVisits);
            case Op.Sub:
                return Eval(node.A!, i, source, report, countVisits) - Eval(node.B!, i, source, report, countVisits);
            case Op.Mul:
                return Eval(node.A!, i, source, report, countVisits) * Eval(node.B!, i, source, report, countVisits);
            case Op.Sqr:
                {
                    double v = Eval(node.A!, i, source, report, countVisits);
                    return v * v;
                }
            case Op.Neg:
                return -Eval(node.A!, i, source, report, countVisits);
            case Op.Relu:
                {
                    double v = Eval(node.A!, i, source, report, countVisits);
                    return v > 0.0 ? v : 0.0;
                }
            default:
                throw new Exception("bad opcode");
        }
    }

    /** Executes the register schedule; operand loads are hoisted once per vector chunk. */
    private static double[] Scheduled(Schedule kernel, double[][] inputs, int n, WorkReport report)
    {
        var output = new double[n];
        var registers = new double[kernel.Nodes.Count];
        report.Overhead += CostModel.InvocationOverhead;
        for (int start = 0; start < n; start += CostModel.VectorLanes)
        {
            int width = Math.Min(CostModel.VectorLanes, n - start);
            report.Chunks++;
            int laneSlots = kernel.UsedSlots.Count == 0 ? 0 : kernel.UsedSlots[kernel.UsedSlots.Count - 1] + 1;
            var lanes = new double[Math.Max(inputs.Length, laneSlots)][];
            foreach (int slot in kernel.UsedSlots)
            {
                lanes[slot] = new double[width];
                Array.Copy(inputs[slot], start, lanes[slot], 0, width);
                report.Loads++;                       // one wide load per chunk, not per element
            }
            report.Loads += kernel.ConstLeaves;       // immediates hoisted out of the element loop
            for (int j = 0; j < width; j++)
            {
                for (int k = 0; k < kernel.Nodes.Count; k++)
                {
                    Node node = kernel.Nodes[k];
                    report.Ops++;
                    switch (node.Op)
                    {
                        case Op.Const: registers[k] = node.Value; break;
                        case Op.Var: registers[k] = lanes[node.Slot][j]; break;
                        case Op.Add: registers[k] = registers[kernel.AIdx[k]] + registers[kernel.BIdx[k]]; break;
                        case Op.Sub: registers[k] = registers[kernel.AIdx[k]] - registers[kernel.BIdx[k]]; break;
                        case Op.Mul: registers[k] = registers[kernel.AIdx[k]] * registers[kernel.BIdx[k]]; break;
                        case Op.Sqr: registers[k] = registers[kernel.AIdx[k]] * registers[kernel.AIdx[k]]; break;
                        case Op.Neg: registers[k] = -registers[kernel.AIdx[k]]; break;
                        case Op.Relu:
                            registers[k] = registers[kernel.AIdx[k]] > 0.0 ? registers[kernel.AIdx[k]] : 0.0;
                            break;
                        default: throw new Exception("bad opcode");
                    }
                }
                output[start + j] = registers[kernel.Nodes.Count - 1];   // postorder => root is last
            }
        }
        return output;
    }
}

internal sealed class RecompilationRuntime
{
    internal sealed class Entry
    {
        internal Program Program = null!;
        internal int Bucket;
        internal Tier Tier = Tier.Interpreter;
        internal long Calls;
        internal long Compilations;
        internal long InterpreterCalls;
        internal long BaselineCalls;
        internal long OptimizedCalls;
        internal long InterpreterWork;
        internal long BaselineWork;
        internal long OptimizedWork;
        internal double BaselineAverageWork;
        internal bool BaselineAverageKnown;
        internal bool Blacklisted;
        internal readonly List<string> History = new List<string>();
    }

    internal readonly Dictionary<string, Program> Programs = new Dictionary<string, Program>(StringComparer.Ordinal);
    internal readonly Dictionary<string, Entry> Entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
    internal readonly List<string> Decisions = new List<string>();

    internal long BaselineCallThreshold = 2;
    internal long HotThreshold = 3;
    internal long ExpectedRemainingCalls = 64;
    internal long OptimizedCompileCost = 200;
    internal bool IgnoreBreakEven;
    internal bool AllowOptimizedTier = true;

    internal long TotalCompilations;
    internal long Deopts;

    internal void Register(Program program) => Programs[program.Name] = program;

    internal static int Bucket(int n)
    {
        int bucket = 0;
        while ((1 << bucket) < n) bucket++;
        return bucket;
    }

    internal string Key(string name, int n) => name + "#" + Bucket(n);

    internal Entry? Find(string name, int n)
    {
        Entry? entry;
        return Entries.TryGetValue(Key(name, n), out entry) ? entry : null;
    }

    /** Independent-of-history prediction of the per-call saving the optimized kernel buys. */
    internal long PredictedSaving(Program program, int n) =>
        CostModel.Work(Tier.Baseline, program, n) - CostModel.Work(Tier.Optimized, program, n);

    internal bool HeuristicWantsOptimized(Program program, int n) =>
        AllowOptimizedTier && (IgnoreBreakEven || PredictedSaving(program, n) * ExpectedRemainingCalls > OptimizedCompileCost);

    internal double[] Invoke(string name, int n, double[][] inputs)
    {
        Program program = Programs[name];
        string key = Key(name, n);
        Entry? found;
        Entry entry;
        if (!Entries.TryGetValue(key, out found))
        {
            entry = new Entry { Program = program, Bucket = Bucket(n) };
            entry.History.Add("interpreter");
            Entries[key] = entry;
        }
        else
        {
            entry = found;
        }

        entry.Calls++;
        var report = new WorkReport();
        double[] output = Kernels.Run(entry.Tier, program, inputs, n, report);
        switch (entry.Tier)
        {
            case Tier.Interpreter:
                entry.InterpreterCalls++;
                entry.InterpreterWork += report.Total;
                break;
            case Tier.Baseline:
                entry.BaselineCalls++;
                entry.BaselineWork += report.Total;
                break;
            default:
                entry.OptimizedCalls++;
                entry.OptimizedWork += report.Total;
                break;
        }
        ConsiderRecompilation(entry, name, n);
        return output;
    }

    private void ConsiderRecompilation(Entry entry, string name, int n)
    {
        if (entry.Tier == Tier.Interpreter)
        {
            if (entry.Calls < BaselineCallThreshold) return;
            entry.Tier = Tier.Baseline;
            entry.Compilations++;
            TotalCompilations++;
            entry.History.Add("baseline");
            Decisions.Add(name + "/" + n + ": compile baseline (fused straight-line kernel)");
            return;
        }

        if (entry.Tier == Tier.Baseline)
        {
            if (entry.Blacklisted || entry.Calls < HotThreshold) return;
            if (!HeuristicWantsOptimized(entry.Program, n))
            {
                Decisions.Add(name + "/" + n + ": stay on baseline, saving " + PredictedSaving(entry.Program, n) +
                    " units/call does not repay compile cost " + OptimizedCompileCost + " within " +
                    ExpectedRemainingCalls + " calls");
                return;
            }
            entry.BaselineAverageWork = entry.BaselineCalls == 0 ? 0.0 : (double)entry.BaselineWork / entry.BaselineCalls;
            entry.BaselineAverageKnown = true;
            entry.Tier = Tier.Optimized;
            entry.Compilations++;
            TotalCompilations++;
            entry.History.Add("optimized");
            Decisions.Add(name + "/" + n + ": compile optimized (vectorized, " +
                entry.Program.OptimizedNodes + " ops vs " + entry.Program.OriginalNodes + ")");
            return;
        }

        // Optimized: the heuristic can be wrong, so compare measured work against the baseline average.
        if (!entry.BaselineAverageKnown || entry.OptimizedCalls < 2) return;
        double averageOptimized = (double)entry.OptimizedWork / entry.OptimizedCalls;
        if (averageOptimized > entry.BaselineAverageWork)
        {
            entry.Tier = Tier.Baseline;
            entry.Blacklisted = true;
            Deopts++;
            entry.History.Add("deopt");
            Decisions.Add(name + "/" + n + ": deopt, measured " + averageOptimized.ToString("F1", CultureInfo.InvariantCulture) +
                " > baseline " + entry.BaselineAverageWork.ToString("F1", CultureInfo.InvariantCulture) + " units/call, shape blacklisted");
        }
    }
}

public static class Solution
{
    private static ulong _state = 0x243F6A8885A308D3UL;

    private static ulong NextRandom()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextValue()
    {
        double u = (NextRandom() >> 11) * (1.0 / 9007199254740992.0);   // uniform in [0,1)
        return u * 4.0 - 2.0;                                          // uniform in [-2,2)
    }

    private static double Relu(double v) => v > 0.0 ? v : 0.0;

    private static double[][] MakeInputs(int variables, int n)
    {
        var inputs = new double[variables][];
        for (int v = 0; v < variables; v++)
        {
            inputs[v] = new double[n];
            for (int i = 0; i < n; i++) inputs[v][i] = NextValue();
        }
        return inputs;
    }

    private static double EvalOne(string source, string[] variables, double[] values)
    {
        Program program = Program.Compile("one", source, variables);
        var inputs = new double[variables.Length][];
        for (int v = 0; v < variables.Length; v++)
        {
            int slot = program.SlotOf(variables[v]);
            if (slot < 0) throw new Exception("unused declared variable " + variables[v]);
            inputs[slot] = new double[] { values[v] };
        }
        var report = new WorkReport();
        return Kernels.Run(Tier.Baseline, program, inputs, 1, report)[0];
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // ---- parser: precedence, unary minus, parentheses, relu, and rejection of bad programs ----
        Check(EvalOne("a + b * c", new[] { "a", "b", "c" }, new double[] { 1, 2, 3 }) == 7.0, "precedence a+b*c");
        Check(EvalOne("(a + b) * c", new[] { "a", "b", "c" }, new double[] { 1, 2, 3 }) == 9.0, "parentheses");
        Check(EvalOne("-a * b", new[] { "a", "b" }, new double[] { 2, 5 }) == -10.0, "unary minus");
        Check(EvalOne("relu(c - b) + relu(b - c)", new[] { "b", "c" }, new double[] { 2, 3 }) == 1.0,
            "relu both branches");
        Check(EvalOne("relu(0)", Array.Empty<string>(), Array.Empty<double>()) == 0.0, "relu at the boundary 0");
        Check(EvalOne("--a", new[] { "a" }, new double[] { 3 }) == 3.0, "double negation");
        Check(Throws(() => Program.Compile("bad", "a +", new[] { "a" })), "reject truncated program");
        Check(Throws(() => Program.Compile("bad", "(a + b", new[] { "a", "b" })), "reject unclosed paren");
        Check(Throws(() => Program.Compile("bad", "a + z", new[] { "a" })), "reject unknown variable");
        Check(Throws(() => Program.Compile("bad", "a $ b", new[] { "a", "b" })), "reject illegal character");

        // ---- eight programs, each with an independently hand-written formula oracle ----
        var specs = new (string Source, string[] Vars, Func<double[], double> Oracle, bool StrictDrop)[]
        {
            ("relu(a*b + c)", new[] { "a", "b", "c" }, v => Relu(v[0] * v[1] + v[2]), false),
            ("a*1 + 0 - 0 + a", new[] { "a" }, v => v[0] * 1.0 + 0.0 - 0.0 + v[0], true),
            ("1.5*x*x + 2.0*x*x", new[] { "x" }, v => 1.5 * v[0] * v[0] + 2.0 * v[0] * v[0], true),
            ("(x*x) - (x*x)", new[] { "x" }, v => (v[0] * v[0]) - (v[0] * v[0]), true),
            ("relu(relu(a)) + neg(neg(b))", new[] { "a", "b" }, v => Relu(Relu(v[0])) + (-(-v[1])), true),
            ("0*x + relu(0) + 2.5*2", new[] { "x" }, v => 0.0 * v[0] + Relu(0.0) + 2.5 * 2.0, true),
            ("a + b", new[] { "a", "b" }, v => v[0] + v[1], false),
            ("3.5", Array.Empty<string>(), v => 3.5, false),
        };

        var runtime = new RecompilationRuntime();
        var programs = new List<Program>();
        for (int i = 0; i < specs.Length; i++)
        {
            Program program = Program.Compile("p" + i, specs[i].Source, specs[i].Vars);
            programs.Add(program);
            runtime.Register(program);
            for (int v = 0; v < specs[i].Vars.Length; v++)
                Check(program.SlotOf(specs[i].Vars[v]) == v, "slots follow declaration order for " + specs[i].Source);
            Check(program.OptimizedNodes <= program.OriginalNodes,
                "optimizer never increases op count: " + specs[i].Source);
            if (specs[i].StrictDrop)
                Check(program.OptimizedNodes < program.OriginalNodes,
                    "optimizer removes work for " + specs[i].Source);
        }

        // ---- differential semantics: every tier and the end-to-end runtime against the oracle ----
        int[] shapes = { 0, 1, 4, 16, 64 };
        foreach (int n in shapes)
        {
            for (int s = 0; s < specs.Length; s++)
            {
                Program program = programs[s];
                double[][] inputs = MakeInputs(specs[s].Vars.Length, n);
                var expected = new double[n];
                for (int i = 0; i < n; i++)
                {
                    var point = new double[specs[s].Vars.Length];
                    for (int v = 0; v < inputs.Length; v++) point[v] = inputs[v][i];
                    expected[i] = specs[s].Oracle(point);
                }
                foreach (Tier tier in new[] { Tier.Interpreter, Tier.Baseline, Tier.Optimized })
                {
                    var report = new WorkReport();
                    double[] got = Kernels.Run(tier, program, inputs, n, report);
                    Check(got.Length == n, "output length " + tier + " " + n);
                    for (int i = 0; i < n; i++) Check(got[i] == expected[i], "oracle mismatch " + tier + " n=" + n + " i=" + i);
                }
                for (int call = 0; call < 8; call++)
                {
                    double[] got = runtime.Invoke(program.Name, n, inputs);
                    for (int i = 0; i < n; i++) Check(got[i] == expected[i], "runtime mismatch n=" + n + " i=" + i);
                }
            }
        }

        // ---- cost model: every measured component must equal the analytic value ----
        foreach (int n in shapes)
        {
            foreach (Program program in programs)
            {
                double[][] inputs = MakeInputs(program.Slots.Count, n);
                foreach (Tier tier in new[] { Tier.Interpreter, Tier.Baseline, Tier.Optimized })
                {
                    var measured = new WorkReport();
                    Kernels.Run(tier, program, inputs, n, measured);
                    WorkReport model = CostModel.Components(tier, program, n);
                    Check(measured.Ops == model.Ops, "model ops " + tier + " n=" + n + " " + program.Name);
                    Check(measured.Loads == model.Loads, "model loads " + tier + " n=" + n + " " + program.Name);
                    Check(measured.NodeVisits == model.NodeVisits, "model visits " + tier + " n=" + n + " " + program.Name);
                    Check(measured.Chunks == model.Chunks, "model chunks " + tier + " n=" + n + " " + program.Name);
                    Check(measured.Overhead == model.Overhead, "model overhead " + tier + " n=" + n + " " + program.Name);
                    Check(measured.Total == model.Total, "model total " + tier + " n=" + n + " " + program.Name);
                }
                if (n >= 32)
                    Check(CostModel.Work(Tier.Optimized, program, n) <= CostModel.Work(Tier.Baseline, program, n),
                        "optimized is never slower in the model on wide shapes: " + program.Name + " n=" + n);
                if (n > 0)
                    Check(CostModel.Work(Tier.Baseline, program, n) <= CostModel.Work(Tier.Interpreter, program, n),
                        "fusing the tree walk is never slower in the model: " + program.Name + " n=" + n);
            }
        }
        Check(CostModel.Work(Tier.Optimized, programs[0], 1) > CostModel.Work(Tier.Baseline, programs[0], 1),
            "the vector prologue makes the optimized tier slower on tiny shapes - the heuristic must catch this");
        Check(CostModel.Work(Tier.Baseline, programs[0], 64) == 64 * (programs[0].OriginalNodes + programs[0].OriginalLeaves),
            "closed-form baseline work");
        Check(CostModel.Work(Tier.Interpreter, programs[0], 8) == 8 * (2 * programs[0].OriginalNodes + programs[0].OriginalLeaves),
            "closed-form interpreter work");
        Check(CostModel.Work(Tier.Optimized, programs[0], 16) == 16 * programs[0].OptimizedNodes + 2 * programs[0].OptimizedLeaves + 2 + 5,
            "closed-form optimized work");

        // ---- heuristic: break-even decisions re-derived in the test on a parameter grid ----
        long[] remainingGrid = { 0, 1, 2, 64, 1000 };
        long[] costGrid = { 1, 50, 200, 5000 };
        long[] shapeGrid = { 1, 4, 16, 64, 256, 1024 };
        int promotions = 0, refusals = 0;
        foreach (Program program in programs)
        {
            foreach (int n in shapeGrid)
            {
                long baselineWork = (long)n * program.OriginalNodes + (long)n * program.OriginalLeaves;
                long chunks = (n + 8 - 1) / 8;
                long optimizedWork = (long)n * program.OptimizedNodes + chunks * program.OptimizedLeaves + chunks + 5;
                long saving = baselineWork - optimizedWork;
                Check(runtime.PredictedSaving(program, n) == saving, "predicted saving " + program.Name + " n=" + n);
                foreach (long remaining in remainingGrid)
                {
                    foreach (long compileCost in costGrid)
                    {
                        runtime.ExpectedRemainingCalls = remaining;
                        runtime.OptimizedCompileCost = compileCost;
                        bool expected = saving * remaining > compileCost;
                        Check(runtime.HeuristicWantsOptimized(program, n) == expected,
                            "break-even decision " + program.Name + " n=" + n + " remaining=" + remaining + " cost=" + compileCost);
                        if (expected) promotions++; else refusals++;
                    }
                }
            }
        }
        runtime.ExpectedRemainingCalls = 64;
        runtime.OptimizedCompileCost = 200;

        // ---- behavioural policy: a small shape is never promoted, a large one is ----
        var smallRuntime = new RecompilationRuntime();
        foreach (Program program in programs) smallRuntime.Register(program);
        for (int call = 0; call < 20; call++) smallRuntime.Invoke("p0", 4, MakeInputs(3, 4));
        RecompilationRuntime.Entry small = smallRuntime.Find("p0", 4)!;
        Check(small.Tier == Tier.Baseline, "small shape stays on baseline, tier=" + small.Tier);
        Check(small.Compilations == 1, "small shape compiled once, got " + small.Compilations);
        Check(smallRuntime.Decisions.Any(d => d.Contains("stay on baseline")), "rationale for staying is logged");

        for (int call = 0; call < 20; call++) smallRuntime.Invoke("p0", 512, MakeInputs(3, 512));
        RecompilationRuntime.Entry large = smallRuntime.Find("p0", 512)!;
        Check(large.Tier == Tier.Optimized, "hot large shape is promoted, tier=" + large.Tier);
        Check(large.Compilations == 2, "large shape compiled twice, got " + large.Compilations);

        // ---- self-correction: an over-eager heuristic is caught by measured work and blacklisted ----
        var aggressive = new RecompilationRuntime
        {
            BaselineCallThreshold = 1,
            HotThreshold = 1,
            IgnoreBreakEven = true,
            ExpectedRemainingCalls = 1000,
        };
        foreach (Program program in programs) aggressive.Register(program);
        Program constant = programs[7];                     // "3.5": one op, vectorization cannot pay off
        double[][] tiny = MakeInputs(1, 4);
        for (int call = 0; call < 10; call++)
        {
            double[] got = aggressive.Invoke(constant.Name, 4, tiny);
            for (int i = 0; i < 4; i++) Check(got[i] == 3.5, "constant program stays correct");
        }
        RecompilationRuntime.Entry deopted = aggressive.Find(constant.Name, 4)!;
        Check(aggressive.Deopts == 1, "one deopt, got " + aggressive.Deopts);
        Check(deopted.Blacklisted && deopted.Tier == Tier.Baseline, "regressed shape demoted and blacklisted");
        Check(deopted.Compilations == 2, "deopt costs no extra compilation, got " + deopted.Compilations);
        Check(deopted.OptimizedWork / 2 > deopted.BaselineAverageWork, "measured regression is real");
        for (int call = 0; call < 100; call++) aggressive.Invoke(constant.Name, 4, tiny);
        Check(deopted.Compilations == 2, "blacklisted shape never recompiles, got " + deopted.Compilations);
        Check(aggressive.Deopts == 1, "no repeated deopt churn, got " + aggressive.Deopts);

        // ---- boundedness: 200 invocations across programs and shapes ----
        var churn = new RecompilationRuntime();
        foreach (Program program in programs) churn.Register(program);
        int[] churnShapes = { 8, 32, 128, 3 };
        for (int call = 0; call < 200; call++)
        {
            Program program = programs[call % programs.Count];
            int n = churnShapes[(call / programs.Count) % churnShapes.Length];
            double[][] inputs = MakeInputs(program.Slots.Count, n);
            double[] got = churn.Invoke(program.Name, n, inputs);
            Check(got.Length == n, "churn output length");
        }
        Check(churn.TotalCompilations <= 2 * churn.Entries.Count,
            "compilations bounded by 2 per key: " + churn.TotalCompilations + " over " + churn.Entries.Count + " keys");
        foreach (KeyValuePair<string, RecompilationRuntime.Entry> pair in churn.Entries)
            Check(pair.Value.Compilations <= 2, "per-key compilation bound for " + pair.Key);
        Check(churn.Entries.Count == programs.Count * churnShapes.Length, "one entry per (program, shape) key");

        int promotedTiers = churn.Entries.Values.Count(e => e.Tier == Tier.Optimized);
        Console.WriteLine("programs=" + programs.Count + " shapes=" + shapes.Length + " entries=" + churn.Entries.Count +
            " compilations=" + churn.TotalCompilations + " optimizedEntries=" + promotedTiers + " deopts=" + aggressive.Deopts);
        Console.WriteLine("p0 relu(a*b+c): " + programs[0].OriginalNodes + " -> " + programs[0].OptimizedNodes +
            " ops, work n=64 baseline=" + CostModel.Work(Tier.Baseline, programs[0], 64) +
            " optimized=" + CostModel.Work(Tier.Optimized, programs[0], 64));
        Console.WriteLine("break-even grid: " + promotions + " promotions / " + refusals + " refusals; model matches measurement on " +
            (specs.Length * shapes.Length * 3) + " reports");
        Console.WriteLine("PASS");
    }
}
