// Task 752 - Implement taint analysis simple propagation for security checks. (Tier: Expert, Beyond - Security & Privacy)
//
// A forward, flow-sensitive, may-taint analysis over a toy basic-block IR. A secret is not a single bit here
// but a *label set*: every untrusted source produces the risk labels {exec, sql, html} (shell injection, SQL
// injection, cross-site scripting), each sanitizer removes exactly the labels it is able to neutralise, and
// each sink is dangerous for a fixed subset of labels. A flow is reported as (block, instruction, sink,
// argument index, label), so using the wrong sanitizer for a sink - escaping the shell and then building a
// SQL string - is *not* silently accepted, which is the whole point of a label-aware analysis.
//
// The fixpoint is the textbook worklist algorithm over the finite lattice of "variable -> set of labels":
// in[b] = union of out[pred], out[b] = transfer_b(in[b]). The transfer function is monotone (it only copies
// sets and removes labels named by a sanitizer), so the iteration terminates; the file asserts termination
// within the lattice-height bound instead of trusting it, asserts the in-state only ever grows, and re-checks
// the fixpoint equations in[b] = union(out[pred]) / out[b] = transfer(in[b]) afterwards.
//
// Independent oracle: an exhaustive path enumeration executes the CFG concretely (`maxSteps` unrolled) and
// unions the flows it observes. The analysis must be a *superset* of that oracle (soundness - taint must
// never be missed) and must be *exactly* equal on acyclic CFGs (no false positives from path merging).
// Simplifications, documented: no inter-procedural analysis, no implicit (control-dependence) flows, no
// string-escape reasoning, sanitizer effectiveness is a table rather than a proof of correctness.
using System;
using System.Collections.Generic;

internal static class Security
{
    internal static readonly string[] RiskLabels = { "exec", "sql", "html" };

    internal static readonly Dictionary<string, string[]> SinkDangerous = new Dictionary<string, string[]>
    {
        { "system", new[] { "exec" } },
        { "sql_query", new[] { "sql" } },
        { "html_out", new[] { "html" } },
    };

    internal static readonly Dictionary<string, string[]> SanitizerClears = new Dictionary<string, string[]>
    {
        { "shell_escape", new[] { "exec" } },
        { "param_query", new[] { "sql" } },
        { "html_escape", new[] { "html" } },
        { "strict_int", RiskLabels },
    };

    internal static bool IsSink(string name) => SinkDangerous.ContainsKey(name);
}

internal sealed class Instr
{
    internal readonly string Kind;      // assign | source | sanitize | sink
    internal readonly string Dst;       // variable written ("" for a sink)
    internal readonly string Arg;       // variable read ("" for a source or a sink)
    internal readonly string Name;      // source / sanitizer / sink name
    internal readonly string[] Args;    // sink arguments

    private Instr(string kind, string dst, string arg, string name, string[] args)
    {
        Kind = kind;
        Dst = dst;
        Arg = arg;
        Name = name;
        Args = args;
    }

    internal static Instr Assign(string dst, string src) => new Instr("assign", dst, src, "", Array.Empty<string>());

    internal static Instr Source(string dst, string name) => new Instr("source", dst, "", name, Array.Empty<string>());

    internal static Instr Sanitize(string dst, string sanitizer, string src) => new Instr("sanitize", dst, src, sanitizer, Array.Empty<string>());

    internal static Instr Sink(string name, params string[] args) => new Instr("sink", "", "", name, args);
}

/** variable -> set of risk labels still attached to it. */
internal sealed class State
{
    private static readonly HashSet<string> Empty = new HashSet<string>();
    private readonly Dictionary<string, HashSet<string>> labels = new Dictionary<string, HashSet<string>>();

    internal State() { }

    private State(State other)
    {
        foreach (KeyValuePair<string, HashSet<string>> pair in other.labels)
        {
            labels[pair.Key] = new HashSet<string>(pair.Value);
        }
    }

    internal State Clone() => new State(this);

    internal HashSet<string> Of(string variable)
        => labels.TryGetValue(variable, out HashSet<string>? set) ? set : Empty;

    internal void Set(string variable, HashSet<string> value) => labels[variable] = value;

    /** this := this union other (used to merge the outgoing states of all predecessors). */
    internal void MergeFrom(State other)
    {
        foreach (KeyValuePair<string, HashSet<string>> pair in other.labels)
        {
            if (!labels.TryGetValue(pair.Key, out HashSet<string>? set))
            {
                labels[pair.Key] = new HashSet<string>(pair.Value);
                continue;
            }
            foreach (string label in pair.Value) set.Add(label);
        }
    }

    /** this is a superset of other. */
    internal bool Covers(State other)
    {
        foreach (KeyValuePair<string, HashSet<string>> pair in other.labels)
        {
            if (!labels.TryGetValue(pair.Key, out HashSet<string>? set)) return false;
            foreach (string label in pair.Value)
            {
                if (!set.Contains(label)) return false;
            }
        }
        return true;
    }

    internal bool Same(State other) => Key() == other.Key();

    internal string Key()
    {
        var names = new List<string>(labels.Keys);
        names.Sort(StringComparer.Ordinal);
        var parts = new List<string>();
        foreach (string name in names)
        {
            var values = new List<string>(labels[name]);
            values.Sort(StringComparer.Ordinal);
            parts.Add(name + "={" + string.Join(",", values) + "}");
        }
        return string.Join(";", parts);
    }
}

internal static class TaintTransfer
{
    internal static void Apply(Instr instr, State state)
    {
        switch (instr.Kind)
        {
            case "assign":
                state.Set(instr.Dst, new HashSet<string>(state.Of(instr.Arg)));
                break;
            case "source":
                state.Set(instr.Dst, new HashSet<string>(Security.RiskLabels));
                break;
            case "sanitize":
                var kept = new HashSet<string>(state.Of(instr.Arg));
                foreach (string cleared in Security.SanitizerClears[instr.Name]) kept.Remove(cleared);
                state.Set(instr.Dst, kept);
                break;
            default:
                break;   // a sink reads its arguments and writes nothing
        }
    }
}

internal sealed class Block
{
    internal readonly int Id;
    internal readonly List<Instr> Body = new List<Instr>();
    internal readonly List<int> Next = new List<int>();

    internal Block(int id) { Id = id; }
}

internal sealed class Program
{
    private readonly List<Block> blocks;
    private readonly List<int>[] predecessors;

    internal Program(List<Block> blocks, int entry)
    {
        this.blocks = blocks;
        Entry = entry;
        predecessors = new List<int>[blocks.Count];
        for (int i = 0; i < blocks.Count; i++) predecessors[i] = new List<int>();
        foreach (Block block in blocks)
        {
            foreach (int successor in block.Next) predecessors[successor].Add(block.Id);
        }
    }

    internal int Entry { get; }
    internal IReadOnlyList<Block> Blocks => blocks;
    internal IReadOnlyList<int> Predecessors(int id) => predecessors[id];

    internal int EdgeCount()
    {
        int total = 0;
        foreach (Block block in blocks) total += block.Next.Count;
        return total;
    }

    internal List<string> Variables()
    {
        var seen = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Block block in blocks)
        {
            foreach (Instr instr in block.Body)
            {
                if (instr.Dst.Length > 0) seen.Add(instr.Dst);
                if (instr.Arg.Length > 0) seen.Add(instr.Arg);
                foreach (string arg in instr.Args) seen.Add(arg);
            }
        }
        return new List<string>(seen);
    }
}

internal sealed class TaintAnalysis
{
    private readonly Program program;
    private readonly Dictionary<int, State> inState = new Dictionary<int, State>();
    private readonly Dictionary<int, State> outState = new Dictionary<int, State>();

    internal readonly SortedSet<string> Hits = new SortedSet<string>(StringComparer.Ordinal);
    internal long TransferSteps;
    internal long BlockVisits;
    internal bool Monotone = true;

    internal TaintAnalysis(Program program) { this.program = program; }

    internal static string Hit(int block, int index, string sink, int arg, string label)
        => "b" + block + "#" + index + "|" + sink + "|arg" + arg + "|" + label;

    internal void Run()
    {
        foreach (Block block in program.Blocks)
        {
            inState[block.Id] = new State();
            outState[block.Id] = new State();
        }
        if (program.Blocks.Count == 0) return;

        var queue = new Queue<int>();
        var queued = new bool[program.Blocks.Count];
        var settledIn = new bool[program.Blocks.Count];
        var settledOut = new bool[program.Blocks.Count];
        queue.Enqueue(program.Entry);
        queued[program.Entry] = true;
        while (queue.Count > 0)
        {
            int id = queue.Dequeue();
            queued[id] = false;
            BlockVisits++;

            var incoming = new State();
            foreach (int pred in program.Predecessors(id)) incoming.MergeFrom(outState[pred]);
            // A block is processed at least once even when its in-state equals the initial empty state,
            // otherwise a block that only reads (no writes) would never be enqueued and its successors
            // would never be visited.
            if (settledIn[id] && incoming.Same(inState[id])) continue;
            if (settledIn[id] && !incoming.Covers(inState[id])) Monotone = false;   // the in-state must only grow
            inState[id] = incoming;
            settledIn[id] = true;

            State outgoing = incoming.Clone();
            foreach (Instr instr in program.Blocks[id].Body) TaintTransfer.Apply(instr, outgoing);
            TransferSteps += program.Blocks[id].Body.Count;
            if (settledOut[id] && outgoing.Same(outState[id])) continue;
            if (settledOut[id] && !outgoing.Covers(outState[id])) Monotone = false;
            outState[id] = outgoing;
            settledOut[id] = true;
            foreach (int successor in program.Blocks[id].Next)
            {
                if (!queued[successor])
                {
                    queued[successor] = true;
                    queue.Enqueue(successor);
                }
            }
        }
    }

    /** Replays every block from its final in-state; a sink hit means a tainted label reaches a dangerous argument. */
    internal void CollectHits()
    {
        foreach (Block block in program.Blocks)
        {
            State state = inState[block.Id].Clone();
            for (int i = 0; i < block.Body.Count; i++)
            {
                Instr instr = block.Body[i];
                TaintTransfer.Apply(instr, state);
                if (instr.Kind != "sink") continue;
                string[] dangerous = Security.SinkDangerous[instr.Name];
                for (int arg = 0; arg < instr.Args.Length; arg++)
                {
                    HashSet<string> labels = state.Of(instr.Args[arg]);
                    foreach (string label in dangerous)
                    {
                        if (labels.Contains(label)) Hits.Add(Hit(block.Id, i, instr.Name, arg, label));
                    }
                }
            }
        }
    }

    /** Re-checks the fixpoint equations independently of how the worklist got there. */
    internal bool EquationsHold()
    {
        foreach (Block block in program.Blocks)
        {
            var expected = new State();
            foreach (int pred in program.Predecessors(block.Id)) expected.MergeFrom(outState[pred]);
            if (!expected.Same(inState[block.Id])) return false;
            State outgoing = inState[block.Id].Clone();
            foreach (Instr instr in block.Body) TaintTransfer.Apply(instr, outgoing);
            if (!outgoing.Same(outState[block.Id])) return false;
        }
        return true;
    }

    internal string Snapshot()
    {
        var parts = new List<string>();
        foreach (Block block in program.Blocks)
        {
            parts.Add(block.Id + ":" + inState[block.Id].Key() + "=>" + outState[block.Id].Key());
        }
        parts.Add("hits:" + string.Join(",", Hits));
        return string.Join("|", parts);
    }

    internal long IterationBound()
    {
        long height = (long)program.Variables().Count * Security.RiskLabels.Length + 1;
        return 1 + height * (program.EdgeCount() + program.Blocks.Count + 1);
    }
}

internal static class PathOracle
{
    /** Concrete execution of every CFG path up to maxSteps; `complete` is false once a bound truncates the search. */
    internal static SortedSet<string> Hits(Program program, int maxSteps, long pathCap, out long paths, out bool complete)
    {
        var hits = new SortedSet<string>(StringComparer.Ordinal);
        long finished = 0;
        bool truncated = false;
        if (program.Blocks.Count > 0)
        {
            Walk(program, program.Entry, new State(), 0, maxSteps, pathCap, hits, ref finished, ref truncated);
        }
        paths = finished;
        complete = !truncated;
        return hits;
    }

    private static void Walk(Program program, int blockId, State state, int steps, int maxSteps, long pathCap,
        SortedSet<string> hits, ref long paths, ref bool truncated)
    {
        if (paths >= pathCap || steps > maxSteps)
        {
            truncated = true;
            return;
        }
        Block block = program.Blocks[blockId];
        State current = state.Clone();
        for (int i = 0; i < block.Body.Count; i++)
        {
            Instr instr = block.Body[i];
            TaintTransfer.Apply(instr, current);
            if (instr.Kind != "sink") continue;
            string[] dangerous = Security.SinkDangerous[instr.Name];
            for (int arg = 0; arg < instr.Args.Length; arg++)
            {
                HashSet<string> labels = current.Of(instr.Args[arg]);
                foreach (string label in dangerous)
                {
                    if (labels.Contains(label)) hits.Add(TaintAnalysis.Hit(block.Id, i, instr.Name, arg, label));
                }
            }
        }
        if (block.Next.Count == 0)
        {
            paths++;
            return;
        }
        foreach (int successor in block.Next)
        {
            Walk(program, successor, current, steps + 1, maxSteps, pathCap, hits, ref paths, ref truncated);
        }
    }
}

internal sealed class Builder
{
    private readonly List<Block> blocks = new List<Block>();
    internal int Entry;

    internal int Add(params Instr[] body)
    {
        var block = new Block(blocks.Count);
        block.Body.AddRange(body);
        blocks.Add(block);
        return block.Id;
    }

    internal void Link(int from, int to) => blocks[from].Next.Add(to);

    internal Program Build() => new Program(blocks, Entry);
}

internal static class Corpus
{
    internal static Program Direct()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sink("system", "raw"));
        return builder.Build();
    }

    internal static Program SanitizedForExec()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sanitize("safe", "shell_escape", "raw"), Instr.Sink("system", "safe"));
        return builder.Build();
    }

    internal static Program WrongSanitizer()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "file_read"), Instr.Sanitize("safe", "shell_escape", "raw"), Instr.Sink("sql_query", "safe"));
        return builder.Build();
    }

    internal static Program CopyChain()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("a", "http_param"), Instr.Assign("b", "a"), Instr.Assign("c", "b"), Instr.Assign("d", "c"),
            Instr.Sink("system", "d"));
        return builder.Build();
    }

    internal static Program BranchJoin()
    {
        var builder = new Builder();
        int head = builder.Add(Instr.Source("raw", "http_param"), Instr.Assign("v", "raw"));
        int cleansed = builder.Add(Instr.Sanitize("v", "strict_int", "v"));
        int untouched = builder.Add();
        int join = builder.Add(Instr.Sink("system", "v"));
        builder.Link(head, cleansed);
        builder.Link(head, untouched);
        builder.Link(cleansed, join);
        builder.Link(untouched, join);
        return builder.Build();
    }

    internal static Program LoopCarried()
    {
        var builder = new Builder();
        int source = builder.Add(Instr.Source("raw", "http_param"));
        int loop = builder.Add(Instr.Assign("acc", "raw"), Instr.Sink("system", "acc"), Instr.Assign("acc", "acc"));
        builder.Link(source, loop);
        builder.Link(loop, loop);
        return builder.Build();
    }

    internal static Program TaintWithoutSink()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Assign("x", "raw"));
        return builder.Build();
    }

    internal static Program ValidatorClearsEverything()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sanitize("i", "strict_int", "raw"),
            Instr.Sink("system", "i"), Instr.Sink("sql_query", "i"), Instr.Sink("html_out", "i"));
        return builder.Build();
    }

    internal static Program ArgumentIndexing()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sink("system", "clean", "raw"), Instr.Sink("html_out", "raw"));
        return builder.Build();
    }

    internal static Program SanitizerAfterSink()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sink("system", "raw"),
            Instr.Sanitize("raw2", "shell_escape", "raw"), Instr.Sink("system", "raw2"));
        return builder.Build();
    }

    internal static Program EverySink()
    {
        var builder = new Builder();
        builder.Add(Instr.Source("raw", "http_param"), Instr.Sink("system", "raw"),
            Instr.Sink("sql_query", "raw"), Instr.Sink("html_out", "raw"));
        return builder.Build();
    }

    internal static Program Empty()
    {
        var builder = new Builder();
        return builder.Build();
    }
}

public static class Solution
{
    public static void Main()
    {
        bool monotone = true;
        bool equations = true;
        bool idempotent = true;
        bool bounded = true;
        bool oracleSound = true;
        bool oracleExact = true;
        int totalHits = 0;

        // Expected flow sets are written out literally (block#instruction|sink|argN|label) rather than derived
        // from the analysis, so a bug in hit construction cannot hide behind a matching expectation.
        var programs = new (string Name, Program Program, string[] Expected)[]
        {
            ("direct source-to-sink", Corpus.Direct(), new[] { "b0#1|system|arg0|exec" }),
            ("shell_escape before system", Corpus.SanitizedForExec(), Array.Empty<string>()),
            ("shell_escape is useless for sql_query", Corpus.WrongSanitizer(), new[] { "b0#2|sql_query|arg0|sql" }),
            ("four-copy propagation chain", Corpus.CopyChain(), new[] { "b0#4|system|arg0|exec" }),
            ("branch join keeps the dirty path", Corpus.BranchJoin(), new[] { "b3#0|system|arg0|exec" }),
            ("loop-carried taint reaches fixpoint", Corpus.LoopCarried(), new[] { "b1#1|system|arg0|exec" }),
            ("tainted but no sink", Corpus.TaintWithoutSink(), Array.Empty<string>()),
            ("strict_int clears every label", Corpus.ValidatorClearsEverything(), Array.Empty<string>()),
            ("argument index is reported", Corpus.ArgumentIndexing(),
                new[] { "b0#1|system|arg1|exec", "b0#2|html_out|arg0|html" }),
            ("sanitizer after the sink does not help", Corpus.SanitizerAfterSink(), new[] { "b0#1|system|arg0|exec" }),
            ("one source, every sink, three labels", Corpus.EverySink(),
                new[] { "b0#1|system|arg0|exec", "b0#2|sql_query|arg0|sql", "b0#3|html_out|arg0|html" }),
        };

        foreach ((string name, Program program, string[] expected) in programs)
        {
            var analysis = new TaintAnalysis(program);
            analysis.Run();
            analysis.CollectHits();

            var repeat = new TaintAnalysis(program);          // same input -> same fixpoint and same flows
            repeat.Run();
            repeat.CollectHits();
            idempotent &= analysis.Snapshot() == repeat.Snapshot();
            monotone &= analysis.Monotone;
            equations &= analysis.EquationsHold();
            bounded &= analysis.BlockVisits <= analysis.IterationBound() && repeat.BlockVisits == analysis.BlockVisits;
            totalHits += analysis.Hits.Count;

            var expectedSet = new SortedSet<string>(expected, StringComparer.Ordinal);
            Check(analysis.Hits.SetEquals(expectedSet),
                name + ": flows {" + string.Join(" ", analysis.Hits) + "} expected {" + string.Join(" ", expectedSet) + "}");

            SortedSet<string> oracle = PathOracle.Hits(program, 32, 200000, out long paths, out bool complete);
            oracleSound &= analysis.Hits.IsSupersetOf(oracle);
            Check(analysis.Hits.IsSupersetOf(oracle), name + ": analysis must not miss an executed flow");
            if (complete)
            {
                oracleExact &= analysis.Hits.SetEquals(oracle);
                Check(analysis.Hits.SetEquals(oracle), name + ": exact on an acyclic CFG (" + paths + " paths)");
            }
            Console.WriteLine("  " + name + ": flows=" + analysis.Hits.Count + " paths=" + paths
                + " visits=" + analysis.BlockVisits + " oracleExact=" + complete);
        }

        var empty = new TaintAnalysis(Corpus.Empty());
        empty.Run();
        empty.CollectHits();
        Check(empty.Hits.Count == 0 && empty.EquationsHold(), "empty program reports nothing and holds its equations");
        Check(PathOracle.Hits(Corpus.Empty(), 32, 1000, out long emptyPaths, out bool emptyComplete).Count == 0
            && emptyPaths == 0 && emptyComplete, "empty program has no paths");

        Check(monotone, "every fixpoint iteration only added labels");
        Check(equations, "in[b] = union(out[pred]) and out[b] = transfer(in[b]) hold at the fixpoint");
        Check(idempotent, "re-running the analysis reproduces the same states and flows");
        Check(bounded, "worklist step count stays inside the lattice-height bound");
        Check(oracleSound, "analysis dominates the concrete path oracle on every program");
        Check(oracleExact, "analysis is exact wherever the oracle enumerated all paths");
        Check(totalHits == 11, "total flows across the corpus: " + totalHits);
        Check(Security.IsSink("system") && !Security.IsSink("assign"), "sink table lookup");

        Console.WriteLine("  corpus: " + programs.Length + " programs, " + totalHits + " flows, empty program clean");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
