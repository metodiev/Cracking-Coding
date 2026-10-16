// Task 753 - Implement control-flow integrity checker skeleton. (Tier: Expert, Beyond - Security & Privacy)
//
// Control-flow integrity (CFI) validates every control transfer of an execution against the static control
// flow graph. This skeleton models a program as basic blocks with three kinds of *indirect* transfer:
//   * a call must land on the callee entry of that call site (a "function signature" check, and a coarse
//     target-set check: only function entries are legitimate call targets);
//   * a return must land on the return site recorded by the matching call, which is enforced with a shadow
//     stack - the standard defence against return-address corruption and ROP chains;
//   * an indirect branch (jump table / virtual dispatch) must land inside the block's own target set.
// Direct branches are checked against the successor list, and every target must be the address of a declared
// block, so a transfer into the middle of a block is rejected (the toy stand-in for instruction alignment).
//
// The CFG is analysed independently of the trace checker: reachability has two implementations (DFS vs
// repeated edge relaxation) that must agree, dominators come from the textbook dataflow fixpoint, and back
// edges are detected twice - by the dominator rule (an edge u->v with v in Dom(u)) and by DFS colouring (an
// edge to a grey node). Agreement of the two back-edge sets is a witness that this CFG is reducible, which is
// what makes a header-based shadow-stack policy safe here. Documented simplification: the checker validates
// transfers, not reachability or instruction semantics, and it reports only the first violation (a trap).
// Honest limitation tested below: an indirect transfer that stays inside the legitimate target set is
// invisible to coarse CFI, so that mutation is expected to be *accepted*, not flagged.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    private static ulong state = 0x753C0DE0FEEDUL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal enum Kind { Entry, Plain, Call, Ret, Indirect }

internal sealed class CfgBlock
{
    internal readonly int Id;
    internal readonly Kind Kind;
    internal readonly List<int> Next = new List<int>();   // static successors; jump table for Indirect; return sites for Ret
    internal int ReturnSite = -1;                         // Call blocks: where the callee returns to

    internal CfgBlock(int id, Kind kind) { Id = id; Kind = kind; }
}

internal sealed class Program
{
    internal const int BaseAddress = 0x1000;
    internal const int Stride = 0x10;

    private readonly List<CfgBlock> blocks;
    private readonly List<int>[] predecessors;

    internal Program(List<CfgBlock> blocks, int entryId)
    {
        this.blocks = blocks;
        Entry = entryId;
        predecessors = new List<int>[blocks.Count];
        for (int i = 0; i < blocks.Count; i++) predecessors[i] = new List<int>();
        foreach (CfgBlock block in blocks)
        {
            foreach (int next in block.Next) predecessors[next].Add(block.Id);
        }
    }

    internal int Entry { get; }
    internal IReadOnlyList<CfgBlock> Blocks => blocks;
    internal CfgBlock this[int id] => blocks[id];
    internal IReadOnlyList<int> Predecessors(int id) => predecessors[id];
    internal int AddressOf(int id) => BaseAddress + id * Stride;
    internal int MidBlockAddress(int id) => BaseAddress + id * Stride + Stride / 2;

    internal bool TryResolve(int address, out int id)
    {
        id = -1;
        int offset = address - BaseAddress;
        if (offset < 0 || offset % Stride != 0) return false;
        int index = offset / Stride;
        if (index >= blocks.Count) return false;
        id = index;
        return true;
    }

    internal int EdgeCount()
    {
        int total = 0;
        foreach (CfgBlock block in blocks) total += block.Next.Count;
        return total;
    }
}

internal sealed class Cfg
{
    private readonly List<CfgBlock> blocks = new List<CfgBlock>();

    internal CfgBlock Add(Kind kind)
    {
        var block = new CfgBlock(blocks.Count, kind);
        blocks.Add(block);
        return block;
    }

    internal void Link(CfgBlock from, CfgBlock to) => from.Next.Add(to.Id);

    internal void Call(CfgBlock from, CfgBlock calleeEntry, CfgBlock returnSite)
    {
        Link(from, calleeEntry);
        from.ReturnSite = returnSite.Id;
    }

    internal Program Build(CfgBlock entry) => new Program(blocks, entry.Id);
}

internal static class Sample
{
    /** main { loop with two backedges; call f_log; call f_add; dispatch } plus f_log and f_add. */
    internal static Program Build()
    {
        var cfg = new Cfg();
        CfgBlock entry = cfg.Add(Kind.Entry);          // 0
        CfgBlock head = cfg.Add(Kind.Plain);           // 1
        CfgBlock loopHeader = cfg.Add(Kind.Plain);     // 2  loop header (two backedges)
        CfgBlock body = cfg.Add(Kind.Plain);           // 3
        CfgBlock back1 = cfg.Add(Kind.Plain);          // 4
        CfgBlock afterLoop = cfg.Add(Kind.Plain);      // 5
        CfgBlock mid = cfg.Add(Kind.Plain);            // 6
        CfgBlock continuer = cfg.Add(Kind.Plain);      // 7
        CfgBlock callLog = cfg.Add(Kind.Call);         // 8
        CfgBlock afterLog = cfg.Add(Kind.Plain);       // 9
        CfgBlock callAdd = cfg.Add(Kind.Call);         // 10
        CfgBlock afterAdd = cfg.Add(Kind.Plain);       // 11
        CfgBlock dispatch = cfg.Add(Kind.Indirect);    // 12  jump table
        CfgBlock arm0 = cfg.Add(Kind.Plain);           // 13
        CfgBlock arm1 = cfg.Add(Kind.Plain);           // 14
        CfgBlock arm2 = cfg.Add(Kind.Plain);           // 15
        CfgBlock join = cfg.Add(Kind.Plain);           // 16
        CfgBlock exit = cfg.Add(Kind.Plain);           // 17
        CfgBlock callLog2 = cfg.Add(Kind.Call);        // 18  second call site of f_log
        CfgBlock logEntry = cfg.Add(Kind.Entry);       // 19
        CfgBlock logBody = cfg.Add(Kind.Plain);        // 20
        CfgBlock logRet = cfg.Add(Kind.Ret);           // 21  returns to 9 or 17
        CfgBlock addEntry = cfg.Add(Kind.Entry);       // 22
        CfgBlock addBody = cfg.Add(Kind.Plain);        // 23
        CfgBlock addRet = cfg.Add(Kind.Ret);           // 24  returns to 11 or 21
        CfgBlock logCall = cfg.Add(Kind.Call);         // 25  nested call inside f_log

        cfg.Link(entry, head);
        cfg.Link(head, loopHeader);
        cfg.Link(loopHeader, body);
        cfg.Link(loopHeader, afterLoop);
        cfg.Link(body, back1);
        cfg.Link(back1, loopHeader);                   // back edge 1
        cfg.Link(afterLoop, mid);
        cfg.Link(mid, continuer);
        cfg.Link(continuer, loopHeader);               // back edge 2
        cfg.Link(continuer, callLog);
        cfg.Call(callLog, logEntry, afterLog);
        cfg.Link(afterLog, callAdd);
        cfg.Call(callAdd, addEntry, afterAdd);
        cfg.Link(afterAdd, dispatch);
        cfg.Link(dispatch, arm0);
        cfg.Link(dispatch, arm1);
        cfg.Link(dispatch, arm2);
        cfg.Link(arm0, join);
        cfg.Link(arm1, join);
        cfg.Link(arm2, callLog2);
        cfg.Call(callLog2, logEntry, exit);
        cfg.Link(join, exit);
        cfg.Link(logEntry, logBody);
        cfg.Link(logBody, logCall);
        cfg.Call(logCall, addEntry, logRet);           // f_add is called from inside f_log too
        cfg.Link(logRet, afterLog);
        cfg.Link(logRet, exit);
        cfg.Link(addEntry, addBody);
        cfg.Link(addBody, addRet);
        cfg.Link(addRet, afterAdd);
        cfg.Link(addRet, logRet);
        return cfg.Build(entry);
    }
}

internal sealed class Violation
{
    internal int Step;
    internal string Kind = "";
    internal string Detail = "";
}

internal sealed class Walk
{
    internal int Start;
    internal readonly List<(int From, int To)> Steps = new List<(int From, int To)>();
    internal readonly HashSet<string> Edges = new HashSet<string>();
    internal int MaxShadowDepth;
}

internal static class Walks
{
    /** Random walk that maintains its own shadow stack, i.e. it only ever performs legitimate transfers. */
    internal static Walk Generate(Program program, int steps)
    {
        var walk = new Walk();
        var stack = new List<int>();
        int current = program.Entry;
        walk.Start = program.AddressOf(current);
        for (int i = 0; i < steps; i++)
        {
            CfgBlock block = program[current];
            if (block.Kind == Kind.Ret)
            {
                if (stack.Count == 0) break;
                int site = stack[stack.Count - 1];
                stack.RemoveAt(stack.Count - 1);
                walk.Steps.Add((program.AddressOf(current), program.AddressOf(site)));
                walk.Edges.Add(current + "->" + site);
                current = site;
                continue;
            }
            if (block.Next.Count == 0) break;             // main exited
            int next = block.Next[SplitMix64.NextInt(block.Next.Count)];
            if (block.Kind == Kind.Call) stack.Add(block.ReturnSite);
            walk.Steps.Add((program.AddressOf(current), program.AddressOf(next)));
            walk.Edges.Add(current + "->" + next);
            walk.MaxShadowDepth = Math.Max(walk.MaxShadowDepth, stack.Count);
            current = next;
        }
        return walk;
    }
}

internal sealed class CfiChecker
{
    private readonly Program program;

    internal CfiChecker(Program program) { this.program = program; }

    internal int StepsVerified;

    /** Validates the trace; returns the first violation (empty when the trace is CFI-clean). */
    internal List<Violation> Check(int startAddress, IReadOnlyList<(int From, int To)> trace)
    {
        var violations = new List<Violation>();
        if (!program.TryResolve(startAddress, out _))
        {
            violations.Add(new Violation { Step = -1, Kind = "bad-start", Detail = Hex(startAddress) });
            return violations;
        }

        var shadowStack = new List<int>();
        for (int i = 0; i < trace.Count; i++)
        {
            int fromAddress = trace[i].From;
            int toAddress = trace[i].To;
            if (!program.TryResolve(fromAddress, out int fromId))
            {
                violations.Add(new Violation { Step = i, Kind = "unknown-source", Detail = Hex(fromAddress) });
                break;
            }
            if (!program.TryResolve(toAddress, out int toId))
            {
                violations.Add(new Violation { Step = i, Kind = "unaligned-target", Detail = Hex(toAddress) });
                break;
            }

            StepsVerified++;
            CfgBlock block = program[fromId];
            string kind = string.Empty;
            switch (block.Kind)
            {
                case Kind.Call:
                    if (block.Next.Count != 1 || block.Next[0] != toId || program[toId].Kind != Kind.Entry)
                    {
                        kind = "call-target";
                    }
                    else
                    {
                        shadowStack.Add(block.ReturnSite);
                    }
                    break;
                case Kind.Ret:
                    if (shadowStack.Count == 0) kind = "stack-underflow";
                    else if (shadowStack[shadowStack.Count - 1] != toId) kind = "return-address";
                    else shadowStack.RemoveAt(shadowStack.Count - 1);
                    break;
                case Kind.Indirect:
                    if (!block.Next.Contains(toId)) kind = "indirect-out-of-table";
                    break;
                default:
                    if (!block.Next.Contains(toId)) kind = "edge";
                    break;
            }

            if (kind.Length > 0)
            {
                violations.Add(new Violation { Step = i, Kind = kind, Detail = Hex(fromAddress) + "->" + Hex(toAddress) });
                break;                                     // trap on the first violation
            }
        }
        return violations;
    }

    private static string Hex(int value) => "0x" + value.ToString("X4");
}

internal static class CfgAnalysis
{
    internal static bool[] ReachableByDfs(Program program, int start)
    {
        var seen = new bool[program.Blocks.Count];
        var stack = new Stack<int>();
        stack.Push(start);
        seen[start] = true;
        while (stack.Count > 0)
        {
            int id = stack.Pop();
            foreach (int next in program[id].Next)
            {
                if (seen[next]) continue;
                seen[next] = true;
                stack.Push(next);
            }
        }
        return seen;
    }

    /** Same question, answered by repeated relaxation over the edge list instead of a DFS. */
    internal static bool[] ReachableByRelaxation(Program program, int start)
    {
        var seen = new bool[program.Blocks.Count];
        seen[start] = true;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int id = 0; id < program.Blocks.Count; id++)
            {
                if (!seen[id]) continue;
                foreach (int next in program[id].Next)
                {
                    if (seen[next]) continue;
                    seen[next] = true;
                    changed = true;
                }
            }
        }
        return seen;
    }

    /** Dom(b) = {b} union the intersection of Dom(pred); computed to a fixpoint over reachable blocks. */
    internal static List<HashSet<int>> Dominators(Program program, bool[] reachable)
    {
        int n = program.Blocks.Count;
        var everything = new HashSet<int>();
        for (int id = 0; id < n; id++)
        {
            if (reachable[id]) everything.Add(id);
        }
        var dom = new List<HashSet<int>>();
        for (int id = 0; id < n; id++) dom.Add(reachable[id] ? new HashSet<int>(everything) : new HashSet<int>());
        dom[program.Entry] = new HashSet<int> { program.Entry };
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int id = 0; id < n; id++)
            {
                if (!reachable[id] || id == program.Entry) continue;
                HashSet<int>? intersection = null;
                foreach (int pred in program.Predecessors(id))
                {
                    if (!reachable[pred]) continue;
                    intersection = intersection == null ? new HashSet<int>(dom[pred]) : Intersect(intersection, dom[pred]);
                }
                if (intersection == null) continue;
                intersection.Add(id);
                if (!intersection.SetEquals(dom[id]))
                {
                    dom[id] = intersection;
                    changed = true;
                }
            }
        }
        return dom;
    }

    internal static HashSet<string> BackEdgesByDominators(Program program, List<HashSet<int>> dom, bool[] reachable)
    {
        var back = new HashSet<string>();
        for (int id = 0; id < program.Blocks.Count; id++)
        {
            if (!reachable[id]) continue;
            foreach (int next in program[id].Next)
            {
                if (dom[id].Contains(next)) back.Add(id + "->" + next);   // the head dominates the tail: a back edge
            }
        }
        return back;
    }

    internal static HashSet<string> BackEdgesByDfs(Program program, int start)
    {
        var color = new int[program.Blocks.Count];         // 0 white, 1 grey, 2 black
        var back = new HashSet<string>();
        Visit(program, start, color, back);
        return back;
    }

    private static void Visit(Program program, int id, int[] color, HashSet<string> back)
    {
        color[id] = 1;
        foreach (int next in program[id].Next)
        {
            if (color[next] == 1) back.Add(id + "->" + next);
            else if (color[next] == 0) Visit(program, next, color, back);
        }
        color[id] = 2;
    }

    private static HashSet<int> Intersect(HashSet<int> left, HashSet<int> right)
    {
        var result = new HashSet<int>();
        foreach (int value in left)
        {
            if (right.Contains(value)) result.Add(value);
        }
        return result;
    }
}

internal sealed class Case
{
    internal string Name = "";
    internal string[] Expected = Array.Empty<string>();   // acceptable violation kinds; empty means "must be accepted"
    internal List<(int From, int To)> Trace = new List<(int From, int To)>();
    internal int Start;
    internal int Step;
}

internal static class Injections
{
    /** Every case is a real walk with exactly one transfer rewritten (or one step inserted). */
    internal static List<Case> Build(Program program, List<Walk> walks)
    {
        var cases = new List<Case>();
        var perClass = new Dictionary<string, int>();

        foreach (Walk walk in walks)
        {
            List<(int From, int To)> steps = walk.Steps;
            if (steps.Count < 4) continue;

            for (int i = 0; i + 1 < steps.Count; i++)
            {
                if (!program.TryResolve(steps[i].From, out int fromId)) continue;
                if (!program.TryResolve(steps[i].To, out int toId)) continue;
                CfgBlock from = program[fromId];

                if (from.Kind == Kind.Plain || from.Kind == Kind.Entry)
                {
                    int replacement = FirstBlockNotIn(program, from, toId);
                    if (replacement >= 0 && Allow(perClass, "edge", 120))
                    {
                        cases.Add(Replaced(walk, i, program.AddressOf(replacement), "edge", new[] { "edge" }));
                    }
                }

                if (from.Kind == Kind.Ret && from.Next.Count >= 2 && Allow(perClass, "return-address", 120))
                {
                    int other = from.Next[0] == toId ? from.Next[1] : from.Next[0];
                    cases.Add(Replaced(walk, i, program.AddressOf(other), "return-address", new[] { "return-address" }));
                }

                if (from.Kind == Kind.Ret && Allow(perClass, "double-return", 60))
                {
                    var trace = new List<(int From, int To)>(steps);
                    trace.Insert(i + 1, steps[i]);
                    cases.Add(new Case
                    {
                        Name = "double-return",
                        Expected = new[] { "return-address", "stack-underflow" },
                        Trace = trace,
                        Start = walk.Start,
                        Step = i + 1,
                    });
                }

                if (from.Kind == Kind.Call && Allow(perClass, "call-target", 60))
                {
                    cases.Add(Replaced(walk, i, program.AddressOf(program.Blocks.Count - 1), "call-target", new[] { "call-target" }));
                }

                if (from.Kind == Kind.Indirect)
                {
                    int outside = FirstBlockNotIn(program, from, toId);
                    if (outside >= 0 && Allow(perClass, "indirect-out-of-table", 60))
                    {
                        cases.Add(Replaced(walk, i, program.AddressOf(outside), "indirect-out-of-table",
                            new[] { "indirect-out-of-table" }));
                    }
                    int inside = FirstBlockIn(program, from, toId);
                    if (inside >= 0 && Allow(perClass, "indirect-within-table (documented blind spot)", 60))
                    {
                        // Coarse CFI cannot see this: the new target is still inside the legitimate table.
                        cases.Add(Replaced(walk, i, program.AddressOf(inside),
                            "indirect-within-table (documented blind spot)", Array.Empty<string>()));
                    }
                }

                if (i % 3 == 0)
                {
                    if (Allow(perClass, "unknown-source", 60))
                    {
                        cases.Add(Replaced(walk, i, steps[i].To, "unknown-source", new[] { "unknown-source" },
                            program.MidBlockAddress(0)));
                    }
                    if (Allow(perClass, "unaligned-target", 60))
                    {
                        cases.Add(Replaced(walk, i, program.MidBlockAddress(1), "unaligned-target",
                            new[] { "unaligned-target" }));
                    }
                }
            }

            // A ret with an empty shadow stack: a single ret step, starting at the ret block.
            for (int i = 0; i < steps.Count; i++)
            {
                if (!program.TryResolve(steps[i].From, out int fromId) || program[fromId].Kind != Kind.Ret) continue;
                if (!Allow(perClass, "stack-underflow", 40)) break;
                cases.Add(new Case
                {
                    Name = "stack-underflow",
                    Expected = new[] { "stack-underflow" },
                    Trace = new List<(int From, int To)> { steps[i] },
                    Start = steps[i].From,
                    Step = 0,
                });
                break;
            }
        }
        return cases;
    }

    private static Case Replaced(Walk walk, int index, int newTarget, string name, string[] expected)
        => Replaced(walk, index, newTarget, name, expected, walk.Steps[index].From);

    private static Case Replaced(Walk walk, int index, int newTarget, string name, string[] expected, int newSource)
    {
        var trace = new List<(int From, int To)>(walk.Steps);
        trace[index] = (newSource, newTarget);
        return new Case { Name = name, Expected = expected, Trace = trace, Start = walk.Start, Step = index };
    }

    private static int FirstBlockNotIn(Program program, CfgBlock block, int current)
    {
        for (int candidate = 0; candidate < program.Blocks.Count; candidate++)
        {
            if (candidate == current || block.Next.Contains(candidate)) continue;
            return candidate;
        }
        return -1;
    }

    private static int FirstBlockIn(Program program, CfgBlock block, int current)
    {
        for (int candidate = 0; candidate < program.Blocks.Count; candidate++)
        {
            if (candidate != current && block.Next.Contains(candidate)) return candidate;
        }
        return -1;
    }

    private static bool Allow(Dictionary<string, int> perClass, string name, int cap)
    {
        perClass.TryGetValue(name, out int count);
        if (count >= cap) return false;
        perClass[name] = count + 1;
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        Program program = Sample.Build();
        var checker = new CfiChecker(program);

        // Independent CFG facts.
        bool[] dfsReach = CfgAnalysis.ReachableByDfs(program, program.Entry);
        bool[] relaxReach = CfgAnalysis.ReachableByRelaxation(program, program.Entry);
        int reachableCount = 0;
        bool sameReach = true;
        for (int i = 0; i < program.Blocks.Count; i++)
        {
            if (dfsReach[i]) reachableCount++;
            sameReach &= dfsReach[i] == relaxReach[i];
        }
        Check(sameReach && reachableCount == program.Blocks.Count, "reachability agrees and covers all blocks");

        List<HashSet<int>> dom = CfgAnalysis.Dominators(program, dfsReach);
        bool entryDominatesAll = true;
        for (int i = 0; i < program.Blocks.Count; i++)
        {
            if (dfsReach[i]) entryDominatesAll &= dom[i].Contains(program.Entry);
        }
        Check(entryDominatesAll, "the entry dominates every reachable block");
        Check(dom[2].Contains(2) && dom[0].SetEquals(new HashSet<int> { 0 }), "dominator sets are reflexive at the entry");
        Check(!dom[3].Contains(5) && dom[5].Contains(2), "loop header dominates the loop exit path but not the body");

        HashSet<string> domBack = CfgAnalysis.BackEdgesByDominators(program, dom, dfsReach);
        HashSet<string> dfsBack = CfgAnalysis.BackEdgesByDfs(program, program.Entry);
        Check(domBack.SetEquals(dfsBack) && domBack.Count >= 3,
            "dominator rule and DFS colouring agree on back edges: {" + string.Join(" ", domBack) + "}");
        Check(domBack.Contains("4->2") && domBack.Contains("7->2"), "both loop backedges target the loop header");
        Check(domBack.Count > 2,
            "modelling calls as CFG edges makes the call inside the loop a back edge to the callee entry");

        // Legitimate traces: every walk is a real execution (calls push, returns pop) until the process exits.
        var walks = new List<Walk>();
        int totalTransfers = 0;
        while (walks.Count < 400 && totalTransfers < 4000)
        {
            Walk generated = Walks.Generate(program, 400);
            if (generated.Steps.Count == 0) break;
            walks.Add(generated);
            totalTransfers += generated.Steps.Count;
        }
        Check(walks.Count >= 30 && totalTransfers >= 800,
            "walk corpus: " + walks.Count + " executions, " + totalTransfers + " transfers");

        var coveredEdges = new HashSet<string>();
        int falsePositives = 0, maxShadowDepth = 0;
        foreach (Walk legit in walks)
        {
            coveredEdges.UnionWith(legit.Edges);
            falsePositives += checker.Check(legit.Start, legit.Steps).Count;
            maxShadowDepth = Math.Max(maxShadowDepth, legit.MaxShadowDepth);
        }
        Check(falsePositives == 0, "0 false positives over " + totalTransfers + " legitimate transfers");
        Check(maxShadowDepth >= 2, "walks nest calls, max shadow depth " + maxShadowDepth);
        double coverage = coveredEdges.Count / (double)program.EdgeCount();
        Check(coverage >= 0.6, "static edge coverage " + coverage.ToString("F3") + " over " + program.EdgeCount() + " edges");
        Check(coveredEdges.Contains("21->9") && coveredEdges.Contains("21->17"),
            "both return sites of f_log are exercised");
        Check(coveredEdges.Contains("16->17") && coveredEdges.Contains("12->15"), "exit and jump-table arms are exercised");

        // An empty trace and a trace from an undeclared address.
        Check(checker.Check(walks[0].Start, Array.Empty<(int, int)>()).Count == 0, "empty trace is trivially clean");
        Check(checker.Check(program.MidBlockAddress(0), Array.Empty<(int, int)>())[0].Kind == "bad-start",
            "a start address in the middle of a block is rejected");

        // Injected attacks, one aggregated check per class.
        List<Case> cases = Injections.Build(program, walks);
        var classes = new List<string>();
        var outcomes = new Dictionary<string, int[]>();
        foreach (Case testCase in cases)
        {
            if (!outcomes.TryGetValue(testCase.Name, out int[]? counters))
            {
                counters = new int[2];
                outcomes[testCase.Name] = counters;
                classes.Add(testCase.Name);
            }
            List<Violation> observed = checker.Check(testCase.Start, testCase.Trace);
            counters[1]++;
            if (testCase.Expected.Length == 0)
            {
                if (observed.Count == 0) counters[0]++;
            }
            else if (observed.Count == 1 && observed[0].Step == testCase.Step
                && Array.IndexOf(testCase.Expected, observed[0].Kind) >= 0)
            {
                counters[0]++;
            }
        }
        Check(cases.Count >= 200, "injection suite size: " + cases.Count);
        foreach (string name in classes)
        {
            int[] counters = outcomes[name];
            Check(counters[1] >= 1 && counters[0] == counters[1],
                "injection class '" + name + "': " + counters[0] + "/" + counters[1] + " behaved as predicted");
        }
        Check(outcomes.ContainsKey("indirect-within-table (documented blind spot)")
            && outcomes.ContainsKey("stack-underflow") && outcomes.ContainsKey("call-target"),
            "the coarse CFI blind spot and the underflow cases were exercised");
        Check(outcomes["return-address"][0] >= 50 && outcomes["edge"][0] >= 50,
            "return corruption and edge violations are caught in bulk");

        Console.WriteLine("  blocks=" + program.Blocks.Count + " edges=" + program.EdgeCount()
            + " backEdges={" + string.Join(" ", domBack) + "} coverage=" + coverage.ToString("F3"));
        Console.WriteLine("  walks: " + walks.Count + " executions, " + totalTransfers + " transfers, shadow depth "
            + maxShadowDepth + ", verified steps " + checker.StepsVerified);
        Console.WriteLine("  injections: " + cases.Count + " cases in " + classes.Count + " classes, all as predicted");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
