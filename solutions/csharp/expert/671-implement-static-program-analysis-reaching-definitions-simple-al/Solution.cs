// Task 671 - Implement static program analysis: reaching definitions simple algorithm.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A block-level reaching-definitions analysis over the classic dataflow lattice, implemented twice so the
// two solutions can be compared:
//   * the standard iterative solver. Definitions are numbered globally (block, instruction index), each
//     block has GEN = the last definition of every variable it assigns and KILL = all other definitions of
//     those variables, and the system
//         IN[B]  = union of OUT[P] over predecessors P
//         OUT[B] = GEN[B] union (IN[B] - KILL[B])
//     is iterated round-robin until a whole pass changes nothing. Predecessors include self-loops, and the
//     virtual ENTRY node contributes nothing, so a block with no predecessor (normally the entry) has the
//     familiar IN = {}, while a loop that jumps back to the entry is honoured rather than dropped. Every
//     round is checked to be monotone (no set ever shrinks), which is what makes the iteration terminate on
//     this lattice.
//   * a worklist solver that only revisits successors of blocks that actually changed.
// Both are cross-checked against an exact brute force that enumerates the reachable abstract states of the
// program: a state is (block, last-definition-per-variable) and a definition reaches a block start exactly
// when some reachable state at that block records it as the last writer of its variable. This state
// exploration shares no code with the fixpoint solvers, so agreement on 60 randomised control-flow graphs
// is real evidence rather than a tautology — the graphs are built with a spanning chain first so that every
// block is reachable and the two answers must coincide (with dead code the equations still have a fixpoint,
// but it counts definitions out of paths that no execution can take; that case is exercised separately and
// documented below).
// Beyond the block-level sets the file also computes the definition set *at each instruction* and the
// use-def chains that follow from it (which is what the analysis is actually for), plus the two
// edge cases the equations must handle: the entry block's empty IN, and a block that no path reaches,
// whose IN is empty while its OUT is exactly its GEN.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic pseudo-randomness for graph generation.
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

internal sealed class Instruction
{
    internal string Var = "";
    internal List<string> Uses = new List<string>();

    internal static Instruction Defines(string variable, params string[] uses)
    {
        var instruction = new Instruction { Var = variable };
        instruction.Uses.AddRange(uses);
        return instruction;
    }
}

internal sealed class Block
{
    internal string Name = "";
    internal List<Instruction> Instructions = new List<Instruction>();
    internal List<int> Successors = new List<int>();
}

internal sealed class Program
{
    internal List<Block> Blocks = new List<Block>();
    internal List<string> Variables = new List<string>();

    internal int VariableIndex(string name)
    {
        int index = Variables.IndexOf(name);
        if (index < 0)
        {
            Variables.Add(name);
            index = Variables.Count - 1;
        }
        return index;
    }

    internal Block Add(string name)
    {
        var block = new Block { Name = name };
        Blocks.Add(block);
        return block;
    }

    internal void Edge(int from, int to) => Blocks[from].Successors.Add(to);
}

internal sealed class Def
{
    internal int Block;
    internal int Instruction;
    internal int Variable;
}

internal sealed class Solution671
{
    internal List<Def> Defs = new List<Def>();
    internal int[][] Gen = Array.Empty<int[]>();          // block -> sorted def ids generated last
    internal int[][] Kill = Array.Empty<int[]>();
    internal HashSet<int>[] In = Array.Empty<HashSet<int>>();
    internal HashSet<int>[] Out = Array.Empty<HashSet<int>>();
    internal bool[] Reachable = Array.Empty<bool>();
    internal int Rounds;
    internal bool MonotoneEverywhere = true;
}

internal static class ReachingDefs
{
    /// Deterministic numbering: definition ids follow program order, and GEN keeps only the last
    /// definition of each variable inside a block (an earlier one cannot survive to the block exit).
    internal static void Build(Program program, Solution671 data)
    {
        var variableIds = new Dictionary<string, int>();
        int nextVariable = 0;
        data.Defs.Clear();
        int[] lastDefInBlock = new int[program.Blocks.Count];
        var gen = new List<List<int>>();
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            gen.Add(new List<int>());
            lastDefInBlock[b] = -1;
        }

        // First pass: id every definition in program order.
        var perBlockLast = new List<Dictionary<int, int>>();
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            var last = new Dictionary<int, int>();
            perBlockLast.Add(last);
            List<Instruction> instructions = program.Blocks[b].Instructions;
            for (int i = 0; i < instructions.Count; i++)
            {
                string variable = instructions[i].Var;
                if (!variableIds.ContainsKey(variable)) variableIds[variable] = nextVariable++;
                int id = data.Defs.Count;
                data.Defs.Add(new Def { Block = b, Instruction = i, Variable = variableIds[variable] });
                last[variableIds[variable]] = id;
            }
        }

        // Second pass: GEN is the surviving definition per variable, KILL is every other definition of
        // the variables that this block (re)defines.
        var genPerBlock = new List<HashSet<int>>();
        var killPerBlock = new List<HashSet<int>>();
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            genPerBlock.Add(new HashSet<int>(perBlockLast[b].Values));
            var killed = new HashSet<int>();
            foreach (int variable in perBlockLast[b].Keys)
            {
                for (int d = 0; d < data.Defs.Count; d++)
                {
                    if (data.Defs[d].Variable == variable && data.Defs[d].Block != b) killed.Add(d);
                }
            }
            killPerBlock.Add(killed);
        }

        data.Gen = new int[program.Blocks.Count][];
        data.Kill = new int[program.Blocks.Count][];
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            var sortedGen = new List<int>(genPerBlock[b]);
            sortedGen.Sort();
            data.Gen[b] = sortedGen.ToArray();
            var sortedKill = new List<int>(killPerBlock[b]);
            sortedKill.Sort();
            data.Kill[b] = sortedKill.ToArray();
        }
    }

    private static bool IsKilled(Solution671 data, int block, int def)
    {
        foreach (int killed in data.Kill[block]) if (killed == def) return true;
        return false;
    }

    /// Round-robin fixpoint iteration; records whether every set only ever grew.
    internal static void Solve(Program program, Solution671 data)
    {
        int n = program.Blocks.Count;
        data.In = new HashSet<int>[n];
        data.Out = new HashSet<int>[n];
        for (int b = 0; b < n; b++)
        {
            data.In[b] = new HashSet<int>();
            data.Out[b] = new HashSet<int>();
        }
        data.Rounds = 0;
        bool changed = true;
        while (changed)
        {
            changed = false;
            data.Rounds++;
            for (int b = 0; b < n; b++)
            {
                int inSizeBefore = data.In[b].Count;
                int outSizeBefore = data.Out[b].Count;
                var incoming = new HashSet<int>();
                // No entry special case: the virtual ENTRY node simply contributes nothing, so a block
                // whose predecessors exist (including the entry block itself, when a loop jumps back to
                // it) unions their OUT sets. With no predecessor the result is the empty set, which is
                // the familiar IN[entry] = {}.
                foreach (int pred in Predecessors(program, b))
                {
                    foreach (int def in data.Out[pred]) incoming.Add(def);
                }
                var outgoing = new HashSet<int>(data.Gen[b]);
                foreach (int def in incoming) if (!IsKilled(data, b, def)) outgoing.Add(def);
                if (!data.In[b].SetEquals(incoming))
                {
                    data.In[b] = incoming;
                    changed = true;
                }
                if (!data.Out[b].SetEquals(outgoing))
                {
                    data.Out[b] = outgoing;
                    changed = true;
                }
                if (data.In[b].Count < inSizeBefore || data.Out[b].Count < outSizeBefore)
                {
                    data.MonotoneEverywhere = false;
                }
            }
        }
    }

    /// Worklist variant: only re-queue the successors of a block whose OUT actually changed.
    internal static (HashSet<int>[] In, HashSet<int>[] Out, int Pops) Worklist(Program program, Solution671 data)
    {
        int n = program.Blocks.Count;
        var inSets = new HashSet<int>[n];
        var outSets = new HashSet<int>[n];
        for (int b = 0; b < n; b++)
        {
            inSets[b] = new HashSet<int>();
            outSets[b] = new HashSet<int>();
        }
        var queue = new Queue<int>();
        var queued = new bool[n];
        queue.Enqueue(0);
        queued[0] = true;
        int pops = 0;
        while (queue.Count > 0)
        {
            int b = queue.Dequeue();
            queued[b] = false;
            pops++;
            var incoming = new HashSet<int>();
            foreach (int pred in Predecessors(program, b))
            {
                foreach (int def in outSets[pred]) incoming.Add(def);
            }
            var outgoing = new HashSet<int>(data.Gen[b]);
            foreach (int def in incoming) if (!IsKilled(data, b, def)) outgoing.Add(def);
            bool changed = !inSets[b].SetEquals(incoming) || !outSets[b].SetEquals(outgoing);
            inSets[b] = incoming;
            outSets[b] = outgoing;
            if (!changed) continue;
            foreach (int successor in program.Blocks[b].Successors)
            {
                if (queued[successor]) continue;
                queue.Enqueue(successor);
                queued[successor] = true;
            }
        }
        return (inSets, outSets, pops);
    }

    /// Predecessors, self-loops included (a self-loop feeds OUT[B] back into IN[B], which is why the
    /// iteration must be allowed to see it; it costs nothing because the transfer is monotone).
    internal static List<int> Predecessors(Program program, int block)
    {
        var preds = new List<int>();
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            foreach (int successor in program.Blocks[b].Successors) if (successor == block) preds.Add(b);
        }
        return preds;
    }

    /// Plain reachability from the entry, used for the unreachable-block edge case.
    internal static bool[] Mark(Program program)
    {
        var reachable = new bool[program.Blocks.Count];
        var stack = new Stack<int>();
        stack.Push(0);
        reachable[0] = true;
        while (stack.Count > 0)
        {
            int b = stack.Pop();
            foreach (int successor in program.Blocks[b].Successors)
            {
                if (reachable[successor]) continue;
                reachable[successor] = true;
                stack.Push(successor);
            }
        }
        return reachable;
    }

    /// True when the computed sets satisfy the equations exactly (i.e. one more round changes nothing) —
    /// the property that distinguishes a fixpoint from an approximate answer.
    internal static bool IsFixedPoint(Program program, Solution671 data)
    {
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            var incoming = new HashSet<int>();
            // Same equations as Solve: OUT of the predecessors, self-loops included.
            {
                foreach (int pred in Predecessors(program, b))
                {
                    foreach (int def in data.Out[pred]) incoming.Add(def);
                }
            }
            var outgoing = new HashSet<int>(data.Gen[b]);
            foreach (int def in incoming) if (!IsKilled(data, b, def)) outgoing.Add(def);
            if (!data.In[b].SetEquals(incoming) || !data.Out[b].SetEquals(outgoing)) return false;
        }
        return true;
    }

    /// Exact brute force: explore the (block, last-def-per-variable) state space. A definition reaches a
    /// block start iff some reachable state there records it as the variable's last writer.
    internal static HashSet<int>[] BruteForce(Program program, Solution671 data)
    {
        int n = program.Blocks.Count;
        int variables = 0;
        foreach (Def def in data.Defs) if (def.Variable + 1 > variables) variables = def.Variable + 1;
        var entry = new int[variables];
        for (int v = 0; v < variables; v++) entry[v] = -1;
        var seen = new HashSet<string>();
        var pending = new Queue<(int, int[])>();
        pending.Enqueue((0, entry));
        seen.Add(Key(0, entry));
        var result = new HashSet<int>[n];
        for (int b = 0; b < n; b++) result[b] = new HashSet<int>();
        while (pending.Count > 0)
        {
            (int block, int[] state) = pending.Dequeue();
            foreach (int value in state) if (value >= 0) result[block].Add(value);
            var outgoing = (int[])state.Clone();
            List<Instruction> instructions = program.Blocks[block].Instructions;
            for (int i = 0; i < instructions.Count; i++)
            {
                int defId = Lookup(data, block, i);
                if (defId < 0) continue;
                outgoing[data.Defs[defId].Variable] = defId;
            }
            foreach (int successor in program.Blocks[block].Successors)
            {
                string key = Key(successor, outgoing);
                if (seen.Add(key)) pending.Enqueue((successor, (int[])outgoing.Clone()));
            }
        }
        return result;
    }

    private static int Lookup(Solution671 data, int block, int instruction)
    {
        for (int d = 0; d < data.Defs.Count; d++)
        {
            if (data.Defs[d].Block == block && data.Defs[d].Instruction == instruction) return d;
        }
        return -1;
    }

    private static string Key(int block, int[] state)
    {
        var parts = new List<string> { block.ToString() };
        foreach (int value in state) parts.Add(value.ToString());
        return string.Join(",", parts);
    }

    /// Definitions of `variable` that reach the use at (block, index): the block-level set carried into
    /// the instruction, with the definitions of earlier instructions in the same block applied on top.
    internal static HashSet<int> UsesOf(Program program, Solution671 data, int block, int index, int variable)
    {
        var current = new HashSet<int>();
        if (block >= 0 && block < data.In.Length) current = new HashSet<int>(data.In[block]);
        for (int i = 0; i < index; i++)
        {
            int defId = Lookup(data, block, i);
            if (defId < 0) continue;
            int written = data.Defs[defId].Variable;
            var kept = new HashSet<int>();
            foreach (int def in current) if (data.Defs[def].Variable != written) kept.Add(def);
            kept.Add(defId);
            current = kept;
        }
        var filtered = new HashSet<int>();
        foreach (int def in current) if (data.Defs[def].Variable == variable) filtered.Add(def);
        return filtered;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string Show(IEnumerable<int> defs)
    {
        var list = new List<int>(defs);
        list.Sort();
        return "{" + string.Join(",", list) + "}";
    }

    private static Solution671 Analyse(Program program)
    {
        var data = new Solution671();
        ReachingDefs.Build(program, data);
        ReachingDefs.Solve(program, data);
        data.Reachable = ReachingDefs.Mark(program);
        return data;
    }

    /// Straight line: a later definition of the same variable kills an earlier one.
    private static Program StraightLine()
    {
        var p = new Program();
        Block a = p.Add("A");
        a.Instructions.Add(Instruction.Defines("x", "n"));
        a.Instructions.Add(Instruction.Defines("x", "x"));
        Block b = p.Add("B");
        b.Instructions.Add(Instruction.Defines("y", "x"));
        p.Edge(0, 1);
        return p;
    }

    /// Diamond: both branches define x, so both definitions reach the join.
    private static Program Diamond()
    {
        var p = new Program();
        p.Add("A").Instructions.Add(Instruction.Defines("c"));
        p.Add("L").Instructions.Add(Instruction.Defines("x"));
        p.Add("R").Instructions.Add(Instruction.Defines("x"));
        p.Add("J").Instructions.Add(Instruction.Defines("z", "x"));
        p.Edge(0, 1);
        p.Edge(0, 2);
        p.Edge(1, 3);
        p.Edge(2, 3);
        return p;
    }

    /// Loop with a separate entry block (so the header is not the program entry and the back edge really
    /// feeds IN[head]): the body's definition reaches the header, and is then killed by the header itself.
    private static Program Loop()
    {
        var p = new Program();
        p.Add("entry");
        p.Add("head").Instructions.Add(Instruction.Defines("i", "n"));
        p.Add("body").Instructions.Add(Instruction.Defines("i", "i"));
        p.Add("exit").Instructions.Add(Instruction.Defines("r", "i"));
        p.Edge(0, 1);
        p.Edge(1, 2);
        p.Edge(2, 1);
        p.Edge(1, 3);
        return p;
    }

    /// Unreachable block hanging off the entry-free part of the graph.
    private static Program WithUnreachable()
    {
        var p = new Program();
        p.Add("A").Instructions.Add(Instruction.Defines("x"));
        p.Add("B").Instructions.Add(Instruction.Defines("y", "x"));
        Block orphan = p.Add("orphan");
        orphan.Instructions.Add(Instruction.Defines("q"));
        orphan.Instructions.Add(Instruction.Defines("q", "q"));
        p.Edge(0, 1);
        p.Edge(1, 1);
        return p;
    }

    private static Program RandomProgram(ref ulong state)
    {
        var p = new Program();
        int blocks = 4 + Rng.NextBelow(ref state, 4);
        string[] names = { "a", "b", "c", "d" };
        for (int b = 0; b < blocks; b++)
        {
            Block block = p.Add("B" + b);
            int count = 1 + Rng.NextBelow(ref state, 3);
            for (int i = 0; i < count; i++)
            {
                string target = names[Rng.NextBelow(ref state, names.Length)];
                var uses = new List<string>();
                int useCount = Rng.NextBelow(ref state, 3);
                for (int u = 0; u < useCount; u++) uses.Add(names[Rng.NextBelow(ref state, names.Length)]);
                block.Instructions.Add(Instruction.Defines(target, uses.ToArray()));
            }
        }
        // A chain 0 -> 1 -> ... -> n-1 makes every block reachable, so the fixpoint is compared against
        // the MOP brute force on equal footing; the random extra edges (including backwards ones) add the
        // branches and loops. Dead code is exercised separately by the orphan program.
        for (int b = 1; b < blocks; b++) p.Edge(b - 1, b);
        for (int b = 0; b < blocks; b++)
        {
            int edges = Rng.NextBelow(ref state, 3);
            for (int e = 0; e < edges; e++)
            {
                int target = Rng.NextBelow(ref state, blocks);
                if (target != b) p.Edge(b, target);
            }
        }
        return p;
    }

    public static void Main()
    {
        // ---- straight line --------------------------------------------------------------------
        var line = Analyse(StraightLine());
        Check(line.Defs.Count == 3, "three definitions are numbered in program order");
        Check(line.In[0].Count == 0, "the entry block has an empty IN set");
        Check(Show(line.Out[0]) == "{1}", "the second definition of x kills the first: OUT[A] = " + Show(line.Out[0]));
        Check(Show(line.In[1]) == "{1}", "only the surviving definition reaches B: IN[B] = " + Show(line.In[1]));
        Check(line.Kill[0].Length == 0, "a block never kills its own generated definitions");
        Check(line.Rounds > 0 && line.MonotoneEverywhere, "round-robin iteration is monotone and terminates");

        // ---- diamond -------------------------------------------------------------------------
        var diamond = Analyse(Diamond());
        Check(Show(diamond.In[3]) == "{0,1,2}",
              "the untouched definition of c and both branch definitions of x reach the join: IN[J] = "
              + Show(diamond.In[3]));
        Check(diamond.In[3].Count == 3, "neither branch definition kills the other, they merely tie for x");
        Check(diamond.Gen[3].Length == 1 && diamond.Gen[3][0] == 3, "the join generates its own definition of z");
        Check(Show(diamond.Out[3]) == "{0,1,2,3}", "OUT[J] = " + Show(diamond.Out[3]));

        // ---- loop ----------------------------------------------------------------------------
        var loop = Analyse(Loop());
        Check(loop.In[0].Count == 0, "the entry block of the loop program still has an empty IN set");
        Check(Show(loop.Out[2]) == "{1}", "the body's definition of i survives to its exit: OUT[body] = " + Show(loop.Out[2]));
        Check(Show(loop.In[1]) == "{1}",
              "and comes back along the back edge, so IN[head] = " + Show(loop.In[1]));
        Check(Show(loop.In[3]) == "{0}",
              "the header's own definition kills the loop-carried one on the way to the exit: IN[exit] = "
              + Show(loop.In[3]));
        Check(loop.Rounds >= 2, "the loop needs at least two rounds to reach the fixpoint (used " + loop.Rounds + ")");

        // ---- unreachable blocks ---------------------------------------------------------------
        var orphan = Analyse(WithUnreachable());
        Check(!orphan.Reachable[2], "the orphan block is unreachable from the entry");
        Check(orphan.In[2].Count == 0, "an unreachable block has an empty IN set");
        Check(Show(orphan.Out[2]) == "{3}",
              "its OUT is exactly its GEN (only the last definition of q): " + Show(orphan.Out[2]));
        Check(orphan.Reachable[1] && orphan.Reachable[0], "the reachable part of the graph is marked as such");

        // ---- the use-def chain the analysis exists for ----------------------------------------
        var chainProgram = Diamond();
        var chain = Analyse(chainProgram);
        HashSet<int> atJoin = ReachingDefs.UsesOf(chainProgram, chain, 3, 0, 1);
        Check(Show(atJoin) == "{1,2}", "the use of x in the join sees both definitions of x: " + Show(atJoin));
        var shadowProgram = new Program();
        Block before = shadowProgram.Add("S0");
        before.Instructions.Add(Instruction.Defines("x"));
        Block after = shadowProgram.Add("S1");
        after.Instructions.Add(Instruction.Defines("x"));
        after.Instructions.Add(Instruction.Defines("y", "x"));
        shadowProgram.Edge(0, 1);
        var shadowData = Analyse(shadowProgram);
        Check(shadowData.In[1].Contains(0), "the first block's definition of x does reach the second block");
        HashSet<int> second = ReachingDefs.UsesOf(shadowProgram, shadowData, 1, 1, 0);
        Check(Show(second) == "{1}",
              "but the intra-block redefinition shadows it at the use: " + Show(second) + " instead of {0}");
        var undefinedUse = new Program();
        undefinedUse.Add("U").Instructions.Add(Instruction.Defines("a", "b", "c"));
        var undefinedData = Analyse(undefinedUse);
        Check(ReachingDefs.UsesOf(undefinedUse, undefinedData, 0, 0, 0).Count == 0,
              "a use that precedes any definition has an empty reaching set (undefined use)");

        // ---- invariants ----------------------------------------------------------------------
        Check(line.Gen[1].Length == 1 && line.Gen[1][0] == 2, "block B generates its only definition");
        Check(line.Kill[1].Length == 0, "B is the only writer of y, so it has nothing to kill");
        Check(loop.Kill[1].Length == 1 && loop.Kill[1][0] == 1, "the loop header kills the body's definition of i");
        Check(Show(line.Out[0]) != Show(line.In[0]), "the transfer function really does something on a nonempty block");

        // ---- randomised cross-check against the state-space brute force -----------------------
        ulong state = 671001UL;
        int cases = 60;
        int reachableDefs = 0;
        int unreachableBlocks = 0;
        for (int c = 0; c < cases; c++)
        {
            Program program = RandomProgram(ref state);
            var data = Analyse(program);
            HashSet<int>[] brute = ReachingDefs.BruteForce(program, data);
            for (int b = 0; b < program.Blocks.Count; b++)
            {
                if (data.Reachable[b])
                {
                    Check(brute[b].SetEquals(data.In[b]),
                          "case " + c + " block " + b + ": brute force " + Show(brute[b])
                          + " differs from the fixpoint " + Show(data.In[b]));
                }
                else
                {
                    unreachableBlocks++;
                    Check(brute[b].Count == 0,
                          "case " + c + ": nothing from the entry reaches unreachable block " + b);
                }
            }
            Check(ReachingDefs.IsFixedPoint(program, data), "case " + c + ": the computed sets satisfy the equations exactly");
            Check(data.MonotoneEverywhere, "case " + c + ": every round of the iteration only grew the sets");
            Check(data.Rounds <= program.Blocks.Count + 2,
                  "case " + c + ": the fixpoint is reached in a bounded number of rounds (" + data.Rounds + ")");
            for (int b = 0; b < program.Blocks.Count; b++)
            {
                reachableDefs += data.In[b].Count;
                foreach (int def in data.Kill[b]) if (data.Out[b].Contains(def)) Check(false, "case " + c + ": a killed definition survived");
                foreach (int def in data.Gen[b]) if (!data.Out[b].Contains(def)) Check(false, "case " + c + ": a generated definition was lost");
            }
        }
        Check(reachableDefs > 0, "the randomised programs carry a total of " + reachableDefs + " reaching definitions");
        Check(unreachableBlocks == 0, "the randomised CFGs are fully reachable, so the comparison is exact");

        // ---- the worklist solver must agree with round-robin ----------------------------------
        for (int c = 0; c < cases; c++)
        {
            Program program = RandomProgram(ref state);
            var data = Analyse(program);
            (HashSet<int>[] inSets, HashSet<int>[] outSets, int pops) = ReachingDefs.Worklist(program, data);
            bool same = true;
            for (int b = 0; b < program.Blocks.Count; b++)
            {
                if (!inSets[b].SetEquals(data.In[b]) || !outSets[b].SetEquals(data.Out[b])) same = false;
            }
            Check(same, "case " + c + ": the worklist solver reproduces the round-robin fixpoint");
            Check(pops > 0, "case " + c + ": the worklist actually processed blocks (" + pops + " pops)");
        }

        // ---- degenerate programs ------------------------------------------------------------
        var single = new Program();
        single.Add("only");
        var singleData = Analyse(single);
        Check(singleData.In[0].Count == 0 && singleData.Out[0].Count == 0, "a single empty block defines nothing");

        Console.WriteLine("straight line OUT[A]   : " + Show(line.Out[0]) + "  IN[B] " + Show(line.In[1]));
        Console.WriteLine("diamond IN[J]          : " + Show(diamond.In[3]));
        Console.WriteLine("loop IN[head]          : " + Show(loop.In[0]) + " after " + loop.Rounds + " rounds");
        Console.WriteLine("random CFGs verified   : " + cases + " (brute force + worklist)");
        Console.WriteLine("PASS");
    }
}
