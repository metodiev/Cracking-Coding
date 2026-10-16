// Task 672 - Implement liveness analysis on small CFG and dead code elimination.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A backward dataflow analysis over a tiny imperative IR (`v = k + uses...` definitions and
// `use v...` side-effecting statements, blocks ending in an optional two-way branch on a variable),
// followed by the optimisation it exists for:
//   * liveness. Each block has USE = variables read before being written in the block and DEF = variables
//     written in it, and the system
//         LIVE_OUT[B] = union of LIVE_IN[S] over successors S
//         LIVE_IN[B]  = USE[B] union (LIVE_OUT[B] - DEF[B])
//     is iterated backwards until a full pass changes nothing; the per-instruction live sets are then
//     recovered by walking each block backwards from LIVE_OUT. Every round is recorded, and the file
//     asserts the sets never shrink (that monotonicity is what makes the iteration converge).
//   * dead code elimination. Walking each block backwards, a definition is deleted when its variable is
//     not live after it; side-effecting statements are never deleted, and neither is a definition whose
//     variable is read by a branch, a print, or any surviving instruction. The deletion loop repeats until
//     nothing more can be removed and reports how many instructions went.
// Nothing here is asserted by inspecting the analyser's own output. Correctness is established by
// execution: a small interpreter runs the CFG over deterministic pseudo-random inputs, and the
// *observable behaviour* — the ordered print trace and the exact sequence of executed blocks — must be
// identical before and after elimination, for every sampled input. On top of that, each deleted
// instruction is re-checked on its own by deleting exactly that one instruction from the original program
// and confirming the behaviour still matches, which validates the individual removals rather than the
// aggregate. The elimination is also verified to be complete (the surviving program has no definition
// whose variable is unlived) and conservative (no side effect is ever dropped).
// Interpreted programs are bounded by a step limit, which both versions reach at the same point because
// liveness guarantees the branch variables keep their values.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic pseudo-randomness for inputs and generated programs.
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextBelow(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    internal static int NextSigned(ref ulong state, int bound) => NextBelow(ref state, 2 * bound + 1) - bound;
}

internal sealed class Ins
{
    internal string Var = "";                    // empty for a side-effecting statement
    internal int Constant;
    internal List<string> Uses = new List<string>();
    internal bool SideEffect;

    internal static Ins Defines(string variable, int constant, params string[] uses)
    {
        var instruction = new Ins { Var = variable, Constant = constant };
        instruction.Uses.AddRange(uses);
        return instruction;
    }

    internal static Ins Effect(params string[] uses)
    {
        var instruction = new Ins { SideEffect = true };
        instruction.Uses.AddRange(uses);
        return instruction;
    }

    internal Ins Clone()
    {
        var copy = new Ins { Var = Var, Constant = Constant, SideEffect = SideEffect };
        copy.Uses.AddRange(Uses);
        return copy;
    }
}

internal sealed class Blk
{
    internal string Name = "";
    internal List<Ins> Instructions = new List<Ins>();
    internal string BranchVar = "";              // empty when the block simply falls through
    internal List<int> Successors = new List<int>();

    internal Blk Clone()
    {
        var copy = new Blk { Name = Name, BranchVar = BranchVar };
        foreach (Ins instruction in Instructions) copy.Instructions.Add(instruction.Clone());
        copy.Successors.AddRange(Successors);
        return copy;
    }
}

internal sealed class Cfg
{
    internal List<Blk> Blocks = new List<Blk>();
    internal HashSet<string> Inputs = new HashSet<string>();

    internal Blk Add(string name) { var block = new Blk { Name = name }; Blocks.Add(block); return block; }
    internal void Edge(int from, int to) => Blocks[from].Successors.Add(to);

    internal Cfg Clone()
    {
        var copy = new Cfg();
        foreach (Blk block in Blocks) copy.Blocks.Add(block.Clone());
        foreach (string input in Inputs) copy.Inputs.Add(input);
        return copy;
    }

    /// Variables that must start with a value: everything read before it is written on the fall-through
    /// path from the entry. Computed by a simple forward scan rather than hard-coded.
    internal void DiscoverInputs()
    {
        Inputs = new HashSet<string>();
        var defined = new HashSet<string>();
        var seen = new HashSet<int>();
        var queue = new Queue<int>();
        queue.Enqueue(0);
        seen.Add(0);
        while (queue.Count > 0)
        {
            int b = queue.Dequeue();
            foreach (Ins instruction in Blocks[b].Instructions)
            {
                foreach (string use in instruction.Uses) if (!defined.Contains(use)) Inputs.Add(use);
                if (instruction.Var.Length > 0) defined.Add(instruction.Var);
            }
            if (Blocks[b].BranchVar.Length > 0 && !defined.Contains(Blocks[b].BranchVar))
            {
                Inputs.Add(Blocks[b].BranchVar);
            }
            foreach (int successor in Blocks[b].Successors)
            {
                if (seen.Add(successor)) queue.Enqueue(successor);
            }
        }
    }
}

internal sealed class Live
{
    internal HashSet<string>[] LiveIn = Array.Empty<HashSet<string>>();
    internal HashSet<string>[] LiveOut = Array.Empty<HashSet<string>>();
    internal int Rounds;
    internal bool Monotone = true;
    internal bool FixedPoint = true;
}

internal sealed class Behaviour
{
    internal List<string> Trace = new List<string>();
    internal List<int> Path = new List<int>();
    internal bool HitStepLimit;
}

internal static class Analysis
{
    internal static Live Solve(Cfg cfg)
    {
        int n = cfg.Blocks.Count;
        var live = new Live
        {
            LiveIn = new HashSet<string>[n],
            LiveOut = new HashSet<string>[n]
        };
        var use = new HashSet<string>[n];
        var def = new HashSet<string>[n];
        for (int b = 0; b < n; b++)
        {
            live.LiveIn[b] = new HashSet<string>();
            live.LiveOut[b] = new HashSet<string>();
            use[b] = new HashSet<string>();
            def[b] = new HashSet<string>();
            foreach (Ins instruction in cfg.Blocks[b].Instructions)
            {
                foreach (string variable in instruction.Uses) if (!def[b].Contains(variable)) use[b].Add(variable);
                if (instruction.Var.Length > 0) def[b].Add(instruction.Var);
            }
            if (cfg.Blocks[b].BranchVar.Length > 0 && !def[b].Contains(cfg.Blocks[b].BranchVar))
            {
                use[b].Add(cfg.Blocks[b].BranchVar);
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            live.Rounds++;
            for (int b = n - 1; b >= 0; b--)
            {
                int inBefore = live.LiveIn[b].Count;
                int outBefore = live.LiveOut[b].Count;
                var outgoing = new HashSet<string>();
                foreach (int successor in cfg.Blocks[b].Successors)
                {
                    foreach (string variable in live.LiveIn[successor]) outgoing.Add(variable);
                }
                var incoming = new HashSet<string>(use[b]);
                foreach (string variable in outgoing) if (!def[b].Contains(variable)) incoming.Add(variable);
                if (!live.LiveOut[b].SetEquals(outgoing) || !live.LiveIn[b].SetEquals(incoming))
                {
                    changed = true;
                }
                live.LiveOut[b] = outgoing;
                live.LiveIn[b] = incoming;
                if (live.LiveIn[b].Count < inBefore || live.LiveOut[b].Count < outBefore) live.Monotone = false;
            }
        }

        // Recompute one more round and confirm nothing moves: a genuine fixpoint, not just a stable print.
        for (int b = 0; b < n; b++)
        {
            var outgoing = new HashSet<string>();
            foreach (int successor in cfg.Blocks[b].Successors)
            {
                foreach (string variable in live.LiveIn[successor]) outgoing.Add(variable);
            }
            var incoming = new HashSet<string>(use[b]);
            foreach (string variable in outgoing) if (!def[b].Contains(variable)) incoming.Add(variable);
            if (!live.LiveOut[b].SetEquals(outgoing) || !live.LiveIn[b].SetEquals(incoming)) live.FixedPoint = false;
        }
        return live;
    }

    /// Live set immediately after each instruction of the given block, in instruction order. The branch
    /// variable is read by the terminator, i.e. *after* the last instruction, so it seeds the walk.
    internal static List<HashSet<string>> InstructionLiveOut(Cfg cfg, Live live, int block)
    {
        var result = new List<HashSet<string>>();
        var current = new HashSet<string>(live.LiveOut[block]);
        if (cfg.Blocks[block].BranchVar.Length > 0) current.Add(cfg.Blocks[block].BranchVar);
        var perInstruction = new List<HashSet<string>>();
        for (int i = cfg.Blocks[block].Instructions.Count - 1; i >= 0; i--)
        {
            perInstruction.Add(new HashSet<string>(current));
            Ins instruction = cfg.Blocks[block].Instructions[i];
            if (instruction.Var.Length > 0) current.Remove(instruction.Var);
            foreach (string variable in instruction.Uses) current.Add(variable);
        }
        for (int i = perInstruction.Count - 1; i >= 0; i--) result.Add(perInstruction[i]);
        return result;
    }

    /// Deletes definitions whose variable is not live afterwards. Returns the (block, instruction index)
    /// pairs that were deleted, in terms of the *original* layout, so the caller can validate each removal.
    internal static List<int[]> Eliminate(Cfg cfg, out int passes)
    {
        int n = cfg.Blocks.Count;
        var originalCount = new int[n];
        var origins = new List<int>[n];
        for (int b = 0; b < n; b++)
        {
            originalCount[b] = cfg.Blocks[b].Instructions.Count;
            origins[b] = new List<int>();
            for (int i = 0; i < originalCount[b]; i++) origins[b].Add(i);
        }

        passes = 0;
        while (true)
        {
            passes++;
            Live live = Solve(cfg);
            int removedThisPass = 0;
            for (int b = 0; b < n; b++)
            {
                List<HashSet<string>> after = InstructionLiveOut(cfg, live, b);
                var kept = new List<Ins>();
                var keptOrigins = new List<int>();
                for (int i = 0; i < cfg.Blocks[b].Instructions.Count; i++)
                {
                    Ins instruction = cfg.Blocks[b].Instructions[i];
                    bool needed = instruction.SideEffect
                                  || (instruction.Var.Length > 0 && after[i].Contains(instruction.Var));
                    if (needed)
                    {
                        kept.Add(instruction);
                        keptOrigins.Add(origins[b][i]);
                    }
                    else
                    {
                        removedThisPass++;
                    }
                }
                cfg.Blocks[b].Instructions = kept;
                origins[b] = keptOrigins;
            }
            if (removedThisPass == 0)
            {
                var removed = new List<int[]>();
                for (int b = 0; b < n; b++)
                {
                    var survivors = new bool[originalCount[b]];
                    foreach (int origin in origins[b]) survivors[origin] = true;
                    for (int i = 0; i < originalCount[b]; i++)
                    {
                        if (!survivors[i]) removed.Add(new[] { b, i });
                    }
                }
                return removed;
            }
        }
    }

    /// Interpreter: evaluates a program for one input assignment and records everything observable.
    internal static Behaviour Execute(Cfg cfg, Dictionary<string, int> inputs, int stepLimit)
    {
        var behaviour = new Behaviour();
        var environment = new Dictionary<string, int>();
        foreach (KeyValuePair<string, int> pair in inputs) environment[pair.Key] = pair.Value;
        int block = 0;
        for (int step = 0; step < stepLimit; step++)
        {
            behaviour.Path.Add(block);
            foreach (Ins instruction in cfg.Blocks[block].Instructions)
            {
                if (instruction.SideEffect)
                {
                    var text = new List<string>();
                    foreach (string use in instruction.Uses) text.Add(use + "=" + Value(environment, use));
                    behaviour.Trace.Add("print " + string.Join(",", text));
                    continue;
                }
                int value = instruction.Constant;
                foreach (string use in instruction.Uses) value += Value(environment, use);
                environment[instruction.Var] = value;
            }
            List<int> successors = cfg.Blocks[block].Successors;
            if (successors.Count == 0) return behaviour;
            if (successors.Count == 1) { block = successors[0]; continue; }
            int condition = Value(environment, cfg.Blocks[block].BranchVar);
            block = condition >= 0 ? successors[0] : successors[1];
        }
        behaviour.HitStepLimit = true;
        return behaviour;
    }

    private static int Value(Dictionary<string, int> environment, string variable)
    {
        int value;
        return environment.TryGetValue(variable, out value) ? value : 0;
    }

    /// Counts definitions whose variable is not live after them — must be zero on an optimised program.
    internal static int DeadDefinitions(Cfg cfg)
    {
        Live live = Solve(cfg);
        int dead = 0;
        for (int b = 0; b < cfg.Blocks.Count; b++)
        {
            List<HashSet<string>> after = InstructionLiveOut(cfg, live, b);
            for (int i = 0; i < cfg.Blocks[b].Instructions.Count; i++)
            {
                Ins instruction = cfg.Blocks[b].Instructions[i];
                if (instruction.SideEffect) continue;
                if (!after[i].Contains(instruction.Var)) dead++;
            }
        }
        return dead;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static string Show(HashSet<string> set)
    {
        var list = new List<string>(set);
        list.Sort();
        return "{" + string.Join(",", list) + "}";
    }

    /// x = 1; x = 2; use x — the first definition is dead on arrival.
    private static Cfg StraightLine()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("x", 1));
        entry.Instructions.Add(Ins.Defines("x", 2));
        entry.Instructions.Add(Ins.Effect("x"));
        return cfg;
    }

    /// A branch whose left arm is ignored: only the right arm's value is printed.
    private static Cfg Branch()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("live", 3));
        entry.Instructions.Add(Ins.Defines("dead", 7));
        Blk head = cfg.Add("H");
        head.BranchVar = "live";
        Blk left = cfg.Add("L");
        left.Instructions.Add(Ins.Defines("picked", 10, "live"));
        Blk right = cfg.Add("R");
        right.Instructions.Add(Ins.Defines("picked", 20));
        Blk join = cfg.Add("J");
        join.Instructions.Add(Ins.Effect("picked"));
        cfg.Edge(0, 1);
        cfg.Edge(1, 2);
        cfg.Edge(1, 3);
        cfg.Edge(2, 4);
        cfg.Edge(3, 4);
        return cfg;
    }

    /// Loop with a counter: the counter is live across the back edge, the accumulator is printed.
    private static Cfg Loop()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("i", 3));
        entry.Instructions.Add(Ins.Defines("unused", 5));
        Blk head = cfg.Add("H");
        head.BranchVar = "i";
        Blk body = cfg.Add("B");
        body.Instructions.Add(Ins.Defines("i", -1, "i"));
        body.Instructions.Add(Ins.Defines("scratch", 1, "i"));
        Blk exit = cfg.Add("X");
        exit.Instructions.Add(Ins.Effect("i"));
        cfg.Edge(0, 1);
        cfg.Edge(1, 2);
        cfg.Edge(1, 3);
        cfg.Edge(2, 1);
        return cfg;
    }

    private static Cfg RandomCfg(ref ulong state)
    {
        var cfg = new Cfg();
        string[] names = { "a", "b", "c", "d" };
        int blocks = 4 + Rng.NextBelow(ref state, 4);
        for (int b = 0; b < blocks; b++)
        {
            Blk block = cfg.Add("B" + b);
            int count = 1 + Rng.NextBelow(ref state, 3);
            for (int i = 0; i < count; i++)
            {
                if (Rng.NextBelow(ref state, 4) == 0)
                {
                    block.Instructions.Add(Ins.Effect(names[Rng.NextBelow(ref state, names.Length)]));
                    continue;
                }
                var uses = new List<string>();
                int useCount = Rng.NextBelow(ref state, 3);
                for (int u = 0; u < useCount; u++) uses.Add(names[Rng.NextBelow(ref state, names.Length)]);
                block.Instructions.Add(Ins.Defines(names[Rng.NextBelow(ref state, names.Length)],
                                                  Rng.NextSigned(ref state, 4), uses.ToArray()));
            }
            if (Rng.NextBelow(ref state, 3) == 0)
            {
                block.BranchVar = names[Rng.NextBelow(ref state, names.Length)];
            }
        }
        for (int b = 1; b < blocks; b++) cfg.Edge(b - 1, b);
        for (int b = 0; b < blocks; b++)
        {
            int edges = Rng.NextBelow(ref state, 3);
            for (int e = 0; e < edges; e++)
            {
                int target = Rng.NextBelow(ref state, blocks);
                if (target != b) cfg.Edge(b, target);
            }
        }
        cfg.DiscoverInputs();
        return cfg;
    }

    private static Dictionary<string, int> InputsFor(Cfg cfg, ref ulong state)
    {
        var inputs = new Dictionary<string, int>();
        foreach (string name in cfg.Inputs) inputs[name] = Rng.NextSigned(ref state, 3);
        return inputs;
    }

    private static bool SameBehaviour(Behaviour a, Behaviour b)
    {
        if (a.Trace.Count != b.Trace.Count) return false;
        for (int i = 0; i < a.Trace.Count; i++) if (a.Trace[i] != b.Trace[i]) return false;
        if (a.Path.Count != b.Path.Count) return false;
        for (int i = 0; i < a.Path.Count; i++) if (a.Path[i] != b.Path[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- straight-line liveness and elimination --------------------------------------------
        Cfg line = StraightLine();
        Live lineLive = Analysis.Solve(line);
        Check(lineLive.FixedPoint && lineLive.Monotone, "the straight-line liveness sets are a monotone fixpoint");
        Check(Show(lineLive.LiveOut[0]) == "{}", "nothing is live at the end of the straight-line program");
        Check(Show(lineLive.LiveIn[0]) == "{}", "so nothing is live at its start either");
        List<HashSet<string>> after = Analysis.InstructionLiveOut(line, lineLive, 0);
        Check(Show(after[0]) == "{}", "the first definition of x is not live after itself: " + Show(after[0]));
        Check(Show(after[1]) == "{x}", "the second one is, because the print reads it: " + Show(after[1]));
        int passes;
        List<int[]> removed = Analysis.Eliminate(line, out passes);
        Check(removed.Count == 1 && line.Blocks[0].Instructions.Count == 2,
              "exactly the overwritten definition is removed (removed " + removed.Count + ")");
        Check(removed[0][0] == 0 && removed[0][1] == 0, "and it is the first instruction");
        Check(line.Blocks[0].Instructions[1].SideEffect, "the side-effecting print survives");

        // ---- branches: only definitions that can reach a use are kept ---------------------------
        Cfg branch = Branch();
        Live branchLive = Analysis.Solve(branch);
        Check(Show(branchLive.LiveIn[1]) == "{live}", "the branch variable is live at the branch: " + Show(branchLive.LiveIn[1]));
        Check(Show(branchLive.LiveIn[3]) == "{}", "nothing flows into the right arm: " + Show(branchLive.LiveIn[3]));
        Check(Show(branchLive.LiveIn[4]) == "{picked}", "the join needs picked: " + Show(branchLive.LiveIn[4]));
        int branchRemoved = Analysis.Eliminate(branch, out passes).Count;
        Check(branchRemoved == 1, "only the unreferenced 'dead' definition goes (removed " + branchRemoved + ")");
        Check(branch.Blocks[0].Instructions.Count == 1, "the definition feeding the branch is kept");
        Check(branch.Blocks[3].Instructions.Count == 1, "and so is the right arm's definition of picked");

        // ---- loops: the counter is live across the back edge -----------------------------------
        Cfg loop = Loop();
        Live loopLive = Analysis.Solve(loop);
        Check(Show(loopLive.LiveIn[1]) == "{i}", "the loop header reads i, and the back edge keeps it live: "
              + Show(loopLive.LiveIn[1]));
        Check(Show(loopLive.LiveOut[2]) == "{i}", "the body's live-out carries the counter round the loop");
        int loopRemoved = Analysis.Eliminate(loop, out passes).Count;
        Check(loopRemoved == 2, "the unused and the scratch definition both go (removed " + loopRemoved + ")");
        Check(loop.Blocks[0].Instructions.Count == 1 && loop.Blocks[2].Instructions.Count == 1,
              "the counter's initialisation and the loop body's update survive");

        // ---- elimination preserves observable behaviour ----------------------------------------
        ulong state = 672001UL;
        int programs = 40;
        int totalRemoved = 0;
        int programsWithRemovals = 0;
        int individualRemovalsChecked = 0;
        for (int p = 0; p < programs; p++)
        {
            Cfg original = RandomCfg(ref state);
            Cfg optimised = original.Clone();
            int gone = Analysis.Eliminate(optimised, out passes).Count;
            totalRemoved += gone;
            if (gone > 0) programsWithRemovals++;
            Check(Analysis.DeadDefinitions(optimised) == 0,
                  "program " + p + ": the optimised program has no dead definitions left");
            Check(passes >= 1, "program " + p + ": elimination ran at least one pass");
            for (int t = 0; t < 20; t++)
            {
                Dictionary<string, int> inputs = InputsFor(original, ref state);
                Behaviour before = Analysis.Execute(original, inputs, 200);
                Behaviour afterRun = Analysis.Execute(optimised, inputs, 200);
                Check(SameBehaviour(before, afterRun),
                      "program " + p + " input " + t + ": the print trace and block path are unchanged"
                      + " (before " + before.Trace.Count + " prints, after " + afterRun.Trace.Count + ")");
            }
        }
        Check(programsWithRemovals >= 20, "elimination did something in " + programsWithRemovals + " of "
              + programs + " random programs");
        Check(totalRemoved >= 40, "the random sweep deleted " + totalRemoved + " instructions in total");

        // ---- every individual deletion is validated on its own ---------------------------------
        ulong perCase = 672999UL;
        for (int p = 0; p < programs; p++)
        {
            Cfg original = RandomCfg(ref perCase);
            Cfg optimised = original.Clone();
            List<int[]> deleted = Analysis.Eliminate(optimised, out passes);
            foreach (int[] pair in deleted)
            {
                Cfg singleDelete = original.Clone();
                singleDelete.Blocks[pair[0]].Instructions.RemoveAt(pair[1]);
                singleDelete.DiscoverInputs();
                for (int t = 0; t < 5; t++)
                {
                    Dictionary<string, int> inputs = InputsFor(original, ref perCase);
                    Check(SameBehaviour(Analysis.Execute(original, inputs, 200),
                                        Analysis.Execute(singleDelete, inputs, 200)),
                          "program " + p + " block " + pair[0] + " instruction " + pair[1]
                          + ": deleting just this instruction changes nothing observable");
                }
                individualRemovalsChecked++;
            }
        }
        Check(individualRemovalsChecked >= 20,
              "the per-deletion replay checked " + individualRemovalsChecked + " individual deletions");

        // ---- degenerate programs --------------------------------------------------------------
        var empty = new Cfg();
        empty.Add("E");
        Live emptyLive = Analysis.Solve(empty);
        Check(emptyLive.LiveIn[0].Count == 0 && emptyLive.LiveOut[0].Count == 0, "an empty block has empty live sets");
        var allDead = new Cfg();
        allDead.Add("E").Instructions.Add(Ins.Defines("z", 4));
        int deadRemoved = Analysis.Eliminate(allDead, out passes).Count;
        Check(deadRemoved == 1 && allDead.Blocks[0].Instructions.Count == 0,
              "a program consisting of one unused definition reduces to nothing");
        var sideEffectOnly = new Cfg();
        sideEffectOnly.Add("E").Instructions.Add(Ins.Effect("q"));
        int effectRemoved = Analysis.Eliminate(sideEffectOnly, out passes).Count;
        Check(effectRemoved == 0, "a side effect is never removed, even when it reads an input");

        Console.WriteLine("straight line: removed " + removed + " of 3, remaining "
              + line.Blocks[0].Instructions.Count);
        Console.WriteLine("random sweep : " + totalRemoved + " removals over " + programs + " programs");
        Console.WriteLine("per-deletion replays verified: " + individualRemovalsChecked);
        Console.WriteLine("PASS");
    }
}
