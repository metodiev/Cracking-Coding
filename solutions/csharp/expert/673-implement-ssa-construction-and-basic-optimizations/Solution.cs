// Task 673 - Implement SSA construction and basic optimizations.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// The minimal-SSA construction of Cytron et al., scaled down to a small imperative IR but following the
// paper's four steps exactly: dominators by iterative intersection, dominance frontiers by the standard
// "walk up the idom chain from each predecessor" rule, phi insertion by the iterated dominance-frontier
// worklist (phi functions are themselves definitions, so the worklist must be re-driven when a phi site
// is added), and renaming as a walk of the dominator tree with a per-variable stack of versions, filling
// each successor's phi arguments from the state at the end of the current block.
// Three groups of properties are then verified rather than trusted:
//   * structural SSA: every versioned name has exactly one definition site, every phi has one argument per
//     predecessor, and (the property that actually defines SSA) the definition of every name dominates
//     every one of its uses — checked against the dominator sets computed independently of the renamer.
//   * semantic: a small interpreter runs the original, the SSA, and the optimised SSA form over identical
//     pseudo-random inputs; the print trace and the whole sequence of executed blocks must be identical.
//   * phi justification: the renamer's reachable-definition cross-check (a state exploration over
//     last-definition-per-variable, the same technique the reaching-definitions task uses) must agree
//     that every surviving phi sits at a block where at least two different definitions can arrive.
// The basic optimisations are constant propagation (folding a definition whose operands all folded, and a
// phi whose arguments all folded to the same constant), trivial-phi elimination (a phi whose surviving
// arguments are one single name is replaced by that name), and dead-code elimination of unused
// definitions; they are iterated to a fixpoint and are only accepted because the interpreter says the
// behaviour did not change. The generated control-flow graphs keep the entry free of incoming edges, the
// assumption minimal SSA rests on (otherwise the phi at the entry would have no argument standing for the
// values coming in from outside).
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
    internal string Var = "";
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
}

internal sealed class Blk
{
    internal string Name = "";
    internal List<Ins> Instructions = new List<Ins>();
    internal string BranchVar = "";
    internal List<int> Successors = new List<int>();
    internal List<int> Predecessors = new List<int>();
}

internal sealed class Cfg
{
    internal List<Blk> Blocks = new List<Blk>();
    internal HashSet<string> Inputs = new HashSet<string>();

    internal Blk Add(string name) { var block = new Blk { Name = name }; Blocks.Add(block); return block; }

    internal void Edge(int from, int to)
    {
        Blocks[from].Successors.Add(to);
        Blocks[to].Predecessors.Add(from);
    }

    internal Cfg Clone()
    {
        var copy = new Cfg();
        for (int b = 0; b < Blocks.Count; b++)
        {
            Blk block = copy.Add(Blocks[b].Name);
            block.BranchVar = Blocks[b].BranchVar;
            foreach (Ins instruction in Blocks[b].Instructions)
            {
                var clone = new Ins { Var = instruction.Var, Constant = instruction.Constant, SideEffect = instruction.SideEffect };
                clone.Uses.AddRange(instruction.Uses);
                block.Instructions.Add(clone);
            }
            foreach (int successor in Blocks[b].Successors) copy.Edge(b, successor);
        }
        copy.DiscoverInputs();
        return copy;
    }

    internal void DiscoverInputs()
    {
        Inputs.Clear();
        var defined = new HashSet<string>();
        var queue = new Queue<int>();
        var seen = new HashSet<int>();
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
            if (Blocks[b].BranchVar.Length > 0 && !defined.Contains(Blocks[b].BranchVar)) Inputs.Add(Blocks[b].BranchVar);
            foreach (int successor in Blocks[b].Successors) if (seen.Add(successor)) queue.Enqueue(successor);
        }
    }
}

internal sealed class Behaviour
{
    internal List<string> Trace = new List<string>();
    internal List<int> Path = new List<int>();
    internal int Steps;
}

internal static class Interp
{
    internal static Behaviour Run(Cfg cfg, Dictionary<string, int> inputs, int stepLimit)
    {
        var behaviour = new Behaviour();
        var environment = new Dictionary<string, int>();
        foreach (KeyValuePair<string, int> pair in inputs) environment[pair.Key] = pair.Value;
        int block = 0;
        for (int step = 0; step < stepLimit; step++)
        {
            behaviour.Steps++;
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
            block = Value(environment, cfg.Blocks[block].BranchVar) >= 0 ? successors[0] : successors[1];
        }
        return behaviour;
    }

    private static int Value(Dictionary<string, int> environment, string variable)
    {
        int value;
        return environment.TryGetValue(variable, out value) ? value : 0;
    }
}

internal sealed class SsaBlock
{
    internal List<SsaIns> Phis = new List<SsaIns>();
    internal List<SsaIns> Instructions = new List<SsaIns>();
    internal string BranchVar = "";
    internal List<int> Successors = new List<int>();
    internal List<int> Predecessors = new List<int>();
}

internal sealed class SsaIns
{
    internal string Var = "";
    internal string Original = "";      // pre-renaming name of a phi definition
    internal int Constant;
    internal List<string> Uses = new List<string>();
    internal List<string> PhiArguments = new List<string>();
    internal bool Phi;
    internal bool SideEffect;

    internal SsaIns Copy()
    {
        var copy = new SsaIns { Var = Var, Original = Original, Constant = Constant, Phi = Phi, SideEffect = SideEffect };
        copy.Uses.AddRange(Uses);
        copy.PhiArguments.AddRange(PhiArguments);
        return copy;
    }
}

internal sealed class SsaProgram
{
    internal List<SsaBlock> Blocks = new List<SsaBlock>();
    internal List<int> DefBlock = new List<int>();       // per definition count: the block that defines it
    internal int Folds;
    internal int DroppedPhis;
    internal int EliminatedDefs;
}

internal static class Ssa
{
    /// Dominator sets by iterative intersection; entry dominates only itself to begin with.
    internal static HashSet<int>[] Dominators(Cfg cfg)
    {
        int n = cfg.Blocks.Count;
        var all = new HashSet<int>();
        for (int b = 0; b < n; b++) all.Add(b);
        var dom = new HashSet<int>[n];
        for (int b = 0; b < n; b++) dom[b] = new HashSet<int>(b == 0 ? new[] { 0 } : all);
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int b = 0; b < n; b++)
            {
                if (b == 0) continue;
                var next = new HashSet<int> { b };
                if (cfg.Blocks[b].Predecessors.Count > 0)
                {
                    HashSet<int>? common = null;
                    foreach (int pred in cfg.Blocks[b].Predecessors)
                    {
                        if (common == null) common = new HashSet<int>(dom[pred]);
                        else common.IntersectWith(dom[pred]);
                    }
                    if (common != null) foreach (int node in common) next.Add(node);
                }
                if (!dom[b].SetEquals(next))
                {
                    dom[b] = next;
                    changed = true;
                }
            }
        }
        return dom;
    }

    internal static int[] ImmediateDominators(Cfg cfg, HashSet<int>[] dom)
    {
        int n = cfg.Blocks.Count;
        var idom = new int[n];
        for (int b = 0; b < n; b++) idom[b] = b == 0 ? 0 : -1;
        for (int b = 1; b < n; b++)
        {
            int best = -1;
            foreach (int candidate in dom[b])
            {
                if (candidate == b) continue;
                if (best < 0 || dom[candidate].Count > dom[best].Count) best = candidate;
            }
            idom[b] = best;
        }
        return idom;
    }

    /// Dominance frontier: for every join block, walk up the idom chain from each predecessor.
    internal static HashSet<int>[] Frontiers(Cfg cfg, HashSet<int>[] dom, int[] idom)
    {
        int n = cfg.Blocks.Count;
        var frontiers = new HashSet<int>[n];
        for (int b = 0; b < n; b++) frontiers[b] = new HashSet<int>();
        for (int b = 0; b < n; b++)
        {
            if (cfg.Blocks[b].Predecessors.Count < 2) continue;
            foreach (int pred in cfg.Blocks[b].Predecessors)
            {
                int runner = pred;
                while (runner != idom[b] && runner >= 0)
                {
                    frontiers[runner].Add(b);
                    if (runner == idom[runner]) break;
                    runner = idom[runner];
                }
            }
        }
        return frontiers;
    }

    /// Iterated dominance frontier: phi functions are definitions too, so new sites re-enter the worklist.
    private static bool[] PhiSites(Cfg cfg, HashSet<int>[] frontiers, string variable)
    {
        int n = cfg.Blocks.Count;
        var placed = new bool[n];
        var queue = new Queue<int>();
        var queued = new bool[n];
        for (int b = 0; b < n; b++)
        {
            foreach (Ins instruction in cfg.Blocks[b].Instructions)
            {
                if (instruction.Var == variable && !queued[b])
                {
                    queued[b] = true;
                    queue.Enqueue(b);
                }
            }
        }
        while (queue.Count > 0)
        {
            int b = queue.Dequeue();
            foreach (int frontier in frontiers[b])
            {
                if (placed[frontier]) continue;
                placed[frontier] = true;
                if (!queued[frontier])
                {
                    queued[frontier] = true;
                    queue.Enqueue(frontier);
                }
            }
        }
        for (int b = 0; b < n; b++)
        {
            if (placed[b] && cfg.Blocks[b].Predecessors.Count < 2) placed[b] = false;
        }
        return placed;
    }

    /// Full construction: phi insertion followed by the dominator-tree renaming walk.
    internal static SsaProgram Build(Cfg cfg)
    {
        int n = cfg.Blocks.Count;
        var program = new SsaProgram();
        for (int b = 0; b < n; b++)
        {
            program.Blocks.Add(new SsaBlock
            {
                BranchVar = cfg.Blocks[b].BranchVar,
                Successors = new List<int>(cfg.Blocks[b].Successors),
                Predecessors = new List<int>(cfg.Blocks[b].Predecessors)
            });
        }

        HashSet<int>[] dom = Dominators(cfg);
        int[] idom = ImmediateDominators(cfg, dom);
        HashSet<int>[] frontiers = Frontiers(cfg, dom, idom);

        var variables = new List<string>();
        for (int b = 0; b < n; b++)
        {
            foreach (Ins instruction in cfg.Blocks[b].Instructions)
            {
                if (instruction.Var.Length > 0 && !variables.Contains(instruction.Var)) variables.Add(instruction.Var);
            }
        }

        // Insert phis, then rename through the dominator tree with a per-variable version stack.
        var stacks = new Dictionary<string, List<string>>();
        var counters = new Dictionary<string, int>();
        foreach (string variable in variables)
        {
            stacks[variable] = new List<string>();
            counters[variable] = 0;
            bool[] sites = PhiSites(cfg, frontiers, variable);
            for (int b = 0; b < n; b++)
            {
                if (!sites[b]) continue;
                var phi = new SsaIns { Var = variable, Original = variable, Phi = true };
                for (int i = 0; i < cfg.Blocks[b].Predecessors.Count; i++) phi.PhiArguments.Add("");
                program.Blocks[b].Phis.Add(phi);
            }
        }

        var children = new List<int>[n];
        for (int b = 0; b < n; b++) children[b] = new List<int>();
        for (int b = 1; b < n; b++)
        {
            if (idom[b] >= 0) children[idom[b]].Add(b);
        }

        Rename(cfg, program, 0, stacks, counters, children);
        for (int b = 0; b < n; b++)
        {
            foreach (SsaIns phi in program.Blocks[b].Phis) program.DefBlock.Add(b);
            foreach (SsaIns instruction in program.Blocks[b].Instructions) program.DefBlock.Add(b);
        }
        return program;
    }

    private static void Rename(Cfg cfg, SsaProgram program, int block, Dictionary<string, List<string>> stacks,
                               Dictionary<string, int> counters, List<int>[] children)
    {
        var pushed = new List<string>();
        foreach (SsaIns phi in program.Blocks[block].Phis)
        {
            string version = phi.Var + "." + counters[phi.Var];
            counters[phi.Var]++;
            stacks[phi.Var].Add(version);
            pushed.Add(phi.Var);
            phi.Var = version;
        }

        foreach (Ins instruction in cfg.Blocks[block].Instructions)
        {
            var renamed = new SsaIns { SideEffect = instruction.SideEffect, Constant = instruction.Constant };
            foreach (string use in instruction.Uses) renamed.Uses.Add(Top(stacks, use));
            if (instruction.Var.Length > 0)
            {
                string source = instruction.Var;
                string version = source + "." + counters[source];
                counters[source]++;
                stacks[source].Add(version);
                pushed.Add(source);
                renamed.Var = version;
            }
            program.Blocks[block].Instructions.Add(renamed);
        }

        foreach (int successor in cfg.Blocks[block].Successors)
        {
            for (int i = 0; i < program.Blocks[successor].Predecessors.Count; i++)
            {
                if (program.Blocks[successor].Predecessors[i] != block) continue;
                foreach (SsaIns phi in program.Blocks[successor].Phis)
                {
                    phi.PhiArguments[i] = Top(stacks, phi.Original);
                }
            }
        }

        // The terminator reads its variable after the block's instructions, so it is renamed last.
        if (cfg.Blocks[block].BranchVar.Length > 0)
        {
            program.Blocks[block].BranchVar = Top(stacks, cfg.Blocks[block].BranchVar);
        }

        foreach (int child in children[block]) Rename(cfg, program, child, stacks, counters, children);

        for (int i = pushed.Count - 1; i >= 0; i--)
        {
            List<string> stack = stacks[pushed[i]];
            stack.RemoveAt(stack.Count - 1);
        }
    }

    private static string Top(Dictionary<string, List<string>> stacks, string variable)
    {
        if (!stacks.TryGetValue(variable, out List<string>? stack) || stack.Count == 0)
        {
            return variable + ".in";       // a read before any definition: the input version of the name
        }
        return stack[stack.Count - 1];
    }

    /// Constant propagation, trivial-phi elimination and dead-code elimination, iterated to a fixpoint.
    internal static void Optimise(SsaProgram program)
    {
        bool changed = true;
        while (changed)
        {
            changed = false;
            var constants = new Dictionary<string, int>();
            bool propagating = true;
            while (propagating)
            {
                propagating = false;
                foreach (SsaBlock block in program.Blocks)
                {
                    foreach (SsaIns phi in block.Phis)
                    {
                        int value;
                        if (constants.ContainsKey(phi.Var)) continue;
                        if (TryPhiConstant(phi, constants, out value))
                        {
                            constants[phi.Var] = value;
                            propagating = true;
                        }
                    }
                    foreach (SsaIns instruction in block.Instructions)
                    {
                        if (instruction.SideEffect || constants.ContainsKey(instruction.Var)) continue;
                        if (instruction.Constant != 0 && instruction.Uses.Count == 0)
                        {
                            constants[instruction.Var] = instruction.Constant;
                            propagating = true;
                            continue;
                        }
                        int value = instruction.Constant;
                        bool all = true;
                        foreach (string use in instruction.Uses)
                        {
                            int one;
                            if (!constants.TryGetValue(use, out one)) { all = false; break; }
                            value += one;
                        }
                        if (all && instruction.Uses.Count > 0)
                        {
                            constants[instruction.Var] = value;
                            propagating = true;
                        }
                    }
                }
            }

            foreach (SsaBlock block in program.Blocks)
            {
                var kept = new List<SsaIns>();
                foreach (SsaIns instruction in block.Instructions)
                {
                    int value;
                    if (constants.TryGetValue(instruction.Var, out value)
                        && (instruction.Uses.Count > 0 || instruction.Constant != value))
                    {
                        instruction.Uses.Clear();
                        instruction.Constant = value;
                        program.Folds++;
                        changed = true;
                    }
                    kept.Add(instruction);
                }
                block.Instructions = kept;

                var keptPhis = new List<SsaIns>();
                foreach (SsaIns phi in block.Phis)
                {
                    int value;
                    if (constants.ContainsKey(phi.Var) && TryPhiConstant(phi, constants, out value))
                    {
                        var replacement = new SsaIns { Var = phi.Var, Constant = value };
                        block.Instructions.Insert(0, replacement);
                        program.Folds++;
                        changed = true;
                        continue;
                    }
                    string single = null!;
                    bool same = true;
                    foreach (string argument in phi.PhiArguments)
                    {
                        if (single == null) single = argument;
                        else if (single != argument) same = false;
                    }
                    if (same && single != null)
                    {
                        string target = single;
                        foreach (SsaBlock other in program.Blocks)
                        {
                            foreach (SsaIns instruction in other.Instructions)
                            {
                                for (int i = 0; i < instruction.Uses.Count; i++)
                                {
                                    if (instruction.Uses[i] == phi.Var) instruction.Uses[i] = target;
                                }
                            }
                            foreach (SsaIns otherPhi in other.Phis)
                            {
                                for (int i = 0; i < otherPhi.PhiArguments.Count; i++)
                                {
                                    if (otherPhi.PhiArguments[i] == phi.Var) otherPhi.PhiArguments[i] = target;
                                }
                            }
                        }
                        program.DroppedPhis++;
                        changed = true;
                        continue;
                    }
                    keptPhis.Add(phi);
                }
                block.Phis = keptPhis;
            }

            // Dead-code elimination: a definition whose name is never read again is removed.
            var used = new HashSet<string>();
            foreach (SsaBlock block in program.Blocks)
            {
                if (block.BranchVar.Length > 0) used.Add(block.BranchVar);
                foreach (SsaIns phi in block.Phis) foreach (string argument in phi.PhiArguments) used.Add(argument);
                foreach (SsaIns instruction in block.Instructions)
                {
                    foreach (string use in instruction.Uses) used.Add(use);
                }
            }
            foreach (SsaBlock block in program.Blocks)
            {
                var kept = new List<SsaIns>();
                foreach (SsaIns instruction in block.Instructions)
                {
                    if (!instruction.SideEffect && instruction.Var.Length > 0 && !used.Contains(instruction.Var))
                    {
                        program.EliminatedDefs++;
                        changed = true;
                        continue;
                    }
                    kept.Add(instruction);
                }
                block.Instructions = kept;
                var keptPhis = new List<SsaIns>();
                foreach (SsaIns phi in block.Phis)
                {
                    if (!used.Contains(phi.Var)) continue;
                    keptPhis.Add(phi);
                }
                block.Phis = keptPhis;
            }
        }
    }

    private static bool TryPhiConstant(SsaIns phi, Dictionary<string, int> constants, out int value)
    {
        value = 0;
        if (phi.PhiArguments.Count == 0) return false;
        bool first = true;
        foreach (string argument in phi.PhiArguments)
        {
            int one;
            if (!constants.TryGetValue(argument, out one)) return false;
            if (first) { value = one; first = false; }
            else if (one != value) return false;
        }
        return true;
    }
}

internal static class SsaInterp
{
    internal static Behaviour Run(SsaProgram program, Dictionary<string, int> inputs, int stepLimit)
    {
        var behaviour = new Behaviour();
        var environment = new Dictionary<string, int>();
        foreach (KeyValuePair<string, int> pair in inputs) environment[pair.Key] = pair.Value;
        int block = 0;
        int arrivedFrom = -1;
        for (int step = 0; step < stepLimit; step++)
        {
            behaviour.Steps++;
            behaviour.Path.Add(block);
            SsaBlock current = program.Blocks[block];
            foreach (SsaIns phi in current.Phis)
            {
                int index = 0;
                for (int i = 0; i < current.Predecessors.Count; i++)
                {
                    if (current.Predecessors[i] == arrivedFrom) index = i;
                }
                environment[phi.Var] = Lookup(environment, phi.PhiArguments[index]);
            }
            foreach (SsaIns instruction in current.Instructions)
            {
                if (instruction.SideEffect)
                {
                    var text = new List<string>();
                    foreach (string use in instruction.Uses) text.Add(use + "=" + Lookup(environment, use));
                    behaviour.Trace.Add("print " + string.Join(",", text));
                    continue;
                }
                int value = instruction.Constant;
                foreach (string use in instruction.Uses) value += Lookup(environment, use);
                environment[instruction.Var] = value;
            }
            if (current.Successors.Count == 0) return behaviour;
            if (current.Successors.Count == 1)
            {
                arrivedFrom = block;
                block = current.Successors[0];
                continue;
            }
            int condition = Lookup(environment, current.BranchVar);
            arrivedFrom = block;
            block = condition >= 0 ? current.Successors[0] : current.Successors[1];
        }
        return behaviour;
    }

    private static int Lookup(Dictionary<string, int> environment, string variable)
    {
        int value;
        if (environment.TryGetValue(variable, out value)) return value;
        // Input versions arrive as "<name>.in"; the harness supplies the unversioned input.
        int dot = variable.LastIndexOf('.');
        if (dot > 0)
        {
            string bare = variable.Substring(0, dot);
            if (environment.TryGetValue(bare, out value)) return value;
            string suffix = variable.Substring(dot + 1);
            if (suffix != "in" && environment.TryGetValue(bare + "." + suffix, out value)) return value;
        }
        return 0;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// Diamond: x is assigned on both arms, so the join needs a phi.
    private static Cfg Diamond()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("c", 1));
        Blk head = cfg.Add("H");
        head.BranchVar = "c";
        Blk left = cfg.Add("L");
        left.Instructions.Add(Ins.Defines("x", 3));
        Blk right = cfg.Add("R");
        right.Instructions.Add(Ins.Defines("x", 5));
        Blk join = cfg.Add("J");
        join.Instructions.Add(Ins.Effect("x"));
        cfg.Edge(0, 1);
        cfg.Edge(1, 2);
        cfg.Edge(1, 3);
        cfg.Edge(2, 4);
        cfg.Edge(3, 4);
        cfg.DiscoverInputs();
        return cfg;
    }

    /// Loop whose body assigns the header variable: the header needs a phi for it.
    private static Cfg LoopWithPhi()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("i", 2));
        Blk head = cfg.Add("H");
        head.BranchVar = "i";
        Blk body = cfg.Add("B");
        body.Instructions.Add(Ins.Defines("i", -1, "i"));
        Blk exit = cfg.Add("X");
        exit.Instructions.Add(Ins.Effect("i"));
        cfg.Edge(0, 1);
        cfg.Edge(1, 2);
        cfg.Edge(1, 3);
        cfg.Edge(2, 1);
        cfg.DiscoverInputs();
        return cfg;
    }

    /// Straight line with folding opportunities and one dead definition.
    private static Cfg Foldable()
    {
        var cfg = new Cfg();
        Blk entry = cfg.Add("E");
        entry.Instructions.Add(Ins.Defines("a", 3));
        entry.Instructions.Add(Ins.Defines("b", 4, "a"));
        entry.Instructions.Add(Ins.Defines("dead", 9, "b"));
        entry.Instructions.Add(Ins.Effect("b"));
        cfg.DiscoverInputs();
        return cfg;
    }

    private static Cfg RandomCfg(ref ulong state)
    {
        var cfg = new Cfg();
        string[] names = { "a", "b", "c" };
        int blocks = 3 + Rng.NextBelow(ref state, 3);
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
                if (Rng.NextBelow(ref state, 2) == 0) uses.Add(names[Rng.NextBelow(ref state, names.Length)]);
                if (Rng.NextBelow(ref state, 2) == 0) uses.Add(names[Rng.NextBelow(ref state, names.Length)]);
                block.Instructions.Add(Ins.Defines(names[Rng.NextBelow(ref state, names.Length)],
                                                  Rng.NextSigned(ref state, 4), uses.ToArray()));
            }
            if (Rng.NextBelow(ref state, 3) == 0) block.BranchVar = names[Rng.NextBelow(ref state, names.Length)];
        }
        for (int b = 1; b < blocks; b++) cfg.Edge(b - 1, b);
        for (int b = 0; b < blocks; b++)
        {
            int edges = Rng.NextBelow(ref state, 3);
            for (int e = 0; e < edges; e++)
            {
                int target = Rng.NextBelow(ref state, blocks);
                // No edge back into the entry: minimal SSA assumes a unique entry with no predecessors
                // (there would otherwise be no phi argument for the outside world). No duplicate edges
                // either: they would put the same predecessor in the list twice for no benefit.
                if (target != b && target != 0 && !cfg.Blocks[b].Successors.Contains(target)) cfg.Edge(b, target);
            }
        }
        cfg.DiscoverInputs();
        return cfg;
    }

    private static Dictionary<string, int> InputsFor(Cfg cfg, ref ulong state)
    {
        var inputs = new Dictionary<string, int>();
        foreach (string name in cfg.Inputs) inputs[name] = Rng.NextSigned(ref state, 3);
        // Every variable gets a value, including the versions the SSA program reads as "<name>.in".
        foreach (string name in new[] { "a", "b", "c" })
        {
            if (!inputs.ContainsKey(name)) inputs[name] = Rng.NextSigned(ref state, 3);
        }
        return inputs;
    }

    private static bool SameBehaviour(Behaviour a, Behaviour b)
    {
        if (a.Trace.Count != b.Trace.Count || a.Path.Count != b.Path.Count) return false;
        for (int i = 0; i < a.Trace.Count; i++) if (Normalise(a.Trace[i]) != Normalise(b.Trace[i])) return false;
        for (int i = 0; i < a.Path.Count; i++) if (a.Path[i] != b.Path[i]) return false;
        return true;
    }

    /// Renaming gives a variable a new versioned name, so "print b=2" in the original and
    /// "print b.in=2" in the SSA form are the same observation; comparison strips the version.
    private static string Normalise(string entry)
    {
        if (!entry.StartsWith("print ")) return entry;
        string[] parts = entry.Substring(6).Split(',');
        var rendered = new List<string>();
        foreach (string part in parts)
        {
            int equals = part.IndexOf('=');
            string name = part.Substring(0, equals);
            int dot = name.LastIndexOf('.');
            if (dot > 0) name = name.Substring(0, dot);
            rendered.Add(name + "=" + part.Substring(equals + 1));
        }
        return "print " + string.Join(",", rendered);
    }

    /// Structural SSA property: every name is defined exactly once, and its definition dominates every
    /// use. Checked with dominators computed here, not with anything the renamer produced.
    private static (int Definitions, int Uses, int Violations) AuditSsa(Cfg cfg, SsaProgram program)
    {
        HashSet<int>[] dom = Ssa.Dominators(cfg);
        var defSite = new Dictionary<string, int>();
        int definitions = 0;
        int violations = 0;
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            foreach (SsaIns phi in program.Blocks[b].Phis)
            {
                if (defSite.ContainsKey(phi.Var)) violations++;
                defSite[phi.Var] = b;
                definitions++;
                if (phi.PhiArguments.Count != cfg.Blocks[b].Predecessors.Count) violations++;
            }
            foreach (SsaIns instruction in program.Blocks[b].Instructions)
            {
                if (instruction.Var.Length == 0) continue;
                if (defSite.ContainsKey(instruction.Var)) violations++;
                defSite[instruction.Var] = b;
                definitions++;
            }
        }
        int uses = 0;
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            var checker = new List<string>();
            foreach (SsaIns instruction in program.Blocks[b].Instructions) checker.AddRange(instruction.Uses);
            if (program.Blocks[b].BranchVar.Length > 0) checker.Add(program.Blocks[b].BranchVar);
            foreach (string use in checker)
            {
                uses++;
                int site;
                if (!defSite.TryGetValue(use, out site)) continue;      // an input version: no definition to dominate
                if (!dom[b].Contains(site)) violations++;
            }

            // A phi argument is read at the end of the matching predecessor, so its definition has to
            // dominate that predecessor rather than the phi's own block.
            foreach (SsaIns phi in program.Blocks[b].Phis)
            {
                for (int i = 0; i < phi.PhiArguments.Count; i++)
                {
                    uses++;
                    int site;
                    if (!defSite.TryGetValue(phi.PhiArguments[i], out site)) continue;
                    int pred = i < cfg.Blocks[b].Predecessors.Count ? cfg.Blocks[b].Predecessors[i] : b;
                    if (!dom[pred].Contains(site)) violations++;
                }
            }
        }
        return (definitions, uses, violations);
    }

    /// Brute-force justification of the phis: explore the (block, last-definition-per-variable) state space
    /// of the *original* program and collect, for every block, which definitions of each variable can
    /// arrive. A phi is justified when at least two different definitions (counting "still the input" as
    /// one of them) can reach its block for its variable.
    private static void AuditPhis(Cfg cfg, SsaProgram program, out int justified, out int unjustified, out string detail)
    {
        var variables = new List<string>();
        foreach (Blk block in cfg.Blocks)
        {
            foreach (Ins instruction in block.Instructions)
            {
                if (instruction.Var.Length > 0 && !variables.Contains(instruction.Var)) variables.Add(instruction.Var);
            }
        }

        var reaching = new Dictionary<string, HashSet<int>>();
        var visited = new HashSet<string>();
        var queue = new Queue<KeyValuePair<int, int[]>>();
        var start = new int[variables.Count];
        for (int v = 0; v < start.Length; v++) start[v] = -1;
        visited.Add(StateKey(0, start));
        queue.Enqueue(new KeyValuePair<int, int[]>(0, start));
        while (queue.Count > 0)
        {
            KeyValuePair<int, int[]> current = queue.Dequeue();
            int block = current.Key;
            int[] state = current.Value;
            for (int v = 0; v < variables.Count; v++)
            {
                string key = block + "|" + variables[v];
                HashSet<int> set;
                if (!reaching.TryGetValue(key, out set))
                {
                    set = new HashSet<int>();
                    reaching[key] = set;
                }
                set.Add(state[v]);
            }
            var outgoing = (int[])state.Clone();
            for (int i = 0; i < cfg.Blocks[block].Instructions.Count; i++)
            {
                Ins instruction = cfg.Blocks[block].Instructions[i];
                if (instruction.Var.Length == 0) continue;
                outgoing[variables.IndexOf(instruction.Var)] = block * 1000 + i + 1;
            }
            foreach (int successor in cfg.Blocks[block].Successors)
            {
                string key = StateKey(successor, outgoing);
                if (visited.Add(key)) queue.Enqueue(new KeyValuePair<int, int[]>(successor, (int[])outgoing.Clone()));
            }
        }

        justified = 0;
        unjustified = 0;
        detail = "";
        for (int b = 0; b < program.Blocks.Count; b++)
        {
            foreach (SsaIns phi in program.Blocks[b].Phis)
            {
                HashSet<int> set;
                int count = reaching.TryGetValue(b + "|" + phi.Original, out set) ? set.Count : 0;
                if (count >= 2)
                {
                    justified++;
                }
                else
                {
                    unjustified++;
                    if (detail.Length == 0) detail = "block " + b + " variable " + phi.Original
                                                     + " has only " + count + " arriving definition(s)";
                }
            }
        }
    }

    private static string StateKey(int block, int[] state) => block + ":" + string.Join(",", state);

    public static void Main()
    {
        // ---- dominators and frontiers ---------------------------------------------------------
        Cfg diamond = Diamond();
        HashSet<int>[] dom = Ssa.Dominators(diamond);
        Check(dom[0].Count == 1 && dom[0].Contains(0), "the entry dominates only itself");
        Check(dom[4].SetEquals(new HashSet<int> { 0, 1, 4 }), "the join is dominated by the entry, the head and itself");
        Check(!dom[2].Contains(3) && !dom[3].Contains(2), "neither arm dominates the other");
        int[] idom = Ssa.ImmediateDominators(diamond, dom);
        Check(idom[4] == 1 && idom[2] == 1 && idom[3] == 1, "the head is the immediate dominator of both arms and the join");
        HashSet<int>[] frontiers = Ssa.Frontiers(diamond, dom, idom);
        Check(frontiers[2].SetEquals(new HashSet<int> { 4 }) && frontiers[3].SetEquals(new HashSet<int> { 4 }),
              "the join is exactly the dominance frontier of both arms");
        Check(frontiers[0].Count == 0 && frontiers[1].Count == 0, "no other block has a frontier in this shape");

        // ---- phi insertion and renaming --------------------------------------------------------
        SsaProgram ssa = Ssa.Build(diamond);
        Check(ssa.Blocks[4].Phis.Count == 1, "the join gets exactly one phi (got " + ssa.Blocks[4].Phis.Count + ")");
        Check(ssa.Blocks[4].Phis[0].PhiArguments.Count == 2, "with one argument per predecessor");
        Check(ssa.Blocks[4].Phis[0].PhiArguments[0] != ssa.Blocks[4].Phis[0].PhiArguments[1],
              "and the two arguments name the two different versions");
        Check(ssa.Blocks[2].Phis.Count == 0 && ssa.Blocks[3].Phis.Count == 0, "the arms need no phis");
        var audit = AuditSsa(diamond, ssa);
        Check(audit.Violations == 0, "structural SSA audit: one definition per name, definition dominates use ("
              + audit.Violations + " violations)");
        Check(audit.Definitions >= 4 && audit.Uses >= 3, "the audit inspected " + audit.Definitions
              + " definitions and " + audit.Uses + " uses");

        // ---- the loop: a phi at the header -----------------------------------------------------
        Cfg loop = LoopWithPhi();
        SsaProgram loopSsa = Ssa.Build(loop);
        Check(loopSsa.Blocks[1].Phis.Count == 1, "the loop header gets a phi (got " + loopSsa.Blocks[1].Phis.Count + ")");
        Check(loopSsa.Blocks[1].Phis[0].PhiArguments.Count == 2, "with one argument per predecessor, including the back edge");
        Check(loopSsa.Blocks[1].Phis[0].PhiArguments[0] == "i.0" && loopSsa.Blocks[1].Phis[0].PhiArguments[1] == "i.2",
              "the phi takes the entry's version on the way in and the body's version on the back edge ("
              + loopSsa.Blocks[1].Phis[0].PhiArguments[0] + ", " + loopSsa.Blocks[1].Phis[0].PhiArguments[1] + ")");
        Check(loopSsa.Blocks[1].BranchVar == "i.1", "the loop header's branch reads the phi's own version");
        Check(loopSsa.Blocks[2].Instructions[0].Uses[0] == "i.1",
              "and the body reads whatever version arrived: " + loopSsa.Blocks[2].Instructions[0].Uses[0]);
        Check(loopSsa.Blocks[0].Phis.Count == 0, "the entry needs no phi");
        Check(AuditSsa(loop, loopSsa).Violations == 0, "the loop's SSA form passes the structural audit");

        // ---- semantic equivalence of SSA -------------------------------------------------------
        ulong state = 673001UL;
        int programs = 40;
        int totalPhis = 0;
        for (int p = 0; p < programs; p++)
        {
            Cfg cfg = RandomCfg(ref state);
            SsaProgram form = Ssa.Build(cfg);
            foreach (SsaBlock block in form.Blocks) totalPhis += block.Phis.Count;
            Check(AuditSsa(cfg, form).Violations == 0,
                  "program " + p + ": the SSA form passes the structural audit (definition dominates use)");
            for (int t = 0; t < 15; t++)
            {
                Dictionary<string, int> inputs = InputsFor(cfg, ref state);
                Behaviour expected = Interp.Run(cfg, inputs, 150);
                Behaviour actualRun = SsaInterp.Run(form, inputs, 150);
                Check(SameBehaviour(expected, actualRun),
                      "program " + p + " input " + t + ": the SSA form computes the same behaviour");
            }
        }
        Check(totalPhis >= 20, "the random sweep inserted " + totalPhis + " phi functions in total");

        // ---- optimisations preserve semantics ---------------------------------------------------
        Cfg fold = Foldable();
        SsaProgram foldSsa = Ssa.Build(fold);
        Ssa.Optimise(foldSsa);
        Check(foldSsa.Folds >= 2, "constant propagation folded " + foldSsa.Folds + " definitions");
        Check(foldSsa.EliminatedDefs >= 1, "and dead-code elimination removed " + foldSsa.EliminatedDefs + " definitions");
        bool foldedSeven = false;
        foreach (SsaIns instruction in foldSsa.Blocks[0].Instructions)
        {
            if (instruction.Var == "b.0" && instruction.Constant == 7 && instruction.Uses.Count == 0) foldedSeven = true;
        }
        Check(foldedSeven, "b = 4 + a with a = 3 folds to the constant 7");

        int optimisedPrograms = 0;
        int justifiedTotal = 0;
        int unjustifiedTotal = 0;
        for (int p = 0; p < 30; p++)
        {
            Cfg cfg = RandomCfg(ref state);
            SsaProgram form = Ssa.Build(cfg);
            SsaProgram before = FormCopy(form);
            Ssa.Optimise(form);
            if (form.Folds > 0 || form.DroppedPhis > 0 || form.EliminatedDefs > 0) optimisedPrograms++;
            int justified;
            int unjustified;
            string detail;
            AuditPhis(cfg, form, out justified, out unjustified, out detail);
            justifiedTotal += justified;
            unjustifiedTotal += unjustified;
            Check(unjustified == 0, "program " + p + ": every surviving phi has two or more arriving definitions ("
                  + detail + ")");
            for (int t = 0; t < 15; t++)
            {
                Dictionary<string, int> inputs = InputsFor(cfg, ref state);
                Check(SameBehaviour(SsaInterp.Run(before, inputs, 150), SsaInterp.Run(form, inputs, 150)),
                      "program " + p + " input " + t + ": the optimised SSA form behaves identically");
                Check(SameBehaviour(Interp.Run(cfg, inputs, 150), SsaInterp.Run(form, inputs, 150)),
                      "program " + p + " input " + t + ": and identical to the original program");
            }
        }
        Check(optimisedPrograms >= 10, "the optimiser changed " + optimisedPrograms + " of 30 random programs");
        Check(unjustifiedTotal == 0, "all " + justifiedTotal + " surviving phis are justified by reaching definitions");

        // ---- degenerate input ------------------------------------------------------------------
        var single = new Cfg();
        single.Add("E").Instructions.Add(Ins.Effect("q"));
        single.DiscoverInputs();
        SsaProgram singleSsa = Ssa.Build(single);
        Check(singleSsa.Blocks[0].Phis.Count == 0, "a single block program needs no phis");
        Check(AuditSsa(single, singleSsa).Violations == 0, "and still passes the audit");

        Console.WriteLine("diamond: " + ssa.Blocks[4].Phis.Count + " phi at the join, loop: "
              + loopSsa.Blocks[1].Phis.Count + " at the header");
        Console.WriteLine("random programs: " + programs + " built, " + totalPhis + " phis inserted");
        Console.WriteLine("optimiser active in " + optimisedPrograms + " of 30 programs, phis justified: " + justifiedTotal);
        Console.WriteLine("PASS");
    }

    private static SsaProgram FormCopy(SsaProgram form)
    {
        var copy = new SsaProgram();
        foreach (SsaBlock block in form.Blocks)
        {
            var clone = new SsaBlock
            {
                BranchVar = block.BranchVar,
                Successors = new List<int>(block.Successors),
                Predecessors = new List<int>(block.Predecessors)
            };
            foreach (SsaIns phi in block.Phis) clone.Phis.Add(phi.Copy());
            foreach (SsaIns instruction in block.Instructions) clone.Instructions.Add(instruction.Copy());
            copy.Blocks.Add(clone);
        }
        return copy;
    }
}
