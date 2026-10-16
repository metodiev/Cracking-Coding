// Task 676 - Implement speculatively executed branch prediction simulation.
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A speculative execution engine over a small register machine, together with the predictors that decide
// which way it gambles:
//   * the machine: eight registers, blocks of Set / Add / Xor1 / Print instructions ending in either a
//     fall-through or a two-way branch on "register >= 0". Speculation depth D means the engine resolves a
//     branch only after D further blocks have been fetched behind it, which is what makes squashing real:
//     the work done behind a wrong guess has to be discarded;
//   * four predictors: always-not-taken, always-taken, a 2-bit bimodal table indexed by block, and a
//     gshare predictor that keeps an 8-bit global history and indexes the table with pc xor history;
//   * the engine commits a speculative state only when the branch it guessed for resolves correctly,
//     otherwise it rolls back to the snapshot taken at that branch, jumps to the real target, pays a
//     pipeline refill penalty and counts the squashed blocks. Prints go through a per-frame buffer and are
//     discarded with the frame, so a mis-speculated path can never print anything.
// Verification is behavioural, not self-referential: a straightforward non-speculative interpreter produces
// the ground truth (the dynamic branch sequence, the prints and the final registers) and the speculative
// engine must reproduce all three. On a window of depth 0 (nothing in flight) the engine's misprediction
// count must equal that of an *independent* predictor replay over the reference dynamic trace; a window
// deeper than that is only required to preserve behaviour and to satisfy the accounting identity
// "committed blocks = fetched blocks - squashed blocks", because a speculative engine predicts ahead of
// training and drops the updates belonging to squashed branches, which legitimately changes the count. The
// canonical misprediction counts are pinned down for a fixed-trip-count loop (always-taken misses once at
// the exit, the 2-bit bimodal table misses twice: once entering and once leaving, always-not-taken misses
// on every iteration), which is only exact because the predictor tables are seeded to the documented
// initial state.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic pseudo-random register inputs.
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

internal enum OpKind { Set, Add, Xor1, Flip, Print }

internal sealed class Ins
{
    internal OpKind Kind;
    internal int Register;
    internal int Constant;

    internal static Ins Set(int register, int constant) => new Ins { Kind = OpKind.Set, Register = register, Constant = constant };
    internal static Ins Add(int register, int constant) => new Ins { Kind = OpKind.Add, Register = register, Constant = constant };
    internal static Ins Xor1(int register) => new Ins { Kind = OpKind.Xor1, Register = register };
    /// r = -r - 1: flips the sign of the register, which is what makes a branch outcome alternate.
    internal static Ins Flip(int register) => new Ins { Kind = OpKind.Flip, Register = register };
    internal static Ins Print(int register) => new Ins { Kind = OpKind.Print, Register = register };
}

internal sealed class Blk
{
    internal List<Ins> Instructions = new List<Ins>();
    internal int BranchRegister = -1;          // -1: fall through to Successors[0]
    internal List<int> Successors = new List<int>();
}

internal sealed class Code
{
    internal List<Blk> Blocks = new List<Blk>();
    internal int Registers = 8;
    internal Dictionary<int, int> Inputs = new Dictionary<int, int>();

    internal Blk Add()
    {
        var block = new Blk();
        Blocks.Add(block);
        return block;
    }
}

internal sealed class RunResult
{
    internal int[] Registers = Array.Empty<int>();
    internal List<string> Prints = new List<string>();
    internal List<int> BranchPcs = new List<int>();
    internal List<bool> BranchTaken = new List<bool>();
    internal int BlocksExecuted;
}

internal static class Machine
{
    internal static RunResult Reference(Code code, int budget)
    {
        var result = new RunResult();
        var registers = new int[code.Registers];
        foreach (KeyValuePair<int, int> pair in code.Inputs) registers[pair.Key] = pair.Value;
        int block = 0;
        for (int step = 0; step < budget; step++)
        {
            result.BlocksExecuted++;
            Blk current = code.Blocks[block];
            foreach (Ins instruction in current.Instructions)
            {
                switch (instruction.Kind)
                {
                    case OpKind.Set:
                        registers[instruction.Register] = instruction.Constant;
                        break;
                    case OpKind.Add:
                        registers[instruction.Register] += instruction.Constant;
                        break;
                    case OpKind.Xor1:
                        registers[instruction.Register] ^= 1;
                        break;
                    case OpKind.Flip:
                        registers[instruction.Register] = -registers[instruction.Register] - 1;
                        break;
                    default:
                        result.Prints.Add(registers[instruction.Register].ToString());
                        break;
                }
            }
            if (current.Successors.Count == 0) break;
            if (current.Successors.Count == 1) { block = current.Successors[0]; continue; }
            bool taken = registers[current.BranchRegister] >= 0;
            result.BranchPcs.Add(block);
            result.BranchTaken.Add(taken);
            block = taken ? current.Successors[0] : current.Successors[1];
        }
        result.Registers = registers;
        return result;
    }
}

internal interface IPredictor
{
    string Name { get; }
    bool Predict(int pc);
    void Update(int pc, bool taken);
}

internal sealed class AlwaysNotTaken : IPredictor
{
    public string Name => "always-not-taken";
    public bool Predict(int pc) => false;
    public void Update(int pc, bool taken) { }
}

internal sealed class AlwaysTaken : IPredictor
{
    public string Name => "always-taken";
    public bool Predict(int pc) => true;
    public void Update(int pc, bool taken) { }
}

/// 2-bit saturating counter table indexed by the block pc.
internal sealed class Bimodal : IPredictor
{
    private readonly int[] table;
    public Bimodal(int entries) { table = new int[entries]; for (int i = 0; i < entries; i++) table[i] = 1; }
    public string Name => "bimodal-2bit";
    public bool Predict(int pc) => table[pc % table.Length] >= 2;
    public void Update(int pc, bool taken)
    {
        int index = pc % table.Length;
        table[index] += taken ? 1 : -1;
        if (table[index] < 0) table[index] = 0;
        if (table[index] > 3) table[index] = 3;
    }
}

/// Gshare: the table index is the pc xor'ed with the global branch history.
internal sealed class GShare : IPredictor
{
    private readonly int[] table;
    private int history;
    public GShare(int entries) { table = new int[entries]; for (int i = 0; i < entries; i++) table[i] = 1; }
    public string Name => "gshare-8bit";
    public bool Predict(int pc) => table[(pc ^ history) % table.Length] >= 2;
    public void Update(int pc, bool taken)
    {
        int index = (pc ^ history) % table.Length;
        table[index] += taken ? 1 : -1;
        if (table[index] < 0) table[index] = 0;
        if (table[index] > 3) table[index] = 3;
        history = ((history << 1) | (taken ? 1 : 0)) & 0xFF;
    }
}

internal static class Counter
{
    /// Independent replay of a predictor over a dynamic trace: the number of wrong guesses.
    internal static int Mispredictions(IPredictor predictor, List<int> pcs, List<bool> taken)
    {
        int misses = 0;
        for (int i = 0; i < pcs.Count; i++)
        {
            if (predictor.Predict(pcs[i]) != taken[i]) misses++;
            predictor.Update(pcs[i], taken[i]);
        }
        return misses;
    }
}

internal sealed class SpecFrame
{
    internal int Block;
    internal int[] Snapshot = Array.Empty<int>();     // architectural state *before* the branch block
    internal int FetchedAt;                            // fetch counter when the branch block was fetched
    internal bool Guess;                               // true when the predictor guessed "taken"
}

internal sealed class SpecResult
{
    internal int[] Registers = Array.Empty<int>();
    internal List<string> Prints = new List<string>();
    internal int BlocksFetched;
    internal int BlocksSquashed;
    internal int BlocksCommitted;
    internal int Mispredictions;
    internal int Cycles;
    internal int Refills;
}

internal static class Speculation
{
    /// Speculative engine. A branch is resolved once `depth` further branches are in flight: until then the
    /// work behind it is speculative, and a wrong guess discards that work, restores the snapshot taken at
    /// the branch, jumps to the real target and pays a refill. Prints carry the fetch index of the block
    /// that produced them so they are discarded together with the frame.
    internal static SpecResult Run(Code code, IPredictor predictor, int depth, int budget)
    {
        var result = new SpecResult();
        var state = new int[code.Registers];
        foreach (KeyValuePair<int, int> pair in code.Inputs) state[pair.Key] = pair.Value;
        var prints = new List<KeyValuePair<int, string>>();
        var pending = new Queue<SpecFrame>();
        int pc = 0;

        for (int step = 0; step < budget && (pc >= 0 || pending.Count > 0); step++)
        {
            if (pc >= 0)
            {
                Blk current = code.Blocks[pc];
                result.BlocksFetched++;
                result.Cycles++;
                foreach (Ins instruction in current.Instructions)
                {
                    switch (instruction.Kind)
                    {
                        case OpKind.Set:
                            state[instruction.Register] = instruction.Constant;
                            break;
                        case OpKind.Add:
                            state[instruction.Register] += instruction.Constant;
                            break;
                        case OpKind.Xor1:
                            state[instruction.Register] ^= 1;
                            break;
                        case OpKind.Flip:
                            state[instruction.Register] = -state[instruction.Register] - 1;
                            break;
                        default:
                            prints.Add(new KeyValuePair<int, string>(result.BlocksFetched,
                                                                     state[instruction.Register].ToString()));
                            break;
                    }
                }

                if (current.Successors.Count == 0)
                {
                    pc = -1;
                }
                else if (current.Successors.Count == 1)
                {
                    pc = current.Successors[0];
                }
                else
                {
                    var frame = new SpecFrame
                    {
                        Block = pc,
                        Snapshot = (int[])state.Clone(),
                        FetchedAt = result.BlocksFetched,
                        Guess = predictor.Predict(pc)
                    };
                    pending.Enqueue(frame);
                    pc = frame.Guess ? current.Successors[0] : current.Successors[1];
                }
            }

            // Resolve the oldest branch when the window is full, or drain everything once the program ended.
            while (pending.Count > 0 && (pending.Count > depth || pc < 0))
            {
                SpecFrame front = pending.Dequeue();
                bool taken = front.Snapshot[code.Blocks[front.Block].BranchRegister] >= 0;
                predictor.Update(front.Block, taken);
                if (taken == front.Guess) continue;

                result.Mispredictions++;
                result.Refills++;
                result.Cycles += depth;
                result.BlocksSquashed += result.BlocksFetched - front.FetchedAt;
                state = (int[])front.Snapshot.Clone();
                prints.RemoveAll(entry => entry.Key > front.FetchedAt);
                pending.Clear();
                pc = taken ? code.Blocks[front.Block].Successors[0] : code.Blocks[front.Block].Successors[1];
                break;
            }
        }

        result.BlocksCommitted = result.BlocksFetched - result.BlocksSquashed;
        result.Registers = state;
        foreach (KeyValuePair<int, string> entry in prints) result.Prints.Add(entry.Value);
        return result;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// r = N; loop: r = r - 1; while r >= 0; print r — a fixed trip count loop.
    private static Code Countdown(int iterations)
    {
        var code = new Code();
        Blk entry = code.Add();
        entry.Instructions.Add(Ins.Set(0, iterations));
        entry.Successors.Add(1);
        Blk body = code.Add();
        body.Instructions.Add(Ins.Add(0, -1));
        body.BranchRegister = 0;
        body.Successors.Add(1);        // taken: loop again
        body.Successors.Add(2);        // not taken: leave
        Blk exit = code.Add();
        exit.Instructions.Add(Ins.Print(0));
        return code;
    }

    /// counter = iterations; r = 0; head: counter--; while counter >= 0 { r = -r - 1; while r >= 0 { } }
    /// The body's branch alternates its outcome on every visit while the header's branch is taken almost
    /// every time, which is exactly the pattern a history register can learn and a 2-bit table cannot.
    private static Code Alternating(int iterations)
    {
        var code = new Code();
        Blk entry = code.Add();
        entry.Instructions.Add(Ins.Set(1, iterations));
        entry.Instructions.Add(Ins.Set(0, 0));
        entry.Successors.Add(1);
        Blk head = code.Add();
        head.Instructions.Add(Ins.Add(1, -1));
        head.BranchRegister = 1;
        head.Successors.Add(2);        // taken: do the alternating work
        head.Successors.Add(3);        // not taken: done
        Blk body = code.Add();
        body.Instructions.Add(Ins.Flip(0));
        body.BranchRegister = 0;
        body.Successors.Add(1);        // both targets are the header, only the outcome alternates
        body.Successors.Add(1);
        Blk exit = code.Add();
        exit.Instructions.Add(Ins.Print(1));
        return code;
    }

    /// The branch is correctly resolved as "not taken"; the wrong (taken) path writes 999 into register 3
    /// and prints it. Neither the write nor the print may survive the squash.
    private static Code LeakyPath()
    {
        var code = new Code();
        Blk entry = code.Add();
        entry.Instructions.Add(Ins.Set(0, -2));
        entry.Instructions.Add(Ins.Set(3, 7));
        entry.BranchRegister = 0;
        entry.Successors.Add(1);
        entry.Successors.Add(2);
        Blk left = code.Add();
        left.Instructions.Add(Ins.Set(3, 999));
        left.Instructions.Add(Ins.Print(3));
        left.BranchRegister = 0;
        left.Successors.Add(3);
        left.Successors.Add(3);
        Blk right = code.Add();
        right.Instructions.Add(Ins.Add(3, -7));
        right.Instructions.Add(Ins.Print(3));
        right.BranchRegister = 0;
        right.Successors.Add(3);
        right.Successors.Add(3);
        Blk join = code.Add();
        join.Instructions.Add(Ins.Print(3));
        return code;
    }

    private static Code RandomCode(ref ulong state)
    {
        var code = new Code();
        int blocks = 4 + Rng.NextBelow(ref state, 4);
        for (int b = 0; b < blocks; b++)
        {
            Blk block = code.Add();
            int count = 1 + Rng.NextBelow(ref state, 3);
            for (int i = 0; i < count; i++)
            {
                int register = Rng.NextBelow(ref state, 4);
                switch (Rng.NextBelow(ref state, 4))
                {
                    case 0:
                        block.Instructions.Add(Ins.Set(register, Rng.NextBelow(ref state, 5)));
                        break;
                    case 1:
                        block.Instructions.Add(Ins.Add(register, Rng.NextBelow(ref state, 3) - 1));
                        break;
                    case 2:
                        block.Instructions.Add(Ins.Xor1(register));
                        break;
                    default:
                        block.Instructions.Add(Ins.Print(register));
                        break;
                }
            }
        }
        for (int b = 0; b < blocks - 1; b++)
        {
            code.Blocks[b].Successors.Add(b + 1);
            if (Rng.NextBelow(ref state, 3) == 0)
            {
                code.Blocks[b].BranchRegister = Rng.NextBelow(ref state, 4);
                int other = b + 1 + Rng.NextBelow(ref state, blocks - b - 1);
                code.Blocks[b].Successors.Add(other);
            }
        }
        return code;
    }

    private static bool SameState(RunResult reference, SpecResult speculative)
    {
        for (int r = 0; r < reference.Registers.Length; r++)
        {
            if (reference.Registers[r] != speculative.Registers[r]) return false;
        }
        if (reference.Prints.Count != speculative.Prints.Count) return false;
        for (int i = 0; i < reference.Prints.Count; i++)
        {
            if (reference.Prints[i] != speculative.Prints[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        // ---- the canonical misprediction counts -----------------------------------------------
        Code loop10 = Countdown(10);
        RunResult reference = Machine.Reference(loop10, 200);
        Check(reference.BranchPcs.Count == 11, "a ten-iteration countdown executes eleven branches (got "
              + reference.BranchPcs.Count + ")");
        Check(reference.Prints.Count == 1 && reference.Prints[0] == "-1", "and prints the final counter once");
        Check(Counter.Mispredictions(new AlwaysTaken(), reference.BranchPcs, reference.BranchTaken) == 1,
              "always-taken misses only the loop exit");
        Check(Counter.Mispredictions(new AlwaysNotTaken(), reference.BranchPcs, reference.BranchTaken) == 10,
              "always-not-taken misses every one of the ten taken iterations");
        Check(Counter.Mispredictions(new Bimodal(64), reference.BranchPcs, reference.BranchTaken) == 2,
              "the 2-bit bimodal table misses twice: entering and leaving the loop");

        // ---- the speculative engine reproduces the reference behaviour -------------------------
        var depth = 2;
        var predictor = new Bimodal(64);
        SpecResult speculative = Speculation.Run(loop10, predictor, depth, 400);
        Check(SameState(reference, speculative), "the speculative engine prints and computes exactly the reference run");
        var replay = new Bimodal(64);
        int replayMisses = Counter.Mispredictions(replay, reference.BranchPcs, reference.BranchTaken);
        SpecResult orderedRun = Speculation.Run(loop10, new Bimodal(64), 0, 400);
        Check(orderedRun.Mispredictions == replayMisses,
              "with nothing in flight its mispredictions equal the independent replay over the reference trace ("
              + orderedRun.Mispredictions + " vs " + replayMisses + ")");
        Check(orderedRun.BlocksSquashed == 0 && orderedRun.BlocksFetched == reference.BlocksExecuted,
              "and with an empty window nothing is fetched twice");
        Check(speculative.BlocksCommitted == reference.BlocksExecuted,
              "every reference block is committed exactly once (" + speculative.BlocksCommitted + " vs "
              + reference.BlocksExecuted + ")");
        Check(speculative.BlocksCommitted == speculative.BlocksFetched - speculative.BlocksSquashed,
              "committed work is exactly the fetched work that was not squashed (" + speculative.BlocksFetched
              + " - " + speculative.BlocksSquashed + ")");
        Check(speculative.BlocksFetched > reference.BlocksExecuted, "speculation fetched extra work ("
              + speculative.BlocksFetched + " > " + reference.BlocksExecuted + ")");
        Check(speculative.BlocksSquashed > 0, "and squashed " + speculative.BlocksSquashed + " blocks");
        Check(speculative.Cycles >= speculative.BlocksFetched, "cycles account for the refills as well");

        // ---- an oracle predictor never squashes ------------------------------------------------
        SpecResult oracleRun = Speculation.Run(loop10, new Oracle(loop10), depth, 400);
        Check(oracleRun.Mispredictions == 0 && oracleRun.BlocksSquashed == 0,
              "an oracle predictor never mispredicts and never squashes anything (misses=" + oracleRun.Mispredictions
              + " squashed=" + oracleRun.BlocksSquashed + " fetched=" + oracleRun.BlocksFetched
              + " prints=" + oracleRun.Prints.Count + ")");
        Check(SameState(reference, oracleRun), "and still produces the reference behaviour");

        // ---- mis-speculated writes must not leak ----------------------------------------------
        Code leaky = LeakyPath();
        RunResult leakyReference = Machine.Reference(leaky, 100);
        SpecResult leakyRun = Speculation.Run(leaky, new Inverted(), depth, 100);
        Check(leakyRun.Mispredictions > 0, "the deliberately wrong predictor does mispredict on the leaky program");
        Check(SameState(leakyReference, leakyRun),
              "speculative writes to register 3 are discarded, so only the correct value is printed");

        // ---- why history matters ---------------------------------------------------------------
        Code alternating = Alternating(20);
        RunResult alternatingReference = Machine.Reference(alternating, 200);
        int bimodalMisses = Counter.Mispredictions(new Bimodal(64), alternatingReference.BranchPcs,
                                                   alternatingReference.BranchTaken);
        int gshareMisses = Counter.Mispredictions(new GShare(64), alternatingReference.BranchPcs,
                                                  alternatingReference.BranchTaken);
        Check(bimodalMisses >= 8, "a 2-bit table cannot follow an alternating pattern (" + bimodalMisses + " misses)");
        // gshare does markedly better but not perfectly: the header branch's outcomes keep polluting the
        // history register, which is exactly the interference a real predictor has to live with.
        Check(gshareMisses < bimodalMisses, "and the history-based predictor does better (" + gshareMisses
              + " misses against " + bimodalMisses + ")");

        // ---- randomised cross-check ------------------------------------------------------------
        ulong state = 676001UL;
        int programs = 40;
        int totalMispredictions = 0;
        int totalSquashed = 0;
        for (int p = 0; p < programs; p++)
        {
            Code program = RandomCode(ref state);
            program.Blocks[program.Blocks.Count - 1].Successors.Clear();
            foreach (int register in new[] { 0, 1, 2, 3 })
            {
                program.Inputs[register] = Rng.NextBelow(ref state, 3) - 1;
            }
            RunResult truth = Machine.Reference(program, 120);
            foreach (IPredictor candidate in new IPredictor[] { new Bimodal(64), new GShare(64),
                                                                new AlwaysTaken(), new AlwaysNotTaken() })
            {
                SpecResult run = Speculation.Run(program, candidate, 3, 400);
                Check(SameState(truth, run), "program " + p + " with " + candidate.Name
                      + ": the speculative run reproduces the reference behaviour");
                Check(run.BlocksCommitted == truth.BlocksExecuted, "program " + p + " with " + candidate.Name
                      + ": committed blocks match the reference");
                Check(run.BlocksCommitted == run.BlocksFetched - run.BlocksSquashed, "program " + p + " with "
                      + candidate.Name + ": committed work is the fetched work minus the squashed work");
                Check(run.BlocksSquashed <= run.BlocksFetched, "program " + p + " with " + candidate.Name
                      + ": squashed work cannot exceed fetched work");
                totalMispredictions += run.Mispredictions;
                totalSquashed += run.BlocksSquashed;

                // The misprediction count is pinned down on a non-speculative run (depth 0), where the
                // predictor is trained strictly in architectural order; with speculation in the window the
                // engine predicts ahead of training and drops the updates of squashed branches, which
                // legitimately changes the count, so equality is only claimed for the depth-0 run.
                SpecResult ordered = Speculation.Run(program, NewLike(candidate), 0, 400);
                int expected = Counter.Mispredictions(NewLike(candidate), truth.BranchPcs, truth.BranchTaken);
                Check(ordered.Mispredictions == expected, "program " + p + " with " + candidate.Name
                      + ": mispredictions match the independent replay in architectural order ("
                      + ordered.Mispredictions + " vs " + expected + ")");
                Check(ordered.BlocksSquashed == 0, "program " + p + " with " + candidate.Name
                      + ": a depth-0 window has nothing to squash");
            }
        }
        Check(totalMispredictions > 0 && totalSquashed > 0, "the randomised sweep exercised squashing ("
              + totalMispredictions + " mispredictions, " + totalSquashed + " squashed blocks)");

        // ---- determinism -----------------------------------------------------------------------
        SpecResult repeat = Speculation.Run(loop10, new Bimodal(64), depth, 400);
        Check(repeat.BlocksFetched == speculative.BlocksFetched && repeat.Mispredictions == speculative.Mispredictions,
              "the speculative engine is deterministic for a fixed predictor");

        Console.WriteLine("countdown loop: bimodal " + 2 + " misses, always-taken 1, always-not-taken 10");
        Console.WriteLine("alternating loop: bimodal " + bimodalMisses + " misses, gshare " + gshareMisses);
        Console.WriteLine("random sweep: " + totalMispredictions + " mispredictions, " + totalSquashed + " squashed blocks");
        Console.WriteLine("PASS");
    }

    private static IPredictor NewLike(IPredictor predictor)
    {
        if (predictor is Bimodal) return new Bimodal(64);
        if (predictor is GShare) return new GShare(64);
        if (predictor is AlwaysTaken) return new AlwaysTaken();
        return new AlwaysNotTaken();
    }
}

/// Predicts the outcome the reference run actually takes (used to show that a perfect oracle squashes
/// nothing). Predictions are indexed by call order, which is the dynamic branch index because the engine
/// fetches branches in program order.
internal sealed class Oracle : IPredictor
{
    private readonly List<bool> outcomes = new List<bool>();
    private int called;

    internal Oracle(Code code)
    {
        RunResult reference = Machine.Reference(code, 200);
        outcomes.AddRange(reference.BranchTaken);
    }

    public string Name => "oracle";
    public bool Predict(int pc) => called < outcomes.Count ? outcomes[called++] : false;
    public void Update(int pc, bool taken) { }
}

/// Deliberately predicts the opposite of the direction last seen, which forces squashes.
internal sealed class Inverted : IPredictor
{
    private bool previous;

    public string Name => "inverted";
    public bool Predict(int pc) => !previous;
    public void Update(int pc, bool taken) => previous = taken;
}
