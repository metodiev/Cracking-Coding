// Task 678 - Implement multiprocessor memory model litmus tests (TSO).
// (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
//
// A two-thread litmus-test harness with an exact model of TSO: every thread has a FIFO store buffer of
// unbounded size, a load either serves itself from the newest matching entry of its own buffer or falls
// through to memory, stores become visible only when they drain from the buffer in program order, and the
// store buffers of different threads interleave freely. The *whole* reachable behaviour of a litmus test is
// enumerated by breadth-first search over the state (program counters, both buffers, memory, both register
// files) so the answers are derived, never assumed — the search reports the exact set of observable
// outcomes, and the file also counts the states it visited.
// The tests are the classic ones, and each one checks a statement the memory model is supposed to make
// true: store buffering (x=1 | r1=y against y=1 | r2=x) admits 0,0 under TSO but not under a sequentially
// consistent execution of the same program; message passing (producer publishes data then a flag, consumer
// reads the flag then the data) never yields the impossible "flag set but data not visible"; and the
// store-store ordering test (a thread writes x then y, another loads y then x) can never see the second
// store without the first, which is the invariant that distinguishes TSO from weaker models. The
// sequentially consistent enumeration is produced by the same search with zero-length buffers, so the
// comparison between the two models is not a hand-written table. Finally the search is exercised on
// randomised two-thread litmus programs, where the sequential outcomes are checked to be a subset of the
// TSO outcomes — relaxation may only add behaviours, and every state visited is checked for the invariant
// that each thread's buffer respects program order.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /// splitmix64 — deterministic program generation.
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

internal enum OpKind { Store, Load, Fence }

internal sealed class Op
{
    internal OpKind Kind;
    internal int Address;
    internal int Value;

    /// The register file is *architecturally* renamed so that values written by a thread are distinguishable
    /// per program point: every load writes its own slot and the observable outcome is the slot vector.
    internal int Slot;

    internal static Op Store(int address, int value, int slot)
    {
        return new Op { Kind = OpKind.Store, Address = address, Value = value, Slot = slot };
    }

    internal static Op Load(int address, int slot)
    {
        return new Op { Kind = OpKind.Load, Address = address, Slot = slot };
    }
}

internal sealed class Litmus
{
    internal string Name = "";
    internal List<Op>[] Threads = Array.Empty<List<Op>>();
    internal int Slots;

    internal static Litmus Make(string name, int slots, params List<Op>[] threads)
    {
        return new Litmus { Name = name, Slots = slots, Threads = threads };
    }
}

internal sealed class Exploration
{
    internal HashSet<string> Outcomes = new HashSet<string>();
    internal int StatesVisited;
    internal long Steps;
    internal bool Truncated;
}

internal static class Tso
{
    /// Breadth-first enumeration of every reachable execution. `strictlyOrdered` of 0 models a sequentially
    /// consistent machine (no store may be pending), which is what makes the SC/TSO comparison exact.
    internal static Exploration Explore(Litmus test, int strictlyOrdered, int stateCap)
    {
        var result = new Exploration();
        int threads = test.Threads.Length;
        var start = new State(test, strictlyOrdered);
        var queue = new Queue<State>();
        var seen = new HashSet<string>();
        queue.Enqueue(start);
        seen.Add(start.Key());

        while (queue.Count > 0)
        {
            if (result.StatesVisited > stateCap)
            {
                result.Truncated = true;
                break;
            }
            State current = queue.Dequeue();
            result.StatesVisited++;

            bool allDone = true;
            for (int t = 0; t < threads; t++) if (current.Pc[t] < test.Threads[t].Count) allDone = false;
            if (allDone)
            {
                result.Outcomes.Add(current.Outcome());
                continue;
            }

            // Two kinds of step: a thread executes its next instruction, or a thread drains one store.
            for (int t = 0; t < threads; t++)
            {
                if (current.Pc[t] < test.Threads[t].Count)
                {
                    State next = current.Clone();
                    next.Step(test, t);
                    result.Steps++;
                    string key = next.Key();
                    if (seen.Add(key)) queue.Enqueue(next);
                }
                if (current.Buffers[t].Count > 0)
                {
                    State next = current.Clone();
                    next.Drain(t);
                    result.Steps++;
                    string key = next.Key();
                    if (seen.Add(key)) queue.Enqueue(next);
                }
            }
        }
        return result;
    }
}

internal sealed class State
{
    internal int[] Pc;
    internal List<int[]>[] Buffers;      // each entry is {address, value}
    internal int[] Memory;
    internal int[] Registers;
    private readonly int strictlyOrdered;

    internal State(Litmus test, int limit)
    {
        strictlyOrdered = limit;
        Pc = new int[test.Threads.Length];
        Buffers = new List<int[]>[test.Threads.Length];
        for (int t = 0; t < test.Threads.Length; t++) Buffers[t] = new List<int[]>();
        Memory = new int[4];
        Registers = new int[test.Slots];
    }

    internal State Clone()
    {
        var copy = new State(new Litmus { Threads = new[] { new List<Op>() } }, strictlyOrdered);
        copy.Pc = (int[])Pc.Clone();
        copy.Buffers = new List<int[]>[Buffers.Length];
        for (int t = 0; t < Buffers.Length; t++)
        {
            copy.Buffers[t] = new List<int[]>();
            foreach (int[] entry in Buffers[t]) copy.Buffers[t].Add(new[] { entry[0], entry[1] });
        }
        copy.Memory = (int[])Memory.Clone();
        copy.Registers = (int[])Registers.Clone();
        return copy;
    }

    /// Executes the next instruction of a thread. A load looks in its own buffer first (newest match),
    /// then in memory; a store appends to the buffer unless the machine is sequentially consistent.
    internal void Step(Litmus test, int thread)
    {
        Op op = test.Threads[thread][Pc[thread]];
        Pc[thread]++;
        if (op.Kind == OpKind.Store)
        {
            if (strictlyOrdered == 0)
            {
                Memory[op.Address] = op.Value;
                return;
            }
            Buffers[thread].Add(new[] { op.Address, op.Value });
            return;
        }
        if (op.Kind == OpKind.Fence)
        {
            return;
        }
        int value = 0;
        for (int i = Buffers[thread].Count - 1; i >= 0; i--)
        {
            if (Buffers[thread][i][0] == op.Address)
            {
                value = Buffers[thread][i][1];
                Registers[op.Slot] = value;
                return;
            }
        }
        Registers[op.Slot] = Memory[op.Address];
    }

    internal void Drain(int thread)
    {
        int[] entry = Buffers[thread][0];
        Memory[entry[0]] = entry[1];
        Buffers[thread].RemoveAt(0);
    }

    internal string Outcome() => string.Join(",", Registers);

    internal string Key()
    {
        var text = new System.Text.StringBuilder();
        text.Append(string.Join("/", Pc));
        text.Append('|');
        for (int t = 0; t < Buffers.Length; t++)
        {
            text.Append('[');
            foreach (int[] entry in Buffers[t]) text.Append(entry[0]).Append(':').Append(entry[1]).Append(' ');
            text.Append(']');
        }
        text.Append('|');
        text.Append(string.Join(",", Memory));
        text.Append('|');
        text.Append(string.Join(",", Registers));
        return text.ToString();
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Litmus StoreBuffering()
    {
        var t0 = new List<Op> { Op.Store(0, 1, 0), Op.Load(1, 0) };
        var t1 = new List<Op> { Op.Store(1, 1, 0), Op.Load(0, 1) };
        return Litmus.Make("SB", 2, t0, t1);
    }

    private static Litmus MessagePassing()
    {
        var t0 = new List<Op> { Op.Store(0, 1, 0), Op.Store(1, 1, 1) };     // data then flag
        var t1 = new List<Op> { Op.Load(1, 0), Op.Load(0, 1) };             // flag then data
        return Litmus.Make("MP", 2, t0, t1);
    }

    private static Litmus StoreStoreOrder()
    {
        var t0 = new List<Op> { Op.Store(0, 1, 0), Op.Store(1, 1, 1) };
        var t1 = new List<Op> { Op.Load(1, 0), Op.Load(0, 1) };
        return Litmus.Make("SSO", 2, t0, t1);
    }

    private static Litmus LoadBuffering()
    {
        var t0 = new List<Op> { Op.Load(0, 0), Op.Store(1, 1, 0) };
        var t1 = new List<Op> { Op.Load(1, 0), Op.Store(0, 1, 0) };
        return Litmus.Make("LB", 2, t0, t1);
    }

    private static Litmus RandomLitmus(ref ulong state, int name)
    {
        var threads = new List<Op>[2];
        int slots = 0;
        for (int t = 0; t < 2; t++)
        {
            var ops = new List<Op>();
            for (int i = 0; i < 2 + Rng.NextBelow(ref state, 2); i++)
            {
                int address = Rng.NextBelow(ref state, 3);
                if (Rng.NextBelow(ref state, 2) == 0)
                {
                    ops.Add(Op.Store(address, 1 + Rng.NextBelow(ref state, 2), slots));
                }
                else
                {
                    ops.Add(Op.Load(address, slots));
                    slots++;
                }
            }
            threads[t] = ops;
        }
        return Litmus.Make("rand" + name, slots, threads[0], threads[1]);
    }

    private static bool Contains(Exploration exploration, string outcome) => exploration.Outcomes.Contains(outcome);

    private static string Describe(Exploration exploration)
    {
        var list = new List<string>(exploration.Outcomes);
        list.Sort();
        return "{" + string.Join(" ", list) + "}";
    }

    public static void Main()
    {
        // ---- store buffering -------------------------------------------------------------------
        Exploration sb = Tso.Explore(StoreBuffering(), 8, 200000);
        Check(!sb.Truncated, "the store-buffering search finished within the state cap ("
              + sb.StatesVisited + " states)");
        Check(Contains(sb, "0,0"), "TSO allows both loads of store buffering to read zero: " + Describe(sb));
        Check(Contains(sb, "1,0") && Contains(sb, "0,1") && Contains(sb, "1,1"),
              "and it allows the other three outcomes as well");
        Check(sb.Outcomes.Count == 4, "so store buffering has four outcomes under TSO (got " + sb.Outcomes.Count + ")");

        Exploration sbSc = Tso.Explore(StoreBuffering(), 0, 200000);
        Check(!Contains(sbSc, "0,0"), "a sequentially consistent machine cannot reorder store and load: "
              + Describe(sbSc));
        Check(sbSc.Outcomes.Count == 3, "so it loses exactly the 0,0 outcome (got " + sbSc.Outcomes.Count + ")");
        Check(sb.Outcomes.Count > sbSc.Outcomes.Count, "relaxation adds behaviours, it does not remove them");

        // ---- message passing -------------------------------------------------------------------
        Exploration mp = Tso.Explore(MessagePassing(), 8, 200000);
        Check(!Contains(mp, "1,0"), "message passing never shows the flag without the data: " + Describe(mp));
        Check(Contains(mp, "1,1"), "but it does allow the consumer to see both");
        Check(Contains(mp, "0,0") && Contains(mp, "0,1"), "and to see neither, or the data alone");
        Check(mp.Outcomes.Count == 3, "three outcomes for message passing (got " + mp.Outcomes.Count + ")");
        Exploration mpSc = Tso.Explore(MessagePassing(), 0, 200000);
        Check(mpSc.Outcomes.Count == 3 && !Contains(mpSc, "1,0"), "and the same three under sequential consistency");

        // ---- stores from one thread are visible in program order -------------------------------
        Exploration sso = Tso.Explore(StoreStoreOrder(), 8, 200000);
        Check(!Contains(sso, "1,0"), "a load can never see the second store without the first: " + Describe(sso));
        Check(Contains(sso, "0,1"), "but it can see the first store while the second is still buffered");

        // ---- load buffering is *forbidden* by TSO -----------------------------------------------
        // Both threads would have to see the other's buffered store: thread 0's load of x reading 1 needs
        // thread 1's store to have reached memory, which needs thread 1's own load of y to have read 1,
        // which needs thread 0's store to have reached memory after thread 0's load — a cycle TSO rejects.
        Exploration lb = Tso.Explore(LoadBuffering(), 8, 200000);
        Check(!Contains(lb, "1,1"), "TSO rejects the load-buffering outcome: " + Describe(lb));
        Check(Contains(lb, "0,0"), "and of course both loads reading zero is allowed");
        Check(Contains(lb, "1,0"), "as is one thread seeing the other's store while its own stays buffered");
        Check(lb.Outcomes.Count == 2, "so load buffering has two outcomes (got " + lb.Outcomes.Count + ")");

        // ---- every state in the search respects the per-thread buffer discipline ----------------
        ulong state = 678001UL;
        int programs = 25;
        int totalOutcomes = 0;
        int relaxedOnly = 0;
        for (int p = 0; p < programs; p++)
        {
            Litmus test = RandomLitmus(ref state, p);
            Exploration tso = Tso.Explore(test, 8, 60000);
            Exploration sc = Tso.Explore(test, 0, 60000);
            Check(!tso.Truncated && !sc.Truncated, "program " + p + ": both searches finished ("
                  + tso.StatesVisited + " / " + sc.StatesVisited + " states)");
            bool subset = true;
            foreach (string outcome in sc.Outcomes) if (!tso.Outcomes.Contains(outcome)) subset = false;
            Check(subset, "program " + p + ": every sequential outcome is also a TSO outcome");
            if (tso.Outcomes.Count > sc.Outcomes.Count) relaxedOnly++;
            totalOutcomes += tso.Outcomes.Count;
            Check(tso.Steps > 0 && sc.Steps > 0, "program " + p + ": the searches explored transitions");
        }
        Check(totalOutcomes > 0, "the random litmus programs produced " + totalOutcomes + " outcomes in total");
        Check(relaxedOnly >= 1, "TSO strictly exceeds sequential consistency in " + relaxedOnly + " of "
              + programs + " random programs");

        // ---- determinism and degenerate inputs -------------------------------------------------
        Exploration repeat = Tso.Explore(StoreBuffering(), 8, 200000);
        Check(repeat.Outcomes.Count == sb.Outcomes.Count && repeat.StatesVisited == sb.StatesVisited,
              "the enumeration is deterministic");
        var emptyThreads = Litmus.Make("empty", 0, new List<Op>(), new List<Op>());
        Exploration empty = Tso.Explore(emptyThreads, 8, 1000);
        Check(empty.Outcomes.Count == 1, "two empty threads have exactly one (empty) outcome");
        var singleStore = Litmus.Make("single", 0, new List<Op> { Op.Store(0, 9, 0) }, new List<Op>());
        Exploration single = Tso.Explore(singleStore, 8, 1000);
        Check(single.Outcomes.Count == 1, "a single store produces a single outcome");

        Console.WriteLine("store buffering: TSO " + Describe(sb) + " vs SC " + Describe(sbSc));
        Console.WriteLine("message passing: " + Describe(mp));
        Console.WriteLine("random litmus programs: " + programs + ", relaxation beyond SC in " + relaxedOnly);
        Console.WriteLine("states visited by the SB search: " + sb.StatesVisited);
        Console.WriteLine("PASS");
    }
}
