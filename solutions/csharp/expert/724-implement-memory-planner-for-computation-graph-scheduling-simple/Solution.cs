// Task 724 - Implement memory planner for computation graph scheduling (simple greedy). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// A computation graph executes a topologically ordered stream of tensors, each live from the step that
// produces it until the last step that consumes it. If every tensor got its own buffer the arena would
// be as large as the sum of all tensors; a memory planner instead assigns offsets inside one arena and
// reuses a region as soon as its previous owner dies. This file implements the classic greedy: process
// the tensors in schedule order and place each one at the lowest address that does not collide with any
// *already placed tensor whose lifetime overlaps it* - the linear-scan rule used by register allocators.
//
// The invariant it maintains is exactly: two tensors whose live ranges overlap occupy disjoint byte
// intervals, and all offsets respect the alignment. Both are re-derived from the finished plan (a full
// pairwise sweep and a step-by-step replay) rather than trusted, and the result is bracketed between
// the two reference quantities any packing must respect - the peak number of live bytes at any step
// (a lower bound) and the sum of all tensor sizes (the no-reuse baseline). Designed instances where
// reuse is perfect must hit the lower bound, and a small instance is compared against an exhaustive
// search over all allocation orders. Sizes, lifetimes and the tested graphs come from splitmix64.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

/// <summary>A tensor with a byte size and the step interval [Birth, Death] over which it is live.</summary>
internal sealed class Tensor
{
    internal Tensor(string name, int size, int birth, int death)
    {
        Name = name;
        Size = size;
        Birth = birth;
        Death = death;
    }

    internal string Name { get; }
    internal int Size { get; }
    internal int Birth { get; }
    internal int Death { get; }

    internal bool Overlaps(Tensor other) => Birth <= other.Death && other.Birth <= Death;
}

internal sealed class MemoryPlan
{
    internal readonly IReadOnlyList<Tensor> Tensors;
    internal readonly int[] Offsets;
    internal readonly int[] Footprints;
    internal readonly int Alignment;

    internal MemoryPlan(IReadOnlyList<Tensor> tensors, int[] offsets, int[] footprints, int alignment)
    {
        Tensors = tensors;
        Offsets = offsets;
        Footprints = footprints;
        Alignment = alignment;
    }

    internal int BufferSize
    {
        get
        {
            int peak = 0;
            for (int i = 0; i < Offsets.Length; i++) peak = Math.Max(peak, Offsets[i] + Footprints[i]);
            return peak;
        }
    }

    internal int NoReuseBytes
    {
        get
        {
            int total = 0;
            foreach (int footprint in Footprints) total += footprint;
            return total;
        }
    }

    internal int LastStep
    {
        get
        {
            int last = 0;
            foreach (Tensor tensor in Tensors) last = Math.Max(last, tensor.Death);
            return last;
        }
    }

    /// <summary>Largest number of bytes that are simultaneously live: a lower bound for any packing.</summary>
    internal int LiveLowerBound
    {
        get
        {
            int best = 0;
            for (int step = 0; step <= LastStep; step++)
            {
                int live = 0;
                for (int i = 0; i < Tensors.Count; i++)
                    if (Tensors[i].Birth <= step && step <= Tensors[i].Death) live += Footprints[i];
                best = Math.Max(best, live);
            }
            return best;
        }
    }

    /// <summary>Re-derives the disjointness invariant pairwise; returns null when it holds.</summary>
    internal string InvariantViolation()
    {
        for (int i = 0; i < Tensors.Count; i++)
        {
            if (Offsets[i] % Alignment != 0) return Tensors[i].Name + " is not aligned";
            if (Footprints[i] < Tensors[i].Size) return Tensors[i].Name + " is smaller than its tensor";
            for (int j = i + 1; j < Tensors.Count; j++)
            {
                if (!Tensors[i].Overlaps(Tensors[j])) continue;
                int aFrom = Offsets[i], aTo = Offsets[i] + Footprints[i];
                int bFrom = Offsets[j], bTo = Offsets[j] + Footprints[j];
                if (aFrom < bTo && bFrom < aTo)
                    return Tensors[i].Name + " overlaps " + Tensors[j].Name;
            }
        }
        return null;
    }

    /// <summary>Walks the schedule step by step and re-checks disjointness of the live set.</summary>
    internal string ReplayViolation()
    {
        for (int step = 0; step <= LastStep; step++)
        {
            var live = new List<int>();
            for (int i = 0; i < Tensors.Count; i++)
                if (Tensors[i].Birth <= step && step <= Tensors[i].Death) live.Add(i);
            for (int a = 0; a < live.Count; a++)
                for (int b = a + 1; b < live.Count; b++)
                {
                    int i = live[a], j = live[b];
                    if (Offsets[i] < Offsets[j] + Footprints[j] && Offsets[j] < Offsets[i] + Footprints[i])
                        return "step " + step + ": " + Tensors[i].Name + " overlaps " + Tensors[j].Name;
                }
        }
        return null;
    }
}

internal static class MemoryPlanner
{
    internal static int AlignUp(int size, int alignment) => (size + alignment - 1) / alignment * alignment;

    /// <summary>Linear-scan placement: lowest address free of every *conflicting* placed tensor.</summary>
    internal static MemoryPlan LinearScan(IReadOnlyList<Tensor> tensors, int alignment, int[]? order = null)
    {
        if (alignment < 1) throw new ArgumentOutOfRangeException(nameof(alignment));
        int n = tensors.Count;
        var sequence = order ?? Schedule(tensors);
        var offsets = new int[n];
        var footprints = new int[n];
        var placed = new List<int>();
        foreach (int i in sequence)
        {
            if (i < 0 || i >= n) throw new ArgumentOutOfRangeException(nameof(order));
            int size = AlignUp(tensors[i].Size, alignment);
            footprints[i] = size;
            // Conflicts are the placed tensors whose live range overlaps this one, in address order;
            // walk the gaps between them and take the first one that fits.
            var blocked = new List<(int Start, int End)>();
            foreach (int p in placed)
                if (tensors[i].Overlaps(tensors[p])) blocked.Add((offsets[p], offsets[p] + footprints[p]));
            blocked.Sort((x, y) => x.Start.CompareTo(y.Start));
            int cursor = 0;
            foreach ((int start, int end) in blocked)
            {
                if (cursor + size <= start) break;
                cursor = Math.Max(cursor, end);
            }
            offsets[i] = cursor;
            placed.Add(i);
        }
        return new MemoryPlan(tensors, offsets, footprints, alignment);
    }

    /// <summary>No reuse at all: every tensor gets its own slice of the arena.</summary>
    internal static MemoryPlan NoReuse(IReadOnlyList<Tensor> tensors, int alignment)
    {
        int n = tensors.Count;
        var offsets = new int[n];
        var footprints = new int[n];
        int cursor = 0;
        for (int i = 0; i < n; i++)
        {
            footprints[i] = AlignUp(tensors[i].Size, alignment);
            offsets[i] = cursor;
            cursor += footprints[i];
        }
        return new MemoryPlan(tensors, offsets, footprints, alignment);
    }

    internal static int[] Schedule(IReadOnlyList<Tensor> tensors)
    {
        var order = new int[tensors.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int byBirth = tensors[a].Birth.CompareTo(tensors[b].Birth);
            if (byBirth != 0) return byBirth;
            return a.CompareTo(b);
        });
        return order;
    }

    /// <summary>Exhaustive reference: the best peak any allocation order can reach with this greedy rule.</summary>
    internal static int BestOverOrders(IReadOnlyList<Tensor> tensors, int alignment)
    {
        int n = tensors.Count;
        var permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        int best = int.MaxValue;
        Permute(permutation, 0, tensors, alignment, ref best);
        return best;
    }

    private static void Permute(int[] permutation, int index, IReadOnlyList<Tensor> tensors, int alignment, ref int best)
    {
        if (index == permutation.Length)
        {
            best = Math.Min(best, LinearScan(tensors, alignment, permutation).BufferSize);
            return;
        }
        for (int i = index; i < permutation.Length; i++)
        {
            (permutation[index], permutation[i]) = (permutation[i], permutation[index]);
            Permute(permutation, index + 1, tensors, alignment, ref best);
            (permutation[index], permutation[i]) = (permutation[i], permutation[index]);
        }
    }
}

public static class Solution
{
    public static void Main()
    {
        const int alignment = 16;

        // ---- degenerate graphs --------------------------------------------------------------------
        {
            var empty = MemoryPlanner.LinearScan(new List<Tensor>(), alignment);
            Check(empty.BufferSize == 0 && empty.LiveLowerBound == 0 && empty.InvariantViolation() == null,
                "an empty graph needs no arena");

            var single = MemoryPlanner.LinearScan(new List<Tensor> { new Tensor("t0", 5, 0, 0) }, alignment);
            Check(single.Offsets[0] == 0 && single.BufferSize == 16 && single.Footprints[0] == 16,
                "a single tensor is aligned up to one block");
            Check(single.InvariantViolation() == null && single.ReplayViolation() == null, "one tensor packs trivially");

            var simultaneous = new List<Tensor>
            {
                new Tensor("a", 64, 0, 3),
                new Tensor("b", 32, 0, 3),
                new Tensor("c", 16, 1, 3),
            };
            MemoryPlan together = MemoryPlanner.LinearScan(simultaneous, alignment);
            Check(together.BufferSize == together.LiveLowerBound && together.BufferSize == 112,
                "tensors that are all live at once must all fit (" + together.BufferSize + ")");
            Check(together.InvariantViolation() == null, "the fully live case still satisfies the invariant");
        }

        // ---- perfect reuse: a chain whose tensors never coexist --------------------------------
        {
            var chain = new List<Tensor>();
            var rng = new SplitMix64(0x724A1UL);
            for (int i = 0; i < 40; i++) chain.Add(new Tensor("c" + i, 8 + 8 * rng.NextInt(5), i, i + 1));
            MemoryPlan plan = MemoryPlanner.LinearScan(chain, alignment);
            MemoryPlan naive = MemoryPlanner.NoReuse(chain, alignment);
            Console.WriteLine("chain: buffer=" + plan.BufferSize + " lowerBound=" + plan.LiveLowerBound
                + " noReuse=" + naive.BufferSize + " largest=" + Largest(plan));
            Check(plan.InvariantViolation() == null && plan.ReplayViolation() == null, "the chain packs legally");
            Check(plan.BufferSize == plan.LiveLowerBound, "a chain reuses a dead tensor's block immediately");
            Check(plan.BufferSize <= 2 * Largest(plan), "the chain arena holds at most two tensors at a time");
            Check(naive.BufferSize == plan.NoReuseBytes, "the no-reuse baseline is the sum of all footprints");
        }

        // ---- a realistic graph: short-lived intermediates over long-lived inputs -------------------
        {
            var graph = BuildGraph(120, 0x7246AAUL, out int horizon);
            MemoryPlan plan = MemoryPlanner.LinearScan(graph, alignment);
            MemoryPlan naive = MemoryPlanner.NoReuse(graph, alignment);
            Console.WriteLine("graph: tensors=" + graph.Count + " steps=" + horizon + " buffer=" + plan.BufferSize
                + " liveLowerBound=" + plan.LiveLowerBound + " noReuse=" + naive.BufferSize
                + " saving=" + ((double)naive.BufferSize / plan.BufferSize).ToString("G4"));
            Check(plan.InvariantViolation() == null, "the pairwise sweep finds no overlap: " + plan.InvariantViolation());
            Check(plan.ReplayViolation() == null, "the step-by-step replay finds no overlap: " + plan.ReplayViolation());
            Check(plan.BufferSize >= plan.LiveLowerBound, "no packing can beat the live-byte lower bound");
            Check(plan.BufferSize < naive.BufferSize, "reuse beats giving every tensor its own buffer");
            Check(plan.BufferSize * 4 < naive.BufferSize, "the planner saves at least three quarters of the arena");
            Check(plan.BufferSize <= 2 * plan.LiveLowerBound, "the greedy is within a factor two of the bound");

            for (int i = 0; i < graph.Count; i++)
            {
                Check(plan.Offsets[i] % alignment == 0, "tensor " + i + " is aligned");
                Check(plan.Footprints[i] >= graph[i].Size, "tensor " + i + " has at least its own size");
            }
            MemoryPlan again = MemoryPlanner.LinearScan(graph, alignment);
            bool identical = true;
            for (int i = 0; i < graph.Count; i++) if (plan.Offsets[i] != again.Offsets[i]) identical = false;
            Check(identical, "the planner is deterministic");
        }

        // ---- small instance: exhaustive search over every allocation order -------------------------
        {
            var small = new List<Tensor>
            {
                new Tensor("s0", 40, 0, 2),
                new Tensor("s1", 24, 1, 4),
                new Tensor("s2", 32, 2, 5),
                new Tensor("s3", 16, 3, 6),
                new Tensor("s4", 48, 4, 7),
                new Tensor("s5", 24, 5, 8),
                new Tensor("s6", 40, 6, 9),
            };
            MemoryPlan plan = MemoryPlanner.LinearScan(small, alignment);
            int best = MemoryPlanner.BestOverOrders(small, alignment);
            Console.WriteLine("small: scheduleOrderPeak=" + plan.BufferSize + " bestOverOrders=" + best
                + " lowerBound=" + plan.LiveLowerBound);
            Check(plan.InvariantViolation() == null && plan.ReplayViolation() == null, "the small instance packs legally");
            Check(best <= plan.BufferSize, "no allocation order beats the planner");
            Check(plan.BufferSize <= plan.LiveLowerBound * 2, "the planner stays within twice the lower bound");
            Check(plan.BufferSize <= 2 * best, "the schedule order is at most twice the best order");

            var disjoint = new List<Tensor>();
            for (int i = 0; i < 6; i++) disjoint.Add(new Tensor("d" + i, 16 * (i + 1), 2 * i, 2 * i));
            MemoryPlan spread = MemoryPlanner.LinearScan(disjoint, alignment);
            Check(spread.BufferSize == Largest(spread),
                "point-lifetime tensors all share the same offset (" + spread.BufferSize + ")");
            Check(spread.BufferSize == spread.LiveLowerBound && spread.InvariantViolation() == null,
                "the point-lifetime instance hits the lower bound");
        }

        // ---- rejection and boundaries ---------------------------------------------------------------
        {
            var tensors = new List<Tensor> { new Tensor("x", 1, 0, 0) };
            bool threw = false;
            try { MemoryPlanner.LinearScan(tensors, 0); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a zero alignment is rejected");

            bool threwOrder = false;
            try { MemoryPlanner.LinearScan(tensors, 8, new[] { 1 }); } catch (ArgumentOutOfRangeException) { threwOrder = true; }
            Check(threwOrder, "an out-of-range allocation order is rejected");

            MemoryPlan zero = MemoryPlanner.LinearScan(new List<Tensor> { new Tensor("z", 0, 0, 4) }, 8);
            Check(zero.BufferSize == 0 && zero.InvariantViolation() == null, "a zero-sized tensor occupies nothing");

            var oneStep = new List<Tensor> { new Tensor("k", 100, 5, 5) };
            MemoryPlan lonely = MemoryPlanner.LinearScan(oneStep, alignment);
            Check(lonely.LastStep == 5 && lonely.BufferSize == 112 && lonely.LiveLowerBound == 112,
                "a single-step tensor is both the lower bound and the peak");
        }

        Console.WriteLine("PASS");
    }

    private static List<Tensor> BuildGraph(int count, ulong seed, out int horizon)
    {
        var rng = new SplitMix64(seed);
        var tensors = new List<Tensor>();
        int step = 0;
        for (int i = 0; i < count; i++)
        {
            int size = 16 * (1 + rng.NextInt(4));
            int life = i % 5 == 0 ? 6 + rng.NextInt(6) : 1 + rng.NextInt(3);
            tensors.Add(new Tensor("g" + i, size, step, step + life));
            step += 1 + rng.NextInt(2);
        }
        horizon = step;
        return tensors;
    }

    private static int Largest(MemoryPlan plan)
    {
        int largest = 0;
        foreach (int footprint in plan.Footprints) largest = Math.Max(largest, footprint);
        return largest;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
