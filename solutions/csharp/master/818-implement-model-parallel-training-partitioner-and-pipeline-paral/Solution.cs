// Task 818 - Implement model parallel training partitioner and pipeline parallel executor. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Two halves, both deterministic and in-process:
//
//   * Partitioner - a model is a chain of L stages, each with a compute cost (its FLOP weight) and an
//     activation footprint per microbatch. Splitting the chain into P *contiguous* groups is the
//     classical linear partition problem "minimise the maximum group cost", here with an extra hard
//     constraint: a group's activation footprint must fit the device memory capacity. The optimum is
//     computed by dynamic programming (dp[p][i] = min over j of max(dp[p-1][j], cost(j..i)) over
//     feasible groups) and checked against an exhaustive enumeration of every cut placement, plus a
//     greedy balancer that is shown never to beat the optimum.
//   * Executor - a 1F1B (one forward, one backward) pipeline over M microbatches. A partition performs
//     one forward *or* one backward for one microbatch at a time, taking as many ticks as the group's
//     cost. Dependencies are explicit: forward at partition p needs the forward at p-1 for the same
//     microbatch, backward at p needs the backward at p+1 (or the forward if p is the last partition).
//     The 1F1B memory rule is enforced as a per-partition in-flight limit: partition p may not start a
//     forward while it already has more than P-p microbatches forwarded but not yet backwarded, which
//     is what keeps the activation memory bounded (the last partition strictly alternates).
//
// Simplifications (documented): the compute model is "a partition's action costs groupFlops ticks"
// rather than real hardware timing; activations are stored as an exact double array rather than a
// tensor library, and the "device" is a per-partition bucket in memory; there is no inter-partition
// communication cost (a real pipeline hides the same amount of transfer as compute); the model is a
// chain of affine stages, which is enough to check the executor against an analytic gradient and a
// finite-difference oracle.
//
// Verified invariants:
//   partitioning - the DP optimum equals the exhaustive optimum over all cut placements, every group is
//       contiguous, covers the chain exactly once, and satisfies the memory capacity (an infeasible
//       instance is reported as infeasible rather than silently mis-partitioned);
//   safety       - the schedule is always legal (one action per partition per tick, forward only after
//       the upstream forward of that microbatch, backward only after the downstream backward), checked
//       per tick with explicit counters;
//   exactness    - the pipelined forward equals a purely sequential execution of the same chain, and
//       the accumulated gradients equal the analytic gradients and the finite-difference oracle;
//   convergence  - a gradient step computed from the pipeline lowers the loss;
//   efficiency   - makespan bounds (>= the 2*M*c critical path, <= the (M+P-1) period times the group
//       cost), the measured bubble fraction, the speedup over serial execution, and the memory
//       advantage of 1F1B over an all-forwards-then-all-backwards schedule.
using System;
using System.Collections.Generic;
using System.Linq;

internal struct SplitMix64
{
    private ulong state;

    public SplitMix64(ulong seed) { state = seed; }

    public ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public int NextInt(int bound) => (int)(Next() % (ulong)bound);

    public double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

// ---- partitioner --------------------------------------------------------------------------------------

internal sealed class StageSpec
{
    public int Cost;
    public int Memory;

    public StageSpec(int cost, int memory)
    {
        Cost = cost;
        Memory = memory;
    }
}

internal sealed class Partition
{
    public bool Feasible = true;
    public readonly List<int[]> Groups = new List<int[]>();
    public long MaxCost;
    public string Reason = "";

    public int GroupCount => Groups.Count;
}

internal static class Partitioner
{
    private static bool GroupFits(IList<StageSpec> stages, int from, int to, long capacity, out long cost, out long memory)
    {
        cost = 0;
        memory = 0;
        for (int i = from; i < to; i++)
        {
            cost += stages[i].Cost;
            memory += stages[i].Memory;
        }
        return memory <= capacity;
    }

    public static Partition Dynamic(IList<StageSpec> stages, int parts, long capacity)
    {
        int n = stages.Count;
        var result = new Partition();
        if (parts <= 0 || parts > n)
        {
            result.Feasible = false;
            result.Reason = "cannot split " + n + " stages into " + parts + " parts";
            return result;
        }
        const long infinity = long.MaxValue / 4;
        var best = new long[parts + 1, n + 1];
        var cut = new int[parts + 1, n + 1];
        for (int p = 0; p <= parts; p++)
            for (int i = 0; i <= n; i++) best[p, i] = infinity;
        best[0, 0] = 0;
        for (int p = 1; p <= parts; p++)
            for (int i = p; i <= n; i++)
                for (int j = p - 1; j < i; j++)
                {
                    if (best[p - 1, j] >= infinity) continue;
                    if (!GroupFits(stages, j, i, capacity, out long cost, out _)) continue;
                    long candidate = Math.Max(best[p - 1, j], cost);
                    if (candidate < best[p, i])
                    {
                        best[p, i] = candidate;
                        cut[p, i] = j;
                    }
                }
        if (best[parts, n] >= infinity)
        {
            result.Feasible = false;
            result.Reason = "no partition respects the memory capacity " + capacity;
            return result;
        }
        var groups = new List<int[]>();
        int end = n;
        for (int p = parts; p >= 1; p--)
        {
            int start = cut[p, end];
            var group = new int[end - start];
            for (int i = start; i < end; i++) group[i - start] = i;
            groups.Add(group);
            end = start;
        }
        groups.Reverse();
        result.Groups.AddRange(groups);
        result.MaxCost = best[parts, n];
        return result;
    }

    // Exhaustive: every way of choosing the P-1 cut positions.
    public static Partition BruteForce(IList<StageSpec> stages, int parts, long capacity)
    {
        int n = stages.Count;
        var result = new Partition();
        long best = long.MaxValue;
        List<int[]> bestGroups = new List<int[]>();
        var cuts = new int[parts - 1];
        bool Feasible(long limit)
        {
            int start = 0;
            for (int c = 0; c <= cuts.Length; c++)
            {
                int end = c < cuts.Length ? cuts[c] : n;
                if (!GroupFits(stages, start, end, capacity, out _, out _)) return false;
                start = end;
            }
            return true;
        }
        long MaxCost()
        {
            long max = 0;
            int start = 0;
            for (int c = 0; c <= cuts.Length; c++)
            {
                int end = c < cuts.Length ? cuts[c] : n;
                GroupFits(stages, start, end, capacity, out long cost, out _);
                max = Math.Max(max, cost);
                start = end;
            }
            return max;
        }
        void Enumerate(int index, int from)
        {
            if (index == cuts.Length)
            {
                if (!Feasible(long.MaxValue)) return;
                long cost = MaxCost();
                if (cost < best)
                {
                    best = cost;
                    bestGroups = Describe(cuts, n);
                }
                return;
            }
            for (int position = from; position <= n - (cuts.Length - index); position++)
            {
                cuts[index] = position;
                Enumerate(index + 1, position + 1);
            }
        }
        if (parts <= 0 || parts > n)
        {
            result.Feasible = false;
            result.Reason = "invalid part count";
            return result;
        }
        if (parts == 1)
        {
            if (!GroupFits(stages, 0, n, capacity, out long cost, out _))
            {
                result.Feasible = false;
                result.Reason = "the whole chain does not fit";
                return result;
            }
            result.Groups.Add(Enumerable.Range(0, n).ToArray());
            result.MaxCost = cost;
            return result;
        }
        Enumerate(0, 1);
        if (bestGroups.Count == 0)
        {
            result.Feasible = false;
            result.Reason = "no feasible partition exists";
            return result;
        }
        result.Groups.AddRange(bestGroups);
        result.MaxCost = best;
        return result;
    }

    private static List<int[]> Describe(int[] cuts, int n)
    {
        var groups = new List<int[]>();
        int start = 0;
        for (int c = 0; c <= cuts.Length; c++)
        {
            int end = c < cuts.Length ? cuts[c] : n;
            groups.Add(Enumerable.Range(start, end - start).ToArray());
            start = end;
        }
        return groups;
    }

    // Greedy contiguous balancer: fill a group until adding the next stage would exceed the running
    // average, then start a new group.
    public static Partition Greedy(IList<StageSpec> stages, int parts, long capacity)
    {
        int n = stages.Count;
        var result = new Partition();
        if (parts <= 0 || parts > n)
        {
            result.Feasible = false;
            result.Reason = "invalid part count";
            return result;
        }
        long total = stages.Sum(s => (long)s.Cost);
        long target = (long)Math.Ceiling(total / (double)parts);
        var groups = new List<int[]>();
        var current = new List<int>();
        long cost = 0, memory = 0;
        for (int i = 0; i < n; i++)
        {
            int groupsLeft = parts - groups.Count;
            int stagesLeft = n - i;
            bool mustCut = groupsLeft > 1 && stagesLeft <= groupsLeft;
            bool overTarget = current.Count > 0 && cost + stages[i].Cost > target && groupsLeft > 1;
            if (mustCut || overTarget)
            {
                groups.Add(current.ToArray());
                current.Clear();
                cost = 0;
                memory = 0;
            }
            current.Add(i);
            cost += stages[i].Cost;
            memory += stages[i].Memory;
            if (memory > capacity)
            {
                result.Feasible = false;
                result.Reason = "greedy produced an infeasible group at stage " + i;
                return result;
            }
        }
        if (current.Count > 0) groups.Add(current.ToArray());
        if (groups.Count != parts)
        {
            result.Feasible = false;
            result.Reason = "greedy produced " + groups.Count + " groups";
            return result;
        }
        result.Groups.AddRange(groups);
        result.MaxCost = groups.Max(g => g.Sum(i => (long)stages[i].Cost));
        return result;
    }
}

// ---- pipeline model -----------------------------------------------------------------------------------

internal sealed class MicroBatch
{
    public double Input;
    public double Target;
}

internal sealed class StageParam
{
    public double A;
    public double B;
}

internal sealed class PipelineExecutor
{
    public readonly List<StageSpec> Stages;
    public readonly List<StageParam> Parameters = new List<StageParam>();
    public readonly Partition Layout;
    public readonly List<MicroBatch> Batches;
    public readonly int Parts;

    public long[] ForwardDoneAt;      // partition x microbatch -> completion tick (0 = not done)
    public long[] BackwardDoneAt;
    public Dictionary<(int Partition, int Batch), double[]> Activations = new Dictionary<(int, int), double[]>();
    public int[] InFlight;
    public long[] FreeAt;

    public long Tick;
    public long LegalActions;
    public long IllegalActions;
    public long HeldActivationTicks;
    public long PeakHeld;
    public bool AllForwardsFirst;
    public long ForwardActions;
    public long BackwardActions;
    public double[] FinalOutputs;
    public double[] Gradients;
    public double FinalLoss;

    public PipelineExecutor(List<StageSpec> stages, List<StageParam> parameters, Partition layout,
        List<MicroBatch> batches, bool allForwardsFirst)
    {
        Stages = stages;
        Parameters = parameters;
        Layout = layout;
        Batches = batches;
        Parts = layout.GroupCount;
        AllForwardsFirst = allForwardsFirst;
        int m = batches.Count;
        ForwardDoneAt = new long[Parts * m];
        BackwardDoneAt = new long[Parts * m];
        InFlight = new int[Parts];
        FreeAt = new long[Parts];
        FinalOutputs = new double[batches.Count];
        Gradients = new double[stages.Count * 2];
    }

    private int GroupCost(int partition) => Layout.Groups[partition].Sum(i => Stages[i].Cost);
    private int GroupSize(int partition) => Layout.Groups[partition].Length;

    private int InFlightLimit(int partition) => AllForwardsFirst ? int.MaxValue : Parts - partition;

    // Readiness uses the *completion* tick: an action may only start once the tick it waited for has
    // actually arrived, which is what makes the schedule legal rather than merely ordered.
    private bool ForwardReady(int partition, int batch)
    {
        long upstream = partition == 0 ? long.MinValue : ForwardDoneAt[(partition - 1) * Batches.Count + batch];
        if (partition == 0) return true;
        return upstream != 0 && upstream <= Tick;
    }

    private bool BackwardReady(int partition, int batch)
    {
        long forward = ForwardDoneAt[partition * Batches.Count + batch];
        if (forward == 0 || forward > Tick) return false;
        if (partition == Parts - 1) return true;
        long downstreamDone = BackwardDoneAt[(partition + 1) * Batches.Count + batch];
        return downstreamDone != 0 && downstreamDone <= Tick;
    }

    private long Held()
    {
        long held = 0;
        foreach (KeyValuePair<(int Partition, int Batch), double[]> entry in Activations)
            held += GroupSize(entry.Key.Partition);
        PeakHeld = Math.Max(PeakHeld, held);
        return held;
    }

    private void DoForward(int partition, int batch, long finishTick)
    {
        double[] inputs = new double[GroupSize(partition) + 1];
        double value = partition == 0
            ? Batches[batch].Input
            : Activations[(partition - 1, batch)][GroupSize(partition - 1)];
        inputs[0] = value;
        for (int k = 0; k < GroupSize(partition); k++)
        {
            int stage = Layout.Groups[partition][k];
            value = Parameters[stage].A * value + Parameters[stage].B;
            inputs[k + 1] = value;
        }
        Activations[(partition, batch)] = inputs;
        ForwardDoneAt[partition * Batches.Count + batch] = finishTick;
        InFlight[partition]++;
        if (partition == Parts - 1) FinalOutputs[batch] = value;
    }

    private double DoBackward(int partition, int batch, long finishTick)
    {
        double[] inputs = Activations[(partition, batch)];
        double gradOut = partition == Parts - 1
            ? 2.0 * (inputs[GroupSize(partition)] - Batches[batch].Target)
            : DownstreamGradient(partition + 1, batch);
        double grad = gradOut;
        for (int k = GroupSize(partition) - 1; k >= 0; k--)
        {
            int stage = Layout.Groups[partition][k];
            double x = inputs[k];
            grad = gradOut * Parameters[stage].A;
            Gradients[stage * 2] += gradOut * x;
            Gradients[stage * 2 + 1] += gradOut;
            gradOut = grad;
        }
        Activations.Remove((partition, batch));
        BackwardDoneAt[partition * Batches.Count + batch] = finishTick;
        InFlight[partition]--;
        return grad;
    }

    private readonly Dictionary<(int, int), double> downstream = new Dictionary<(int, int), double>();

    // The gradient a partition hands upstream is stored under that partition's own index, because it is
    // the input gradient of *its* group.
    private double DownstreamGradient(int partition, int batch)
        => downstream.TryGetValue((partition, batch), out double value) ? value : 0;

    public void Run(long maxTicks)
    {
        int m = Batches.Count;
        PeakHeld = 0;
        while (!Done() && Tick < maxTicks)
        {
            Tick++;
            for (int partition = 0; partition < Parts; partition++)
            {
                if (FreeAt[partition] > Tick) continue;
                int chosenBatch = -1;
                bool chooseBackward = false;
                for (int batch = 0; batch < m; batch++)
                {
                    if (BackwardDoneAt[partition * m + batch] == 0 && BackwardReady(partition, batch))
                    {
                        chosenBatch = batch;
                        chooseBackward = true;
                        break;
                    }
                }
                if (!chooseBackward && InFlight[partition] < InFlightLimit(partition))
                {
                    for (int batch = 0; batch < m; batch++)
                    {
                        if (ForwardDoneAt[partition * m + batch] == 0 && ForwardReady(partition, batch))
                        {
                            chosenBatch = batch;
                            break;
                        }
                    }
                }
                if (chosenBatch < 0) continue;
                long finish = Tick + GroupCost(partition);
                if (chooseBackward)
                {
                    double grad = DoBackward(partition, chosenBatch, finish);
                    downstream[(partition, chosenBatch)] = grad;
                    BackwardActions++;
                }
                else
                {
                    DoForward(partition, chosenBatch, finish);
                    ForwardActions++;
                }
                FreeAt[partition] = finish;
                LegalActions++;
                CheckLegal(partition, chosenBatch, chooseBackward);
            }
            HeldActivationTicks += Held();
        }
        FinalLoss = Loss();
    }

    private void CheckLegal(int partition, int batch, bool backward)
    {
        if (backward)
        {
            if (ForwardDoneAt[partition * Batches.Count + batch] == 0) IllegalActions++;
            if (partition < Parts - 1 && BackwardDoneAt[(partition + 1) * Batches.Count + batch] == 0) IllegalActions++;
        }
        else
        {
            if (partition > 0 && ForwardDoneAt[(partition - 1) * Batches.Count + batch] == 0) IllegalActions++;
        }
    }

    private bool Done() => BackwardActions == (long)Parts * Batches.Count;

    public double Loss()
    {
        double total = 0;
        foreach (MicroBatch batch in Batches)
        {
            double value = batch.Input;
            for (int stage = 0; stage < Stages.Count; stage++) value = Parameters[stage].A * value + Parameters[stage].B;
            total += (value - batch.Target) * (value - batch.Target);
        }
        return total;
    }

    public long Makespan => FreeAt.Length == 0 ? 0 : FreeAt.Max();
}

// ---- checks -------------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xB1A5EUL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        // ---- partitioner against exhaustive enumeration ----------------------------------------
        int compared = 0;
        double worstGreedyRatio = 1.0;
        int greedyStrictlyWorse = 0;
        for (int trial = 0; trial < 60; trial++)
        {
            int stageCount = 4 + trial % 5;
            int partCount = 1 + trial % 4;
            if (partCount > stageCount) continue;
            var specs = new List<StageSpec>();
            for (int i = 0; i < stageCount; i++) specs.Add(new StageSpec(1 + Rand(20), 1 + Rand(6)));
            long capacity = specs.Sum(s => (long)s.Memory);

            Partition dp = Partitioner.Dynamic(specs, partCount, capacity);
            Partition brute = Partitioner.BruteForce(specs, partCount, capacity);
            Check(dp.Feasible && brute.Feasible, "both partitioners found the instance feasible");
            Check(dp.MaxCost == brute.MaxCost, $"DP optimum {dp.MaxCost} equals brute force {brute.MaxCost}");
            Check(dp.GroupCount == partCount, "the DP produced exactly P groups");
            Check(Flatten(dp).SequenceEqual(Enumerable.Range(0, stageCount)), "groups cover the chain exactly once");
            Check(dp.Groups.All(g => g.Sum(i => (long)specs[i].Memory) <= capacity), "every group fits the capacity");

            Partition greedy = Partitioner.Greedy(specs, partCount, capacity);
            if (greedy.Feasible)
            {
                Check(greedy.MaxCost >= dp.MaxCost, "a heuristic never beats the optimum");
                worstGreedyRatio = Math.Max(worstGreedyRatio, greedy.MaxCost / (double)dp.MaxCost);
                if (greedy.MaxCost > dp.MaxCost) greedyStrictlyWorse++;
            }
            compared++;
        }
        Check(compared >= 40, "compared enough partition instances: " + compared);
        // Equal stages: every balancer should find the optimum.
        for (int partCount = 1; partCount <= 6; partCount++)
        {
            var equalSpecs = Enumerable.Range(0, 12).Select(_ => new StageSpec(5, 1)).ToList();
            Partition evenSplit = Partitioner.Dynamic(equalSpecs, partCount, 100);
            Check(evenSplit.MaxCost == 5L * ((12 + partCount - 1) / partCount),
                "equal stages split evenly into " + partCount + " parts: " + evenSplit.MaxCost);
        }
        // Memory feasibility is a hard constraint: shrink the capacity until the instance is infeasible.
        var tight = new List<StageSpec> { new StageSpec(4, 5), new StageSpec(4, 5), new StageSpec(4, 5), new StageSpec(4, 5) };
        Check(Partitioner.Dynamic(tight, 4, 5).Feasible, "each stage alone fits a tight capacity");
        Check(!Partitioner.Dynamic(tight, 2, 5).Feasible, "merging two stages breaks the capacity");
        Check(!Partitioner.Dynamic(tight, 3, 5).Feasible, "no feasible 3 way split exists");
        Check(Partitioner.Dynamic(tight, 4, 5).Groups.Count(g => g.Length == 1) == 4, "feasible split is one stage per part");
        // Degenerate part counts.
        Check(!Partitioner.Dynamic(tight, 5, 100).Feasible, "more parts than stages is infeasible");
        Check(!Partitioner.Dynamic(tight, 0, 100).Feasible, "zero parts is infeasible");
        Check(Partitioner.Dynamic(tight, 1, 20).MaxCost == 16, "one part holds the whole chain");
        Check(!Partitioner.Dynamic(tight, 1, 19).Feasible, "the whole chain must fit the capacity");

        // ---- pipeline correctness against the sequential oracle --------------------------------
        var layout = BuildLayout();
        var parameters = BuildParameters(layout.Stages.Count);
        var batches = BuildBatches(8);
        var executor = new PipelineExecutor(layout.Stages, Clone(parameters), layout.Partition, batches, false);
        executor.Run(10000);
        Check(executor.IllegalActions == 0, "the 1F1B schedule never violated a dependency");
        Check(executor.ForwardActions == (long)layout.Partition.GroupCount * batches.Count
            && executor.BackwardActions == (long)layout.Partition.GroupCount * batches.Count,
            "every microbatch was forwarded and backwarded once per partition");

        var sequentialOutputs = new List<double>();
        foreach (MicroBatch batch in batches)
        {
            double value = batch.Input;
            for (int stage = 0; stage < layout.Stages.Count; stage++)
                value = parameters[stage].A * value + parameters[stage].B;
            sequentialOutputs.Add(value);
        }
        Check(executor.FinalOutputs.SequenceEqual(sequentialOutputs),
            "the pipelined forward equals a purely sequential execution of the same chain");
        var repeatRun = new PipelineExecutor(layout.Stages, Clone(parameters), layout.Partition, batches, false);
        repeatRun.Run(10000);
        Check(Math.Abs(repeatRun.Loss() - executor.Loss()) < 1e-12 && repeatRun.Makespan == executor.Makespan
            && repeatRun.FinalOutputs.SequenceEqual(executor.FinalOutputs), "the pipeline is deterministic");

        // ---- gradients: analytic, finite difference and pipelined must agree -------------------
        double[] pipelined = executor.Gradients;
        double[] analytic = AnalyticGradient(parameters, batches);
        Check(pipelined.Length == analytic.Length, "one gradient pair per stage");
        for (int i = 0; i < analytic.Length; i++)
            Check(Math.Abs(pipelined[i] - analytic[i]) <= 1e-9 * Math.Max(1.0, Math.Abs(analytic[i])),
                $"pipelined gradient [{i}] matches the analytic one: {pipelined[i]:F9} vs {analytic[i]:F9}");
        Check(pipelined.Any(g => Math.Abs(g) > 1e-6), "the gradient is not trivially zero");
        double worstRelative = 0;
        for (int stage = 0; stage < parameters.Count; stage++)
        {
            worstRelative = Math.Max(worstRelative, RelativeFiniteDifference(parameters, batches, stage, true));
            worstRelative = Math.Max(worstRelative, RelativeFiniteDifference(parameters, batches, stage, false));
        }
        Check(worstRelative < 1e-5, "finite difference agrees with the analytic gradient: " + worstRelative.ToString("E2"));

        // ---- efficiency: makespan bounds, bubbles, speedup and memory --------------------------
        var equalLayout = EqualLayout(8, 4, 3);
        var equalParams = BuildParameters(8);
        var equalBatches = BuildBatches(8);
        var equal = new PipelineExecutor(equalLayout.Stages, Clone(equalParams), equalLayout.Partition, equalBatches, false);
        equal.Run(100000);
        int parts = equalLayout.Partition.GroupCount;
        int microbatchCount = equalBatches.Count;
        long stageCost = equalLayout.Partition.MaxCost;
        long criticalPath = 2L * microbatchCount * equalLayout.Partition.MaxCost;
        long periodBound = 2L * (microbatchCount + parts - 1) * equalLayout.Partition.MaxCost;
        Check(equal.Makespan >= criticalPath, $"makespan {equal.Makespan} at least the critical path {criticalPath}");
        Check(equal.Makespan <= periodBound + equalLayout.Partition.MaxCost,
            $"makespan {equal.Makespan} near the 1F1B period bound {periodBound}");
        long serial = 2L * microbatchCount * parts * equalLayout.Partition.MaxCost;
        double speedup = serial / (double)equal.Makespan;
        Check(speedup > 1.5, "pipeline speedup over serial execution: " + speedup.ToString("F3"));
        Check(speedup <= parts, "speedup cannot exceed the number of partitions");
        double bubble = 1.0 - criticalPath / (double)equal.Makespan;
        Check(bubble < (parts - 1.0) / microbatchCount, "bubble fraction stays below (P-1)/M: " + bubble.ToString("F4"));

        var greedySchedule = new PipelineExecutor(equalLayout.Stages, Clone(equalParams), equalLayout.Partition, equalBatches, true);
        greedySchedule.Run(100000);
        Check(greedySchedule.Makespan >= equal.Makespan, "all forwards first is never faster than 1F1B");
        Check(greedySchedule.PeakHeld > equal.PeakHeld, $"1F1B uses less activation memory: {equal.PeakHeld} vs {greedySchedule.PeakHeld}");

        // ---- microbatch count smaller than the pipeline depth ----------------------------------
        var fewBatches = BuildBatches(2);
        var small = new PipelineExecutor(equalLayout.Stages, Clone(equalParams), equalLayout.Partition, fewBatches, false);
        small.Run(100000);
        Check(small.IllegalActions == 0 && small.ForwardActions == (long)parts * 2, "two microbatches through four partitions");
        Check(small.Makespan >= 2L * 2 * equalLayout.Partition.MaxCost, "short pipelines are bubble dominated");
        double shortBubble = 1.0 - 2L * 2 * equalLayout.Partition.MaxCost / (double)small.Makespan;
        Check(shortBubble > bubble, "fewer microbatches means a bigger bubble fraction");
        var single = new PipelineExecutor(equalLayout.Stages, Clone(equalParams), equalLayout.Partition, BuildBatches(1), false);
        single.Run(100000);
        Check(single.IllegalActions == 0 && single.ForwardActions == parts && single.BackwardActions == parts,
            "a single microbatch still traverses every partition exactly once");

        // ---- one partition, and the all forwards first comparison -------------------------------
        var singleLayout = EqualLayout(8, 1, 3);
        var singlePart = new PipelineExecutor(singleLayout.Stages, Clone(singleLayout.Parameters), singleLayout.Partition, equalBatches, false);
        singlePart.Run(100000);
        Check(singlePart.IllegalActions == 0, "a single partition has no cross partition dependency");
        Check(singlePart.Makespan <= 2L * microbatchCount * singlePart.Layout.MaxCost + singlePart.Layout.MaxCost,
            "one partition has no bubbles: " + singlePart.Makespan
            + " <= " + (2L * microbatchCount * singlePart.Layout.MaxCost) + " + cost");
        Check(singlePart.PeakHeld <= 8, "one partition holds at most one microbatch's activations at a time");

        // ---- balanced versus unbalanced partition ----------------------------------------------
        var lopsided = BuildLopsidedLayout();
        var balancedRun = new PipelineExecutor(layout.Stages, Clone(parameters), layout.Partition, batches, false);
        balancedRun.Run(100000);
        var lopsidedRun = new PipelineExecutor(lopsided.Stages, Clone(lopsided.Parameters), lopsided.Partition, batches, false);
        lopsidedRun.Run(100000);
        Check(balancedRun.Makespan <= lopsidedRun.Makespan,
            $"the balanced partition is no slower: {balancedRun.Makespan} vs {lopsidedRun.Makespan}");

        // ---- a gradient step must lower the loss ------------------------------------------------
        var trained = Clone(parameters);
        var stepExecutor = new PipelineExecutor(layout.Stages, trained, layout.Partition, batches, false);
        double lossBefore = stepExecutor.Loss();
        stepExecutor.Run(10000);
        for (int stage = 0; stage < trained.Count; stage++)
        {
            trained[stage].A -= 1e-4 * stepExecutor.Gradients[stage * 2];
            trained[stage].B -= 1e-4 * stepExecutor.Gradients[stage * 2 + 1];
        }
        var afterStep = new PipelineExecutor(layout.Stages, trained, layout.Partition, batches, false);
        double lossAfter = afterStep.Loss();
        Check(lossAfter < lossBefore, $"a pipeline gradient step lowers the loss: {lossBefore:F6} -> {lossAfter:F6}");

        Console.WriteLine($"818 partitioner: DP optimum equals brute force on {compared} instances (worst greedy ratio "
            + $"{worstGreedyRatio:F3}, greedy strictly worse on {greedyStrictlyWorse}), {parts} partitions x {microbatchCount} "
            + $"microbatches -> makespan {equal.Makespan} (critical path {criticalPath}, period bound {periodBound}), "
            + $"speedup {speedup:F3}x, bubble {bubble:F4}, peak activations {equal.PeakHeld} vs {greedySchedule.PeakHeld} "
            + $"all-forwards-first, gradient vs finite difference {worstRelative:E2}");
        Console.WriteLine("PASS");
    }

    private static int[] Flatten(Partition partition) => partition.Groups.SelectMany(g => g).ToArray();

    private static double[] AnalyticGradient(List<StageParam> parameters, List<MicroBatch> batches)
    {
        var gradient = new double[parameters.Count * 2];
        foreach (MicroBatch batch in batches)
        {
            var values = new double[parameters.Count + 1];
            values[0] = batch.Input;
            for (int stage = 0; stage < parameters.Count; stage++)
                values[stage + 1] = parameters[stage].A * values[stage] + parameters[stage].B;
            double grad = 2.0 * (values[parameters.Count] - batch.Target);
            for (int stage = parameters.Count - 1; stage >= 0; stage--)
            {
                gradient[stage * 2] += grad * values[stage];
                gradient[stage * 2 + 1] += grad;
                grad *= parameters[stage].A;
            }
        }
        return gradient;
    }

    private static double RelativeFiniteDifference(List<StageParam> parameters, List<MicroBatch> batches, int stage, bool onA)
    {
        double epsilon = 1e-6;
        double[] analytic = AnalyticGradient(parameters, batches);
        double Loss()
        {
            double total = 0;
            foreach (MicroBatch batch in batches)
            {
                double value = batch.Input;
                for (int s = 0; s < parameters.Count; s++) value = parameters[s].A * value + parameters[s].B;
                total += (value - batch.Target) * (value - batch.Target);
            }
            return total;
        }
        double original = onA ? parameters[stage].A : parameters[stage].B;
        if (onA) parameters[stage].A = original + epsilon; else parameters[stage].B = original + epsilon;
        double plus = Loss();
        if (onA) parameters[stage].A = original - epsilon; else parameters[stage].B = original - epsilon;
        double minus = Loss();
        if (onA) parameters[stage].A = original; else parameters[stage].B = original;
        double numeric = (plus - minus) / (2 * epsilon);
        double expected = analytic[stage * 2 + (onA ? 0 : 1)];
        return Math.Abs(numeric - expected) / Math.Max(1.0, Math.Abs(expected));
    }

    private static List<StageParam> BuildParameters(int count)
    {
        var parameters = new List<StageParam>();
        for (int i = 0; i < count; i++)
            parameters.Add(new StageParam { A = 0.6 + 0.05 * i, B = 0.1 * (i + 1) });
        return parameters;
    }

    private static List<StageParam> Clone(List<StageParam> parameters)
        => parameters.Select(p => new StageParam { A = p.A, B = p.B }).ToList();

    private static List<MicroBatch> BuildBatches(int count)
    {
        var batches = new List<MicroBatch>();
        for (int i = 0; i < count; i++)
            batches.Add(new MicroBatch { Input = 0.5 + 0.25 * i, Target = 1.0 + 0.1 * i });
        return batches;
    }

    private sealed class Layout
    {
        public List<StageSpec> Stages = new List<StageSpec>();
        public List<StageParam> Parameters = new List<StageParam>();
        public Partition Partition = new Partition();
    }

    private static Layout BuildLayout()
    {
        var stages = new List<StageSpec>
        {
            new StageSpec(3, 2), new StageSpec(2, 1), new StageSpec(5, 3), new StageSpec(2, 2),
            new StageSpec(4, 1), new StageSpec(3, 2)
        };
        var layout = new Layout
        {
            Stages = stages,
            Parameters = BuildParameters(stages.Count),
            Partition = Partitioner.Dynamic(stages, 3, stages.Sum(s => (long)s.Memory))
        };
        Check(layout.Partition.Feasible, "the reference layout is feasible");
        return layout;
    }

    private static Layout BuildLopsidedLayout()
    {
        var stages = new List<StageSpec>
        {
            new StageSpec(3, 2), new StageSpec(2, 1), new StageSpec(5, 3), new StageSpec(2, 2),
            new StageSpec(4, 1), new StageSpec(3, 2)
        };
        var partition = new Partition { MaxCost = 19 };
        partition.Groups.Add(new[] { 0 });
        partition.Groups.Add(new[] { 1, 2, 3, 4 });
        partition.Groups.Add(new[] { 5 });
        return new Layout { Stages = stages, Parameters = BuildParameters(stages.Count), Partition = partition };
    }

    private static Layout EqualLayout(int stageCount, int parts, int cost)
    {
        var stages = Enumerable.Range(0, stageCount).Select(_ => new StageSpec(cost, 1)).ToList();
        return new Layout
        {
            Stages = stages,
            Parameters = BuildParameters(stageCount),
            Partition = Partitioner.Dynamic(stages, parts, stages.Sum(s => (long)s.Memory))
        };
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
