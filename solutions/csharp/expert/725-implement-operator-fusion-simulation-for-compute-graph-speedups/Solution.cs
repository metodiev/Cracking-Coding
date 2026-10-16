// Task 725 - Implement operator fusion simulation for compute graph speedups. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// In a tensor graph every "elementwise" op (add, mul, relu, sigmoid) reads each input and writes each
// output through memory. A chain of elementwise ops can be fused into a single kernel that streams over
// the data once: the intermediates never touch memory, they live in registers. This file builds the
// graph, partitions it into fusion groups (the connected components of the elementwise-only subgraph,
// so a fan-out stays in one kernel; every shape-changing op - dot, reduce - is its own group and always
// breaks a chain), evaluates it both ways and compares:
//
//   traffic(unfused) = sum over ops of  (elements read from memory + elements written)
//   traffic(fused)   = sum over groups of (external elements read + materialised elements written)
//
// Documented simplifications: tensors are flat vectors of doubles (plus scalars for dot/reduce), so
// there is no broadcasting, tiling or layout choice, and the model counts elements moved rather than
// cycles. The fused evaluator is a simulation: it walks each element through the group's ops in
// topological order keeping one register per intermediate, exactly what a fused kernel would compute.
// That makes the strongest available check possible: the fused result must equal the unfused result
// bit-for-bit on every graph, because the per-element operation order is identical. Register pressure
// is measured separately as the peak number of intermediates alive at once inside a group.
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

    internal double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class OpNode
{
    internal OpNode(string kind, int[] inputs, int output)
    {
        Kind = kind;
        Inputs = inputs;
        Output = output;
    }

    internal string Kind { get; }
    internal int[] Inputs { get; }
    internal int Output { get; }

    internal bool Elementwise => Kind == "add" || Kind == "mul" || Kind == "relu" || Kind == "sigmoid";
}

internal sealed class Graph
{
    internal readonly List<OpNode> Ops = new List<OpNode>();
    internal readonly List<int> TensorElements = new List<int>();

    internal int AddTensor(int elements)
    {
        TensorElements.Add(elements);
        return TensorElements.Count - 1;
    }

    /// <summary>Ops are stored in topological order, so every input must precede the new output.</summary>
    internal void AddOp(string kind, int output, params int[] inputs)
    {
        foreach (int input in inputs)
            if (input >= output) throw new ArgumentException("ops must be listed in topological order");
        Ops.Add(new OpNode(kind, inputs, output));
    }

    internal bool IsGraphOutput(int tensor)
    {
        foreach (OpNode op in Ops) foreach (int input in op.Inputs) if (input == tensor) return false;
        return true;
    }
}

internal sealed class FusionGroup
{
    internal readonly List<int> OpIndices = new List<int>();
    internal readonly List<int> Inputs = new List<int>();
    internal readonly List<int> Outputs = new List<int>();
    internal int PeakRegisters;

    internal bool IsFused => OpIndices.Count > 1;
}

internal static class Fusion
{
    internal static List<FusionGroup> Partition(Graph graph)
    {
        int n = graph.Ops.Count;
        var producer = new Dictionary<int, int>();
        for (int i = 0; i < n; i++) producer[graph.Ops[i].Output] = i;

        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int Find(int x)
        {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        }
        for (int i = 0; i < n; i++)
        {
            if (!graph.Ops[i].Elementwise) continue;
            foreach (int input in graph.Ops[i].Inputs)
            {
                if (!producer.TryGetValue(input, out int source) || !graph.Ops[source].Elementwise) continue;
                int a = Find(i), b = Find(source);
                if (a != b) parent[a] = b;
            }
        }

        var groups = new List<FusionGroup>();
        var byRoot = new Dictionary<int, FusionGroup>();
        for (int i = 0; i < n; i++)
        {
            if (!graph.Ops[i].Elementwise)
            {
                var solo = new FusionGroup();
                solo.OpIndices.Add(i);
                groups.Add(solo);
                continue;
            }
            int root = Find(i);
            if (!byRoot.TryGetValue(root, out FusionGroup? group))
            {
                group = new FusionGroup();
                byRoot[root] = group;
                groups.Add(group);
            }
            group.OpIndices.Add(i);
        }

        foreach (FusionGroup group in groups)
        {
            var inside = new HashSet<int>();
            foreach (int opIndex in group.OpIndices) inside.Add(graph.Ops[opIndex].Output);
            foreach (int opIndex in group.OpIndices)
            {
                foreach (int input in graph.Ops[opIndex].Inputs)
                    if (!inside.Contains(input) && !group.Inputs.Contains(input)) group.Inputs.Add(input);
            }
            foreach (int opIndex in group.OpIndices)
            {
                int output = graph.Ops[opIndex].Output;
                bool exported = graph.IsGraphOutput(output);
                for (int j = 0; j < graph.Ops.Count && !exported; j++)
                {
                    if (group.OpIndices.Contains(j)) continue;      // a consumer inside the group is not a boundary
                    foreach (int input in graph.Ops[j].Inputs) if (input == output) { exported = true; break; }
                }
                if (exported && !group.Outputs.Contains(output)) group.Outputs.Add(output);
            }
            group.PeakRegisters = RegisterPressure(graph, group);
        }
        groups.Sort((a, b) => a.OpIndices[0].CompareTo(b.OpIndices[0]));
        return groups;
    }

    /// <summary>
    /// Peak number of internal values alive at once. A value is live from the op that produces it until
    /// (but not including) the last op inside the group that consumes it; materialised group outputs go
    /// to memory and do not occupy a register. A chain therefore needs one register where a fan-out needs
    /// as many as the widest simultaneously reachable set of intermediates.
    /// </summary>
    private static int RegisterPressure(Graph graph, FusionGroup group)
    {
        var inside = new HashSet<int>();
        foreach (int opIndex in group.OpIndices) inside.Add(graph.Ops[opIndex].Output);
        foreach (int output in group.Outputs) inside.Remove(output);
        var birth = new Dictionary<int, int>();
        var death = new Dictionary<int, int>();
        for (int position = 0; position < group.OpIndices.Count; position++)
        {
            int output = graph.Ops[group.OpIndices[position]].Output;
            if (!inside.Contains(output)) continue;
            birth[output] = position;
            death[output] = position;
        }
        for (int position = 0; position < group.OpIndices.Count; position++)
            foreach (int input in graph.Ops[group.OpIndices[position]].Inputs)
                if (death.ContainsKey(input)) death[input] = Math.Max(death[input], position);

        int peak = 0;
        for (int position = 0; position < group.OpIndices.Count; position++)
        {
            int live = 0;
            foreach (KeyValuePair<int, int> entry in birth)
                if (entry.Value <= position && position < death[entry.Key]) live++;
            peak = Math.Max(peak, live);
        }
        return peak;
    }

    internal static long UnfusedTraffic(Graph graph)
    {
        long bytes = 0;
        foreach (OpNode op in graph.Ops)
        {
            foreach (int input in op.Inputs) bytes += graph.TensorElements[input] * 8L;
            bytes += graph.TensorElements[op.Output] * 8L;
        }
        return bytes;
    }

    internal static long FusedTraffic(Graph graph, List<FusionGroup> groups)
    {
        long bytes = 0;
        foreach (FusionGroup group in groups)
        {
            foreach (int input in group.Inputs) bytes += graph.TensorElements[input] * 8L;
            foreach (int output in group.Outputs) bytes += graph.TensorElements[output] * 8L;
        }
        return bytes;
    }

    internal static Dictionary<int, double[]> EvaluateUnfused(Graph graph, Dictionary<int, double[]> values)
    {
        var result = new Dictionary<int, double[]>(values);
        foreach (OpNode op in graph.Ops) result[op.Output] = Apply(op, result, graph);
        return result;
    }

    /// <summary>Streams every element through a group's ops in order, intermediates held in registers.</summary>
    internal static Dictionary<int, double[]> EvaluateFused(Graph graph, List<FusionGroup> groups, Dictionary<int, double[]> values)
    {
        var result = new Dictionary<int, double[]>(values);
        foreach (FusionGroup group in groups)
        {
            int leading = group.OpIndices[0];
            if (!graph.Ops[leading].Elementwise)
            {
                OpNode only = graph.Ops[leading];
                result[only.Output] = Apply(only, result, graph);
                continue;
            }
            int width = graph.TensorElements[graph.Ops[leading].Output];
            var registers = new double[group.OpIndices.Count][];
            for (int r = 0; r < registers.Length; r++) registers[r] = new double[width];
            for (int element = 0; element < width; element++)
            {
                for (int position = 0; position < group.OpIndices.Count; position++)
                {
                    OpNode op = graph.Ops[group.OpIndices[position]];
                    double first = Slot(registers, group, result, graph, position, op.Inputs[0], element);
                    switch (op.Kind)
                    {
                        case "add":
                            registers[position][element] = first + Slot(registers, group, result, graph, position, op.Inputs[1], element);
                            break;
                        case "mul":
                            registers[position][element] = first * Slot(registers, group, result, graph, position, op.Inputs[1], element);
                            break;
                        case "relu":
                            registers[position][element] = first > 0.0 ? first : 0.0;
                            break;
                        default:
                            registers[position][element] = 1.0 / (1.0 + Math.Exp(-first));
                            break;
                    }
                }
            }
            for (int position = 0; position < group.OpIndices.Count; position++)
            {
                int output = graph.Ops[group.OpIndices[position]].Output;
                if (group.Outputs.Contains(output)) result[output] = registers[position];
            }
        }
        return result;
    }

    /// <summary>Reads an input of the op at <paramref name="position"/>: a register if produced inside, else memory.</summary>
    private static double Slot(double[][] registers, FusionGroup group, Dictionary<int, double[]> result, Graph graph,
        int position, int tensor, int element)
    {
        for (int p = 0; p < position; p++)
            if (graph.Ops[group.OpIndices[p]].Output == tensor) return registers[p][element];
        return result[tensor][element];
    }

    internal static double[] Apply(OpNode op, Dictionary<int, double[]> env, Graph graph)
    {
        var output = new double[graph.TensorElements[op.Output]];
        switch (op.Kind)
        {
            case "add": for (int i = 0; i < output.Length; i++) output[i] = env[op.Inputs[0]][i] + env[op.Inputs[1]][i]; break;
            case "mul": for (int i = 0; i < output.Length; i++) output[i] = env[op.Inputs[0]][i] * env[op.Inputs[1]][i]; break;
            case "relu":
                for (int i = 0; i < output.Length; i++) output[i] = env[op.Inputs[0]][i] > 0.0 ? env[op.Inputs[0]][i] : 0.0;
                break;
            case "sigmoid":
                for (int i = 0; i < output.Length; i++) output[i] = 1.0 / (1.0 + Math.Exp(-env[op.Inputs[0]][i]));
                break;
            case "dot":
                {
                    double sum = 0.0;
                    for (int i = 0; i < graph.TensorElements[op.Inputs[0]]; i++) sum += env[op.Inputs[0]][i] * env[op.Inputs[1]][i];
                    output[0] = sum;
                    break;
                }
            case "reduce":
                {
                    double sum = 0.0;
                    for (int i = 0; i < graph.TensorElements[op.Inputs[0]]; i++) sum += env[op.Inputs[0]][i];
                    output[0] = sum;
                    break;
                }
            default: throw new ArgumentException("unknown op " + op.Kind);
        }
        return output;
    }
}

public static class Solution
{
    public static void Main()
    {
        const int n = 64;

        // ---- an eight-op unary chain: one group, one read and one write ----------------------------
        {
            Graph graph = UnaryChain(8, n);
            List<FusionGroup> groups = Fusion.Partition(graph);
            long unfused = Fusion.UnfusedTraffic(graph);
            long fused = Fusion.FusedTraffic(graph, groups);
            Console.WriteLine("chain: groups=" + groups.Count + " unfusedBytes=" + unfused + " fusedBytes=" + fused
                + " speedup=" + ((double)unfused / fused).ToString("G4") + " registers=" + groups[0].PeakRegisters);
            Check(groups.Count == 1 && groups[0].OpIndices.Count == 8, "a unary elementwise chain fuses into one group");
            Check(groups[0].PeakRegisters == 1, "a chain needs one register, not eight tensors (" + groups[0].PeakRegisters + ")");
            Check(groups[0].Inputs.Count == 1 && groups[0].Outputs.Count == 1,
                "only the head input and the tail output cross the boundary");
            Check(fused == 2L * n * 8, "the fused kernel moves one input and one output (" + fused + " bytes)");
            Check(unfused == 2L * 8 * n * 8, "the unfused version touches a tensor per op");
            Check(unfused / fused >= 5, "fusion removes most of the traffic (" + unfused + " -> " + fused + ")");
            Check(CountOps(groups) == graph.Ops.Count, "the partition covers every op exactly once");
            Dictionary<int, double[]> inputs = Inputs(graph, 0x725644AUL);
            Check(AllEqual(graph, Fusion.EvaluateUnfused(graph, inputs), Fusion.EvaluateFused(graph, groups, inputs)),
                "the fused chain equals the unfused chain bit-for-bit");
            Check(Materialised(graph, groups, inputs).Count == 2,
                "the fused kernel keeps exactly the input and the output in memory");
        }

        // ---- fan-out inside a group: two consumers, still one kernel -------------------------------
        {
            var graph = new Graph();
            int a = graph.AddTensor(n), c = graph.AddTensor(n), d = graph.AddTensor(n);
            int x = graph.AddTensor(n), b1 = graph.AddTensor(n), b2 = graph.AddTensor(n), outT = graph.AddTensor(n);
            graph.AddOp("relu", x, a);
            graph.AddOp("mul", b1, x, c);
            graph.AddOp("add", b2, x, d);
            graph.AddOp("add", outT, b1, b2);
            List<FusionGroup> groups = Fusion.Partition(graph);
            Check(groups.Count == 1, "a fan-out of elementwise ops is still a single kernel");
            Check(groups[0].PeakRegisters == 2, "the fan-out keeps two intermediates alive between ops ("
                + groups[0].PeakRegisters + ") where a chain needs one");
            Check(groups[0].Inputs.Count == 3, "the group reads three external tensors");
            Check(groups[0].Outputs.Count == 1, "the group materialises only its output");
            Check(Fusion.FusedTraffic(graph, groups) == 4L * n * 8, "four tensors cross the group boundary");
            Check(Fusion.FusedTraffic(graph, groups) * 2 < Fusion.UnfusedTraffic(graph), "the fan-out still pays off");
            Dictionary<int, double[]> inputs = Inputs(graph, 0x725FA0UL);
            Check(AllEqual(graph, Fusion.EvaluateUnfused(graph, inputs), Fusion.EvaluateFused(graph, groups, inputs)),
                "the fan-out evaluates identically fused and unfused");
        }

        // ---- a shape-changing op always breaks the chain --------------------------------------------
        {
            var graph = new Graph();
            int a = graph.AddTensor(n), b = graph.AddTensor(n), c = graph.AddTensor(n);
            int x = graph.AddTensor(n), scalar = graph.AddTensor(1), y = graph.AddTensor(n);
            int relu = graph.AddTensor(n), sum = graph.AddTensor(1);
            graph.AddOp("sigmoid", x, a);
            graph.AddOp("dot", scalar, x, b);
            graph.AddOp("mul", y, x, c);
            graph.AddOp("relu", relu, y);
            graph.AddOp("reduce", sum, relu);
            List<FusionGroup> groups = Fusion.Partition(graph);
            long unfused = Fusion.UnfusedTraffic(graph);
            long fused = Fusion.FusedTraffic(graph, groups);
            Console.WriteLine("boundaries: groups=" + groups.Count + " unfusedBytes=" + unfused + " fusedBytes=" + fused);
            Check(groups.Count == 3, "the dot and the reduce split the graph into three kernels (" + groups.Count + ")");
            bool boundariesSolo = true, elementwiseChain = false;
            foreach (FusionGroup group in groups)
            {
                string kind = graph.Ops[group.OpIndices[0]].Kind;
                if ((kind == "dot" || kind == "reduce") && group.OpIndices.Count != 1) boundariesSolo = false;
                if (kind == "sigmoid") elementwiseChain = group.OpIndices.Count == 3;
            }
            Check(boundariesSolo, "shape-changing ops are never fused with anything else");
            Check(elementwiseChain, "the three elementwise ops around the dot stay in one kernel");
            Check(fused < unfused, "fusion reduces traffic across the split (" + fused + " < " + unfused + ")");
            Check(fused == 8L * (4L * n + 2L * n + 1 + n + 1),
                "the split traffic is exactly the three kernels' boundary traffic");
            Check(unfused == 8L * (2L * n + 2L * n + 1 + 3L * n + 2L * n + n + 1),
                "the unfused traffic is one read and one write per op");
            Dictionary<int, double[]> inputs = Inputs(graph, 0x7255A17UL);
            Check(AllEqual(graph, Fusion.EvaluateUnfused(graph, inputs), Fusion.EvaluateFused(graph, groups, inputs)),
                "the split graph evaluates identically");
        }

        // ---- a hand-computed two-op example, independent of both evaluators ------------------------
        {
            var graph = new Graph();
            int a = graph.AddTensor(3), b = graph.AddTensor(3);
            int t1 = graph.AddTensor(3), outT = graph.AddTensor(3);
            graph.AddOp("add", t1, a, b);
            graph.AddOp("sigmoid", outT, t1);
            var inputs = new Dictionary<int, double[]>
            {
                [a] = new[] { -2.0, 0.0, 3.0 },
                [b] = new[] { 1.0, 0.5, -1.0 },
            };
            List<FusionGroup> groups = Fusion.Partition(graph);
            Dictionary<int, double[]> fused = Fusion.EvaluateFused(graph, groups, inputs);
            var expected = new double[3];
            for (int i = 0; i < 3; i++) expected[i] = 1.0 / (1.0 + Math.Exp(-(inputs[a][i] + inputs[b][i])));
            Check(Same(expected, fused[outT]), "a hand-computed add + sigmoid matches the fused kernel exactly");
            Check(Fusion.UnfusedTraffic(graph) == 3L * 8 * 5, "the unfused traffic follows the op-by-op formula");
            Check(Fusion.FusedTraffic(graph, groups) == 3L * 8 * 3, "the fused traffic counts three tensors");
        }

        // ---- degenerate graphs ----------------------------------------------------------------------
        {
            var graph = new Graph();
            int a = graph.AddTensor(n), b = graph.AddTensor(n), outT = graph.AddTensor(1);
            graph.AddOp("dot", outT, a, b);
            List<FusionGroup> groups = Fusion.Partition(graph);
            Check(groups.Count == 1 && groups[0].OpIndices.Count == 1 && !groups[0].IsFused,
                "a single op is a group of one and is not a fused kernel");
            Check(Fusion.FusedTraffic(graph, groups) == Fusion.UnfusedTraffic(graph),
                "with nothing to fuse the traffic is unchanged");

            var empty = new Graph();
            empty.AddTensor(n);
            Check(Fusion.Partition(empty).Count == 0 && Fusion.UnfusedTraffic(empty) == 0, "an empty graph costs nothing");
            Check(Inputs(empty, 1UL).Count == 1, "an empty graph still has its input");
            Check(Fusion.EvaluateFused(empty, Fusion.Partition(empty), Inputs(empty, 1UL)).Count == 1,
                "evaluating an empty graph returns its inputs");

            bool threw = false;
            try { new Graph().AddOp("add", 0, 0); } catch (ArgumentException) { threw = true; }
            Check(threw, "a non-topological op is rejected at construction");
        }

        // ---- reproducibility -------------------------------------------------------------------------
        {
            Graph graph = UnaryChain(6, n);
            List<FusionGroup> first = Fusion.Partition(graph);
            List<FusionGroup> second = Fusion.Partition(graph);
            bool sameGroups = first.Count == second.Count;
            for (int i = 0; i < first.Count && sameGroups; i++)
            {
                if (first[i].OpIndices.Count != second[i].OpIndices.Count) sameGroups = false;
                else for (int k = 0; k < first[i].OpIndices.Count; k++)
                        if (first[i].OpIndices[k] != second[i].OpIndices[k]) sameGroups = false;
            }
            Check(sameGroups, "the partition is reproducible");
            Dictionary<int, double[]> inputs = Inputs(graph, 0x7256EEDUL);
            int last = graph.Ops[graph.Ops.Count - 1].Output;
            Check(Same(Fusion.EvaluateUnfused(graph, inputs)[last], Fusion.EvaluateFused(graph, first, inputs)[last]),
                "a repeated evaluation is bit-identical");
        }

        Console.WriteLine("PASS");
    }

    private static Graph UnaryChain(int length, int width)
    {
        var graph = new Graph();
        int previous = graph.AddTensor(width);
        for (int i = 0; i < length; i++)
        {
            int output = graph.AddTensor(width);
            graph.AddOp(i % 2 == 0 ? "relu" : "sigmoid", output, previous);
            previous = output;
        }
        return graph;
    }

    private static Dictionary<int, double[]> Inputs(Graph graph, ulong seed)
    {
        var rng = new SplitMix64(seed);
        var produced = new HashSet<int>();
        foreach (OpNode op in graph.Ops) produced.Add(op.Output);
        var values = new Dictionary<int, double[]>();
        for (int tensor = 0; tensor < graph.TensorElements.Count; tensor++)
        {
            if (produced.Contains(tensor)) continue;
            var data = new double[graph.TensorElements[tensor]];
            for (int i = 0; i < data.Length; i++) data[i] = 2.0 * rng.NextUnit() - 1.0;
            values[tensor] = data;
        }
        return values;
    }

    private static int CountOps(List<FusionGroup> groups)
    {
        int total = 0;
        foreach (FusionGroup group in groups) total += group.OpIndices.Count;
        return total;
    }

    /// <summary>Tensors the fused evaluation actually holds in memory: the inputs plus the group outputs.</summary>
    private static Dictionary<int, double[]> Materialised(Graph graph, List<FusionGroup> groups, Dictionary<int, double[]> inputs)
    {
        return Fusion.EvaluateFused(graph, groups, inputs);
    }

    private static bool AllEqual(Graph graph, Dictionary<int, double[]> a, Dictionary<int, double[]> b)
    {
        foreach (OpNode op in graph.Ops)
        {
            int tensor = op.Output;
            if (!b.ContainsKey(tensor)) continue;      // a fused kernel never materialises that intermediate
            if (!Same(a[tensor], b[tensor])) return false;
        }
        return true;
    }

    private static bool Same(double[] a, double[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (BitConverter.DoubleToInt64Bits(a[i]) != BitConverter.DoubleToInt64Bits(b[i])) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
