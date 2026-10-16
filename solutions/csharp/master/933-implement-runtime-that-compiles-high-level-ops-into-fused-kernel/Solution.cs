// Task 933 - Implement runtime that compiles high-level ops into fused kernels with autotuning. (Tier: Master, Beyond — Systems & Performance)
//
// A tiny operator graph (matmul, bias, relu, scale - plus one deliberately unsupported op) is compiled in two steps.
// Fusion groups the chain: a matmul opens a group and absorbs the elementwise ops that follow it, so the epilogue is
// applied to each output element as it is produced and no intermediate tensor is ever materialised; a run of purely
// elementwise ops fuses into a single traversal, while an unsupported op closes the current group and is executed by
// the reference path (and recorded as a fallback instead of being silently dropped). Each group then gets an
// autotuned tiling: candidate (BM, BN) block shapes are scored by a deterministic cost model (work plus per-block
// launch/setup cost plus a register-pressure term), the argmin is chosen and the decision is memoised under the shape
// key so a repeated compile with the same shapes costs nothing. Because both executors apply the operations in the
// same order per output element, the fused result is required to be bit-for-bit identical to the unfused one.
//
// Simplifications: double precision only (no real device, no vectorisation, no memory hierarchy model), a linear
// chain rather than a DAG, one small candidate set for the tuner, a synthetic cost model instead of measurements.
// Verified: fused and unfused outputs are bit-identical for random graphs, fusion lowers the kernel and intermediate
// counts, the tuner's choice equals an exhaustive scan of the candidates, the memoised decision avoids re-tuning,
// a single-op graph performs no fusion, an elementwise-only graph fuses into one kernel, an unsupported op is
// reported as a fallback and still computed, and every run is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

internal enum Kind { MatMul, Bias, Relu, Scale, Unsupported }

internal sealed class Op
{
    internal Kind Kind;
    internal double[] Data = Array.Empty<double>();                  // matmul weights (K x N) or the bias vector
    internal double Scalar;
    internal int M, K, N;
    internal static Op MatMul(int m, int k, int n, ref ulong state)
    {
        var w = new double[k * n];
        for (int i = 0; i < w.Length; i++) w[i] = Rng.Signed(ref state);
        return new Op { Kind = Kind.MatMul, Data = w, M = m, K = k, N = n };
    }
    internal static Op Bias(int n, ref ulong state)
    {
        var b = new double[n];
        for (int i = 0; i < n; i++) b[i] = Rng.Signed(ref state);
        return new Op { Kind = Kind.Bias, Data = b, N = n };
    }
    internal static Op Relu() => new Op { Kind = Kind.Relu };
    internal static Op Scale(double factor) => new Op { Kind = Kind.Scale, Scalar = factor };
    internal static Op Unsupported(int n) => new Op { Kind = Kind.Unsupported, N = n };
}

/** Groups the operator chain into kernels and autotunes the matmul tiling. */
internal sealed class Runtime
{
    internal readonly Dictionary<string, int[]> TuningCache = new Dictionary<string, int[]>();
    internal int Tunings;
    /** Fusion: a matmul group absorbs the elementwise ops that follow; elementwise runs fuse together. */
    internal List<List<int>> Groups(List<Op> graph, out List<int> fallbacks)
    {
        var groups = new List<List<int>>();
        fallbacks = new List<int>();
        var current = new List<int>();
        for (int i = 0; i < graph.Count; i++)
        {
            switch (graph[i].Kind)
            {
                case Kind.MatMul:
                    if (current.Count > 0) groups.Add(current);
                    current = new List<int> { i };
                    break;
                case Kind.Unsupported:
                    if (current.Count > 0) groups.Add(current);
                    current = new List<int>();
                    fallbacks.Add(i);
                    groups.Add(new List<int> { i });
                    break;
                default:
                    current.Add(i);
                    break;
            }
        }
        if (current.Count > 0) groups.Add(current);
        return groups;
    }
    /** Deterministic cost model: work, per-block launch cost, register pressure. */
    internal double ModelCost(int m, int n, int k, int bm, int bn) =>
        (double)m * n * k + 6.0 * Math.Ceiling(m / (double)bm) * Math.Ceiling(n / (double)bn) + 2.0 * Math.Max(bm, bn);
    internal int[] Tune(int m, int n, int k, string key)
    {
        if (TuningCache.TryGetValue(key, out int[] cached)) return cached;
        var candidates = new List<int[]>();
        foreach (int bm in new[] { 1, 2, 4, 8, 16 })
            foreach (int bn in new[] { 1, 2, 4, 8, 16 })
                if (bm <= m && bn <= n) candidates.Add(new[] { bm, bn });
        int[] best = candidates.OrderBy(c => ModelCost(m, n, k, c[0], c[1])).ThenBy(c => c[0]).ThenBy(c => c[1]).First();
        TuningCache[key] = best;
        Tunings++;
        return best;
    }
    private static double Epilogue(List<Op> graph, List<int> group, double value, int column)
    {
        for (int g = 1; g < group.Count; g++)
        {
            Op op = graph[group[g]];
            if (op.Kind == Kind.Bias) value += op.Data[column];
            else if (op.Kind == Kind.Relu) value = Math.Max(0.0, value);
            else if (op.Kind == Kind.Scale) value *= op.Scalar;
        }
        return value;
    }
    /** Unfused execution: every op is its own kernel and every op materialises a tensor. */
    internal double[] RunUnfused(List<Op> graph, double[] input, out int launches, out int intermediates)
    {
        double[] data = input;
        launches = 0; intermediates = 0;
        foreach (Op op in graph)
        {
            launches++;
            double[] next;
            switch (op.Kind)
            {
                case Kind.MatMul:
                    next = new double[op.M * op.N];
                    for (int i = 0; i < op.M; i++)
                        for (int j = 0; j < op.N; j++)
                        {
                            double acc = 0;
                            for (int t = 0; t < op.K; t++) acc += data[i * op.K + t] * op.Data[t * op.N + j];
                            next[i * op.N + j] = acc;
                        }
                    break;
                case Kind.Bias:
                    next = data.Select((v, i) => v + op.Data[i % op.Data.Length]).ToArray();
                    break;
                case Kind.Relu:
                    next = data.Select(v => Math.Max(0.0, v)).ToArray();
                    break;
                case Kind.Scale:
                    next = data.Select(v => v * op.Scalar).ToArray();
                    break;
                default:
                    next = data.Reverse().ToArray();                     // the unsupported op is emulated by a reversal
                    break;
            }
            data = next;
            intermediates++;
        }
        return data;
    }
    /** Fused execution: one kernel per group, no intermediate tensors inside a group. */
    internal double[] RunFused(List<Op> graph, double[] input, out int launches, out List<int> fallbacks)
    {
        List<List<int>> groups = Groups(graph, out fallbacks);
        double[] data = input;
        launches = 0;
        foreach (List<int> group in groups)
        {
            launches++;
            Op head = graph[group[0]];
            if (head.Kind == Kind.MatMul)
            {
                int[] tile = Tune(head.M, head.N, head.K, head.M + "x" + head.K + "x" + head.N);
                var next = new double[head.M * head.N];
                for (int bi = 0; bi < head.M; bi += tile[0])
                    for (int bj = 0; bj < head.N; bj += tile[1])
                        for (int i = bi; i < Math.Min(head.M, bi + tile[0]); i++)
                            for (int j = bj; j < Math.Min(head.N, bj + tile[1]); j++)
                            {
                                double acc = 0;
                                for (int t = 0; t < head.K; t++) acc += data[i * head.K + t] * head.Data[t * head.N + j];
                                next[i * head.N + j] = Epilogue(graph, group, acc, j);
                            }
                data = next;
            }
            else if (head.Kind == Kind.Unsupported)
            { data = data.Reverse().ToArray(); }
            else
            {
                var next = new double[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    double value = data[i];
                    foreach (int index in group)
                    {
                        Op op = graph[index];
                        if (op.Kind == Kind.Bias) value += op.Data[i % op.Data.Length];
                        else if (op.Kind == Kind.Relu) value = Math.Max(0.0, value);
                        else if (op.Kind == Kind.Scale) value *= op.Scalar;
                    }
                    next[i] = value;
                }
                data = next;
            }
        }
        return data;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 933UL;
    /** Independent brute-force scan of the same cost model, written out without LINQ. */
    private static int[] ScanBest(int m, int n, int k)
    {
        int[] best = { 1, 1 };
        double bestCost = double.PositiveInfinity;
        for (int bm = 1; bm <= 16; bm *= 2)
            for (int bn = 1; bn <= 16; bn *= 2)
            {
                if (bm > m || bn > n) continue;
                double cost = (double)m * n * k + 6.0 * Math.Ceiling(m / (double)bm) * Math.Ceiling(n / (double)bn) + 2.0 * Math.Max(bm, bn);
                if (cost < bestCost) { bestCost = cost; best = new[] { bm, bn }; }
            }
        return best;
    }
    private static List<Op> RandomGraph(ref ulong state, int m, int k, int n)
    {
        var graph = new List<Op> { Op.MatMul(m, k, n, ref state) };
        for (int extra = 0; extra < 3; extra++)
        {
            switch (Rng.Next(ref state) % 3)
            {
                case 0: graph.Add(Op.Bias(n, ref state)); break;
                case 1: graph.Add(Op.Relu()); break;
                default: graph.Add(Op.Scale(0.5 + Rng.Next(ref state) % 3)); break;
            }
        }
        return graph;
    }
    public static void Main()
    {
        var runtime = new Runtime();
        // 1. fused and unfused executions agree bit for bit on random graphs
        var comparisons = new List<int[]>();
        int mismatches = 0;
        for (int trial = 0; trial < 6; trial++)
        {
            List<Op> graph = RandomGraph(ref _state, 3 + trial, 2 + trial, 3 + trial);
            var input = new double[graph[0].M * graph[0].K];
            for (int i = 0; i < input.Length; i++) input[i] = Rng.Signed(ref _state);
            double[] unfused = runtime.RunUnfused(graph, (double[])input.Clone(), out int uLaunches, out int intermediates);
            double[] fused = runtime.RunFused(graph, (double[])input.Clone(), out int fLaunches, out List<int> fallbacks);
            if (!unfused.SequenceEqual(fused)) mismatches++;
            comparisons.Add(new[] { uLaunches, fLaunches, intermediates, fallbacks.Count });
        }
        Check(mismatches == 0, "six random graphs produce bit-identical fused and unfused outputs");
        Check(comparisons.All(c => c[1] <= c[0]), "fusion never increases the kernel count (" + string.Join(", ", comparisons.Select(c => c[1] + "/" + c[0])) + ")");
        Check(comparisons.Any(c => c[1] < c[0]), "fusion strictly reduces the kernel count on at least one graph");
        Check(comparisons.All(c => c[2] >= c[1]), "the unfused path materialises at least as many tensors as the fused path launches");

        // 2. autotuning: the chosen tile equals an exhaustive scan, and the decision is memoised
        var sizes = new[] { new[] { 8, 8, 8 }, new[] { 16, 4, 12 }, new[] { 3, 5, 7 }, new[] { 1, 1, 1 }, new[] { 32, 2, 30 } };
        bool tuningMatches = true;
        foreach (int[] triple in sizes)
        {
            int[] chosen = runtime.Tune(triple[0], triple[2], triple[1], triple[0] + "x" + triple[1] + "x" + triple[2] + "-check");
            int[] scanned = ScanBest(triple[0], triple[2], triple[1]);
            tuningMatches &= chosen[0] == scanned[0] && chosen[1] == scanned[1];
        }
        Check(tuningMatches, "the tuner's choice equals the exhaustive scan for " + sizes.Length + " shapes (" + runtime.Tunings + " tunings)");
        int before = runtime.Tunings;
        runtime.Tune(8, 8, 8, "8x8x8-check");
        Check(runtime.Tunings == before, "a repeated compile of the same shape is served from the tuning cache");
        runtime.Tune(9, 9, 9, "9x9x9-check");
        Check(runtime.Tunings == before + 1 && runtime.TuningCache.Count >= 6,
            "a new shape triggers exactly one tuning and the cache holds a decision per shape key (" + runtime.TuningCache.Count + " entries)");

        // 3. boundary cases: single op, elementwise-only chain and an unsupported op
        var single = new List<Op> { Op.MatMul(4, 4, 4, ref _state) };
        double[] singleInput = Enumerable.Range(0, 16).Select(i => Rng.Signed(ref _state)).ToArray();
        double[] singleFused = runtime.RunFused(single, (double[])singleInput.Clone(), out int singleLaunches, out _);
        Check(singleLaunches == 1 && singleFused.SequenceEqual(runtime.RunUnfused(single, (double[])singleInput.Clone(), out _, out _)), "a single-op graph launches one kernel and performs no fusion (" + singleLaunches + ")");
        var elementwise = new List<Op> { Op.Bias(4, ref _state), Op.Relu(), Op.Scale(2.0) };
        double[] vector = new double[4];
        double[] elementwiseFused = runtime.RunFused(elementwise, vector, out int elementwiseLaunches, out _);
        Check(elementwiseLaunches == 1 && elementwiseFused.SequenceEqual(runtime.RunUnfused(elementwise, new double[4], out _, out _)), "an elementwise-only chain fuses into one kernel");
        var withUnsupported = new List<Op> { Op.MatMul(2, 2, 2, ref _state), Op.Relu(), Op.Unsupported(4), Op.Scale(0.5) };
        double[] mixedInput = new double[4];
        double[] mixedFused = runtime.RunFused(withUnsupported, (double[])mixedInput.Clone(), out int mixedLaunches, out List<int> mixedFallbacks);
        Check(mixedFallbacks.Count == 1 && mixedFallbacks[0] == 2, "the unsupported op is reported as a fallback at index 2");
        Check(mixedFused.SequenceEqual(runtime.RunUnfused(withUnsupported, (double[])mixedInput.Clone(), out _, out _)) && mixedLaunches == 3, "the graph with a fallback still evaluates correctly in three kernels");
        var reluOnly = new List<Op> { Op.Relu() };
        Check(runtime.RunFused(reluOnly, new[] { -2.0, 3.0 }, out _, out _).SequenceEqual(new[] { 0.0, 3.0 }), "a ReLU group zeroes negative values (boundary)");

        // 4. reproducibility
        var replay = new Runtime();
        _state = 933UL;
        List<Op> graphA = RandomGraph(ref _state, 5, 4, 5);
        var replayInput = new double[graphA[0].M * graphA[0].K];
        for (int i = 0; i < replayInput.Length; i++) replayInput[i] = Rng.Signed(ref _state);
        double[] firstRun = runtime.RunFused(graphA, replayInput, out _, out _);
        double[] secondRun = replay.RunFused(graphA, replayInput, out _, out _);
        Check(firstRun.SequenceEqual(secondRun) && runtime.Tune(5, 5, 4, "5x4x5-check").SequenceEqual(replay.Tune(5, 5, 4, "5x4x5-check")), "the fused execution and the tuning decision are reproducible");

        Console.WriteLine("fusion: 6 graphs, " + comparisons.Sum(c => c[1]) + " fused kernels vs " + comparisons.Sum(c => c[0]) + " unfused kernels, " + comparisons.Sum(c => c[2]) + " unfused intermediates, 0 output mismatches");
        Console.WriteLine("autotune: " + runtime.Tunings + " tunings, cache " + runtime.TuningCache.Count + " entries, choices match the exhaustive scan; fallbacks " + mixedFallbacks.Count + "; single-op and elementwise-only graphs behave");
        Console.WriteLine("PASS");
    }
}
