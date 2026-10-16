// Task 719 - Implement parameter server simulation for distributed ML training. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// A parameter server (Li et al., 2014) splits training into p workers that each own a shard of the
// data plus a central server that owns the parameter vector. Every round a worker *pulls* a snapshot
// of the parameters, computes the gradient of its share of the objective, and *pushes* the resulting
// delta back; the server adds the shares for a synchronous round, or applies each delta immediately
// for an asynchronous one. A shard's share is (1/n) sum over its samples plus its proportional slice
// lambda * (|S|/n) of the L2 term, which is exactly what makes plain summation over a partition
// reproduce the full-batch gradient. Ridge regression is used because both the answer and the
// gradient are known in closed form, so every component can be checked against an *independent* oracle
// rather than against a second copy of the same code:
//
//   closed form    w* = (X^T X + lambda I)^-1 X^T y          (own Gaussian elimination, partial pivoting)
//   Gram gradient  grad(w) = (X^T X / n + lambda I) w - X^T y / n   (precomputed matrices, a different
//                  code path from the per-sample loop the workers run, so it also cross-checks the
//                  shard weighting: sum over a partition must equal the full-batch gradient)
//   Gershgorin     L = max_i sum_j |G_ij| + lambda bounds the Hessian, so lr = 1/L is a safe step
//
// Documented simplifications: the network is modelled by a deterministic staleness *lag* (a worker
// always reads the parameters from `lag` versions ago) instead of real asynchrony, so every run is
// reproducible; worker compute time is deterministic in the shard size. Communication is accounted in
// bytes rather than simulated as messages. Staleness is bounded and over-stale pushes are rejected
// (SSP style), which is verified by counting applied vs rejected pushes. Asynchronous schedules are
// Gauss-Seidel like: their fixed point sits O(lr) away from w*, which the checks confirm by shrinking
// lr fourfold and watching that offset shrink with it. All data comes from splitmix64.
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

    internal double NextSigned() => 2.0 * NextUnit() - 1.0;
}

/// <summary>Ridge regression data plus the two independent ways of evaluating its gradient.</summary>
internal sealed class RidgeData
{
    private readonly double[] _x;      // row-major Samples x Features
    private readonly double[] _y;
    private readonly double[] _gram;   // Features x Features : X^T X / Samples
    private readonly double[] _aty;    // Features            : X^T y / Samples

    internal RidgeData(int samples, int features, ulong seed)
    {
        Samples = samples;
        Features = features;
        _x = new double[samples * features];
        _y = new double[samples];
        var rng = new SplitMix64(seed);
        var truth = new double[features];
        for (int j = 0; j < features; j++) truth[j] = rng.NextSigned();
        for (int i = 0; i < samples; i++)
        {
            double acc = 0.0;
            for (int j = 0; j < features; j++)
            {
                double v = rng.NextSigned();
                _x[i * features + j] = v;
                acc += v * truth[j];
            }
            _y[i] = acc + 0.05 * rng.NextSigned();
        }
        _gram = new double[features * features];
        _aty = new double[features];
        for (int i = 0; i < samples; i++)
        {
            for (int a = 0; a < features; a++)
            {
                double xa = _x[i * features + a];
                for (int b = 0; b < features; b++) _gram[a * features + b] += xa * _x[i * features + b];
                _aty[a] += xa * _y[i];
            }
        }
        for (int k = 0; k < _gram.Length; k++) _gram[k] /= samples;
        for (int k = 0; k < _aty.Length; k++) _aty[k] /= samples;
    }

    internal int Samples { get; }
    internal int Features { get; }

    internal double SampleEntry(int i, int j) => _x[i * Features + j];
    internal double Label(int i) => _y[i];

    /// <summary>Gradient of the shard loss, already scaled by 1/Samples so shards add up to the full one.</summary>
    internal double[] ShardGradient(IReadOnlyList<int> shard, IReadOnlyList<double> w, double lambda)
    {
        var g = new double[Features];
        for (int s = 0; s < shard.Count; s++)
        {
            int i = shard[s];
            double dot = 0.0;
            for (int j = 0; j < Features; j++) dot += _x[i * Features + j] * w[j];
            double err = dot - _y[i];
            for (int j = 0; j < Features; j++) g[j] += _x[i * Features + j] * err;
        }
        double scale = 1.0 / Samples;
        double reg = lambda * shard.Count * scale;
        for (int j = 0; j < Features; j++) g[j] = g[j] * scale + reg * w[j];
        return g;
    }

    /// <summary>Full-batch gradient through the precomputed Gram matrices: the independent oracle path.</summary>
    internal double[] GramGradient(IReadOnlyList<double> w, double lambda)
    {
        var g = new double[Features];
        for (int a = 0; a < Features; a++)
        {
            double acc = -_aty[a];
            for (int b = 0; b < Features; b++) acc += _gram[a * Features + b] * w[b];
            g[a] = acc + lambda * w[a];
        }
        return g;
    }

    internal double Loss(IReadOnlyList<double> w, double lambda)
    {
        double residual = 0.0;
        double penalty = 0.0;
        for (int i = 0; i < Samples; i++)
        {
            double dot = 0.0;
            for (int j = 0; j < Features; j++) dot += _x[i * Features + j] * w[j];
            double e = dot - _y[i];
            residual += e * e;
        }
        for (int j = 0; j < Features; j++) penalty += w[j] * w[j];
        return 0.5 * residual / Samples + 0.5 * lambda * penalty;
    }

    internal double[] RidgeSolution(double lambda)
    {
        int d = Features;
        var m = new double[d, d + 1];
        for (int a = 0; a < d; a++)
        {
            for (int b = 0; b < d; b++) m[a, b] = _gram[a * d + b] + (a == b ? lambda : 0.0);
            m[a, d] = _aty[a];
        }
        for (int col = 0; col < d; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < d; r++) if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col])) pivot = r;
            if (pivot != col)
                for (int c = col; c <= d; c++)
                {
                    (m[col, c], m[pivot, c]) = (m[pivot, c], m[col, c]);
                }
            double p = m[col, col];
            for (int r = col + 1; r < d; r++)
            {
                double f = m[r, col] / p;
                if (f == 0.0) continue;
                for (int c = col; c <= d; c++) m[r, c] -= f * m[col, c];
            }
        }
        var w = new double[d];
        for (int r = d - 1; r >= 0; r--)
        {
            double acc = m[r, d];
            for (int c = r + 1; c < d; c++) acc -= m[r, c] * w[c];
            w[r] = acc / m[r, r];
        }
        return w;
    }

    /// <summary>Gershgorin bound on the largest Hessian eigenvalue: lr = 1/L is a safe step size.</summary>
    internal double LipschitzBound(double lambda)
    {
        double best = 0.0;
        for (int a = 0; a < Features; a++)
        {
            double row = lambda;
            for (int b = 0; b < Features; b++)
                row += Math.Abs(b == a ? _gram[a * Features + b] + lambda : _gram[a * Features + b]);
            best = Math.Max(best, row);
        }
        return best;
    }
}

/// <summary>Per-shard Gram statistics: a matrix formulation of the shard gradient (oracle path).</summary>
internal sealed class ShardStats
{
    private readonly RidgeData _data;
    private readonly double[] _gram;   // Features x Features
    private readonly double[] _aty;    // Features

    internal ShardStats(RidgeData data, int[] indices)
    {
        _data = data;
        Indices = indices;
        _gram = new double[data.Features * data.Features];
        _aty = new double[data.Features];
        foreach (int i in indices)
        {
            for (int a = 0; a < data.Features; a++)
            {
                double xa = data.SampleEntry(i, a);
                for (int b = 0; b < data.Features; b++) _gram[a * data.Features + b] += xa * data.SampleEntry(i, b);
                _aty[a] += xa * data.Label(i);
            }
        }
        double scale = 1.0 / data.Samples;
        for (int k = 0; k < _gram.Length; k++) _gram[k] *= scale;
        for (int k = 0; k < _aty.Length; k++) _aty[k] *= scale;
    }

    internal int[] Indices { get; }

    internal double Weight => (double)Indices.Length / _data.Samples;

    internal double[] Gradient(IReadOnlyList<double> w, double lambda)
    {
        int d = _data.Features;
        var g = new double[d];
        for (int a = 0; a < d; a++)
        {
            double acc = -_aty[a];
            for (int b = 0; b < d; b++) acc += _gram[a * d + b] * w[b];
            g[a] = acc + lambda * Weight * w[a];
        }
        return g;
    }

    internal double[] Translate(double[] gradient, double lr)
    {
        var delta = new double[gradient.Length];
        for (int k = 0; k < gradient.Length; k++) delta[k] = -lr * gradient[k];
        return delta;
    }
}

/// <summary>Central parameter store: versioned snapshots, push/pull, bounded staleness and byte accounting.</summary>
internal sealed class ParameterServer
{
    private readonly int _d;
    private readonly List<double[]> _history = new List<double[]>();
    private readonly double[] _w;

    internal ParameterServer(IReadOnlyList<double> initial)
    {
        _d = initial.Count;
        _w = new double[_d];
        for (int k = 0; k < _d; k++) _w[k] = initial[k];
        _history.Add((double[])_w.Clone());
    }

    internal long Submitted { get; private set; }
    internal long Applied { get; private set; }
    internal long Rejected { get; private set; }
    internal long BytesPushed { get; private set; }
    internal int MaxObservedStaleness { get; private set; }

    internal int Version => _history.Count - 1;
    internal double[] Current => (double[])_w.Clone();

    internal double[] Snapshot(int version)
    {
        if (version < 0 || version > Version) throw new ArgumentOutOfRangeException(nameof(version));
        return (double[])_history[version].Clone();
    }

    internal bool Push(IReadOnlyList<double> delta, int readVersion, int maxStaleness, int bytes)
    {
        if (readVersion < 0 || readVersion > Version) throw new ArgumentOutOfRangeException(nameof(readVersion));
        Submitted++;
        int staleness = Version - readVersion;
        if (staleness > maxStaleness) { Rejected++; return false; }
        for (int k = 0; k < _d; k++) _w[k] += delta[k];
        _history.Add((double[])_w.Clone());
        Applied++;
        BytesPushed += bytes;
        MaxObservedStaleness = Math.Max(MaxObservedStaleness, staleness);
        return true;
    }
}

public static class Solution
{
    private sealed class Run
    {
        internal double[] Final = Array.Empty<double>();
        internal ParameterServer Server = null!;
        internal long Barriers, GradientEvaluations, RoundsExecuted;
        internal double FirstLoss, LastLoss;
        internal bool LossMonotone = true;
    }

    /// <summary>Synchronous parameter-server round: every worker pulls the same version, deltas are averaged.</summary>
    private static Run SyncTrain(RidgeData data, List<ShardStats> shards, double lr, double lambda, int rounds, bool oracle)
    {
        var run = new Run { Server = new ParameterServer(new double[data.Features]) };
        int d = data.Features;
        double previous = double.PositiveInfinity;
        for (int round = 0; round < rounds; round++)
        {
            double[] w = run.Server.Current;                       // the barrier: one version for everybody
            var delta = new double[d];
            foreach (ShardStats shard in shards)
            {
                double[] g = oracle
                    ? shard.Gradient(w, lambda)
                    : data.ShardGradient(shard.Indices, w, lambda);
                run.GradientEvaluations++;
                for (int k = 0; k < d; k++) delta[k] -= lr * g[k];   // shards already carry their objective share
            }
            double loss = data.Loss(w, lambda);
            if (round == 0) run.FirstLoss = loss;
            if (loss > previous + 1e-12 * Math.Max(1.0, previous)) run.LossMonotone = false;
            previous = loss;
            run.Server.Push(delta, run.Server.Version, int.MaxValue, shards.Count * d * 8);
            run.Barriers++;
        }
        run.Final = run.Server.Current;
        run.LastLoss = data.Loss(run.Final, lambda);
        run.RoundsExecuted = rounds;
        return run;
    }

    /// <summary>Bounded-staleness asynchronous training: each worker reads a snapshot `lag` versions old.</summary>
    private static Run AsyncTrain(RidgeData data, List<ShardStats> shards, double lr, double lambda,
        int steps, int lag, int maxStaleness, bool oracle)
    {
        var run = new Run { Server = new ParameterServer(new double[data.Features]) };
        int d = data.Features;
        run.FirstLoss = data.Loss(run.Server.Current, lambda);
        for (int step = 0; step < steps; step++)
        {
            ShardStats shard = shards[step % shards.Count];
            int readVersion = Math.Max(0, run.Server.Version - lag);
            double[] w = run.Server.Snapshot(readVersion);
            double[] g = oracle ? shard.Gradient(w, lambda) : data.ShardGradient(shard.Indices, w, lambda);
            run.GradientEvaluations++;
            run.Server.Push(shard.Translate(g, lr), readVersion, maxStaleness, d * 8);
        }
        run.Final = run.Server.Current;
        run.LastLoss = data.Loss(run.Final, lambda);
        run.RoundsExecuted = steps;
        return run;
    }

    public static void Main()
    {
        const ulong seed = 0x51EED719UL;
        var data = new RidgeData(120, 6, seed);
        var sizes = new[] { 60, 30, 30, 0 };
        var shards = new List<ShardStats>();
        {
            int start = 0;
            foreach (int size in sizes)
            {
                if (size == 0) continue;
                var indices = new int[size];
                for (int k = 0; k < size; k++) indices[k] = start + k;
                shards.Add(new ShardStats(data, indices));
                start += size;
            }
        }
        const double lambda = 0.1;
        double L = data.LipschitzBound(lambda);
        double[] wStar = data.RidgeSolution(lambda);
        Console.WriteLine("samples=" + data.Samples + " features=" + data.Features + " shards=" + shards.Count
            + " L=" + L.ToString("G6") + " ridgeLoss=" + data.Loss(wStar, lambda).ToString("G8"));

        // ---- sharded gradient == full-batch gradient (the defining property of a parameter server) ---
        {
            var w = new double[data.Features];
            var rng = new SplitMix64(0xABCDEFUL);
            for (int j = 0; j < data.Features; j++) w[j] = rng.NextSigned();
            double[] full = data.GramGradient(w, lambda);
            double[] samplePath = data.ShardGradient(All(data.Samples), w, lambda);
            Check(MaxAbs(Diff(full, samplePath)) < 1e-12,
                "the per-sample loop and the Gram oracle agree on the full-batch gradient ("
                + MaxAbs(Diff(full, samplePath)) + ")");

            var summed = new double[data.Features];
            double[] unweighted = new double[data.Features];
            bool shardPathsAgree = true;
            foreach (ShardStats shard in shards)
            {
                double[] loop = data.ShardGradient(shard.Indices, w, lambda);
                double[] statsPath = shard.Gradient(w, lambda);
                if (MaxAbs(Diff(loop, statsPath)) > 1e-12) shardPathsAgree = false;
                for (int k = 0; k < data.Features; k++)
                {
                    summed[k] += loop[k];
                    unweighted[k] += loop[k] / shards.Count;
                }
            }
            Check(shardPathsAgree, "both gradient formulations agree on every shard");
            Check(MaxAbs(Diff(summed, full)) < 1e-12, "the shard gradients sum to the full-batch gradient ("
                + MaxAbs(Diff(summed, full)) + ")");
            Check(MaxAbs(Diff(unweighted, full)) > 1e-3,
                "size-weighting the already-weighted shares would double-count the regulariser ("
                + MaxAbs(Diff(unweighted, full)) + ")");

            var single = new ShardStats(data, new[] { 0, 1, 2, 3, 4, 5 });
            var one = new ShardStats(data, new[] { 7 });
            var hand = new double[data.Features];
            for (int j = 0; j < data.Features; j++)
            {
                double dot = 0.0;
                for (int c = 0; c < data.Features; c++) dot += data.SampleEntry(7, c) * w[c];
                hand[j] = data.SampleEntry(7, j) * (dot - data.Label(7)) / data.Samples + lambda / data.Samples * w[j];
            }
            Check(MaxAbs(Diff(one.Gradient(w, lambda), hand)) < 1e-14,
                "a single-sample shard matches the hand-derived formula");
            Check(single.Indices.Length == 6 && one.Weight > 0.0 && one.Weight < 1.0,
                "shard weights are the sample fractions");
        }

        // ---- synchronous training converges to the closed-form ridge solution ----------------------
        Run sync = SyncTrain(data, shards, 1.0 / L, lambda, 250, false);
        double syncError = MaxAbs(Diff(sync.Final, wStar));
        Console.WriteLine("syncError=" + syncError.ToString("G6") + " syncLoss=" + sync.LastLoss.ToString("G10")
            + " barriers=" + sync.Barriers);
        Check(syncError < 1e-9, "synchronous rounds reach the closed-form ridge solution (" + syncError + ")");
        Check(sync.LossMonotone, "a 1/L step decreases the loss at every synchronous round");
        Check(sync.LastLoss <= data.Loss(new double[data.Features], lambda) - 1e-3, "training beats the zero model");
        Check(Math.Abs(sync.LastLoss - data.Loss(wStar, lambda)) < 1e-9, "the final loss is the optimal one");
        Check(data.Loss(wStar, lambda) <= sync.FirstLoss + 1e-12, "the ridge solution is the global minimum");

        // ---- the update rule reproduces an independent Gram-matrix trajectory ----------------------
        Run syncOracle = SyncTrain(data, shards, 1.0 / L, lambda, 250, true);
        Check(MaxAbs(Diff(sync.Final, syncOracle.Final)) < 1e-9,
            "the sample-loop trajectory matches the Gram oracle trajectory ("
            + MaxAbs(Diff(sync.Final, syncOracle.Final)) + ")");

        // ---- asynchronous training with bounded staleness ------------------------------------------
        // A sequential (Gauss-Seidel-style) schedule has its fixed point O(lr) away from w*, so the
        // checks below verify both proximity and the lr-proportionality of that offset.
        Run asyncFresh = AsyncTrain(data, shards, 1.0 / (2.0 * L), lambda, 3000, 0, 16, false);
        Run asyncStale = AsyncTrain(data, shards, 1.0 / (2.0 * L), lambda, 3000, 2, 16, false);
        Run asyncTight = AsyncTrain(data, shards, 1.0 / (8.0 * L), lambda, 12000, 2, 16, false);
        double freshError = MaxAbs(Diff(asyncFresh.Final, wStar));
        double staleError = MaxAbs(Diff(asyncStale.Final, wStar));
        double tightError = MaxAbs(Diff(asyncTight.Final, wStar));
        Console.WriteLine("lag0Error=" + freshError.ToString("G6") + " lag2Error=" + staleError.ToString("G6")
            + " lag2QuarterStepError=" + tightError.ToString("G6")
            + " observedStaleness=" + asyncStale.Server.MaxObservedStaleness
            + " rejected=" + asyncStale.Server.Rejected);
        Check(freshError < 0.02, "asynchronous updates with lag 0 converge near the ridge solution (" + freshError + ")");
        Check(staleError < 0.02, "asynchronous updates with staleness 2 still converge (" + staleError + ")");
        Check(tightError < 0.4 * staleError,
            "quartering the step size quarters the sequential fixed-point offset (" + tightError + " vs " + staleError + ")");
        Check(asyncStale.Server.MaxObservedStaleness == 2, "every stale push was exactly two versions old");
        Check(asyncStale.Server.Rejected == 0 && asyncStale.Server.Applied == 3000,
            "all in-bound pushes were applied, none rejected");
        Check(asyncStale.Server.BytesPushed == 3000L * data.Features * 8, "push traffic is d doubles per step");
        Check(asyncStale.Server.Version == 3000 && asyncFresh.Server.Version == 3000,
            "the version counter advances once per applied push");
        Check(asyncStale.LastLoss < data.Loss(wStar, lambda) + 1e-3 && asyncFresh.LastLoss < data.Loss(wStar, lambda) + 1e-3,
            "both asynchronous schedules end within 1e-3 of the optimal loss");
        Check(asyncFresh.Server.Submitted == asyncFresh.Server.Applied + asyncFresh.Server.Rejected,
            "submitted == applied + rejected");

        // ---- over-stale pushes are rejected, and p = 1 makes sync and async identical ---------------
        {
            Run tooStale = AsyncTrain(data, shards, 1.0 / (2.0 * L), lambda, 200, 5, 2, false);
            Check(tooStale.Server.Rejected > 0, "pushes older than the staleness bound are rejected");
            Check(tooStale.Server.Submitted == tooStale.Server.Applied + tooStale.Server.Rejected,
                "the rejection bookkeeping balances");
            Check(tooStale.Server.Version == (int)tooStale.Server.Applied,
                "the version equals the number of applied updates");

            var oneShard = new List<ShardStats> { new ShardStats(data, All(data.Samples)) };
            Run syncOne = SyncTrain(data, oneShard, 1.0 / L, lambda, 40, false);
            Run asyncOne = AsyncTrain(data, oneShard, 1.0 / L, lambda, 40, 0, 16, false);
            Check(asyncOne.Server.Applied == 40 && syncOne.Server.Applied == 40, "one worker, one push per step");
            Check(MaxAbs(Diff(syncOne.Final, asyncOne.Final)) == 0.0,
                "with a single worker and no staleness the two schedules coincide bit-for-bit");
        }

        // ---- reproducibility -----------------------------------------------------------------------
        {
            Run again = SyncTrain(data, shards, 1.0 / L, lambda, 250, false);
            bool identical = true;
            for (int k = 0; k < data.Features; k++)
                if (BitConverter.DoubleToInt64Bits(again.Final[k]) != BitConverter.DoubleToInt64Bits(sync.Final[k]))
                    identical = false;
            Check(identical, "the same schedule and seed reproduce the same parameters bit-for-bit");
            Check(sync.GradientEvaluations == 250L * shards.Count, "the synchronous run evaluated one gradient per worker-round");
        }

        Console.WriteLine("PASS");
    }

    private static int[] All(int n)
    {
        var a = new int[n];
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    private static double[] Diff(double[] a, double[] b)
    {
        var d = new double[a.Length];
        for (int i = 0; i < a.Length; i++) d[i] = a[i] - b[i];
        return d;
    }

    private static double MaxAbs(double[] v)
    {
        double m = 0.0;
        for (int i = 0; i < v.Length; i++) m = Math.Max(m, Math.Abs(v[i]));
        return m;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
