// Task 927 - Implement model-backed materialized views updating with streaming features. (Tier: Master, Beyond — Machine Learning & AI)
//
// A materialised view backed by a *model*: it holds predictions of a least-squares line y = a + b*x and is refreshed
// from a stream of deltas. The model keeps only the sufficient statistics (n, sum x, sum y, sum xx, sum xy), so an
// insert adds its row and a delete subtracts it again, and a refresh of the view costs O(#query points) instead of
// O(#rows). Streaming features are served from a sliding window implemented as a ring buffer whose outgoing row is
// subtracted from the same statistics, which keeps the window exact without re-scanning history. Every incremental
// result is cross-checked against a full recompute from the raw rows (the independent oracle), with a relative
// tolerance because the two accumulation orders differ in round-off.
//
// Simplifications: one entity, scalar x, a linear model (no regularisation, no nonlinear refit), in-memory stream,
// fixed window, no persistence or scheduling. Verified: incremental statistics equal the brute-force recompute after
// every batch, deletions undo exactly, the view matches the closed-form predictions, the sliding window holds exactly
// the last rows, an empty batch is bit-for-bit a no-op, an empty model / a single row / zero variance all report "no
// model" instead of dividing by zero, and the stream is reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
    internal static double Gaussian(ref ulong s)
    {
        double u = ((Next(ref s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double v = (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}

/** Sufficient statistics of a linear model, driven forward by inserts and backward by deletes. */
internal sealed class IncrementalModel
{
    internal long N;
    internal double Sx, Sy, Sxx, Sxy;
    internal int Updates;
    internal void Update(double x, double y, int sign)
    {
        N += sign;
        Sx += sign * x; Sy += sign * y; Sxx += sign * x * x; Sxy += sign * x * y;
        Updates++;
    }
    /** Coefficients, or null when the fit is undefined (fewer than two rows, or no variance in x). */
    internal double[] Fit()
    {
        if (N < 2) return null;
        double variance = Sxx - Sx * Sx / N;
        if (Math.Abs(variance) < 1e-12) return null;
        double slope = (Sxy - Sx * Sy / N) / variance;
        return new[] { Sy / N - slope * Sx / N, slope };
    }
    internal double Predict(double x, double fallback)
    {
        double[] coefficients = Fit();
        return coefficients == null ? fallback : coefficients[0] + coefficients[1] * x;
    }
}

/** Materialised view of the model at fixed query points, refreshed by streaming deltas. */
internal sealed class MaterializedView
{
    internal readonly double[] QueryPoints;
    internal double[] Values;
    internal int Refreshes;
    internal readonly IncrementalModel Model = new IncrementalModel();
    internal MaterializedView(double[] queryPoints) { QueryPoints = queryPoints; Values = new double[queryPoints.Length]; }
    internal void Apply(IReadOnlyList<double[]> batch)
    {
        foreach (double[] row in batch) Model.Update(row[0], row[1], 1);
        Refresh();
    }
    internal void Remove(IReadOnlyList<double[]> batch)
    {
        foreach (double[] row in batch) Model.Update(row[0], row[1], -1);
        Refresh();
    }
    internal void Refresh()
    {
        double fallback = Model.N > 0 ? Model.Sy / Model.N : 0.0;
        for (int i = 0; i < QueryPoints.Length; i++) Values[i] = Model.Predict(QueryPoints[i], fallback);
        Refreshes++;
    }
}

/** Sliding-window statistics over the last `capacity` rows (the ring buffer subtracts whatever it drops). */
internal sealed class SlidingWindow
{
    private readonly int _capacity;
    private readonly Queue<double[]> _rows = new Queue<double[]>();
    internal readonly IncrementalModel Window = new IncrementalModel();
    internal SlidingWindow(int capacity) { _capacity = capacity; }
    internal void Push(double[] row)
    {
        _rows.Enqueue(row); Window.Update(row[0], row[1], 1);
        if (_rows.Count > _capacity)
        {
            double[] dropped = _rows.Dequeue();
            Window.Update(dropped[0], dropped[1], -1);
        }
    }
    internal int Count => _rows.Count;
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static ulong _state = 927UL;
    /** Independent oracle: textbook least squares recomputed from the raw rows. */
    private static double[] OracleFit(List<double[]> rows)
    {
        int n = rows.Count;
        if (n < 2) return null;
        double sx = rows.Sum(r => r[0]), sy = rows.Sum(r => r[1]);
        double sxx = rows.Sum(r => r[0] * r[0]), sxy = rows.Sum(r => r[0] * r[1]);
        double variance = sxx - sx * sx / n;
        if (Math.Abs(variance) < 1e-12) return null;
        double slope = (sxy - sx * sy / n) / variance;
        return new[] { sy / n - slope * sx / n, slope };
    }
    private static bool Close(double a, double b) => Math.Abs(a - b) <= 1e-9 * Math.Max(1.0, Math.Abs(a) + Math.Abs(b));
    private static List<double[]> Batch(ref ulong state, int size)
    {
        var rows = new List<double[]>();
        for (int i = 0; i < size; i++)
        {
            double x = Rng.Signed(ref state) * 5.0;
            rows.Add(new[] { x, 2.5 - 0.75 * x + 0.2 * Rng.Gaussian(ref state) });
        }
        return rows;
    }
    public static void Main()
    {
        const int window = 25;
        var queryPoints = new[] { 0.0, 1.0, 2.5, 5.0, 10.0 };
        var view = new MaterializedView(queryPoints);
        var rows = new List<double[]>();
        var sliding = new SlidingWindow(window);
        int statMismatches = 0, viewMismatches = 0, windowMismatches = 0;

        // 1. five streaming batches: cross-check statistics, the view and the sliding window each time
        for (int batch = 0; batch < 5; batch++)
        {
            List<double[]> incoming = Batch(ref _state, 20);
            view.Apply(incoming);
            rows.AddRange(incoming);
            foreach (double[] row in incoming) sliding.Push(row);
            double[] oracle = OracleFit(rows);
            if (!Close(view.Model.Sx, rows.Sum(r => r[0])) || !Close(view.Model.Sy, rows.Sum(r => r[1])) ||
                !Close(view.Model.Sxx, rows.Sum(r => r[0] * r[0])) || !Close(view.Model.Sxy, rows.Sum(r => r[0] * r[1])) ||
                view.Model.N != rows.Count) statMismatches++;
            if (!Close(view.Values[3], oracle[0] + oracle[1] * queryPoints[3])) viewMismatches++;
            List<double[]> tail = rows.Skip(rows.Count - window).ToList();
            if (!Close(sliding.Window.Sx, tail.Sum(r => r[0])) || !Close(sliding.Window.Sy, tail.Sum(r => r[1])) ||
                sliding.Count != Math.Min(window, rows.Count)) windowMismatches++;
        }
        Check(statMismatches == 0, "incremental statistics equal the recompute after all five batches");
        double[] valuesAfterBatches = (double[])view.Values.Clone();
        Check(viewMismatches == 0, "the materialised view matches the closed-form predictions after every batch");
        Check(windowMismatches == 0, "the sliding window holds exactly the last " + window + " rows of the stream");
        Check(view.Model.Fit()[1] < 0 && Math.Abs(view.Model.Fit()[1] + 0.75) < 0.2,
            "the fitted slope recovers the generating slope (" + view.Model.Fit()[1].ToString("F4") + " vs -0.75)");

        // 2. deletions undo an insert exactly, and a re-apply restores it
        var reference = new List<double[]>(rows);
        List<double[]> undo = rows.Take(25).ToList();
        view.Remove(undo);
        reference.RemoveRange(0, 25);
        double[] afterDelete = OracleFit(reference);
        Check(view.Model.N == reference.Count && Close(view.Model.Sx, reference.Sum(r => r[0])) && Close(view.Model.Sy, reference.Sum(r => r[1])),
            "deleting 25 rows subtracts them from the statistics exactly");
        Check(Close(view.Values[4], afterDelete[0] + afterDelete[1] * queryPoints[4]), "the view follows the deleted rows out");
        view.Apply(undo);
        Check(view.Model.N == rows.Count, "re-applying the deleted rows restores the row count (" + view.Model.N + ")");
        Check(Close(view.Model.Fit()[0], OracleFit(rows)[0]), "the restored model matches the oracle again");
        Check(view.Refreshes == 7, "each batch, deletion and restore refreshed the view (" + view.Refreshes + " refreshes)");

        // 3. boundary cases: empty batch, empty model, a single row and zero variance
        int refreshesBefore = view.Refreshes;
        string snapshot = string.Join(",", view.Values.Select(v => v.ToString("R")));
        view.Apply(new List<double[]>());
        Check(view.Refreshes > refreshesBefore && string.Join(",", view.Values.Select(v => v.ToString("R"))) == snapshot,
            "an empty batch refreshes without changing a single prediction (bit-for-bit)");
        var empty = new IncrementalModel();
        Check(empty.Fit() == null && empty.Predict(3.0, -1.0) == -1.0, "an empty model is undefined and falls back");
        var single = new IncrementalModel();
        single.Update(2.0, 5.0, 1);
        Check(single.Fit() == null && single.Predict(2.0, 7.0) == 7.0, "a single row gives no slope (boundary)");
        var flat = new IncrementalModel();
        for (int i = 0; i < 5; i++) flat.Update(1.0, i, 1);
        Check(flat.Fit() == null, "zero variance in x is detected instead of dividing by zero");

        // 4. reproducibility of the stream and the view
        _state = 927UL;
        var replay = new MaterializedView(queryPoints);
        var replayRows = new List<double[]>();
        for (int batch = 0; batch < 5; batch++)
        {
            List<double[]> incoming = Batch(ref _state, 20);
            replay.Apply(incoming); replayRows.AddRange(incoming);
        }
        Check(replay.Model.N == rows.Count && replayRows[0][0] == rows[0][0] &&
            Enumerable.Range(0, queryPoints.Length).All(i => replay.Values[i] == valuesAfterBatches[i]),
            "the same seed reproduces the stream, the statistics and the view bit for bit");

        double[] final = view.Model.Fit() ?? new[] { 0.0, 0.0 };
        Console.WriteLine("stream: " + rows.Count + " rows in 5 batches, window " + window + ", mismatches stats " + statMismatches +
            " view " + viewMismatches + " window " + windowMismatches);
        Console.WriteLine("model: intercept " + final[0].ToString("F4") + ", slope " + final[1].ToString("F4") +
            " (generator 2.5 / -0.75); deletes undo exactly; empty/single/flat inputs report no model; " + view.Refreshes + " refreshes");
        Console.WriteLine("PASS");
    }
}
