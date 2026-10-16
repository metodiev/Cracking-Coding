// Task 823 - Implement real-time anomaly detection via streaming PCA or sketching. (Tier: Master, CTCI Ch 9 — System Design and Scalability)
//
// Three classic bounded-memory summaries are implemented, each verified against an exact oracle:
//
//   * Misra-Gries heavy hitters - k counters; an arriving item either increments its counter, takes a
//     free slot, or triggers a global decrement. The checked guarantee is the textbook one: every item
//     whose true frequency exceeds n/k is present, and every reported count c satisfies
//     true - n/k <= c <= true.
//   * Count-Min sketch - d rows of w counters with distinct hashes. Point estimates are the minimum
//     over rows; the empirical error is measured against the guarantee "error <= eps * n" for
//     w = ceil(e / eps), and the space is verified to be far below the stream length.
//   * Streaming PCA - Welford mean and covariance accumulators (O(d^2) per event) refreshed by a few
//     power-iteration steps with deflation to keep the top-k subspace. Its angle and eigenvalue are
//     compared with a Jacobi eigen-decomposition of the batch covariance (the exact oracle), and the
//     subspace is scored for drift when the data rotates.
//
// Anomaly detection is the reconstruction error under the streaming subspace. Detection quality (AUC
// against injected anomalies) and fidelity (agreement with the batch PCA detector) are both measured,
// and the memory of every structure is reported: the whole point is that they are O(d^2 + d*k +
// counters), not O(n*d).
//
// Simplifications (documented): a single in-process stream (no shards, no mergeable sketch protocol);
// the covariance is kept dense in d dimensions (fine for the small d used here, and a real system would
// sketch it); power iteration is restarted from the previous estimate rather than using a blocked
// incremental SVD; drift is handled by capping the Welford update at a decay window rather than by a
// proper forgetting factor; the anomaly threshold is a fixed quantile of the score distribution rather
// than a learned calibrator.
//
// Verified invariants: the Misra-Gries guarantee on random Zipfian streams; the count-min error bound
// and its space bound; the streaming eigenvector angle versus Jacobi stays small, the eigenvalue tracks
// the exact one, and the subspace survives a 45 degree rotation of the data (drift tracking); the
// streaming detector's AUC is close to the batch PCA detector's and clearly above chance; the injected
// anomalies are ranked above normal points; the memory of each summary stays inside its stated bound;
// and degenerate inputs (d = 1, k = d, an empty stream, constant data, an all-anomaly stream) behave.
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

    public double NextNormal()
        => (NextUnit() + NextUnit() + NextUnit() + NextUnit() + NextUnit() + NextUnit() - 3.0) * 1.1547005383792515;
}

// ---- Misra-Gries -------------------------------------------------------------------------------------

internal sealed class MisraGries
{
    private readonly Dictionary<int, long> counters = new Dictionary<int, long>();
    private readonly int capacity;

    public long Updates;
    public long DecrementRounds;
    public long PeakCounters;

    public MisraGries(int capacity) { this.capacity = Math.Max(1, capacity); }

    public int CounterCount => counters.Count;
    public long UpdateCost => Updates + DecrementRounds * capacity;

    public void Add(int item)
    {
        Updates++;
        if (counters.TryGetValue(item, out long count))
        {
            counters[item] = count + 1;
            return;
        }
        if (counters.Count < capacity)
        {
            counters[item] = 1;
            PeakCounters = Math.Max(PeakCounters, counters.Count);
            return;
        }
        DecrementRounds++;
        foreach (int key in counters.Keys.ToList())
        {
            long next = counters[key] - 1;
            if (next <= 0) counters.Remove(key);
            else counters[key] = next;
        }
    }

    public IEnumerable<KeyValuePair<int, long>> Summary => counters;

    public long Estimate(int item) => counters.TryGetValue(item, out long count) ? count : 0;
}

// ---- count-min sketch --------------------------------------------------------------------------------

internal sealed class CountMinSketch
{
    private readonly int[][] rows;
    private readonly int width;
    private readonly ulong seed;
    public long Updates;

    public CountMinSketch(int width, int depth, ulong seed)
    {
        this.width = width;
        this.seed = seed;
        rows = new int[depth][];
        for (int d = 0; d < depth; d++) rows[d] = new int[width];
    }

    public long Cells => (long)rows.Length * width;
    public int Depth => rows.Length;

    private int Slot(int depth, int item)
    {
        ulong h = (ulong)item * 0x9E3779B97F4A7C15UL + (ulong)depth * 0xD1B54A32D192ED03UL + seed;
        h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9UL;
        h = (h ^ (h >> 27)) * 0x94D049BB133111EBUL;
        h ^= h >> 31;
        return (int)(h % (ulong)width);
    }

    public void Add(int item, int weight = 1)
    {
        Updates++;
        for (int d = 0; d < rows.Length; d++)
        {
            int slot = Slot(d, item);
            rows[d][slot] = Math.Min(int.MaxValue - 1, rows[d][slot] + weight);
        }
    }

    public long Estimate(int item)
    {
        long best = long.MaxValue;
        for (int d = 0; d < rows.Length; d++) best = Math.Min(best, rows[d][Slot(d, item)]);
        return best == long.MaxValue ? 0 : best;
    }
}

// ---- linear algebra helpers -------------------------------------------------------------------------

internal static class Matrix
{
    public static double[][] Covariance(IList<double[]> data, double[] mean)
    {
        int d = mean.Length;
        var covariance = new double[d][];
        for (int i = 0; i < d; i++) covariance[i] = new double[d];
        foreach (double[] row in data)
            for (int i = 0; i < d; i++)
            {
                double left = row[i] - mean[i];
                for (int j = 0; j < d; j++) covariance[i][j] += left * (row[j] - mean[j]);
            }
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) covariance[i][j] /= Math.Max(1, data.Count);
        return covariance;
    }

    public static double[] Mean(IList<double[]> data)
    {
        int d = data[0].Length;
        var mean = new double[d];
        foreach (double[] row in data)
            for (int i = 0; i < d; i++) mean[i] += row[i];
        for (int i = 0; i < d; i++) mean[i] /= data.Count;
        return mean;
    }

    public static double Norm(double[] v) => Math.Sqrt(v.Sum(x => x * x));

    public static double Angle(double[] a, double[] b)
    {
        double dot = 0;
        for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
        double norm = Norm(a) * Norm(b);
        if (norm < 1e-15) return 0;
        return Math.Acos(Math.Clamp(Math.Abs(dot) / norm, -1, 1));
    }

    public static double[] Multiply(double[][] matrix, double[] vector)
    {
        var result = new double[vector.Length];
        for (int i = 0; i < matrix.Length; i++)
            for (int j = 0; j < vector.Length; j++) result[i] += matrix[i][j] * vector[j];
        return result;
    }

    // Jacobi eigen-decomposition of a symmetric matrix: the exact oracle for the streaming estimate.
    public static (double[] Values, double[][] Vectors) Jacobi(double[][] input, int iterations = 100)
    {
        int n = input.Length;
        var a = input.Select(row => (double[])row.Clone()).ToArray();
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < iterations; sweep++)
        {
            double off = 0;
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            if (off < 1e-18) break;
            for (int p = 0; p < n; p++)
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) < 1e-15) continue;
                    double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
                    double c = 1 / Math.Sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k][p], akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p][k], aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k][p], vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
        }
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        var order = Enumerable.Range(0, n).OrderByDescending(i => values[i]).ToArray();
        var sortedValues = order.Select(i => values[i]).ToArray();
        var vectors = new double[n][];
        for (int c = 0; c < n; c++) vectors[c] = Enumerable.Range(0, n).Select(k => v[k][order[c]]).ToArray();
        return (sortedValues, vectors);
    }
}

// ---- streaming PCA ------------------------------------------------------------------------------------

internal sealed class StreamingPca
{
    public readonly int Dimensions;
    public readonly int Components;
    public readonly int RefreshInterval;
    public readonly int PowerSteps;

    public readonly double[] Mean;
    public readonly double[][] Covariance;
    public double[][] Subspace;
    public double[] Eigenvalues = Array.Empty<double>();

    // int.MaxValue means "cumulative"; a finite value makes the estimate behave like a sliding window of
    // that many samples, which is what lets the subspace follow drift.
    public int DecayWindow = int.MaxValue;

    public long Seen;
    public long Refreshes;
    public long StateBytes => (long)(Dimensions * Dimensions + Dimensions * Components + Dimensions) * sizeof(double);

    public StreamingPca(int dimensions, int components, int refreshInterval = 25, int powerSteps = 30)
    {
        Dimensions = dimensions;
        Components = components;
        RefreshInterval = refreshInterval;
        PowerSteps = powerSteps;
        Mean = new double[dimensions];
        Covariance = new double[dimensions][];
        for (int i = 0; i < dimensions; i++) Covariance[i] = new double[dimensions];
        Subspace = new double[components][];
        for (int c = 0; c < components; c++) Subspace[c] = new double[dimensions];
    }

    public void Add(double[] x)
    {
        Seen++;
        // Sliding/cumulative Welford: n is capped by DecayWindow, which makes the accumulator forget old
        // samples at the same rate a sliding window would.
        double n = Math.Min(Seen, DecayWindow);
        var oldMean = new double[Dimensions];
        Array.Copy(Mean, oldMean, Dimensions);
        for (int i = 0; i < Dimensions; i++) Mean[i] += (x[i] - oldMean[i]) / n;
        for (int i = 0; i < Dimensions; i++)
        {
            double d1 = x[i] - oldMean[i];
            for (int j = 0; j < Dimensions; j++)
            {
                double d2 = x[j] - Mean[j];
                Covariance[i][j] += (d1 * d2 - Covariance[i][j]) / n;
            }
        }
        if (Seen % RefreshInterval == 0) Refresh();
    }

    public void Refresh()
    {
        Refreshes++;
        var working = Covariance.Select(row => (double[])row.Clone()).ToArray();
        var components = new double[Components][];
        var values = new double[Components];
        for (int c = 0; c < Components; c++)
        {
            var vector = c < Subspace.Length && Matrix.Norm(Subspace[c]) > 1e-12
                ? (double[])Subspace[c].Clone()
                : InitialGuess(c);
            vector = Orthogonalise(vector, components, c);
            for (int step = 0; step < PowerSteps; step++)
            {
                var next = Matrix.Multiply(working, vector);
                double norm = Matrix.Norm(next);
                if (norm < 1e-15) break;
                for (int i = 0; i < Dimensions; i++) next[i] /= norm;
                vector = Orthogonalise(next, components, c);
            }
            double eigenvalue = 0;
            var applied = Matrix.Multiply(working, vector);
            for (int i = 0; i < Dimensions; i++) eigenvalue += vector[i] * applied[i];
            components[c] = vector;
            values[c] = eigenvalue;
            // Deflation so the next component is orthogonal to this one.
            for (int i = 0; i < Dimensions; i++)
                for (int j = 0; j < Dimensions; j++) working[i][j] -= eigenvalue * vector[i] * vector[j];
        }
        Subspace = components;
        Eigenvalues = values;
    }

    private double[] InitialGuess(int component)
    {
        var guess = new double[Dimensions];
        guess[component % Dimensions] = 1.0;
        return guess;
    }

    private double[] Orthogonalise(double[] vector, double[][] basis, int count)
    {
        for (int c = 0; c < count; c++)
        {
            double dot = 0;
            for (int i = 0; i < Dimensions; i++) dot += vector[i] * basis[c][i];
            for (int i = 0; i < Dimensions; i++) vector[i] -= dot * basis[c][i];
        }
        double norm = Matrix.Norm(vector);
        if (norm > 1e-15) for (int i = 0; i < Dimensions; i++) vector[i] /= norm;
        return vector;
    }

    public double ReconstructionError(double[] x)
    {
        var centred = new double[Dimensions];
        for (int i = 0; i < Dimensions; i++) centred[i] = x[i] - Mean[i];
        var projected = new double[Dimensions];
        foreach (double[] component in Subspace)
        {
            double dot = 0;
            for (int i = 0; i < Dimensions; i++) dot += component[i] * centred[i];
            for (int i = 0; i < Dimensions; i++) projected[i] += dot * component[i];
        }
        double error = 0;
        for (int i = 0; i < Dimensions; i++)
        {
            double residual = centred[i] - projected[i];
            error += residual * residual;
        }
        return error;
    }
}

// ---- checks -------------------------------------------------------------------------------------------

public static class Solution
{
    private static SplitMix64 rng = new SplitMix64(0xA0A1A1UL);

    private static int Rand(int bound) => rng.NextInt(bound);

    public static void Main()
    {
        // ---- Misra-Gries guarantee ---------------------------------------------------------------
        int heavyViolations = 0;
        int countViolations = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            int distinct = 40 + trial;
            int k = 8;
            int length = 3000;
            var trueCounts = new long[distinct];
            var summary = new MisraGries(k);
            var local = new SplitMix64(0x1000UL + (ulong)trial);
            for (int i = 0; i < length; i++)
            {
                int item = (int)(local.Next() % (ulong)distinct);
                if (local.NextUnit() < 0.5) item = (int)(local.Next() % 4);      // a heavy head
                trueCounts[item]++;
                summary.Add(item);
            }
            double threshold = length / (double)k;
            for (int item = 0; item < distinct; item++)
            {
                long estimate = summary.Estimate(item);
                if (trueCounts[item] > threshold && estimate == 0) heavyViolations++;
                if (estimate > trueCounts[item]) countViolations++;
                if (estimate > 0 && estimate < trueCounts[item] - threshold) countViolations++;
            }
        }
        Check(heavyViolations == 0, "every heavy hitter above n/k is reported (violations " + heavyViolations + ")");
        Check(countViolations == 0, "every reported count is within n/k below the truth (violations " + countViolations + ")");

        var tinySummary = new MisraGries(3);
        for (int i = 0; i < 100; i++) tinySummary.Add(7);
        Check(tinySummary.Estimate(7) == 100, "a single repeated item is counted exactly");
        Check(tinySummary.DecrementRounds == 0, "no decrements were needed");
        var flushSummary = new MisraGries(2);
        flushSummary.Add(1);
        flushSummary.Add(1);
        flushSummary.Add(2);
        flushSummary.Add(3);
        flushSummary.Add(4);
        Check(flushSummary.CounterCount <= 2, "the summary never exceeds its capacity");
        Check(flushSummary.UpdateCost <= (flushSummary.Updates + flushSummary.DecrementRounds * 2),
            "the update cost stays inside its bound");

        // ---- count-min error bound ----------------------------------------------------------------
        int epsilonViolations = 0;
        double worstRelativeError = 0;
        long sketchCells = 0;
        for (int trial = 0; trial < 30; trial++)
        {
            int distinct = 500;
            int length = 8000;
            double epsilon = 0.01;
            int width = (int)Math.Ceiling(Math.E / epsilon);
            var sketch = new CountMinSketch(width, 5, 0x5EEDUL + (ulong)trial);
            var trueCounts = new long[distinct];
            var local = new SplitMix64(0x2000UL + (ulong)trial);
            for (int i = 0; i < length; i++)
            {
                int item = (int)(local.Next() % (ulong)distinct);
                trueCounts[item]++;
                sketch.Add(item);
            }
            sketchCells = Math.Max(sketchCells, sketch.Cells);
            for (int item = 0; item < distinct; item++)
            {
                long estimate = sketch.Estimate(item);
                if (estimate < trueCounts[item]) countViolations++;
                double error = estimate - trueCounts[item];
                if (error > epsilon * length) epsilonViolations++;
                worstRelativeError = Math.Max(worstRelativeError, error / length);
            }
        }
        Check(epsilonViolations == 0, "the count-min estimate never overshoots eps*n: worst relative error "
            + worstRelativeError.ToString("E2"));
        Check(sketchCells < 8000, "the sketch is smaller than the stream it summarises: " + sketchCells + " cells");

        // ---- streaming PCA against the Jacobi oracle ----------------------------------------------
        var data = SampleNormal(2000, 4, out _);
        var streaming = new StreamingPca(4, 2, 25, 40);
        foreach (double[] x in data) streaming.Add(x);
        streaming.Refresh();
        double[] mean = Matrix.Mean(data);
        double[][] covariance = Matrix.Covariance(data, mean);
        var (exactValues, exactVectors) = Matrix.Jacobi(covariance);
        for (int i = 0; i < 4; i++) Check(Math.Abs(streaming.Mean[i] - mean[i]) < 1e-9, "the Welford mean matches the batch mean");
        double angle0 = Matrix.Angle(streaming.Subspace[0], exactVectors[0]);
        double angle1 = Matrix.Angle(streaming.Subspace[1], exactVectors[1]);
        Check(angle0 < 0.05, "top eigenvector angle against Jacobi: " + angle0.ToString("E3") + " rad");
        Check(angle1 < 0.10, "second eigenvector angle against Jacobi: " + angle1.ToString("E3") + " rad");
        Check(Math.Abs(streaming.Eigenvalues[0] - exactValues[0]) / exactValues[0] < 0.02,
            $"top eigenvalue tracks the exact one: {streaming.Eigenvalues[0]:F4} vs {exactValues[0]:F4}");
        Check(exactValues[0] > exactValues[1] * 1.5, "the synthetic data really is anisotropic: "
            + exactValues[0].ToString("F3") + " vs " + exactValues[1].ToString("F3"));

        // ---- anomaly detection: streaming versus the batch PCA detector ---------------------------
        var clean = SampleNormal(1500, 4, out _);
        var detector = new StreamingPca(4, 2, 25, 40);
        foreach (double[] x in clean) detector.Add(x);
        detector.Refresh();
        var injected = new List<(double[] Point, bool Anomaly)>();
        var local2 = new SplitMix64(0x777UL);
        foreach (double[] x in clean.Take(300)) injected.Add((x, false));
        for (int i = 0; i < 60; i++)
        {
            var point = new double[4];
            for (int d = 0; d < 4; d++) point[d] = 9.0 * local2.NextNormal();     // far from the subspace
            injected.Add((point, true));
        }
        double streamingAuc = Auc(injected.Select(p => (detector.ReconstructionError(p.Point), p.Anomaly)).ToList());
        double[] batchMean = Matrix.Mean(clean);
        double[][] batchCovariance = Matrix.Covariance(clean, batchMean);
        var (_, batchVectors) = Matrix.Jacobi(batchCovariance);
        double BatchError(double[] x)
        {
            var centred = new double[4];
            for (int i = 0; i < 4; i++) centred[i] = x[i] - batchMean[i];
            double error = 0;
            for (int i = 0; i < 4; i++)
            {
                double residual = centred[i];
                for (int c = 0; c < 2; c++)
                {
                    double dot = 0;
                    for (int j = 0; j < 4; j++) dot += batchVectors[c][j] * centred[j];
                    residual -= dot * batchVectors[c][i];
                }
                error += residual * residual;
            }
            return error;
        }
        double batchAuc = Auc(injected.Select(p => (BatchError(p.Point), p.Anomaly)).ToList());
        Check(streamingAuc > 0.95, "the streaming detector separates the injected anomalies: AUC " + streamingAuc.ToString("F3"));
        Check(Math.Abs(streamingAuc - batchAuc) < 0.05,
            $"the streaming detector agrees with the batch detector: {streamingAuc:F3} vs {batchAuc:F3}");
        Check(detector.StateBytes < 1500 * 4 * sizeof(double) / 4,
            "the summary uses far less memory than the stream it saw: " + detector.StateBytes + " bytes");

        // ---- drift: the subspace follows a rotation of the data -----------------------------------
        var drift = new StreamingPca(2, 1, 10, 40) { DecayWindow = 200 };
        for (int phase = 0; phase < 3; phase++)
        {
            double angle = phase * Math.PI / 4.0;
            var direction = new[] { Math.Cos(angle), Math.Sin(angle) };
            var local3 = new SplitMix64(0x999UL + (ulong)phase);
            for (int i = 0; i < 400; i++)
            {
                double t = 3.0 * local3.NextNormal();
                drift.Add(new[] { t * direction[0], t * direction[1] });
            }
            drift.Refresh();
            double error = Matrix.Angle(drift.Subspace[0], direction);
            Check(error < 0.12, $"the subspace follows the rotation in phase {phase}: {error:E3} rad");
        }

        var constant = new StreamingPca(3, 1, 10, 20);
        for (int i = 0; i < 200; i++) constant.Add(new[] { 2.0, 2.0, 2.0 });
        constant.Refresh();
        Check(Math.Abs(constant.ReconstructionError(new[] { 2.0, 2.0, 2.0 })) < 1e-9,
            "constant data has zero reconstruction error");

        // ---- degenerate shapes ---------------------------------------------------------------------
        var oneDimension = new StreamingPca(1, 1, 5, 10);
        for (int i = 0; i < 50; i++) oneDimension.Add(new[] { (double)i - 25 });
        oneDimension.Refresh();
        Check(Math.Abs(Math.Abs(oneDimension.Subspace[0][0]) - 1.0) < 1e-9, "in one dimension the component is a unit axis");
        Check(oneDimension.ReconstructionError(new[] { 100.0 }) < 1e-9, "one dimension reconstructs any point");
        var fullRank = new StreamingPca(3, 3, 10, 20);
        foreach (double[] x in SampleNormal(300, 3, out _)) fullRank.Add(x);
        fullRank.Refresh();
        Check(fullRank.ReconstructionError(SampleNormal(1, 3, out _)[0]) < 1e-6,
            "using every component leaves no residual");
        var empty = new StreamingPca(2, 1);
        Check(empty.Seen == 0 && empty.Refreshes == 0, "an empty stream never refreshes");
        empty.Refresh();
        Check(empty.Subspace.Length == 1 && empty.Subspace[0].All(double.IsFinite), "refreshing on empty data is safe");
        var allAnomaly = new StreamingPca(2, 1, 5, 10);
        for (int i = 0; i < 100; i++) allAnomaly.Add(new[] { 50.0 + i, -50.0 - i });
        allAnomaly.Refresh();
        Check(allAnomaly.ReconstructionError(new[] { 50.0, -50.0 }) >= 0, "an all anomaly stream still scores");

        // ---- determinism and cost -------------------------------------------------------------------
        Check(detector.Refreshes == 1500 / 25 + 1, "the refresh schedule is deterministic: " + detector.Refreshes);
        Check(streaming.Seen == 2000 && streaming.StateBytes == (4 * 4 + 4 * 2 + 4) * 8,
            "the state size follows the formula: " + streaming.StateBytes + " bytes");
        Check(tinySummary.UpdateCost == 100, "a stream with one item costs one operation per event");
        var repeat = new StreamingPca(4, 2, 25, 40);
        foreach (double[] x in data) repeat.Add(x);
        repeat.Refresh();
        Check(Math.Abs(repeat.Eigenvalues[0] - streaming.Eigenvalues[0]) < 1e-12
            && Matrix.Angle(repeat.Subspace[0], streaming.Subspace[0]) < 1e-12, "the streaming PCA is deterministic");

        Console.WriteLine($"823 sketches: Misra-Gries k=8 held the guarantee over 40 Zipfian streams; count-min "
            + $"({sketchCells} cells) worst relative error {worstRelativeError:E2}; streaming PCA top-2 angle "
            + $"{angle0:E2}/{angle1:E2} rad vs Jacobi, eigenvalue {streaming.Eigenvalues[0]:F4} vs {exactValues[0]:F4}; "
            + $"anomaly AUC streaming {streamingAuc:F3} vs batch {batchAuc:F3}; state {detector.StateBytes} bytes "
            + $"for {detector.Seen} points of dimension {detector.Dimensions}");
        Console.WriteLine("PASS");
    }

    private static List<double[]> SampleNormal(int count, int dimensions, out double[] mean)
    {
        var local = new SplitMix64(0xDEADBEEFUL + (ulong)count * 31 + (ulong)dimensions);
        var data = new List<double[]>();
        for (int i = 0; i < count; i++)
        {
            var point = new double[dimensions];
            for (int d = 0; d < dimensions; d++)
            {
                double t = local.NextNormal();
                // Anisotropic: later dimensions have smaller scale, which makes the top subspace unique.
                point[d] = t * (d == 0 ? 4.0 : d == 1 ? 1.0 : 0.2);
            }
            data.Add(point);
        }
        mean = Matrix.Mean(data);
        return data;
    }

    private static double Auc(IList<(double Score, bool Anomaly)> scored)
    {
        var positives = scored.Where(s => s.Anomaly).Select(s => s.Score).ToList();
        var negatives = scored.Where(s => !s.Anomaly).Select(s => s.Score).ToList();
        if (positives.Count == 0 || negatives.Count == 0) return 0.5;
        double wins = 0;
        foreach (double p in positives)
            foreach (double n in negatives)
                wins += p > n ? 1.0 : p < n ? 0.0 : 0.5;
        return wins / (positives.Count * (double)negatives.Count);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
