// Task 718 - Implement gradient compression techniques (top-k) and error feedback. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Top-k sparsification keeps the k largest-magnitude coordinates of a gradient and ships them as
// (index, value) pairs; the rest are dropped. On its own that is a *biased* compressor: a coordinate
// that rarely makes the cut is never transmitted, so the cumulative update drifts away from the true
// gradient sum without bound. Error feedback (Seide et al., 2014) fixes this by keeping the dropped
// part as a local residual e: each step compresses x = g + e instead of g and then sets e' = x - c.
// Because c_i is either exactly 0 or exactly x_i, the subtraction x_i - c_i is *exact* in IEEE
// arithmetic, which yields the telescoping invariant this file is built around:
//
//     sum_t c_t = sum_t g_t - e_T        (coordinate-wise, up to summation roundoff)
//
// so every dropped component is remembered and retried on the next step, and the residual stays
// bounded by a single truncation step instead of growing with the step count.
//
// Verified here: canonical (index-sorted, deduplicated) support, exact sparsity, the k = 0 / k >= n /
// empty / single-element boundaries and a rejected negative k; a brute-force oracle that enumerates
// all C(n,k) supports of a tiny vector and must not beat the chosen one; the per-step identity
// "compressed + residual == x" bit-for-bit; the telescoping invariant; and the bias experiment in a
// fixed random direction, where the no-feedback error grows linearly with the step count while the
// feedback error stays at the size of one truncation step. All randomness comes from splitmix64.
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

/// <summary>A compressed gradient: k (index, value) pairs with strictly increasing indices.</summary>
internal sealed class SparseGradient
{
    internal const int BytesPerEntry = 4 + 8;   // int32 index + float64 value

    internal readonly int[] Indices;
    internal readonly double[] Values;

    internal SparseGradient(int[] indices, double[] values)
    {
        Indices = indices;
        Values = values;
    }

    internal int Count => Indices.Length;

    internal int Bytes => Count * BytesPerEntry;

    internal double ValueAt(int index)
    {
        for (int j = 0; j < Indices.Length; j++) if (Indices[j] == index) return Values[j];
        return 0.0;
    }

    /// <summary>Squared L2 distance between the dense expansion of this sparse vector and a reference.</summary>
    internal double SquaredError(IReadOnlyList<double> reference)
    {
        double sum = 0.0;
        int next = 0;
        for (int i = 0; i < reference.Count; i++)
        {
            double kept = next < Indices.Length && Indices[next] == i ? Values[next++] : 0.0;
            double d = reference[i] - kept;
            sum += d * d;
        }
        return sum;
    }

    internal void AddTo(double[] target)
    {
        for (int j = 0; j < Indices.Length; j++) target[Indices[j]] += Values[j];
    }

    internal bool StrictlyIncreasingIndices()
    {
        for (int j = 1; j < Indices.Length; j++) if (Indices[j] <= Indices[j - 1]) return false;
        return true;
    }

    internal bool IndicesWithin(int size)
    {
        foreach (int i in Indices) if (i < 0 || i >= size) return false;
        return true;
    }
}

internal static class TopK
{
    /// <summary>
    /// The k largest-magnitude entries of <paramref name="vector"/>, ties broken by ascending index so
    /// that the support is canonical, returned with the indices sorted ascending.
    /// </summary>
    internal static SparseGradient Compress(IReadOnlyList<double> vector, int k)
    {
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k), "k must be non-negative");
        int n = vector.Count;
        int take = Math.Min(k, n);
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            double ma = Math.Abs(vector[a]);
            double mb = Math.Abs(vector[b]);
            if (ma != mb) return ma > mb ? -1 : 1;
            return a.CompareTo(b);
        });
        Array.Sort(order, 0, take, Comparer<int>.Create((a, b) => a.CompareTo(b)));
        var indices = new int[take];
        var values = new double[take];
        for (int j = 0; j < take; j++)
        {
            indices[j] = order[j];
            values[j] = vector[order[j]];
        }
        return new SparseGradient(indices, values);
    }
}

/// <summary>Top-k with a local error-feedback residual: e' = (g + e) - topk(g + e).</summary>
internal sealed class ErrorFeedbackCompressor
{
    private readonly double[] _residual;

    internal ErrorFeedbackCompressor(int size, int k)
    {
        if (size < 0) throw new ArgumentOutOfRangeException(nameof(size));
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k));
        _residual = new double[size];
        K = k;
    }

    internal int K { get; }
    internal int Size => _residual.Length;
    internal long Steps { get; private set; }

    internal double[] ResidualCopy() => (double[])_residual.Clone();

    internal double ResidualNorm()
    {
        double sum = 0.0;
        for (int i = 0; i < _residual.Length; i++) sum += _residual[i] * _residual[i];
        return Math.Sqrt(sum);
    }

    internal SparseGradient Step(IReadOnlyList<double> gradient)
    {
        if (gradient.Count != _residual.Length) throw new ArgumentException("size mismatch", nameof(gradient));
        var x = new double[_residual.Length];
        for (int i = 0; i < x.Length; i++) x[i] = gradient[i] + _residual[i];
        SparseGradient compressed = TopK.Compress(x, K);
        int next = 0;
        for (int i = 0; i < x.Length; i++)
        {
            // x_i - c_i is exact: c_i is either x_i itself or 0, and both subtractions are exact.
            _residual[i] = next < compressed.Count && compressed.Indices[next] == i
                ? x[i] - compressed.Values[next]
                : x[i];
            if (_residual[i] == 0.0) _residual[i] = 0.0;   // normalise -0.0 so checksums stay stable
            if (next < compressed.Count && compressed.Indices[next] == i) next++;
        }
        Steps++;
        return compressed;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- boundaries: empty input, k = 0, k = n, k > n, single element, negative k --------------
        {
            Check(TopK.Compress(Array.Empty<double>(), 3).Count == 0, "an empty gradient compresses to nothing");
            var v = new[] { 1.0, -2.0, 3.0 };
            Check(TopK.Compress(v, 0).Count == 0, "k = 0 drops every coordinate");
            Check(TopK.Compress(v, 3).SquaredError(v) == 0.0, "k = n is lossless");
            Check(TopK.Compress(v, 99).Count == 3, "k > n is clamped to n");
            var single = TopK.Compress(new[] { -0.5 }, 1);
            Check(single.Count == 1 && single.Values[0] == -0.5, "a single coordinate is transmitted verbatim");
            bool threw = false;
            try { TopK.Compress(v, -1); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a negative k is rejected instead of silently clamped");
        }

        // ---- canonical support, ties, and magnitude (not sign) ordering ---------------------------
        {
            var ties = new[] { 1.0, -1.0, 0.5, -1.0 };
            SparseGradient pick = TopK.Compress(ties, 2);
            Check(pick.Indices[0] == 0 && pick.Indices[1] == 1, "equal magnitudes break ties by ascending index");
            Check(pick.StrictlyIncreasingIndices() && pick.IndicesWithin(4), "the support is sorted and in range");
            SparseGradient signed = TopK.Compress(new[] { -4.0, 0.2, -0.1, 3.0 }, 2);
            Check(signed.Count == 2 && signed.ValueAt(0) == -4.0 && signed.ValueAt(3) == 3.0,
                "magnitude, not signed value, decides the survivors");
            SparseGradient zeros = TopK.Compress(new[] { 0.0, 0.0, 0.0 }, 2);
            Check(zeros.Count == 2 && zeros.SquaredError(new[] { 0.0, 0.0, 0.0 }) == 0.0,
                "an all-zero gradient still occupies exactly k slots");
        }

        // ---- brute-force oracle: no subset of size k may beat the chosen support -------------------
        {
            var v = new[] { 0.3, -1.7, 2.1, 0.9, -0.4, 1.2 };
            const int k = 2;
            SparseGradient picked = TopK.Compress(v, k);
            double bestError = double.PositiveInfinity;
            int bestMask = -1;
            for (int mask = 0; mask < 1 << v.Length; mask++)
            {
                if (PopCount(mask) != k) continue;
                double err = 0.0;
                for (int i = 0; i < v.Length; i++) if (((mask >> i) & 1) == 0) err += v[i] * v[i];
                if (err < bestError - 1e-15) { bestError = err; bestMask = mask; }
            }
            int support = 0;
            for (int j = 0; j < picked.Count; j++) support |= 1 << picked.Indices[j];
            Check(support == bestMask, "the support is the brute-force argmin over all C(n,k) supports");
            Check(Math.Abs(picked.SquaredError(v) - bestError) < 1e-18, "its error equals the enumerated optimum");
        }

        // ---- the feedback experiment: a fixed random direction, T scaled steps --------------------
        const int n = 32, k2 = 4, T = 200;
        const ulong seed = 0xC0FFEE123456789UL;
        var direction = new double[n];
        var stream = new double[T][];
        {
            var rng = new SplitMix64(seed);
            for (int i = 0; i < n; i++) direction[i] = 0.4 + rng.NextUnit();   // distinct, positive
            for (int t = 0; t < T; t++)
            {
                double s = 1.0 + 0.05 * rng.NextSigned();
                var g = new double[n];
                for (int i = 0; i < n; i++) g[i] = s * direction[i];
                stream[t] = g;
            }
        }

        var sumGradients = new double[n];
        var sumWithFeedback = new double[n];
        var sumWithoutFeedback = new double[n];
        var compressor = new ErrorFeedbackCompressor(n, k2);
        var seenWith = new HashSet<string>();
        var seenWithout = new HashSet<string>();
        bool identityHeld = true, sparsityHeld = true;
        double maxStepError = 0.0, maxResidual = 0.0, errorWithoutAt50 = 0.0, errorWithAt50 = 0.0;
        for (int t = 0; t < T; t++)
        {
            double[] g = stream[t];
            for (int i = 0; i < n; i++) sumGradients[i] += g[i];

            double[] residualBefore = compressor.ResidualCopy();
            SparseGradient c = compressor.Step(g);
            double[] residualAfter = compressor.ResidualCopy();
            c.AddTo(sumWithFeedback);
            for (int i = 0; i < n; i++)
            {
                double x = g[i] + residualBefore[i];
                if (c.ValueAt(i) + residualAfter[i] != x) identityHeld = false;   // bit-exact, no tolerance
            }
            maxStepError = Math.Max(maxStepError, Math.Sqrt(c.SquaredError(g)));
            maxResidual = Math.Max(maxResidual, compressor.ResidualNorm());
            if (c.Count != k2 || !c.StrictlyIncreasingIndices()) sparsityHeld = false;
            seenWith.Add(Support(c));

            SparseGradient plain = TopK.Compress(g, k2);
            plain.AddTo(sumWithoutFeedback);
            seenWithout.Add(Support(plain));
            if (t == 49)
            {
                errorWithoutAt50 = Norm(Diff(sumWithoutFeedback, sumGradients));
                errorWithAt50 = Norm(Diff(sumWithFeedback, sumGradients));
            }
        }
        Check(identityHeld, "compressed + residual reproduces x bit-for-bit at every step");
        Check(sparsityHeld, "every transmitted message carries exactly k sorted coordinates");
        Check(compressor.Steps == T, "the compressor really ran T steps");
        Check(seenWithout.Count == 1, "without feedback the same k coordinates are retransmitted forever");

        double errorWith = Norm(Diff(sumWithFeedback, sumGradients));
        double errorWithout = Norm(Diff(sumWithoutFeedback, sumGradients));
        double directionNorm = Norm(direction);
        double gradientNorm = Norm(sumGradients);
        Console.WriteLine("withFeedbackBias=" + errorWith.ToString("G6") + " noFeedbackBias=" + errorWithout.ToString("G6")
            + " ratio=" + (errorWithout / errorWith).ToString("G6") + " maxResidual=" + maxResidual.ToString("G6")
            + " distinctSupports=" + seenWith.Count + "/" + seenWithout.Count
            + " bias50=" + errorWithAt50.ToString("G6") + "/" + errorWithoutAt50.ToString("G6")
            + " dirNorm=" + directionNorm.ToString("G6"));

        Check(errorWith < 0.05 * gradientNorm,
            "the cumulative bias is under 5% of the transmitted update (" + errorWith + " vs " + gradientNorm + ")");
        Check(errorWithout > 10.0 * errorWith, "without feedback the bias is at least an order larger");
        Check(errorWithout > 3.0 * errorWithoutAt50,
            "the no-feedback bias grows with the number of steps (" + errorWithoutAt50 + " -> " + errorWithout + ")");
        Check(errorWith <= 1.6 * errorWithAt50,
            "the feedback bias is a steady state, not a growing quantity (" + errorWithAt50 + " -> " + errorWith + ")");
        Check(seenWith.Count > 1, "error feedback rotates which coordinates are transmitted");
        Check(maxResidual <= 0.10 * gradientNorm, "the residual stays a small fraction of the total update");
        Check(maxStepError <= 3.0 * maxResidual + 1e-12,
            "one compression step never moves more than a few residuals away from the gradient");

        // ---- compression accounting and reproducibility -------------------------------------------
        Check(T * k2 * SparseGradient.BytesPerEntry < T * n * 8, "the compressed stream is smaller than the dense one");
        var second = new ErrorFeedbackCompressor(n, k2);
        var secondSum = new double[n];
        for (int t = 0; t < T; t++) second.Step(stream[t]).AddTo(secondSum);
        bool reproducible = true;
        for (int i = 0; i < n; i++)
            if (BitConverter.DoubleToInt64Bits(secondSum[i]) != BitConverter.DoubleToInt64Bits(sumWithFeedback[i]))
                reproducible = false;
        Check(reproducible, "the same stream reproduces the same accumulated update bit-for-bit");
        Console.WriteLine("steps=" + T + " k=" + k2 + " n=" + n + " bytesCompressed=" + T * k2 * SparseGradient.BytesPerEntry
            + " bytesDense=" + T * n * 8);

        Console.WriteLine("PASS");
    }

    private static string Support(SparseGradient g)
    {
        var sb = new System.Text.StringBuilder();
        foreach (int i in g.Indices) sb.Append(i).Append(',');
        return sb.ToString();
    }

    private static double[] Diff(double[] a, double[] b)
    {
        var d = new double[a.Length];
        for (int i = 0; i < a.Length; i++) d[i] = a[i] - b[i];
        return d;
    }

    private static double Norm(double[] v)
    {
        double sum = 0.0;
        for (int i = 0; i < v.Length; i++) sum += v[i] * v[i];
        return Math.Sqrt(sum);
    }

    private static int PopCount(int mask)
    {
        int c = 0;
        while (mask != 0) { c += mask & 1; mask >>= 1; }
        return c;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
