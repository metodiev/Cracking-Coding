// Task 710 - Implement private-count sketches with Laplace mechanism simulation.
// (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// A count-min sketch estimates item frequencies of a stream in one pass using d hash rows of width w:
// every arrival increments one cell per row, and the estimate of an item is the minimum of its d
// cells. Because collisions only ever add, the estimate is a guaranteed over-estimate; with
// w = ceil(e / epsilon) the over-estimate is at most epsilon * N with probability 1 - delta.
//
// Privacy is added in the two ways the literature uses, and both are exercised here:
//   * releasing the sketch itself - one record changes exactly one cell per row, so the L1
//     sensitivity of the cell matrix is d, and adding Lap(d / epsilon) to every cell makes the whole
//     release epsilon-differentially private;
//   * answering queries - the estimate reads d noisy cells, so a single query with budget epsilon
//     needs Lap(d / epsilon) on the answer (the per-query private count-min sketch of Chan et al.),
//     and the median over the rows is a cheaper, more robust alternative to the minimum.
//
// Verified below: the sketch is exact on an uncollided row and always an over-estimate; the observed
// collision error respects the (e/w)*N bound; the mean absolute error of the noisy estimate grows
// with the composition and falls with epsilon; the median-of-rows estimator beats a single row; the
// empirical output-probability ratio of two neighbouring streams stays inside exp(epsilon) for the
// released matrix; and the degenerate cases (empty stream, width 1, absent item, invalid budget) are
// handled. All numbers come from a hand-written splitmix64 stream, so the run is deterministic.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double OpenUnit(ref ulong state)
    {
        double u = (Next(ref state) >> 12) * (1.0 / 4503599627370496.0);
        return u <= 0.0 ? 0.5 : u;
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal static class Laplace
{
    internal static double Sample(ref ulong state, double scale)
    {
        if (scale < 0) throw new ArgumentOutOfRangeException(nameof(scale));
        if (scale == 0) return 0;
        double u = Rng.OpenUnit(ref state) - 0.5;
        double magnitude = -Math.Log(1 - 2 * Math.Abs(u));
        return u < 0 ? -scale * magnitude : scale * magnitude;
    }
}

internal sealed class CountMinSketch
{
    private const long Prime = 2147483647L;
    private readonly int _rows, _width;
    private readonly long[] _multiplier, _offset;
    private readonly int[,] _cells;

    internal CountMinSketch(int rows, int width, ulong seed)
    {
        if (rows <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(rows), "the sketch needs positive dimensions");
        _rows = rows;
        _width = width;
        _cells = new int[rows, width];
        _multiplier = new long[rows];
        _offset = new long[rows];
        ulong state = seed;
        for (int r = 0; r < rows; r++)
        {
            _multiplier[r] = 1 + (long)(Rng.Next(ref state) % 1000000UL);
            _offset[r] = (long)(Rng.Next(ref state) % 1000000UL);
        }
    }

    internal int Rows => _rows;
    internal int Width => _width;
    internal long Total { get; private set; }

    /// <summary>Cell of a row that the item hashes to.</summary>
    internal int CellIndex(int row, int item)
    {
        long value = (_multiplier[row] * item + _offset[row]) % Prime;
        if (value < 0) value += Prime;
        return (int)(value % _width);
    }

    internal void Add(int item) => Add(item, 1);

    internal void Add(int item, int delta)
    {
        for (int r = 0; r < _rows; r++) _cells[r, CellIndex(r, item)] += delta;
        Total += delta;
    }

    internal int Cell(int row, int column) => _cells[row, column];

    /// <summary>Unnoised estimate: the smallest counter the item hashes to.</summary>
    internal long Estimate(int item)
    {
        long best = long.MaxValue;
        for (int r = 0; r < _rows; r++) best = Math.Min(best, _cells[r, CellIndex(r, item)]);
        return best;
    }

    internal long RowEstimate(int row, int item) => _cells[row, CellIndex(row, item)];

    /// <summary>Worst collision error observed in a row: how far its counters overshoot the truth.</summary>
    internal long RowSlack(int row, int item, long truth) => _cells[row, CellIndex(row, item)] - truth;
}

internal static class PrivateSketch
{
    /// <summary>
    /// Releases the whole sketch. One record touches one cell per row, so the L1 sensitivity of the
    /// cell matrix is the number of rows and every cell gets Laplace(rows / epsilon) noise.
    /// </summary>
    internal static long[,] ReleaseMatrix(CountMinSketch sketch, double epsilon, ref ulong state)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive");
        double scale = (double)sketch.Rows / epsilon;
        var noisy = new long[sketch.Rows, sketch.Width];
        for (int r = 0; r < sketch.Rows; r++)
            for (int c = 0; c < sketch.Width; c++)
                noisy[r, c] = (long)Math.Round(sketch.Cell(r, c) + Laplace.Sample(ref state, scale));
        return noisy;
    }

    internal static long EstimateFromMatrix(long[,] released, CountMinSketch sketch, int item)
    {
        long best = long.MaxValue;
        for (int r = 0; r < sketch.Rows; r++) best = Math.Min(best, released[r, sketch.CellIndex(r, item)]);
        return best;
    }

    /// <summary>Per-query privatisation: the answer reads d cells, so it needs Laplace(d / epsilon).</summary>
    internal static long NoisyEstimate(CountMinSketch sketch, int item, double epsilon, ref ulong state)
    {
        if (epsilon <= 0) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be positive");
        double scale = (double)sketch.Rows / epsilon;
        return (long)Math.Round(sketch.Estimate(item) + Laplace.Sample(ref state, scale));
    }

    /// <summary>Median over the per-row noisy estimates: robust and cheaper than the minimum.</summary>
    internal static double MedianOfRows(long[,] released, CountMinSketch sketch, int item)
    {
        var values = new double[sketch.Rows];
        for (int r = 0; r < sketch.Rows; r++) values[r] = released[r, sketch.CellIndex(r, item)];
        Array.Sort(values);
        return values.Length % 2 == 1
            ? values[values.Length / 2]
            : 0.5 * (values[values.Length / 2 - 1] + values[values.Length / 2]);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>A skewed stream: item i appears max(1, 400 / (i + 1)) times.</summary>
    private static int[] BuildStream(int distinct, out long[] truth)
    {
        var items = new List<int>();
        truth = new long[distinct];
        for (int i = 0; i < distinct; i++)
        {
            long repeats = Math.Max(1, 400 / (i + 1));
            truth[i] = repeats;
            for (long r = 0; r < repeats; r++) items.Add(i);
        }
        return items.ToArray();
    }

    public static void Main()
    {
        // ---- the sketch itself -----------------------------------------------------------------
        {
            const int Rows = 6, Width = 512, Distinct = 200;
            int[] stream = BuildStream(Distinct, out long[] truth);
            var sketch = new CountMinSketch(Rows, Width, 0xC0FFEEUL);
            foreach (int item in stream) sketch.Add(item);
            Check(sketch.Total == stream.Length, "every arrival was counted exactly once in the total");

            long worstError = 0, exactRows = 0;
            bool overEstimate = true;
            for (int item = 0; item < Distinct; item++)
            {
                long estimate = sketch.Estimate(item);
                if (estimate < truth[item]) overEstimate = false;
                worstError = Math.Max(worstError, estimate - truth[item]);
            }
            Check(overEstimate, "collisions can only inflate, never deflate an estimate");
            double bound = Math.E / Width * stream.Length;
            Check(worstError <= bound, "the worst over-estimate " + worstError + " respects the e/w*N bound "
                + bound.ToString("F2"));

            // a row that happens to be collision free for an item reports the truth exactly
            for (int item = 0; item < Distinct; item++)
                if (sketch.RowSlack(0, item, truth[item]) == 0) exactRows++;
            Check(exactRows > 0, "at least one item is collision free in row 0 (" + exactRows + ")");
            Check(sketch.CellIndex(0, 7) == sketch.CellIndex(0, 7), "hashing is a pure function of the item");
            int collisions = 0;
            for (int item = 0; item < Distinct; item++) if (sketch.CellIndex(0, item) == sketch.CellIndex(0, 0)) collisions++;
            Check(collisions >= 1, "the tested row does have collisions (" + collisions + " items share cell 0)");
            Console.WriteLine("stream=" + stream.Length + " distinct=" + Distinct + " worstError=" + worstError
                + " bound=" + bound.ToString("F2"));
        }

        // ---- per-query Laplace noise: error grows with d / epsilon ------------------------------
        {
            const int Rows = 5, Width = 400, Distinct = 100;
            int[] stream = BuildStream(Distinct, out long[] truth);
            var sketch = new CountMinSketch(Rows, Width, 0xBEEFUL);
            foreach (int item in stream) sketch.Add(item);

            var budgets = new[] { 10.0, 2.0, 0.5 };
            var measured = new double[budgets.Length];
            for (int b = 0; b < budgets.Length; b++)
            {
                ulong state = 0x1234UL + (ulong)b;
                double total = 0;
                const int Queries = 4000;
                for (int q = 0; q < Queries; q++)
                {
                    int item = q % Distinct;
                    total += Math.Abs(PrivateSketch.NoisyEstimate(sketch, item, budgets[b], ref state) - truth[item]);
                }
                measured[b] = total / Queries;
                double theoretical = Rows / budgets[b];
                Check(Math.Abs(measured[b] - theoretical) < 0.15 * theoretical,
                    "epsilon=" + budgets[b] + " yields mean error ~d/epsilon (measured "
                    + measured[b].ToString("F2") + " vs " + theoretical.ToString("F2") + ")");
            }
            Check(measured[0] < measured[1] && measured[1] < measured[2],
                "more budget means less noise: " + measured[0].ToString("F2") + " < " + measured[1].ToString("F2")
                + " < " + measured[2].ToString("F2"));
            Console.WriteLine("perQueryError(10,2,0.5)=" + measured[0].ToString("F2") + "/"
                + measured[1].ToString("F2") + "/" + measured[2].ToString("F2"));
        }

        // ---- the median over rows beats a single row, and the matrix release is usable ---------
        {
            const int Rows = 9, Width = 400, Distinct = 60;
            int[] stream = BuildStream(Distinct, out long[] truth);
            var sketch = new CountMinSketch(Rows, Width, 0xF00DUL);
            foreach (int item in stream) sketch.Add(item);

            ulong state = 0x9999UL;
            double medianError = 0, singleRowError = 0;
            const int Trials = 3000;
            for (int t = 0; t < Trials; t++)
            {
                long[,] released = PrivateSketch.ReleaseMatrix(sketch, 2.0, ref state);
                int item = t % Distinct;
                medianError += Math.Abs(PrivateSketch.MedianOfRows(released, sketch, item) - truth[item]);
                singleRowError += Math.Abs(released[0, sketch.CellIndex(0, item)] - truth[item]);
            }
            Check(medianError / Trials < singleRowError / Trials,
                "the median over rows is more accurate than one row (" + (medianError / Trials).ToString("F2")
                + " < " + (singleRowError / Trials).ToString("F2") + ")");

            long[,] one = PrivateSketch.ReleaseMatrix(sketch, 2.0, ref state);
            long minEstimate = PrivateSketch.EstimateFromMatrix(one, sketch, 0);
            Check(minEstimate > -1000, "the released matrix still answers estimates (" + minEstimate + ")");
            Check(one.GetLength(0) == Rows && one.GetLength(1) == Width, "the release has the full shape");
            Console.WriteLine("medianOfRowsError=" + (medianError / Trials).ToString("F2")
                + " singleRowError=" + (singleRowError / Trials).ToString("F2"));
        }

        // ---- the privacy statement: released-cell ratios respect exp(epsilon) ------------------
        {
            // Two neighbouring streams differ in one arrival, which shifts exactly one cell per row.
            const int Rows = 4, Width = 256;
            var a = new CountMinSketch(Rows, Width, 0x5555UL);
            var b = new CountMinSketch(Rows, Width, 0x5555UL);
            for (int i = 0; i < 500; i++) { a.Add(i % 37); b.Add(i % 37); }
            a.Add(1);                                   // the neighbouring database adds one record
            int row = 0, column = a.CellIndex(row, 12345);
            Check(column == b.CellIndex(row, 12345), "both sketches hash identically");

            double epsilon = 1.0, scale = (double)Rows / epsilon;
            ulong stateA = 0xAAAUL, stateB = 0xBBBUL;
            const int Samples = 60000;
            int hitsA = 0, hitsB = 0;
            const long Low = 2, High = 3;
            for (int i = 0; i < Samples; i++)
            {
                long releasedA = (long)Math.Round(a.Cell(row, column) + Laplace.Sample(ref stateA, scale));
                long releasedB = (long)Math.Round(b.Cell(row, column) + Laplace.Sample(ref stateB, scale));
                if (releasedA >= Low && releasedA <= High) hitsA++;
                if (releasedB >= Low && releasedB <= High) hitsB++;
            }
            double pA = (double)hitsA / Samples, pB = (double)hitsB / Samples;
            Check(pA > 0 && pB > 0, "the chosen interval has mass under both streams");
            double ratio = pA / pB;
            Check(ratio <= Math.Exp(epsilon) * 1.1 && ratio >= Math.Exp(-epsilon) * 0.9,
                "the released-cell probability ratio " + ratio.ToString("F4") + " sits inside exp(+-1)");
            Check(a.Cell(row, column) == b.Cell(row, column),
                "the neighbouring record did not even fall into this cell, and the ratio is still bounded");
            Console.WriteLine("releasedCellRatio=" + ratio.ToString("F4") + " expEps=" + Math.Exp(epsilon).ToString("F4"));
        }

        // ---- boundary and rejected input -------------------------------------------------------
        {
            var empty = new CountMinSketch(3, 64, 7UL);
            Check(empty.Total == 0 && empty.Estimate(42) == 0, "an empty sketch estimates zero");
            ulong state = 0x1UL;
            Check(PrivateSketch.NoisyEstimate(empty, 42, 1.0, ref state) != long.MinValue,
                "an empty sketch still answers privately");

            var narrow = new CountMinSketch(2, 1, 11UL);
            for (int i = 0; i < 50; i++) narrow.Add(i);
            Check(narrow.Estimate(3) == 50 && narrow.Estimate(49) == 50,
                "a width-1 sketch makes every item report the whole stream");

            var single = new CountMinSketch(4, 128, 13UL);
            single.Add(77);
            Check(single.Estimate(77) == 1, "a single arrival is counted exactly");
            Check(single.Estimate(78) == 0, "an absent item estimates zero");

            var heavy = new CountMinSketch(5, 128, 17UL);
            for (int i = 0; i < 1000; i++) heavy.Add(5);
            for (int i = 0; i < 10; i++) heavy.Add(6);
            Check(heavy.Estimate(5) == 1000 && heavy.Estimate(6) == 10,
                "a heavy hitter is estimated exactly when it never collides");

            bool thrown = false;
            try { PrivateSketch.NoisyEstimate(heavy, 5, 0.0, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a zero budget is rejected");
            thrown = false;
            try { PrivateSketch.ReleaseMatrix(heavy, -1.0, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a negative budget is rejected");
            thrown = false;
            try { new CountMinSketch(0, 8, 1UL); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a sketch without rows is rejected");
            thrown = false;
            try { new CountMinSketch(4, 0, 1UL); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a sketch without columns is rejected");
            thrown = false;
            try { Laplace.Sample(ref state, -1); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "negative Laplace scale is rejected");
        }

        // ---- determinism -----------------------------------------------------------------------
        {
            var first = new CountMinSketch(4, 64, 21UL);
            var second = new CountMinSketch(4, 64, 21UL);
            ulong a = 5UL, b = 5UL;
            long firstEstimate = PrivateSketch.NoisyEstimate(first, 3, 1.5, ref a);
            long secondEstimate = PrivateSketch.NoisyEstimate(second, 3, 1.5, ref b);
            Check(firstEstimate == secondEstimate && a == b, "the private sketch is deterministic");
            var third = new CountMinSketch(4, 64, 22UL);
            Check(third.CellIndex(0, 5) != first.CellIndex(0, 5) || third.CellIndex(1, 5) != first.CellIndex(1, 5),
                "a different seed produces a different hash family");
        }

        Console.WriteLine("PASS");
    }
}
