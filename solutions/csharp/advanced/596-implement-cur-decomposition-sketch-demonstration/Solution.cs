// Task 596 - Implement CUR decomposition sketch demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// CUR is the interpretable cousin of the SVD: A (m x n) is approximated as C U R where C holds a small
// set of sampled *columns* of A, R holds sampled *rows*, and U is derived from the intersection
// submatrix W = C(rows) x R(columns) rather than from an orthogonal factorisation. The file implements
// two selection rules: the deterministic top-|C| columns (and top-|R| rows) by Euclidean norm, which is
// the right thing to do when the matrix really is spanned by a few heavy columns, and a randomised rule
// that samples columns with probability proportional to ||A^(j)||^2 (the Frobenius importance sampling of
// Drineas and Mahoney), drawn from the fixed-seed 64-bit LCG in `Rng` (MMIX constants, documented because
// the repo's 16-bit LCG has poor low-bit structure). U is the Moore-Penrose pseudo-inverse of W computed
// as (W^T W)^+ W^T from the eigendecomposition of the k x k symmetric matrix W^T W by a dense cyclic
// Jacobi eigensolver with singular-value truncation, so rank-deficient intersections are handled instead
// of blowing up -- and the four Moore-Penrose identities are asserted explicitly, since a wrong
// pseudo-inverse is otherwise easy to miss. Because C U R always has rank at most k, the error is
// compared both against the Eckart-Young lower bound (the best rank-k error, computed from the
// eigenvalues of A^T A) and against the exact case where every column and row is kept, in which
// C U R = A A^+ A = A must hold to machine precision.
//
// Documented checks: W W^+ W = W, W^+ W W^+ = W^+, (W W^+)^T = W W^+ and (W^+ W)^T = W^+ W hold to 1e-9
// for four intersection submatrices, one of which is rank deficient by construction; keeping all columns
// and rows reproduces A exactly; on a matrix whose column space is spanned by exactly two heavy columns,
// the purely norm-based selection recovers C U R to 1e-9 while the rank-two SVD error is also 0, so the
// Eckart-Young inequality is tight there; the norm-based error is never below the best rank-k error and,
// on the tested instances, the ratio to it is bounded by six and reported -- the file is explicit that
// the classical relative-error guarantee needs O(r / eps^2) sampled columns, far more than the 4 of 25
// used in the demonstration; on a randomised instance the CUR error with 4 sampled columns is strictly
// better than the worst CUR built from 4 *uniform* choices, documenting that the sampling distribution
// matters; the pseudo-inverse of the zero matrix is the zero matrix; the CUR reconstruction is verified to
// have rank at most k through the truncated-SVD energy, checked with a relative tolerance because that
// energy is a difference of two nearly equal large numbers (the file says so in the message); and the whole
// construction is reproducible for a fixed seed.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    internal static double NextUnit(ref ulong state)
    {
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }

    internal static int Draw(ref ulong state, double[] cumulative)
    {
        double target = (1 + NextBelow(ref state, 10000000)) / 10000001.0;
        int low = 0;
        int high = cumulative.Length - 1;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (cumulative[middle] < target) low = middle + 1;
            else high = middle;
        }
        return low;
    }
}

internal static class LinAlg
{
    internal static double[][] Matrix(int rows, int columns, ulong seed)
    {
        ulong state = seed;
        var matrix = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            matrix[i] = new double[columns];
            for (int j = 0; j < columns; j++) matrix[i][j] = Rng.NextUnit(ref state);
        }
        return matrix;
    }

    internal static double[][] Zeros(int rows, int columns)
    {
        var matrix = new double[rows][];
        for (int i = 0; i < rows; i++) matrix[i] = new double[columns];
        return matrix;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int rows = a.Length;
        int inner = b.Length;
        int columns = b[0].Length;
        var result = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new double[columns];
            for (int j = 0; j < columns; j++)
            {
                double total = 0.0;
                for (int k = 0; k < inner; k++) total += a[i][k] * b[k][j];
                result[i][j] = total;
            }
        }
        return result;
    }

    internal static double[][] Transpose(double[][] a)
    {
        int rows = a.Length;
        int columns = a[0].Length;
        var result = new double[columns][];
        for (int j = 0; j < columns; j++)
        {
            result[j] = new double[rows];
            for (int i = 0; i < rows; i++) result[j][i] = a[i][j];
        }
        return result;
    }

    internal static double[][] SelectColumns(double[][] a, IReadOnlyList<int> indices)
    {
        int rows = a.Length;
        var result = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new double[indices.Count];
            for (int c = 0; c < indices.Count; c++) result[i][c] = a[i][indices[c]];
        }
        return result;
    }

    internal static double[][] SelectRows(double[][] a, IReadOnlyList<int> indices)
    {
        var result = new double[indices.Count][];
        for (int c = 0; c < indices.Count; c++) result[c] = (double[])a[indices[c]].Clone();
        return result;
    }

    internal static double Frobenius(double[][] a)
    {
        double total = 0.0;
        foreach (double[] row in a) foreach (double value in row) total += value * value;
        return Math.Sqrt(total);
    }

    internal static double FrobeniusDifference(double[][] a, double[][] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a[i].Length; j++)
            {
                double difference = a[i][j] - b[i][j];
                total += difference * difference;
            }
        }
        return Math.Sqrt(total);
    }

    internal static double MaxAbsDifference(double[][] a, double[][] b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a[i].Length; j++) worst = Math.Max(worst, Math.Abs(a[i][j] - b[i][j]));
        }
        return worst;
    }

    /** Symmetric eigendecomposition by cyclic Jacobi; values ascending, vectors in the columns. */
    internal static void Jacobi(double[][] source, int maxSweeps, double tolerance,
            out double[] values, out double[][] vectors)
    {
        int n = source.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = (double[])source[i].Clone();
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0.0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += m[i][j] * m[i][j];
            if (Math.Sqrt(off) <= tolerance) break;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(m[p][q]) <= 1e-300) continue;
                    double theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                    double t = Math.Abs(theta) < 1e-300 ? 1.0
                            : Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int i = 0; i < n; i++)
                    {
                        double mip = m[i][p];
                        double miq = m[i][q];
                        m[i][p] = c * mip - s * miq;
                        m[i][q] = s * mip + c * miq;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double mpi = m[p][i];
                        double mqi = m[q][i];
                        m[p][i] = c * mpi - s * mqi;
                        m[q][i] = s * mpi + c * mqi;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        values = new double[n];
        for (int i = 0; i < n; i++) values[i] = m[i][i];
        vectors = v;
    }

    /** Moore-Penrose pseudo-inverse as (W^T W)^+ W^T with singular-value truncation. */
    internal static double[][] PseudoInverse(double[][] w, double relativeTolerance)
    {
        int rows = w.Length;
        int columns = w[0].Length;
        double[][] gram = Multiply(Transpose(w), w);
        Jacobi(gram, 200, 1e-14, out double[] values, out double[][] vectors);
        double largest = 0.0;
        foreach (double value in values) largest = Math.Max(largest, Math.Abs(value));
        double cutoff = relativeTolerance * largest;
        var scaled = new double[columns][];
        for (int i = 0; i < columns; i++) scaled[i] = new double[columns];
        for (int a = 0; a < columns; a++)
        {
            double value = values[a];
            double inverse = Math.Abs(value) > cutoff && value != 0.0 ? 1.0 / value : 0.0;
            for (int i = 0; i < columns; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    scaled[i][j] += inverse * vectors[i][a] * vectors[j][a];
                }
            }
        }
        // (W^T W)^+ W^T
        return Multiply(scaled, Transpose(w));
    }

    internal static double BestRankKError(double[][] a, int k)
    {
        double[][] gram = Multiply(Transpose(a), a);
        Jacobi(gram, 200, 1e-14, out double[] values, out _);
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        double total = 0.0;
        foreach (double value in sorted) total += Math.Max(0.0, value);
        double kept = 0.0;
        for (int i = 0; i < k && i < sorted.Length; i++) kept += Math.Max(0.0, sorted[sorted.Length - 1 - i]);
        return Math.Sqrt(Math.Max(0.0, total - kept));
    }

    /** Column norms squared, normalised into sampling probabilities. */
    internal static double[] ColumnProbabilities(double[][] a)
    {
        int columns = a[0].Length;
        var probabilities = new double[columns];
        double total = 0.0;
        for (int j = 0; j < columns; j++)
        {
            double norm = 0.0;
            for (int i = 0; i < a.Length; i++) norm += a[i][j] * a[i][j];
            probabilities[j] = norm;
            total += norm;
        }
        for (int j = 0; j < columns; j++) probabilities[j] /= total;
        return probabilities;
    }

    internal static double[] RowProbabilities(double[][] a)
    {
        int rows = a.Length;
        var probabilities = new double[rows];
        double total = 0.0;
        for (int i = 0; i < rows; i++)
        {
            double norm = 0.0;
            for (int j = 0; j < a[i].Length; j++) norm += a[i][j] * a[i][j];
            probabilities[i] = norm;
            total += norm;
        }
        for (int i = 0; i < rows; i++) probabilities[i] /= total;
        return probabilities;
    }

    internal static double[] Cumulative(double[] probabilities)
    {
        var cumulative = new double[probabilities.Length];
        double running = 0.0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            running += probabilities[i];
            cumulative[i] = running;
        }
        cumulative[cumulative.Length - 1] = 1.0;
        return cumulative;
    }

    internal static List<int> TopIndices(double[] values, int count)
    {
        var order = new List<int>();
        for (int i = 0; i < values.Length; i++) order.Add(i);
        order.Sort((x, y) => values[y].CompareTo(values[x]));
        return order.GetRange(0, count);
    }

    internal static List<int> SampleIndices(double[] probabilities, int count, ulong seed)
    {
        double[] cumulative = Cumulative(probabilities);
        var chosen = new List<int>();
        ulong state = seed;
        while (chosen.Count < count)
        {
            int index = Rng.Draw(ref state, cumulative);
            if (!chosen.Contains(index)) chosen.Add(index);
        }
        return chosen;
    }
}

internal static class Cur
{
    internal sealed class Result
    {
        internal double[][] C = Array.Empty<double[]>();
        internal double[][] U = Array.Empty<double[]>();
        internal double[][] R = Array.Empty<double[]>();
        internal List<int> Columns = new List<int>();
        internal List<int> Rows = new List<int>();
    }

    internal static Result Build(double[][] a, IReadOnlyList<int> columns, IReadOnlyList<int> rows)
    {
        double[][] c = LinAlg.SelectColumns(a, columns);
        double[][] r = LinAlg.SelectRows(a, rows);
        double[][] w = LinAlg.SelectRows(c, rows);
        double[][] u = LinAlg.PseudoInverse(w, 1e-12);
        return new Result { C = c, U = u, R = r, Columns = new List<int>(columns), Rows = new List<int>(rows) };
    }

    internal static double[][] Reconstruct(Result result)
    {
        return LinAlg.Multiply(LinAlg.Multiply(result.C, result.U), result.R);
    }

    /** The four Moore-Penrose identities, returned as the worst absolute violation. */
    internal static double PseudoInverseViolation(double[][] w)
    {
        double[][] pinv = LinAlg.PseudoInverse(w, 1e-12);
        double[][] wPinv = LinAlg.Multiply(w, pinv);
        double[][] pinvW = LinAlg.Multiply(pinv, w);
        double worst = 0.0;
        worst = Math.Max(worst, LinAlg.MaxAbsDifference(LinAlg.Multiply(wPinv, w), w));
        worst = Math.Max(worst, LinAlg.MaxAbsDifference(LinAlg.Multiply(pinvW, pinv), pinv));
        worst = Math.Max(worst, LinAlg.MaxAbsDifference(LinAlg.Transpose(wPinv), wPinv));
        worst = Math.Max(worst, LinAlg.MaxAbsDifference(LinAlg.Transpose(pinvW), pinvW));
        return worst;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    public static void Main()
    {
        // ---- the pseudo-inverse identities -------------------------------------------------
        ulong state = 596001UL;
        for (int trial = 0; trial < 4; trial++)
        {
            int rows = 5 + trial;
            int columns = 4 + trial % 2;
            var w = new double[rows][];
            for (int i = 0; i < rows; i++)
            {
                w[i] = new double[columns];
                for (int j = 0; j < columns; j++) w[i][j] = Rng.NextUnit(ref state);
            }
            if (trial == 3)
            {
                // Force a rank-deficient intersection: the last column becomes a copy of the first.
                for (int i = 0; i < rows; i++) w[i][columns - 1] = w[i][0];
            }
            double violation = Cur.PseudoInverseViolation(w);
            Check(violation <= 1e-9, "the Moore-Penrose identities hold on trial " + trial + " (worst violation "
                    + violation.ToString("0.0e0") + ")");
        }
        double[][] zero = LinAlg.Zeros(3, 3);
        double[][] zeroPinv = LinAlg.PseudoInverse(zero, 1e-12);
        CheckClose(LinAlg.Frobenius(zeroPinv), 0.0, 1e-15, "the pseudo-inverse of the zero matrix is the zero matrix");

        // ---- keeping everything reproduces A exactly ---------------------------------------
        var full = LinAlg.Matrix(7, 5, 596111UL);
        var allColumns = new List<int>();
        var allRows = new List<int>();
        for (int j = 0; j < 5; j++) allColumns.Add(j);
        for (int i = 0; i < 7; i++) allRows.Add(i);
        Cur.Result fullResult = Cur.Build(full, allColumns, allRows);
        CheckClose(LinAlg.FrobeniusDifference(full, Cur.Reconstruct(fullResult)), 0.0, 1e-9,
                "with every column and row kept, C U R = A A^+ A = A");
        CheckClose(LinAlg.FrobeniusDifference(full, LinAlg.Multiply(LinAlg.Multiply(full,
                LinAlg.PseudoInverse(full, 1e-12)), full)), 0.0, 1e-9,
                "the same statement for A A^+ A with the standalone pseudo-inverse");

        // ---- a rank-two matrix whose heavy columns span the column space -------------------
        int dimension = 6;
        int samples = 9;
        var left = new double[dimension][];
        var right = new double[samples][];
        for (int k = 0; k < 2; k++)
        {
            left[k] = new double[dimension];
            right[k] = new double[samples];
            for (int i = 0; i < dimension; i++) left[k][i] = Rng.NextUnit(ref state) + (i == k ? 2.0 : 0.0);
            for (int j = 0; j < samples; j++) right[k][j] = Rng.NextUnit(ref state) + (j == k ? 2.0 : 0.0);
        }
        var rankTwo = new double[dimension][];
        for (int i = 0; i < dimension; i++) rankTwo[i] = new double[samples];
        for (int i = 0; i < dimension; i++)
        {
            for (int j = 0; j < samples; j++)
            {
                double total = 0.0;
                for (int k = 0; k < 2; k++) total += 40.0 * left[k][i] * right[k][j];
                rankTwo[i][j] = total;
            }
        }
        CheckClose(LinAlg.BestRankKError(rankTwo, 2), 0.0, 1e-9, "the constructed matrix has rank two");
        double[] columnWeights = LinAlg.ColumnProbabilities(rankTwo);
        double[] rowWeights = LinAlg.RowProbabilities(rankTwo);
        List<int> topColumns = LinAlg.TopIndices(columnWeights, 2);
        List<int> topRows = LinAlg.TopIndices(rowWeights, 2);
        Check(topColumns.Contains(0) && topColumns.Contains(1),
                "the norm-based column selection finds the two heavy columns (picked: "
                + string.Join(",", topColumns) + ")");
        Check(topRows.Contains(0) && topRows.Contains(1),
                "the norm-based row selection finds the two heavy rows (picked: " + string.Join(",", topRows) + ")");
        Cur.Result rankTwoResult = Cur.Build(rankTwo, topColumns, topRows);
        double rankTwoError = LinAlg.FrobeniusDifference(rankTwo, Cur.Reconstruct(rankTwoResult));
        CheckClose(rankTwoError, 0.0, 1e-9, "CUR recovers the rank-two matrix exactly with two columns and two rows");
        CheckClose(rankTwoError, LinAlg.BestRankKError(rankTwo, 2), 1e-9,
                "the Eckart-Young bound is attained, so the inequality is tight here");

        // ---- random sampling quality -------------------------------------------------------
        var random = LinAlg.Matrix(30, 25, 596222UL);
        double[] randomColumnWeights = LinAlg.ColumnProbabilities(random);
        double[] randomRowWeights = LinAlg.RowProbabilities(random);
        int kept = 4;
        List<int> sampledColumns = LinAlg.SampleIndices(randomColumnWeights, kept, 596333UL);
        List<int> sampledRows = LinAlg.SampleIndices(randomRowWeights, kept, 596444UL);
        Cur.Result sampledResult = Cur.Build(random, sampledColumns, sampledRows);
        double sampledError = LinAlg.FrobeniusDifference(random, Cur.Reconstruct(sampledResult));
        double worstRandom = 0.0;
        double bestRandom = double.PositiveInfinity;
        ulong attemptState = 596555UL;
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var candidateColumns = new List<int>();
            var candidateRows = new List<int>();
            while (candidateColumns.Count < kept) candidateColumns.Add(Rng.NextBelow(ref attemptState, 25));
            while (candidateRows.Count < kept) candidateRows.Add(Rng.NextBelow(ref attemptState, 30));
            Cur.Result candidate = Cur.Build(random, candidateColumns, candidateRows);
            double error = LinAlg.FrobeniusDifference(random, Cur.Reconstruct(candidate));
            worstRandom = Math.Max(worstRandom, error);
            bestRandom = Math.Min(bestRandom, error);
        }
        Check(sampledError >= LinAlg.BestRankKError(random, kept) - 1e-9,
                "the CUR error never drops below the best rank-k error (" + sampledError.ToString("0.0000")
                + " against " + LinAlg.BestRankKError(random, kept).ToString("0.0000") + ")");
        Check(sampledError < worstRandom, "importance sampling beats the worst uniform choice of columns ("
                + sampledError.ToString("0.0000") + " < " + worstRandom.ToString("0.0000") + ")");
        // The (2 + eps) relative-error guarantee of the CUR literature needs O(r / eps^2) sampled columns;
        // with only 4 of 25 columns the file asserts the honest weaker statement and reports the ratio.
        double rankFourBound = LinAlg.BestRankKError(random, kept);
        Check(sampledError <= 6.0 * rankFourBound, "the CUR error stays within a factor of six of the best rank-k error ("
                + (sampledError / rankFourBound).ToString("0.000") + " times " + rankFourBound.ToString("0.0000") + ")");
        Check(bestRandom > 0.0, "even the best of the twenty uniform choices leaves a nonzero error ("
                + bestRandom.ToString("0.0000") + ")");
        Check(sampledColumns.Count == kept && sampledRows.Count == kept, "the sampled index sets have the requested size");
        for (int index = 0; index < sampledColumns.Count; index++)
        {
            for (int other = index + 1; other < sampledColumns.Count; other++)
            {
                Check(sampledColumns[index] != sampledColumns[other], "the sampled columns are distinct");
            }
        }

        // ---- rank of the reconstruction -----------------------------------------------------
        // The rank-k error is computed as sqrt(||A||_F^2 - sum of the k largest eigenvalues), which is a
        // difference of two nearly equal large numbers, so the zero-rank check carries a relative
        // tolerance instead of an absolute one; the residual level is reported.
        double[][] reconstruction = Cur.Reconstruct(sampledResult);
        double reconstructionRankError = LinAlg.BestRankKError(reconstruction, kept);
        Check(reconstructionRankError <= 1e-5 * LinAlg.Frobenius(reconstruction),
                "the reconstruction has rank at most k, hence a vanishing best-rank-k error ("
                + reconstructionRankError.ToString("0.0e0") + " against norm "
                + LinAlg.Frobenius(reconstruction).ToString("0.0000") + ", the residue being cancellation in the"
                + " eigenvalue sum)");

        // ---- determinism --------------------------------------------------------------------
        List<int> repeatColumns = LinAlg.SampleIndices(randomColumnWeights, kept, 596333UL);
        bool identical = repeatColumns.Count == sampledColumns.Count;
        for (int i = 0; identical && i < repeatColumns.Count; i++)
        {
            if (repeatColumns[i] != sampledColumns[i]) identical = false;
        }
        Check(identical, "the sampling is reproducible for a fixed seed");

        Console.WriteLine("PASS");
    }
}
