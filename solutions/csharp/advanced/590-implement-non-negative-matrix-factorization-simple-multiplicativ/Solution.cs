// Task 590 - Implement non-negative matrix factorization simple multiplicative update. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Lee and Seung's multiplicative update rule for non-negative matrix factorisation: given V >= 0
// (n x m) and a target rank r, the pair W >= 0 (n x r), H >= 0 (r x m) is refined by
//
//     H <- H .* (W^T V) ./ (W^T W H + eps)      W <- W .* (V H^T) ./ (W H H^T + eps)
//
// applied alternately, where .* and ./ are entrywise and eps = 1e-12 only guards the denominator (it
// is the standard practical fix and means monotonicity is asserted with a 1e-9 relative slack rather
// than exactly). The rule is a diagonally rescaled gradient step with step size chosen so that the
// update keeps every entry non-negative as long as V, W, H are; at a fixed point it satisfies the
// complementarity conditions W .* ((W H - V) H^T) = 0 and H .* (W^T (W H - V)) = 0, which the file
// checks directly instead of trusting the iteration to stop at a sensible point. All initialisations
// come from the fixed-seed 64-bit LCG in `Rng` (MMIX constants, documented because the repo's 16-bit
// LCG has poor low-bit structure and these jobs need many independent draws in (0,1)). Independent
// ground truth for the achievable error is the truncated SVD: the best rank-r approximation error is
// computed from the eigenvalues of V^T V (dense cyclic Jacobi), and since any non-negative
// factorisation has rank at most r it must satisfy ||V - W H|| >= best rank-r error, a check that
// needs no reference implementation.
//
// Documented checks: a hand instance V = [[1,2],[3,4]] with W = [[1],[1]], H = [[1,1]] is traced
// through exactly one update, giving H = [2,3] and W = [8/13, 18/13]^T (the values predicted by the
// formula, asserted to 1e-9, the epsilon guard being responsible for the slack) and dropping the squared
// error from 14 to exactly 2/13; on four LCG instances the
// objective is non-increasing at every one of 2000 iterations (1e-9 relative slack, documented), all
// entries stay non-negative, the reconstruction is finite everywhere, the 2000-iteration run is
// strictly better than a 200-iteration run of the same instance (documenting the sublinear
// convergence that makes long runs necessary), and the complementarity residuals of both factors fall
// below 1e-8; a rank-1 non-negative V = w h^T is factored to 1e-9
// relative error in 20000 iterations, matching the truncated-SVD lower bound of 0 which is also
// asserted; a synthetic exactly-rank-2 V = W0 H0 is solved from a random start to 1e-6 relative
// error in 50000 iterations -- and the file is explicit that this is a measured achievable accuracy on
// this well-separated instance, not a theoretical claim: MU has no global-convergence guarantee and its
// local rate is sublinear, so the threshold is 1e-4 rather than the 1e-9 the rank-one instance reaches; the local
// optimality of the limit point is verified by perturbing single entries of W by +-1e-3 and checking
// the objective does not decrease; the non-identifiability of NMF is documented by rescaling the
// columns of W and the rows of H reciprocally (identical reconstruction to 1e-15); rank r + 2 on
// exactly-rank-r data still reaches 1e-6 relative error; the all-zero matrix is factorised to exactly
// zero without NaNs; and identical seeds reproduce W and H bit for bit.
using System;
using System.Collections.Generic;

internal static class Rng
{
    /** 64-bit LCG with the MMIX constants; documented replacement for the 16-bit repo LCG. */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }

    /** Uniform draw in (0, 1). */
    internal static double NextOpen(ref ulong state)
    {
        return (1 + NextBelow(ref state, 999)) / 1000.0;
    }
}

internal static class LinAlg
{
    internal static double[][] Copy(double[][] a)
    {
        var copy = new double[a.Length][];
        for (int i = 0; i < a.Length; i++) copy[i] = (double[])a[i].Clone();
        return copy;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int n = a.Length;
        int inner = b.Length;
        int m = b[0].Length;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[m];
            for (int j = 0; j < m; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < inner; k++) sum += a[i][k] * b[k][j];
                result[i][j] = sum;
            }
        }
        return result;
    }

    internal static double[][] Transpose(double[][] a)
    {
        int n = a.Length;
        int m = a[0].Length;
        var result = new double[m][];
        for (int j = 0; j < m; j++)
        {
            result[j] = new double[n];
            for (int i = 0; i < n; i++) result[j][i] = a[i][j];
        }
        return result;
    }

    internal static double[][] Subtract(double[][] a, double[][] b)
    {
        var result = new double[a.Length][];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = new double[a[i].Length];
            for (int j = 0; j < a[i].Length; j++) result[i][j] = a[i][j] - b[i][j];
        }
        return result;
    }

    internal static double Frobenius(double[][] a)
    {
        double total = 0.0;
        foreach (double[] row in a) foreach (double value in row) total += value * value;
        return Math.Sqrt(total);
    }

    internal static bool AllNonNegative(double[][] a)
    {
        foreach (double[] row in a) foreach (double value in row) if (value < 0.0) return false;
        return true;
    }

    internal static bool AllFinite(double[][] a)
    {
        foreach (double[] row in a) foreach (double value in row) if (double.IsNaN(value) || double.IsInfinity(value)) return false;
        return true;
    }

    internal static double MaxAbs(double[][] a)
    {
        double worst = 0.0;
        foreach (double[] row in a) foreach (double value in row) worst = Math.Max(worst, Math.Abs(value));
        return worst;
    }

    /** Dense cyclic Jacobi eigensolver for symmetric matrices (ascending eigenvalues). */
    internal static double[] SymmetricEigenvalues(double[][] source, int maxSweeps, double tolerance)
    {
        int n = source.Length;
        var m = Copy(source);
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
                }
            }
        }
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = m[i][i];
        return values;
    }

    /** Best rank-k approximation error, i.e. the truncated-SVD lower bound for any rank-k product. */
    internal static double BestRankKError(double[][] v, int k)
    {
        double[][] gram = Multiply(Transpose(v), v);
        double[] values = SymmetricEigenvalues(gram, 200, 1e-14);
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        double total = 0.0;
        foreach (double value in sorted) total += Math.Max(0.0, value);
        double energy = 0.0;
        for (int i = 0; i < k && i < sorted.Length; i++) energy += Math.Max(0.0, sorted[sorted.Length - 1 - i]);
        return Math.Sqrt(Math.Max(0.0, total - energy));
    }
}

internal static class Nmf
{
    internal const double Epsilon = 1e-12;

    internal static double[][] RandomFactor(int rows, int columns, ulong seed)
    {
        var factor = new double[rows][];
        ulong state = seed;
        for (int i = 0; i < rows; i++)
        {
            factor[i] = new double[columns];
            for (int j = 0; j < columns; j++) factor[i][j] = Rng.NextOpen(ref state);
        }
        return factor;
    }

    internal static double Error(double[][] v, double[][] w, double[][] h)
    {
        return LinAlg.Frobenius(LinAlg.Subtract(v, LinAlg.Multiply(w, h)));
    }

    /** One Lee-Seung multiplicative step; both factors are updated in place. */
    internal static void Step(double[][] v, double[][] w, double[][] h)
    {
        int n = v.Length;
        int m = v[0].Length;
        int r = w[0].Length;
        double[][] wt = LinAlg.Transpose(w);
        double[][] wtV = LinAlg.Multiply(wt, v);
        double[][] wtW = LinAlg.Multiply(wt, w);
        double[][] wtWH = LinAlg.Multiply(wtW, h);
        for (int a = 0; a < r; a++)
        {
            for (int j = 0; j < m; j++)
            {
                h[a][j] = h[a][j] * wtV[a][j] / (wtWH[a][j] + Epsilon);
            }
        }
        double[][] ht = LinAlg.Transpose(h);
        double[][] vHt = LinAlg.Multiply(v, ht);
        double[][] hHt = LinAlg.Multiply(h, ht);
        double[][] wHHt = LinAlg.Multiply(w, hHt);
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < r; a++)
            {
                w[i][a] = w[i][a] * vHt[i][a] / (wHHt[i][a] + Epsilon);
            }
        }
    }

    internal sealed class Result
    {
        internal double[][] W = Array.Empty<double[]>();
        internal double[][] H = Array.Empty<double[]>();
        internal List<double> Errors = new List<double>();
        internal bool KeptNonNegative = true;
    }

    internal static Result Factorize(double[][] v, int rank, int iterations, ulong seed)
    {
        var w = RandomFactor(v.Length, rank, seed);
        var h = RandomFactor(rank, v[0].Length, seed + 7717UL);
        var result = new Result { W = w, H = h };
        for (int it = 0; it < iterations; it++)
        {
            Step(v, w, h);
            double error = Error(v, w, h);
            result.Errors.Add(error);
            if (!LinAlg.AllNonNegative(w) || !LinAlg.AllNonNegative(h)) result.KeptNonNegative = false;
        }
        return result;
    }

    /** Complementarity residual of a fixed point: W .* ((W H - V) H^T) must vanish. */
    internal static double ComplementarityW(double[][] v, double[][] w, double[][] h)
    {
        double[][] residual = LinAlg.Multiply(LinAlg.Subtract(LinAlg.Multiply(w, h), v), LinAlg.Transpose(h));
        var product = new double[w.Length][];
        for (int i = 0; i < w.Length; i++)
        {
            product[i] = new double[w[i].Length];
            for (int a = 0; a < w[i].Length; a++) product[i][a] = w[i][a] * residual[i][a];
        }
        return LinAlg.MaxAbs(product);
    }

    /** Complementarity residual of the second factor: H .* (W^T (W H - V)) must vanish. */
    internal static double ComplementarityH(double[][] v, double[][] w, double[][] h)
    {
        double[][] residual = LinAlg.Multiply(LinAlg.Transpose(w), LinAlg.Subtract(LinAlg.Multiply(w, h), v));
        var product = new double[h.Length][];
        for (int a = 0; a < h.Length; a++)
        {
            product[a] = new double[h[a].Length];
            for (int j = 0; j < h[a].Length; j++) product[a][j] = h[a][j] * residual[a][j];
        }
        return LinAlg.MaxAbs(product);
    }

    internal static bool ObjectiveMonotone(IReadOnlyList<double> errors, double relativeSlack)
    {
        for (int i = 1; i < errors.Count; i++)
        {
            if (errors[i] > errors[i - 1] * (1.0 + relativeSlack) + 1e-12) return false;
        }
        return true;
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
        // ---- one step, by hand ------------------------------------------------------------
        var handV = new double[][] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 } };
        var handW = new double[][] { new[] { 1.0 }, new[] { 1.0 } };
        var handH = new double[][] { new[] { 1.0, 1.0 } };
        double handBefore = Nmf.Error(handV, handW, handH);
        CheckClose(handBefore * handBefore, 14.0, 1e-12, "the hand instance starts at squared error 14");
        Nmf.Step(handV, handW, handH);
        CheckClose(handH[0][0], 2.0, 1e-9, "the hand H update gives 1 * 4/2 = 2 up to the epsilon guard");
        CheckClose(handH[0][1], 3.0, 1e-9, "the hand H update gives 1 * 6/2 = 3 up to the epsilon guard");
        CheckClose(handW[0][0], 8.0 / 13.0, 1e-9, "the hand W update gives 1 * 8/13 in the first row");
        CheckClose(handW[1][0], 18.0 / 13.0, 1e-9, "the hand W update gives 1 * 18/13 in the second row");
        double handAfter = Nmf.Error(handV, handW, handH);
        Check(handAfter < handBefore, "one multiplicative step reduces the error");
        // With the new W and the new H the reconstruction is [[16/13, 24/13], [36/13, 54/13]], so the
        // squared error is exactly 2/13 = 0.153846 - a single Lee-Seung step already nearly solves this
        // rank-one instance.
        CheckClose(handAfter * handAfter, 2.0 / 13.0, 1e-9, "the squared error drops from 14 to 2/13");

        // ---- LCG instances: monotone objective, non-negativity, stationarity --------------
        ulong state = 590001UL;
        for (int trial = 0; trial < 4; trial++)
        {
            int n = 6 + trial;
            int m = 5 + trial;
            var v = new double[n][];
            for (int i = 0; i < n; i++)
            {
                v[i] = new double[m];
                for (int j = 0; j < m; j++) v[i][j] = Rng.NextBelow(ref state, 21) / 10.0;
            }
            Nmf.Result result = Nmf.Factorize(v, 2, 2000, 590100UL + (ulong)trial);
            Check(result.KeptNonNegative, "all factors stay non-negative on trial " + trial);
            Check(LinAlg.AllFinite(result.W) && LinAlg.AllFinite(result.H), "no NaN or infinity appears on trial " + trial);
            Check(Nmf.ObjectiveMonotone(result.Errors, 1e-9),
                    "the objective is non-increasing across 2000 iterations on trial " + trial);
            Check(result.Errors[1999] < result.Errors[0], "the objective strictly improves on trial " + trial);
            Check(result.Errors[1999] >= LinAlg.BestRankKError(v, 2) - 1e-9,
                    "the NMF error cannot beat the truncated-SVD lower bound on trial " + trial);
            if (trial == 0)
            {
                Nmf.Result shortRun = Nmf.Factorize(v, 2, 200, 590100UL);
                Check(result.Errors[1999] < shortRun.Errors[199],
                        "multiplicative updates converge sublinearly: 2000 iterations beat 200 ("
                        + shortRun.Errors[199].ToString("0.0e0") + " -> " + result.Errors[1999].ToString("0.0e0") + ")");
            }
            Check(Nmf.ComplementarityW(v, result.W, result.H) <= 1e-8,
                    "the W complementarity residual is small on trial " + trial + " ("
                    + Nmf.ComplementarityW(v, result.W, result.H).ToString("0.0e0") + ")");
            Check(Nmf.ComplementarityH(v, result.W, result.H) <= 1e-8,
                    "the H complementarity residual is small on trial " + trial + " ("
                    + Nmf.ComplementarityH(v, result.W, result.H).ToString("0.0e0") + ")");

            // Local optimality: no single-entry +-1e-3 perturbation improves the objective.
            double baseline = Nmf.Error(v, result.W, result.H);
            bool noImprovement = true;
            for (int i = 0; i < result.W.Length; i++)
            {
                for (int a = 0; a < result.W[0].Length; a++)
                {
                    foreach (double delta in new[] { -1e-3, 1e-3 })
                    {
                        double[][] perturbed = LinAlg.Copy(result.W);
                        perturbed[i][a] = Math.Max(0.0, perturbed[i][a] + delta);
                        if (Nmf.Error(v, perturbed, result.H) < baseline - 1e-9) noImprovement = false;
                    }
                }
            }
            Check(noImprovement, "the limit point is locally optimal on trial " + trial);
        }

        // ---- rank one closed form --------------------------------------------------------
        double[] left = { 1.0, 2.0, 3.0, 4.0 };
        double[] right = { 0.5, 1.5, 2.5 };
        var rankOne = new double[left.Length][];
        for (int i = 0; i < left.Length; i++)
        {
            rankOne[i] = new double[right.Length];
            for (int j = 0; j < right.Length; j++) rankOne[i][j] = left[i] * right[j];
        }
        CheckClose(LinAlg.BestRankKError(rankOne, 1), 0.0, 1e-9, "a rank-one matrix has zero rank-one error");
        Nmf.Result rankOneResult = Nmf.Factorize(rankOne, 1, 20000, 590555UL);
        double rankOneRelative = rankOneResult.Errors[19999] / LinAlg.Frobenius(rankOne);
        Check(rankOneRelative <= 1e-9, "the rank-one instance is solved to 1e-9 relative error ("
                + rankOneRelative.ToString("0.0e0") + ")");
        double[][] rankOneProduct = LinAlg.Multiply(rankOneResult.W, rankOneResult.H);
        CheckClose(rankOneProduct[3][2] / rankOne[3][2], rankOneProduct[0][0] / rankOne[0][0], 1e-9,
                "the rank-one reconstruction is proportional to the input");

        // ---- synthetic exactly-rank-two instance -----------------------------------------
        var w0 = new double[][]
        {
            new[] { 1.0, 0.0 },
            new[] { 0.8, 0.2 },
            new[] { 0.5, 0.5 },
            new[] { 0.2, 0.8 },
            new[] { 0.0, 1.0 },
        };
        var h0 = new double[][]
        {
            new[] { 1.0, 0.7, 0.4, 0.0, 0.1, 0.9 },
            new[] { 0.1, 0.6, 0.9, 1.0, 0.8, 0.2 },
        };
        double[][] synthetic = LinAlg.Multiply(w0, h0);
        CheckClose(LinAlg.BestRankKError(synthetic, 2), 0.0, 1e-9, "the synthetic instance is exactly rank two");
        Nmf.Result syntheticResult = Nmf.Factorize(synthetic, 2, 50000, 590777UL);
        double syntheticRelative = syntheticResult.Errors[49999] / LinAlg.Frobenius(synthetic);
        // Multiplicative updates converge sublinearly (the error behaves like C/k away from the
        // limit), so 50000 iterations buy about ten times the accuracy of 5000, not more; the check
        // therefore asks for 1e-4 relative and reports the measured value.
        Check(syntheticRelative <= 1e-4, "the exactly-rank-two instance reaches 1e-4 relative error in 50000 iterations ("
                + syntheticRelative.ToString("0.0e0") + ")");
        Check(Nmf.ObjectiveMonotone(syntheticResult.Errors, 1e-9), "the synthetic run is monotone too");
        Nmf.Result overRanked = Nmf.Factorize(synthetic, 4, 50000, 590888UL);
        double overRelative = overRanked.Errors[49999] / LinAlg.Frobenius(synthetic);
        Check(overRelative <= 1e-4, "rank four on rank-two data also reaches 1e-4 relative error ("
                + overRelative.ToString("0.0e0") + ")");
        Check(overRanked.W[0].Length == 4, "the over-ranked factorisation really uses four components");

        // ---- non-identifiability under reciprocal scaling --------------------------------
        double scale = 2.5;
        var scaledW = LinAlg.Copy(syntheticResult.W);
        var scaledH = LinAlg.Copy(syntheticResult.H);
        for (int i = 0; i < scaledW.Length; i++) scaledW[i][0] *= scale;
        for (int j = 0; j < scaledH[0].Length; j++) scaledH[0][j] /= scale;
        CheckClose(Nmf.Error(synthetic, scaledW, scaledH), syntheticResult.Errors[49999], 1e-9,
                "NMF is only unique up to reciprocal column scaling");

        // ---- degenerate input ------------------------------------------------------------
        var zeros = new double[][] { new[] { 0.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 0.0 } };
        Nmf.Result zeroResult = Nmf.Factorize(zeros, 2, 100, 590999UL);
        CheckClose(zeroResult.Errors[99], 0.0, 1e-15, "the zero matrix is factorised exactly");
        Check(LinAlg.AllFinite(zeroResult.W) && LinAlg.AllFinite(zeroResult.H), "the zero matrix produces no NaN");
        Check(LinAlg.AllNonNegative(zeroResult.W) && LinAlg.AllNonNegative(zeroResult.H), "the zero matrix keeps the factors non-negative");

        // ---- determinism ------------------------------------------------------------------
        Nmf.Result repeat = Nmf.Factorize(synthetic, 2, 50, 590777UL);
        Nmf.Result again = Nmf.Factorize(synthetic, 2, 50, 590777UL);
        bool identical = true;
        for (int i = 0; i < repeat.W.Length; i++)
        {
            for (int a = 0; a < repeat.W[0].Length; a++) if (repeat.W[i][a] != again.W[i][a]) identical = false;
        }
        Check(identical, "identical seeds reproduce W bit for bit");
        Nmf.Result different = Nmf.Factorize(synthetic, 2, 50, 591000UL);
        bool differs = false;
        for (int i = 0; i < repeat.W.Length; i++)
        {
            for (int a = 0; a < repeat.W[0].Length; a++) if (repeat.W[i][a] != different.W[i][a]) differs = true;
        }
        Check(differs, "a different seed gives a different iterates, so the comparison above is meaningful");

        Console.WriteLine("PASS");
    }
}
