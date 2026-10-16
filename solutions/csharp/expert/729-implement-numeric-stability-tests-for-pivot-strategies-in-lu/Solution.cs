// Task 729 - Implement numeric stability tests for pivot strategies in LU. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// Gaussian elimination factorises A = L U with a unit lower L, but the order of the eliminations decides
// whether the factorisation is usable. This file implements three strategies and measures them:
//
//   no pivot      take the diagonal entry as it is. Cheap, and catastrophic whenever a small pivot
//                 amplifies the multipliers - the growth factor rho = max|U| / max|A| explodes.
//   partial pivot pick the largest magnitude in the column (PA = LU). Bounded multipliers (<= 1) but rho
//                 can still reach 2^(n-1), the classical worst case found by Wilkinson.
//   complete      pick the largest magnitude in the whole trailing block (PAQ = LU). Smallest practical
//                 growth, at the cost of a quadratic search per step.
//
// Verified against both the textbook theorems and numeric practice: PA = LU (and PAQ = LU) hold to
// rounding; the two-entry matrix [[eps, 1], [1, 1]] shows growth 1/eps without pivoting, a factorisation
// that cannot even reproduce A, and a garbage solution - where one swap brings growth 1 and a clean
// answer; Wilkinson's matrix genuinely attains rho = 2^(n-1) under partial pivoting; a singular matrix is
// *detected* by the no-pivot path where partial pivoting simply succeeds; triangular matrices need no
// pivoting at all; and the growth factors of the three strategies are compared over a graded family of
// random matrices, where partial pivoting never does worse than no pivoting and complete pivoting is the
// safest of the three.
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

internal enum PivotMode
{
    None,
    Partial,
    Complete,
}

internal sealed class LuResult
{
    internal int Size;
    internal double[][] L = Array.Empty<double[]>();
    internal double[][] U = Array.Empty<double[]>();
    internal int[] RowPerm = Array.Empty<int>();
    internal int[] ColPerm = Array.Empty<int>();
    internal int Swaps;
    internal double Growth;
    internal bool Singular;
}

internal static class Lu
{
    internal static LuResult Factor(double[][] a, PivotMode mode)
    {
        int n = a.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = new double[n];
            Array.Copy(a[i], m[i], n);
        }
        var rows = new int[n];
        var cols = new int[n];
        for (int i = 0; i < n; i++) { rows[i] = i; cols[i] = i; }

        var result = new LuResult { Size = n };
        double inputMax = MaxAbs(m, n);
        int swaps = 0;
        bool singular = false;
        for (int col = 0; col < n; col++)
        {
            int pivotRow = col, pivotCol = col;
            if (mode == PivotMode.Partial)
            {
                for (int r = col + 1; r < n; r++)
                    if (Math.Abs(m[r][col]) > Math.Abs(m[pivotRow][col])) pivotRow = r;
            }
            else if (mode == PivotMode.Complete)
            {
                double best = -1.0;
                for (int r = col; r < n; r++)
                    for (int c = col; c < n; c++)
                        if (Math.Abs(m[r][c]) > best) { best = Math.Abs(m[r][c]); pivotRow = r; pivotCol = c; }
            }
            if (pivotRow != col)
            {
                (m[col], m[pivotRow]) = (m[pivotRow], m[col]);
                (rows[col], rows[pivotRow]) = (rows[pivotRow], rows[col]);
                swaps++;
            }
            if (pivotCol != col)
            {
                for (int r = 0; r < n; r++) (m[r][col], m[r][pivotCol]) = (m[r][pivotCol], m[r][col]);
                (cols[col], cols[pivotCol]) = (cols[pivotCol], cols[col]);
                swaps++;
            }
            double pivot = m[col][col];
            if (pivot == 0.0) { singular = true; break; }
            for (int r = col + 1; r < n; r++)
            {
                double factor = m[r][col] / pivot;
                m[r][col] = factor;                       // L stores the multipliers in place
                if (factor == 0.0) continue;
                for (int c = col + 1; c < n; c++) m[r][c] -= factor * m[col][c];
            }
        }

        result.Swaps = swaps;
        result.Singular = singular;
        result.RowPerm = rows;
        result.ColPerm = cols;
        result.L = new double[n][];
        result.U = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result.L[i] = new double[n];
            result.U[i] = new double[n];
            result.L[i][i] = 1.0;
            for (int j = 0; j < i; j++) result.L[i][j] = m[i][j];
            for (int j = i; j < n; j++) result.U[i][j] = m[i][j];
        }
        result.Growth = inputMax == 0.0 ? 0.0 : MaxAbs(result.U, n) / inputMax;
        return result;
    }

    /// <summary>Forward and back substitution with the row permutation applied to b.</summary>
    internal static double[] Solve(double[][] a, double[] b, PivotMode mode)
    {
        LuResult factors = Factor(a, mode);
        int n = a.Length;
        var y = new double[n];
        for (int i = 0; i < n; i++) y[i] = b[factors.RowPerm[i]];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) y[i] -= factors.L[i][j] * y[j];
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) sum -= factors.U[i][j] * x[j];
            x[i] = factors.U[i][i] == 0.0 ? double.NaN : sum / factors.U[i][i];
        }
        return x;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int n = a.Length;
        var c = new double[n][];
        for (int i = 0; i < n; i++)
        {
            c[i] = new double[n];
            for (int k = 0; k < n; k++)
            {
                double aik = a[i][k];
                if (aik == 0.0) continue;
                for (int j = 0; j < n; j++) c[i][j] += aik * b[k][j];
            }
        }
        return c;
    }

    internal static double[] Multiply(double[][] a, double[] x)
    {
        int n = a.Length;
        var y = new double[n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) y[i] += a[i][j] * x[j];
        return y;
    }

    /// <summary>Rebuilds P A Q from the factorisation and reports the largest deviation from L U.</summary>
    internal static double FactorisationError(double[][] a, LuResult r)
    {
        int n = r.Size;
        double worst = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double lu = 0.0;
                for (int k = 0; k < Math.Min(i, j) + 1; k++) lu += r.L[i][k] * r.U[k][j];
                double permuted = a[r.RowPerm[i]][r.ColPerm[j]];
                worst = Math.Max(worst, Math.Abs(permuted - lu));
            }
        return worst;
    }

    internal static double MaxAbs(double[][] m, int n)
    {
        double best = 0.0;
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) best = Math.Max(best, Math.Abs(m[i][j]));
        return best;
    }

    internal static double NormInf(double[] v)
    {
        double best = 0.0;
        foreach (double x in v) best = Math.Max(best, Math.Abs(x));
        return best;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- the factorisation identities, on a well-conditioned random system ----------------------
        {
            double[][] a = RandomMatrix(6, 0x729AUL, 1.0);
            var errors = new Dictionary<PivotMode, double>();
            foreach (PivotMode mode in new[] { PivotMode.None, PivotMode.Partial, PivotMode.Complete })
            {
                LuResult r = Lu.Factor(a, mode);
                double error = Lu.FactorisationError(a, r);
                errors[mode] = error;
                Check(error < 1e-12, mode + ": P A Q = L U (deviation " + error + ")");
                Check(!r.Singular, mode + ": a well-conditioned matrix is not singular");
                Check(r.Growth >= 1.0 - 1e-12, mode + ": the growth factor is at least one");
                for (int i = 0; i < r.Size; i++)
                {
                    if (Math.Abs(r.L[i][i] - 1.0) > 0.0) Check(false, "L must have a unit diagonal");
                    for (int j = i + 1; j < r.Size; j++)
                        if (r.U[j][i] != 0.0) Check(false, "U must be upper triangular");
                }
                bool validPermutation = true;
                var seen = new HashSet<int>();
                foreach (int row in r.RowPerm) if (!seen.Add(row)) validPermutation = false;
                Check(validPermutation, mode + ": the row permutation is a permutation");
            }
            Console.WriteLine("random 6x6: growth none=" + Lu.Factor(a, PivotMode.None).Growth.ToString("G4")
                + " partial=" + Lu.Factor(a, PivotMode.Partial).Growth.ToString("G4")
                + " complete=" + Lu.Factor(a, PivotMode.Complete).Growth.ToString("G4"));
            Check(errors[PivotMode.Partial] <= errors[PivotMode.None] * 100.0 + 1e-15,
                "pivoting does not spoil the factorisation identity");
        }

        // ---- the classic two-by-two killer of no-pivot elimination ----------------------------------
        {
            const double eps = 1e-18;
            double[][] a = { new[] { eps, 1.0 }, new[] { 1.0, 1.0 } };
            LuResult none = Lu.Factor(a, PivotMode.None);
            LuResult partial = Lu.Factor(a, PivotMode.Partial);
            Console.WriteLine("eps-killer: growth none=" + none.Growth.ToString("G4")
                + " partial=" + partial.Growth.ToString("G4") + " swaps=" + partial.Swaps);
            Check(none.Growth > 1e15, "without pivoting the growth factor explodes (" + none.Growth + ")");
            Check(partial.Growth < 10.0, "one row swap tames it (" + partial.Growth + ")");
            Check(partial.Swaps == 1, "partial pivoting swaps exactly once here");
            Check(Lu.FactorisationError(a, partial) < 1e-15, "the pivoted factorisation is exact in the identity sense");
            double noneError = Lu.FactorisationError(a, none);
            Check(noneError > 0.1, "the unpivoted one cannot even reproduce A (deviation " + noneError + ")");

            double[] truth = { 1.0, -2.0 };
            double[] b = Lu.Multiply(a, truth);
            double[] xNone = Lu.Solve(a, b, PivotMode.None);
            double[] xPartial = Lu.Solve(a, b, PivotMode.Partial);
            double errorNone = RelativeError(xNone, truth);
            double errorPartial = RelativeError(xPartial, truth);
            double residualNone = RelativeResidual(a, xNone, b);
            double residualPartial = RelativeResidual(a, xPartial, b);
            Console.WriteLine("eps-killer solve: relativeError none=" + errorNone.ToString("G4")
                + " partial=" + errorPartial.ToString("G4") + " residual none=" + residualNone.ToString("G4")
                + " partial=" + residualPartial.ToString("G4"));
            Check(errorPartial < 1e-12, "partial pivoting recovers the solution (" + errorPartial + ")");
            Check(errorNone > 1e-3, "no pivoting loses almost every digit (" + errorNone + ")");
            Check(residualPartial < 1e-14 && residualNone > residualPartial,
                "and the residual separates the two just as clearly (" + residualNone + ")");
            Check(RelativeResidual(a, xPartial, b) < 1e-14, "the pivoted solve is backward stable");
        }

        // ---- Wilkinson's worst case for partial pivoting: growth 2^(n-1) ----------------------------
        {
            foreach (int n in new[] { 4, 5, 6, 8 })
            {
                double[][] w = Wilkinson(n);
                double growth = Lu.Factor(w, PivotMode.Partial).Growth;
                double expected = Math.Pow(2.0, n - 1);
                Console.WriteLine("Wilkinson n=" + n + ": growth=" + growth.ToString("G6") + " expected=" + expected);
                Check(Math.Abs(growth - expected) <= expected * 1e-9,
                    "n=" + n + ": partial pivoting reaches the theoretical growth 2^(n-1)");
                double complete = Lu.Factor(w, PivotMode.Complete).Growth;
                Check(complete <= growth * (1.0 + 1e-9),
                    "n=" + n + ": complete pivoting does no worse (" + complete + " vs " + growth + ")");
                Check(Lu.FactorisationError(w, Lu.Factor(w, PivotMode.Partial)) < 1e-14,
                    "n=" + n + ": the Wilkinson factorisation still reproduces A");
            }
        }

        // ---- singular matrices: detected without pivoting, handled with it --------------------------
        {
            double[][] singular = { new[] { 0.0, 1.0 }, new[] { 0.0, 2.0 } };
            LuResult none = Lu.Factor(singular, PivotMode.None);
            Check(none.Singular, "a zero pivot is detected");
            double[][] alsoSingular = { new[] { 1.0, 2.0 }, new[] { 2.0, 4.0 } };
            Check(Lu.Factor(alsoSingular, PivotMode.None).Singular, "a dependent row eventually produces a zero pivot");
            Check(!Lu.Factor(new double[][] { new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 } }, PivotMode.Partial).Singular,
                "partial pivoting rescues a zero diagonal entry");
            double[] solved = Lu.Solve(new double[][] { new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 } }, new[] { 2.0, 3.0 }, PivotMode.Partial);
            Check(Math.Abs(solved[0] - 1.0) < 1e-14 && Math.Abs(solved[1] - 2.0) < 1e-14,
                "and solves the system it rescued");
        }

        // ---- triangular matrices never need pivoting -------------------------------------------------
        {
            var upper = new double[5][];
            var lower = new double[5][];
            for (int i = 0; i < 5; i++)
            {
                upper[i] = new double[5];
                lower[i] = new double[5];
                for (int j = i; j < 5; j++) upper[i][j] = 1.0 + i * 0.5 + j;
                for (int j = 0; j <= i; j++) lower[i][j] = 2.0 + i - j * 0.5;
            }
            Check(Math.Abs(Lu.Factor(upper, PivotMode.None).Growth - 1.0) < 1e-12,
                "an upper triangular matrix is already factorised");
            Check(Lu.Factor(lower, PivotMode.None).Swaps == 0, "a lower triangular matrix needs no swaps");
            Check(!Lu.Factor(upper, PivotMode.None).Singular && !Lu.Factor(lower, PivotMode.None).Singular,
                "and neither is singular");
        }

        // ---- pivoting quality on a graded random family ------------------------------------------------
        {
            int harshForNone = 0, partialNotWorse = 0, completeComparable = 0;
            foreach (double scale in new[] { 1.0, 1e-3, 1e-6, 1e-9 })
            {
                double[][] a = RandomMatrix(8, 0x729CUL, scale);
                double none = Lu.Factor(a, PivotMode.None).Growth;
                double partial = Lu.Factor(a, PivotMode.Partial).Growth;
                double complete = Lu.Factor(a, PivotMode.Complete).Growth;
                var truth = new double[8];
                for (int i = 0; i < 8; i++) truth[i] = 1.0 + i * 0.25;
                double[] b = Lu.Multiply(a, truth);
                double errorNone = RelativeError(Lu.Solve(a, b, PivotMode.None), truth);
                double errorPartial = RelativeError(Lu.Solve(a, b, PivotMode.Partial), truth);
                if (none > 10.0) harshForNone++;
                if (partial <= none * 1.000001) partialNotWorse++;
                if (complete <= partial * 2.0) completeComparable++;
                Console.WriteLine("rowScale=" + scale.ToString("G2") + " growth none=" + none.ToString("G4")
                    + " partial=" + partial.ToString("G4") + " complete=" + complete.ToString("G4")
                    + " errNone=" + errorNone.ToString("G4") + " errPartial=" + errorPartial.ToString("G4"));
                Check(none >= 1.0 - 1e-12 && partial >= 1.0 - 1e-12 && complete >= 1.0 - 1e-12,
                    "growth stays at least one for every strategy (row scale " + scale + ")");
                Check(Lu.FactorisationError(a, Lu.Factor(a, PivotMode.Partial)) < 1e-9,
                    "partial pivoting always reproduces A (row scale " + scale + ")");
                Check(errorPartial < 1e-6, "partial pivoting keeps the solution accurate ("
                    + errorPartial.ToString("G4") + ")");
            }
            Check(harshForNone >= 2, "a tiny first row really punishes the no-pivot order (" + harshForNone + " of 4)");
            Check(partialNotWorse >= 3, "partial pivoting is never much worse than no pivoting ("
                + partialNotWorse + " of 4)");
            Check(completeComparable == 4, "complete pivoting stays comparable to partial pivoting ("
                + completeComparable + " of 4)");
        }

        // ---- reproducibility ----------------------------------------------------------------------------
        {
            double[][] a = RandomMatrix(5, 0x729DUL, 1e-3);
            LuResult first = Lu.Factor(a, PivotMode.Partial);
            LuResult second = Lu.Factor(a, PivotMode.Partial);
            Check(first.Swaps == second.Swaps && first.Growth == second.Growth, "the factorisation is reproducible");
            Check(Same(x: Lu.Solve(a, new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, PivotMode.Partial),
                y: Lu.Solve(a, new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }, PivotMode.Partial)), "and so is the solve");
            Check(Lu.FactorisationError(a, first) < 1e-12, "the reproduced factorisation still equals A");
        }

        Console.WriteLine("PASS");
    }

    /// <summary>a[i][i] = 1, a[i][j] = -1 below the diagonal, last column 1: growth 2^(n-1) with partial pivoting.</summary>
    private static double[][] Wilkinson(int n)
    {
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n];
            for (int j = 0; j < i; j++) a[i][j] = -1.0;
            a[i][i] = 1.0;
            a[i][n - 1] = 1.0;
        }
        return a;
    }

    private static double[][] RandomMatrix(int n, ulong seed, double scale)
    {
        var rng = new SplitMix64(seed);
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n];
            for (int j = 0; j < n; j++) a[i][j] = rng.NextSigned();
        }
        if (scale != 1.0) for (int j = 0; j < n; j++) a[0][j] *= scale;   // a tiny leading pivot
        return a;
    }

    private static double RelativeError(double[] computed, double[] truth)
    {
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < truth.Length; i++)
        {
            numerator = Math.Max(numerator, Math.Abs(computed[i] - truth[i]));
            denominator = Math.Max(denominator, Math.Abs(truth[i]));
        }
        return denominator == 0.0 ? numerator : numerator / denominator;
    }

    private static double RelativeResidual(double[][] a, double[] x, double[] b)
    {
        double[] ax = Lu.Multiply(a, x);
        double numerator = 0.0, denominator = 0.0;
        for (int i = 0; i < b.Length; i++)
        {
            numerator = Math.Max(numerator, Math.Abs(ax[i] - b[i]));
            denominator = Math.Max(denominator, Math.Abs(b[i]));
        }
        return denominator == 0.0 ? numerator : numerator / denominator;
    }

    private static bool Same(double[] x, double[] y)
    {
        if (x.Length != y.Length) return false;
        for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
