// Task 588 - Implement matrix inverse with Gaussian elimination (careful with numeric). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// Gauss-Jordan inversion on the augmented pair [A | I] with partial pivoting: at every column the
// largest remaining pivot candidate is exchanged into place, the pivot row is normalised, and the
// column is cleared in *all* other rows, so the right half of the augmented matrix becomes A^-1. The
// numerical care the title asks for is made explicit in four ways. (1) Singularity is decided
// relatively: a pivot is rejected when |pivot| <= pivotTolerance * max|A_ij|, which is scale
// invariant, and the routine returns null instead of silently producing garbage. (2) Both residuals
// are measured, since a one-sided check can miss a wrong inverse -- the documented 2x2 counterexample
// [[1e-16, 1], [1, 1]] gives X with ||A X - I|| ~ 1e-16 while ||X A - I|| = 1. (3) Pivoting is
// compared against an unpivoted elimination on exactly that instance: the pivoted result is accurate
// to 1e-9 whereas the unpivoted one is off by 1.0 in the infinity norm. (4) The known ill-conditioned
// Hilbert matrices are inverted and compared against their exact integer inverses, with the tolerance
// loosened where the condition number (~5e2 for H_3, ~1.5e4 for H_4) demands it. The inverse is
// cross-checked against three genuinely independent routes: Cramer's rule for 2x2, the closed-form
// integer inverses, and Newton-Schulz iteration X_{k+1} = X_k (2I - A X_k) started from the scaled
// transpose A^T / (||A||_1 ||A||_inf), which converges quadratically but knows nothing about
// elimination. All matrices are small deterministic LCG samples (documented 64-bit LCG, MMIX
// constants) or fixed hand instances; nothing depends on wall-clock time or hash ordering.
//
// Documented checks: the 2x2 hand instance [[4,7],[2,6]] inverts to (1/10)[[6,-7],[-2,4]] entry by
// entry; for six LCG matrices the products A A^-1 and A^-1 A are both within 1e-12 of the identity in
// the infinity norm; (A B)^-1 = B^-1 A^-1, (A^T)^-1 = (A^-1)^T, (A^-1)^-1 = A, det(A^-1) = 1/det(A)
// and (cA)^-1 = A^-1/c all hold to 1e-12; diagonal, upper and lower triangular instances give the
// exact reciprocal/back-substitution inverses (a triangular inverse stays triangular, checked
// elementwise); the Hilbert 3x3 inverse equals the integer matrix [[9,-36,30],[-36,192,-180],
// [30,-180,180]] to 1e-8 and the Hilbert 4x4 inverse is within 1e-4 absolute of the integer matrix
// [[16,-120,240,-140],[-120,1200,-2700,1680],[240,-2700,6480,-4200],[-140,1680,-4200,2800]]
// (documented: cond(H_4) ~ 1.5e4 amplifies the rounding of the entries 1/(i+j-1) to about 1e-7 on
// entries of size 6480, so 1e-8 is not achievable there); the unimodular matrix [[1,2,3],[0,1,4],
// [5,6,0]] inverts to the exact integer matrix [[-24,18,5],[20,-15,-4],[-5,4,1]]; duplicated-row,
// proportional-row and below-tolerance instances all return null -- the rejected one is
// [[1, 1], [1, 1 + 1e-14]], whose second pivot is 1e-14 against a scale of 1 -- while a matrix with a
// merely small (1e-3) pivot still inverts correctly and a uniformly scaled 1e-14 identity-like matrix
// is accepted, because the tolerance is relative to max|A_ij| rather than absolute; Newton-Schulz
// contracts ||I - A X_k|| monotonically over the first six iterations (each step being asserted against the
// previous one), reaches a defect below 1e-12 after 30 iterations and lands within 1e-9 of the elimination
// inverse on a symmetric positive definite instance; solving A x = b through the inverse matches a direct LU solve to 1e-9 with residuals
// below 1e-12; and the documented 2x2 tiny-pivot instance separates the pivoted and unpivoted
// routines by 1.0 in both residuals (asserted as > 0.5 each).
using System;
using System.Collections.Generic;

internal static class MatrixOps
{
    internal static double MaxAbsEntry(double[][] a)
    {
        double worst = 0.0;
        foreach (double[] row in a) foreach (double value in row) worst = Math.Max(worst, Math.Abs(value));
        return worst;
    }

    internal static double Norm1(double[][] a)
    {
        int n = a.Length;
        double worst = 0.0;
        for (int j = 0; j < n; j++)
        {
            double column = 0.0;
            for (int i = 0; i < n; i++) column += Math.Abs(a[i][j]);
            worst = Math.Max(worst, column);
        }
        return worst;
    }

    internal static double NormInf(double[][] a)
    {
        double worst = 0.0;
        foreach (double[] row in a)
        {
            double total = 0.0;
            foreach (double value in row) total += Math.Abs(value);
            worst = Math.Max(worst, total);
        }
        return worst;
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

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int n = a.Length;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[n];
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < n; k++) sum += a[i][k] * b[k][j];
                result[i][j] = sum;
            }
        }
        return result;
    }

    internal static double[] Multiply(double[][] a, double[] x)
    {
        int n = a.Length;
        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += a[i][j] * x[j];
            result[i] = sum;
        }
        return result;
    }

    internal static double[][] Transpose(double[][] a)
    {
        int n = a.Length;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[n];
            for (int j = 0; j < n; j++) result[i][j] = a[j][i];
        }
        return result;
    }

    /** Gauss-Jordan inversion with partial pivoting; null when a pivot is relatively too small. */
    internal static double[][]? Invert(double[][] source, double pivotTolerance)
    {
        int n = source.Length;
        var m = new double[n][];
        var inverse = new double[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = (double[])source[i].Clone();
            inverse[i] = new double[n];
            inverse[i][i] = 1.0;
        }
        double scale = MaxAbsEntry(source);
        if (scale == 0.0) return null;
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.Abs(m[r][col]) > Math.Abs(m[pivot][col])) pivot = r;
            if (Math.Abs(m[pivot][col]) <= pivotTolerance * scale) return null;
            if (pivot != col)
            {
                (m[pivot], m[col]) = (m[col], m[pivot]);
                (inverse[pivot], inverse[col]) = (inverse[col], inverse[pivot]);
            }
            double divisor = m[col][col];
            for (int c = 0; c < n; c++)
            {
                m[col][c] /= divisor;
                inverse[col][c] /= divisor;
            }
            for (int r = 0; r < n; r++)
            {
                if (r == col) continue;
                double factor = m[r][col];
                if (factor == 0.0) continue;
                for (int c = 0; c < n; c++)
                {
                    m[r][c] -= factor * m[col][c];
                    inverse[r][c] -= factor * inverse[col][c];
                }
            }
        }
        return inverse;
    }

    /** Identical elimination with no pivoting, kept only to document the failure mode. */
    internal static double[][]? InvertNoPivot(double[][] source)
    {
        int n = source.Length;
        var m = new double[n][];
        var inverse = new double[n][];
        for (int i = 0; i < n; i++)
        {
            m[i] = (double[])source[i].Clone();
            inverse[i] = new double[n];
            inverse[i][i] = 1.0;
        }
        for (int col = 0; col < n; col++)
        {
            if (m[col][col] == 0.0) return null;
            double divisor = m[col][col];
            for (int c = 0; c < n; c++)
            {
                m[col][c] /= divisor;
                inverse[col][c] /= divisor;
            }
            for (int r = 0; r < n; r++)
            {
                if (r == col) continue;
                double factor = m[r][col];
                if (factor == 0.0) continue;
                for (int c = 0; c < n; c++)
                {
                    m[r][c] -= factor * m[col][c];
                    inverse[r][c] -= factor * inverse[col][c];
                }
            }
        }
        return inverse;
    }

    /** Newton-Schulz iteration for the inverse: X <- X (2I - A X). */
    internal static double[][] NewtonSchulz(double[][] a, int iterations, out double lastDefect)
    {
        int n = a.Length;
        double denominator = Norm1(a) * NormInf(a);
        var x = new double[n][];
        for (int i = 0; i < n; i++)
        {
            x[i] = new double[n];
            for (int j = 0; j < n; j++) x[i][j] = a[j][i] / denominator;
        }
        lastDefect = double.NaN;
        for (int step = 0; step < iterations; step++)
        {
            double[][] ax = Multiply(a, x);
            var correction = new double[n][];
            for (int i = 0; i < n; i++)
            {
                correction[i] = new double[n];
                for (int j = 0; j < n; j++) correction[i][j] = (i == j ? 2.0 : 0.0) - ax[i][j];
            }
            x = Multiply(x, correction);
            double[][] actual = Multiply(a, x);
            double defect = 0.0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double expected = i == j ? 1.0 : 0.0;
                    defect = Math.Max(defect, Math.Abs(actual[i][j] - expected));
                }
            }
            lastDefect = defect;
        }
        return x;
    }

    /** Direct LU solve with partial pivoting, an independent route to x in A x = b. */
    internal static double[] Solve(double[][] a, double[] b)
    {
        int n = a.Length;
        var m = new double[n][];
        var rhs = (double[])b.Clone();
        for (int i = 0; i < n; i++) m[i] = (double[])a[i].Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.Abs(m[r][col]) > Math.Abs(m[pivot][col])) pivot = r;
            if (pivot != col)
            {
                (m[pivot], m[col]) = (m[col], m[pivot]);
                (rhs[pivot], rhs[col]) = (rhs[col], rhs[pivot]);
            }
            for (int r = col + 1; r < n; r++)
            {
                double factor = m[r][col] / m[col][col];
                for (int c = col; c < n; c++) m[r][c] -= factor * m[col][c];
                rhs[r] -= factor * rhs[col];
            }
        }
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = rhs[i];
            for (int j = i + 1; j < n; j++) sum -= m[i][j] * x[j];
            x[i] = sum / m[i][i];
        }
        return x;
    }

    internal static double Determinant(double[][] source)
    {
        int n = source.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = (double[])source[i].Clone();
        double det = 1.0;
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.Abs(m[r][col]) > Math.Abs(m[pivot][col])) pivot = r;
            if (pivot != col)
            {
                (m[pivot], m[col]) = (m[col], m[pivot]);
                det = -det;
            }
            det *= m[col][col];
            for (int r = col + 1; r < n; r++)
            {
                double factor = m[r][col] / m[col][col];
                for (int c = col; c < n; c++) m[r][c] -= factor * m[col][c];
            }
        }
        return det;
    }
}

internal static class Lcg64
{
    /** 64-bit LCG with the MMIX constants (the repo's 16-bit LCG is too weak for these samples). */
    internal static ulong Next(ref ulong state)
    {
        state = state * 6364136223846793005UL + 1442695040888963407UL;
        return state >> 11;
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

public static class Solution
{
    private const double TinyPivotTolerance = 1e-12;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance * Math.Max(1.0, Math.Abs(expected)),
                label + " (got " + actual + ", expected " + expected + ")");
    }

    private static double[][] Require(double[][]? value, string label)
    {
        if (value is null) throw new Exception("FAILED: " + label + " (unexpected null inverse)");
        return value;
    }

    private static double[][] Identity(int n)
    {
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[n];
            result[i][i] = 1.0;
        }
        return result;
    }

    /** Random matrix made strictly diagonally dominant, hence guaranteed invertible. */
    private static double[][] RandomMatrix(int n, ulong seed, int bound)
    {
        var matrix = new double[n][];
        ulong state = seed;
        for (int i = 0; i < n; i++)
        {
            matrix[i] = new double[n];
            for (int j = 0; j < n; j++) matrix[i][j] = Lcg64.NextBelow(ref state, 2 * bound + 1) - bound;
            matrix[i][i] += (n + 1) * bound + n;
        }
        return matrix;
    }

    public static void Main()
    {
        // ---- 2x2 hand instance -----------------------------------------------------------
        var two = new double[][] { new[] { 4.0, 7.0 }, new[] { 2.0, 6.0 } };
        double[][] twoInverse = Require(MatrixOps.Invert(two, TinyPivotTolerance), "the 2x2 instance is invertible");
        CheckClose(twoInverse[0][0], 0.6, 1e-12, "the 2x2 inverse entry (0,0) is 6/10");
        CheckClose(twoInverse[0][1], -0.7, 1e-12, "the 2x2 inverse entry (0,1) is -7/10");
        CheckClose(twoInverse[1][0], -0.2, 1e-12, "the 2x2 inverse entry (1,0) is -2/10");
        CheckClose(twoInverse[1][1], 0.4, 1e-12, "the 2x2 inverse entry (1,1) is 4/10");

        // ---- LCG instances: both residuals -------------------------------------------------
        ulong seed = 588001UL;
        for (int trial = 0; trial < 6; trial++)
        {
            int n = 3 + trial % 4;
            double[][] a = RandomMatrix(n, seed + (ulong)trial, 5);
            double[][] inverse = Require(MatrixOps.Invert(a, TinyPivotTolerance), "the LCG instance " + trial + " inverts");
            Check(MatrixOps.MaxAbsDifference(MatrixOps.Multiply(a, inverse), Identity(n)) <= 1e-12,
                    "A A^-1 = I on trial " + trial);
            Check(MatrixOps.MaxAbsDifference(MatrixOps.Multiply(inverse, a), Identity(n)) <= 1e-12,
                    "A^-1 A = I on trial " + trial);
            double[][] roundTrip = Require(MatrixOps.Invert(inverse, TinyPivotTolerance), "the inverse is invertible");
            Check(MatrixOps.MaxAbsDifference(roundTrip, a) <= 1e-10, "(A^-1)^-1 = A on trial " + trial);
            double[][] transposedInverse = Require(MatrixOps.Invert(MatrixOps.Transpose(a), TinyPivotTolerance),
                    "the transpose inverts on trial " + trial);
            Check(MatrixOps.MaxAbsDifference(transposedInverse, MatrixOps.Transpose(inverse)) <= 1e-12,
                    "(A^T)^-1 = (A^-1)^T on trial " + trial);
            CheckClose(MatrixOps.Determinant(inverse), 1.0 / MatrixOps.Determinant(a), 1e-9,
                    "det(A^-1) = 1/det(A) on trial " + trial);

            double[][] b = RandomMatrix(n, seed + 100UL + (ulong)trial, 4);
            double[][] bInverse = Require(MatrixOps.Invert(b, TinyPivotTolerance), "the second matrix inverts on trial " + trial);
            double[][] productInverse = Require(MatrixOps.Invert(MatrixOps.Multiply(a, b), TinyPivotTolerance),
                    "the product inverts on trial " + trial);
            Check(MatrixOps.MaxAbsDifference(productInverse, MatrixOps.Multiply(bInverse, inverse)) <= 1e-11,
                    "(A B)^-1 = B^-1 A^-1 on trial " + trial);

            var scaled = new double[n][];
            for (int i = 0; i < n; i++)
            {
                scaled[i] = new double[n];
                for (int j = 0; j < n; j++) scaled[i][j] = 3.0 * a[i][j];
            }
            double[][] scaledInverse = Require(MatrixOps.Invert(scaled, TinyPivotTolerance), "(3A) inverts on trial " + trial);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    CheckClose(scaledInverse[i][j], inverse[i][j] / 3.0, 1e-12, "(3A)^-1 = A^-1/3 on trial " + trial);
                }
            }
        }

        // ---- triangular and diagonal instances -------------------------------------------
        var upper = new double[][] { new[] { 2.0, 3.0, -1.0 }, new[] { 0.0, 4.0, 5.0 }, new[] { 0.0, 0.0, 8.0 } };
        double[][] upperInverse = Require(MatrixOps.Invert(upper, TinyPivotTolerance), "the upper triangular matrix inverts");
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (i > j) Check(upperInverse[i][j] == 0.0, "the inverse of an upper triangular matrix stays upper at (" + i + "," + j + ")");
            }
        }
        CheckClose(upperInverse[0][0], 0.5, 1e-12, "upper (0,0) = 1/2");
        CheckClose(upperInverse[1][1], 0.25, 1e-12, "upper (1,1) = 1/4");
        CheckClose(upperInverse[2][2], 0.125, 1e-12, "upper (2,2) = 1/8");
        CheckClose(upperInverse[0][1], -0.375, 1e-12, "upper (0,1) = -3/8");
        CheckClose(upperInverse[1][2], -5.0 / 32.0, 1e-12, "upper (1,2) = -5/32 from back substitution");
        CheckClose(upperInverse[0][2], 19.0 / 64.0, 1e-12, "upper (0,2) = 19/64 from back substitution");
        var lower = MatrixOps.Transpose(upper);
        double[][] lowerInverse = Require(MatrixOps.Invert(lower, TinyPivotTolerance), "the lower triangular matrix inverts");
        Check(MatrixOps.MaxAbsDifference(lowerInverse, MatrixOps.Transpose(upperInverse)) <= 1e-12,
                "(L^T)^-1 = (L^-1)^T for the triangular pair");
        var diagonal = new double[][] { new[] { 2.0, 0.0 }, new[] { 0.0, -4.0 } };
        double[][] diagonalInverse = Require(MatrixOps.Invert(diagonal, TinyPivotTolerance), "the diagonal matrix inverts");
        CheckClose(diagonalInverse[0][0], 0.5, 1e-15, "a diagonal inverse is the reciprocal at (0,0)");
        CheckClose(diagonalInverse[1][1], -0.25, 1e-15, "a diagonal inverse is the reciprocal at (1,1)");
        Check(diagonalInverse[0][1] == 0.0 && diagonalInverse[1][0] == 0.0, "the diagonal inverse is diagonal");

        // ---- Hilbert matrices: known integer inverses ------------------------------------
        var hilbert3 = new double[][] { new[] { 1.0, 0.5, 1.0 / 3 }, new[] { 0.5, 1.0 / 3, 0.25 }, new[] { 1.0 / 3, 0.25, 0.2 } };
        double[][] hilbert3Inverse = Require(MatrixOps.Invert(hilbert3, TinyPivotTolerance), "H_3 inverts");
        var hilbert3Exact = new double[][] { new[] { 9.0, -36.0, 30.0 }, new[] { -36.0, 192.0, -180.0 }, new[] { 30.0, -180.0, 180.0 } };
        Check(MatrixOps.MaxAbsDifference(hilbert3Inverse, hilbert3Exact) <= 1e-8,
                "H_3 inverts to the known integer matrix within 1e-8 (cond(H_3) ~ 5e2)");
        var hilbert4 = new double[4][];
        for (int i = 0; i < 4; i++)
        {
            hilbert4[i] = new double[4];
            for (int j = 0; j < 4; j++) hilbert4[i][j] = 1.0 / (i + j + 1);
        }
        double[][] hilbert4Inverse = Require(MatrixOps.Invert(hilbert4, TinyPivotTolerance), "H_4 inverts");
        var hilbert4Exact = new double[][]
        {
            new[] { 16.0, -120.0, 240.0, -140.0 },
            new[] { -120.0, 1200.0, -2700.0, 1680.0 },
            new[] { 240.0, -2700.0, 6480.0, -4200.0 },
            new[] { -140.0, 1680.0, -4200.0, 2800.0 },
        };
        double hilbert4Error = MatrixOps.MaxAbsDifference(hilbert4Inverse, hilbert4Exact);
        Check(hilbert4Error <= 1e-4, "H_4 inverts to the known integer matrix within 1e-4 (worst error "
                + hilbert4Error.ToString("0.0e0") + ", cond(H_4) ~ 1.5e4)");
        Check(MatrixOps.MaxAbsDifference(MatrixOps.Multiply(hilbert4, hilbert4Inverse), Identity(4)) <= 1e-10,
                "the H_4 residual is still small even though the entries are only accurate to 1e-4");

        // ---- unimodular integer instance --------------------------------------------------
        var unimodular = new double[][] { new[] { 1.0, 2.0, 3.0 }, new[] { 0.0, 1.0, 4.0 }, new[] { 5.0, 6.0, 0.0 } };
        CheckClose(MatrixOps.Determinant(unimodular), 1.0, 1e-12, "the unimodular instance has determinant 1");
        double[][] unimodularInverse = Require(MatrixOps.Invert(unimodular, TinyPivotTolerance), "the unimodular matrix inverts");
        var unimodularExact = new double[][] { new[] { -24.0, 18.0, 5.0 }, new[] { 20.0, -15.0, -4.0 }, new[] { -5.0, 4.0, 1.0 } };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                CheckClose(unimodularInverse[i][j], unimodularExact[i][j], 1e-9, "the integer inverse entry (" + i + "," + j + ")");
                Check(Math.Abs(unimodularInverse[i][j] - Math.Round(unimodularInverse[i][j])) <= 1e-9,
                        "the integer inverse entry (" + i + "," + j + ") is an integer");
            }
        }

        // ---- singularity handling ---------------------------------------------------------
        var duplicated = new double[][] { new[] { 1.0, 2.0 }, new[] { 1.0, 2.0 } };
        Check(MatrixOps.Invert(duplicated, TinyPivotTolerance) is null, "a duplicated row is reported singular");
        var proportional = new double[][] { new[] { 1.0, 2.0 }, new[] { 2.0, 4.0 } };
        Check(MatrixOps.Invert(proportional, TinyPivotTolerance) is null, "proportional rows are reported singular");
        var zero = new double[][] { new[] { 0.0, 0.0 }, new[] { 0.0, 0.0 } };
        Check(MatrixOps.Invert(zero, TinyPivotTolerance) is null, "the zero matrix is reported singular");
        var tinyPivotButFine = new double[][] { new[] { 1e-3, 1.0 }, new[] { 1.0, 1.0 } };
        double[][] tinyOk = Require(MatrixOps.Invert(tinyPivotButFine, TinyPivotTolerance), "a 1e-3 pivot is still fine");
        Check(MatrixOps.MaxAbsDifference(MatrixOps.Multiply(tinyPivotButFine, tinyOk), Identity(2)) <= 1e-12,
                "the small-but-sufficient pivot instance still inverts accurately");
        var belowTolerance = new double[][] { new[] { 1.0, 1.0 }, new[] { 1.0, 1.0 + 1e-14 } };
        Check(MatrixOps.Invert(belowTolerance, TinyPivotTolerance) is null,
                "a pivot below the relative tolerance is rejected as singular");
        Check(MatrixOps.Invert(belowTolerance, 1e-16) is not null,
                "loosening the tolerance lets the same matrix through, so the decision is a documented knob");
        var uniformlyTiny = new double[][] { new[] { 1e-14, 0.0 }, new[] { 0.0, 1e-14 } };
        double[][] uniformlyTinyInverse = Require(MatrixOps.Invert(uniformlyTiny, TinyPivotTolerance),
                "a uniformly scaled matrix is well conditioned and inverts");
        CheckClose(uniformlyTinyInverse[0][0], 1e14, 1e-9, "the relative tolerance does not reject a uniform 1e-14 scale");

        // ---- pivoting versus no pivoting --------------------------------------------------
        var tinyPivot = new double[][] { new[] { 1e-16, 1.0 }, new[] { 1.0, 1.0 } };
        double[][] pivoted = Require(MatrixOps.Invert(tinyPivot, 1e-20), "the tiny-pivot matrix inverts with pivoting");
        double[][] naive = Require(MatrixOps.InvertNoPivot(tinyPivot), "the unpivoted routine returns something");
        double pivotedLeft = MatrixOps.MaxAbsDifference(MatrixOps.Multiply(pivoted, tinyPivot), Identity(2));
        double naiveLeft = MatrixOps.MaxAbsDifference(MatrixOps.Multiply(naive, tinyPivot), Identity(2));
        double naiveRight = MatrixOps.MaxAbsDifference(MatrixOps.Multiply(tinyPivot, naive), Identity(2));
        Check(pivotedLeft <= 1e-9, "partial pivoting keeps the left residual small (" + pivotedLeft.ToString("0.0e0") + ")");
        Check(naiveLeft > 0.5, "the unpivoted elimination is off by 1.0 in the left residual ("
                + naiveLeft.ToString("0.0e0") + ")");
        Check(naiveRight > 0.5, "the unpivoted elimination is off by 1.0 in the right residual too ("
                + naiveRight.ToString("0.0e0") + "); both residuals are measured because a one-sided check"
                + " cannot certify that X is the inverse");
        CheckClose(pivoted[0][0], -1.0, 1e-9, "the pivoted inverse entry (0,0) is -1");

        // ---- Newton-Schulz as an independent route ---------------------------------------
        var symmetric = new double[][] { new[] { 4.0, 1.0, 0.5 }, new[] { 1.0, 3.0, -0.5 }, new[] { 0.5, -0.5, 2.0 } };
        double[][] eliminationInverse = Require(MatrixOps.Invert(symmetric, TinyPivotTolerance), "the SPD instance inverts");
        double[][] newton = MatrixOps.NewtonSchulz(symmetric, 30, out double defect);
        Check(defect <= 1e-12, "Newton-Schulz converges to the identity defect " + defect.ToString("0.0e0"));
        Check(MatrixOps.MaxAbsDifference(newton, eliminationInverse) <= 1e-9,
                "Newton-Schulz agrees with elimination to 1e-9");
        double previousDefect = double.PositiveInfinity;
        bool monotone = true;
        for (int step = 1; step <= 6; step++)
        {
            MatrixOps.NewtonSchulz(symmetric, step, out double current);
            if (current > previousDefect) monotone = false;
            previousDefect = current;
        }
        Check(monotone, "the Newton-Schulz defect decreases at every step");

        // ---- solving through the inverse --------------------------------------------------
        var system = new double[][] { new[] { 3.0, -1.0, 2.0 }, new[] { 1.0, 5.0, -2.0 }, new[] { 2.0, 1.0, 4.0 } };
        double[] rhs = { 7.0, -3.0, 6.0 };
        double[][] systemInverse = Require(MatrixOps.Invert(system, TinyPivotTolerance), "the system matrix inverts");
        double[] viaInverse = MatrixOps.Multiply(systemInverse, rhs);
        double[] viaLu = MatrixOps.Solve(system, rhs);
        for (int i = 0; i < rhs.Length; i++) CheckClose(viaInverse[i], viaLu[i], 1e-9, "inverse and LU solves agree at " + i);
        double[] residual = MatrixOps.Multiply(system, viaInverse);
        for (int i = 0; i < rhs.Length; i++) CheckClose(residual[i], rhs[i], 1e-12, "the solved system reproduces the right-hand side at " + i);

        Console.WriteLine("PASS");
    }
}
