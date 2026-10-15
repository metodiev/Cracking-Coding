// Task 375 - Gaussian elimination with partial pivoting (doubles). (Tier: Intermediate, CTCI Ch 6)
using System;

/// <summary>
/// Solve A x = b for a small dense double matrix.  Forward elimination with partial
/// pivoting (swap in the row with the largest |pivot|), then back substitution.
/// Documented conventions:
///   * a pivot column whose remaining entries are all &lt;= EPS in absolute value is
///     treated as a free column - the system then has no unique solution;
///   * after elimination, a row with an all-zero coefficient part but |rhs| &gt; EPS is
///     reported "inconsistent";
///   * otherwise a rank-deficient consistent system is reported "infinite";
///   * results are computed in doubles, so verification by substitution uses the
///     documented tolerance TOL = 1e-9 (absolute) - the small hand-checked systems
///     below have exact integer solutions that the elimination reproduces to well
///     within that tolerance.
/// Sentinel: the status string ("unique", "infinite", "inconsistent") plus x = null for
/// anything that is not "unique".
///
/// Includes systems that require a pivot swap (leading zero in the first column) and a
/// 4x4 diagonally dominant system, plus LCG cross-checks: a random integer matrix A and
/// integer x* give b = A x*; the solver must recover x* whenever the elimination reports
/// a unique solution.
/// </summary>
public static class Solution
{
    private const double EPS = 1e-12;     // pivot/inconsistency threshold for the elimination
    private const double TOL = 1e-9;      // tolerance for verifying a solution by substitution

    private sealed class Lcg
    {
        private int seed;

        public Lcg(int seed)
        {
            this.seed = seed;
        }

        public int Next(int modulus)
        {
            seed = (seed * 75 + 74) % 65537;
            return seed % modulus;
        }
    }

    /// <summary>status in {"unique", "infinite", "inconsistent"}; x is null unless unique.</summary>
    private sealed class SolveResult
    {
        public string Status { get; }
        public double[]? X { get; }

        public SolveResult(string status, double[]? x)
        {
            Status = status;
            X = x;
        }
    }

    private static SolveResult Solve(double[][] matrix, double[] rhs)
    {
        int n = matrix.Length;
        var augmented = new double[n][];
        for (int i = 0; i < n; i++)
        {
            augmented[i] = new double[n + 1];
            for (int j = 0; j < n; j++) augmented[i][j] = matrix[i][j];
            augmented[i][n] = rhs[i];
        }

        int rank = 0;
        for (int col = 0; col < n; col++)
        {
            int pivotRow = -1;
            double pivotSize = 0.0;
            for (int row = rank; row < n; row++)
            {
                if (Math.Abs(augmented[row][col]) > pivotSize)
                {
                    pivotSize = Math.Abs(augmented[row][col]);
                    pivotRow = row;
                }
            }
            if (pivotRow == -1 || pivotSize <= EPS) continue;         // free column
            double[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++)
            {
                double factor = augmented[row][col] / augmented[rank][col];
                if (factor != 0.0)
                {
                    for (int k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
                }
            }
            rank++;
        }

        for (int row = rank; row < n; row++)
        {
            if (Math.Abs(augmented[row][n]) > EPS) return new SolveResult("inconsistent", null);
        }
        if (rank < n) return new SolveResult("infinite", null);

        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double total = augmented[row][n];
            for (int col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
            x[row] = total / augmented[row][row];
        }
        return new SolveResult("unique", x);
    }

    /// <summary>max |A x - b| - the independent substitution check.</summary>
    private static double Residual(double[][] matrix, double[] x, double[] rhs)
    {
        double worst = 0.0;
        for (int i = 0; i < matrix.Length; i++)
        {
            double total = 0.0;
            for (int j = 0; j < matrix[i].Length; j++) total += matrix[i][j] * x[j];
            worst = Math.Max(worst, Math.Abs(total - rhs[i]));
        }
        return worst;
    }

    private static (double[][] matrix, double[] rhs, int[] planted) LcgSystem(Lcg lcg)
    {
        int size = 2 + lcg.Next(3);
        var matrix = new double[size][];
        var intMatrix = new int[size][];
        for (int i = 0; i < size; i++)
        {
            matrix[i] = new double[size];
            intMatrix[i] = new int[size];
            for (int j = 0; j < size; j++)
            {
                intMatrix[i][j] = lcg.Next(11) - 5;
                matrix[i][j] = intMatrix[i][j];
            }
        }
        var planted = new int[size];
        for (int i = 0; i < size; i++) planted[i] = lcg.Next(11) - 5;
        var rhs = new double[size];
        for (int i = 0; i < size; i++)
        {
            int total = 0;
            for (int j = 0; j < size; j++) total += intMatrix[i][j] * planted[j];
            rhs[i] = total;
        }
        return (matrix, rhs, planted);
    }

    public static void Main()
    {
        // 2x2, unique integer solution (2, 1)
        SolveResult result = Solve(new[] { new[] { 1.0, 1.0 }, new[] { 1.0, -1.0 } }, new[] { 3.0, 1.0 });
        Check(result.Status == "unique", "2x2 unique");
        Check(Math.Abs(result.X![0] - 2) < TOL && Math.Abs(result.X[1] - 1) < TOL, "2x2 solution");

        // needs a pivot swap: first pivot is zero
        result = Solve(new[] { new[] { 0.0, 1.0 }, new[] { 1.0, 0.0 } }, new[] { 2.0, 3.0 });
        Check(result.Status == "unique", "pivot swap unique");
        Check(Math.Abs(result.X![0] - 3) < TOL && Math.Abs(result.X[1] - 2) < TOL, "pivot swap solution");

        // 3x3 permutation system: x = (1, 2, 3)
        result = Solve(
            new[] { new[] { 0.0, 0.0, 1.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 1.0, 0.0, 0.0 } },
            new[] { 3.0, 2.0, 1.0 });
        Check(result.Status == "unique", "permutation unique");
        for (int i = 0; i < 3; i++) Check(Math.Abs(result.X![i] - (i + 1)) < TOL, "permutation " + i);

        // classic 3x3 with solution (5, 3, -2): checked by hand
        double[][] matrix3 = { new[] { 1.0, 1.0, 1.0 }, new[] { 0.0, 2.0, 5.0 }, new[] { 2.0, 5.0, -1.0 } };
        double[] rhs3 = { 6.0, -4.0, 27.0 };
        Check(5 + 3 - 2 == 6 && 2 * 3 + 5 * -2 == -4 && 2 * 5 + 5 * 3 + 2 == 27, "hand check 3x3");
        result = Solve(matrix3, rhs3);
        Check(result.Status == "unique", "3x3 unique");
        Check(Math.Abs(result.X![0] - 5) < TOL && Math.Abs(result.X[1] - 3) < TOL
            && Math.Abs(result.X[2] + 2) < TOL, "3x3 solution");
        Check(Residual(matrix3, result.X!, rhs3) < TOL, "3x3 residual");

        // 4x4 diagonally dominant with solution (1, 2, 3, 4)
        double[][] matrix4 =
        {
            new[] { 2.0, 1.0, 1.0, 1.0 }, new[] { 1.0, 2.0, 1.0, 1.0 },
            new[] { 1.0, 1.0, 2.0, 1.0 }, new[] { 1.0, 1.0, 1.0, 2.0 },
        };
        double[] rhs4 = { 11.0, 12.0, 13.0, 14.0 };
        result = Solve(matrix4, rhs4);
        Check(result.Status == "unique", "4x4 unique");
        for (int i = 0; i < 4; i++) Check(Math.Abs(result.X![i] - (i + 1)) < TOL, "4x4 " + i);
        Check(Residual(matrix4, result.X!, rhs4) < TOL, "4x4 residual");

        // rank-deficient consistent -> infinite
        result = Solve(new[] { new[] { 1.0, 1.0 }, new[] { 2.0, 2.0 } }, new[] { 1.0, 2.0 });
        Check(result.Status == "infinite" && result.X == null, "infinite sentinel");
        // rank-deficient inconsistent -> inconsistent
        result = Solve(new[] { new[] { 1.0, 1.0 }, new[] { 2.0, 2.0 } }, new[] { 1.0, 2.5 });
        Check(result.Status == "inconsistent" && result.X == null, "inconsistent sentinel");
        // three-way dependent 3x3: row3 = row1 + row2
        result = Solve(
            new[] { new[] { 1.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 0.0 } },
            new[] { 1.0, 2.0, 3.0 });
        Check(result.Status == "infinite" && result.X == null, "3x3 dependent infinite");
        result = Solve(
            new[] { new[] { 1.0, 0.0, 0.0 }, new[] { 0.0, 1.0, 0.0 }, new[] { 1.0, 1.0, 0.0 } },
            new[] { 1.0, 2.0, 4.0 });
        Check(result.Status == "inconsistent" && result.X == null, "3x3 dependent inconsistent");

        // --- LCG cross-checks: b = A x* must recover x* -----------------------
        var lcg = new Lcg(37513);
        int uniqueCount = 0;
        for (int trial = 0; trial < 80; trial++)
        {
            (double[][] matrix, double[] rhs, int[] planted) = LcgSystem(lcg);
            result = Solve(matrix, rhs);
            if (result.Status != "unique") continue;      // singular random matrix; skipped
            uniqueCount++;
            for (int i = 0; i < planted.Length; i++)
            {
                Check(Math.Abs(result.X![i] - planted[i]) < 1e-6, "recover planted " + trial);
            }
            Check(Residual(matrix, result.X!, rhs) < TOL, "random residual " + trial);
        }
        Check(uniqueCount >= 40, "enough unique systems, got " + uniqueCount);
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
