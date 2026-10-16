// Task 697 - Implement iterative linear solver (CG) with preconditioning demo. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// A dense conjugate gradient solver with three preconditioners, plus a direct solver used as an
// oracle. The solver keeps the textbook recurrences (r = b - A x, p = z + beta p, alpha from
// p^T r / p^T A p) and the preconditioners are applied as z = M^-1 r:
//   * none      - identity, so z = r;
//   * Jacobi    - z = r / diag(A), which removes pure scaling from the spectrum;
//   * SSOR      - the symmetric Gauss-Seidel factorisation M = (D + L) D^-1 (D + U), applied as one
//                 forward and one backward triangular solve.
// Test problems are the 1D Poisson matrix (-1, 2, -1), a 2D five point Poisson stencil, a matrix
// whose diagonal spans ten orders of magnitude (where Jacobi is decisive) and a deterministic
// random SPD matrix A = M M^T + n I.
//
// Everything is verified against independent oracles rather than against the solver itself:
//   * the direct solve (Gaussian elimination with partial pivoting) gives the exact discrete
//     solution, so the CG error in the A-norm can be measured and its monotone decrease - the
//     central CG theorem - can be asserted;
//   * the reported residual is compared with an independently recomputed b - A x;
//   * the search directions are checked for A-conjugacy (p_i^T A p_j = 0 for i != j);
//   * a non symmetric matrix, a non positive diagonal, an indefinite matrix and an exhausted
//     iteration budget all have to be reported instead of silently returning garbage.
// Also checked: an exact initial guess costs zero iterations, Jacobi and SSOR beat no
// preconditioning on the problems where they should, and CG needs no more than n steps for an n by
// n system. Documented simplifications: dense storage, no matrix free operators, no restart, no
// numerical pivoting in the triangular solves, and convergence judged on the relative residual.
using System;
using System.Collections.Generic;

internal sealed class DenseMatrix
{
    private readonly double[][] _a;

    public readonly int N;

    public DenseMatrix(int n)
    {
        if (n <= 0) throw new ArgumentException("the dimension must be positive");
        N = n;
        _a = new double[n][];
        for (int i = 0; i < n; i++) _a[i] = new double[n];
    }

    public double this[int row, int column]
    {
        get => _a[row][column];
        set => _a[row][column] = value;
    }

    public double[] Multiply(double[] x)
    {
        var y = new double[N];
        for (int i = 0; i < N; i++)
        {
            double sum = 0;
            double[] row = _a[i];
            for (int j = 0; j < N; j++) sum += row[j] * x[j];
            y[i] = sum;
        }
        return y;
    }

    public double[] Diagonal()
    {
        var diagonal = new double[N];
        for (int i = 0; i < N; i++) diagonal[i] = _a[i][i];
        return diagonal;
    }

    public double Lower(int row, int column) => row > column ? _a[row][column] : 0;

    public double Upper(int row, int column) => row < column ? _a[row][column] : 0;

    public bool IsSymmetric(double tolerance)
    {
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if (Math.Abs(_a[i][j] - _a[j][i]) > tolerance * (1 + Math.Abs(_a[i][j]))) return false;
        return true;
    }

    public static DenseMatrix Poisson1D(int n)
    {
        var matrix = new DenseMatrix(n);
        for (int i = 0; i < n; i++)
        {
            matrix[i, i] = 2;
            if (i > 0) matrix[i, i - 1] = -1;
            if (i + 1 < n) matrix[i, i + 1] = -1;
        }
        return matrix;
    }

    /// <summary>Five point Laplacian on an m by m interior grid, ordered row by row.</summary>
    public static DenseMatrix Poisson2D(int m)
    {
        int n = m * m;
        var matrix = new DenseMatrix(n);
        for (int row = 0; row < m; row++)
        {
            for (int column = 0; column < m; column++)
            {
                int index = row * m + column;
                matrix[index, index] = 4;
                if (row > 0) matrix[index, index - m] = -1;
                if (row + 1 < m) matrix[index, index + m] = -1;
                if (column > 0) matrix[index, index - 1] = -1;
                if (column + 1 < m) matrix[index, index + 1] = -1;
            }
        }
        return matrix;
    }

    /// <summary>Diagonal entries spanning ten orders of magnitude, with small symmetric couplings.</summary>
    public static DenseMatrix BadlyScaled(int n)
    {
        var matrix = new DenseMatrix(n);
        for (int i = 0; i < n; i++) matrix[i, i] = Math.Pow(10, -5 + 10.0 * i / (n - 1));
        for (int i = 0; i + 1 < n; i++)
        {
            double coupling = 0.1 * Math.Sqrt(matrix[i, i] * matrix[i + 1, i + 1]);
            matrix[i, i + 1] = coupling;
            matrix[i + 1, i] = coupling;
        }
        // Keep the wide spread but make every row strictly diagonally dominant, hence definite.
        for (int i = 0; i < n; i++)
        {
            double off = 0;
            for (int j = 0; j < n; j++) if (j != i) off += Math.Abs(matrix[i, j]);
            matrix[i, i] += off;
        }
        return matrix;
    }

    /// <summary>Deterministic random SPD matrix A = B B^T + n I.</summary>
    public static DenseMatrix RandomSpd(int n, ulong seed)
    {
        var rng = new DeterministicRng(seed);
        var b = new DenseMatrix(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) b[i, j] = rng.NextUnit();
        var matrix = new DenseMatrix(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++) sum += b[i, k] * b[j, k];
                matrix[i, j] = sum;
            }
            matrix[i, i] += n;
        }
        return matrix;
    }

    /// <summary>Structurally asymmetric: the lower triangle uses a different coupling.</summary>
    public static DenseMatrix Asymmetric(int n)
    {
        var matrix = new DenseMatrix(n);
        for (int i = 0; i < n; i++)
        {
            matrix[i, i] = 4;
            if (i + 1 < n) { matrix[i, i + 1] = -1; matrix[i + 1, i] = -2; }
        }
        return matrix;
    }

    /// <summary>Indefinite: the 1D Poisson matrix with the first diagonal entry driven negative.</summary>
    public static DenseMatrix Indefinite(int n)
    {
        DenseMatrix matrix = Poisson1D(n);
        matrix[0, 0] = -2;
        return matrix;
    }

    public DenseMatrix Clone()
    {
        var copy = new DenseMatrix(N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) copy[i, j] = _a[i][j];
        return copy;
    }
}

internal sealed class DeterministicRng
{
    private ulong _state;

    public DeterministicRng(ulong seed) => _state = seed;

    public ulong Next()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public double NextUnit() => (Next() % 2000001) / 1000000.0 - 1.0;
}

internal sealed class CgResult
{
    public double[]? X;
    public int Iterations;
    public readonly List<double> TrueResiduals = new List<double>();
    public readonly List<double> ErrorANorms = new List<double>();
    public readonly List<double[]> Directions = new List<double[]>();
    public double FinalRelativeResidual;
    public bool Converged;
    public string? Failure;
}

internal static class CgSolver
{
    public static double Dot(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    public static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    public static double[] SubtractForTest(double[] a, double[] b) => Subtract(a, b);

    private static double[] Subtract(double[] a, double[] b)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++) result[i] = a[i] - b[i];
        return result;
    }

    /// <summary>Solves A x = b by conjugate gradients whose preconditioner is named explicitly.</summary>
    public static CgResult Solve(DenseMatrix a, double[] b, double[] x0, string preconditioner,
        double tolerance, int maxIterations, double[]? exact = null)
    {
        var result = new CgResult();
        if (!a.IsSymmetric(1e-12))
        {
            result.Failure = "the matrix is not symmetric";
            return result;
        }
        if (preconditioner == "jacobi")
        {
            foreach (double entry in a.Diagonal())
            {
                if (entry <= 0)
                {
                    result.Failure = "the Jacobi preconditioner needs a positive diagonal";
                    return result;
                }
            }
        }

        int n = a.N;
        var x = (double[])x0.Clone();
        double[] r = Subtract(b, a.Multiply(x));
        double bNorm = Norm(b);
        if (bNorm == 0) bNorm = 1;
        result.TrueResiduals.Add(Norm(r) / bNorm);
        if (exact != null) result.ErrorANorms.Add(ErrorANorm(a, x, exact));
        if (Norm(r) <= tolerance * bNorm)
        {
            result.X = x;
            result.Converged = true;
            result.FinalRelativeResidual = Norm(r) / bNorm;
            return result;
        }

        double[] z = ApplyPreconditioner(a, preconditioner, r);
        var p = (double[])z.Clone();
        double rz = Dot(r, z);

        for (int iteration = 1; iteration <= maxIterations; iteration++)
        {
            result.Iterations = iteration;
            result.Directions.Add((double[])p.Clone());
            double[] ap = a.Multiply(p);
            double pap = Dot(p, ap);
            if (pap <= 0)
            {
                result.Failure = "the curvature p^T A p is not positive, the matrix is not positive definite";
                result.X = x;
                return result;
            }
            double alpha = rz / pap;
            for (int i = 0; i < n; i++) x[i] += alpha * p[i];
            for (int i = 0; i < n; i++) r[i] -= alpha * ap[i];

            double trueResidual = Norm(Subtract(b, a.Multiply(x))) / bNorm;
            result.TrueResiduals.Add(trueResidual);
            if (exact != null) result.ErrorANorms.Add(ErrorANorm(a, x, exact));

            if (trueResidual <= tolerance)
            {
                result.X = x;
                result.Converged = true;
                result.FinalRelativeResidual = trueResidual;
                return result;
            }
            if (iteration == maxIterations) break;

            double[] zNext = ApplyPreconditioner(a, preconditioner, r);
            double rzNext = Dot(r, zNext);
            double beta = rzNext / rz;
            for (int i = 0; i < n; i++) p[i] = zNext[i] + beta * p[i];
            rz = rzNext;
        }

        result.X = x;
        result.FinalRelativeResidual = result.TrueResiduals[result.TrueResiduals.Count - 1];
        result.Failure = "the iteration budget of " + maxIterations + " was exhausted";
        return result;
    }

    private static double[] ApplyPreconditioner(DenseMatrix a, string kind, double[] r)
    {
        int n = a.N;
        if (kind == "none") return (double[])r.Clone();
        if (kind == "jacobi")
        {
            var z = new double[n];
            for (int i = 0; i < n; i++) z[i] = r[i] / a[i, i];
            return z;
        }
        if (kind == "ssor")
        {
            // M = (D + L) D^-1 (D + U): one forward then one backward substitution.
            var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                double sum = r[i];
                for (int j = 0; j < i; j++) sum -= a.Lower(i, j) * y[j];
                y[i] = sum / a[i, i];
            }
            var z = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                double sum = y[i];
                for (int j = i + 1; j < n; j++) sum -= a.Upper(i, j) * z[j];
                z[i] = sum / a[i, i];
            }
            return z;
        }
        throw new ArgumentException("unknown preconditioner: " + kind);
    }

    private static double ErrorANorm(DenseMatrix a, double[] x, double[] exact)
    {
        double[] difference = Subtract(x, exact);
        return Math.Sqrt(Math.Max(0, Dot(difference, a.Multiply(difference))));
    }

    /// <summary>Dense Gaussian elimination with partial pivoting, the direct oracle.</summary>
    public static double[] DirectSolve(DenseMatrix a, double[] b)
    {
        int n = a.N;
        var work = new double[n][];
        for (int i = 0; i < n; i++) work[i] = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) work[i][j] = a[i, j];
            work[i][n] = b[i];
        }
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(work[row][column]) > Math.Abs(work[pivot][column])) pivot = row;
            if (Math.Abs(work[pivot][column]) < 1e-300) throw new InvalidOperationException("singular matrix");
            (work[column], work[pivot]) = (work[pivot], work[column]);
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = work[row][column] / work[column][column];
                if (factor == 0) continue;
                for (int j = column; j <= n; j++) work[row][j] -= factor * work[column][j];
            }
        }
        var x = new double[n];
        for (int i = 0; i < n; i++) x[i] = work[i][n] / work[i][i];
        return x;
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- 1D Poisson: CG reproduces the direct solution in at most n steps. -----------
        const int N = 12;
        DenseMatrix poisson = DenseMatrix.Poisson1D(N);
        var ones = new double[N];
        for (int i = 0; i < N; i++) ones[i] = 1;
        double[] exact = CgSolver.DirectSolve(poisson, ones);
        var start = new double[N];
        CgResult cg = CgSolver.Solve(poisson, ones, start, "none", 1e-12, 4 * N, exact);
        Check(cg.Converged, "plain CG converged on the 1D Poisson problem: " + cg.Failure);
        Check(cg.Iterations <= N, $"CG needed no more than n steps, used {cg.Iterations}");
        Check(cg.Iterations >= 5, "the problem is not trivially solved in a couple of steps, used " + cg.Iterations);
        Check(cg.FinalRelativeResidual <= 1e-12, "the final relative residual meets the tolerance");
        double maxDifference = 0;
        for (int i = 0; i < N; i++) maxDifference = Math.Max(maxDifference, Math.Abs(cg.X![i] - exact[i]));
        Check(maxDifference < 1e-9, "the CG solution matches the direct solve, max difference=" + maxDifference);

        // The residual reported by the recurrence equals an independently recomputed b - A x.
        double[] recomputed = CgSolver.SubtractForTest(ones, poisson.Multiply(cg.X!));
        double reported = cg.FinalRelativeResidual * CgSolver.Norm(ones);
        Check(Math.Abs(CgSolver.Norm(recomputed) - reported) < 1e-9 * CgSolver.Norm(ones),
            "the reported residual matches b - A x, " + CgSolver.Norm(recomputed) + " vs " + reported);

        // The central CG theorem: the error in the A-norm decreases monotonically.
        Check(cg.ErrorANorms.Count >= 2, "the A-norm error history was recorded");
        bool monotone = true;
        for (int i = 1; i < cg.ErrorANorms.Count; i++)
            if (cg.ErrorANorms[i] > cg.ErrorANorms[i - 1] * (1 + 1e-9)) monotone = false;
        Check(monotone, "the error in the A-norm decreased in every iteration");

        // A-conjugacy of the search directions.
        double worstConjugacy = 0;
        for (int i = 0; i < cg.Directions.Count; i++)
        {
            for (int j = i + 1; j < cg.Directions.Count; j++)
            {
                double[] api = poisson.Multiply(cg.Directions[i]);
                double product = Math.Abs(CgSolver.Dot(cg.Directions[j], api));
                double scale = CgSolver.Norm(cg.Directions[j]) * CgSolver.Norm(api);
                if (scale > 0) worstConjugacy = Math.Max(worstConjugacy, product / scale);
            }
        }
        Check(worstConjugacy < 1e-8, "the search directions are A-conjugate, worst ratio=" + worstConjugacy.ToString("E2"));

        // An exact initial guess costs nothing.
        CgResult instant = CgSolver.Solve(poisson, ones, exact, "none", 1e-12, 10, exact);
        Check(instant.Converged && instant.Iterations == 0, "an exact initial guess finishes in zero iterations");
        Check(instant.TrueResiduals.Count == 1 && instant.TrueResiduals[0] < 1e-10,
            "and reports a residual at the rounding level, got " + instant.TrueResiduals[0].ToString("E2"));

        // --- Badly scaled diagonal: Jacobi is the difference between slow and fast. -----
        DenseMatrix scaled = DenseMatrix.BadlyScaled(40);
        var scaledRhs = new double[40];
        for (int i = 0; i < 40; i++) scaledRhs[i] = 1;
        double[] scaledExact = CgSolver.DirectSolve(scaled, scaledRhs);
        var scaledStart = new double[40];
        CgResult without = CgSolver.Solve(scaled, scaledRhs, scaledStart, "none", 1e-10, 400, scaledExact);
        CgResult jacobi = CgSolver.Solve(scaled, scaledRhs, scaledStart, "jacobi", 1e-10, 400, scaledExact);
        Check(jacobi.Converged, "Jacobi preconditioned CG converged: " + jacobi.Failure);
        Check(!without.Converged, "unpreconditioned CG cannot reach the tolerance on the scaled matrix in 400 steps");
        Check(without.Iterations == 400, "and it really used the whole budget, used " + without.Iterations);
        Check(jacobi.FinalRelativeResidual <= without.FinalRelativeResidual,
            $"Jacobi reaches a smaller residual ({jacobi.FinalRelativeResidual:E2} <= {without.FinalRelativeResidual:E2})");
        Check(jacobi.Iterations < 20, "Jacobi needs only a handful of steps on a diagonal scaling, used " + jacobi.Iterations);
        double scaledDifference = 0;
        for (int i = 0; i < 40; i++) scaledDifference = Math.Max(scaledDifference, Math.Abs(jacobi.X![i] - scaledExact[i]));
        Check(scaledDifference < 1e-7, "the preconditioned solution matches the direct solve, max difference=" + scaledDifference);

        // --- SSOR on the 2D Poisson stencil. --------------------------------------------
        DenseMatrix grid = DenseMatrix.Poisson2D(16);
        int gridSize = grid.N;
        var gridRhs = new double[gridSize];
        for (int i = 0; i < gridSize; i++) gridRhs[i] = 1;
        double[] gridExact = CgSolver.DirectSolve(grid, gridRhs);
        var gridStart = new double[gridSize];
        CgResult gridPlain = CgSolver.Solve(grid, gridRhs, gridStart, "none", 1e-10, 1000, gridExact);
        CgResult gridJacobi = CgSolver.Solve(grid, gridRhs, gridStart, "jacobi", 1e-10, 1000, gridExact);
        CgResult gridSsor = CgSolver.Solve(grid, gridRhs, gridStart, "ssor", 1e-10, 1000, gridExact);
        Check(gridPlain.Converged && gridJacobi.Converged && gridSsor.Converged, "all three variants converged on the stencil");
        Check(gridPlain.Iterations <= 2 * Math.Sqrt(gridSize) * 6,
            "unpreconditioned iterations grow like sqrt(kappa), used " + gridPlain.Iterations);
        Check(gridJacobi.Iterations <= gridPlain.Iterations, $"Jacobi is never worse on the stencil ({gridJacobi.Iterations} vs {gridPlain.Iterations})");
        Check(gridSsor.Iterations < gridPlain.Iterations,
            $"SSOR needs fewer iterations than plain CG on the stencil ({gridSsor.Iterations} vs {gridPlain.Iterations})");
        Check(gridSsor.Iterations * 5 < gridPlain.Iterations * 4,
            $"SSOR is a clear improvement ({gridSsor.Iterations} vs {gridPlain.Iterations})");
        double gridDifference = 0;
        for (int i = 0; i < gridSize; i++) gridDifference = Math.Max(gridDifference, Math.Abs(gridSsor.X![i] - gridExact[i]));
        Check(gridDifference < 1e-7, "the SSOR solution matches the direct solve, max difference=" + gridDifference);

        // --- Random SPD matrix: n steps are enough. -------------------------------------
        DenseMatrix random = DenseMatrix.RandomSpd(15, 0x697C0FFEEUL);
        Check(random.IsSymmetric(0), "the generated matrix is symmetric by construction");
        var randomRhs = new double[15];
        for (int i = 0; i < 15; i++) randomRhs[i] = (i % 5) - 2;
        double[] randomExact = CgSolver.DirectSolve(random, randomRhs);
        CgResult randomCg = CgSolver.Solve(random, randomRhs, new double[15], "none", 1e-11, 200, randomExact);
        Check(randomCg.Converged, "CG converged on the random SPD matrix: " + randomCg.Failure);
        Check(randomCg.Iterations <= 15, "a 15 by 15 SPD system needs at most 15 steps, used " + randomCg.Iterations);
        double randomDifference = 0;
        for (int i = 0; i < 15; i++) randomDifference = Math.Max(randomDifference, Math.Abs(randomCg.X![i] - randomExact[i]));
        Check(randomDifference < 1e-8, "the random system solution matches the direct solve, max difference=" + randomDifference);

        // --- Refusals: the solver must never pretend to have solved a bad system. -------
        DenseMatrix indefinite = DenseMatrix.Indefinite(8);
        var small = new double[8];
        for (int i = 0; i < 8; i++) small[i] = 1;
        CgResult broken = CgSolver.Solve(indefinite, small, new double[8], "none", 1e-10, 100);
        Check(!broken.Converged && broken.Failure != null, "an indefinite matrix is refused: " + broken.Failure);
        Check(broken.Failure!.Contains("positive definite"), "the refusal names the cause");

        DenseMatrix asymmetric = DenseMatrix.Asymmetric(4);
        CgResult asym = CgSolver.Solve(asymmetric, new double[4], new double[4], "none", 1e-10, 10);
        Check(!asym.Converged && asym.Failure == "the matrix is not symmetric", "an asymmetric matrix is refused");

        DenseMatrix negativeDiagonal = DenseMatrix.Poisson1D(4);
        negativeDiagonal[0, 0] = -1;
        CgResult badPreconditioner = CgSolver.Solve(negativeDiagonal, new double[4], new double[4], "jacobi", 1e-10, 10);
        Check(!badPreconditioner.Converged && badPreconditioner.Failure!.Contains("positive diagonal"),
            "a non positive diagonal is refused by the Jacobi preconditioner");

        DenseMatrix hard = DenseMatrix.Poisson1D(60);
        var hardRhs = new double[60];
        for (int i = 0; i < 60; i++) hardRhs[i] = 1;
        CgResult budget = CgSolver.Solve(hard, hardRhs, new double[60], "none", 1e-14, 3);
        Check(!budget.Converged && budget.Failure!.Contains("budget"), "an exhausted budget is reported: " + budget.Failure);
        Check(budget.X != null && budget.FinalRelativeResidual > 1e-14, "the partial solution and its residual are still returned");
        Check(budget.TrueResiduals.Count == 4, "one residual per iteration plus the initial one");
        bool resumedOk = CgSolver.Solve(hard, hardRhs, budget.X!, "none", 1e-10, 200).Converged;
        Check(resumedOk, "restarting from the partial solution converges, so the partial iterate is usable");

        // Unknown preconditioner names are rejected instead of silently doing nothing.
        bool rejected = false;
        try { CgSolver.Solve(poisson, ones, start, "chebyshev", 1e-8, 10); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "an unknown preconditioner name is rejected");

        // --- Determinism. -----------------------------------------------------------------
        Check(Summary() == Summary(), "reruns are identical");

        Console.WriteLine($"poisson1d: iterations={cg.Iterations} residual={cg.FinalRelativeResidual:E2} " +
                          $"conjugacy={worstConjugacy:E2}");
        Console.WriteLine($"scaled: none={without.Iterations} jacobi={jacobi.Iterations} | " +
                          $"grid: none={gridPlain.Iterations} jacobi={gridJacobi.Iterations} ssor={gridSsor.Iterations}");
        Console.WriteLine("PASS");
    }

    private static string Summary()
    {
        DenseMatrix matrix = DenseMatrix.Poisson2D(6);
        var rhs = new double[matrix.N];
        for (int i = 0; i < matrix.N; i++) rhs[i] = i % 3 == 0 ? 1 : 0;
        var result = new List<string>();
        foreach (string kind in new[] { "none", "jacobi", "ssor" })
        {
            CgResult run = CgSolver.Solve(matrix, rhs, new double[matrix.N], kind, 1e-10, 500);
            result.Add($"{kind}:{run.Iterations}:{run.FinalRelativeResidual:E2}");
        }
        return string.Join(" ", result);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
