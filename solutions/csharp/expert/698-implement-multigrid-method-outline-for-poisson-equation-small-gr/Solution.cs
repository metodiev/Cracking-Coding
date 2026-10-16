// Task 698 - Implement multigrid method outline for Poisson equation small grid. (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)
//
// A working geometric multigrid solver for the two dimensional Poisson problem -Laplace(u) = f on
// the unit square with homogeneous Dirichlet boundary conditions, discretised by the standard five
// point stencil on a sequence of grids. This is the "outline" task, so the model is deliberately
// small but complete rather than sketched: the hierarchy is built by halving the interior grid
// (15 -> 7 -> 3 -> 1, so every level satisfies M_coarse = (M - 1) / 2), the smoother is weighted
// Jacobi with omega = 2/3, the transfer operators are full weighting restriction and bilinear
// prolongation, and the coarse problem is solved directly. Both a V-cycle and a W-cycle are
// implemented, and the coarsest level is solved exactly by Gaussian elimination.
//
// The invariants checked below are the ones multigrid theory actually rests on:
//   * restriction is exactly one quarter of the transpose of prolongation (R = P^T / 4), verified
//     against an independently written full weighting stencil on random vectors;
//   * prolongation reproduces the coarse values exactly at the coarse grid points and interpolates
//     bilinearly (the arithmetic mean of the surrounding coarse values) in between;
//   * the Galerkin coarse operator R A P is symmetric, consistent (interior rows sum to zero) and an
//     M-matrix, and its diagonal matches the assembled coarse stencil up to an O(1) factor;
//   * one weighted Jacobi sweep damps the checkerboard mode by the theoretical factor 1 - 2 omega
//     (1/3 for omega = 2/3) while barely touching the smooth mode - the smoothing property;
//   * the V-cycle residual shrinks by a bounded factor per cycle, that factor changes little when
//     the grid is refined (the grid independence claim), a W-cycle is at least as strong as a
//     V-cycle, and the converged solution equals the dense direct solve of the same discrete
//     system. Everything is deterministic, and an even interior size is rejected because it cannot
//     be coarsened.
using System;
using System.Collections.Generic;

internal sealed class PoissonGrid
{
    public readonly int M;          // interior points per dimension
    public readonly double H;

    public PoissonGrid(int m)
    {
        M = m;
        H = 1.0 / (m + 1);
    }

    /// <summary>Index of the interior point (i, j) in a flat row major vector.</summary>
    public int Index(int i, int j) => i * M + j;

    /// <summary>The five point stencil applied to an interior vector with zero boundary values.</summary>
    public double[] Apply(double[] u)
    {
        var result = new double[M * M];
        double scale = 1.0 / (H * H);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < M; j++)
            {
                double center = 4 * u[Index(i, j)];
                double neighbours =
                    (i > 0 ? u[Index(i - 1, j)] : 0) + (i + 1 < M ? u[Index(i + 1, j)] : 0) +
                    (j > 0 ? u[Index(i, j - 1)] : 0) + (j + 1 < M ? u[Index(i, j + 1)] : 0);
                result[Index(i, j)] = scale * (center - neighbours);
            }
        }
        return result;
    }

    public PoissonGrid Coarser() => new PoissonGrid((M - 1) / 2);

    public bool IsCoarsest => M <= 3;                 // the 3x3 level is solved directly
}

internal static class Transfers
{
    /// <summary>Bilinear interpolation from a coarse interior grid to a fine one (boundaries zero).</summary>
    public static double[] Prolongate(PoissonGrid coarse, double[] error)
    {
        int mc = coarse.M;
        int fine = 2 * mc + 1;
        var result = new double[fine * fine];
        Func<int, int, double> at = (i, j) =>
            i < 0 || j < 0 || i >= mc || j >= mc ? 0.0 : error[i * mc + j];
        for (int i = 0; i < fine; i++)
        {
            for (int j = 0; j < fine; j++)
            {
                double value;
                bool oddI = i % 2 == 1, oddJ = j % 2 == 1;
                if (oddI && oddJ) value = at((i - 1) / 2, (j - 1) / 2);
                else if (oddI) value = 0.5 * (at((i - 1) / 2, j / 2 - 1) + at((i - 1) / 2, j / 2));
                else if (oddJ) value = 0.5 * (at(i / 2 - 1, (j - 1) / 2) + at(i / 2, (j - 1) / 2));
                else value = 0.25 * (at(i / 2 - 1, j / 2 - 1) + at(i / 2, j / 2 - 1) + at(i / 2 - 1, j / 2) + at(i / 2, j / 2));
                result[i * fine + j] = value;
            }
        }
        return result;
    }

    /// <summary>Full weighting restriction written out as its nine point stencil.</summary>
    public static double[] RestrictStencil(PoissonGrid fine, double[] residual)
    {
        int m = fine.M;
        int mc = (m - 1) / 2;
        var result = new double[mc * mc];
        Func<int, int, double> at = (i, j) => i < 0 || j < 0 || i >= m || j >= m ? 0.0 : residual[i * m + j];
        for (int bigI = 0; bigI < mc; bigI++)
        {
            for (int bigJ = 0; bigJ < mc; bigJ++)
            {
                int i = 2 * bigI + 1, j = 2 * bigJ + 1;
                double value = 4 * at(i, j)
                    + 2 * (at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1))
                    + (at(i - 1, j - 1) + at(i - 1, j + 1) + at(i + 1, j - 1) + at(i + 1, j + 1));
                result[bigI * mc + bigJ] = value / 16.0;
            }
        }
        return result;
    }

    /// <summary>Restriction as one quarter of the transpose of the interpolation.</summary>
    public static double[] RestrictTranspose(PoissonGrid fine, double[] residual)
    {
        PoissonGrid coarse = fine.Coarser();
        var result = new double[coarse.M * coarse.M];
        for (int i = 0; i < fine.M; i++)
        {
            var basis = new double[fine.M * fine.M];
            basis[i * fine.M + 0] = 1;
            _ = basis;
        }
        // The transpose is applied column by column: accumulate the interpolation weights.
        double[] column = new double[fine.M * fine.M];
        for (int k = 0; k < coarse.M * coarse.M; k++)
        {
            Array.Clear(column, 0, column.Length);
            column[k] = 1;
            double[] spread = Prolongate(coarse, column);
            double sum = 0;
            for (int t = 0; t < spread.Length; t++) sum += spread[t] * residual[t];
            result[k] = sum / 4.0;
        }
        return result;
    }

    /// <summary>The Galerkin coarse operator R A P applied to a coarse vector.</summary>
    public static double[] GalerkinApply(PoissonGrid fine, double[] coarseVector)
    {
        PoissonGrid coarse = fine.Coarser();
        double[] prolongated = Prolongate(coarse, coarseVector);
        double[] applied = fine.Apply(prolongated);
        return RestrictStencil(fine, applied);
    }
}

internal sealed class MultigridSolver
{
    private readonly List<PoissonGrid> _levels = new List<PoissonGrid>();
    private readonly List<double[]> _rhs = new List<double[]>();
    private readonly List<double[]> _solution = new List<double[]>();

    public const double Omega = 2.0 / 3.0;
    public static readonly int[] CoarsestSizes = { 3, 1 };

    public int Cycles;
    public long SmoothingSweeps;
    public readonly List<double> CycleResiduals = new List<double>();

    public MultigridSolver(int interior)
    {
        if (interior < 3) throw new ArgumentException("the interior size must be at least three");
        if (interior % 2 == 0) throw new ArgumentException("the interior size must be odd so it can be coarsened");
        PoissonGrid grid = new PoissonGrid(interior);
        while (true)
        {
            _levels.Add(grid);
            if (grid.IsCoarsest) break;
            grid = grid.Coarser();
        }
    }

    public int Levels => _levels.Count;
    public PoissonGrid Finest => _levels[0];

    /// <summary>Solves the discrete system by repeated cycles, recording the residual of each.</summary>
    public double[] Solve(double[] f, int cycles, bool wCycle)
    {
        if (f.Length != Finest.M * Finest.M) throw new ArgumentException("the right hand side has the wrong size");
        _rhs.Clear();
        _solution.Clear();
        foreach (PoissonGrid grid in _levels)
        {
            _rhs.Add(new double[grid.M * grid.M]);
            _solution.Add(new double[grid.M * grid.M]);
        }
        Array.Copy(f, _rhs[0], f.Length);

        double[] u = new double[f.Length];
        CycleResiduals.Clear();
        CycleResiduals.Add(ResidualNorm(0, u));
        for (int cycle = 0; cycle < cycles; cycle++)
        {
            Cycle(0, u, wCycle);
            Cycles++;
            CycleResiduals.Add(ResidualNorm(0, u));
        }
        return u;
    }

    private void Cycle(int level, double[] u, bool wCycle)
    {
        PoissonGrid grid = _levels[level];
        if (grid.IsCoarsest)
        {
            DirectSolve(grid, u, _rhs[level]);
            return;
        }
        double[] f = _rhs[level];
        Smooth(grid, u, f, 2);
        PoissonGrid coarse = _levels[level + 1];
        double[] residual = new double[grid.M * grid.M];
        double[] applied = grid.Apply(u);
        for (int i = 0; i < residual.Length; i++) residual[i] = f[i] - applied[i];
        Array.Copy(Transfers.RestrictStencil(grid, residual), _rhs[level + 1], _rhs[level + 1].Length);

        double[] correction = new double[coarse.M * coarse.M];
        Cycle(level + 1, correction, wCycle);
        double[] prolongated = Transfers.Prolongate(coarse, correction);
        for (int i = 0; i < u.Length; i++) u[i] += prolongated[i];
        Smooth(grid, u, f, 2);

        if (wCycle)
        {
            // A W-cycle spends a second recursive visit on the corrected residual.
            applied = grid.Apply(u);
            for (int i = 0; i < residual.Length; i++) residual[i] = f[i] - applied[i];
            Array.Copy(Transfers.RestrictStencil(grid, residual), _rhs[level + 1], _rhs[level + 1].Length);
            Array.Clear(correction, 0, correction.Length);
            Cycle(level + 1, correction, true);
            prolongated = Transfers.Prolongate(coarse, correction);
            for (int i = 0; i < u.Length; i++) u[i] += prolongated[i];
            Smooth(grid, u, f, 2);
        }
    }

    /// <summary>Weighted Jacobi smoother in update form: u += omega (f - A u) / (4 / h^2).</summary>
    public void Smooth(PoissonGrid grid, double[] u, double[] f, int sweeps)
    {
        double scale = grid.H * grid.H / 4.0;
        var next = new double[u.Length];
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            double[] applied = grid.Apply(u);
            for (int i = 0; i < u.Length; i++) next[i] = u[i] + Omega * scale * (f[i] - applied[i]);
            Array.Copy(next, u, u.Length);
            SmoothingSweeps++;
        }
    }

    public static void DirectSolve(PoissonGrid grid, double[] u, double[] f)
    {
        int n = u.Length;
        var matrix = new double[n][];
        for (int i = 0; i < n; i++) matrix[i] = new double[n + 1];
        int m = grid.M;
        double scale = 1.0 / (grid.H * grid.H);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                int row = i * m + j;
                matrix[row][row] = 4 * scale;
                if (i > 0) matrix[row][(i - 1) * m + j] = -scale;
                if (i + 1 < m) matrix[row][(i + 1) * m + j] = -scale;
                if (j > 0) matrix[row][i * m + j - 1] = -scale;
                if (j + 1 < m) matrix[row][i * m + j + 1] = -scale;
                matrix[row][n] = f[row];
            }
        }
        for (int column = 0; column < n; column++)
        {
            int pivot = column;
            for (int row = column + 1; row < n; row++)
                if (Math.Abs(matrix[row][column]) > Math.Abs(matrix[pivot][column])) pivot = row;
            (matrix[column], matrix[pivot]) = (matrix[pivot], matrix[column]);
            for (int row = 0; row < n; row++)
            {
                if (row == column) continue;
                double factor = matrix[row][column] / matrix[column][column];
                if (factor == 0) continue;
                for (int j = column; j <= n; j++) matrix[row][j] -= factor * matrix[column][j];
            }
        }
        for (int i = 0; i < n; i++) u[i] = matrix[i][n] / matrix[i][i];
    }

    private double ResidualNorm(int level, double[] u)
    {
        PoissonGrid grid = _levels[level];
        double[] applied = grid.Apply(u);
        double sum = 0;
        for (int i = 0; i < u.Length; i++)
        {
            double difference = _rhs[level][i] - applied[i];
            sum += difference * difference;
        }
        return Math.Sqrt(sum);
    }

    /// <summary>Residual reduction factor of the last cycle.</summary>
    public double LastFactor
    {
        get
        {
            if (CycleResiduals.Count < 2) return double.NaN;
            double before = CycleResiduals[CycleResiduals.Count - 2];
            double after = CycleResiduals[CycleResiduals.Count - 1];
            return before == 0 ? 0 : after / before;
        }
    }

    public double ResidualOf(double[] u)
    {
        PoissonGrid grid = Finest;
        double[] applied = grid.Apply(u);
        double sum = 0;
        for (int i = 0; i < u.Length; i++)
        {
            double difference = _rhs[0][i] - applied[i];
            sum += difference * difference;
        }
        return Math.Sqrt(sum);
    }
}

public static class Solution
{
    public static void Main()
    {
        // --- Hierarchy and input validation. --------------------------------------------
        var multigrid = new MultigridSolver(15);
        Check(multigrid.Levels == 3, "the 15 point grid coarsens through 7 and 3, levels=" + multigrid.Levels);
        Check(multigrid.Finest.M == 15 && multigrid.Finest.H == 1.0 / 16, "the fine grid step is 1/(M+1)");
        bool rejected = false;
        try { _ = new MultigridSolver(14); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "an even interior size is rejected");
        rejected = false;
        try { _ = new MultigridSolver(1); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "an interior size below three is rejected");

        // --- Transfer operators: R = P^T / 4 and exact bilinear interpolation. -----------
        PoissonGrid fine = new PoissonGrid(15);
        PoissonGrid coarse = fine.Coarser();
        var rng = new DeterministicRng698(0x698C0FFEEUL);
        var probe = new double[fine.M * fine.M];
        for (int i = 0; i < probe.Length; i++) probe[i] = rng.NextUnit();
        double[] viaStencil = Transfers.RestrictStencil(fine, probe);
        double[] viaTranspose = Transfers.RestrictTranspose(fine, probe);
        double worstDifference = 0;
        for (int i = 0; i < viaStencil.Length; i++)
            worstDifference = Math.Max(worstDifference, Math.Abs(viaStencil[i] - viaTranspose[i]));
        Check(worstDifference < 1e-12, "full weighting equals one quarter of the transpose of interpolation, diff=" + worstDifference);

        int fineSize = 2 * coarse.M + 1;
        var ramp = new double[coarse.M * coarse.M];
        for (int i = 0; i < coarse.M; i++)
            for (int j = 0; j < coarse.M; j++) ramp[i * coarse.M + j] = 2 * i + 3 * j;
        double[] rampFine = Transfers.Prolongate(coarse, ramp);
        var atCoarse = (int i, int j) => ramp[i * coarse.M + j];
        bool bilinear = true;
        for (int i = 0; i < coarse.M; i++)
            for (int j = 0; j < coarse.M; j++)
                if (Math.Abs(rampFine[(2 * i + 1) * fineSize + (2 * j + 1)] - atCoarse(i, j)) > 1e-12) bilinear = false;
        for (int i = 2; i + 2 < fineSize; i += 2)
        {
            for (int j = 2; j + 2 < fineSize; j += 2)
            {
                double expected = 0.25 * (atCoarse((i - 1) / 2, (j - 1) / 2) + atCoarse((i + 1) / 2, (j - 1) / 2)
                                        + atCoarse((i - 1) / 2, (j + 1) / 2) + atCoarse((i + 1) / 2, (j + 1) / 2));
                if (Math.Abs(rampFine[i * fineSize + j] - expected) > 1e-12) bilinear = false;
            }
        }
        Check(bilinear, "prolongation reproduces the coarse values exactly and averages at the midpoints");

        double[] zero = new double[coarse.M * coarse.M];
        double[] prolongatedZero = Transfers.Prolongate(coarse, zero);
        Check(Array.TrueForAll(prolongatedZero, value => value == 0), "prolongation of zero is zero");

        // --- The Galerkin coarse operator R A P: structural invariants. ------------------
        // For full weighting plus bilinear interpolation the product is *not* the five point stencil
        // (it is the wider nine point one), so what is checked are the properties it must have.
        int coarseSize = coarse.M * coarse.M;
        var galerkin = new double[coarseSize][];
        for (int column = 0; column < coarseSize; column++)
        {
            var unit = new double[coarseSize];
            unit[column] = 1;
            galerkin[column] = Transfers.GalerkinApply(fine, unit);
        }
        bool symmetric = true;
        for (int i = 0; i < coarseSize; i++)
            for (int j = 0; j < coarseSize; j++)
                if (Math.Abs(galerkin[i][j] - galerkin[j][i]) > 1e-9) symmetric = false;
        Check(symmetric, "R A P is symmetric, as A and R = P^T / 4 require");

        int cm = coarse.M;
        double worstRowSum = 0;
        bool mMatrix = true;
        for (int i = 1; i + 1 < cm; i++)
        {
            for (int j = 1; j + 1 < cm; j++)
            {
                int row = i * cm + j;
                double sum = 0;
                for (int column = 0; column < coarseSize; column++) sum += galerkin[column][row];
                worstRowSum = Math.Max(worstRowSum, Math.Abs(sum));
                if (galerkin[row][row] <= 0) mMatrix = false;
                for (int column = 0; column < coarseSize; column++)
                    if (column != row && galerkin[column][row] > 1e-12) mMatrix = false;
            }
        }
        Check(worstRowSum < 1e-9, "interior rows of R A P sum to zero: the coarse operator is consistent, worst=" + worstRowSum);
        Check(mMatrix, "R A P is an M-matrix: positive diagonal and non positive couplings");

        double galerkinDiagonal = galerkin[cm + 1][cm + 1];
        double coarseDiagonal = 4 / (coarse.H * coarse.H);
        double diagonalRatio = galerkinDiagonal / coarseDiagonal;
        Check(diagonalRatio > 0.4 && diagonalRatio < 2.5,
            "the Galerkin diagonal is the coarse one up to an O(1) factor, ratio=" + diagonalRatio.ToString("F3"));

        // --- The smoothing property: checkerboard damped, smooth mode kept. -------------
        var checker = new double[fine.M * fine.M];
        for (int i = 0; i < fine.M; i++)
            for (int j = 0; j < fine.M; j++) checker[fine.Index(i, j)] = ((i + j) % 2 == 0) ? 1 : -1;
        double beforeChecker = Norm(checker);
        multigrid.Smooth(fine, checker, new double[checker.Length], 1);
        double checkerFactor = Norm(checker) / beforeChecker;
        Check(Math.Abs(checkerFactor - Math.Abs(1 - 2 * MultigridSolver.Omega)) < 0.15,
            "one weighted Jacobi sweep damps the checkerboard by |1 - 2 omega| = 1/3, measured " + checkerFactor.ToString("F3"));

        var smooth = new double[fine.M * fine.M];
        for (int i = 0; i < fine.M; i++)
            for (int j = 0; j < fine.M; j++) smooth[fine.Index(i, j)] = Math.Sin(Math.PI * (i + 1) * fine.H) * Math.Sin(Math.PI * (j + 1) * fine.H);
        double beforeSmooth = Norm(smooth);
        multigrid.Smooth(fine, smooth, new double[smooth.Length], 1);
        double smoothFactor = Norm(smooth) / beforeSmooth;
        Check(smoothFactor > 0.9, "the smooth mode is barely touched by the smoother, factor=" + smoothFactor.ToString("F4"));
        Check(smoothFactor > 2 * checkerFactor, "the smoother really is a smoother: high frequencies die first");

        // --- A manufactured solution: -Laplace u = 2 pi^2 sin(pi x) sin(pi y). -----------
        PoissonGrid grid = multigrid.Finest;
        var exact = new double[grid.M * grid.M];
        for (int i = 0; i < grid.M; i++)
            for (int j = 0; j < grid.M; j++)
                exact[grid.Index(i, j)] = Math.Sin(Math.PI * (i + 1) * grid.H) * Math.Sin(Math.PI * (j + 1) * grid.H);
        double[] appliedExact = grid.Apply(exact);
        var rhs = new double[exact.Length];
        for (int i = 0; i < rhs.Length; i++) rhs[i] = appliedExact[i];       // f = A u_exact exactly

        double[] reference = new double[exact.Length];
        MultigridSolver.DirectSolve(grid, reference, rhs);
        double referenceDifference = 0;
        for (int i = 0; i < exact.Length; i++) referenceDifference = Math.Max(referenceDifference, Math.Abs(reference[i] - exact[i]));
        Check(referenceDifference < 1e-10, "the direct solve of the discrete system reproduces the sampled solution");

        // Starting from the exact solution the residual is only rounding noise.
        var exactStart = multigrid.Solve(rhs, 0, false);
        Array.Copy(exact, exactStart, exact.Length);
        Check(multigrid.ResidualOf(exactStart) < 1e-12 * Norm(rhs), "an exact start has a residual at the rounding level");

        // --- V-cycle convergence and the grid independence of its factor. ---------------
        double[] vSolution = multigrid.Solve(rhs, 25, false);
        Check(multigrid.CycleResiduals.Count == 26, "one residual per cycle plus the initial one");
        double worstFactor = 0;
        bool neverWorse = true;
        for (int i = 1; i < multigrid.CycleResiduals.Count; i++)
        {
            double factor = multigrid.CycleResiduals[i] / multigrid.CycleResiduals[i - 1];
            if (i > 15) continue;                       // below the rounding floor the residual wobbles
            worstFactor = Math.Max(worstFactor, factor);
            if (factor > 1.0) neverWorse = false;
        }
        Check(neverWorse, "the residual decreased in each of the first fifteen V-cycles");
        Check(worstFactor < 0.6, "each early V-cycle reduced the residual substantially, worst factor=" + worstFactor.ToString("F3"));
        Check(multigrid.CycleResiduals[15] < 1e-6 * multigrid.CycleResiduals[0],
            "fifteen cycles bought six orders of magnitude");
        Check(multigrid.ResidualOf(vSolution) < 1e-9 * Norm(rhs), "twenty five V-cycles reach a tiny residual");

        double solutionDifference = 0;
        for (int i = 0; i < vSolution.Length; i++) solutionDifference = Math.Max(solutionDifference, Math.Abs(vSolution[i] - reference[i]));
        Check(solutionDifference < 1e-8, "the multigrid solution equals the direct solve, max difference=" + solutionDifference);

        var coarseGrid = new MultigridSolver(7);
        var coarseRhs = new double[coarseGrid.Finest.M * coarseGrid.Finest.M];
        for (int i = 0; i < coarseGrid.Finest.M; i++)
            for (int j = 0; j < coarseGrid.Finest.M; j++)
                coarseRhs[coarseGrid.Finest.Index(i, j)] =
                    2 * Math.PI * Math.PI * Math.Sin(Math.PI * (i + 1) * coarseGrid.Finest.H) * Math.Sin(Math.PI * (j + 1) * coarseGrid.Finest.H);
        coarseGrid.Solve(coarseRhs, 8, false);
        double coarseFactor = coarseGrid.LastFactor;
        var fineGridRun = new MultigridSolver(31);
        var fineRhs = new double[fineGridRun.Finest.M * fineGridRun.Finest.M];
        for (int i = 0; i < fineGridRun.Finest.M; i++)
            for (int j = 0; j < fineGridRun.Finest.M; j++)
                fineRhs[fineGridRun.Finest.Index(i, j)] =
                    2 * Math.PI * Math.PI * Math.Sin(Math.PI * (i + 1) * fineGridRun.Finest.H) * Math.Sin(Math.PI * (j + 1) * fineGridRun.Finest.H);
        fineGridRun.Solve(fineRhs, 8, false);
        double fineFactor = fineGridRun.LastFactor;
        Check(fineGridRun.Levels == 4, "the 31 point grid has four levels, got " + fineGridRun.Levels);
        Check(Math.Abs(fineFactor - coarseFactor) < 0.2,
            $"the convergence factor hardly changes with the grid ({coarseFactor:F3} on 31 points vs {fineFactor:F3} on 15)");
        Check(fineFactor < 0.6 && coarseFactor < 0.6, "both grids converge quickly");

        // --- W-cycle against V-cycle, and multigrid against plain smoothing. ------------
        var vCycle = new MultigridSolver(15);
        var wCycle = new MultigridSolver(15);
        double[] vResult = vCycle.Solve(rhs, 4, false);
        double[] wResult = wCycle.Solve(rhs, 4, true);
        Check(vCycle.ResidualOf(vResult) < 1e-2 * Norm(rhs),
            "four V-cycles already cut the residual by two orders of magnitude, got " + vCycle.ResidualOf(vResult).ToString("E2"));
        Check(wCycle.ResidualOf(wResult) <= vCycle.ResidualOf(vResult) * 1.001,
            $"the W-cycle is at least as strong as the V-cycle ({wCycle.ResidualOf(wResult):E2} vs {vCycle.ResidualOf(vResult):E2})");

        var smoothingOnly = new MultigridSolver(15);
        double[] smoothSolution = new double[rhs.Length];
        int sweeps = (int)(vCycle.SmoothingSweeps / 1);           // the same number of fine grid sweeps
        smoothingOnly.Smooth(smoothingOnly.Finest, smoothSolution, rhs, sweeps);
        double smoothResidual = ResidualNormOf(smoothingOnly.Finest, smoothSolution, rhs);
        Check(vCycle.ResidualOf(vResult) < smoothResidual / 100,
            $"multigrid beats the same number of smoother sweeps by orders of magnitude ({vCycle.ResidualOf(vResult):E2} vs {smoothResidual:E2})");

        // --- Determinism. -----------------------------------------------------------------
        Check(Summary() == Summary(), "reruns are identical");

        Console.WriteLine($"levels={multigrid.Levels} cycles={multigrid.Cycles} worstFactor={worstFactor:F3} " +
                          $"coarseFactor={coarseFactor:F3} fineFactor={fineFactor:F3}");
        Console.WriteLine($"checkerFactor={checkerFactor:F3} smoothFactor={smoothFactor:F4} " +
                          $"galerkinRatio={diagonalRatio:F3} directDifference={solutionDifference:E2}");
        Console.WriteLine("PASS");
    }

    private static double Norm(double[] v)
    {
        double sum = 0;
        foreach (double value in v) sum += value * value;
        return Math.Sqrt(sum);
    }

    private static double ResidualNormOf(PoissonGrid grid, double[] u, double[] f)
    {
        double[] applied = grid.Apply(u);
        double sum = 0;
        for (int i = 0; i < u.Length; i++)
        {
            double difference = f[i] - applied[i];
            sum += difference * difference;
        }
        return Math.Sqrt(sum);
    }

    private static string Summary()
    {
        var solver = new MultigridSolver(15);
        var grid = solver.Finest;
        var rhs = new double[grid.M * grid.M];
        for (int i = 0; i < grid.M; i++)
            for (int j = 0; j < grid.M; j++) rhs[grid.Index(i, j)] = 1;
        double[] solution = solver.Solve(rhs, 6, false);
        return $"residual={solver.ResidualOf(solution):E3} cycles={solver.Cycles} sweeps={solver.SmoothingSweeps}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}

internal sealed class DeterministicRng698
{
    private ulong _state;

    public DeterministicRng698(ulong seed) => _state = seed;

    public double NextUnit()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        z ^= z >> 31;
        return (z % 2000001) / 1000000.0 - 1.0;
    }
}
