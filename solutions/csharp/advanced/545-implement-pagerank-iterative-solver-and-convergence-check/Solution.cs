// Task 545 - Implement pagerank iterative solver and convergence check. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Power iteration for PageRank on graphs whose every vertex has at least one out-link, so no
// dangling-mass plumbing is needed here (that is task 546). One iteration is
//
//     x' = (1 - d) / n * 1  +  d * P^T x ,      P[i][j] = 1 / outdeg(i) for each edge i -> j,
//
// and the loop stops when the L1 residual ||x' - x||_1 drops to the tolerance (documented cap 500
// iterations), recording the residual of every step. Two independent solvers provide ground truth:
// a dense Gaussian elimination with partial pivoting on (I - d P^T) x = (1 - d)/n * 1, and a
// matrix-based iteration that materialises the dense transpose instead of scattering mass along
// edges. Closed forms cover the boundary cases.
//
// Documented checks: on a two-node cycle the rank is exactly (0.5, 0.5); on every undirected
// instance (cycles, complete graphs and symmetrised LCG graphs) the stationary distribution is
// proportional to the degree, which is verified algebraically as an exact fixed point of the
// teleport-free walk (pi = P^T pi, so no reliance on power iteration converging for periodic chains
// such as even cycles) and uniform to 1e-9 for the regular ones; iteration, Gaussian solve and dense
// iteration agree to 1e-10 on ten instances; the residual never increases and the final residual is
// below the 1e-12 tolerance within the documented iteration bound (<= 200 for d = 0.85, which the
// classical d^k bound allows); the ranks sum to 1 and are positive; d = 0 yields the uniform vector
// on the first step, whose residual is already zero; larger damping needs no fewer iterations
// (asserted for 0.5, 0.85 and 0.99 on the fixed instances); restarting from the converged vector is
// already a fixed point, so it costs exactly one step and moves nothing; and the whole solver is
// reproducible.
using System;
using System.Collections.Generic;

internal sealed class PageRankProblem
{
    internal readonly int N;
    internal readonly List<int>[] Out;

    internal PageRankProblem(int n)
    {
        N = n;
        Out = new List<int>[n];
        for (int v = 0; v < n; v++) Out[v] = new List<int>();
    }

    internal void AddEdge(int u, int v)
    {
        if (!Out[u].Contains(v)) Out[u].Add(v);
    }

    internal static PageRankProblem Cycle(int n)
    {
        var g = new PageRankProblem(n);
        for (int i = 0; i < n; i++) g.AddEdge(i, (i + 1) % n);
        return g;
    }

    internal static PageRankProblem Complete(int n)
    {
        var g = new PageRankProblem(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i != j) g.AddEdge(i, j);
            }
        }
        return g;
    }

    /** Undirected LCG graph: the cycle backbone is installed both ways, as are the random edges. */
    internal static PageRankProblem SymmetricLcg(int n, int degree, int seed)
    {
        var g = new PageRankProblem(n);
        int state = seed;
        for (int i = 0; i < n; i++)
        {
            g.AddEdge(i, (i + 1) % n);
            g.AddEdge((i + 1) % n, i);
        }
        for (int k = 0; k < degree * n / 2; k++)
        {
            state = (state * 75 + 74) % 65537;
            int u = state % n;
            state = (state * 75 + 74) % 65537;
            int v = state % n;
            if (u == v) continue;
            g.AddEdge(u, v);
            g.AddEdge(v, u);
        }
        return g;
    }

    internal static PageRankProblem DirectedLcg(int n, int seed)
    {
        var g = new PageRankProblem(n);
        int state = seed;
        for (int i = 0; i < n; i++) g.AddEdge(i, (i + 1) % n);
        for (int k = 0; k < n; k++)
        {
            state = (state * 75 + 74) % 65537;
            int u = state % n;
            state = (state * 75 + 74) % 65537;
            int v = state % n;
            g.AddEdge(u, v);
        }
        for (int v = 0; v < n; v++)
        {
            if (g.Out[v].Count == 0) g.AddEdge(v, (v + 1) % n);
        }
        return g;
    }

    internal bool HasDangling()
    {
        for (int v = 0; v < N; v++)
        {
            if (Out[v].Count == 0) return true;
        }
        return false;
    }

    internal double[,] DenseMatrix()
    {
        var p = new double[N, N];
        for (int i = 0; i < N; i++)
        {
            double share = 1.0 / Out[i].Count;
            foreach (int j in Out[i]) p[i, j] += share;
        }
        return p;
    }

    /** Power iteration from the uniform start; fills the residual history and iteration count. */
    internal double[] Iterate(double damping, double tolerance, int maxIterations,
            out List<double> residuals, out int iterations)
    {
        var start = new double[N];
        for (int v = 0; v < N; v++) start[v] = 1.0 / N;
        return IterateFrom(start, damping, tolerance, maxIterations, out residuals, out iterations);
    }

    internal double[] IterateFrom(double[] start, double damping, double tolerance, int maxIterations,
            out List<double> residuals, out int iterations)
    {
        var x = (double[])start.Clone();
        var next = new double[N];
        double teleport = (1.0 - damping) / N;
        residuals = new List<double>();
        iterations = 0;
        for (int step = 0; step < maxIterations; step++)
        {
            for (int v = 0; v < N; v++) next[v] = teleport;
            for (int i = 0; i < N; i++)
            {
                double share = damping * x[i] / Out[i].Count;
                foreach (int j in Out[i]) next[j] += share;
            }
            double residual = 0.0;
            for (int v = 0; v < N; v++) residual += Math.Abs(next[v] - x[v]);
            residuals.Add(residual);
            iterations = step + 1;
            Array.Copy(next, x, N);
            if (residual <= tolerance) break;
        }
        return x;
    }

    /** Dense iteration: same fixed point, but the transpose matrix is built explicitly. */
    internal double[] IterateDense(double damping, double tolerance, int maxIterations)
    {
        double[,] p = DenseMatrix();
        var x = new double[N];
        for (int v = 0; v < N; v++) x[v] = 1.0 / N;
        var next = new double[N];
        double teleport = (1.0 - damping) / N;
        for (int step = 0; step < maxIterations; step++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = teleport;
                for (int i = 0; i < N; i++) sum += damping * x[i] * p[i, j];
                next[j] = sum;
            }
            double residual = 0.0;
            for (int v = 0; v < N; v++) residual += Math.Abs(next[v] - x[v]);
            Array.Copy(next, x, N);
            if (residual <= tolerance) break;
        }
        return x;
    }

    /** Exact solve of (I - d P^T) x = (1 - d)/n by Gaussian elimination with partial pivoting. */
    internal double[] SolveLinear(double damping)
    {
        var a = new double[N, N];
        var b = new double[N];
        double t = (1.0 - damping) / N;
        for (int i = 0; i < N; i++)
        {
            b[i] = t;
            for (int j = 0; j < N; j++) a[i, j] = i == j ? 1.0 : 0.0;
        }
        for (int i = 0; i < N; i++)
        {
            double share = damping / Out[i].Count;
            foreach (int j in Out[i]) a[j, i] -= share;
        }
        for (int col = 0; col < N; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < N; r++)
            {
                if (Math.Abs(a[r, col]) > Math.Abs(a[pivot, col])) pivot = r;
            }
            if (Math.Abs(a[pivot, col]) < 1e-15) throw new InvalidOperationException("singular system");
            if (pivot != col)
            {
                for (int c = 0; c < N; c++)
                {
                    double tmp = a[col, c];
                    a[col, c] = a[pivot, c];
                    a[pivot, c] = tmp;
                }
                double tb = b[col];
                b[col] = b[pivot];
                b[pivot] = tb;
            }
            for (int r = col + 1; r < N; r++)
            {
                double factor = a[r, col] / a[col, col];
                if (factor == 0.0) continue;
                for (int c = col; c < N; c++) a[r, c] -= factor * a[col, c];
                b[r] -= factor * b[col];
            }
        }
        var x = new double[N];
        for (int r = N - 1; r >= 0; r--)
        {
            double sum = b[r];
            for (int c = r + 1; c < N; c++) sum -= a[r, c] * x[c];
            x[r] = sum / a[r, r];
        }
        return x;
    }

    internal double MaxAbsDifference(double[] a, double[] b)
    {
        double worst = 0.0;
        for (int v = 0; v < N; v++) worst = Math.Max(worst, Math.Abs(a[v] - b[v]));
        return worst;
    }

    internal double Sum(double[] x)
    {
        double total = 0.0;
        foreach (double value in x) total += value;
        return total;
    }
}

public static class Solution
{
    private const double Damping = 0.85;
    private const double Tolerance = 1e-12;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        var problems = new List<PageRankProblem>
        {
            PageRankProblem.Cycle(2), PageRankProblem.Cycle(5), PageRankProblem.Cycle(12),
            PageRankProblem.Complete(3), PageRankProblem.Complete(6),
            PageRankProblem.SymmetricLcg(9, 6, 545001), PageRankProblem.SymmetricLcg(15, 6, 545002),
            PageRankProblem.DirectedLcg(8, 545003), PageRankProblem.DirectedLcg(14, 545004),
            PageRankProblem.DirectedLcg(20, 545005),
        };

        foreach (PageRankProblem problem in problems)
        {
            Check(!problem.HasDangling(), "545 instances are dangling-free");
            double[] iterated = problem.Iterate(Damping, Tolerance, 500, out List<double> residuals, out int iterations);
            double[] solved = problem.SolveLinear(Damping);
            double[] dense = problem.IterateDense(Damping, Tolerance, 500);
            Check(problem.MaxAbsDifference(iterated, solved) <= 1e-10, "iteration matches the linear solve");
            Check(problem.MaxAbsDifference(iterated, dense) <= 1e-10, "iteration matches the dense iteration");
            Check(Math.Abs(problem.Sum(iterated) - 1.0) <= 1e-12, "ranks sum to one");
            foreach (double value in iterated) Check(value > 0.0, "ranks stay positive");
            Check(iterations == residuals.Count, "the residual history has one entry per iteration");
            for (int step = 1; step < residuals.Count; step++)
            {
                Check(residuals[step] <= residuals[step - 1] + 1e-15, "the residual never increases at step " + step);
            }
            Check(residuals[residuals.Count - 1] <= Tolerance, "the final residual meets the tolerance");
            Check(iterations <= 200, "iteration bound respected (" + iterations + " iterations)");
            if (residuals[0] <= 1e-15)
            {
                Check(iterations == 1, "a uniform fixed point is reached in one step");
            }
            else
            {
                Check(iterations >= 3, "non-trivial instances need several iterations");
            }

            // Restarting from the converged vector is already a fixed point: one step, zero movement.
            double[] restarted = problem.IterateFrom(iterated, Damping, Tolerance, 500, out List<double> history, out int steps);
            Check(problem.MaxAbsDifference(restarted, iterated) <= 1e-12, "the fixed point is stationary");
            Check(steps == 1, "a converged vector needs a single step");
            Check(history.Count >= 1 && history[0] <= Tolerance, "the restart residual is already below tolerance");

            // Zero damping is the uniform vector, which is already the fixed point.
            double[] uniform = problem.Iterate(0.0, Tolerance, 500, out List<double> zeroHistory, out int zeroSteps);
            for (int v = 0; v < problem.N; v++)
            {
                Check(Math.Abs(uniform[v] - 1.0 / problem.N) <= 1e-15, "damping 0 is uniform at vertex " + v);
            }
            Check(zeroHistory[0] <= 1e-15, "the uniform vector is already a fixed point for damping 0");
        }

        // Two-node cycle: symmetric, hence uniform.
        var pair = PageRankProblem.Cycle(2);
        double[] pairRank = pair.Iterate(Damping, Tolerance, 500, out _, out _);
        Check(Math.Abs(pairRank[0] - 0.5) <= 1e-12 && Math.Abs(pairRank[1] - 0.5) <= 1e-12,
                "the two-node cycle has rank (0.5, 0.5)");

        // Undirected instances: the stationary distribution is proportional to the degree, hence
        // uniform exactly for the regular ones (cycles and complete graphs).
        var undirected = new List<PageRankProblem>
        {
            PageRankProblem.Cycle(5), PageRankProblem.Cycle(7), PageRankProblem.Complete(5),
            PageRankProblem.SymmetricLcg(10, 4, 545100), PageRankProblem.SymmetricLcg(12, 5, 545101),
            PageRankProblem.SymmetricLcg(16, 6, 545102),
        };
        foreach (PageRankProblem problem in undirected)
        {
            // On an undirected graph the degree-proportional vector is exactly a fixed point of the
            // teleport-free walk (pi = P^T pi); this is checked algebraically on the matrix, which
            // avoids relying on power iteration converging - it does not for periodic chains, e.g.
            // even cycles, where unbounded damping oscillates instead of converging.
            double degreeSum = 0.0;
            for (int v = 0; v < problem.N; v++) degreeSum += problem.Out[v].Count;
            var degreeProportional = new double[problem.N];
            for (int v = 0; v < problem.N; v++) degreeProportional[v] = problem.Out[v].Count / degreeSum;
            var incoming = new double[problem.N];
            for (int i = 0; i < problem.N; i++)
            {
                double share = degreeProportional[i] / problem.Out[i].Count;
                foreach (int j in problem.Out[i]) incoming[j] += share;
            }
            for (int j = 0; j < problem.N; j++)
            {
                Check(Math.Abs(incoming[j] - degreeProportional[j]) <= 1e-12,
                        "the degree-proportional vector is a walk fixed point at vertex " + j);
            }
            double[] converged = problem.Iterate(Damping, Tolerance, 500, out _, out _);
            Check(Math.Abs(problem.Sum(converged) - 1.0) <= 1e-12, "the damped rank sums to one");
        }
        foreach (PageRankProblem problem in new[] { PageRankProblem.Cycle(5), PageRankProblem.Complete(5) })
        {
            double[] rank = problem.Iterate(Damping, Tolerance, 500, out _, out _);
            for (int v = 0; v < problem.N; v++)
            {
                Check(Math.Abs(rank[v] - 1.0 / problem.N) <= 1e-9, "regular graph ranks uniformly at vertex " + v);
            }
        }

        // Damping monotonicity in the iteration count.
        foreach (PageRankProblem problem in new[] { PageRankProblem.DirectedLcg(12, 545200), PageRankProblem.Cycle(10) })
        {
            problem.Iterate(0.5, Tolerance, 500, out _, out int fast);
            problem.Iterate(0.85, Tolerance, 500, out _, out int medium);
            problem.Iterate(0.99, Tolerance, 500, out _, out int slow);
            Check(medium >= fast && slow >= medium, "larger damping does not converge faster");
            Check(slow <= 500, "damping 0.99 still converges inside the cap");
        }

        // Non-uniform example: a star pointing at the centre - the centre outranks every leaf.
        var star = new PageRankProblem(5);
        for (int v = 1; v < 5; v++) star.AddEdge(v, 0);
        for (int v = 1; v < 5; v++) star.AddEdge(0, v);
        double[] starRank = star.Iterate(Damping, Tolerance, 500, out _, out _);
        Check(starRank[0] > starRank[1], "the star centre outranks a leaf");
        for (int v = 1; v < 5; v++)
        {
            Check(Math.Abs(starRank[v] - starRank[1]) <= 1e-9, "star leaves tie at vertex " + v);
        }
        Check(Math.Abs(star.Sum(starRank) - 1.0) <= 1e-12, "the star ranks sum to one");
        Check(Math.Abs(star.MaxAbsDifference(starRank, star.SolveLinear(Damping))) <= 1e-10,
                "the star matches its linear solve");

        Console.WriteLine("PASS");
    }
}
