// Task 570 - Implement linear programming simplex pivot demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
//
// A dense tableau simplex for the LP max c.x subject to A x <= b, x >= 0 with b >= 0 (so the slack
// basis is a feasible start and no phase I is needed). The tableau has one row per constraint plus a
// reduced-cost row, and one column per structural variable, per slack variable and for the right-hand
// side; a pivot normalises the leaving row and eliminates the entering column from every other row,
// and the objective value is read off as the negated reduced-cost-row right-hand side. The pivot rule
// is Bland's (enter the smallest-indexed column with a positive reduced cost, leave on the smallest
// index among tied minimum ratios), which provably terminates because it cannot revisit a basis -
// the file demonstrates why this matters by running Beale's classic cycling instance with the Dantzig
// (most positive reduced cost) rule instead and showing that it exceeds the iteration cap, while
// Bland's rule finishes in a handful of pivots. Every solution is validated against two independent
// methods: complete vertex enumeration (solve each of the C(m + n, n) active-set systems by Gaussian
// elimination, keep the feasible ones, take the best objective) and the dual solution recovered from
// the final tableau's slack columns (y_i is the reduced cost of slack i, i.e. c_B^T B^{-1} e_i), for
// which strong duality b.y = c.x, dual feasibility A^T y >= c and complementary slackness are checked.
//
// Documented checks: the textbook LP max 3x + 2y subject to x + y <= 4, x + 3y <= 6 has optimum 12 at
// (4, 0) - the simplex returns exactly that vertex with slacks (0, 2); six deterministic random LPs
// (four constraints, three variables, positive right-hand side) agree with vertex enumeration to
// 1e-7 whenever the optimum is finite, and every returned x is feasible; the dual recovered from the
// tableau is non-negative, dual feasible and satisfies strong duality and complementary slackness;
// Beale's cycling instance reaches 0.05 at (1/25, 0, 1, 0) under Bland's rule in at most twenty
// pivots, while the Dantzig rule on the same instance hits the 1000-pivot cap without terminating;
// two unbounded LPs (max x with x - y <= 0, and max x + y with x - y <= 1) are reported unbounded
// rather than looped over; a degenerate LP with ratio ties still terminates under Bland's rule and
// matches enumeration; and pivot traces are reproducible.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }
}

internal static class Linear
{
    /** Gaussian elimination with partial pivoting; null when the matrix is singular. */
    internal static double[]? Solve(double[][] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var a = new double[n][];
        var x = new double[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n + 1];
            Array.Copy(matrix[i], a[i], n);
            a[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(a[row][col]) > Math.Abs(a[pivot][col])) pivot = row;
            }
            if (Math.Abs(a[pivot][col]) < 1e-12) return null;
            (a[col], a[pivot]) = (a[pivot], a[col]);
            for (int row = 0; row < n; row++)
            {
                if (row == col) continue;
                double factor = a[row][col] / a[col][col];
                if (factor == 0.0) continue;
                for (int k = col; k <= n; k++) a[row][k] -= factor * a[col][k];
            }
        }
        for (int i = 0; i < n; i++) x[i] = a[i][n] / a[i][i];
        return x;
    }
}

internal static class Simplex
{
    internal sealed class Result
    {
        internal double[] X = Array.Empty<double>();
        internal double Objective;
        internal bool Unbounded;
        internal bool HitCap;
        internal int Pivots;
        internal List<(int Entering, int Leaving)> Trace = new List<(int, int)>();
        internal double[] Dual = Array.Empty<double>();
        internal double[][] FinalTableau = Array.Empty<double[]>();
    }

    /**
     * max c.x subject to A x <= b, x >= 0. bland selects Bland's anti-cycling rule, otherwise the
     * Dantzig rule picks the most positive reduced cost.
     */
    internal static Result Solve(double[][] a, double[] b, double[] c, bool bland, int maxPivots)
    {
        int m = b.Length;
        int n = c.Length;
        int cols = n + m + 1; // structural, slack, rhs
        var t = new double[m + 1][];
        for (int i = 0; i <= m; i++) t[i] = new double[cols];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++) t[i][j] = a[i][j];
            t[i][n + i] = 1.0;
            t[i][cols - 1] = b[i];
        }
        for (int j = 0; j < n; j++) t[m][j] = c[j];
        var basis = new int[m];
        for (int i = 0; i < m; i++) basis[i] = n + i;

        var result = new Result();
        while (result.Pivots < maxPivots)
        {
            int entering = -1;
            if (bland)
            {
                for (int j = 0; j < n + m; j++)
                {
                    if (t[m][j] > 1e-11)
                    {
                        entering = j;
                        break;
                    }
                }
            }
            else
            {
                double best = 1e-11;
                for (int j = 0; j < n + m; j++)
                {
                    if (t[m][j] > best)
                    {
                        best = t[m][j];
                        entering = j;
                    }
                }
            }
            if (entering < 0) break; // optimal

            int leaving = -1;
            double bestRatio = double.PositiveInfinity;
            for (int i = 0; i < m; i++)
            {
                if (t[i][entering] <= 1e-11) continue;
                double ratio = t[i][cols - 1] / t[i][entering];
                if (ratio < bestRatio - 1e-12 || (ratio <= bestRatio + 1e-12
                        && (leaving < 0 || basis[i] < basis[leaving])))
                {
                    if (ratio < bestRatio - 1e-12)
                    {
                        bestRatio = ratio;
                        leaving = i;
                    }
                    else if (leaving < 0 || basis[i] < basis[leaving])
                    {
                        leaving = i;
                    }
                }
            }
            if (leaving < 0)
            {
                result.Unbounded = true;
                return Finish(result, t, basis, n, m, cols);
            }

            double pivot = t[leaving][entering];
            for (int k = 0; k < cols; k++) t[leaving][k] /= pivot;
            for (int i = 0; i <= m; i++)
            {
                if (i == leaving) continue;
                double factor = t[i][entering];
                if (factor == 0.0) continue;
                for (int k = 0; k < cols; k++) t[i][k] -= factor * t[leaving][k];
            }
            basis[leaving] = entering;
            result.Trace.Add((entering, leaving));
            result.Pivots++;
        }
        if (result.Pivots >= maxPivots) result.HitCap = true;
        return Finish(result, t, basis, n, m, cols);
    }

    private static Result Finish(Result result, double[][] t, int[] basis, int n, int m, int cols)
    {
        var x = new double[n];
        for (int i = 0; i < m; i++)
        {
            if (basis[i] < n) x[basis[i]] = t[i][cols - 1];
        }
        result.X = x;
        result.Objective = -t[m][cols - 1];
        var dual = new double[m];
        // For slack i, c_j - z_j = -c_B^T B^{-1} e_i = -y_i, so the dual variable is the negated
        // reduced cost of the slack column.
        for (int i = 0; i < m; i++) dual[i] = -t[m][n + i];
        result.Dual = dual;
        var snapshot = new double[m + 1][];
        for (int i = 0; i <= m; i++) snapshot[i] = (double[])t[i].Clone();
        result.FinalTableau = snapshot;
        return result;
    }
}

internal static class Vertices
{
    /** Enumerate all vertices of {A x <= b, x >= 0} and return the best objective (best of all feasible). */
    internal static double BestByEnumeration(double[][] a, double[] b, double[] c)
    {
        int m = b.Length;
        int n = c.Length;
        int constraints = m + n;
        var rows = new double[constraints][];
        var rhs = new double[constraints];
        for (int i = 0; i < m; i++)
        {
            rows[i] = a[i];
            rhs[i] = b[i];
        }
        for (int j = 0; j < n; j++)
        {
            var row = new double[n];
            row[j] = -1.0;
            rows[m + j] = row;
            rhs[m + j] = 0.0;
        }
        double best = double.NegativeInfinity;
        var index = new int[n];
        void Recurse(int start, int depth)
        {
            if (depth == n)
            {
                var matrix = new double[n][];
                var right = new double[n];
                for (int i = 0; i < n; i++)
                {
                    matrix[i] = rows[index[i]];
                    right[i] = rhs[index[i]];
                }
                double[]? x = Linear.Solve(matrix, right);
                if (x == null) return;
                for (int i = 0; i < constraints; i++)
                {
                    double lhs = 0.0;
                    for (int j = 0; j < n; j++) lhs += rows[i][j] * x[j];
                    if (lhs > rhs[i] + 1e-9) return;
                }
                double value = 0.0;
                for (int j = 0; j < n; j++) value += c[j] * x[j];
                if (value > best) best = value;
                return;
            }
            for (int i = start; i + (n - depth) <= constraints; i++)
            {
                index[depth] = i;
                Recurse(i + 1, depth + 1);
            }
        }
        Recurse(0, 0);
        return best;
    }

    /** Dual optimum: min b.y subject to A^T y >= c, y >= 0, enumerated as a vertex of (-A^T y <= -c, y >= 0). */
    internal static double DualByEnumeration(double[][] a, double[] b, double[] c)
    {
        int m = b.Length;
        int n = c.Length;
        var rows = new double[n + m][];
        var rhs = new double[n + m];
        for (int j = 0; j < n; j++)
        {
            var row = new double[m];
            for (int i = 0; i < m; i++) row[i] = -a[i][j];
            rows[j] = row;
            rhs[j] = -c[j];
        }
        for (int i = 0; i < m; i++)
        {
            var row = new double[m];
            row[i] = -1.0;
            rows[n + i] = row;
            rhs[n + i] = 0.0;
        }
        var negB = new double[m];
        for (int i = 0; i < m; i++) negB[i] = -b[i];
        double maxNeg = Vertices.BestByEnumeration(rows, rhs, negB);
        return double.IsNegativeInfinity(maxNeg) ? double.PositiveInfinity : -maxNeg;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double[][] Constraints(params double[][] rows)
    {
        return rows;
    }

    internal static void Main()
    {
        // The textbook LP.
        double[][] a = Constraints(new[] { 1.0, 1.0 }, new[] { 1.0, 3.0 });
        double[] b = { 4.0, 6.0 };
        double[] c = { 3.0, 2.0 };
        Simplex.Result result = Simplex.Solve(a, b, c, true, 1000);
        Check(!result.Unbounded && !result.HitCap, "the textbook LP is solved to optimality");
        Check(Math.Abs(result.Objective - 12.0) <= 1e-9, "the textbook optimum is 12 (" + result.Objective + ")");
        Check(Math.Abs(result.X[0] - 4.0) <= 1e-9 && Math.Abs(result.X[1]) <= 1e-9, "the optimum is at (4, 0)");
        Check(Math.Abs(result.Dual[0] - 3.0) <= 1e-9 && Math.Abs(result.Dual[1]) <= 1e-9,
                "the tableau dual is (3, 0), the true optimum of the dual");
        Check(Math.Abs(result.Dual[0] * 4.0 + result.Dual[1] * 6.0 - 12.0) <= 1e-9, "strong duality holds here");
        Check(result.Pivots >= 1 && result.Pivots <= 10, "the textbook LP takes a few pivots (" + result.Pivots + ")");

        // Random LPs: simplex versus vertex enumeration, feasibility and duality.
        ulong state = 570123UL;
        int bounded = 0;
        int unboundedRandom = 0;
        for (int trial = 0; trial < 6; trial++)
        {
            int m = 4;
            int n = 3;
            var rows = new double[m][];
            var rhs = new double[m];
            var objective = new double[n];
            for (int i = 0; i < m; i++)
            {
                rows[i] = new double[n];
                for (int j = 0; j < n; j++) rows[i][j] = 2.0 * Rng.NextUnit(ref state) - 1.0;
                rhs[i] = 1.0 + 4.0 * Rng.NextUnit(ref state);
            }
            for (int j = 0; j < n; j++) objective[j] = -1.0 + 2.0 * Rng.NextUnit(ref state);
            Simplex.Result r = Simplex.Solve(rows, rhs, objective, true, 1000);
            Check(!r.HitCap, "Bland's rule terminates on random LP " + trial);
            if (r.Unbounded)
            {
                unboundedRandom++;
                continue;
            }
            bounded++;
            double enumerated = Vertices.BestByEnumeration(rows, rhs, objective);
            Check(Math.Abs(r.Objective - enumerated) <= 1e-7, "simplex matches vertex enumeration on LP " + trial
                    + " (" + r.Objective + " vs " + enumerated + ")");
            for (int i = 0; i < m; i++)
            {
                double lhs = 0.0;
                for (int j = 0; j < n; j++) lhs += rows[i][j] * r.X[j];
                Check(lhs <= rhs[i] + 1e-9, "the returned point is feasible for constraint " + i);
            }
            for (int j = 0; j < n; j++) Check(r.X[j] >= -1e-9, "the returned point is non-negative");
            double dualObjective = 0.0;
            for (int i = 0; i < m; i++) dualObjective += rhs[i] * r.Dual[i];
            Check(Math.Abs(dualObjective - r.Objective) <= 1e-7, "strong duality on LP " + trial);
            Check(Math.Abs(Vertices.DualByEnumeration(rows, rhs, objective) - r.Objective) <= 1e-7,
                    "the enumerated dual optimum equals the primal optimum on LP " + trial);
            for (int i = 0; i < m; i++) Check(r.Dual[i] >= -1e-9, "the tableau dual is non-negative");
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < m; i++) sum += rows[i][j] * r.Dual[i];
                Check(sum >= objective[j] - 1e-7, "dual feasibility at column " + j);
            }
            for (int i = 0; i < m; i++)
            {
                double slack = rhs[i];
                for (int j = 0; j < n; j++) slack -= rows[i][j] * r.X[j];
                if (slack > 1e-7) Check(Math.Abs(r.Dual[i]) <= 1e-7, "complementary slackness on row " + i);
            }
        }
        Check(bounded >= 4, "most random LPs were bounded (" + bounded + " of 6, " + unboundedRandom + " unbounded)");

        // Beale's cycling instance: Bland terminates, Dantzig cycles.
        double[][] bealeA = Constraints(
                new[] { 0.25, -60.0, -0.04, 9.0 },
                new[] { 0.5, -90.0, -0.02, 3.0 },
                new[] { 0.0, 0.0, 1.0, 0.0 });
        double[] bealeB = { 0.0, 0.0, 1.0 };
        double[] bealeC = { 0.75, -150.0, 0.02, -6.0 };
        Simplex.Result bland = Simplex.Solve(bealeA, bealeB, bealeC, true, 1000);
        Check(!bland.Unbounded && !bland.HitCap, "Bland's rule terminates on Beale's instance");
        Check(Math.Abs(bland.Objective - 0.05) <= 1e-7, "Beale's optimum is 0.05 (" + bland.Objective + ")");
        Check(Math.Abs(bland.X[0] - 0.04) <= 1e-6 && Math.Abs(bland.X[2] - 1.0) <= 1e-6,
                "Beale's optimum is at (1/25, 0, 1, 0)");
        Check(bland.Pivots <= 20, "Bland's rule needs at most twenty pivots (" + bland.Pivots + ")");
        Simplex.Result dantzig = Simplex.Solve(bealeA, bealeB, bealeC, false, 1000);
        Check(dantzig.HitCap, "the Dantzig rule cycles on Beale's instance and hits the 1000-pivot cap");
        Check(dantzig.Pivots == 1000, "the cycling run used the whole pivot budget");

        // Unbounded LPs.
        Simplex.Result unboundedOne = Simplex.Solve(Constraints(new[] { 1.0, -1.0 }), new[] { 0.0 },
                new[] { 1.0, 0.0 }, true, 1000);
        Check(unboundedOne.Unbounded, "max x subject to x - y <= 0 is unbounded");
        Simplex.Result unboundedTwo = Simplex.Solve(Constraints(new[] { 1.0, -1.0 }), new[] { 1.0 },
                new[] { 1.0, 1.0 }, true, 1000);
        Check(unboundedTwo.Unbounded, "max x + y subject to x - y <= 1 is unbounded");

        // A degenerate LP with ratio ties still terminates and is optimal.
        double[][] degenerateA = Constraints(
                new[] { 1.0, 1.0 },
                new[] { 1.0, 0.0 },
                new[] { 0.0, 1.0 });
        double[] degenerateB = { 2.0, 2.0, 2.0 };
        double[] degenerateC = { 1.0, 1.0 };
        Simplex.Result degenerate = Simplex.Solve(degenerateA, degenerateB, degenerateC, true, 1000);
        Check(!degenerate.Unbounded && !degenerate.HitCap && degenerate.Pivots > 0,
                "the degenerate LP terminates with at least one pivot");
        Check(Math.Abs(degenerate.Objective - 2.0) <= 1e-9, "the degenerate LP has optimum 2");
        Check(Math.Abs(degenerate.Objective - Vertices.BestByEnumeration(degenerateA, degenerateB, degenerateC)) <= 1e-9,
                "the degenerate LP matches enumeration");

        // Reproducibility of the pivot trace.
        Simplex.Result repeated = Simplex.Solve(a, b, c, true, 1000);
        Check(repeated.Pivots == result.Pivots, "the pivot count is reproducible");
        bool sameTrace = repeated.Trace.Count == result.Trace.Count;
        for (int i = 0; sameTrace && i < result.Trace.Count; i++)
        {
            sameTrace = repeated.Trace[i].Entering == result.Trace[i].Entering
                    && repeated.Trace[i].Leaving == result.Trace[i].Leaving;
        }
        Check(sameTrace, "the pivot trace is reproducible");
        Check(result.FinalTableau.Length == 3 && result.FinalTableau[0].Length == a[0].Length + b.Length + 1,
                "the final tableau has the expected shape");

        Console.WriteLine("PASS");
    }
}
