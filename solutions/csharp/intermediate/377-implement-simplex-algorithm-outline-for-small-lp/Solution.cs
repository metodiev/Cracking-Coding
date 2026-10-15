// Task 377 - Simplex algorithm for small LPs (outline). (Tier: Intermediate, CTCI Ch 6)
using System;
using System.Collections.Generic;

/// <summary>
/// Standard form: maximise c.x subject to A x &lt;= b, x &gt;= 0.  The tableau simplex
/// starts from the slack basis (x = 0) when b &gt;= 0.  Rows with b_i &lt; 0 are negated
/// (their slack coefficient becomes -1) and get an artificial variable; a phase-1
/// objective (-1 on the artificial columns, i.e. minimising their sum) drives them out,
/// and a positive phase-1 optimum means the LP is infeasible.  Phase 2 then runs the real
/// objective with the artificial columns forbidden from entering.
///
/// Bland's rule is used for both iterations: the entering variable is the smallest index
/// with a positive reduced cost and the leaving variable breaks ratio ties by the
/// smallest basic-variable index.  This guarantees termination (no cycling); an iteration
/// cap is kept as a failure sentinel and is never hit by the test problems.
///
/// Documented sentinels (the API returns a status string; x is null and value NaN unless
/// "optimal"):
///   * "optimal"    -&gt; x is the optimal vertex,
///   * "unbounded"  -&gt; some entering column has no positive tableau coefficient,
///   * "infeasible" -&gt; a positive phase-1 optimum (the constraints contradict).
///
/// Re-derivation note: the task brief suggests "max 3x + 2y s.t. x+y&lt;=4, x+3y&lt;=6 -&gt;
/// x=3, y=1, value 11".  That point is the intersection of the two constraint lines, but
/// the vertex (4, 0) satisfies both constraints (4+0=4, 4+0&lt;=6) and gives
/// 3*4 + 2*0 = 12 &gt; 11, so the true optimum is 12 at (4, 0) - the code asserts 12.
///
/// The objective values are small exact integers here; comparisons use the documented
/// tolerance 1e-9 because the tableau arithmetic is in doubles.  A Cramer-rule vertex
/// enumeration is used as an independent oracle for the LCG-generated bounded LPs.
/// </summary>
public static class Solution
{
    private const double EPS = 1e-9;
    private const int MaxIterations = 10000;

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

    /// <summary>status in {"optimal", "unbounded", "infeasible"}; X null / Value NaN unless optimal.</summary>
    private sealed class LpResult
    {
        public string Status { get; }
        public double[]? X { get; }
        public double Value { get; }

        public LpResult(string status, double[]? x, double value)
        {
            Status = status;
            X = x;
            Value = value;
        }
    }

    /// <summary>Primal simplex core (Bland's rule). Mutates the tableau; returns (status, value).</summary>
    private static (string status, double value) RunSimplex(
        double[][] tableau, double[] rhs, int[] basis, double[] cost, bool[] canEnter)
    {
        int rows = tableau.Length;
        int columns = cost.Length;

        var reduced = (double[])cost.Clone();
        double value = 0.0;
        for (int i = 0; i < rows; i++)
        {
            double basicCost = cost[basis[i]];
            if (basicCost != 0.0)
            {
                value += basicCost * rhs[i];
                for (int j = 0; j < columns; j++) reduced[j] -= basicCost * tableau[i][j];
            }
        }

        int iterations = 0;
        while (true)
        {
            iterations++;
            if (iterations > MaxIterations) throw new Exception("simplex iteration cap exceeded");
            int entering = -1;
            for (int j = 0; j < columns; j++)                 // Bland: smallest improving index
            {
                if (canEnter[j] && reduced[j] > EPS)
                {
                    entering = j;
                    break;
                }
            }
            if (entering == -1) return ("optimal", value);

            int leaving = -1;
            double bestRatio = 0.0;
            for (int i = 0; i < rows; i++)
            {
                if (tableau[i][entering] > EPS)
                {
                    double ratio = rhs[i] / tableau[i][entering];
                    if (leaving == -1 || ratio < bestRatio - 1e-12)
                    {
                        leaving = i;
                        bestRatio = ratio;
                    }
                    else if (Math.Abs(ratio - bestRatio) <= 1e-12 && basis[i] < basis[leaving])
                    {
                        leaving = i;                          // Bland ratio tie-break
                        bestRatio = ratio;
                    }
                }
            }
            if (leaving == -1) return ("unbounded", double.NaN);

            double pivot = tableau[leaving][entering];
            for (int j = 0; j < columns; j++) tableau[leaving][j] /= pivot;
            rhs[leaving] /= pivot;
            for (int i = 0; i < rows; i++)
            {
                if (i != leaving && tableau[i][entering] != 0.0)
                {
                    double factor = tableau[i][entering];
                    for (int j = 0; j < columns; j++) tableau[i][j] -= factor * tableau[leaving][j];
                    rhs[i] -= factor * rhs[leaving];
                }
            }
            double pivotFactor = reduced[entering];
            for (int j = 0; j < columns; j++) reduced[j] -= pivotFactor * tableau[leaving][j];
            value += pivotFactor * rhs[leaving];
            basis[leaving] = entering;
        }
    }

    /// <summary>max c.x s.t. A x &lt;= b, x &gt;= 0.  Returns an LpResult (see sentinel doc).</summary>
    private static LpResult Simplex(int[] objective, int[][] constraints, int[] bounds)
    {
        int n = objective.Length;
        int m = constraints.Length;
        int artificialCount = 0;
        for (int i = 0; i < m; i++) if (bounds[i] < 0) artificialCount++;
        int columns = n + m + artificialCount;

        var tableau = new double[m][];
        var rhs = new double[m];
        var basis = new int[m];
        var artificialOfRow = new int[m];
        for (int i = 0; i < m; i++) artificialOfRow[i] = -1;
        int nextArtificial = n + m;
        for (int i = 0; i < m; i++)
        {
            tableau[i] = new double[columns];
            if (bounds[i] >= 0)
            {
                for (int j = 0; j < n; j++) tableau[i][j] = constraints[i][j];
                tableau[i][n + i] = 1.0;                      // slack basis
                basis[i] = n + i;
                rhs[i] = bounds[i];
            }
            else
            {
                for (int j = 0; j < n; j++) tableau[i][j] = -constraints[i][j];
                tableau[i][n + i] = -1.0;
                tableau[i][nextArtificial] = 1.0;             // artificial basis
                artificialOfRow[i] = nextArtificial;
                basis[i] = nextArtificial;
                nextArtificial++;
                rhs[i] = -bounds[i];
            }
        }

        if (artificialCount > 0)
        {
            var phaseOneCost = new double[columns];
            for (int i = 0; i < m; i++)
            {
                if (artificialOfRow[i] != -1) phaseOneCost[artificialOfRow[i]] = -1.0;
            }
            var allColumns = new bool[columns];
            for (int j = 0; j < columns; j++) allColumns[j] = true;
            (string statusOne, double valueOne) = RunSimplex(tableau, rhs, basis, phaseOneCost, allColumns);
            if (statusOne != "optimal") throw new Exception("phase 1 must be bounded");
            if (valueOne < -EPS) return new LpResult("infeasible", null, double.NaN);
        }

        var phaseTwoCost = new double[columns];
        for (int j = 0; j < n; j++) phaseTwoCost[j] = objective[j];
        var canEnter = new bool[columns];
        for (int j = 0; j < n + m; j++) canEnter[j] = true;   // artificials stay out
        (string status, double value) = RunSimplex(tableau, rhs, basis, phaseTwoCost, canEnter);
        if (status != "optimal") return new LpResult(status, null, double.NaN);

        var x = new double[n];
        for (int i = 0; i < m; i++)
        {
            if (basis[i] < n) x[basis[i]] = rhs[i];
        }
        return new LpResult("optimal", x, value);
    }

    /// <summary>Exact-ish recursive determinant for tiny matrices (n &lt;= 3 in this task).</summary>
    private static double Determinant(double[][] matrix)
    {
        int size = matrix.Length;
        if (size == 1) return matrix[0][0];
        double total = 0.0;
        for (int j = 0; j < size; j++)
        {
            var minor = new double[size - 1][];
            for (int r = 1; r < size; r++)
            {
                minor[r - 1] = new double[size - 1];
                int at = 0;
                for (int c = 0; c < size; c++)
                {
                    if (c != j) minor[r - 1][at++] = matrix[r][c];
                }
            }
            total += (j % 2 == 0 ? 1 : -1) * matrix[0][j] * Determinant(minor);
        }
        return total;
    }

    private static void CollectCombinations(int n, int k, int start, int[] combo, int at, List<int[]> output)
    {
        if (at == k)
        {
            output.Add((int[])combo.Clone());
            return;
        }
        for (int i = start; i <= n - (k - at); i++)
        {
            combo[at] = i;
            CollectCombinations(n, k, i + 1, combo, at + 1, output);
        }
    }

    /// <summary>Cramer-rule vertex enumeration for bounded feasible LPs.</summary>
    private static double? BruteForceLp(int[] objective, int[][] constraints, int[] bounds)
    {
        int n = objective.Length;
        int m = constraints.Length;
        var hyperplanes = new List<double[]>();
        var hyperRhs = new List<double>();
        for (int i = 0; i < m; i++)
        {
            var row = new double[n];
            for (int j = 0; j < n; j++) row[j] = constraints[i][j];
            hyperplanes.Add(row);
            hyperRhs.Add(bounds[i]);
        }
        for (int j = 0; j < n; j++)
        {
            var unit = new double[n];
            unit[j] = 1.0;
            hyperplanes.Add(unit);
            hyperRhs.Add(0.0);
        }

        var combos = new List<int[]>();
        CollectCombinations(hyperplanes.Count, n, 0, new int[n], 0, combos);

        double tolerance = 1e-7;
        double? best = null;
        foreach (int[] combo in combos)
        {
            var matrix = new double[n][];
            var rhs = new double[n];
            for (int i = 0; i < n; i++)
            {
                matrix[i] = hyperplanes[combo[i]];
                rhs[i] = hyperRhs[combo[i]];
            }
            double det = Determinant(matrix);
            if (Math.Abs(det) < 1e-12) continue;
            var x = new double[n];
            for (int col = 0; col < n; col++)
            {
                var replaced = new double[n][];
                for (int r = 0; r < n; r++)
                {
                    replaced[r] = new double[n];
                    for (int c = 0; c < n; c++) replaced[r][c] = (c == col) ? rhs[r] : matrix[r][c];
                }
                x[col] = Determinant(replaced) / det;
            }
            bool feasible = true;
            for (int j = 0; j < n && feasible; j++) if (x[j] < -tolerance) feasible = false;
            for (int i = 0; i < m && feasible; i++)
            {
                double lhs = 0.0;
                for (int j = 0; j < n; j++) lhs += constraints[i][j] * x[j];
                if (lhs > bounds[i] + tolerance) feasible = false;
            }
            if (!feasible) continue;
            double value = 0.0;
            for (int j = 0; j < n; j++) value += objective[j] * x[j];
            if (best == null || value > best) best = value;
        }
        return best;
    }

    private static (int[] objective, int[][] constraints, int[] bounds) LcgBoundedLp(Lcg lcg)
    {
        int n = 1 + lcg.Next(2);
        int m = 2 + lcg.Next(2);
        var constraints = new int[m][];
        for (int i = 0; i < m; i++)
        {
            constraints[i] = new int[n];
            for (int j = 0; j < n; j++) constraints[i][j] = lcg.Next(9);
        }
        for (int j = 0; j < n; j++)                           // every column positive
        {
            bool positive = false;
            for (int i = 0; i < m; i++) if (constraints[i][j] > 0) positive = true;
            if (!positive) constraints[lcg.Next(m)][j] = 1 + lcg.Next(3);
        }
        var bounds = new int[m];
        for (int i = 0; i < m; i++) bounds[i] = 1 + lcg.Next(20);
        var objective = new int[n];
        for (int j = 0; j < n; j++) objective[j] = lcg.Next(10);
        return (objective, constraints, bounds);
    }

    public static void Main()
    {
        // the brief's LP, re-derived: optimum (4, 0) with value 12, NOT (3, 1)/11
        LpResult result = Simplex(new[] { 3, 2 }, new[] { new[] { 1, 1 }, new[] { 1, 3 } }, new[] { 4, 6 });
        Check(result.Status == "optimal", "brief LP optimal");
        Check(Math.Abs(result.Value - 12.0) < EPS, "brief LP value 12, got " + result.Value);
        Check(Math.Abs(result.X![0] - 4.0) < EPS && Math.Abs(result.X[1]) < EPS, "brief LP vertex (4, 0)");
        Check(BruteForceLp(new[] { 3, 2 }, new[] { new[] { 1, 1 }, new[] { 1, 3 } }, new[] { 4, 6 }) == 12.0,
            "brief LP oracle 12");

        // hand-solved: vertices (0,0)=0, (2.5,0)=5, (0,4)=12, (1,3)=11 -> 12 at (0,4)
        result = Simplex(new[] { 2, 3 }, new[] { new[] { 1, 1 }, new[] { 2, 1 } }, new[] { 4, 5 });
        Check(result.Status == "optimal" && Math.Abs(result.Value - 12.0) < EPS, "second LP 12");
        Check(Math.Abs(result.X![0]) < EPS && Math.Abs(result.X[1] - 4.0) < EPS, "second LP vertex");
        Check(BruteForceLp(new[] { 2, 3 }, new[] { new[] { 1, 1 }, new[] { 2, 1 } }, new[] { 4, 5 }) == 12.0,
            "second LP oracle");

        // 3 variables: optima (4,1,0), (2,2,1), (0,3,2) all give 14; Bland picks one
        result = Simplex(new[] { 3, 2, 4 }, new[] { new[] { 1, 1, 1 }, new[] { 2, 1, 3 } }, new[] { 5, 9 });
        Check(result.Status == "optimal" && Math.Abs(result.Value - 14.0) < EPS, "3-var LP 14");
        Check(Math.Abs(result.X![0] + result.X[1] + result.X[2] - 5.0) < EPS, "3-var sum is 5");
        Check(result.X[0] > -EPS && result.X[1] > -EPS && result.X[2] > -EPS, "3-var non-negative");
        Check(2 * result.X[0] + result.X[1] + 3 * result.X[2] <= 9 + 1e-7, "3-var second constraint");
        Check(BruteForceLp(new[] { 3, 2, 4 }, new[] { new[] { 1, 1, 1 }, new[] { 2, 1, 3 } }, new[] { 5, 9 }) == 14.0,
            "3-var LP oracle");

        // tie along x + y <= 1 (degenerate objective direction)
        result = Simplex(new[] { 1, 1 }, new[] { new[] { 1, 1 } }, new[] { 1 });
        Check(result.Status == "optimal" && Math.Abs(result.Value - 1.0) < EPS, "tie LP");
        Check(Math.Abs(result.X![0] + result.X[1] - 1.0) < EPS, "tie LP sum");

        // duplicate constraints (degenerate basis)
        result = Simplex(new[] { 1 }, new[] { new[] { 1 }, new[] { 1 } }, new[] { 3, 3 });
        Check(result.Status == "optimal" && Math.Abs(result.Value - 3.0) < EPS, "duplicate LP");

        // --- sentinels ---------------------------------------------------------
        result = Simplex(new[] { 1 }, Array.Empty<int[]>(), Array.Empty<int>());
        Check(result.Status == "unbounded" && result.X == null, "no constraints -> unbounded");
        result = Simplex(new[] { 1 }, new[] { new[] { -1 } }, new[] { -1 });        // x >= 1
        Check(result.Status == "unbounded" && result.X == null, "x >= 1 -> unbounded");
        result = Simplex(new[] { 1 }, new[] { new[] { 1 }, new[] { -1 } }, new[] { 1, -3 });   // x<=1, x>=3
        Check(result.Status == "infeasible" && result.X == null, "x <= 1 and x >= 3 infeasible");

        // feasible LP with a negative bound (phase 1 needed, artificial leaves the basis)
        result = Simplex(new[] { 1 }, new[] { new[] { 1 }, new[] { -1 } }, new[] { 5, -2 });   // 2 <= x <= 5
        Check(result.Status == "optimal" && Math.Abs(result.X![0] - 5.0) < EPS, "negative bound max x");
        result = Simplex(new[] { -1 }, new[] { new[] { 1 }, new[] { -1 } }, new[] { 5, -2 });
        Check(result.Status == "optimal" && Math.Abs(result.Value + 2.0) < EPS, "negative bound min x");

        // --- LCG cross-checks against vertex enumeration -----------------------
        var lcg = new Lcg(37713);
        for (int trial = 0; trial < 40; trial++)
        {
            (int[] objective, int[][] constraints, int[] bounds) = LcgBoundedLp(lcg);
            result = Simplex(objective, constraints, bounds);
            Check(result.Status == "optimal", "random LP optimal " + trial);
            double? oracle = BruteForceLp(objective, constraints, bounds);
            Check(oracle != null, "random LP has an oracle vertex " + trial);
            Check(Math.Abs(result.Value - oracle!.Value) < 1e-6, "random LP value " + trial);
            for (int i = 0; i < constraints.Length; i++)
            {
                double lhs = 0.0;
                for (int j = 0; j < objective.Length; j++) lhs += constraints[i][j] * result.X![j];
                Check(lhs <= bounds[i] + 1e-7, "random LP feasibility " + trial);
            }
            foreach (double component in result.X!)
            {
                Check(component > -1e-7, "random LP non-negative " + trial);
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
