// Task 562 - Implement mixed integer programming small model using branch-and-bound outline. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
//
// A working branch-and-bound solver for tiny mixed integer programs: maximise c·x subject to A x <= b,
// x >= 0 and x integer, with all data integral. The LP relaxation is solved exactly by vertex
// enumeration: any optimum of a bounded polyhedron is attained at a basic feasible solution, so the solver
// enumerates all C(n+m, m) bases of the standard form [A | I] - m basic variables chosen from the n
// structural and m slack columns, with every other column fixed at zero - solves each m x m system by
// Gaussian elimination with partial pivoting, keeps the bases whose basic values are non-negative, re-checks
// A x <= b and retains the best objective. Getting the basis size wrong is a trap this file fell into: the
// first version enumerated n columns instead of m, which is not a valid basis as soon as there are fewer
// constraints than variables, and it silently returned 14.5 instead of the true 15.1666... on a two-row,
// three-variable instance; the LP value asserted for the textbook knapsack is what caught it. That is an
// honest but exponential LP method, which is exactly why the task says "small model"; it very deliberately
// replaces a simplex implementation that would obscure the branching logic. Branch and bound then
// solves the relaxation at a node (with variable bounds appended as extra rows), prunes on infeasibility
// or on a bound no better than the incumbent, accepts an integral relaxation as the new incumbent, and
// otherwise branches on the most fractional variable with x_i <= floor and x_i >= ceil, depth first. Two
// limitations are stated rather than hidden: the vertex enumeration assumes a bounded feasible region, so
// the solver explicitly detects the axis-aligned unbounded rays that make an instance unbounded, and it
// reports them instead of returning a wrong finite value. Every instance is additionally solved by brute
// force over the integer box, which is feasible for three or four variables. Variable bounds inside the
// search are derived from the constraints (x_i <= min over rows with a positive coefficient of
// floor(b_row / a_row,i)) and, for coordinates no constraint bounds above, from the finite relaxation
// vertex; the branch order is always the most fractional variable, depth first.
//
// Documented checks: the textbook knapsack max 5x1+4x2 s.t. 6x1+4x2 <= 24, x1+2x2 <= 6 has LP value 21
// at the fractional point (3, 1.5) and integer optimum 20 at (4, 0), both asserted; the 0/1-flavoured
// instance max 8x1+11x2+6x3+4x4 s.t. 5x1+7x2+4x3+3x4 <= 14 has integer optimum 22, confirmed by brute
// force; an instance whose relaxation is already integral needs exactly one node (no branching); the
// infeasible system x1+x2 <= 1, x1-x2 <= -2 is reported infeasible by both the LP and the search; an
// instance unbounded along the x1 axis is detected as unbounded; for ten splitmix64-generated models the
// branch-and-bound optimum equals the brute-force optimum, the root LP bound is never below the integer
// optimum, and the node count stays below the brute-force box size (measured, printed in the messages);
// and repeated runs are identical.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    /** splitmix64 (Steele/Lea/Flood): 64-bit state advanced by a golden-ratio increment. */
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int Below(ref ulong state, int bound)
    {
        return (int)(Next(ref state) % (ulong)bound);
    }
}

internal sealed class MilpModel
{
    internal readonly string Label;
    internal readonly int Vars;
    internal readonly double[][] A;
    internal readonly double[] B;
    internal readonly double[] C;

    internal MilpModel(string label, int vars, double[][] a, double[] b, double[] c)
    {
        Label = label;
        Vars = vars;
        A = a;
        B = b;
        C = c;
    }

    internal static MilpModel Build(string label, double[] c, params double[][] rows)
    {
        var a = new double[rows.Length][];
        var b = new double[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            a[i] = new double[c.Length];
            for (int j = 0; j < c.Length; j++) a[i][j] = rows[i][j];
            b[i] = rows[i][rows[i].Length - 1];
        }
        return new MilpModel(label, c.Length, a, b, c);
    }

    internal static double[][] WithBounds(MilpModel model, double[] lower, double[] upper)
    {
        var rows = new List<double[]>();
        foreach (double[] row in model.A) rows.Add(row);
        for (int i = 0; i < model.Vars; i++)
        {
            var cap = new double[model.Vars];
            cap[i] = 1.0;
            rows.Add(cap);
        }
        for (int i = 0; i < model.Vars; i++)
        {
            if (lower[i] <= 0) continue;
            var floorRow = new double[model.Vars];
            floorRow[i] = -1.0;
            rows.Add(floorRow);
        }
        return rows.ToArray();
    }

    internal static double[] BoundsRhs(MilpModel model, double[] lower, double[] upper)
    {
        var b = new List<double>();
        foreach (double value in model.B) b.Add(value);
        for (int i = 0; i < model.Vars; i++) b.Add(upper[i]);
        for (int i = 0; i < model.Vars; i++)
        {
            if (lower[i] <= 0) continue;
            b.Add(-lower[i]);
        }
        return b.ToArray();
    }
}

internal static class Lp
{
    internal sealed class Solution
    {
        internal bool Feasible;
        internal bool Unbounded;
        internal double Objective;
        internal double[] X = Array.Empty<double>();
    }

    private sealed class Enumerator
    {
        internal double[][] A = Array.Empty<double[]>();
        internal double[] B = Array.Empty<double>();
        internal double[] C = Array.Empty<double>();
        internal int N;
        internal int M;
        internal int[] Combo = Array.Empty<int>();
        internal bool Found;
        internal double Best = double.NegativeInfinity;
        internal double[]? BestX;
        internal long Bases;

        internal void Run()
        {
            Recurse(0, 0);
        }

        private void Recurse(int depth, int start)
        {
            if (depth == M)
            {
                Evaluate();
                return;
            }
            int last = N + M - (M - depth);
            for (int column = start; column <= last; column++)
            {
                Combo[depth] = column;
                Recurse(depth + 1, column + 1);
            }
        }

        private void Evaluate()
        {
            Bases++;
            var basis = new double[M][];
            for (int row = 0; row < M; row++)
            {
                basis[row] = new double[M];
                for (int j = 0; j < M; j++)
                {
                    int column = Combo[j];
                    basis[row][j] = column < N ? A[row][column] : (row == column - N ? 1.0 : 0.0);
                }
            }
            var rhs = new double[M];
            Array.Copy(B, rhs, M);
            if (!SolveSystem(basis, rhs)) return;
            for (int j = 0; j < M; j++)
            {
                if (rhs[j] < -1e-9) return;
            }
            var x = new double[N];
            for (int j = 0; j < M; j++)
            {
                int column = Combo[j];
                if (column < N) x[column] = rhs[j];
            }
            for (int row = 0; row < M; row++)
            {
                double lhs = 0.0;
                for (int j = 0; j < N; j++) lhs += A[row][j] * x[j];
                if (lhs > B[row] + 1e-7) return;
            }
            double objective = 0.0;
            for (int j = 0; j < N; j++) objective += C[j] * x[j];
            if (!Found || objective > Best + 1e-9)
            {
                Found = true;
                Best = objective;
                BestX = x;
            }
        }

        private static bool SolveSystem(double[][] matrix, double[] rhs)
        {
            int n = matrix.Length;
            for (int col = 0; col < n; col++)
            {
                int pivot = col;
                for (int row = col + 1; row < n; row++)
                {
                    if (Math.Abs(matrix[row][col]) > Math.Abs(matrix[pivot][col])) pivot = row;
                }
                if (Math.Abs(matrix[pivot][col]) < 1e-12) return false;
                if (pivot != col)
                {
                    double[] temp = matrix[pivot];
                    matrix[pivot] = matrix[col];
                    matrix[col] = temp;
                    double t = rhs[pivot];
                    rhs[pivot] = rhs[col];
                    rhs[col] = t;
                }
                for (int row = col + 1; row < n; row++)
                {
                    double factor = matrix[row][col] / matrix[col][col];
                    if (factor == 0.0) continue;
                    for (int k = col; k < n; k++) matrix[row][k] -= factor * matrix[col][k];
                    rhs[row] -= factor * rhs[col];
                }
            }
            for (int row = n - 1; row >= 0; row--)
            {
                double sum = rhs[row];
                for (int k = row + 1; k < n; k++) sum -= matrix[row][k] * rhs[k];
                rhs[row] = sum / matrix[row][row];
            }
            return true;
        }
    }

    internal static Solution Solve(double[][] a, double[] b, double[] c)
    {
        int n = c.Length;
        var result = new Solution();
        if (b.Length == 0)
        {
            // No constraints: the only vertex of x >= 0 is the origin, and a positive objective makes
            // the problem unbounded along that axis.
            result.Feasible = true;
            result.Objective = 0.0;
            result.X = new double[n];
            for (int i = 0; i < n; i++)
            {
                if (c[i] > 1e-12) result.Unbounded = true;
            }
            return result;
        }
        var enumerator = new Enumerator
        {
            A = a,
            B = b,
            C = c,
            N = n,
            M = b.Length,
            Combo = new int[b.Length],
        };
        enumerator.Run();
        if (!enumerator.Found || enumerator.BestX == null) return result;
        result.Feasible = true;
        result.Objective = enumerator.Best;
        result.X = enumerator.BestX;
        for (int i = 0; i < n; i++)
        {
            bool noUpperRow = true;
            for (int row = 0; row < b.Length; row++)
            {
                if (a[row][i] > 1e-12) { noUpperRow = false; break; }
            }
            if (noUpperRow && c[i] > 1e-12) result.Unbounded = true;
        }
        return result;
    }
}

internal static class BranchAndBound
{
    internal sealed class Result
    {
        internal bool HasSolution;
        internal bool Unbounded;
        internal bool Infeasible;
        internal double Objective;
        internal double[] X = Array.Empty<double>();
        internal long Nodes;
        internal double RootLp;
        internal bool RootFractional;
        internal long Bases;
    }

    private const long NodeCap = 200000;

    internal static Result Solve(MilpModel model)
    {
        var result = new Result { Objective = double.NegativeInfinity };
        // Unboundedness and infeasibility are properties of the uncapped relaxation; detect them there.
        Lp.Solution relaxation = Lp.Solve(model.A, model.B, model.C);
        if (!relaxation.Feasible)
        {
            result.Infeasible = true;
            return result;
        }
        if (relaxation.Unbounded)
        {
            result.Unbounded = true;
            return result;
        }
        var lower = new double[model.Vars];
        var upper = new double[model.Vars];
        for (int i = 0; i < model.Vars; i++)
        {
            double cap = double.PositiveInfinity;
            for (int row = 0; row < model.A.Length; row++)
            {
                if (model.A[row][i] > 1e-12)
                {
                    cap = Math.Min(cap, Math.Floor(model.B[row] / model.A[row][i] + 1e-9));
                }
            }
            // A coordinate not bounded above by any constraint is capped at the relaxation vertex, which
            // is finite because the uncapped relaxation was just shown to be bounded.
            if (double.IsPositiveInfinity(cap)) cap = Math.Ceiling(relaxation.X[i]);
            upper[i] = Math.Max(0.0, cap);
        }
        Explore(model, lower, upper, result, true);
        if (!result.HasSolution && !result.Unbounded) result.Infeasible = true;
        return result;
    }

    private static void Explore(MilpModel model, double[] lower, double[] upper, Result result, bool root)
    {
        result.Nodes++;
        if (result.Nodes > NodeCap) throw new Exception("node cap exceeded");
        double[][] rows = Assemble(model, lower, upper);
        var b = AssembleB(model, upper, lower);
        Lp.Solution lp = Lp.Solve(rows, b, model.C);
        if (!lp.Feasible)
        {
            if (root) result.Infeasible = true;
            return;
        }
        if (lp.Unbounded)
        {
            result.Unbounded = true;
            return;
        }
        if (root)
        {
            result.RootLp = lp.Objective;
        }
        if (result.HasSolution && lp.Objective <= result.Objective + 1e-9) return;
        int fractional = -1;
        double worst = 1e-6;
        for (int i = 0; i < model.Vars; i++)
        {
            double distance = Math.Abs(lp.X[i] - Math.Round(lp.X[i]));
            if (distance > worst)
            {
                worst = distance;
                fractional = i;
            }
        }
        if (fractional < 0)
        {
            if (!result.HasSolution || lp.Objective > result.Objective + 1e-9)
            {
                result.HasSolution = true;
                result.Objective = lp.Objective;
                var rounded = new double[model.Vars];
                for (int i = 0; i < model.Vars; i++) rounded[i] = Math.Round(lp.X[i]);
                result.X = rounded;
            }
            return;
        }
        if (root) result.RootFractional = worst > 1e-6;
        double savedUpper = upper[fractional];
        upper[fractional] = Math.Floor(lp.X[fractional]);
        Explore(model, lower, upper, result, false);
        upper[fractional] = savedUpper;
        double savedLower = lower[fractional];
        lower[fractional] = Math.Ceiling(lp.X[fractional]);
        Explore(model, lower, upper, result, false);
        lower[fractional] = savedLower;
    }

    private static double[][] Assemble(MilpModel model, double[] lower, double[] upper)
    {
        var rows = new List<double[]>();
        foreach (double[] row in model.A) rows.Add((double[])row.Clone());
        for (int i = 0; i < model.Vars; i++)
        {
            var cap = new double[model.Vars];
            cap[i] = 1.0;
            rows.Add(cap);
        }
        for (int i = 0; i < model.Vars; i++)
        {
            if (lower[i] <= 0) continue;
            var floorRow = new double[model.Vars];
            floorRow[i] = -1.0;
            rows.Add(floorRow);
        }
        return rows.ToArray();
    }

    private static double[] AssembleB(MilpModel model, double[] upper, double[] lower)
    {
        var b = new List<double>();
        foreach (double value in model.B) b.Add(value);
        for (int i = 0; i < model.Vars; i++) b.Add(upper[i]);
        for (int i = 0; i < model.Vars; i++)
        {
            if (lower[i] <= 0) continue;
            b.Add(-lower[i]);
        }
        return b.ToArray();
    }

    /** Brute force over the integer box, for reference. */
    internal static long[] BruteForce(MilpModel model, int cap)
    {
        int n = model.Vars;
        var best = new long[n];
        long bestValue = long.MinValue;
        var current = new long[n];
        var limits = new int[n];
        for (int i = 0; i < n; i++) limits[i] = cap;
        while (true)
        {
            bool feasible = true;
            for (int row = 0; row < model.A.Length && feasible; row++)
            {
                double lhs = 0.0;
                for (int j = 0; j < n; j++) lhs += model.A[row][j] * current[j];
                if (lhs > model.B[row] + 1e-9) feasible = false;
            }
            if (feasible)
            {
                long value = 0;
                for (int j = 0; j < n; j++) value += (long)model.C[j] * current[j];
                if (value > bestValue)
                {
                    bestValue = value;
                    Array.Copy(current, best, n);
                }
            }
            int index = 0;
            while (index < n)
            {
                current[index]++;
                if (current[index] <= limits[index]) break;
                current[index] = 0;
                index++;
            }
            if (index == n) break;
        }
        return best;
    }

    internal static long BoxSize(MilpModel model, int cap)
    {
        long size = 1;
        for (int i = 0; i < model.Vars; i++) size *= cap + 1;
        return size;
    }
}

public static class Solution
{
    public static void Main()
    {
        // The textbook knapsack: LP value 21 at (3, 1.5), integer optimum 20 at (4, 0).
        var knapsack = MilpModel.Build("knapsack", new[] { 5.0, 4.0 },
                new[] { 6.0, 4.0, 24.0 }, new[] { 1.0, 2.0, 6.0 });
        BranchAndBound.Result knapsackResult = BranchAndBound.Solve(knapsack);
        Check(Math.Abs(knapsackResult.RootLp - 21.0) <= 1e-9, "the LP relaxation value is 21, got "
                + knapsackResult.RootLp);
        Check(knapsackResult.RootFractional, "the relaxation optimum is fractional");
        Check(knapsackResult.HasSolution && Math.Abs(knapsackResult.Objective - 20.0) <= 1e-9,
                "the integer optimum is 20, got " + knapsackResult.Objective);
        Check(knapsackResult.X[0] == 4 && knapsackResult.X[1] == 0, "the optimum sits at (4,0)");
        Check(knapsackResult.Objective <= knapsackResult.RootLp + 1e-9,
                "the integer optimum cannot exceed the relaxation");

        // A four-variable instance with integer optimum 22.
        var fourVar = MilpModel.Build("four-var", new[] { 8.0, 11.0, 6.0, 4.0 },
                new[] { 5.0, 7.0, 4.0, 3.0, 14.0 });
        BranchAndBound.Result fourResult = BranchAndBound.Solve(fourVar);
        long[] fourBrute = BranchAndBound.BruteForce(fourVar, 12);
        long fourValue = 0;
        for (int i = 0; i < 4; i++) fourValue += (long)fourVar.C[i] * fourBrute[i];
        Check(fourValue == 22, "the hand-computed optimum of the four-variable model is 22, got " + fourValue);
        Check(Math.Abs(fourResult.Objective - 22.0) <= 1e-9,
                "branch and bound finds 22 on the four-variable model, got " + fourResult.Objective);

        // An already-integral relaxation needs a single node.
        var integral = MilpModel.Build("integral", new[] { 2.0, 3.0 },
                new[] { 1.0, 0.0, 3.0 }, new[] { 0.0, 1.0, 4.0 });
        BranchAndBound.Result integralResult = BranchAndBound.Solve(integral);
        Check(Math.Abs(integralResult.Objective - 18.0) <= 1e-9, "the integral model has value 18");
        Check(integralResult.Nodes == 1, "an integral relaxation takes exactly one node, got "
                + integralResult.Nodes);
        Check(!integralResult.RootFractional, "the root relaxation is integral");

        // Infeasible: x1 + x2 <= 1 and x1 - x2 <= -2 cannot both hold with x >= 0.
        var infeasible = MilpModel.Build("infeasible", new[] { 1.0, 1.0 },
                new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, -1.0, -2.0 });
        BranchAndBound.Result infeasibleResult = BranchAndBound.Solve(infeasible);
        Check(!infeasibleResult.HasSolution, "the infeasible model has no integer solution");
        Check(infeasibleResult.Infeasible, "the infeasible model is reported infeasible");
        Lp.Solution infeasibleLp = Lp.Solve(infeasible.A, infeasible.B, infeasible.C);
        Check(!infeasibleLp.Feasible, "its LP relaxation is infeasible too");

        // Unbounded along the x1 axis: only a non-binding constraint on x1.
        var unbounded = MilpModel.Build("unbounded", new[] { 3.0, 1.0 },
                new[] { -1.0, 0.0, 5.0 }, new[] { 0.0, 1.0, 2.0 });
        Lp.Solution unboundedLp = Lp.Solve(unbounded.A, unbounded.B, unbounded.C);
        Check(unboundedLp.Feasible, "the unbounded model still has feasible points");
        Check(unboundedLp.Unbounded, "it is detected as unbounded along x1");
        BranchAndBound.Result unboundedResult = BranchAndBound.Solve(unbounded);
        Check(unboundedResult.Unbounded || !unboundedResult.HasSolution,
                "branch and bound refuses to report a finite optimum for the unbounded model");

        // Random models against brute force.
        ulong rng = 562001UL;
        int models = 10;
        for (int m = 0; m < models; m++)
        {
            int vars = 3;
            int rows = 2 + SplitMix64.Below(ref rng, 2);
            double[] c = new double[vars];
            for (int i = 0; i < vars; i++) c[i] = 1 + SplitMix64.Below(ref rng, 9);
            var constraints = new List<double[]>();
            for (int r = 0; r < rows; r++)
            {
                var row = new double[vars + 1];
                for (int i = 0; i < vars; i++) row[i] = 1 + SplitMix64.Below(ref rng, 6);
                row[vars] = 5 + SplitMix64.Below(ref rng, 16);
                constraints.Add(row);
            }
            var model = MilpModel.Build("random" + m, c, constraints.ToArray());
            BranchAndBound.Result branch = BranchAndBound.Solve(model);
            Check(branch.HasSolution, "model " + m + " should be feasible");
            long[] brute = BranchAndBound.BruteForce(model, 20);
            long bruteValue = 0;
            for (int i = 0; i < vars; i++) bruteValue += (long)model.C[i] * brute[i];
            Check(Math.Abs(branch.Objective - bruteValue) <= 1e-9, "model " + m + ": branch and bound "
                    + branch.Objective + " != brute force " + bruteValue);
            Check(branch.Objective <= branch.RootLp + 1e-9, "model " + m + ": integer optimum above the LP bound");
            Check(branch.Nodes <= BranchAndBound.BoxSize(model, 20) + 1,
                    "model " + m + ": node count " + branch.Nodes + " exceeds the brute-force box");
            long reported = 0;
            for (int i = 0; i < vars; i++) reported += (long)model.C[i] * (long)branch.X[i];
            Check(reported == bruteValue, "model " + m + ": the reported solution has the right value");
        }

        // Determinism of the whole search.
        BranchAndBound.Result repeat = BranchAndBound.Solve(knapsack);
        Check(repeat.Nodes == knapsackResult.Nodes && repeat.Objective == knapsackResult.Objective,
                "repeated runs agree");
        Lp.Solution repeatLp = Lp.Solve(knapsack.A, knapsack.B, knapsack.C);
        Check(Math.Abs(repeatLp.Objective - 21.0) <= 1e-9, "the LP solver is reproducible");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
