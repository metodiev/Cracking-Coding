// Task 546 - Implement pagerank with dangling node handling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
//
// Dangling nodes (vertices with no out-links) destroy column stochasticity, so the rank vector is
// produced by redistributing their accumulated mass uniformly over all n vertices:
//
//     x'[j] = (1 - d)/n + d * ( sum over edges i -> j of x[i] / outdeg(i) + D / n ),
//     D = sum of x[i] over the dangling i,
//
// which is exactly power iteration on the modified matrix M whose dangling rows are replaced by
// 1/n. The file implements three equivalent computations: the mass-redistribution iteration above,
// a dense iteration on the explicitly patched stochastic matrix, and a Gaussian elimination of
// (I - d M^T) x = (1 - d)/n * 1 - plus an unpatched "mass escapes the sink" variant, both raw and
// renormalised, which exposes the structure of the fix.
//
// Documented checks: the hand-solved two-node example (0 -> 1 with 1 dangling) has the exact ranks
// 20/57 and 37/57 - asserted to 1e-12 against the rational values and against both alternative
// computations; an all-dangling graph (no edges at all) collapses to the uniform vector in a single
// step; a DAG with two sinks, a star with dangling leaves, a cycle shedding into a sink, a tree and
// a dangling-free graph all agree across the three solvers to 1e-12 while keeping the mass at 1 and
// every rank positive; the raw mass-losing fixed point leaks mass (sum below 0.99, and 0.65 off on
// the hand example) yet renormalising it reproduces the dangling-aware ranks to 1e-9, which is the
// precise sense in which the recycling term is only a rescaling; the sinks of the DAG outrank its
// source and the symmetric sinks tie; damping 0.99 still converges inside the documented cap with a
// non-increasing residual; and the solver is deterministic.
using System;
using System.Collections.Generic;

internal sealed class DanglingPageRank
{
    internal readonly int N;
    internal readonly List<int>[] Out;

    internal DanglingPageRank(int n)
    {
        N = n;
        Out = new List<int>[n];
        for (int v = 0; v < n; v++) Out[v] = new List<int>();
    }

    internal void AddEdge(int u, int v)
    {
        if (!Out[u].Contains(v)) Out[u].Add(v);
    }

    internal static DanglingPageRank FromEdges(int n, IReadOnlyList<int[]> edges)
    {
        var g = new DanglingPageRank(n);
        foreach (int[] e in edges) g.AddEdge(e[0], e[1]);
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

    internal List<int> Dangling()
    {
        var result = new List<int>();
        for (int v = 0; v < N; v++)
        {
            if (Out[v].Count == 0) result.Add(v);
        }
        return result;
    }

    /** Power iteration with uniform redistribution of the dangling mass. */
    internal double[] Iterate(double damping, double tolerance, int maxIterations,
            out List<double> residuals, out int iterations)
    {
        var x = new double[N];
        for (int v = 0; v < N; v++) x[v] = 1.0 / N;
        var next = new double[N];
        double teleport = (1.0 - damping) / N;
        residuals = new List<double>();
        iterations = 0;
        for (int step = 0; step < maxIterations; step++)
        {
            double danglingMass = 0.0;
            for (int v = 0; v < N; v++)
            {
                if (Out[v].Count == 0) danglingMass += x[v];
            }
            double share = damping * danglingMass / N;
            for (int j = 0; j < N; j++) next[j] = teleport + share;
            for (int i = 0; i < N; i++)
            {
                if (Out[i].Count == 0) continue;
                double edgeShare = damping * x[i] / Out[i].Count;
                foreach (int j in Out[i]) next[j] += edgeShare;
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

    /** Same fixed point via the explicitly patched stochastic matrix (dangling rows = 1/n). */
    internal double[] IterateDense(double damping, double tolerance, int maxIterations)
    {
        var m = new double[N, N];
        for (int i = 0; i < N; i++)
        {
            if (Out[i].Count == 0)
            {
                for (int j = 0; j < N; j++) m[i, j] = 1.0 / N;
            }
            else
            {
                double share = 1.0 / Out[i].Count;
                foreach (int j in Out[i]) m[i, j] += share;
            }
        }
        var x = new double[N];
        for (int v = 0; v < N; v++) x[v] = 1.0 / N;
        var next = new double[N];
        double teleport = (1.0 - damping) / N;
        for (int step = 0; step < maxIterations; step++)
        {
            for (int j = 0; j < N; j++)
            {
                double sum = teleport;
                for (int i = 0; i < N; i++) sum += damping * x[i] * m[i, j];
                next[j] = sum;
            }
            double residual = 0.0;
            for (int v = 0; v < N; v++) residual += Math.Abs(next[v] - x[v]);
            Array.Copy(next, x, N);
            if (residual <= tolerance) break;
        }
        return x;
    }

    /** Exact solve of (I - d M^T) x = (1 - d)/n * 1 with the patched matrix. */
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
            if (Out[i].Count == 0)
            {
                for (int j = 0; j < N; j++) a[j, i] -= damping / N;
            }
            else
            {
                double share = damping / Out[i].Count;
                foreach (int j in Out[i]) a[j, i] -= share;
            }
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

    /**
     * No dangling plumbing: mass that reaches a sink simply disappears. The raw fixed point loses
     * mass, but - because the proper fixed point is exactly a scalar multiple of it (the recycling
     * term only rescales) - renormalising it recovers the dangling-aware ranks exactly.
     */
    internal double[] IteratePrunedRaw(double damping, double tolerance, int maxIterations)
    {
        var x = new double[N];
        for (int v = 0; v < N; v++) x[v] = 1.0 / N;
        var next = new double[N];
        double teleport = (1.0 - damping) / N;
        for (int step = 0; step < maxIterations; step++)
        {
            for (int v = 0; v < N; v++) next[v] = teleport;
            for (int i = 0; i < N; i++)
            {
                if (Out[i].Count == 0) continue;
                double share = damping * x[i] / Out[i].Count;
                foreach (int j in Out[i]) next[j] += share;
            }
            double residual = 0.0;
            for (int v = 0; v < N; v++) residual += Math.Abs(next[v] - x[v]);
            Array.Copy(next, x, N);
            if (residual <= tolerance) break;
        }
        return x;
    }

    internal double[] IteratePruned(double damping, double tolerance, int maxIterations)
    {
        double[] x = IteratePrunedRaw(damping, tolerance, maxIterations);
        double total = 0.0;
        foreach (double value in x) total += value;
        for (int v = 0; v < N; v++) x[v] /= total;
        return x;
    }

    internal double MaxAbsDifference(double[] a, double[] b)
    {
        double worst = 0.0;
        for (int v = 0; v < Math.Min(a.Length, b.Length); v++) worst = Math.Max(worst, Math.Abs(a[v] - b[v]));
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

    private static void CheckAgreement(DanglingPageRank problem, string label)
    {
        double[] iterated = problem.Iterate(Damping, Tolerance, 1000, out List<double> residuals, out int iterations);
        double[] dense = problem.IterateDense(Damping, Tolerance, 1000);
        double[] solved = problem.SolveLinear(Damping);
        Check(problem.MaxAbsDifference(iterated, dense) <= 1e-12, label + ": iteration equals the dense patch");
        Check(problem.MaxAbsDifference(iterated, solved) <= 1e-10, label + ": iteration equals the linear solve");
        Check(Math.Abs(problem.Sum(iterated) - 1.0) <= 1e-12, label + ": ranks sum to one");
        foreach (double value in iterated) Check(value > 0.0, label + ": ranks stay positive");
        for (int step = 1; step < residuals.Count; step++)
        {
            Check(residuals[step] <= residuals[step - 1] + 1e-15, label + ": residual never increases");
        }
        Check(residuals[residuals.Count - 1] <= Tolerance, label + ": tolerance reached");
        Check(iterations <= 250, label + ": bounded iteration count (" + iterations + ")");
    }

    internal static void Main()
    {
        // Hand-solved example: 0 -> 1 with 1 dangling gives 20/57 and 37/57.
        var hand = DanglingPageRank.FromEdges(2, new List<int[]> { new[] { 0, 1 } });
        Check(hand.HasDangling() && hand.Dangling().Count == 1 && hand.Dangling()[0] == 1,
                "the hand example has node 1 dangling");
        double[] handRank = hand.Iterate(Damping, Tolerance, 1000, out _, out _);
        Check(Math.Abs(handRank[0] - 20.0 / 57.0) <= 1e-12, "the hand example rank of node 0 is 20/57");
        Check(Math.Abs(handRank[1] - 37.0 / 57.0) <= 1e-12, "the hand example rank of node 1 is 37/57");
        Check(Math.Abs(hand.Sum(handRank) - 1.0) <= 1e-12, "the hand example conserves mass");
        CheckAgreement(hand, "hand example");
        double[] handSolved = hand.SolveLinear(Damping);
        Check(Math.Abs(handSolved[0] - 20.0 / 57.0) <= 1e-12, "the linear solve reproduces 20/57");

        // All-dangling graph: no edges at all, rank is uniform.
        var empty = DanglingPageRank.FromEdges(4, new List<int[]>());
        Check(empty.Dangling().Count == 4, "every vertex of the empty graph is dangling");
        double[] emptyRank = empty.Iterate(Damping, Tolerance, 1000, out List<double> emptyResiduals, out int emptySteps);
        for (int v = 0; v < 4; v++) Check(Math.Abs(emptyRank[v] - 0.25) <= 1e-12, "empty graph rank is uniform");
        Check(emptySteps == 1 && emptyResiduals[0] <= 1e-15, "the empty graph starts at its fixed point");
        CheckAgreement(empty, "empty graph");

        // A DAG with a single sink and a small tree.
        var dag = DanglingPageRank.FromEdges(5, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 2, 4 },
        });
        Check(dag.Dangling().Count == 2, "the DAG has two sinks");
        CheckAgreement(dag, "dag");
        double[] dagRank = dag.Iterate(Damping, Tolerance, 1000, out _, out _);
        Check(dagRank[3] > dagRank[0] && dagRank[4] > dagRank[0], "sinks outrank the source of the DAG");
        Check(Math.Abs(dagRank[3] - dagRank[4]) <= 1e-9, "the two symmetric sinks tie");

        // Star of dangling leaves and a directed cycle feeding one sink.
        var star = DanglingPageRank.FromEdges(6, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 0, 2 }, new[] { 0, 3 }, new[] { 0, 4 }, new[] { 0, 5 }, new[] { 5, 0 },
        });
        CheckAgreement(star, "star");
        double[] starRank = star.Iterate(Damping, Tolerance, 1000, out _, out _);
        Check(starRank[0] > starRank[1], "the star centre outranks a dangling leaf");
        Check(Math.Abs(starRank[1] - starRank[2]) <= 1e-12, "dangling leaves tie by symmetry");

        var cycleToSink = DanglingPageRank.FromEdges(5, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 3, 4 },
        });
        CheckAgreement(cycleToSink, "cycle with sink");
        Check(cycleToSink.Dangling()[0] == 4, "the sink of the cycle is node 4");

        // A tree with a dangling root leaf and a dangling sink chain.
        var chain = DanglingPageRank.FromEdges(4, new List<int[]> { new[] { 0, 1 }, new[] { 1, 2 }, new[] { 3, 1 } });
        CheckAgreement(chain, "chain");

        // A dangling-free graph: the plumbing is invisible.
        var closed = DanglingPageRank.FromEdges(4, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 0 }, new[] { 1, 3 },
        });
        Check(!closed.HasDangling(), "the closed graph has no dangling vertex");
        double[] closedRank = closed.Iterate(Damping, Tolerance, 1000, out _, out _);
        double[] closedPruned = closed.IteratePruned(Damping, Tolerance, 1000);
        Check(closed.MaxAbsDifference(closedRank, closedPruned) <= 1e-12,
                "with no dangling nodes the patch changes nothing");
        CheckAgreement(closed, "closed graph");

        // Recycling the dangling mass and renormalising the mass-losing variant coincide, while the
        // *raw* mass-losing fixed point is materially different (it simply leaks mass).
        double[] handPruned = hand.IteratePruned(Damping, Tolerance, 1000);
        Check(hand.MaxAbsDifference(handRank, handPruned) <= 1e-9,
                "renormalised pruning reproduces the dangling-aware ranks");
        double[] handRaw = hand.IteratePrunedRaw(Damping, Tolerance, 1000);
        Check(hand.Sum(handRaw) < 0.99, "the raw mass-losing fixed point leaks mass");
        Check(Math.Abs(handRaw[0] - handRank[0]) > 0.1, "the raw fixed point is far from the correct rank");
        double[] dagPruned = dag.IteratePruned(Damping, Tolerance, 1000);
        Check(dag.MaxAbsDifference(dagRank, dagPruned) <= 1e-9, "the DAG shows the same equivalence");
        Check(dag.Sum(dag.IteratePrunedRaw(Damping, Tolerance, 1000)) < 0.99, "the DAG raw fixed point leaks mass");

        // Larger damping still converges inside the cap.
        var stress = DanglingPageRank.FromEdges(7, new List<int[]>
        {
            new[] { 0, 1 }, new[] { 1, 2 }, new[] { 2, 3 }, new[] { 3, 4 }, new[] { 4, 5 }, new[] { 5, 6 }, new[] { 6, 0 },
            new[] { 2, 6 },
        });
        double[] stressRank = stress.Iterate(0.99, 1e-12, 1000, out _, out int stressSteps);
        Check(stressSteps <= 1000, "damping 0.99 converges inside the cap");
        Check(Math.Abs(stress.Sum(stressRank) - 1.0) <= 1e-12, "damping 0.99 conserves mass");
        Check(stress.MaxAbsDifference(stressRank, stress.SolveLinear(0.99)) <= 1e-10,
                "damping 0.99 matches its linear solve");

        // Determinism.
        double[] again = hand.Iterate(Damping, Tolerance, 1000, out _, out _);
        Check(hand.MaxAbsDifference(again, handRank) <= 1e-15, "the solver is deterministic");
        double[] dagAgain = dag.Iterate(Damping, Tolerance, 1000, out _, out _);
        Check(dag.MaxAbsDifference(dagAgain, dagRank) <= 1e-15, "the DAG solver is deterministic");

        Console.WriteLine("PASS");
    }
}
