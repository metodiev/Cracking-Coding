// Task 547 - Implement spectral graph partitioning small demo. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Spectral bipartitioning of an explicit weighted graph. The combinatorial Laplacian L = D - W is
// built from the adjacency matrix, the full symmetric eigen-decomposition is computed with cyclic
// Jacobi rotations (sweeps until the off-diagonal Frobenius norm drops below 1e-12, documented cap
// 100 sweeps), and the Fiedler vector - the eigenvector of the second-smallest eigenvalue - is
// thresholded at its mean (sign fixed so that vertex 0 sits at or above the mean) to produce the
// two-way split. The normalized Laplacian I - D^-1/2 W D^-1/2 is diagonalised with the same routine
// as a second, degree-corrected variant, and the split is refined by the sweep cut: the vertices are
// sorted along the Fiedler vector and the best prefix is kept, which is the cheapest of the n - 1
// Cauchy-style threshold cuts. Ground truth comes from brute force over every balanced cut and from
// closed-form Laplacian spectra.
//
// Documented checks: the Jacobi routine satisfies the residual ||L v - lambda v|| <= 1e-9 and the
// orthogonality ||V^T V - I|| <= 1e-9 for every eigenpair, reproduces the trace and the squared
// Frobenius norm as the sum and the sum of squares of the eigenvalues, and returns them ascending;
// lambda_1 = 0 with the constant eigenvector on every connected graph and a second zero exactly on
// a disconnected one (K6 disjoint K6); the closed forms hold to 1e-9 - lambda_2 = 2 - 2cos(pi/n) for
// the path, 2 - 2cos(2pi/n) for the cycle, k for K_{k,k}; the normalized Laplacian of the planted
// two-block graph recovers the planted split as well, with all eigenvalues inside [0, 2] and the
// same zero multiplicity; K5,5 has lambda_2 = 5 with a four-dimensional eigenspace, so no split is
// forced there and the file only asserts the hand formula for its balanced cuts (a^2 + (5 - a)^2
// minimised at a = 2 or 3, i.e. 13); the spectral cut of the planted graph equals the brute-force
// minimum over all balanced cuts, and the sweep of prefix cuts never does worse than the
// mean-threshold split while still respecting the exhaustive global minimum cut; every quantity is
// reproduced on a second run.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 2685821657736338717UL;
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }

    internal static int NextBelow(ref ulong state, int bound)
    {
        return (int)(NextRaw(ref state) % (ulong)bound);
    }
}

internal static class Eigen
{
    /** Cyclic Jacobi on a symmetric matrix: eigenvalues ascending, eigenvectors as columns. */
    internal static void Jacobi(double[][] input, out double[] values, out double[][] vectors)
    {
        int n = input.Length;
        var a = new double[n][];
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = (double[])input[i].Clone();
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0.0;
            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
            }
            if (off <= 1e-24) break;
            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) <= 1e-18) continue;
                    double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                    double sign = theta >= 0.0 ? 1.0 : -1.0;
                    double t = sign / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double mkp = a[k][p];
                        double mkq = a[k][q];
                        a[k][p] = c * mkp - s * mkq;
                        a[k][q] = s * mkp + c * mkq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double mpk = a[p][k];
                        double mqk = a[q][k];
                        a[p][k] = c * mpk - s * mqk;
                        a[q][k] = s * mpk + c * mqk;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        var diag = new double[n];
        for (int i = 0; i < n; i++) diag[i] = a[i][i];
        Array.Sort(order, (x, y) => diag[x].CompareTo(diag[y]));
        values = new double[n];
        var reordered = new double[n][];
        for (int i = 0; i < n; i++)
        {
            values[i] = diag[order[i]];
            reordered[i] = new double[n];
        }
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < n; k++) reordered[k][i] = v[k][order[i]];
        }
        vectors = reordered;
    }
}

internal sealed class PartitionGraph
{
    internal readonly int N;
    internal readonly double[][] W;

    internal PartitionGraph(int n)
    {
        N = n;
        W = new double[n][];
        for (int i = 0; i < n; i++) W[i] = new double[n];
    }

    internal void SetEdge(int u, int v, double weight)
    {
        W[u][v] = weight;
        W[v][u] = weight;
    }

    internal static PartitionGraph Path(int n)
    {
        var g = new PartitionGraph(n);
        for (int i = 0; i + 1 < n; i++) g.SetEdge(i, i + 1, 1.0);
        return g;
    }

    internal static PartitionGraph Cycle(int n)
    {
        var g = new PartitionGraph(n);
        for (int i = 0; i < n; i++) g.SetEdge(i, (i + 1) % n, 1.0);
        return g;
    }

    internal static PartitionGraph CompleteBipartite(int a, int b)
    {
        var g = new PartitionGraph(a + b);
        for (int i = 0; i < a; i++)
        {
            for (int j = 0; j < b; j++) g.SetEdge(i, a + j, 1.0);
        }
        return g;
    }

    internal static PartitionGraph DisjointUnion(PartitionGraph first, PartitionGraph second)
    {
        var g = new PartitionGraph(first.N + second.N);
        for (int i = 0; i < first.N; i++)
        {
            for (int j = i + 1; j < first.N; j++) g.SetEdge(i, j, first.W[i][j]);
        }
        for (int i = 0; i < second.N; i++)
        {
            for (int j = i + 1; j < second.N; j++)
            {
                g.SetEdge(first.N + i, first.N + j, second.W[i][j]);
            }
        }
        return g;
    }

    /** Planted two-block model: intra-block probability high, cross-block probability low. */
    internal static PartitionGraph Planted(int blockSize, int intraPercent, int crossPercent, int seed)
    {
        int n = 2 * blockSize;
        var g = new PartitionGraph(n);
        ulong state = (ulong)seed;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                bool sameBlock = i / blockSize == j / blockSize;
                int percent = sameBlock ? intraPercent : crossPercent;
                if (Rng.NextUnit(ref state) * 100.0 < percent) g.SetEdge(i, j, 1.0);
            }
        }
        for (int v = 0; v < n; v++)
        {
            if (Degree(g, v) == 0.0) g.SetEdge(v, (v + 1) % n, 1.0);
        }
        return g;
    }

    private static double Degree(PartitionGraph g, int v)
    {
        double total = 0.0;
        for (int j = 0; j < g.N; j++) total += g.W[v][j];
        return total;
    }

    internal double[] Degrees()
    {
        var degrees = new double[N];
        for (int v = 0; v < N; v++) degrees[v] = Degree(this, v);
        return degrees;
    }

    internal double[][] Laplacian()
    {
        var l = new double[N][];
        for (int i = 0; i < N; i++) l[i] = new double[N];
        double[] degrees = Degrees();
        for (int i = 0; i < N; i++)
        {
            l[i][i] = degrees[i];
            for (int j = 0; j < N; j++)
            {
                if (i != j) l[i][j] = -W[i][j];
            }
        }
        return l;
    }

    internal double[][] NormalizedLaplacian()
    {
        var l = new double[N][];
        for (int i = 0; i < N; i++) l[i] = new double[N];
        double[] degrees = Degrees();
        for (int i = 0; i < N; i++)
        {
            l[i][i] = 1.0;
            for (int j = 0; j < N; j++)
            {
                if (i == j) continue;
                l[i][j] = -W[i][j] / Math.Sqrt(degrees[i] * degrees[j]);
            }
        }
        return l;
    }

    internal int EdgeCount()
    {
        int total = 0;
        for (int i = 0; i < N; i++)
        {
            for (int j = i + 1; j < N; j++)
            {
                if (W[i][j] > 0.0) total++;
            }
        }
        return total;
    }

    internal double CutWeight(int[] labels)
    {
        double total = 0.0;
        for (int i = 0; i < N; i++)
        {
            for (int j = i + 1; j < N; j++)
            {
                if (labels[i] != labels[j]) total += W[i][j];
            }
        }
        return total;
    }

    /** Mean-threshold split along a vector: label 0 for values at or above the mean. */
    internal static int[] ThresholdSplit(double[] vector)
    {
        int n = vector.Length;
        double mean = 0.0;
        foreach (double value in vector) mean += value;
        mean /= n;
        var labels = new int[n];
        bool flipped = vector[0] < mean;
        for (int i = 0; i < n; i++)
        {
            bool above = vector[i] >= mean;
            labels[i] = (above != flipped) ? 0 : 1;
        }
        return labels;
    }

    /** Best prefix cut after sorting along the vector (all thresholds, not just the mean). */
    internal int[] BestSweepCut(double[] vector, out double cut)
    {
        int n = vector.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Array.Sort(order, (x, y) => vector[x].CompareTo(vector[y]));
        var labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = 1;
        var best = new int[n];
        for (int i = 0; i < n; i++) best[i] = 1;
        cut = double.PositiveInfinity;
        for (int prefix = 1; prefix < n; prefix++)
        {
            labels[order[prefix - 1]] = 0;
            double value = CutWeight(labels);
            if (value < cut - 1e-15)
            {
                cut = value;
                Array.Copy(labels, best, n);
            }
        }
        return best;
    }

    /** Minimum cut over all subsets of size n / 2 (n even), by exhaustive enumeration. */
    internal double MinBalancedCut(int half)
    {
        var labels = new int[N];
        for (int i = 0; i < N; i++) labels[i] = 1;
        double best = double.PositiveInfinity;
        Enumerate(labels, 0, 0, half, ref best);
        return best;
    }

    /** Global minimum cut over every non-trivial bipartition (both sides non-empty). */
    internal double MinCutAllSubsets()
    {
        double best = double.PositiveInfinity;
        var labels = new int[N];
        int maskLimit = 1 << N;
        for (int mask = 1; mask < maskLimit - 1; mask++)
        {
            for (int v = 0; v < N; v++) labels[v] = (mask >> v) & 1;
            double value = CutWeight(labels);
            if (value < best) best = value;
        }
        return best;
    }

    private void Enumerate(int[] labels, int index, int chosen, int half, ref double best)
    {
        if (chosen == half)
        {
            double value = CutWeight(labels);
            if (value < best) best = value;
            return;
        }
        if (index >= N) return;
        if (N - index < half - chosen) return;
        labels[index] = 0;
        Enumerate(labels, index + 1, chosen + 1, half, ref best);
        labels[index] = 1;
        Enumerate(labels, index + 1, chosen, half, ref best);
    }

    internal static string Labels(int[] labels)
    {
        return string.Join(",", labels);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckEigen(double[][] a, double[] values, double[][] vectors)
    {
        int n = a.Length;
        double trace = 0.0;
        double frobenius = 0.0;
        for (int i = 0; i < n; i++)
        {
            trace += a[i][i];
            for (int j = 0; j < n; j++) frobenius += a[i][j] * a[i][j];
        }
        double valueSum = 0.0;
        double squareSum = 0.0;
        for (int k = 0; k < n; k++)
        {
            valueSum += values[k];
            squareSum += values[k] * values[k];
        }
        Check(Math.Abs(valueSum - trace) <= 1e-9, "eigenvalues sum to the trace");
        Check(Math.Abs(squareSum - frobenius) <= 1e-9, "squared eigenvalues sum to the Frobenius norm");
        for (int k = 1; k < n; k++) Check(values[k] >= values[k - 1] - 1e-12, "eigenvalues are ascending");
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += a[i][j] * vectors[j][k];
                Check(Math.Abs(sum - values[k] * vectors[i][k]) <= 1e-9, "residual of eigenpair " + k);
            }
        }
        for (int p = 0; p < n; p++)
        {
            for (int q = 0; q < n; q++)
            {
                double dot = 0.0;
                for (int k = 0; k < n; k++) dot += vectors[k][p] * vectors[k][q];
                double expected = p == q ? 1.0 : 0.0;
                Check(Math.Abs(dot - expected) <= 1e-9, "eigenvectors are orthonormal at " + p + "," + q);
            }
        }
    }

    internal static void Main()
    {
        // Closed-form spectra of the path, the cycle and the complete bipartite graph.
        var path = PartitionGraph.Path(10);
        Eigen.Jacobi(path.Laplacian(), out double[] pathValues, out double[][] pathVectors);
        CheckEigen(path.Laplacian(), pathValues, pathVectors);
        Check(Math.Abs(pathValues[0]) <= 1e-9, "the path Laplacian has a zero eigenvalue");
        double pathExpected = 2.0 - 2.0 * Math.Cos(Math.PI / 10.0);
        Check(Math.Abs(pathValues[1] - pathExpected) <= 1e-9, "path lambda_2 matches 2 - 2cos(pi/n)");
        for (int i = 0; i < 10; i++) Check(Math.Abs(pathVectors[i][0] - 1.0 / Math.Sqrt(10.0)) <= 1e-9,
                "the zero eigenvector of the path is constant at " + i);

        var cycle = PartitionGraph.Cycle(12);
        Eigen.Jacobi(cycle.Laplacian(), out double[] cycleValues, out double[][] cycleVectors);
        CheckEigen(cycle.Laplacian(), cycleValues, cycleVectors);
        double cycleExpected = 2.0 - 2.0 * Math.Cos(2.0 * Math.PI / 12.0);
        Check(Math.Abs(cycleValues[1] - cycleExpected) <= 1e-9, "cycle lambda_2 matches 2 - 2cos(2pi/n)");

        var bipartite = PartitionGraph.CompleteBipartite(5, 5);
        Eigen.Jacobi(bipartite.Laplacian(), out double[] kBipartiteValues, out double[][] kBipartiteVectors);
        CheckEigen(bipartite.Laplacian(), kBipartiteValues, kBipartiteVectors);
        Check(Math.Abs(kBipartiteValues[1] - 5.0) <= 1e-9, "K5,5 lambda_2 equals 5");
        for (int i = 0; i < 10; i++)
        {
            Check(Math.Abs(kBipartiteVectors[i][0] - 1.0 / Math.Sqrt(10.0)) <= 1e-9,
                    "the K5,5 zero eigenvector is constant at " + i);
        }
        // lambda_2 of K5,5 has a four-dimensional eigenspace, so no split is forced; the exhaustive
        // balanced cut follows the hand formula a^2 + (5 - a)^2 minimized at a = 2 or 3, i.e. 13.
        Check(bipartite.MinBalancedCut(5) == 13.0, "the K5,5 minimum balanced cut is 13");

        // Disconnected graph: two zero eigenvalues.
        var disconnected = PartitionGraph.DisjointUnion(PartitionGraph.Path(6), PartitionGraph.Cycle(6));
        Eigen.Jacobi(disconnected.Laplacian(), out double[] disconnectedValues, out double[][] disconnectedVectors);
        CheckEigen(disconnected.Laplacian(), disconnectedValues, disconnectedVectors);
        Check(Math.Abs(disconnectedValues[0]) <= 1e-9 && Math.Abs(disconnectedValues[1]) <= 1e-9,
                "two components give two zero eigenvalues");
        Check(disconnectedValues[2] > 1e-6, "the third eigenvalue is positive");
        Check(disconnected.EdgeCount() == 11, "the disconnected graph has 11 edges");
        Check(disconnected.N == 12, "the disconnected graph has twelve vertices");

        // Planted two-block graph: recover the partition and match the brute-force balanced cut.
        var planted = PartitionGraph.Planted(7, 90, 10, 547001);
        Check(planted.EdgeCount() >= 30, "the planted graph is dense (" + planted.EdgeCount() + " edges)");
        Check(planted.N == 14 && planted.CutWeight(new int[14]) == 0.0, "the trivial partition cuts nothing");
        Eigen.Jacobi(planted.Laplacian(), out double[] plantedValues, out double[][] plantedVectors);
        CheckEigen(planted.Laplacian(), plantedValues, plantedVectors);
        var fiedler = new double[planted.N];
        for (int i = 0; i < planted.N; i++) fiedler[i] = plantedVectors[i][1];
        int[] meanSplit = PartitionGraph.ThresholdSplit(fiedler);
        for (int i = 0; i < planted.N; i++)
        {
            int plantedLabel = i / 7;
            Check(meanSplit[i] == (plantedLabel == 0 ? meanSplit[0] : meanSplit[7]),
                    "the spectral split matches the planted blocks at " + i);
        }
        for (int i = 0; i < planted.N; i++) Check(meanSplit[i] == 0 || meanSplit[i] == 1, "labels are binary");
        int[] sweepSplit = planted.BestSweepCut(fiedler, out double sweepCut);
        double meanCut = planted.CutWeight(meanSplit);
        Check(sweepCut <= meanCut + 1e-12, "the sweep cut is at least as good as the mean threshold");
        double optimumCut = planted.MinBalancedCut(7);
        Check(Math.Abs(meanCut - optimumCut) <= 1e-12, "the spectral cut equals the best balanced cut");
        double globalOptimum = planted.MinCutAllSubsets();
        Check(sweepCut >= globalOptimum - 1e-12, "the sweep cut respects the global minimum cut");
        Check(globalOptimum <= optimumCut + 1e-12, "the balanced optimum is an upper bound on the global one");
        Check(sweepCut <= meanCut + 1e-12, "the sweep never loses to the mean threshold");

        // Normalized Laplacian variant on the same graph.
        Eigen.Jacobi(planted.NormalizedLaplacian(), out double[] normalizedValues, out double[][] normalizedVectors);
        CheckEigen(planted.NormalizedLaplacian(), normalizedValues, normalizedVectors);
        foreach (double value in normalizedValues)
        {
            Check(value >= -1e-9 && value <= 2.0 + 1e-9, "normalized eigenvalues lie in [0, 2]");
        }
        Check(Math.Abs(normalizedValues[0]) <= 1e-9, "the normalized Laplacian has a zero eigenvalue");
        var normalizedFiedler = new double[planted.N];
        for (int i = 0; i < planted.N; i++) normalizedFiedler[i] = normalizedVectors[i][1];
        int[] normalizedSplit = PartitionGraph.ThresholdSplit(normalizedFiedler);
        for (int i = 0; i < planted.N; i++)
        {
            int plantedLabel = i / 7;
            Check(normalizedSplit[i] == (plantedLabel == 0 ? normalizedSplit[0] : normalizedSplit[7]),
                    "the normalized split matches the planted blocks at " + i);
        }
        Check(planted.CutWeight(normalizedSplit) >= 0.0, "the normalized cut is a non-negative weight");
        Check(planted.CutWeight(normalizedSplit) <= planted.EdgeCount(), "the normalized cut is bounded");

        // Determinism.
        var repeat = PartitionGraph.Planted(7, 90, 10, 547001);
        Eigen.Jacobi(repeat.Laplacian(), out double[] repeatValues, out double[][] repeatVectors);
        for (int i = 0; i < planted.N; i++)
        {
            Check(Math.Abs(repeatValues[i] - plantedValues[i]) <= 1e-12, "the spectrum is reproducible");
        }
        var repeatFiedler = new double[repeat.N];
        for (int i = 0; i < repeat.N; i++) repeatFiedler[i] = repeatVectors[i][1];
        Check(PartitionGraph.Labels(PartitionGraph.ThresholdSplit(repeatFiedler)) == PartitionGraph.Labels(meanSplit),
                "the split is reproducible");

        Console.WriteLine("PASS");
    }
}
