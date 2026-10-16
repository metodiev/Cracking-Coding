// Task 589 - Implement spectral clustering using sparse eigen solver (power iteration multiple vectors). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Spectral clustering driven by a sparse eigensolver. The graph is kept as adjacency lists and the
// only operations ever performed on it are matrix-vector products: `ApplyLaplacian` computes
// (D - A) x by touching exactly n diagonal entries plus 2m off-diagonal ones (the file counts those
// accesses and asserts the count, so "sparse" is a checked property rather than a claim), and
// `ApplyNormalizedLaplacian` computes (I - D^-1/2 A D^-1/2) x by scaling with the inverse square-root
// degrees. The smallest eigenvectors of L are obtained by orthogonal iteration on the shifted,
// PSD-flipped operator M = sigma I - L with sigma = 2 max_degree >= ||L||_2 (Gershgorin), so the
// *largest* eigenvectors of M are the smallest of L, and using a k-dimensional block with a
// twice-repeated modified Gram-Schmidt step ("power iteration multiple vectors") is what makes the
// second and third vectors reachable at all. Ground truth for every eigenvalue comes from two
// places: a dense cyclic Jacobi eigensolver over the explicitly materialised dense Laplacian, and
// closed forms -- the path graph P_n has eigenvalues 2 - 2cos(pi k / n) with eigenvectors
// cos(pi k (i + 1/2) / n), which pins the Fiedler value and vector down analytically. Clustering is
// the classical two-step recipe: embed the vertices in the rows of the k computed vectors and split
// by the sign of the Fiedler coordinate (for k = 2), which is then compared against exhaustive
// search over all 2^(n-1) bisections of small graphs and against the 2/vol-normalised cut value.
//
// Documented checks: Jacobi reproduces the textbook eigenvalues {1,3} of [[2,1],[1,2]] and all ten
// closed-form eigenvalues of the P_10 Laplacian to 1e-9; the sparse products agree with dense
// matrix-vector multiplication to 1e-12 and touch exactly 2m + n entries; L 1 = 0 and
// L_sym D^(1/2) 1 = 0 to 1e-12; the single-vector power iteration converges to the trivial
// eigenvalue 0 with the uniform vector 1/sqrt(n) up to sign; for P_10 the two-vector block iteration
// returns lambda_1 = 0 and lambda_2 = 2 - 2cos(pi/10) = 0.0979 to 1e-9, its columns are orthonormal
// to 1e-12, both residuals ||L q - lambda q|| are below 1e-9, and the Fiedler vector matches the
// closed form cos(pi (i + 1/2) / n) to 1e-6 after normalisation and sign fixing; on three disjoint
// 4-cliques the three computed vectors span exactly the component-indicator space, verified by
// comparing the two projectors in the Frobenius norm against 1e-6, while each indicator vector
// itself is already an eigenvector of L with eigenvalue 0; the two-blob graph (two 5-cliques joined
// by one bridge) is clustered perfectly by the sign of the Fiedler coordinate, and the resulting
// normalised cut 2/21 equals the exhaustive minimum over all 512 bisections (also reached by the
// four-vertex instance, 2/13 over 128 bisections) while the ratio cut 1/4 + 1/4 = 0.5 on the eight-vertex
// graph attains the textbook bound 4/n = 0.5; every normalised-Laplacian eigenvalue of the tested graphs
// lies in [0,2] to 1e-9 with a nullity of exactly one, since the single bridge keeps the graph connected,
// and the largest one is strictly below 2 because the cliques are not bipartite; the shifted Rayleigh
// quotient is monotone non-decreasing along the iteration and the residual
// after 5000 iterations is strictly below the residual after 20, so the solver is documented as
// converging gradually rather than magically; and the whole computation is reproducible because every
// starting block comes from the fixed-seed 64-bit LCG (MMIX constants) documented in `Lcg64`.
using System;
using System.Collections.Generic;

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

    internal static double NextUnit(ref ulong state)
    {
        return (NextBelow(ref state, 2001) / 1000.0) - 1.0;
    }
}

internal sealed class SparseGraph
{
    internal readonly int N;
    internal readonly List<int>[] Adjacency;
    internal readonly int[] Degree;
    internal int LastTouchCount;

    internal SparseGraph(int n)
    {
        N = n;
        Adjacency = new List<int>[n];
        Degree = new int[n];
        for (int v = 0; v < n; v++) Adjacency[v] = new List<int>();
    }

    internal void AddEdge(int u, int v)
    {
        if (u == v || Adjacency[u].Contains(v)) return;
        Adjacency[u].Add(v);
        Adjacency[v].Add(u);
        Degree[u]++;
        Degree[v]++;
    }

    internal int EdgeCount
    {
        get
        {
            int total = 0;
            for (int v = 0; v < N; v++) total += Adjacency[v].Count;
            return total / 2;
        }
    }

    internal int MaxDegree()
    {
        int worst = 0;
        foreach (int value in Degree) worst = Math.Max(worst, value);
        return worst;
    }

    /** (D - A) x, counting every matrix entry touched. */
    internal double[] ApplyLaplacian(double[] x)
    {
        var result = new double[N];
        LastTouchCount = 0;
        for (int v = 0; v < N; v++)
        {
            double sum = Degree[v] * x[v];
            LastTouchCount++;
            foreach (int u in Adjacency[v])
            {
                sum -= x[u];
                LastTouchCount++;
            }
            result[v] = sum;
        }
        return result;
    }

    /** (I - D^-1/2 A D^-1/2) x. */
    internal double[] ApplyNormalizedLaplacian(double[] x)
    {
        var result = new double[N];
        for (int v = 0; v < N; v++)
        {
            if (Degree[v] == 0)
            {
                result[v] = x[v];
                continue;
            }
            double neighbourSum = 0.0;
            foreach (int u in Adjacency[v])
            {
                if (Degree[u] > 0) neighbourSum += x[u] / Math.Sqrt(Degree[u]);
            }
            result[v] = x[v] - neighbourSum / Math.Sqrt(Degree[v]);
        }
        return result;
    }

    internal double[] ApplyShiftedLaplacian(double[] x, double shift)
    {
        double[] lx = ApplyLaplacian(x);
        var result = new double[N];
        for (int v = 0; v < N; v++) result[v] = shift * x[v] - lx[v];
        return result;
    }

    internal double[] ApplyAdjacency(double[] x)
    {
        var result = new double[N];
        for (int v = 0; v < N; v++)
        {
            double sum = 0.0;
            foreach (int u in Adjacency[v]) sum += x[u];
            result[v] = sum;
        }
        return result;
    }

    internal double[][] DenseLaplacian()
    {
        var matrix = new double[N][];
        for (int i = 0; i < N; i++)
        {
            matrix[i] = new double[N];
            matrix[i][i] = Degree[i];
            foreach (int u in Adjacency[i]) matrix[i][u] = -1.0;
        }
        return matrix;
    }

    internal double[][] DenseNormalizedLaplacian()
    {
        var matrix = new double[N][];
        for (int i = 0; i < N; i++)
        {
            matrix[i] = new double[N];
            matrix[i][i] = 1.0;
            foreach (int u in Adjacency[i])
            {
                if (Degree[i] > 0 && Degree[u] > 0) matrix[i][u] = -1.0 / Math.Sqrt(Degree[i] * (double)Degree[u]);
            }
        }
        return matrix;
    }

    /** Two cliques of the given size joined by exactly one bridge edge (deterministic). */
    internal static SparseGraph TwoBlobs(int size)
    {
        var graph = new SparseGraph(2 * size);
        for (int block = 0; block < 2; block++)
        {
            int start = block * size;
            for (int i = 0; i < size; i++)
            {
                for (int j = i + 1; j < size; j++) graph.AddEdge(start + i, start + j);
            }
        }
        graph.AddEdge(0, size);
        return graph;
    }

    internal static SparseGraph Path(int n)
    {
        var graph = new SparseGraph(n);
        for (int v = 0; v + 1 < n; v++) graph.AddEdge(v, v + 1);
        return graph;
    }

    internal static SparseGraph DisjointCliques(int blocks, int size)
    {
        var graph = new SparseGraph(blocks * size);
        for (int block = 0; block < blocks; block++)
        {
            int start = block * size;
            for (int i = 0; i < size; i++)
            {
                for (int j = i + 1; j < size; j++) graph.AddEdge(start + i, start + j);
            }
        }
        return graph;
    }
}

internal static class Spectral
{
    /** Dense cyclic Jacobi eigensolver for symmetric matrices; vectors are the columns. */
    internal static void Jacobi(double[][] source, int maxSweeps, double tolerance,
            out double[] values, out double[][] vectors)
    {
        int n = source.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = (double[])source[i].Clone();
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < maxSweeps; sweep++)
        {
            double off = 0.0;
            for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += m[i][j] * m[i][j];
            if (Math.Sqrt(off) <= tolerance) break;
            for (int p = 0; p < n - 1; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(m[p][q]) <= 1e-300) continue;
                    double theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                    double t = Math.Abs(theta) < 1e-300 ? 1.0 : Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int i = 0; i < n; i++)
                    {
                        double mip = m[i][p];
                        double miq = m[i][q];
                        m[i][p] = c * mip - s * miq;
                        m[i][q] = s * mip + c * miq;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double mpi = m[p][i];
                        double mqi = m[q][i];
                        m[p][i] = c * mpi - s * mqi;
                        m[q][i] = s * mpi + c * mqi;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        double vip = v[i][p];
                        double viq = v[i][q];
                        v[i][p] = c * vip - s * viq;
                        v[i][q] = s * vip + c * viq;
                    }
                }
            }
        }
        values = new double[n];
        for (int i = 0; i < n; i++) values[i] = m[i][i];
        vectors = v;
    }

    /** Twice-repeated modified Gram-Schmidt on the columns of x, in place. */
    internal static void Orthonormalize(double[][] x)
    {
        int n = x.Length;
        int k = x[0].Length;
        for (int pass = 0; pass < 2; pass++)
        {
            for (int j = 0; j < k; j++)
            {
                for (int p = 0; p < j; p++)
                {
                    double dot = 0.0;
                    for (int i = 0; i < n; i++) dot += x[i][j] * x[i][p];
                    for (int i = 0; i < n; i++) x[i][j] -= dot * x[i][p];
                }
                double norm = 0.0;
                for (int i = 0; i < n; i++) norm += x[i][j] * x[i][j];
                norm = Math.Sqrt(norm);
                for (int i = 0; i < n; i++) x[i][j] /= norm;
            }
        }
    }

    internal static double[][] RandomBlock(int n, int k, ulong seed)
    {
        var x = new double[n][];
        ulong state = seed;
        for (int i = 0; i < n; i++)
        {
            x[i] = new double[k];
            for (int j = 0; j < k; j++) x[i][j] = Lcg64.NextUnit(ref state);
        }
        return x;
    }

    /** Orthogonal iteration on sigma I - L: the k largest directions of M are the k smallest of L. */
    internal static double[][] OrthogonalIteration(SparseGraph graph, int k, int iterations, ulong seed)
    {
        double shift = 2.0 * graph.MaxDegree();
        var x = RandomBlock(graph.N, k, seed);
        Orthonormalize(x);
        for (int it = 0; it < iterations; it++)
        {
            var z = new double[graph.N][];
            for (int j = 0; j < k; j++)
            {
                var column = new double[graph.N];
                for (int i = 0; i < graph.N; i++) column[i] = x[i][j];
                double[] product = graph.ApplyShiftedLaplacian(column, shift);
                for (int i = 0; i < graph.N; i++)
                {
                    if (j == 0) z[i] = new double[k];
                    z[i][j] = product[i];
                }
            }
            Orthonormalize(z);
            x = z;
        }
        return x;
    }

    /** Ritz values lambda_j = q_j^T L q_j for the columns of q. */
    internal static double[] RitzValues(SparseGraph graph, double[][] q)
    {
        int k = q[0].Length;
        var values = new double[k];
        for (int j = 0; j < k; j++)
        {
            var column = new double[graph.N];
            for (int i = 0; i < graph.N; i++) column[i] = q[i][j];
            double[] lx = graph.ApplyLaplacian(column);
            double total = 0.0;
            for (int i = 0; i < graph.N; i++) total += column[i] * lx[i];
            values[j] = total;
        }
        return values;
    }

    internal static double Residual(SparseGraph graph, double[] vector, double lambda, bool normalized)
    {
        double[] lx = normalized ? graph.ApplyNormalizedLaplacian(vector) : graph.ApplyLaplacian(vector);
        double worst = 0.0;
        for (int i = 0; i < vector.Length; i++) worst = Math.Max(worst, Math.Abs(lx[i] - lambda * vector[i]));
        return worst;
    }

    internal static double RayleightQuotient(SparseGraph graph, double[] vector, bool normalized)
    {
        double[] lx = normalized ? graph.ApplyNormalizedLaplacian(vector) : graph.ApplyLaplacian(vector);
        double dot = 0.0;
        double norm = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            dot += vector[i] * lx[i];
            norm += vector[i] * vector[i];
        }
        return dot / norm;
    }

    /** Unit-norm Fiedler-like vector: normalised Laplacian eigenvector for the second smallest value. */
    internal static double[] NormalizedFiedler(SparseGraph graph, int iterations, ulong seed, out double eigenvalue)
    {
        double[][] basis = OrthogonalIterationNormalized(graph, 2, iterations, seed);
        var vector = new double[graph.N];
        for (int i = 0; i < graph.N; i++) vector[i] = basis[i][1];
        eigenvalue = RayleighNormalized(graph, basis, 1);
        return vector;
    }

    internal static double[][] OrthogonalIterationNormalized(SparseGraph graph, int k, int iterations, ulong seed)
    {
        const double Shift = 2.0;
        var x = RandomBlock(graph.N, k, seed);
        Orthonormalize(x);
        for (int it = 0; it < iterations; it++)
        {
            var z = new double[graph.N][];
            for (int j = 0; j < k; j++)
            {
                var column = new double[graph.N];
                for (int i = 0; i < graph.N; i++) column[i] = x[i][j];
                double[] lx = graph.ApplyNormalizedLaplacian(column);
                for (int i = 0; i < graph.N; i++)
                {
                    if (j == 0) z[i] = new double[k];
                    z[i][j] = Shift * column[i] - lx[i];
                }
            }
            Orthonormalize(z);
            x = z;
        }
        return x;
    }

    private static double RayleighNormalized(SparseGraph graph, double[][] q, int column)
    {
        var vector = new double[graph.N];
        for (int i = 0; i < graph.N; i++) vector[i] = q[i][column];
        return RayleightQuotient(graph, vector, true);
    }

    /** Sign cut with a canonical global flip so that vertex 0 lands on the positive side. */
    internal static bool[] SignCut(double[] vector)
    {
        bool flip = vector[0] < 0.0;
        var side = new bool[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            double value = flip ? -vector[i] : vector[i];
            side[i] = value >= 0.0;
        }
        return side;
    }

    internal static double NormalizedCut(SparseGraph graph, bool[] side)
    {
        double cut = 0.0;
        double volumeS = 0.0;
        double volumeT = 0.0;
        for (int v = 0; v < graph.N; v++)
        {
            if (side[v]) volumeS += graph.Degree[v];
            else volumeT += graph.Degree[v];
            foreach (int u in graph.Adjacency[v])
            {
                if (u > v && side[u] != side[v]) cut += 1.0;
            }
        }
        if (volumeS == 0.0 || volumeT == 0.0) return double.PositiveInfinity;
        return cut / volumeS + cut / volumeT;
    }

    internal static double RatioCut(SparseGraph graph, bool[] side)
    {
        double cut = 0.0;
        int countS = 0;
        int countT = 0;
        for (int v = 0; v < graph.N; v++)
        {
            if (side[v]) countS++;
            else countT++;
            foreach (int u in graph.Adjacency[v])
            {
                if (u > v && side[u] != side[v]) cut += 1.0;
            }
        }
        if (countS == 0 || countT == 0) return double.PositiveInfinity;
        return cut / countS + cut / countT;
    }

    /** Exhaustive minimum normalised cut over all non-trivial bisections. */
    internal static double BestNormalizedCut(SparseGraph graph, out bool[] bestSide)
    {
        int n = graph.N;
        bestSide = new bool[n];
        double best = double.PositiveInfinity;
        int subsetCount = 1 << (n - 1);
        for (int mask = 0; mask < subsetCount; mask++)
        {
            var side = new bool[n];
            side[0] = false;
            for (int v = 1; v < n; v++) side[v] = (mask >> (v - 1) & 1) == 1;
            double value = NormalizedCut(graph, side);
            if (value < best)
            {
                best = value;
                bestSide = side;
            }
        }
        return best;
    }

    internal static bool SamePartition(bool[] a, bool[] b)
    {
        bool same = true;
        bool flipped = true;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) same = false;
            if (a[i] == b[i]) flipped = false;
        }
        return same || flipped;
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
}

public static class Solution
{
    private const int SlowIterations = 5000;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckClose(double actual, double expected, double tolerance, string label)
    {
        Check(Math.Abs(actual - expected) <= tolerance, label + " (got " + actual + ", expected " + expected + ")");
    }

    public static void Main()
    {
        // ---- ground-truth eigensolver ----------------------------------------------------
        Spectral.Jacobi(new[] { new[] { 2.0, 1.0 }, new[] { 1.0, 2.0 } }, 100, 1e-14, out double[] smallValues, out _);
        var sortedSmall = (double[])smallValues.Clone();
        Array.Sort(sortedSmall);
        CheckClose(sortedSmall[0], 1.0, 1e-12, "Jacobi gives eigenvalue 1 for [[2,1],[1,2]]");
        CheckClose(sortedSmall[1], 3.0, 1e-12, "Jacobi gives eigenvalue 3 for [[2,1],[1,2]]");

        const int PathN = 10;
        SparseGraph path = SparseGraph.Path(PathN);
        Spectral.Jacobi(path.DenseLaplacian(), 200, 1e-14, out double[] pathValues, out _);
        var pathSorted = (double[])pathValues.Clone();
        Array.Sort(pathSorted);
        for (int k = 0; k < PathN; k++)
        {
            CheckClose(pathSorted[k], 2.0 - 2.0 * Math.Cos(Math.PI * k / PathN), 1e-9,
                    "the P_10 Laplacian eigenvalue " + k + " matches 2 - 2cos(pi k / n)");
        }

        // ---- sparse products --------------------------------------------------------------
        ulong state = 589001UL;
        var probe = new double[PathN];
        for (int i = 0; i < PathN; i++) probe[i] = Lcg64.NextUnit(ref state);
        double[] sparseProduct = path.ApplyLaplacian(probe);
        Check(path.LastTouchCount == 2 * path.EdgeCount + PathN,
                "the sparse Laplacian touches exactly 2m + n entries (" + path.LastTouchCount + ")");
        double[][] densePath = path.DenseLaplacian();
        for (int i = 0; i < PathN; i++)
        {
            double dense = 0.0;
            for (int j = 0; j < PathN; j++) dense += densePath[i][j] * probe[j];
            CheckClose(sparseProduct[i], dense, 1e-12, "the sparse product equals the dense product at " + i);
        }
        var ones = new double[PathN];
        for (int i = 0; i < PathN; i++) ones[i] = 1.0;
        double[] laplacianOnes = path.ApplyLaplacian(ones);
        for (int i = 0; i < PathN; i++) CheckClose(laplacianOnes[i], 0.0, 1e-12, "L 1 = 0 at " + i);
        var sqrtDegrees = new double[PathN];
        for (int i = 0; i < PathN; i++) sqrtDegrees[i] = Math.Sqrt(path.Degree[i]);
        double[] normalizedOnSqrtDegrees = path.ApplyNormalizedLaplacian(sqrtDegrees);
        for (int i = 0; i < PathN; i++)
        {
            CheckClose(normalizedOnSqrtDegrees[i], 0.0, 1e-12, "L_sym D^(1/2) 1 = 0 at " + i);
        }

        // ---- single-vector power iteration ------------------------------------------------
        var uniform = new double[PathN];
        ulong powerState = 589777UL;
        for (int i = 0; i < PathN; i++) uniform[i] = Lcg64.NextUnit(ref powerState);
        for (int it = 0; it < SlowIterations; it++)
        {
            double[] next = path.ApplyShiftedLaplacian(uniform, 2.0 * path.MaxDegree());
            double norm = 0.0;
            for (int i = 0; i < PathN; i++) norm += next[i] * next[i];
            norm = Math.Sqrt(norm);
            for (int i = 0; i < PathN; i++) uniform[i] = next[i] / norm;
        }
        double trivial = Spectral.RayleightQuotient(path, uniform, false);
        CheckClose(trivial, 0.0, 1e-9, "the plain power iteration lands on the trivial eigenvalue 0");
        bool allSameMagnitude = true;
        for (int i = 0; i < PathN; i++)
        {
            if (Math.Abs(Math.Abs(uniform[i]) - 1.0 / Math.Sqrt(PathN)) > 1e-9) allSameMagnitude = false;
        }
        Check(allSameMagnitude, "the trivial eigenvector is the uniform vector 1/sqrt(n) up to sign");
        Check(Spectral.Residual(path, uniform, 0.0, false) <= 1e-9, "the trivial eigenvector has a tiny residual");

        // ---- block iteration: Fiedler value and vector against the closed form -------------
        double[][] pathBasis = Spectral.OrthogonalIteration(path, 2, SlowIterations, 589123UL);
        double[] pathRitz = Spectral.RitzValues(path, pathBasis);
        CheckClose(pathRitz[0], 0.0, 1e-9, "the block iteration finds the trivial eigenvalue first");
        double fiedlerClosedForm = 2.0 - 2.0 * Math.Cos(Math.PI / PathN);
        CheckClose(pathRitz[1], fiedlerClosedForm, 1e-9,
                "the Fiedler value of P_10 is 2 - 2cos(pi/10) = 0.097887");
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double dot = 0.0;
                for (int r = 0; r < PathN; r++) dot += pathBasis[r][i] * pathBasis[r][j];
                CheckClose(dot, i == j ? 1.0 : 0.0, 1e-12, "the block columns are orthonormal at (" + i + "," + j + ")");
            }
        }
        for (int j = 0; j < 2; j++)
        {
            var column = new double[PathN];
            for (int i = 0; i < PathN; i++) column[i] = pathBasis[i][j];
            Check(Spectral.Residual(path, column, pathRitz[j], false) <= 1e-9, "the residual of block column " + j + " is below 1e-9");
        }
        var fiedlerVector = new double[PathN];
        for (int i = 0; i < PathN; i++) fiedlerVector[i] = pathBasis[i][1];
        double sign = fiedlerVector[0] >= 0.0 ? 1.0 : -1.0;
        double analyticNorm = 0.0;
        for (int i = 0; i < PathN; i++) analyticNorm += Math.Pow(Math.Cos(Math.PI * (i + 0.5) / PathN), 2.0);
        analyticNorm = Math.Sqrt(analyticNorm);
        for (int i = 0; i < PathN; i++)
        {
            double expected = sign * Math.Cos(Math.PI * (i + 0.5) / PathN) / analyticNorm;
            CheckClose(fiedlerVector[i], expected, 1e-6,
                    "the Fiedler vector of P_10 matches cos(pi(i+1/2)/n) at " + i);
        }

        // ---- convergence behaviour --------------------------------------------------------
        double[][] fewIterations = Spectral.OrthogonalIteration(path, 2, 20, 589123UL);
        double residualFew = Spectral.Residual(path, Column(fewIterations, 1), Spectral.RitzValues(path, fewIterations)[1], false);
        double residualMany = Spectral.Residual(path, Column(pathBasis, 1), pathRitz[1], false);
        Check(residualMany < residualFew, "more iterations strictly reduce the residual (" + residualFew.ToString("0.0e0")
                + " -> " + residualMany.ToString("0.0e0") + ")");
        var trace = new List<double>();
        var rayleighVector = new double[PathN];
        ulong traceState = 589999UL;
        for (int i = 0; i < PathN; i++) rayleighVector[i] = Lcg64.NextUnit(ref traceState);
        double previous = double.NegativeInfinity;
        bool monotone = true;
        for (int it = 0; it < 40; it++)
        {
            double[] shifted = path.ApplyShiftedLaplacian(rayleighVector, 2.0 * path.MaxDegree());
            double quotient = 0.0;
            double norm = 0.0;
            for (int i = 0; i < PathN; i++)
            {
                quotient += rayleighVector[i] * shifted[i];
                norm += rayleighVector[i] * rayleighVector[i];
            }
            quotient /= norm;
            trace.Add(quotient);
            if (quotient < previous - 1e-12) monotone = false;
            previous = quotient;
            double productNorm = 0.0;
            for (int i = 0; i < PathN; i++) productNorm += shifted[i] * shifted[i];
            productNorm = Math.Sqrt(productNorm);
            for (int i = 0; i < PathN; i++) rayleighVector[i] = shifted[i] / productNorm;
        }
        Check(monotone, "the shifted Rayleigh quotient is monotone along the power iteration");
        Check(trace[trace.Count - 1] > trace[0], "the shifted Rayleigh quotient grows over 40 iterations");

        // ---- disjoint components: the computed span is the indicator span ----------------
        SparseGraph cliques = SparseGraph.DisjointCliques(3, 4);
        double[][] cliqueBasis = Spectral.OrthogonalIteration(cliques, 3, 200, 589321UL);
        double[] cliqueRitz = Spectral.RitzValues(cliques, cliqueBasis);
        for (int j = 0; j < 3; j++)
        {
            CheckClose(cliqueRitz[j], 0.0, 1e-9, "the disconnected graph has a zero eigenvalue at " + j);
        }
        var indicatorProjector = new double[cliques.N][];
        for (int i = 0; i < cliques.N; i++) indicatorProjector[i] = new double[cliques.N];
        for (int block = 0; block < 3; block++)
        {
            var indicator = new double[cliques.N];
            for (int i = 0; i < 4; i++) indicator[block * 4 + i] = 1.0 / Math.Sqrt(4.0);
            double[] product = cliques.ApplyLaplacian(indicator);
            for (int i = 0; i < cliques.N; i++) CheckClose(product[i], 0.0, 1e-12, "the indicator of component " + block + " is a null vector");
            for (int i = 0; i < cliques.N; i++)
            {
                for (int j = 0; j < cliques.N; j++) indicatorProjector[i][j] += indicator[i] * indicator[j];
            }
        }
        double projectorError = 0.0;
        for (int i = 0; i < cliques.N; i++)
        {
            for (int j = 0; j < cliques.N; j++)
            {
                double computed = 0.0;
                for (int c = 0; c < 3; c++) computed += cliqueBasis[i][c] * cliqueBasis[j][c];
                projectorError += Math.Pow(computed - indicatorProjector[i][j], 2.0);
            }
        }
        Check(Math.Sqrt(projectorError) <= 1e-6,
                "the computed span equals the component-indicator span (Frobenius error "
                + Math.Sqrt(projectorError).ToString("0.0e0") + ")");

        // ---- spectral clustering versus exhaustive search --------------------------------
        SparseGraph blobs = SparseGraph.TwoBlobs(5);
        double[] blobFiedler = Spectral.NormalizedFiedler(blobs, SlowIterations, 589444UL, out double blobEigenvalue);
        bool[] blobSides = Spectral.SignCut(blobFiedler);
        var trueSides = new bool[blobs.N];
        for (int v = 0; v < blobs.N; v++) trueSides[v] = v < 5;
        bool partitionMatches = Spectral.SamePartition(blobSides, trueSides);
        double spectralCut = Spectral.NormalizedCut(blobs, blobSides);
        double bruteCut = Spectral.BestNormalizedCut(blobs, out bool[] bruteSides);
        Check(blobs.EdgeCount == 21, "the two-blob graph has 21 edges (twenty within the cliques plus the bridge)");
        Check(blobEigenvalue > 0.0 && blobEigenvalue < 0.2,
                "the normalised Fiedler value of the two-blob graph is small but nonzero ("
                + blobEigenvalue.ToString("0.0000") + "), because the single bridge still couples the cliques");
        Check(spectralCut >= bruteCut - 1e-12, "the spectral cut cannot beat the exhaustive optimum");
        CheckClose(spectralCut, bruteCut, 1e-9, "the sign cut attains the exhaustive minimum normalised cut");
        CheckClose(spectralCut, 2.0 / 21.0, 1e-9, "the minimum normalised cut of the two-blob graph is 2/21");
        Check(partitionMatches, "the sign of the normalised Fiedler coordinate recovers the two blobs exactly");

        SparseGraph small = SparseGraph.TwoBlobs(4);
        double[][] smallBasis = Spectral.OrthogonalIteration(small, 2, SlowIterations, 589222UL);
        bool[] smallSides = Spectral.SignCut(Column(smallBasis, 1));
        double smallSpectral = Spectral.NormalizedCut(small, smallSides);
        double smallBrute = Spectral.BestNormalizedCut(small, out bool[] smallBruteSides);
        CheckClose(smallSpectral, smallBrute, 1e-9, "the 8-vertex instance also attains its exhaustive optimum");
        CheckClose(smallBrute, 2.0 / 13.0, 1e-9, "the 8-vertex minimum normalised cut is 2/13");
        var badSide = new bool[small.N];
        for (int v = 0; v < small.N; v++) badSide[v] = v % 3 == 0;
        Check(Spectral.NormalizedCut(small, badSide) > smallSpectral + 0.1,
                "a scrambled partition scores strictly worse than the spectral one");
        Check(!Spectral.SamePartition(smallSides, badSide), "the scrambled partition is genuinely different");
        Check(Spectral.NormalizedCut(small, smallBruteSides) <= smallBrute + 1e-12, "the exhaustive side really is optimal");

        // ---- ratio cut bound and normalised Laplacian spectrum ---------------------------
        double ratio = Spectral.RatioCut(small, smallSides);
        CheckClose(ratio, 0.5, 1e-9, "the bridge cut of the 8-vertex graph has ratio cut 1/4 + 1/4 = 0.5");
        Check(ratio >= 4.0 / small.N - 1e-12, "the ratio cut attains the textbook bound 4/n = 0.5");
        Spectral.Jacobi(small.DenseNormalizedLaplacian(), 200, 1e-14, out double[] normalizedValues, out _);
        var normalizedSorted = (double[])normalizedValues.Clone();
        Array.Sort(normalizedSorted);
        foreach (double value in normalizedSorted)
        {
            Check(value >= -1e-9 && value <= 2.0 + 1e-9, "the normalised Laplacian spectrum lies in [0,2] ("
                    + value.ToString("0.000000") + ")");
        }
        CheckClose(normalizedSorted[0], 0.0, 1e-9, "the trivial normalised eigenvalue is 0");
        int nullity = 0;
        foreach (double value in normalizedSorted) if (value < 1e-9) nullity++;
        Check(nullity == 1, "the bridge keeps the graph connected, so the normalised nullity is exactly 1");
        Check(normalizedSorted[small.N - 1] < 2.0 - 1e-3,
                "the largest normalised eigenvalue is strictly below 2 because the cliques are not bipartite ("
                + normalizedSorted[small.N - 1].ToString("0.0000") + ")");

        // ---- determinism ------------------------------------------------------------------
        double[][] repeat = Spectral.OrthogonalIteration(small, 2, 50, 589222UL);
        double[][] again = Spectral.OrthogonalIteration(small, 2, 50, 589222UL);
        Check(Spectral.MaxAbsDifference(repeat, again) == 0.0, "identical seeds give bit-identical blocks");

        Console.WriteLine("PASS");
    }

    private static double[] Column(double[][] matrix, int column)
    {
        var result = new double[matrix.Length];
        for (int i = 0; i < matrix.Length; i++) result[i] = matrix[i][column];
        return result;
    }
}
