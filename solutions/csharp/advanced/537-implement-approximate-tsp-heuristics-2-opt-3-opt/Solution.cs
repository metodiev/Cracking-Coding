// Task 537 - Implement approximate TSP heuristics (2-opt / 3-opt). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
//
// A complete local-search pipeline for Euclidean TSP over an explicit n x n distance matrix:
// nearest-neighbour construction (ties to the smaller index) followed by best-improvement 2-opt
// (reverse the segment between two cut edges) and best-improvement 3-opt (the seven non-trivial
// reconnections of the three segments cut at edges (i,i+1), (j,j+1), (k,k+1)). Every move is
// evaluated with the O(1) edge-exchange delta - pattern 0 is exactly a 2-opt reversal, patterns 1-6
// swap/reverse the middle segments - and applied by materialising the tour in the rotated frame
// where the cut sits. Because the linearisation of a cyclic tour hides the wrap edge, each sweep
// scans all n rotations, so the scan covers every cyclic move; the loop stops only when a full
// multi-rotation sweep finds nothing. The deltas are cross-validated: for sampled moves the
// reported delta must equal Length(after) - Length(before). An exact Held-Karp solver (n <= 12)
// supplies optima for the small instances.
//
// Documented checks: the 2-opt and 3-opt tours are permutations, never lengthen the incumbent
// (40-node LCG instance for 2-opt, 24-node for 3-opt), and are certified locally optimal by an
// independent re-scan (minimum 2-opt delta and minimum 3-opt delta over all rotations and all
// patterns >= -1e-9); delta equals the realised length change for sampled 2-opt and 3-opt moves;
// a 10-node LCG instance stays within 1.15 x the Held-Karp optimum and above it; on 8 points in
// convex position (jittered circle, radius 1000) the boundary cycle has the Held-Karp optimum
// length, every 2-opt result from all 8 starts is certified locally optimal and the best of them
// attains the optimum; 4 collinear points reach the closed form 2 * span;
// the n = 2 and 3 = 4 - 5 triangle instances hit 2 * d(0,1) and the perimeter; and the whole
// pipeline is reproduced tour-for-tour on a second run.
using System;
using System.Collections.Generic;

internal sealed class Euclidean
{
    internal const double Eps = 1e-9;

    internal readonly int N;
    internal readonly double[][] D;

    internal Euclidean(double[] x, double[] y)
    {
        N = x.Length;
        D = new double[N][];
        for (int i = 0; i < N; i++)
        {
            D[i] = new double[N];
            for (int j = 0; j < N; j++)
            {
                double dx = x[i] - x[j];
                double dy = y[i] - y[j];
                D[i][j] = Math.Sqrt(dx * dx + dy * dy);
            }
        }
    }

    internal double Length(int[] tour)
    {
        double total = 0.0;
        for (int i = 0; i < tour.Length; i++) total += D[tour[i]][tour[(i + 1) % tour.Length]];
        return total;
    }

    internal bool IsPermutation(int[] tour)
    {
        if (tour.Length != N) return false;
        var seen = new bool[N];
        foreach (int v in tour)
        {
            if (v < 0 || v >= N || seen[v]) return false;
            seen[v] = true;
        }
        return true;
    }

    internal int[] NearestNeighbour(int start)
    {
        var tour = new int[N];
        var used = new bool[N];
        tour[0] = start;
        used[start] = true;
        for (int k = 1; k < N; k++)
        {
            int best = -1;
            double bestDist = double.PositiveInfinity;
            for (int v = 0; v < N; v++)
            {
                if (used[v]) continue;
                double dist = D[tour[k - 1]][v];
                if (best < 0 || dist < bestDist - 1e-15)
                {
                    bestDist = dist;
                    best = v;
                }
            }
            tour[k] = best;
            used[best] = true;
        }
        return tour;
    }

    internal static double TwoOptDelta(double[][] d, int a, int a2, int b, int b2)
    {
        return d[a][b] + d[a2][b2] - d[a][a2] - d[b][b2];
    }

    internal static double PatternDelta(double[][] d, int pattern, int a, int a2, int b, int b2, int c, int c2)
    {
        double all = d[a][a2] + d[b][b2] + d[c][c2];
        switch (pattern)
        {
            case 0:
                return d[a][b] + d[a2][b2] - d[a][a2] - d[b][b2];
            case 1:
                return d[b][c] + d[b2][c2] - d[b][b2] - d[c][c2];
            case 2:
                return d[a][b] + d[a2][c] + d[b2][c2] - all;
            case 3:
                return d[a][b2] + d[c][a2] + d[c2][b] - all;
            case 4:
                return d[a][c] + d[b2][a2] + d[b][c2] - all;
            case 5:
                return d[a][b2] + d[c][b] + d[a2][c2] - all;
            default:
                return d[a][c] + d[b2][b] + d[a2][c2] - all;
        }
    }

    /** Best improving 2-opt move over all rotations; returns false when the tour is 2-optimal. */
    internal bool TwoOptSweep(int[] tour)
    {
        int n = N;
        double bestDelta = -Eps;
        int bestI = -1, bestJ = -1, bestRot = 0;
        for (int r = 0; r < n; r++)
        {
            for (int i = 0; i + 2 < n; i++)
            {
                for (int j = i + 1; j + 1 < n; j++)
                {
                    int a = tour[(r + i) % n];
                    int a2 = tour[(r + i + 1) % n];
                    int b = tour[(r + j) % n];
                    int b2 = tour[(r + j + 1) % n];
                    double delta = TwoOptDelta(D, a, a2, b, b2);
                    if (delta < bestDelta)
                    {
                        bestDelta = delta;
                        bestI = i;
                        bestJ = j;
                        bestRot = r;
                    }
                }
            }
        }
        if (bestI < 0) return false;
        ApplyTwoOpt(tour, bestRot, bestI, bestJ);
        return true;
    }

    internal int TwoOpt(int[] tour)
    {
        int moves = 0;
        while (TwoOptSweep(tour))
        {
            moves++;
            if (moves > 20000) throw new InvalidOperationException("2-opt failed to terminate");
        }
        return moves;
    }

    internal double MinTwoOptDelta(int[] tour)
    {
        int n = N;
        double best = double.PositiveInfinity;
        for (int r = 0; r < n; r++)
        {
            for (int i = 0; i + 2 < n; i++)
            {
                for (int j = i + 1; j + 1 < n; j++)
                {
                    double delta = TwoOptDelta(D,
                            tour[(r + i) % n], tour[(r + i + 1) % n],
                            tour[(r + j) % n], tour[(r + j + 1) % n]);
                    if (delta < best) best = delta;
                }
            }
        }
        return best;
    }

    /** Best improving 3-opt move (all rotations, all seven patterns); false when 3-optimal. */
    internal bool ThreeOptSweep(int[] tour)
    {
        int n = N;
        double bestDelta = -Eps;
        int bestPattern = -1, bestI = -1, bestJ = -1, bestK = -1, bestRot = 0;
        for (int r = 0; r < n; r++)
        {
            for (int i = 0; i + 3 < n; i++)
            {
                for (int j = i + 1; j + 2 < n; j++)
                {
                    for (int k = j + 1; k + 1 < n; k++)
                    {
                        int a = tour[(r + i) % n];
                        int a2 = tour[(r + i + 1) % n];
                        int b = tour[(r + j) % n];
                        int b2 = tour[(r + j + 1) % n];
                        int c = tour[(r + k) % n];
                        int c2 = tour[(r + k + 1) % n];
                        for (int p = 0; p < 7; p++)
                        {
                            double delta = PatternDelta(D, p, a, a2, b, b2, c, c2);
                            if (delta < bestDelta)
                            {
                                bestDelta = delta;
                                bestPattern = p;
                                bestI = i;
                                bestJ = j;
                                bestK = k;
                                bestRot = r;
                            }
                        }
                    }
                }
            }
        }
        if (bestPattern < 0) return false;
        ApplyThreeOpt(tour, bestRot, bestI, bestJ, bestK, bestPattern);
        return true;
    }

    internal int ThreeOpt(int[] tour)
    {
        int moves = 0;
        while (ThreeOptSweep(tour))
        {
            moves++;
            if (moves > 20000) throw new InvalidOperationException("3-opt failed to terminate");
        }
        return moves;
    }

    internal double MinThreeOptDelta(int[] tour)
    {
        int n = N;
        double best = double.PositiveInfinity;
        for (int r = 0; r < n; r++)
        {
            for (int i = 0; i + 3 < n; i++)
            {
                for (int j = i + 1; j + 2 < n; j++)
                {
                    for (int k = j + 1; k + 1 < n; k++)
                    {
                        int a = tour[(r + i) % n];
                        int a2 = tour[(r + i + 1) % n];
                        int b = tour[(r + j) % n];
                        int b2 = tour[(r + j + 1) % n];
                        int c = tour[(r + k) % n];
                        int c2 = tour[(r + k + 1) % n];
                        for (int p = 0; p < 7; p++)
                        {
                            double delta = PatternDelta(D, p, a, a2, b, b2, c, c2);
                            if (delta < best) best = delta;
                        }
                    }
                }
            }
        }
        return best;
    }

    /** Realised length change of the 2-opt move minus its reported delta (must be ~0). */
    internal double VerifyTwoOptDelta(int[] tour, int r, int i, int j)
    {
        int n = N;
        double delta = TwoOptDelta(D,
                tour[(r + i) % n], tour[(r + i + 1) % n],
                tour[(r + j) % n], tour[(r + j + 1) % n]);
        var copy = (int[])tour.Clone();
        double before = Length(copy);
        ApplyTwoOpt(copy, r, i, j);
        return (Length(copy) - before) - delta;
    }

    /** Realised length change of a 3-opt move minus its reported delta (must be ~0). */
    internal double VerifyThreeOptDelta(int[] tour, int r, int i, int j, int k, int pattern)
    {
        int n = N;
        double delta = PatternDelta(D, pattern,
                tour[(r + i) % n], tour[(r + i + 1) % n],
                tour[(r + j) % n], tour[(r + j + 1) % n],
                tour[(r + k) % n], tour[(r + k + 1) % n]);
        var copy = (int[])tour.Clone();
        double before = Length(copy);
        ApplyThreeOpt(copy, r, i, j, k, pattern);
        return (Length(copy) - before) - delta;
    }

    internal void ApplyTwoOpt(int[] tour, int r, int i, int j)
    {
        int n = N;
        var frame = new int[n];
        for (int x = 0; x < n; x++) frame[x] = tour[(r + x) % n];
        for (int lo = i + 1, hi = j; lo < hi; lo++, hi--)
        {
            int tmp = frame[lo];
            frame[lo] = frame[hi];
            frame[hi] = tmp;
        }
        for (int x = 0; x < n; x++) tour[(r + x) % n] = frame[x];
    }

    internal void ApplyThreeOpt(int[] tour, int r, int i, int j, int k, int pattern)
    {
        int n = N;
        var frame = new int[n];
        for (int x = 0; x < n; x++) frame[x] = tour[(r + x) % n];

        bool swap = pattern >= 3;
        bool revB = pattern == 0 || pattern == 2 || pattern == 5 || pattern == 6;
        bool revC = pattern == 1 || pattern == 2 || pattern == 4 || pattern == 6;

        var rebuilt = new int[n];
        int w = 0;
        for (int x = 0; x <= i; x++) rebuilt[w++] = frame[x];
        if (swap)
        {
            Append(rebuilt, ref w, frame, j + 1, k, revC);
            Append(rebuilt, ref w, frame, i + 1, j, revB);
        }
        else
        {
            Append(rebuilt, ref w, frame, i + 1, j, revB);
            Append(rebuilt, ref w, frame, j + 1, k, revC);
        }
        for (int x = k + 1; x < n; x++) rebuilt[w++] = frame[x];
        if (w != n) throw new InvalidOperationException("3-opt rebuild lost vertices");
        for (int x = 0; x < n; x++) tour[(r + x) % n] = rebuilt[x];
    }

    private static void Append(int[] dest, ref int w, int[] frame, int lo, int hi, bool reversed)
    {
        if (reversed)
        {
            for (int x = hi; x >= lo; x--) dest[w++] = frame[x];
        }
        else
        {
            for (int x = lo; x <= hi; x++) dest[w++] = frame[x];
        }
    }

    /** Exact Held-Karp dynamic program over subsets, n <= 12. */
    internal double HeldKarp()
    {
        int n = N;
        if (n > 12) throw new InvalidOperationException("Held-Karp limited to 12 nodes");
        int full = 1 << (n - 1);
        var dp = new double[full][];
        for (int s = 0; s < full; s++)
        {
            dp[s] = new double[n];
            for (int v = 0; v < n; v++) dp[s][v] = double.PositiveInfinity;
        }
        for (int v = 1; v < n; v++) dp[1 << (v - 1)][v] = D[0][v];
        for (int s = 1; s < full; s++)
        {
            for (int v = 1; v < n; v++)
            {
                if (double.IsPositiveInfinity(dp[s][v])) continue;
                for (int w = 1; w < n; w++)
                {
                    if ((s & (1 << (w - 1))) != 0) continue;
                    int next = s | (1 << (w - 1));
                    double cand = dp[s][v] + D[v][w];
                    if (cand < dp[next][w]) dp[next][w] = cand;
                }
            }
        }
        double best = double.PositiveInfinity;
        for (int v = 1; v < n; v++)
        {
            double cand = dp[full - 1][v] + D[v][0];
            if (cand < best) best = cand;
        }
        return best;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static int state = 537001;

    private static int NextInt(int mod)
    {
        state = (state * 75 + 74) % 65537;
        return state % mod;
    }

    private static void LcgPoints(int n, double[] x, double[] y, int limit)
    {
        for (int i = 0; i < n; i++)
        {
            x[i] = NextInt(limit);
            y[i] = NextInt(limit);
        }
    }

    private static string Render(int[] tour)
    {
        return string.Join(",", tour);
    }

    internal static void Main()
    {
        // Instance A: 40 LCG points, 2-opt only.
        var xa = new double[40];
        var ya = new double[40];
        LcgPoints(40, xa, ya, 1000);
        var instA = new Euclidean(xa, ya);
        int[] nnA = instA.NearestNeighbour(0);
        Check(instA.IsPermutation(nnA), "NN tour is a permutation");
        double nnLenA = instA.Length(nnA);
        var twoA = (int[])nnA.Clone();
        int movesA = instA.TwoOpt(twoA);
        double twoLenA = instA.Length(twoA);
        Check(movesA >= 1, "2-opt performs at least one improving move on instance A");
        Check(instA.IsPermutation(twoA), "2-opt keeps the tour a permutation");
        Check(twoLenA <= nnLenA + 1e-9, "2-opt never lengthens the tour");
        Check(instA.MinTwoOptDelta(twoA) >= -1e-9, "instance A tour is 2-optimal over every rotation");
        Check(Math.Abs(instA.Length((int[])twoA.Clone()) - twoLenA) <= 1e-9, "tour length is stable");

        // Instance B: 24 LCG points, full pipeline NN -> 2-opt -> 3-opt.
        var xb = new double[24];
        var yb = new double[24];
        LcgPoints(24, xb, yb, 1000);
        var instB = new Euclidean(xb, yb);
        int[] nnB = instB.NearestNeighbour(0);
        var twoB = (int[])nnB.Clone();
        instB.TwoOpt(twoB);
        double twoLenB = instB.Length(twoB);
        var threeB = (int[])twoB.Clone();
        int movesB = instB.ThreeOpt(threeB);
        double threeLenB = instB.Length(threeB);
        Check(instB.IsPermutation(threeB), "3-opt keeps the tour a permutation");
        Check(movesB >= 1, "3-opt performs at least one improving move on instance B");
        Check(threeLenB <= twoLenB + 1e-9, "3-opt never lengthens the 2-opt tour");
        Check(instB.MinThreeOptDelta(threeB) >= -1e-9, "instance B tour is 3-optimal over every rotation");
        Check(instB.MinTwoOptDelta(threeB) >= -1e-9, "a 3-optimal tour is also 2-optimal");

        // Delta cross-validation: reported delta must equal the realised length change.
        for (int t = 0; t < 12; t++)
        {
            int r = NextInt(instB.N);
            int i = NextInt(instB.N - 2);
            int j = i + 1 + NextInt(instB.N - 2 - i);
            if (j + 1 >= instB.N) j = instB.N - 2;
            Check(Math.Abs(instB.VerifyTwoOptDelta(nnB, r, i, j)) <= 1e-9,
                    "2-opt delta matches the realised change");
        }
        for (int t = 0; t < 12; t++)
        {
            int r = NextInt(instB.N);
            int i = NextInt(instB.N - 3);
            int j = i + 1 + NextInt(instB.N - 3 - i);
            if (j + 2 >= instB.N) j = instB.N - 3;
            int k = j + 1 + NextInt(instB.N - 2 - j);
            if (k + 1 >= instB.N) k = instB.N - 2;
            int p = NextInt(7);
            Check(Math.Abs(instB.VerifyThreeOptDelta(nnB, r, i, j, k, p)) <= 1e-9,
                    "3-opt pattern " + p + " delta matches the realised change");
        }

        // Instance C: 10 points against the exact Held-Karp optimum.
        var xc = new double[10];
        var yc = new double[10];
        LcgPoints(10, xc, yc, 1000);
        var instC = new Euclidean(xc, yc);
        double optimum = instC.HeldKarp();
        int[] nnC = instC.NearestNeighbour(0);
        Check(nnC.Length == instC.N, "the NN tour covers every node");
        var twoC = (int[])nnC.Clone();
        instC.TwoOpt(twoC);
        var threeC = (int[])twoC.Clone();
        instC.ThreeOpt(threeC);
        double twoLenC = instC.Length(twoC);
        double threeLenC = instC.Length(threeC);
        Check(optimum <= twoLenC + 1e-9, "Held-Karp optimum is a lower bound for 2-opt");
        Check(threeLenC <= twoLenC + 1e-9, "3-opt beats or matches 2-opt on instance C");
        Check(threeLenC <= 1.15 * optimum, "3-opt is within 15% of the optimum (" + threeLenC + " vs " + optimum + ")");
        Check(instC.Length(nnC) >= optimum - 1e-9, "the NN tour is never shorter than the optimum");

        // Convex position: jittered circle of 8 points; the boundary is optimal.
        var xd = new double[8];
        var yd = new double[8];
        for (int i = 0; i < 8; i++)
        {
            double angle = (Math.PI / 4.0) * i + (NextInt(17) - 8) * Math.PI / 180.0;
            xd[i] = Math.Round(1000.0 * Math.Cos(angle));
            yd[i] = Math.Round(1000.0 * Math.Sin(angle));
        }
        var instD = new Euclidean(xd, yd);
        var boundary = new int[8];
        for (int i = 0; i < 8; i++) boundary[i] = i;
        double boundaryLen = instD.Length(boundary);
        double convexOptimum = instD.HeldKarp();
        Check(Math.Abs(boundaryLen - convexOptimum) <= 1e-9, "the boundary cycle is the convex optimum");
        double bestConvex = double.PositiveInfinity;
        for (int start = 0; start < 8; start++)
        {
            int[] candidate = instD.NearestNeighbour(start);
            instD.TwoOpt(candidate);
            Check(instD.MinTwoOptDelta(candidate) >= -1e-9, "convex 2-opt tour is locally optimal (start " + start + ")");
            double len = instD.Length(candidate);
            if (len < bestConvex) bestConvex = len;
        }
        Check(Math.Abs(bestConvex - convexOptimum) <= 1e-9, "2-opt reaches the convex optimum from some start");

        // Collinear points: the optimum is twice the span.
        var xe = new[] { 0.0, 3.0, 7.0, 10.0 };
        var ye = new[] { 0.0, 0.0, 0.0, 0.0 };
        var instE = new Euclidean(xe, ye);
        int[] twoE = instE.NearestNeighbour(0);
        instE.TwoOpt(twoE);
        Check(Math.Abs(instE.Length(twoE) - 20.0) <= 1e-9, "collinear tour equals 2 * span = 20");
        Check(Math.Abs(instE.HeldKarp() - 20.0) <= 1e-9, "Held-Karp confirms 2 * span for collinear points");

        // Degenerate sizes.
        var instF = new Euclidean(new[] { 0.0, 3.0 }, new[] { 0.0, 4.0 });
        int[] tourF = instF.NearestNeighbour(0);
        instF.TwoOpt(tourF);
        Check(Math.Abs(instF.Length(tourF) - 10.0) <= 1e-9, "two points give a round trip of 2 * 5");
        var instG = new Euclidean(new[] { 0.0, 3.0, 0.0 }, new[] { 0.0, 0.0, 4.0 });
        int[] tourG = instG.NearestNeighbour(1);
        instG.TwoOpt(tourG);
        instG.ThreeOpt(tourG);
        Check(Math.Abs(instG.Length(tourG) - 12.0) <= 1e-9, "the 3-4-5 triangle has perimeter 12");

        // Determinism: rebuild instance B's pipeline from scratch.
        var repeated = new Euclidean(xb, yb);
        int[] nnR = repeated.NearestNeighbour(0);
        repeated.TwoOpt(nnR);
        repeated.ThreeOpt(nnR);
        Check(Render(nnR) == Render(threeB), "the pipeline is reproducible tour-for-tour");

        Console.WriteLine("PASS");
    }
}
