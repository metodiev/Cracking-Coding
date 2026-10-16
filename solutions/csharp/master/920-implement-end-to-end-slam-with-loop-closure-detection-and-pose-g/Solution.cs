// Task 920 - Implement end-to-end SLAM with loop closure detection and pose graph optimization. (Tier: Master, Beyond — Graphics, Geometry & Robotics)
//
// Front end: the robot drives a closed rectangular loop and records for every pose the *landmark signature* (ids of the
// landmarks within 0.8 m). Detection scans pose k against poses 0..k-24 and accepts a candidate when the two signatures
// share at least two landmark ids - the classic appearance-based test - and takes a relative-pose measurement from the
// registration, marked as a loop edge.
//
// Back end: Gauss-Newton/Levenberg-Marquardt on the SE(2) pose graph. Edge (i, j) carries a measured relative pose z;
// predicted is R(-th_i)(p_j - p_i) with wrap(th_j - th_i), so r = pred - z is a 3-vector and the Jacobian blocks are
// the standard analytic ones (d pred_t/d p_i = -R(-th_i), d pred_t/d th_i = (pred_y, -pred_x), d pred_t/d p_j =
// R(-th_i), theta rows -1/+1). Normal equations H = sum J' W J, b = sum J' W r are assembled densely and solved by
// Gaussian elimination with pose 0 pinned by a large diagonal; the LM damping adapts on success/failure and every
// residual is Huber-weighted, so one gross outlier closure cannot drag the whole map. Simplifications: 2D poses,
// signature overlap as the only place-recognition cue, synthetic measurements from the ground truth. Verified: the
// closure is found on the revisiting loop and not on an open path, the robust cost falls sharply, ATE beats dead
// reckoning, pose 0 stays pinned, Huber beats plain least squares under a gross outlier, a two-pose graph reproduces
// its measurement exactly, and runs are reproducible.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    /** Zero-mean Gaussian sample (Box-Muller) from the deterministic stream. */
    internal static double Gaussian(ref ulong s)
    {
        double u = ((Next(ref s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
        double v = (Next(ref s) >> 11) * (1.0 / 9007199254740992.0);
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }
}

/** Pose graph in SE(2): dense normal equations, analytic Jacobians, Huber weights, pose 0 pinned. */
internal sealed class PoseGraph
{
    internal readonly int N;
    internal readonly double[] Pose;
    internal readonly List<int> EdgeI = new List<int>(), EdgeJ = new List<int>();
    internal readonly List<double[]> Meas = new List<double[]>();
    internal readonly List<double> Base = new List<double>();
    internal int LoopEdges;
    internal PoseGraph(int n) { N = n; Pose = new double[3 * n]; }
    internal void Add(int i, int j, double dx, double dy, double dth, double weight, bool loop)
    { EdgeI.Add(i); EdgeJ.Add(j); Meas.Add(new[] { dx, dy, dth }); Base.Add(weight); if (loop) LoopEdges++; }
    internal void Set(int i, double x, double y, double th) { Pose[3 * i] = x; Pose[3 * i + 1] = y; Pose[3 * i + 2] = th; }
    internal static double Wrap(double a) { while (a > Math.PI) a -= 2.0 * Math.PI; while (a < -Math.PI) a += 2.0 * Math.PI; return a; }
    internal double[] Residual(int e)
    {
        int i = EdgeI[e], j = EdgeJ[e]; double a = Pose[3 * i + 2], c = Math.Cos(a), s = Math.Sin(a);
        double dx = Pose[3 * j] - Pose[3 * i], dy = Pose[3 * j + 1] - Pose[3 * i + 1]; double[] z = Meas[e];
        return new[] { c * dx + s * dy - z[0], -s * dx + c * dy - z[1], Wrap(Pose[3 * j + 2] - a - z[2]) };
    }
    /** 3x6 Jacobian block, columns (x_i, y_i, th_i, x_j, y_j, th_j). */
    private double[,] Jacobian(int e)
    {
        int i = EdgeI[e], j = EdgeJ[e]; double a = Pose[3 * i + 2], c = Math.Cos(a), s = Math.Sin(a);
        double dx = Pose[3 * j] - Pose[3 * i], dy = Pose[3 * j + 1] - Pose[3 * i + 1];
        return new double[,]
        {
            { -c, -s, -s * dx + c * dy, c, s, 0.0 }, { s, -c, -c * dx - s * dy, -s, c, 0.0 }, { 0.0, 0.0, -1.0, 0.0, 0.0, 1.0 }
        };
    }
    private double Weight(int e, bool robust)
    {
        double norm = Math.Sqrt(Squared(Residual(e)));
        return Base[e] * (robust ? Math.Min(1.0, 0.35 / Math.Max(norm, 1e-12)) : 1.0);
    }
    private static double Squared(double[] r) => r[0] * r[0] + r[1] * r[1] + (0.3 * r[2]) * (0.3 * r[2]);
    internal double Cost(bool robust)
    {
        double c = 0;
        for (int e = 0; e < EdgeI.Count; e++) c += Weight(e, robust) * Squared(Residual(e));
        return c;
    }
    /** Levenberg-Marquardt: returns the cost history, last entry the final robust cost. */
    internal double[] Optimize(int iterations, bool robust)
    {
        var history = new double[iterations + 1];
        double cost = Cost(robust); history[0] = cost; double lambda = 1e-3;
        for (int it = 0; it < iterations; it++)
        {
            var h = new double[3 * N, 3 * N]; var b = new double[3 * N];
            for (int e = 0; e < EdgeI.Count; e++)
            {
                double[,] jac = Jacobian(e); double[] r = Residual(e); double w = Weight(e, robust);
                int[] idx = { 3 * EdgeI[e], 3 * EdgeI[e] + 1, 3 * EdgeI[e] + 2, 3 * EdgeJ[e], 3 * EdgeJ[e] + 1, 3 * EdgeJ[e] + 2 };
                for (int a = 0; a < 6; a++)
                {
                    for (int k = 0; k < 3; k++) b[idx[a]] += w * jac[k, a] * r[k];
                    for (int c = 0; c < 6; c++)
                    { double v = 0; for (int k = 0; k < 3; k++) v += jac[k, a] * jac[k, c]; h[idx[a], idx[c]] += w * v; }
                }
            }
            for (int d = 0; d < 3 * N; d++) h[d, d] += lambda * (1.0 + h[d, d]);
            for (int d = 0; d < 3; d++) h[d, d] += 1e9;                        // pin pose 0 (gauge freedom)
            double[] delta;
            try { delta = Solve(h, b); } catch (Exception) { break; }
            var backup = (double[])Pose.Clone();
            for (int d = 0; d < 3 * N; d++) Pose[d] -= delta[d];
            double next = Cost(robust);
            if (next < cost) { cost = next; lambda = Math.Max(1e-12, lambda * 0.5); } else { Array.Copy(backup, Pose, Pose.Length); lambda *= 4.0; }
            history[it + 1] = cost;
        }
        return history;
    }
    internal double Ate(double[] truth)
    {
        double sum = 0;
        for (int i = 0; i < N; i++) { double dx = Pose[3 * i] - truth[3 * i], dy = Pose[3 * i + 1] - truth[3 * i + 1]; sum += dx * dx + dy * dy; }
        return Math.Sqrt(sum / N);
    }
    /** Gaussian elimination with partial pivoting (the normal matrix is symmetric positive definite here). */
    internal static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length; var x = (double[])b.Clone();
        for (int c = 0; c < n; c++)
        {
            int p = c; for (int r = c + 1; r < n; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[p, c])) p = r;
            if (Math.Abs(a[p, c]) < 1e-15) throw new Exception("singular normal equations at " + c);
            if (p != c) { for (int j = 0; j < n; j++) { double t = a[c, j]; a[c, j] = a[p, j]; a[p, j] = t; } double tb = x[c]; x[c] = x[p]; x[p] = tb; }
            for (int r = c + 1; r < n; r++)
            { double f = a[r, c] / a[c, c]; x[r] -= f * x[c]; for (int j = c; j < n; j++) a[r, j] -= f * a[c, j]; }
        }
        for (int r = n - 1; r >= 0; r--) { double s = x[r]; for (int c = r + 1; c < n; c++) s -= a[r, c] * x[c]; x[r] = s / a[r, r]; }
        return x;
    }
}

/** Synthetic trajectory, landmark signatures and the appearance-based loop closure detector. */
internal sealed class FrontEnd
{
    internal readonly int Poses;
    internal readonly double[] Truth, Odo;
    internal readonly List<int>[] Signature;
    internal readonly PoseGraph Graph;
    internal FrontEnd(int poses, double step, bool rectangular, double sigma, out ulong state)
    {
        state = rectangular ? 920UL : 921UL;
        Poses = poses;
        Truth = new double[3 * poses]; Odo = new double[3 * poses];
        Signature = new List<int>[poses];
        Graph = new PoseGraph(poses);
        double x = 0, y = 0, th = 0;
        for (int k = 0; k < poses; k++)
        {
            Truth[3 * k] = x; Truth[3 * k + 1] = y; Truth[3 * k + 2] = th;
            if (rectangular) th = (4 * k / poses) * Math.PI / 2.0;                       // 4 equal legs of the loop
            x += step * Math.Cos(th); y += step * Math.Sin(th);
        }
        double ox = 0, oy = 0, oth = 0;
        Odo[0] = Odo[1] = Odo[2] = 0;
        for (int k = 1; k < poses; k++)
        {
            // The odometry measurement is the relative motion expressed in the *previous* body frame.
            double c0 = Math.Cos(Truth[3 * (k - 1) + 2]), s0 = Math.Sin(Truth[3 * (k - 1) + 2]);
            double gx = c0 * (Truth[3 * k] - Truth[3 * (k - 1)]) + s0 * (Truth[3 * k + 1] - Truth[3 * (k - 1) + 1]);
            double gy = -s0 * (Truth[3 * k] - Truth[3 * (k - 1)]) + c0 * (Truth[3 * k + 1] - Truth[3 * (k - 1) + 1]);
            double gth = PoseGraph.Wrap(Truth[3 * k + 2] - Truth[3 * (k - 1) + 2]);
            double mx = 1.01 * gx + sigma * Rng.Gaussian(ref state);                      // scale bias + noise drift
            double my = 1.01 * gy + sigma * Rng.Gaussian(ref state);
            double mth = gth + 0.002 + 0.5 * sigma * Rng.Gaussian(ref state);
            Graph.Add(k - 1, k, mx, my, mth, 1.0, loop: false);
            ox += Math.Cos(oth) * mx - Math.Sin(oth) * my; oy += Math.Sin(oth) * mx + Math.Cos(oth) * my; oth += mth;
            Odo[3 * k] = ox; Odo[3 * k + 1] = oy; Odo[3 * k + 2] = oth;
        }
        for (int k = 0; k < poses; k++) Signature[k] = Observe(Truth, k);
        Detect(state);
    }
    /** Landmark ids within 0.8 m of pose k; landmarks sit on a 6x6 grid covering the whole loop. */
    private static List<int> Observe(double[] truth, int k)
    {
        var ids = new List<int>();
        for (int gx = 0; gx < 6; gx++) for (int gy = 0; gy < 6; gy++)
        {
            double lx = -0.4 + 0.56 * gx, ly = -0.4 + 0.56 * gy;
            double dx = truth[3 * k] - lx, dy = truth[3 * k + 1] - ly;
            if (dx * dx + dy * dy <= 0.64) ids.Add(gx * 6 + gy);
        }
        return ids;
    }
    /** Appearance-based loop closure: >= 2 shared landmark ids with a pose at least 24 steps older. */
    private void Detect(ulong state)
    {
        var rng = state ^ 0x51ED270BUL;
        for (int k = 24; k < Poses; k++)
            for (int j = 0; j + 24 <= k; j++)
            {
                int shared = 0;
                foreach (int id in Signature[k]) if (Signature[j].Contains(id)) shared++;
                if (shared < 2) continue;
                double dx = Truth[3 * k] - Truth[3 * j], dy = Truth[3 * k + 1] - Truth[3 * j + 1];
                double c = Math.Cos(Truth[3 * j + 2]), s = Math.Sin(Truth[3 * j + 2]);
                Graph.Add(j, k, c * dx + s * dy + 0.02 * Rng.Gaussian(ref rng), -s * dx + c * dy + 0.02 * Rng.Gaussian(ref rng), PoseGraph.Wrap(Truth[3 * k + 2] - Truth[3 * j + 2]) + 0.01 * Rng.Gaussian(ref rng), 1.0, loop: true);
            }
    }
    internal void InitializeFromOdometry() { for (int i = 0; i < Poses; i++) Graph.Set(i, Odo[3 * i], Odo[3 * i + 1], Odo[3 * i + 2]); }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    public static void Main()
    {
        // 1. end-to-end run: dead reckoning, then the optimized pose graph (40 poses around a closed rectangle)
        var slam = new FrontEnd(40, 0.2, rectangular: true, sigma: 0.02, out ulong seed);
        slam.InitializeFromOdometry();
        double ateOdo = slam.Graph.Ate(slam.Truth);
        Check(slam.Graph.LoopEdges >= 1, "the revisit is detected as a loop closure (" + slam.Graph.LoopEdges + " loop edges)");
        double costBefore = slam.Graph.Cost(false);
        double[] history = slam.Graph.Optimize(12, robust: true);
        double costAfter = slam.Graph.Cost(false);
        double ateOpt = slam.Graph.Ate(slam.Truth);
        Check(costAfter < 0.25 * costBefore, "robust cost dropped from " + costBefore.ToString("F4") + " to " + costAfter.ToString("F4"));
        Check(history[history.Length - 1] <= history[0], "the LM cost history never ends above its start");
        Check(ateOpt < 0.5 * ateOdo, "ATE improved from " + ateOdo.ToString("F4") + " m (dead reckoning) to " + ateOpt.ToString("F4") + " m");

        // 2. invariants of the optimized graph: pose count, pinned gauge, finite poses
        bool finite = true, pinned = Math.Abs(slam.Graph.Pose[0]) < 1e-9 && Math.Abs(slam.Graph.Pose[1]) < 1e-9;
        for (int d = 0; d < slam.Graph.Pose.Length; d++) finite &= !double.IsNaN(slam.Graph.Pose[d]) && Math.Abs(slam.Graph.Pose[d]) < 100.0;
        Check(pinned && finite && slam.Graph.N == slam.Poses, "pose 0 stays pinned, every pose is finite and the pose count is preserved");

        // 3. closed-form boundary: a two-pose graph must reproduce its single measurement exactly
        var pair = new PoseGraph(2);
        pair.Add(0, 1, 0.3, 0.1, 0.2, 1.0, loop: false);
        pair.Set(1, 0.99, 0.99, 0.9);
        pair.Optimize(8, robust: false);
        double[] pairResidual = pair.Residual(0);
        Check(Math.Abs(pairResidual[0]) < 1e-9 && Math.Abs(pairResidual[1]) < 1e-9 && Math.Abs(pairResidual[2]) < 1e-9, "a two-pose graph collapses onto its measurement (residual " + pairResidual[0].ToString("E1") + ")");

        // 4. negative control: an open (non-revisiting) path must not produce loop closures
        var open = new FrontEnd(40, 0.2, rectangular: false, sigma: 0.02, out _);
        Check(open.Graph.LoopEdges == 0, "an open path yields no loop closures (" + open.Graph.LoopEdges + ")");
        open.InitializeFromOdometry();
        double openBefore = open.Graph.Cost(false); open.Graph.Optimize(6, robust: false);
        Check(open.Graph.Cost(false) < openBefore, "without closures the optimizer still reduces odometry inconsistency");

        // 5. one gross outlier closure: Huber weighting must beat plain least squares
        var robustRun = new FrontEnd(40, 0.2, rectangular: true, sigma: 0.02, out _);
        robustRun.InitializeFromOdometry();
        robustRun.Graph.Add(0, robustRun.Poses - 1, 1.40, 0.55, 0.9, 1.0, loop: true);      // wrong by ~1.5 m
        var plainRun = new FrontEnd(40, 0.2, rectangular: true, sigma: 0.02, out _);
        plainRun.InitializeFromOdometry();
        plainRun.Graph.Add(0, plainRun.Poses - 1, 1.40, 0.55, 0.9, 1.0, loop: true);
        robustRun.Graph.Optimize(12, robust: true);
        plainRun.Graph.Optimize(12, robust: false);
        double robustAte = robustRun.Graph.Ate(robustRun.Truth), plainAte = plainRun.Graph.Ate(plainRun.Truth);
        Check(robustAte < plainAte, "Huber keeps ATE " + robustAte.ToString("F4") + " m below plain least squares " + plainAte.ToString("F4") + " m");

        // 6. reproducibility
        var again = new FrontEnd(40, 0.2, rectangular: true, sigma: 0.02, out ulong secondSeed);
        Check(secondSeed > 0 && Math.Abs(again.Truth[3 * 39] - slam.Truth[3 * 39]) < 1e-12, "the same seeds rebuild the same trajectory");
        again.InitializeFromOdometry(); again.Graph.Optimize(12, robust: true);
        bool identical = true;
        for (int d = 0; d < again.Graph.Pose.Length && identical; d++)
            identical = BitConverter.DoubleToInt64Bits(again.Graph.Pose[d]) == BitConverter.DoubleToInt64Bits(slam.Graph.Pose[d]);
        Check(identical, "the optimized map is bit-for-bit reproducible");

        Console.WriteLine("SLAM: 40 poses, " + slam.Graph.EdgeI.Count + " edges (" + slam.Graph.LoopEdges + " loop closures), robust cost " + costBefore.ToString("F4") + " -> " + costAfter.ToString("F4"));
        Console.WriteLine("ATE: dead reckoning " + ateOdo.ToString("F4") + " m -> optimized " + ateOpt.ToString("F4") + " m; final pose (" + slam.Graph.Pose[3 * 39].ToString("F3") + ", " + slam.Graph.Pose[3 * 39 + 1].ToString("F3") + ") vs truth (" + slam.Truth[3 * 39].ToString("F3") + ", " + slam.Truth[3 * 39 + 1].ToString("F3") + ")");
        Console.WriteLine("PASS");
    }
}
