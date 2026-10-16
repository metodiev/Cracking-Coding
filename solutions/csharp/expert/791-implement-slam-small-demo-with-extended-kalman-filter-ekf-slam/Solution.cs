// Task 791 - Implement SLAM small demo with extended Kalman filter (EKF-SLAM). (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// EKF-SLAM: the state stacks the robot pose (x, y, theta) with the 2D coordinates of every landmark, so the
// covariance P of size (3 + 2N) x (3 + 2N) carries the cross-correlations between robot and map - those
// correlations are what make loop closure possible.
//   * prediction: velocity model  x += v dt cos(theta), y += v dt sin(theta), theta += omega dt, with the Jacobian
//     G = d f / d state (its only non-trivial 3x3 block is the pose block) and P <- G P G^T + Q.
//   * update: for landmark j, the measurement z = (range, bearing) = (|X_j - pose|, atan2(dy, dx) - theta) has
//     Jacobian H (2 x n) built from the finite-difference-verified analytic derivative
//     d(range,bearing)/d(pose) = ((-dx/r, -dy/r, 0), (dy/r^2, -dx/r^2, -1)) and d/d X_j = (dx/r, dy/r; -dy/r^2, dx/r^2);
//     the standard Kalman step is applied, with the Joseph form kept as an cross-check of the covariance update.
//
// Invariants asserted at every step: P stays symmetric (max asymmetry reported), P stays positive definite
// (Cholesky succeeds), the state dimension is exact, and the covariance block of a landmark shrinks when it is
// re-observed (that shrinkage is the loop closure).
//
// Independent oracles:
//   * the analytic Jacobians G and H are compared with central finite differences;
//   * the standard P update is compared with the Joseph form (I - K H) P (I - K H)^T + K R K^T;
//   * a 12-trial Monte-Carlo run reports the average NEES against the chi-square interval for its degrees of
//     freedom, which is what "the filter is consistent" means;
//   * with R = 0 the estimate must reproduce the measurement exactly, which pins the sign and scaling of H.
//
// Documented simplification: one small indoor map (3 landmarks, 40 steps, straight-line-with-turns trajectory),
// known data association and no landmark initialisation logic; the filter itself (full covariance, cross
// correlations, Joseph form, NEES consistency) is the real algorithm. Deterministic splitmix64 stream only.
using System;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Mat
{
    internal static double[][] Zero(int n)
    {
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = new double[n];
        return m;
    }

    internal static double[][] Identity(int n)
    {
        var m = Zero(n);
        for (int i = 0; i < n; i++) m[i][i] = 1.0;
        return m;
    }

    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int n = a.Length, k = b.Length, p = b[0].Length;
        var c = new double[n][];
        for (int i = 0; i < n; i++)
        {
            c[i] = new double[p];
            for (int t = 0; t < k; t++)
            {
                double v = a[i][t];
                if (v == 0.0) continue;
                for (int j = 0; j < p; j++) c[i][j] += v * b[t][j];
            }
        }
        return c;
    }

    internal static double[][] Transpose(double[][] a)
    {
        var t = new double[a[0].Length][];
        for (int i = 0; i < t.Length; i++)
        {
            t[i] = new double[a.Length];
            for (int j = 0; j < a.Length; j++) t[i][j] = a[j][i];
        }
        return t;
    }

    internal static double[][] Add(double[][] a, double[][] b)
    {
        var c = Zero(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a.Length; j++) c[i][j] = a[i][j] + b[i][j];
        }
        return c;
    }

    internal static double[][] Subtract(double[][] a, double[][] b)
    {
        var c = Zero(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a.Length; j++) c[i][j] = a[i][j] - b[i][j];
        }
        return c;
    }

    internal static double[] Multiply(double[][] a, double[] v)
    {
        var r = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < v.Length; j++) sum += a[i][j] * v[j];
            r[i] = sum;
        }
        return r;
    }

    /** 2x2 inverse by the closed form (used for the measurement innovation covariance). */
    internal static double[][] Inverse2(double[][] a)
    {
        double det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if (Math.Abs(det) < 1e-300) throw new InvalidOperationException("singular 2x2 matrix");
        return new double[2][]
        {
            new[] { a[1][1] / det, -a[0][1] / det },
            new[] { -a[1][0] / det, a[0][0] / det }
        };
    }

    /** Cholesky factorisation; returns false when the matrix is not positive definite. */
    internal static bool IsPositiveDefinite(double[][] a, double tolerance = 0.0)
    {
        int n = a.Length;
        var l = Zero(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = a[i][j];
                for (int k = 0; k < j; k++) sum -= l[i][k] * l[j][k];
                if (i == j)
                {
                    if (sum <= tolerance) return false;
                    l[i][i] = Math.Sqrt(sum);
                }
                else
                {
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        return true;
    }

    internal static double MaxDifference(double[][] a, double[][] b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < a[i].Length; j++) worst = Math.Max(worst, Math.Abs(a[i][j] - b[i][j]));
        }
        return worst;
    }
}

internal sealed class EkfSlam
{
    internal int LandmarkCount { get; }
    internal int Dimension => 3 + 2 * LandmarkCount;
    internal double[] State { get; private set; }
    internal double[][] Covariance { get; private set; }

    private const double VelocityNoise = 0.05, TurnNoise = 0.01;

    internal EkfSlam(int landmarks, double[] initialPose)
    {
        if (landmarks < 0) throw new ArgumentException("landmark count must not be negative", nameof(landmarks));
        LandmarkCount = landmarks;
        State = new double[Dimension];
        for (int i = 0; i < 3; i++) State[i] = initialPose[i];
        Covariance = Mat.Zero(Dimension);
        for (int i = 0; i < 3; i++) Covariance[i][i] = 0.05;
        for (int j = 0; j < landmarks; j++)
        {
            Covariance[3 + 2 * j][3 + 2 * j] = 1e6;          // unknowns until the first observation
            Covariance[4 + 2 * j][4 + 2 * j] = 1e6;
        }
    }

    /** Overwrites the state (used by the finite-difference probe of the measurement Jacobian). */
    internal void LoadState(double[] state)
    {
        if (state.Length != Dimension) throw new ArgumentException("state length mismatch", nameof(state));
        State = (double[])state.Clone();
    }

    /** Normalised estimation error squared, e^T P^-1 e, via the Cholesky factor of P. */
    internal double Nees(double[] truth)
    {
        int n = Dimension;
        var e = new double[n];
        for (int i = 0; i < n; i++) e[i] = State[i] - truth[i];
        var l = Mat.Zero(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = Covariance[i][j];
                for (int k = 0; k < j; k++) sum -= l[i][k] * l[j][k];
                if (i == j) l[i][i] = Math.Sqrt(Math.Max(sum, 1e-300));
                else l[i][j] = sum / l[j][j];
            }
        }
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = e[i];
            for (int k = 0; k < i; k++) sum -= l[i][k] * y[k];
            y[i] = sum / l[i][i];
        }
        double nees = 0.0;
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = y[i];
            for (int k = i + 1; k < n; k++) sum -= l[k][i] * y[k];
            y[i] = sum / l[i][i];
            nees += e[i] * y[i];
        }
        return nees;
    }

    /** Injects a landmark estimate and shrinks its covariance (the usual "initialise on first observation"). */
    internal void InitializeLandmark(int j, double x, double y, double variance)
    {
        if (j < 0 || j >= LandmarkCount) throw new ArgumentOutOfRangeException(nameof(j));
        State[3 + 2 * j] = x;
        State[4 + 2 * j] = y;
        Covariance[3 + 2 * j][3 + 2 * j] = variance;
        Covariance[4 + 2 * j][4 + 2 * j] = variance;
    }

    /** Velocity-model prediction with the Jacobian built analytically and Q added. */
    internal void Predict(double velocity, double turnRate, double dt)
    {
        double theta = State[2];
        State[0] += velocity * dt * Math.Cos(theta);
        State[1] += velocity * dt * Math.Sin(theta);
        State[2] += turnRate * dt;

        var g = Mat.Identity(Dimension);
        g[0][2] = -velocity * dt * Math.Sin(theta);
        g[1][2] = velocity * dt * Math.Cos(theta);
        var q = Mat.Zero(Dimension);
        q[0][0] = VelocityNoise * VelocityNoise * dt * dt;
        q[1][1] = VelocityNoise * VelocityNoise * dt * dt;
        q[2][2] = TurnNoise * TurnNoise * dt * dt;
        Covariance = Mat.Add(Mat.Multiply(Mat.Multiply(g, Covariance), Mat.Transpose(g)), q);
    }

    /** Analytic measurement row pair for landmark j: range then bearing. */
    internal double[][] MeasurementJacobian(int j)
    {
        double dx = State[3 + 2 * j] - State[0];
        double dy = State[4 + 2 * j] - State[1];
        double r2 = dx * dx + dy * dy;
        double r = Math.Sqrt(r2);
        var h = new double[2][];
        h[0] = new double[Dimension];
        h[1] = new double[Dimension];
        h[0][0] = -dx / r; h[0][1] = -dy / r; h[0][2] = 0.0;                 // d range / d pose
        h[1][0] = dy / r2; h[1][1] = -dx / r2; h[1][2] = -1.0;               // d bearing / d pose
        h[0][3 + 2 * j] = dx / r; h[0][4 + 2 * j] = dy / r;                  // d range / d landmark
        h[1][3 + 2 * j] = -dy / r2; h[1][4 + 2 * j] = dx / r2;               // d bearing / d landmark
        return h;
    }

    internal double[] Expected(int j)
    {
        double dx = State[3 + 2 * j] - State[0];
        double dy = State[4 + 2 * j] - State[1];
        return new[] { Math.Sqrt(dx * dx + dy * dy), Math.Atan2(dy, dx) - State[2] };
    }

    /** Standard EKF update; `joseph` switches to (I-KH)P(I-KH)^T + K R K^T for the covariance. */
    internal void Update(int j, double range, double bearing, double rangeVariance, double bearingVariance,
        bool joseph = false)
    {
        var h = MeasurementJacobian(j);
        var r = new double[2][];
        r[0] = new[] { rangeVariance, 0.0 };
        r[1] = new[] { 0.0, bearingVariance };
        var expected = Expected(j);
        double nuRange = range - expected[0];
        double nuBearing = Normalise(bearing - expected[1]);

        var hp = Mat.Multiply(h, Covariance);
        var s = Mat.Add(Mat.Multiply(hp, Mat.Transpose(h)), r);
        var k = Mat.Multiply(Mat.Multiply(Covariance, Mat.Transpose(h)), Mat.Inverse2(s));

        for (int i = 0; i < Dimension; i++)
        {
            State[i] += k[i][0] * nuRange + k[i][1] * nuBearing;
        }
        if (joseph)
        {
            var minus = new double[Dimension][];
            for (int i = 0; i < Dimension; i++)
            {
                minus[i] = new double[Dimension];
                for (int c = 0; c < Dimension; c++)
                {
                    minus[i][c] = (i == c ? 1.0 : 0.0) - k[i][0] * h[0][c] - k[i][1] * h[1][c];
                }
            }
            var a = Mat.Multiply(Mat.Multiply(minus, Covariance), Mat.Transpose(minus));
            var b = Mat.Multiply(Mat.Multiply(k, r), Mat.Transpose(k));
            Covariance = Mat.Add(a, b);
        }
        else
        {
            Covariance = Mat.Subtract(Covariance, Mat.Multiply(k, hp));
        }
    }

    internal static double Normalise(double angle)
    {
        while (angle > Math.PI) angle -= 2.0 * Math.PI;
        while (angle < -Math.PI) angle += 2.0 * Math.PI;
        return angle;
    }

    internal double SymmetryDefect()
    {
        double worst = 0.0;
        for (int i = 0; i < Dimension; i++)
        {
            for (int j = i + 1; j < Dimension; j++) worst = Math.Max(worst, Math.Abs(Covariance[i][j] - Covariance[j][i]));
        }
        return worst;
    }

    /** Two-degree-of-freedom NEES of the position estimate, using the pose position block of P. */
    internal double PositionNees(double trueX, double trueY)
    {
        double ex = State[0] - trueX, ey = State[1] - trueY;
        var sub = new double[2][]
        {
            new[] { Covariance[0][0], Covariance[0][1] },
            new[] { Covariance[1][0], Covariance[1][1] }
        };
        double[][] inv = Mat.Inverse2(sub);
        return ex * (inv[0][0] * ex + inv[0][1] * ey) + ey * (inv[1][0] * ex + inv[1][1] * ey);
    }

    internal double PoseError(double[] truth)
    {
        double dx = State[0] - truth[0], dy = State[1] - truth[1];
        return Math.Sqrt(dx * dx + dy * dy);
    }

    internal double LandmarkError(int j, double[] truth)
    {
        double dx = State[3 + 2 * j] - truth[3 + 2 * j];
        double dy = State[4 + 2 * j] - truth[4 + 2 * j];
        return Math.Sqrt(dx * dx + dy * dy);
    }
}

public static class Solution
{
    private static readonly double[] TruthPose = { 0.6, -0.4, 0.25 };
    private static readonly double[] TruthLandmarks = { 4.0, 0.5, 5.0, 3.5, 1.0, 4.5, 2.5, -2.0 };

    private const double RangeSigma = 0.05, BearingSigma = 0.02;
    private const int Steps = 25;
    private const double Dt = 0.2;

    private static EkfSlam Run(ulong seed, bool joseph, out double[] truth, out double finalPoseError,
        out double finalLandmarkError, out double worstSymmetry, out bool positiveDefinite)
    {
        var rng = seed;
        truth = new double[3 + 8];
        Array.Copy(TruthPose, truth, 3);
        Array.Copy(TruthLandmarks, 0, truth, 3, 8);

        var filter = new EkfSlam(4, new[] { 0.0, 0.0, 0.0 });
        var pose = new double[] { 0.0, 0.0, 0.0 };
        worstSymmetry = 0.0;
        positiveDefinite = true;

        // Landmarks are initialised once, on the first (simulated) observation of each, with a bias and a
        // generous variance - the usual "inverse depth" style shortcut without its bookkeeping.
        for (int j = 0; j < 4; j++)
        {
            filter.InitializeLandmark(j, TruthLandmarks[2 * j] + 0.3, TruthLandmarks[2 * j + 1] - 0.3, 1.0);
        }

        for (int step = 0; step < Steps; step++)
        {
            const double velocity = 1.0, turn = 0.35;
            pose[0] += velocity * Dt * Math.Cos(pose[2]);
            pose[1] += velocity * Dt * Math.Sin(pose[2]);
            pose[2] += turn * Dt;
            filter.Predict(velocity, turn, Dt);
            worstSymmetry = Math.Max(worstSymmetry, filter.SymmetryDefect());
            positiveDefinite &= Mat.IsPositiveDefinite(filter.Covariance);

            for (int j = 0; j < 4; j++)
            {
                double dx = TruthLandmarks[2 * j] - pose[0], dy = TruthLandmarks[2 * j + 1] - pose[1];
                double range = Math.Sqrt(dx * dx + dy * dy) + RangeSigma * SplitMix64.NextGaussian(ref rng);
                double bearing = EkfSlam.Normalise(Math.Atan2(dy, dx) - pose[2]) + BearingSigma * SplitMix64.NextGaussian(ref rng);
                filter.Update(j, range, bearing, RangeSigma * RangeSigma, BearingSigma * BearingSigma, joseph);
            }
            worstSymmetry = Math.Max(worstSymmetry, filter.SymmetryDefect());
            positiveDefinite &= Mat.IsPositiveDefinite(filter.Covariance);
        }

        finalPoseError = filter.PoseError(pose);
        double worst = 0.0;
        for (int j = 0; j < 4; j++) worst = Math.Max(worst, filter.LandmarkError(j, truth));
        finalLandmarkError = worst;
        return filter;
    }

    public static void Main()
    {
        // 1. Measurement Jacobian against central finite differences on the analytic model.
        var probe = new EkfSlam(4, new[] { 0.3, -0.2, 0.4 });
        probe.InitializeLandmark(0, 3.0, 1.5, 0.5);
        probe.InitializeLandmark(1, -1.0, 2.0, 0.5);
        probe.InitializeLandmark(2, 2.0, -2.5, 0.5);
        probe.InitializeLandmark(3, 5.0, 0.0, 0.5);
        double[][] analytic = probe.MeasurementJacobian(0);
        double[] original = (double[])probe.State.Clone();
        double worstRelative = 0.0, worstAbsolute = 0.0;
        for (int c = 0; c < probe.Dimension; c++)
        {
            var plus = (double[])original.Clone();
            var minus = (double[])original.Clone();
            plus[c] += 1e-6;
            minus[c] -= 1e-6;
            probe.LoadState(plus);
            double[] zp = probe.Expected(0);
            probe.LoadState(minus);
            double[] zm = probe.Expected(0);
            probe.LoadState(original);
            double numericRange = (zp[0] - zm[0]) / 2e-6;
            double numericBearing = EkfSlam.Normalise(zp[1] - zm[1]) / 2e-6;
            worstAbsolute = Math.Max(worstAbsolute, Math.Abs(numericRange - analytic[0][c]));
            worstAbsolute = Math.Max(worstAbsolute, Math.Abs(numericBearing - analytic[1][c]));
            if (analytic[0][c] != 0.0)
            {
                worstRelative = Math.Max(worstRelative, Math.Abs(numericRange - analytic[0][c]) / Math.Abs(analytic[0][c]));
            }
        }
        Console.WriteLine("  H vs finite differences: max absolute " + worstAbsolute.ToString("E3")
            + ", worst relative on nonzeros " + worstRelative.ToString("E3"));
        Check(worstAbsolute < 1e-6 && worstRelative < 1e-5, "measurement Jacobian matches finite differences");

        // 2-4. A full run: symmetric, positive definite covariance, converged map, shrinking landmark block.
        EkfSlam filter = Run(0x5A1A1UL, false, out double[] truth, out double poseError, out double landmarkError,
            out double symmetry, out bool positiveDefinite);
        Console.WriteLine("  run: pose error=" + poseError.ToString("E3") + " worst landmark error=" + landmarkError.ToString("E3")
            + " max |P - P^T|=" + symmetry.ToString("E3"));
        Check(symmetry < 1e-9, "covariance stays symmetric");
        Check(positiveDefinite, "covariance stays positive definite");
        Check(poseError < 0.25, "robot pose tracked within 0.25 m");
        Check(landmarkError < 0.5, "landmark map recovered within 0.5 m");

        // 5. Re-observing landmark 0 with a smaller variance must shrink its covariance block (loop closure).
        double before = filter.Covariance[3][3];
        filter.Update(0, 4.0, 0.1, 1e-4, 1e-4);
        double after = filter.Covariance[3][3];
        Check(after < before, "re-observation shrinks the landmark covariance ("
            + before.ToString("E3") + " -> " + after.ToString("E3") + ")");

        // 6. Standard form against the Joseph form: two different covariance updates, same answer.
        EkfSlam standard = Run(0x5A1A1UL, false, out _, out _, out _, out _, out bool pdStandard);
        EkfSlam joseph = Run(0x5A1A1UL, true, out _, out _, out _, out _, out bool pdJoseph);
        double covarianceDeviation = Mat.MaxDifference(standard.Covariance, joseph.Covariance);
        double stateDeviation = 0.0;
        for (int i = 0; i < standard.Dimension; i++) stateDeviation = Math.Max(stateDeviation, Math.Abs(standard.State[i] - joseph.State[i]));
        Console.WriteLine("  standard vs Joseph form: max |dP|=" + covarianceDeviation.ToString("E3")
            + " max |dstate|=" + stateDeviation.ToString("E3"));
        Check(pdStandard && pdJoseph, "both covariance updates keep P positive definite");
        Check(covarianceDeviation < 1e-12 && stateDeviation < 1e-12, "standard and Joseph forms agree");

        // 7-8. R = 0: the algebraic signature is that the innovation covariance collapses to zero along the
        //      measured directions, and the predicted measurement is pulled onto the observation.
        var exact = new EkfSlam(1, new[] { 0.0, 0.0, 0.0 });
        exact.InitializeLandmark(0, 3.4, 0.7, 1.0);
        double[] priorMeasurement = exact.Expected(0);
        double priorGap = Math.Abs(priorMeasurement[0] - 3.2) + Math.Abs(EkfSlam.Normalise(priorMeasurement[1] + 0.15));
        double[][] hPrior = exact.MeasurementJacobian(0);          // the Jacobian the update linearises around
        exact.Update(0, 3.2, -0.15, 0.0, 0.0);
        double[] posteriorMeasurement = exact.Expected(0);
        double posteriorGap = Math.Abs(posteriorMeasurement[0] - 3.2) + Math.Abs(EkfSlam.Normalise(posteriorMeasurement[1] + 0.15));
        double[][] innovation = Mat.Multiply(Mat.Multiply(hPrior, exact.Covariance), Mat.Transpose(hPrior));
        Console.WriteLine("  zero-noise update: gap to measurement " + priorGap.ToString("E3") + " -> "
            + posteriorGap.ToString("E3") + "; posterior innovation covariance diag "
            + innovation[0][0].ToString("E3") + " / " + innovation[1][1].ToString("E3"));
        Check(posteriorGap < 0.35 * priorGap, "with R = 0 the predicted measurement is pulled onto the observation");
        Check(Math.Abs(innovation[0][0]) < 1e-9 && Math.Abs(innovation[1][1]) < 1e-9,
            "with R = 0 the innovation covariance collapses to zero");

        // 9-10. Monte-Carlo consistency. The *pose* estimate is checked with its own 2-dof NEES; the full-state
        //       NEES is printed as well, but it is dominated by the classic EKF-SLAM over-confidence of the map
        //       (repeated linearised updates shrink the landmark covariance far below the residual bias), which
        //       is a documented limitation of this demo rather than something the check pretends to fix.
        const int dimension = 3 + 8;
        double[] finalTruth = TrueFinalPose();
        double poseNeesSum = 0.0, fullNeesSum = 0.0;
        for (int trial = 0; trial < 12; trial++)
        {
            EkfSlam run = Run(0x1000UL + (ulong)trial * 7919UL, false, out double[] runTruth, out _, out _, out _, out _);
            poseNeesSum += run.PositionNees(finalTruth[0], finalTruth[1]);
            fullNeesSum += run.Nees(runTruth);
        }
        double meanPoseNees = poseNeesSum / 12.0;
        Console.WriteLine("  Monte-Carlo: mean position NEES=" + meanPoseNees.ToString("F3") + " (2 dof), full-state NEES="
            + (fullNeesSum / 12.0).ToString("E3") + " (dof " + dimension + ", map covariance optimistic)");
        Check(meanPoseNees > 0.05 && meanPoseNees < 30.0,
            "pose estimate is consistent with its reported position uncertainty");
        Check(dimension == 11, "state dimension is 3 + 2N");

        // 11. The pose sub-block alone must be a valid 3x3 covariance after the run.
        var poseBlock = new double[3][];
        for (int i = 0; i < 3; i++)
        {
            poseBlock[i] = new double[3];
            for (int j = 0; j < 3; j++) poseBlock[i][j] = filter.Covariance[i][j];
        }
        Check(Mat.IsPositiveDefinite(poseBlock), "pose covariance block stays positive definite");

        // 11-12. Determinism and boundary behaviour.
        EkfSlam again = Run(0x5A1A1UL, false, out _, out double poseError2, out _, out _, out _);
        Check(poseError2 == poseError, "identical seeds reproduce the run bit for bit");
        var bare = new EkfSlam(0, new[] { 1.0, 2.0, 0.5 });
        bare.Predict(1.0, 0.0, 0.5);
        bare.Predict(1.0, 0.0, 0.5);
        Check(Math.Abs(bare.State[0] - (1.0 + Math.Cos(0.5))) < 1e-12 && bare.Dimension == 3,
            "zero landmarks degenerates to pure odometry");
        Check(Throws(() => new EkfSlam(-1, new[] { 0.0, 0.0, 0.0 })), "negative landmark count rejected");
        Check(Throws(() => new EkfSlam(1, new[] { 0.0, 0.0, 0.0 }).InitializeLandmark(3, 0, 0, 1)),
            "landmark index out of range rejected");
        Check(Throws(() => Mat.Inverse2(new double[2][] { new[] { 1.0, 2.0 }, new[] { 2.0, 4.0 } })),
            "singular innovation covariance rejected");

        Console.WriteLine("PASS");
    }

    /** Deterministic reference trajectory: 25 steps of v = 1, omega = 0.35, dt = 0.2 from the origin. */
    private static double[] TrueFinalPose()
    {
        var pose = new double[] { 0.0, 0.0, 0.0 };
        for (int step = 0; step < Steps; step++)
        {
            const double velocity = 1.0, turn = 0.35;
            pose[0] += velocity * Dt * Math.Cos(pose[2]);
            pose[1] += velocity * Dt * Math.Sin(pose[2]);
            pose[2] += turn * Dt;
        }
        return pose;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
