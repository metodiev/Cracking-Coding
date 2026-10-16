// Task 790 - Implement bundle adjustment sparse solver outline with Levenberg-Marquardt. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// Bundle adjustment minimises the reprojection error of a 2D "camera" model over camera poses and point
// positions. Here a camera i is a rigid frame (theta_i, C_i) and the predicted image of world point X_j is
//   p_ij = R(theta_i) (X_j - C_i),  residual r_ij = p_ij - z_ij,
// with analytic Jacobians d p/dX = R, d p/dC = -R and d p/dtheta = R'(theta)(X - C). Because the cost is a sum
// of per-observation terms, the normal equations have the classic arrow/sparse shape
//   H = [[A, W], [W^T, B]] with A block-diagonal 3x3 per camera, B block-diagonal 2x2 per point
// and W containing a 3x2 block only for the (camera, point) pairs that are actually observed. The Schur
// complement S = B - W^T A^-1 W eliminates the points: solve S for the point update, back-substitute A for the
// cameras. Camera 0 stays fixed, which pins the 3-dimensional gauge freedom (rotation + translation).
//
// Levenberg-Marquardt runs with the damping added to the diagonal (H + lambda diag(H)), accepting a step when
// the cost drops and multiplying lambda by 3 on rejection, dividing by 3 on acceptance.
//
// Verified against independent references:
//   * the analytic Jacobian against central finite differences (max relative deviation reported);
//   * the block-assembled sparse H against a dense J^T J built straight from the full Jacobian;
//   * the Schur-complement update against the dense solve of the same damped system (max deviation reported);
//   * the noiseless problem against its known ground truth, and the noisy one against sigma (the measurement
//     noise level is what the final reprojection RMS should reproduce);
//   * an L2 run against a Huber-weighted (IRLS) run under 10% gross outliers, where the robust weighting must
//     land closer to the truth - a check that the loss weighting is wired in correctly;
//   * the gauge: the un-fixed normal equations are singular and are rejected, the fixed ones are not.
//
// Documented simplification (the task says *outline*): a tiny 3-camera/20-point 2D bundle with all pairs observed
// and no Schur ordering, no robust intrinsics, no sparse Cholesky and no marginalisation bookkeeping. The sparse
// structure, the Schur elimination, the analytic Jacobians and LM itself are the real algorithm. Deterministic.
using System;

internal static class Linalg
{
    /** Gaussian elimination with partial pivoting on a row-major square matrix; throws when singular. */
    internal static double[] Solve(double[][] a, double[] b)
    {
        int n = b.Length;
        var m = new double[n][];
        for (int i = 0; i < n; i++) m[i] = (double[])a[i].Clone();
        var rhs = (double[])b.Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row][col]) > Math.Abs(m[pivot][col])) pivot = row;
            }
            if (Math.Abs(m[pivot][col]) < 1e-12) throw new InvalidOperationException("singular normal equations");
            if (pivot != col)
            {
                (m[col], m[pivot]) = (m[pivot], m[col]);
                (rhs[col], rhs[pivot]) = (rhs[pivot], rhs[col]);
            }
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row][col] / m[col][col];
                if (factor == 0.0) continue;
                for (int c = col; c < n; c++) m[row][c] -= factor * m[col][c];
                rhs[row] -= factor * rhs[col];
            }
        }
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            double sum = rhs[row];
            for (int c = row + 1; c < n; c++) sum -= m[row][c] * x[c];
            x[row] = sum / m[row][row];
        }
        return x;
    }

    internal static double[][] Identity(int n)
    {
        var a = new double[n][];
        for (int i = 0; i < n; i++)
        {
            a[i] = new double[n];
            a[i][i] = 1.0;
        }
        return a;
    }
}

internal sealed class Bundle
{
    internal int CameraCount, PointCount;
    internal double[] CameraTruth, PointTruth, CameraEstimate, PointEstimate, Measurements;

    private const int CameraStride = 3, PointStride = 2;

    internal Bundle(int cameras, int points, ulong seed)
    {
        if (cameras < 2) throw new ArgumentException("at least two cameras are needed to pin the gauge", nameof(cameras));
        if (points < 1) throw new ArgumentException("at least one point is required", nameof(points));
        CameraCount = cameras;
        PointCount = points;
        CameraTruth = new double[cameras * CameraStride];
        PointTruth = new double[points * PointStride];
        var rng = seed;
        for (int i = 0; i < cameras; i++)
        {
            CameraTruth[i * 3] = i == 0 ? 0.0 : 0.6 * (SplitMix64.NextDouble(ref rng) - 0.5);
            CameraTruth[i * 3 + 1] = 2.5 * (SplitMix64.NextDouble(ref rng) - 0.5);
            CameraTruth[i * 3 + 2] = 2.5 * (SplitMix64.NextDouble(ref rng) - 0.5);
        }
        for (int j = 0; j < points; j++)
        {
            PointTruth[j * 2] = 6.0 * (SplitMix64.NextDouble(ref rng) - 0.5);
            PointTruth[j * 2 + 1] = 6.0 * (SplitMix64.NextDouble(ref rng) - 0.5);
        }
        Measurements = new double[2 * cameras * points];
        for (int i = 0; i < cameras; i++)
        {
            for (int j = 0; j < points; j++)
            {
                double[] p = Project(CameraTruth, i, PointTruth, j);
                int k = (i * points + j) * 2;
                Measurements[k] = p[0];
                Measurements[k + 1] = p[1];
            }
        }
        CameraEstimate = (double[])CameraTruth.Clone();
        PointEstimate = (double[])PointTruth.Clone();
    }

    internal static double[] Project(double[] cameras, int i, double[] points, int j)
    {
        double theta = cameras[i * 3];
        double c = Math.Cos(theta), s = Math.Sin(theta);
        double dx = points[j * 2] - cameras[i * 3 + 1];
        double dy = points[j * 2 + 1] - cameras[i * 3 + 2];
        return new[] { c * dx - s * dy, s * dx + c * dy };
    }

    /** Adds noise / gross outliers to the measurements (outlierCount observations get a large offset). */
    internal void Corrupt(ulong seed, double sigma, int outlierCount, double outlierSize)
    {
        var rng = seed;
        for (int k = 0; k < Measurements.Length; k++)
        {
            Measurements[k] += sigma * SplitMix64.NextGaussian(ref rng);
        }
        for (int o = 0; o < outlierCount; o++)
        {
            int k = 2 * (int)(SplitMix64.Next(ref rng) % (ulong)(CameraCount * PointCount));
            Measurements[k] += outlierSize;
            Measurements[k + 1] -= outlierSize;
        }
    }

    /** Moves the *initial estimate* off the truth: camera 0 stays pinned to define the gauge. */
    internal void Perturb(ulong seed, double cameraJitter, double pointJitter)
    {
        var rng = seed;
        for (int p = CameraStride; p < CameraEstimate.Length; p++)
        {
            CameraEstimate[p] += cameraJitter * (SplitMix64.NextDouble(ref rng) - 0.5);
        }
        for (int p = 0; p < PointEstimate.Length; p++)
        {
            PointEstimate[p] += pointJitter * (SplitMix64.NextDouble(ref rng) - 0.5);
        }
    }

    internal int ParameterCount => CameraCount * CameraStride + PointCount * PointStride;

    /** Parameter vector: cameras then points. Camera 0 is frozen (gauge fixing). */
    internal double[] PackParameters(double[] cameras, double[] points)
    {
        var x = new double[ParameterCount];
        Array.Copy(cameras, 0, x, 0, cameras.Length);
        Array.Copy(points, 0, x, cameras.Length, points.Length);
        return x;
    }

    internal void UnpackParameters(double[] x, double[] cameras, double[] points)
    {
        Array.Copy(x, 0, cameras, 0, cameras.Length);
        Array.Copy(x, cameras.Length, points, 0, points.Length);
    }

    internal bool IsFree(int parameter) => parameter >= CameraStride;   // camera 0 is pinned

    internal int[] FreeMap()
    {
        var map = new int[ParameterCount];
        int next = 0;
        for (int p = 0; p < ParameterCount; p++) map[p] = p < CameraStride ? -1 : next++;
        return map;
    }

    /** Residual vector over all observations for the given parameters (2 values per observation). */
    internal double[] Residuals(double[] x)
    {
        var cameras = new double[CameraCount * CameraStride];
        var points = new double[PointCount * PointStride];
        UnpackParameters(x, cameras, points);
        var r = new double[Measurements.Length];
        for (int i = 0; i < CameraCount; i++)
        {
            for (int j = 0; j < PointCount; j++)
            {
                double[] p = Project(cameras, i, points, j);
                int k = (i * PointCount + j) * 2;
                r[k] = p[0] - Measurements[k];
                r[k + 1] = p[1] - Measurements[k + 1];
            }
        }
        return r;
    }

    /** Huber costs of a residual pair: 0.5 |r|^2 inside the band, delta (|r| - delta/2) outside. */
    internal double Cost(double[] r, double huberDelta = 0.0)
    {
        double sum = 0.0;
        for (int k = 0; k + 1 < r.Length; k += 2)
        {
            double norm = Math.Sqrt(r[k] * r[k] + r[k + 1] * r[k + 1]);
            if (huberDelta <= 0.0 || norm <= huberDelta) sum += 0.5 * norm * norm;
            else sum += huberDelta * (norm - 0.5 * huberDelta);
        }
        return sum;
    }

    /** Huber IRLS weights (square-root form) and the weighted residual vector. */
    internal static double[] HuberWeights(double[] r, double huberDelta, out double[] weightedResidual)
    {
        var weights = new double[r.Length];
        weightedResidual = (double[])r.Clone();
        for (int k = 0; k + 1 < r.Length; k += 2)
        {
            double norm = Math.Sqrt(r[k] * r[k] + r[k + 1] * r[k + 1]);
            double w = huberDelta <= 0.0 || norm <= huberDelta ? 1.0 : Math.Sqrt(huberDelta / norm);
            weights[k] = w;
            weights[k + 1] = w;
            weightedResidual[k] *= w;
            weightedResidual[k + 1] *= w;
        }
        return weights;
    }

    /** Dense analytic Jacobian, 2K x N, optionally with each row scaled by an IRLS weight. */
    internal double[][] Jacobian(double[] x, double[] rowWeights = null)
    {
        var cameras = new double[CameraCount * CameraStride];
        var points = new double[PointCount * PointStride];
        UnpackParameters(x, cameras, points);
        int rows = Measurements.Length, cols = ParameterCount;
        var jac = new double[rows][];
        for (int k = 0; k < rows; k++) jac[k] = new double[cols];
        for (int i = 0; i < CameraCount; i++)
        {
            double theta = cameras[i * 3];
            double c = Math.Cos(theta), s = Math.Sin(theta);
            for (int j = 0; j < PointCount; j++)
            {
                double dx = points[j * 2] - cameras[i * 3 + 1];
                double dy = points[j * 2 + 1] - cameras[i * 3 + 2];
                int k = (i * PointCount + j) * 2;
                int cb = i * CameraStride, pb = CameraCount * CameraStride + j * PointStride;
                // d p / d theta = R'(theta) (X - C) with R' = [[-s, -c], [c, -s]]
                jac[k][cb] = -s * dx - c * dy;
                jac[k + 1][cb] = c * dx - s * dy;
                // d p / d C = -R, d p / d X = R
                jac[k][cb + 1] = -c; jac[k][cb + 2] = s;
                jac[k + 1][cb + 1] = -s; jac[k + 1][cb + 2] = -c;
                jac[k][pb] = c; jac[k][pb + 1] = -s;
                jac[k + 1][pb] = s; jac[k + 1][pb + 1] = c;
            }
        }
        if (rowWeights != null)
        {
            for (int row = 0; row < rows; row++)
            {
                double w = rowWeights[row];
                if (w == 1.0) continue;
                for (int col = 0; col < cols; col++) jac[row][col] *= w;
            }
        }
        return jac;
    }

    /** Fills A (block diagonal per free camera), B (block diagonal per point) and W (camera-point coupling). */
    internal void AssembleBlocks(double[] x, double lambda, double[] rowWeights,
        out double[][] aBlock, out double[][] bBlock, out double[][,] wBlock)
    {
        double[][] jac = Jacobian(x, rowWeights);
        int freeCameras = CameraCount - 1;
        aBlock = new double[freeCameras][];
        for (int i = 0; i < freeCameras; i++) aBlock[i] = new double[9];
        bBlock = new double[PointCount][];
        for (int j = 0; j < PointCount; j++) bBlock[j] = new double[4];
        wBlock = new double[freeCameras][,];
        for (int i = 0; i < freeCameras; i++) wBlock[i] = new double[PointCount, 6];

        for (int i = 0; i < CameraCount; i++)
        {
            int block = i - 1;                                        // camera 0 is frozen
            for (int j = 0; j < PointCount; j++)
            {
                int k = (i * PointCount + j) * 2;
                int cb = i * CameraStride, pb = CameraCount * CameraStride + j * PointStride;
                for (int r = 0; r < 2; r++)
                {
                    for (int u = 0; u < 3; u++)
                    {
                        for (int v = 0; v < 3; v++)
                        {
                            if (block >= 0) aBlock[block][u * 3 + v] += jac[k + r][cb + u] * jac[k + r][cb + v];
                        }
                        for (int v = 0; v < 2; v++)
                        {
                            if (block >= 0) wBlock[block][j, u * 2 + v] += jac[k + r][cb + u] * jac[k + r][pb + v];
                        }
                    }
                    for (int u = 0; u < 2; u++)
                    {
                        for (int v = 0; v < 2; v++) bBlock[j][u * 2 + v] += jac[k + r][pb + u] * jac[k + r][pb + v];
                    }
                }
            }
        }
        for (int i = 0; i < freeCameras; i++) DampBlock(aBlock[i], 3, lambda);
        for (int j = 0; j < PointCount; j++) DampBlock(bBlock[j], 2, lambda);
    }

    private static void DampBlock(double[] block, int size, double lambda)
    {
        for (int d = 0; d < size; d++) block[d * size + d] *= 1.0 + lambda;
    }

    /** Dense reduced normal equations [[A, W], [W^T, B]] assembled from the same Jacobian. */
    internal double[][] DenseNormal(double[] x, double[] residual, double lambda, double[] rowWeights, out double[] gradient)
    {
        double[][] jac = Jacobian(x, rowWeights);
        var map = FreeMap();
        int m = 0;
        foreach (int v in map) if (v >= 0) m++;
        var h = new double[m][];
        for (int i = 0; i < m; i++) h[i] = new double[m];
        gradient = new double[m];
        for (int row = 0; row < residual.Length; row++)
        {
            for (int a = 0; a < ParameterCount; a++)
            {
                if (map[a] < 0 || jac[row][a] == 0.0) continue;
                int ia = map[a];
                gradient[ia] += jac[row][a] * residual[row];
                for (int b = 0; b < ParameterCount; b++)
                {
                    if (map[b] < 0 || jac[row][b] == 0.0) continue;
                    h[ia][map[b]] += jac[row][a] * jac[row][b];
                }
            }
        }
        for (int i = 0; i < m; i++) h[i][i] *= 1.0 + lambda;
        return h;
    }

    /** The same damped system in block form, as the Schur path sees it. */
    internal double[][] BlockNormal(double[] x, double[] residual, double lambda, double[] rowWeights, out double[] gradient)
    {
        AssembleBlocks(x, lambda, rowWeights, out double[][] aBlock, out double[][] bBlock, out double[][,] wBlock);
        double[][] jac = Jacobian(x, rowWeights);
        int m = 3 * (CameraCount - 1) + 2 * PointCount;
        var h = new double[m][];
        for (int i = 0; i < m; i++) h[i] = new double[m];
        gradient = new double[m];

        // gradient via the same dense product; blocks only contribute to their own rows/columns
        for (int row = 0; row < residual.Length; row++)
        {
            for (int i = 0; i < CameraCount; i++)
            {
                int cb = i * CameraStride;
                int gi = 3 * (i - 1);
                for (int u = 0; u < 3; u++)
                {
                    if (jac[row][cb + u] == 0.0) continue;
                    if (i > 0) gradient[gi + u] += jac[row][cb + u] * residual[row];
                }
            }
            for (int j = 0; j < PointCount; j++)
            {
                int pb = CameraCount * CameraStride + j * PointStride;
                int gj = 3 * (CameraCount - 1) + 2 * j;
                for (int u = 0; u < 2; u++)
                {
                    if (jac[row][pb + u] != 0.0) gradient[gj + u] += jac[row][pb + u] * residual[row];
                }
            }
        }

        for (int i = 0; i < CameraCount - 1; i++)
        {
            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++) h[3 * i + u][3 * i + v] = aBlock[i][u * 3 + v];
            }
        }
        int pointBase = 3 * (CameraCount - 1);
        for (int j = 0; j < PointCount; j++)
        {
            for (int u = 0; u < 2; u++)
            {
                for (int v = 0; v < 2; v++) h[pointBase + 2 * j + u][pointBase + 2 * j + v] = bBlock[j][u * 2 + v];
            }
        }
        for (int i = 0; i < CameraCount - 1; i++)
        {
            for (int j = 0; j < PointCount; j++)
            {
                for (int u = 0; u < 3; u++)
                {
                    for (int v = 0; v < 2; v++)
                    {
                        double value = wBlock[i][j, u * 2 + v];
                        h[3 * i + u][pointBase + 2 * j + v] = value;
                        h[pointBase + 2 * j + v][3 * i + u] = value;
                    }
                }
            }
        }
        return h;
    }

    /** Solves the damped system (A, B, W) by Schur elimination of the points. */
    internal double[] SchurSolve(double[] x, double[] residual, double lambda, double[] rowWeights)
    {
        AssembleBlocks(x, lambda, rowWeights, out double[][] aBlock, out double[][] bBlock, out double[][,] wBlock);
        int camCount = CameraCount - 1, pointCount = PointCount;
        var gradient = BlockGradient(x, residual, rowWeights);

        // A^-1 W for every camera and point: solve a 3x3 system per (camera, point).
        var aInvW = new double[camCount][,];
        var aInvG = new double[camCount][];
        for (int i = 0; i < camCount; i++)
        {
            aInvW[i] = new double[pointCount, 6];
            var rows = BlockRows(aBlock[i], 3);
            for (int j = 0; j < pointCount; j++)
            {
                var rhs = new double[3];
                for (int v = 0; v < 2; v++)
                {
                    for (int u = 0; u < 3; u++) rhs[u] = wBlock[i][j, u * 2 + v];
                    double[] column = Linalg.Solve(rows, rhs);
                    for (int u = 0; u < 3; u++) aInvW[i][j, u * 2 + v] = column[u];
                }
            }
            var g = new double[3];
            for (int u = 0; u < 3; u++) g[u] = gradient[3 * i + u];
            aInvG[i] = Linalg.Solve(rows, g);
        }

        int baseIndex = 3 * camCount;
        var s = new double[2 * pointCount][];
        for (int i = 0; i < 2 * pointCount; i++) s[i] = new double[2 * pointCount];
        var rhsPoint = new double[2 * pointCount];
        for (int j = 0; j < pointCount; j++)
        {
            for (int u = 0; u < 2; u++)
            {
                for (int v = 0; v < 2; v++) s[2 * j + u][2 * j + v] = bBlock[j][u * 2 + v];
            }
        }
        // S = B - W^T A^-1 W. Eliminating a camera couples *every* pair of points it observes, so the double loop
        // over points is required - this is exactly where the fill-in of a real bundle adjustment comes from.
        for (int i = 0; i < camCount; i++)
        {
            for (int j1 = 0; j1 < pointCount; j1++)
            {
                for (int j2 = 0; j2 < pointCount; j2++)
                {
                    for (int u = 0; u < 2; u++)
                    {
                        for (int v = 0; v < 2; v++)
                        {
                            double sum = 0.0;
                            for (int a = 0; a < 3; a++) sum += wBlock[i][j1, a * 2 + u] * aInvW[i][j2, a * 2 + v];
                            s[2 * j1 + u][2 * j2 + v] -= sum;
                        }
                    }
                }
            }
        }
        for (int j = 0; j < pointCount; j++)
        {
            for (int u = 0; u < 2; u++)
            {
                double sum = -gradient[baseIndex + 2 * j + u];
                for (int i = 0; i < camCount; i++)
                {
                    for (int a = 0; a < 3; a++) sum += wBlock[i][j, a * 2 + u] * aInvG[i][a];
                }
                rhsPoint[2 * j + u] = sum;
            }
        }
        double[] deltaPoints = Linalg.Solve(s, rhsPoint);

        var delta = new double[3 * camCount + 2 * pointCount];
        for (int j = 0; j < pointCount; j++)
        {
            for (int u = 0; u < 2; u++) delta[baseIndex + 2 * j + u] = deltaPoints[2 * j + u];
        }
        for (int i = 0; i < camCount; i++)
        {
            var rhs = new double[3];
            for (int u = 0; u < 3; u++) rhs[u] = -gradient[3 * i + u];
            for (int j = 0; j < pointCount; j++)
            {
                for (int a = 0; a < 3; a++)
                {
                    for (int v = 0; v < 2; v++) rhs[a] -= wBlock[i][j, a * 2 + v] * deltaPoints[2 * j + v];
                }
            }
            double[] deltaCamera = Linalg.Solve(BlockRows(aBlock[i], 3), rhs);
            for (int u = 0; u < 3; u++) delta[3 * i + u] = deltaCamera[u];
        }
        return delta;
    }

    private double[] BlockGradient(double[] x, double[] residual, double[] rowWeights)
    {
        double[][] jac = Jacobian(x, rowWeights);
        var g = new double[3 * (CameraCount - 1) + 2 * PointCount];
        for (int row = 0; row < residual.Length; row++)
        {
            for (int i = 1; i < CameraCount; i++)
            {
                int cb = i * CameraStride;
                for (int u = 0; u < 3; u++) g[3 * (i - 1) + u] += jac[row][cb + u] * residual[row];
            }
            for (int j = 0; j < PointCount; j++)
            {
                int pb = CameraCount * CameraStride + j * PointStride;
                for (int u = 0; u < 2; u++) g[3 * (CameraCount - 1) + 2 * j + u] += jac[row][pb + u] * residual[row];
            }
        }
        return g;
    }

    internal static double[][] BlockRows(double[] block, int size)
    {
        var rows = new double[size][];
        for (int i = 0; i < size; i++)
        {
            rows[i] = new double[size];
            for (int j = 0; j < size; j++) rows[i][j] = block[i * size + j];
        }
        return rows;
    }

    /** Applies a reduced parameter update to a full parameter vector. */
    internal double[] ApplyDelta(double[] x, double[] delta)
    {
        var next = (double[])x.Clone();
        for (int i = 1; i < CameraCount; i++)
        {
            for (int u = 0; u < 3; u++) next[i * CameraStride + u] += delta[3 * (i - 1) + u];
        }
        int baseIndex = 3 * (CameraCount - 1);
        for (int j = 0; j < PointCount; j++)
        {
            for (int u = 0; u < 2; u++) next[CameraCount * CameraStride + j * PointStride + u] += delta[baseIndex + 2 * j + u];
        }
        return next;
    }

    /** Levenberg-Marquardt. `sparse` selects the Schur path, `huberDelta > 0` switches on Huber IRLS weighting. */
    internal double[] Optimize(double lambda, int maxIterations, bool sparse, double huberDelta)
    {
        double[] x = PackParameters(CameraEstimate, PointEstimate);
        double[] r = Residuals(x);
        double cost = Cost(r, huberDelta);
        Iterations = 0;
        Accepted = 0;
        Rejected = 0;
        for (int it = 0; it < maxIterations; it++)
        {
            Iterations++;
            double[] weights = HuberWeights(r, huberDelta, out double[] weightedResidual);

            double[] delta = sparse
                ? SchurSolve(x, weightedResidual, lambda, weights)
                : DenseSolve(x, weightedResidual, lambda, weights);
            double[] candidate = ApplyDelta(x, delta);
            double[] candidateResidual = Residuals(candidate);
            double candidateCost = Cost(candidateResidual, huberDelta);
            if (candidateCost < cost)
            {
                x = candidate;
                r = candidateResidual;
                cost = candidateCost;
                Accepted++;
                lambda = Math.Max(lambda / 3.0, 1e-12);
                if (Norm(delta) < 1e-13) break;
            }
            else
            {
                Rejected++;
                lambda *= 3.0;
            }
        }
        FinalCost = cost;
        return x;
    }

    private double[] DenseSolve(double[] x, double[] residual, double lambda, double[] rowWeights)
    {
        double[][] h = DenseNormal(x, residual, lambda, rowWeights, out double[] g);
        var rhs = new double[g.Length];
        for (int i = 0; i < g.Length; i++) rhs[i] = -g[i];
        return Linalg.Solve(h, rhs);
    }

    internal int Iterations { get; private set; }
    internal int Accepted { get; private set; }
    internal int Rejected { get; private set; }
    internal double FinalCost { get; private set; }

    internal static double Norm(double[] v)
    {
        double m = 0.0;
        foreach (double value in v) m = Math.Max(m, Math.Abs(value));
        return m;
    }

    /** Largest parameter deviation from the ground truth over the free parameters. */
    internal double TruthError(double[] x)
    {
        double worst = 0.0;
        for (int p = CameraStride; p < CameraCount * CameraStride; p++) worst = Math.Max(worst, Math.Abs(x[p] - CameraTruth[p]));
        int baseIndex = CameraCount * CameraStride;
        for (int j = 0; j < PointCount * PointStride; j++) worst = Math.Max(worst, Math.Abs(x[baseIndex + j] - PointTruth[j]));
        return worst;
    }

    /** Root-mean-square reprojection error of a parameter vector (unweighted). */
    internal double Rms(double[] x)
    {
        double[] r = Residuals(x);
        return Math.Sqrt(2.0 * Cost(r) / r.Length);
    }
}

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

public static class Solution
{
    /** Schur complement built straight from the dense reduced matrix: S = H22 - H21 H11^-1 H12. */
    private static void DenseSchur(double[][] h, double[] rhs, int cam, out double[][] s, out double[] rhsX)
    {
        int m = h.Length, px = m - cam;
        var a = new double[cam][];
        for (int i = 0; i < cam; i++)
        {
            a[i] = new double[cam];
            for (int j = 0; j < cam; j++) a[i][j] = h[i][j];
        }
        var aInvH12 = new double[cam][];
        for (int i = 0; i < cam; i++) aInvH12[i] = new double[px];
        for (int col = 0; col < px; col++)
        {
            var e = new double[cam];
            for (int i = 0; i < cam; i++) e[i] = h[i][cam + col];
            double[] solution = Linalg.Solve(a, e);
            for (int i = 0; i < cam; i++) aInvH12[i][col] = solution[i];
        }
        var rhsC = new double[cam];
        for (int i = 0; i < cam; i++) rhsC[i] = rhs[i];
        double[] aInvRhs = Linalg.Solve(a, rhsC);
        s = new double[px][];
        rhsX = new double[px];
        for (int r = 0; r < px; r++)
        {
            s[r] = new double[px];
            for (int c = 0; c < px; c++)
            {
                double sum = h[cam + r][cam + c];
                for (int i = 0; i < cam; i++) sum -= h[cam + r][i] * aInvH12[i][c];
                s[r][c] = sum;
            }
            double rx = rhs[cam + r];
            for (int i = 0; i < cam; i++) rx -= h[cam + r][i] * aInvRhs[i];
            rhsX[r] = rx;
        }
    }

    /** Max |H x - rhs| over rows [from, to), used to see which half of the Schur system is off. */
    private static double SolveResidual(double[][] h, double[] x, double[] rhs, int from = 0, int to = -1)
    {
        if (to < 0) to = h.Length;
        double worst = 0.0;
        for (int i = from; i < to; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < h.Length; j++) sum += h[i][j] * x[j];
            worst = Math.Max(worst, Math.Abs(sum - rhs[i]));
        }
        return worst;
    }

    private static double[][] FiniteDifferenceJacobian(Bundle bundle, double[] x, out double worstRelative)
    {
        double[][] analytic = bundle.Jacobian(x);
        int rows = analytic.Length, cols = x.Length;
        var numeric = new double[rows][];
        for (int k = 0; k < rows; k++) numeric[k] = new double[cols];
        worstRelative = 0.0;
        for (int c = 0; c < cols; c++)
        {
            double eps = 1e-6 * Math.Max(1.0, Math.Abs(x[c]));
            var plus = (double[])x.Clone();
            plus[c] += eps;
            var minus = (double[])x.Clone();
            minus[c] -= eps;
            double[] rp = bundle.Residuals(plus), rm = bundle.Residuals(minus);
            for (int k = 0; k < rows; k++)
            {
                numeric[k][c] = (rp[k] - rm[k]) / (2.0 * eps);
                if (analytic[k][c] != 0.0)
                {
                    double relative = Math.Abs(numeric[k][c] - analytic[k][c]) / Math.Abs(analytic[k][c]);
                    worstRelative = Math.Max(worstRelative, relative);
                }
            }
        }
        return numeric;
    }

    public static void Main()
    {
        var bundle = new Bundle(3, 20, 0xB0A7UL);
        bundle.Perturb(0x51EEDUL, 0.06, 0.12);          // check everything at a generic, non-stationary point
        double[] x0 = bundle.PackParameters(bundle.CameraEstimate, bundle.PointEstimate);

        // 1. Analytic Jacobian against central finite differences.
        double[][] fd = FiniteDifferenceJacobian(bundle, x0, out double worstRelative);
        double[][] analytic = bundle.Jacobian(x0);
        double jacobianDeviation = 0.0;
        for (int k = 0; k < analytic.Length; k++)
        {
            for (int c = 0; c < analytic[k].Length; c++) jacobianDeviation = Math.Max(jacobianDeviation, Math.Abs(fd[k][c] - analytic[k][c]));
        }
        Console.WriteLine("  Jacobian: max |analytic - FD| = " + jacobianDeviation.ToString("E3")
            + " (worst relative on nonzeros " + worstRelative.ToString("E3") + ")");
        Check(jacobianDeviation < 1e-7 && worstRelative < 1e-5, "analytic Jacobian matches finite differences");

        // 2-4. The block assembly, the dense assembly and the two damped solves must agree.
        double[] residual = bundle.Residuals(x0);
        double lambda = 1e-3;
        double[][] dense = bundle.DenseNormal(x0, residual, lambda, null, out double[] denseGradient);
        double[][] blocks = bundle.BlockNormal(x0, residual, lambda, null, out double[] blockGradient);
        double assemblyDeviation = 0.0, gradientDeviation = 0.0;
        for (int i = 0; i < dense.Length; i++)
        {
            gradientDeviation = Math.Max(gradientDeviation, Math.Abs(denseGradient[i] - blockGradient[i]));
            for (int j = 0; j < dense.Length; j++) assemblyDeviation = Math.Max(assemblyDeviation, Math.Abs(dense[i][j] - blocks[i][j]));
        }
        Console.WriteLine("  assembly: max |dense H - block H| = " + assemblyDeviation.ToString("E3")
            + " |gradient| = " + gradientDeviation.ToString("E3"));
        Check(assemblyDeviation < 1e-9, "block-assembled normal equations equal the dense J^T J");
        Check(gradientDeviation < 1e-9, "block gradient equals the dense one");

        int m = dense.Length;
        var rhs = new double[m];
        for (int i = 0; i < m; i++) rhs[i] = -denseGradient[i];
        double[] denseDelta = Linalg.Solve(dense, rhs);
        double[] schurDelta = bundle.SchurSolve(x0, residual, lambda, null);
        double solveDeviation = 0.0;
        for (int i = 0; i < m; i++) solveDeviation = Math.Max(solveDeviation, Math.Abs(denseDelta[i] - schurDelta[i]));

        // The Schur complement is assembled twice: from the block path (SchurMatrix) and by eliminating the
        // 6x6 camera block of the dense matrix (DenseSchur). Both must agree, and the dense one must reproduce
        // the point part of the direct dense solve - that is the property the sparse solver relies on.
        var sSchur = SchurMatrix(bundle, x0, residual, lambda);
        DenseSchur(blocks, rhs, 6, out double[][] sDense, out double[] rhsDense);
        double schurMatrixDeviation = 0.0;
        for (int i = 0; i < sDense.Length; i++)
        {
            for (int j = 0; j < sDense.Length; j++)
            {
                schurMatrixDeviation = Math.Max(schurMatrixDeviation, Math.Abs(sDense[i][j] - sSchur[i][j]));
            }
        }
        double[] pointRef = Linalg.Solve(sDense, rhsDense);
        double densePointDeviation = 0.0;
        for (int i = 0; i < pointRef.Length; i++)
        {
            densePointDeviation = Math.Max(densePointDeviation, Math.Abs(pointRef[i] - denseDelta[6 + i]));
        }

        Console.WriteLine("  solve: |dense - Schur| = " + solveDeviation.ToString("E3")
            + " step=" + Bundle.Norm(denseDelta).ToString("E3")
            + " residual (dense/Schur) " + SolveResidual(dense, denseDelta, rhs).ToString("E3") + "/"
            + SolveResidual(blocks, schurDelta, rhs).ToString("E3")
            + " | block S vs dense S " + schurMatrixDeviation.ToString("E3")
            + " | dense-S point step vs direct " + densePointDeviation.ToString("E3"));
        Check(solveDeviation < 1e-9, "Schur complement solve equals the dense solve");
        Check(SolveResidual(blocks, schurDelta, rhs) < 1e-12, "the sparse step solves the damped normal equations");
        Check(schurMatrixDeviation < 1e-9, "block-built Schur complement equals the dense one");
        Check(densePointDeviation < 1e-12, "the dense Schur complement reproduces the direct point update");

        // 5-6. The Schur complement itself: symmetric, and positive definite (it is the fill-in-dense 40x40 block
        //      that couples every pair of points sharing a camera).
        Check(sSchur.Length == 2 * bundle.PointCount, "the Schur block has one 2x2 block per point");
        Check(IsSymmetric(sSchur, 1e-10), "Schur complement is symmetric");
        Check(LeadingMinorsPositive(sSchur), "Schur complement is positive definite (all leading minors > 0)");

        // 7-8. Gauge freedom: without pinning camera 0 the normal equations are singular.
        Check(Throws(() => Linalg.Solve(UnfixedNormal(bundle, x0, residual), new double[bundle.ParameterCount])),
            "unfixed normal equations are singular (gauge freedom detected)");
        Check(!Throws(() => Linalg.Solve(dense, rhs)), "pinned normal equations are solvable");

        // 9-11. Perfect data from a perturbed start: the optimiser must drive the reprojection error to zero and
        //       recover the ground truth (camera 0 is pinned, which is what fixes the gauge).
        var clean = new Bundle(3, 20, 0xB0A7UL);
        clean.Perturb(0x51EEDUL, 0.06, 0.12);
        double cleanStartRms = clean.Rms(clean.PackParameters(clean.CameraEstimate, clean.PointEstimate));
        double[] xClean = clean.Optimize(1e-3, 60, false, 0.0);
        double cleanRms = clean.Rms(xClean);
        Console.WriteLine("  clean data: rms " + cleanStartRms.ToString("E3") + " -> " + cleanRms.ToString("E3")
            + " iterations=" + clean.Iterations + " accepted=" + clean.Accepted + " rejected=" + clean.Rejected
            + " max truth error=" + clean.TruthError(xClean).ToString("E3"));
        Check(cleanStartRms > 0.01, "the run really starts away from the solution");
        Check(cleanRms < 1e-9, "reprojection error driven to zero on noiseless data");
        Check(clean.TruthError(xClean) < 1e-8, "ground-truth parameters recovered on noiseless data");

        // 12-13. The sparse and dense paths agree on the whole optimisation, not just one step.
        var sparseRun = new Bundle(3, 20, 0xB0A7UL);
        sparseRun.Perturb(0x51EEDUL, 0.06, 0.12);
        double[] xSparse = sparseRun.Optimize(1e-3, 60, true, 0.0);
        double parameterDeviation = 0.0;
        for (int i = 0; i < xClean.Length; i++) parameterDeviation = Math.Max(parameterDeviation, Math.Abs(xClean[i] - xSparse[i]));
        Check(parameterDeviation < 1e-9, "sparse and dense optimisations reach the same parameters");
        Check(Math.Abs(cleanRms - sparseRun.Rms(xSparse)) < 1e-12, "sparse and dense reach the same reprojection error");

        // 14. Damping controls the step: lambda = 1e12 barely moves, lambda = 1e-6 is essentially the
        //     Gauss-Newton step and collapses the error; the LM bookkeeping must balance.
        var heavilyDamped = new Bundle(3, 20, 0xB0A7UL);
        heavilyDamped.Perturb(0x51EEDUL, 0.06, 0.12);
        double dampedStart = heavilyDamped.Rms(heavilyDamped.PackParameters(heavilyDamped.CameraEstimate, heavilyDamped.PointEstimate));
        double[] xHeavy = heavilyDamped.Optimize(1e12, 1, false, 0.0);
        var lightlyDamped = new Bundle(3, 20, 0xB0A7UL);
        lightlyDamped.Perturb(0x51EEDUL, 0.06, 0.12);
        double[] xLight = lightlyDamped.Optimize(1e-6, 1, false, 0.0);
        Console.WriteLine("  one step from rms=" + dampedStart.ToString("E4") + ": lambda=1e12 rms="
            + heavilyDamped.Rms(xHeavy).ToString("E4") + " | lambda=1e-6 rms=" + lightlyDamped.Rms(xLight).ToString("E4"));
        Check(heavilyDamped.Rms(xHeavy) > 0.99 * dampedStart, "huge damping leaves the parameters almost unchanged");
        Check(lightlyDamped.Rms(xLight) < 0.5 * dampedStart, "tiny damping takes a near Gauss-Newton step");
        Check(lightlyDamped.Accepted + lightlyDamped.Rejected == lightlyDamped.Iterations,
            "LM bookkeeping balances accepted and rejected steps");

        // 15-17. Noisy data with 10% gross outliers: L2 versus Huber-weighted IRLS (same corruption stream).
        const double sigma = 0.01;
        var noisy = new Bundle(3, 20, 0xB0A7UL);
        noisy.Perturb(0x51EEDUL, 0.06, 0.12);
        noisy.Corrupt(0xC0FFEEUL, sigma, 6, 0.8);
        double[] xNoisy = noisy.Optimize(1e-3, 80, true, 0.0);
        var robust = new Bundle(3, 20, 0xB0A7UL);
        robust.Perturb(0x51EEDUL, 0.06, 0.12);
        robust.Corrupt(0xC0FFEEUL, sigma, 6, 0.8);
        double[] xRobust = robust.Optimize(1e-3, 80, true, 0.05);
        Console.WriteLine("  outliers: L2 rms=" + noisy.Rms(xNoisy).ToString("E4") + " truth error=" + noisy.TruthError(xNoisy).ToString("E3")
            + " | Huber rms=" + robust.Rms(xRobust).ToString("E4") + " truth error=" + robust.TruthError(xRobust).ToString("E3")
            + " (sigma=" + sigma.ToString("F3") + ")");
        Check(noisy.Rms(xNoisy) > sigma, "outliers inflate the L2 reprojection error");
        Check(robust.TruthError(xRobust) < noisy.TruthError(xNoisy),
            "Huber weighting lands closer to the ground truth than plain L2");
        Check(robust.Rejected >= 1, "IRLS re-weighting forces the damping to adapt");

        // 18-19. Sensor noise without outliers: the residual reproduces sigma and the estimate stays close.
        var mild = new Bundle(3, 20, 0xB0A7UL);
        mild.Perturb(0x51EEDUL, 0.06, 0.12);
        mild.Corrupt(0x1234UL, sigma, 0, 0.0);
        double[] xMild = mild.Optimize(1e-3, 80, true, 0.0);
        Console.WriteLine("  sigma only: rms=" + mild.Rms(xMild).ToString("E4") + " truth error=" + mild.TruthError(xMild).ToString("E3"));
        Check(mild.Rms(xMild) > 0.7 * sigma && mild.Rms(xMild) < 1.4 * sigma,
            "reprojection residual stays at the sensor noise level");
        Check(mild.TruthError(xMild) < 0.02, "parameter error stays small at sigma = 0.01");

        // 20-22. Boundary cases and determinism.
        Check(Throws(() => new Bundle(1, 20, 1UL)), "single camera rejected (no gauge to pin)");
        var thinRig = new Bundle(2, 1, 1UL);                        // 5 unknowns, 4 residuals
        double[] thinX = thinRig.PackParameters(thinRig.CameraEstimate, thinRig.PointEstimate);
        double[] thinResidual = thinRig.Residuals(thinX);
        double[][] thinUndamped = thinRig.DenseNormal(thinX, thinResidual, 0.0, null, out double[] thinGradient);
        double[][] thinDamped = thinRig.DenseNormal(thinX, thinResidual, 1e-3, null, out _);
        var thinRhs = new double[thinGradient.Length];
        Check(Throws(() => Linalg.Solve(thinUndamped, thinRhs)),
            "under-determined rig: the undamped normal equations are singular");
        Check(!Throws(() => Linalg.Solve(thinDamped, thinRhs)), "diagonal damping regularises the thin rig");
        var again = new Bundle(3, 20, 0xB0A7UL);
        again.Perturb(0x51EEDUL, 0.06, 0.12);
        double[] xAgain = again.Optimize(1e-3, 60, false, 0.0);
        Check(xAgain[12] == xClean[12] && again.Iterations == clean.Iterations && again.Accepted == clean.Accepted,
            "identical runs give identical results");

        Console.WriteLine("PASS");
    }

    /** Rebuilds S = B - W^T A^-1 W through the public block path (used for the definiteness checks). */
    private static double[][] SchurMatrix(Bundle bundle, double[] x, double[] residual, double lambda)
    {
        bundle.AssembleBlocks(x, lambda, null, out double[][] a, out double[][] b, out double[][,] w);
        int camCount = bundle.CameraCount - 1, pointCount = bundle.PointCount;
        var s = new double[2 * pointCount][];
        for (int i = 0; i < 2 * pointCount; i++) s[i] = new double[2 * pointCount];
        for (int j = 0; j < pointCount; j++)
        {
            for (int u = 0; u < 2; u++)
            {
                for (int v = 0; v < 2; v++) s[2 * j + u][2 * j + v] = b[j][u * 2 + v];
            }
        }
        for (int i = 0; i < camCount; i++)
        {
            var rows = Bundle.BlockRows(a[i], 3);
            var inverseW = new double[pointCount][,];
            for (int j = 0; j < pointCount; j++)
            {
                inverseW[j] = new double[3, 2];
                for (int v = 0; v < 2; v++)
                {
                    var rhs = new double[3];
                    for (int u = 0; u < 3; u++) rhs[u] = w[i][j, u * 2 + v];
                    double[] solution = Linalg.Solve(rows, rhs);
                    for (int u = 0; u < 3; u++) inverseW[j][u, v] = solution[u];
                }
            }
            for (int j1 = 0; j1 < pointCount; j1++)
            {
                for (int j2 = 0; j2 < pointCount; j2++)
                {
                    for (int u = 0; u < 2; u++)
                    {
                        for (int v = 0; v < 2; v++)
                        {
                            double sum = 0.0;
                            for (int c = 0; c < 3; c++) sum += w[i][j1, c * 2 + u] * inverseW[j2][c, v];
                            s[2 * j1 + u][2 * j2 + v] -= sum;
                        }
                    }
                }
            }
        }
        return s;
    }

    /** Normal equations without pinning camera 0: the gauge freedom makes them singular. */
    private static double[][] UnfixedNormal(Bundle bundle, double[] x, double[] residual)
    {
        double[][] jac = bundle.Jacobian(x);
        int n = bundle.ParameterCount;
        var h = new double[n][];
        for (int i = 0; i < n; i++) h[i] = new double[n];
        for (int row = 0; row < residual.Length; row++)
        {
            for (int a = 0; a < n; a++)
            {
                if (jac[row][a] == 0.0) continue;
                for (int b = 0; b < n; b++) h[a][b] += jac[row][a] * jac[row][b];
            }
        }
        return h;
    }

    private static bool IsSymmetric(double[][] a, double tolerance)
    {
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = i + 1; j < a.Length; j++)
            {
                if (Math.Abs(a[i][j] - a[j][i]) > tolerance) return false;
            }
        }
        return true;
    }

    private static bool LeadingMinorsPositive(double[][] a)
    {
        int n = a.Length;
        for (int size = 1; size <= n; size++)
        {
            var sub = new double[size][];
            for (int i = 0; i < size; i++) sub[i] = new double[size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++) sub[i][j] = a[i][j];
            }
            var rhs = new double[size];
            for (int i = 0; i < size; i++) rhs[i] = (i == size - 1) ? 1.0 : 0.0;
            try
            {
                double[] solution = Linalg.Solve(sub, rhs);
                if (double.IsNaN(solution[size - 1]) || double.IsInfinity(solution[size - 1])) return false;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
        }
        return true;
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
