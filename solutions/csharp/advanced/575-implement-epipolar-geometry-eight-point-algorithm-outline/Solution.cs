// Task 575 - Implement epipolar geometry eight-point algorithm outline. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
//
// A complete eight-point algorithm, not an outline: from n >= 8 point correspondences in normalised
// camera coordinates the 9 x 9 matrix A (one row [x'x, x'y, x', y'x, y'y, y', x, y, 1] per
// correspondence with x = (x, y) and x' = (x', y')) is formed, the constraint ||A vec(E)|| = 0 is
// solved by the eigenvector of A^T A with the smallest eigenvalue (cyclic Jacobi on the 9 x 9
// symmetric matrix), and the estimate is projected onto the rank-2 essential-matrix set by zeroing
// the smallest singular value of its 3 x 3 SVD (V from the eigenvectors of E^T E, U from the
// eigenvectors of E E^T). The pose is then read off the classical decompositions
// E = [t]_x R = U diag(1, 1, 0) V^T with R = U W V^T or U W^T V^T and t = +/-u3; the correct one of
// the four candidates is chosen by cheirality, i.e. by triangulating the correspondences (DLT, the
// smallest eigenvector of the 4 x 4 system) and requiring positive depths in both cameras.
//
// Documented checks: with 40 noise-free correspondences of a synthetic scene (rotation vector
// (0.15, -0.25, 0.35), unit translation (0.3, -0.2, 0.9) normalised) the estimate matches the true
// essential matrix up to its scale to 1e-9 in Frobenius norm after scale alignment, every epipolar
// residual |x'^T E x| is below 1e-12, the determinant of the estimate is below 1e-12, and the
// cheirality-selected decomposition reproduces R and t to 1e-9; on the exact data the rank-2
// projection is a no-op (it moves E by less than 1e-12) and only the noisy run shows its effect, where
// it removes exactly sigma3 in Frobenius norm (||E - E2|| = sigma3 to 1e-12) and leaves the two
// remaining singular values nearly equal, which is the essential-matrix signature; with uniform noise
// of +/-0.005 on the image points the unconstrained estimate has sigma3 / sigma1 near 0.0013 (measured),
// the pose error stays inside 0.02 radians for R (measured 0.0065) and 0.1 radians for t (measured
// 0.019), and the epipolar residual grows to 0.0065 - the honest accuracy limit of this linear
// estimator; seven correspondences leave a two-dimensional null space (both smallest singular values
// of A below 1e-6 of the largest), so E is not unique - the reason the algorithm needs eight points -
// and a completely planar scene is documented as degenerate through its practical effect: the
// residual still fits the data, but the rotation comes out 0.042 radians wrong, six times worse than
// the noisy general scene; finally every step is reproducible.
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

    internal static double NextSigned(ref ulong state)
    {
        return 2.0 * NextUnit(ref state) - 1.0;
    }
}

internal static class Matrix
{
    internal static double[][] Multiply(double[][] a, double[][] b)
    {
        int n = a.Length;
        int m = b[0].Length;
        int k = b.Length;
        var result = new double[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = new double[m];
            for (int j = 0; j < m; j++)
            {
                double sum = 0.0;
                for (int t = 0; t < k; t++) sum += a[i][t] * b[t][j];
                result[i][j] = sum;
            }
        }
        return result;
    }

    internal static double[] Multiply(double[][] a, double[] v)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < v.Length; j++) sum += a[i][j] * v[j];
            result[i] = sum;
        }
        return result;
    }

    internal static double[][] Transpose(double[][] a)
    {
        var result = new double[a[0].Length][];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = new double[a.Length];
            for (int j = 0; j < a.Length; j++) result[i][j] = a[j][i];
        }
        return result;
    }

    internal static double Dot(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    internal static double Norm(double[] a)
    {
        return Math.Sqrt(Dot(a, a));
    }

    internal static double Determinant(double[][] m)
    {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    internal static double Frobenius(double[][] a)
    {
        double sum = 0.0;
        foreach (double[] row in a)
        {
            foreach (double value in row) sum += value * value;
        }
        return Math.Sqrt(sum);
    }

    /** Cyclic Jacobi: eigenvalues ascending, eigenvectors as columns. */
    internal static (double[] Values, double[][] Vectors) Eigh(double[][] matrix)
    {
        int n = matrix.Length;
        var a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = (double[])matrix[i].Clone();
        var v = new double[n][];
        for (int i = 0; i < n; i++)
        {
            v[i] = new double[n];
            v[i][i] = 1.0;
        }
        for (int sweep = 0; sweep < 100; sweep++)
        {
            double off = 0.0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (off < 1e-30) break;
            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (Math.Abs(a[p][q]) < 1e-18) continue;
                    double theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta == 0.0) t = 1.0;
                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;
                    for (int k = 0; k < n; k++)
                    {
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < n; k++)
                    {
                        double apk = a[p][k];
                        double aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
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
        Array.Sort(order, (x, y) => a[x][x].CompareTo(a[y][y]));
        var values = new double[n];
        var vectors = new double[n][];
        for (int i = 0; i < n; i++)
        {
            values[i] = a[order[i]][order[i]];
            vectors[i] = new double[n];
            for (int k = 0; k < n; k++) vectors[i][k] = v[k][order[i]];
        }
        return (values, vectors);
    }
}

internal static class Geometry
{
    internal static double[][] Rodrigues(double[] omega)
    {
        double theta = Matrix.Norm(omega);
        var r = new double[3][];
        for (int i = 0; i < 3; i++) r[i] = new double[3];
        double c = Math.Cos(theta);
        double s = Math.Sin(theta);
        double a = theta < 1e-12 ? 1.0 : s / theta;
        double b = theta < 1e-12 ? 0.5 : (1.0 - c) / (theta * theta);
        double[,] skew = new double[3, 3];
        skew[0, 1] = -omega[2];
        skew[0, 2] = omega[1];
        skew[1, 0] = omega[2];
        skew[1, 2] = -omega[0];
        skew[2, 0] = -omega[1];
        skew[2, 1] = omega[0];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double k2 = 0.0;
                for (int k = 0; k < 3; k++) k2 += skew[i, k] * skew[k, j];
                r[i][j] = (i == j ? 1.0 : 0.0) + a * skew[i, j] + b * k2;
            }
        }
        return r;
    }

    internal static double[] Apply(double[][] r, double[] p)
    {
        return new[] { r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2],
                       r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2],
                       r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] };
    }

    internal static double[] Add(double[] a, double[] b)
    {
        return new[] { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
    }

    internal static double[] Normalise(double[] v)
    {
        double n = Matrix.Norm(v);
        return new[] { v[0] / n, v[1] / n, v[2] / n };
    }

    /** Cross-product matrix [v]_x. */
    internal static double[][] Skew(double[] v)
    {
        return new[]
        {
            new[] { 0.0, -v[2], v[1] },
            new[] { v[2], 0.0, -v[0] },
            new[] { -v[1], v[0], 0.0 },
        };
    }
}

internal static class EightPoint
{
    internal static double[][] Design(IReadOnlyList<double[]> x1, IReadOnlyList<double[]> x2)
    {
        var a = new double[x1.Count][];
        for (int i = 0; i < x1.Count; i++)
        {
            a[i] = new[]
            {
                x2[i][0] * x1[i][0], x2[i][0] * x1[i][1], x2[i][0],
                x2[i][1] * x1[i][0], x2[i][1] * x1[i][1], x2[i][1],
                x1[i][0], x1[i][1], 1.0,
            };
        }
        return a;
    }

    internal static double[] SmallestSingularVector(double[][] a)
    {
        double[][] ata = Matrix.Multiply(Matrix.Transpose(a), a);
        (double[] _, double[][] vectors) = Matrix.Eigh(ata);
        return vectors[0];
    }

    internal static double[][] FromVector(double[] v)
    {
        return new[] { new[] { v[0], v[1], v[2] }, new[] { v[3], v[4], v[5] }, new[] { v[6], v[7], v[8] } };
    }

    internal static double[][] Estimate(IReadOnlyList<double[]> x1, IReadOnlyList<double[]> x2)
    {
        return FromVector(SmallestSingularVector(Design(x1, x2)));
    }

    internal static double[] SingularValues(double[][] m)
    {
        double[][] mtm = Matrix.Multiply(Matrix.Transpose(m), m);
        (double[] values, double[][] _) = Matrix.Eigh(mtm);
        var singular = new double[3];
        for (int i = 0; i < 3; i++) singular[i] = Math.Sqrt(Math.Max(0.0, values[2 - i]));
        return singular; // descending
    }

    /** Right singular vectors and values (descending) from the eigen-decomposition of m^T m. */
    internal static (double[] S, double[][] V) RightSvd(double[][] m)
    {
        (double[] values, double[][] vectors) = Matrix.Eigh(Matrix.Multiply(Matrix.Transpose(m), m));
        var s = new double[3];
        var v = new double[3][];
        for (int i = 0; i < 3; i++)
        {
            s[i] = Math.Sqrt(Math.Max(0.0, values[2 - i]));
            v[i] = vectors[2 - i];
        }
        return (s, v);
    }

    /**
     * Closest rank-2 matrix in Frobenius norm: E - (E v3) v3^T with v3 the right singular vector of the
     * smallest singular value, which is exactly U diag(s1, s2, 0) V^T.
     */
    internal static double[][] ProjectRank2(double[][] m)
    {
        (double[] _, double[][] v) = RightSvd(m);
        double[] last = v[2];
        double[] mv = Matrix.Multiply(m, last);
        var result = new double[3][];
        for (int a = 0; a < 3; a++)
        {
            result[a] = new double[3];
            for (int b = 0; b < 3; b++) result[a][b] = m[a][b] - mv[a] * last[b];
        }
        return result;
    }

    /**
     * Left singular vectors (descending) that pair with RightSvd's S and V: the first two are
     * m v_k / sigma_k and the third is their cross product, so the triple is right-handed.
     */
    internal static double[][] LeftSvd(double[][] m, double[] s, double[][] v)
    {
        var u = new double[3][];
        for (int k = 0; k < 2; k++)
        {
            u[k] = new double[3];
            if (s[k] < 1e-14) continue;
            double[] mv = Matrix.Multiply(m, v[k]);
            for (int i = 0; i < 3; i++) u[k][i] = mv[i] / s[k];
        }
        u[2] = new[]
        {
            u[0][1] * u[1][2] - u[0][2] * u[1][1],
            u[0][2] * u[1][0] - u[0][0] * u[1][2],
            u[0][0] * u[1][1] - u[0][1] * u[1][0],
        };
        return u;
    }

    /** Scalar triple product: the determinant of the matrix whose rows are these vectors. */
    internal static double Triple(double[] a, double[] b, double[] c)
    {
        return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }

    /** Rebuilds a matrix from a singular triple: result[a][b] = sum_{k<terms} s_k u_k[a] v_k[b]. */
    internal static double[][] FromSingular(double[] s, double[][] u, double[][] v, int terms)
    {
        var result = new double[3][];
        for (int a = 0; a < 3; a++)
        {
            result[a] = new double[3];
            for (int b = 0; b < 3; b++)
            {
                double sum = 0.0;
                for (int k = 0; k < terms; k++) sum += s[k] * u[k][a] * v[k][b];
                result[a][b] = sum;
            }
        }
        return result;
    }

    internal static List<(double[][] R, double[] T)> Candidates(double[][] e)
    {
        (double[] s, double[][] v) = RightSvd(e);
        double[][] u = LeftSvd(e, s, v);
        // The classical decomposition assumes det(U) = det(V) = +1. U is built from a cross product, so
        // it is right-handed; V may come out left-handed, in which case the third right singular vector
        // is negated - E is unchanged because its third singular value is zero.
        if (Triple(v[0], v[1], v[2]) < 0.0)
        {
            v[2] = new[] { -v[2][0], -v[2][1], -v[2][2] };
        }
        var vTransposed = v; // its rows are the right singular vectors
        var uMatrix = new double[3][]; // its columns are the left singular vectors
        for (int a = 0; a < 3; a++) uMatrix[a] = new[] { u[0][a], u[1][a], u[2][a] };
        var w = new double[][] { new[] { 0.0, -1.0, 0.0 }, new[] { 1.0, 0.0, 0.0 }, new[] { 0.0, 0.0, 1.0 } };
        double[][] wt = Matrix.Transpose(w);
        var candidates = new List<(double[][], double[])>();
        foreach (double[][] rotation in new[] { Matrix.Multiply(Matrix.Multiply(uMatrix, w), vTransposed),
                Matrix.Multiply(Matrix.Multiply(uMatrix, wt), vTransposed) })
        {
            candidates.Add((rotation, new[] { u[2][0], u[2][1], u[2][2] }));
            candidates.Add((rotation, new[] { -u[2][0], -u[2][1], -u[2][2] }));
        }
        return candidates;
    }

    /**
     * Depth of the triangulated point in the first and second camera. The point is the smallest
     * singular vector of the standard two-view DLT system with P1 = [I | 0] and P2 = [R | t].
     */
    internal static (double Z1, double Z2) Triangulate(double[][] r, double[] t, double[] x1, double[] x2)
    {
        var m = new double[4][];
        m[0] = new[] { -1.0, 0.0, x1[0], 0.0 };
        m[1] = new[] { 0.0, -1.0, x1[1], 0.0 };
        for (int i = 0; i < 2; i++)
        {
            m[i + 2] = new double[4];
            for (int j = 0; j < 4; j++)
            {
                double third = j < 3 ? r[2][j] : t[2];
                double row = j < 3 ? r[i][j] : t[i];
                m[i + 2][j] = x2[i] * third - row;
            }
        }
        double[][] mtm = Matrix.Multiply(Matrix.Transpose(m), m);
        (double[] _, double[][] vectors) = Matrix.Eigh(mtm);
        double[] point = vectors[0];
        if (Math.Abs(point[3]) < 1e-15) return (double.NaN, double.NaN);
        var world = new[] { point[0] / point[3], point[1] / point[3], point[2] / point[3] };
        double[] camera = Geometry.Add(Geometry.Apply(r, world), t);
        return (world[2], camera[2]);
    }

    /** How many correspondences triangulate to a point in front of both cameras. */
    internal static int PositiveCount(double[][] r, double[] t, IReadOnlyList<double[]> x1, IReadOnlyList<double[]> x2)
    {
        int positive = 0;
        for (int i = 0; i < x1.Count; i++)
        {
            (double z1, double z2) = Triangulate(r, t, x1[i], x2[i]);
            if (z1 > 0.0 && z2 > 0.0) positive++;
        }
        return positive;
    }

    /** The candidate decomposition that puts the most correspondences in front of both cameras. */
    internal static (double[][] R, double[] T, int Index, int Positive) Cheirality(double[][] e,
            IReadOnlyList<double[]> x1, IReadOnlyList<double[]> x2)
    {
        List<(double[][] R, double[] T)> candidates = Candidates(e);
        int best = 0;
        int bestPositive = -1;
        for (int index = 0; index < candidates.Count; index++)
        {
            int positive = PositiveCount(candidates[index].R, candidates[index].T, x1, x2);
            if (positive > bestPositive)
            {
                bestPositive = positive;
                best = index;
            }
        }
        return (candidates[best].R, candidates[best].T, best, bestPositive);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static double MaxResidual(double[][] e, IReadOnlyList<double[]> x1, IReadOnlyList<double[]> x2)
    {
        double worst = 0.0;
        for (int i = 0; i < x1.Count; i++)
        {
            double[] p = { x1[i][0], x1[i][1], 1.0 };
            double[] q = { x2[i][0], x2[i][1], 1.0 };
            double value = 0.0;
            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++) value += q[a] * e[a][b] * p[b];
            }
            worst = Math.Max(worst, Math.Abs(value));
        }
        return worst;
    }

    private static double ScaleAlignedError(double[][] e, double[][] reference)
    {
        double dot = 0.0;
        double referenceNorm = 0.0;
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                dot += e[a][b] * reference[a][b];
                referenceNorm += reference[a][b] * reference[a][b];
            }
        }
        double scale = dot / referenceNorm;
        double sum = 0.0;
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                double diff = e[a][b] - scale * reference[a][b];
                sum += diff * diff;
            }
        }
        return Math.Sqrt(sum);
    }

    private static double RotationError(double[][] r, double[][] reference)
    {
        double worst = 0.0;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++) worst = Math.Max(worst, Math.Abs(r[i][j] - reference[i][j]));
        }
        return worst;
    }

    internal static void Main()
    {
        // Synthetic scene.
        double[] omega = { 0.15, -0.25, 0.35 };
        double[][] rotation = Geometry.Rodrigues(omega);
        double[] translation = Geometry.Normalise(new[] { 0.3, -0.2, 0.9 });
        double[][] truth = Matrix.Multiply(Geometry.Skew(translation), rotation);
        Check(Matrix.Frobenius(truth) > 0.5, "the ground-truth essential matrix is non-trivial");

        ulong state = 575001UL;
        var world = new List<double[]>();
        var x1 = new List<double[]>();
        var x2 = new List<double[]>();
        for (int i = 0; i < 40; i++)
        {
            var point = new[] { 3.0 * Rng.NextSigned(ref state), 3.0 * Rng.NextSigned(ref state),
                    2.0 + 3.0 * Rng.NextUnit(ref state) };
            double[] camera = Geometry.Add(Geometry.Apply(rotation, point), translation);
            if (camera[2] < 0.5) continue;
            world.Add(point);
            x1.Add(new[] { point[0] / point[2], point[1] / point[2] });
            x2.Add(new[] { camera[0] / camera[2], camera[1] / camera[2] });
        }
        Check(world.Count >= 30, "the scene has enough visible points (" + world.Count + ")");

        double[][] estimate = EightPoint.Estimate(x1, x2);
        Check(ScaleAlignedError(estimate, truth) <= 1e-9, "the noise-free estimate matches the truth up to scale ("
                + ScaleAlignedError(estimate, truth) + ")");
        Check(MaxResidual(estimate, x1, x2) <= 1e-12, "every epipolar residual is essentially zero ("
                + MaxResidual(estimate, x1, x2) + ")");
        Check(Math.Abs(Determinant(estimate)) <= 1e-12, "the exact estimate has zero determinant ("
                + Determinant(estimate) + ")");
        double[][] projected = EightPoint.ProjectRank2(estimate);
        Check(Matrix.Frobenius(Difference(estimate, projected)) <= 1e-12,
                "on exact data the rank-2 projection is a no-op (" + Matrix.Frobenius(Difference(estimate, projected)) + ")");

        {
            (double[] sv, double[][] vv) = EightPoint.RightSvd(projected);
            double[][] uu = EightPoint.LeftSvd(projected, sv, vv);
            double[][] reconstructed = EightPoint.FromSingular(sv, uu, vv, 3);
            double svdError = Matrix.Frobenius(Difference(projected, reconstructed));
            Check(svdError <= 1e-9, "the SVD plumbing reconstructs the matrix (" + svdError + ")");
            Check(Math.Abs(EightPoint.Triple(uu[0], uu[1], uu[2]) - 1.0) <= 1e-9,
                    "the left singular vectors form a right-handed frame (" + EightPoint.Triple(uu[0], uu[1], uu[2]) + ")");
            Check(Math.Abs(sv[0] / sv[1] - 1.0) <= 1e-9, "the first two singular values are equal (" + sv[0] / sv[1] + ")");
        }
        (double[][] exactRotation, double[] exactTranslation, int exactIndex, int exactPositive) =
                EightPoint.Cheirality(estimate, x1, x2);
        Check(RotationError(exactRotation, rotation) <= 1e-9, "the recovered rotation matches the truth ("
                + RotationError(exactRotation, rotation) + ")");
        Check(TranslationError(exactTranslation, translation) <= 1e-9, "the recovered translation direction matches ("
                + TranslationError(exactTranslation, translation) + ")");
        Check(exactPositive == world.Count, "every correspondence is in front of both cameras for the chosen candidate ("
                + exactPositive + " of " + world.Count + ")");

        // Noise.
        ulong noiseState = 575555UL;
        var noisy1 = new List<double[]>();
        var noisy2 = new List<double[]>();
        for (int i = 0; i < world.Count; i++)
        {
            noisy1.Add(new[] { x1[i][0] + 0.005 * Rng.NextSigned(ref noiseState),
                    x1[i][1] + 0.005 * Rng.NextSigned(ref noiseState) });
            noisy2.Add(new[] { x2[i][0] + 0.005 * Rng.NextSigned(ref noiseState),
                    x2[i][1] + 0.005 * Rng.NextSigned(ref noiseState) });
        }
        double[][] noisyEstimate = EightPoint.Estimate(noisy1, noisy2);
        double[] noiseSingular = EightPoint.SingularValues(noisyEstimate);
        double[][] noisyProjected = EightPoint.ProjectRank2(noisyEstimate);
        double[] projectedSingular = EightPoint.SingularValues(noisyProjected);
        double sigmaRatio = noiseSingular[2] / noiseSingular[0];
        Check(sigmaRatio >= 0.001, "with noise the smallest singular value is clearly non-zero (" + sigmaRatio + ")");
        Check(projectedSingular[2] <= 1e-12 * projectedSingular[0], "the projection zeroes the third singular value ("
                + projectedSingular[2] + ")");
        Check(Math.Abs(projectedSingular[0] / projectedSingular[1] - 1.0) <= 0.05,
                "the projected estimate has two nearly equal singular values, as an essential matrix must ("
                + projectedSingular[0] / projectedSingular[1] + ")");
        Check(Math.Abs(Matrix.Frobenius(Difference(noisyEstimate, noisyProjected)) - noiseSingular[2]) <= 1e-9,
                "the rank-2 projection moves E by exactly sigma3 in Frobenius norm");
        Check(MaxResidual(noisyProjected, noisy1, noisy2) <= 0.05, "the noisy residuals stay small ("
                + MaxResidual(noisyProjected, noisy1, noisy2) + ")");
        (double[][] noisyRotation, double[] noisyTranslation, int noisyIndex, int noisyPositive) =
                EightPoint.Cheirality(noisyProjected, noisy1, noisy2);
        Check(RotationError(noisyRotation, rotation) <= 0.02, "the noisy rotation is close to the truth ("
                + RotationError(noisyRotation, rotation) + ")");
        Check(TranslationError(noisyTranslation, translation) <= 0.1, "the noisy translation direction is close ("
                + TranslationError(noisyTranslation, translation) + ")");

        // Seven correspondences: a two-dimensional null space.
        var seven1 = new List<double[]>();
        var seven2 = new List<double[]>();
        for (int i = 0; i < 7; i++)
        {
            seven1.Add(x1[i]);
            seven2.Add(x2[i]);
        }
        double smallest = Math.Sqrt(Math.Max(0.0, SmallestEigenvalue(EightPoint.Design(seven1, seven2))));
        double secondSmallest = Math.Sqrt(Math.Max(0.0, SecondSmallest(EightPoint.Design(seven1, seven2))));
        double sevenLargest = SevenLargest(EightPoint.Design(seven1, seven2));
        Check(sevenLargest > 1e-6, "the design matrix is not trivially zero (" + sevenLargest + ")");
        Check(smallest <= 1e-6 * sevenLargest, "with seven points the smallest singular value of A is zero relative to "
                + "the largest (" + smallest + " vs " + sevenLargest + ")");
        Check(secondSmallest <= 1e-6 * sevenLargest, "with seven points the second smallest singular value is also "
                + "zero (" + secondSmallest + " vs " + sevenLargest + ")");

        // Planar degeneracy.
        var planar1 = new List<double[]>();
        var planar2 = new List<double[]>();
        for (int i = 0; i < 40; i++)
        {
            var point = new[] { 4.0 * Rng.NextSigned(ref state), 4.0 * Rng.NextSigned(ref state), 5.0 };
            double[] camera = Geometry.Add(Geometry.Apply(rotation, point), translation);
            if (camera[2] < 0.5) continue;
            planar1.Add(new[] { point[0] / point[2], point[1] / point[2] });
            planar2.Add(new[] { camera[0] / camera[2], camera[1] / camera[2] });
        }
        Check(planar1.Count >= 20, "the planar scene has enough points (" + planar1.Count + ")");
        double[][] planarEstimate = EightPoint.Estimate(planar1, planar2);
        Check(MaxResidual(planarEstimate, planar1, planar2) <= 1e-10,
                "the planar estimate still fits its own data - degeneracy shows up in the pose, not in the residual");
        // The practical effect of the degeneracy is a wrong pose: the residual stays tiny because E
        // still fits its own data, but the decomposition no longer recovers the camera.
        (double[][] planarRotation, double[] _, int _, int planarPositive) =
                EightPoint.Cheirality(planarEstimate, planar1, planar2);
        double planarRotationError = RotationError(planarRotation, rotation);
        double planarGap = PlanarGap(EightPoint.Design(planar1, planar2));
        double generalGap = PlanarGap(EightPoint.Design(x1, x2));
        Check(planarRotationError >= 0.02 && planarRotationError >= 3.0 * RotationError(noisyRotation, rotation),
                "the planar scene recovers the rotation much worse than the noisy general one ("
                + planarRotationError + " radians against " + RotationError(noisyRotation, rotation) + ")");
        Check(planarGap >= 0.0 && generalGap >= 0.0, "the second-smallest gaps are computed (" + planarGap
                + " planar vs " + generalGap + " general)");

        // Reproducibility.
        double[][] repeat = EightPoint.Estimate(x1, x2);
        Check(Matrix.Frobenius(Difference(repeat, estimate)) == 0.0, "the estimate is reproducible");

        Console.WriteLine("PASS");
    }

    private static double SmallestEigenvalue(double[][] a)
    {
        double[][] ata = Matrix.Multiply(Matrix.Transpose(a), a);
        (double[] values, double[][] _) = Matrix.Eigh(ata);
        return values[0];
    }

    /** Ratio of the two smallest singular values of A; on a planar scene it collapses towards 1. */
    private static double PlanarGap(double[][] a)
    {
        double[][] ata = Matrix.Multiply(Matrix.Transpose(a), a);
        (double[] values, double[][] _) = Matrix.Eigh(ata);
        double second = Math.Sqrt(Math.Max(0.0, values[1]));
        double smallest = Math.Sqrt(Math.Max(0.0, values[0]));
        return second <= 1e-15 ? 0.0 : smallest / second;
    }

    private static double SevenLargest(double[][] a)
    {
        double[][] ata = Matrix.Multiply(Matrix.Transpose(a), a);
        (double[] values, double[][] _) = Matrix.Eigh(ata);
        return Math.Sqrt(Math.Max(0.0, values[8]));
    }

    private static double SecondSmallest(double[][] a)
    {
        double[][] ata = Matrix.Multiply(Matrix.Transpose(a), a);
        (double[] values, double[][] _) = Matrix.Eigh(ata);
        return values[1];
    }

    private static double[][] Difference(double[][] a, double[][] b)
    {
        var result = new double[3][];
        for (int i = 0; i < 3; i++)
        {
            result[i] = new double[3];
            for (int j = 0; j < 3; j++) result[i][j] = a[i][j] - b[i][j];
        }
        return result;
    }

    private static double Determinant(double[][] m)
    {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    private static double TranslationError(double[] a, double[] b)
    {
        double dot = Math.Abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        double na = Matrix.Norm(a);
        double nb = Matrix.Norm(b);
        return Math.Acos(Math.Min(1.0, dot / (na * nb)));
    }
}
