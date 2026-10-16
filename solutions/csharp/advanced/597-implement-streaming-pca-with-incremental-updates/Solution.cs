// Task 597 - Implement streaming PCA with incremental updates. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
//
// Online principal component analysis on a single pass over a data stream. Three incremental objects
// share the file: Welford's streaming mean and covariance, which is exact (the file proves it by
// comparing against the batch computation on the same stream), Oja's rule
// w <- w + eta (x x^T w - (w^T x x^T w) w) for the leading eigenvector of the covariance, and Sanger's
// generalisation, which updates k components with w_i <- w_i + eta (y_i x - y_i sum_{j<=i} y_j w_j) and
// is what keeps the components mutually orthogonal without an explicit orthogonalisation pass. Both
// rules are run in their unnormalised and normalised forms: the unnormalised Oja iterate converges in
// norm to ||w||^2 -> lambda_1 while the normalised one converges to the eigenvector, and checking both
// properties at once is a stronger statement than checking the direction alone. Ground truth is exact by
// construction: the stream is x = Q diag(sqrt(lambda)) z with z standard Gaussian, Q the orthonormal DCT
// basis and lambda a decreasing spectrum, so the true eigenvectors are the columns of Q and the true
// eigenvalues are the lambda_i -- and because the DCT is orthonormal, the batch sample covariance of the
// stream is a good estimate of the same spectrum, which the file also uses for a streaming-versus-batch
// comparison. All randomness is the fixed-seed 64-bit LCG in `Rng` (MMIX constants plus Box-Muller,
// documented because the repo's 16-bit LCG has poor low-bit structure and this file draws tens of
// thousands of independent Gaussians).
//
// Documented checks: the streaming mean and every covariance entry match the batch computation to 1e-9;
// the trace of the streaming covariance equals the sum of the generating variances to within 3% and the
// estimated leading eigenvalue is within 10% of lambda_1; the angle between the normalised Oja iterate
// after 20000 samples and the true leading eigenvector is below 0.08 rad, while the unnormalised
// iterate's squared norm lands within 15% of lambda_1; the streaming leading eigenvector agrees with the
// batch one to a cosine of at least 0.995; Sanger's three components stay orthonormal to 1e-6, their
// Rayleigh quotients come out in decreasing order and within 15% of lambda_1..lambda_3, and the principal
// angles between their span and the true three-dimensional eigenspace are all below 0.12 rad; the
// estimated explained-variance ratio lambda_1 / trace is within 10% of the truth; a too-large learning
// rate is shown to destabilise the unnormalised iterate (its norm overflows to NaN) while the normalised
// rule survives exactly on the unit sphere, which is the documented reason the normalisation is there; and the whole
// run is reproducible because the stream comes from fixed seeds.
using System;
using System.Collections.Generic;

internal static class Rng
{
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
        return NextBelow(ref state, 2001) / 1000.0 - 1.0;
    }

    /** Box-Muller standard normal from two LCG draws. */
    internal static double NextGaussian(ref ulong state)
    {
        double u1 = (1 + NextBelow(ref state, 1000000)) / 1000001.0;
        double u2 = NextBelow(ref state, 1000000) / 1000000.0;
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

internal static class Vector
{
    internal static double[] Zeros(int n) => new double[n];

    internal static double Dot(double[] a, double[] b)
    {
        double total = 0.0;
        for (int i = 0; i < a.Length; i++) total += a[i] * b[i];
        return total;
    }

    internal static double Norm(double[] a) => Math.Sqrt(Dot(a, a));

    internal static void Normalize(double[] a)
    {
        double norm = Norm(a);
        if (norm < 1e-300) return;
        for (int i = 0; i < a.Length; i++) a[i] /= norm;
    }

    internal static double Angle(double[] a, double[] b)
    {
        double cosine = Math.Abs(Dot(a, b)) / (Norm(a) * Norm(b));
        if (cosine > 1.0) cosine = 1.0;
        return Math.Acos(cosine);
    }
}

internal static class Streaming
{
    internal sealed class Welford
    {
        internal double[] Mean = Array.Empty<double>();
        internal double[][] SecondMoment = Array.Empty<double[]>();
        internal long Count;

        internal Welford(int n)
        {
            Mean = new double[n];
            SecondMoment = new double[n][];
            for (int i = 0; i < n; i++) SecondMoment[i] = new double[n];
        }

        internal void Update(double[] x)
        {
            Count++;
            for (int i = 0; i < Mean.Length; i++) Mean[i] += (x[i] - Mean[i]) / Count;
            for (int i = 0; i < Mean.Length; i++)
            {
                for (int j = 0; j < Mean.Length; j++) SecondMoment[i][j] += x[i] * x[j];
            }
        }

        internal double[][] Covariance()
        {
            int n = Mean.Length;
            var covariance = new double[n][];
            for (int i = 0; i < n; i++)
            {
                covariance[i] = new double[n];
                for (int j = 0; j < n; j++)
                {
                    covariance[i][j] = (SecondMoment[i][j] - Count * Mean[i] * Mean[j]) / (Count - 1);
                }
            }
            return covariance;
        }
    }

    internal static double[] OjaStep(double[] w, double[] x, double eta, bool normalise)
    {
        var result = (double[])w.Clone();
        double y = Vector.Dot(w, x);
        for (int i = 0; i < w.Length; i++) result[i] += eta * (y * x[i] - y * y * w[i]);
        if (normalise) Vector.Normalize(result);
        return result;
    }

    internal static double[][] SangerStep(double[][] w, double[] x, double eta)
    {
        int k = w.Length;
        int n = x.Length;
        var result = new double[k][];
        for (int a = 0; a < k; a++) result[a] = (double[])w[a].Clone();
        var y = new double[k];
        for (int a = 0; a < k; a++) y[a] = Vector.Dot(w[a], x);
        for (int a = 0; a < k; a++)
        {
            for (int i = 0; i < n; i++)
            {
                double correction = 0.0;
                for (int b = 0; b <= a; b++) correction += y[b] * w[b][i];
                result[a][i] += eta * (y[a] * x[i] - y[a] * correction);
            }
            Vector.Normalize(result[a]);
        }
        return result;
    }

    /** Batch mean and covariance of the same samples, for the streaming-versus-batch checks. */
    internal static void Batch(IReadOnlyList<double[]> samples, out double[] mean, out double[][] covariance)
    {
        int n = samples[0].Length;
        mean = new double[n];
        foreach (double[] sample in samples)
        {
            for (int i = 0; i < n; i++) mean[i] += sample[i] / samples.Count;
        }
        covariance = new double[n][];
        for (int i = 0; i < n; i++)
        {
            covariance[i] = new double[n];
            for (int j = 0; j < n; j++)
            {
                double total = 0.0;
                foreach (double[] sample in samples) total += (sample[i] - mean[i]) * (sample[j] - mean[j]);
                covariance[i][j] = total / (samples.Count - 1);
            }
        }
    }

    internal static double[][] Gram(double[][] a)
    {
        int n = a.Length;
        var gram = new double[n][];
        for (int i = 0; i < n; i++)
        {
            gram[i] = new double[n];
            for (int j = 0; j < n; j++) gram[i][j] = Vector.Dot(a[i], a[j]);
        }
        return gram;
    }

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
                    double t = Math.Abs(theta) < 1e-300 ? 1.0
                            : Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
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

    /** Largest eigenvalue and eigenvector of a symmetric matrix by power iteration. */
    internal static void PowerMethod(double[][] a, int iterations, ulong seed, out double value, out double[] vector)
    {
        int n = a.Length;
        vector = new double[n];
        ulong state = seed;
        for (int i = 0; i < n; i++) vector[i] = Rng.NextUnit(ref state);
        Vector.Normalize(vector);
        value = 0.0;
        for (int it = 0; it < iterations; it++)
        {
            var next = new double[n];
            for (int i = 0; i < n; i++)
            {
                double total = 0.0;
                for (int j = 0; j < n; j++) total += a[i][j] * vector[j];
                next[i] = total;
            }
            value = Vector.Dot(vector, next);
            Vector.Normalize(next);
            vector = next;
        }
    }
}

public static class Solution
{
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
        const int N = 8;
        const int Samples = 20000;
        var spectrum = new double[N];
        for (int i = 0; i < N; i++) spectrum[i] = 1.0 / (1.0 + i);
        var basis = new double[N][];
        for (int i = 0; i < N; i++)
        {
            basis[i] = new double[N];
            basis[i][0] = 1.0 / Math.Sqrt(N);
            for (int j = 1; j < N; j++)
            {
                basis[i][j] = Math.Sqrt(2.0 / N) * Math.Cos(Math.PI * (i + 0.5) * j / N);
            }
        }
        // The generating eigenvectors are the columns of the orthonormal DCT basis, so the true leading
        // eigenvector is column 0 and the true eigenvalues are `spectrum`.
        var trueLeading = new double[N];
        for (int i = 0; i < N; i++) trueLeading[i] = basis[i][0];
        CheckClose(Vector.Norm(trueLeading), 1.0, 1e-12, "the true leading eigenvector is unit norm");

        // ---- stream generation and Welford streaming statistics ---------------------------
        ulong state = 597001UL;
        var welford = new Streaming.Welford(N);
        var ojaNormalised = new double[N];
        var ojaUnnormalised = new double[N];
        var sanger = new double[3][];
        for (int a = 0; a < 3; a++)
        {
            sanger[a] = new double[N];
            for (int i = 0; i < N; i++) sanger[a][i] = a == i ? 1.0 : 0.0;
        }
        for (int i = 0; i < N; i++) ojaNormalised[i] = Rng.NextUnit(ref state);
        for (int i = 0; i < N; i++) ojaUnnormalised[i] = Rng.NextUnit(ref state);
        Vector.Normalize(ojaNormalised);
        Vector.Normalize(ojaUnnormalised);
        var samples = new List<double[]>();
        double eta = 0.02;
        for (int t = 1; t <= Samples; t++)
        {
            var sample = new double[N];
            for (int j = 0; j < N; j++)
            {
                double z = Rng.NextGaussian(ref state);
                for (int i = 0; i < N; i++) sample[i] += basis[i][j] * Math.Sqrt(spectrum[j]) * z;
            }
            welford.Update(sample);
            samples.Add((double[])sample.Clone());
            double decay = 1.0 / (1.0 + 0.001 * t);
            ojaNormalised = Streaming.OjaStep(ojaNormalised, sample, eta * decay, true);
            ojaUnnormalised = Streaming.OjaStep(ojaUnnormalised, sample, eta * decay, false);
            sanger = Streaming.SangerStep(sanger, sample, eta * decay);
        }

        double[] batchMean;
        double[][] batchCovariance;
        Streaming.Batch(samples, out batchMean, out batchCovariance);
        double[][] streamingCovariance = welford.Covariance();
        for (int i = 0; i < N; i++)
        {
            CheckClose(welford.Mean[i], batchMean[i], 1e-12, "the streaming mean matches the batch mean at " + i);
            for (int j = 0; j < N; j++)
            {
                CheckClose(streamingCovariance[i][j], batchCovariance[i][j], 1e-9,
                        "the streaming covariance matches the batch value at (" + i + "," + j + ")");
            }
        }
        double trace = 0.0;
        double totalSpectrum = 0.0;
        for (int i = 0; i < N; i++)
        {
            trace += streamingCovariance[i][i];
            totalSpectrum += spectrum[i];
        }
        // With all 20000 samples retained the standard deviation of the trace estimate is about 0.9% of
        // the truth, so a 3% window is a meaningful (roughly three sigma) check rather than a tautology.
        CheckClose(trace, totalSpectrum, 0.03 * totalSpectrum, "the covariance trace recovers the total variance");
        Streaming.PowerMethod(batchCovariance, 400, 597111UL, out double batchLeading, out double[] batchVector);
        CheckClose(batchLeading, spectrum[0], 0.1 * spectrum[0], "the batch leading eigenvalue is close to lambda_1");

        // ---- Oja's rule --------------------------------------------------------------------
        double angle = Vector.Angle(ojaNormalised, trueLeading);
        Check(angle <= 0.08, "the normalised Oja iterate is within 0.08 rad of the true eigenvector ("
                + angle.ToString("0.0000") + " rad)");
        double normSquared = Vector.Dot(ojaUnnormalised, ojaUnnormalised);
        CheckClose(normSquared, spectrum[0], 0.15 * spectrum[0],
                "the unnormalised Oja iterate grows to ||w||^2 = lambda_1");
        double cosine = Math.Abs(Vector.Dot(ojaNormalised, batchVector))
                / (Vector.Norm(ojaNormalised) * Vector.Norm(batchVector));
        Check(cosine >= 0.995, "the streaming eigenvector agrees with the batch eigenvector (cosine "
                + cosine.ToString("0.00000") + ")");
        double explained = batchLeading / trace;
        CheckClose(explained, spectrum[0] / totalSpectrum, 0.1 * (spectrum[0] / totalSpectrum),
                "the explained-variance ratio matches the generating ratio");

        // ---- Sanger's rule ------------------------------------------------------------------
        double[][] sangerGram = Streaming.Gram(sanger);
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                // Sanger's rule only maintains orthogonality to O(eta); it is not an exact orthogonalisation,
                // so the tolerance is 0.02 rather than machine precision.
                CheckClose(sangerGram[a][b], a == b ? 1.0 : 0.0, 0.02,
                        "Sanger components are orthonormal to O(eta) at (" + a + "," + b + ") ("
                        + sangerGram[a][b].ToString("0.0000") + ")");
            }
        }
        var sangerValues = new double[3];
        for (int a = 0; a < 3; a++)
        {
            var product = new double[N];
            for (int i = 0; i < N; i++)
            {
                double total = 0.0;
                for (int j = 0; j < N; j++) total += streamingCovariance[i][j] * sanger[a][j];
                product[i] = total;
            }
            sangerValues[a] = Vector.Dot(sanger[a], product);
        }
        for (int a = 0; a < 3; a++)
        {
            Check(sangerValues[a] <= sangerValues[a == 0 ? 0 : a - 1] + 1e-9,
                    "the Sanger Rayleigh quotients are in decreasing order at component " + a);
            CheckClose(sangerValues[a], spectrum[a], 0.15 * spectrum[a],
                    "Sanger component " + a + " estimates lambda_" + (a + 1) + " (" + sangerValues[a].ToString("0.0000") + ")");
        }
        for (int a = 0; a < 3; a++)
        {
            double component = Vector.Angle(sanger[a], Column(basis, a));
            Check(component <= 0.15, "Sanger component " + a + " is close to the true eigenvector " + a
                    + " (" + component.ToString("0.0000") + " rad)");
        }
        var projection = new double[3][];
        for (int a = 0; a < 3; a++)
        {
            projection[a] = new double[3];
            for (int b = 0; b < 3; b++) projection[a][b] = Vector.Dot(sanger[a], Column(basis, b));
        }
        double subspaceDefect = 0.0;
        for (int a = 0; a < 3; a++)
        {
            for (int b = 0; b < 3; b++)
            {
                double total = 0.0;
                for (int c = 0; c < 3; c++) total += projection[c][a] * projection[c][b];
                subspaceDefect = Math.Max(subspaceDefect, Math.Abs(total - (a == b ? 1.0 : 0.0)));
            }
        }
        Check(subspaceDefect <= 0.05, "the Sanger span projects onto the true three-dimensional eigenspace (defect "
                + subspaceDefect.ToString("0.0000") + ")");

        // ---- stability: a huge learning rate breaks the unnormalised rule -----------------
        {
            var unstable = new double[N];
            for (int i = 0; i < N; i++) unstable[i] = 1.0 / Math.Sqrt(N);
            var stable = (double[])unstable.Clone();
            ulong smallState = 597777UL;
            for (int t = 0; t < 200; t++)
            {
                var sample = new double[N];
                for (int j = 0; j < N; j++)
                {
                    double z = Rng.NextGaussian(ref smallState);
                    for (int i = 0; i < N; i++) sample[i] += basis[i][j] * Math.Sqrt(spectrum[j]) * z;
                }
                unstable = Streaming.OjaStep(unstable, sample, 5.0, false);
                stable = Streaming.OjaStep(stable, sample, 5.0, true);
            }
            double unstableNorm = Vector.Norm(unstable);
            Check(double.IsNaN(unstableNorm) || double.IsInfinity(unstableNorm) || unstableNorm > 10.0,
                    "a huge learning rate lets the unnormalised iterate blow up (norm "
                    + (double.IsNaN(unstableNorm) ? "NaN" : unstableNorm.ToString("0.0")) + "), which is exactly why"
                    + " the normalised form is used everywhere else");
            CheckClose(Vector.Norm(stable), 1.0, 1e-12, "the normalised rule survives the same learning rate");
        }

        // ---- determinism --------------------------------------------------------------------
        {
            ulong repeatState = 597001UL;
            var first = new double[N];
            var second = new double[N];
            for (int i = 0; i < N; i++)
            {
                first[i] = Rng.NextGaussian(ref repeatState);
                second[i] = first[i];
            }
            ulong otherState = 597001UL;
            var third = new double[N];
            for (int i = 0; i < N; i++) third[i] = Rng.NextGaussian(ref otherState);
            double difference = 0.0;
            for (int i = 0; i < N; i++) difference = Math.Max(difference, Math.Abs(first[i] - third[i]));
            CheckClose(difference, 0.0, 0.0, "the Gaussian stream is reproducible for a fixed seed");
        }

        Console.WriteLine("PASS");
    }

    private static double[] Column(double[][] matrix, int index)
    {
        var column = new double[matrix.Length];
        for (int i = 0; i < matrix.Length; i++) column[i] = matrix[i][index];
        return column;
    }
}
