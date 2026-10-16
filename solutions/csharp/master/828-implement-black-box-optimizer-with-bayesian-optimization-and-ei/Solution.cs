// Task 828 - Implement black-box optimizer with Bayesian optimization and EI acquisition.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// Bayesian optimization of a black-box function with an exact Gaussian process surrogate and the analytic
// expected-improvement acquisition, implemented from the linear algebra up:
//
//   * kernel  k(a,b) = sigma_f^2 * exp(-||a-b||^2 / (2 l^2))   (squared exponential, isotropic);
//   * posterior over n observations:  Mu = K_*^T (K + sigma_n^2 I)^-1 y  and
//     Var = k(x,x) - K_*^T (K + sigma_n^2 I)^-1 K_*, both from the Cholesky factor of K + sigma_n^2 I
//     (forward/backward substitution - the inverse is never formed);
//   * expected improvement for minimisation, EI(x) = (f_best - mu) Phi(z) + sigma phi(z), z = (f_best - mu)/sigma,
//     maximised over a dense grid, one observation per iteration and a refit after every observation.
//
// Verified invariants (each recomputed here, never assumed):
//   * Phi comes from a Gauss-Legendre quadrature of the standard normal density and agrees to 1e-12 with the
//     published constants (0.5, 0.8413447460685429, 0.9750021048517795, ...);
//   * the Cholesky factor satisfies L L^T = K + sigma_n^2 I to 1e-12, and its log-determinant agrees with an
//     independent LU factorisation with partial pivoting;
//   * the posterior mean and variance agree to 1e-9 with a *separately implemented* dense Gaussian elimination
//     solve of the same system - a different algorithm, not a restatement of the same arithmetic;
//   * the posterior variance is non-negative, never exceeds the prior variance, and for a single observation
//     equals the closed form sigma_f^2 sigma_n^2 / (sigma_f^2 + sigma_n^2) at the observation;
//   * a near-noiseless posterior interpolates its observations, and far from all data it relaxes to the prior;
//   * EI is non-negative, reduces to the deterministic improvement when sigma -> 0, and the analytic formula
//     agrees to 1e-6 with a kink-splitting trapezoid quadrature of E[max(f_best - f, 0)] that never uses Phi;
//   * on the black box f(x) = (x - 0.6180339887498949)^2 (exact minimum 0) the optimizer reaches < 1e-5 within
//     25 acquisitions, its regret is non-increasing, and it beats a cost-matched uniform design;
//   * repeated runs are bit-for-bit identical, and every malformed input is rejected - including the singular
//     case of duplicated inputs at zero noise, which the Cholesky factorisation detects.
//
// Documented simplification: one-dimensional input (the linear algebra is dimension-general, the demo is 1-D),
// fixed hyperparameters instead of marginal-likelihood optimisation, grid search instead of continuous
// trust-region acquisition maximisation, and a 30-evaluation budget. The GP algebra, the EI identity and the
// sequential loop are the real algorithms; production systems add hyperparameter fitting, batch (q-EI)
// suggestions and constrained or parallel acquisition.
using System;
using System.Collections.Generic;

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
}

/** Standard normal pdf in closed form and cdf by high-order Gauss-Legendre quadrature of the density. */
internal static class Normal
{
    private static readonly double[] Nodes =
    {
        -0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363
    };

    private static readonly double[] Weights =
    {
        0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763
    };

    /**
     * Integral of the standard normal density from lo to hi: composite 8-point Gauss-Legendre with panels of
     * half a unit. One panel alone would leave ~1e-9 of quadrature error at |z| ~ 2; the composite rule is exact
     * to ~1e-15, which is what the constants below are checked against.
     */
    internal static double DensityIntegral(double lo, double hi)
    {
        if (hi <= lo) return 0.0;
        int panels = Math.Max(1, (int)Math.Ceiling((hi - lo) / 0.5));
        double h = (hi - lo) / panels;
        double total = 0.0;
        for (int p = 0; p < panels; p++)
        {
            double a = lo + p * h;
            double half = 0.5 * h;
            double mid = a + half;
            double sum = 0.0;
            for (int i = 0; i < Nodes.Length; i++) sum += Weights[i] * Pdf(mid + half * Nodes[i]);
            total += sum * half;
        }
        return total;
    }

    internal static double Pdf(double z) => Math.Exp(-0.5 * z * z) * 0.3989422804014327;

    internal static double Cdf(double z)
    {
        if (z <= -12.0) return 0.0;
        if (z >= 12.0) return 1.0;
        double tail = DensityIntegral(0.0, Math.Abs(z));
        return z >= 0.0 ? 0.5 + tail : 0.5 - tail;
    }
}

/** Exact GP regression: squared-exponential kernel, fixed hyperparameters, Cholesky-based posterior. */
internal sealed class Gp
{
    private readonly List<double[]> _xs = new List<double[]>();
    private readonly List<double> _ys = new List<double>();
    private double[] _alpha = Array.Empty<double>();
    private double[] _lower = Array.Empty<double>();
    private int _n;

    internal double LengthScale { get; }
    internal double SignalVariance { get; }
    internal double NoiseVariance { get; }
    internal int Count => _n;
    internal double[] Lower => _lower;
    internal IReadOnlyList<double> Observations => _ys;

    internal Gp(double lengthScale, double signalVariance, double noiseVariance)
    {
        if (lengthScale <= 0.0) throw new ArgumentException("the length scale must be positive", nameof(lengthScale));
        if (signalVariance <= 0.0) throw new ArgumentException("the signal variance must be positive", nameof(signalVariance));
        if (noiseVariance < 0.0) throw new ArgumentException("the noise variance must not be negative", nameof(noiseVariance));
        LengthScale = lengthScale;
        SignalVariance = signalVariance;
        NoiseVariance = noiseVariance;
    }

    internal double Kernel(double[] a, double[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("kernel arguments must have the same dimension");
        double d = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            d += diff * diff;
        }
        return SignalVariance * Math.Exp(-0.5 * d / (LengthScale * LengthScale));
    }

    /** Builds the posterior from scratch; duplicated inputs at zero noise make the kernel singular. */
    internal void Fit(IReadOnlyList<double[]> xs, IReadOnlyList<double> ys)
    {
        if (xs.Count != ys.Count) throw new ArgumentException("xs and ys must have the same length");
        // Copy first: callers may hand in the internal lists (Add does), and clearing them in place would
        // silently empty the new posterior.
        var newXs = new List<double[]>(xs);
        var newYs = new List<double>(ys);
        _xs.Clear();
        _ys.Clear();
        _xs.AddRange(newXs);
        _ys.AddRange(newYs);
        _n = _xs.Count;
        if (_n == 0)
        {
            _alpha = Array.Empty<double>();
            _lower = Array.Empty<double>();
            return;
        }
        _lower = Cholesky(SystemMatrix(), _n);
        var target = new double[_n];
        for (int i = 0; i < _n; i++) target[i] = _ys[i];
        _alpha = SolveFromCholesky(_lower, target, _n);
    }

    internal void Add(double[] x, double y)
    {
        _xs.Add(x);
        _ys.Add(y);
        Fit(_xs, _ys);
    }

    /** Lower-triangular Cholesky factor, row-major, of a symmetric positive definite matrix. */
    internal static double[] Cholesky(double[,] a, int n)
    {
        var l = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = a[i, j];
                for (int p = 0; p < j; p++) sum -= l[i * n + p] * l[j * n + p];
                if (i == j)
                {
                    if (!(sum > 0.0)) throw new ArgumentException("the kernel matrix is not positive definite");
                    l[i * n + j] = Math.Sqrt(sum);
                }
                else
                {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }
        return l;
    }

    /** Solves (L L^T) x = b with one forward and one backward substitution. */
    internal static double[] SolveFromCholesky(double[] lower, double[] b, int n)
    {
        var z = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = b[i];
            for (int p = 0; p < i; p++) sum -= lower[i * n + p] * z[p];
            z[i] = sum / lower[i * n + i];
        }
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = z[i];
            for (int p = i + 1; p < n; p++) sum -= lower[p * n + i] * x[p];
            x[i] = sum / lower[i * n + i];
        }
        return x;
    }

    /** K + sigma_n^2 I; exposed so the tests can re-derive the posterior with another solver. */
    internal double[,] SystemMatrix()
    {
        var k = new double[_n, _n];
        for (int i = 0; i < _n; i++)
        {
            for (int j = 0; j < _n; j++) k[i, j] = Kernel(_xs[i], _xs[j]);
            k[i, i] += NoiseVariance;
        }
        return k;
    }

    internal double[] CrossCovariance(double[] x)
    {
        if (_n > 0 && x.Length != _xs[0].Length) throw new ArgumentException("dimension mismatch");
        var v = new double[_n];
        for (int i = 0; i < _n; i++) v[i] = Kernel(_xs[i], x);
        return v;
    }

    internal double Mean(double[] x)
    {
        if (_n == 0) return 0.0;
        double[] v = CrossCovariance(x);
        double s = 0.0;
        for (int i = 0; i < _n; i++) s += v[i] * _alpha[i];
        return s;
    }

    internal double Variance(double[] x)
    {
        if (_n == 0) return SignalVariance;
        double[] v = CrossCovariance(x);
        var z = new double[_n];
        for (int i = 0; i < _n; i++)
        {
            double sum = v[i];
            for (int p = 0; p < i; p++) sum -= _lower[i * _n + p] * z[p];
            z[i] = sum / _lower[i * _n + i];
        }
        double s = 0.0;
        for (int i = 0; i < _n; i++) s += z[i] * z[i];
        return Math.Max(0.0, SignalVariance - s);
    }

    internal double LogDeterminant()
    {
        double s = 0.0;
        for (int i = 0; i < _n; i++) s += Math.Log(_lower[i * _n + i]);
        return 2.0 * s;
    }

    internal double LogMarginalLikelihood()
    {
        if (_n == 0) return 0.0;
        double quadratic = 0.0;
        for (int i = 0; i < _n; i++) quadratic += _ys[i] * _alpha[i];
        return -0.5 * (quadratic + LogDeterminant() + _n * Math.Log(2.0 * Math.PI));
    }
}

internal sealed class BoResult
{
    internal double[] Trajectory = Array.Empty<double>();
    internal double Best;
    internal double BestPoint;
    internal int Evaluations;
}

public static class Solution
{
    private const double Golden = 0.6180339887498949;
    private const int GridPoints = 4097;

    /** The black box: a smooth function whose exact minimum is 0 at x = Golden. */
    private static double BlackBox(double x) => (x - Golden) * (x - Golden);

    /** Analytic expected improvement for minimisation. */
    private static double ExpectedImprovement(double best, double mean, double sigma)
    {
        if (sigma <= 1e-12) return Math.Max(0.0, best - mean);
        double z = (best - mean) / sigma;
        return (best - mean) * Normal.Cdf(z) + sigma * Normal.Pdf(z);
    }

    /**
     * Independent oracle for EI: E[max(best - f, 0)] for f ~ N(mean, sigma^2), by the trapezoid rule on a grid
     * split exactly at the kink t = (best - mean)/sigma. It never uses Phi, so it checks the closed form instead
     * of restating it; the fixed step keeps the quadrature error near 4e-7.
     */
    private static double ExpectedImprovementQuadrature(double best, double mean, double sigma)
    {
        const double step = 5e-4;
        double kink = (best - mean) / sigma;
        double lo = -9.0;
        double hi = Math.Min(9.0, kink);
        if (hi <= lo) return 0.0;
        int steps = Math.Max(2, (int)Math.Ceiling((hi - lo) / step));
        double h = (hi - lo) / steps;
        double total = 0.0;
        double previous = 0.0;
        for (int i = 0; i <= steps; i++)
        {
            double t = lo + i * h;
            double value = Math.Max(0.0, best - (mean + sigma * t)) * Normal.Pdf(t);
            if (i > 0) total += 0.5 * (previous + value) * h;
            previous = value;
        }
        return total;
    }

    /** Dense Gaussian elimination with partial pivoting: the independent solver used to check the Cholesky path. */
    private static double[] DenseSolve(double[,] a, double[] b, int n)
    {
        var m = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) m[i, j] = a[i, j];
            m[i, n] = b[i];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row, col]) > Math.Abs(m[pivot, col])) pivot = row;
            }
            if (m[pivot, col] == 0.0) throw new ArgumentException("singular system");
            for (int j = 0; j <= n; j++)
            {
                double t = m[col, j];
                m[col, j] = m[pivot, j];
                m[pivot, j] = t;
            }
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row, col] / m[col, col];
                for (int j = col; j <= n; j++) m[row, j] -= factor * m[col, j];
            }
        }
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = m[i, n];
            for (int j = i + 1; j < n; j++) sum -= m[i, j] * x[j];
            x[i] = sum / m[i, i];
        }
        return x;
    }

    /** log|A| by LU with partial pivoting: an independent check of the Cholesky log-determinant. */
    private static double DenseLogDet(double[,] a, int n)
    {
        var m = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) m[i, j] = a[i, j];
        }
        double logSum = 0.0;
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(m[row, col]) > Math.Abs(m[pivot, col])) pivot = row;
            }
            if (m[pivot, col] == 0.0) throw new ArgumentException("singular system");
            if (pivot != col)
            {
                for (int j = 0; j < n; j++)
                {
                    double t = m[col, j];
                    m[col, j] = m[pivot, j];
                    m[pivot, j] = t;
                }
            }
            logSum += Math.Log(Math.Abs(m[col, col]));
            for (int row = col + 1; row < n; row++)
            {
                double factor = m[row, col] / m[col, col];
                for (int j = col; j < n; j++) m[row, j] -= factor * m[col, j];
            }
        }
        return logSum;
    }

    /**
     * The sequential loop: a drawn initial design, then `iterations` rounds of (maximise EI on the grid, evaluate,
     * refit). The incumbent is carried forward, so its regret can only decrease.
     */
    private static BoResult RunBo(ulong seed, int designSize, int iterations)
    {
        var rng = seed;
        var optimizer = new Gp(0.2, 1.0, 1e-8);
        var initial = new List<double[]>();
        var initialY = new List<double>();
        var trajectory = new List<double>();
        double best = double.PositiveInfinity;
        for (int i = 0; i < designSize; i++)
        {
            double x = SplitMix64.NextDouble(ref rng);
            trajectory.Add(x);
            initial.Add(new[] { x });
            double value = BlackBox(x);
            initialY.Add(value);
            best = Math.Min(best, value);
        }
        optimizer.Fit(initial, initialY);
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            double argMax = 0.0;
            double argValue = -1.0;
            for (int g = 0; g < GridPoints; g++)
            {
                double x = g / (double)(GridPoints - 1);
                double ei = ExpectedImprovement(best, optimizer.Mean(new[] { x }), Math.Sqrt(optimizer.Variance(new[] { x })));
                if (ei > argValue)
                {
                    argValue = ei;
                    argMax = x;
                }
            }
            double value = BlackBox(argMax);
            trajectory.Add(argMax);
            optimizer.Add(new[] { argMax }, value);
            best = Math.Min(best, value);
        }
        var result = new BoResult { Trajectory = trajectory.ToArray(), Best = best, Evaluations = optimizer.Count };
        double bestPoint = 0.0;
        double bestValue = double.PositiveInfinity;
        foreach (double x in result.Trajectory)
        {
            double v = BlackBox(x);
            if (v < bestValue)
            {
                bestValue = v;
                bestPoint = x;
            }
        }
        result.BestPoint = bestPoint;
        return result;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        // 1-3. The normal cdf matches published constants, the density integrates to one, and it is exact at zero.
        double[] probes = { 0.0, 1.0, -1.0, 1.96, -3.0 };
        double[] known = { 0.5, 0.8413447460685429, 0.1586552539314571, 0.9750021048517795, 0.0013498980316301 };
        double cdfError = 0.0;
        for (int i = 0; i < probes.Length; i++) cdfError = Math.Max(cdfError, Math.Abs(Normal.Cdf(probes[i]) - known[i]));
        Console.WriteLine("  normal cdf: worst error against known constants=" + cdfError.ToString("E3"));
        Check(cdfError < 1e-12, "the cdf agrees with the published constants to 1e-12");
        Check(Math.Abs(Normal.DensityIntegral(-10.0, 10.0) - 1.0) < 1e-12, "the density integrates to one over +-10");
        Check(Normal.Cdf(0.0) == 0.5, "the cdf is exactly one half at zero");

        // 4-5. Kernel sanity: symmetry, self-covariance, monotone decay.
        var gp = new Gp(0.2, 1.0, 1e-6);
        var p1 = new[] { 0.1 };
        var p2 = new[] { 0.55 };
        Check(gp.Kernel(p1, p1) == 1.0 && gp.Kernel(p1, p2) == gp.Kernel(p2, p1),
            "the kernel is symmetric with k(x,x) = sigma_f^2");
        Check(gp.Kernel(p1, new[] { 0.2 }) > gp.Kernel(p1, new[] { 0.9 }), "the kernel decays with distance");

        // 6-7. Prior: no data means mean zero and the full signal variance, so EI is the closed prior form.
        var prior = new Gp(0.2, 1.0, 1e-6);
        prior.Fit(new List<double[]>(), new List<double>());
        Check(prior.Mean(p1) == 0.0 && prior.Variance(p1) == 1.0, "the empty posterior is the prior");
        double priorEi = ExpectedImprovement(0.0, prior.Mean(p1), Math.Sqrt(prior.Variance(p1)));
        Check(Math.Abs(priorEi - 1.0 / Math.Sqrt(2.0 * Math.PI)) < 1e-12,
            "EI at the prior mean is sigma / sqrt(2 pi) in closed form");

        // 8-9. Cholesky: L L^T reproduces the system matrix, and the log-determinant matches LU.
        var rng = 0xA1B2UL;
        int n = 7;
        var xs = new List<double[]>();
        var ys = new List<double>();
        for (int i = 0; i < n; i++)
        {
            double x = 0.08 + 0.13 * i;
            xs.Add(new[] { x });
            ys.Add(BlackBox(x) + 0.01 * (SplitMix64.NextDouble(ref rng) - 0.5));
        }
        var fit = new Gp(0.25, 1.0, 1e-4);
        fit.Fit(xs, ys);
        double[,] system = fit.SystemMatrix();
        double[] lower = fit.Lower;
        double worstFactor = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < n; p++) sum += lower[i * n + p] * lower[j * n + p];
                worstFactor = Math.Max(worstFactor, Math.Abs(sum - system[i, j]));
            }
        }
        Console.WriteLine("  cholesky: worst |L L^T - (K + noise I)| = " + worstFactor.ToString("E3")
            + " logdet=" + fit.LogDeterminant().ToString("F6"));
        Check(worstFactor < 1e-12, "the Cholesky factor reproduces the kernel matrix");
        Check(Math.Abs(fit.LogDeterminant() - DenseLogDet(system, n)) < 1e-9,
            "the Cholesky log-determinant matches an independent LU factorisation");

        // 10-11. Posterior against an independent dense solver, at training points and at fresh points.
        var denseAlpha = DenseSolve(system, ys.ToArray(), n);
        var testPoints = new[] { 0.0, 0.2, 0.35, Golden, 0.9, 1.0 };
        double worstMean = 0.0;
        for (int i = 0; i < n; i++)
        {
            double[] v = fit.CrossCovariance(xs[i]);
            double dense = 0.0;
            for (int j = 0; j < n; j++) dense += v[j] * denseAlpha[j];
            worstMean = Math.Max(worstMean, Math.Abs(dense - fit.Mean(xs[i])));
        }
        foreach (double x in testPoints)
        {
            double[] v = fit.CrossCovariance(new[] { x });
            double dense = 0.0;
            for (int j = 0; j < n; j++) dense += v[j] * denseAlpha[j];
            worstMean = Math.Max(worstMean, Math.Abs(dense - fit.Mean(new[] { x })));
        }
        Console.WriteLine("  posterior mean: worst difference from the dense solver=" + worstMean.ToString("E3"));
        Check(worstMean < 1e-9, "the posterior mean matches the independent dense solve");

        double worstVariance = 0.0;
        foreach (double x in testPoints)
        {
            double[] v = fit.CrossCovariance(new[] { x });
            double[] solution = DenseSolve(system, v, n);
            double dense = 1.0;
            for (int j = 0; j < n; j++) dense -= v[j] * solution[j];
            worstVariance = Math.Max(worstVariance, Math.Abs(dense - fit.Variance(new[] { x })));
        }
        Console.WriteLine("  posterior variance: worst difference from the dense solver=" + worstVariance.ToString("E3"));
        Check(worstVariance < 1e-9, "the posterior variance matches the independent dense solve");

        // 12-13. Variance bounds, and the closed form at the observation of a one-point posterior.
        bool bounded = true;
        for (int i = 0; i <= 400; i++)
        {
            double v = fit.Variance(new[] { i / 400.0 });
            if (v < -1e-15 || v > 1.0 + 1e-12) bounded = false;
        }
        Check(bounded, "the posterior variance stays inside [0, sigma_f^2]");
        var one = new Gp(0.3, 1.0, 1e-4);
        one.Fit(new List<double[]> { new[] { 0.4 } }, new List<double> { 0.25 });
        Check(Math.Abs(one.Variance(new[] { 0.4 }) - 1e-4 / 1.0001) < 1e-12,
            "with one observation the variance there is sigma_f^2 sigma_n^2 / (sigma_f^2 + sigma_n^2)");

        // 14. With negligible noise the posterior interpolates the observations.
        var tight = new Gp(0.25, 1.0, 1e-10);
        tight.Fit(xs, ys);
        double worstInterpolation = 0.0;
        for (int i = 0; i < n; i++) worstInterpolation = Math.Max(worstInterpolation, Math.Abs(tight.Mean(xs[i]) - ys[i]));
        Console.WriteLine("  interpolation: worst |mu(x_i) - y_i| = " + worstInterpolation.ToString("E3"));
        Check(worstInterpolation < 1e-6, "a near-noiseless posterior interpolates its observations");

        // 15-17. The analytic EI agrees with quadrature, is non-negative, and degenerates correctly.
        double worstEi = 0.0;
        for (int i = 0; i < 12; i++)
        {
            double best = -0.2 + 0.1 * i;
            double mean = 0.05 * i - 0.3;
            double sigma = 0.05 + 0.04 * i;
            double analytic = ExpectedImprovement(best, mean, sigma);
            double quadrature = ExpectedImprovementQuadrature(best, mean, sigma);
            worstEi = Math.Max(worstEi, Math.Abs(analytic - quadrature) / Math.Max(1e-6, quadrature));
        }
        Console.WriteLine("  EI: worst relative difference from the quadrature oracle=" + worstEi.ToString("E3"));
        Check(worstEi < 1e-6, "the analytic EI matches the quadrature oracle");
        bool nonNegative = true;
        for (int i = -100; i <= 100; i += 10)
        {
            double mean = i / 50.0;
            if (ExpectedImprovement(0.0, mean, 0.3) < 0.0) nonNegative = false;
            if (ExpectedImprovementQuadrature(0.0, mean, 0.3) < -1e-12) nonNegative = false;
        }
        Check(nonNegative, "EI is non-negative everywhere");
        Check(ExpectedImprovement(-0.5, 0.0, 0.0) == 0.0 && ExpectedImprovement(0.5, 0.0, 0.0) == 0.5,
            "with zero uncertainty EI is the deterministic improvement clamped at zero");

        // 18-19. Acquisition behaviour on a fitted posterior: uncertainty and EI prefer the unexplored point.
        double varianceAtObservation = fit.Variance(xs[0]);
        double varianceAtFresh = fit.Variance(new[] { -0.5 });
        Check(varianceAtFresh > varianceAtObservation, "an unexplored point has more posterior uncertainty");
        double eiFresh = ExpectedImprovement(fit.Observations[0], fit.Mean(new[] { -0.5 }), Math.Sqrt(varianceAtFresh));
        double eiObserved = ExpectedImprovement(fit.Observations[0], fit.Mean(xs[0]), Math.Sqrt(varianceAtObservation));
        Check(eiFresh > eiObserved, "so EI prefers the unexplored point over the incumbent's location");

        // 20-23. Sequential optimization: bookkeeping, optimum, regret monotonicity.
        var bo = RunBo(0xB0B0UL, 5, 25);
        Console.WriteLine("  optimization: best after " + bo.Evaluations + " evaluations=" + bo.Best.ToString("E6")
            + " at x=" + bo.BestPoint.ToString("F6") + " (exact minimum 0 at " + Golden.ToString("F6") + ")");
        Check(bo.Evaluations == 30, "every evaluation produced exactly one observation");
        Check(bo.Best < 1e-5, "Bayesian optimization reaches a value below 1e-5 within 25 acquisitions");
        Check(Math.Abs(bo.BestPoint - Golden) < 1e-3, "the winning point is within 1e-3 of the true minimiser");
        double initialBest = double.PositiveInfinity;
        double finalBest = double.PositiveInfinity;
        for (int i = 0; i < bo.Trajectory.Length; i++)
        {
            double v = BlackBox(bo.Trajectory[i]);
            finalBest = Math.Min(finalBest, v);
            if (i < 5) initialBest = Math.Min(initialBest, v);
        }
        Console.WriteLine("  regret: initial design=" + initialBest.ToString("E6")
            + " after acquisitions=" + bo.Best.ToString("E6"));
        Check(bo.Best == finalBest, "the reported incumbent is the minimum over the whole trajectory");
        Check(bo.Best < initialBest, "the EI acquisitions strictly improve on the initial design");

        // 24. Same budget, no model: a uniform design of the same size is far worse.
        var uniformRng = 0x5EEDUL;
        double uniformBest = double.PositiveInfinity;
        for (int i = 0; i < bo.Evaluations; i++)
        {
            uniformBest = Math.Min(uniformBest, BlackBox(SplitMix64.NextDouble(ref uniformRng)));
        }
        Console.WriteLine("  cost-matched uniform design best=" + uniformBest.ToString("E6"));
        Check(bo.Best < uniformBest, "the surrogate-directed search beats a cost-matched uniform design");

        // 25. Determinism of the whole loop.
        var again = RunBo(0xB0B0UL, 5, 25);
        bool identical = again.Trajectory.Length == bo.Trajectory.Length && again.Best == bo.Best;
        for (int i = 0; identical && i < bo.Trajectory.Length; i++)
        {
            if (again.Trajectory[i] != bo.Trajectory[i]) identical = false;
        }
        Check(identical, "repeating the run reproduces the trajectory bit for bit");

        // 26. Marginal likelihood: the Cholesky quadratic form matches an independent dense solve.
        double[] yVector = ys.ToArray();
        double[] denseSolution = DenseSolve(system, yVector, n);
        double quadratic = 0.0;
        for (int i = 0; i < n; i++) quadratic += yVector[i] * denseSolution[i];
        double expectedLogLikelihood = -0.5 * (quadratic + DenseLogDet(system, n) + n * Math.Log(2.0 * Math.PI));
        Console.WriteLine("  log marginal likelihood=" + fit.LogMarginalLikelihood().ToString("F9"));
        Check(Math.Abs(fit.LogMarginalLikelihood() - expectedLogLikelihood) < 1e-9,
            "the log marginal likelihood matches the dense-solve expression");

        // 27-30. Input validation, including the singular duplicate-input case.
        Check(Throws(() => new Gp(0.0, 1.0, 0.0)), "a non-positive length scale is rejected");
        Check(Throws(() => new Gp(0.2, -1.0, 0.0)), "a non-positive signal variance is rejected");
        Check(Throws(() => new Gp(0.2, 1.0, -1e-9)), "a negative noise variance is rejected");
        Check(Throws(() => fit.Fit(new List<double[]> { new[] { 0.1 } }, new List<double>())),
            "mismatched observation lengths are rejected");
        Check(Throws(() => new Gp(0.2, 1.0, 0.0).Fit(
            new List<double[]> { new[] { 0.3 }, new[] { 0.3 } }, new List<double> { 1.0, 1.0 })),
            "duplicated inputs at zero noise are singular and rejected");
        Check(Throws(() => new Gp(0.2, 1.0, 0.0).Fit(
            new List<double[]> { new[] { 0.3 } }, new List<double> { 1.0, 2.0 })), "extra targets are rejected");
        Check(Throws(() => gp.Kernel(new[] { 0.1 }, new[] { 0.1, 0.2 })), "a kernel dimension mismatch is rejected");

        // 31-34. A single observation: interpolated at the point, prior weight everywhere else.
        var single = new Gp(0.3, 1.0, 1e-6);
        single.Fit(new List<double[]> { new[] { 0.4 } }, new List<double> { 1.25 });
        Check(single.Count == 1, "one observation means one posterior row");
        Check(Math.Abs(single.Mean(new[] { 0.4 }) - 1.25 / 1.000001) < 1e-12,
            "the prediction at the observation shrinks by the closed-form noise ratio sigma_f^2/(sigma_f^2+sigma_n^2)");
        var sharp = new Gp(0.3, 1.0, 1e-12);
        sharp.Fit(new List<double[]> { new[] { 0.4 } }, new List<double> { 1.25 });
        Check(Math.Abs(sharp.Mean(new[] { 0.4 }) - 1.25) < 1e-9, "a single noiseless observation is interpolated");
        Check(single.Variance(new[] { 0.4 }) < single.Variance(new[] { 0.9 }),
            "the posterior is more confident at the observation than far away");
        Check(Math.Abs(single.Mean(new[] { 5.0 })) < 1e-3, "far from all data the posterior falls back to the prior mean");

        Console.WriteLine("PASS");
    }
}
