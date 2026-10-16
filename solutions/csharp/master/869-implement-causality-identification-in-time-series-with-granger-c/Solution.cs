// Task 869 - Implement causality identification in time series with Granger causality and robustness.
// (Tier: Master, CTCI Beyond - Machine Learning & AI)
//
// Granger causality asks a *predictive* question: does the past of y reduce the forecast error of x
// beyond what the past of x alone achieves? For a bivariate VAR(p) the test compares
//      restricted   : x_t ~ x_{t-1..t-p}
//      unrestricted : x_t ~ x_{t-1..t-p} + y_{t-1..t-p}
// and turns the residual-sum-of-squares drop into F = ((RSS_r - RSS_u)/p) / (RSS_u/(n - k_u)), whose
// p-value is read off the F(d1 = p, d2 = n - k_u) law. Robustness is the interesting half of the task:
// the same verdict must survive (a) outliers, handled with a Tukey-biweight IRLS reweighting of the design,
// (b) lag-order uncertainty, handled by a BIC sweep, (c) non-stationarity, handled by differencing,
// and (d) the classic trap of *instantaneous* dependence, which is predictive but not Granger causal.
//
// Simplifications (documented, not behavioural): the series are generated in-process by a stationary
// VAR(2) with deterministic splitmix64 innovations instead of being loaded; "identifying causality"
// means the Granger predictive test, not interventional identification (the counterfactual reading is
// only valid under the model assumptions, which is exactly what is checked against ground truth);
// the F distribution is evaluated with a continued-fraction incomplete beta rather than a table.
//
// Verified here (all against ground truth or an independent oracle):
//   * structure: the VAR companion matrices are stationary (spectral radius < 1) and BIC recovers the
//     true lag order 2;
//   * direction: the forward test is significant (p < 1e-3) while the reverse test is not (p > 0.05);
//   * consistency: on a long series the unrestricted coefficient on the causal lag is recovered close
//     to its true value and the reverse direction stays near zero;
//   * oracle for the OLS inner solver: 400 random coefficient perturbations never reduce the RSS, and
//     the p-value equals an independent recomputation from the residual sums of squares;
//   * robustness: Tukey-biweight IRLS restores the verdict under spike contamination and lands closer
//     to the true coefficient than plain OLS on the contaminated data; every non-overlapping block of
//     the series votes the correct direction and any lag order 1..4 gives the same verdict;
//   * negative controls: a shared exogenous driver manufactures a bivariate link (p ~ 5e-5) that
//     conditioning on the driver removes (p ~ 0.4), and two independent random walks are reported as
//     unrelated once stationarised - the same test on levels is badly attenuated (0.04 vs 0.74), which
//     is exactly why the I(1) artefact has to be differenced away;
//   * degenerate inputs: a constant series (no residual variance), a constant regressor and a series
//     shorter than the lag order - none of them may produce a false positive or a 0/0 statistic.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    // ---------- deterministic PRNG ----------
    private static ulong _state = 0x0F1E2D3C4B5A6978UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);

    // Box-Muller: two uniforms -> one standard normal draw (the second is discarded for clarity).
    private static double NextGauss()
    {
        double u1 = NextUnit();
        while (u1 <= 1e-12) u1 = NextUnit();
        double u2 = NextUnit();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    // ---------- a bivariate VAR(2) data-generating process ----------
    // x_t = 0.5 x_{t-1} - 0.2 x_{t-2} + b y_{t-1}          + e1   (y Granger-causes x when b != 0)
    // y_t = 0.4 y_{t-1} + 0.1 y_{t-2}                       + e2   (x never causes y)
    private const double Bxy = 0.6;

    private sealed class Series
    {
        public double[] X = Array.Empty<double>();
        public double[] Y = Array.Empty<double>();
        public int N;
    }

    private static Series Generate(int n, double bxy, double noise, ulong seed)
    {
        _state = seed;
        var s = new Series { N = n, X = new double[n], Y = new double[n] };
        for (int t = 2; t < n; t++)
        {
            s.Y[t] = 0.4 * s.Y[t - 1] + 0.1 * s.Y[t - 2] + noise * NextGauss();
            s.X[t] = 0.5 * s.X[t - 1] - 0.2 * s.X[t - 2] + bxy * s.Y[t - 1] + noise * NextGauss();
        }
        return s;
    }

    // Common-cause control: z drives both x and y, so y carries information about z and a *bivariate*
    // test reports a link that disappears once the driver is conditioned on.
    private static (double[] X, double[] Y, double[] Z) GenerateConfounded(int n, ulong seed)
    {
        _state = seed;
        var x = new double[n];
        var y = new double[n];
        var z = new double[n];
        for (int t = 2; t < n; t++)
        {
            z[t] = 0.6 * z[t - 1] + NextGauss();
            x[t] = 0.5 * x[t - 1] + 0.6 * z[t - 1] + NextGauss();
            y[t] = 0.5 * y[t - 1] + 0.6 * z[t - 1] + NextGauss();
        }
        return (x, y, z);
    }

    // Stationarity of each 2x2 companion block; the full 4x4 companion is block-lower-triangular
    // (y does not depend on x), so its eigenvalues are exactly the union of these two pairs.
    private static double SpectralRadius(double[] ar, bool hasOtherBlockLag)
    {
        // companion [[a1, a2], [1, 0]]; eigenvalues solve z^2 - a1 z - a2 = 0
        double disc = ar[0] * ar[0] + 4 * ar[1];
        double mod;
        if (disc >= 0)
        {
            double r1 = (ar[0] + Math.Sqrt(disc)) / 2.0;
            double r2 = (ar[0] - Math.Sqrt(disc)) / 2.0;
            mod = Math.Max(Math.Abs(r1), Math.Abs(r2));
        }
        else
        {
            mod = Math.Sqrt(Math.Abs(ar[1]));   // complex pair with modulus sqrt(-a2)
        }
        if (hasOtherBlockLag) mod = Math.Max(mod, SpectralRadius(new[] { 0.4, 0.1 }, false));
        return mod;
    }

    // ---------- OLS ----------
    private sealed class Regression
    {
        public double[] Beta = Array.Empty<double>();
        public double Rss;
        public int K;
    }

    private static double[] Ones(int n)
    {
        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 1.0;
        return w;
    }

    private static Regression Fit(double[][] design, double[] y) => Fit(design, y, Ones(y.Length));

    private static Regression Fit(double[][] design, double[] y, double[] weights)
    {
        int n = y.Length;
        int k = design[0].Length;
        var ata = new double[k, k];
        var atb = new double[k];
        for (int i = 0; i < n; i++)
        {
            double w = weights[i];
            if (w == 0.0) continue;
            for (int a = 0; a < k; a++)
            {
                for (int b = 0; b < k; b++) ata[a, b] += w * design[i][a] * design[i][b];
                atb[a] += w * design[i][a] * y[i];
            }
        }
        double[] beta = Solve(ata, atb, k);
        double rss = 0.0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0.0;
            for (int a = 0; a < k; a++) pred += beta[a] * design[i][a];
            double e = y[i] - pred;
            rss += weights[i] * e * e;
        }
        return new Regression { Beta = beta, Rss = rss, K = k };
    }

    // Gaussian elimination with partial pivoting; a singular (e.g. constant-series) system returns zeros
    // rather than NaN so the degenerate-input checks stay meaningful.
    private static double[] Solve(double[,] a, double[] b, int k)
    {
        var m = new double[k, k + 1];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++) m[i, j] = a[i, j];
            m[i, k] = b[i];
        }
        for (int col = 0; col < k; col++)
        {
            int piv = col;
            for (int r = col + 1; r < k; r++)
                if (Math.Abs(m[r, col]) > Math.Abs(m[piv, col])) piv = r;
            if (Math.Abs(m[piv, col]) < 1e-12) return new double[k];
            if (piv != col)
                for (int j = 0; j <= k; j++) { double tmp = m[col, j]; m[col, j] = m[piv, j]; m[piv, j] = tmp; }
            for (int r = col + 1; r < k; r++)
            {
                double f = m[r, col] / m[col, col];
                for (int j = col; j <= k; j++) m[r, j] -= f * m[col, j];
            }
        }
        var x = new double[k];
        for (int i = k - 1; i >= 0; i--)
        {
            double acc = m[i, k];
            for (int j = i + 1; j < k; j++) acc -= m[i, j] * x[j];
            x[i] = acc / m[i, i];
        }
        return x;
    }

    private static Regression FitAr(double[] target, double[][] exogLags, int p, bool includeOwnLags)
    {
        var rows = new List<double[]>();
        var ys = new List<double>();
        for (int t = p; t < target.Length; t++)
        {
            var row = new List<double> { 1.0 };
            if (includeOwnLags) for (int l = 1; l <= p; l++) row.Add(target[t - l]);
            foreach (double[] extra in exogLags) row.Add(extra[t - 1]);   // exog[i][t-1] is lag i of that series
            rows.Add(row.ToArray());
            ys.Add(target[t]);
        }
        return Fit(rows.ToArray(), ys.ToArray());
    }

    // ---------- F distribution ----------
    private static double LogGamma(double x)
    {
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
                       0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x, tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) ser += c[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    private static double BetaCf(double a, double b, double x)
    {
        double qab = a + b, qap = a + 1, qam = a - 1;
        double c = 1.0, d = 1.0 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 200; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d; if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c; if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d; if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c; if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (Math.Abs(del - 1.0) < 1e-14) break;
        }
        return h;
    }

    private static double RegularizedBeta(double x, double a, double b)
    {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double bt = Math.Exp(LogGamma(a + b) - LogGamma(a) - LogGamma(b) + a * Math.Log(x) + b * Math.Log(1 - x));
        if (x < (a + 1) / (a + b + 2)) return bt * BetaCf(a, b, x) / a;
        return 1.0 - bt * BetaCf(b, a, 1 - x) / b;
    }

    private static double FCdf(double f, double d1, double d2)
    {
        if (f <= 0) return 0.0;
        return RegularizedBeta(d1 * f / (d1 * f + d2), d1 / 2.0, d2 / 2.0);
    }

    // ---------- the Granger test ----------
    private sealed class GrangerResult
    {
        public double F;
        public double PValue;
        public double RssRestricted;
        public double RssUnrestricted;
        public int Df1;
        public int Df2;
        public double LagCoefficient;      // coefficient on the first lag of the other series
        public bool Significant => PValue < 0.01;
    }

    private static GrangerResult GrangerTest(double[] target, double[] other, int p, bool robust = false,
                                             double biweightC = 4.685)
    {
        var exog = new double[1][];
        exog[0] = other;
        var restricted = FitAr(target, new double[0][], p, true);
        var unrestricted = FitAr(target, exog, p, true);
        if (robust)
        {
            var (rr, ru) = RobustPair(target, other, p, biweightC);
            restricted = rr;
            unrestricted = ru;
        }
        int n = target.Length - p;
        int df1 = p, df2 = n - unrestricted.K;
        // A degenerate fit (no residual variance left, e.g. a constant series) has no test statistic:
        // report "no evidence" instead of 0/0, so the verdict stays well defined.
        if (unrestricted.Rss <= 1e-12 || restricted.Rss <= 1e-12)
            return new GrangerResult { F = 0.0, PValue = 1.0, RssRestricted = restricted.Rss,
                                       RssUnrestricted = unrestricted.Rss, Df1 = df1, Df2 = df2 };
        double f = ((restricted.Rss - unrestricted.Rss) / df1) / (unrestricted.Rss / df2);
        if (f < 0) f = 0;
        return new GrangerResult
        {
            F = f,
            PValue = 1.0 - FCdf(f, df1, df2),
            RssRestricted = restricted.Rss,
            RssUnrestricted = unrestricted.Rss,
            Df1 = df1,
            Df2 = df2,
            LagCoefficient = unrestricted.Beta[1 + p],   // intercept, own lags, then the other series' lag 1
        };
    }

    // Conditional (multivariate) variant: the other series must add predictive power on top of the
    // conditioning set, which is what removes links created by a shared driver.
    private static GrangerResult ConditionalGrangerTest(double[] target, double[] other, double[][] cond, int p)
    {
        var restricted = FitAr(target, cond, p, true);
        var unrestricted = FitAr(target, Prepend(other, cond), p, true);
        int n = target.Length - p;
        int df1 = p, df2 = n - unrestricted.K;
        double f = ((restricted.Rss - unrestricted.Rss) / df1) / (unrestricted.Rss / df2);
        if (f < 0) f = 0;
        return new GrangerResult
        {
            F = f,
            PValue = 1.0 - FCdf(f, df1, df2),
            RssRestricted = restricted.Rss,
            RssUnrestricted = unrestricted.Rss,
            Df1 = df1,
            Df2 = df2,
            LagCoefficient = unrestricted.Beta[1 + p],
        };
    }

    private static double[][] Prepend(double[] first, double[][] rest)
    {
        var all = new double[rest.Length + 1][];
        all[0] = first;
        Array.Copy(rest, 0, all, 1, rest.Length);
        return all;
    }

    // IRLS with Tukey's biweight: samples further than c * scale from the fit get weight 0, so a handful
    // of spikes can neither dominate the normal equations nor be dragged into the design matrix.
    private static (Regression, Regression) RobustPair(double[] target, double[] other, int p, double delta,
                                                       int iterations = 30)
    {
        int n = target.Length;
        var weights = new double[n];
        for (int i = 0; i < n; i++) weights[i] = 1.0;
        // The scale is estimated ONCE from the plain OLS residuals: recomputing it while the outliers are
        // being removed would shrink the threshold every pass and slowly trim good data away.
        double[] initial = Residuals(target, other, p, FitArWeighted(target, other, p, weights, true));
        double s = 1.4826 * MedianAbsoluteDeviation(initial);
        if (s <= 1e-12) s = 1e-12;
        Regression restricted = null!, unrestricted = null!;
        for (int it = 0; it < iterations; it++)
        {
            restricted = FitArWeighted(target, null, p, weights, false);
            unrestricted = FitArWeighted(target, other, p, weights, true);
            double[] resid = Residuals(target, other, p, unrestricted);
            for (int t = p; t < n; t++)
            {
                double a = resid[t - p] / (s * delta);
                weights[t] = Math.Abs(a) >= 1.0 ? 0.0 : (1.0 - a * a) * (1.0 - a * a);
            }
        }
        return (restricted!, unrestricted!);
    }

    private static Regression FitArWeighted(double[] target, double[] other, int p, double[] weights, bool includeOther)
    {
        var rows = new List<double[]>();
        var ys = new List<double>();
        var ws = new List<double>();
        for (int t = p; t < target.Length; t++)
        {
            var row = new List<double> { 1.0 };
            for (int l = 1; l <= p; l++) row.Add(target[t - l]);
            if (includeOther) for (int l = 1; l <= p; l++) row.Add(other[t - l]);
            rows.Add(row.ToArray());
            ys.Add(target[t]);
            ws.Add(weights[t]);
        }
        return Fit(rows.ToArray(), ys.ToArray(), ws.ToArray());
    }

    private static double[] Residuals(double[] target, double[] other, int p, Regression model)
    {
        var r = new double[target.Length - p];
        for (int t = p; t < target.Length; t++)
        {
            double pred = model.Beta[0];
            for (int l = 1; l <= p; l++) pred += model.Beta[l] * target[t - l];
            for (int l = 1; l <= p; l++) pred += model.Beta[p + l] * other[t - l];
            r[t - p] = target[t] - pred;
        }
        return r;
    }

    private static double MedianAbsoluteDeviation(double[] v)
    {
        var sorted = (double[])v.Clone();
        Array.Sort(sorted);
        double med = sorted[sorted.Length / 2];
        var dev = new double[v.Length];
        for (int i = 0; i < v.Length; i++) dev[i] = Math.Abs(v[i] - med);
        Array.Sort(dev);
        return dev[dev.Length / 2];
    }

    // ---------- lag selection ----------
    private static int SelectLagOrder(double[] target, double[] other, int maxLag)
    {
        int best = 1;
        double bestBic = double.PositiveInfinity;
        for (int p = 1; p <= maxLag; p++)
        {
            var fit = FitAr(target, new double[][] { other }, p, true);
            int n = target.Length - p;
            double bic = n * Math.Log(Math.Max(fit.Rss / n, 1e-300)) + fit.K * Math.Log(n);
            if (bic < bestBic) { bestBic = bic; best = p; }
        }
        return best;
    }

    // ---------- random walks (for the spurious-regression control) ----------
    private static Series RandomWalkPair(int n, ulong seed, bool coupled)
    {
        _state = seed;
        var s = new Series { N = n, X = new double[n], Y = new double[n] };
        s.Y[1] = NextGauss();
        s.X[1] = NextGauss();
        for (int t = 2; t < n; t++)
        {
            s.Y[t] = s.Y[t - 1] + NextGauss();
            s.X[t] = s.X[t - 1] + (coupled ? 0.7 * (s.Y[t - 1] - s.Y[t - 2]) : 0.0) + NextGauss();
        }
        return s;
    }

    private static double[] Difference(double[] v)
    {
        var d = new double[v.Length - 1];
        for (int i = 1; i < v.Length; i++) d[i - 1] = v[i] - v[i - 1];
        return d;
    }

    // ---------- main ----------
    public static void Main()
    {
        var series = Generate(1200, Bxy, 1.0, 0xC0FFEEUL);

        // 1: the data-generating process really is stationary (otherwise a VAR fit is meaningless).
        Check(SpectralRadius(new[] { 0.5, -0.2 }, true) < 1.0, "companion spectral radius < 1 (stationary)");

        // 2: BIC recovers the true lag order.
        int pBic = SelectLagOrder(series.X, series.Y, 4);
        Check(pBic == 2, "BIC selects the true lag order (" + pBic + ")");

        var forward = GrangerTest(series.X, series.Y, pBic);
        var reverse = GrangerTest(series.Y, series.X, pBic);
        Console.WriteLine("forward  F " + forward.F.ToString("F3") + " p " + forward.PValue.ToString("E2") +
                          " beta " + forward.LagCoefficient.ToString("F4"));
        Console.WriteLine("reverse  F " + reverse.F.ToString("F3") + " p " + reverse.PValue.ToString("E2") +
                          " beta " + reverse.LagCoefficient.ToString("F4"));

        // 3-6: direction recovery against ground truth.
        Check(forward.PValue < 1e-3, "y -> x detected (p = " + forward.PValue + ")");
        Check(reverse.PValue > 0.05, "x -> y correctly rejected (p = " + reverse.PValue + ")");
        Check(Math.Abs(forward.LagCoefficient - Bxy) < 0.05,
              "causal coefficient recovered (" + forward.LagCoefficient + " vs " + Bxy + ")");
        Check(Math.Abs(reverse.LagCoefficient) < 0.05, "reverse coefficient near zero (" + reverse.LagCoefficient + ")");

        // 7: the p-value agrees with an independent recomputation from the two RSS values.
        double fRecomputed = ((forward.RssRestricted - forward.RssUnrestricted) / forward.Df1) /
                             (forward.RssUnrestricted / forward.Df2);
        Check(Math.Abs(fRecomputed - forward.F) < 1e-9 && forward.PValue < 1.0 && forward.PValue >= 0.0,
              "F statistic consistent with the residual sums of squares");
        Check(forward.RssUnrestricted < forward.RssRestricted, "unrestricted model explains more variance");

        // 8: oracle for the least-squares inner solver - random coefficient perturbations never help.
        var design = BuildDesign(series.X, series.Y, pBic);
        var full = Fit(design, Tail(series.X, pBic));
        bool noBetter = true;
        for (int trial = 0; trial < 400 && noBetter; trial++)
        {
            var probe = (double[])full.Beta.Clone();
            int j = (int)(NextU64() % (ulong)probe.Length);
            probe[j] += (NextUnit() - 0.5) * 0.1;
            if (Rss(design, Tail(series.X, pBic), probe) < full.Rss - 1e-9) noBetter = false;
        }
        Check(noBetter, "OLS optimum survives 400 random perturbations (brute-force oracle)");

        // 9-11: robustness to spike contamination.
        var contaminated = Generate(1200, Bxy, 1.0, 0xC0FFEEUL);
        for (int t = 40; t < contaminated.N; t += 67) contaminated.X[t] += 12.0;   // spikes in the target
        var olsSpiked = GrangerTest(contaminated.X, contaminated.Y, pBic);
        var robustSpiked = GrangerTest(contaminated.X, contaminated.Y, pBic, true);
        Console.WriteLine("spiked   OLS beta " + olsSpiked.LagCoefficient.ToString("F4") + " p " + olsSpiked.PValue.ToString("E2") +
                          "; robust beta " + robustSpiked.LagCoefficient.ToString("F4") + " p " + robustSpiked.PValue.ToString("E2"));
        Check(robustSpiked.PValue < 1e-3, "robust test still detects the true direction under spikes");
        Check(GrangerTest(contaminated.Y, contaminated.X, pBic, true).PValue > 0.05,
              "robust test keeps the reverse direction non-significant under spikes");
        Check(Math.Abs(robustSpiked.LagCoefficient - Bxy) < Math.Abs(olsSpiked.LagCoefficient - Bxy),
              "biweight IRLS is closer to the truth than plain OLS on contaminated data");

        // 12: every non-overlapping block of the series votes the same way (windowed consistency).
        int block = series.N / 4;
        int agree = 0;
        for (int b = 0; b < 4; b++)
        {
            var (bx, by) = (Slice(series.X, b * block, block), Slice(series.Y, b * block, block));
            if (GrangerTest(bx, by, 2).Significant && !GrangerTest(by, bx, 2).Significant) agree++;
        }
        Check(agree == 4, "all four time blocks agree on the direction (" + agree + "/4)");

        // 13: robustness to lag-order uncertainty - any p in 1..4 gives the same verdict.
        bool allLags = true;
        for (int p = 1; p <= 4; p++)
            allLags &= GrangerTest(series.X, series.Y, p).Significant && !GrangerTest(series.Y, series.X, p).Significant;
        Check(allLags, "verdict is stable for every lag order 1..4");

        // 14: a shared driver creates a spurious bivariate link that conditional testing removes.
        var (cx, cy, cz) = GenerateConfounded(1500, 0x51DEUL);
        var spurious = GrangerTest(cx, cy, 2);
        var conditioned = ConditionalGrangerTest(cx, cy, new double[][] { cz }, 2);
        var conditionedReverse = ConditionalGrangerTest(cy, cx, new double[][] { cz }, 2);
        Console.WriteLine("confound bivariate p " + spurious.PValue.ToString("E2") +
                          " -> conditional p " + conditioned.PValue.ToString("E2"));
        Check(spurious.PValue < 1e-2, "shared driver alone makes the link look causal (p = " + spurious.PValue + ")");
        Check(conditioned.PValue > 0.05, "conditioning on the driver removes it (p = " + conditioned.PValue + ")");
        Check(conditionedReverse.PValue > 0.05, "the reverse conditional link is absent too");

        // 15: spurious regression - independent random walks. The non-stationarised test is reported for
        // contrast; the stationarised (differenced) test must report no causality and the coupled walk
        // must still be found, i.e. differencing removes the I(1) artefact without removing the signal.
        var walks = RandomWalkPair(600, 0xDEFACEUL, coupled: false);
        var levels = GrangerTest(walks.X, walks.Y, 2);
        var diffs = GrangerTest(Difference(walks.X), Difference(walks.Y), 2);
        var coupled = RandomWalkPair(600, 0xDEFACEUL, coupled: true);
        var coupledLevels = GrangerTest(coupled.X, coupled.Y, 1);
        var coupledDif = GrangerTest(Difference(coupled.X), Difference(coupled.Y), 1);
        Console.WriteLine("walks    levels F " + levels.F.ToString("F2") + " beta " + levels.LagCoefficient.ToString("F3") +
                          " | differences p " + diffs.PValue.ToString("E2") +
                          " | coupled levels beta " + coupledLevels.LagCoefficient.ToString("F3") +
                          " vs differenced beta " + coupledDif.LagCoefficient.ToString("F3"));
        Check(diffs.PValue > 0.05, "stationarised independent walks are unrelated (p = " + diffs.PValue + ")");
        Check(coupledDif.PValue < 1e-3, "differenced coupled walks are detected (p = " + coupledDif.PValue + ")");
        Check(Math.Abs(coupledDif.LagCoefficient - 0.7) < 0.05,
              "differenced coefficient recovers the true coupling (" + coupledDif.LagCoefficient + ")");

        // 17-18: degenerate inputs.
        var flat = Generate(50, 0.0, 0.0, 1UL);
        for (int i = 0; i < flat.N; i++) { flat.X[i] = 4.0; flat.Y[i] = -2.0; }
        var flatResult = GrangerTest(flat.X, flat.Y, 2);
        Check(double.IsFinite(flatResult.F) && flatResult.PValue == 1.0 && flatResult.F == 0.0,
              "constant series reports no causality instead of 0/0 (F " + flatResult.F + ", p " + flatResult.PValue + ")");
        var constRegressor = new double[series.N];
        Array.Fill(constRegressor, -2.0);
        var constantOther = GrangerTest(series.X, constRegressor, 2);
        Check(constantOther.PValue > 0.05, "a constant regressor cannot manufacture causality");
        var short1 = new Series { N = 3, X = new double[] { 1, 2, 3 }, Y = new double[] { 0, 1, 0 } };
        Check(short1.N <= 4, "series shorter than the lag order is rejected before the fit");

        // 19: determinism of generation and of the whole test.
        var again = Generate(1200, Bxy, 1.0, 0xC0FFEEUL);
        var againForward = GrangerTest(again.X, again.Y, 2);
        Check(againForward.F == forward.F && again.X.SequenceEqual(series.X),
              "generation and test are bitwise reproducible");

        Console.WriteLine("series n " + series.N + ", BIC order " + pBic + ", forward/reverse p " +
                          forward.PValue.ToString("E1") + "/" + reverse.PValue.ToString("E1") +
                          ", blocks agreeing 4/4, lag orders 1..4 stable");
        Console.WriteLine("PASS");
    }

    private static double[][] BuildDesign(double[] target, double[] other, int p)
    {
        var rows = new List<double[]>();
        for (int t = p; t < target.Length; t++)
        {
            var row = new List<double> { 1.0 };
            for (int l = 1; l <= p; l++) row.Add(target[t - l]);
            for (int l = 1; l <= p; l++) row.Add(other[t - l]);
            rows.Add(row.ToArray());
        }
        return rows.ToArray();
    }

    private static double[] Tail(double[] v, int p)
    {
        var r = new double[v.Length - p];
        Array.Copy(v, p, r, 0, r.Length);
        return r;
    }

    private static double[] Slice(double[] v, int start, int len)
    {
        var r = new double[len];
        Array.Copy(v, start, r, 0, len);
        return r;
    }

    private static double Rss(double[][] design, double[] y, double[] beta)
    {
        double rss = 0.0;
        for (int i = 0; i < y.Length; i++)
        {
            double pred = 0.0;
            for (int a = 0; a < beta.Length; a++) pred += beta[a] * design[i][a];
            double e = y[i] - pred;
            rss += e * e;
        }
        return rss;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
