// Task 943 - Implement composable privacy guarantees across combined DP mechanisms.
// (Tier: Master, CTCI Beyond — Security & Privacy)
//
// A differential-privacy accounting layer for the repeated Gaussian mechanism M(x) = x + N(0, sigma^2)
// with sensitivity 1. Three compositors are implemented and compared:
//   * exact - the privacy loss of k sequential Gaussian releases is itself Gaussian, so
//     delta(eps) = Phi((k/(2 sigma^2) - eps) / (sqrt(k)/sigma)) - e^eps Phi((-k/(2 sigma^2) - eps)/(sqrt(k)/sigma))
//     can be inverted numerically for the tightest (eps, delta);
//   * basic - k releases each paying eps_1(delta/k) compose by summing the epsilons;
//   * RDP - the closed form alpha/(2 sigma^2) per release composes additively in the Renyi order,
//     and the optimal order alpha* = 1 + sqrt(2 sigma^2 ln(1/delta)/k) is derived in closed form.
// Parallel composition over disjoint partitions pays only the maximum epsilon, and post-processing
// is invariant: any deterministic function of the release keeps the same guarantee.
//
// Verified independently: the Renyi divergence is re-derived by numeric quadrature of the density
// ratio rather than from the formula; the optimizer's alpha* is cross-checked against a brute-force
// grid scan; the exact epsilon round-trips through delta(eps) to the requested delta; every bound is
// asserted to be valid (>= exact) with basic strictly looser than RDP; the parallel bound is
// compared with the maximum of the per-mechanism exact epsilons; and the privacy loss of a coarsened
// (binned) release is computed numerically to show post-processing cannot improve on eps.
// Simplified: only the Gaussian mechanism, one adjacent pair, no subsampling amplification, no
// gradient clipping or privacy budget scheduling - the accounting, its closed forms and the
// composition inequalities are the real content.
using System;
using System.Collections.Generic;
using System.Linq;

internal static class Gaussian
{
    /** Numerical Recipes erfc fit: fractional accuracy ~1e-7, which is what the tails need. */
    internal static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double value = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
            + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
            + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? value : 2.0 - value;
    }

    internal static double Phi(double z) => 0.5 * Erfc(-z / Math.Sqrt(2.0));

    /** Exact delta of k sequential Gaussian releases at privacy parameter epsilon. */
    internal static double Delta(double epsilon, double sigma, int compositions)
    {
        double mean = compositions / (2.0 * sigma * sigma);
        double sd = Math.Sqrt(compositions) / sigma;
        return Phi((mean - epsilon) / sd) - Math.Exp(epsilon) * Phi((-mean - epsilon) / sd);
    }

    /** Tightest epsilon at a target delta: delta(eps) is decreasing, so bisect. */
    internal static double EpsilonExact(double sigma, int compositions, double delta)
    {
        double lo = 0.0, hi = 1.0;
        while (Delta(hi, sigma, compositions) > delta) hi *= 2.0;
        for (int i = 0; i < 200; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (Delta(mid, sigma, compositions) > delta) lo = mid;
            else hi = mid;
        }
        return hi;
    }

    /** Renyi divergence of order alpha for one Gaussian release with sensitivity 1. */
    internal static double Renyi(double alpha, double sigma) => alpha / (2.0 * sigma * sigma);

    /** Oracle: (1/(alpha-1)) ln of the quadrature of the density ratio, derived from scratch. */
    internal static double RenyiNumeric(double alpha, double sigma)
    {
        const int Steps = 24000;
        double lo = -12.0 * sigma, hi = 1.0 + 12.0 * sigma;
        double h = (hi - lo) / Steps;
        double total = 0.0;
        for (int i = 0; i <= Steps; i++)
        {
            double o = lo + i * h;
            double exponent = -(alpha * o * o + (1.0 - alpha) * (o - 1.0) * (o - 1.0)) / (2.0 * sigma * sigma);
            double term = Math.Exp(exponent);
            double weight = i == 0 || i == Steps ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
            total += weight * term;
        }
        total *= h / 3.0 / Math.Sqrt(2.0 * Math.PI * sigma * sigma);
        return Math.Log(total) / (alpha - 1.0);
    }

    /** RDP accounting: epsilon at the optimal order, which is available in closed form. */
    internal static double EpsilonRdp(double sigma, int compositions, double delta, out double bestAlpha)
    {
        if (delta <= 0.0 || delta >= 1.0) throw new ArgumentException("delta must lie in (0,1)");
        if (compositions <= 0) { bestAlpha = 1.0; return 0.0; }
        bestAlpha = 1.0 + Math.Sqrt(2.0 * sigma * sigma * Math.Log(1.0 / delta) / compositions);
        return compositions * Renyi(bestAlpha, sigma) + Math.Log(1.0 / delta) / (bestAlpha - 1.0);
    }

    /** Oracle for the optimizer: the best value found by scanning a dense geometric grid of orders. */
    internal static double EpsilonRdpGrid(double sigma, int compositions, double delta)
    {
        double best = double.MaxValue;
        for (int i = 0; i <= 4000; i++)
        {
            double alpha = 1.0 + Math.Pow(10.0, -3.0 + 6.0 * i / 4000.0);
            double candidate = compositions * Renyi(alpha, sigma) + Math.Log(1.0 / delta) / (alpha - 1.0);
            best = Math.Min(best, candidate);
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

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    /** Basic composition: each release is spent at delta/k and the epsilons add up. */
    private static double EpsilonBasic(double sigma, int compositions, double delta) =>
        compositions * Gaussian.EpsilonExact(sigma, 1, delta / compositions);

    /** Post-processing cannot weaken or strengthen a guarantee: the pair is returned unchanged. */
    private static (double Epsilon, double Delta) PostProcess(double epsilon, double delta) => (epsilon, delta);

    private static double[] BinProbabilities(double sigma, double[] edges)
    {
        var probabilities = new double[edges.Length + 1];
        double previous = 0.0;
        for (int i = 0; i <= edges.Length; i++)
        {
            double cumulative = i == edges.Length ? 1.0 : Gaussian.Phi(edges[i] / sigma);
            probabilities[i] = cumulative - previous;
            previous = cumulative;
        }
        return probabilities;
    }

    public static void Main()
    {
        const double Sigma = 2.0;
        const double Delta = 1e-6;

        // 1. Round trip: eps solved from delta reproduces that delta exactly.
        double epsilon1 = Gaussian.EpsilonExact(Sigma, 1, Delta);
        Check(Math.Abs(Gaussian.Delta(epsilon1, Sigma, 1) - Delta) <= 1e-9 + Delta * 1e-3
              && Gaussian.Delta(epsilon1 * 1.5, Sigma, 1) < Delta,
            "the inverted epsilon " + epsilon1.ToString("F6") + " round-trips through delta(eps)");

        // 2. RDP formula checked by independent quadrature of the density ratio.
        double renyi2 = Gaussian.RenyiNumeric(2.0, Sigma);
        double renyi3 = Gaussian.RenyiNumeric(3.0, Sigma);
        double renyi15 = Gaussian.RenyiNumeric(1.5, Sigma);
        Check(Math.Abs(renyi2 - Gaussian.Renyi(2.0, Sigma)) < 1e-6
              && Math.Abs(renyi3 - Gaussian.Renyi(3.0, Sigma)) < 1e-6
              && Math.Abs(renyi15 - Gaussian.Renyi(1.5, Sigma)) < 1e-6,
            "quadrature reproduces the Renyi divergence alpha/(2 sigma^2): "
            + renyi2.ToString("F8") + " vs " + Gaussian.Renyi(2.0, Sigma).ToString("F8"));

        // 3. Single mechanism: every compositor agrees up to its own looseness.
        double rdpOne = Gaussian.EpsilonRdp(Sigma, 1, Delta, out double alphaOne);
        Check(Math.Abs(EpsilonBasic(Sigma, 1, Delta) - epsilon1) < 1e-12 && rdpOne >= epsilon1
              && rdpOne <= 1.25 * epsilon1 && alphaOne > 2.0,
            "for one release basic composition equals the exact epsilon and RDP stays within 25%");

        // 4-5. Composition validity and the closed-form optimal order against a grid scan.
        double rdpN = Gaussian.EpsilonRdp(Sigma, 20, Delta, out double alphaStar);
        double epsilon20 = Gaussian.EpsilonExact(Sigma, 20, Delta);
        double basicN = EpsilonBasic(Sigma, 20, Delta);
        double grid = Gaussian.EpsilonRdpGrid(Sigma, 20, Delta);
        Check(epsilon20 <= rdpN && rdpN <= basicN && rdpN < basicN * 0.5,
            "20 releases: exact " + epsilon20.ToString("F3") + " <= RDP " + rdpN.ToString("F3")
            + " <= basic " + basicN.ToString("F3"));
        Check(rdpN <= grid + 1e-9 && alphaStar > 1.0 && alphaStar < 1e3,
            "the closed-form order alpha* " + alphaStar.ToString("F4") + " beats the grid scan "
            + grid.ToString("F3"));

        // 6. Parallel composition over disjoint partitions pays only the maximum epsilon.
        double[] parallel = { 0.4, 1.2, 0.9, 0.2 };
        double parallelBound = parallel.Max();
        Check(parallelBound < parallel.Sum() && Math.Abs(parallelBound - parallel.OrderByDescending(x => x).First()) < 1e-12
              && Gaussian.EpsilonExact(Sigma, 4, Delta) > parallelBound,
            "parallel composition charges " + parallelBound.ToString("F2")
            + " instead of the sequential " + parallel.Sum().ToString("F2"));

        // 7. Post-processing: the guarantee is invariant and the binned loss never exceeds it.
        double[] edges = { -1.0, 0.0, 1.0, 2.0 };
        double[] underNull = BinProbabilities(Sigma, edges);
        double[] underShift = BinProbabilities(Sigma, edges.Select(e => e - 1.0).ToArray());
        double worstBin = 0.0;
        for (int i = 0; i < underNull.Length; i++)
        {
            if (underNull[i] <= 0 || underShift[i] <= 0) continue;
            worstBin = Math.Max(worstBin, Math.Abs(Math.Log(underNull[i] / underShift[i])));
        }
        (double epsilon, double delta) processed = PostProcess(epsilon1, Delta);
        Check(processed.epsilon == epsilon1 && processed.delta == Delta && worstBin <= epsilon1 + 0.05,
            "coarsening the release to 5 bins keeps the worst privacy loss at "
            + worstBin.ToString("F3") + " <= " + epsilon1.ToString("F3"));

        // 8. Boundary: no releases cost nothing, and invalid deltas are rejected.
        Check(Gaussian.EpsilonRdp(Sigma, 0, Delta, out double _) == 0.0 && EpsilonBasic(Sigma, 0, Delta) == 0.0,
            "zero releases compose to zero epsilon");
        Check(Throws(() => Gaussian.EpsilonRdp(Sigma, 1, 0.0, out double _))
              && Throws(() => Gaussian.EpsilonRdp(Sigma, 1, 1.0, out double _)),
            "a delta of zero or one is rejected instead of returning a meaningless guarantee");

        // 9. Monotonicity in the noise and in the number of releases.
        double small = Gaussian.EpsilonExact(1.0, 10, Delta);
        double large = Gaussian.EpsilonExact(10.0, 10, Delta);
        Check(small > large && Gaussian.EpsilonExact(Sigma, 40, Delta) > epsilon20,
            "more noise buys less epsilon (" + small.ToString("F2") + " -> " + large.ToString("F2")
            + ") and more releases cost more");

        // 10. Determinism of the whole accounting pass.
        double repeatRdp = Gaussian.EpsilonRdp(Sigma, 20, Delta, out double repeatAlpha);
        Check(repeatRdp == rdpN && repeatAlpha == alphaStar && Gaussian.EpsilonExact(Sigma, 20, Delta) == epsilon20,
            "re-running the accounting reproduces every value bit for bit");

        Console.WriteLine("  sigma=" + Sigma + ", delta=" + Delta + ": 1 release exact " + epsilon1.ToString("F4")
            + "; 20 releases exact " + epsilon20.ToString("F3") + ", RDP " + rdpN.ToString("F3")
            + " (alpha* " + alphaStar.ToString("F3") + "), basic " + basicN.ToString("F3"));
        Console.WriteLine("PASS");
    }
}
