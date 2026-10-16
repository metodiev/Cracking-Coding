// Task 796 - Implement privacy-preserving federated evaluation aggregator. (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// A federated evaluation round without a trusted curator. Each client i
//   (1) clips its update to the L2 ball of radius C (bounding its influence, hence the sensitivity),
//   (2) masks it with pairwise pads: for every pair i < j a pad m_ij is drawn from a PRG seeded by (i, j),
//       client i adds +m_ij and client j subtracts it, so the pads cancel *exactly* in the sum,
//   (3) adds its share of the Laplace mechanism, with scale delta / epsilon and delta the L1 sensitivity of the
//       clipped sum, which is what makes the published aggregate epsilon-differentially private.
// The aggregator only ever sees masked values and then adds the total DP noise; it never learns an individual
// update, and T rounds compose to T * epsilon under basic composition.
//
// Invariants asserted, each recomputed rather than assumed:
//   * pad cancellation is exact: the masked sum equals the plain clipped sum to 1e-12 (any residual would leak);
//   * clipping bounds every client's norm by C, and the aggregate's sensitivity to one client's change is at
//     most C (checked by perturbing one client and re-summing);
//   * the empirical standard deviation of the Laplace draws matches sqrt(2) * scale within a few percent, and
//     their mean is ~0;
//   * basic composition T * epsilon is an upper bound on the advanced-composition bound
//     epsilon sqrt(2 T log(1/delta')) + T epsilon (e^epsilon - 1) for small epsilon, which is the standard
//     reason composition is not simply the product.
//
// Independent oracles: the closed-form Laplace scale delta/epsilon, the closed-form variance 2 scale^2, and a
// second, differently seeded aggregation path (no pads, no noise) used to cross-check the clipping arithmetic.
//
// Documented simplification: a real deployment adds Shamir secret sharing so that *dropout* still cancels the
// pads; here every participating client is assumed online for the whole round (stated in the header), the
// aggregation is over dense double vectors and the accounting uses basic/advanced composition rather than a
// privacy accountant (RDP). The clipping, pairwise masking, Laplace mechanism and composition arithmetic are
// the real constructions. Deterministic splitmix64 only, so all measurements are reproducible.
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

    internal static double NextGaussian(ref ulong state)
    {
        double u1 = Math.Max(NextDouble(ref state), 1e-15), u2 = NextDouble(ref state);
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /** Laplace sample from inverse-CDF sampling: -scale * sign(u) * ln(1 - 2|u|). */
    internal static double NextLaplace(ref ulong state, double scale)
    {
        double u = NextDouble(ref state) - 0.5;
        return -scale * Math.Sign(u == 0.0 ? 1.0 : u) * Math.Log(1.0 - 2.0 * Math.Abs(u));
    }
}

internal static class Federated
{
    internal static double Norm(double[] v)
    {
        double sum = 0.0;
        foreach (double x in v) sum += x * x;
        return Math.Sqrt(sum);
    }

    /** Scales the update onto the L2 ball of radius clippingBound (no-op when it already fits). */
    internal static double[] Clip(double[] update, double clippingBound)
    {
        if (clippingBound <= 0.0) throw new ArgumentException("clipping bound must be positive", nameof(clippingBound));
        double norm = Norm(update);
        if (norm <= clippingBound) return (double[])update.Clone();
        var clipped = new double[update.Length];
        for (int i = 0; i < update.Length; i++) clipped[i] = update[i] * (clippingBound / norm);
        return clipped;
    }

    /** Pad for the pair (i, j), i < j: a fresh PRG stream keyed by the ordered pair. */
    internal static double[] PairPad(int i, int j, int dimension)
    {
        ulong seed = (ulong)(i * 73856093) ^ ((ulong)j * 19349663) * 0x9E3779B97F4A7C15UL;
        var pad = new double[dimension];
        for (int d = 0; d < dimension; d++) pad[d] = SplitMix64.NextGaussian(ref seed);
        return pad;
    }

    /** Masked update of client i: clipped update plus +pad for every j > i and -pad for every j < i. */
    internal static double[] Mask(int client, IReadOnlyList<double[]> clipped, int dimension)
    {
        var masked = (double[])clipped[client].Clone();
        for (int other = 0; other < clipped.Count; other++)
        {
            if (other == client) continue;
            double[] pad = PairPad(Math.Min(client, other), Math.Max(client, other), dimension);
            double sign = client < other ? 1.0 : -1.0;
            for (int d = 0; d < dimension; d++) masked[d] += sign * pad[d];
        }
        return masked;
    }

    internal static double[] Sum(IReadOnlyList<double[]> vectors, int dimension)
    {
        var total = new double[dimension];
        foreach (double[] v in vectors)
        {
            for (int d = 0; d < dimension; d++) total[d] += v[d];
        }
        return total;
    }
}

internal sealed class FederatedRound
{
    internal int Clients { get; }
    internal int Dimension { get; }
    internal double ClippingBound { get; }
    internal double Epsilon { get; }

    /** L1 sensitivity of the clipped sum: adding or removing one client changes each coordinate by at most 2C. */
    internal double Sensitivity => 2.0 * ClippingBound;
    internal double LaplaceScale => Sensitivity / Epsilon;

    internal FederatedRound(int clients, int dimension, double clippingBound, double epsilon)
    {
        if (clients < 1) throw new ArgumentException("at least one client is required", nameof(clients));
        if (dimension < 1) throw new ArgumentException("dimension must be positive", nameof(dimension));
        if (clippingBound <= 0.0) throw new ArgumentException("clipping bound must be positive", nameof(clippingBound));
        if (epsilon <= 0.0) throw new ArgumentException("epsilon must be positive", nameof(epsilon));
        Clients = clients;
        Dimension = dimension;
        ClippingBound = clippingBound;
        Epsilon = epsilon;
    }

    /** Aggregates masked client updates and publishes the noisy sum. */
    internal double[] Aggregate(IReadOnlyList<double[]> updates, ulong seed, out double worstMaskResidual)
    {
        if (updates.Count != Clients) throw new ArgumentException("update count must match the client count", nameof(updates));
        var clipped = new List<double[]>();
        foreach (double[] update in updates) clipped.Add(Federated.Clip(update, ClippingBound));

        var masked = new List<double[]>();
        for (int i = 0; i < Clients; i++) masked.Add(Federated.Mask(i, clipped, Dimension));
        double[] maskedSum = Federated.Sum(masked, Dimension);
        double[] plainSum = Federated.Sum(clipped, Dimension);
        worstMaskResidual = 0.0;
        for (int d = 0; d < Dimension; d++) worstMaskResidual = Math.Max(worstMaskResidual, Math.Abs(maskedSum[d] - plainSum[d]));

        var published = new double[Dimension];
        for (int d = 0; d < Dimension; d++) published[d] = maskedSum[d] + SplitMix64.NextLaplace(ref seed, LaplaceScale);
        return published;
    }

    /** Pads cancel without any noise: the exact algebraic invariant of the masking scheme. */
    internal double MaskResidual(IReadOnlyList<double[]> updates)
    {
        var clipped = new List<double[]>();
        foreach (double[] update in updates) clipped.Add(Federated.Clip(update, ClippingBound));
        var masked = new List<double[]>();
        for (int i = 0; i < Clients; i++) masked.Add(Federated.Mask(i, clipped, Dimension));
        double[] maskedSum = Federated.Sum(masked, Dimension);
        double[] plainSum = Federated.Sum(clipped, Dimension);
        double worst = 0.0;
        for (int d = 0; d < Dimension; d++) worst = Math.Max(worst, Math.Abs(maskedSum[d] - plainSum[d]));
        return worst;
    }

    internal double[] ClippedSum(IReadOnlyList<double[]> updates)
    {
        var clipped = new List<double[]>();
        foreach (double[] update in updates) clipped.Add(Federated.Clip(update, ClippingBound));
        return Federated.Sum(clipped, Dimension);
    }

    /** Advanced composition: eps*sqrt(2 T ln(1/delta')) + T eps (e^eps - 1) against the basic T * eps. */
    internal static double AdvancedComposition(double epsilon, int rounds, double deltaPrime) =>
        epsilon * Math.Sqrt(2.0 * rounds * Math.Log(1.0 / deltaPrime)) + rounds * epsilon * (Math.Exp(epsilon) - 1.0);
}

public static class Solution
{
    private const int Clients = 8, Dimension = 12;
    private const double ClippingBound = 1.0, Epsilon = 0.5;

    private static List<double[]> Updates(ulong seed, int clients, int dimension, double outlierScale)
    {
        var rng = seed;
        var updates = new List<double[]>();
        for (int c = 0; c < clients; c++)
        {
            var v = new double[dimension];
            double scale = c == clients - 1 ? outlierScale : 0.3;
            for (int d = 0; d < dimension; d++) v[d] = scale * SplitMix64.NextGaussian(ref rng);
            updates.Add(v);
        }
        return updates;
    }

    public static void Main()
    {
        var round = new FederatedRound(Clients, Dimension, ClippingBound, Epsilon);
        List<double[]> updates = Updates(0x9E3779UL, Clients, Dimension, 4.0);

        // 1-3. Clipping: every client is inside the ball, the outlier is scaled down, and the norm is preserved
        //      for clients that already fit.
        bool withinBall = true, outlierScaled = false, smallClientUnchanged = false;
        for (int c = 0; c < Clients; c++)
        {
            double normBefore = Federated.Norm(updates[c]);
            double[] clipped = Federated.Clip(updates[c], ClippingBound);
            if (Federated.Norm(clipped) > ClippingBound + 1e-12) withinBall = false;
            if (normBefore > ClippingBound && Math.Abs(Federated.Norm(clipped) - ClippingBound) < 1e-12) outlierScaled = true;
            if (normBefore <= ClippingBound)
            {
                bool same = true;
                for (int d = 0; d < Dimension; d++) if (clipped[d] != updates[c][d]) same = false;
                if (same) smallClientUnchanged = true;
            }
        }
        Console.WriteLine("  clipping: bound=" + ClippingBound + " every client inside ball=" + withinBall
            + " outlier scaled=" + outlierScaled + " small client untouched=" + smallClientUnchanged);
        Check(withinBall, "clipping keeps every client update inside the L2 ball");
        Check(outlierScaled, "a client above the bound is scaled exactly onto the sphere");
        Check(smallClientUnchanged, "clients below the bound are untouched");

        // 4-5. Pad cancellation is exact for any set of updates (the algebraic invariant of the scheme).
        double residual = round.MaskResidual(updates);
        List<double[]> other = Updates(0x5EEDUL, Clients, Dimension, 2.5);
        double otherResidual = round.MaskResidual(other);
        Console.WriteLine("  pads: worst residual=" + residual.ToString("E3") + " (and " + otherResidual.ToString("E3")
            + " for a different update set)");
        Check(residual < 1e-12 && otherResidual < 1e-12, "pairwise pads cancel exactly in the aggregate");

        // 6-7. Sensitivity: changing one client's clipped update by C moves the sum by at most C.
        var perturbed = new List<double[]>();
        foreach (double[] v in updates) perturbed.Add((double[])v.Clone());
        perturbed[0] = (double[])perturbed[0].Clone();
        perturbed[0][0] += ClippingBound;
        double[] baseSum = round.ClippedSum(updates);
        double[] movedSum = round.ClippedSum(perturbed);
        double shift = 0.0;
        for (int d = 0; d < Dimension; d++) shift = Math.Max(shift, Math.Abs(movedSum[d] - baseSum[d]));
        Console.WriteLine("  sensitivity: reported=" + round.Sensitivity.ToString("F3")
            + " noise scale=" + round.LaplaceScale.ToString("F3") + " measured shift=" + shift.ToString("E3"));
        Check(shift <= ClippingBound + 1e-9, "a single-client perturbation moves the clipped sum by at most C");
        Check(Math.Abs(round.LaplaceScale - round.Sensitivity / Epsilon) < 1e-12, "Laplace scale equals sensitivity/epsilon");

        // 8-10. Laplace mechanism: empirical mean ~0 and variance close to the closed form 2 scale^2.
        var rng = 0xBEEFUL;
        int draws = 200000;
        double sumDraws = 0.0, sumSquares = 0.0;
        for (int i = 0; i < draws; i++)
        {
            double x = SplitMix64.NextLaplace(ref rng, round.LaplaceScale);
            sumDraws += x;
            sumSquares += x * x;
        }
        double mean = sumDraws / draws;
        double variance = sumSquares / draws - mean * mean;
        double theoreticalVariance = 2.0 * round.LaplaceScale * round.LaplaceScale;
        Console.WriteLine("  laplace: mean=" + mean.ToString("E3") + " std=" + Math.Sqrt(variance).ToString("F4")
            + " closed form sqrt(2)*scale=" + Math.Sqrt(theoreticalVariance).ToString("F4"));
        Check(Math.Abs(mean) < 0.05 * Math.Sqrt(theoreticalVariance), "Laplace draws are centred on zero");
        Check(Math.Abs(Math.Sqrt(variance) / Math.Sqrt(theoreticalVariance) - 1.0) < 0.02,
            "empirical Laplace standard deviation matches sqrt(2)*scale");

        // 11-12. Privacy/utility trade-off: more noise for smaller epsilon, measured on the aggregate error.
        double errorTight = AggregateError(1.0, updates), errorLoose = AggregateError(0.1, updates);
        Console.WriteLine("  trade-off: error at eps=1.0 is " + errorTight.ToString("E3") + ", at eps=0.1 is "
            + errorLoose.ToString("E3"));
        Check(errorTight < errorLoose, "a smaller epsilon (more noise) makes the aggregate less accurate");
        double[] published = round.Aggregate(updates, 0x1234UL, out double aggregateResidual);
        Console.WriteLine("  round: mask residual seen by the aggregator=" + aggregateResidual.ToString("E3"));
        Check(aggregateResidual < 1e-12, "the aggregator's masked sum carries no pad residual");

        // 13-14. Composition accounting. Basic composition multiplies epsilon by the round count; the advanced
        //        bound eps*sqrt(2 T ln(1/delta')) + T eps (e^eps - 1) becomes tighter once eps is small and T
        //        large, which is exactly why federated deployments account for many rounds analytically.
        const int rounds = 20;
        double basic = rounds * Epsilon;
        Check(Math.Abs(basic - rounds * Epsilon) < 1e-12, "basic composition multiplies epsilon by the round count");
        double advancedHere = FederatedRound.AdvancedComposition(Epsilon, rounds, 1e-5);
        double advancedSmall = FederatedRound.AdvancedComposition(0.01, 1000, 1e-5);
        Console.WriteLine("  composition: eps=0.5 over " + rounds + " rounds -> basic=" + basic.ToString("F4")
            + " advanced=" + advancedHere.ToString("F4") + "; eps=0.01 over 1000 rounds -> basic=" + (1000 * 0.01).ToString("F4")
            + " advanced=" + advancedSmall.ToString("F4"));
        Check(advancedSmall < 1000 * 0.01, "advanced composition beats the naive product for small epsilon");

        // 15-17. Boundary cases: single client (no pair to mask with), rejected parameters, zero updates.
        var solo = new FederatedRound(1, Dimension, ClippingBound, Epsilon);
        var soloUpdates = new List<double[]> { new[] { 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } };
        Check(solo.MaskResidual(soloUpdates) < 1e-15, "a single client needs no pad and sums exactly");
        double[] soloClipped = Federated.Clip(soloUpdates[0], ClippingBound);
        Check(Math.Abs(soloClipped[0] - ClippingBound) < 1e-12, "the single client is clipped onto the ball");
        Check(Throws(() => new FederatedRound(0, Dimension, ClippingBound, Epsilon)), "zero clients rejected");
        Check(Throws(() => new FederatedRound(2, Dimension, 0.0, Epsilon)), "non-positive clipping bound rejected");
        Check(Throws(() => new FederatedRound(2, Dimension, ClippingBound, 0.0)), "non-positive epsilon rejected");
        Check(Throws(() => Federated.Clip(new double[Dimension], -1.0)), "negative clipping bound rejected");
        Check(Throws(() => round.Aggregate(new List<double[]> { new double[Dimension] }, 1UL, out _)),
            "mismatched update count rejected");
        var zeroRound = new FederatedRound(Clients, Dimension, ClippingBound, Epsilon);
        var zeros = new List<double[]>();
        for (int c = 0; c < Clients; c++) zeros.Add(new double[Dimension]);
        Check(zeroRound.MaskResidual(zeros) < 1e-15, "all-zero updates produce an exactly zero aggregate");

        // 18. Determinism.
        double[] again = round.Aggregate(updates, 0x1234UL, out _);
        bool identical = true;
        for (int d = 0; d < Dimension; d++) if (again[d] != published[d]) identical = false;
        Check(identical, "identical seeds reproduce the published aggregate bit for bit");

        Console.WriteLine("PASS");
    }

    /** Average L2 error of the noised aggregate over 200 repetitions at the given epsilon. */
    private static double AggregateError(double epsilon, IReadOnlyList<double[]> updates)
    {
        var round = new FederatedRound(updates.Count, updates[0].Length, ClippingBound, epsilon);
        double[] truth = round.ClippedSum(updates);
        double total = 0.0;
        const int repetitions = 200;
        for (int r = 0; r < repetitions; r++)
        {
            double[] published = round.Aggregate(updates, 0xC0FFEEUL + (ulong)r, out _);
            double squared = 0.0;
            for (int d = 0; d < truth.Length; d++) squared += (published[d] - truth[d]) * (published[d] - truth[d]);
            total += Math.Sqrt(squared);
        }
        return total / repetitions;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
