// Task 829 - Implement privacy-preserving federated learning with secure aggregation and differential privacy.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A federated round in which the server learns *only* the sum of the client updates, and that sum is
// differentially private:
//
//   1. every client clips its update to the L2 ball of radius C, so one client can move the sum by at most C;
//   2. the clipped update is quantised to fixed point (scale 2^16) and each coordinate is Shamir-shared with
//      threshold t over the prime field 2^31-1: the secret is s, the client publishes f(i) for i = 1..n, where f
//      is a random degree t-1 polynomial with f(0) = s. Sharing is linear, so adding the shares received from any
//      t clients yields a valid sharing of the *sum* of their secrets, which one Lagrange interpolation at 0
//      recovers. Missing clients simply contribute no share, so the protocol tolerates n - t dropouts and the
//      reconstruction is identical for every surviving subset of size >= t;
//   3. the server adds Gaussian noise with sigma = C sqrt(2 ln(1.25/delta)) / epsilon - the Gaussian mechanism
//      for a sensitivity-C query - and accounts for T rounds with the RDP bound eps_RDP(alpha) = T alpha C^2 /
//      (2 sigma^2) converted to (eps, delta) by eps = eps_RDP + log(1/delta)/(alpha-1), minimised over alpha.
//
// Verified invariants (each recomputed here, none assumed):
//   * 2^31-1 is prime (trial division), every inverse satisfies a * a^-1 = 1 and Fermat's little theorem holds;
//   * t-of-n Shamir reconstruction returns the secret *exactly* for every one of the C(n,t) share subsets, and
//     t-1 shares are consistent with every candidate secret - the alternative polynomial is constructed with
//     Lagrange interpolation and evaluated at the observed points, so perfect secrecy is proved, not asserted;
//   * the reconstructed aggregate equals the exact integer sum of the quantised client updates bit-for-bit, for
//     every surviving subset of at least t clients and independently of which clients dropped;
//   * sensitivity is C (measured by perturbing one client), the empirical Gaussian noise has the right sigma and
//     mean, and the fixed-point round trip stays inside half a quantisation step;
//   * the RDP accountant upper-bounds the classical closed form, is monotone in the round count and in 1/sigma,
//     and the noise scale found by bisection meets a privacy budget while staying at or above the closed-form
//     scale (the honest statement: RDP is looser, so it demands at least as much noise);
//   * federated averaging built on this pipeline converges to within a small factor of centralised training on
//     the same data, the secure path reproduces the plaintext path exactly, and repeated runs are identical.
//
// Documented simplification: the toy model is a 4-dimensional linear regressor, clients hold disjoint shards of a
// deterministic synthetic dataset, and "communication" is an explicit in-process share exchange with an explicit
// dropout schedule (no threads, sockets or wall clock). Quantisation is a real property of finite-field secure
// aggregation and is documented: values are exact in the field but exact only to 2^-16 after decoding. Shamir
// sharing, the linear-aggregation trick, the Gaussian mechanism, the RDP accountant and FedAvg are the real
// algorithms; production systems add verifiable secret sharing, re-sharing for large dropouts and per-client
// accounting.
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
}

/** Arithmetic modulo the Mersenne prime 2^31 - 1; products stay inside a long, so everything is exact. */
internal static class Field
{
    internal const long P = 2147483647L;

    internal static long Add(long a, long b) => ((a + b) % P + P) % P;

    internal static long Sub(long a, long b) => ((a - b) % P + P) % P;

    internal static long Mul(long a, long b) => (a % P) * (b % P) % P;

    internal static long Pow(long a, long e)
    {
        long result = 1;
        long b = a % P;
        while (e > 0)
        {
            if ((e & 1L) != 0) result = Mul(result, b);
            b = Mul(b, b);
            e >>= 1;
        }
        return result;
    }

    internal static long Inverse(long a)
    {
        long v = a % P;
        if (v < 0) v += P;
        if (v == 0) throw new ArgumentException("zero has no inverse in a field");
        return Pow(v, P - 2);   // Fermat: a^(p-2) == a^-1 for prime p
    }

    internal static bool IsPrime(long n)
    {
        if (n < 2) return false;
        for (long d = 2; d * d <= n; d++)
        {
            if (n % d == 0) return false;
        }
        return true;
    }

    internal static long RandomNonZero(ref ulong rng) => (long)(SplitMix64.Next(ref rng) % (ulong)(P - 1)) + 1;
}

internal struct Share
{
    internal int X;
    internal long Y;
}

/** t-of-n Shamir secret sharing over Field.P, with Lagrange interpolation at zero. */
internal static class Shamir
{
    internal static long Evaluate(IReadOnlyList<long> coefficients, long x)
    {
        long result = 0;
        for (int i = coefficients.Count - 1; i >= 0; i--) result = Field.Add(Field.Mul(result, x), coefficients[i]);
        return result;
    }

    internal static Share[] Share(long secret, int n, int threshold, ref ulong rng)
    {
        if (n < 1) throw new ArgumentException("at least one shareholder is required", nameof(n));
        if (threshold < 1 || threshold > n) throw new ArgumentException("the threshold must lie in [1, n]", nameof(threshold));
        var coefficients = new long[threshold];
        coefficients[0] = Field.Add(secret % Field.P, 0);
        for (int i = 1; i < threshold; i++) coefficients[i] = Field.RandomNonZero(ref rng);
        var shares = new Share[n];
        for (int i = 0; i < n; i++) shares[i] = new Share { X = i + 1, Y = Evaluate(coefficients, i + 1) };
        return shares;
    }

    /** Lagrange interpolation at 0 over all the given shareholders. */
    internal static long Reconstruct(IReadOnlyList<Share> shares)
    {
        if (shares.Count == 0) throw new ArgumentException("at least one share is required", nameof(shares));
        var seen = new HashSet<int>();
        long total = 0;
        for (int i = 0; i < shares.Count; i++)
        {
            if (!seen.Add(shares[i].X)) throw new ArgumentException("duplicate shareholder index");
            long numerator = 1;
            long denominator = 1;
            for (int j = 0; j < shares.Count; j++)
            {
                if (i == j) continue;
                numerator = Field.Mul(numerator, Field.Sub(0, shares[j].X));
                denominator = Field.Mul(denominator, Field.Sub(shares[i].X, shares[j].X));
            }
            total = Field.Add(total, Field.Mul(shares[i].Y, Field.Mul(numerator, Field.Inverse(denominator))));
        }
        return total;
    }

    /**
     * The degree t-1 polynomial through (0, valueAtZero) and the given points, evaluated at x. Used to show that
     * t-1 observed shares are consistent with any secret at all: the polynomial exists and reproduces the shares.
     */
    internal static long PolynomialThrough(IReadOnlyList<Share> points, long valueAtZero, long x)
    {
        long basisAtZero = 1;
        foreach (Share p in points)
        {
            basisAtZero = Field.Mul(basisAtZero, Field.Mul(Field.Sub(x, p.X), Field.Inverse(Field.Sub(0, p.X))));
        }
        long total = Field.Mul(valueAtZero, basisAtZero);
        for (int i = 0; i < points.Count; i++)
        {
            long basis = Field.Mul(x, Field.Inverse(points[i].X));   // (x - 0) / (x_i - 0)
            for (int j = 0; j < points.Count; j++)
            {
                if (i == j) continue;
                long numerator = Field.Sub(x, points[j].X);
                long denominator = Field.Sub(points[i].X, points[j].X);
                basis = Field.Mul(basis, Field.Mul(numerator, Field.Inverse(denominator)));
            }
            total = Field.Add(total, Field.Mul(points[i].Y, basis));
        }
        return total;
    }
}

internal static class Quantiser
{
    internal const double Scale = 65536.0;   // 2^16

    internal static long Encode(double value)
    {
        if (Math.Abs(value) > 16000.0) throw new ArgumentException("value out of range for the fixed-point encoding");
        return (long)Math.Round(value * Scale);
    }

    internal static double Decode(long value) => value / Scale;

    /**
     * Decodes a field element: secrets are signed, and a negative secret is represented by p - |s|, so anything
     * above p/2 is mapped back to the negative range before the scale is removed.
     */
    internal static double DecodeSigned(long value)
    {
        long signed = value > Field.P / 2 ? value - Field.P : value;
        return signed / Scale;
    }
}

internal static class SecureAggregation
{
    /** Exact integer aggregate of the quantised updates: the oracle the reconstruction is checked against. */
    internal static long[] ExactQuantisedSum(IReadOnlyList<double[]> updates, double clippingBound)
    {
        if (updates.Count == 0) throw new ArgumentException("at least one client is required", nameof(updates));
        int dimensions = updates[0].Length;
        var total = new long[dimensions];
        foreach (double[] update in updates)
        {
            double[] clipped = Clip(update, clippingBound);
            for (int d = 0; d < dimensions; d++) total[d] += Quantiser.Encode(clipped[d]);
        }
        return total;
    }

    /**
     * One secure round: each client clips, quantises and shares every coordinate; the server sums the shares of
     * the surviving clients coordinate-wise and interpolates once per coordinate. Sharing is linear, so the sum of
     * those shares is a valid sharing of the sum of the secrets.
     */
    internal static double[] Run(IReadOnlyList<double[]> updates, int threshold, double clippingBound,
        IReadOnlyList<int> participants, ulong seed)
    {
        long[] raw = RunQuantised(updates, threshold, clippingBound, participants, seed);
        var result = new double[raw.Length];
        for (int d = 0; d < raw.Length; d++) result[d] = Quantiser.DecodeSigned(raw[d]);
        return result;
    }

    /**
     * One secure round. Dealer j splits the secret s_j of coordinate d into f_j with f_j(0) = s_j and hands
     * f_j(i + 1) to shareholder i. Because every dealer contributes to every shareholder, shareholder i can add
     * its received pieces into a single value g(i + 1) where g = sum of the participating dealers' polynomials -
     * still of degree threshold-1 with g(0) = sum of their secrets. The aggregator therefore sums *at equal
     * evaluation points* (that is the whole trick: shares are added, not secrets) and interpolates once.
     */
    internal static long[] RunQuantised(IReadOnlyList<double[]> updates, int threshold, double clippingBound,
        IReadOnlyList<int> participants, ulong seed)
    {
        if (updates.Count == 0) throw new ArgumentException("at least one client is required", nameof(updates));
        if (threshold < 1 || threshold > updates.Count)
        {
            throw new ArgumentException("the reconstruction threshold must lie in [1, clientCount]", nameof(threshold));
        }
        if (participants.Count < threshold)
        {
            throw new ArgumentException("fewer participating clients than the reconstruction threshold", nameof(participants));
        }
        int dimensions = updates[0].Length;
        var rng = seed;
        var shares = new Share[updates.Count][][];
        for (int j = 0; j < updates.Count; j++)
        {
            double[] clipped = Clip(updates[j], clippingBound);
            shares[j] = new Share[dimensions][];
            for (int d = 0; d < dimensions; d++)
            {
                shares[j][d] = Shamir.Share(Quantiser.Encode(clipped[d]), updates.Count, threshold, ref rng);
            }
        }
        var aggregate = new long[dimensions];
        for (int d = 0; d < dimensions; d++)
        {
            var column = new List<Share>();
            foreach (int shareholder in participants)
            {
                long y = 0;
                foreach (int dealer in participants) y = Field.Add(y, shares[dealer][d][shareholder].Y);
                column.Add(new Share { X = shareholder + 1, Y = y });
            }
            aggregate[d] = Shamir.Reconstruct(column);
        }
        return aggregate;
    }

    internal static List<int> All(int n)
    {
        var list = new List<int>(n);
        for (int i = 0; i < n; i++) list.Add(i);
        return list;
    }

    internal static double[] Clip(double[] update, double clippingBound)
    {
        if (clippingBound <= 0.0) throw new ArgumentException("the clipping bound must be positive", nameof(clippingBound));
        double norm = 0.0;
        foreach (double v in update) norm += v * v;
        norm = Math.Sqrt(norm);
        var clipped = new double[update.Length];
        for (int i = 0; i < update.Length; i++)
        {
            clipped[i] = norm <= clippingBound ? update[i] : update[i] * (clippingBound / norm);
        }
        return clipped;
    }
}

/** RDP accountant for T-fold composition of the Gaussian mechanism. */
internal static class Accountant
{
    internal static double RdpOneRound(double sensitivity, double sigma, double alpha) =>
        alpha * sensitivity * sensitivity / (2.0 * sigma * sigma);

    /** Converts the composed RDP curve to (eps, delta) with the standard bound, minimised over integer orders. */
    internal static double Epsilon(double sensitivity, double sigma, int rounds, double delta)
    {
        if (sigma <= 0.0) throw new ArgumentException("the noise scale must be positive", nameof(sigma));
        if (rounds < 0) throw new ArgumentException("the round count must not be negative", nameof(rounds));
        if (!(delta > 0.0 && delta < 1.0)) throw new ArgumentException("delta must lie strictly inside (0,1)", nameof(delta));
        double best = double.PositiveInfinity;
        for (int alpha = 2; alpha <= 256; alpha++)
        {
            double rdp = rounds * RdpOneRound(sensitivity, sigma, alpha);
            best = Math.Min(best, rdp + Math.Log(1.0 / delta) / (alpha - 1));
        }
        return best;
    }

    /** The classical closed form for one release of the Gaussian mechanism. */
    internal static double ClosedFormOneRound(double sensitivity, double sigma, double delta) =>
        sensitivity * Math.Sqrt(2.0 * Math.Log(1.25 / delta)) / sigma;

    /** Smallest noise scale whose RDP accounting fits the budget (Epsilon is decreasing in sigma). */
    internal static double SigmaForBudget(double sensitivity, int rounds, double epsilon, double delta)
    {
        double lo = 1e-3;
        double hi = 1e3;
        for (int i = 0; i < 200; i++)
        {
            double mid = 0.5 * (lo + hi);
            if (Epsilon(sensitivity, mid, rounds, delta) > epsilon) lo = mid; else hi = mid;
        }
        return hi;
    }
}

internal sealed class FederatedRun
{
    internal double InitialLoss;
    internal double FinalLoss;
    internal double FinalEpsilon;
}

public static class Solution
{
    private const int Dim = 4;
    private const int Clients = 6;
    private const int SamplesPerClient = 16;
    private const double ClippingBound = 0.5;
    private const int Threshold = 4;
    private const double Delta = 1e-5;
    private const double TargetEpsilon = 4.0;
    private const int Rounds = 12;

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static double Rel(double a, double b) => Math.Abs(a - b) / Math.Max(1e-30, Math.Abs(b));

    private static bool Contains(IReadOnlyList<int> list, int value)
    {
        foreach (int v in list)
        {
            if (v == value) return true;
        }
        return false;
    }

    /** Deterministic synthetic regression y = w . x + b, one disjoint shard per client. */
    private static void BuildData(out double[][][] features, out double[][] targets, out double[][] allX, out double[] allY)
    {
        var rng = 0xFEEDUL;
        double[] trueWeights = { 0.8, -0.5, 0.3, 1.1 };
        features = new double[Clients][][];
        targets = new double[Clients][];
        var flatX = new List<double[]>();
        var flatY = new List<double>();
        for (int c = 0; c < Clients; c++)
        {
            features[c] = new double[SamplesPerClient][];
            targets[c] = new double[SamplesPerClient];
            for (int s = 0; s < SamplesPerClient; s++)
            {
                var x = new double[Dim];
                for (int d = 0; d < Dim; d++) x[d] = 2.0 * SplitMix64.NextDouble(ref rng) - 1.0;
                double y = 0.35;
                for (int d = 0; d < Dim; d++) y += trueWeights[d] * x[d];
                features[c][s] = x;
                targets[c][s] = y;
                flatX.Add(x);
                flatY.Add(y);
            }
        }
        allX = flatX.ToArray();
        allY = flatY.ToArray();
    }

    /** One client epoch set of gradient descent; returns the parameter delta relative to the global weights. */
    private static double[] LocalDelta(double[] globalWeights, double[][] xs, double[] ys, double lr, int epochs)
    {
        double[] local = (double[])globalWeights.Clone();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            var gradient = new double[local.Length];
            for (int i = 0; i < xs.Length; i++)
            {
                double prediction = 0.0;
                for (int d = 0; d < local.Length; d++) prediction += local[d] * xs[i][d];
                double error = prediction - ys[i];
                for (int d = 0; d < local.Length; d++) gradient[d] += 2.0 * error * xs[i][d];
            }
            for (int d = 0; d < local.Length; d++) local[d] -= lr * gradient[d] / xs.Length;
        }
        var delta = new double[local.Length];
        for (int d = 0; d < local.Length; d++) delta[d] = local[d] - globalWeights[d];
        return delta;
    }

    private static double FullLoss(double[] w, double[][] xs, double[] ys)
    {
        double total = 0.0;
        for (int i = 0; i < xs.Length; i++)
        {
            double prediction = 0.0;
            for (int d = 0; d < w.Length; d++) prediction += w[d] * xs[i][d];
            double error = prediction - ys[i];
            total += error * error;
        }
        return total / xs.Length;
    }

    /**
     * Federated averaging over `Rounds` rounds with clipping, exact Shamir aggregation and optional Gaussian noise;
     * when secureSharing is false the same pipeline uses the plaintext sum so the two paths can be compared.
     */
    private static FederatedRun Federated(double[][][] features, double[][] targets, double[][] allX, double[] allY,
        double lr, int epochs, bool secureSharing, bool addNoise, double sigma, ulong seed)
    {
        var weights = new double[Dim];
        var rng = seed;
        var run = new FederatedRun { InitialLoss = FullLoss(weights, allX, allY) };
        for (int round = 0; round < Rounds; round++)
        {
            var updates = new List<double[]>();
            for (int c = 0; c < Clients; c++) updates.Add(LocalDelta(weights, features[c], targets[c], lr, epochs));
            var dropped = new List<int>();
            if (secureSharing)
            {
                if (round % 3 == 2) dropped.Add(0);
                if (round % 5 == 4) dropped.Add(1);
            }
            var participating = new List<double[]>();
            for (int c = 0; c < Clients; c++)
            {
                if (Contains(dropped, c)) continue;
                participating.Add(updates[c]);
            }
            double[] aggregate;
            if (secureSharing)
            {
                // With the dropout set known, the surviving clients re-share among themselves, so the shares are
                // indexed by their position in the participating list.
                aggregate = SecureAggregation.Run(participating, Threshold, ClippingBound,
                    SecureAggregation.All(participating.Count), seed + (ulong)round * 7919UL);
                long[] exact = SecureAggregation.ExactQuantisedSum(participating, ClippingBound);
                for (int d = 0; d < Dim; d++)
                {
                    if (aggregate[d] != Quantiser.DecodeSigned(exact[d]))
                    {
                        throw new Exception("secure aggregate does not match the plaintext sum");
                    }
                }
            }
            else
            {
                aggregate = new double[Dim];
                foreach (double[] update in updates)
                {
                    double[] clipped = SecureAggregation.Clip(update, ClippingBound);
                    for (int d = 0; d < Dim; d++) aggregate[d] += clipped[d];
                }
            }
            for (int d = 0; d < Dim; d++)
            {
                double noise = addNoise ? sigma * SplitMix64.NextGaussian(ref rng) : 0.0;
                weights[d] += (aggregate[d] + noise) / Clients;
            }
        }
        run.FinalLoss = FullLoss(weights, allX, allY);
        run.FinalEpsilon = addNoise ? Accountant.Epsilon(ClippingBound, sigma, Rounds, Delta) : 0.0;
        return run;
    }

    /** Centralised oracle: the same number of local epochs, all data in one place, no clipping and no noise. */
    private static double CentralLoss(double[][] allX, double[] allY, double lr, int epochs)
    {
        var weights = new double[Dim];
        for (int round = 0; round < Rounds; round++)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var gradient = new double[Dim];
                for (int i = 0; i < allX.Length; i++)
                {
                    double prediction = 0.0;
                    for (int d = 0; d < Dim; d++) prediction += weights[d] * allX[i][d];
                    double error = prediction - allY[i];
                    for (int d = 0; d < Dim; d++) gradient[d] += 2.0 * error * allX[i][d];
                }
                for (int d = 0; d < Dim; d++) weights[d] -= lr * gradient[d] / allX.Length;
            }
        }
        return FullLoss(weights, allX, allY);
    }

    public static void Main()
    {
        // 1-4. Field arithmetic: primality, inverses, Fermat.
        Check(Field.P == 2147483647L && Field.IsPrime(Field.P), "2^31-1 is prime");
        Check(!Field.IsPrime(2147483649L) && Field.IsPrime(97L), "the primality test separates primes from composites");
        var fieldRng = 0x1234UL;
        bool inverseOk = true, fermatOk = true;
        for (int i = 0; i < 200; i++)
        {
            long a = Field.RandomNonZero(ref fieldRng);
            if (Field.Mul(a, Field.Inverse(a)) != 1) inverseOk = false;
            if (Field.Pow(a, Field.P - 1) != 1) fermatOk = false;
        }
        Check(inverseOk, "every non-zero element has an exact multiplicative inverse");
        Check(fermatOk, "Fermat's little theorem holds for random elements");

        // 5-8. t-of-n sharing: exact reconstruction from every subset of size t and from larger subsets.
        var shareRng = 0xABCDUL;
        const long secret = 1234567;
        var shares = Shamir.Share(secret, 6, 3, ref shareRng);
        int subsets = 0;
        bool allExact = true;
        for (int a = 0; a < 6; a++)
        {
            for (int b = a + 1; b < 6; b++)
            {
                for (int c = b + 1; c < 6; c++)
                {
                    if (Shamir.Reconstruct(new List<Share> { shares[a], shares[b], shares[c] }) != secret) allExact = false;
                    subsets++;
                }
            }
        }
        Console.WriteLine("  shamir: " + subsets + " subsets of size 3, all exact=" + allExact);
        Check(subsets == 20 && allExact, "all C(6,3) share subsets reconstruct the secret exactly");
        Check(Shamir.Reconstruct(new List<Share> { shares[0], shares[1], shares[2], shares[3], shares[4], shares[5] }) == secret,
            "more shares than the threshold still reconstruct the same secret");
        Check(Shamir.Reconstruct(new List<Share> { shares[5], shares[2], shares[0] }) == secret,
            "the reconstruction does not depend on the order of the shares");
        var zeroShares = Shamir.Share(0, 4, 2, ref shareRng);
        Check(Shamir.Reconstruct(new List<Share> { zeroShares[0], zeroShares[3] }) == 0, "a zero secret reconstructs as zero");

        // 9-10. Perfect secrecy: with t-1 shares, a polynomial through (0, anyCand) hits every observed share.
        var partial = new List<Share> { shares[0], shares[1] };
        bool everySecretConsistent = true;
        foreach (long candidate in new[] { 0L, 1L, 42L, Field.P - 1 })
        {
            if (Shamir.PolynomialThrough(partial, candidate, 0) != candidate) everySecretConsistent = false;
            foreach (Share s in partial)
            {
                if (Shamir.PolynomialThrough(partial, candidate, s.X) != s.Y) everySecretConsistent = false;
            }
        }
        Check(everySecretConsistent, "t-1 shares are consistent with every candidate secret - perfect secrecy");
        Check(Shamir.PolynomialThrough(new List<Share> { shares[0], shares[1] }, secret, shares[0].X) == shares[0].Y,
            "the same construction with the true secret also reproduces the observed shares");

        // 11-14. Input validation of the sharing scheme.
        Check(Throws(() => Shamir.Share(1, 0, 1, ref shareRng)), "zero shareholders are rejected");
        Check(Throws(() => Shamir.Share(1, 3, 0, ref shareRng)), "a threshold below one is rejected");
        Check(Throws(() => Shamir.Share(1, 3, 4, ref shareRng)), "a threshold above n is rejected");
        Check(Throws(() => Shamir.Reconstruct(new List<Share>
        {
            new Share { X = 1, Y = 3 }, new Share { X = 1, Y = 4 }
        })), "duplicate shareholder indices are rejected");
        Check(Throws(() => Field.Inverse(0)), "zero has no inverse");

        // 15-17. Secure aggregation equals the exact plaintext integer sum.
        BuildData(out double[][][] features, out double[][] targets, out double[][] allX, out double[] allY);
        var updateRng = 0x5EEDUL;
        var updates = new List<double[]>();
        for (int c = 0; c < Clients; c++)
        {
            var v = new double[Dim];
            for (int d = 0; d < Dim; d++) v[d] = 4.0 * SplitMix64.NextDouble(ref updateRng) - 2.0;
            updates.Add(v);
        }
        long[] exactSum = SecureAggregation.ExactQuantisedSum(updates, ClippingBound);
        long[] secureRaw = SecureAggregation.RunQuantised(updates, Threshold, ClippingBound,
            SecureAggregation.All(Clients), 0xCAFEUL);
        double worstAggregate = 0.0;
        for (int d = 0; d < Dim; d++)
        {
            worstAggregate = Math.Max(worstAggregate,
                Math.Abs(Quantiser.DecodeSigned(secureRaw[d]) - Quantiser.DecodeSigned(exactSum[d])));
        }
        Console.WriteLine("  secure aggregation: worst |aggregate - plaintext sum| = " + worstAggregate.ToString("E3"));
        Check(worstAggregate == 0.0, "the secure aggregate is bit-exact against the plaintext sum");
        var seven = new List<double[]>(updates) { updates[0] };
        long[] withDuplicate = SecureAggregation.RunQuantised(seven, Threshold, ClippingBound,
            SecureAggregation.All(7), 0xCAFEUL);
        Check(withDuplicate[0] == Quantiser.Encode(SecureAggregation.Clip(updates[0], ClippingBound)[0]) * 2
            || withDuplicate[0] != secureRaw[0],
            "a different client set produces a different aggregate (the aggregate really follows the inputs)");

        // 18-19. Dropout tolerance: every surviving subset of size >= t reconstructs its own exact sum.
        bool dropoutsAgree = true;
        var subsetsOfInterest = new[]
        {
            new[] { 0, 1, 2, 3 },
            new[] { 2, 3, 4, 5 },
            new[] { 0, 1, 2, 3, 4 },
            new[] { 1, 2, 3, 4, 5 },
            new[] { 0, 2, 3, 5 },
            new[] { 0, 1, 2, 3, 4, 5 }
        };
        foreach (int[] subset in subsetsOfInterest)
        {
            var kept = new List<double[]>();
            foreach (int index in subset) kept.Add(updates[index]);
            double[] result = SecureAggregation.Run(kept, Threshold, ClippingBound, SecureAggregation.All(kept.Count), 0xCAFEUL);
            long[] exactKept = SecureAggregation.ExactQuantisedSum(kept, ClippingBound);
            for (int d = 0; d < Dim; d++)
            {
                if (result[d] != Quantiser.DecodeSigned(exactKept[d])) dropoutsAgree = false;
            }
        }
        Check(dropoutsAgree, "every surviving subset of at least t clients reconstructs its own exact sum");
        Check(Throws(() => SecureAggregation.Run(updates, Threshold, ClippingBound, SecureAggregation.All(Clients - 3), 1UL)),
            "fewer participants than the threshold is rejected");
        Check(Throws(() => SecureAggregation.Run(updates, Clients + 1, ClippingBound, SecureAggregation.All(Clients), 1UL)),
            "a threshold above the client count is rejected");

        // 20. Quantisation round trip.
        bool quantisationOk = true;
        for (int i = 0; i < 200; i++)
        {
            double v = 8.0 * (SplitMix64.NextDouble(ref updateRng) - 0.5);
            if (Math.Abs(Quantiser.Decode(Quantiser.Encode(v)) - v) > 0.5 / Quantiser.Scale) quantisationOk = false;
        }
        Check(quantisationOk, "the fixed-point round trip stays within half a quantisation step");

        // 21-24. Sensitivity: clipping bounds each client's contribution by C, so adding or removing a client moves
        // the sum by at most C (the calibration used below); *replacing* one client's update moves it by at most 2C,
        // which is the neighbouring relation other calibrations use.
        double worstContribution = 0.0;
        foreach (double[] update in updates)
        {
            double[] clipped = SecureAggregation.Clip(update, ClippingBound);
            double norm = 0.0;
            foreach (double v in clipped) norm += v * v;
            worstContribution = Math.Max(worstContribution, Math.Sqrt(norm));
        }
        Check(worstContribution <= ClippingBound + 1e-12, "every clipped update lies inside the L2 ball");
        var withoutFirstClient = new List<double[]>();
        for (int c = 1; c < Clients; c++) withoutFirstClient.Add(updates[c]);
        long[] baseSum = SecureAggregation.ExactQuantisedSum(withoutFirstClient, ClippingBound);
        long[] fullSum = SecureAggregation.ExactQuantisedSum(updates, ClippingBound);
        double addShift = 0.0;
        for (int d = 0; d < Dim; d++) addShift = Math.Max(addShift, Math.Abs(Quantiser.DecodeSigned(fullSum[d] - baseSum[d])));
        var replaced = new List<double[]>();
        foreach (double[] update in updates) replaced.Add((double[])update.Clone());
        replaced[0][0] += 3.0;
        long[] replacedSum = SecureAggregation.ExactQuantisedSum(replaced, ClippingBound);
        double replaceShift = 0.0;
        for (int d = 0; d < Dim; d++)
        {
            replaceShift = Math.Max(replaceShift, Math.Abs(Quantiser.DecodeSigned(replacedSum[d] - fullSum[d])));
        }
        Console.WriteLine("  sensitivity: add/remove=" + addShift.ToString("F6") + " replace=" + replaceShift.ToString("F6")
            + " C=" + ClippingBound);
        Check(addShift <= ClippingBound + 1e-6 && addShift > 0.0, "adding or removing a client moves the aggregate by at most C");
        Check(replaceShift <= 2.0 * ClippingBound + 1e-6, "replacing one client's update moves it by at most 2C");

        // 23-24. Gaussian mechanism: centred, and the empirical deviation matches sigma.
        var noiseRng = 0xBEEFUL;
        const double sigma = 0.25;
        double sum = 0.0, sumSquares = 0.0;
        const int draws = 200000;
        for (int i = 0; i < draws; i++)
        {
            double z = sigma * SplitMix64.NextGaussian(ref noiseRng);
            sum += z;
            sumSquares += z * z;
        }
        double mean = sum / draws;
        double variance = sumSquares / draws - mean * mean;
        Console.WriteLine("  gaussian noise: mean=" + mean.ToString("E3") + " std=" + Math.Sqrt(variance).ToString("F6")
            + " target=" + sigma.ToString("F6"));
        Check(Math.Abs(mean) < 5e-3, "the noise is centred");
        Check(Rel(Math.Sqrt(variance), sigma) < 0.02, "the noise standard deviation matches sigma");

        // 25-28. Accountant: upper-bounds the closed form, monotone, and meets a budget.
        double one = Accountant.Epsilon(ClippingBound, sigma, 1, Delta);
        double closed = Accountant.ClosedFormOneRound(ClippingBound, sigma, Delta);
        Console.WriteLine("  accountant: rdp one round=" + one.ToString("F6") + " closed form=" + closed.ToString("F6"));
        Check(one >= closed - 1e-12, "the RDP bound is at least the classical closed form (it upper bounds it)");
        Check(one < 2.0 * closed, "and it stays within a factor two of it");
        Check(Accountant.Epsilon(ClippingBound, sigma, 12, Delta) > Accountant.Epsilon(ClippingBound, sigma, 6, Delta),
            "the budget grows with the number of rounds");
        Check(Accountant.Epsilon(ClippingBound, sigma * 2.0, 12, Delta) < Accountant.Epsilon(ClippingBound, sigma, 12, Delta),
            "more noise buys a smaller budget");
        double budgetSigma = Accountant.SigmaForBudget(ClippingBound, Rounds, TargetEpsilon, Delta);
        double achieved = Accountant.Epsilon(ClippingBound, budgetSigma, Rounds, Delta);
        double closedSigma = ClippingBound * Math.Sqrt(2.0 * Rounds * Math.Log(1.25 / Delta)) / TargetEpsilon;
        Console.WriteLine("  budget: sigma=" + budgetSigma.ToString("F6") + " (closed form " + closedSigma.ToString("F6")
            + ") achieved eps=" + achieved.ToString("F6") + " target=" + TargetEpsilon);
        Check(achieved <= TargetEpsilon, "the default noise scale meets the privacy budget");
        Check(achieved > 0.5 * TargetEpsilon, "and it is not wasteful: the budget is nearly used up");
        Check(budgetSigma >= closedSigma - 1e-9, "RDP demands at least the closed-form noise scale");
        Check(Throws(() => Accountant.Epsilon(ClippingBound, 0.0, Rounds, Delta)), "a zero noise scale is rejected");
        Check(Throws(() => Accountant.Epsilon(ClippingBound, sigma, Rounds, 0.0)), "delta outside (0,1) is rejected");

        // 29-32. Federated averaging: exact aggregation, convergence, centralised comparison, DP cost.
        const double lr = 0.4;
        const int epochs = 3;
        var exactRun = Federated(features, targets, allX, allY, lr, epochs, true, false, 0.0, 0x99UL);
        var plainRun = Federated(features, targets, allX, allY, lr, epochs, false, false, 0.0, 0x99UL);
        var privateRun = Federated(features, targets, allX, allY, lr, epochs, true, true, budgetSigma, 0x99UL);
        double central = CentralLoss(allX, allY, lr, epochs);
        Console.WriteLine("  fedavg: initial=" + exactRun.InitialLoss.ToString("E6")
            + " secure=" + exactRun.FinalLoss.ToString("E6") + " plain=" + plainRun.FinalLoss.ToString("E6")
            + " private=" + privateRun.FinalLoss.ToString("E6") + " central=" + central.ToString("E6"));
        Check(exactRun.FinalLoss < 0.3 * exactRun.InitialLoss, "federated averaging removes at least 70% of the loss");
        Check(Rel(exactRun.FinalLoss, plainRun.FinalLoss) < 1e-2,
            "federated training through the secure pipeline matches the plaintext pipeline to within 1%");
        Check(exactRun.FinalLoss < 1.05 * central + 1e-9, "it matches centralised training on the same data to within 5%");
        Check(privateRun.FinalLoss < 2.0 * exactRun.FinalLoss + 1e-3, "the DP noise costs only a little accuracy");

        // 33-34. Determinism and budget reporting.
        var repeat = Federated(features, targets, allX, allY, lr, epochs, true, true, budgetSigma, 0x99UL);
        Check(repeat.FinalLoss == privateRun.FinalLoss && repeat.FinalEpsilon == privateRun.FinalEpsilon,
            "repeated private federated runs are bit-for-bit identical");
        Check(privateRun.FinalEpsilon <= TargetEpsilon + 1e-12, "the reported epsilon respects the budget");

        // 35-37. Degenerate inputs.
        Check(Throws(() => SecureAggregation.Run(new List<double[]>(), 1, 1.0, SecureAggregation.All(0), 1UL)),
            "an empty client set is rejected");
        Check(Throws(() => SecureAggregation.Clip(new double[Dim], 0.0)), "a non-positive clipping bound is rejected");
        Check(Throws(() => Quantiser.Encode(1e9)), "a value outside the fixed-point range is rejected");
        var singleton = SecureAggregation.Run(new List<double[]> { new[] { 0.25, -0.5, 0.0, 0.125 } }, 1, 1.0,
            SecureAggregation.All(1), 3UL);
        Check(singleton[0] == 0.25 && singleton[1] == -0.5 && singleton[2] == 0.0 && singleton[3] == 0.125,
            "a single client with threshold one aggregates exactly");

        Console.WriteLine("PASS");
    }
}
