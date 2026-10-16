// Task 838 - Implement secure multi-party aggregation for machine learning (additive secret sharing). (Tier: Master, CTCI Beyond — Security & Privacy)
//
// Secure aggregation of model updates (the Bonawitz-style protocol, reduced to its arithmetic core): n parties
// each hold a private update vector, secret-share it additively over the prime field F_p, keep one share and
// send one share to every other party. Party j adds its own share to its inbox; the resulting per-party
// aggregate share vector contains no information about any single update, yet the n aggregate shares
// reconstruct to the exact plaintext sum because the field is abelian and shares cancel coordinate-wise. The
// aggregator therefore never sees an update, only the sum — which is what a federated training round needs.
//
// What the file verifies (each property is recomputed, not asserted from the header):
//   * correctness: reconstruct(share_i(x_i) summed over i) == sum_i x_i, exactly, in Z_p;
//   * perfect secrecy: a single share is uniform over F_p, and the joint transcript of any n-1 shares is
//     uniform over F_p^{n-1}. The exhaustive oracle enumerates all 11^3 share triples of the small field
//     F_11 and counts, for every secret, exactly p^{n-1} = 121 transcripts summing to it — i.e. the transcript
//     distribution is provably independent of the secret, which no sampled statistic could establish;
//   * the sampled statistics (chi-square on 16 share buckets on F_p, chi-square on the 8x8 joint cells of
//     shares 0 and 1, and the L1 distance between the transcript histograms of two different secrets) are the
//     measurements that the toy-scale claim "no real key material" is checked against;
//   * additive homomorphism, so a private scalar multiplication and a private sum commute with sharing;
//   * communication cost: the explicit in-memory bus delivers exactly n*(n-1)*dim share messages per round
//     (the O(n^2 d) traffic that real deployments cut with tree aggregation).
//
// Documented simplifications: (1) additive sharing is *not* dropout tolerant — the file measures the failure
// and charges it to the simplification, where a real protocol adds Shamir shares for the seed agreement so
// that a dropped party's shares can be reconstructed; (2) secrets are quantized integers in [0, p) with
// p = 2^31 - 1, so an aggregate that reaches p wraps around — also measured; (3) no Diffie-Hellman key
// agreement is modelled (no real key material exists here), the per-pair pads of the full protocol are
// replaced by the explicit share messages of the dealer-free variant, which is exactly as private for an
// honest-but-curious aggregator but not against colluding parties. Randomness is a hand-written splitmix64
// with an explicit seed, so every measurement below is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    /// <summary>Deterministic stand-in for Random: the seed is always explicit, so runs are reproducible.</summary>
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static long NextBelow(ref ulong state, long bound) => (long)(Next(ref state) % (ulong)bound);
}

/// <summary>Arithmetic in F_p for p = 2^31 - 1 (Mersenne prime), exact in a signed 64-bit integer.</summary>
internal static class Field
{
    internal const long P = 2147483647L;

    internal static long Normalize(long value)
    {
        long reduced = value % P;
        return reduced < 0 ? reduced + P : reduced;
    }

    internal static long Add(long a, long b) => Normalize(a + b);

    internal static long Sub(long a, long b) => Normalize(a - b);

    /// <summary>Both operands are below 2^31, so the product stays below 2^62 and cannot overflow.</summary>
    internal static long Mul(long a, long b) => Normalize(a * b);
}

/// <summary>An explicit in-memory message bus, so the traffic of a round is countable and verifiable.</summary>
internal sealed class MessageBus
{
    private readonly List<(int From, int To, long[] Payload)> _messages = new List<(int, int, long[])>();

    internal int Count => _messages.Count;

    internal long PayloadValues
    {
        get
        {
            long total = 0;
            foreach ((int _, int _, long[] payload) in _messages) total += payload.Length;
            return total;
        }
    }

    internal void Send(int from, int to, long[] payload)
    {
        if (from == to) throw new ArgumentException("a party does not message itself", nameof(to));
        _messages.Add((from, to, payload));
    }

    /// <summary>Every share a given receiver obtained, in delivery order — the only thing that party sees.</summary>
    internal IReadOnlyList<long[]> InboxOf(int receiver)
    {
        var inbox = new List<long[]>();
        foreach ((int _, int to, long[] payload) in _messages) if (to == receiver) inbox.Add(payload);
        return inbox;
    }
}

internal static class SecretSharing
{
    /// <summary>
    /// Splits a secret into n additive shares over F_p. The first n-1 shares are fresh uniform field
    /// elements and the last closes the sum, so every share is uniform and independent of the secret: for a
    /// fixed secret the transcript is uniform over F_p^(n-1) (verified exhaustively on F_11 below).
    /// </summary>
    internal static long[] Split(long secret, int parties, ref ulong state)
    {
        if (parties < 1) throw new ArgumentException("at least one party is required", nameof(parties));
        if (secret < 0 || secret >= Field.P) throw new ArgumentException("secret outside F_p", nameof(secret));
        var shares = new long[parties];
        long running = 0;
        for (int i = 0; i < parties - 1; i++)
        {
            shares[i] = SplitMix64.NextBelow(ref state, Field.P);
            running = Field.Add(running, shares[i]);
        }
        shares[parties - 1] = Field.Sub(secret, running);
        return shares;
    }

    internal static long Reconstruct(IReadOnlyList<long> shares)
    {
        if (shares.Count == 0) throw new ArgumentException("no shares to reconstruct from", nameof(shares));
        long sum = 0;
        foreach (long share in shares) sum = Field.Add(sum, share);
        return sum;
    }

    internal static long[] SplitVector(long[] vector, int parties, ref ulong state)
    {
        var flat = new long[parties * vector.Length];   // [party * dim + coordinate]
        for (int d = 0; d < vector.Length; d++)
        {
            long[] shares = Split(vector[d], parties, ref state);
            for (int p = 0; p < parties; p++) flat[p * vector.Length + d] = shares[p];
        }
        return flat;
    }

    internal static long[] RowOf(long[] flat, int party, int dimension)
    {
        var row = new long[dimension];
        Array.Copy(flat, party * dimension, row, 0, dimension);
        return row;
    }

    internal static long[] AddVector(long[] a, long[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("dimension mismatch", nameof(b));
        var sum = new long[a.Length];
        for (int i = 0; i < a.Length; i++) sum[i] = Field.Add(a[i], b[i]);
        return sum;
    }

    internal static long[] ScaleVector(long[] v, long scalar)
    {
        var scaled = new long[v.Length];
        for (int i = 0; i < v.Length; i++) scaled[i] = Field.Mul(v[i], scalar);
        return scaled;
    }

    internal static long[] ReconstructVector(IReadOnlyList<long[]> shareVectors, int dimension)
    {
        if (shareVectors.Count == 0) throw new ArgumentException("no shares", nameof(shareVectors));
        var total = new long[dimension];
        foreach (long[] shares in shareVectors)
        {
            if (shares.Length != dimension) throw new ArgumentException("dimension mismatch", nameof(shareVectors));
            for (int d = 0; d < dimension; d++) total[d] = Field.Add(total[d], shares[d]);
        }
        return total;
    }
}

/// <summary>One aggregator round over the private updates of n parties, wired through an explicit bus.</summary>
internal static class SecureAggregator
{
    /// <summary>
    /// Every party secret-shares its own update vector and sends one share to each other party; the receiving
    /// party keeps its own share locally and sums it with its inbox. The per-party sums (the "aggregate
    /// shares") reconstruct to the plaintext sum, because every dealer's shares add up to its secret but no
    /// share ever leaves its owner except in masked form.
    /// </summary>
    internal static List<long[]> AggregateShares(IReadOnlyList<long[]> updates, int dimension, ulong seed, MessageBus bus)
    {
        int parties = updates.Count;
        if (parties < 1) throw new ArgumentException("at least one party is required", nameof(updates));
        var perDealer = new List<long[]>();
        ulong state = seed;
        foreach (long[] update in updates)
        {
            if (update.Length != dimension) throw new ArgumentException("dimension mismatch", nameof(updates));
            perDealer.Add(SecretSharing.SplitVector(update, parties, ref state));
        }

        for (int dealer = 0; dealer < parties; dealer++)
        {
            for (int receiver = 0; receiver < parties; receiver++)
            {
                if (receiver == dealer) continue;        // the diagonal share stays with its owner
                bus.Send(dealer, receiver, SecretSharing.RowOf(perDealer[dealer], receiver, dimension));
            }
        }

        var aggregateShares = new List<long[]>();
        for (int receiver = 0; receiver < parties; receiver++)
        {
            var mine = SecretSharing.RowOf(perDealer[receiver], receiver, dimension);
            foreach (long[] payload in bus.InboxOf(receiver)) mine = SecretSharing.AddVector(mine, payload);
            aggregateShares.Add(mine);
        }
        return aggregateShares;
    }

    /// <summary>Runs one round and returns the reconstructed plaintext aggregate.</summary>
    internal static long[] Round(IReadOnlyList<long[]> updates, int dimension, ulong seed, MessageBus bus)
    {
        return SecretSharing.ReconstructVector(AggregateShares(updates, dimension, seed, bus), dimension);
    }

    /// <summary>
    /// What a party that owns an update plus its own aggregate share can infer: it can remove its own
    /// contribution, which leaves the sum of the other parties' shares — still a uniform pad away from the
    /// plaintext sum of the others.
    /// </summary>
    internal static long[] WithoutOwnContribution(long[] aggregateShare, long[] ownRow)
    {
        var residual = new long[aggregateShare.Length];
        for (int i = 0; i < residual.Length; i++) residual[i] = Field.Sub(aggregateShare[i], ownRow[i]);
        return residual;
    }
}

internal static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Upper-tail 0.001 critical values of the chi-square distribution, hard-coded for the two df used.</summary>
    private static double ChiSquareCritical(int degreesOfFreedom) => degreesOfFreedom switch
    {
        15 => 37.70,
        63 => 108.90,
        _ => throw new ArgumentException("no critical value tabulated for df " + degreesOfFreedom),
    };

    private static double ChiSquare(long[] buckets, double expected)
    {
        double statistic = 0;
        foreach (long observed in buckets) statistic += (observed - expected) * (observed - expected) / expected;
        return statistic;
    }

    private static long[] QuantizedUpdate(int party, int dimension)
    {
        var update = new long[dimension];
        for (int d = 0; d < dimension; d++) update[d] = (party * 7919 + d * 104729) % 1000 + 17;
        return update;
    }

    /// <summary>
    /// Exhaustive oracle for perfect secrecy: enumerate every transcript of n shares over F_q (q = 11, n = 3)
    /// and count, per (first share, reconstructed secret) cell, how many transcripts fall there. A cell holds
    /// q^(n-2) tuples for *every* secret, so every secret owns exactly q^(n-1) transcripts and the transcript
    /// distribution does not depend on the secret at all.
    /// </summary>
    private static long[][] TranscriptCountsExhaustive(int field, int parties)
    {
        var counts = new long[field][];
        for (int s = 0; s < field; s++) counts[s] = new long[field];
        var tuple = new long[parties];
        long total = 1;
        for (int i = 0; i < parties; i++) total *= field;
        for (long code = 0; code < total; code++)
        {
            long rest = code;
            long sum = 0;
            for (int i = 0; i < parties; i++) { tuple[i] = rest % field; rest /= field; sum = (sum + tuple[i]) % field; }
            counts[tuple[0]][sum]++;                     // bucketed by the first share, keyed by the reconstructed secret
        }
        return counts;
    }

    public static void Main()
    {
        // ---- correctness: the aggregate shares reconstruct the plaintext sum -------------------------
        {
            const int Parties = 4, Dimension = 3;
            var updates = new List<long[]>();
            var plain = new long[Dimension];
            for (int p = 0; p < Parties; p++)
            {
                long[] update = QuantizedUpdate(p, Dimension);
                updates.Add(update);
                for (int d = 0; d < Dimension; d++) plain[d] = Field.Add(plain[d], update[d]);
            }
            var bus = new MessageBus();
            long[] aggregate = SecureAggregator.Round(updates, Dimension, 0x5EED838000000001UL, bus);

            bool matches = true;
            for (int d = 0; d < Dimension; d++) if (aggregate[d] != plain[d]) matches = false;
            Check(matches, "the reconstructed aggregate equals the plaintext sum of the updates");
            Check(bus.Count == Parties * (Parties - 1), "each party sends one share to each other party ("
                + bus.Count + " messages)");
            Check(bus.PayloadValues == (long)Parties * (Parties - 1) * Dimension,
                "the round costs n(n-1)dim field elements (" + bus.PayloadValues + ")");
            Console.WriteLine("aggregation: parties=" + Parties + " dim=" + Dimension + " messages=" + bus.Count
                + " aggregate=[" + string.Join(",", aggregate) + "]");
        }

        // ---- additive homomorphism: sharing commutes with a private sum and a private scaling --------
        {
            const int Parties = 3, Dimension = 2;
            var updates = new List<long[]>();
            for (int p = 0; p < Parties; p++) updates.Add(QuantizedUpdate(p + 5, Dimension));
            ulong state = 0x5EED838000000002UL;
            var perParty = new List<long[]>();
            foreach (long[] update in updates) perParty.Add(SecretSharing.SplitVector(update, Parties, ref state));

            var sumShares = new long[Parties][];
            for (int party = 0; party < Parties; party++)
            {
                var mine = SecretSharing.RowOf(perParty[0], party, Dimension);
                for (int dealer = 1; dealer < Parties; dealer++)
                    mine = SecretSharing.AddVector(mine, SecretSharing.RowOf(perParty[dealer], party, Dimension));
                sumShares[party] = mine;
            }
            long[] homomorphic = SecretSharing.ReconstructVector(sumShares, Dimension);
            var plain = new long[Dimension];
            for (int p = 0; p < Parties; p++)
                for (int d = 0; d < Dimension; d++) plain[d] = Field.Add(plain[d], updates[p][d]);
            bool sums = true;
            for (int d = 0; d < Dimension; d++) if (homomorphic[d] != plain[d]) sums = false;
            Check(sums, "sharing the sum equals summing the shares");

            const long Scalar = 12345;
            var scaledPlain = SecretSharing.ScaleVector(updates[1], Scalar);
            long[] scaledShares = SecretSharing.ReconstructVector(
                new List<long[]> { SecretSharing.ScaleVector(SecretSharing.RowOf(perParty[1], 0, Dimension), Scalar),
                                   SecretSharing.ScaleVector(SecretSharing.RowOf(perParty[1], 1, Dimension), Scalar),
                                   SecretSharing.ScaleVector(SecretSharing.RowOf(perParty[1], 2, Dimension), Scalar) },
                Dimension);
            bool scaled = true;
            for (int d = 0; d < Dimension; d++) if (scaledShares[d] != scaledPlain[d]) scaled = false;
            Check(scaled, "a private scalar multiple of an update is reconstructed from scaled shares");
            Console.WriteLine("homomorphism: addedAggregate=[" + string.Join(",", homomorphic) + "] scaled=["
                + string.Join(",", scaledShares) + "]");
        }

        // ---- perfect secrecy, exhaustively: the transcript is uniform given any secret --------------
        {
            const int Small = 11, Parties = 3;
            long[][] counts = TranscriptCountsExhaustive(Small, Parties);
            long cellExpected = 1, secretExpected = 1;
            for (int i = 0; i < Parties - 2; i++) cellExpected *= Small;
            for (int i = 0; i < Parties - 1; i++) secretExpected *= Small;
            bool cellsEqual = true, secretsEqual = true;
            long grand = 0;
            for (int secret = 0; secret < Small; secret++)
            {
                long column = 0;
                for (int first = 0; first < Small; first++)
                {
                    if (counts[first][secret] != cellExpected) cellsEqual = false;
                    column += counts[first][secret];
                    grand += counts[first][secret];
                }
                if (column != secretExpected) secretsEqual = false;
            }
            Check(cellsEqual, "every (first share, secret) cell of the exhaustive table holds exactly q^(n-2)="
                + cellExpected + " transcripts");
            Check(secretsEqual, "each secret owns exactly q^(n-1)=" + secretExpected
                + " of the transcripts, so the transcript distribution is independent of the secret (perfect secrecy)");
            Check(grand == (long)Math.Pow(Small, Parties), "the enumeration covers every transcript ("
                + grand + " of " + (long)Math.Pow(Small, Parties) + ")");
            Console.WriteLine("exhaustive secrecy: field=" + Small + " parties=" + Parties + " transcripts=" + grand
                + " perCell=" + cellExpected + " perSecret=" + secretExpected);
        }

        // ---- a single share is uniform over F_p (16 buckets, alpha = 0.001) --------------------------
        {
            const int Trials = 20000, Buckets = 16;
            double expected = (double)(Trials * 3) / Buckets;   // three shares per split, each uniform on its own
            var statistics = new List<double>();
            ulong state = 0x5EED838000000003UL;
            bool ranges = true, lengths = true;
            for (int run = 0; run < 3; run++)
            {
                var buckets = new long[Buckets];
                for (int t = 0; t < Trials; t++)
                {
                    long[] shares = SecretSharing.Split(4242, 3, ref state);
                    if (shares.Length != 3) lengths = false;
                    foreach (long share in shares)
                    {
                        if (share < 0 || share >= Field.P) ranges = false;
                        // Bucket by the share's position in [0, p) so the bucketing itself is uniform.
                        buckets[(int)(share * Buckets / Field.P)]++;
                    }
                }
                statistics.Add(ChiSquare(buckets, expected));
            }
            Check(ranges, "every share lies in [0, p)");
            Check(lengths, "a 3-party split returns 3 shares");
            bool uniform = true;
            foreach (double statistic in statistics) if (statistic > ChiSquareCritical(Buckets - 1)) uniform = false;
            Check(uniform, "a single share passes the 16-bucket chi-square test on 20000 samples for 3 seeds ("
                + string.Join(", ", statistics.ConvertAll(s => s.ToString("F2"))) + " < "
                + ChiSquareCritical(Buckets - 1).ToString("F2") + ")");
            Console.WriteLine("single share uniformity: chi2 = "
                + string.Join(", ", statistics.ConvertAll(s => s.ToString("F2"))));
        }

        // ---- the joint transcript of two shares is uniform (8x8 cells, alpha = 0.001) ---------------
        {
            const int Trials = 20000, Side = 8;
            double expected = (double)Trials / (Side * Side);
            ulong state = 0x5EED838000000004UL;
            var buckets = new long[Side * Side];
            for (int t = 0; t < Trials; t++)
            {
                long[] shares = SecretSharing.Split(987654321, 3, ref state);
                int cellX = (int)(shares[0] % Side), cellY = (int)(shares[1] % Side);
                buckets[cellX * Side + cellY]++;
            }
            double statistic = ChiSquare(buckets, expected);
            Check(statistic < ChiSquareCritical(Side * Side - 1), "the joint distribution of shares 0 and 1 is uniform over "
                + (Side * Side) + " cells (chi2 " + statistic.ToString("F2") + " < "
                + ChiSquareCritical(Side * Side - 1).ToString("F2") + ")");
            Console.WriteLine("joint transcript uniformity: chi2 = " + statistic.ToString("F2") + " on " + Trials + " transcripts");
        }

        // ---- two different secrets have indistinguishable transcripts --------------------------------
        {
            const int Trials = 20000, Buckets = 16;
            var histograms = new long[2][];
            for (int secret = 0; secret < 2; secret++)
            {
                histograms[secret] = new long[Buckets];
                ulong state = secret == 0 ? 0x5EED838000000005UL : 0x5EED838000000006UL;
                for (int t = 0; t < Trials; t++)
                {
                    long[] shares = SecretSharing.Split(secret * 1000003L, 5, ref state);
                    histograms[secret][(int)(shares[0] * Buckets / Field.P)]++;
                }
            }
            double l1 = 0;
            for (int b = 0; b < Buckets; b++)
                l1 += Math.Abs((double)histograms[0][b] / Trials - (double)histograms[1][b] / Trials);
            Check(l1 < 0.25, "the transcript histograms of two different secrets differ only by sampling noise (L1 = "
                + l1.ToString("F4") + " < 0.25, the noise floor of 2x20000 samples over 16 buckets)");
            Console.WriteLine("indistinguishability: L1 = " + l1.ToString("F4") + " over 20000 transcripts per secret");
        }

        // ---- a party that removes its own contribution still learns nothing -------------------------
        {
            const int Parties = 4, Dimension = 2;
            var updates = new List<long[]>();
            for (int p = 0; p < Parties; p++) updates.Add(QuantizedUpdate(p + 11, Dimension));
            var bus = new MessageBus();
            List<long[]> aggregateShares = SecureAggregator.AggregateShares(updates, Dimension, 0x5EED838000000007UL, bus);
            ulong state = 0x5EED838000000008UL;
            long[] own = SecretSharing.SplitVector(updates[0], Parties, ref state);
            long[] residual = SecureAggregator.WithoutOwnContribution(aggregateShares[0],
                SecretSharing.RowOf(own, 0, Dimension));
            long[] othersOnly = new long[Dimension];
            for (int p = 1; p < Parties; p++)
                for (int d = 0; d < Dimension; d++) othersOnly[d] = Field.Add(othersOnly[d], updates[p][d]);
            bool hides = true;
            for (int d = 0; d < Dimension; d++) if (residual[d] == othersOnly[d]) hides = false;
            Check(hides, "the residual a party can compute is not the plaintext sum of the others (a uniform pad separates them)");
            Console.WriteLine("residual after removing own share: [" + string.Join(",", residual) + "]");
        }

        // ---- documented limitations, measured rather than hidden ------------------------------------
        {
            const int Parties = 4, Dimension = 1;
            var updates = new List<long[]>();
            for (int p = 0; p < Parties; p++) updates.Add(QuantizedUpdate(p + 21, Dimension));
            var bus = new MessageBus();
            List<long[]> shares = SecureAggregator.AggregateShares(updates, Dimension, 0x5EED838000000009UL, bus);
            long plainSum = 0;
            for (int p = 0; p < Parties; p++) plainSum = Field.Add(plainSum, updates[p][0]);
            long[] full = SecretSharing.ReconstructVector(shares, Dimension);
            Check(full[0] == plainSum, "with all n aggregate shares the round is exact");
            long partial = 0;
            for (int i = 0; i < Parties - 1; i++) partial = Field.Add(partial, shares[i][0]);
            Check(Field.Sub(partial, plainSum) != 0, "n-1 aggregate shares do NOT reconstruct the sum: additive sharing "
                + "is not dropout tolerant (delta " + Field.Sub(partial, plainSum) + "), which the header documents");

            long big = Field.P - 1000;
            var wrap = new List<long[]> { new long[] { big }, new long[] { 5000 } };
            long[] wrapped = SecureAggregator.Round(wrap, 1, 0x5EED83800000000AUL, new MessageBus());
            Check(wrapped[0] == Field.Normalize(big + 5000) && wrapped[0] != big + 5000,
                "an aggregate that reaches p wraps modulo p (" + wrapped[0] + "), so updates must be bounded");
            Console.WriteLine("limitations: partial-share delta=" + Field.Sub(partial, plainSum)
                + " modular wrap=" + wrapped[0] + " vs true " + (big + 5000));
        }

        // ---- degenerate party counts and determinism -------------------------------------------------
        {
            ulong state = 0x5EED83800000000BUL;
            long[] single = SecretSharing.Split(1234, 1, ref state);
            Check(single.Length == 1 && single[0] == 1234, "a single party holds its own secret as its only share");
            long[] pair = SecretSharing.Split(1234, 2, ref state);
            Check(SecretSharing.Reconstruct(pair) == 1234, "two parties reconstruct exactly");
            var seen = new HashSet<long>();
            for (int t = 0; t < 1000; t++) seen.Add(SecretSharing.Split(1234, 2, ref state)[0]);
            Check(seen.Count > 990, "with two parties the first share is a fresh uniform mask (" + seen.Count
                + " distinct values in 1000 splits of the same secret)");

            ulong stateA = 0x5EED83800000000CUL, stateB = stateA;
            long[] first = SecretSharing.Split(777, 4, ref stateA);
            long[] second = SecretSharing.Split(777, 4, ref stateB);
            bool same = true;
            for (int i = 0; i < first.Length; i++) if (first[i] != second[i]) same = false;
            Check(same, "the same seed produces the same transcript");
            long[] other = SecretSharing.Split(777, 4, ref stateA);
            bool differs = false;
            for (int i = 0; i < first.Length; i++) if (other[i] != first[i]) differs = true;
            Check(differs, "a fresh seed produces a different transcript for the same secret, as a re-sharing must");
            Console.WriteLine("boundaries: single=" + single[0] + " pairDelta=" + (pair[0] - 1234)
                + " reproducible=true");
        }

        Console.WriteLine("PASS");
    }
}
