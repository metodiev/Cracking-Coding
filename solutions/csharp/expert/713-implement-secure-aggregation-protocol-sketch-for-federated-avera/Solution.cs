// Task 713 - Implement secure aggregation protocol sketch for federated averaging.
// (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// Secure aggregation (Bonawitz et al.) lets a server learn the *sum* of the clients' vectors without
// learning any individual vector. Every pair of clients (i, k) shares a mask vector m_ik; client i
// adds +m_ik for every k > i and subtracts m_ki for every k < i, so the masks cancel pairwise in
// the total and the server's sum of masked vectors is exactly the sum of the plaintexts - but no
// single masked vector is meaningful on its own. Dropouts are handled by secret sharing: every
// client secret-shares the masks it holds with the threshold Shamir scheme, so when clients vanish
// the survivors can reconstruct exactly the missing masks and subtract their contribution.
//
// Documented simplification: the pairwise seeds are produced by the deterministic splitmix64 stream
// (keyed by the pair) instead of a Diffie-Hellman exchange, and the Shamir shares carry the mask
// vectors themselves rather than key material; the arithmetic that matters - mask cancellation,
// dropout recovery by threshold reconstruction, and tamper detection - is the real protocol.
//
// Verified below: the masks necessarily cancel (their total is zero mod p), every masked vector
// differs from its plaintext (so the server never sees a client's data), the aggregate equals the
// plaintext sum exactly, one and two dropouts are repaired to the exact sum over the survivors,
// Shamir reconstruction works for every t-subset while t-1 shares leave the secret undetermined,
// a tampered share is caught by its checksum, and the degenerate configurations (one client, one
// dimension, threshold equal to the client count, everybody dropping) behave sensibly.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static long NextMod(ref ulong state, long modulus) => (long)(Next(ref state) % (ulong)modulus);
}

/// <summary>Modular arithmetic over a prime field, implemented with long arithmetic.</summary>
internal static class Field
{
    internal const long Prime = 1000000007L;

    internal static long Normalize(long value)
    {
        long reduced = value % Prime;
        return reduced < 0 ? reduced + Prime : reduced;
    }

    internal static long Add(long a, long b) => Normalize(Normalize(a) + Normalize(b));

    internal static long Subtract(long a, long b) => Normalize(Normalize(a) - Normalize(b));

    internal static long Multiply(long a, long b) => Normalize(Normalize(a) * Normalize(b));

    internal static long Power(long basis, long exponent)
    {
        long result = 1, b = Normalize(basis), e = exponent;
        while (e > 0)
        {
            if ((e & 1) == 1) result = Multiply(result, b);
            b = Multiply(b, b);
            e >>= 1;
        }
        return result;
    }

    internal static long Inverse(long value)
    {
        long v = Normalize(value);
        if (v == 0) throw new ArgumentException("zero has no inverse in the field");
        return Power(v, Prime - 2);                  // Fermat: p is prime
    }
}

/// <summary>Threshold secret sharing over the prime field.</summary>
internal static class Shamir
{
    /// <summary>Splits a secret into shares[1..count]; any <paramref name="threshold"/> of them rebuild it.</summary>
    internal static long[] Split(long secret, int threshold, int count, ref ulong state)
    {
        if (threshold <= 0 || threshold > count) throw new ArgumentOutOfRangeException(nameof(threshold));
        var coefficients = new long[threshold];
        coefficients[0] = Field.Normalize(secret);
        for (int i = 1; i < threshold; i++) coefficients[i] = Rng.NextMod(ref state, Field.Prime);
        var shares = new long[count + 1];             // index 0 is unused: share i belongs to x = i
        for (int x = 1; x <= count; x++)
        {
            long value = 0, power = 1;
            for (int i = 0; i < threshold; i++)
            {
                value = Field.Add(value, Field.Multiply(coefficients[i], power));
                power = Field.Multiply(power, x);
            }
            shares[x] = value;
        }
        return shares;
    }

    /// <summary>Lagrange interpolation at x = 0 over the given (x, y) pairs.</summary>
    internal static long Reconstruct(IReadOnlyList<(int X, long Y)> points)
    {
        if (points.Count == 0) throw new ArgumentException("no shares were supplied", nameof(points));
        long secret = 0;
        for (int i = 0; i < points.Count; i++)
        {
            long numerator = 1, denominator = 1;
            for (int j = 0; j < points.Count; j++)
            {
                if (i == j) continue;
                if (points[i].X == points[j].X) throw new ArgumentException("duplicate share indices");
                numerator = Field.Multiply(numerator, Field.Normalize(-points[j].X));
                denominator = Field.Multiply(denominator, Field.Subtract(points[i].X, points[j].X));
            }
            long weight = Field.Multiply(numerator, Field.Inverse(denominator));
            secret = Field.Add(secret, Field.Multiply(points[i].Y, weight));
        }
        return secret;
    }
}

internal sealed class SecureAggregation
{
    private readonly int _clients, _dimension, _threshold;
    private readonly long[][] _values;
    private readonly long[][][] _pairMasks;        // [i][k][d] mask shared by clients i and k
    private readonly long[][][][] _shareOf;        // [owner][other][holder][d] share of the pair mask
    private readonly ulong[] _checksums;           // [owner * clients + holder] commitment to a share bundle
    internal long[]? LastAggregate;

    internal SecureAggregation(long[][] clientValues, int threshold, ulong seed)
    {
        if (clientValues.Length == 0) throw new ArgumentException("at least one client is required", nameof(clientValues));
        _clients = clientValues.Length;
        _dimension = clientValues[0].Length;
        _threshold = threshold;
        for (int i = 0; i < _clients; i++)
        {
            if (clientValues[i].Length != _dimension) throw new ArgumentException("all vectors must have the same length");
            foreach (long value in clientValues[i])
                if (value < 0 || value >= Field.Prime) throw new ArgumentOutOfRangeException(nameof(clientValues), "values must be field elements");
        }
        if (threshold <= 0 || threshold > _clients) throw new ArgumentOutOfRangeException(nameof(threshold));
        _values = clientValues;

        ulong state = seed;
        _pairMasks = new long[_clients][][];
        for (int i = 0; i < _clients; i++)
        {
            _pairMasks[i] = new long[_clients][];
            for (int k = 0; k < _clients; k++)
            {
                _pairMasks[i][k] = new long[_dimension];
                if (i >= k) continue;                            // the pair (i, k) is built once, for i < k
                ulong pairState = seed ^ ((ulong)(i * 1000003 + k) * 0x9E3779B97F4A7C15UL);
                for (int d = 0; d < _dimension; d++) _pairMasks[i][k][d] = Rng.NextMod(ref pairState, Field.Prime) + 1;
            }
        }
        // Mirror the pair masks so that both owners see the same vector.
        for (int i = 0; i < _clients; i++)
            for (int k = 0; k < i; k++)
                _pairMasks[i][k] = _pairMasks[k][i];

        _shareOf = new long[_clients][][][];
        _checksums = new ulong[_clients * _clients];
        for (int owner = 0; owner < _clients; owner++)
        {
            _shareOf[owner] = new long[_clients][][];
            for (int other = 0; other < _clients; other++)
            {
                if (other == owner) continue;
                _shareOf[owner][other] = new long[_clients][];
                for (int holder = 0; holder < _clients; holder++) _shareOf[owner][other][holder] = new long[_dimension];
                for (int d = 0; d < _dimension; d++)
                {
                    // One Shamir instance per mask coordinate; share x is the holder index plus one.
                    long[] column = Shamir.Split(_pairMasks[owner][other][d], _threshold, _clients, ref state);
                    for (int holder = 0; holder < _clients; holder++) _shareOf[owner][other][holder][d] = column[holder + 1];
                }
            }
            for (int holder = 0; holder < _clients; holder++) _checksums[owner * _clients + holder] = BundleChecksum(owner, holder);
        }
    }

    private ulong BundleChecksum(int owner, int holder)
    {
        var flat = new List<long>();
        for (int other = 0; other < _clients; other++)
        {
            if (other == owner) continue;
            for (int d = 0; d < _dimension; d++) flat.Add(_shareOf[owner][other][holder][d]);
        }
        return Checksum(flat.ToArray());
    }

    private static ulong Checksum(long[] share)
    {
        ulong hash = 0xCBF29CE484222325UL;
        foreach (long value in share)
        {
            hash ^= (ulong)value;
            hash *= 0x100000001B3UL;
        }
        return hash;
    }

    internal int Clients => _clients;
    internal int Dimension => _dimension;
    internal long[] ValueOf(int client) => _values[client];
    internal long[] MaskOf(int owner, int other) => _pairMasks[owner][other];
    internal bool ShareIsIntact(int owner, int holder) =>
        BundleChecksum(owner, holder) == _checksums[owner * _clients + holder];

    /// <summary>Corrupts a share to model a malicious holder; the checksum comparison must catch it.</summary>
    internal void TamperShare(int owner, int other, int holder, int coordinate, long delta)
    {
        _shareOf[owner][other][holder][coordinate] =
            Field.Add(_shareOf[owner][other][holder][coordinate], delta);
    }

    private long[] ClientMask(int client)
    {
        var mask = new long[_dimension];
        for (int k = 0; k < _clients; k++)
        {
            if (k == client) continue;
            long[] pair = _pairMasks[client][k];
            for (int d = 0; d < _dimension; d++)
            {
                mask[d] = k > client
                    ? Field.Add(mask[d], pair[d])
                    : Field.Subtract(mask[d], pair[d]);
            }
        }
        return mask;
    }

    /// <summary>The masked vector a client submits.</summary>
    internal long[] Masked(int client)
    {
        var mask = ClientMask(client);
        var submitted = new long[_dimension];
        for (int d = 0; d < _dimension; d++) submitted[d] = Field.Add(_values[client][d], mask[d]);
        return submitted;
    }

    /// <summary>Checks that the masks cancel: the sum of the client masks is zero.</summary>
    internal bool MasksCancel()
    {
        for (int d = 0; d < _dimension; d++)
        {
            long total = 0;
            for (int client = 0; client < _clients; client++) total = Field.Add(total, ClientMask(client)[d]);
            if (total != 0) return false;
        }
        return true;
    }

    /// <summary>
    /// Aggregates the submissions of the online clients, reconstructing the masks of every dropped
    /// client from its shares and adding the correction terms to the survivors' submissions.
    /// </summary>
    internal long[] Aggregate(IReadOnlyList<int> online, IReadOnlyList<int> dropped)
    {
        var total = new long[_dimension];
        foreach (int client in online)
        {
            var submitted = Masked(client);
            for (int d = 0; d < _dimension; d++) total[d] = Field.Add(total[d], submitted[d]);
        }
        if (dropped.Count == 0)
        {
            LastAggregate = total;
            return total;
        }
        if (online.Count > 0 && online.Count < _threshold)
            throw new InvalidOperationException("too few online clients to reconstruct the dropped masks");
        foreach (int gone in dropped)
        {
            foreach (int survivor in online)
            {
                for (int d = 0; d < _dimension; d++)
                {
                    var column = new List<(int X, long Y)>();
                    foreach (int holder in online) column.Add((holder + 1, _shareOf[gone][survivor][holder][d]));
                    long recovered = Shamir.Reconstruct(column);
                    if (recovered != _pairMasks[gone][survivor][d])
                        throw new InvalidOperationException("the reconstructed mask does not match the original");
                    long correction = survivor > gone ? recovered : Field.Normalize(-recovered);
                    total[d] = Field.Add(total[d], correction);
                }
            }
        }
        LastAggregate = total;
        return total;
    }

    /// <summary>The plaintext sum over a set of clients, for verification.</summary>
    internal long[] PlaintextSum(IReadOnlyList<int> clients)
    {
        var total = new long[_dimension];
        foreach (int client in clients)
            for (int d = 0; d < _dimension; d++) total[d] = Field.Add(total[d], _values[client][d]);
        return total;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool Same(long[] a, long[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static string Show(long[] values)
    {
        var parts = new List<string>();
        foreach (long v in values) parts.Add(v.ToString());
        return "[" + string.Join(",", parts) + "]";
    }

    private static long[][] RandomVectors(int clients, int dimension, int magnitude, ulong seed)
    {
        var values = new long[clients][];
        var state = seed;
        for (int i = 0; i < clients; i++)
        {
            values[i] = new long[dimension];
            for (int d = 0; d < dimension; d++) values[i][d] = Rng.NextMod(ref state, magnitude);
        }
        return values;
    }

    public static void Main()
    {
        // ---- the masks cancel and hide the individual vectors ----------------------------------
        {
            var protocol = new SecureAggregation(RandomVectors(8, 4, 1000, 0xA1CEUL), 5, 0x5EEDUL);
            Check(protocol.Clients == 8 && protocol.Dimension == 4, "the setup has 8 clients of dimension 4");
            Check(protocol.MasksCancel(), "the sum of all client masks is zero, so the masks cancel");

            int hidden = 0, exponents = 0;
            for (int i = 0; i < protocol.Clients; i++)
            {
                long[] masked = protocol.Masked(i);
                long[] plain = protocol.ValueOf(i);
                bool differs = false;
                for (int d = 0; d < protocol.Dimension; d++) if (masked[d] != plain[d]) differs = true;
                if (differs) hidden++;
                if (masked.Length == protocol.Dimension) exponents++;
            }
            Check(hidden == protocol.Clients, "every masked vector differs from its plaintext (" + hidden + "/8)");
            Check(exponents == 8, "every submission has the right shape");

            var everyone = new List<int>();
            for (int i = 0; i < protocol.Clients; i++) everyone.Add(i);
            long[] aggregate = protocol.Aggregate(everyone, new List<int>());
            Check(Same(aggregate, protocol.PlaintextSum(everyone)),
                "with no dropouts the aggregate is the plaintext sum " + Show(aggregate));
            Console.WriteLine("clients=8 dimension=4 aggregate=" + Show(aggregate));
        }

        // ---- dropouts are repaired by threshold reconstruction ---------------------------------
        {
            const int Clients = 7, Dimension = 3, Threshold = 4;
            var protocol = new SecureAggregation(RandomVectors(Clients, Dimension, 100000, 0xBEEFUL), Threshold, 0xF00DUL);
            var online = new List<int> { 0, 2, 3, 4, 5 };
            var dropped = new List<int> { 1, 6 };
            Check(online.Count >= Threshold, "enough clients stay online to reconstruct");
            long[] repaired = protocol.Aggregate(online, dropped);
            Check(Same(repaired, protocol.PlaintextSum(online)),
                "two dropped clients are repaired exactly " + Show(repaired));

            var oneOnline = new List<int> { 0, 1, 2, 3, 4, 5 };
            var oneDropped = new List<int> { 6 };
            Check(Same(protocol.Aggregate(oneOnline, oneDropped), protocol.PlaintextSum(oneOnline)),
                "a single dropout is repaired exactly as well");

            var tooFew = new List<int> { 0, 1, 2 };
            bool thrown = false;
            try { protocol.Aggregate(tooFew, new List<int> { 3 }); }
            catch (InvalidOperationException) { thrown = true; }
            Check(thrown, "fewer online clients than the threshold cannot repair a dropout and say so");

            var all = new List<int> { 0, 1, 2, 3, 4, 5, 6 };
            var gone = new List<int> { 0, 1, 2, 3, 4, 5, 6 };
            long[] empty = protocol.Aggregate(new List<int>(), all);
            Check(Same(empty, new long[Dimension]), "with everybody dropping the aggregate is the empty sum");
            Check(Same(protocol.Aggregate(all, new List<int>()), protocol.PlaintextSum(all)),
                "and with everybody online it is the full sum");
            Console.WriteLine("dropoutsRepaired=2 survivingSum=" + Show(repaired));
        }

        // ---- Shamir threshold reconstruction ---------------------------------------------------
        {
            const int Count = 6, Threshold = 3;
            ulong state = 0x51A1UL;
            bool everySubset = true, undetermined = true;
            int undeterminedTrials = 0;
            for (int trial = 0; trial < 20; trial++)
            {
                long secret = Rng.NextMod(ref state, Field.Prime);
                long[] shares = Shamir.Split(secret, Threshold, Count, ref state);
                for (int a = 1; a <= Count && everySubset; a++)
                    for (int b = a + 1; b <= Count && everySubset; b++)
                        for (int c = b + 1; c <= Count && everySubset; c++)
                        {
                            var points = new List<(int, long)> { (a, shares[a]), (b, shares[b]), (c, shares[c]) };
                            if (Shamir.Reconstruct(points) != secret) everySubset = false;
                        }
                // Two shares leave the secret undetermined: a guessed third share agrees only by chance.
                int correctGuesses = 0;
                var pair = new List<(int, long)> { (1, shares[1]), (2, shares[2]) };
                for (int guess = 0; guess < 200; guess++)
                {
                    long fake = Rng.NextMod(ref state, Field.Prime);
                    var points = new List<(int, long)>(pair) { (3, fake) };
                    if (Shamir.Reconstruct(points) == secret) correctGuesses++;
                }
                undeterminedTrials += correctGuesses;
                if (correctGuesses > 2) undetermined = false;
            }
            Check(everySubset, "every 3-subset of shares rebuilds the secret");
            Check(undetermined, "two shares plus a guess almost never rebuild it (" + undeterminedTrials + "/4000)");

            bool thrown = false;
            try { Shamir.Split(1, 0, 5, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a threshold of zero is rejected");
            thrown = false;
            try { Shamir.Split(1, 6, 5, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a threshold above the share count is rejected");
            thrown = false;
            try { Shamir.Reconstruct(new List<(int, long)>()); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "reconstructing from no shares is rejected");
            thrown = false;
            try { Shamir.Reconstruct(new List<(int, long)> { (1, 5), (1, 6) }); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "duplicate share indices are rejected");
            Console.WriteLine("shamirThreshold=" + Threshold + " wrongGuesses=" + undeterminedTrials);
        }

        // ---- the checksum catches tampering ----------------------------------------------------
        {
            var protocol = new SecureAggregation(RandomVectors(5, 2, 500, 0x7A17UL), 3, 0x1DEAUL);
            bool allIntact = true;
            for (int owner = 0; owner < protocol.Clients; owner++)
                for (int holder = 0; holder < protocol.Clients; holder++)
                    if (!protocol.ShareIsIntact(owner, holder)) allIntact = false;
            Check(allIntact, "every honest share passes its checksum");
            Check(protocol.MasksCancel(), "the fixture's masks cancel");

            // A malicious holder flips a share; the checksum comparison must catch it, and the
            // reconstruction must refuse to accept the corrupted mask.
            protocol.TamperShare(0, 1, 2, 0, 12345);
            Check(!protocol.ShareIsIntact(0, 2), "the checksum catches the tampered share");
            Check(protocol.ShareIsIntact(0, 3), "the untouched shares still verify");
            var survivors = new List<int> { 1, 2, 3, 4 };
            bool caught = false;
            try { protocol.Aggregate(survivors, new List<int> { 0 }); }
            catch (InvalidOperationException) { caught = true; }
            Check(caught, "aggregation refuses the mask reconstructed from the tampered share");
            Console.WriteLine("integrityChecks=" + (protocol.Clients * protocol.Clients) + " tamperDetected=" + caught);
        }

        // ---- degenerate configurations and the field itself ------------------------------------
        {
            var single = new SecureAggregation(new[] { new long[] { 42, 7 } }, 1, 0x5UL);
            Check(Same(single.Masked(0), new long[] { 42, 7 }), "a single client needs no mask");
            Check(Same(single.Aggregate(new List<int> { 0 }, new List<int>()), new long[] { 42, 7 }),
                "and its aggregate is its own vector");

            var oneDimension = new SecureAggregation(RandomVectors(6, 1, 100, 0x33UL), 4, 0x44UL);
            Check(oneDimension.MasksCancel() && oneDimension.Dimension == 1, "a one-dimensional setup still cancels");
            var allOnline = new List<int>();
            for (int i = 0; i < oneDimension.Clients; i++) allOnline.Add(i);
            Check(Same(oneDimension.Aggregate(allOnline, new List<int>()), oneDimension.PlaintextSum(allOnline)),
                "the one-dimensional aggregate is right");

            var fullThreshold = new SecureAggregation(RandomVectors(3, 2, 50, 0x55UL), 3, 0x66UL);
            allOnline = new List<int> { 0, 1, 2 };
            Check(Same(fullThreshold.Aggregate(allOnline, new List<int>()), fullThreshold.PlaintextSum(allOnline)),
                "a threshold equal to the client count works with everybody online");

            Check(Field.Add(Field.Prime - 1, 1) == 0, "field addition wraps correctly");
            Check(Field.Multiply(Field.Inverse(7), 7) == 1, "the field inverse is correct");
            Check(Field.Power(2, 10) == 1024, "field exponentiation matches ordinary arithmetic");
            Check(Field.Subtract(3, 5) == Field.Prime - 2, "field subtraction stays in range");

            bool thrown = false;
            try { new SecureAggregation(new[] { new long[] { Field.Prime } }, 1, 1UL); }
            catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a value outside the field is rejected");
            thrown = false;
            try { new SecureAggregation(new[] { new long[] { 1 }, new long[] { 1, 2 } }, 1, 1UL); }
            catch (ArgumentException) { thrown = true; }
            Check(thrown, "mismatched vector lengths are rejected");
            thrown = false;
            try { Field.Inverse(0); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "inverting zero is rejected");
            thrown = false;
            try { new SecureAggregation(Array.Empty<long[]>(), 1, 1UL); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a protocol without clients is rejected");
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            var first = new SecureAggregation(RandomVectors(4, 3, 999, 0x99UL), 3, 0xABCDUL);
            var second = new SecureAggregation(RandomVectors(4, 3, 999, 0x99UL), 3, 0xABCDUL);
            var all = new List<int> { 0, 1, 2, 3 };
            Check(Same(first.Aggregate(all, new List<int>()), second.Aggregate(all, new List<int>())),
                "the protocol is deterministic");
            Check(Same(first.Masked(2), second.Masked(2)), "and so are the masked vectors");
        }

        Console.WriteLine("PASS");
    }
}
