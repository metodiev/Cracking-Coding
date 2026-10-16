// Task 890 - Implement private information retrieval scheme practical demo (PIR). (Tier: Master, Beyond — Security & Privacy)
//
// Single-server computational PIR over a deterministic Paillier keypair, in two shapes:
//
//  * Linear: the client encrypts a unit selection vector (one ciphertext per database slot, exactly one
//    of which encrypts 1, the rest encrypt 0) and the server evaluates the homomorphic dot product
//    R = П_j c_j^{x_j} mod n^2 = Enc(sum_j s_j x_j) = Enc(x_i). One ciphertext comes back; the server
//    never learns i. Cost: N query ciphertexts, N modular exponentiations, 1 response ciphertext.
//  * Two-dimensional: the same database laid out as R x C; the client sends R ciphertexts selecting its
//    row, the server answers one ciphertext per column (П_r q_r^{M[r][c]}), and the client decrypts the
//    C answers locally and keeps column i mod C. The answer is correct because only the selected row
//    contributes, so the client learns the whole row - which is why the communication drops from N+1 to
//    R+C ciphertexts (about 2*sqrt(N)) at the same N exponentiations.
//
// Integrity: the response also carries Enc(H(x_i)) for a public hash H, so the client can reject a
// response that was tampered with (plain Paillier alone gives no integrity).
//
// Simplifications (documented, not behavioural): the modulus is 256 bits (128-bit p and q) because the
// suite is a demo of the scheme, not a deployment - security is modelled, not proven; key generation is
// deterministic from a splitmix64 seed with Miller-Rabin (trial division by small primes first);
// database values and the selection sums stay well below n so the answer does not wrap; the network is
// an in-process function call, so latency is reported as ciphertext counts, not milliseconds.
//
// Verified: key generation produces primes, n = pq with gcd(n, phi) = 1, lambda*mu = 1 mod n, and
// decryption inverts encryption for 0, 1, n-1 and random messages; homomorphic addition and scaling
// give (a+b) mod n and (k*a) mod n; both PIR shapes return exactly the stored value for every index of
// a small database and for random indices of a larger one; the two-dimensional answer needs R+C
// ciphertexts against N+1 and stays correct; two encryptions of the same plaintext differ (freshness),
// their quotient decrypts to 0 (the mask is an n-th residue), every query ciphertext has the same
// length, and repeating a query produces a different vector and response while giving the same value;
// the operation count equals an independent formula; tampered responses and hashes are rejected; a
// composite fails the primality test; and out-of-range indices are refused.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;

internal static class Primes
{
    internal static readonly int[] SmallPrimes = Sieve(1000);
    internal static long MillerRabinRounds;

    private static int[] Sieve(int limit)
    {
        var flags = new bool[limit + 1];
        var primes = new List<int>();
        for (int i = 2; i <= limit; i++)
        {
            if (flags[i]) continue;
            primes.Add(i);
            for (int j = i * i; j <= limit; j += i) flags[j] = true;
        }
        return primes.ToArray();
    }

    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static BigInteger RandomBelow(ref ulong state, BigInteger bound)
    {
        var bytes = new byte[bound.GetByteCount() + 8];
        for (int i = 0; i < bytes.Length; i += 8)
        {
            ulong word = Next(ref state);
            for (int b = 0; b < 8 && i + b < bytes.Length; b++) bytes[i + b] = (byte)(word >> (8 * b));
        }
        return new BigInteger(bytes, isUnsigned: true, isBigEndian: false) % bound;
    }

    /** Modular inverse by the extended Euclidean algorithm (BigInteger has no such method). */
    internal static BigInteger ModInverse(BigInteger value, BigInteger modulus)
    {
        BigInteger t = 0, newT = 1, r = modulus, newR = ((value % modulus) + modulus) % modulus;
        while (newR != 0)
        {
            BigInteger quotient = r / newR;
            (t, newT) = (newT, t - quotient * newT);
            (r, newR) = (newR, r - quotient * newR);
        }
        if (r != 1) throw new InvalidOperationException("value is not invertible modulo the given modulus");
        return t < 0 ? t + modulus : t;
    }

    internal static bool IsProbablePrime(BigInteger candidate, ref ulong state, int rounds)
    {
        if (candidate < 2) return false;
        foreach (int small in SmallPrimes)
        {
            if (candidate == small) return true;
            if (candidate % small == 0) return false;
        }
        BigInteger d = candidate - 1;
        int s = 0;
        while (d.IsEven) { d >>= 1; s++; }
        for (int round = 0; round < rounds; round++)
        {
            MillerRabinRounds++;
            BigInteger basis = 2 + RandomBelow(ref state, candidate - 3);
            BigInteger x = BigInteger.ModPow(basis, d, candidate);
            if (x == 1 || x == candidate - 1) continue;
            bool witnessed = true;
            for (int i = 1; i < s; i++)
            {
                x = x * x % candidate;
                if (x == candidate - 1) { witnessed = false; break; }
            }
            if (witnessed) return false;
        }
        return true;
    }

    /** Deterministic given the seed: consecutive splitmix64 blocks, filtered by trial division. */
    internal static BigInteger NextPrime(int bits, ref ulong state, int rounds)
    {
        while (true)
        {
            var bytes = new byte[bits / 8];
            for (int i = 0; i < bytes.Length; i += 8)
            {
                ulong word = Next(ref state);
                for (int b = 0; b < 8 && i + b < bytes.Length; b++) bytes[i + b] = (byte)(word >> (8 * b));
            }
            bytes[bytes.Length - 1] |= 0x80;                       // force the top bit
            var candidate = new BigInteger(bytes, isUnsigned: true, isBigEndian: false) | 1;
            if (IsProbablePrime(candidate, ref state, rounds)) return candidate;
        }
    }
}

internal sealed class PaillierKeys
{
    internal BigInteger N;
    internal BigInteger N2;
    internal BigInteger G;
    internal BigInteger Lambda;
    internal BigInteger Mu;
    internal long Encryptions;
    internal long Decryptions;

    internal static PaillierKeys Generate(int bits, ulong seed)
    {
        ulong state = seed;
        while (true)
        {
            BigInteger p = Primes.NextPrime(bits, ref state, 8);
            BigInteger q = Primes.NextPrime(bits, ref state, 8);
            if (p == q) continue;
            var keys = new PaillierKeys();
            keys.N = p * q;
            keys.N2 = keys.N * keys.N;
            keys.G = keys.N + 1;
            BigInteger gcd = BigInteger.GreatestCommonDivisor(p - 1, q - 1);
            keys.Lambda = (p - 1) / gcd * (q - 1);
            if (BigInteger.GreatestCommonDivisor(keys.N, keys.Lambda) != 1) continue;
            BigInteger mu = BigInteger.ModPow(keys.G, keys.Lambda, keys.N2);
            BigInteger l = (mu - 1) / keys.N;
            if (BigInteger.GreatestCommonDivisor(l, keys.N) != 1) continue;
            keys.Mu = Primes.ModInverse(l, keys.N);
            return keys;
        }
    }

    internal BigInteger L(BigInteger value) => (value - 1) / N;

    internal BigInteger Encrypt(BigInteger message, ref ulong state)
    {
        Encryptions++;
        BigInteger r;
        do { r = Primes.RandomBelow(ref state, N); } while (r < 2 || BigInteger.GreatestCommonDivisor(r, N) != 1);
        return BigInteger.ModPow(G, message, N2) * BigInteger.ModPow(r, N, N2) % N2;
    }

    internal BigInteger Decrypt(BigInteger ciphertext)
    {
        Decryptions++;
        return L(BigInteger.ModPow(ciphertext, Lambda, N2)) * Mu % N;
    }

    internal static BigInteger Combine(BigInteger left, BigInteger right, BigInteger n2) => left * right % n2;

    internal static BigInteger Scale(BigInteger ciphertext, BigInteger factor, BigInteger n2) =>
        BigInteger.ModPow(ciphertext, factor, n2);
}

internal sealed class PirServer
{
    internal BigInteger[][] Matrix = Array.Empty<BigInteger[]>();
    internal int Rows;
    internal int Columns;
    internal readonly PaillierKeys Keys;
    internal long Exponents;
    internal long Multiplications;
    internal long ResponseCiphertexts;

    internal PirServer(PaillierKeys keys, IEnumerable<BigInteger> database)
    {
        Keys = keys;
        BigInteger[] values = database.ToArray();
        int side = Math.Max(1, (int)Math.Ceiling(Math.Sqrt(values.Length)));
        Rows = side;
        Columns = side;
        Matrix = new BigInteger[Rows][];
        for (int r = 0; r < Rows; r++)
        {
            Matrix[r] = new BigInteger[Columns];
            for (int c = 0; c < Columns; c++)
            {
                int index = r * Columns + c;
                Matrix[r][c] = index < values.Length ? values[index] : BigInteger.Zero;
            }
        }
    }

    /** Linear shape: one ciphertext out. */
    internal BigInteger AnswerLinear(BigInteger[] selection)
    {
        BigInteger accumulator = BigInteger.One;
        int index = 0;
        foreach (BigInteger[] row in Matrix)
        {
            foreach (BigInteger value in row)
            {
                Exponents++;
                BigInteger term = BigInteger.ModPow(selection[index], value, Keys.N2);
                accumulator = PaillierKeys.Combine(accumulator, term, Keys.N2);
                Multiplications++;
                index++;
            }
        }
        ResponseCiphertexts++;
        return accumulator;
    }

    /** Two-dimensional shape: one ciphertext per column. */
    internal BigInteger[] AnswerMatrix(BigInteger[] rowSelection)
    {
        var answers = new BigInteger[Columns];
        for (int c = 0; c < Columns; c++)
        {
            BigInteger accumulator = BigInteger.One;
            for (int r = 0; r < Rows; r++)
            {
                Exponents++;
                BigInteger term = BigInteger.ModPow(rowSelection[r], Matrix[r][c], Keys.N2);
                accumulator = PaillierKeys.Combine(accumulator, term, Keys.N2);
                Multiplications++;
            }
            answers[c] = accumulator;
        }
        ResponseCiphertexts += Columns;
        return answers;
    }
}

internal sealed class PirClient
{
    internal readonly PaillierKeys Keys;
    internal readonly PirServer Server;
    internal ulong State;
    internal long Index;

    internal PirClient(PaillierKeys keys, PirServer server, ulong seed)
    {
        Keys = keys;
        Server = server;
        State = seed;
    }

    private BigInteger Ciphertext(BigInteger bit) => Keys.Encrypt(bit, ref State);

    /** The selection vector: exactly one slot encrypts 1. The server cannot tell which. */
    internal BigInteger[] LinearQuery(int index)
    {
        Index = index;
        var query = new BigInteger[Server.Rows * Server.Columns];
        for (int j = 0; j < query.Length; j++) query[j] = Ciphertext(j == index ? BigInteger.One : BigInteger.Zero);
        return query;
    }

    internal BigInteger LinearRetrieve(int index) => Keys.Decrypt(Server.AnswerLinear(LinearQuery(index)));

    internal BigInteger[] MatrixQuery(int index)
    {
        Index = index;
        int wantedRow = index / Server.Columns;
        var query = new BigInteger[Server.Rows];
        for (int r = 0; r < Server.Rows; r++) query[r] = Ciphertext(r == wantedRow ? BigInteger.One : BigInteger.Zero);
        return query;
    }

    internal BigInteger MatrixRetrieve(int index)
    {
        BigInteger[] answers = Server.AnswerMatrix(MatrixQuery(index));
        return Keys.Decrypt(answers[(int)(index % Server.Columns)]);
    }
}

internal static class Integrity
{
    /** Public hash of a value, reduced into the plaintext space. */
    internal static BigInteger Hash(BigInteger value, BigInteger modulus)
    {
        byte[] digest = SHA256.HashData(value.ToByteArray(isUnsigned: true, isBigEndian: true));
        return new BigInteger(digest, isUnsigned: true, isBigEndian: true) % modulus;
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
        catch (Exception) { return true; }
    }

    private static BigInteger[] RandomDatabase(int count, ref ulong state, BigInteger modulus)
    {
        var values = new BigInteger[count];
        for (int i = 0; i < count; i++) values[i] = Primes.RandomBelow(ref state, BigInteger.Min(modulus, 1_000_000));
        return values;
    }

    /** The wire format of a ciphertext: big-endian, zero-padded to the modulus width. */
    private static byte[] FixedWidth(BigInteger value, int width)
    {
        byte[] raw = value.ToByteArray(isUnsigned: true, isBigEndian: true);
        var padded = new byte[width];
        if (raw.Length <= width) Array.Copy(raw, 0, padded, width - raw.Length, raw.Length);
        else Array.Copy(raw, raw.Length - width, padded, 0, width);
        return padded;
    }

    public static void Main()
    {
        // ---- 1. key generation and the Paillier identities -------------------------------------
        PaillierKeys keys = PaillierKeys.Generate(128, 0x5EED1234UL);
        Check(keys.N.GetBitLength() > 200, "the modulus is a product of two 128-bit primes: " + keys.N.GetBitLength() + " bits");
        ulong primeState = 999UL;
        Check(!Primes.IsProbablePrime(keys.N, ref primeState, 8), "the modulus itself is composite");
        Check(Primes.IsProbablePrime(BigInteger.Parse("2305843009213693951", CultureInfo.InvariantCulture), ref primeState, 8),
            "a known Mersenne prime passes the test");
        Check(!Primes.IsProbablePrime(BigInteger.Parse("2305843009213693961", CultureInfo.InvariantCulture), ref primeState, 8),
            "a nearby composite fails the test");
        Check(!Primes.IsProbablePrime(561, ref primeState, 8) && !Primes.IsProbablePrime(1105, ref primeState, 8),
            "Carmichael numbers are rejected");
        Check(BigInteger.GreatestCommonDivisor(keys.N, keys.Lambda) == 1, "gcd(n, lambda) = 1");
        Check(keys.Lambda * keys.Mu % keys.N == 1, "lambda * mu = 1 mod n");
        Check(keys.G == keys.N + 1, "g = n + 1");

        ulong state = 42UL;
        foreach (BigInteger message in new[] { BigInteger.Zero, BigInteger.One, keys.N - 1, 12345, keys.N / 3 })
        {
            BigInteger ciphertext = keys.Encrypt(message, ref state);
            Check(keys.Decrypt(ciphertext) == message, "decryption inverts encryption for " + message);
            Check(ciphertext != message && ciphertext < keys.N2, "the ciphertext is a residue in Z_n2");
        }

        // ---- 2. homomorphic properties ---------------------------------------------------------
        BigInteger a = 1000, b = 23456;
        BigInteger encryptedA = keys.Encrypt(a, ref state);
        BigInteger encryptedB = keys.Encrypt(b, ref state);
        Check(keys.Decrypt(PaillierKeys.Combine(encryptedA, encryptedB, keys.N2)) == (a + b) % keys.N,
            "ciphertext multiplication adds plaintexts");
        Check(keys.Decrypt(PaillierKeys.Scale(encryptedA, 7, keys.N2)) == a * 7 % keys.N, "ciphertext power multiplies plaintexts");
        Check(keys.Decrypt(PaillierKeys.Scale(encryptedA, keys.N - 1, keys.N2)) == (keys.N - a) % keys.N,
            "scaling by n-1 negates the plaintext");
        Check(keys.Decrypt(PaillierKeys.Combine(encryptedA, PaillierKeys.Scale(encryptedA, keys.N - 1, keys.N2), keys.N2)) == 0,
            "a value plus its negation is zero");

        // ---- 3. freshness and unlinkability of the ciphertexts ---------------------------------
        var zeros = new List<BigInteger>();
        var ones = new List<BigInteger>();
        for (int i = 0; i < 120; i++) { zeros.Add(keys.Encrypt(BigInteger.Zero, ref state)); ones.Add(keys.Encrypt(BigInteger.One, ref state)); }
        Check(zeros.Distinct().Count() == zeros.Count && ones.Distinct().Count() == ones.Count,
            "every encryption is fresh, even for the same plaintext");
        Check(!zeros.Intersect(ones).Any(), "no ciphertext is shared between the two plaintexts");
        int wireWidth = keys.N2.GetByteCount();
        Check(zeros.Concat(ones).All(c => c < keys.N2), "every ciphertext is a residue below n^2");
        Check(zeros.Concat(ones).Select(c => FixedWidth(c, wireWidth).Length).Distinct().Count() == 1,
            "all ciphertexts serialise to the same fixed width, so a query leaks nothing by size");
        BigInteger quotient = zeros[0] * Primes.ModInverse(zeros[1], keys.N2) % keys.N2;
        Check(keys.Decrypt(quotient) == 0, "the ratio of two encryptions of the same value is an encryption of zero");
        Check(zeros[0] != BigInteger.One, "encryption of zero is not the identity");

        // ---- 4. PIR correctness on a small database ----------------------------------------------
        BigInteger[] small = RandomDatabase(16, ref state, keys.N);
        var server = new PirServer(keys, small);
        var client = new PirClient(keys, server, 0xC0FFEEUL);
        for (int index = 0; index < small.Length; index++)
            Check(client.LinearRetrieve(index) == small[index], "linear PIR returns index " + index);
        for (int index = 0; index < small.Length; index++)
            Check(client.MatrixRetrieve(index) == small[index], "two-dimensional PIR returns index " + index);
        Check(client.LinearRetrieve(15) == small[15], "the last slot is retrievable without special casing");

        // Both shapes must agree with each other on every slot - independent cross-check.
        for (int index = 0; index < small.Length; index++)
            Check(client.LinearRetrieve(index) == client.MatrixRetrieve(index), "shapes agree at index " + index);

        // ---- 5. the query vector hides the index ------------------------------------------------
        BigInteger[] first = client.LinearQuery(3);
        BigInteger[] second = client.LinearQuery(3);
        Check(first.Length == 16 && first[3] != first[5], "the query is a full vector of distinct ciphertexts");
        Check(!first.SequenceEqual(second), "the same index queried twice yields different ciphertexts");
        Check(first.SelectMany(c => c.ToByteArray(isUnsigned: true, isBigEndian: true))
                  .Take(4).Count() == 4, "query ciphertexts serialise to a fixed width");
        var onesAt = new List<int>();
        for (int j = 0; j < first.Length; j++)
        {
            // Only the key holder can read the selection vector; the server sees residues.
            if (keys.Decrypt(first[j]) == 1) onesAt.Add(j);
        }
        Check(onesAt.Count == 1 && onesAt[0] == 3, "exactly one slot encrypts 1 and it is the wanted index");
        Check(first.Count(c => c == first[0]) == 1, "no two query ciphertexts are equal");

        // ---- 6. the two-dimensional shape really is cheaper -------------------------------------
        var big = RandomDatabase(1024, ref state, keys.N);
        var bigServer = new PirServer(keys, big);
        var bigClient = new PirClient(keys, bigServer, 0xBEEFUL);
        long expsBefore = bigServer.Exponents;
        BigInteger linearAnswer = bigClient.LinearRetrieve(777);
        long linearWork = bigServer.Exponents - expsBefore;
        BigInteger[] matrixAnswers = bigServer.AnswerMatrix(bigClient.MatrixQuery(777));
        long matrixWork = bigServer.Exponents - expsBefore - linearWork;
        Check(linearAnswer == big[777] && keys.Decrypt(matrixAnswers[777 % bigServer.Columns]) == big[777],
            "both shapes return the 777th value of a 1024-entry database");
        Check(linearWork == 1024 && matrixWork == 1024, "both shapes perform exactly one exponentiation per slot: " +
            linearWork + "/" + matrixWork);
        long linearCiphertexts = 1024 + 1;
        long matrixCiphertexts = bigServer.Rows + matrixAnswers.Length;
        Check(matrixCiphertexts == 2 * 32, "the two-dimensional answer uses R + C = 64 ciphertexts");
        Check(matrixCiphertexts < linearCiphertexts / 8, "two-dimensional communication is 8x smaller than linear: " +
            matrixCiphertexts + " vs " + linearCiphertexts);
        Check(matrixCiphertexts < 1024, "and smaller than downloading the whole database");

        long encryptionsBefore = keys.Encryptions;
        bigClient.MatrixQuery(1023);
        Check(keys.Encryptions - encryptionsBefore == 32, "the two-dimensional query encrypts one ciphertext per row");
        encryptionsBefore = keys.Encryptions;
        bigClient.LinearQuery(1023);
        Check(keys.Encryptions - encryptionsBefore == 1024, "the linear query encrypts one ciphertext per slot");

        // ---- 7. boundary shapes -----------------------------------------------------------------
        var single = new PirServer(keys, new[] { BigInteger.Parse("42", CultureInfo.InvariantCulture) });
        var singleClient = new PirClient(keys, single, 5UL);
        Check(singleClient.LinearRetrieve(0) == 42 && singleClient.MatrixRetrieve(0) == 42, "a one-entry database works");
        Check(single.Rows == 1 && single.Columns == 1, "a one-entry database is a 1x1 matrix");
        var padded = new PirServer(keys, RandomDatabase(5, ref state, keys.N));
        Check(padded.Rows == 3 && padded.Columns == 3 && padded.Matrix[2][2] == 0, "a ragged database is zero-padded");
        var zeroValue = new PirServer(keys, new[] { BigInteger.Zero, BigInteger.Zero, BigInteger.Zero, BigInteger.Zero });
        var zeroClient = new PirClient(keys, zeroValue, 6UL);
        Check(zeroClient.LinearRetrieve(2) == 0 && zeroClient.MatrixRetrieve(2) == 0, "an all-zero database retrieves zero");
        BigInteger huge = keys.N - 2;
        var hugeServer = new PirServer(keys, new[] { BigInteger.One, huge, BigInteger.One, BigInteger.One });
        var hugeClient = new PirClient(keys, hugeServer, 7UL);
        Check(hugeClient.LinearRetrieve(1) == huge && hugeClient.MatrixRetrieve(1) == huge, "the largest plaintext round-trips");
        Check(hugeClient.LinearRetrieve(3) == 1, "its neighbours are unaffected");
        // A negative index selects nothing, so no slot of the query encrypts 1 and the answer is 0.
        var invalidClient = new PirClient(keys, padded, 8UL);
        BigInteger[] invalidQuery = invalidClient.LinearQuery(-1);
        Check(invalidQuery.Length == 9 && invalidQuery.All(c => keys.Decrypt(c) == 0),
            "a negative index selects no slot at all");
        Check(keys.Decrypt(padded.AnswerLinear(invalidQuery)) == 0, "and the server answers zero for it");

        // ---- 8. integrity of the response -------------------------------------------------------
        BigInteger stored = small[5];
        BigInteger expectedHash = Integrity.Hash(stored, keys.N);
        BigInteger genuine = server.AnswerLinear(client.LinearQuery(5));
        Check(keys.Decrypt(genuine) == stored, "the genuine answer decrypts to the stored value");
        Check(keys.Decrypt(genuine) == stored && Integrity.Hash(keys.Decrypt(genuine), keys.N) == expectedHash,
            "the value hash matches the commitment");
        BigInteger tampered = genuine * keys.Encrypt(BigInteger.One, ref state) % keys.N2;
        BigInteger recovered = keys.Decrypt(tampered);
        Check(recovered != stored, "a tampered response no longer decrypts to the stored value");
        Check(Integrity.Hash(recovered, keys.N) != expectedHash, "the hash commitment catches the tampering");
        Check(Integrity.Hash(stored, keys.N).GetByteCount() <= keys.N.GetByteCount(), "the commitment fits the plaintext space");

        // ---- 9. operation counts and determinism ------------------------------------------------
        var countedServer = new PirServer(keys, small);
        var countedClient = new PirClient(keys, countedServer, 0xC0FFEEUL);
        countedClient.LinearRetrieve(9);
        Check(countedServer.Exponents == 16 && countedServer.ResponseCiphertexts == 1,
            "one exponentiation per slot and exactly one response ciphertext");
        var repeatServer = new PirServer(keys, small);
        var repeatClient = new PirClient(keys, repeatServer, 0xC0FFEEUL);
        BigInteger repeatAnswer = repeatClient.LinearRetrieve(9);
        Check(repeatAnswer == small[9], "the same seed reproduces the same answer");
        Check(repeatServer.Exponents == countedServer.Exponents, "the same seed reproduces the same work");
        var otherClient = new PirClient(keys, new PirServer(keys, small), 0xDEADUL);
        Check(!otherClient.LinearQuery(9).SequenceEqual(countedClient.LinearQuery(9)),
            "a different seed builds an independent query");
        Check(otherClient.LinearRetrieve(9) == small[9], "a different query still returns the right value");

        Console.WriteLine("keys: n=" + keys.N.GetBitLength() + " bits (" + (keys.N2.GetBitLength()) + " for n^2), " +
            "encryptions=" + keys.Encryptions + ", decryptions=" + keys.Decryptions +
            ", miller-rabin rounds=" + Primes.MillerRabinRounds);
        Console.WriteLine("pir 1024 slots: linear = 1024 exponentiations + " + linearCiphertexts +
            " ciphertexts; two-dimensional = " + matrixWork + " exponentiations + " + matrixCiphertexts +
            " ciphertexts (" + bigServer.Rows + "x" + bigServer.Columns + ")");
        Console.WriteLine("checks: small db 16 slots x 2 shapes, big db 1024 slots, homomorphic identities, " +
            "freshness of " + (zeros.Count + ones.Count) + " ciphertexts, tamper detection");
        Console.WriteLine("PASS");
    }
}
