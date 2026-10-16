// Task 711 - Implement homomorphic encryption simple operations demo (Paillier).
// (Tier: Expert, CTCI Beyond — Security & Privacy)
//
// Paillier encryption is additively homomorphic modulo n = p*q. The public key is (n, g = n + 1) and
// the secret key is lambda = lcm(p-1, q-1) together with mu = L(g^lambda mod n^2)^-1 mod n, where
// L(x) = (x - 1) / n. A message m is encrypted with a random unit r as c = g^m * r^n mod n^2; the
// randomness is what makes the scheme probabilistic, because r^n mod n^2 hides m completely while
// the decryption L(c^lambda mod n^2) * mu mod n cancels r exactly. Two plaintext operations become
// cheap ciphertext operations:
//     addition of plaintexts   <->  multiplication of ciphertexts  (c1 * c2 mod n^2)
//     scaling by a constant k  <->  exponentiation of a ciphertext (c^k mod n^2)
//
// Verified below: an exhaustive round trip for *every* message of a tiny key (p = 11, q = 17, so all
// 187 residues), the range and shape of ciphertexts, the probabilistic behaviour (the same message
// gets many different ciphertexts, all of which decrypt the same), homomorphic addition and scaling
// against plaintext arithmetic on two keys, associativity and commutativity of the homomorphic sum,
// a homomorphic batch sum, the r = 1 special case that reduces the scheme to g^m, and every way the
// input can be malformed (message out of range, rg not coprime with n, bad keys).
using System;
using System.Collections.Generic;
using System.Numerics;

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

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class PaillierKey
{
    internal BigInteger N { get; }
    internal BigInteger NSquared { get; }
    internal BigInteger G { get; }
    internal BigInteger Lambda { get; }
    internal BigInteger Mu { get; }

    private PaillierKey(BigInteger n, BigInteger nsq, BigInteger g, BigInteger lambda, BigInteger mu)
    {
        N = n;
        NSquared = nsq;
        G = g;
        Lambda = lambda;
        Mu = mu;
    }

    /// <summary>Builds a key from two distinct primes and derives lambda and mu.</summary>
    internal static PaillierKey FromPrimes(BigInteger p, BigInteger q)
    {
        if (p <= 2 || q <= 2) throw new ArgumentException("the factors must be odd primes");
        if (p == q) throw new ArgumentException("the factors must differ");
        BigInteger n = p * q;
        BigInteger nsq = n * n;
        BigInteger g = n + 1;
        BigInteger lambda = (p - 1) * (q - 1) / BigInteger.GreatestCommonDivisor(p - 1, q - 1);
        BigInteger mu = ModInverse(L(BigInteger.ModPow(g, lambda, nsq), n), n);
        return new PaillierKey(n, nsq, g, lambda, mu);
    }

    /// <summary>L(x) = (x - 1) / n, which is an exact integer division for the values used here.</summary>
    private static BigInteger L(BigInteger x, BigInteger n) => (x - 1) / n;

    internal static BigInteger ModInverse(BigInteger value, BigInteger modulus)
    {
        BigInteger a = ((value % modulus) + modulus) % modulus;
        if (a == 0) throw new ArgumentException("zero has no inverse");
        BigInteger t = 0, newT = 1, r = modulus, newR = a;
        while (newR != 0)
        {
            BigInteger quotient = r / newR;
            (t, newT) = (newT, t - quotient * newT);
            (r, newR) = (newR, r - quotient * newR);
        }
        if (r != 1) throw new ArgumentException("the value is not invertible modulo the given modulus");
        return ((t % modulus) + modulus) % modulus;
    }

    internal BigInteger Encrypt(BigInteger message, BigInteger randomness)
    {
        if (message < 0 || message >= N)
            throw new ArgumentOutOfRangeException(nameof(message), "a Paillier message must be a residue in [0, n)");
        if (randomness <= 0 || randomness >= N || BigInteger.GreatestCommonDivisor(randomness, N) != 1)
            throw new ArgumentException("the randomness must be a unit modulo n", nameof(randomness));
        return BigInteger.ModPow(G, message, NSquared) * BigInteger.ModPow(randomness, N, NSquared) % NSquared;
    }

    /// <summary>Encrypts with a randomness drawn from the deterministic stream.</summary>
    internal BigInteger Encrypt(BigInteger message, ref ulong state)
    {
        BigInteger randomness;
        BigInteger span = N - 2;
        do
        {
            randomness = 2 + (BigInteger)(Rng.Next(ref state) % (ulong)span);   // in [2, n-1]
        }
        while (BigInteger.GreatestCommonDivisor(randomness, N) != 1);
        return Encrypt(message, randomness);
    }

    internal BigInteger Decrypt(BigInteger ciphertext)
    {
        if (ciphertext <= 0 || ciphertext >= NSquared)
            throw new ArgumentOutOfRangeException(nameof(ciphertext), "a ciphertext lives in [1, n^2)");
        return L(BigInteger.ModPow(ciphertext, Lambda, NSquared), N) * Mu % N;
    }

    /// <summary>Ciphertext product = plaintext sum.</summary>
    internal BigInteger HomomorphicAdd(BigInteger c1, BigInteger c2) => c1 * c2 % NSquared;

    /// <summary>Ciphertext power = plaintext scaled by k.</summary>
    internal BigInteger HomomorphicScale(BigInteger ciphertext, BigInteger k)
    {
        if (k < 0) throw new ArgumentOutOfRangeException(nameof(k), "negative scalars need the modular inverse of the ciphertext");
        return BigInteger.ModPow(ciphertext, k, NSquared);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool IsPrime(long value)
    {
        if (value < 2) return false;
        if (value % 2 == 0) return value == 2;
        for (long d = 3; d * d <= value; d += 2) if (value % d == 0) return false;
        return true;
    }

    public static void Main()
    {
        // ---- exhaustive round trip on the smallest usable key --------------------------------
        {
            var key = PaillierKey.FromPrimes(11, 17);
            Check(key.N == 187 && key.NSquared == 34969, "the tiny key has n = 187 and n^2 = 34969");
            Check(IsPrime(11) && IsPrime(17), "the fixtures are primes");
            // The key is only usable if L(g^lambda mod n^2) really is invertible modulo n.
            BigInteger sanity = (BigInteger.ModPow(key.G, key.Lambda, key.NSquared) - 1) / key.N;
            Check(sanity * key.Mu % key.N == 1, "mu is the inverse of L(g^lambda mod n^2)");

            ulong state = 0xBADC0DEUL;
            bool everyResidue = true;
            for (int m = 0; m < (int)key.N; m++)
            {
                BigInteger ciphertext = key.Encrypt(m, ref state);
                if (key.Decrypt(ciphertext) != m) everyResidue = false;
                if (ciphertext < 1 || ciphertext >= key.NSquared) everyResidue = false;
            }
            Check(everyResidue, "every residue of the tiny key survives the round trip");
            Check(key.Decrypt(key.Encrypt(0, ref state)) == 0, "zero encrypts and decrypts");
            Check(key.Decrypt(key.Encrypt(key.N - 1, ref state)) == key.N - 1, "the largest residue works too");

            // probabilistic: identical messages, different ciphertexts
            var seen = new HashSet<BigInteger>();
            bool allDecrypt = true;
            for (int i = 0; i < 120; i++)
            {
                BigInteger ciphertext = key.Encrypt(42, ref state);
                seen.Add(ciphertext);
                if (key.Decrypt(ciphertext) != 42) allDecrypt = false;
            }
            Check(allDecrypt, "all 120 ciphertexts of the same message decrypt to it");
            Check(seen.Count >= 60, "the scheme is probabilistic: " + seen.Count
                + " distinct ciphertexts out of a space of phi(187) = 160 units");

            BigInteger withOne = key.Encrypt(42, 1);
            Check(withOne == BigInteger.ModPow(key.G, 42, key.NSquared),
                "r = 1 reduces encryption to g^m, as the construction says");
            Check(key.Decrypt(withOne) == 42, "that degenerate ciphertext is still decryptable");
            Console.WriteLine("tinyKey residues=187 distinctCiphertexts=" + seen.Count);
        }

        // ---- homomorphic addition and scaling -------------------------------------------------
        {
            var key = PaillierKey.FromPrimes(1019, 1031);
            Check(key.N == 1019 * 1031, "the second key multiplies to " + key.N);
            ulong state = 0xB16B00B5UL;
            const int Trials = 40;
            bool addOk = true, scaleOk = true;
            for (int t = 0; t < Trials; t++)
            {
                BigInteger a = Rng.NextInt(ref state, 500);
                BigInteger b = Rng.NextInt(ref state, 500);
                BigInteger ca = key.Encrypt(a, ref state);
                BigInteger cb = key.Encrypt(b, ref state);
                if (key.Decrypt(key.HomomorphicAdd(ca, cb)) != (a + b) % key.N) addOk = false;
                if (key.Decrypt(key.HomomorphicScale(ca, b)) != a * b % key.N) scaleOk = false;
            }
            Check(addOk, "ciphertext products decrypt to plaintext sums");
            Check(scaleOk, "ciphertext powers decrypt to plaintext multiples");

            BigInteger x = Rng.NextInt(ref state, 500), y = Rng.NextInt(ref state, 500), z = Rng.NextInt(ref state, 500);
            BigInteger cx = key.Encrypt(x, ref state), cy = key.Encrypt(y, ref state), cz = key.Encrypt(z, ref state);
            Check(key.Decrypt(key.HomomorphicAdd(cx, cy)) == key.Decrypt(key.HomomorphicAdd(cy, cx)),
                "the homomorphic sum is commutative");
            BigInteger left = key.HomomorphicAdd(key.HomomorphicAdd(cx, cy), cz);
            BigInteger right = key.HomomorphicAdd(cx, key.HomomorphicAdd(cy, cz));
            Check(key.Decrypt(left) == key.Decrypt(right) && key.Decrypt(left) == (x + y + z) % key.N,
                "the homomorphic sum is associative and correct");
            Check(key.Decrypt(key.HomomorphicScale(cx, 0)) == 0, "scaling by zero yields the encryption of zero");
            Check(key.Decrypt(key.HomomorphicScale(cx, 1)) == x, "scaling by one is the identity");

            // a batch: 40 encrypted values summed without ever decrypting them individually
            BigInteger accumulator = key.Encrypt(0, ref state);
            BigInteger plaintextSum = 0;
            for (int i = 0; i < 40; i++)
            {
                BigInteger value = Rng.NextInt(ref state, 100);
                plaintextSum += value;
                accumulator = key.HomomorphicAdd(accumulator, key.Encrypt(value, ref state));
            }
            Check(key.Decrypt(accumulator) == plaintextSum % key.N,
                "the encrypted batch sum equals the plaintext sum (" + plaintextSum % key.N + ")");
            Console.WriteLine("secondKey n=" + key.N + " batchSum=" + plaintextSum % key.N);
        }

        // ---- rejected input ---------------------------------------------------------------------
        {
            var key = PaillierKey.FromPrimes(11, 17);
            ulong state = 0x5A5AUL;
            bool thrown = false;
            try { key.Encrypt(-1, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a negative message is rejected");
            thrown = false;
            try { key.Encrypt(key.N, ref state); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a message outside [0, n) is rejected");
            thrown = false;
            try { key.Encrypt(5, key.N); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "randomness sharing a factor with n is rejected");
            thrown = false;
            try { key.Encrypt(5, 0); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "zero randomness is rejected");
            thrown = false;
            try { key.Decrypt(0); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "a ciphertext outside [1, n^2) is rejected");
            thrown = false;
            try { key.Decrypt(key.NSquared); } catch (ArgumentOutOfRangeException) { thrown = true; }
            Check(thrown, "an oversized ciphertext is rejected");
            thrown = false;
            try { PaillierKey.FromPrimes(13, 13); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "repeated factors are rejected");
            thrown = false;
            try { PaillierKey.ModInverse(4, 8); } catch (ArgumentException) { thrown = true; }
            Check(thrown, "a non-invertible value is reported instead of returning nonsense");
            Check(PaillierKey.ModInverse(3, 11) == 4, "the modular inverse of 3 modulo 11 is 4");
        }

        // ---- determinism ------------------------------------------------------------------------
        {
            var key = PaillierKey.FromPrimes(1019, 1031);
            ulong a = 0x777UL, b = 0x777UL;
            BigInteger first = key.Encrypt(12345, ref a);
            BigInteger second = key.Encrypt(12345, ref b);
            Check(first == second && a == b, "the same seed produces the same ciphertext and stream position");

            // With a six-digit modulus the ciphertext space is large enough that repetition is rare.
            var distinct = new HashSet<BigInteger>();
            for (int i = 0; i < 200; i++) distinct.Add(key.Encrypt(7, ref a));
            Check(distinct.Count >= 195, "a big key gives a fresh ciphertext almost every time (" + distinct.Count + "/200)");

            var other = PaillierKey.FromPrimes(1033, 1051);
            Check(other.N != key.N && other.Lambda != key.Lambda, "different primes give different keys");
            Check(other.Decrypt(other.Encrypt(999, ref b)) == 999, "the third key round trips as well");
            Console.WriteLine("thirdKey n=" + other.N + " distinctBigKey=" + distinct.Count);
        }

        Console.WriteLine("PASS");
    }
}
