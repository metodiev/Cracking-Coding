// Task 385 - Implement RSA key generation and basic encrypt/decrypt (demo). (Tier: Intermediate, CTCI Beyond — Security & Privacy)
using System;

public static class Solution
{
    private static long[] ExtendedGcd(long a, long b)
    {
        if (b == 0) return new long[] { a, 1, 0 };
        var prev = ExtendedGcd(b, a % b);
        return new long[] { prev[0], prev[2], prev[1] - (a / b) * prev[2] };
    }

    private static long ModInverse(long a, long modulus)
    {
        var eg = ExtendedGcd(((a % modulus) + modulus) % modulus, modulus);
        if (eg[0] != 1) throw new InvalidOperationException("modular inverse does not exist");
        return ((eg[1] % modulus) + modulus) % modulus;
    }

    private static long Gcd(long a, long b)
    {
        while (b != 0)
        {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /// <summary>Trial division, used only to reject obviously bad demo primes.</summary>
    private static bool IsSmallPrime(long n)
    {
        if (n < 2) return false;
        for (long i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    private static long ModPower(long baseValue, long exponent, long modulus)
    {
        long result = 1 % modulus;
        long b = ((baseValue % modulus) + modulus) % modulus;
        long e = exponent;
        while (e > 0)
        {
            if ((e & 1) == 1) result = (long)((UInt128)result * (UInt128)b % (UInt128)modulus);
            b = (long)((UInt128)b * (UInt128)b % (UInt128)modulus);
            e >>= 1;
        }
        return result;
    }

    /// <summary>Returns {n, e, d, phi}; throws on bad demo parameters.</summary>
    private static long[] Keygen(long p, long q, long e)
    {
        if (!IsSmallPrime(p) || !IsSmallPrime(q) || p == q)
            throw new InvalidOperationException("p and q must be distinct primes");
        long n = p * q;
        long phi = (p - 1) * (q - 1);
        if (Gcd(e, phi) != 1)
            throw new InvalidOperationException("public exponent must be coprime with phi(n)");
        return new long[] { n, e, ModInverse(e, phi), phi };
    }

    private static long Encrypt(long message, long e, long n)
    {
        if (message < 0 || message >= n)
            throw new InvalidOperationException("message must satisfy 0 <= m < n");
        return ModPower(message, e, n);
    }

    private static long Decrypt(long ciphertext, long d, long n)
    {
        if (ciphertext < 0 || ciphertext >= n)
            throw new InvalidOperationException("ciphertext must satisfy 0 <= c < n");
        return ModPower(ciphertext, d, n);
    }

    private static string Join(long[] values) => string.Join(",", values);

    public static void Main()
    {
        // classic small keypair p=61, q=53 (n = 3233, phi = 3120, e = 17, d = 2753)
        var keys = Keygen(61, 53, 17);
        long n = keys[0], e = keys[1], d = keys[2], phi = keys[3];
        Check(Join(keys) == "3233,17,2753,3120", "keygen 61,53,17");
        Check(n == 61 * 53 && phi == 60 * 52, "n and phi");
        Check(e * d % phi == 1, "e*d = 1 (mod phi)");
        Check(d != e, "d differs from e");

        // round trip for several messages, including the boundaries 0 and n-1
        foreach (long message in new long[] { 0, 1, 2, 42, 65, 1000, 3232 })
            Check(Decrypt(Encrypt(message, e, n), d, n) == message, "round trip " + message);
        Check(Encrypt(0, e, n) == 0 && Encrypt(1, e, n) == 1, "fixed points 0 and 1");
        Check(Encrypt(n - 1, e, n) == n - 1, "(-1)^e = -1 for odd e");
        foreach (long message in new long[] { 2, 42, 65, 1000 })
            Check(Encrypt(message, e, n) != message, "ciphertext differs " + message);

        // wrong key: applying the public exponent to the ciphertext does not recover the message
        long ciphertext42 = Encrypt(42, e, n);
        Check(ciphertext42 == 2557, "c(42) = 2557");
        Check(Decrypt(ciphertext42, e, n) == 1001 && Decrypt(ciphertext42, e, n) != 42,
                "wrong key gives 1001, not 42");

        // second keypair with e = 3: p=101, q=113 -> n=11413, phi=11200, d=7467
        var keys2 = Keygen(101, 113, 3);
        long n2 = keys2[0], e2 = keys2[1], d2 = keys2[2], phi2 = keys2[3];
        Check(Join(keys2) == "11413,3,7467,11200", "keygen 101,113,3");
        Check(e2 * d2 % phi2 == 1, "e*d = 1 (mod phi) second key");
        foreach (long message in new long[] { 0, 1, 2, 5, 42, 999, 11412 })
            Check(Decrypt(Encrypt(message, e2, n2), d2, n2) == message, "round trip 2 " + message);
        long ciphertext7 = Encrypt(7, e2, n2);
        Check(Decrypt(ciphertext7, e2, n2) == 8652 && Decrypt(ciphertext7, e2, n2) != 7,
                "second key wrong key gives 8652");

        // keygen rejects bad demo parameters (documented errors)
        var badParams = new[]
        {
            new long[] { 7, 13, 3 }, new long[] { 9, 53, 17 },
            new long[] { 61, 61, 17 }, new long[] { 61, 53, 4 },
        };
        foreach (var bad in badParams)
        {
            bool threw = false;
            try { Keygen(bad[0], bad[1], bad[2]); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "bad keygen params rejected");
        }

        // encryption domain is checked
        foreach (long message in new long[] { 3233, -1 })
        {
            bool threw = false;
            try { Encrypt(message, e, n); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "message outside [0, n) rejected");
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
