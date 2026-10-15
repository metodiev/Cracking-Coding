// Task 380 - Modular inverse (extended Euclid and Fermat) + fast modular exponentiation. (Tier: Intermediate, CTCI Ch 6)
using System;

/// <summary>
/// Two independent inverse implementations:
///   * extended Euclid: g = gcd(a, m) and Bezout coefficients x, y with a*x + m*y = g;
///     when g == 1, x mod m is the inverse.  Works for ANY modulus m;
///   * Fermat: for PRIME p and a not divisible by p, inv = a^(p-2) mod p.
/// Both must agree on every coprime pair, and a * inv(a, m) % m == 1 on LCG values.
///
/// Documented failure: when gcd(a, m) != 1 no inverse exists; the API returns null (the
/// documented sentinel) instead of silently returning garbage.  The m = 1 case is
/// documented as out of scope (every residue is 0 mod 1) and rejected.
///
/// Fast modular exponentiation is binary (square-and-multiply) with the usual contract
/// ModPow(a, 0, m) = 1 mod m; hand-checked: 2^10 = 1024, 3^0 = 1, cross-checked against a
/// naive O(e) loop for small exponents, checked against the identity
/// pow(a, x+y) = pow(a, x) * pow(a, y) mod m, and Fermat's little theorem a^(p-1) = 1
/// mod p is asserted for several primes.
///
/// All arithmetic is exact long arithmetic (products stay below 1.1e18); no floats.
/// </summary>
public static class Solution
{
    private sealed class Lcg
    {
        private int seed;

        public Lcg(int seed)
        {
            this.seed = seed;
        }

        public int Next(int modulus)
        {
            seed = (seed * 75 + 74) % 65537;
            return seed % modulus;
        }
    }

    /// <summary>Fast binary modular exponentiation; exponents &gt;= 0 only (documented).</summary>
    public static long PowMod(long baseValue, long exponent, long modulus)
    {
        if (exponent < 0) throw new ArgumentException("negative exponents are not supported");
        if (modulus <= 0) throw new ArgumentException("modulus must be positive");
        long result = 1 % modulus;
        baseValue = ((baseValue % modulus) + modulus) % modulus;
        while (exponent > 0)
        {
            if ((exponent & 1) == 1) result = result * baseValue % modulus;
            baseValue = baseValue * baseValue % modulus;
            exponent >>= 1;
        }
        return result;
    }

    public static long Gcd(long a, long b)
    {
        while (b != 0)
        {
            long t = a % b;
            a = b;
            b = t;
        }
        return a < 0 ? -a : a;
    }

    /// <summary>Iterative extended Euclid: returns (g, x, y) with a*x + b*y = g &gt;= 0.</summary>
    public static (long g, long x, long y) ExtendedGcd(long a, long b)
    {
        long oldR = a, r = b;
        long oldS = 1, s = 0;
        long oldT = 0, t = 1;
        while (r != 0)
        {
            long q = oldR / r;
            (oldR, r) = (r, oldR - q * r);
            (oldS, s) = (s, oldS - q * s);
            (oldT, t) = (t, oldT - q * t);
        }
        if (oldR < 0) return (-oldR, -oldS, -oldT);
        return (oldR, oldS, oldT);
    }

    /// <summary>Inverse of a modulo m via extended Euclid, or null when gcd(a, m) != 1.</summary>
    public static long? ModInverseEuclid(long a, long m)
    {
        if (m <= 1) throw new ArgumentException("m >= 2 required (m = 1 is out of scope)");
        long value = ((a % m) + m) % m;
        (long g, long x, _) = ExtendedGcd(value, m);
        if (g != 1) return null;
        return ((x % m) + m) % m;
    }

    /// <summary>Inverse of a modulo the PRIME p via a^(p-2) mod p; null when p divides a.</summary>
    public static long? ModInverseFermat(long a, long p)
    {
        if (p <= 1) throw new ArgumentException("p > 1 required");
        long value = ((a % p) + p) % p;
        if (value == 0) return null;
        return PowMod(value, p - 2, p);
    }

    /// <summary>Independent O(exponent) oracle for small exponents.</summary>
    public static long PowNaive(long baseValue, long exponent, long modulus)
    {
        long result = 1 % modulus;
        for (long i = 0; i < exponent; i++) result = result * (baseValue % modulus) % modulus;
        return result;
    }

    public static void Main()
    {
        // --- hand-checked small cases -----------------------------------------
        Check(ModInverseEuclid(3, 11) == 4L, "inv(3, 11) = 4");
        Check(ModInverseEuclid(1, 11) == 1L, "inv(1, 11) = 1");
        Check(ModInverseEuclid(1, 7) == 1L, "inv(1, 7) = 1");
        Check(ModInverseEuclid(1, 1000000000L) == 1L, "inv(1, 1e9) = 1");
        Check(ModInverseEuclid(2, 5) == 3L, "inv(2, 5) = 3");
        Check(ModInverseEuclid(10, 17) == 12L, "inv(10, 17) = 12");
        Check(ModInverseEuclid(4, 8) == null, "gcd(4, 8) != 1 -> null");
        Check(ModInverseEuclid(6, 9) == null, "gcd(6, 9) != 1 -> null");
        Check(ModInverseEuclid(0, 5) == null, "gcd(0, 5) != 1 -> null");

        Check(ModInverseFermat(3, 11) == 4L, "fermat inv(3, 11) = 4");
        Check(ModInverseFermat(5, 101) == 81L, "fermat inv(5, 101) = 81");
        Check(ModInverseFermat(11, 11) == null, "p divides a -> null");

        // --- fast exponentiation, hand-checked ---------------------------------
        Check(PowMod(2, 10, 1000000007L) == 1024L, "2^10 = 1024");
        Check(PowMod(3, 0, 1000000007L) == 1L, "3^0 = 1");
        Check(PowMod(0, 0, 7) == 1L, "0^0 = 1 mod 7");
        Check(PowMod(7, 1, 5) == 2L, "7 mod 5");
        Check(PowMod(5, 3, 13) == 8L, "5^3 mod 13");
        Check(PowMod(2, 1000, 1000000007L) == 688423210L, "2^1000 mod p (recomputed)");
        Check(PowMod(3, 200, 1000000007L) == 136318165L, "3^200 mod p (recomputed)");

        // binary vs naive loop for small exponents
        var lcg = new Lcg(38013);
        for (int trial = 0; trial < 60; trial++)
        {
            long baseValue = lcg.Next(1000);
            long exponent = lcg.Next(40);
            long modulus = lcg.Next(999) + 2;
            Check(PowMod(baseValue, exponent, modulus) == PowNaive(baseValue, exponent, modulus),
                "binary vs naive " + trial);
        }

        // pow(a, x + y) = pow(a, x) * pow(a, y) mod m
        for (int trial = 0; trial < 40; trial++)
        {
            long baseValue = lcg.Next(1000000006) + 1L;
            long x = lcg.Next(60);
            long y = lcg.Next(60);
            long modulus = 1000000007L;
            Check(PowMod(baseValue, x + y, modulus)
                == PowMod(baseValue, x, modulus) * PowMod(baseValue, y, modulus) % modulus,
                "pow additive identity " + trial);
        }

        // --- extended Euclid identity a*inv = 1 (mod m) on LCG values ----------
        for (int trial = 0; trial < 60; trial++)
        {
            long m = lcg.Next(998) + 2L;
            long a = lcg.Next(1000);
            long? inverse = ModInverseEuclid(a, m);
            if (Gcd(a, m) == 1)
            {
                Check(inverse != null, "coprime pair has inverse " + a + " " + m);
                Check(a * inverse!.Value % m == 1, "a * inv = 1 mod m " + a + " " + m);
            }
            else
            {
                Check(inverse == null, "non-coprime pair has no inverse " + a + " " + m);
            }
        }

        // --- Euclid and Fermat agree for primes --------------------------------
        foreach (long prime in new long[] { 7, 11, 13, 101, 65537, 1000000007L })
        {
            for (int trial = 0; trial < 12; trial++)
            {
                long a = lcg.Next((int)Math.Min(prime - 1, 65536)) + 1;
                if (a % prime == 0) continue;
                long? euclid = ModInverseEuclid(a, prime);
                long? fermat = ModInverseFermat(a, prime);
                Check(euclid != null && euclid == fermat, "euclid == fermat " + a + " " + prime);
                Check(a * euclid!.Value % prime == 1, "inverse verifies " + a + " " + prime);
            }
        }

        // Fermat's little theorem: a^(p-1) = 1 mod p for p prime, p not dividing a
        foreach (long prime in new long[] { 7, 11, 13, 101, 1000000007L })
        {
            foreach (long a in new long[] { 1, 2, 3, prime - 1, prime - 2 })
            {
                if (a % prime == 0) continue;
                Check(PowMod(a, prime - 1, prime) == 1, "fermat little theorem " + a + " " + prime);
            }
        }

        // Wilson-flavoured sanity for a small prime: (19-1)! = -1 mod 19
        long factorial = 1;
        for (int i = 1; i <= 18; i++) factorial = factorial * i % 19;
        Check(factorial == 18, "Wilson 18! = -1 mod 19");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
