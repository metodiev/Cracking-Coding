// Task 384 - Implement discrete logarithm baby-step giant-step. (Tier: Intermediate, CTCI Ch 6)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static long ModPower(long baseValue, long exponent, long modulus)
    {
        long result = 1 % modulus;
        long b = ((baseValue % modulus) + modulus) % modulus;
        long e = exponent;
        while (e > 0)
        {
            if ((e & 1) == 1) result = result * b % modulus;
            b = b * b % modulus;
            e >>= 1;
        }
        return result;
    }

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

    /// <summary>
    /// Smallest x >= 0 with g^x = h (mod p); -1 when no such x exists.
    /// Baby steps list g^j -> j for j in [0, m), m = ceil(sqrt(p)); giant steps walk
    /// h * (g^(-m))^i. The first hit with the smallest i and j is the minimal solution.
    /// </summary>
    private static long DiscreteLog(long g, long h, long p)
    {
        if (p <= 1) throw new InvalidOperationException("modulus must exceed 1");
        long target = ((h % p) + p) % p;
        long m = (long)Math.Sqrt(p);
        while (m * m < p) m++;
        var baby = new Dictionary<long, long>();
        long value = 1 % p;
        for (long j = 0; j < m; j++)
        {
            if (!baby.ContainsKey(value)) baby[value] = j;
            value = value * g % p;
        }
        long giantFactor = ModPower(ModInverse(g, p), m, p);
        long gamma = target;
        for (long i = 0; i <= m; i++)
        {
            if (baby.TryGetValue(gamma, out long j)) return i * m + j;
            gamma = gamma * giantFactor % p;
        }
        return -1;
    }

    /// <summary>Reference oracle: minimal x in [0, p), or -1; O(p).</summary>
    private static long BruteForceDiscreteLog(long g, long h, long p)
    {
        long target = ((h % p) + p) % p;
        long value = 1 % p;
        for (long x = 0; x < p; x++)
        {
            if (value == target) return x;
            value = value * g % p;
        }
        return -1;
    }

    private static int seed = 9871;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        // hand-verified small problems
        Check(DiscreteLog(2, 8, 13) == 3 && ModPower(2, 3, 13) == 8, "2^x = 8 (13)");
        Check(DiscreteLog(3, 13, 17) == 4 && ModPower(3, 4, 17) == 13, "3^x = 13 (17)");
        Check(DiscreteLog(2, 10, 19) == 17 && ModPower(2, 17, 19) == 10, "2^x = 10 (19)");
        Check(DiscreteLog(2, 5, 11) == 4 && ModPower(2, 4, 11) == 5, "2^x = 5 (11)");
        Check(DiscreteLog(2, 1, 13) == 0, "g^0 = 1 always");

        // documented failure sentinel: no solution when h is outside the subgroup <g>
        Check(DiscreteLog(2, 3, 7) == -1, "<2> = {1,2,4} (7) misses 3");
        Check(BruteForceDiscreteLog(2, 3, 7) == -1, "brute force agrees (2,3,7)");
        Check(DiscreteLog(3, 2, 11) == -1, "<3> = {1,3,4,5,9} (11) misses 2");
        Check(BruteForceDiscreteLog(3, 2, 11) == -1, "brute force agrees (3,2,11)");
        Check(DiscreteLog(1, 2, 7) == -1, "powers of 1 are all 1");
        Check(DiscreteLog(2, 0, 7) == -1, "0 is never a power of g");

        // cross-check against brute force for small primes
        long[] primes = { 13, 17, 19, 97, 101, 103, 211, 1009 };
        long[] targets = { 1, 2, 3, 5, 10, 42, 123 };
        foreach (long p in primes)
        {
            for (long g = 2; g < Math.Min(p, 6); g++)
            {
                foreach (long h in targets)
                {
                    long expected = BruteForceDiscreteLog(g, h, p);
                    Check(DiscreteLog(g, h, p) == expected, "cross-check " + g + "," + h + "," + p);
                }
            }
        }

        // LCG triples: h = g^x is solvable and the answer must satisfy g^answer = h
        for (int trial = 0; trial < 60; trial++)
        {
            long p = primes[NextSeed() % primes.Length];
            long g = 2 + NextSeed() % (p - 2);
            long x = NextSeed() % p;
            long h = ModPower(g, x, p);
            long found = DiscreteLog(g, h, p);
            Check(found != -1 && ModPower(g, found, p) == h, "LCG solvable " + trial);
            Check(found == BruteForceDiscreteLog(g, h, p), "LCG minimal " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
