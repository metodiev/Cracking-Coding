// Task 381 - Chinese remainder theorem solver for pairwise coprime moduli. (Tier: Intermediate, CTCI Ch 6)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>Returns {g, x, y} with a*x + b*y == g == gcd(a, b).</summary>
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

    /// <summary>Merge two congruences; throws when the moduli are not coprime.</summary>
    private static long[] CrtTwo(long a1, long m1, long a2, long m2)
    {
        if (ExtendedGcd(m1, m2)[0] != 1)
            throw new InvalidOperationException($"moduli {m1} and {m2} are not coprime");
        long t = ((((a2 - a1) % m2) * ModInverse(m1, m2)) % m2 + m2) % m2;
        long modulus = m1 * m2;
        return new long[] { ((a1 + m1 * t) % modulus + modulus) % modulus, modulus };
    }

    /// <summary>
    /// Solve a system of congruences with pairwise coprime moduli.
    /// Returns {x, m}: x the smallest non-negative solution, m the product of the moduli.
    /// Throws for an empty system or non-coprime moduli.
    /// </summary>
    private static long[] Crt(long[][] congruences)
    {
        if (congruences.Length == 0) throw new InvalidOperationException("empty congruence system");
        long a = 0, m = 1;
        foreach (var c in congruences)
        {
            long residue = c[0], modulus = c[1];
            if (modulus <= 0) throw new InvalidOperationException("modulus must be positive");
            var merged = CrtTwo(a, m, ((residue % modulus) + modulus) % modulus, modulus);
            a = merged[0];
            m = merged[1];
        }
        return new long[] { a, m };
    }

    private static int seed = 4212;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static string Join(long[] values) => string.Join(",", values);

    public static void Main()
    {
        // classic worked example: 23 = 2 (3), 23 = 3 (5), 23 = 2 (7)
        var classic = Crt(new[] {
            new long[] { 2, 3 }, new long[] { 3, 5 }, new long[] { 2, 7 },
        });
        Check(Join(classic) == "23,105", "classic (23, 105)");
        Check(classic[0] % 3 == 2 && classic[0] % 5 == 3 && classic[0] % 7 == 2, "substitution classic");

        // a single congruence is the identity; merge order must not matter
        Check(Join(Crt(new[] { new long[] { 4, 9 } })) == "4,9", "single congruence");
        Check(Join(Crt(new[] { new long[] { 23, 105 } })) == "23,105", "already combined");
        Check(Join(Crt(new[] {
            new long[] { 2, 7 }, new long[] { 3, 5 }, new long[] { 2, 3 },
        })) == "23,105", "merge order");
        Check(Join(Crt(new[] { new long[] { 0, 2 }, new long[] { 1, 3 } })) == "4,6", "small system");

        // non-coprime moduli are rejected (documented error), even when consistent
        var bad = new[]
        {
            new[] { new long[] { 1, 4 }, new long[] { 2, 6 } },
            new[] { new long[] { 0, 2 }, new long[] { 0, 4 } },
            new[] { new long[] { 1, 9 }, new long[] { 1, 3 } },
        };
        foreach (var system in bad)
        {
            bool threw = false;
            try { Crt(system); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "non-coprime rejected");
        }

        // LCG-driven systems: cross-check by substitution and by brute force where feasible
        int[] primes = { 3, 5, 7, 11, 13, 17, 19, 23 };
        int verifiedCount = 0;
        for (int trial = 0; trial < 40; trial++)
        {
            int count = 2 + NextSeed() % 3;
            var chosen = new List<int>();
            while (chosen.Count < count)
            {
                int p = primes[NextSeed() % primes.Length];
                if (!chosen.Contains(p)) chosen.Add(p);
            }
            var system = new long[count][];
            for (int i = 0; i < count; i++)
            {
                int p = chosen[i];
                system[i] = new long[] { NextSeed() % p, p };
            }
            var solved = Crt(system);
            foreach (var c in system)
                Check(solved[0] % c[1] == c[0], "substitution trial " + trial);
            long product = 1;
            foreach (var c in system) product *= c[1];
            Check(solved[1] == product && solved[0] < solved[1], "product trial " + trial);
            if (solved[1] <= 100000)
            {
                long brute = 0;
                while (brute < solved[1])
                {
                    bool ok = true;
                    foreach (var c in system)
                        if (brute % c[1] != c[0]) { ok = false; break; }
                    if (ok) break;
                    brute++;
                }
                Check(brute == solved[0], "brute force trial " + trial);
                verifiedCount++;
            }
        }
        Check(verifiedCount >= 10, "enough brute-force checks");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
