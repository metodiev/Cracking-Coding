// Task 382 - Miller-Rabin primality test with deterministic small bases. (Tier: Intermediate, CTCI Ch 6)
using System;

public static class Solution
{
    private static readonly ulong[] DeterministicBases = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    private static ulong PowMod(ulong baseValue, ulong exponent, ulong modulus)
    {
        ulong result = 1 % modulus;
        ulong b = baseValue % modulus;
        ulong e = exponent;
        while (e > 0)
        {
            if ((e & 1) == 1) result = (ulong)((UInt128)result * b % modulus);
            b = (ulong)((UInt128)b * b % modulus);
            e >>= 1;
        }
        return result;
    }

    /// <summary>One-base strong test: true when the base fails to prove n composite.</summary>
    private static bool IsStrongProbablePrime(ulong n, ulong baseValue)
    {
        if (n < 2) return false;
        ulong d = n - 1;
        int s = 0;
        while ((d & 1) == 0) { d >>= 1; s++; }
        ulong x = PowMod(baseValue, d, n);
        if (x == 1 || x == n - 1) return true;
        for (int r = 1; r < s; r++)
        {
            x = (ulong)((UInt128)x * x % n);
            if (x == n - 1) return true;
        }
        return false;
    }

    /// <summary>Deterministic Miller-Rabin (the twelve-base set is valid for n below 3.3e24).</summary>
    private static bool IsPrime(ulong n)
    {
        if (n < 2) return false;
        foreach (ulong p in DeterministicBases)
            if (n % p == 0) return n == p;
        foreach (ulong a in DeterministicBases)
            if (!IsStrongProbablePrime(n, a)) return false;
        return true;
    }

    /// <summary>Reference oracle for the cross-check; O(sqrt(n)).</summary>
    private static bool IsPrimeTrialDivision(ulong n)
    {
        if (n < 2) return false;
        for (ulong i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    public static void Main()
    {
        foreach (ulong p in DeterministicBases)
            Check(IsPrime(p), "base is prime " + p);

        ulong[] knownPrimes = { 41, 97, 541, 7919, 104729, 1299709, 2147483647,
            2305843009213693951, 67280421310721 };
        foreach (ulong p in knownPrimes) Check(IsPrime(p), "known prime " + p);

        ulong[] knownComposites = { 0, 1, 4, 9, 15, 21, 25, 49, 91, 121, 341, 561, 1105, 1729,
            4294967297 };
        foreach (ulong c in knownComposites) Check(!IsPrime(c), "known composite " + c);
        Check(11ul * 31 == 341, "341 = 11*31");
        Check(3ul * 11 * 17 == 561, "561 = 3*11*17");
        Check(5ul * 13 * 17 == 1105, "1105 = 5*13*17");
        Check(7ul * 13 * 19 == 1729, "1729 = 7*13*19");
        Check(641ul * 6700417 == 4294967297, "2^32+1 = 641*6700417");

        // 2047 = 23*89: the base-2 test alone would call it prime; extra bases expose it
        Check(23ul * 89 == 2047, "2047 = 23*89");
        Check(IsStrongProbablePrime(2047, 2), "2047 passes base 2");
        Check(!IsStrongProbablePrime(2047, 3), "2047 fails base 3");
        Check(!IsPrime(2047), "2047 composite");

        // 3215031751 = 151*751*28351 passes the first four bases but not the whole set
        Check(151ul * 751 * 28351 == 3215031751, "3215031751 = 151*751*28351");
        foreach (ulong b in new ulong[] { 2, 3, 5, 7 })
            Check(IsStrongProbablePrime(3215031751, b), "3215031751 passes base " + b);
        Check(!IsStrongProbablePrime(3215031751, 11), "3215031751 fails base 11");
        Check(!IsPrime(3215031751), "3215031751 composite");

        // cross-check against trial division for every n <= 10000
        int primeCount = 0;
        for (ulong n = 0; n <= 10000; n++)
        {
            Check(IsPrime(n) == IsPrimeTrialDivision(n), "trial division agrees at " + n);
            if (n > 1 && IsPrime(n)) primeCount++;
        }
        Check(primeCount == 1229, "pi(10000) = 1229");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
