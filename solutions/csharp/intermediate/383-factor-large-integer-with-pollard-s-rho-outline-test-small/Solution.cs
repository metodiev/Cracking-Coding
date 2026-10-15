// Task 383 - Factor a large integer with Pollard's rho (outline / small tests). (Tier: Intermediate, CTCI Ch 6)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static readonly ulong[] MillerRabinBases = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

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

    private static ulong Gcd(ulong a, ulong b)
    {
        while (b != 0)
        {
            ulong t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /// <summary>Deterministic Miller-Rabin (same twelve-base set as task 382).</summary>
    private static bool IsPrime(ulong n)
    {
        if (n < 2) return false;
        foreach (ulong p in MillerRabinBases)
            if (n % p == 0) return n == p;
        ulong d = n - 1;
        int s = 0;
        while ((d & 1) == 0) { d >>= 1; s++; }
        foreach (ulong a in MillerRabinBases)
        {
            ulong x = PowMod(a, d, n);
            if (x == 1 || x == n - 1) continue;
            bool witnessed = true;
            for (int r = 1; r < s; r++)
            {
                x = (ulong)((UInt128)x * x % n);
                if (x == n - 1) { witnessed = false; break; }
            }
            if (witnessed) return false;
        }
        return true;
    }

    private static ulong F(ulong x, ulong c, ulong n)
    {
        return (ulong)(((UInt128)x * x + c) % n);
    }

    /// <summary>Floyd cycle-finding variant; retries with the next c when the gcd is n.</summary>
    private static ulong PollardRho(ulong n)
    {
        if (n % 2 == 0) return 2;
        if (n % 3 == 0) return 3;
        for (ulong c = 1; c < 100; c++)
        {
            ulong x = 2, y = 2, d = 1;
            while (d == 1)
            {
                x = F(x, c, n);
                y = F(F(y, c, n), c, n);
                d = Gcd(x > y ? x - y : y - x, n);
            }
            if (d != n) return d;
        }
        throw new InvalidOperationException("Pollard's rho failed to find a factor");
    }

    private static void Split(ulong value, List<ulong> outList)
    {
        if (value == 1) return;
        if (IsPrime(value))
        {
            outList.Add(value);
            return;
        }
        ulong factor = PollardRho(value);
        Split(factor, outList);
        Split(value / factor, outList);
    }

    /// <summary>Complete prime factorisation in ascending order ([] for 1).</summary>
    private static ulong[] Factorize(ulong n)
    {
        var outList = new List<ulong>();
        Split(n, outList);
        outList.Sort();
        return outList.ToArray();
    }

    private static string Join(ulong[] values) => string.Join(",", values);

    public static void Main()
    {
        // hand-checked factorisations
        Check(Join(Factorize(8051)) == "83,97", "8051 = 83*97");
        Check(Join(Factorize(10403)) == "101,103", "10403 = 101*103");
        Check(Join(Factorize(4294967297)) == "641,6700417", "2^32+1");
        Check(Join(Factorize(123456789)) == "3,3,3607,3803", "123456789");
        Check(Join(Factorize(600851475143)) == "71,839,1471,6857", "600851475143");
        Check(Join(Factorize(1099511627777)) == "257,4278255361", "2^40+1");
        Check(83ul * 97 == 8051 && 101ul * 103 == 10403, "small products");
        Check(641ul * 6700417 == 4294967297, "641*6700417");
        Check(9ul * 13717421 == 123456789 && 3607ul * 3803 == 13717421, "123456789 factors");
        Check(71ul * 839 * 1471 * 6857 == 600851475143, "Project Euler 3 factors");
        Check(257ul * 4278255361 == 1099511627777, "257*4278255361");

        // primes and 1 are handled without recursion
        Check(Factorize(1).Length == 0, "Factorize(1)");
        Check(Join(Factorize(2)) == "2", "Factorize(2)");
        Check(Join(Factorize(97)) == "97", "Factorize(97)");
        Check(Join(Factorize(97ul * 97)) == "97,97", "Factorize(97^2)");

        // cross-check: product of factors == n, every factor prime, list ascending
        ulong[] samples = { 8051, 10403, 4294967297, 123456789, 600851475143, 1099511627777,
            999983, 1, 4, 51, 65536, 1048577 };
        foreach (ulong n in samples)
        {
            var factors = Factorize(n);
            ulong product = 1;
            foreach (ulong factor in factors)
            {
                Check(IsPrime(factor), "factor prime " + n + "/" + factor);
                product *= factor;
            }
            Check(product == n, "product " + n);
            for (int i = 1; i < factors.Length; i++)
                Check(factors[i - 1] <= factors[i], "ascending " + n);
        }
        Check(Join(Factorize(999983)) == "999983", "999983 is prime");
        var powers = Factorize(65536);
        Check(powers.Length == 16, "65536 has 16 factors");
        foreach (ulong p in powers) Check(p == 2, "factor of 65536 is 2");
        Check(Join(Factorize(1048577)) == "17,61681", "2^20+1 = 17*61681");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
