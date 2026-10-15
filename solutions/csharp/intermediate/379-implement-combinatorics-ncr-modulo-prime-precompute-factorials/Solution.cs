// Task 379 - Binomial coefficients C(n, k) mod prime p via precomputed factorials. (Tier: Intermediate, CTCI Ch 6)
using System;
using System.Numerics;

/// <summary>
/// For a prime p = 1_000_000_007 factorials and inverse factorials are precomputed once:
/// invFact[n] = fact[n]^(p-2) mod p (Fermat), then downward invFact[i] = invFact[i+1]*(i+1).
/// Then C(n, k) = fact[n] * invFact[k] * invFact[n-k] mod p in O(1) per query, valid for
/// 0 &lt;= k &lt;= n &lt; p (Lucas is not needed below p).
///
/// Hand-verified assertions: C(10,3) = 120, C(20,10) = 184756, C(n,0) = C(n,n) = 1,
/// k &gt; n -&gt; 0 (the convention used everywhere in this repo), k &lt; 0 -&gt; 0.
///
/// Independent verification paths:
///   * a multiplicative loop C(n,k) = prod_{i=1..k} (n-k+i) * inv(i) mod p that never
///     touches the factorial tables;
///   * additive Pascal rows built by repeated addition mod p (no modular inverses at all)
///     for n &lt;= 140, cross-checked on LCG (n, k) pairs;
///   * the known constants C(100,50) mod p = 538992043 and C(1000,500) mod p = 159835829,
///     each obtained independently (multiplicative loop + exact BigInteger reduction of
///     the true value) and asserted here.
///
/// All modular products stay below 1.1e18 (signed 64-bit safe); the BigInteger path is a
/// separate belt-and-braces check of the two constants.  No floating point anywhere.
/// </summary>
public static class Solution
{
    private const long Mod = 1000000007L;

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

    /// <summary>Fast modular exponentiation (binary).</summary>
    public static long ModPow(long baseValue, long exponent, long modulus)
    {
        long result = 1 % modulus;
        baseValue %= modulus;
        while (exponent > 0)
        {
            if ((exponent & 1) == 1) result = result * baseValue % modulus;
            baseValue = baseValue * baseValue % modulus;
            exponent >>= 1;
        }
        return result;
    }

    /// <summary>C(n, k) mod Mod with precomputed tables.</summary>
    private sealed class Binomial
    {
        private readonly long[] fact;
        private readonly long[] invFact;
        private readonly int limit;

        public Binomial(int limit)
        {
            this.limit = limit;
            fact = new long[limit + 1];
            fact[0] = 1;
            for (int i = 1; i <= limit; i++) fact[i] = fact[i - 1] * i % Mod;
            invFact = new long[limit + 1];
            invFact[limit] = ModPow(fact[limit], Mod - 2, Mod);
            for (int i = limit; i > 0; i--) invFact[i - 1] = invFact[i] * i % Mod;
        }

        public long Choose(int n, int k)
        {
            if (k < 0 || n < 0 || k > n) return 0;
            if (n > limit) throw new ArgumentException("n above the precomputed limit");
            return fact[n] * invFact[k] % Mod * invFact[n - k] % Mod;
        }
    }

    /// <summary>Independent path: running product with per-step modular inverses.</summary>
    private static long ChooseMultiplicative(int n, int k)
    {
        if (k < 0 || n < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        long result = 1;
        for (int i = 1; i <= k; i++)
        {
            result = result * ((n - k + i) % Mod) % Mod;
            result = result * ModPow(i, Mod - 2, Mod) % Mod;
        }
        return result;
    }

    /// <summary>Additive Pascal row mod Mod - no division or inversion used at all.</summary>
    private static long[] PascalRow(int n)
    {
        var row = new long[] { 1 };
        for (int step = 0; step < n; step++)
        {
            var next = new long[row.Length + 1];
            next[0] = 1;
            next[next.Length - 1] = 1;
            for (int i = 1; i < row.Length; i++) next[i] = (row[i - 1] + row[i]) % Mod;
            row = next;
        }
        return row;
    }

    /// <summary>Exact C(n, k) as a BigInteger, reduced mod Mod - a third fully independent path.</summary>
    private static long ChooseBigInteger(int n, int k)
    {
        if (k < 0 || n < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        BigInteger numerator = 1;
        BigInteger denominator = 1;
        for (int i = 1; i <= k; i++)
        {
            numerator *= n - k + i;
            denominator *= i;
        }
        BigInteger exact = numerator / denominator;
        BigInteger reduced = exact % Mod;
        return (long)reduced;
    }

    public static void Main()
    {
        var table = new Binomial(1000);

        Check(table.Choose(10, 3) == 120, "C(10,3) = 120");
        Check(table.Choose(20, 10) == 184756, "C(20,10) = 184756");
        Check(table.Choose(5, 0) == 1 && table.Choose(5, 5) == 1, "C(n,0) = C(n,n) = 1");
        Check(table.Choose(0, 0) == 1, "C(0,0) = 1");
        Check(table.Choose(3, 5) == 0, "k > n -> 0");
        Check(table.Choose(3, -1) == 0, "k < 0 -> 0");
        Check(table.Choose(0, 1) == 0, "C(0,1) = 0");

        // symmetry C(n, k) == C(n, n-k)
        foreach (int n in new[] { 1, 2, 7, 40, 999, 1000 })
        {
            foreach (int k in new[] { 0, 1, n / 2, n - 1, n })
            {
                Check(table.Choose(n, k) == table.Choose(n, n - k), "symmetry " + n + " " + k);
            }
        }

        // real (non-modular) checks where the exact value is known
        Check(ChooseMultiplicative(10, 3) == 120, "multiplicative C(10,3)");
        Check(ChooseMultiplicative(20, 10) == 184756, "multiplicative C(20,10)");
        Check(ChooseMultiplicative(52, 5) == 2598960, "poker hands C(52,5)");

        // known constants, each verified by two independent paths
        Check(table.Choose(100, 50) == 538992043L, "C(100,50) mod p");
        Check(ChooseMultiplicative(100, 50) == 538992043L, "multiplicative C(100,50)");
        Check(ChooseBigInteger(100, 50) == 538992043L, "BigInteger C(100,50)");
        Check(table.Choose(1000, 500) == 159835829L, "C(1000,500) mod p");
        Check(ChooseMultiplicative(1000, 500) == 159835829L, "multiplicative C(1000,500)");
        Check(ChooseBigInteger(1000, 500) == 159835829L, "BigInteger C(1000,500)");

        // additive Pascal rows: completely different arithmetic path for smaller n
        var lcg = new Lcg(37913);
        for (int trial = 0; trial < 60; trial++)
        {
            int n = lcg.Next(141);
            int k = lcg.Next(n + 2) - 1;              // includes -1, 0, n and n+1
            long expected = (k < 0 || k > n) ? 0 : PascalRow(n)[k];
            Check(table.Choose(n, k) == expected, "pascal cross-check " + n + " " + k);
            Check(ChooseMultiplicative(n, k) == expected, "pascal multiplicative " + n + " " + k);
        }
        foreach (int n in new[] { 1, 2, 3, 10, 50, 140 })
        {
            long[] row = PascalRow(n);
            for (int k = 0; k <= n; k++)
            {
                Check(table.Choose(n, k) == row[k], "full pascal row " + n + " " + k);
                Check(ChooseMultiplicative(n, k) == row[k], "full pascal multiplicative " + n + " " + k);
            }
        }

        // Pascal identity C(n, k) = C(n-1, k-1) + C(n-1, k) on the modular tables
        for (int trial = 0; trial < 40; trial++)
        {
            int n = lcg.Next(1001);
            int k = lcg.Next(n + 1);
            if (k < 1 || k > n) continue;
            Check(table.Choose(n, k) == (table.Choose(n - 1, k - 1) + table.Choose(n - 1, k)) % Mod,
                "pascal identity " + n + " " + k);
        }

        // binomial theorem sanity: sum_k C(n, k) = 2^n mod p
        foreach (int n in new[] { 5, 30, 200, 1000 })
        {
            long total = 0;
            for (int k = 0; k <= n; k++) total = (total + table.Choose(n, k)) % Mod;
            Check(total == ModPow(2, n, Mod), "binomial theorem " + n);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
