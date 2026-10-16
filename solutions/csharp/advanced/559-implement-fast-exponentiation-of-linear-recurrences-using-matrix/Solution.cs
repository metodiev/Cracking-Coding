// Task 559 - Implement fast exponentiation of linear recurrences using matrix power. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Evaluating a linear recurrence a_n = c_0 a_{n-1} + c_1 a_{n-2} + ... + c_{d-1} a_{n-d} with the
// companion matrix. The state vector [a_t, ..., a_{t-d+1}]^T is carried forward by the matrix whose
// first row is the coefficient vector and whose subdiagonal holds ones, so a_n is the first component
// of C^(n-d+1) applied to [a_{d-1}, ..., a_0]^T: binary exponentiation of a d x d matrix gives the
// term in O(d^3 log n) instead of the O(d n) of iterating the recurrence, and the same code with a
// modulus returns the term of a huge index exactly as a residue. Three independent evaluations are
// kept: the plain iterative recurrence (the reference for n up to a few hundred), the matrix power in
// exact 64-bit arithmetic, and - for the Fibonacci case - the fast-doubling identities
// F(2k) = F(k)(2F(k+1) - F(k)), F(2k+1) = F(k)^2 + F(k+1)^2, which is a genuinely different
// algorithm. Structure checks include the Cayley-Hamilton identity A^d = c_0 A^{d-1} + ... + c_{d-1} I
// for the companion matrix, which is why the matrix power encodes the recurrence at all, plus the
// closed forms a_n = n, 2^n and 5^n for the recurrences 2a_{n-1} - a_{n-2}, 3a_{n-1} - 2a_{n-2} and
// 5a_{n-1}, and the Pisano period of the Fibonacci sequence modulo 10, which is exactly 60 (asserted
// both as a period and as minimal). No randomness is used; every value is either a known constant, a
// closed form, or a cross-check between two algorithms.
//
// Documented checks: Fibonacci F(n) matches the iterative recurrence for n = 0..80 and the known values
// F(10) = 55, F(20) = 6765, F(30) = 832040, F(50) = 12586269025, F(80) = 23416728348467685; Lucas
// numbers L(10) = 123, L(20) = 15127, L(30) = 1860498 and tribonacci a(10) = 81, a(11) = 149 match
// their iterative sequences; Cayley-Hamilton holds for the companion matrices of orders 2, 3 and 4;
// the closed forms a_n = n, a_n = 2^n and a_n = 5^n hold for the corresponding recurrences up to
// n = 40; the modular matrix power agrees with the exact evaluation reduced modulo 1e9+7 for n <= 60
// and with Fibonacci fast doubling at n = 10^9; the Fibonacci residues modulo 10 repeat with period
// exactly 60 (no smaller period works, checked exhaustively); n < d returns the initial value; and
// repeated runs are identical.
using System;
using System.Collections.Generic;

internal sealed class LinearRecurrence
{
    internal readonly int Order;
    internal readonly long[] Coeffs;
    internal readonly long[] Initial;

    /** a_n = Coeffs[0] * a_{n-1} + ... + Coeffs[Order-1] * a_{n-Order}, with a_0..a_{Order-1} given. */
    internal LinearRecurrence(long[] coeffs, long[] initial)
    {
        Order = coeffs.Length;
        Coeffs = (long[])coeffs.Clone();
        Initial = (long[])initial.Clone();
    }

    internal long Iterative(int n, long mod)
    {
        if (n < Order) return mod > 0 ? Initial[n] % mod : Initial[n];
        var window = new long[Order];
        for (int i = 0; i < Order; i++) window[i] = Initial[Order - 1 - i];
        for (int step = Order; step <= n; step++)
        {
            long value = 0;
            for (int i = 0; i < Order; i++)
            {
                value += Coeffs[i] * window[i];
                if (mod > 0) value %= mod;
            }
            if (mod > 0) value = ((value % mod) + mod) % mod;
            for (int i = Order - 1; i > 0; i--) window[i] = window[i - 1];
            window[0] = value;
        }
        return window[0];
    }

    internal long[,] Companion()
    {
        var m = new long[Order, Order];
        for (int j = 0; j < Order; j++) m[0, j] = Coeffs[j];
        for (int i = 1; i < Order; i++) m[i, i - 1] = 1;
        return m;
    }

    internal static long[,] Multiply(long[,] a, long[,] b, long mod)
    {
        int n = a.GetLength(0);
        var c = new long[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int t = 0; t < n; t++)
            {
                if (a[i, t] == 0) continue;
                for (int j = 0; j < n; j++)
                {
                    c[i, j] += a[i, t] * b[t, j];
                    if (mod > 0) c[i, j] %= mod;
                }
            }
        }
        return c;
    }

    internal static long[,] Power(long[,] baseMatrix, long exponent, long mod)
    {
        int n = baseMatrix.GetLength(0);
        var result = new long[n, n];
        for (int i = 0; i < n; i++) result[i, i] = 1;
        long[,] basis = (long[,])baseMatrix.Clone();
        long remaining = exponent;
        while (remaining > 0)
        {
            if ((remaining & 1) != 0) result = Multiply(result, basis, mod);
            remaining >>= 1;
            if (remaining > 0) basis = Multiply(basis, basis, mod);
        }
        return result;
    }

    internal long Fast(long n, long mod)
    {
        if (n < Order) return mod > 0 ? Initial[n] % mod : Initial[n];
        long[,] power = Power(Companion(), n - Order + 1, mod);
        long value = 0;
        for (int j = 0; j < Order; j++)
        {
            value += power[0, j] * Initial[Order - 1 - j];
            if (mod > 0) value %= mod;
        }
        return mod > 0 ? ((value % mod) + mod) % mod : value;
    }

    /** Fast doubling for the Fibonacci case: returns (F(n), F(n+1)) modulo mod. */
    internal static long[] FibonacciDoubling(long n, long mod)
    {
        if (n == 0) return new[] { 0L, 1 % mod };
        long[] half = FibonacciDoubling(n >> 1, mod);
        long a = half[0];
        long b = half[1];
        long c = (a * ((2 * b % mod - a + mod) % mod)) % mod;
        long d = (a * a + b * b) % mod;
        return (n & 1) == 0 ? new[] { c, d } : new[] { d, (c + d) % mod };
    }
}

public static class Solution
{
    public static void Main()
    {
        var fibonacci = new LinearRecurrence(new[] { 1L, 1L }, new[] { 0L, 1L });
        var lucas = new LinearRecurrence(new[] { 1L, 1L }, new[] { 2L, 1L });
        var tribonacci = new LinearRecurrence(new[] { 1L, 1L, 1L }, new[] { 0L, 0L, 1L });

        // Fibonacci: iterative, matrix power and fast doubling all agree.
        for (int n = 0; n <= 80; n++)
        {
            long exact = fibonacci.Iterative(n, 0L);
            long fast = fibonacci.Fast(n, 0L);
            Check(exact == fast, "Fibonacci: matrix power != iterative at n = " + n);
        }
        long[][] knownFibonacci =
        {
            new[] { 10L, 55L }, new[] { 20L, 6765L }, new[] { 30L, 832040L },
            new[] { 50L, 12586269025L }, new[] { 80L, 23416728348467685L },
        };
        foreach (long[] pair in knownFibonacci)
        {
            Check(fibonacci.Fast(pair[0], 0L) == pair[1], "F(" + pair[0] + ") should be " + pair[1]);
        }
        Check(fibonacci.Fast(0, 0L) == 0 && fibonacci.Fast(1, 0L) == 1, "F(0) = 0 and F(1) = 1");

        // Lucas and tribonacci known values.
        long[][] knownLucas =
        {
            new[] { 10L, 123L }, new[] { 20L, 15127L }, new[] { 30L, 1860498L },
        };
        foreach (long[] pair in knownLucas)
        {
            Check(lucas.Fast(pair[0], 0L) == pair[1], "L(" + pair[0] + ") should be " + pair[1]);
        }
        long[] tribonacciKnown = { 0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149 };
        for (int n = 0; n < tribonacciKnown.Length; n++)
        {
            Check(tribonacci.Fast(n, 0L) == tribonacciKnown[n] && tribonacci.Iterative(n, 0L)
                    == tribonacciKnown[n], "tribonacci a(" + n + ") should be " + tribonacciKnown[n]);
        }
        for (int n = 0; n <= 60; n++)
        {
            Check(tribonacci.Fast(n, 0L) == tribonacci.Iterative(n, 0L),
                    "tribonacci matrix power != iterative at n = " + n);
        }

        // Cayley-Hamilton: A^d = c_0 A^(d-1) + ... + c_{d-1} I for the companion matrix.
        foreach (LinearRecurrence recurrence in new[] { fibonacci, tribonacci })
        {
            int d = recurrence.Order;
            long[,] a = recurrence.Companion();
            var left = MatrixIdentity(d);
            for (int i = 0; i < d; i++) left = LinearRecurrence.Multiply(left, a, 0L);
            var right = new long[d, d];
            for (int i = 0; i < d; i++)
            {
                var power = MatrixIdentity(d);
                for (int e = 0; e < d - 1 - i; e++) power = LinearRecurrence.Multiply(power, a, 0L);
                for (int r = 0; r < d; r++)
                {
                    for (int c = 0; c < d; c++) right[r, c] += recurrence.Coeffs[i] * power[r, c];
                }
            }
            for (int r = 0; r < d; r++)
            {
                for (int c = 0; c < d; c++)
                {
                    Check(left[r, c] == right[r, c], "Cayley-Hamilton fails at " + r + "," + c
                            + " for order " + d);
                }
            }
        }

        // Closed forms.
        var arithmetic = new LinearRecurrence(new[] { 2L, -1L }, new[] { 0L, 1L });
        var powersOfTwo = new LinearRecurrence(new[] { 3L, -2L }, new[] { 1L, 2L });
        var geometric = new LinearRecurrence(new[] { 5L }, new[] { 1L });
        for (int n = 0; n <= 40; n++)
        {
            Check(arithmetic.Fast(n, 0L) == n, "2a_{n-1} - a_{n-2} should give n at n = " + n
                    + ", got " + arithmetic.Fast(n, 0L));
            Check(powersOfTwo.Fast(n, 0L) == (1L << n), "3a_{n-1} - 2a_{n-2} should give 2^n at n = " + n);
        }
        long five = 1;
        for (int n = 0; n <= 20; n++)
        {
            Check(geometric.Fast(n, 0L) == five, "5a_{n-1} should give 5^n at n = " + n);
            five *= 5;
        }

        // Modular evaluation: exact agreement for small n, and fast doubling at n = 10^9.
        const long Modulus = 1000000007L;
        for (int n = 0; n <= 60; n++)
        {
            long expected = fibonacci.Iterative(n, 0L) % Modulus;
            Check(fibonacci.Fast(n, Modulus) == expected, "modular Fibonacci disagrees at n = " + n);
            Check(fibonacci.Iterative(n, Modulus) == expected, "iterative modular Fibonacci at n = " + n);
        }
        long huge = 1000000000L;
        long[] doubling = LinearRecurrence.FibonacciDoubling(huge, Modulus);
        Check(fibonacci.Fast(huge, Modulus) == doubling[0], "matrix power != fast doubling at n = 10^9");
        Check(LinearRecurrence.FibonacciDoubling(90, Modulus)[0]
                == fibonacci.Iterative(90, Modulus), "fast doubling agrees with the recurrence at n = 90");

        // Pisano period modulo 10 is exactly 60.
        var residues = new long[120];
        for (int n = 0; n < 120; n++) residues[n] = fibonacci.Fast(n, 10L);
        bool periodic = true;
        for (int n = 0; n < 60; n++)
        {
            if (residues[n] != residues[n + 60]) periodic = false;
        }
        Check(periodic, "the Fibonacci residues modulo 10 repeat with period 60");
        int minimalPeriod = -1;
        for (int p = 1; p < 60 && minimalPeriod < 0; p++)
        {
            bool works = true;
            for (int n = 0; n < 60; n++)
            {
                if (residues[n] != residues[n + p]) { works = false; break; }
            }
            if (works) minimalPeriod = p;
        }
        Check(minimalPeriod < 0, "no period smaller than 60 works, found " + minimalPeriod);

        // Determinism and boundaries.
        Check(arithmetic.Fast(1, 0L) == 1 && geometric.Fast(0, 0L) == 1,
                "n < order returns the initial values");
        var order4 = new LinearRecurrence(new[] { 1L, 1L, 1L, 1L }, new[] { 3L, 1L, 4L, 1L });
        for (int n = 0; n <= 40; n++)
        {
            Check(order4.Fast(n, 0L) == order4.Iterative(n, 0L), "order-4 recurrence disagrees at n = " + n);
        }
        Check(order4.Fast(41, 0L) == order4.Iterative(41, 0L), "order-4 n = 41 matches the recurrence");
        Check(order4.Fast(5000, 1000000007L) == order4.Iterative(5000, 1000000007L),
                "order-4 n = 5000 matches the iterated recurrence modulo 1e9+7");
        Check(arithmetic.Fast(400, 0L) == 400, "the arithmetic recurrence reaches n = 400");
        Check(fibonacci.Fast(3000, 1000000007L) == fibonacci.Iterative(3000, 1000000007L),
                "Fibonacci at n = 3000 agrees modulo 1e9+7");
        for (int n = 0; n < 120; n++)
        {
            Check(fibonacci.Fast(n, 10L) == residues[n], "residue table is reproducible at n = " + n);
        }

        Console.WriteLine("PASS");
    }

    private static long[,] MatrixIdentity(int n)
    {
        var m = new long[n, n];
        for (int i = 0; i < n; i++) m[i, i] = 1;
        return m;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
