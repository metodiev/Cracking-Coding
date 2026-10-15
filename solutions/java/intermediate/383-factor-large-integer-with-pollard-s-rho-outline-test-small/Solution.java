/** Task 383 - Factor a large integer with Pollard's rho (outline / small tests). (Tier: Intermediate, CTCI Ch 6) */
import java.math.BigInteger;
import java.util.*;

public class Solution {
    static final long[] MILLER_RABIN_BASES = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /** Deterministic Miller-Rabin (same twelve-base set as task 382). */
    static boolean isPrime(long n) {
        if (n < 2) return false;
        for (long p : MILLER_RABIN_BASES)
            if (n % p == 0) return n == p;
        long d = n - 1;
        int s = 0;
        while ((d & 1) == 0) { d >>= 1; s++; }
        for (long a : MILLER_RABIN_BASES) {
            long x = BigInteger.valueOf(a)
                    .modPow(BigInteger.valueOf(d), BigInteger.valueOf(n)).longValue();
            if (x == 1 || x == n - 1) continue;
            boolean witnessed = true;
            for (int r = 1; r < s; r++) {
                x = BigInteger.valueOf(x).multiply(BigInteger.valueOf(x))
                        .mod(BigInteger.valueOf(n)).longValue();
                if (x == n - 1) { witnessed = false; break; }
            }
            if (witnessed) return false;
        }
        return true;
    }

    private static long f(long x, long c, long n) {
        return BigInteger.valueOf(x).multiply(BigInteger.valueOf(x))
                .add(BigInteger.valueOf(c)).mod(BigInteger.valueOf(n)).longValue();
    }

    /** Floyd cycle-finding variant; retries with the next c when the gcd is n. */
    static long pollardRho(long n) {
        if (n % 2 == 0) return 2;
        if (n % 3 == 0) return 3;
        for (long c = 1; c < 100; c++) {
            long x = 2, y = 2, d = 1;
            while (d == 1) {
                x = f(x, c, n);
                y = f(f(y, c, n), c, n);
                d = gcd(Math.abs(x - y), n);
            }
            if (d != n) return d;
        }
        throw new IllegalStateException("Pollard's rho failed to find a factor");
    }

    private static void split(long value, List<Long> out) {
        if (value == 1) return;
        if (isPrime(value)) {
            out.add(value);
            return;
        }
        long factor = pollardRho(value);
        split(factor, out);
        split(value / factor, out);
    }

    /** Complete prime factorisation in ascending order ([] for 1). */
    static long[] factorize(long n) {
        List<Long> out = new ArrayList<>();
        split(n, out);
        Collections.sort(out);
        long[] result = new long[out.size()];
        for (int i = 0; i < out.size(); i++) result[i] = out.get(i);
        return result;
    }

    private static void checkEquals(long[] actual, long[] expected, String label) {
        check(Arrays.equals(actual, expected), label);
    }

    public static void main(String[] args) {
        // hand-checked factorisations
        checkEquals(factorize(8051), new long[] { 83, 97 }, "8051 = 83*97");
        checkEquals(factorize(10403), new long[] { 101, 103 }, "10403 = 101*103");
        checkEquals(factorize(4294967297L), new long[] { 641, 6700417 }, "2^32+1");
        checkEquals(factorize(123456789), new long[] { 3, 3, 3607, 3803 }, "123456789");
        checkEquals(factorize(600851475143L), new long[] { 71, 839, 1471, 6857 }, "600851475143");
        checkEquals(factorize(1099511627777L), new long[] { 257, 4278255361L }, "2^40+1");
        check(83 * 97 == 8051 && 101 * 103 == 10403, "small products");
        check(641L * 6700417 == 4294967297L, "641*6700417");
        check(9 * 13717421 == 123456789 && 3607 * 3803 == 13717421, "123456789 factors");
        check(71L * 839 * 1471 * 6857 == 600851475143L, "Project Euler 3 factors");
        check(257L * 4278255361L == 1099511627777L, "257*4278255361");

        // primes and 1 are handled without recursion
        check(factorize(1).length == 0, "factorize(1)");
        checkEquals(factorize(2), new long[] { 2 }, "factorize(2)");
        checkEquals(factorize(97), new long[] { 97 }, "factorize(97)");
        checkEquals(factorize(97 * 97), new long[] { 97, 97 }, "factorize(97^2)");

        // cross-check: product of factors == n, every factor prime, list ascending
        long[] samples = { 8051, 10403, 4294967297L, 123456789, 600851475143L, 1099511627777L,
                999983, 1, 4, 51, 65536, 1048577 };
        for (long n : samples) {
            long[] factors = factorize(n);
            long product = 1;
            for (long factor : factors) {
                check(isPrime(factor), "factor prime " + n + "/" + factor);
                product *= factor;
            }
            check(product == n, "product " + n);
            for (int i = 1; i < factors.length; i++)
                check(factors[i - 1] <= factors[i], "ascending " + n);
        }
        checkEquals(factorize(999983), new long[] { 999983 }, "999983 is prime");
        long[] powers = factorize(65536);
        check(powers.length == 16, "65536 has 16 factors");
        for (long p : powers) check(p == 2, "factor of 65536 is 2");
        checkEquals(factorize(1048577), new long[] { 17, 61681 }, "2^20+1 = 17*61681");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
