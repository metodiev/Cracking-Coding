/** Task 382 - Miller-Rabin primality test with deterministic small bases. (Tier: Intermediate, CTCI Ch 6) */
import java.math.BigInteger;

public class Solution {
    static final long[] DETERMINISTIC_BASES = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };

    private static long powMod(long base, long exponent, long modulus) {
        return BigInteger.valueOf(base % modulus)
                .modPow(BigInteger.valueOf(exponent), BigInteger.valueOf(modulus))
                .longValue();
    }

    /** One-base strong test: true when the base fails to prove n composite. */
    static boolean isStrongProbablePrime(long n, long base) {
        if (n < 2) return false;
        long d = n - 1;
        int s = 0;
        while ((d & 1) == 0) { d >>= 1; s++; }
        long x = powMod(base, d, n);
        if (x == 1 || x == n - 1) return true;
        for (int r = 1; r < s; r++) {
            x = (long) (BigInteger.valueOf(x).multiply(BigInteger.valueOf(x))
                    .mod(BigInteger.valueOf(n)).longValue());
            if (x == n - 1) return true;
        }
        return false;
    }

    /** Deterministic Miller-Rabin for n < 3.3e24 using the twelve fixed bases. */
    static boolean isPrime(long n) {
        if (n < 2) return false;
        for (long p : DETERMINISTIC_BASES)
            if (n % p == 0) return n == p;
        for (long a : DETERMINISTIC_BASES)
            if (!isStrongProbablePrime(n, a)) return false;
        return true;
    }

    /** Reference oracle for the cross-check; O(sqrt(n)). */
    static boolean isPrimeTrialDivision(long n) {
        if (n < 2) return false;
        for (long i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    public static void main(String[] args) {
        for (long p : DETERMINISTIC_BASES)
            check(isPrime(p), "base is prime " + p);

        long[] knownPrimes = { 41, 97, 541, 7919, 104729, 1299709, 2147483647L,
                2305843009213693951L, 67280421310721L };
        for (long p : knownPrimes) check(isPrime(p), "known prime " + p);

        long[] knownComposites = { 0, 1, 4, 9, 15, 21, 25, 49, 91, 121, 341, 561, 1105, 1729,
                4294967297L };
        for (long c : knownComposites) check(!isPrime(c), "known composite " + c);
        check(11 * 31 == 341, "341 = 11*31");
        check(3 * 11 * 17 == 561, "561 = 3*11*17");
        check(5 * 13 * 17 == 1105, "1105 = 5*13*17");
        check(7 * 13 * 19 == 1729, "1729 = 7*13*19");
        check(641L * 6700417L == 4294967297L, "2^32+1 = 641*6700417");

        // 2047 = 23*89: the base-2 test alone would call it prime; extra bases expose it
        check(23 * 89 == 2047, "2047 = 23*89");
        check(isStrongProbablePrime(2047, 2), "2047 passes base 2");
        check(!isStrongProbablePrime(2047, 3), "2047 fails base 3");
        check(!isPrime(2047), "2047 composite");

        // 3215031751 = 151*751*28351 passes the first four bases but not the whole set
        check(151L * 751L * 28351L == 3215031751L, "3215031751 = 151*751*28351");
        for (long b : new long[] { 2, 3, 5, 7 })
            check(isStrongProbablePrime(3215031751L, b), "3215031751 passes base " + b);
        check(!isStrongProbablePrime(3215031751L, 11), "3215031751 fails base 11");
        check(!isPrime(3215031751L), "3215031751 composite");

        // cross-check against trial division for every n <= 10000
        int primeCount = 0;
        for (int n = 0; n <= 10000; n++) {
            check(isPrime(n) == isPrimeTrialDivision(n), "trial division agrees at " + n);
            if (n > 1 && isPrime(n)) primeCount++;
        }
        check(primeCount == 1229, "pi(10000) = 1229");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
