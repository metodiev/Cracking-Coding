/** Task 384 - Implement discrete logarithm baby-step giant-step. (Tier: Intermediate, CTCI Ch 6) */
import java.util.*;

public class Solution {
    static long modPower(long base, long exponent, long modulus) {
        long result = 1 % modulus;
        long b = ((base % modulus) + modulus) % modulus;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = result * b % modulus;
            b = b * b % modulus;
            e >>= 1;
        }
        return result;
    }

    static long[] extendedGcd(long a, long b) {
        if (b == 0) return new long[] { a, 1, 0 };
        long[] prev = extendedGcd(b, a % b);
        return new long[] { prev[0], prev[2], prev[1] - (a / b) * prev[2] };
    }

    static long modInverse(long a, long modulus) {
        long[] eg = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
        if (eg[0] != 1) throw new IllegalArgumentException("modular inverse does not exist");
        return ((eg[1] % modulus) + modulus) % modulus;
    }

    /**
     * Smallest x >= 0 with g^x = h (mod p); -1 when no such x exists.
     * Baby steps list g^j -> j for j in [0, m), m = ceil(sqrt(p)); giant steps walk
     * h * (g^(-m))^i. The first hit with the smallest i and j is the minimal solution.
     */
    static long discreteLog(long g, long h, long p) {
        if (p <= 1) throw new IllegalArgumentException("modulus must exceed 1");
        long target = ((h % p) + p) % p;
        long m = (long) Math.sqrt(p - 1);
        while (m * m < p - 1) m++;
        Map<Long, Long> baby = new HashMap<>();
        long value = 1 % p;
        for (long j = 0; j < m; j++) {
            if (!baby.containsKey(value)) baby.put(value, j);
            value = value * g % p;
        }
        long giantFactor = modPower(modInverse(g, p), m, p);
        long gamma = target;
        for (long i = 0; i <= m; i++) {
            Long j = baby.get(gamma);
            if (j != null) return i * m + j;
            gamma = gamma * giantFactor % p;
        }
        return -1;
    }

    /** Reference oracle: minimal x in [0, p), or -1; O(p). */
    static long bruteForceDiscreteLog(long g, long h, long p) {
        long target = ((h % p) + p) % p;
        long value = 1 % p;
        for (long x = 0; x < p; x++) {
            if (value == target) return x;
            value = value * g % p;
        }
        return -1;
    }

    private static int seed = 9871;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        // hand-verified small problems
        check(discreteLog(2, 8, 13) == 3 && modPower(2, 3, 13) == 8, "2^x = 8 (13)");
        check(discreteLog(3, 13, 17) == 4 && modPower(3, 4, 17) == 13, "3^x = 13 (17)");
        check(discreteLog(2, 10, 19) == 17 && modPower(2, 17, 19) == 10, "2^x = 10 (19)");
        check(discreteLog(2, 5, 11) == 4 && modPower(2, 4, 11) == 5, "2^x = 5 (11)");
        check(discreteLog(2, 1, 13) == 0, "g^0 = 1 always");

        // documented failure sentinel: no solution when h is outside the subgroup <g>
        check(discreteLog(2, 3, 7) == -1, "<2> = {1,2,4} (7) misses 3");
        check(bruteForceDiscreteLog(2, 3, 7) == -1, "brute force agrees (2,3,7)");
        check(discreteLog(3, 2, 11) == -1, "<3> = {1,3,4,5,9} (11) misses 2");
        check(bruteForceDiscreteLog(3, 2, 11) == -1, "brute force agrees (3,2,11)");
        check(discreteLog(1, 2, 7) == -1, "powers of 1 are all 1");
        check(discreteLog(2, 0, 7) == -1, "0 is never a power of g");

        // cross-check against brute force for small primes
        long[] primes = { 13, 17, 19, 97, 101, 103, 211, 1009 };
        long[] targets = { 1, 2, 3, 5, 10, 42, 123 };
        for (long p : primes) {
            for (long g = 2; g < Math.min(p, 6); g++) {
                for (long h : targets) {
                    long expected = bruteForceDiscreteLog(g, h, p);
                    check(discreteLog(g, h, p) == expected, "cross-check " + g + "," + h + "," + p);
                }
            }
        }

        // LCG triples: h = g^x is solvable and the answer must satisfy g^answer = h
        for (int trial = 0; trial < 60; trial++) {
            long p = primes[nextSeed() % primes.length];
            long g = 2 + nextSeed() % (p - 2);
            long x = nextSeed() % p;
            long h = modPower(g, x, p);
            long found = discreteLog(g, h, p);
            check(found != -1 && modPower(g, found, p) == h, "LCG solvable " + trial);
            check(found == bruteForceDiscreteLog(g, h, p), "LCG minimal " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
