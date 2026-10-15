/** Task 380 - Modular inverse (extended Euclid and Fermat) + fast modular exponentiation. (Tier: Intermediate, CTCI Ch 6) */

/**
 * Two independent inverse implementations:
 *   * extended Euclid: g = gcd(a, m) and Bezout coefficients x, y with a*x + m*y = g;
 *     when g == 1, x mod m is the inverse.  Works for ANY modulus m;
 *   * Fermat: for PRIME p and a not divisible by p, inv = a^(p-2) mod p.
 * Both must agree on every coprime pair, and a * inv(a, m) % m == 1 on LCG values.
 *
 * Documented failure: when gcd(a, m) != 1 no inverse exists; the API returns null
 * (the documented sentinel) instead of silently returning garbage.  The m = 1 case is
 * documented as out of scope (every residue is 0 mod 1) and rejected.
 *
 * Fast modular exponentiation is binary (square-and-multiply) with the usual contract
 * powMod(a, 0, m) = 1 mod m; hand-checked: 2^10 = 1024, 3^0 = 1, cross-checked against
 * a naive O(e) loop for small exponents, checked against the identity
 * pow(a, x+y) = pow(a, x) * pow(a, y) mod m, and Fermat's little theorem a^(p-1) = 1
 * mod p is asserted for several primes.
 *
 * All arithmetic is exact long arithmetic (products stay below 1.1e18); no floats.
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

public class Solution {
    /** Fast binary modular exponentiation; exponents >= 0 only (documented). */
    static long powMod(long base, long exponent, long modulus) {
        if (exponent < 0) throw new IllegalArgumentException("negative exponents unsupported");
        if (modulus <= 0) throw new IllegalArgumentException("modulus must be positive");
        long result = 1 % modulus;
        base %= modulus;
        while (exponent > 0) {
            if ((exponent & 1) == 1) result = result * base % modulus;
            base = base * base % modulus;
            exponent >>= 1;
        }
        return result;
    }

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /** Iterative extended Euclid: returns {g, x, y} with a*x + b*y = g >= 0. */
    static long[] extendedGcd(long a, long b) {
        long oldR = a, r = b;
        long oldS = 1, s = 0;
        long oldT = 0, t = 1;
        while (r != 0) {
            long q = oldR / r;
            long temp = oldR - q * r;
            oldR = r;
            r = temp;
            temp = oldS - q * s;
            oldS = s;
            s = temp;
            temp = oldT - q * t;
            oldT = t;
            t = temp;
        }
        if (oldR < 0) {
            oldR = -oldR;
            oldS = -oldS;
            oldT = -oldT;
        }
        return new long[]{oldR, oldS, oldT};
    }

    /** Inverse of a modulo m via extended Euclid, or null when gcd(a, m) != 1. */
    static Long modInverseEuclid(long a, long m) {
        if (m <= 1) throw new IllegalArgumentException("m >= 2 required (m = 1 is out of scope)");
        long value = ((a % m) + m) % m;
        long[] egcd = extendedGcd(value, m);
        if (egcd[0] != 1) return null;
        return ((egcd[1] % m) + m) % m;
    }

    /** Inverse of a modulo the PRIME p via a^(p-2) mod p; null when p divides a. */
    static Long modInverseFermat(long a, long p) {
        if (p <= 1) throw new IllegalArgumentException("p > 1 required");
        long value = ((a % p) + p) % p;
        if (value == 0) return null;
        return powMod(value, p - 2, p);
    }

    /** Independent O(exponent) oracle for small exponents. */
    static long powNaive(long base, long exponent, long modulus) {
        long result = 1 % modulus;
        for (long i = 0; i < exponent; i++) result = result * (base % modulus) % modulus;
        return result;
    }

    public static void main(String[] args) {
        // --- hand-checked small cases -----------------------------------------
        check(modInverseEuclid(3, 11) == 4L, "inv(3, 11) = 4");
        check(modInverseEuclid(1, 11) == 1L, "inv(1, 11) = 1");
        check(modInverseEuclid(1, 7) == 1L, "inv(1, 7) = 1");
        check(modInverseEuclid(1, 1000000000L) == 1L, "inv(1, 1e9) = 1");
        check(modInverseEuclid(2, 5) == 3L, "inv(2, 5) = 3");
        check(modInverseEuclid(10, 17) == 12L, "inv(10, 17) = 12");
        check(modInverseEuclid(4, 8) == null, "gcd(4, 8) != 1 -> null");
        check(modInverseEuclid(6, 9) == null, "gcd(6, 9) != 1 -> null");
        check(modInverseEuclid(0, 5) == null, "gcd(0, 5) != 1 -> null");

        check(modInverseFermat(3, 11) == 4L, "fermat inv(3, 11) = 4");
        check(modInverseFermat(5, 101) == 81L, "fermat inv(5, 101) = 81");
        check(modInverseFermat(11, 11) == null, "p divides a -> null");

        // --- fast exponentiation, hand-checked ---------------------------------
        check(powMod(2, 10, 1000000007L) == 1024L, "2^10 = 1024");
        check(powMod(3, 0, 1000000007L) == 1L, "3^0 = 1");
        check(powMod(0, 0, 7) == 1L, "0^0 = 1 mod 7");
        check(powMod(7, 1, 5) == 2L, "7 mod 5");
        check(powMod(5, 3, 13) == 8L, "5^3 mod 13");
        check(powMod(2, 1000, 1000000007L) == 688423210L, "2^1000 mod p (recomputed)");
        check(powMod(3, 200, 1000000007L) == 136318165L, "3^200 mod p (recomputed)");

        // binary vs naive loop for small exponents
        Lcg lcg = new Lcg(38013);
        for (int trial = 0; trial < 60; trial++) {
            long base = lcg.next(1000);
            long exponent = lcg.next(40);
            long modulus = lcg.next(999) + 2;
            check(powMod(base, exponent, modulus) == powNaive(base, exponent, modulus),
                    "binary vs naive " + trial);
        }

        // pow(a, x + y) = pow(a, x) * pow(a, y) mod m
        for (int trial = 0; trial < 40; trial++) {
            long base = lcg.next(1000000006) + 1L;
            long x = lcg.next(60);
            long y = lcg.next(60);
            long modulus = 1000000007L;
            check(powMod(base, x + y, modulus)
                    == powMod(base, x, modulus) * powMod(base, y, modulus) % modulus,
                    "pow additive identity " + trial);
        }

        // --- extended Euclid identity a*inv = 1 (mod m) on LCG values ----------
        for (int trial = 0; trial < 60; trial++) {
            long m = lcg.next(998) + 2L;
            long a = lcg.next(1000);
            Long inverse = modInverseEuclid(a, m);
            if (gcd(a, m) == 1) {
                check(inverse != null, "coprime pair has inverse " + a + " " + m);
                check(a * inverse % m == 1, "a * inv = 1 mod m " + a + " " + m);
            } else {
                check(inverse == null, "non-coprime pair has no inverse " + a + " " + m);
            }
        }

        // --- Euclid and Fermat agree for primes --------------------------------
        long[] primes = {7, 11, 13, 101, 65537, 1000000007L};
        for (long prime : primes) {
            for (int trial = 0; trial < 12; trial++) {
                long a = lcg.next((int) Math.min(prime - 1, 65536)) + 1;
                if (a % prime == 0) continue;
                Long euclid = modInverseEuclid(a, prime);
                Long fermat = modInverseFermat(a, prime);
                check(euclid != null && euclid.equals(fermat), "euclid == fermat " + a + " " + prime);
                check(a * euclid % prime == 1, "inverse verifies " + a + " " + prime);
            }
        }

        // Fermat's little theorem: a^(p-1) = 1 mod p for p prime, p not dividing a
        for (long prime : primes) {
            for (long a : new long[]{1, 2, 3, prime - 1, prime - 2}) {
                if (a % prime == 0) continue;
                check(powMod(a, prime - 1, prime) == 1, "fermat little theorem " + a + " " + prime);
            }
        }

        // Wilson-flavoured sanity for a small prime: (19-1)! = -1 mod 19
        long factorial = 1;
        for (int i = 1; i <= 18; i++) factorial = factorial * i % 19;
        check(factorial == 18, "Wilson 18! = -1 mod 19");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
