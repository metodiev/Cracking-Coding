/** Task 385 - Implement RSA key generation and basic encrypt/decrypt (demo). (Tier: Intermediate, CTCI Beyond — Security & Privacy) */
import java.math.BigInteger;
import java.util.*;

public class Solution {
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

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /** Trial division, used only to reject obviously bad demo primes. */
    static boolean isSmallPrime(long n) {
        if (n < 2) return false;
        for (long i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    static long modPower(long base, long exponent, long modulus) {
        return BigInteger.valueOf(base)
                .modPow(BigInteger.valueOf(exponent), BigInteger.valueOf(modulus)).longValue();
    }

    /** Returns {n, e, d, phi}; throws IllegalArgumentException on bad demo parameters. */
    static long[] keygen(long p, long q, long e) {
        if (!isSmallPrime(p) || !isSmallPrime(q) || p == q)
            throw new IllegalArgumentException("p and q must be distinct primes");
        long n = p * q;
        long phi = (p - 1) * (q - 1);
        if (gcd(e, phi) != 1)
            throw new IllegalArgumentException("public exponent must be coprime with phi(n)");
        return new long[] { n, e, modInverse(e, phi), phi };
    }

    static long encrypt(long message, long e, long n) {
        if (message < 0 || message >= n)
            throw new IllegalArgumentException("message must satisfy 0 <= m < n");
        return modPower(message, e, n);
    }

    static long decrypt(long ciphertext, long d, long n) {
        if (ciphertext < 0 || ciphertext >= n)
            throw new IllegalArgumentException("ciphertext must satisfy 0 <= c < n");
        return modPower(ciphertext, d, n);
    }

    public static void main(String[] args) {
        // classic small keypair p=61, q=53 (n = 3233, phi = 3120, e = 17, d = 2753)
        long[] keys = keygen(61, 53, 17);
        long n = keys[0], e = keys[1], d = keys[2], phi = keys[3];
        check(Arrays.equals(keys, new long[] { 3233, 17, 2753, 3120 }), "keygen 61,53,17");
        check(n == 61 * 53 && phi == 60 * 52, "n and phi");
        check(e * d % phi == 1, "e*d = 1 (mod phi)");
        check(d != e, "d differs from e");

        // round trip for several messages, including the boundaries 0 and n-1
        long[] messages = { 0, 1, 2, 42, 65, 1000, 3232 };
        for (long message : messages)
            check(decrypt(encrypt(message, e, n), d, n) == message, "round trip " + message);
        check(encrypt(0, e, n) == 0 && encrypt(1, e, n) == 1, "fixed points 0 and 1");
        check(encrypt(n - 1, e, n) == n - 1, "(-1)^e = -1 for odd e");
        for (long message : new long[] { 2, 42, 65, 1000 })
            check(encrypt(message, e, n) != message, "ciphertext differs " + message);

        // wrong key: applying the public exponent to the ciphertext does not recover the message
        long ciphertext42 = encrypt(42, e, n);
        check(ciphertext42 == 2557, "c(42) = 2557");
        check(decrypt(ciphertext42, e, n) == 1001 && decrypt(ciphertext42, e, n) != 42,
                "wrong key gives 1001, not 42");

        // second keypair with e = 3: p=101, q=113 -> n=11413, phi=11200, d=7467
        long[] keys2 = keygen(101, 113, 3);
        long n2 = keys2[0], e2 = keys2[1], d2 = keys2[2], phi2 = keys2[3];
        check(Arrays.equals(keys2, new long[] { 11413, 3, 7467, 11200 }), "keygen 101,113,3");
        check(e2 * d2 % phi2 == 1, "e*d = 1 (mod phi) second key");
        for (long message : new long[] { 0, 1, 2, 5, 42, 999, 11412 })
            check(decrypt(encrypt(message, e2, n2), d2, n2) == message, "round trip 2 " + message);
        long ciphertext7 = encrypt(7, e2, n2);
        check(decrypt(ciphertext7, e2, n2) == 8652 && decrypt(ciphertext7, e2, n2) != 7,
                "second key wrong key gives 8652");

        // keygen rejects bad demo parameters (documented errors)
        long[][] bad = { { 7, 13, 3 }, { 9, 53, 17 }, { 61, 61, 17 }, { 61, 53, 4 } };
        for (long[] params : bad) {
            boolean threw = false;
            try {
                keygen(params[0], params[1], params[2]);
            } catch (IllegalArgumentException ex) {
                threw = true;
            }
            check(threw, "bad keygen params rejected");
        }

        // encryption domain is checked
        for (long message : new long[] { 3233, -1 }) {
            boolean threw = false;
            try {
                encrypt(message, e, n);
            } catch (IllegalArgumentException ex) {
                threw = true;
            }
            check(threw, "message outside [0, n) rejected");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
