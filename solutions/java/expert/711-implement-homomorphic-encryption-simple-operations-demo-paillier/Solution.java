/** Task 711 - Implement homomorphic encryption simple operations demo (Paillier).
 * (Tier: Expert, CTCI Beyond -- Security & Privacy)
 *
 * Paillier's scheme is the standard additive homomorphism: with n = p q and g = n + 1 the public key is
 * (n, g) and the secret key is lambda = lcm(p - 1, q - 1) together with mu = (L(g^lambda mod n^2))^-1
 * mod n, where L(x) = (x - 1) / n. Encryption of m in [0, n) with randomness r coprime to n is
 *
 *     c = (1 + m n) r^n mod n^2 ,     decryption m = L(c^lambda mod n^2) mu mod n ,
 *
 * and the two properties demonstrated here follow directly: multiplying ciphertexts adds the
 * plaintexts (c1 c2 decrypts to m1 + m2 mod n) and raising a ciphertext to a power multiplies the
 * plaintext (c^k decrypts to k m mod n). Multiplying by a fresh r'^n re-randomises the ciphertext
 * without changing its plaintext, which is what makes the scheme semantically secure.
 *
 * Simplified: a *demo* key of about 62 bits (n^2 fits in a signed long), because the whole point here
 * is the algebra rather than cryptographic strength; a real key needs thousands of bits and hence
 * arbitrary-precision arithmetic, so no security is claimed. All modular arithmetic uses a
 * double-and-add mulmod that is exact for moduli below 2^62, and the primes are found deterministically
 * by trial division, so no BigInteger is needed.
 *
 * Verified: decryption round trips for boundary plaintexts (0 and n - 1) and random ones, the additive
 * homomorphism on many pairs - including the modular wrap-around when the sum exceeds n - scalar
 * multiplication, re-randomisation (same plaintext, different ciphertext), summing k ciphertexts, the
 * key invariants n = p q, gcd(n, lambda) = 1 and mu L(g^lambda) = 1 mod n, the negative control that a
 * wrong lambda decrypts incorrectly, and rejection of plaintexts outside [0, n).
 */
import java.util.*;

class PaillierDemo {
    final long p;
    final long q;
    final long n;
    final long nSquared;
    final long g;
    final long lambda;
    final long mu;
    private long state;

    PaillierDemo(long seed) {
        long first = 40001;
        long second = 0;
        while (true) {
            first = nextPrime(first);
            second = nextPrime(first + 2);
            long candidate = first * second;
            long phiLike = lcm(first - 1, second - 1);
            if (candidate < (1L << 31) && gcd(candidate, phiLike) == 1) break;
            first += 2;
        }
        p = first;
        q = second;
        n = p * q;
        nSquared = n * n;
        if (nSquared >= (1L << 62) || nSquared <= 0) {
            throw new IllegalStateException("demo modulus too large for exact long arithmetic");
        }
        g = n + 1;
        lambda = lcm(p - 1, q - 1);
        long residue = modPow(g, lambda, nSquared);
        long l = (residue - 1) / n;
        mu = modInverse(l, n);
        state = seed;
    }

    long modulus() {
        return n;
    }

    long modulusSquared() {
        return nSquared;
    }

    /** c = (1 + m n) r^n mod n^2. */
    long encrypt(long message) {
        if (message < 0 || message >= n) throw new IllegalArgumentException("plaintext out of range");
        long r = coprimeRandom();
        long factor = modPow(r, n, nSquared);
        return mulmod((1 + message * n) % nSquared, factor, nSquared);
    }

    long decrypt(long ciphertext) {
        return decryptWith(ciphertext, lambda);
    }

    /** Decryption with an explicit exponent, so the checks can show a wrong key failing. */
    long decryptWith(long ciphertext, long exponent) {
        if (ciphertext <= 0 || ciphertext >= nSquared) {
            throw new IllegalArgumentException("ciphertext out of range");
        }
        long residue = modPow(ciphertext, exponent, nSquared);
        return mulmod((residue - 1) / n, mu, n);
    }

    /** Homomorphic addition: E(a) E(b) decrypts to a + b mod n. */
    long add(long first, long second) {
        return mulmod(first, second, nSquared);
    }

    /** Homomorphic scalar multiplication: E(a)^k decrypts to k a mod n. */
    long scale(long ciphertext, long scalar) {
        if (scalar < 0) throw new IllegalArgumentException("scalar must be non-negative");
        return modPow(ciphertext, scalar, nSquared);
    }

    /** Re-randomisation: multiplies by a fresh r^n, leaving the plaintext untouched. */
    long rerandomize(long ciphertext) {
        long r = coprimeRandom();
        return mulmod(ciphertext, modPow(r, n, nSquared), nSquared);
    }

    private long coprimeRandom() {
        while (true) {
            long candidate = 2 + Math.floorMod(nextLong(), n - 2);
            if (gcd(candidate, n) == 1) return candidate;
        }
    }

    private long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    static long lcm(long a, long b) {
        return a / gcd(a, b) * b;
    }

    static boolean isPrime(long value) {
        if (value < 2) return false;
        for (long divisor = 2; divisor * divisor <= value; divisor++) {
            if (value % divisor == 0) return false;
        }
        return true;
    }

    static long nextPrime(long from) {
        long candidate = from % 2 == 0 ? from + 1 : from;
        while (!isPrime(candidate)) candidate += 2;
        return candidate;
    }

    /** Exact for m < 2^62: doubling avoids every 128-bit intermediate. */
    static long mulmod(long a, long b, long m) {
        long result = 0;
        long left = a % m;
        long right = b % m;
        while (right > 0) {
            if ((right & 1) != 0) result = (result + left) % m;
            left = (left << 1) % m;
            right >>= 1;
        }
        return result;
    }

    static long modPow(long base, long exponent, long m) {
        long result = 1 % m;
        long current = base % m;
        long power = exponent;
        while (power > 0) {
            if ((power & 1) != 0) result = mulmod(result, current, m);
            current = mulmod(current, current, m);
            power >>= 1;
        }
        return result;
    }

    static long modInverse(long value, long m) {
        long oldR = value % m;
        long r = m;
        long oldS = 1;
        long s = 0;
        while (r != 0) {
            long quotient = oldR / r;
            long tmp = oldR - quotient * r;
            oldR = r;
            r = tmp;
            tmp = oldS - quotient * s;
            oldS = s;
            s = tmp;
        }
        if (oldR != 1) throw new IllegalArgumentException("not invertible: " + value);
        return ((oldS % m) + m) % m;
    }
}

public class Solution {
    public static void main(String[] args) {
        PaillierDemo paillier = new PaillierDemo(0x9E3779B9L);
        long n = paillier.modulus();

        // --- the key itself -----------------------------------------------------
        check(paillier.n == paillier.p * paillier.q, "n must be the product of the two primes");
        check(PaillierDemo.isPrime(paillier.p) && PaillierDemo.isPrime(paillier.q),
                "the key factors must be prime");
        check(PaillierDemo.gcd(n, paillier.lambda) == 1, "gcd(n, lambda) must be one");
        check(paillier.p != paillier.q, "the two primes must differ");
        check(paillier.nSquared < (1L << 62), "the demo modulus must fit exact long arithmetic");
        check(PaillierDemo.mulmod(paillier.modulusSquared() - 1, paillier.modulusSquared() - 1,
                        paillier.modulusSquared())
                        == PaillierDemo.modPow(paillier.modulusSquared() - 1, 2, paillier.modulusSquared()),
                "mulmod agrees with modPow");
        long luCheck = PaillierDemo.mulmod(
                (PaillierDemo.modPow(paillier.g, paillier.lambda, paillier.modulusSquared()) - 1) / n,
                paillier.mu, n);
        check(luCheck == 1, "mu must invert L(g^lambda): " + luCheck);

        // --- boundary plaintexts and round trips --------------------------------
        check(paillier.decrypt(paillier.encrypt(0)) == 0, "zero must survive a round trip");
        check(paillier.decrypt(paillier.encrypt(n - 1)) == n - 1, "n - 1 must survive a round trip");
        check(paillier.decrypt(paillier.encrypt(1)) == 1, "one must survive a round trip");
        for (int trial = 0; trial < 25; trial++) {
            long message = Math.floorMod(trial * 104729L + 12345L, n);
            check(paillier.decrypt(paillier.encrypt(message)) == message,
                    "round trip failed for " + message);
        }

        // --- additive homomorphism ---------------------------------------------
        long a = 1234567;
        long b = n - 500;                       // forces the modular wrap-around when added to a
        long cipherA = paillier.encrypt(a);
        long cipherB = paillier.encrypt(b);
        check(paillier.decrypt(paillier.add(cipherA, cipherB)) == (a + b) % n,
                "E(a) E(b) must decrypt to a + b mod n");
        check(paillier.decrypt(cipherA) == a && paillier.decrypt(cipherB) == b,
                "ciphertexts are untouched by the homomorphic operation");
        for (int trial = 0; trial < 12; trial++) {
            long first = Math.floorMod(trial * 2654435761L, n);
            long second = Math.floorMod(trial * 40503L + 17, n);
            long sum = paillier.decrypt(paillier.add(paillier.encrypt(first), paillier.encrypt(second)));
            check(sum == (first + second) % n, "homomorphic sum failed at trial " + trial);
        }

        // --- scalar multiplication ---------------------------------------------
        check(paillier.decrypt(paillier.scale(cipherA, 5)) == (5 * a) % n, "scalar multiplication");
        check(paillier.decrypt(paillier.scale(cipherA, 0)) == 0, "multiplying by zero encrypts zero");
        check(paillier.decrypt(paillier.scale(cipherA, 1)) == a, "multiplying by one is a no-op");
        check(paillier.decrypt(paillier.scale(cipherA, n - 1)) == (n - a) % n,
                "multiplying by n - 1 negates modulo n");

        // --- re-randomisation ---------------------------------------------------
        int distinct = 0;
        for (int trial = 0; trial < 8; trial++) {
            long fresh = paillier.rerandomize(cipherA);
            check(paillier.decrypt(fresh) == a, "re-randomisation must keep the plaintext");
            if (fresh != cipherA) distinct++;
        }
        check(distinct >= 6, "re-randomisation must change the ciphertext: " + distinct + "/8");

        // --- summing many ciphertexts (the private-aggregation use case) --------
        long expectedSum = 0;
        long running = paillier.encrypt(0);
        for (int i = 1; i <= 20; i++) {
            expectedSum = (expectedSum + i * 1000L) % n;
            running = paillier.add(running, paillier.encrypt(i * 1000L));
        }
        check(paillier.decrypt(running) == expectedSum,
                "summing twenty ciphertexts must decrypt to the sum of the plaintexts");

        // --- negative control: a wrong key does not decrypt ---------------------
        int wrongKeyFailures = 0;
        for (int trial = 0; trial < 5; trial++) {
            long message = 1000 + trial;
            long ciphertext = paillier.encrypt(message);
            long wrong = paillier.decryptWith(ciphertext, paillier.lambda + 1);
            if (wrong != message) wrongKeyFailures++;
        }
        check(wrongKeyFailures >= 4, "the wrong lambda must fail: " + wrongKeyFailures + "/5");

        // --- rejected input ------------------------------------------------------
        check(rejects(() -> paillier.encrypt(n)), "plaintext n rejected");
        check(rejects(() -> paillier.encrypt(-1)), "negative plaintext rejected");
        check(rejects(() -> paillier.decrypt(0)), "ciphertext zero rejected");
        check(rejects(() -> paillier.decrypt(paillier.modulusSquared())), "ciphertext n^2 rejected");
        check(rejects(() -> paillier.scale(cipherA, -3)), "negative scalar rejected");
        check(rejects(() -> PaillierDemo.modInverse(4, 8)), "non-invertible value rejected");

        // --- determinism ----------------------------------------------------------
        PaillierDemo rebuilt = new PaillierDemo(0x9E3779B9L);
        check(rebuilt.n == paillier.n && rebuilt.g == paillier.g && rebuilt.mu == paillier.mu,
                "the same seed must derive the same key");
        check(new PaillierDemo(4242L).encrypt(4242) == new PaillierDemo(4242L).encrypt(4242),
                "the same seed must derive the same ciphertext stream");

        System.out.println("demo key: p = " + paillier.p + ", q = " + paillier.q + ", n = " + n
                + " (" + Long.toBinaryString(n).length() + " bits), n^2 = " + paillier.modulusSquared());
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
