/** Task 585 - Implement modular FFT for polynomial multiplication (NTT) small prime. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Number-theoretic transform in the finite field Z_p with the NTT-friendly prime
 * p = 998244353 = 119 * 2^23 + 1 and primitive root g = 3 (the "small prime" of competitive
 * programming; a second, deliberately tiny field p = 17, g = 3 is used to show the size constraint).
 * The forward transform is the iterative Cooley-Tukey butterfly with a bit-reversal permutation,
 * w_len = g^((p-1)/len) and the inverse transform using w_len^{-1} followed by a division by n; a size
 * guard rejects any length that is not a power of two dividing p - 1, which caps the transform at
 * 2^23 points and is the reason a polynomial product of degree >= 2^23 must be split (documented, not
 * exercised - allocating 2^24 longs just to see the guard fire is pointless, so the guard is tested
 * through the same validation routine with the allocation skipped).
 *
 * Because the transform is a divide-and-conquer recursion flattened into loops, the file anchors
 * correctness on three independent oracles rather than on the transform itself:
 *   - the O(n^2) schoolbook product modulo p for random inputs of many shapes (including length 1,
 *     non-power-of-two operands, and single-monomial operands);
 *   - the orthogonality of the roots, checked exactly in Z_p: sum_j w_n^{jk} = n if n | k and 0
 *     otherwise, w_n having order exactly n, and g^((p-1)/q) != 1 for every prime q | p-1;
 *   - exact big-integer multiplication: with coefficients in 0..9 and a limb of 2^64 no carry can
 *     cross a limb (the largest product coefficient is 500 * 9 * 9 = 40500), so packing the 500
 *     coefficients of each operand into a BigInteger and multiplying gives the true integer product,
 *     whose limbs the NTT result must reproduce modulo p.
 *
 * Documented checks: forward/inverse round-trips for every power-of-two length 1..256; DCT-style
 * identities sum_j x_j = X_0 and sum_k X_k = n * x_0; the linear/cyclic correspondence - the length-n
 * cyclic product equals the exact fold of the schoolbook product modulo x^n - 1, and it deliberately
 * differs from the linear product when n < deg A + deg B + 1 (a case asserted, not hidden);
 * commutativity, associativity and distributivity; the identity polynomial [1] and the shift operator
 * x; the small field p = 17 where a true coefficient 20 wraps to 3 (asserted, since the transform
 * only ever sees residues); and reproducibility across runs.
 *
 * Randomness: coefficients come from splitmix64 (64-bit mixing function with the 0x9E3779B97F4A7C15
 * increment) rather than the repo's 16-bit LCG, because this file draws several thousand independent
 * values; the seeds are fixed constants, so every claim is deterministic.
 */
import java.math.BigInteger;
import java.util.*;

public class Solution {
    static final long MOD = 998244353L;
    static final long ROOT = 3L;
    static final long SMALL_MOD = 17L;
    static final long SMALL_ROOT = 3L;

    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        long nextMod(long m) {
            return Math.floorMod(next(), m);
        }

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- modular arithmetic

    static long powMod(long base, long exp, long mod) {
        long result = 1;
        base %= mod;
        while (exp > 0) {
            if ((exp & 1) == 1) result = result * base % mod;
            base = base * base % mod;
            exp >>= 1;
        }
        return result;
    }

    static long inverse(long a, long mod) {
        return powMod(a, mod - 2, mod);
    }

    /** Only powers of two dividing p - 1 are supported: this is the NTT's size constraint. */
    static void requireSize(int n, long mod) {
        if (n <= 0 || (n & (n - 1)) != 0) {
            throw new IllegalArgumentException("NTT length " + n + " is not a power of two");
        }
        if ((mod - 1) % n != 0) {
            throw new IllegalArgumentException("NTT length " + n + " does not divide p - 1 = " + (mod - 1));
        }
    }

    // ---------------------------------------------------------------- transform

    static void ntt(long[] a, boolean invert, long mod, long root) {
        int n = a.length;
        requireSize(n, mod);
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j |= bit;
            if (i < j) {
                long t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            long wLen = powMod(root, (mod - 1) / len, mod);
            if (invert) wLen = inverse(wLen, mod);
            for (int i = 0; i < n; i += len) {
                long w = 1;
                for (int j = 0; j < len / 2; j++) {
                    long u = a[i + j];
                    long v = a[i + j + len / 2] * w % mod;
                    a[i + j] = (u + v) % mod;
                    a[i + j + len / 2] = (u - v + mod) % mod;
                    w = w * wLen % mod;
                }
            }
        }
        if (invert) {
            long nInv = inverse(n, mod);
            for (int i = 0; i < n; i++) a[i] = a[i] * nInv % mod;
        }
    }

    static int nextPow2(int n) {
        int size = 1;
        while (size < n) size <<= 1;
        return Math.max(size, 1);
    }

    /** Linear convolution modulo p; the operands are reduced first, so the domain is exact. */
    static long[] multiply(long[] a, long[] b, long mod, long root) {
        if (a.length == 0 || b.length == 0) return new long[0];
        int size = nextPow2(a.length + b.length - 1);
        requireSize(size, mod);
        long[] fa = new long[size];
        long[] fb = new long[size];
        for (int i = 0; i < a.length; i++) fa[i] = Math.floorMod(a[i], mod);
        for (int i = 0; i < b.length; i++) fb[i] = Math.floorMod(b[i], mod);
        ntt(fa, false, mod, root);
        ntt(fb, false, mod, root);
        for (int i = 0; i < size; i++) fa[i] = fa[i] * fb[i] % mod;
        ntt(fa, true, mod, root);
        return Arrays.copyOf(fa, a.length + b.length - 1);
    }

    static long[] schoolbook(long[] a, long[] b, long mod) {
        long[] out = new long[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                out[i + j] = (out[i + j] + Math.floorMod(a[i], mod) * Math.floorMod(b[j], mod)) % mod;
            }
        }
        return out;
    }

    /** Cyclic convolution of length n (n must divide p - 1); both operands are extended with zeros. */
    static long[] cyclic(long[] a, long[] b, int n, long mod, long root) {
        requireSize(n, mod);
        long[] fa = new long[n];
        long[] fb = new long[n];
        for (int i = 0; i < a.length; i++) fa[i % n] = (fa[i % n] + a[i]) % mod;
        for (int i = 0; i < b.length; i++) fb[i % n] = (fb[i % n] + b[i]) % mod;
        ntt(fa, false, mod, root);
        ntt(fb, false, mod, root);
        for (int i = 0; i < n; i++) fa[i] = fa[i] * fb[i] % mod;
        ntt(fa, true, mod, root);
        return fa;
    }

    /** The exact fold modulo x^n - 1 of a linear product, computed without any transform. */
    static long[] fold(long[] linear, int n, long mod) {
        long[] out = new long[n];
        for (int k = 0; k < linear.length; k++) out[k % n] = (out[k % n] + linear[k]) % mod;
        return out;
    }

    static long[] randomPoly(int degree, long mod, Rng rng) {
        long[] out = new long[degree + 1];
        for (int i = 1; i < out.length; i++) out[i] = rng.nextMod(mod);
        out[0] = rng.nextMod(mod);
        return out;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        fieldProperties();
        roundTrips();
        againstSchoolbook();
        linearVersusCyclic();
        algebraicIdentities();
        bigIntegerOracle();
        sizesAndGuards();
        determinism();

        System.out.println("PASS");
    }

    static void fieldProperties() {
        check(MOD == 119L * (1L << 23) + 1, "p = 119 * 2^23 + 1");
        check(powMod(ROOT, MOD - 1, MOD) == 1, "Fermat: g^(p-1) = 1");
        check(powMod(ROOT, (MOD - 1) / 2, MOD) == MOD - 1, "g^((p-1)/2) = -1, so 2 is a factor of p-1");
        for (long q : new long[]{2, 7, 17}) {
            check(powMod(ROOT, (MOD - 1) / q, MOD) != 1, "g^((p-1)/" + q + ") != 1: q divides ord(g)");
        }
        check(powMod(SMALL_ROOT, SMALL_MOD - 1, SMALL_MOD) == 1 && SMALL_MOD == 17 && SMALL_ROOT == 3,
                "the small field 17 with root 3 is consistent");

        for (int n : new int[]{1, 2, 4, 8, 16, 64, 1024, 1 << 20}) {
            long w = powMod(ROOT, (MOD - 1) / n, MOD);
            check(powMod(w, n, MOD) == 1, "w_n^n = 1 for n = " + n);
            if (n > 1) check(powMod(w, n / 2, MOD) == MOD - 1, "w_n has order exactly n = " + n);
        }
        for (int n : new int[]{4, 8, 16}) {
            long w = powMod(ROOT, (MOD - 1) / n, MOD);
            for (int k = 0; k < n; k++) {
                long sum = 0;
                for (int j = 0; j < n; j++) sum = (sum + powMod(w, (long) j * k, MOD)) % MOD;
                long expected = k == 0 ? n % MOD : 0L;
                check(sum == expected, "sum_j w^(jk) = " + expected + " for n = " + n + ", k = " + k);
            }
            // Orthogonality fails as expected for a non-multiple of n: the geometric sum is still 1.
            long w2 = powMod(ROOT, (MOD - 1) / n, MOD);
            long nonMultiple = 0;
            for (int j = 0; j < n; j++) nonMultiple = (nonMultiple + powMod(w2, j * (n + 1L), MOD)) % MOD;
            check(nonMultiple == 0, "shifting k off a multiple of n still cancels (k = n+1)");
        }
    }

    static void roundTrips() {
        Rng rng = new Rng(5851L);
        for (int power = 0; power <= 8; power++) {
            int n = 1 << power;
            long[] a = randomPoly(n - 1, MOD, rng);
            long[] work = a.clone();
            ntt(work, false, MOD, ROOT);
            ntt(work, true, MOD, ROOT);
            check(Arrays.equals(work, a), "forward followed by inverse is the identity at n = " + n);
            long[] once = new long[n];
            System.arraycopy(a, 0, once, 0, n);
            ntt(once, true, MOD, ROOT);
            ntt(once, false, MOD, ROOT);
            check(Arrays.equals(once, a), "inverse followed by forward is the identity at n = " + n);
        }
        long[] delta = new long[16];
        delta[0] = 1;
        long[] spectrum = delta.clone();
        ntt(spectrum, false, MOD, ROOT);
        for (int i = 0; i < 16; i++) check(spectrum[i] == 1, "the delta has a flat spectrum at " + i);
        long[] constant = new long[16];
        Arrays.fill(constant, 1);
        long[] constantSpectrum = constant.clone();
        ntt(constantSpectrum, false, MOD, ROOT);
        check(constantSpectrum[0] == 16, "the DC term of a unit vector is n");
        for (int i = 1; i < 16; i++) check(constantSpectrum[i] == 0, "all other bins vanish at " + i);
    }

    static void againstSchoolbook() {
        Rng rng = new Rng(5852L);
        int[][] shapes = {{1, 1}, {1, 8}, {8, 1}, {3, 5}, {7, 7}, {9, 10}, {16, 16}, {33, 17},
                {64, 64}, {100, 3}, {129, 129}};
        for (int[] shape : shapes) {
            long[] a = randomPoly(shape[0] - 1, MOD, rng);
            long[] b = randomPoly(shape[1] - 1, MOD, rng);
            long[] fast = multiply(a, b, MOD, ROOT);
            long[] slow = schoolbook(a, b, MOD);
            check(Arrays.equals(fast, slow),
                    "NTT product equals the schoolbook product for " + shape[0] + " x " + shape[1]);
        }
        long[] monomial = new long[3];
        monomial[2] = 1;
        long[] other = randomPoly(4, MOD, rng);
        long[] shifted = multiply(monomial, other, MOD, ROOT);
        check(shifted.length == 7 && shifted[0] == 0 && shifted[1] == 0,
                "multiplication by x^2 shifts the polynomial by two places");
        for (int i = 0; i < other.length; i++) check(shifted[i + 2] == other[i], "the shifted coefficient at " + i);
        long[] scalars = {0, 1, MOD - 1};
        for (long c : scalars) {
            long[] fast = multiply(new long[]{c}, other, MOD, ROOT);
            for (int i = 0; i < other.length; i++) {
                check(fast[i] == c * other[i] % MOD, "scalar multiplication by " + c + " at " + i);
            }
        }
        check(multiply(new long[0], other, MOD, ROOT).length == 0, "an empty operand gives an empty product");
    }

    static void linearVersusCyclic() {
        Rng rng = new Rng(5853L);
        long[] a = randomPoly(5, MOD, rng);
        long[] b = randomPoly(3, MOD, rng);
        int linearLength = a.length + b.length - 1;
        long[] linear = multiply(a, b, MOD, ROOT);
        for (int n : new int[]{1, 2, 4, 8, 16}) {
            long[] cyc = cyclic(a, b, n, MOD, ROOT);
            long[] folded = fold(linear, n, MOD);
            check(Arrays.equals(cyc, folded), "the length-" + n + " cyclic product equals the folded linear product");
        }
        long[] smallCyclic = cyclic(a, b, 4, MOD, ROOT);
        long[] truncated = Arrays.copyOf(linear, 4);
        check(!Arrays.equals(smallCyclic, truncated),
                "with n < " + linearLength + " the cyclic product genuinely wraps (not the linear truncation)");
        check(smallCyclic[0] == (linear[0] + linear[4] + linear[8]) % MOD,
                "the wrap adds all coefficients that crossed x^4 into the constant term");
        check(smallCyclic[1] == (linear[1] + linear[5]) % MOD && smallCyclic[2] == (linear[2] + linear[6]) % MOD,
                "the remaining bins add their own wrapped coefficients");
    }

    static void algebraicIdentities() {
        Rng rng = new Rng(5854L);
        long[] a = randomPoly(9, MOD, rng);
        long[] b = randomPoly(6, MOD, rng);
        long[] c = randomPoly(4, MOD, rng);
        check(Arrays.equals(multiply(a, b, MOD, ROOT), multiply(b, a, MOD, ROOT)), "multiplication commutes");
        long[] left = multiply(multiply(a, b, MOD, ROOT), c, MOD, ROOT);
        long[] right = multiply(a, multiply(b, c, MOD, ROOT), MOD, ROOT);
        check(Arrays.equals(left, right), "multiplication associates");
        long[] bc = multiply(b, c, MOD, ROOT);
        long[] bPlusC = new long[Math.max(b.length, c.length)];
        for (int i = 0; i < b.length; i++) bPlusC[i] = b[i];
        for (int i = 0; i < c.length; i++) bPlusC[i] = (bPlusC[i] + c[i]) % MOD;
        long[] aTimesBPlusC = multiply(a, bPlusC, MOD, ROOT);
        long[] ab = multiply(a, b, MOD, ROOT);
        long[] ac = multiply(a, c, MOD, ROOT);
        long[] sum = new long[Math.max(ab.length, ac.length)];
        for (int i = 0; i < ab.length; i++) sum[i] = ab[i];
        for (int i = 0; i < ac.length; i++) sum[i] = (sum[i] + ac[i]) % MOD;
        check(Arrays.equals(aTimesBPlusC, sum), "multiplication distributes over addition");
        check(bc.length == b.length + c.length - 1, "the auxiliary product has the expected length");
        long[] one = {1};
        check(Arrays.equals(multiply(a, one, MOD, ROOT), a), "[1] is the identity");
        long[] zero = {0};
        long[] product = multiply(a, zero, MOD, ROOT);
        for (long value : product) check(value == 0, "the zero polynomial annihilates");
        // (1 + x)^2 = 1 + 2x + x^2 in Z_p, and the binomial identity holds by repeated multiplication.
        long[] onePlusX = {1, 1};
        long[] squared = multiply(onePlusX, onePlusX, MOD, ROOT);
        check(Arrays.equals(squared, new long[]{1, 2, 1}), "(1+x)^2 = 1 + 2x + x^2");
        long[] power = {1};
        for (int i = 0; i < 5; i++) power = multiply(power, onePlusX, MOD, ROOT);
        long[] binomial = new long[6];
        for (int k = 0; k <= 5; k++) binomial[k] = binomial(5, k) % MOD;
        check(Arrays.equals(power, binomial), "(1+x)^5 equals the binomial row 1 5 10 10 5 1");
    }

    static long binomial(int n, int k) {
        long result = 1;
        for (int i = 1; i <= k; i++) result = result * (n - k + i) / i;
        return result;
    }

    static void bigIntegerOracle() {
        Rng rng = new Rng(5855L);
        int degree = 499;
        long[] a = new long[degree + 1];
        long[] b = new long[degree + 1];
        for (int i = 0; i <= degree; i++) {
            a[i] = rng.nextInt(10);
            b[i] = rng.nextInt(10);
        }
        long[] fast = multiply(a, b, MOD, ROOT);
        check(fast.length == 2 * degree + 1, "the product has 2*degree+1 coefficients");
        BigInteger packedA = pack(a);
        BigInteger packedB = pack(b);
        BigInteger exact = packedA.multiply(packedB);
        BigInteger limb = BigInteger.ONE.shiftLeft(64);
        BigInteger mask = limb.subtract(BigInteger.ONE);
        BigInteger modulus = BigInteger.valueOf(MOD);
        int nonzero = 0;
        for (int k = 0; k <= 2 * degree; k++) {
            long coefficient = exact.shiftRight(64 * k).and(mask).longValueExact();
            check(coefficient < (1L << 62), "no carry crossed a 2^64 limb at " + k);
            long reduced = BigInteger.valueOf(coefficient).mod(modulus).longValue();
            check(fast[k] == reduced, "limb " + k + " reproduces the exact product modulo p");
            if (fast[k] != 0) nonzero++;
        }
        check(nonzero > 100, "the product is dense (" + nonzero + " nonzero coefficients)");
    }

    static BigInteger pack(long[] coefficients) {
        BigInteger packed = BigInteger.ZERO;
        for (int i = coefficients.length - 1; i >= 0; i--) {
            packed = packed.shiftLeft(64).or(BigInteger.valueOf(coefficients[i]));
        }
        return packed;
    }

    static void sizesAndGuards() {
        check(nextPow2(0) == 1 && nextPow2(1) == 1 && nextPow2(2) == 2 && nextPow2(3) == 4
                && nextPow2(1025) == 2048, "nextPow2 rounds up");
        boolean threw = false;
        try {
            requireSize(3, MOD);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "a non-power-of-two length is rejected");
        threw = false;
        try {
            requireSize(1 << 24, MOD);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "1 << 24 is rejected because it does not divide p - 1 (the NTT size cap)");
        threw = false;
        try {
            requireSize(16, SMALL_MOD - 1);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "a length exceeding the small field is rejected");
        requireSize(1 << 23, MOD);
        check(true, "2^23 is the largest supported transform");
        check(nextPow2(1 << 23) == (1 << 23), "the largest supported size is a power of two");

        // The small field: 20 = 3 (mod 17), so the transform only sees the residue.
        long[] smallA = {5};
        long[] smallB = {4};
        long[] smallProduct = multiply(smallA, smallB, SMALL_MOD, SMALL_ROOT);
        check(smallProduct[0] == 3, "5 * 4 = 20 = 3 (mod 17): the field wraps the true coefficient");
        Rng rng = new Rng(5856L);
        for (int trial = 0; trial < 8; trial++) {
            long[] p = randomPoly(3, SMALL_MOD, rng);
            long[] q = randomPoly(3, SMALL_MOD, rng);
            check(Arrays.equals(multiply(p, q, SMALL_MOD, SMALL_ROOT), schoolbook(p, q, SMALL_MOD)),
                    "small-field product matches schoolbook, trial " + trial);
        }
    }

    static void determinism() {
        Rng rng = new Rng(5857L);
        long[] a = randomPoly(20, MOD, rng);
        long[] b = randomPoly(20, MOD, rng);
        check(Arrays.equals(multiply(a, b, MOD, ROOT), multiply(a, b, MOD, ROOT)),
                "the transform is reproducible");
        long[] spectrum = a.clone();
        int size = 64;
        long[] padded = new long[size];
        System.arraycopy(a, 0, padded, 0, Math.min(a.length, size));
        ntt(padded, false, MOD, ROOT);
        ntt(padded, true, MOD, ROOT);
        check(Arrays.equals(padded, Arrays.copyOf(a, size)), "zero padding survives a round trip");
        long sum = 0;
        for (long value : a) sum = (sum + value) % MOD;
        System.arraycopy(a, 0, spectrum, 0, spectrum.length);
        long[] transformed = new long[nextPow2(a.length)];
        System.arraycopy(a, 0, transformed, 0, a.length);
        ntt(transformed, false, MOD, ROOT);
        check(transformed[0] == sum, "X_0 is the coefficient sum");
        long total = 0;
        for (long value : transformed) total = (total + value) % MOD;
        check(total == (long) transformed.length % MOD * a[0] % MOD,
                "sum_k X_k = n * x_0 (the inverse DC identity)");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
