/** Task 756 - Implement side-channel resistant constant-time modular exponentiation.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A textbook square-and-multiply loop multiplies only when the exponent bit is 1, so the number of
 * multiplications equals the Hamming weight of the secret exponent: the timing (or power trace) leaks
 * 1 bit per bit of the exponent. Two standard defences are implemented here and measured:
 *
 *  * square-and-multiply-*always*: every iteration squares, multiplies *with a dummy accumulator* and
 *    then folds the real product in through an arithmetic mask (mask = 0 or -1), so there is no
 *    data-dependent branch and no data-dependent number of multiplies;
 *  * the Montgomery ladder: both ladder rungs are always computed (two squarings and one product per
 *    bit) and the bit only selects which result goes into which register, again with a masked select.
 *
 * The measurements are taken on a deterministic operation-count model (squares, multiplies, selects),
 * which is how constant-time behaviour is demonstrated without a wall clock. The model of the
 * variable-time version is also used by a small attacker: observing the operation count shrinks the
 * exponent search space from 2^k to C(k, weight), while the constant-time versions give a count that is
 * independent of the exponent, so the space does not shrink at all.
 *
 * Simplified: the modulus is a toy 32-bit prime so that a*b fits in a long and "% " needs no Montgomery
 * reduction; windowing, blinding and CRT are omitted. Base and exponent are ordinary long values, never
 * real key material. Both masked versions are "constant-time" with respect to *modelled* operations only.
 *
 * Verified: both ladders agree with a repeated-multiplication brute force and with a variable-time
 * reference; exponent 0, 1, modulus 1 and base 0 boundaries; Fermat's little theorem; the identities
 * a^(e1+e2) and (a^e1)^e2; flat operation counts across every exponent of a given width; the exact
 * Hamming-weight leak of the naive version, and its absence in the ladder; and the attacker's search
 * space collapsing for the naive version but not for the constant-time one.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** Modular exponentiation variants plus the operation counter standing in for time or power. */
final class ConstantTime {

    long squares, multiplies, selects;

    /** Toy modulus arithmetic: a, b < 2^31 so the product fits in a long. */
    static long mulmod(long a, long b, long modulus) { return (a * b) % modulus; }

    /** Branch-free select; mask must be 0 (unless = ifClear) or -1 (unless = ifSet). */
    static long select(long mask, long ifSet, long ifClear) { return (ifSet & mask) | (ifClear & ~mask); }

    static long maskOf(long bit) { return -(bit & 1L); }

    /** Naive square-and-multiply: multiplies only on set bits, which is exactly the leak. */
    long variableTimePow(long base, long exponent, long modulus) {
        long result = 1 % modulus;
        long power = base % modulus;
        long e = exponent;
        while (e > 0) {
            if ((e & 1L) != 0) {
                result = mulmod(result, power, modulus);
                multiplies++;
            }
            power = mulmod(power, power, modulus);
            squares++;
            e >>= 1;
        }
        return result;
    }

    /** Square-and-multiply-always: identical work for every exponent of the same width. */
    long maskedPow(long base, long exponent, long modulus, int width) {
        long acc = 1 % modulus;
        long power = base % modulus;
        for (int bit = width - 1; bit >= 0; bit--) {
            acc = mulmod(acc, acc, modulus);
            squares++;
            long product = mulmod(acc, power, modulus);       // always computed, sometimes discarded
            multiplies++;
            acc = select(maskOf(exponent >>> bit), product, acc);
            selects++;
        }
        return acc;
    }

    /** Montgomery ladder: 2 squarings + 1 product + 2 selects per bit, whatever the exponent is. */
    long ladderPow(long base, long exponent, long modulus, int width) {
        long r0 = 1 % modulus;
        long r1 = base % modulus;
        for (int bit = width - 1; bit >= 0; bit--) {
            long evenSquare = mulmod(r0, r0, modulus);        // rung for bit 0
            squares++;
            long oddSquare = mulmod(r1, r1, modulus);         // rung for bit 1
            squares++;
            long cross = mulmod(r0, r1, modulus);
            multiplies++;
            long mask = maskOf(exponent >>> bit);
            r0 = select(mask, cross, evenSquare);
            r1 = select(mask, oddSquare, cross);
            selects += 2;
        }
        return r0;
    }

    long total() { return squares + multiplies + selects; }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Independent oracle: multiply `exponent` times, one multiplication per step. */
    static long bruteForcePow(long base, long exponent, long modulus) {
        long acc = 1 % modulus;
        for (long i = 0; i < exponent; i++) acc = ConstantTime.mulmod(acc, base, modulus);
        return acc;
    }

    /** Is `value` prime? Trial division is plenty for the toy moduli used here. */
    static boolean isPrime(long value) {
        if (value < 2) return false;
        for (long d = 2; d * d <= value; d++) if (value % d == 0) return false;
        return true;
    }

    static int bitWidth(long exponent) {
        int width = 1;
        while ((1L << width) <= exponent) width++;
        return width;
    }

    /** Which exponents of this width produce the observed operation count? */
    static List<Long> candidateSet(long observed, int width, boolean constantTime) {
        List<Long> candidates = new ArrayList<>();
        for (long e = 1; e < (1L << width); e++) {
            ConstantTime model = new ConstantTime();
            if (constantTime) model.ladderPow(1, e, 1_000_000_007L, width);
            else model.variableTimePow(1, e, 1_000_000_007L);
            if (model.total() == observed) candidates.add(e);
        }
        return candidates;
    }

    public static void main(String[] args) {
        Prng rng = new Prng(0x600D5EEDL);
        long prime = 2_147_483_647L;                       // 2^31 - 1, a Mersenne prime
        check(isPrime(prime), "the toy modulus is prime: " + prime);
        check(prime < (1L << 31), "the toy modulus keeps a*b inside a long");

        // 1. the masked select is a real conditional move: mask 0 picks the "clear" side, -1 the other
        for (long bit = 0; bit < 2; bit++) {
            long mask = ConstantTime.maskOf(bit);
            check(mask == 0 || mask == -1, "the mask is 0 or -1, got " + mask);
            check(ConstantTime.select(mask, 12345, 999) == (bit == 1 ? 12345 : 999),
                    "select follows the mask for bit " + bit);
        }
        check(ConstantTime.select(0, 0, 0) == 0, "select of zeros is zero");

        // 2. correctness against a repeated-multiplication brute force on small exponents
        int bruteChecked = 0;
        for (int trial = 0; trial < 60; trial++) {
            long base = rng.nextInt(1000) + 2;
            long exponent = rng.nextInt(60);
            long modulus = 2_147_483_647L;
            long expected = bruteForcePow(base, exponent, modulus);
            ConstantTime masked = new ConstantTime();
            ConstantTime ladder = new ConstantTime();
            int width = Math.max(1, bitWidth(exponent));
            check(masked.maskedPow(base, exponent, modulus, width) == expected,
                    "masked pow matches brute force for " + base + "^" + exponent);
            check(ladder.ladderPow(base, exponent, modulus, width) == expected,
                    "ladder matches brute force for " + base + "^" + exponent);
            bruteChecked++;
        }
        check(bruteChecked == 60, "60 exponentiation results cross-checked against brute force");

        // 3. correctness against the naive reference on full-width exponents
        int wideChecked = 0;
        for (int trial = 0; trial < 200; trial++) {
            long base = rng.nextInt(1 << 20) + 2;
            long exponent = (long) (rng.nextInt(Integer.MAX_VALUE));
            long expected = new ConstantTime().variableTimePow(base, exponent, prime);
            check(new ConstantTime().maskedPow(base, exponent, prime, 31) == expected,
                    "masked pow agrees on " + base + "^" + exponent);
            check(new ConstantTime().ladderPow(base, exponent, prime, 31) == expected,
                    "ladder agrees on " + base + "^" + exponent);
            wideChecked++;
        }
        check(wideChecked == 200, "200 full-width exponentiations agreed with the reference");

        // 4. boundaries: exponent 0 and 1, modulus 1, base 0 and base 1
        check(new ConstantTime().maskedPow(7, 0, prime, 1) == 1, "a^0 == 1");
        check(new ConstantTime().ladderPow(7, 0, prime, 1) == 1, "ladder a^0 == 1");
        check(new ConstantTime().maskedPow(7, 1, prime, 1) == 7, "a^1 == a");
        check(new ConstantTime().ladderPow(7, 1, prime, 1) == 7, "ladder a^1 == a");
        check(new ConstantTime().maskedPow(12345, 99, 1, 7) == 0, "everything is 0 modulo 1");
        check(new ConstantTime().maskedPow(0, 5, prime, 3) == 0, "0^e == 0 for e > 0");
        check(new ConstantTime().maskedPow(1, 1_234_567, prime, 21) == 1, "1^e == 1");
        check(new ConstantTime().maskedPow(prime + 5, 1, prime, 1) == 5, "the base is reduced modulo m");

        // 5. Fermat's little theorem and the exponent identities
        int fermatChecked = 0;
        for (int trial = 0; trial < 20; trial++) {
            long base = rng.nextInt(1_000_000) + 2;
            check(new ConstantTime().ladderPow(base, prime - 1, prime, 31) == 1,
                    "a^(p-1) == 1 mod p for base " + base);
            fermatChecked++;
        }
        check(fermatChecked == 20, "Fermat's little theorem holds for 20 bases");
        for (int trial = 0; trial < 20; trial++) {
            long base = rng.nextInt(1000) + 2;
            long e1 = rng.nextInt(4000), e2 = rng.nextInt(4000);
            long left = new ConstantTime().ladderPow(base, e1 + e2, prime, 14);
            long right = ConstantTime.mulmod(new ConstantTime().ladderPow(base, e1, prime, 13),
                    new ConstantTime().ladderPow(base, e2, prime, 13), prime);
            check(left == right, "a^(e1+e2) == a^e1 * a^e2 for " + base + "^(" + e1 + "+" + e2 + ")");
        }

        // 6. flat operation counts: every exponent of a fixed width costs exactly the same
        int width = 12;
        long maskedReference = -1, ladderReference = -1, referenceSquares = -1;
        long minTotal = Long.MAX_VALUE, maxTotal = Long.MIN_VALUE;
        for (long exponent = 0; exponent < (1L << width); exponent++) {
            ConstantTime masked = new ConstantTime();
            ConstantTime ladder = new ConstantTime();
            masked.maskedPow(3, exponent, prime, width);
            ladder.ladderPow(3, exponent, prime, width);
            if (maskedReference < 0) {
                maskedReference = masked.total();
                ladderReference = ladder.total();
                referenceSquares = masked.squares;
            }
            check(masked.total() == maskedReference, "square-and-multiply-always is flat for exponent "
                    + exponent + ": " + masked.total() + " vs " + maskedReference);
            check(ladder.total() == ladderReference, "the ladder is flat for exponent " + exponent + ": "
                    + ladder.total() + " vs " + ladderReference);
            check(masked.squares == referenceSquares, "one squaring per bit for exponent " + exponent);
            minTotal = Math.min(minTotal, masked.total());
            maxTotal = Math.max(maxTotal, masked.total());
        }
        check(minTotal == maxTotal, "zero variance over all " + (1 << width) + " exponents of width " + width);
        check(maskedReference == width * 3L, "square-and-multiply-always: 3 modelled ops per bit, got "
                + maskedReference);
        check(ladderReference == width * 5L, "the ladder: 5 modelled ops per bit, got " + ladderReference);

        // 7. the naive version leaks the Hamming weight of the exponent, and only that
        for (long exponent : new long[]{1, 2, 3, 8, 255, 1023, 1_000_000_007L}) {
            ConstantTime naive = new ConstantTime();
            naive.variableTimePow(3, exponent, prime);
            int weight = Long.bitCount(exponent);
            int bits = bitWidth(exponent);
            check(naive.multiplies == weight, "naive multiplies == Hamming weight for exponent " + exponent);
            check(naive.squares == bits, "naive squares == bit length for exponent " + exponent);
        }

        // 8. the attacker's view: the naive count shrinks the search space, the ladder's does not
        int attackWidth = 8;
        long secret = 0b1011_0110L;                        // hidden exponent, weight 5
        ConstantTime observed = new ConstantTime();
        observed.variableTimePow(3, secret, prime);
        List<Long> naiveCandidates = candidateSet(observed.total(), attackWidth, false);
        ConstantTime ladderModel = new ConstantTime();
        ladderModel.ladderPow(3, secret, prime, attackWidth);
        List<Long> ladderCandidates = candidateSet(ladderModel.total(), attackWidth, true);
        check(naiveCandidates.contains(secret), "the attacker's candidate set contains the secret exponent");
        check(naiveCandidates.size() <= 255 / 4, "the naive leak removes at least three quarters of the "
                + "space: " + naiveCandidates.size() + " candidates remain");
        check(ladderCandidates.size() == (1 << attackWidth) - 1,
                "the ladder reveals nothing: all exponents of the width remain, got " + ladderCandidates.size());
        check(Long.bitCount(secret) == 5 && bitWidth(secret) == attackWidth, "the attack scenario is set up");

        System.out.println("modulus: " + prime + " (31-bit prime), uniform exponent width: " + width);
        System.out.println("constant-time cost: " + maskedReference + " modelled ops per exponent (3 per bit), "
                + "variance 0 over " + (1 << width) + " exponents");
        System.out.println("naive leak: squares + Hamming weight; search space 255 -> " + naiveCandidates.size()
                + " with the naive version, 255 -> " + ladderCandidates.size() + " with the ladder");
        System.out.println("PASS");
    }
}
