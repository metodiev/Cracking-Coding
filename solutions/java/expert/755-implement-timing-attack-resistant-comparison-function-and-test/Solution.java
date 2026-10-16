/** Task 755 - Implement a timing attack resistant comparison function and test.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A byte-wise comparison that returns as soon as two bytes differ leaks *where* the strings differ:
 * an attacker who can measure the time (or an operation counter standing in for it) recovers the
 * secret one byte at a time. The classic fix is to touch every byte and fold the difference into an
 * accumulator:
 *
 *     diff |= a[i] ^ b[i]     everywhere, and only then  diff == 0
 *
 * so the number of executed operations is a function of the length alone. This file implements both
 * comparators and a *timing model*: a counter of the primitive operations the CPU would perform per
 * element (load of a, load of b, xor, or/cmp, branch), which is the standard way to demonstrate timing
 * behaviour deterministically and without a real clock.
 *
 * The chosen index of a mismatch is also modelled, because leaking the length is the one thing this
 * construction cannot hide: comparing equal-length secrets is constant-time, while a length-based early
 * exit still reveals *that* the lengths differ (documented, and asserted, rather than hidden).
 *
 * Simplified: a Java method cannot be verified constant-time by wall-clock measurement, so the time is
 * the count of modelled primitives, and the "attacker" is a correlation analysis over that counter
 * rather than a real side-channel capture. The comparison operates on a capped toy secret length.
 *
 * Verified: correctness against Arrays.equals on random inputs; the constant-time version performs an
 * identical operation count for every mismatch position and every pair of equal-length inputs; the
 * naive version's count is a strictly increasing function of the first mismatch index, and a
 * prefix-guessing attacker recovers the secret completely from that leak while recovering nothing from
 * the constant-time one; length handling and empty-input boundaries.
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

/** The timing model: counts the primitive operations a byte-wise loop would execute. */
final class TimingModel {
    long loads, xors, ors, branches;
    long total;

    void touch() { loads += 2; total += 2; }        // reading a[i] and b[i]

    void fold() { xors++; ors++; total += 2; }      // diff |= a[i] ^ b[i]

    void branch() { branches++; total++; }

    @Override
    public String toString() {
        return "loads=" + loads + " xors=" + xors + " ors=" + ors + " branches=" + branches
                + " total=" + total;
    }
}

/** Comparison primitives. Both are correct; only the operation count differs. */
final class Comparators {

    /** Early exit on the first difference - this is the leak. */
    static boolean naiveEquals(byte[] a, byte[] b, TimingModel model) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            model.touch();
            boolean same = a[i] == b[i];
            model.branch();
            if (!same) return false;
            model.fold();
        }
        return true;
    }

    /** Touches every byte and folds the difference; the work is a function of the length alone. */
    static boolean constantTimeEquals(byte[] a, byte[] b, TimingModel model) {
        if (a.length != b.length) return false;
        int diff = 0;
        for (int i = 0; i < a.length; i++) {
            model.touch();
            diff |= a[i] ^ b[i];
            model.fold();
            model.branch();                          // the loop branch, taken the same way every time
        }
        return diff == 0;
    }

    static boolean constantTimeEquals(byte[] a, byte[] b) { return constantTimeEquals(a, b, new TimingModel()); }

    static boolean naiveEquals(byte[] a, byte[] b) { return naiveEquals(a, b, new TimingModel()); }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static byte[] bytes(int... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) out[i] = (byte) values[i];
        return out;
    }

    static byte[] randomBytes(int length, Prng rng) {
        byte[] out = new byte[length];
        for (int i = 0; i < length; i++) out[i] = (byte) rng.nextInt(256);
        return out;
    }

    /**
     * The attacker. The naive comparator's response is 5 ops per skipped byte plus 3, so the candidate
     * byte that extends the matching prefix is the unique slowest response; the constant-time comparator
     * answers in the same time for every candidate, so no candidate stands out and the attack stops.
     * Returns how many leading bytes were recovered.
     */
    static int recoverPrefix(byte[] padding, byte[] secret, boolean constantTime) {
        byte[] guess = padding.clone();
        int matched = 0;
        while (matched < secret.length) {
            long slowest = -1, fastest = Long.MAX_VALUE;
            int bestByte = -1;
            for (int candidate = 0; candidate < 256; candidate++) {
                byte[] attempt = guess.clone();
                attempt[matched] = (byte) candidate;
                TimingModel model = new TimingModel();
                boolean equal = constantTime ? Comparators.constantTimeEquals(attempt, secret, model)
                        : Comparators.naiveEquals(attempt, secret, model);
                if (equal) return secret.length;              // the whole secret was already guessed
                slowest = Math.max(slowest, model.total);
                fastest = Math.min(fastest, model.total);
                if (model.total == slowest) bestByte = candidate;
            }
            if (slowest == fastest || bestByte < 0) break;    // no usable signal
            guess[matched] = (byte) bestByte;
            matched++;
        }
        return matched;
    }

    public static void main(String[] args) {
        Prng rng = new Prng(0x7A11E5EEDL);

        // 1. correctness against the library comparison, on random inputs
        int compared = 0;
        for (int trial = 0; trial < 400; trial++) {
            int length = 1 + rng.nextInt(33);
            byte[] a = randomBytes(length, rng);
            byte[] b = rng.nextInt(2) == 0 ? a.clone() : randomBytes(length, rng);
            check(Comparators.constantTimeEquals(a, b) == Arrays.equals(a, b), "constant-time equals matches");
            check(Comparators.naiveEquals(a, b) == Arrays.equals(a, b), "naive equals matches");
            compared++;
        }
        check(compared == 400, "400 random comparisons agreed with Arrays.equals");
        check(Comparators.constantTimeEquals(bytes(0, 1, 2), bytes(0, 1, 2)), "identical inputs compare equal");
        check(!Comparators.constantTimeEquals(bytes(0, 1, 2), bytes(0, 1, 3)), "differing inputs differ");
        check(!Comparators.constantTimeEquals(bytes(0, 1), bytes(0, 1, 2)), "different lengths are unequal");
        check(Comparators.constantTimeEquals(new byte[0], new byte[0]), "two empty arrays are equal");
        check(!Comparators.constantTimeEquals(new byte[0], bytes(0)), "empty vs non-empty is unequal");

        // 2. the constant-time version's work depends only on the length, never on the mismatch position
        byte[] secret = randomBytes(32, rng);
        long reference = -1;
        for (int mismatch = -1; mismatch < secret.length; mismatch++) {
            byte[] attempt = secret.clone();
            if (mismatch >= 0) attempt[mismatch] = (byte) (attempt[mismatch] ^ 0xFF);
            TimingModel model = new TimingModel();
            Comparators.constantTimeEquals(attempt, secret, model);
            if (reference < 0) reference = model.total;
            check(model.total == reference, "constant-time work is flat: mismatch at " + mismatch + " gave "
                    + model.total + " ops, expected " + reference);
            check(model.loads == 64 && model.xors == 32, "every byte was loaded and folded: " + model);
        }
        check(reference == 32 * 5, "32-byte comparison costs 5 modelled ops per byte, got " + reference);

        // 3. the naive version leaks the mismatch position: a strictly increasing operation count
        long previous = -1;
        for (int mismatch = 0; mismatch < secret.length; mismatch++) {
            byte[] attempt = secret.clone();
            attempt[mismatch] = (byte) (attempt[mismatch] ^ 0xFF);
            TimingModel model = new TimingModel();
            boolean equal = Comparators.naiveEquals(attempt, secret, model);
            check(!equal, "the naive comparator still reports inequality");
            check(model.total == 5L * mismatch + 3, "the naive count is 5 ops per skipped byte plus 3: "
                    + model.total + " for a mismatch at " + mismatch);
            check(model.total > previous, "the naive count grows with the mismatch position");
            previous = model.total;
        }
        check(previous == 5L * 31 + 3, "the last mismatch costs one extra op per skipped byte: " + previous);

        // 4. flatness is measured as exactly zero variance, not "looks similar"
        long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
        Prng flatRng = new Prng(0xF1A7L);
        for (int trial = 0; trial < 500; trial++) {
            byte[] other = randomBytes(16, flatRng);
            TimingModel model = new TimingModel();
            Comparators.constantTimeEquals(other, randomBytes(16, flatRng), model);
            min = Math.min(min, model.total);
            max = Math.max(max, model.total);
        }
        check(min == max, "constant-time comparison has zero timing variance: [" + min + ", " + max + "]");

        // 5. the model is attacker-usable: a prefix search recovers the secret from the naive leak only
        byte[] zeroFree = new byte[24];
        for (int i = 0; i < zeroFree.length; i++) {
            zeroFree[i] = (byte) (1 + rng.nextInt(255));
        }
        byte[] padding = new byte[zeroFree.length];               // the attacker pads with zero bytes
        int naiveRecovered = recoverPrefix(padding, zeroFree, false);
        int constantRecovered = recoverPrefix(padding, zeroFree, true);
        check(naiveRecovered == zeroFree.length, "the naive leak gives up the whole secret: "
                + naiveRecovered + "/" + zeroFree.length);
        check(constantRecovered == 0, "the constant-time comparator gives up nothing: " + constantRecovered);

        // 6. equal inputs cost the same in both comparators, and lengths are still observable
        byte[] same = randomBytes(24, new Prng(0x11L));
        TimingModel constantModel = new TimingModel();
        TimingModel naiveModel = new TimingModel();
        check(Comparators.constantTimeEquals(same, same, constantModel), "equal arrays are equal");
        check(Comparators.naiveEquals(same, same, naiveModel), "equal arrays are equal in both");
        check(constantModel.total == naiveModel.total, "equal inputs cost the same in both comparators: "
                + constantModel.total + " vs " + naiveModel.total);
        check(constantModel.branches == same.length, "one loop branch per byte: " + constantModel.branches);
        TimingModel shortModel = new TimingModel();
        Comparators.constantTimeEquals(new byte[8], same, shortModel);
        check(shortModel.total == 0, "a length mismatch returns before any byte work: " + shortModel.total);

        System.out.println("constant-time 32-byte compare: " + reference + " modelled ops, variance 0 over 500 "
                + "random inputs");
        System.out.println("naive compare: " + previous + " ops for a mismatch in the last byte, 3 for a "
                + "mismatch in the first");
        System.out.println("prefix recovery: naive " + naiveRecovered + "/" + zeroFree.length
                + " bytes, constant-time " + constantRecovered + "/" + zeroFree.length + " bytes");
        System.out.println("PASS");
    }
}
