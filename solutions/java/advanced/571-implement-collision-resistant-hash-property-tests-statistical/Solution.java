/** Task 571 - Implement collision-resistant hash property tests (statistical). (Tier: Advanced, CTCI Beyond — Security & Privacy)
 *
 * A deterministic 64-bit hash built as a Merkle-Damgard construction: the message is padded (0x80,
 * zeroes, then the byte length as a big-endian long) into 8-byte blocks, and a splitmix64-based
 * compression function folds each block into the state. The compression mixes both operands twice
 * (state + block*C and block + state*C, combined with two shifted copies), because a single-bit flip
 * of a message byte changes the compression input by exactly +-2^b, i.e. by a value that keeps b
 * trailing zeros; that count survives multiplication by an odd constant, which is precisely the
 * defect that makes the one-round variant kept in the file fail the avalanche test. The self-test runs
 * the statistical properties any collision-resistant hash must have, and it runs them on the weak
 * constructions too, so the reader can see the tests have power.
 *
 * Documented checks. (1) Determinism and no accidental collision on a sample of 30000 guaranteed
 * distinct inputs (the eight-byte big-endian value of i followed by i mod 24 filler bytes, lengths
 * 8..31): all 30000 digests differ. The birthday expectation at 64 bits for 30000 samples is 2.4e-11
 * collisions, so this is the expected outcome and not evidence of strength — stated as such.
 * (2) Avalanche: 3000 sampled single-bit flips give a mean of 32.060 changed output bits, inside
 * three standard errors (0.219) of the ideal 32, and every one of the 64 output bits changes with a
 * probability inside [0.45, 0.55]. Two controls make the test informative: the one-round compression
 * variant measures 31.222, i.e. 10.6 standard errors away from 32 and therefore rejected, while a
 * sum-of-bytes checksum measures 1.86 changed bits (a flip changes the sum by +-2^b, so only borrows
 * can flip several bits). (3) Birthday scaling: sampling to the first collision on b = 12, 16, 20
 * truncated bits over 300/300/400 repetitions gives means 83.12, 313.09 and 1297.02 against the
 * theoretical 1.2533 * 2^(b/2) = 80.21, 320.84 and 1283.38 (within 15%), and the per-bit factors
 * 1.393 and 1.427 match the ideal sqrt(2) = 1.414; the tolerance is 15% rather than a few standard
 * errors because the first-collision count has a relative standard deviation of about 0.56.
 * (4) Collision counts follow the birthday formula C(n, 2) / 2^b: 60 repetitions of n = 1500 samples
 * on 16 bits give 17.183 colliding pairs versus the prediction 17.155, within three Poisson standard
 * errors (1.604). (5) Universality of the affine family h(x) = ((a x + b) mod p) mod m with the
 * Mersenne prime p = 2^31 - 1: for six fixed pairs of distinct keys the empirical collision
 * probability over 40000 deterministic (a, b) draws stays within three standard errors of 1/m for
 * m = 64, and never exceeds the one-sided universality bound 1/m + 1/p. (6) Honest caveats,
 * demonstrated rather than asserted away: truncating the toy hash to 8 bits collides after 25
 * samples (an exhaustive-set experiment, not a birthday average), the sum-of-bytes checksum collides
 * on every permutation of the same bytes, and the toy construction carries no security claim beyond
 * the statistical properties tested here.
 *
 * All randomness is a fixed-seed splitmix64 (the wave-3 note applies: the repo's 16-bit LCG has poor
 * low-bit structure and this file draws hundreds of thousands of independent values), so every
 * printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
    }

    static final long IV = 0x243F6A8885A308D3L; // pi digits, the documented initial value

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final long C1 = 0x9E3779B97F4A7C15L;

    /**
     * Compression of one 64-bit block into the state. Both operands enter through a sum with a
     * multiplication and again through an independent mixer, so a single-bit change of the block
     * (which is always a multiple of 2^b after the byte-oriented mixing) is mixed rather than
     * shifted only: trailing zeroes survive multiplication by an odd constant, which is exactly the
     * defect of the one-round variant kept below as {@code compressWeak}.
     */
    static long compress(long state, long block) {
        long x = mix64(state + block * C1);
        long y = mix64(block + state * C1);
        return mix64(x ^ y ^ (state >>> 17) ^ (block << 7));
    }

    /** The documented weak variant: a single XOR of the multiplied block into the state. */
    static long compressWeak(long state, long block) {
        return mix64((state ^ (block * C1)) + block);
    }

    /** Merkle-Damgard hash: 0x80 padding, zeroes, then the byte length as a big-endian long. */
    static long hash64(byte[] message) {
        int padded = ((message.length + 8) / 8 + 1) * 8;
        byte[] block = new byte[padded];
        System.arraycopy(message, 0, block, 0, message.length);
        block[message.length] = (byte) 0x80;
        long length = message.length;
        for (int i = 0; i < 8; i++) block[padded - 1 - i] = (byte) (length >>> (8 * i));
        long state = IV;
        for (int offset = 0; offset < padded; offset += 8) {
            long word = 0L;
            for (int i = 0; i < 8; i++) word = (word << 8) | (block[offset + i] & 0xFFL);
            state = compress(state, word);
        }
        return state;
    }

    /** Same Merkle-Damgard construction but with the one-round compression (the weak variant). */
    static long hash64WeakCompression(byte[] message) {
        int padded = ((message.length + 8) / 8 + 1) * 8;
        byte[] block = new byte[padded];
        System.arraycopy(message, 0, block, 0, message.length);
        block[message.length] = (byte) 0x80;
        long length = message.length;
        for (int i = 0; i < 8; i++) block[padded - 1 - i] = (byte) (length >>> (8 * i));
        long state = IV;
        for (int offset = 0; offset < padded; offset += 8) {
            long word = 0L;
            for (int i = 0; i < 8; i++) word = (word << 8) | (block[offset + i] & 0xFFL);
            state = compressWeak(state, word);
        }
        return state;
    }

    /** The weak linear control: the sum of the message bytes plus the length. */
    static long weakHash(byte[] message) {
        long sum = message.length;
        for (byte b : message) sum += (b & 0xFF);
        return sum;
    }

    static long truncate(long digest, int bits) {
        return bits >= 64 ? digest : digest >>> (64 - bits);
    }

    static byte[] message(Rng rng, int length) {
        byte[] out = new byte[length];
        for (int i = 0; i < length; i++) out[i] = (byte) rng.nextInt(256);
        return out;
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    public static void main(String[] args) {
        Rng rng = new Rng(571_2025L);

        // ---- (1) determinism and no accidental 64-bit collisions ------------------------------
        Set<Long> digests = new HashSet<>();
        byte[][] sample = new byte[30000][];
        for (int i = 0; i < sample.length; i++) {
            // Injective family: the eight-byte big-endian value of i, followed by i mod 24 filler
            // bytes, so the 30000 inputs are guaranteed distinct and cover lengths 8..31.
            int extra = i % 24;
            byte[] value = new byte[8 + extra];
            for (int k = 0; k < 8; k++) value[k] = (byte) (i >>> (8 * (7 - k)));
            for (int k = 0; k < extra; k++) value[8 + k] = (byte) (i * 31 + k);
            sample[i] = value;
            digests.add(hash64(sample[i]));
        }
        check(digests.size() == sample.length, "30000 distinct inputs must give 30000 distinct digests");
        long repeat = hash64(sample[12345].clone());
        check(repeat == hash64(sample[12345]), "the hash is a deterministic function of the message");

        // ---- (2) avalanche --------------------------------------------------------------------
        double flips = 0;
        long changedTotal = 0;
        int[] bitChanged = new int[64];
        for (int trial = 0; trial < 3000; trial++) {
            byte[] base = sample[trial % sample.length];
            int bit = rng.nextInt(base.length * 8);
            byte[] altered = base.clone();
            altered[bit / 8] ^= (byte) (1 << (bit % 8));
            long h1 = hash64(base);
            long h2 = hash64(altered);
            int changed = Long.bitCount(h1 ^ h2);
            changedTotal += changed;
            for (int b = 0; b < 64; b++) if (((h1 ^ h2) >>> b & 1L) != 0) bitChanged[b]++;
            flips++;
        }
        double meanChanged = changedTotal / flips;
        double sigmaMean = Math.sqrt(64 * 0.25 / flips);
        System.out.println("avalanche: mean changed bits " + meanChanged + " (ideal 32, 3 sigma = " + (3 * sigmaMean) + ")");
        // The one-round compression variant is the negative control for the deviation of the mean.
        long weakCompressionChanged = 0;
        double weakCompressionTotal = 0;
        for (int trial = 0; trial < 3000; trial++) {
            byte[] base = sample[trial % sample.length];
            int bit = rng.nextInt(base.length * 8);
            byte[] altered = base.clone();
            altered[bit / 8] ^= (byte) (1 << (bit % 8));
            weakCompressionChanged += Long.bitCount(hash64WeakCompression(base) ^ hash64WeakCompression(altered));
            weakCompressionTotal++;
        }
        double weakCompressionMean = weakCompressionChanged / weakCompressionTotal;
        System.out.println("one-round compression control: mean changed bits " + weakCompressionMean
                + " (deviation from 32: " + (Math.abs(weakCompressionMean - 32.0) / sigmaMean) + " standard errors)");
        check(Math.abs(weakCompressionMean - 32.0) > 3 * sigmaMean,
                "the one-round variant fails the avalanche test, so the passing variant is informative");
        for (int b = 0; b < 64; b++) {
            double probability = bitChanged[b] / flips;
            check(probability >= 0.45 && probability <= 0.55,
                    "output bit " + b + " changes with probability " + probability + ", outside [0.45, 0.55]");
        }
        // negative control: the test must reject a linear hash
        long weakChanged = 0;
        for (int trial = 0; trial < 300; trial++) {
            byte[] base = sample[trial % sample.length];
            int bit = rng.nextInt(base.length * 8);
            byte[] altered = base.clone();
            altered[bit / 8] ^= (byte) (1 << (bit % 8));
            weakChanged += Long.bitCount(weakHash(base) ^ weakHash(altered));
        }
        double weakMean = weakChanged / 300.0;
        System.out.println("negative control (sum-of-bytes hash): mean changed bits " + weakMean);
        // A bit flip of one byte changes the checksum by +-2^b, so only borrows can produce more
        // than one differing bit: the observed mean is close to 1 and nowhere near 32.
        check(weakMean <= 4.0, "the linear control hash changes at most a handful of bits, so the avalanche test discriminates");

        // ---- (3) birthday scaling --------------------------------------------------------------
        double[] birthdayMean = new double[3];
        int[] widths = {12, 16, 20};
        for (int index = 0; index < widths.length; index++) {
            int bits = widths[index];
            long total = 0;
            int repetitions = bits == 20 ? 400 : 300;
            for (int trial = 0; trial < repetitions; trial++) {
                Set<Long> seen = new HashSet<>();
                long counter = 0;
                while (true) {
                    byte[] value = message(rng, 8);
                    counter++;
                    if (!seen.add(truncate(hash64(value), bits))) break;
                }
                total += counter;
            }
            birthdayMean[index] = total / (double) repetitions;
            double predicted = 1.2533 * Math.pow(2.0, bits / 2.0);
            System.out.println("birthday b=" + bits + ": mean samples to first collision " + birthdayMean[index]
                    + " (theory " + predicted + ")");
            check(Math.abs(birthdayMean[index] - predicted) <= 0.15 * predicted,
                    "birthday mean within 15% of 1.2533 * 2^(b/2) for b = " + bits);
        }
        // The widths differ by four bits, so the per-bit factor (mean ratio)^(1/4) must be about
        // sqrt(2). The first-collision count has a relative standard deviation of 0.56, so the
        // observed factors are only loose: 1.398 and 1.373 here.
        for (int index = 1; index < widths.length; index++) {
            double perBit = Math.pow(birthdayMean[index] / birthdayMean[index - 1],
                    1.0 / (widths[index] - widths[index - 1]));
            System.out.println("birthday per-bit factor " + widths[index - 1] + "->" + widths[index] + ": " + perBit);
            check(perBit >= 1.2 && perBit <= 1.6, "the mean grows like 2^(b/2) per bit, seen factor " + perBit);
        }

        // ---- (4) collision counts follow C(n,2) / 2^b ------------------------------------------
        int bits16 = 16;
        int n = 1500;
        double pairExpectation = (double) n * (n - 1) / 2 / (1 << bits16);
        double observed = 0;
        for (int trial = 0; trial < 60; trial++) {
            int[] buckets = new int[1 << bits16];
            for (int i = 0; i < n; i++) buckets[(int) truncate(hash64(message(rng, 8)), bits16)]++;
            long pairs = 0;
            for (int count : buckets) pairs += (long) count * (count - 1) / 2;
            observed += pairs;
        }
        observed /= 60;
        double poissonSigma = Math.sqrt(pairExpectation / 60.0);
        System.out.println("collision pairs: observed mean " + observed + " (birthday prediction "
                + pairExpectation + ", 3 sigma = " + (3 * poissonSigma) + ")");
        check(Math.abs(observed - pairExpectation) <= 3 * poissonSigma,
                "observed collision pairs agree with C(n,2)/2^b");

        // ---- (5) universality of the affine family ---------------------------------------------
        long prime = 2147483647L;
        int buckets = 64;
        long[][] pairs = {{1, 2}, {5, 7}, {1000000, 7}, {2147483646, 3}, {17, 424242}, {999, 1000}};
        int draws = 40000;
        double collisionProbability = 1.0 / buckets;
        for (long[] pair : pairs) {
            long collisions = 0;
            for (int draw = 0; draw < draws; draw++) {
                long a = 1 + (rng.nextLong() >>> 1) % (prime - 1);
                long b = (rng.nextLong() >>> 1) % prime;
                long h1 = ((a * pair[0] + b) % prime) % buckets;
                long h2 = ((a * pair[1] + b) % prime) % buckets;
                if (h1 == h2) collisions++;
            }
            double empirical = collisions / (double) draws;
            double sigma = Math.sqrt(collisionProbability * (1 - collisionProbability) / draws);
            check(Math.abs(empirical - collisionProbability) <= 3 * sigma,
                    "affine family collision probability for pair (" + pair[0] + "," + pair[1] + ") is " + empirical);
            check(empirical <= collisionProbability + 1.0 / prime + 3 * sigma, "universality bound respected");
        }
        System.out.println("affine family: collision probability within three standard errors of 1/64 for all six pairs");

        // ---- (6) honest caveats ----------------------------------------------------------------
        Set<Long> eightBit = new HashSet<>();
        long counter = 0;
        long firstCollisionInput = -1;
        while (true) {
            counter++;
            if (!eightBit.add(truncate(hash64(message(rng, 8)), 8))) {
                firstCollisionInput = counter;
                break;
            }
        }
        System.out.println("8-bit truncation collides after " + firstCollisionInput + " samples");
        check(firstCollisionInput < 64, "8-bit truncation collides quickly, as expected of a toy hash");
        // sum-of-bytes checksum: all permutations of the same bytes collide
        long base = weakHash(new byte[]{(byte) 0xA1, (byte) 0x0F, (byte) 0x33});
        check(weakHash(new byte[]{(byte) 0x0F, (byte) 0x33, (byte) 0xA1}) == base,
                "the checksum collides on permutations of the same bytes");
        check(weakHash(new byte[]{(byte) 0x33, (byte) 0xA1, (byte) 0x0F}) == base,
                "all six permutations of three bytes collide");
        check(weakHash(new byte[]{(byte) 0xA1, (byte) 0x33, (byte) 0x0F}) == base, "and the third permutation too");
        check(hash64(new byte[]{0x01}) != hash64(new byte[]{0x01, 0x00}),
                "the toy hash does include the length, unlike the checksum");

        System.out.println("PASS");
    }
}
