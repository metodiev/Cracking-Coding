/** Task 945 - Implement certified random number generator with proof of uniformity (small demo).
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A randomness service that ships a certificate. The core is splitmix64 - a bijective 64-bit finaliser -
 *  plus Lemire's multiply-shift bounded draw: the 64x64 -> 128 bit product is formed with an unsigned high
 *  word, the low word decides whether the draw is rejected, and the high word is returned, so the output is
 *  uniform on [0, n) with no modulo bias and a rejection probability below n / 2^64. Uniformity is *proved*
 *  where it can be: a 16-bit LCG of full period 65536 is expanded exhaustively through a 16-bit replica of
 *  the very same transform, so every output can be counted exactly rather than sampled, and a low-entropy
 *  counter source is shown to fail the identical statistical certificate.
 *  Simplified: the by-exhaustion proof covers a 16-bit state space, the 64-bit generator is certified only
 *  statistically (chi-square with a Wilson-Hilferty critical value), and no cryptographic strength is claimed.
 *  Verified: the period is exactly 65536 with no repeats, power-of-two bounds never reject and hit every
 *  value 4096 times, exhaustion counts differ by at most one, the 10-bin, 4! permutation and 16x16 pairwise
 *  certificates pass, counts reconcile with sum(O^2/E) - N, and the biased source is rejected.
 */
import java.util.*;
import java.util.function.LongSupplier;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    /** splitmix64: 64-bit state, bijective finaliser, the deterministic source of every draw. */
    static final class SplitMix64 implements LongSupplier {
        long state;
        SplitMix64(long seed) { state = seed; }
        public long getAsLong() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
    }

    static long unsignedMultiplyHigh(long a, long b) {
        return Math.multiplyHigh(a, b) + ((a >> 63) & b) + ((b >> 63) & a);
    }

    /** Lemire's bounded draw with rejection: unbiased by construction, rejection rate below n / 2^64. */
    static final class Sampler {
        final LongSupplier source;
        long calls, rejections;

        Sampler(LongSupplier source) { this.source = source; }

        long bounded(long n) {
            if (n <= 0) throw new IllegalArgumentException("bound must be positive: " + n);
            long threshold = Long.remainderUnsigned(-n, n);          // 2^64 mod n, unsigned
            while (true) {
                long x = source.getAsLong();
                long low = x * n;
                calls++;
                if (Long.compareUnsigned(low, threshold) < 0) {
                    rejections++;                                    // the low word sits in the biased tail
                    continue;
                }
                return unsignedMultiplyHigh(x, n);
            }
        }

        double rejectionRate() { return calls == 0 ? 0 : (double) rejections / calls; }
    }

    /** 16-bit replica of the same multiply-shift transform, small enough to exhaust the whole state space. */
    static final class SmallSampler {
        static final int SPACE = 1 << 16;
        final int bound, threshold;
        long calls, rejections;
        long[] counts;

        SmallSampler(int bound) {
            this.bound = bound;
            this.threshold = SPACE % bound;
            this.counts = new long[bound];
        }

        /** Returns the draw, or -1 when the value is rejected. */
        int draw(long x) {
            long product = x * bound;
            calls++;
            if ((int) (product & 0xFFFF) < threshold) {
                rejections++;
                return -1;
            }
            int value = (int) (product >>> 16);
            counts[value]++;
            return value;
        }
    }

    /** Wilson-Hilferty approximation of the upper chi-square critical value at normal quantile z. */
    static double chiSquareCritical(int df, double z) {
        double term = 2.0 / (9.0 * df), cube = 1 - term + z * Math.sqrt(term);
        return df * cube * cube * cube;
    }

    static double chiSquare(long[] counts, double expected) {
        double statistic = 0;
        for (long c : counts) statistic += (c - expected) * (c - expected) / expected;
        return statistic;
    }

    static long[] histogram(Sampler sampler, long n, int draws) {
        long[] counts = new long[(int) n];
        for (int i = 0; i < draws; i++) counts[(int) sampler.bounded(n)]++;
        return counts;
    }

    static long lcg(long state) { return (1103515245L * state + 12345L) & 0xFFFF; }

    /** Brent's cycle detection: returns {mu, lambda} for the LCG state sequence. */
    static long[] brent(long seed) {
        long power = 1, lambda = 1, tortoise = lcg(seed), hare = lcg(lcg(seed));
        while (tortoise != hare) {
            if (power == lambda) { tortoise = hare; power *= 2; lambda = 0; }
            hare = lcg(hare);
            lambda++;
        }
        long mu = 0;
        tortoise = seed;
        hare = seed;
        for (long i = 0; i < lambda; i++) hare = lcg(hare);
        while (tortoise != hare) { tortoise = lcg(tortoise); hare = lcg(hare); mu++; }
        return new long[] {mu, lambda};
    }

    /** A low-entropy source: only eight distinct words ever leave it, all far below 2^64. */
    static final class Counter implements LongSupplier {
        long value = 1;
        public long getAsLong() {
            long out = value;
            value = value == 8 ? 1 : value + 1;
            return out;
        }
    }

    static int[] shuffled(int size, Sampler sampler) {
        int[] permutation = new int[size];
        for (int i = 0; i < size; i++) permutation[i] = i;
        for (int i = size - 1; i > 0; i--) {
            int j = (int) sampler.bounded(i + 1);
            int swap = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = swap;
        }
        return permutation;
    }

    /** Factorial-number-system index of a permutation, a bijection onto [0, n!). */
    static int permutationIndex(int[] permutation) {
        int index = 0;
        for (int i = 0; i < permutation.length; i++) {
            int smaller = 0;
            for (int j = i + 1; j < permutation.length; j++) if (permutation[j] < permutation[i]) smaller++;
            index = index * (permutation.length - i) + smaller;
        }
        return index;
    }

    static final class Certificate {
        final String source;
        final long bins, draws, rejections;
        final double statistic, critical;
        final boolean certified;

        Certificate(String source, long bins, long draws, long rejections, double statistic, double critical) {
            this.source = source; this.bins = bins; this.draws = draws; this.rejections = rejections;
            this.statistic = statistic; this.critical = critical; this.certified = statistic < critical;
        }

        String line() {
            return source + ": " + bins + " bins over " + draws + " draws, chi2 "
                    + String.format("%.2f", statistic) + " < " + String.format("%.2f", critical) + " -> "
                    + (certified ? "certified" : "REJECTED");
        }
    }

    static Certificate certify(String name, LongSupplier source, int bins, int draws) {
        Sampler sampler = new Sampler(source);
        long[] counts = histogram(sampler, bins, draws);
        return new Certificate(name, bins, draws, sampler.rejections,
                chiSquare(counts, (double) draws / bins), chiSquareCritical(bins - 1, 3.0902));
    }

    public static void main(String[] args) {
        long first = new SplitMix64(0).getAsLong();
        check(first == 0xE220A8397B1DCDAFL && new SplitMix64(7).getAsLong() != new SplitMix64(8).getAsLong()
                        && new SplitMix64(42).getAsLong() == new SplitMix64(42).getAsLong(),
                "splitmix64 must reproduce the published first output for seed 0: " + Long.toHexString(first));

        long[] cycle = brent(0);
        boolean[] visited = new boolean[1 << 16];
        long state = 0;
        int repeats = 0;
        for (int i = 0; i < (1 << 16); i++) {
            state = lcg(state);
            if (visited[(int) state]) repeats++;
            visited[(int) state] = true;
        }
        check(cycle[0] == 0 && cycle[1] == 1 << 16 && repeats == 0 && state == 0,
                "the LCG must have a pure cycle of length 65536: " + cycle[0] + "/" + cycle[1] + ", repeats "
                        + repeats);

        SmallSampler sixteen = new SmallSampler(16), ten = new SmallSampler(10);
        long walk = 0;
        for (int i = 0; i < (1 << 16); i++) {
            walk = lcg(walk);
            sixteen.draw(walk);
            ten.draw(walk);
        }
        long min = Long.MAX_VALUE, max = 0;
        for (long c : sixteen.counts) {
            min = Math.min(min, c);
            max = Math.max(max, c);
        }
        long tenMin = Long.MAX_VALUE, tenMax = 0;
        for (long c : ten.counts) {
            tenMin = Math.min(tenMin, c);
            tenMax = Math.max(tenMax, c);
        }
        check(sixteen.rejections == 0 && sixteen.calls == 1 << 16 && min == 4096 && max == 4096
                        && ten.rejections == 6 && tenMin == 6553 && tenMax == 6553
                        && Long.remainderUnsigned(-16L, 16L) == 0 && Long.remainderUnsigned(-3L, 3L) == 1,
                "by exhaustion the rejection must flatten the partition: 16 bins at [" + min + ", " + max
                        + "], 10 bins at [" + tenMin + ", " + tenMax + "] after " + ten.rejections + " rejections");

        Certificate tenCert = certify("splitmix64", new SplitMix64(0x5EED), 10, 200_000);
        long[] counts = histogram(new Sampler(new SplitMix64(0x5EED)), 10, 200_000);
        double rederived = 0;
        for (long c : counts) rederived += (double) c * c / 20_000 - 20_000;
        check(tenCert.certified && tenCert.statistic > 0 && tenCert.statistic < tenCert.critical
                        && Math.abs(tenCert.statistic - rederived) < 1e-9 && tenCert.rejections <= 1,
                "a 10-bin draw must be certified and reconcile with sum(O^2/E) - N: " + tenCert.statistic
                        + " vs " + rederived);

        Sampler shuffler = new Sampler(new SplitMix64(11));
        long[] permutations = new long[24];
        for (int i = 0; i < 24_000; i++) permutations[permutationIndex(shuffled(4, shuffler))]++;
        long shuffleTotal = 0;
        for (long c : permutations) shuffleTotal += c;
        double shuffleStatistic = chiSquare(permutations, 1_000);
        check(shuffleTotal == 24_000 && shuffleStatistic < chiSquareCritical(23, 3.0902)
                        && permutationIndex(new int[] {0, 1, 2, 3}) == 0
                        && permutationIndex(new int[] {3, 2, 1, 0}) == 23,
                "the shuffle must hit all 24 permutations evenly: " + shuffleStatistic + " of 24,000 draws");

        Sampler pairs = new Sampler(new SplitMix64(99));
        long[] cells = new long[256];
        for (int i = 0; i < 64_000; i++) {
            int left = (int) pairs.bounded(16), right = (int) pairs.bounded(16);
            cells[left * 16 + right]++;
        }
        double pairStatistic = chiSquare(cells, 250);
        check(pairStatistic < chiSquareCritical(255, 3.0902),
                "the 16x16 pairwise table must show no serial structure: " + pairStatistic);

        Certificate biased = certify("period-8 counter", new Counter(), 10, 65_536);
        Sampler rejected = new Sampler(new Counter());
        for (int i = 0; i < 65_536; i++) rejected.bounded(10);
        check(!biased.certified && biased.statistic > 1_000 && rejected.calls == 65_536
                        && rejected.rejections == 0,
                "a low-entropy source must fail the identical certificate: " + biased.line());

        Sampler boundary = new Sampler(new SplitMix64(3));
        boolean threw = false;
        try { boundary.bounded(0); } catch (IllegalArgumentException e) { threw = true; }
        long[] two = histogram(new Sampler(new SplitMix64(5)), 2, 40_000);
        check(boundary.bounded(1) == 0 && two[0] > 19_000 && two[1] > 19_000 && boundary.bounded(2) <= 1
                        && boundary.bounded(Long.MAX_VALUE) >= 0 && boundary.bounded(1L << 62) >= 0 && threw,
                "n = 1, n = 2 and huge bounds must behave while n = 0 is rejected: " + Arrays.toString(two));

        Sampler tight = new Sampler(new SplitMix64(12345));
        for (int i = 0; i < 5_000; i++) tight.bounded(3);
        check(tight.rejectionRate() <= 0.5 && tight.calls >= 5_000 && unsignedMultiplyHigh(-1L, -1L) == -2L
                        && unsignedMultiplyHigh(1L << 63, 2L) == 1L
                        && Math.multiplyHigh(1L << 63, 2L) == -1L,
                "rejection bookkeeping and the unsigned 128-bit product must be exact: rate "
                        + tight.rejectionRate() + ", hi(-1,-1) " + unsignedMultiplyHigh(-1L, -1L));

        Sampler shufflerAgain = new Sampler(new SplitMix64(11));
        long[] permutationsAgain = new long[24];
        for (int i = 0; i < 24_000; i++) permutationsAgain[permutationIndex(shuffled(4, shufflerAgain))]++;
        check(certify("splitmix64", new SplitMix64(0x5EED), 10, 200_000).statistic == tenCert.statistic
                        && chiSquare(permutationsAgain, 1_000) == shuffleStatistic
                        && chiSquareCritical(9, 3.0902) > 27 && chiSquareCritical(9, 3.0902) < 29
                        && chiSquareCritical(23, 3.0902) > 49 && chiSquareCritical(23, 3.0902) < 51,
                "the generator, statistic and critical value must all be reproducible");

        System.out.println("certificates: " + tenCert.line() + " | 4! permutations chi2 "
                + String.format("%.2f", shuffleStatistic) + " | 16x16 pairs chi2 "
                + String.format("%.2f", pairStatistic));
        System.out.println("negative control: " + biased.line());
        System.out.println("proof by exhaustion: period " + cycle[1] + ", 16 bins at exactly 4096, 10 bins in ["
                + tenMin + ", " + tenMax + "], splitmix64(0) = " + Long.toHexString(first));
        System.out.println("PASS");
    }
}
