/** Task 838 - Implement secure multi-party aggregation for machine learning (additive secret sharing). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Setting: n data-owning parties each hold a private update vector and a single server wants only the SUM
 * (the aggregate gradient an ML aggregator applies). The protocol is the textbook secure-aggregation
 * stack, in four rounds:
 *
 *   1. additive sharing  - party i splits each coordinate into n additive shares summing to v_i[c] over
 *                          F_p and hands share h to holder h; adding the shares pointwise adds the
 *                          secrets pointwise, so the total is the sum of all holders' accumulated values
 *                          b_h. This is the "additive secret sharing" of the task title.
 *   2. pairwise masking  - holder h opens y_h = b_h + SUM_{j>h} m(h,j) - SUM_{j<h} m(j,h), where m(a,b)
 *                          is a field element both holders derive from one global seed. Each y_h is
 *                          uniform on its own, and the masks cancel in SUM_h y_h, so the server's only
 *                          usable quantity is the aggregate.
 *   3. dropout recovery  - before opening, holder h threshold-shares y_h with a t-of-n Shamir sharing.
 *                          A dropped holder's update is therefore recovered from any t live holders,
 *                          while an honest holder still only publishes one field element.
 *   4. MAC verification  - every published y_h carries the SPDZ-style tag alpha * y_h with a globally
 *                          shared alpha, so a holder that publishes y_h + delta is caught unless it also
 *                          guesses alpha, i.e. it succeeds with probability 1/p.
 *
 * Simplifications, stated honestly: this is an in-process toy model - no sockets, no threads, no key
 * material, traffic is counted rather than transmitted, the masks come from splitmix64 instead of a real
 * PRG, alpha is reconstructed by a trusted verifier (real SPDZ checks it against a shared random value
 * and never reveals alpha), and the field prime is kept below 2^31 so products fit in a long.
 *
 * Invariants verified below. Safety: the server's sum equals the true total over *all* parties, including
 * the dropped ones, and any corrupted opening breaks the tag equation. Liveness: aggregation completes
 * with up to n-t dropped holders and is rejected when fewer than t holders answer. Secrecy: the masked
 * openings are uniform and independent of the aggregate - measured by exhaustive enumeration over F_97,
 * not asserted. Cost: rounds, published values and reconstructed values are counted.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every transcript is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long nextBelow(long bound) { return Math.floorMod(nextLong(), bound); }
}

/** Arithmetic in F_p. p stays below 2^31 so a*b fits in a long without a 128-bit multiply. */
final class Field {
    final long p;

    Field(long prime) {
        if (prime < 3) throw new IllegalArgumentException("prime too small");
        this.p = prime;
    }

    long add(long a, long b) { return Math.floorMod(a + b, p); }

    long sub(long a, long b) { return Math.floorMod(a - b, p); }

    long mul(long a, long b) { return Math.floorMod(a, p) * Math.floorMod(b, p) % p; }

    long norm(long a) { return Math.floorMod(a, p); }

    long pow(long base, long exponent) {
        long result = 1, b = norm(base), e = exponent;
        while (e > 0) {
            if ((e & 1L) == 1L) result = mul(result, b);
            b = mul(b, b);
            e >>>= 1;
        }
        return result;
    }

    long inverse(long a) {
        long v = norm(a);
        if (v == 0) throw new ArithmeticException("division by zero in F_" + p);
        return pow(v, p - 2);
    }
}

/** Lagrange interpolation at x = 0 over distinct (public) evaluation points. */
final class Lagrange {
    static long atZero(Field f, int[] xs, long[] ys) {
        if (xs.length != ys.length || xs.length == 0) throw new IllegalArgumentException("bad sample");
        long total = 0;
        for (int i = 0; i < xs.length; i++) {
            long numerator = 1, denominator = 1;
            for (int j = 0; j < xs.length; j++) {
                if (i == j) continue;
                numerator = f.mul(numerator, f.norm(-xs[j]));
                denominator = f.mul(denominator, f.sub(xs[i], xs[j]));
            }
            total = f.add(total, f.mul(ys[i], f.mul(numerator, f.inverse(denominator))));
        }
        return total;
    }
}

/** Additive n-of-n secret sharing: shares[h][c] are uniform and sum to secret[c] over F_p. */
final class AdditiveSharing {
    static long[][] share(Field f, int holders, long[] secret, SplitMix64 rng) {
        if (holders < 2) throw new IllegalArgumentException("additive sharing needs n >= 2");
        long[][] shares = new long[holders][secret.length];
        for (int c = 0; c < secret.length; c++) {
            long accumulated = 0;
            for (int h = 0; h < holders - 1; h++) {
                shares[h][c] = rng.nextBelow(f.p);
                accumulated = f.add(accumulated, shares[h][c]);
            }
            shares[holders - 1][c] = f.sub(secret[c], accumulated);
        }
        return shares;
    }
}

/** A t-of-n Shamir sharing of one d-dimensional vector; shares[h][c] is holder h's piece of coordinate c. */
final class ShamirSharing {
    final Field field;
    final int holders;
    final int threshold;
    final long[][] shares;

    private ShamirSharing(Field field, int holders, int threshold, long[][] shares) {
        this.field = field;
        this.holders = holders;
        this.threshold = threshold;
        this.shares = shares;
    }

    static ShamirSharing deal(Field field, int holders, int threshold, long[] secret, SplitMix64 rng) {
        if (holders < 2) throw new IllegalArgumentException("need at least two holders");
        if (threshold < 2 || threshold > holders) {
            throw new IllegalArgumentException("threshold must satisfy 2 <= t <= n, got t=" + threshold
                    + " n=" + holders);
        }
        long[][] shares = new long[holders][secret.length];
        for (int c = 0; c < secret.length; c++) {
            long[] coefficients = new long[threshold];
            coefficients[0] = field.norm(secret[c]);
            for (int k = 1; k < threshold; k++) coefficients[k] = rng.nextBelow(field.p);
            for (int h = 0; h < holders; h++) shares[h][c] = evaluate(field, coefficients, h + 1);
        }
        return new ShamirSharing(field, holders, threshold, shares);
    }

    /** Horner evaluation of the sharing polynomial at the holder's public point (holder + 1). */
    static long evaluate(Field f, long[] coefficients, long x) {
        long value = 0;
        for (int k = coefficients.length - 1; k >= 0; k--) value = f.add(f.mul(value, x), coefficients[k]);
        return value;
    }

    /** Reconstruct coordinate c from exactly t holders: their points are holder + 1. */
    long reconstruct(int[] subset, int coordinate) {
        if (subset.length != threshold) throw new IllegalArgumentException("need exactly t shares");
        int[] xs = new int[subset.length];
        long[] ys = new long[subset.length];
        for (int i = 0; i < subset.length; i++) {
            xs[i] = subset[i] + 1;
            ys[i] = shares[subset[i]][coordinate];
        }
        return Lagrange.atZero(field, xs, ys);
    }
}

/** Pairwise masks m(a, b), a < b, all derived from one global seed; both holders know them. */
final class MaskMatrix {
    final Field field;
    final int holders;
    private final long[][] masks;

    MaskMatrix(Field field, int holders, long seed) {
        this.field = field;
        this.holders = holders;
        this.masks = new long[holders][holders];
        SplitMix64 rng = new SplitMix64(seed);
        for (int a = 0; a < holders; a++) {
            for (int b = a + 1; b < holders; b++) masks[a][b] = rng.nextBelow(field.p);
        }
    }

    long mask(int a, int b) {
        if (a == b) throw new IllegalArgumentException("no self mask");
        return a < b ? masks[a][b] : masks[b][a];
    }
}

/** One protocol run: the aggregate, the transcript the server sees, and the measured cost. */
final class AggregateRun {
    final long[] total;
    final long[][] opened;
    final long[][] masked;
    final long[][] contribution;
    final long[] tags;
    final long[] liveOnlyTotal;
    final int reconstructed;
    final int publishedValues;
    final int sharesRevealed;
    final int rounds;
    final boolean macVerified;

    AggregateRun(long[] total, long[][] opened, long[][] masked, long[][] contribution, long[] tags,
            long[] liveOnlyTotal, int reconstructed, int publishedValues, int sharesRevealed, int rounds,
            boolean macVerified) {
        this.total = total;
        this.opened = opened;
        this.masked = masked;
        this.contribution = contribution;
        this.tags = tags;
        this.liveOnlyTotal = liveOnlyTotal;
        this.reconstructed = reconstructed;
        this.publishedValues = publishedValues;
        this.sharesRevealed = sharesRevealed;
        this.rounds = rounds;
        this.macVerified = macVerified;
    }
}

/** The four-round aggregator: additive share, mask, threshold-share the masked update, open and verify. */
final class SecureAggregator {
    static final int ROUNDS = 4;

    /**
     * @param inputs      one private vector per party (all of the same dimension)
     * @param threshold   t of the t-of-n Shamir sharing used for dropout recovery
     * @param live        which holders answer; the others dropped out
     * @param maskSeed    seed of the pairwise mask matrix
     * @param macValue    the global MAC key alpha held by the verifier
     * @param tamperAt    holder whose published opening is corrupted, or -1
     * @param tamperBy    delta added to that opening
     * @param guessedAlpha alpha the adversary assumes when forging the tag
     */
    static AggregateRun run(Field f, long[][] inputs, int threshold, boolean[] live, long maskSeed,
            long macValue, int tamperAt, long tamperBy, long guessedAlpha) {
        int n = inputs.length;
        int d = inputs[0].length;
        if (threshold < 2 || threshold > n) throw new IllegalArgumentException("2 <= t <= n required");
        SplitMix64 rng = new SplitMix64(maskSeed ^ 0x2545F4914F6CDD1DL);

        // Round 1: every party additively shares its update; holder h accumulates one share from each party.
        long[][] contribution = new long[n][d];
        for (int i = 0; i < n; i++) {
            if (inputs[i].length != d) throw new IllegalArgumentException("dimension mismatch");
            long[][] shares = AdditiveSharing.share(f, n, inputs[i], rng);
            for (int h = 0; h < n; h++) {
                for (int c = 0; c < d; c++) contribution[h][c] = f.add(contribution[h][c], shares[h][c]);
            }
        }

        // Round 2: pairwise masks make each opening uniform while cancelling in the sum.
        MaskMatrix masks = new MaskMatrix(f, n, maskSeed);
        long[][] masked = new long[n][d];
        for (int h = 0; h < n; h++) {
            for (int c = 0; c < d; c++) {
                long value = contribution[h][c];
                for (int j = h + 1; j < n; j++) value = f.add(value, masks.mask(h, j));
                for (int j = 0; j < h; j++) value = f.sub(value, masks.mask(j, h));
                masked[h][c] = value;
            }
        }

        // Round 3: each holder threshold-shares its masked update so a dropout stays recoverable.
        ShamirSharing[] recoveries = new ShamirSharing[n];
        for (int h = 0; h < n; h++) recoveries[h] = ShamirSharing.deal(f, n, threshold, masked[h], rng);
        int[] helpers = liveHolders(live, n, threshold);

        // Round 4: open the live updates, reconstruct the dropped ones, verify every tag.
        long[][] opened = new long[n][d];
        int reconstructed = 0, published = 0, sharesRevealed = 0;
        for (int h = 0; h < n; h++) {
            if (live[h]) {
                for (int c = 0; c < d; c++) opened[h][c] = masked[h][c];
                if (h == tamperAt) opened[h][0] = f.add(opened[h][0], tamperBy);
                published++;
            } else {
                for (int c = 0; c < d; c++) opened[h][c] = recoveries[h].reconstruct(helpers, c);
                reconstructed++;
                sharesRevealed += threshold;
            }
        }

        long[] total = new long[d];
        long[] liveOnly = new long[d];
        for (int c = 0; c < d; c++) {
            for (int h = 0; h < n; h++) total[c] = f.add(total[c], opened[h][c]);
            for (int h = 0; h < n; h++) if (live[h]) liveOnly[c] = f.add(liveOnly[c], opened[h][c]);
        }

        long[] tags = new long[n];
        if (d == 0) {
            return new AggregateRun(total, opened, masked, contribution, tags, liveOnly, reconstructed,
                    published, sharesRevealed, ROUNDS, true);
        }
        for (int h = 0; h < n; h++) {
            long claimed = opened[h][0];
            long alpha = h == tamperAt ? guessedAlpha : macValue;
            tags[h] = f.mul(f.norm(alpha), f.norm(claimed));
        }
        long tagSum = 0, openSum = 0;
        for (int h = 0; h < n; h++) {
            tagSum = f.add(tagSum, tags[h]);
            openSum = f.add(openSum, opened[h][0]);
        }
        boolean macVerified = tagSum == f.mul(f.norm(macValue), openSum);
        return new AggregateRun(total, opened, masked, contribution, tags, liveOnly, reconstructed,
                published, sharesRevealed, ROUNDS, macVerified);
    }

    /** The t lowest-numbered live holders, or a liveness failure when fewer than t answered. */
    static int[] liveHolders(boolean[] live, int n, int threshold) {
        int[] helpers = new int[threshold];
        int filled = 0;
        for (int h = 0; h < n && filled < threshold; h++) if (live[h]) helpers[filled++] = h;
        if (filled < threshold) {
            throw new IllegalStateException("only " + filled + " of " + n + " holders answered, need "
                    + threshold);
        }
        return helpers;
    }
}

public class Solution {
    static final Field BIG = new Field(1_000_000_007L);
    static final Field SMALL = new Field(97L);

    public static void main(String[] args) {
        int shareChecks = 0;

        // ---- 1. additive sharing: shares sum to the secret and are uniform on their own -----------
        long[] boundarySecrets = {0L, 1L, BIG.p - 1, 42L, 987654321L};
        for (long secret : boundarySecrets) {
            long[][] sh = AdditiveSharing.share(BIG, 5, new long[]{secret}, new SplitMix64(1000 + secret));
            long sum = 0;
            for (int h = 0; h < 5; h++) sum = BIG.add(sum, sh[h][0]);
            check(sum == BIG.norm(secret), "additive shares of " + secret + " must sum back");
            shareChecks++;
        }
        check(shareChecks == boundarySecrets.length, "all boundary secrets round-tripped");
        long[][] twoWay = AdditiveSharing.share(SMALL, 2, new long[]{30}, new SplitMix64(5L));
        check(SMALL.add(twoWay[0][0], twoWay[1][0]) == 30, "the 2-of-2 minimum case must work");

        // ---- 2. a clean run reproduces the plaintext sum and never opens a bare share ------------
        long[][] inputs = {
            {3, 5},
            {7, 11},
            {13, 17},
            {19, 23},
            {29, 31},
        };
        int n = inputs.length;
        boolean[] allLive = allLive(n);
        AggregateRun clean = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, -1, 0, 0);
        long[] oracle = new long[2];
        for (long[] in : inputs) {
            for (int c = 0; c < 2; c++) oracle[c] = BIG.add(oracle[c], in[c]);
        }
        check(Arrays.equals(clean.total, oracle),
                "aggregate must equal the plaintext sum, got " + Arrays.toString(clean.total));
        check(clean.macVerified, "honest openings must satisfy the tag equation");
        check(clean.rounds == 4 && clean.publishedValues == n && clean.reconstructed == 0,
                "a clean run publishes n values in 4 rounds and reconstructs nothing");
        check(Arrays.equals(clean.liveOnlyTotal, clean.total),
                "with every holder live the plain sum of openings is already the aggregate");
        boolean masked = true;
        for (int h = 0; h < n; h++) {
            if (clean.opened[h][0] == clean.contribution[h][0]) masked = false;
        }
        check(masked, "every published value must actually be masked by its pairwise masks");

        // ---- 3. secrecy measurement: one mask makes the opening exactly uniform (F_97) ------------
        Map<Long, Integer> single = new TreeMap<>();
        for (long m = 0; m < SMALL.p; m++) single.merge(SMALL.add(30L, m), 1, Integer::sum);
        check(single.size() == SMALL.p, "the masked opening must cover the whole field");
        boolean once = true;
        for (int count : single.values()) once &= count == 1;
        check(once, "a single mask makes the opening uniform: every value is equally likely");
        Map<String, Integer> joint = pairwiseView(SMALL, 10L, 20L, 7L);
        check(joint.size() == SMALL.p * SMALL.p,
                "two holders' masked view must cover F_97^2 exactly, got " + joint.size());
        boolean allOnce = true;
        for (int count : joint.values()) allOnce &= count == 1;
        check(allOnce, "the pair view is uniform, so it leaks nothing about the aggregate");
        Map<Long, Integer> thirdGivenFirst = new TreeMap<>();
        long b0 = 10, b1 = 20, b2 = 30, m12 = 7;
        for (long m01 = 0; m01 < SMALL.p; m01++) {
            for (long m02 = 0; m02 < SMALL.p; m02++) {
                long y0 = SMALL.add(SMALL.add(b0, m01), m02);
                if (y0 != 5L) continue;
                long y2 = SMALL.sub(SMALL.sub(b2, m02), m12);
                thirdGivenFirst.merge(y2, 1, Integer::sum);
            }
        }
        check(thirdGivenFirst.size() == SMALL.p && thirdGivenFirst.values().stream().allMatch(c -> c == 1),
                "one observed opening leaves the remaining holder's value uniform over F_97");

        // ---- 4. Shamir sharing: any t-1 pieces are uniform, measured over the whole space --------
        Map<String, Integer> histA = shareTupleHistogram(SMALL, 5, 3, 3L);
        Map<String, Integer> histB = shareTupleHistogram(SMALL, 5, 3, 91L);
        check(histA.equals(histB), "the (t-1)-share distribution must not depend on the secret");
        check(histA.size() == SMALL.p * SMALL.p && histA.values().stream().allMatch(c -> c == 1),
                "the t-1 share space must be covered exactly once, got " + histA.size());
        ShamirSharing vector = ShamirSharing.deal(BIG, 5, 3, new long[]{5, 99}, new SplitMix64(11L));
        check(vector.reconstruct(new int[]{0, 1, 2}, 1) == 99
                && vector.reconstruct(new int[]{2, 3, 4}, 1) == 99,
                "any t-subset must interpolate the same secret");

        // ---- 5. dropout tolerance: n - t holders may stop answering ------------------------------
        boolean[] withDropouts = {true, false, true, false, true};
        AggregateRun droppy = SecureAggregator.run(BIG, inputs, 3, withDropouts, 77L, 1234567L, -1, 0, 0);
        check(Arrays.equals(droppy.total, oracle),
                "the dropped holders' updates must still be recovered: " + Arrays.toString(droppy.total));
        check(droppy.reconstructed == 2 && droppy.publishedValues == 3 && droppy.sharesRevealed == 6,
                "two dropped holders cost two reconstructions and six revealed shares");
        check(droppy.macVerified, "a dropout run must still verify its tags");
        check(!Arrays.equals(droppy.liveOnlyTotal, droppy.total),
                "summing only the live openings must be wrong when holders drop out");

        boolean[] tooFew = {false, true, true, false, false};
        expectThrows(() -> SecureAggregator.run(BIG, inputs, 3, tooFew, 77L, 1234567L, -1, 0, 0),
                "fewer than t live holders must abort the aggregation");
        expectThrows(() -> SecureAggregator.run(BIG, inputs, 6, allLive, 77L, 1234567L, -1, 0, 0),
                "a threshold above the party count");
        expectThrows(() -> SecureAggregator.run(BIG, inputs, 1, allLive, 77L, 1234567L, -1, 0, 0),
                "threshold 1 hides nothing");

        // ---- 6. the MAC: a corrupted opening is caught unless alpha is known ---------------------
        AggregateRun guessed = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, 1, 9L, 999L);
        check(!guessed.macVerified, "a wrong alpha guess must be detected");
        AggregateRun clairvoyant = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, 1, 9L,
                1234567L);
        check(clairvoyant.macVerified,
                "an adversary that knows alpha forges successfully, which is why alpha stays secret");
        AggregateRun deltaOne = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, 2, 1L, 0L);
        check(!deltaOne.macVerified, "an opening shifted by one must break the tag equation");
        AggregateRun deltaMinusOne = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, 4,
                BIG.p - 1, 0L);
        check(!deltaMinusOne.macVerified, "a negative shift must break the tag equation too");

        int trials = 2000, forgeries = 0;
        boolean[] fourLive = allLive(4);
        for (int t = 0; t < trials; t++) {
            long guess = new SplitMix64(90000L + t).nextBelow(SMALL.p);
            AggregateRun attempt = SecureAggregator.run(SMALL, new long[][]{{2}, {4}, {6}, {8}}, 3,
                    fourLive, 5150L, 42L, 0, 1L, guess);
            if (attempt.macVerified) forgeries++;
        }
        check(forgeries * 25 <= trials, "the measured forgery rate must stay near 1/p, saw " + forgeries
                + "/" + trials);

        // ---- 7. degenerate shapes, validation and determinism ------------------------------------
        AggregateRun empty = SecureAggregator.run(BIG, new long[][]{{}, {}}, 2, new boolean[]{true, true},
                3L, 5L, -1, 0, 0);
        check(empty.total.length == 0, "an empty feature vector aggregates to nothing");
        AggregateRun scalar = SecureAggregator.run(BIG, new long[][]{{5}, {7}}, 2, new boolean[]{true, true},
                11L, 7L, -1, 0, 0);
        check(scalar.total.length == 1 && scalar.total[0] == 12, "the single-coordinate two-party case");
        expectThrows(() -> AdditiveSharing.share(BIG, 1, new long[]{1}, new SplitMix64(1L)),
                "additive sharing with one holder");
        expectThrows(() -> ShamirSharing.deal(BIG, 4, 3, new long[]{1}, new SplitMix64(1L))
                .reconstruct(new int[]{0, 1}, 0), "reconstruction below t");
        expectThrows(() -> BIG.inverse(0), "inverting zero");
        expectThrows(() -> SecureAggregator.run(BIG, new long[][]{{1, 2}, {3}}, 2, new boolean[]{true, true},
                1L, 1L, -1, 0, 0), "mismatched input dimensions");

        AggregateRun again = SecureAggregator.run(BIG, inputs, 3, allLive, 77L, 1234567L, -1, 0, 0);
        check(Arrays.equals(again.total, clean.total) && again.opened[3][1] == clean.opened[3][1],
                "the same seed must reproduce an identical transcript");
        AggregateRun otherSeed = SecureAggregator.run(BIG, inputs, 3, allLive, 78L, 1234567L, -1, 0, 0);
        check(Arrays.equals(otherSeed.total, clean.total) && otherSeed.opened[3][1] != clean.opened[3][1],
                "a different mask seed keeps the aggregate but changes the openings");

        System.out.println("additive sharing verified for " + shareChecks + " boundary secrets (n-th of n)");
        System.out.println("secrecy: 1 mask -> " + single.size() + " equiprobable openings; 2 masks -> "
                + joint.size() + " equiprobable pair views over F_97; t-1 shares -> " + histA.size()
                + " equiprobable tuples");
        System.out.println("aggregate " + Arrays.toString(clean.total) + " in " + clean.rounds
                + " rounds with " + clean.publishedValues + " published values; dropout run recovered "
                + droppy.reconstructed + " holders from " + droppy.sharesRevealed + " revealed shares");
        System.out.println("MAC forgery rate " + forgeries + "/" + trials + " (ideal 1/p = "
                + (1.0 / SMALL.p) + ")");
        System.out.println("PASS");
    }

    static boolean[] allLive(int n) {
        boolean[] live = new boolean[n];
        Arrays.fill(live, true);
        return live;
    }

    /**
     * Exhaustive enumeration of the two-party view for three holders: y0 = b0 + m01 + m02 and
     * y1 = b1 - m01 + m12, sweeping every (m01, m02) pair over F_p. A map that covers the whole space
     * exactly once is the measured statement "the openings are uniform and independent of the total".
     */
    static Map<String, Integer> pairwiseView(Field f, long b0, long b1, long m12) {
        Map<String, Integer> histogram = new TreeMap<>();
        for (long m01 = 0; m01 < f.p; m01++) {
            for (long m02 = 0; m02 < f.p; m02++) {
                long y0 = f.add(f.add(b0, m01), m02);
                long y1 = f.add(f.sub(b1, m01), m12);
                histogram.merge(y0 + ":" + y1, 1, Integer::sum);
            }
        }
        return histogram;
    }

    /** Histogram of the first t-1 Shamir shares over every coefficient tuple: a full enumeration. */
    static Map<String, Integer> shareTupleHistogram(Field f, int holders, int threshold, long secret) {
        Map<String, Integer> histogram = new TreeMap<>();
        long[] coefficients = new long[threshold];
        coefficients[0] = f.norm(secret);
        long total = 1;
        for (int i = 1; i < threshold; i++) total *= f.p;
        for (long code = 0; code < total; code++) {
            long rest = code;
            for (int i = 1; i < threshold; i++) {
                coefficients[i] = rest % f.p;
                rest /= f.p;
            }
            StringBuilder key = new StringBuilder();
            for (int h = 0; h < threshold - 1; h++) {
                key.append(ShamirSharing.evaluate(f, coefficients, h + 1)).append(',');
            }
            histogram.merge(key.toString(), 1, Integer::sum);
        }
        return histogram;
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
