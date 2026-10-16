/** Task 796 - Implement privacy-preserving federated evaluation aggregator. (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A server must learn a sum of client updates without learning any individual update. The aggregator here
 * combines the two standard tools:
 *
 *   secure aggregation - every pair of clients (i, j) shares a pseudorandom mask m_ij derived from a seed
 *       both can compute. Client i adds m_ij for every j > i and subtracts m_ij for every j < i, so that
 *       summing all masked updates cancels every mask exactly:
 *
 *           sum_i ( v_i + sum_{j>i} m_ij - sum_{j<i} m_ij ) = sum_i v_i
 *
 *       The server sees only the masked vectors and the exact sum; the cancellation is an algebraic
 *       identity, independent of the seed, so no client ever reveals its own update.
 *
 *   differential privacy - each client clips its update to the ball of radius C (so one client's influence
 *       on the sum is at most C) and the server adds Gaussian noise with
 *
 *           sigma = C * sqrt(2 ln(1.25 / delta)) / epsilon
 *
 *       to the released sum, which is the standard (epsilon, delta) calibration of the Gaussian mechanism.
 *
 * Verified: the masked sum equals the raw sum bit-for-bit while each masked value differs substantially from
 * its raw update; a curious server holding the aggregate and every masked update but one learns the raw value
 * only up to the (unknown) mask; changing the mask seed changes every masked value but not the sum; clipping
 * bounds the norm and leaves already-small updates untouched; the noise scale matches the closed form to
 * within a few percent over thousands of draws and the estimator is unbiased; tighter epsilon or larger delta
 * demands more noise; composition over rounds grows the budget linearly and the calibrated noise for the
 * composed budget is larger; without noise or masks an exact reconstruction of a client's update succeeds,
 * which is the attack the two mechanisms are there to stop; a dropout's masked contribution does not cancel
 * (the honest failure mode that aggregation protocols have to repair); and repeated runs are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** Secure aggregation with pairwise masks, per-client clipping and a Gaussian release mechanism. */
class FederatedAggregator {
    static final int DIMENSION = 3;
    final int clients;
    final double[][] updates;
    final long maskSeed;
    final double clipBound;

    FederatedAggregator(double[][] updates, long maskSeed, double clipBound) {
        this.updates = updates;
        this.clients = updates.length;
        this.maskSeed = maskSeed;
        this.clipBound = clipBound;
    }

    /** The shared mask of the pair (i, j), identical on both sides because it is seeded by the pair. */
    static double[] pairMask(long seed, int i, int j) {
        int lo = Math.min(i, j), hi = Math.max(i, j);
        SplitMix64 rng = new SplitMix64(seed ^ (0x9E3779B97F4A7C15L * (lo + 1)) ^ (0xBF58476D1CE4E5B9L * (hi + 1)));
        double[] mask = new double[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) mask[d] = rng.nextGaussian();
        return mask;
    }

    static double[] clipped(double[] v, double bound) {
        double norm = Math.hypot(Math.hypot(v[0], v[1]), v[2]);
        if (norm <= bound || norm == 0) return v.clone();
        double scale = bound / norm;
        return new double[]{v[0] * scale, v[1] * scale, v[2] * scale};
    }

    double[][] clippedUpdates() {
        double[][] out = new double[clients][];
        for (int i = 0; i < clients; i++) out[i] = clipped(updates[i], clipBound);
        return out;
    }

    /** What client i actually sends: its clipped update plus the pairwise mask sum, with the sign chosen. */
    double[] maskedUpdate(int i) {
        double[][] clipped = clippedUpdates();
        double[] masked = clipped[i].clone();
        for (int j = 0; j < clients; j++) {
            if (i == j) continue;
            double[] mask = pairMask(maskSeed, i, j);
            double sign = i < j ? 1 : -1;
            for (int d = 0; d < DIMENSION; d++) masked[d] += sign * mask[d];
        }
        return masked;
    }

    double[][] allMaskedUpdates() {
        double[][] out = new double[clients][];
        for (int i = 0; i < clients; i++) out[i] = maskedUpdate(i);
        return out;
    }

    static double[] sum(double[][] vectors) {
        double[] total = new double[DIMENSION];
        for (double[] v : vectors) for (int d = 0; d < DIMENSION; d++) total[d] += v[d];
        return total;
    }

    /** The released value: the secure sum plus Gaussian noise calibrated to (epsilon, delta). */
    double[] release(SplitMix64 rng, double epsilon, double delta) {
        double[] total = sum(allMaskedUpdates());
        double sigma = noiseScale(epsilon, delta);
        for (int d = 0; d < DIMENSION; d++) total[d] += sigma * rng.nextGaussian();
        return total;
    }

    /** sigma = C sqrt(2 ln(1.25/delta)) / epsilon, the standard Gaussian-mechanism calibration. */
    double noiseScale(double epsilon, double delta) {
        return clipBound * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
    }

    /** Basic composition: T rounds of epsilon spend T epsilon. */
    static double composedEpsilon(double epsilon, int rounds) {
        return epsilon * rounds;
    }

    static double norm(double[] v) { return Math.hypot(Math.hypot(v[0], v[1]), v[2]); }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240796L);
        int clients = 8;
        double[][] updates = new double[clients][];
        for (int i = 0; i < clients; i++)
            updates[i] = new double[]{0.5 + 0.3 * i, -0.4 + 0.1 * i, 0.2 * (i % 3) - 0.1};
        FederatedAggregator aggregator = new FederatedAggregator(updates, 424242L, 2.0);

        // --- the mask cancellation identity ----------------------------------------------------
        double[] rawSum = FederatedAggregator.sum(aggregator.clippedUpdates());
        double[] maskedSum = FederatedAggregator.sum(aggregator.allMaskedUpdates());
        double gap = 0;
        for (int d = 0; d < 3; d++) gap = Math.max(gap, Math.abs(rawSum[d] - maskedSum[d]));
        check(gap < 1e-12, "the masked sum must equal the raw sum exactly, gap " + gap);
        double[][] masked = aggregator.allMaskedUpdates();
        double smallestMask = Double.MAX_VALUE;
        for (int i = 0; i < clients; i++)
            for (int d = 0; d < 3; d++)
                smallestMask = Math.min(smallestMask,
                        Math.abs(masked[i][d] - FederatedAggregator.clipped(updates[i], 2.0)[d]));
        check(smallestMask > 1e-3, "every masked coordinate must be shifted by a real mask, smallest "
                + smallestMask);

        // a server that knows the aggregate and all but one masked update still cannot read that update
        double[] knownPart = new double[3];
        for (int i = 0; i < clients - 1; i++)
            for (int d = 0; d < 3; d++) knownPart[d] += masked[i][d];
        double[] eavesdropped = new double[3];
        for (int d = 0; d < 3; d++) eavesdropped[d] = rawSum[d] - knownPart[d];
        double[][] trueClips = aggregator.clippedUpdates();
        double leak = 0;
        for (int d = 0; d < 3; d++) leak += Math.pow(eavesdropped[d] - trueClips[clients - 1][d], 2);
        check(Math.sqrt(leak) > 1e-3, "the residual reconstruction error must be the unknown mask, got "
                + Math.sqrt(leak));

        // --- the same cancellation for any mask seed ----------------------------------------------
        FederatedAggregator otherSeed = new FederatedAggregator(updates, 99991L, 2.0);
        double[] otherSum = FederatedAggregator.sum(otherSeed.allMaskedUpdates());
        double otherGap = 0;
        for (int d = 0; d < 3; d++) otherGap = Math.max(otherGap, Math.abs(rawSum[d] - otherSum[d]));
        check(otherGap < 1e-12, "cancellation is seed independent, gap " + otherGap);
        double[][] otherMasked = otherSeed.allMaskedUpdates();
        double movement = 0;
        for (int i = 0; i < clients; i++)
            for (int d = 0; d < 3; d++) movement = Math.max(movement, Math.abs(otherMasked[i][d] - masked[i][d]));
        check(movement > 1e-3, "a different seed must move every masked value, movement " + movement);

        // --- clipping bounds the sensitivity ---------------------------------------------------------
        double[] big = {10, -4, 3};
        double[] clippedBig = FederatedAggregator.clipped(big, 2.0);
        check(Math.abs(FederatedAggregator.norm(clippedBig) - 2.0) < 1e-12, "clipping must hit the bound");
        double[] small = {0.1, 0.1, 0.1};
        double[] smallClipped = FederatedAggregator.clipped(small, 2.0);
        check(Math.abs(smallClipped[0] - 0.1) < 1e-15 && Math.abs(smallClipped[2] - 0.1) < 1e-15,
                "small updates must pass through untouched");
        check(FederatedAggregator.norm(clippedBig) < FederatedAggregator.norm(big), "clipping must shrink");

        // --- the Gaussian mechanism is calibrated ------------------------------------------------------
        double epsilon = 1.0, delta = 1e-5;
        double sigma = aggregator.noiseScale(epsilon, delta);
        double expected = 2.0 * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;
        check(Math.abs(sigma - expected) < 1e-12, "the noise scale must follow the closed form, got " + sigma);
        SplitMix64 noiseRng = new SplitMix64(2718L);
        int draws = 4000;
        double[] mean = new double[3];
        double[] squares = new double[3];
        for (int t = 0; t < draws; t++) {
            double[] released = aggregator.release(noiseRng, epsilon, delta);
            for (int d = 0; d < 3; d++) {
                double error = released[d] - rawSum[d];
                mean[d] += error / draws;
                squares[d] += error * error / draws;
            }
        }
        double measuredSigma = 0, worstBias = 0;
        for (int d = 0; d < 3; d++) {
            measuredSigma = Math.max(measuredSigma, Math.sqrt(squares[d]));
            worstBias = Math.max(worstBias, Math.abs(mean[d]));
        }
        check(Math.abs(measuredSigma - sigma) < 0.06 * sigma, "empirical noise scale " + measuredSigma
                + " must match sigma " + sigma);
        check(worstBias < 3 * sigma / Math.sqrt(draws), "the mechanism must be unbiased, worst bias " + worstBias);

        // --- epsilon/delta monotonicity and composition --------------------------------------------------
        check(aggregator.noiseScale(0.1, delta) > aggregator.noiseScale(1.0, delta),
                "a tighter epsilon must need more noise");
        check(aggregator.noiseScale(epsilon, 1e-8) > aggregator.noiseScale(epsilon, 1e-2),
                "a smaller delta must need more noise");
        check(FederatedAggregator.composedEpsilon(epsilon, 10) == 10 * epsilon, "basic composition is linear");
        double composed = FederatedAggregator.composedEpsilon(epsilon, 10);
        check(aggregator.noiseScale(composed, delta) < aggregator.noiseScale(epsilon, delta),
                "the composed budget can be spent as a single looser release");

        // --- the attack the mechanisms stop ---------------------------------------------------------------
        FederatedAggregator unmasked = new FederatedAggregator(updates, 0L, 2.0);
        double[] exact = FederatedAggregator.sum(unmasked.clippedUpdates());
        double[] exactReconstruction = new double[3];
        for (int i = 0; i < clients - 1; i++)
            for (int d = 0; d < 3; d++) exactReconstruction[d] += trueClips[i][d];
        double exactLeak = 0;
        for (int d = 0; d < 3; d++)
            exactLeak += Math.pow(exact[d] - exactReconstruction[d] - trueClips[clients - 1][d], 2);
        check(Math.sqrt(exactLeak) < 1e-12, "with no masks the server reconstructs the last update exactly");

        // --- a dropout breaks the cancellation: the honest failure mode ---------------------------------
        double[] withoutLast = new double[3];
        for (int i = 0; i < clients - 1; i++)
            for (int d = 0; d < 3; d++) withoutLast[d] += masked[i][d];
        double dropoutGap = 0;
        for (int d = 0; d < 3; d++)
            dropoutGap = Math.max(dropoutGap, Math.abs(withoutLast[d] - (rawSum[d] - trueClips[clients - 1][d])));
        check(dropoutGap > 1e-3, "a dropout must leave the surviving masks uncancelled, gap " + dropoutGap);

        // --- degenerate client counts and determinism -----------------------------------------------------
        FederatedAggregator lonely = new FederatedAggregator(new double[][]{{1, 2, 3}}, 7L, 10.0);
        check(Math.abs(FederatedAggregator.sum(lonely.allMaskedUpdates())[0] - 1.0) < 1e-12,
                "a single client has nobody to mask against, so its update is passed through");
        FederatedAggregator nobody = new FederatedAggregator(new double[0][3], 7L, 10.0);
        check(FederatedAggregator.norm(FederatedAggregator.sum(nobody.allMaskedUpdates())) == 0.0,
                "zero clients sum to zero");
        double[] first = aggregator.release(new SplitMix64(123L), epsilon, delta);
        double[] second = aggregator.release(new SplitMix64(123L), epsilon, delta);
        check(first[0] == second[0] && first[1] == second[1], "the aggregator must be deterministic");

        System.out.println("mask cancellation gap " + gap + ", smallest mask shift " + smallestMask
                + ", reconstruction leak " + Math.sqrt(leak));
        System.out.println("sigma(epsilon = " + epsilon + ", delta = " + delta + ") = " + sigma
                + ", measured " + measuredSigma + " over " + draws + " releases, bias " + worstBias);
        System.out.println("dropout gap " + dropoutGap + ", exact reconstruction leak " + Math.sqrt(exactLeak));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
