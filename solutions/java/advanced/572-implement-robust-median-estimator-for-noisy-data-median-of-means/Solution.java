/** Task 572 - Implement robust median estimator for noisy data (median-of-means). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * The median-of-means (MoM) estimator: split the n samples into k contiguous blocks, take the k block
 * means and return their median — concretely the lower middle order statistic, 0-based rank
 * (k-1)/2, so no interpolation is used and the estimator is a deterministic order statistic. MoM
 * trades two constants for robustness: its error is governed by the variance rather than needing
 * sub-Gaussian tails, and it tolerates up to one half of the samples being corrupted with unbounded
 * magnitude. The implementation is deliberately simple (partition, average, order statistic) and the
 * self-test cross-checks every result against an independently written order-statistic computation.
 *
 * Documented checks. (1) Hand values: 99 zeros plus one 1000 with k = 10 gives block means (nine
 * zeros, one 100) whose lower middle is exactly 0, while the sample mean is exactly 10; and
 * {1, 2, 3, 4} with k = 2 gives block means {1.5, 3.5} and the estimator 1.5, while k = 4 reduces MoM
 * to the lower middle sample 2 (not the interpolated 2.5). (2) Equivariance: over 200 random datasets
 * MoM(x + c) = MoM(x) + c and MoM(c x) = c MoM(x) for c > 0 hold exactly, whereas the negative scale
 * case fails on 90 of them, because the median index flips under reversal — the failure is asserted
 * to occur rather than hidden. (3) Heavy tails: a contaminated normal (95% N(0,1) plus 5% N(0,100))
 * with n = 200, k = 20, over 2000 replications, gives MoM an RMS error of 0.202 against 1.579 for the
 * sample mean (7.8 times smaller; the asserted factor is 5), a failure rate Pr(|estimate| > 1) of
 * 0.002 for MoM versus 0.509 for the mean. (4) The Chebyshev-then-binomial bound is checked against
 * the experiment: block means have variance sigma^2/b, so each is at least eps away with probability
 * p <= sigma^2/(b eps^2), and MoM fails only when at least k/2 blocks are bad, giving
 * Pr(failure) <= P(Binomial(k, p) >= k/2) computed exactly by dynamic programming (0.2447 for
 * sigma = 1, eps = 0.5, b = 10, k = 20). The observed failure rate over 4000 replications is 0, far
 * below the bound: the bound holds but is very loose here, which the file states instead of claiming
 * tightness. (5) The 1/2 breakdown point: one-sided corruption of the leading samples leaves MoM
 * intact at 20% and 40% (failure rates 0.0 and 0.0) but defeats it at 60% (failure rate 1.0), which
 * is asserted as the failure it is. A nuance that is documented rather than smoothed over: a
 * *symmetric* 60% corruption does not break MoM (failure rate 0.0) because only six of the twelve
 * corrupted blocks are shifted each way, so a same-direction majority is what the estimator actually
 * cannot survive. (6) Structural checks: MoM equals the independently computed median of its block
 * means for k = 1..10 on a 101-sample dataset, every sample is covered by exactly one block, the
 * size-weighted block means reproduce the sample mean exactly, and permuting the partition by seven
 * positions moves the estimate only from -0.1272 to -0.1880 (MoM depends on the partition, reported).
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws millions of
 * independent normals, so the repo's 16-bit LCG is not used) converted to standard normals by
 * Box-Muller; every printed number is reproducible.
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

        /** Box-Muller from two uniforms (deterministic given the fixed seed). */
        double nextGaussian() {
            double u1 = nextDouble();
            if (u1 < 1e-300) u1 = 1e-300;
            double u2 = nextDouble();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /**
     * Median of the k block means: the lower middle order statistic, i.e. 0-based rank (k-1)/2. No
     * interpolation is used, so the estimator is a deterministic order statistic of the data.
     */
    static double medianOfMeans(double[] data, int k) {
        int n = data.length;
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        double[] means = new double[k];
        int index = 0;
        for (int block = 0; block < k; block++) {
            int from = (int) ((long) block * n / k);
            int to = (int) ((long) (block + 1) * n / k);
            if (to <= from) throw new IllegalArgumentException("empty block, need k <= n");
            double sum = 0.0;
            for (int i = from; i < to; i++) {
                sum += data[i];
                index++;
            }
            means[block] = sum / (to - from);
        }
        if (index != n) throw new AssertionError("the partition must cover every sample exactly once");
        return orderStatistic(means, (k - 1) / 2);
    }

    /** Independent order-statistic computation (0-based rank r). */
    static double orderStatistic(double[] values, int rank) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        return copy[rank];
    }

    static double mean(double[] data) {
        double sum = 0.0;
        for (double v : data) sum += v;
        return sum / data.length;
    }

    static double[] blockMeans(double[] data, int k) {
        int n = data.length;
        double[] means = new double[k];
        for (int block = 0; block < k; block++) {
            int from = (int) ((long) block * n / k);
            int to = (int) ((long) (block + 1) * n / k);
            double sum = 0.0;
            for (int i = from; i < to; i++) sum += data[i];
            means[block] = sum / (to - from);
        }
        return means;
    }

    /** P(Binomial(k, p) >= k/2) computed exactly by dynamic programming. */
    static double failureBound(int k, double p) {
        double[] distribution = new double[k + 1];
        distribution[0] = 1.0;
        for (int trial = 0; trial < k; trial++) {
            double[] next = new double[k + 1];
            for (int i = 0; i <= trial; i++) {
                next[i] += distribution[i] * (1 - p);
                next[i + 1] += distribution[i] * p;
            }
            distribution = next;
        }
        double tail = 0.0;
        for (int i = k - k / 2; i <= k; i++) tail += distribution[i];
        return Math.min(1.0, tail);
    }

    public static void main(String[] args) {
        Rng rng = new Rng(572_2025L);

        // ---- (1) hand values -------------------------------------------------------------------
        double[] corrupted = new double[100];
        corrupted[99] = 1000.0;
        check(medianOfMeans(corrupted, 10) == 0.0, "hand case 1: block means are nine zeros and one 100");
        check(Math.abs(mean(corrupted) - 10.0) <= 1e-12, "hand case 1: the sample mean is 10");
        double[] four = {1, 2, 3, 4};
        check(Math.abs(medianOfMeans(four, 2) - 1.5) <= 1e-12, "hand case 2: block means 1.5 and 3.5");
        // k = n reduces MoM to the lower middle sample, 2 here, not the interpolated 2.5.
        check(Math.abs(medianOfMeans(four, 4) - 2.0) <= 1e-12, "k = n: MoM is the lower middle sample");

        // ---- (2) equivariance ------------------------------------------------------------------
        int shiftFailures = 0;
        int positiveScaleFailures = 0;
        int negativeScaleFailures = 0;
        for (int trial = 0; trial < 200; trial++) {
            int n = 20 + rng.nextInt(60);
            double[] data = new double[n];
            for (int i = 0; i < n; i++) data[i] = rng.nextGaussian() * 3.0 + 1.5;
            int k = 2 + rng.nextInt(Math.min(8, n - 1));
            double base = medianOfMeans(data, k);
            double c = Math.rint(rng.nextDouble() * 20.0) - 10.0;
            double[] shifted = data.clone();
            for (int i = 0; i < n; i++) shifted[i] += c;
            if (Math.abs(medianOfMeans(shifted, k) - (base + c)) > 1e-9) shiftFailures++;
            double s = 0.5 + Math.rint(rng.nextDouble() * 9.0);
            double[] scaled = data.clone();
            for (int i = 0; i < n; i++) scaled[i] *= s;
            if (Math.abs(medianOfMeans(scaled, k) - base * s) > 1e-9) positiveScaleFailures++;
            double[] flipped = data.clone();
            for (int i = 0; i < n; i++) flipped[i] = -flipped[i];
            if (Math.abs(medianOfMeans(flipped, k) + base) > 1e-9) negativeScaleFailures++;
        }
        check(shiftFailures == 0, "MoM is shift equivariant");
        check(positiveScaleFailures == 0, "MoM is scale equivariant for positive factors");
        System.out.println("equivariance: negative scale fails on " + negativeScaleFailures
                + "/200 datasets (the median index flips under reversal)");
        check(negativeScaleFailures > 0, "documented caveat: MoM(x) with x -> -x is not simply negated");

        // ---- (3) heavy tails -------------------------------------------------------------------
        int replications = 2000;
        double sumSquareMom = 0;
        double sumSquareMean = 0;
        int momFailures = 0;
        int meanFailures = 0;
        for (int trial = 0; trial < replications; trial++) {
            double[] data = new double[200];
            for (int i = 0; i < 200; i++) {
                double z = rng.nextGaussian();
                data[i] = rng.nextDouble() < 0.05 ? z * 100.0 : z;
            }
            double estimate = medianOfMeans(data, 20);
            double sampleMean = mean(data);
            sumSquareMom += estimate * estimate;
            sumSquareMean += sampleMean * sampleMean;
            if (Math.abs(estimate) > 1.0) momFailures++;
            if (Math.abs(sampleMean) > 1.0) meanFailures++;
        }
        double rmsMom = Math.sqrt(sumSquareMom / replications);
        double rmsMean = Math.sqrt(sumSquareMean / replications);
        System.out.println("contaminated normal: RMS error MoM " + rmsMom + ", mean " + rmsMean
                + "; failure rate MoM " + (momFailures / (double) replications)
                + ", mean " + (meanFailures / (double) replications));
        check(rmsMean >= 5 * rmsMom, "MoM's RMS error is at least 5 times smaller on the contaminated normal");
        check(momFailures / (double) replications <= 0.05, "MoM's failure rate stays at or below 5%");
        check(meanFailures / (double) replications >= 0.30, "the sample mean's failure rate is at least 30%");

        // ---- (4) the binomial bound ------------------------------------------------------------
        int boundReplications = 4000;
        int n = 200;
        int k = 20;
        double blockSize = n / (double) k;
        double eps = 0.5;
        double pBad = 1.0 / (blockSize * eps * eps); // Chebyshev with sigma = 1
        double bound = failureBound(k, pBad);
        int failures = 0;
        for (int trial = 0; trial < boundReplications; trial++) {
            double[] data = new double[n];
            for (int i = 0; i < n; i++) data[i] = rng.nextGaussian();
            if (Math.abs(medianOfMeans(data, k)) > eps) failures++;
        }
        double observedRate = failures / (double) boundReplications;
        double sigma = Math.sqrt(bound * (1 - bound) / boundReplications);
        System.out.println("binomial bound: p_bad " + pBad + ", bound " + bound + ", observed " + observedRate
                + " (" + ((observedRate - bound) / sigma) + " standard errors from the bound)");
        check(observedRate <= bound + 3 * sigma, "the observed failure rate respects the binomial bound");

        // ---- (5) the 1/2 breakdown point -------------------------------------------------------
        double[] fractions = {0.2, 0.4, 0.6};
        for (double fraction : fractions) {
            int bad = (int) Math.rint(fraction * 200);
            int violations = 0;
            for (int trial = 0; trial < 200; trial++) {
                double[] data = new double[200];
                for (int i = 0; i < 200; i++) data[i] = rng.nextGaussian();
                // One-sided corruption of the leading `bad` samples: this is the hard case, because
                // every corrupted block mean is shifted the same way. (A symmetric corruption of the
                // same size does NOT defeat MoM at 60%: with 12 corrupted blocks split evenly between
                // +1e6 and -1e6 only 6 of them are shifted each way, so the median still lands on a
                // clean block - which is why the test uses a same-direction majority.)
                for (int i = 0; i < bad; i++) data[i] = 1e6;
                if (Math.abs(medianOfMeans(data, 20)) > 1.0) violations++;
            }
            double rate = violations / 200.0;
            System.out.println("one-sided corruption fraction " + fraction + ": failure rate " + rate);
            if (fraction <= 0.4) {
                check(rate <= 0.05, "MoM survives a same-direction corruption fraction of " + fraction);
            } else {
                check(rate >= 0.80, "documented breakdown: a corruption fraction of 1/2 + epsilon defeats MoM");
            }
        }
        // symmetric corruption of 60% does not break MoM, as the comment above explains
        int symmetricViolations = 0;
        for (int trial = 0; trial < 200; trial++) {
            double[] data = new double[200];
            for (int i = 0; i < 200; i++) data[i] = rng.nextGaussian();
            for (int i = 0; i < 120; i++) data[i] = (i % 2 == 0 ? 1 : -1) * 1e6;
            if (Math.abs(medianOfMeans(data, 20)) > 1.0) symmetricViolations++;
        }
        System.out.println("symmetric corruption fraction 0.6: failure rate " + (symmetricViolations / 200.0));
        check(symmetricViolations <= 20, "a symmetric majority is not enough to break the median of means");

        // ---- (6) structural checks -------------------------------------------------------------
        double[] data = new double[101];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextGaussian();
        for (int blocks = 1; blocks <= 10; blocks++) {
            double[] means = blockMeans(data, blocks);
            double expected = orderStatistic(means, (blocks - 1) / 2);
            check(Math.abs(medianOfMeans(data, blocks) - expected) <= 1e-12,
                    "MoM equals the median of its block means for k = " + blocks);
            // The blocks may differ by one sample, so the *size-weighted* average of the block means
            // is the identity that must hold exactly (a plain average would not).
            double weighted = 0.0;
            for (int block = 0; block < blocks; block++) {
                int from = (int) ((long) block * data.length / blocks);
                int to = (int) ((long) (block + 1) * data.length / blocks);
                weighted += (to - from) * means[block];
            }
            check(Math.abs(weighted / data.length - mean(data)) <= 1e-9,
                    "the size-weighted block means average to the sample mean for k = " + blocks);
        }
        // partition sensitivity: a rotation of the data changes the estimate only slightly
        double[] rotated = new double[data.length];
        int shift = 7;
        for (int i = 0; i < data.length; i++) rotated[i] = data[(i + shift) % data.length];
        double base = medianOfMeans(data, 10);
        double rotatedEstimate = medianOfMeans(rotated, 10);
        System.out.println("partition sensitivity: " + base + " versus " + rotatedEstimate
                + " (difference " + Math.abs(base - rotatedEstimate) + ")");
        check(Math.abs(base - rotatedEstimate) <= 1.0, "a different partition moves the estimate only slightly");

        System.out.println("PASS");
    }
}
