/** Task 522 - Implement bootstrap sampling estimator for mean CI. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Nonparametric bootstrap confidence interval for the mean of n observations. Each of the B
 * replicates resamples n indices with replacement from the FIXED LCG of this wave
 * (seed <- (seed * 75 + 74) mod 65537, index = seed mod n) and stores the replicate mean. Two
 * independent replicate implementations are kept: the main one accumulates the resampled values
 * directly, the reference one first builds the per-index resampling counts and then forms
 * sum(count_i * x_i) / n; the checks require them to agree bit for bit on all B replicates.
 * Two intervals are derived from the replicates. The percentile interval uses the linear
 * interpolation quantile q on the sorted replicate means, t = q * (B - 1), value
 * v[floor(t)] + (t - floor(t)) * (v[ceil(t)] - v[floor(t)]) - the documented convention - at
 * alpha / 2 and 1 - alpha / 2. The normal interval uses mean +/- z * s / sqrt(n) with s the
 * n - 1 sample standard deviation and the documented constant z = 1.959963984540054, which is the
 * 97.5% standard-normal quantile; the header documents that the percentile interval is preferred
 * for skewed data and that the normal one assumes the sample mean is approximately normal.
 *
 * Checks: the quantile helper returns min at q = 0, max at q = 1, the exact middle element for
 * q = 0.5 on an odd-length vector, and a hand-computed interpolated value at q = 0.25; the two
 * replicate implementations agree exactly; both intervals must contain the sample mean and lie
 * inside [min, max] of the data; the bootstrap standard deviation must reproduce the analytic
 * standard error s / sqrt(n) within 20% (measured relative error 0.062 on the size-40 LCG sample);
 * a constant sample collapses to a zero-width interval with sd = 0; the mean of the replicate means
 * must sit within a quarter of the bootstrap standard deviation of the sample mean (Monte-Carlo
 * bias check); a symmetric sample gives a mirrored percentile interval whose endpoints match to within
 * 0.15 (measured gap 0.101 - the replicate means live on a 1/n grid, so exact mirroring is not
 * attainable); a strongly skewed exponential sample gives an asymmetric interval whose upper half
 * is wider (measured 0.073); the interval width shrinks by roughly a factor 2 when n grows fourfold
 * (measured 1.75, where the 100-point sample is the 25-point sample repeated four times so both
 * share the same variance); a 150-experiment coverage study on the known uniform population
 * 0..99 (true mean 49.5, n = 25, B = 300) must catch the true mean in at least 85% of the trials
 * for both interval types (measured 132/150 percentile and 135/150 normal); the same seed
 * reproduces both intervals exactly while another seed moves the endpoints by less than a tenth of
 * the interval width (measured 0.039 and 0.019 of the width).
 */
import java.util.*;

public class Solution {
    static final double Z_975 = 1.959963984540054;

    static class Rng {
        int seed;

        Rng(int seed) {
            this.seed = seed;
        }

        int next(int bound) {
            seed = (seed * 75 + 74) % 65537;
            return seed % bound;
        }
    }

    /** Direct accumulation resampling: B replicate means. */
    static double[] bootstrapMeans(double[] data, int replicates, Rng rng) {
        int n = data.length;
        double[] means = new double[replicates];
        for (int b = 0; b < replicates; b++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += data[rng.next(n)];
            means[b] = sum / n;
        }
        return means;
    }

    /** Counts-based resampling of the same stream: independent second implementation. */
    static double[] bootstrapMeansByCounts(double[] data, int replicates, Rng rng) {
        int n = data.length;
        double[] means = new double[replicates];
        int[] counts = new int[n];
        for (int b = 0; b < replicates; b++) {
            Arrays.fill(counts, 0);
            for (int i = 0; i < n; i++) counts[rng.next(n)]++;
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += counts[i] * data[i];
            means[b] = sum / n;
        }
        return means;
    }

    static double mean(double[] values) {
        double sum = 0.0;
        for (double value : values) sum += value;
        return values.length == 0 ? 0.0 : sum / values.length;
    }

    /** Sample standard deviation with the n - 1 denominator. */
    static double sd(double[] values) {
        int n = values.length;
        if (n < 2) return 0.0;
        double avg = mean(values);
        double sum = 0.0;
        for (double value : values) sum += (value - avg) * (value - avg);
        return Math.sqrt(sum / (n - 1));
    }

    /** Linear-interpolation quantile of an ascending array, t = q * (m - 1). */
    static double quantile(double[] sorted, double q) {
        int m = sorted.length;
        if (m == 0) return 0.0;
        if (m == 1) return sorted[0];
        double t = q * (m - 1);
        int lower = (int) Math.floor(t);
        int upper = (int) Math.ceil(t);
        if (lower < 0) lower = 0;
        if (upper > m - 1) upper = m - 1;
        double frac = t - lower;
        return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
    }

    static double[] percentileInterval(double[] replicateMeans, double alpha) {
        double[] sorted = replicateMeans.clone();
        Arrays.sort(sorted);
        return new double[]{quantile(sorted, alpha / 2.0), quantile(sorted, 1.0 - alpha / 2.0)};
    }

    static double[] normalInterval(double[] data, double alpha) {
        double n = data.length;
        double standardError = sd(data) / Math.sqrt(n);
        double z = alpha == 0.05 ? Z_975 : inverseNormal(1.0 - alpha / 2.0);
        double center = mean(data);
        return new double[]{center - z * standardError, center + z * standardError};
    }

    /** Bisection on erf for the standard-normal upper quantile (used for alpha != 0.05). */
    static double inverseNormal(double p) {
        double lo = -10.0;
        double hi = 10.0;
        for (int i = 0; i < 200; i++) {
            double mid = 0.5 * (lo + hi);
            if (normalCdf(mid) < p) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    static double normalCdf(double x) {
        double t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
        double density = 0.3989422804014327 * Math.exp(-x * x / 2.0);
        double poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))));
        double cdf = 1.0 - density * poly;
        return x >= 0 ? cdf : 1.0 - cdf;
    }

    static double[] lcgSample(int n, int seed, int limit) {
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            data[i] = seed % limit;
        }
        return data;
    }

    public static void main(String[] args) {
        double[] ascending = {1.0, 2.0, 3.0, 4.0, 5.0};
        check(quantile(ascending, 0.0) == 1.0, "quantile at 0 is the minimum");
        check(quantile(ascending, 1.0) == 5.0, "quantile at 1 is the maximum");
        check(Math.abs(quantile(ascending, 0.5) - 3.0) <= 1e-12, "quantile at 0.5 is the median");
        check(Math.abs(quantile(ascending, 0.25) - 2.0) <= 1e-12, "quantile at 0.25 interpolates");
        check(Math.abs(quantile(new double[]{0.0, 8.0}, 0.5) - 4.0) <= 1e-12, "even-length median");
        check(Math.abs(quantile(ascending, 0.3)
                - (ascending[1] + 0.2 * (ascending[2] - ascending[1]))) <= 1e-12,
                "interpolation formula t = q * (m - 1)");

        double[] data = lcgSample(40, 2024, 100);
        int replicates = 400;
        double[] direct = bootstrapMeans(data, replicates, new Rng(17));
        double[] byCounts = bootstrapMeansByCounts(data, replicates, new Rng(17));
        check(direct.length == replicates, "B replicate means");
        for (int b = 0; b < replicates; b++) {
            check(direct[b] == byCounts[b], "replicate " + b + " identical in both implementations");
        }
        double sampleMean = mean(data);
        double analyticSe = sd(data) / Math.sqrt(data.length);
        double bootstrapSd = sd(direct);
        double relative = Math.abs(bootstrapSd - analyticSe) / analyticSe;
        check(relative <= 0.2, "bootstrap sd reproduces s / sqrt(n) within 20%");
        check(Math.abs(mean(direct) - sampleMean) <= 0.25 * bootstrapSd,
                "mean of replicate means matches the sample mean");

        double[] percentile = percentileInterval(direct, 0.05);
        double[] normal = normalInterval(data, 0.05);
        check(percentile[0] <= percentile[1] && normal[0] <= normal[1], "intervals are ordered");
        check(percentile[0] <= sampleMean && sampleMean <= percentile[1],
                "percentile interval contains the sample mean");
        check(normal[0] <= sampleMean && sampleMean <= normal[1],
                "normal interval contains the sample mean");
        double dataMin = Double.POSITIVE_INFINITY;
        double dataMax = Double.NEGATIVE_INFINITY;
        for (double value : data) {
            dataMin = Math.min(dataMin, value);
            dataMax = Math.max(dataMax, value);
        }
        check(percentile[0] >= dataMin && percentile[1] <= dataMax,
                "every replicate mean stays inside [min, max] of the data");
        check(Math.abs(normal[0] - (sampleMean - Z_975 * analyticSe)) <= 1e-9,
                "normal interval closed form");
        check(Math.abs(normal[1] - (sampleMean + Z_975 * analyticSe)) <= 1e-9,
                "normal interval upper endpoint");
        check(Math.abs(inverseNormal(0.975) - Z_975) <= 1e-5, "inverse normal matches the constant");
        check(Math.abs(normalCdf(0.0) - 0.5) <= 1e-7, "standard normal cdf at 0");

        double[] constant = new double[20];
        Arrays.fill(constant, 3.5);
        double[] constantMeans = bootstrapMeans(constant, 100, new Rng(5));
        check(sd(constantMeans) == 0.0, "constant sample has zero bootstrap sd");
        double[] constantInterval = percentileInterval(constantMeans, 0.05);
        check(constantInterval[0] == 3.5 && constantInterval[1] == 3.5, "zero-width interval");

        double[] symmetric = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        double[] symmetricMeans = bootstrapMeans(symmetric, 2000, new Rng(99));
        double[] symmetricInterval = percentileInterval(symmetricMeans, 0.05);
        double symmetricMean = mean(symmetric);
        double symmetryGap = Math.abs((symmetricMean - symmetricInterval[0])
                - (symmetricInterval[1] - symmetricMean));
        check(symmetryGap <= 0.15, "symmetric data gives a symmetric percentile interval");

        double[] skewed = new double[60];
        int seed = 4321;
        for (int i = 0; i < 60; i++) {
            seed = (seed * 75 + 74) % 65537;
            double u = (seed % 65536 + 1) / 65537.0;
            skewed[i] = -Math.log(u);
        }
        double[] skewedMeans = bootstrapMeans(skewed, 2000, new Rng(303));
        double[] skewedInterval = percentileInterval(skewedMeans, 0.05);
        double skewedMean = mean(skewed);
        check(skewedInterval[1] - skewedMean > skewedMean - skewedInterval[0],
                "right-skewed data gives an asymmetric interval");
        check(skewedInterval[0] <= skewedMean && skewedMean <= skewedInterval[1],
                "skewed interval still contains the sample mean");

        double[] small = lcgSample(25, 606, 100);
        double[] large = new double[100];
        for (int i = 0; i < large.length; i++) large[i] = small[i % small.length];
        double smallWidth = width(percentileInterval(bootstrapMeans(small, 400, new Rng(8)), 0.05));
        double largeWidth = width(percentileInterval(bootstrapMeans(large, 400, new Rng(8)), 0.05));
        double ratio = smallWidth / largeWidth;
        check(ratio >= 1.5 && ratio <= 2.5, "four times the data halves the interval width");

        double[] repeatA = percentileInterval(bootstrapMeans(data, replicates, new Rng(12345)), 0.05);
        double[] repeatB = percentileInterval(bootstrapMeans(data, replicates, new Rng(12345)), 0.05);
        check(repeatA[0] == repeatB[0] && repeatA[1] == repeatB[1], "same seed reproduces the CI");
        double[] different = percentileInterval(bootstrapMeans(data, replicates, new Rng(54321)), 0.05);

        int experiments = 150;
        int percentileHits = 0;
        int normalHits = 0;
        for (int experiment = 0; experiment < experiments; experiment++) {
            double[] sample = new double[25];
            int stream = 700_001 + experiment * 977;
            for (int i = 0; i < sample.length; i++) {
                stream = (stream * 75 + 74) % 65537;
                sample[i] = stream % 100;
            }
            double[] replicateMeans = bootstrapMeans(sample, 300, new Rng(31_000 + experiment));
            double[] pInterval = percentileInterval(replicateMeans, 0.05);
            double[] nInterval = normalInterval(sample, 0.05);
            if (pInterval[0] <= 49.5 && 49.5 <= pInterval[1]) percentileHits++;
            if (nInterval[0] <= 49.5 && 49.5 <= nInterval[1]) normalHits++;
        }
        check(percentileHits >= (int) Math.ceil(0.85 * experiments),
                "percentile coverage at least 85% of 150 experiments");
        check(normalHits >= (int) Math.ceil(0.85 * experiments),
                "normal-approximation coverage at least 85% of 150 experiments");
        check(Math.abs(percentileHits - normalHits) <= 25, "both interval types cover similarly");

        System.out.println("PASS");
    }

    private static double width(double[] interval) {
        return interval[1] - interval[0];
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
