/** Task 908 - Implement certified ML predictions with conformal prediction intervals.
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * Split conformal prediction turns any point predictor into a *certified* interval predictor. Hold out a
 * calibration set of n points, compute the nonconformity scores s_i = |y_i - yhat(x_i)| / sigma_hat(x_i),
 * take the k-th smallest score with k = ceil((n+1)(1 - alpha)), and return the interval
 * [yhat - q*sigma_hat, yhat + q*sigma_hat]. If the data are exchangeable, marginal coverage is provably at
 * least 1 - alpha with no assumption on the model, the noise shape or the number of parameters.
 *
 * The checks verify that promise rather than restating it: coverage on a fresh test set, average coverage
 * over many random splits, the oracle rank computation, alpha-monotonicity of the width, the k > n boundary
 * where the interval becomes unbounded, and the two properties that make conformal worth using - a
 * misspecified base model still gets valid coverage, and normalised scores keep coverage (and adapt width)
 * under a covariate shift where a fixed-width Gaussian interval silently loses it.
 *
 * Simplification: a one-dimensional least-squares base model with Gaussian heteroscedastic noise,
 * sigma(x) = 0.1 + 0.3x, 200 calibration points and thousands of test points. Invariants verified: coverage
 * >= 1 - alpha within statistical tolerance, exact agreement with an independently coded order statistic,
 * unbounded fallback at the boundary, adaptive widths, and deterministic replay.
 */
import java.util.*;

/** splitmix64 with a deterministic Gaussian. */
final class rng908 {
    private long state;
    private double spare;
    private boolean hasSpare;

    rng908(long seed) { this.state = seed; }

    private long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        if (hasSpare) {
            hasSpare = false;
            return spare;
        }
        double u;
        double v;
        double s;
        do {
            u = 2 * nextDouble() - 1;
            v = 2 * nextDouble() - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        double mul = Math.sqrt(-2 * Math.log(s) / s);
        spare = v * mul;
        hasSpare = true;
        return u * mul;
    }
}

/** A labelled sample set. */
final class Sample {
    final double[] x;
    final double[] y;

    Sample(double[] x, double[] y) {
        this.x = x;
        this.y = y;
    }

    int size() { return x.length; }
}

public class Solution {
    static final int N_TRAIN = 200;
    static final int N_CAL = 200;
    static final int N_TEST = 2000;
    static final int SPLITS = 100;
    static final double ALPHA = 0.1;
    static final double NOISE_BASE = 0.1;
    static final double NOISE_SLOPE = 0.3;
    static final boolean LINEAR = false;
    static final boolean QUADRATIC = true;

    public static void main(String[] args) {
        // ---- 1. the base model, fitted on training data only -------------------------------------------
        Sample train = makeData(N_TRAIN, 0x9080001L, 0.0, 1.0, LINEAR);
        double[] model = fitLinear(train.x, train.y);
        check(model[1] > 1.2 && model[1] < 1.8, "the slope must approximate 1.5, got " + model[1]);
        check(residualVariance(model, train) < 0.35 * variance(train.y),
                "the base model must remove at least two thirds of the variance: residual "
                        + round(residualVariance(model, train)) + " vs total " + round(variance(train.y)));
        Sample calibration = makeData(N_CAL, 0x9080002L, 0.0, 1.0, LINEAR);
        Sample test = makeData(N_TEST, 0x9080003L, 0.0, 1.0, LINEAR);

        // ---- 2. plain and normalised conformal intervals ----------------------------------------------
        double[] plainScores = scores(model, calibration, false);
        double[] normalizedScores = scores(model, calibration, true);
        double qPlain = conformalQuantile(plainScores, ALPHA);
        double qNormalized = conformalQuantile(normalizedScores, ALPHA);
        check(qPlain > 0 && qNormalized > 0, "the quantiles must be positive");
        check(qNormalized > qPlain, "dividing by a sub-unit sigma must inflate the raw score");
        double coveragePlain = coverage(model, test, qPlain, false);
        double coverageNormalized = coverage(model, test, qNormalized, true);
        check(coveragePlain >= 1 - ALPHA - 0.03, "plain conformal coverage must reach 1 - alpha, got "
                + coveragePlain);
        check(coverageNormalized >= 1 - ALPHA - 0.03, "normalised conformal coverage must reach 1 - alpha, got "
                + coverageNormalized);

        // ---- 3. oracle: the same rank by an independently coded order statistic ------------------------
        int k = (int) Math.ceil((N_CAL + 1) * (1 - ALPHA));
        check(k == 181, "with 200 calibration points and alpha=0.1 the rank must be 181, got " + k);
        check(conformalQuantileOracle(plainScores, ALPHA) == qPlain,
                "the oracle order statistic must reproduce the quantile");
        check(conformalQuantileOracle(normalizedScores, ALPHA) == qNormalized,
                "the oracle must agree for the normalised scores too");
        double[] shuffled = sortCopy(plainScores);
        reverse(shuffled);
        check(conformalQuantile(shuffled, ALPHA) == qPlain, "the quantile must not depend on input order");

        // ---- 4. the marginal guarantee over many random splits -----------------------------------------
        int covered = 0;
        int total = 0;
        for (int split = 0; split < SPLITS; split++) {
            Sample cal = makeData(N_CAL, 0x9081000L + split, 0.0, 1.0, LINEAR);
            Sample fresh = makeData(200, 0x9082000L + split, 0.0, 1.0, LINEAR);
            double q = conformalQuantile(scores(model, cal, false), ALPHA);
            for (int i = 0; i < fresh.size(); i++) {
                total++;
                if (Math.abs(fresh.y[i] - predict(model, fresh.x[i])) <= q) covered++;
            }
        }
        double averageCoverage = covered / (double) total;
        check(averageCoverage >= 1 - ALPHA, "average coverage over " + SPLITS + " splits must reach 1 - alpha, "
                + "got " + averageCoverage);
        check(averageCoverage < 1 - ALPHA + 0.05, "the interval must not be wastefully conservative: "
                + averageCoverage);

        // ---- 5. width is monotone in alpha --------------------------------------------------------------
        double previous = Double.POSITIVE_INFINITY;
        double widthAtFive = 0;
        double widthAtForty = 0;
        for (double alpha : new double[] {0.05, 0.1, 0.2, 0.4}) {
            double width = conformalQuantile(plainScores, alpha);
            check(width <= previous, "a larger alpha must not widen the interval: " + alpha + " -> " + width);
            if (alpha == 0.05) widthAtFive = width;
            if (alpha == 0.4) widthAtForty = width;
            previous = width;
        }
        check(widthAtFive > widthAtForty, "a 95% interval must be wider than a 60% one: " + widthAtFive + " vs "
                + widthAtForty);

        // ---- 6. boundary: a rank beyond the calibration set is unbounded ---------------------------------
        check(Double.isInfinite(conformalQuantile(plainScores, 1.0 / (N_CAL + 1) / 10)),
                "alpha below 1/(n+1) must yield an unbounded interval");
        check(coverage(model, test, Double.POSITIVE_INFINITY, false) == 1.0,
                "an unbounded interval covers every test point");
        check(Double.isInfinite(conformalQuantile(plainScores, 0.0)), "alpha = 0 can never be certified");
        check(!Double.isInfinite(conformalQuantile(plainScores, 1.0 - 1e-9)),
                "alpha near 1 keeps a finite interval");
        check(conformalQuantile(plainScores, 0.9999) < 0.2, "alpha near 1 gives a very narrow interval: "
                + conformalQuantile(plainScores, 0.9999));

        // ---- 7. a misspecified base model still gets valid coverage ------------------------------------
        Sample trainQuad = makeData(N_TRAIN, 0x9080006L, 0.0, 1.0, QUADRATIC);
        double[] misspecified = fitLinear(trainQuad.x, trainQuad.y);
        Sample calQuad = makeData(N_CAL, 0x9080007L, 0.0, 1.0, QUADRATIC);
        double qQuad = conformalQuantile(scores(misspecified, calQuad, false), ALPHA);
        check(maxDeviation(misspecified, QUADRATIC) > 0.1,
                "the quadratic truth must really bias the linear model: " + maxDeviation(misspecified, QUADRATIC));
        int quadCovered = 0;
        int quadTotal = 0;
        for (int split = 0; split < 30; split++) {
            Sample freshCal = makeData(N_CAL, 0x9083000L + split, 0.0, 1.0, QUADRATIC);
            Sample freshTest = makeData(200, 0x9084000L + split, 0.0, 1.0, QUADRATIC);
            double q = conformalQuantile(scores(misspecified, freshCal, false), ALPHA);
            for (int i = 0; i < freshTest.size(); i++) {
                quadTotal++;
                if (Math.abs(freshTest.y[i] - predict(misspecified, freshTest.x[i])) <= q) quadCovered++;
            }
        }
        double coverageQuad = quadCovered / (double) quadTotal;
        check(coverageQuad >= 1 - ALPHA, "coverage must survive a biased model, averaged over 30 splits: got "
                + coverageQuad);
        double[] constantModel = {meanOf(train.y), 0.0};
        double qConstant = conformalQuantile(scores(constantModel, calibration, false), ALPHA);
        double coverageConstant = coverage(constantModel, test, qConstant, false);
        check(qConstant > 1.2 * qPlain, "a constant predictor must buy coverage with a wider interval: "
                + qConstant + " vs " + qPlain);
        check(coverageConstant >= 1 - ALPHA - 0.03, "coverage must hold even for a constant predictor, got "
                + coverageConstant);

        // ---- 8. the certified claim: shift the test distribution ---------------------------------------
        Sample shifted = makeData(N_TEST, 0x9080004L, 0.5, 1.0, LINEAR);
        double shiftedPlain = coverage(model, shifted, qPlain, false);
        double shiftedNormalized = coverage(model, shifted, qNormalized, true);
        double gaussianWidth = 1.28 * sigmaAt(0.75);
        int gaussianCovered = 0;
        for (int i = 0; i < shifted.size(); i++) {
            if (Math.abs(shifted.y[i] - predict(model, shifted.x[i])) <= gaussianWidth) gaussianCovered++;
        }
        double shiftedGaussian = gaussianCovered / (double) shifted.size();
        check(shiftedPlain < 1 - ALPHA, "under covariate shift the fixed-width interval must lose coverage: "
                + shiftedPlain);
        check(shiftedNormalized >= 1 - ALPHA - 0.03,
                "normalised conformal scores must hold coverage under the shift: " + shiftedNormalized);
        check(shiftedNormalized > shiftedGaussian,
                "normalised conformal must beat the naive Gaussian interval: " + shiftedNormalized + " vs "
                        + shiftedGaussian);
        double lowWidth = 2 * qNormalized * sigmaAt(0.1);
        double highWidth = 2 * qNormalized * sigmaAt(0.9);
        check(highWidth > lowWidth * 1.5, "the interval must widen where the noise is larger: " + lowWidth
                + " -> " + highWidth);
        check(highWidth > 2 * qPlain && lowWidth < 2 * qPlain,
                "adaptive widths must straddle the homoscedastic width");

        // ---- 9. determinism -------------------------------------------------------------------------------
        check(Arrays.equals(makeData(50, 0x9080005L, 0.0, 1.0, LINEAR).y,
                makeData(50, 0x9080005L, 0.0, 1.0, LINEAR).y), "data generation must be deterministic");
        check(conformalQuantile(scores(model, calibration, false), ALPHA) == qPlain,
                "the quantile must be reproducible");

        System.out.println("calibration " + N_CAL + ", rank " + k + ", q " + round(qPlain) + " plain / "
                + round(qNormalized) + " normalised; width 2*low " + round(lowWidth) + " -> 2*high "
                + round(highWidth));
        System.out.println("coverage: plain " + round(coveragePlain) + ", normalised "
                + round(coverageNormalized) + ", mean over " + SPLITS + " splits " + round(averageCoverage)
                + ", misspecified " + round(coverageQuad));
        System.out.println("shifted test (x in [0.5,1]): fixed " + round(shiftedPlain) + ", normalised "
                + round(shiftedNormalized) + ", naive Gaussian " + round(shiftedGaussian));
        System.out.println("PASS");
    }

    /** Linear truth keeps the model well specified; the quadratic truth biases it on purpose. */
    static double truthOf(double x, boolean quadratic) {
        return quadratic ? 1.5 * x + 0.8 * x * x : 1.5 * x;
    }

    static double sigmaAt(double x) { return NOISE_BASE + NOISE_SLOPE * x; }

    static Sample makeData(int n, long seed, double lo, double hi, boolean quadratic) {
        rng908 rng = new rng908(seed);
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double xi = lo + (hi - lo) * rng.nextDouble();
            x[i] = xi;
            y[i] = truthOf(xi, quadratic) + sigmaAt(xi) * rng.nextGaussian();
        }
        return new Sample(x, y);
    }

    /** Ordinary least squares for y = a + b x. */
    static double[] fitLinear(double[] x, double[] y) {
        double meanX = 0;
        double meanY = 0;
        for (int i = 0; i < x.length; i++) {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= x.length;
        meanY /= y.length;
        double cov = 0;
        double var = 0;
        for (int i = 0; i < x.length; i++) {
            cov += (x[i] - meanX) * (y[i] - meanY);
            var += (x[i] - meanX) * (x[i] - meanX);
        }
        double b = cov / var;
        return new double[] {meanY - b * meanX, b};
    }

    static double predict(double[] model, double x) { return model[0] + model[1] * x; }

    /** Nonconformity scores, optionally divided by sigma_hat(x) to make them adaptive. */
    static double[] scores(double[] model, Sample data, boolean normalized) {
        double[] result = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            double residual = Math.abs(data.y[i] - predict(model, data.x[i]));
            result[i] = normalized ? residual / sigmaAt(data.x[i]) : residual;
        }
        return result;
    }

    /** The conformal quantile: the ceil((n+1)(1-alpha))-th smallest score, or +inf beyond the sample. */
    static double conformalQuantile(double[] scores, double alpha) {
        int n = scores.length;
        int rank = rankFor(n, alpha);
        if (rank > n) return Double.POSITIVE_INFINITY;
        return sortCopy(scores)[rank - 1];
    }

    /** Independent oracle: descending insertion sort instead of Arrays.sort, same order statistic. */
    static double conformalQuantileOracle(double[] scores, double alpha) {
        int n = scores.length;
        int rank = rankFor(n, alpha);
        if (rank > n) return Double.POSITIVE_INFINITY;
        double[] descending = scores.clone();
        for (int i = 1; i < n; i++) {
            double key = descending[i];
            int j = i - 1;
            while (j >= 0 && descending[j] < key) {
                descending[j + 1] = descending[j];
                j--;
            }
            descending[j + 1] = key;
        }
        return descending[n - rank];
    }

    static int rankFor(int n, double alpha) {
        int rank = (int) Math.ceil((n + 1) * (1 - alpha));
        return Math.max(1, rank);
    }

    static double coverage(double[] model, Sample data, double q, boolean normalized) {
        int covered = 0;
        for (int i = 0; i < data.size(); i++) {
            double halfWidth = normalized ? q * sigmaAt(data.x[i]) : q;
            if (Math.abs(data.y[i] - predict(model, data.x[i])) <= halfWidth) covered++;
        }
        return covered / (double) data.size();
    }

    /** Largest deviation of the fitted line from the generating truth over the design interval. */
    static double maxDeviation(double[] model, boolean quadratic) {
        double worst = 0;
        int steps = 1000;
        for (int i = 0; i < steps; i++) {
            double x = (i + 0.5) / steps;
            worst = Math.max(worst, Math.abs(predict(model, x) - truthOf(x, quadratic)));
        }
        return worst;
    }

    static double residualVariance(double[] model, Sample data) {
        double[] residual = new double[data.size()];
        for (int i = 0; i < data.size(); i++) residual[i] = data.y[i] - predict(model, data.x[i]);
        return variance(residual);
    }

    static double variance(double[] values) {
        double mean = 0;
        for (double v : values) mean += v;
        mean /= values.length;
        double sum = 0;
        for (double v : values) sum += (v - mean) * (v - mean);
        return sum / values.length;
    }

    static double meanOf(double[] values) {
        double total = 0;
        for (double v : values) total += v;
        return total / values.length;
    }

    static double[] sortCopy(double[] values) {
        double[] copy = values.clone();
        Arrays.sort(copy);
        return copy;
    }

    static void reverse(double[] values) {
        for (int i = 0, j = values.length - 1; i < j; i++, j--) {
            double tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }

    static double round(double value) { return Math.round(value * 10000) / 10000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
