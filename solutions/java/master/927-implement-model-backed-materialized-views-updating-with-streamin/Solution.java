/** Task 927 - Implement model-backed materialized views updating with streaming features.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A materialized view whose value comes from a linear model over streaming features. The stream is folded into
 * O(d) running sums, so an update never touches the view; instead the view is invalidated and recomputed
 * lazily on the next read, which is exactly the incremental-maintenance contract of a materialized view: reads
 * see a consistent value computed from the current state, and the recomputation happens once per read epoch
 * rather than once per update. A drift monitor compares the streaming means with the training means in units
 * of the training standard deviation and raises an alert once the shift exceeds a threshold.
 * Simplified: a single linear model with fixed weights, no retraining, and whole-feature mean monitoring rather
 * than per-feature distribution testing. Verified: the incremental state always equals a full recomputation
 * over the whole stream; the materialized view equals the model applied to the recomputed means; the refresh
 * count is exactly one per read epoch after an update and zero when nothing changed; drift alerts fire only
 * past the threshold (including the boundary case); an empty stream and a single sample behave; and the run is
 * reproducible.
 */
import java.util.*;

/** An online feature aggregator plus the lazily materialised model view. */
final class StreamView {
    final int dimensions;
    final double bias;
    final double[] weights, trainingMean, trainingStd;
    final double threshold;

    long count;
    final double[] sums;
    int refreshes, updates, alerts;
    Double view;                        // null means the materialised view is stale

    StreamView(double bias, double[] weights, double[] trainingMean, double[] trainingStd, double threshold) {
        this.bias = bias;
        this.weights = weights;
        this.trainingMean = trainingMean;
        this.trainingStd = trainingStd;
        this.threshold = threshold;
        this.dimensions = weights.length;
        this.sums = new double[dimensions];
    }

    /** Fold one sample into the running sums; the view is only invalidated. */
    void add(double[] x) {
        count++;
        updates++;
        for (int i = 0; i < dimensions; i++) sums[i] += x[i];
        view = null;
    }

    /** Read the view, recomputing it at most once per epoch. */
    double read() {
        if (view == null) {
            view = evaluate();
            refreshes++;
        }
        return view;
    }

    /** The model applied to the current feature means (an empty stream reads as the bias). */
    double evaluate() {
        if (count == 0) return bias;
        double prediction = bias;
        for (int i = 0; i < dimensions; i++) prediction += weights[i] * mean(i);
        return prediction;
    }

    double mean(int feature) { return count == 0 ? 0 : sums[feature] / count; }

    /** Largest standardised shift of any feature away from its training mean. */
    double driftScore() {
        if (count == 0) return 0;
        double worst = 0;
        for (int i = 0; i < dimensions; i++) {
            if (trainingStd[i] == 0) continue;
            worst = Math.max(worst, Math.abs(mean(i) - trainingMean[i]) / trainingStd[i]);
        }
        return worst;
    }

    boolean monitor() {
        if (driftScore() > threshold) {
            alerts++;
            return true;
        }
        return false;
    }

    /** Independent oracle: fold the whole stream again from scratch. */
    static double[] recomputeMeans(List<double[]> stream, int dimensions) {
        double[] sums = new double[dimensions];
        for (double[] x : stream) for (int i = 0; i < dimensions; i++) sums[i] += x[i];
        for (int i = 0; i < dimensions; i++) sums[i] /= stream.isEmpty() ? 1 : stream.size();
        return sums;
    }
}

public class Solution {
    static final int DIMENSIONS = 4, SAMPLES = 120;
    static final double[] WEIGHTS = {0.8, -0.5, 1.2, 0.3}, TRAINING_MEAN = {1.0, 2.0, 0.5, -1.0};
    static final double[] TRAINING_STD = {0.5, 0.25, 0.3, 0.4};
    static final double BIAS = 0.25;

    /** Deterministic samples: the first 80 follow the training distribution, the rest are shifted. */
    static List<double[]> stream(int shifted) {
        List<double[]> out = new ArrayList<>();
        for (int t = 0; t < SAMPLES; t++) {
            double[] x = new double[DIMENSIONS];
            for (int i = 0; i < DIMENSIONS; i++) {
                double noise = 0.4 * Math.sin(1.7 * t + 2.3 * i);
                double shift = t >= 80 ? shifted : 0;
                x[i] = TRAINING_MEAN[i] + shift + TRAINING_STD[i] * noise;
            }
            out.add(x);
        }
        return out;
    }

    public static void main(String[] args) {
        List<double[]> stream = stream(30);                      // all features move by +30 sigma/10 after sample 80
        StreamView model = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        List<double[]> seen = new ArrayList<>();
        int refreshesPerEpoch = -1, reads = 0, wrongMeans = 0;

        for (int t = 0; t < stream.size(); t++) {
            model.add(stream.get(t));
            seen.add(stream.get(t));
            if (t % 10 == 9) {
                int before = model.refreshes;
                double a = model.read();
                double b = model.read();                          // a second read in the same epoch must not recompute
                reads += 2;
                double[] means = StreamView.recomputeMeans(seen, DIMENSIONS);
                for (int i = 0; i < DIMENSIONS; i++) if (Math.abs(means[i] - model.mean(i)) > 1e-12) wrongMeans++;
                double closedForm = BIAS;
                for (int i = 0; i < DIMENSIONS; i++) closedForm += WEIGHTS[i] * means[i];
                check(Math.abs(a - closedForm) < 1e-12 && a == b, "the view must equal the model applied to recomputed "
                        + "means and be stable within an epoch, got " + a + " vs " + closedForm);
                check(model.refreshes - before == 1, "exactly one refresh per read epoch, got " + (model.refreshes - before));
                if (refreshesPerEpoch < 0) refreshesPerEpoch = model.refreshes - before;
                model.monitor();
            }
        }
        double[] truth = StreamView.recomputeMeans(stream, DIMENSIONS);
        double closedForm = BIAS;
        for (int i = 0; i < DIMENSIONS; i++) closedForm += WEIGHTS[i] * truth[i];
        int refreshesAfterUpdates = model.refreshes;

        System.out.println("streaming view: " + SAMPLES + " samples, " + DIMENSIONS + " features, " + model.updates
                + " updates, " + model.refreshes + " refreshes for " + reads + " reads, drift alerts " + model.alerts
                + ", worst shift " + String.format("%.3f", model.driftScore()));
        System.out.println("view " + String.format("%.6f", model.read()) + " vs closed form "
                + String.format("%.6f", closedForm) + ", means " + String.format("%.4f", model.mean(0)) + "/"
                + String.format("%.4f", model.mean(2)) + ", stale reads after the last update "
                + (refreshesAfterUpdates - model.refreshes + 1));

        check(wrongMeans == 0, "the incremental means must equal a full recomputation");
        check(Math.abs(model.read() - closedForm) < 1e-12, "the final view must match the closed form");
        check(refreshesPerEpoch == 1 && reads == 2 * (SAMPLES / 10), "reads must be served from one refresh per epoch");
        check(model.refreshes * 8 <= model.updates, "maintenance must be far cheaper than the update stream: "
                + model.refreshes + " refreshes for " + model.updates + " updates");
        check(model.alerts >= 1 && model.alerts <= 6, "the shifted stream must raise alerts, but only after the "
                + "running mean has moved, got " + model.alerts);
        check(model.driftScore() > 3.0, "the final drift score must exceed the threshold, got " + model.driftScore());

        // A steady stream must never alert; a shift just under the threshold must not either.
        StreamView steady = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        for (double[] x : stream(0)) steady.add(x);
        check(!steady.monitor() && steady.alerts == 0, "an unshifted stream must not alert, worst shift "
                + steady.driftScore());
        StreamView borderline = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        for (int t = 0; t < 40; t++) {                            // means 0.7495 sigma away: just inside the threshold
            double[] x = new double[DIMENSIONS];
            for (int i = 0; i < DIMENSIONS; i++) x[i] = TRAINING_MEAN[i] + (i == 0 ? 0.7495 * TRAINING_STD[0] : 0);
            borderline.add(x);
        }
        check(borderline.driftScore() < 3.0 && !borderline.monitor(),
                "a shift below the threshold must stay quiet, score " + borderline.driftScore());
        StreamView overLine = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        for (int t = 0; t < 40; t++) {
            double[] x = new double[DIMENSIONS];
            for (int i = 0; i < DIMENSIONS; i++) x[i] = TRAINING_MEAN[i] + (i == 0 ? 3.25 * TRAINING_STD[0] : 0);
            overLine.add(x);
        }
        check(overLine.driftScore() > 3.0 && overLine.monitor(), "a shift past the threshold must alert, score "
                + overLine.driftScore());

        // Boundaries: an empty stream reads the bias, a single sample reads the model at that sample.
        StreamView empty = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        check(empty.read() == BIAS && empty.refreshes == 1 && empty.mean(0) == 0 && empty.driftScore() == 0,
                "an empty stream must materialise to the bias");
        StreamView single = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        single.add(new double[] {1, 2, 3, 4});
        double singleExpected = BIAS + WEIGHTS[0] * 1 + WEIGHTS[1] * 2 + WEIGHTS[2] * 3 + WEIGHTS[3] * 4;
        check(Math.abs(single.read() - singleExpected) < 1e-12 && single.mean(2) == 3,
                "a single sample must be averaged by one, got " + single.read());

        StreamView replay = new StreamView(BIAS, WEIGHTS, TRAINING_MEAN, TRAINING_STD, 3.0);
        for (double[] x : stream) replay.add(x);
        check(Math.abs(replay.read() - model.read()) < 1e-12 && replay.driftScore() == model.driftScore(),
                "the streaming view must be reproducible");

        System.out.println("boundary: empty stream reads bias " + BIAS + ", single sample reads "
                + String.format("%.4f", single.read()) + ", unshifted alerts " + steady.alerts
                + ", borderline alerts " + borderline.alerts + ", past-threshold alerts " + overLine.alerts);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
