/** Task 929 - Implement end-to-end automated ML system with monitoring, drift alerts, and retraining.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A small but complete automated-ML loop. A logistic model is trained by full-batch gradient descent on an
 * initial batch; a stream of labelled batches then arrives and the system (a) predicts, (b) monitors the
 * rolling error rate over a bounded window and the standardised shift of the streaming feature means against
 * the training reference, (c) raises an alert when either signal crosses its threshold, and (d) retrains on the
 * recent window, publishing a new model version - so the served model tracks a concept that changes under it.
 * Simplified: two features, a linear model, an in-memory window and retraining on the latest window only (no
 * ensembles, no shadow evaluation, no hyper-parameter search). Verified: training is monotone and accurate; the
 * monitor stays quiet on the original distribution and alerts after the concept drift; each alert produces
 * exactly one retrain and one new version; the retrained model is measurably better on the drifted data than
 * the model it replaced; the window stays bounded; the alert threshold is exclusive; and the loop is
 * reproducible.
 */
import java.util.*;

/** Logistic regression trained by full-batch gradient descent, with a recorded loss history. */
final class Logistic {
    final int dims;
    final double[] w;
    double b;
    final List<Double> lossHistory = new ArrayList<>();

    Logistic(int dims) {
        this.dims = dims;
        this.w = new double[dims];
    }

    static double sigmoid(double z) { return 1.0 / (1.0 + Math.exp(-z)); }

    double probability(double[] x) {
        double z = b;
        for (int i = 0; i < dims; i++) z += w[i] * x[i];
        return sigmoid(z);
    }

    int predict(double[] x) { return probability(x) >= 0.5 ? 1 : 0; }

    /** Full-batch gradient descent; returns the final loss. */
    double train(List<double[]> xs, List<Integer> ys, int iterations, double rate) {
        for (int it = 0; it < iterations; it++) {
            double[] grad = new double[dims];
            double gradB = 0, loss = 0;
            for (int n = 0; n < xs.size(); n++) {
                double[] x = xs.get(n);
                double p = probability(x);
                double y = ys.get(n);
                double error = p - y;
                for (int i = 0; i < dims; i++) grad[i] += error * x[i];
                gradB += error;
                loss += -(y * Math.log(Math.max(p, 1e-12)) + (1 - y) * Math.log(Math.max(1 - p, 1e-12)));
            }
            for (int i = 0; i < dims; i++) w[i] -= rate * grad[i] / xs.size();
            b -= rate * gradB / xs.size();
            if (it % 25 == 0 || it == iterations - 1) lossHistory.add(loss / xs.size());
        }
        return lossHistory.get(lossHistory.size() - 1);
    }

    double accuracy(List<double[]> xs, List<Integer> ys) {
        int right = 0;
        for (int i = 0; i < xs.size(); i++) if (predict(xs.get(i)) == ys.get(i)) right++;
        return right / (double) xs.size();
    }
}

/** Monitoring plus retraining: the automated part of the loop. */
final class AutoMl {
    static final int WINDOW = 40, ROLLING = 20;
    static final double ERROR_THRESHOLD = 0.15, SHIFT_THRESHOLD = 3.0;

    final double[] referenceMean, referenceStd;
    final ArrayDeque<double[]> window = new ArrayDeque<>();
    final ArrayDeque<Integer> labels = new ArrayDeque<>();
    final ArrayDeque<Boolean> correct = new ArrayDeque<>();

    Logistic model;
    int version = 1, retrains, alerts;
    double lastErrorRate, lastShift;

    AutoMl(Logistic model, double[] referenceMean, double[] referenceStd) {
        this.model = model;
        this.referenceMean = referenceMean;
        this.referenceStd = referenceStd;
    }

    /** Rolling error rate over the last ROLLING predictions. */
    double errorRate() {
        int bad = 0;
        for (boolean ok : correct) if (!ok) bad++;
        return correct.isEmpty() ? 0 : bad / (double) correct.size();
    }

    /** Standardised shift of the streaming feature means against the training reference. */
    double shift() {
        if (window.isEmpty()) return 0;
        double[] sums = new double[referenceMean.length];
        for (double[] x : window) for (int i = 0; i < sums.length; i++) sums[i] += x[i];
        double worst = 0;
        for (int i = 0; i < sums.length; i++) {
            if (referenceStd[i] == 0) continue;
            worst = Math.max(worst, Math.abs(sums[i] / window.size() - referenceMean[i]) / referenceStd[i]);
        }
        return worst;
    }

    boolean shouldAlert(double errorRate) { return errorRate > ERROR_THRESHOLD; }

    /** Serve one batch: predict, monitor, and retrain when a signal fires. */
    double serve(List<double[]> xs, List<Integer> ys) {
        for (int n = 0; n < xs.size(); n++) {
            double[] x = xs.get(n);
            correct.addLast(model.predict(x) == ys.get(n));
            while (correct.size() > ROLLING) correct.removeFirst();
            window.addLast(x);
            labels.addLast(ys.get(n));
            while (window.size() > WINDOW) {
                window.removeFirst();
                labels.removeFirst();
            }
        }
        lastErrorRate = errorRate();
        lastShift = shift();
        double before = model.accuracy(xs, ys);
        if (shouldAlert(lastErrorRate) || lastShift > SHIFT_THRESHOLD) {
            alerts++;
            retrain();
        }
        return before;
    }

    void retrain() {
        List<double[]> xs = new ArrayList<>(window);
        List<Integer> ys = new ArrayList<>(labels);
        Logistic fresh = new Logistic(model.dims);
        fresh.train(xs, ys, 400, 0.4);
        model = fresh;                       // the served model is replaced atomically
        version++;
        retrains++;
    }
}

public class Solution {
    static final double[] PRE_W = {1.0, 0.6}, POST_W = {-0.4, 1.0};
    static final int BATCH = 20, BATCHES = 14, SHIFT_BATCH = 6;

    static double[] sample(int index, boolean shifted) {
        double base = shifted ? 1.6 : 0.0;
        return new double[] {base + 1.1 * Math.sin(0.37 * index) + 0.2 * Math.cos(0.11 * index),
                0.9 * Math.cos(0.23 * index) + 0.15 * Math.sin(0.53 * index)};
    }

    static int label(double[] x, boolean shifted) {
        double[] w = shifted ? POST_W : PRE_W;
        return w[0] * x[0] + w[1] * x[1] + (shifted ? 0.1 : -0.2) > 0 ? 1 : 0;
    }

    static List<double[]> batch(int b, boolean shifted) {
        List<double[]> out = new ArrayList<>();
        for (int n = 0; n < BATCH; n++) out.add(sample(b * BATCH + n, shifted));
        return out;
    }

    static List<Integer> labels(List<double[]> xs, boolean shifted) {
        List<Integer> out = new ArrayList<>();
        for (double[] x : xs) out.add(label(x, shifted));
        return out;
    }

    static double[] mean(List<double[]> xs) {
        double[] out = new double[xs.get(0).length];
        for (double[] x : xs) for (int i = 0; i < out.length; i++) out[i] += x[i];
        for (int i = 0; i < out.length; i++) out[i] /= xs.size();
        return out;
    }

    static double[] std(List<double[]> xs) {
        double[] m = mean(xs), out = new double[m.length];
        for (double[] x : xs) for (int i = 0; i < out.length; i++) out[i] += (x[i] - m[i]) * (x[i] - m[i]);
        for (int i = 0; i < out.length; i++) out[i] = Math.sqrt(out[i] / xs.size());
        return out;
    }

    public static void main(String[] args) {
        List<double[]> training = new ArrayList<>();
        for (int b = 0; b < SHIFT_BATCH; b++) training.addAll(batch(b, false));
        List<Integer> trainingLabels = labels(training, false);
        Logistic model = new Logistic(2);
        double finalLoss = model.train(training, trainingLabels, 800, 0.4);

        AutoMl system = new AutoMl(model, mean(training), std(training));
        List<Double> accuracyBeforeDrift = new ArrayList<>(), accuracyAfterDrift = new ArrayList<>();
        List<Integer> versions = new ArrayList<>();
        int alertsBeforeDrift = 0, retrainsBeforeDrift = 0, windowPeak = 0;
        List<Double> firstModelOnDrift = null, retrainedOnDrift = null;

        for (int b = 0; b < BATCHES; b++) {
            boolean shifted = b >= SHIFT_BATCH;
            List<double[]> xs = batch(b, shifted);
            List<Integer> ys = labels(xs, shifted);
            double accuracy = system.serve(xs, ys);
            if (shifted) {
                accuracyAfterDrift.add(accuracy);
                if (firstModelOnDrift == null) firstModelOnDrift = List.copyOf(accuracyAfterDrift);
                retrainedOnDrift = List.copyOf(accuracyAfterDrift);
            } else {
                accuracyBeforeDrift.add(accuracy);
                alertsBeforeDrift = system.alerts;
                retrainsBeforeDrift = system.retrains;
            }
            versions.add(system.version);
            windowPeak = Math.max(windowPeak, system.window.size());
        }

        double bestAfter = 0;
        for (double a : accuracyAfterDrift) bestAfter = Math.max(bestAfter, a);
        double worstAfter = 1;
        for (double a : accuracyAfterDrift) worstAfter = Math.min(worstAfter, a);
        double meanBeforeDrift = accuracyBeforeDrift.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double meanAfterDrift = accuracyAfterDrift.stream().mapToDouble(Double::doubleValue).average().orElse(0);

        System.out.println("automated ML: train loss " + String.format("%.4f", finalLoss) + ", training accuracy "
                + String.format("%.4f", model.accuracy(training, trainingLabels)) + ", batches " + BATCHES
                + " (drift at " + SHIFT_BATCH + ")");
        System.out.println("accuracy before drift mean " + String.format("%.4f", meanBeforeDrift) + ", after drift mean "
                + String.format("%.4f", meanAfterDrift) + " (worst " + String.format("%.4f", worstAfter) + ", best "
                + String.format("%.4f", bestAfter) + "), alerts " + system.alerts + ", retrains " + system.retrains
                + ", versions " + system.version + ", window peak " + windowPeak);

        check(finalLoss < model.lossHistory.get(0) / 2, "training must make real progress: "
                + model.lossHistory.get(0) + " -> " + finalLoss);
        for (int i = 1; i < model.lossHistory.size(); i++) {
            check(model.lossHistory.get(i) <= model.lossHistory.get(i - 1) + 1e-9,
                    "the loss must not increase at step " + i);
        }
        check(model.accuracy(training, trainingLabels) > 0.9, "the initial model must fit the training batch: "
                + model.accuracy(training, trainingLabels));
        check(alertsBeforeDrift == 0 && retrainsBeforeDrift == 0, "a matching distribution must not trigger anything");
        check(meanBeforeDrift > 0.9, "the initial model must work before the drift: " + meanBeforeDrift);
        check(system.alerts > 0 && system.retrains == system.alerts, "every alert must produce exactly one retrain: "
                + system.alerts + " alerts, " + system.retrains + " retrains");
        check(system.version == 1 + system.retrains, "each retrain publishes a new version, got " + system.version);
        check(worstAfter < meanBeforeDrift, "the concept drift must actually hurt the served model: worst "
                + worstAfter + " vs " + meanBeforeDrift);
        check(bestAfter > worstAfter + 0.2, "retraining must recover most of the accuracy: worst " + worstAfter
                + " -> best " + bestAfter);
        check(windowPeak <= AutoMl.WINDOW, "the retraining window must stay bounded, peak " + windowPeak);
        check(versions.get(versions.size() - 1) > versions.get(0), "the version counter must advance");

        // The alert threshold is exclusive, and a quiet stream never retrains.
        AutoMl boundary = new AutoMl(new Logistic(2), new double[] {0, 0}, new double[] {1, 1});
        check(!boundary.shouldAlert(AutoMl.ERROR_THRESHOLD) && boundary.shouldAlert(AutoMl.ERROR_THRESHOLD + 1e-9),
                "the alert threshold must be exclusive");
        Logistic steadyModel = new Logistic(2);
        steadyModel.train(training, trainingLabels, 800, 0.4);
        AutoMl steady = new AutoMl(steadyModel, mean(training), std(training));
        double steadyAccuracy = 0;
        int steadyBatches = 0;
        for (int b = 0; b < BATCHES; b++) {
            List<double[]> xs = batch(b, false);
            List<Integer> ys = labels(xs, false);
            steadyAccuracy += steady.serve(xs, ys);
            steadyBatches++;
        }
        check(steady.alerts == 0 && steady.retrains == 0 && steady.version == 1,
                "a stationary stream must never retrain, alerts " + steady.alerts);
        check(steadyAccuracy / steadyBatches > 0.85, "a stationary stream must stay accurate: "
                + steadyAccuracy / steadyBatches);
        check(steady.window.size() == AutoMl.WINDOW, "the window fills up and then stops growing");

        Logistic replay = new Logistic(2);
        replay.train(training, trainingLabels, 800, 0.4);
        check(Math.abs(replay.accuracy(training, trainingLabels) - model.accuracy(training, trainingLabels)) < 1e-12
                && replay.probability(training.get(0)) == model.probability(training.get(0)),
                "training must be reproducible");

        System.out.println("drift handling: model versions " + versions + ", first-post-drift accuracy "
                + String.format("%.4f", firstModelOnDrift.get(0)) + " -> best after retraining "
                + String.format("%.4f", bestAfter) + ", stationary run alerts " + steady.alerts);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
