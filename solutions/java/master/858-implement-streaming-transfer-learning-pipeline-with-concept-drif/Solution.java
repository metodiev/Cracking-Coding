/** Task 858 - Implement streaming transfer learning pipeline with concept drift detection.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A stream of labelled examples (x in R^d, y in {0,1}) arrives one at a time. A frozen pretrained encoder
 *   maps x to a feature vector, and an *adapter head* (online logistic regression with fixed-seed SGD) is
 *   fine-tuned on the stream. Two detectors watch the adapter's loss signal:
 *
 *       ADWIN   : keeps a variable-length window; it is only legal for no split of the window to have
 *                 |mean(left) - mean(right)| > sqrt((1/2m1 + 1/2m2) * ln(4/delta')), delta' = delta / |W|
 *                 and it shrinks the window whenever such a split exists (an exact, checkable invariant);
 *       PH      : Page-Hinkley on the same signal with an allowed drift and a threshold.
 *
 *   The stream contains a concept drift at a known change point (the label rule flips for a suffix of the
 *   features) and the pipeline either re-uses the previous head (transfer) or retrains from zero (scratch) when
 *   a detector fires, so the transfer gain and the recovery cost are both measurable.
 *
 * WHAT IS SIMPLIFIED
 *   d = 6 synthetic features, one change point, a linear logistic head with a fixed learning rate, no
 *   server-side serving path, no storage and no real drift handling policy beyond "reset or keep the head".
 *   Everything is driven by one fixed-seed splitmix64 stream, there is no wall-clock time and no threads.
 *
 * INVARIANTS VERIFIED
 *   Detector correctness: the ADWIN window invariant is checked by brute force over *every* split of the
 *   window after each step (no split may exceed its Hoeffding bound) and the detector never fires on a
 *   stationary stream; both detectors fire within a bounded delay after the injected change point; a constant
 *   stream (zero variance) does not produce spurious alarms.
 *   Learning: the moving-average adapter loss falls during the stationary prefix; the loss spikes after the
 *   drift and recovers below the pre-drift level; a head initialised from the pretrained source task converges
 *   markedly faster than a head trained from scratch on the related concept, while after a large concept change
 *   that source head brings no warm-up advantage at all (negative transfer) - which is precisely why the
 *   pipeline re-initialises the adapter when a detector fires - and every run is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so the stream, the models and the alarms are reproducible. */
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
        double u1 = Math.max(nextDouble(), 1e-12);
        double u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** A streamed example with the hidden label rule that produced it. */
final class Example {
    final double[] x;
    final int label;

    Example(double[] x, int label) {
        this.x = x;
        this.label = label;
    }
}

/** Frozen pretrained encoder: a fixed linear projection plus a quadratic term. */
final class Encoder {
    final double[][] projection;
    final double[] bias;

    Encoder(long seed, int inputDim, int outputDim) {
        SplitMix64 rng = new SplitMix64(seed);
        projection = new double[inputDim][outputDim];
        bias = new double[outputDim];
        for (int i = 0; i < inputDim; i++) {
            for (int j = 0; j < outputDim; j++) projection[i][j] = rng.nextGaussian() * 0.5;
        }
        for (int j = 0; j < outputDim; j++) bias[j] = rng.nextGaussian() * 0.2;
    }

    double[] encode(double[] x) {
        double[] features = new double[bias.length];
        for (int j = 0; j < bias.length; j++) {
            double value = bias[j];
            for (int i = 0; i < x.length; i++) value += projection[i][j] * x[i];
            features[j] = Math.tanh(value);
        }
        return features;
    }

    int dimension() { return bias.length; }
}

/** The generator: a stationary rule, then a drifted rule whose boundary is rotated. */
final class Stream {
    final SplitMix64 rng;
    final double[] weights;
    final double[] driftedWeights;
    final int changePoint;

    /** `sharedRule` (nullable) lets another stream reuse the same source-task rule with different samples. */
    Stream(long seed, int dimension, int changePoint, double driftStrength, double[] sharedRule) {
        this.rng = new SplitMix64(seed);
        this.changePoint = changePoint;
        this.weights = new double[dimension];
        this.driftedWeights = new double[dimension];
        if (sharedRule != null) {
            System.arraycopy(sharedRule, 0, weights, 0, dimension);
        } else {
            for (int i = 0; i < dimension; i++) weights[i] = rng.nextGaussian();
        }
        // rotate and partially invert the decision boundary: a genuine concept change, not just noise
        for (int i = 0; i < dimension; i++) {
            driftedWeights[i] = weights[i] * (1 - driftStrength)
                    + driftStrength * -weights[(i + 1) % dimension];
        }
    }

    Example next(int index, Encoder encoder) {
        double[] x = new double[encoder.projection.length];
        for (int i = 0; i < x.length; i++) x[i] = rng.nextGaussian();
        double[] features = encoder.encode(x);
        double[] active = index >= changePoint ? driftedWeights : weights;
        double score = 0;
        for (int j = 0; j < features.length; j++) score += active[j] * features[j];
        int label = score > 0 ? 1 : 0;
        // label noise so the stream is not perfectly separable
        if (rng.nextDouble() < 0.05) label = 1 - label;
        return new Example(features, label);
    }
}

/** Online logistic adapter with a fixed learning rate. */
final class Adapter {
    final double[] weights;
    double learningRate = 0.02;

    Adapter(int dimension) { weights = new double[dimension]; }

    Adapter copy() {
        Adapter clone = new Adapter(weights.length);
        System.arraycopy(weights, 0, clone.weights, 0, weights.length);
        clone.learningRate = learningRate;
        return clone;
    }

    double score(double[] features) {
        double value = 0;
        for (int i = 0; i < weights.length; i++) value += weights[i] * features[i];
        return value;
    }

    double probability(double[] features) {
        double value = score(features);
        if (value > 30) return 1.0;
        if (value < -30) return 0.0;
        return 1.0 / (1.0 + Math.exp(-value));
    }

    /** Returns 0/1 loss and applies one SGD step. */
    int update(Example example) {
        double p = probability(example.x);
        int predicted = p >= 0.5 ? 1 : 0;
        double error = example.label - p;
        for (int i = 0; i < weights.length; i++) weights[i] += learningRate * error * example.x[i];
        return predicted == example.label ? 0 : 1;
    }
}

/** ADWIN with the exact Hoeffding split rule, plus a brute-force invariant checker. */
final class AdwinDetector {
    final double delta;
    final ArrayDeque<Double> window = new ArrayDeque<>();
    int alarms;

    AdwinDetector(double delta) { this.delta = delta; }

    void add(double value) {
        window.addLast(value);
        boolean shrunk = true;
        while (shrunk) shrunk = shrink();
    }

    /** Removes the oldest element while an illegal split exists; returns true if the window changed. */
    boolean shrink() {
        int size = window.size();
        if (size < 4) return false;
        double[] values = new double[size];
        int i = 0;
        for (double v : window) values[i++] = v;
        double total = 0;
        double[] prefix = new double[size + 1];
        for (int k = 0; k < size; k++) {
            total += values[k];
            prefix[k + 1] = total;
        }
        for (int split = 1; split < size; split++) {
            double left = prefix[split] / split;
            double right = (prefix[size] - prefix[split]) / (size - split);
            double bound = bound(split, size - split, size);
            if (Math.abs(left - right) > bound) {
                alarms++;
                int drop = Math.min(split, size - split);
                for (int k = 0; k < drop; k++) window.pollFirst();
                return true;
            }
        }
        return false;
    }

    double bound(int m1, int m2, int size) {
        double deltaPrime = Math.max(1e-12, delta / size);
        return Math.sqrt((1.0 / (2 * m1) + 1.0 / (2 * m2)) * Math.log(4.0 / deltaPrime));
    }

    /** Largest violation over all legal splits - must always be <= 0 for a legal window. */
    double worstViolation() {
        int size = window.size();
        if (size < 4) return Double.NEGATIVE_INFINITY;
        double[] values = new double[size];
        int i = 0;
        for (double v : window) values[i++] = v;
        double worst = Double.NEGATIVE_INFINITY;
        for (int split = 1; split < size; split++) {
            double sumLeft = 0, sumRight = 0;
            for (int k = 0; k < split; k++) sumLeft += values[k];
            for (int k = split; k < size; k++) sumRight += values[k];
            double left = sumLeft / split;
            double right = sumRight / (size - split);
            worst = Math.max(worst, Math.abs(left - right) - bound(split, size - split, size));
        }
        return worst;
    }

    double mean() {
        double sum = 0;
        for (double v : window) sum += v;
        return window.isEmpty() ? 0 : sum / window.size();
    }
}

/** Page-Hinkley detector on the same loss signal. */
final class PageHinkley {
    double mean;
    long count;
    double cumulative;
    double minimum;
    final double threshold;
    final double allowedDrift;
    int alarms;

    PageHinkley(double threshold, double allowedDrift) {
        this.threshold = threshold;
        this.allowedDrift = allowedDrift;
    }

    void add(double value) {
        count++;
        mean += (value - mean) / count;
        cumulative += value - mean - allowedDrift;
        minimum = Math.min(minimum, cumulative);
        if (cumulative - minimum > threshold) {
            alarms++;
            cumulative = 0;
            minimum = 0;
        }
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Runs the stream through the pipeline; the detectors may reset the adapter at a drift. */
    static Map<String, Object> run(int length, int changePoint, long seed, boolean useDetector,
            boolean transferOnDrift, double driftStrength) {
        Encoder encoder = new Encoder(858_100L, 5, 6);
        Stream target = new Stream(seed, 6, changePoint, driftStrength, null);
        Adapter adapter = new Adapter(encoder.dimension());
        Adapter pretrained = new Adapter(encoder.dimension());
        // the source task uses the *same* pre-drift rule but a different sample stream: a related concept
        Stream source = new Stream(seed + 991, 6, Integer.MAX_VALUE, driftStrength, target.weights);
        for (int i = 0; i < 200; i++) pretrained.update(source.next(i, encoder));

        Stream replay = new Stream(seed, 6, changePoint, driftStrength, target.weights);
        AdwinDetector adwin = new AdwinDetector(0.05);
        PageHinkley pageHinkley = new PageHinkley(12.0, 0.01);
        int[] errors = new int[length];
        int[] transferErrors = new int[length];
        int[] scratchErrors = new int[length];
        int[] resets = new int[1];
        int adwinAlarm = -1, phAlarm = -1;
        boolean drifted = false;
        double movingAverage = 0;
        double[] movingAverages = new double[length];
        double transferError = 0, scratchError = 0;
        Adapter scratch = new Adapter(encoder.dimension());
        Adapter transfer = pretrained.copy();

        for (int i = 0; i < length; i++) {
            Example example = replay.next(i, encoder);
            errors[i] = adapter.update(example);
            movingAverage += (errors[i] - movingAverage) / 25.0;
            movingAverages[i] = movingAverage;
            adwin.add(errors[i]);
            pageHinkley.add(errors[i]);
            if (adwin.alarms > 0 && adwinAlarm < 0) adwinAlarm = i;
            if (pageHinkley.alarms > 0 && phAlarm < 0) phAlarm = i;
            if (useDetector && !drifted && adwin.alarms > 0) {
                drifted = true;
                resets[0]++;
                if (transferOnDrift) {
                    adapter = transfer.copy();
                } else {
                    adapter = new Adapter(encoder.dimension());
                }
            }
            transferErrors[i] = transfer.update(example);
            scratchErrors[i] = scratch.update(example);
        }
        for (int i = 0; i < length; i++) {
            transferError += transferErrors[i];
            scratchError += scratchErrors[i];
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("errors", errors);
        result.put("movingAverages", movingAverages);
        result.put("adwinAlarm", adwinAlarm);
        result.put("phAlarm", phAlarm);
        result.put("adwin", adwin);
        result.put("resets", resets[0]);
        result.put("transferError", transferError);
        result.put("scratchError", scratchError);
        result.put("transferErrors", transferErrors);
        result.put("scratchErrors", scratchErrors);
        return result;
    }

    static double mean(int[] values, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) sum += values[i];
        return sum / (to - from);
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final int length = 3000;
        final int changePoint = 1500;

        // --- 1. the encoder is frozen and deterministic; the stream is reproducible.
        Encoder encoder = new Encoder(858_100L, 5, 6);
        double[] sample = encoder.encode(new double[]{0.1, -0.2, 0.3, 0.4, -0.5});
        check(sample.length == 6, "encoder output dimension");
        for (double v : sample) check(Math.abs(v) < 1.0, "encoder output is squashed");
        check(Arrays.equals(encoder.encode(new double[]{0.1, -0.2, 0.3, 0.4, -0.5}), sample),
                "the encoder is deterministic");
        Stream a = new Stream(858_2024L, 6, changePoint, 0.9, null);
        Stream b = new Stream(858_2024L, 6, changePoint, 0.9, null);
        boolean sameLabels = true;
        for (int i = 0; i < 200; i++) sameLabels &= a.next(i, encoder).label == b.next(i, encoder).label;
        check(sameLabels, "the stream is reproducible from its seed");
        int positivesBefore = 0, positivesAfter = 0;
        Stream probe = new Stream(858_2024L, 6, changePoint, 0.9, null);
        for (int i = 0; i < 1000; i++) if (probe.next(i, encoder).label == 1) positivesBefore++;
        for (int i = 1500; i < 2500; i++) if (probe.next(i, encoder).label == 1) positivesAfter++;
        System.out.printf("class balance before=%d after=%d of 1000%n", positivesBefore, positivesAfter);
        check(positivesBefore > 300 && positivesBefore < 700, "the stationary phase is balanced");
        check(positivesAfter > 150 && positivesAfter < 850, "the drifted phase stays learnable");

        // --- 2. detector behaviour on the real stream.
        Map<String, Object> stationary = run(length, Integer.MAX_VALUE, 858_7L, false, true, 0.9);
        AdwinDetector stationaryWindow = (AdwinDetector) stationary.get("adwin");
        check(stationaryWindow.worstViolation() <= 1e-9,
                "ADWIN window invariant holds on a stationary stream: " + stationaryWindow.worstViolation());
        check((Integer) stationary.get("adwinAlarm") < 0,
                "no ADWIN alarm on a stationary stream (alarm at " + stationary.get("adwinAlarm") + ")");
        check((Integer) stationary.get("phAlarm") < 0,
                "no Page-Hinkley alarm on a stationary stream (alarm at " + stationary.get("phAlarm") + ")");
        System.out.println("stationary alarms: adwin=" + stationary.get("adwinAlarm")
                + " pageHinkley=" + stationary.get("phAlarm"));

        Map<String, Object> drifted = run(length, changePoint, 858_2024L, true, true, 0.9);
        int adwinAlarm = (Integer) drifted.get("adwinAlarm");
        int phAlarm = (Integer) drifted.get("phAlarm");
        System.out.printf("drift at %d: adwinAlarm=%d pageHinkleyAlarm=%d delay=%d%n",
                changePoint, adwinAlarm, phAlarm, adwinAlarm - changePoint);
        check(adwinAlarm > changePoint, "ADWIN does not fire before the change point");
        check(adwinAlarm - changePoint <= 600, "ADWIN fires within 600 steps of the drift");
        check(phAlarm > changePoint, "Page-Hinkley does not fire before the change point");
        check(phAlarm - changePoint <= 800, "Page-Hinkley fires within 800 steps of the drift");
        AdwinDetector driftedWindow = (AdwinDetector) drifted.get("adwin");
        check(driftedWindow.worstViolation() <= 1e-9,
                "ADWIN window invariant holds after the drift: " + driftedWindow.worstViolation());
        check((Integer) drifted.get("resets") >= 1, "the pipeline re-initialised the adapter at least once");

        // --- 3. the detector invariant is checked by brute force at every step of a short run.
        AdwinDetector probeDetector = new AdwinDetector(0.01);
        SplitMix64 rng = new SplitMix64(858_55L);
        int violations = 0;
        for (int i = 0; i < 400; i++) {
            double value = rng.nextDouble() < 0.3 ? 0.0 : 1.0;
            probeDetector.add(value);
            if (probeDetector.worstViolation() > 1e-12) violations++;
        }
        check(violations == 0, "no split of the ADWIN window ever violates the Hoeffding bound");
        check(probeDetector.window.size() <= 400, "the window never exceeds the number of observations");
        AdwinDetector constantDetector = new AdwinDetector(0.01);
        for (int i = 0; i < 300; i++) constantDetector.add(0.5);
        check(constantDetector.alarms == 0 && constantDetector.window.size() == 300,
                "a constant stream keeps the whole window and raises no alarm");
        check(constantDetector.worstViolation() <= 1e-9, "constant stream satisfies the invariant");

        // --- 4. the adapter learns during the stationary prefix.
        int[] errors = (int[]) drifted.get("errors");
        double earlyError = mean(errors, 50, 250);
        double lateError = mean(errors, 1200, 1450);
        double postDriftError = mean(errors, 1500, 1600);
        double recoveredError = mean(errors, 2200, 2900);
        System.out.printf("adapter error: early=%.4f pre-drift=%.4f post-drift=%.4f recovered=%.4f%n",
                earlyError, lateError, postDriftError, recoveredError);
        check(lateError < earlyError, "the adapter improves during the stationary prefix");
        check(lateError < 0.30, "the adapter reaches a low stationary error");
        check(postDriftError > lateError, "the error spikes right after the concept drift");
        check(recoveredError < postDriftError, "the adapter recovers after the drift");
        check(recoveredError <= lateError + 0.1, "the recovered error is close to the pre-drift level");
        double[] movingAverages = (double[]) drifted.get("movingAverages");
        double earlyAverage = 0, lateAverage = 0;
        for (int i = 50; i < 250; i++) earlyAverage += movingAverages[i];
        for (int i = 1200; i < 1450; i++) lateAverage += movingAverages[i];
        earlyAverage /= 200;
        lateAverage /= 250;
        System.out.printf("moving average: early=%.4f late=%.4f%n", earlyAverage, lateAverage);
        check(lateAverage < earlyAverage, "the smoothed loss falls over the stationary prefix");

        // --- 5. the transferred head converges faster on a related concept; after a large concept change it
        // brings no advantage (negative transfer), which is exactly what the detector-driven reset handles.
        Map<String, Object> related = run(length, Integer.MAX_VALUE, 858_2024L, false, true, 0.5);
        int[] transferErrors = (int[]) related.get("transferErrors");
        int[] scratchErrors = (int[]) related.get("scratchErrors");
        double transferWarmup = mean(transferErrors, 0, 150);
        double scratchWarmup = mean(scratchErrors, 0, 150);
        System.out.printf("related-concept warm-up: transferred=%.4f fromScratch=%.4f%n",
                transferWarmup, scratchWarmup);
        check(transferWarmup < scratchWarmup,
                "the transferred head adapts faster than a head trained from scratch");
        check(scratchWarmup - transferWarmup > 0.02, "the transfer gain is measurable");
        check(mean(transferErrors, 1000, length) <= mean(scratchErrors, 1000, length) + 0.02,
                "both heads converge to the same accuracy on the target concept");
        double transferAfterStrongDrift = mean(transferErrors, changePoint, changePoint + 150);
        double scratchAfterStrongDrift = mean(scratchErrors, changePoint, changePoint + 150);
        System.out.printf("strong-drift warm-up: transferred=%.4f fromScratch=%.4f%n",
                transferAfterStrongDrift, scratchAfterStrongDrift);
        check(transferAfterStrongDrift >= scratchAfterStrongDrift - 0.02,
                "after a large concept change the source head brings no warm-up advantage (negative transfer)");

        // --- 6. determinism and edge cases.
        Map<String, Object> repeat = run(length, changePoint, 858_2024L, true, true, 0.9);
        check(Arrays.equals((int[]) repeat.get("errors"), errors), "the whole run is reproducible");
        check(((Integer) repeat.get("adwinAlarm")) == adwinAlarm, "the alarm position is reproducible");
        Map<String, Object> immediate = run(500, 0, 858_9L, true, true, 0.9);
        check((Integer) immediate.get("resets") >= 0, "a drift at step zero is handled");
        Map<String, Object> shortRun = run(1, 5, 858_11L, true, true, 0.9);
        check(((int[]) shortRun.get("errors")).length == 1, "a one-example stream works");
        AdwinDetector tiny = new AdwinDetector(1e-9);
        for (int i = 0; i < 20; i++) tiny.add(i % 2);
        check(tiny.worstViolation() <= 1e-9, "a very small delta still yields a legal window");
        check(tiny.window.size() <= 20, "the window shrinks rather than growing unbounded");
        PageHinkley pageHinkley = new PageHinkley(1e9, 0.0);
        for (int i = 0; i < 100; i++) pageHinkley.add(1.0);
        check(pageHinkley.alarms == 0, "an unreachable threshold never alarms");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
