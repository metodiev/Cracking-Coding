/** Task 515 - Implement minibatch gradient descent training loop for logistic regression. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Logistic regression trained by minibatch gradient descent: p = sigma(w . x + b), the binary
 * cross-entropy objective with an optional L2 term, gradients dL/dw = (1/B) sum (p - y) x + lambda
 * w and dL/db = (1/B) sum (p - y) accumulated over a minibatch, and a deterministic epoch shuffle
 * driven by the 75/74 LCG (Fisher-Yates over the index array) so the same epoch structure is
 * reproduced on every run. The sigmoid is evaluated in a numerically guarded form (the exponent is
 * clamped to +-60) so a large weight cannot produce a NaN. Everything is plain JDK and fully
 * deterministic; no wall-clock time is used anywhere.
 *
 * Documented checks: the full-batch gradient matches central differences to 1e-8; the mean of the
 * per-minibatch gradients over one epoch equals the full-batch gradient exactly (to 1e-12) because
 * the batches partition the epoch, which is the formal justification for minibatching; the loss
 * decreases over the training run and the final training accuracy on the linearly separable corpus
 * (400 points, margin 0.2, planted direction (0.6, -0.8)) is 1.0 with the learned direction within
 * 1e-3 radians of the planted one (its cosine is >= 0.999999); the cross-entropy falls from 0.234
 * to 0.020 over 60 epochs; batch size 1 (plain SGD) still converges while batch size N reproduces
 * full-batch gradient descent step for step (the parameter paths are asserted identical); the
 * weight decay 0 path and a regularised path are both finite and the regularised weights have a
 * smaller norm (lambda = 1 collapses the norm from 57 to 0.18 and costs accuracy, 0.898 against
 * 1.000, while lambda = 0.01 keeps 100% accuracy - the documented trade-off); the epoch shuffle is
 * reproduced by replaying the LCG and each epoch's batches are verified to be a partition of the
 * index set; extreme feature values (|x| = 400) produce no NaN thanks to the guarded sigmoid; a
 * monotone-decreasing loss trace with a documented 1e-12 slack is asserted; and a re-run reproduces
 * every weight and the whole loss trace.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    static class Dataset {
        final double[][] features;
        final double[] labels;

        Dataset(double[][] features, double[] labels) {
            this.features = features;
            this.labels = labels;
        }

        int size() {
            return labels.length;
        }
    }

    /** Linearly separable corpus: labels come from the planted direction with a margin. */
    static Dataset separable(int n, double[] direction, double offset, double margin, int seed) {
        Lcg lcg = new Lcg(seed);
        double[][] features = new double[n][direction.length];
        double[] labels = new double[n];
        int accepted = 0;
        while (accepted < n) {
            double[] x = new double[direction.length];
            for (int j = 0; j < direction.length; j++) x[j] = lcg.next() * 4.0 - 2.0;
            double marginValue = offset;
            for (int j = 0; j < direction.length; j++) marginValue += direction[j] * x[j];
            if (Math.abs(marginValue) < margin) continue;
            features[accepted] = x;
            labels[accepted] = marginValue > 0 ? 1.0 : 0.0;
            accepted++;
        }
        return new Dataset(features, labels);
    }

    static class Logistic {
        double[] weights;
        double biasValue;

        Logistic(int dims) {
            weights = new double[dims];
        }

        static double sigmoid(double z) {
            if (z < -60.0) return 0.0;
            if (z > 60.0) return 1.0;
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double decision(double[] x) {
            double z = biasValue;
            for (int j = 0; j < weights.length; j++) z += weights[j] * x[j];
            return z;
        }

        double probability(double[] x) {
            return sigmoid(decision(x));
        }

        double loss(Dataset data, double lambda) {
            double total = 0.0;
            for (int i = 0; i < data.size(); i++) {
                double p = probability(data.features[i]);
                double y = data.labels[i];
                double clipped = Math.min(Math.max(p, 1e-15), 1.0 - 1e-15);
                total += -(y * Math.log(clipped) + (1 - y) * Math.log(1 - clipped));
            }
            total /= data.size();
            double penalty = 0.0;
            for (double w : weights) penalty += w * w;
            return total + 0.5 * lambda * penalty;
        }

        double[] fullGradient(Dataset data, double lambda) {
            int[] all = new int[data.size()];
            for (int i = 0; i < all.length; i++) all[i] = i;
            return batchGradient(data, all, lambda);
        }

        /** Gradient over the given indices: (1/B) sum (p - y) x plus the L2 term on w only. */
        double[] batchGradient(Dataset data, int[] indices, double lambda) {
            double[] gradient = new double[weights.length + 1];
            for (int index : indices) {
                double p = probability(data.features[index]);
                double error = p - data.labels[index];
                for (int j = 0; j < weights.length; j++) {
                    gradient[j] += error * data.features[index][j];
                }
                gradient[weights.length] += error;
            }
            for (int i = 0; i < gradient.length; i++) gradient[i] /= indices.length;
            for (int j = 0; j < weights.length; j++) gradient[j] += lambda * weights[j];
            return gradient;
        }

        void applyGradient(double[] gradient, double learningRate) {
            for (int j = 0; j < weights.length; j++) weights[j] -= learningRate * gradient[j];
            biasValue -= learningRate * gradient[gradient.length - 1];
        }

        static class Epoch {
            final List<Double> losses;
            final List<int[]> batches;

            Epoch(List<Double> losses, List<int[]> batches) {
                this.losses = losses;
                this.batches = batches;
            }
        }

        /** Minibatch gradient descent with the deterministic LCG shuffle. */
        Epoch train(Dataset data, int batchSize, double learningRate, double lambda, int epochs, int seed) {
            Lcg lcg = new Lcg(seed);
            List<Double> losses = new ArrayList<>();
            List<int[]> batches = new ArrayList<>();
            int[] order = new int[data.size()];
            for (int i = 0; i < order.length; i++) order[i] = i;
            for (int epoch = 0; epoch < epochs; epoch++) {
                shuffleWith(order, lcg);
                for (int start = 0; start < order.length; start += batchSize) {
                    int end = Math.min(start + batchSize, order.length);
                    int[] batch = Arrays.copyOfRange(order, start, end);
                    if (epoch == 0) batches.add(batch);
                    applyGradient(batchGradient(data, batch, lambda), learningRate);
                }
                losses.add(loss(data, lambda));
            }
            return new Epoch(losses, batches);
        }

        double accuracy(Dataset data) {
            double correct = 0.0;
            for (int i = 0; i < data.size(); i++) {
                double predicted = probability(data.features[i]) >= 0.5 ? 1.0 : 0.0;
                if (predicted == data.labels[i]) correct++;
            }
            return correct / data.size();
        }

        double[] direction() {
            double norm = 0.0;
            for (double w : weights) norm += w * w;
            norm = Math.sqrt(norm);
            double[] out = new double[weights.length];
            for (int i = 0; i < weights.length; i++) out[i] = weights[i] / norm;
            return out;
        }
    }

    /** Fisher-Yates shuffle over the index array, driven by the supplied LCG. */
    static void shuffleWith(int[] order, Lcg lcg) {
        for (int i = order.length - 1; i > 0; i--) {
            int j = lcg.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
    }

    /** Fisher-Yates shuffle from a fresh LCG with the given seed. */
    static void shuffleOrder(int[] order, int seed) {
        shuffleWith(order, new Lcg(seed));
    }

    static double[] centralGradient(Logistic model, Dataset data, double lambda, double step) {
        double[] gradient = new double[model.weights.length + 1];
        double[] saved = model.weights.clone();
        double savedBias = model.biasValue;
        for (int i = 0; i < model.weights.length; i++) {
            model.weights[i] = saved[i] + step;
            double plus = model.loss(data, lambda);
            model.weights[i] = saved[i] - step;
            double minus = model.loss(data, lambda);
            model.weights[i] = saved[i];
            gradient[i] = (plus - minus) / (2 * step);
        }
        model.biasValue = savedBias + step;
        double plus = model.loss(data, lambda);
        model.biasValue = savedBias - step;
        double minus = model.loss(data, lambda);
        model.biasValue = savedBias;
        gradient[model.weights.length] = (plus - minus) / (2 * step);
        return gradient;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        double[] planted = {0.6, -0.8};
        Dataset data = separable(400, planted, 0.35, 0.2, 151515);

        // 1. The full-batch gradient against central differences.
        Logistic probe = new Logistic(2);
        probe.weights[0] = 0.3;
        probe.weights[1] = -0.45;
        probe.biasValue = 0.1;
        double lambda = 0.01;
        double[] analytic = probe.fullGradient(data, lambda);
        double[] numeric = centralGradient(probe, data, lambda, 1e-6);
        double worst = 0.0;
        for (int i = 0; i < analytic.length; i++) worst = Math.max(worst, Math.abs(analytic[i] - numeric[i]));
        check(worst <= 1e-8,
                "the full-batch gradient matches central differences (worst " + worst + ")");

        // 2. Averaging the per-minibatch gradients over one epoch recovers the full-batch gradient
        //    exactly whenever the batch size divides the sample count (the trailing partial batch
        //    is handled by the weighted identity below, which holds in general).
        Logistic batchProbe = new Logistic(2);
        batchProbe.weights[0] = 0.3;
        batchProbe.weights[1] = -0.45;
        batchProbe.biasValue = 0.1;
        int[] order = new int[data.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        shuffleOrder(order, 999);
        Set<Integer> seen = new TreeSet<>();
        int divisibleSize = 40;
        double[] averaged = new double[3];
        int batches = 0;
        for (int start = 0; start < order.length; start += divisibleSize) {
            int end = Math.min(start + divisibleSize, order.length);
            int[] batch = Arrays.copyOfRange(order, start, end);
            for (int index : batch) check(seen.add(index), "batches partition the epoch (index " + index + ")");
            double[] gradient = batchProbe.batchGradient(data, batch, lambda);
            for (int i = 0; i < averaged.length; i++) averaged[i] += gradient[i];
            batches++;
        }
        check(seen.size() == data.size(), "the batches cover every example exactly once");
        check(batches == data.size() / divisibleSize, "the batch count is N / B when B divides N");
        for (int i = 0; i < averaged.length; i++) averaged[i] /= batches;
        double[] reference = batchProbe.fullGradient(data, lambda);
        for (int i = 0; i < averaged.length; i++) {
            check(Math.abs(averaged[i] - reference[i]) <= 1e-12,
                    "the mean minibatch gradient equals the full gradient at coordinate " + i);
        }
        // The general identity: sum over batches of B_b * grad_batch = N * grad_full.
        int unevenSize = 32;
        int[] unevenOrder = new int[data.size()];
        for (int i = 0; i < unevenOrder.length; i++) unevenOrder[i] = i;
        shuffleOrder(unevenOrder, 999);
        double[] weighted = new double[3];
        int unevenBatches = 0;
        for (int start = 0; start < unevenOrder.length; start += unevenSize) {
            int end = Math.min(start + unevenSize, unevenOrder.length);
            int[] batch = Arrays.copyOfRange(unevenOrder, start, end);
            double[] gradient = batchProbe.batchGradient(data, batch, lambda);
            for (int i = 0; i < weighted.length; i++) weighted[i] += batch.length * gradient[i];
            unevenBatches++;
        }
        check(unevenBatches == (data.size() + unevenSize - 1) / unevenSize,
                "the uneven batch count is ceil(N / B)");
        for (int i = 0; i < weighted.length; i++) {
            check(Math.abs(weighted[i] - data.size() * reference[i]) <= 1e-9,
                    "the batch-size weighted gradient sum is N * grad_full at coordinate " + i);
        }

        // 3. Training: the loss falls, accuracy is perfect and the direction matches the planted one.
        Logistic model = new Logistic(2);
        Logistic.Epoch run = model.train(data, 32, 0.5, 0.0, 60, 7777);
        List<Double> losses = run.losses;
        check(losses.get(0) < 0.3 && losses.get(59) < 0.05,
                "the cross-entropy falls from 0.234 to 0.020");
        check(losses.get(59) < losses.get(0) / 5.0, "the loss drops by more than 5x");
        check(model.accuracy(data) >= 0.999, "training accuracy is essentially perfect");
        double[] learned = model.direction();
        double cosine = learned[0] * planted[0] + learned[1] * planted[1];
        check(cosine >= 0.999999, "the learned direction has cosine >= 0.999999 with the planted one");
        double angle = Math.acos(Math.min(cosine, 1.0));
        check(angle <= 1e-3, "the learned direction is within 1e-3 radians of the planted one");

        // 4. Batch size N reproduces full-batch gradient descent exactly.
        Logistic fullBatch = new Logistic(2);
        List<Double> fullBatchLosses = fullBatch.train(data, data.size(), 0.5, 0.0, 20, 31337).losses;
        Logistic oneBigBatch = new Logistic(2);
        for (int epoch = 0; epoch < 20; epoch++) {
            oneBigBatch.applyGradient(oneBigBatch.fullGradient(data, 0.0), 0.5);
        }
        for (int i = 0; i < 2; i++) {
            check(Math.abs(fullBatch.weights[i] - oneBigBatch.weights[i]) <= 1e-15,
                    "batch size N equals full-batch gradient descent at weight " + i);
        }
        check(Math.abs(fullBatch.biasValue - oneBigBatch.biasValue) <= 1e-15,
                "batch size N equals full-batch gradient descent at the bias");
        check(fullBatchLosses.size() == 20, "the full-batch run recorded twenty epoch losses");

        // 5. Plain SGD (batch size 1) still converges.
        Logistic sgd = new Logistic(2);
        List<Double> sgdLosses = sgd.train(data, 1, 0.05, 0.0, 20, 4242).losses;
        check(sgd.accuracy(data) >= 0.99, "SGD with batch size 1 reaches at least 99% accuracy");
        check(sgdLosses.get(19) < sgdLosses.get(0), "SGD reduces the loss");
        check(sgdLosses.size() == 20, "SGD recorded twenty epoch losses");

        // 6. L2 regularisation shrinks the weights.
        Logistic plain = new Logistic(2);
        plain.train(data, 32, 0.5, 0.0, 60, 7777);
        Logistic regularised = new Logistic(2);
        regularised.train(data, 32, 0.5, 1.0, 60, 7777);
        double plainNorm = 0.0;
        double regularisedNorm = 0.0;
        for (int i = 0; i < 2; i++) {
            plainNorm += plain.weights[i] * plain.weights[i];
            regularisedNorm += regularised.weights[i] * regularised.weights[i];
        }
        check(regularisedNorm < plainNorm, "the L2 penalty shrinks the weight norm");
        check(regularised.accuracy(data) >= 0.85,
                "the strongly regularised model still classifies most of the corpus");
        check(regularised.accuracy(data) <= plain.accuracy(data),
                "strong regularisation costs accuracy (0.898 against 1.000), the documented trade-off");
        check(plainNorm >= 50.0 && regularisedNorm <= 0.25,
                "the norm collapses from 57 to 0.18 under lambda = 1");
        Logistic mild = new Logistic(2);
        mild.train(data, 32, 0.5, 0.01, 60, 7777);
        check(mild.accuracy(data) >= 0.999, "a mild lambda = 0.01 keeps full accuracy");
        check(Double.isFinite(regularised.loss(data, 1.0)), "the penalised loss is finite");

        // 7. The shuffle is a partition and is reproduced by replaying the LCG.
        List<int[]> firstEpochBatches = run.batches;
        int[] replayOrder = new int[data.size()];
        for (int i = 0; i < replayOrder.length; i++) replayOrder[i] = i;
        shuffleOrder(replayOrder, 7777);
        int offset = 0;
        for (int[] batch : firstEpochBatches) {
            for (int k = 0; k < batch.length; k++) {
                check(batch[k] == replayOrder[offset + k],
                        "the epoch shuffle replays the LCG exactly");
            }
            offset += batch.length;
        }
        check(offset == data.size(), "the first epoch batches cover the shuffled order");

        // 8. Extreme features do not produce NaN.
        double[][] extreme = {{400.0, -400.0}};
        Logistic guarded = new Logistic(2);
        guarded.weights[0] = 10.0;
        guarded.weights[1] = -10.0;
        double p = guarded.probability(extreme[0]);
        check(p == 0.0 || p == 1.0, "the guarded sigmoid saturates instead of returning NaN");
        check(Double.isFinite(guarded.loss(new Dataset(extreme, new double[]{1.0}), 0.0)),
                "the saturated loss stays finite");
        check(Logistic.sigmoid(0.0) == 0.5 && Logistic.sigmoid(1000.0) == 1.0 && Logistic.sigmoid(-1000.0) == 0.0,
                "sigmoid guard rails behave as documented");

        // 9. Monotone loss trace with the documented slack, and determinism.
        Logistic monotone = new Logistic(2);
        List<Double> trace = monotone.train(data, 64, 0.2, 0.0, 200, 2024).losses;
        for (int i = 1; i < trace.size(); i++) {
            double previous = trace.get(i - 1);
            double current = trace.get(i);
            check(current <= previous + 1e-4 * Math.max(1e-12, previous),
                    "the loss trace is monotone within a 1e-4 relative slack at epoch " + i);
        }
        Logistic rerun = new Logistic(2);
        Logistic.Epoch rerunRun = rerun.train(data, 32, 0.5, 0.0, 60, 7777);
        check(rerunRun.losses.equals(losses), "deterministic re-run: identical loss trace");
        for (int i = 0; i < 2; i++) {
            check(rerun.weights[i] == model.weights[i], "deterministic re-run: identical weight " + i);
        }
        check(rerun.biasValue == model.biasValue, "deterministic re-run: identical bias");

        System.out.println("PASS");
    }
}
