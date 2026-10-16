/** Task 715 - Implement sparse neural network training sketch with magnitude pruning.
 * (Tier: Expert, CTCI Beyond -- Machine Learning & AI)
 *
 * A two layer network (input -> ReLU hidden -> softmax) is trained by full batch gradient descent while
 * a *mask* decides which weights are allowed to be non-zero. Magnitude pruning removes the globally
 * smallest weights by absolute value and then fine-tunes what is left; keeping the mask explicit is the
 * whole point, because the pruned weights must stay exactly zero during training - their gradients are
 * zeroed before the update, so a pruned weight can never come back to life.
 *
 * The implementation is deliberately small enough to verify end to end: the analytic backpropagation
 * gradient is checked against central finite differences of the loss, the pruned weights are checked to
 * be exactly zero after further training, the removed set is checked to be exactly the smallest
 * magnitudes of the active set (so pruning really is magnitude pruning and not something else), pruning
 * everything leaves only the biases (the loss then equals the bias-only loss computed by hand), and
 * fine-tuning after a prune lowers the loss compared with the pruned model.
 *
 * Simplified: one hidden layer, a handful of features and a separable synthetic task, full batch
 * gradient descent instead of minibatches, and a single prune-and-finetune cycle rather than a full
 * iterative lottery-ticket search; no optimiser momentum or weight decay. Everything is initialised and
 * iterated from a fixed-seed splitmix64 stream, so runs are reproducible.
 */
import java.util.*;

class SparseMlp {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    final double[][] w1;
    final double[] b1;
    final double[][] w2;
    final double[] b2;
    final boolean[][] mask1;
    final boolean[][] mask2;
    private final double learningRate;
    private final double[][] gradientW1;
    private final double[] gradientB1;
    private final double[][] gradientW2;
    private final double[] gradientB2;

    SparseMlp(int inputSize, int hiddenSize, int outputSize, double learningRate, long seed) {
        if (inputSize < 1 || hiddenSize < 1 || outputSize < 2) {
            throw new IllegalArgumentException("network sizes must be positive");
        }
        if (!(learningRate > 0)) throw new IllegalArgumentException("learning rate must be positive");
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        w1 = new double[hiddenSize][inputSize];
        b1 = new double[hiddenSize];
        w2 = new double[outputSize][hiddenSize];
        b2 = new double[outputSize];
        mask1 = new boolean[hiddenSize][inputSize];
        mask2 = new boolean[outputSize][hiddenSize];
        gradientW1 = new double[hiddenSize][inputSize];
        gradientB1 = new double[hiddenSize];
        gradientW2 = new double[outputSize][hiddenSize];
        gradientB2 = new double[outputSize];
        long state = seed;
        double scale = 1.0 / Math.sqrt(Math.max(inputSize, 1));
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                state = mix(state + h * 131L + i);
                w1[h][i] = scale * (((state >>> 11) * 0x1.0p-53) * 2 - 1);
                mask1[h][i] = true;
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                state = mix(state + 17L + o * 313L + h);
                w2[o][h] = scale * (((state >>> 11) * 0x1.0p-53) * 2 - 1);
                mask2[o][h] = true;
            }
        }
    }

    private static long mix(long value) {
        long z = value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int weightCount() {
        return hiddenSize * inputSize + outputSize * hiddenSize;
    }

    /** Softmax probabilities for one example. */
    double[] forward(double[] x) {
        if (x.length != inputSize) throw new IllegalArgumentException("input size mismatch");
        double[] hidden = new double[hiddenSize];
        for (int h = 0; h < hiddenSize; h++) {
            double sum = b1[h];
            for (int i = 0; i < inputSize; i++) sum += w1[h][i] * x[i];
            hidden[h] = Math.max(0, sum);
        }
        double[] scores = new double[outputSize];
        double best = Double.NEGATIVE_INFINITY;
        for (int o = 0; o < outputSize; o++) {
            double sum = b2[o];
            for (int h = 0; h < hiddenSize; h++) sum += w2[o][h] * hidden[h];
            scores[o] = sum;
            best = Math.max(best, sum);
        }
        double total = 0;
        for (int o = 0; o < outputSize; o++) {
            scores[o] = Math.exp(scores[o] - best);
            total += scores[o];
        }
        for (int o = 0; o < outputSize; o++) scores[o] /= total;
        return scores;
    }

    double loss(double[][] features, int[] labels) {
        double total = 0;
        for (int sample = 0; sample < labels.length; sample++) {
            double[] probabilities = forward(features[sample]);
            total -= Math.log(Math.max(probabilities[labels[sample]], 1e-300));
        }
        return total / labels.length;
    }

    double accuracy(double[][] features, int[] labels) {
        int correct = 0;
        for (int sample = 0; sample < labels.length; sample++) {
            double[] probabilities = forward(features[sample]);
            int predicted = probabilities[0] >= probabilities[1] ? 0 : 1;
            if (predicted == labels[sample]) correct++;
        }
        return correct / (double) labels.length;
    }

    /** Mean gradient over the data set; masked weights get a zero derivative by construction. */
    void computeGradient(double[][] features, int[] labels) {
        for (double[] row : gradientW1) Arrays.fill(row, 0);
        Arrays.fill(gradientB1, 0);
        for (double[] row : gradientW2) Arrays.fill(row, 0);
        Arrays.fill(gradientB2, 0);
        int samples = labels.length;
        for (int sample = 0; sample < samples; sample++) {
            double[] x = features[sample];
            double[] hidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++) {
                double sum = b1[h];
                for (int i = 0; i < inputSize; i++) sum += w1[h][i] * x[i];
                hidden[h] = sum;
            }
            double[] activation = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++) activation[h] = Math.max(0, hidden[h]);
            double[] scores = new double[outputSize];
            double best = Double.NEGATIVE_INFINITY;
            for (int o = 0; o < outputSize; o++) {
                double sum = b2[o];
                for (int h = 0; h < hiddenSize; h++) sum += w2[o][h] * activation[h];
                scores[o] = sum;
                best = Math.max(best, sum);
            }
            double total = 0;
            for (int o = 0; o < outputSize; o++) {
                scores[o] = Math.exp(scores[o] - best);
                total += scores[o];
            }
            for (int o = 0; o < outputSize; o++) scores[o] /= total;
            double[] deltaOutput = new double[outputSize];
            for (int o = 0; o < outputSize; o++) {
                deltaOutput[o] = (scores[o] - (o == labels[sample] ? 1 : 0)) / samples;
            }
            for (int o = 0; o < outputSize; o++) {
                gradientB2[o] += deltaOutput[o];
                for (int h = 0; h < hiddenSize; h++) {
                    gradientW2[o][h] += deltaOutput[o] * activation[h];
                }
            }
            double[] deltaHidden = new double[hiddenSize];
            for (int h = 0; h < hiddenSize; h++) {
                double sum = 0;
                for (int o = 0; o < outputSize; o++) sum += w2[o][h] * deltaOutput[o];
                deltaHidden[h] = hidden[h] > 0 ? sum : 0;
            }
            for (int h = 0; h < hiddenSize; h++) {
                gradientB1[h] += deltaHidden[h];
                for (int i = 0; i < inputSize; i++) gradientW1[h][i] += deltaHidden[h] * x[i];
            }
        }
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (!mask1[h][i]) gradientW1[h][i] = 0;
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (!mask2[o][h]) gradientW2[o][h] = 0;
            }
        }
    }

    /** One gradient descent step on the whole data set, respecting the mask. */
    void trainStep(double[][] features, int[] labels) {
        computeGradient(features, labels);
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (mask1[h][i]) w1[h][i] -= learningRate * gradientW1[h][i];
            }
            b1[h] -= learningRate * gradientB1[h];
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (mask2[o][h]) w2[o][h] -= learningRate * gradientW2[o][h];
            }
            b2[o] -= learningRate * gradientB2[o];
        }
    }

    void train(double[][] features, int[] labels, int steps) {
        for (int step = 0; step < steps; step++) trainStep(features, labels);
    }

    /** Analytic gradient of the mean loss, flattened as [w1..., b1..., w2..., b2...]. */
    double[] flatGradient(double[][] features, int[] labels) {
        computeGradient(features, labels);
        double[] flat = new double[weightCount() + hiddenSize + outputSize];
        int cursor = 0;
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) flat[cursor++] = gradientW1[h][i];
        }
        for (int h = 0; h < hiddenSize; h++) flat[cursor++] = gradientB1[h];
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) flat[cursor++] = gradientW2[o][h];
        }
        for (int o = 0; o < outputSize; o++) flat[cursor++] = gradientB2[o];
        return flat;
    }

    /** Finite differences of the loss with respect to one weight: the gradient oracle. */
    double numericWeightGradient(double[][] features, int[] labels, int layer, int first, int second,
                                 double epsilon) {
        double saved = readWeight(layer, first, second);
        writeWeight(layer, first, second, saved + epsilon);
        double up = loss(features, labels);
        writeWeight(layer, first, second, saved - epsilon);
        double down = loss(features, labels);
        writeWeight(layer, first, second, saved);
        return (up - down) / (2 * epsilon);
    }

    private double readWeight(int layer, int first, int second) {
        return layer == 1 ? w1[first][second] : w2[first][second];
    }

    private void writeWeight(int layer, int first, int second, double value) {
        if (layer == 1) w1[first][second] = value;
        else w2[first][second] = value;
    }

    /** Global magnitude pruning: the fraction of the *active* weights with the smallest |value| dies. */
    int prune(double fraction) {
        if (fraction < 0 || fraction > 1) throw new IllegalArgumentException("fraction in [0,1]");
        List<int[]> active = new ArrayList<>();
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) if (mask1[h][i]) active.add(new int[]{1, h, i});
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) if (mask2[o][h]) active.add(new int[]{2, o, h});
        }
        active.sort((left, right) -> {
            double leftValue = Math.abs(readWeight(left[0], left[1], left[2]));
            double rightValue = Math.abs(readWeight(right[0], right[1], right[2]));
            int byMagnitude = Double.compare(leftValue, rightValue);
            if (byMagnitude != 0) return byMagnitude;
            if (left[0] != right[0]) return Integer.compare(left[0], right[0]);
            if (left[1] != right[1]) return Integer.compare(left[1], right[1]);
            return Integer.compare(left[2], right[2]);
        });
        int toRemove = (int) Math.round(fraction * active.size());
        for (int index = 0; index < toRemove; index++) {
            int[] target = active.get(index);
            if (target[0] == 1) {
                mask1[target[1]][target[2]] = false;
                w1[target[1]][target[2]] = 0;
            } else {
                mask2[target[1]][target[2]] = false;
                w2[target[1]][target[2]] = 0;
            }
        }
        return toRemove;
    }

    double sparsity() {
        int dead = 0;
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) if (!mask1[h][i]) dead++;
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) if (!mask2[o][h]) dead++;
        }
        return dead / (double) weightCount();
    }

    /** The largest magnitude still alive, and the smallest, for the pruning-order checks. */
    double smallestActiveMagnitude() {
        double smallest = Double.MAX_VALUE;
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (mask1[h][i]) smallest = Math.min(smallest, Math.abs(w1[h][i]));
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (mask2[o][h]) smallest = Math.min(smallest, Math.abs(w2[o][h]));
            }
        }
        return smallest;
    }

    double largestDeadMagnitude() {
        double largest = 0;
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (!mask1[h][i]) largest = Math.max(largest, Math.abs(w1[h][i]));
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (!mask2[o][h]) largest = Math.max(largest, Math.abs(w2[o][h]));
            }
        }
        return largest;
    }

    /** All weights must be exactly zero where the mask is off. */
    boolean maskRespected() {
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (!mask1[h][i] && w1[h][i] != 0.0) return false;
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (!mask2[o][h] && w2[o][h] != 0.0) return false;
            }
        }
        return true;
    }

    double prunedGradientMagnitude(double[][] features, int[] labels) {
        computeGradient(features, labels);
        double total = 0;
        for (int h = 0; h < hiddenSize; h++) {
            for (int i = 0; i < inputSize; i++) {
                if (!mask1[h][i]) total += Math.abs(gradientW1[h][i]);
            }
        }
        for (int o = 0; o < outputSize; o++) {
            for (int h = 0; h < hiddenSize; h++) {
                if (!mask2[o][h]) total += Math.abs(gradientW2[o][h]);
            }
        }
        return total;
    }
}

public class Solution {
    static final int FEATURES = 4;
    static final int SAMPLES = 40;

    public static void main(String[] args) {
        double[][] features = new double[SAMPLES][FEATURES];
        int[] labels = new int[SAMPLES];
        long state = 0x715L;
        for (int sample = 0; sample < SAMPLES; sample++) {
            for (int k = 0; k < FEATURES; k++) {
                state = state * 6364136223846793005L + 1442695040888963407L;
                features[sample][k] = ((state >>> 11) * 0x1.0p-53);
            }
            labels[sample] = features[sample][0] + features[sample][1] > 1.0 ? 1 : 0;
        }
        int positives = 0;
        for (int label : labels) positives += label;
        check(positives >= 10 && positives <= SAMPLES - 10, "both classes are populated: " + positives);

        SparseMlp network = new SparseMlp(FEATURES, 12, 2, 0.5, 0x7155EEDL);
        check(network.weightCount() == 12 * FEATURES + 2 * 12, "weight accounting");
        check(network.sparsity() == 0.0, "a fresh network is dense");

        // --- the gradient oracle -------------------------------------------------
        double[] flat = network.flatGradient(features, labels);
        double numericError = 0;
        for (int h = 0; h < 3; h++) {
            for (int i = 0; i < FEATURES; i++) {
                double numeric = network.numericWeightGradient(features, labels, 1, h, i, 1e-5);
                double analytic = flat[h * FEATURES + i];
                numericError = Math.max(numericError,
                        Math.abs(numeric - analytic) / Math.max(1e-3, Math.abs(analytic)));
            }
        }
        for (int o = 0; o < 2; o++) {
            for (int h = 0; h < 3; h++) {
                double numeric = network.numericWeightGradient(features, labels, 2, o, h, 1e-5);
                double analytic = flat[12 * FEATURES + 12 + o * 12 + h];
                numericError = Math.max(numericError,
                        Math.abs(numeric - analytic) / Math.max(1e-3, Math.abs(analytic)));
            }
        }
        check(numericError < 1e-6, "backpropagation must match finite differences: " + numericError);

        // --- training on the dense network ---------------------------------------
        double initialLoss = network.loss(features, labels);
        network.train(features, labels, 400);
        double trainedLoss = network.loss(features, labels);
        check(trainedLoss < 0.3 * initialLoss, "training must reduce the loss: " + trainedLoss);
        check(network.accuracy(features, labels) >= 0.9,
                "the dense network must learn the task: " + network.accuracy(features, labels));

        // --- pruning is magnitude pruning ----------------------------------------
        int active = network.weightCount();
        int removed = network.prune(0.0);
        check(removed == 0 && network.sparsity() == 0.0, "pruning nothing changes nothing");
        removed = network.prune(0.5);
        check(removed == Math.round(0.5 * active), "half of the active weights should die: " + removed);
        check(Math.abs(network.sparsity() - 0.5) < 1e-9,
                "half of all weights should be dead: " + network.sparsity());
        check(network.largestDeadMagnitude() <= network.smallestActiveMagnitude(),
                "pruning must remove the smallest magnitudes first");
        check(network.maskRespected(), "pruned weights must be exactly zero");
        double prunedLoss = network.loss(features, labels);
        check(prunedLoss >= trainedLoss - 1e-9, "pruning cannot improve the training loss by itself");

        // --- fine-tuning respects the mask ---------------------------------------
        check(network.prunedGradientMagnitude(features, labels) == 0.0,
                "pruned weights must have exactly zero gradient");
        network.train(features, labels, 300);
        check(network.maskRespected(), "fine-tuning must not revive pruned weights");
        check(network.prunedGradientMagnitude(features, labels) == 0.0,
                "pruned weights keep a zero gradient after training");
        double fineTunedLoss = network.loss(features, labels);
        check(fineTunedLoss < prunedLoss, "fine-tuning must lower the loss: " + fineTunedLoss + " vs "
                + prunedLoss);
        double sparseAccuracy = network.accuracy(features, labels);
        check(sparseAccuracy >= 0.9, "the pruned network must still classify: " + sparseAccuracy);

        // --- pruning everything leaves only the biases ---------------------------
        SparseMlp hollow = new SparseMlp(FEATURES, 12, 2, 0.5, 0x7155EEDL);
        int totalRemoved = hollow.prune(1.0);
        check(totalRemoved == hollow.weightCount(), "pruning everything removes every weight");
        check(hollow.sparsity() == 1.0, "the network is fully sparse");
        check(hollow.maskRespected(), "all weights are zero");
        double biasOnlyLoss = hollow.loss(features, labels);
        double expectedBiasLoss = Math.log(2.0);
        check(Math.abs(biasOnlyLoss - expectedBiasLoss) < 1e-9,
                "with zero weights the softmax is uniform, loss " + biasOnlyLoss + " vs " + expectedBiasLoss);
        hollow.train(features, labels, 200);
        check(hollow.maskRespected(), "training a fully pruned network keeps every weight at zero");
        double baseRate = positives / (double) SAMPLES;
        double classEntropy = -(baseRate * Math.log(baseRate) + (1 - baseRate) * Math.log(1 - baseRate));
        double hollowLoss = hollow.loss(features, labels);
        check(hollowLoss <= biasOnlyLoss + 1e-12, "training the biases cannot increase the loss");
        check(hollowLoss >= classEntropy - 1e-9,
                "a bias-only model cannot beat the class entropy " + classEntropy + ": " + hollowLoss);

        // --- deeper sparsity degrades, and is measured honestly -------------------
        SparseMlp aggressive = new SparseMlp(FEATURES, 12, 2, 0.5, 0x7155EEDL);
        aggressive.train(features, labels, 400);
        aggressive.prune(0.95);
        aggressive.train(features, labels, 200);
        check(aggressive.maskRespected(), "aggressive pruning still respects the mask");
        check(Math.abs(aggressive.sparsity() - 0.95) <= 1.0 / aggressive.weightCount() + 1e-9,
                "sparsity accounting at 95%: " + aggressive.sparsity());
        check(aggressive.loss(features, labels) <= biasOnlyLoss + 1e-9,
                "a 95% sparse network is still better than guessing");

        // --- rejection and determinism --------------------------------------------
        check(rejects(() -> new SparseMlp(0, 4, 2, 0.1, 1L)), "zero features rejected");
        check(rejects(() -> new SparseMlp(4, 4, 1, 0.1, 1L)), "one output class rejected");
        check(rejects(() -> new SparseMlp(4, 4, 2, 0.0, 1L)), "zero learning rate rejected");
        check(rejects(() -> network.prune(-0.1)), "negative prune fraction rejected");
        check(rejects(() -> network.prune(1.5)), "prune fraction above one rejected");
        check(rejects(() -> network.forward(new double[FEATURES + 1])), "input size mismatch rejected");
        SparseMlp rebuilt = new SparseMlp(FEATURES, 12, 2, 0.5, 0x7155EEDL);
        check(Math.abs(rebuilt.loss(features, labels) - initialLoss) < 1e-12,
                "the same seed must give the same initialisation");
        rebuilt.train(features, labels, 400);
        check(Math.abs(rebuilt.loss(features, labels) - trainedLoss) < 1e-12,
                "training must be deterministic");

        System.out.println("loss " + String.format(Locale.ROOT, "%.5f", initialLoss) + " -> "
                + String.format(Locale.ROOT, "%.5f", trainedLoss) + " -> prune 50% -> "
                + String.format(Locale.ROOT, "%.5f", prunedLoss) + " -> fine-tune -> "
                + String.format(Locale.ROOT, "%.5f", fineTunedLoss) + ", accuracy "
                + String.format(Locale.ROOT, "%.3f", sparseAccuracy));
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
