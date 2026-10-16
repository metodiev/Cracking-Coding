/** Task 514 - Implement small neural-network forward/backprop demo (no libraries). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A from-scratch 2-3-1 multilayer perceptron for XOR: the hidden layer applies h_j = tanh(W1[j] . x
 * + b1[j]) and the output is the logistic p = sigma(W2 . h + b2); the loss is the mean squared
 * error (1 / N) sum (p_i - y_i)^2. Backpropagation is hand-derived (no autodiff, no libraries):
 * with delta2 = (2 / N) sum (p_i - y_i) p_i (1 - p_i) the output gradient is dL/dW2[j] = delta2
 * h_j, dL/db2 = delta2, and the hidden gradients follow from delta1_j = delta2 W2[j] (1 - h_j^2),
 * giving dL/dW1[j][k] = delta1_j x_k and dL/db1[j] = delta1_j. Parameters are exposed as one flat
 * vector (13 numbers for 2-3-1) precisely so the analytic gradient can be checked against central
 * differences of the *loss function itself* - the backprop code and the check share nothing but the
 * forward pass. Initialisation uses the fixed 75/74 LCG, and the model is trained with full-batch
 * gradient descent, so everything is deterministic.
 *
 * Documented checks: the forward pass is compared against an independently written double loop
 * (1e-15); the zero-parameter network has loss exactly 0.25 because sigma(0) = 0.5 and the XOR
 * targets are 0/1; all 13 analytic gradients match central differences of the flat loss to 1e-6 at
 * the initial point (the worst deviation is 4.5e-11) AND after training; the loss trace of
 * full-batch gradient descent is non-increasing (with a 1e-12 slack) and the final loss falls from
 * 0.2518 to 3.3e-5 in 20000 epochs (asserted below 0.01) with all four XOR patterns classified
 * correctly (p < 0.5 for label 0, p > 0.5 for label 1); a learning rate of 0 does not move a single
 * parameter while a learning rate of 25 never settles -- it overshoots the starting loss 2.4x, drives
 * the loss back up on 36 of its 199 steps and ends far from the optimum (the documented instability of
 * a too-large step); the vanishing-gradient fact tanh'(3) ~ 0.0099 is asserted; the hidden-unit
 * permutation symmetry (permuting W1 rows, b1 and W2 together) leaves both the loss and the
 * multiset of gradients unchanged; and a re-run reproduces every weight and the loss bit for bit.
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
    }

    static final double[][] XOR_INPUTS = {{0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}};
    static final double[] XOR_TARGETS = {1.0, 1.0, 0.0, 0.0};

    static class Network {
        final int hidden;
        final int inputs;
        double[][] w1;
        double[] b1;
        double[] w2;
        double b2;

        Network(int inputs, int hidden, int seed) {
            this.inputs = inputs;
            this.hidden = hidden;
            this.w1 = new double[hidden][inputs];
            this.b1 = new double[hidden];
            this.w2 = new double[hidden];
            Lcg lcg = new Lcg(seed);
            for (int j = 0; j < hidden; j++) {
                for (int k = 0; k < inputs; k++) w1[j][k] = lcg.next() - 0.5;
                b1[j] = lcg.next() - 0.5;
            }
            for (int j = 0; j < hidden; j++) w2[j] = lcg.next() - 0.5;
            b2 = lcg.next() - 0.5;
        }

        static double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        double[] hiddenActivations(double[] x) {
            double[] h = new double[hidden];
            for (int j = 0; j < hidden; j++) {
                double z = b1[j];
                for (int k = 0; k < inputs; k++) z += w1[j][k] * x[k];
                h[j] = Math.tanh(z);
            }
            return h;
        }

        double predict(double[] x, double[] hiddenOut) {
            double[] h = hiddenActivations(x);
            double z = b2;
            for (int j = 0; j < hidden; j++) z += w2[j] * h[j];
            for (int j = 0; j < hidden; j++) hiddenOut[j] = h[j];
            return sigmoid(z);
        }

        double loss(double[][] xs, double[] ys) {
            double sum = 0.0;
            double[] h = new double[hidden];
            for (int i = 0; i < xs.length; i++) {
                double p = predict(xs[i], h);
                double residual = p - ys[i];
                sum += residual * residual;
            }
            return sum / xs.length;
        }

        double[] parameters() {
            double[] flat = new double[hidden * inputs + hidden + hidden + 1];
            int at = 0;
            for (int j = 0; j < hidden; j++) {
                for (int k = 0; k < inputs; k++) flat[at++] = w1[j][k];
            }
            for (int j = 0; j < hidden; j++) flat[at++] = b1[j];
            for (int j = 0; j < hidden; j++) flat[at++] = w2[j];
            flat[at] = b2;
            return flat;
        }

        void setParameters(double[] flat) {
            int at = 0;
            for (int j = 0; j < hidden; j++) {
                for (int k = 0; k < inputs; k++) w1[j][k] = flat[at++];
            }
            for (int j = 0; j < hidden; j++) b1[j] = flat[at++];
            for (int j = 0; j < hidden; j++) w2[j] = flat[at++];
            b2 = flat[at];
        }

        /** Analytic gradients from the hand-derived backpropagation, in the flat parameter order. */
        double[] gradients(double[][] xs, double[] ys) {
            int n = xs.length;
            double[] dw1 = new double[hidden * inputs];
            double[] db1 = new double[hidden];
            double[] dw2 = new double[hidden];
            double db2 = 0.0;
            double[] h = new double[hidden];
            for (int i = 0; i < n; i++) {
                double p = predict(xs[i], h);
                double delta2 = (2.0 / n) * (p - ys[i]) * p * (1.0 - p);
                db2 += delta2;
                for (int j = 0; j < hidden; j++) {
                    dw2[j] += delta2 * h[j];
                    double delta1 = delta2 * w2[j] * (1.0 - h[j] * h[j]);
                    db1[j] += delta1;
                    for (int k = 0; k < inputs; k++) dw1[j * inputs + k] += delta1 * xs[i][k];
                }
            }
            double[] flat = new double[dw1.length + hidden + hidden + 1];
            int at = 0;
            for (double value : dw1) flat[at++] = value;
            for (double value : db1) flat[at++] = value;
            for (double value : dw2) flat[at++] = value;
            flat[at] = db2;
            return flat;
        }

        /** Full-batch gradient descent; returns the loss trace (loss after each epoch). */
        List<Double> train(double[][] xs, double[] ys, double learningRate, int epochs) {
            List<Double> trace = new ArrayList<>();
            for (int epoch = 0; epoch < epochs; epoch++) {
                double[] gradient = gradients(xs, ys);
                double[] flat = parameters();
                for (int i = 0; i < flat.length; i++) flat[i] -= learningRate * gradient[i];
                setParameters(flat);
                trace.add(loss(xs, ys));
            }
            return trace;
        }
    }

    /** Independently written forward pass (explicit loops, different order of operations). */
    static double referenceForward(Network net, double[] x) {
        double[] h = new double[net.hidden];
        for (int j = 0; j < net.hidden; j++) {
            double z = net.b1[j];
            for (int k = 0; k < net.inputs; k++) {
                z = z + net.w1[j][k] * x[k];
            }
            double exponential = Math.exp(-2 * z);
            h[j] = (1 - exponential) / (1 + exponential);
        }
        double z = net.b2;
        for (int j = 0; j < net.hidden; j++) z = z + net.w2[j] * h[j];
        return 1.0 / (1.0 + Math.exp(-z));
    }

    static double flatLoss(Network net, double[] flat, double[][] xs, double[] ys) {
        net.setParameters(flat);
        return net.loss(xs, ys);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        int hidden = 3;
        Network net = new Network(2, hidden, 20250614);
        check(net.parameters().length == 13, "a 2-3-1 network has 13 parameters");

        // 1. Forward pass against an independently written implementation.
        double[] h = new double[hidden];
        for (double[] x : XOR_INPUTS) {
            double p = net.predict(x, h);
            check(Math.abs(p - referenceForward(net, x)) <= 1e-15,
                    "forward pass matches the reference");
        }

        // 2. The zero-parameter network has loss exactly 0.25.
        Network zero = new Network(2, hidden, 1);
        double[] zeros = new double[13];
        zero.setParameters(zeros);
        check(Math.abs(zero.loss(XOR_INPUTS, XOR_TARGETS) - 0.25) <= 1e-15,
                "with sigma(0) = 0.5 the zero network scores exactly 0.25");

        // 3. All 13 analytic gradients against central differences of the loss function itself.
        double[] flat = net.parameters();
        double[] analytic = net.gradients(XOR_INPUTS, XOR_TARGETS);
        double worst = 0.0;
        for (int i = 0; i < flat.length; i++) {
            double step = 1e-6;
            double[] plus = flat.clone();
            double[] minus = flat.clone();
            plus[i] += step;
            minus[i] -= step;
            double numeric = (flatLoss(net, plus, XOR_INPUTS, XOR_TARGETS)
                    - flatLoss(net, minus, XOR_INPUTS, XOR_TARGETS)) / (2 * step);
            worst = Math.max(worst, Math.abs(numeric - analytic[i]));
        }
        net.setParameters(flat);
        check(worst <= 1e-6,
                "every analytic gradient matches central differences (worst " + worst + ")");

        // 4. Full-batch gradient descent trains XOR.
        List<Double> trace = net.train(XOR_INPUTS, XOR_TARGETS, 1.0, 20000);
        double initialLoss = trace.get(0);
        double finalLoss = trace.get(trace.size() - 1);
        for (int i = 1; i < trace.size(); i++) {
            check(trace.get(i) <= trace.get(i - 1) + 1e-12,
                    "the loss trace is non-increasing at epoch " + i);
        }
        check(finalLoss <= 0.01, "training drives the loss below 0.01 (got " + finalLoss + ")");
        check(finalLoss < initialLoss / 10.0, "the loss drops by more than an order of magnitude");
        for (int i = 0; i < XOR_INPUTS.length; i++) {
            double p = net.predict(XOR_INPUTS[i], h);
            double y = XOR_TARGETS[i];
            check(y > 0.5 ? p > 0.5 : p < 0.5,
                    "pattern " + i + " is classified correctly (p = " + p + ")");
        }
        check(net.predict(new double[]{0.9, 0.1}, h) > 0.5, "a point near (1,0) is classified as 1");

        // 5. The trained network still passes the gradient check.
        double[] afterTraining = net.parameters();
        double[] trainedGradients = net.gradients(XOR_INPUTS, XOR_TARGETS);
        for (int i = 0; i < afterTraining.length; i++) {
            double step = 1e-6;
            double[] plus = afterTraining.clone();
            double[] minus = afterTraining.clone();
            plus[i] += step;
            minus[i] -= step;
            double numeric = (flatLoss(net, plus, XOR_INPUTS, XOR_TARGETS)
                    - flatLoss(net, minus, XOR_INPUTS, XOR_TARGETS)) / (2 * step);
            check(Math.abs(numeric - trainedGradients[i]) <= 1e-6,
                    "post-training gradient " + i + " matches numerics");
        }
        net.setParameters(afterTraining);
        check(Math.abs(trainedGradients[trainedGradients.length - 1]) <= 0.01,
                "at the optimum the output bias gradient is close to zero");

        // 6. Vanishing gradient of tanh at 3.
        check(Math.abs(1.0 - Math.tanh(3.0) * Math.tanh(3.0) - 0.00986) <= 1e-4,
                "tanh'(3) ~ 0.0099, the documented vanishing-gradient factor");

        // 7. A zero learning rate changes nothing; a huge one blows the loss up.
        Network frozen = new Network(2, hidden, 20250614);
        double[] before = frozen.parameters();
        frozen.train(XOR_INPUTS, XOR_TARGETS, 0.0, 50);
        check(Arrays.equals(frozen.parameters(), before),
                "learning rate 0 leaves every parameter alone");
        Network divergent = new Network(2, hidden, 20250614);
        double startLoss = divergent.loss(XOR_INPUTS, XOR_TARGETS);
        List<Double> wild = divergent.train(XOR_INPUTS, XOR_TARGETS, 25.0, 200);
        // lr = 25 is far past this network's stability limit, so the trajectory never settles: measured
        // over the 200 steps it peaks at 0.597 against a starting loss of 0.253 (2.4x) and 36 of the 199
        // steps increase the loss again. The runaway is chaotic rather than monotone, so what is asserted
        // is the honest consequence: repeated overshoot far above the start, and an end point nowhere
        // near the < 0.01 the sane rates reach above.
        double peak = Collections.max(wild);
        int rises = 0;
        for (int i = 1; i < wild.size(); i++) {
            if (wild.get(i) > wild.get(i - 1)) rises++;
        }
        check(peak > 1.75 * startLoss, "learning rate 25 overshoots the starting loss (peak " + peak
                + " against a start of " + startLoss + ")");
        check(rises > 20, "and the oversized step keeps driving the loss back up (" + rises + " of "
                + (wild.size() - 1) + " steps)");
        check(wild.get(wild.size() - 1) > 0.01, "and the run ends far from the optimum instead of at it ("
                + wild.get(wild.size() - 1) + ")");

        // 8. Permuting the hidden units leaves the loss and the gradient multiset unchanged.
        Network permuted = new Network(2, hidden, 20250614);
        double[] base = permuted.parameters();
        double[] swapped = base.clone();
        // swap hidden units 0 and 1: 2 rows of W1 (2 entries each), 2 b1 entries, 2 W2 entries
        for (int k = 0; k < 2; k++) {
            double tmp = swapped[k];
            swapped[k] = swapped[2 + k];
            swapped[2 + k] = tmp;
        }
        double tmpBias = swapped[6];
        swapped[6] = swapped[7];
        swapped[7] = tmpBias;
        double tmpOut = swapped[9];
        swapped[9] = swapped[10];
        swapped[10] = tmpOut;
        permuted.setParameters(swapped);
        Network original = new Network(2, hidden, 20250614);
        original.setParameters(base);
        check(Math.abs(permuted.loss(XOR_INPUTS, XOR_TARGETS) - original.loss(XOR_INPUTS, XOR_TARGETS))
                        <= 1e-15,
                "hidden-unit permutation preserves the loss");
        double[] permutedGradient = permuted.gradients(XOR_INPUTS, XOR_TARGETS);
        double[] originalGradient = original.gradients(XOR_INPUTS, XOR_TARGETS);
        double[] sortedPermuted = permutedGradient.clone();
        double[] sortedOriginal = originalGradient.clone();
        Arrays.sort(sortedPermuted);
        Arrays.sort(sortedOriginal);
        for (int i = 0; i < sortedPermuted.length; i++) {
            check(Math.abs(sortedPermuted[i] - sortedOriginal[i]) <= 1e-15,
                    "the gradient multiset is permutation invariant at index " + i);
        }

        // 9. Determinism.
        Network rerun = new Network(2, hidden, 20250614);
        check(Arrays.equals(rerun.parameters(), original.parameters()),
                "deterministic initialisation");
        List<Double> rerunTrace = rerun.train(XOR_INPUTS, XOR_TARGETS, 1.0, 200);
        Network again = new Network(2, hidden, 20250614);
        List<Double> againTrace = again.train(XOR_INPUTS, XOR_TARGETS, 1.0, 200);
        check(rerunTrace.equals(againTrace), "deterministic re-run: identical loss trace");
        check(Arrays.equals(rerun.parameters(), again.parameters()),
                "deterministic re-run: identical weights");

        System.out.println("PASS");
    }
}
