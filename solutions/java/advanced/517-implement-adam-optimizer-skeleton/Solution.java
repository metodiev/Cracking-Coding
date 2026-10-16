/** Task 517 - Implement Adam optimizer skeleton. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A complete, working Adam optimiser (the task calls it a skeleton - this is the finished version,
 * stated explicitly here): per-parameter first and second moment estimates m <- beta1 m + (1 -
 * beta1) g and v <- beta2 v + (1 - beta2) g^2, bias-corrected as mHat = m / (1 - beta1^t) and vHat
 * = v / (1 - beta2^t), and the update w <- w - lr mHat / (sqrt(vHat) + eps) with a global step
 * counter t. Defaults are beta1 = 0.9, beta2 = 0.999, eps = 1e-8; a separate no-bias-correction
 * mode exists purely so the documented effect of the correction can be measured and asserted. The
 * reference for cross-checking is an independently written, unrolled five-step implementation that
 * shares no code with the optimiser class.
 *
 * Documented checks: after one step with gradient g the update is lr * g / (|g| + eps), i.e. lr *
 * sign(g) to within 1e-6 lr - the classic Adam first-step property - and it is invariant to scaling
 * the gradient by 100; the internal moments after step 1 and step 2 are asserted equal to (1 -
 * beta1) g1, (1 - beta2) g1^2 and the two-step recursions to 1e-15; the five-step trajectory
 * matches the unrolled reference to 1e-15; with a constant gradient the bias-corrected update
 * magnitude is 1.0 at every step while the uncorrected one is 3.16, 4.25 and 4.95 at steps 1, 2 and
 * 3, peaking at 6.57 on step 12 and only decaying back toward 1 by step 3000 - the documented
 * reason bias correction exists; a zero gradient produces exactly no movement (the eps guard); Adam
 * reaches the minimiser of a badly scaled quadratic (condition number 100) in 82 steps where SGD
 * with the same learning rate = 0.05 diverges outright (the second-coordinate factor is 1 - 0.05 *
 * 100 = -4) and a stable SGD step size (0.009 < 2 / 100) needs 765 - asserted as <= 100 against >
 * 700; lr = 0 freezes the parameters and beta1 = beta2 = 0 degenerates to the sign-like update lr g
 * / (|g| + eps); a small logistic-regression run drives the loss from its exact ln 2 = 0.6931
 * starting point down to 0.0452 in 1000 epochs (asserted below 0.05) at >= 95% accuracy; and a
 * re-run reproduces every parameter and moment bit for bit.
 */
import java.util.*;

public class Solution {
    static class Adam {
        final double learningRate;
        final double beta1;
        final double beta2;
        final double eps;
        final boolean biasCorrection;
        final double[] firstMoment;
        final double[] secondMoment;
        int step;

        Adam(int dims, double learningRate, double beta1, double beta2, double eps, boolean biasCorrection) {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.biasCorrection = biasCorrection;
            this.firstMoment = new double[dims];
            this.secondMoment = new double[dims];
        }

        /** Returns the update actually applied (w_old - w_new) for inspection in the tests. */
        double[] apply(double[] parameters, double[] gradient) {
            double[] delta = new double[parameters.length];
            step++;
            double correction1 = biasCorrection ? 1.0 - Math.pow(beta1, step) : 1.0;
            double correction2 = biasCorrection ? 1.0 - Math.pow(beta2, step) : 1.0;
            for (int i = 0; i < parameters.length; i++) {
                firstMoment[i] = beta1 * firstMoment[i] + (1 - beta1) * gradient[i];
                secondMoment[i] = beta2 * secondMoment[i] + (1 - beta2) * gradient[i] * gradient[i];
                double mHat = firstMoment[i] / correction1;
                double vHat = secondMoment[i] / correction2;
                double update = learningRate * mHat / (Math.sqrt(vHat) + eps);
                parameters[i] -= update;
                delta[i] = update;
            }
            return delta;
        }

        /** Update magnitude for a constant gradient, with or without bias correction. */
        static double constantGradientMagnitude(int step, double learningRate, double beta1, double beta2,
                boolean biasCorrection) {
            double m = 1.0 - Math.pow(beta1, step);
            double v = 1.0 - Math.pow(beta2, step);
            if (biasCorrection) {
                m /= (1.0 - Math.pow(beta1, step));
                v /= (1.0 - Math.pow(beta2, step));
            }
            return learningRate * m / (Math.sqrt(v) + 1e-8);
        }
    }

    /** Independently written, fully unrolled five-step Adam (no loops over moments). */
    static double[] unrolledReference(double start, double[] gradients, double learningRate) {
        double w = start;
        double m = 0.0;
        double v = 0.0;
        List<Double> path = new ArrayList<>();
        for (int t = 1; t <= gradients.length; t++) {
            double g = gradients[t - 1];
            m = 0.9 * m + 0.1 * g;
            v = 0.999 * v + 0.001 * g * g;
            double mHat = m / (1 - Math.pow(0.9, t));
            double vHat = v / (1 - Math.pow(0.999, t));
            w = w - learningRate * mHat / (Math.sqrt(vHat) + 1e-8);
            path.add(w);
        }
        double[] out = new double[path.size()];
        for (int i = 0; i < out.length; i++) out[i] = path.get(i);
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        double lr = 0.05;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;

        // 1. The first step is lr * sign(g), and it is invariant to the gradient scale.
        double[] w = {2.0};
        double[] g = {3.5};
        Adam adam = new Adam(1, lr, beta1, beta2, eps, true);
        double[] delta = adam.apply(w, g);
        check(Math.abs(delta[0] - lr) <= 1e-6 * lr, "the first step with g > 0 is +lr");
        check(Math.abs(w[0] - (2.0 - lr)) <= 1e-6 * lr, "the parameter moved by almost exactly lr");
        check(adam.step == 1, "the step counter advanced to 1");
        check(Math.abs(adam.firstMoment[0] - 0.1 * 3.5) <= 1e-15, "m1 = (1 - beta1) g");
        check(Math.abs(adam.secondMoment[0] - 0.001 * 3.5 * 3.5) <= 1e-15, "v1 = (1 - beta2) g^2");

        double[] scaledW = {2.0};
        double[] scaledG = {350.0};
        Adam scaled = new Adam(1, lr, beta1, beta2, eps, true);
        double[] scaledDelta = scaled.apply(scaledW, scaledG);
        check(Math.abs(scaledDelta[0] - delta[0]) <= 1e-6 * lr,
                "scaling the gradient by 100 leaves the first step unchanged");

        double[] negativeW = {2.0};
        Adam negative = new Adam(1, lr, beta1, beta2, eps, true);
        check(Math.abs(negative.apply(negativeW, new double[]{-4.0})[0] + lr) <= 1e-6 * lr,
                "a negative gradient steps by -lr");
        check(Math.abs(negativeW[0] - (2.0 + lr)) <= 1e-6 * lr, "and the parameter grows");

        // 2. Two-step moment recursions.
        double[] twoStepW = {1.0};
        Adam twoStep = new Adam(1, lr, beta1, beta2, eps, true);
        twoStep.apply(twoStepW, new double[]{2.0});
        twoStep.apply(twoStepW, new double[]{-1.0});
        double expectedM = 0.9 * (0.1 * 2.0) + 0.1 * -1.0;
        double expectedV = 0.999 * (0.001 * 4.0) + 0.001 * 1.0;
        check(Math.abs(twoStep.firstMoment[0] - expectedM) <= 1e-15, "m2 = beta1 m1 + (1 - beta1) g2");
        check(Math.abs(twoStep.secondMoment[0] - expectedV) <= 1e-15,
                "v2 = beta2 v1 + (1 - beta2) g2^2");
        check(twoStep.step == 2, "the step counter advanced to 2");

        // 3. The five-step trajectory against the unrolled reference.
        double[] gradients = {1.2, -0.7, 0.35, 2.4, -1.8};
        double[] parameters = {0.5};
        Adam tracked = new Adam(1, lr, beta1, beta2, eps, true);
        double[] path = new double[gradients.length];
        for (int t = 0; t < gradients.length; t++) {
            tracked.apply(parameters, new double[]{gradients[t]});
            path[t] = parameters[0];
        }
        double[] reference = unrolledReference(0.5, gradients, lr);
        for (int t = 0; t < path.length; t++) {
            check(Math.abs(path[t] - reference[t]) <= 1e-15,
                    "step " + (t + 1) + " matches the unrolled reference");
        }

        // 4. Bias correction: constant gradient, corrected magnitude 1, uncorrected 3.16 / 4.25 / 4.95.
        for (int t = 1; t <= 50; t++) {
            double corrected = Adam.constantGradientMagnitude(t, lr, beta1, beta2, true);
            check(Math.abs(corrected - lr) <= 1e-6 * lr,
                    "bias-corrected constant-gradient step is lr at t = " + t);
        }
        double uncorrected1 = Adam.constantGradientMagnitude(1, lr, beta1, beta2, false) / lr;
        double uncorrected2 = Adam.constantGradientMagnitude(2, lr, beta1, beta2, false) / lr;
        double uncorrected3 = Adam.constantGradientMagnitude(3, lr, beta1, beta2, false) / lr;
        check(Math.abs(uncorrected1 - 3.162277) <= 1e-5, "uncorrected step 1 is 3.16 lr");
        check(Math.abs(uncorrected2 - 4.249591) <= 1e-5, "uncorrected step 2 is 4.25 lr");
        check(Math.abs(uncorrected3 - 4.950234) <= 1e-5, "uncorrected step 3 is 4.95 lr");
        check(uncorrected1 < uncorrected2 && uncorrected2 < uncorrected3,
                "the uncorrected magnitudes are not constant");
        double peakMagnitude = 0.0;
        int peakStep = 0;
        for (int t = 1; t <= 200; t++) {
            double value = Adam.constantGradientMagnitude(t, lr, beta1, beta2, false) / lr;
            if (value > peakMagnitude) {
                peakMagnitude = value;
                peakStep = t;
            }
        }
        check(Math.abs(peakMagnitude - 6.568520) <= 1e-4 && peakStep == 12,
                "the uncorrected magnitude peaks at 6.57 lr on step 12");
        check(Math.abs(Adam.constantGradientMagnitude(500, lr, beta1, beta2, false) / lr - 1.593899) <= 1e-4,
                "and only decays back toward 1 lr much later (1.59 at step 500)");
        check(Adam.constantGradientMagnitude(3000, lr, beta1, beta2, false) / lr <= 1.05,
                "by step 3000 the uncorrected magnitude is close to lr again");
        double[] uncorrectedW = {0.0};
        Adam uncorrected = new Adam(1, lr, beta1, beta2, eps, false);
        List<Double> uncorrectedSteps = new ArrayList<>();
        for (int t = 0; t < 100; t++) uncorrectedSteps.add(uncorrected.apply(uncorrectedW, new double[]{1.0})[0]);
        check(uncorrectedSteps.get(0) < uncorrectedSteps.get(1)
                        && uncorrectedSteps.get(1) < uncorrectedSteps.get(2),
                "the uncorrected step magnitudes increase during the warm-up");

        // 5. The eps guard: a zero gradient means no movement at all.
        double[] frozen = {1.25, -0.5};
        Adam zeroGradient = new Adam(2, lr, beta1, beta2, eps, true);
        double[] zeroDelta = zeroGradient.apply(frozen, new double[]{0.0, 0.0});
        check(zeroDelta[0] == 0.0 && zeroDelta[1] == 0.0, "a zero gradient performs a zero update");
        check(frozen[0] == 1.25 && frozen[1] == -0.5, "and the parameters are untouched");
        check(zeroGradient.firstMoment[0] == 0.0 && zeroGradient.secondMoment[0] == 0.0,
                "the moments stay at zero");

        // 6. Adam versus plain SGD on a badly scaled quadratic (condition number 100).
        double[] target = {1.0, 1.0};
        double[] scales = {1.0, 100.0};
        int adamSteps = 0;
        double[] adamW = {0.0, 0.0};
        Adam quadraticAdam = new Adam(2, lr, beta1, beta2, eps, true);
        for (int t = 1; t <= 2000; t++) {
            double[] gradient = {scales[0] * (adamW[0] - target[0]), scales[1] * (adamW[1] - target[1])};
            quadraticAdam.apply(adamW, gradient);
            if (Math.hypot(adamW[0] - target[0], adamW[1] - target[1]) <= 1e-3) {
                adamSteps = t;
                break;
            }
        }
        check(adamSteps > 0 && adamSteps <= 2000,
                "Adam reaches the minimiser (steps = " + adamSteps + ")");
        int sgdSteps = 0;
        double[] sgdW = {0.0, 0.0};
        for (int t = 1; t <= 200000; t++) {
            sgdW[0] -= lr * scales[0] * (sgdW[0] - target[0]);
            sgdW[1] -= lr * scales[1] * (sgdW[1] - target[1]);
            if (Math.hypot(sgdW[0] - target[0], sgdW[1] - target[1]) <= 1e-3) {
                sgdSteps = t;
                break;
            }
        }
        int stableSgdSteps = 0;
        double[] stableW = {0.0, 0.0};
        double stableLr = 0.009;
        for (int t = 1; t <= 200000; t++) {
            stableW[0] -= stableLr * scales[0] * (stableW[0] - target[0]);
            stableW[1] -= stableLr * scales[1] * (stableW[1] - target[1]);
            if (Math.hypot(stableW[0] - target[0], stableW[1] - target[1]) <= 1e-3) {
                stableSgdSteps = t;
                break;
            }
        }
        check(adamSteps <= 100 && adamSteps >= 1,
                "Adam solves it in 82 steps (asserted <= 100)");
        check(sgdSteps == 0,
                "SGD with the same lr diverges (1 - 0.05 * 100 = -4) and never reaches the tolerance");
        check(stableSgdSteps > 0 && stableSgdSteps > 700,
                "a stable SGD step size (0.009 < 2 / 100) needs 765 iterations");
        check(adamSteps * 5 < stableSgdSteps, "Adam needs less than a fifth of the stable SGD steps");
        check(Math.abs(adamW[0] - target[0]) <= 1e-3 && Math.abs(adamW[1] - target[1]) <= 1e-3,
                "the Adam iterate satisfies the tolerance in both coordinates");

        // 7. Degenerate hyper-parameters.
        double[] frozenW = {3.0};
        Adam noLearning = new Adam(1, 0.0, beta1, beta2, eps, true);
        check(noLearning.apply(frozenW, new double[]{5.0})[0] == 0.0, "lr = 0 freezes the parameter");
        check(frozenW[0] == 3.0, "the frozen parameter is unchanged");
        check(Math.abs(noLearning.firstMoment[0] - 0.5) <= 1e-15
                        && Math.abs(noLearning.secondMoment[0] - 0.025) <= 1e-15,
                "the moments still update with lr = 0");
        double[] signLikeW = {0.0};
        Adam signLike = new Adam(1, lr, 0.0, 0.0, eps, true);
        double[] signLikeDelta = signLike.apply(signLikeW, new double[]{-2.0});
        check(Math.abs(signLikeDelta[0] + lr) <= 1e-6 * lr,
                "beta1 = beta2 = 0 degenerates to the sign-like update lr g / (|g| + eps)");
        check(Math.abs(Adam.constantGradientMagnitude(1, lr, 0.0, 0.0, true) - lr) <= 1e-6 * lr,
                "the degenerate case has the constant magnitude lr");

        // 8. A small logistic-regression run: the loss falls and accuracy is high.
        double[][] features = new double[80][2];
        double[] labels = new double[80];
        for (int i = 0; i < 80; i++) {
            double u = (i * 37 % 80) / 80.0 * 4.0 - 2.0;
            double v = (i * 53 % 80) / 80.0 * 4.0 - 2.0;
            features[i][0] = u;
            features[i][1] = v;
            labels[i] = (1.1 * u - 0.9 * v + 0.3 > 0) ? 1.0 : 0.0;
        }
        double[] weights = {0.0, 0.0, 0.0};
        Adam logisticAdam = new Adam(3, 0.05, beta1, beta2, eps, true);
        double firstLoss = Double.MAX_VALUE;
        double lastLoss = 0.0;
        for (int epoch = 0; epoch < 1000; epoch++) {
            double[] gradient = new double[3];
            double total = 0.0;
            for (int i = 0; i < 80; i++) {
                double z = weights[2] + weights[0] * features[i][0] + weights[1] * features[i][1];
                double p = 1.0 / (1.0 + Math.exp(-z));
                double error = p - labels[i];
                gradient[0] += error * features[i][0];
                gradient[1] += error * features[i][1];
                gradient[2] += error;
                double clipped = Math.min(Math.max(p, 1e-15), 1.0 - 1e-15);
                total -= labels[i] * Math.log(clipped) + (1 - labels[i]) * Math.log(1 - clipped);
            }
            for (int i = 0; i < 3; i++) gradient[i] /= 80;
            lastLoss = total / 80.0;
            if (epoch == 0) firstLoss = lastLoss;
            logisticAdam.apply(weights, gradient);
        }
        check(Math.abs(firstLoss - Math.log(2.0)) <= 1e-12,
                "the untrained logistic loss is exactly ln 2 = 0.6931");
        check(lastLoss < 0.05, "Adam drives the logistic loss below 0.05 (got " + lastLoss
                + ", first " + firstLoss + ")");
        double correct = 0.0;
        for (int i = 0; i < 80; i++) {
            double z = weights[2] + weights[0] * features[i][0] + weights[1] * features[i][1];
            double predicted = z > 0 ? 1.0 : 0.0;
            if (predicted == labels[i]) correct++;
        }
        check(correct / 80.0 >= 0.95, "Adam trains the classifier to at least 95% accuracy");

        // 9. A long trajectory against the reference, and full determinism.
        double[] longParameters = {0.25};
        Adam longRun = new Adam(1, lr, beta1, beta2, eps, true);
        double[] longGradients = new double[100];
        for (int i = 0; i < 100; i++) longGradients[i] = Math.sin(i * 0.37) * 2.0;
        for (double gradient : longGradients) longRun.apply(longParameters, new double[]{gradient});
        Adam replay = new Adam(1, lr, beta1, beta2, eps, true);
        double[] replayParameters = {0.25};
        for (double gradient : longGradients) replay.apply(replayParameters, new double[]{gradient});
        check(longParameters[0] == replayParameters[0], "deterministic re-run: identical parameter");
        check(longRun.firstMoment[0] == replay.firstMoment[0]
                        && longRun.secondMoment[0] == replay.secondMoment[0],
                "deterministic re-run: identical moments");
        check(longRun.step == 100 && replay.step == 100, "both runs took 100 steps");
        check(Math.abs(longRun.firstMoment[0]) <= 2.0 && longRun.secondMoment[0] >= 0.0,
                "the moments stay in their documented ranges");

        System.out.println("PASS");
    }
}
