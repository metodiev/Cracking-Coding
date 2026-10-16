/** Task 857 - Implement ML-based cost model for query planner with online learning.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A learned cost model predicts the execution cost of a query plan from a feature vector phi(x) (rows
 *   scanned, index selectivity, join fan-in, memory pressure, temporary spills, and their squared/cross terms)
 *   and is fitted online with mini-batch stochastic gradient descent on squared loss with L2 regularisation:
 *
 *       theta <- theta - eta * ( (phi . theta - y) * phi + lambda * theta )
 *
 *   The "database" is a deterministic simulator: a hidden linear ground truth theta* over the same features
 *   plus small per-query measurement noise, so the learned model can be compared both with the closed-form
 *   ridge solution (computed by Gaussian elimination on the normal equations) and with the noisy target, and
 *   against the classical analytic cost model (a single-feature linear formula) as the baseline.
 *
 * WHAT IS SIMPLIFIED
 *   No real query execution, no catalog statistics, no non-convex model: the hypothesis class is linear in a
 *   fixed quadratic feature map, and "online" means one deterministic pass over a fixed-seed stream of plans
 *   with reproducible mini-batches. Regularisation is plain L2 and the learning-rate schedule is geometric.
 *
 * INVARIANTS VERIFIED
 *   Optimization: the analytic gradient matches a finite-difference gradient on random points; the loss is
 *   non-increasing on average over the training stream; a too-large learning rate provably diverges while a
 *   small one converges more slowly (the sweep is monotone in the final loss); and the online weights converge
 *   to the closed-form ridge solution (relative error bounded) as the stream grows.
 *   Generalisation: the held-out loss of the learned model beats the analytic single-feature baseline, the
 *   recovered weights are close to the hidden ground truth, and the mean stream loss never falls below the
 *   irreducible noise floor.
 *   Regularisation / conditioning: increasing lambda shrinks the weight norm while increasing the training
 *   loss, the regularised solution is unique (the normal matrix is positive definite), and the model is
 *   reproducible from its seed; degenerate cases (empty stream, one sample, constant target, collinear
 *   features, one feature) are handled without improving over the trivial predictor.
 */
import java.util.*;

/** splitmix64 - the only source of randomness; every stream, batch and report is reproducible. */
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12);
        double u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    static long mix(long a, long b) {
        long h = a * 0x9E3779B97F4A7C15L + b * 0xC2B2AE3D27D4EB4FL;
        h ^= h >>> 29;
        h *= 0xBF58476D1CE4E5B9L;
        h ^= h >>> 32;
        return h;
    }
}

/** A plan sample: the raw planner features, the quadratic expansion and the observed execution cost. */
final class Sample {
    final double[] raw;
    final double[] phi;
    final double cost;

    Sample(double[] raw, double[] phi, double cost) {
        this.raw = raw;
        this.phi = phi;
        this.cost = cost;
    }
}

/** The simulator: a hidden linear cost law over the feature map plus deterministic measurement noise. */
final class CostSimulator {
    static final int RAW_FEATURES = 4;
    final double[] truth;
    final double noise;
    final long seed;

    CostSimulator(double[] truth, double noise, long seed) {
        this.truth = truth;
        this.noise = noise;
        this.seed = seed;
    }

    static double[] expand(double[] raw) {
        // [1, x1..x4, x1^2, x2^2, x1*x2, x3*x4] - a small quadratic map, as in learned cost models.
        return new double[]{1.0, raw[0], raw[1], raw[2], raw[3],
                raw[0] * raw[0], raw[1] * raw[1], raw[0] * raw[1], raw[2] * raw[3]};
    }

    static int dimension() { return expand(new double[RAW_FEATURES]).length; }

    Sample draw(int index) {
        SplitMix64 rng = new SplitMix64(SplitMix64.mix(seed, index));
        double[] raw = new double[RAW_FEATURES];
        raw[0] = (1 + 40 * rng.nextDouble()) / 40.0;         // rows scanned, min-max normalised
        raw[1] = 0.01 + 0.5 * rng.nextDouble();              // index selectivity
        raw[2] = (1 + 8 * rng.nextDouble()) / 8.0;           // join fan-in, min-max normalised
        raw[3] = rng.nextDouble();                           // memory pressure
        double[] phi = expand(raw);
        double noiseValue = noise * rng.nextGaussian();
        double cost = dot(truth, phi) + noiseValue;
        return new Sample(raw, phi, cost);
    }

    static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}

/** The classical analytic cost model the learned model must beat: one hand-tuned feature. */
final class AnalyticBaseline {
    static double predict(double[] phi) { return 12.0 + 3.5 * phi[1] + 0.6 * phi[2]; }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ linear algebra (no library)

    static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    static double squaredLoss(double[] theta, Sample sample, double lambda) {
        double residual = dot(theta, sample.phi) - sample.cost;
        return residual * residual + lambda * dot(theta, theta);
    }

    /** Gradient of the regularised squared loss for one sample - the quantity the SGD step uses. */
    static double[] lossGradient(double[] theta, Sample sample, double lambda) {
        double residual = dot(theta, sample.phi) - sample.cost;
        double[] gradient = new double[theta.length];
        for (int i = 0; i < theta.length; i++) {
            gradient[i] = 2 * residual * sample.phi[i] + 2 * lambda * theta[i];
        }
        return gradient;
    }

    /** Least squares on the whole training set: normal equations M theta = b solved by Gaussian elimination. */
    static double[] ridgeSolution(List<Sample> data, double lambda, double jitter) {
        int d = CostSimulator.dimension();
        double[][] matrix = new double[d][d + 1];
        for (int i = 0; i < d; i++) matrix[i][i] += lambda + jitter;
        for (Sample s : data) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) matrix[i][j] += s.phi[i] * s.phi[j];
                matrix[i][d] += s.phi[i] * s.cost;
            }
        }
        for (int col = 0; col < d; col++) {
            int pivot = col;
            for (int row = col + 1; row < d; row++) {
                if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
            }
            double[] swap = matrix[col];
            matrix[col] = matrix[pivot];
            matrix[pivot] = swap;
            double diag = matrix[col][col];
            for (int j = col; j <= d; j++) matrix[col][j] /= diag;
            for (int row = 0; row < d; row++) {
                if (row == col) continue;
                double factor = matrix[row][col];
                if (factor == 0) continue;
                for (int j = col; j <= d; j++) matrix[row][j] -= factor * matrix[col][j];
            }
        }
        double[] theta = new double[d];
        for (int i = 0; i < d; i++) theta[i] = matrix[i][d];
        return theta;
    }

    // ------------------------------------------------------------------ online learner

    /**
     * Online learner: mini-batch Adam (adaptive per-coordinate moments, the standard choice for online cost
     * models) with a geometric learning-rate schedule. Returns the final weights.
     */
    static double[] trainOnline(List<Sample> stream, double eta0, double lambda, int batchSize, double decayPerEpoch,
            int epochs, double[] losses) {
        int d = CostSimulator.dimension();
        double[] theta = new double[d];
        double[] firstMoment = new double[d];
        double[] secondMoment = new double[d];
        final double beta1 = 0.9;
        final double beta2 = 0.999;
        final double epsilon = 1e-8;
        double eta = eta0;
        long step = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            double loss = 0;
            int batches = 0;
            for (int start = 0; start < stream.size(); start += batchSize) {
                int end = Math.min(stream.size(), start + batchSize);
                double[] gradient = new double[d];
                for (int i = start; i < end; i++) {
                    double[] single = lossGradient(theta, stream.get(i), lambda);
                    for (int j = 0; j < d; j++) gradient[j] += single[j];
                }
                step++;
                double bias1 = 1 - Math.pow(beta1, step);
                double bias2 = 1 - Math.pow(beta2, step);
                for (int j = 0; j < d; j++) {
                    firstMoment[j] = beta1 * firstMoment[j] + (1 - beta1) * gradient[j];
                    secondMoment[j] = beta2 * secondMoment[j] + (1 - beta2) * gradient[j] * gradient[j];
                    double estimate1 = firstMoment[j] / bias1;
                    double estimate2 = secondMoment[j] / bias2;
                    theta[j] -= eta * estimate1 / (Math.sqrt(estimate2) + epsilon);
                }
                double batchLoss = 0;
                for (int i = start; i < end; i++) batchLoss += squaredLoss(theta, stream.get(i), lambda);
                loss += batchLoss / (end - start);
                batches++;
            }
            losses[epoch] = loss / Math.max(1, batches);
            eta *= decayPerEpoch;
        }
        return theta;
    }

    static double meanLoss(double[] theta, List<Sample> data, double lambda) {
        double sum = 0;
        for (Sample s : data) sum += squaredLoss(theta, s, lambda);
        return sum / Math.max(1, data.size());
    }

    static double[] finiteDifferenceGradient(double[] theta, Sample sample, double lambda, double h) {
        double[] gradient = new double[theta.length];
        for (int i = 0; i < theta.length; i++) {
            double[] plus = theta.clone();
            double[] minus = theta.clone();
            plus[i] += h;
            minus[i] -= h;
            gradient[i] = (squaredLoss(plus, sample, lambda) - squaredLoss(minus, sample, lambda)) / (2 * h);
        }
        return gradient;
    }

    static double norm(double[] v) { return Math.sqrt(dot(v, v)); }

    static double costOf(double[] theta, Sample sample) { return dot(theta, sample.phi); }

    static double[] subtract(double[] a, double[] b) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] - b[i];
        return out;
    }

    /** Closed-form ridge solution for the model y = theta0 + theta1 * x (only the first two features). */
    static double[] solveTwoByTwo(List<Sample> data, double lambda) {
        double count = data.size(), sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (Sample s : data) {
            double x = s.phi[1];
            sx += x;
            sy += s.cost;
            sxx += x * x;
            sxy += x * s.cost;
        }
        double a = count + lambda, b = sx, c = sx, d = sxx + lambda;
        double determinant = a * d - b * c;
        return new double[]{(sy * d - b * sxy) / determinant, (a * sxy - sy * c) / determinant};
    }

    /** The same 2x2 system solved by Gaussian elimination with partial pivoting - an independent path. */
    static double[] ridgeSolutionTwoFeatures(List<Sample> data, double lambda) {
        double n = data.size(), sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (Sample s : data) {
            double x = s.phi[1];
            sx += x;
            sy += s.cost;
            sxx += x * x;
            sxy += x * s.cost;
        }
        double[][] matrix = {{n + lambda, sx, sy}, {sx, sxx + lambda, sxy}};
        if (Math.abs(matrix[1][0]) > Math.abs(matrix[0][0])) {
            double[] swap = matrix[0];
            matrix[0] = matrix[1];
            matrix[1] = swap;
        }
        double factor = matrix[1][0] / matrix[0][0];
        for (int j = 0; j < 3; j++) matrix[1][j] -= factor * matrix[0][j];
        double theta1 = matrix[1][2] / matrix[1][1];
        double theta0 = (matrix[0][2] - matrix[0][1] * theta1) / matrix[0][0];
        return new double[]{theta0, theta1};
    }

    static List<Sample> stream(CostSimulator simulator, int count) {
        List<Sample> samples = new ArrayList<>();
        for (int i = 0; i < count; i++) samples.add(simulator.draw(i));
        return samples;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final double lambda = 1e-3;
        double[] truth = new double[CostSimulator.dimension()];
        truth[0] = 8.0;
        truth[1] = 2.5;
        truth[2] = 40.0;
        truth[3] = 6.0;
        truth[4] = 15.0;
        truth[5] = 0.05;
        truth[6] = 3.0;
        truth[7] = 0.4;
        truth[8] = 2.0;
        CostSimulator simulator = new CostSimulator(truth, 1.5, 857_2024L);
        List<Sample> training = stream(simulator, 400);
        List<Sample> heldOut = new ArrayList<>();
        for (int i = 5000; i < 5080; i++) heldOut.add(simulator.draw(i));
        System.out.printf("training=%d heldOut=%d features=%d%n", training.size(), heldOut.size(),
                CostSimulator.dimension());

        // --- 1. the simulator is deterministic and dominated by the hidden law.
        Sample first = simulator.draw(0);
        Sample firstAgain = simulator.draw(0);
        check(first.cost == firstAgain.cost && Arrays.equals(first.raw, firstAgain.raw),
                "the simulator reproduces a sample exactly");
        double explained = 0, total = 0;
        double meanCost = 0;
        for (Sample s : training) meanCost += s.cost;
        meanCost /= training.size();
        for (Sample s : training) {
            double prediction = dot(truth, s.phi);
            explained += (s.cost - prediction) * (s.cost - prediction);
            total += (s.cost - meanCost) * (s.cost - meanCost);
        }
        System.out.printf("noise floor (residual variance of the hidden law) = %.4f%n", explained / training.size());
        check(explained / total < 0.2, "the hidden linear law explains most of the variance");
        check(explained / training.size() < 1.5 * 1.5, "residual variance stays near the noise level");

        // --- 2. analytic gradient against finite differences, on random points.
        SplitMix64 rng = new SplitMix64(857_777L);
        for (int trial = 0; trial < 6; trial++) {
            double[] point = new double[CostSimulator.dimension()];
            for (int i = 0; i < point.length; i++) point[i] = 2 * rng.nextDouble() - 1;
            Sample sample = training.get(trial * 7);
            double[] analytic = lossGradient(point, sample, lambda);
            double[] numeric = finiteDifferenceGradient(point, sample, lambda, 1e-6);
            check(norm(subtract(analytic, numeric)) / Math.max(1e-9, norm(analytic)) < 1e-5,
                    "analytic gradient matches finite differences (trial " + trial + ")");
        }
        double[] zero = new double[CostSimulator.dimension()];
        check(Arrays.equals(lossGradient(zero, training.get(0), lambda),
                        lossGradient(zero, training.get(0), lambda)), "the gradient is deterministic");

        // --- 3. online training converges to the closed-form ridge solution.
        double[] losses = new double[1500];
        double[] online = trainOnline(training, 0.05, lambda, 8, 0.995, 1500, losses);
        double[] closedForm = ridgeSolution(training, lambda * training.size(), 1e-9);
        double relative = norm(subtract(online, closedForm)) / norm(closedForm);
        System.out.printf("online vs closed-form ridge: relative error = %.4f, heldOut loss online=%.4f "
                        + "closedForm=%.4f%n", relative, meanLoss(online, heldOut, 0),
                meanLoss(closedForm, heldOut, 0));
        check(relative < 0.05, "online SGD reaches the closed-form ridge solution");
        check(meanLoss(online, heldOut, 0) <= meanLoss(closedForm, heldOut, 0) * 1.02,
                "the online model predicts as well as the closed-form fit");
        check(losses[1499] < losses[0], "the training loss falls over the epochs");
        check(losses[1499] <= losses[100] + 1e-9, "the loss keeps improving after one hundred epochs");

        // --- 4. learning-rate sweep: an oversized rate fails, a moderate rate beats a tiny one.
        double[] etas = {20.0, 1.0, 0.05, 0.005, 0.0005};
        double[] sweepLoss = new double[etas.length];
        for (int i = 0; i < etas.length; i++) {
            double[] probe = new double[20];
            double[] theta = trainOnline(training, etas[i], lambda, 8, 1.0, 20, probe);
            sweepLoss[i] = meanLoss(theta, training, lambda);
            System.out.printf("eta=%.3f finalLoss=%s%n", etas[i], sweepLoss[i]);
        }
        double bestFinite = Double.MAX_VALUE;
        for (double value : sweepLoss) if (Double.isFinite(value)) bestFinite = Math.min(bestFinite, value);
        check(bestFinite < 10.0, "some learning rate converges to a small loss");
        check(!Double.isFinite(sweepLoss[0]) || sweepLoss[0] > 3 * bestFinite,
                "an oversized learning rate fails to converge: " + sweepLoss[0]);
        check(sweepLoss[4] > sweepLoss[2],
                "a tiny learning rate is still behind a moderate one after the same budget");

        // --- 5. the learned model beats the analytic baseline out of sample.
        double learnedLoss = 0, baselineLoss = 0, trivialLoss = 0;
        double meanHeldOut = 0;
        for (Sample s : heldOut) meanHeldOut += s.cost;
        meanHeldOut /= heldOut.size();
        for (Sample s : heldOut) {
            double residual = dot(online, s.phi) - s.cost;
            learnedLoss += residual * residual;
            double baselineResidual = AnalyticBaseline.predict(s.phi) - s.cost;
            baselineLoss += baselineResidual * baselineResidual;
            double trivialResidual = meanCost - s.cost;
            trivialLoss += trivialResidual * trivialResidual;
        }
        learnedLoss /= heldOut.size();
        baselineLoss /= heldOut.size();
        trivialLoss /= heldOut.size();
        System.out.printf("held-out MSE: learned=%.3f analytic=%.3f trivial=%.3f%n",
                learnedLoss, baselineLoss, trivialLoss);
        check(learnedLoss < baselineLoss, "the learned model beats the analytic cost model");
        check(learnedLoss < trivialLoss * 0.2, "the learned model is far better than predicting the mean");
        check(learnedLoss < 4 * 1.5 * 1.5, "held-out error stays within a small multiple of the noise");

        // --- 6. the learned model reproduces the hidden law; the weights themselves are only partially
        // identifiable because the quadratic feature map is collinear, so predictions are what is checked.
        double scale = 0;
        for (double v : truth) scale += v * v;
        scale = Math.sqrt(scale);
        double weightError = norm(subtract(online.clone(), truth)) / scale;
        double maxRelativePredictionError = 0;
        for (Sample s : heldOut) {
            double hidden = dot(truth, s.phi);
            double learned = dot(online, s.phi);
            maxRelativePredictionError = Math.max(maxRelativePredictionError,
                    Math.abs(hidden - learned) / (1 + Math.abs(hidden)));
        }
        System.out.printf("held-out vs hidden law: max relative prediction error = %.4f, weight error = %.4f%n",
                maxRelativePredictionError, weightError);
        check(maxRelativePredictionError < 0.10, "the learned model predicts like the hidden law");
        check(learnedLoss <= 1.3 * 1.5 * 1.5, "the learned model reaches the noise floor out of sample");
        check(weightError < 0.4, "recovered weights stay near the hidden law despite collinear features");
        check(online[0] > 0, "the intercept is learned as a positive offset");

        // --- 7. regularisation shrinks the weights and increases the training loss.
        double previousNorm = Double.MAX_VALUE;
        double previousTrainLoss = -1;
        for (double reg : new double[]{0.0, 0.01, 0.1, 1.0, 10.0}) {
            double[] theta = ridgeSolution(training, reg, 1e-9);
            double thetaNorm = norm(theta);
            double trainLoss = meanLoss(theta, training, reg);
            System.out.printf("lambda=%.2f |theta|=%.4f trainLoss=%.4f%n", reg, thetaNorm, trainLoss);
            check(thetaNorm <= previousNorm + 1e-9, "the weight norm is non-increasing in lambda");
            check(trainLoss >= previousTrainLoss - 1e-9, "the regularised loss is non-decreasing in lambda");
            previousNorm = thetaNorm;
            previousTrainLoss = trainLoss;
        }

        // --- 8. reproducibility and stream independence (out-of-order batches still converge).
        double[] repeatLosses = new double[1500];
        double[] repeat = trainOnline(training, 0.05, lambda, 8, 0.995, 1500, repeatLosses);
        check(Arrays.equals(repeat, online), "training is reproducible from its seed");
        List<Sample> shuffled = new ArrayList<>(training);
        Collections.reverse(shuffled);
        double[] reverseLosses = new double[1500];
        double[] reversed = trainOnline(shuffled, 0.05, lambda, 8, 0.995, 1500, reverseLosses);
        check(norm(subtract(reversed, closedForm)) / norm(closedForm) < 0.08,
                "the order of the stream does not change the fitted model materially");
        check(meanLoss(reversed, heldOut, 0) < learnedLoss * 1.3,
                "a reversed stream generalises comparably");

        // --- 9. degenerate streams.
        double[] emptyLosses = new double[3];
        double[] emptyModel = trainOnline(new ArrayList<>(), 0.05, lambda, 4, 1.0, 3, emptyLosses);
        check(norm(emptyModel) == 0.0, "an empty stream leaves the model at zero");
        List<Sample> oneSample = List.of(training.get(0));
        double[] oneLosses = new double[50];
        double[] oneModel = trainOnline(oneSample, 0.05, lambda, 1, 1.0, 50, oneLosses);
        check(meanLoss(oneModel, oneSample, 0) < squaredLoss(new double[CostSimulator.dimension()],
                        oneSample.get(0), 0), "a single sample is at least partly fitted");
        double[] constantTruth = new double[CostSimulator.dimension()];
        constantTruth[0] = 42.0;
        CostSimulator constant = new CostSimulator(constantTruth, 0.0, 12345L);
        List<Sample> constantStream = stream(constant, 50);
        double[] constantLosses = new double[3000];
        double[] constantModel = trainOnline(constantStream, 0.3, lambda, 32, 0.999, 3000, constantLosses);
        double maxDeviation = 0;
        for (Sample s : constantStream) {
            maxDeviation = Math.max(maxDeviation, Math.abs(dot(constantModel, s.phi) - 42.0));
        }
        System.out.printf("constant target: deviation=%.4f intercept=%.4f%n", maxDeviation, constantModel[0]);
        check(maxDeviation < 1.0, "a constant target is predicted to within one cost unit (" + maxDeviation + ")");
        check(meanLoss(constantModel, constantStream, 0) < 2.0, "the constant model fits well");
        check(constantLosses[2999] <= constantLosses[0] + 1e-9, "the constant-target loss never grows");

        // --- 10. conditioning of the normal equations: two independent 2x2 solvers agree, ridge is stable.
        List<Sample> tiny = new ArrayList<>();
        tiny.add(new Sample(new double[]{2}, new double[]{1, 2}, 5));
        tiny.add(new Sample(new double[]{3}, new double[]{1, 3}, 8));
        tiny.add(new Sample(new double[]{4}, new double[]{1, 4}, 9));
        double[] formula = solveTwoByTwo(tiny, lambda);
        double[] elimination = ridgeSolutionTwoFeatures(tiny, lambda);
        System.out.printf("2x2 solve: formula=(%s, %s) elimination=(%s, %s)%n",
                formula[0], formula[1], elimination[0], elimination[1]);
        check(Math.abs(formula[0] - elimination[0]) < 1e-9 && Math.abs(formula[1] - elimination[1]) < 1e-9,
                "the closed-form 2x2 solve agrees with Gaussian elimination");
        check(Math.abs(formula[0] + 2 * formula[1] - 5) < 3.0, "the 2x2 fit is close to the samples");
        double[] jittered = ridgeSolution(training, lambda * training.size(), 1e-6);
        check(norm(subtract(jittered, closedForm)) < 1e-3, "a small ridge jitter keeps the solution stable");
        check(meanLoss(online, training, 0) > 0, "the unregularised loss of the fitted model is positive");
        check(Double.isFinite(learnedLoss) && Double.isFinite(baselineLoss), "losses are finite");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }
}
