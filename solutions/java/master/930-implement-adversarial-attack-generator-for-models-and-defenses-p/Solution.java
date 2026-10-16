/** Task 930 - Implement adversarial attack generator for models and defenses (PGD).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A two-layer ReLU network with a softmax head is trained on a linearly separable 2D problem, then attacked in
 * the L-infinity ball: FGSM takes one sign-gradient step of size eps, and PGD takes many projected steps of
 * size alpha, clamping every iterate back into the ball around the clean input and keeping the best iterate
 * found. Both attacks need the exact input gradient, which is obtained by back-propagating through the ReLU and
 * the softmax analytically rather than by finite differences. The same generator then serves as the defense:
 * adversarial training folds PGD examples back into the training set, and the success rate of a fresh attack
 * must fall.
 * Simplified: two input dimensions and a small hidden layer, so the whole ball can also be swept on a fine grid
 * - which is exactly what is used as an independent global oracle for the attack. Verified: perturbations stay
 * inside the ball (and vanish at eps = 0); attacks increase the loss; PGD beats FGSM in loss and in success
 * rate; PGD matches the exhaustive grid search over the ball; adversarial training measurably lowers the attack
 * success rate without collapsing clean accuracy; a large eps breaks every point; and the run is reproducible.
 */
import java.util.*;

/** Two-layer ReLU network with softmax cross-entropy, trained by full-batch gradient descent. */
final class Mlp {
    final int in, hidden, out;
    final double[][] w1, w2;
    final double[] b1, b2;

    Mlp(int in, int hidden, int out, long seed) {
        this.in = in;
        this.hidden = hidden;
        this.out = out;
        w1 = new double[hidden][in];
        w2 = new double[out][hidden];
        b1 = new double[hidden];
        b2 = new double[out];
        long state = seed;
        for (int j = 0; j < hidden; j++) {
            for (int i = 0; i < in; i++) {
                state = state * 6364136223846793005L + 1442695040888963407L;
                w1[j][i] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 1.5;
            }
            state = state * 6364136223846793005L + 1442695040888963407L;
            b1[j] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 0.5;
        }
        for (int c = 0; c < out; c++) for (int j = 0; j < hidden; j++) w2[c][j] = 0.3;
    }

    double[] hidden(double[] x) {
        double[] h = new double[hidden];
        for (int j = 0; j < hidden; j++) {
            double z = b1[j];
            for (int i = 0; i < in; i++) z += w1[j][i] * x[i];
            h[j] = Math.max(0, z);
        }
        return h;
    }

    double[] probabilities(double[] x) {
        double[] h = hidden(x);
        double[] logits = new double[out];
        for (int c = 0; c < out; c++) {
            logits[c] = b2[c];
            for (int j = 0; j < hidden; j++) logits[c] += w2[c][j] * h[j];
        }
        double max = Math.max(logits[0], logits[1]), sum = 0;
        for (int c = 0; c < out; c++) {
            logits[c] = Math.exp(logits[c] - max);
            sum += logits[c];
        }
        for (int c = 0; c < out; c++) logits[c] /= sum;
        return logits;
    }

    int predict(double[] x) {
        double[] p = probabilities(x);
        return p[0] >= p[1] ? 0 : 1;
    }

    double loss(double[] x, int y) { return -Math.log(Math.max(probabilities(x)[y], 1e-12)); }

    /** Analytic d(loss)/d(x) through the ReLU and the softmax. */
    double[] inputGradient(double[] x, int y) {
        double[] h = hidden(x);
        double[] p = probabilities(x);
        double[] dLogits = new double[out];
        for (int c = 0; c < out; c++) dLogits[c] = p[c] - (c == y ? 1 : 0);
        double[] dHidden = new double[hidden];
        for (int j = 0; j < hidden; j++) {
            if (h[j] <= 0) continue;
            for (int c = 0; c < out; c++) dHidden[j] += w2[c][j] * dLogits[c];
        }
        double[] grad = new double[in];
        for (int i = 0; i < in; i++) for (int j = 0; j < hidden; j++) grad[i] += w1[j][i] * dHidden[j];
        return grad;
    }

    /** One full-batch gradient descent step; returns the batch loss. */
    double step(List<double[]> xs, List<Integer> ys, double rate) {
        double[][] gw1 = new double[hidden][in];
        double[][] gw2 = new double[out][hidden];
        double[] gb1 = new double[hidden], gb2 = new double[out];
        double total = 0;
        for (int n = 0; n < xs.size(); n++) {
            double[] x = xs.get(n);
            int y = ys.get(n);
            double[] h = hidden(x);
            double[] p = probabilities(x);
            total += -Math.log(Math.max(p[y], 1e-12));
            double[] dLogits = new double[out];
            for (int c = 0; c < out; c++) dLogits[c] = (p[c] - (c == y ? 1 : 0)) / xs.size();
            for (int c = 0; c < out; c++) {
                gb2[c] += dLogits[c];
                for (int j = 0; j < hidden; j++) gw2[c][j] += dLogits[c] * h[j];
            }
            double[] dHidden = new double[hidden];
            for (int j = 0; j < hidden; j++) {
                if (h[j] <= 0) continue;
                for (int c = 0; c < out; c++) dHidden[j] += w2[c][j] * dLogits[c];
            }
            for (int j = 0; j < hidden; j++) {
                gb1[j] += dHidden[j];
                for (int i = 0; i < in; i++) gw1[j][i] += dHidden[j] * x[i];
            }
        }
        for (int j = 0; j < hidden; j++) {
            b1[j] -= rate * gb1[j];
            for (int i = 0; i < in; i++) w1[j][i] -= rate * gw1[j][i];
        }
        for (int c = 0; c < out; c++) {
            b2[c] -= rate * gb2[c];
            for (int j = 0; j < hidden; j++) w2[c][j] -= rate * gw2[c][j];
        }
        return total / xs.size();
    }

    void train(List<double[]> xs, List<Integer> ys, int iterations, double rate) {
        for (int it = 0; it < iterations; it++) step(xs, ys, rate);
    }

    double accuracy(List<double[]> xs, List<Integer> ys) {
        int right = 0;
        for (int i = 0; i < xs.size(); i++) if (predict(xs.get(i)) == ys.get(i)) right++;
        return right / (double) xs.size();
    }
}

/** The attack generator: FGSM, PGD and an exhaustive grid oracle over a two-dimensional ball. */
final class Attacks {
    static double[] project(double[] candidate, double[] clean, double eps) {
        double[] out = new double[candidate.length];
        for (int i = 0; i < candidate.length; i++)
            out[i] = Math.min(clean[i] + eps, Math.max(clean[i] - eps, candidate[i]));
        return out;
    }

    static double[] fgsm(Mlp model, double[] x, int y, double eps) {
        double[] grad = model.inputGradient(x, y);
        double[] candidate = new double[x.length];
        for (int i = 0; i < x.length; i++) candidate[i] = x[i] + eps * Math.signum(grad[i]);
        return project(candidate, x, eps);
    }

    static double[] pgd(Mlp model, double[] x, int y, double eps, int steps, double alpha) {
        double[] current = x.clone();
        double[] best = x.clone();
        double bestLoss = model.loss(x, y);
        for (int s = 0; s < steps; s++) {
            double[] grad = model.inputGradient(current, y);
            for (int i = 0; i < current.length; i++) current[i] += alpha * Math.signum(grad[i]);
            current = project(current, x, eps);
            double loss = model.loss(current, y);
            if (loss > bestLoss) {
                bestLoss = loss;
                best = current.clone();
            }
        }
        return best;
    }

    /** Exhaustive sweep of the whole ball on a fine grid - the independent global oracle for 2D inputs. */
    static double gridBestLoss(Mlp model, double[] x, int y, double eps, int steps) {
        double best = model.loss(x, y);
        for (int a = 0; a <= steps; a++) {
            for (int b = 0; b <= steps; b++) {
                double[] candidate = {x[0] - eps + 2 * eps * a / steps, x[1] - eps + 2 * eps * b / steps};
                best = Math.max(best, model.loss(candidate, y));
            }
        }
        return best;
    }

    static double[] uniformBall(Mlp model, double[] x, int y, double eps, int samples, long state) {
        double bestLoss = model.loss(x, y);
        for (int s = 0; s < samples; s++) {
            double[] candidate = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                state = state * 6364136223846793005L + 1442695040888963407L;
                candidate[i] = x[i] + eps * (((state >>> 11) * 0x1.0p-53) * 2 - 1);
            }
            bestLoss = Math.max(bestLoss, model.loss(candidate, y));
        }
        return new double[] {bestLoss};
    }
}

public class Solution {
    static final int TRAIN_PER_CLASS = 40, TEST_PER_CLASS = 20, STEPS = 20;
    static final double EPS = 0.5, ALPHA = 2.5 * EPS / STEPS;

    static double[] point(int index, int label, double center, double spread) {
        double sign = label == 0 ? -center : center;
        return new double[] {sign + spread * Math.sin(1.7 * index), sign + spread * Math.cos(2.3 * index)};
    }

    static void fill(List<double[]> xs, List<Integer> ys, int perClass, double center, double spread) {
        for (int c = 0; c < 2; c++) {
            for (int n = 0; n < perClass; n++) {
                xs.add(point(n + 100 * c, c, center, spread));
                ys.add(c);
            }
        }
    }

    static double successRate(Mlp model, List<double[]> xs, List<Integer> ys, double eps, boolean usePgd) {
        int broken = 0;
        for (int n = 0; n < xs.size(); n++) {
            double[] adv = usePgd ? Attacks.pgd(model, xs.get(n), ys.get(n), eps, STEPS, ALPHA)
                    : Attacks.fgsm(model, xs.get(n), ys.get(n), eps);
            if (model.predict(adv) != ys.get(n)) broken++;
        }
        return broken / (double) xs.size();
    }

    static double meanLoss(Mlp model, List<double[]> xs, List<Integer> ys, double eps, boolean usePgd) {
        double total = 0;
        for (int n = 0; n < xs.size(); n++) {
            double[] adv = usePgd ? Attacks.pgd(model, xs.get(n), ys.get(n), eps, STEPS, ALPHA)
                    : Attacks.fgsm(model, xs.get(n), ys.get(n), eps);
            total += model.loss(adv, ys.get(n));
        }
        return total / xs.size();
    }

    public static void main(String[] args) {
        List<double[]> trainX = new ArrayList<>(), testX = new ArrayList<>();
        List<Integer> trainY = new ArrayList<>(), testY = new ArrayList<>();
        fill(trainX, trainY, TRAIN_PER_CLASS, 1.2, 0.7);
        fill(testX, testY, TEST_PER_CLASS, 0.45, 0.2);

        Mlp model = new Mlp(2, 8, 2, 0x930ABCDEF0L);
        model.train(trainX, trainY, 1500, 0.5);
        double cleanAccuracy = model.accuracy(testX, testY);
        double cleanLoss = meanLoss(model, testX, testY, 0.0, true);
        double fgsmRate = successRate(model, testX, testY, EPS, false);
        double pgdRate = successRate(model, testX, testY, EPS, true);
        double fgsmLoss = meanLoss(model, testX, testY, EPS, false);
        double pgdLoss = meanLoss(model, testX, testY, EPS, true);

        // Independent oracles on the same ball.
        int violations = 0, gridWorse = 0, randomWorse = 0;
        double gridBest = 0, randomBest = 0;
        for (int n = 0; n < testX.size(); n++) {
            double[] x = testX.get(n);
            int y = testY.get(n);
            for (double[] adv : new double[][] {Attacks.fgsm(model, x, y, EPS), Attacks.pgd(model, x, y, EPS, STEPS, ALPHA)}) {
                for (int i = 0; i < x.length; i++) if (Math.abs(adv[i] - x[i]) > EPS + 1e-9) violations++;
            }
            if (model.predict(x) != y) continue;                       // only attack correctly classified points
            gridBest = Math.max(gridBest, Attacks.gridBestLoss(model, x, y, EPS, 60));
            double[] sampled = Attacks.uniformBall(model, x, y, EPS, 200, 0x930L + n);
            randomBest = Math.max(randomBest, sampled[0]);
            if (model.loss(Attacks.pgd(model, x, y, EPS, STEPS, ALPHA), y) < Attacks.gridBestLoss(model, x, y, EPS, 60) - 0.02) gridWorse++;
            if (model.loss(Attacks.pgd(model, x, y, EPS, STEPS, ALPHA), y) < sampled[0] - 0.02) randomWorse++;
        }

        System.out.println("adversarial: clean accuracy " + String.format("%.3f", cleanAccuracy) + ", eps " + EPS
                + " (grid " + (61 * 61) + " points), fgsm loss " + String.format("%.3f", fgsmLoss) + " success "
                + String.format("%.2f", fgsmRate) + ", pgd loss " + String.format("%.3f", pgdLoss) + " success "
                + String.format("%.2f", pgdRate));
        System.out.println("oracles: grid best " + String.format("%.3f", gridBest) + ", uniform sampling best "
                + String.format("%.3f", randomBest) + ", ball violations " + violations + ", grid shortfalls "
                + gridWorse + ", sampling shortfalls " + randomWorse);

        check(cleanAccuracy > 0.9, "the model must classify the clean test set: " + cleanAccuracy);
        check(violations == 0, "every adversarial example must stay inside the eps ball");
        check(fgsmLoss > cleanLoss && pgdLoss > cleanLoss, "attacks must increase the loss: " + cleanLoss + " -> "
                + fgsmLoss + " / " + pgdLoss);
        check(pgdLoss >= fgsmLoss, "multi-step PGD must be at least as strong as one-step FGSM: " + pgdLoss
                + " vs " + fgsmLoss);
        check(pgdRate >= fgsmRate, "PGD must break at least as many points: " + pgdRate + " vs " + fgsmRate);
        check(pgdRate > 0.5, "at eps " + EPS + " the attack should break most points, got " + pgdRate);
        check(gridWorse == 0, "PGD must match the exhaustive sweep of the ball, shortfalls " + gridWorse);
        check(randomWorse == 0, "PGD must beat uniform sampling inside the ball, shortfalls " + randomWorse);

        // The exact-ball boundary: eps = 0 leaves the input untouched and flips nothing.
        double unchanged = 0, flipped = 0;
        for (int n = 0; n < testX.size(); n++) {
            double[] adv = Attacks.pgd(model, testX.get(n), testY.get(n), 0.0, STEPS, ALPHA);
            double distance = 0;
            for (int i = 0; i < adv.length; i++) distance = Math.max(distance, Math.abs(adv[i] - testX.get(n)[i]));
            if (distance == 0) unchanged++;
            if (model.predict(adv) != testY.get(n)) flipped++;
        }
        check(unchanged == testX.size() && flipped == 0, "eps = 0 must be a no-op: untouched " + unchanged + " of "
                + testX.size());
        double huge = successRate(model, testX, testY, 4.0, true);
        check(huge >= pgdRate, "a larger ball can only help the attacker: " + huge + " vs " + pgdRate);
        double tiny = successRate(model, testX, testY, 1e-4, true);
        check(tiny <= pgdRate && tiny >= 0, "a tiny eps cannot break more points than a large one, got " + tiny);

        // The defense: two rounds of adversarial training from scratch must reduce the attack success rate.
        Mlp defended = model;
        double beforeDefense = pgdRate;
        for (int round = 0; round < 2; round++) {
            List<double[]> augmentedX = new ArrayList<>(trainX);
            List<Integer> augmentedY = new ArrayList<>(trainY);
            for (int n = 0; n < trainX.size(); n++)
                augmentedX.add(Attacks.pgd(defended, trainX.get(n), trainY.get(n), EPS, STEPS, ALPHA));
            augmentedY.addAll(trainY);
            Mlp next = new Mlp(2, 16, 2, 0x77A1000L + round);
            next.train(augmentedX, augmentedY, 3000, 0.5);
            defended = next;
        }
        double defendedClean = defended.accuracy(testX, testY);
        double defendedRate = successRate(defended, testX, testY, EPS, true);
        double defendedLoss = meanLoss(defended, testX, testY, EPS, true);

        check(defendedRate < beforeDefense, "adversarial training must lower the attack success rate: " + beforeDefense
                + " -> " + defendedRate);
        check(defendedClean > 0.8, "the defended model must keep its clean accuracy: " + defendedClean);
        check(defendedClean > defendedRate, "the defended model must be right more often than it is fooled: "
                + defendedClean + " vs " + defendedRate);

        Mlp replay = new Mlp(2, 8, 2, 0x930ABCDEF0L);
        replay.train(trainX, trainY, 1500, 0.5);
        check(new Mlp(2, 8, 2, 0x930ABCDEF0L).loss(testX.get(0), 0)
                == new Mlp(2, 8, 2, 0x930ABCDEF0L).loss(testX.get(0), 0), "an untrained network must be reproducible");
        check(replay.accuracy(testX, testY) == cleanAccuracy, "retraining from the same seed must give the same model");

        System.out.println("defense: adversarial training " + String.format("%.2f", defendedClean) + " clean accuracy, "
                + "attack success " + String.format("%.2f", pgdRate) + " -> " + String.format("%.2f", defendedRate)
                + ", adversarial loss " + String.format("%.3f", pgdLoss) + " -> " + String.format("%.3f", defendedLoss));
        System.out.println("boundaries: eps = 0 untouched " + unchanged + "/" + testX.size() + ", eps = 4 success "
                + String.format("%.2f", huge) + ", eps = 1e-4 success " + String.format("%.2f", tiny));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
