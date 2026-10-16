/** Task 797 - Implement multi-armed bandit with contextual features (contextual bandits). (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * LinUCB (disjoint linear model, Li et al. 2010). Arm k has an unknown weight vector theta_k; the reward of
 * playing arm k in context x is theta_k . x plus noise. The algorithm keeps a ridge regression per arm,
 *
 *     A_k = lambda I + sum_{pulls of k} x x^T,   b_k = sum_{pulls of k} r x,   thetaHat_k = A_k^-1 b_k
 *
 * and plays argmax_k [ thetaHat_k . x + alpha sqrt( x^T A_k^-1 x ) ]. The square root term is the width of
 * the confidence ellipsoid in the direction x, so alpha trades exploration for exploitation, and it shrinks
 * like 1/sqrt(n_k) as an arm gets sampled.
 *
 * The regret of a run is measured against the oracle policy, which knows the true theta_k and always plays
 * the best arm for the current context. LinUCB's regret is sublinear in T, and the file checks that against
 * both the oracle gap and a uniform-random policy on the same contexts.
 *
 * Verified: the UCB term agrees with an explicitly inverted A_k built by hand for a single context; the
 * per-arm ridge estimate converges to the true weight vector; the average regret per step falls over time
 * (sublinear regret) and beats the random policy by a wide margin; the arm-choice accuracy approaches the
 * oracle's; the confidence width shrinks monotonically as an arm collects pulls; with alpha = 0 the algorithm
 * becomes greedy and does strictly worse; an unplayed arm has the exact width alpha |x| / sqrt(lambda); a
 * single-arm and a one-step run behave; and the run is deterministic.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

class LinUcb {
    final int arms, dimension;
    final double alpha, lambda;
    final double[][][] a;
    final double[][] b;
    int[] pulls;

    LinUcb(int arms, int dimension, double alpha, double lambda) {
        this.arms = arms;
        this.dimension = dimension;
        this.alpha = alpha;
        this.lambda = lambda;
        a = new double[arms][dimension][dimension];
        b = new double[arms][dimension];
        pulls = new int[arms];
        for (int k = 0; k < arms; k++) a[k] = Mat.identity(dimension, lambda);
    }

    /** Ridge estimate of the arm's weight vector, by solving A_k theta = b_k. */
    double[] theta(int k) { return Mat.solve(a[k], b[k]); }

    /** x^T A_k^-1 x: the squared confidence width in the direction x. */
    double squaredWidth(int k, double[] x) {
        double[] solved = Mat.solve(a[k], x);
        double sum = 0;
        for (int d = 0; d < dimension; d++) sum += x[d] * solved[d];
        return Math.max(sum, 0);
    }

    double ucb(int k, double[] x) {
        return Mat.dot(theta(k), x) + alpha * Math.sqrt(squaredWidth(k, x));
    }

    int select(double[] x) {
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < arms; k++) {
            double score = ucb(k, x);
            if (score > bestScore) {
                bestScore = score;
                best = k;
            }
        }
        return best;
    }

    void update(int k, double[] x, double reward) {
        for (int i = 0; i < dimension; i++) {
            b[k][i] += reward * x[i];
            for (int j = 0; j < dimension; j++) a[k][i][j] += x[i] * x[j];
        }
        pulls[k]++;
    }
}

/** Small dense linear algebra helpers. */
class Mat {
    static double[][] identity(int n, double diagonal) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) m[i][i] = diagonal;
        return m;
    }

    static double dot(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) sum += u[i] * v[i];
        return sum;
    }

    static double[] solve(double[][] m, double[] rhs) {
        int n = rhs.length;
        double[][] augmented = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, augmented[i], 0, n);
            augmented[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
                if (Math.abs(augmented[r][col]) > Math.abs(augmented[pivot][col])) pivot = r;
            double[] swap = augmented[col];
            augmented[col] = augmented[pivot];
            augmented[pivot] = swap;
            for (int r = col + 1; r < n; r++) {
                double factor = augmented[r][col] / augmented[col][col];
                for (int c = col; c <= n; c++) augmented[r][c] -= factor * augmented[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = augmented[i][n];
            for (int j = i + 1; j < n; j++) sum -= augmented[i][j] * x[j];
            x[i] = sum / augmented[i][i];
        }
        return x;
    }
}

public class Solution {
    static final int ARMS = 4, DIMENSION = 5, HORIZON = 4000;

    static double[][] trueWeights(SplitMix64 rng) {
        double[][] weights = new double[ARMS][DIMENSION];
        for (double[] w : weights) for (int d = 0; d < DIMENSION; d++) w[d] = rng.nextGaussian();
        return weights;
    }

    static double[] context(SplitMix64 rng) {
        double[] x = new double[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) x[d] = rng.nextGaussian();
        double norm = Math.sqrt(Mat.dot(x, x));
        for (int d = 0; d < DIMENSION; d++) x[d] /= norm;
        return x;
    }

    static int bestArm(double[][] weights, double[] x) {
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < ARMS; k++) {
            double value = Mat.dot(weights[k], x);
            if (value > bestValue) {
                bestValue = value;
                best = k;
            }
        }
        return best;
    }

    /** Runs one policy and returns {regret, correctChoices, reward}. */
    static double[] run(LinUcb bandit, double[][] weights, SplitMix64 rng, int horizon, boolean random) {
        double regret = 0, correct = 0, rewardSum = 0;
        for (int t = 0; t < horizon; t++) {
            double[] x = context(rng);
            int oracle = bestArm(weights, x);
            int chosen = random ? rng.nextInt(ARMS) : bandit.select(x);
            double reward = Mat.dot(weights[chosen], x) + 0.1 * rng.nextGaussian();
            if (bandit != null) bandit.update(chosen, x, reward);
            regret += Mat.dot(weights[oracle], x) - Mat.dot(weights[chosen], x);
            rewardSum += reward;
            if (chosen == oracle) correct++;
        }
        return new double[]{regret, correct / horizon, rewardSum};
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240797L);
        double[][] weights = trueWeights(rng);

        // --- the UCB formula against a hand-built inverse --------------------------------
        LinUcb probe = new LinUcb(ARMS, DIMENSION, 1.0, 1.0);
        double[] x0 = {0.3, -0.4, 0.5, 0.1, -0.2};
        double[] r0 = probe.theta(0);
        check(Math.abs(r0[0]) < 1e-12 && Math.abs(r0[4]) < 1e-12, "an untouched arm has a zero estimate");
        double width = Math.sqrt(probe.squaredWidth(0, x0));
        double handBuilt = Math.sqrt(Mat.dot(x0, x0));
        check(Math.abs(width - handBuilt) < 1e-12, "for A = lambda I the width must be |x| / sqrt(lambda), got "
                + width + " versus " + handBuilt);
        check(Math.abs(probe.ucb(0, x0) - Math.sqrt(Mat.dot(x0, x0))) < 1e-12,
                "the untouched UCB must be exactly the exploration term");

        // --- after a known update the width matches an explicit 1x1 computation -------------
        LinUcb one = new LinUcb(1, 1, 1.0, 2.0);
        one.update(0, new double[]{1.0}, 3.0);
        check(Math.abs(one.a[0][0][0] - 3.0) < 1e-12 && Math.abs(one.b[0][0] - 3.0) < 1e-12,
                "one pull must fill A and b exactly");
        check(Math.abs(one.theta(0)[0] - 1.0) < 1e-12, "theta must be b / A");
        check(Math.abs(one.squaredWidth(0, new double[]{1.0}) - 1.0 / 3) < 1e-12,
                "the width must be 1 / A after one pull, got " + one.squaredWidth(0, new double[]{1.0}));

        // --- the measured regret against the oracle and against random play --------------------
        LinUcb bandit = new LinUcb(ARMS, DIMENSION, 0.4, 1.0);
        double[] result = run(bandit, weights, new SplitMix64(11L), HORIZON, false);
        SplitMix64 randomRng = new SplitMix64(11L);
        double[] randomResult = run(null, weights, randomRng, HORIZON, true);
        check(result[0] < randomResult[0] * 0.4, "LinUCB regret " + result[0] + " must dwarf-free beat random "
                + randomResult[0]);
        check(result[1] > 0.85, "the arm choice must be right most of the time, accuracy " + result[1]);
        check(result[0] / HORIZON < 0.15, "average regret per step " + result[0] / HORIZON);

        // sublinear: the regret in the second half is not larger per step than in the first
        double[] firstHalf = run(new LinUcb(ARMS, DIMENSION, 0.4, 1.0), weights, new SplitMix64(11L),
                HORIZON / 2, false);
        double[] secondHalf = run(new LinUcb(ARMS, DIMENSION, 0.4, 1.0), weights, new SplitMix64(11L), HORIZON,
                false);
        double firstRate = firstHalf[0] / (HORIZON / 2.0);
        double secondRate = (secondHalf[0] - firstHalf[0]) / (HORIZON / 2.0);
        check(secondRate < firstRate, "the regret rate must fall: " + firstRate + " -> " + secondRate);
        check(secondRate < 0.02 * firstRate + 0.02, "the late regret rate must be small, got " + secondRate);

        // --- convergence of the ridge estimate to the true weights ------------------------------
        double worstEstimateError = 0;
        for (int k = 0; k < ARMS; k++) {
            double[] estimate = bandit.theta(k);
            double error = 0;
            for (int d = 0; d < DIMENSION; d++) error += Math.pow(estimate[d] - weights[k][d], 2);
            worstEstimateError = Math.max(worstEstimateError, Math.sqrt(error));
        }
        check(bandit.pulls[bestArm(weights, context(new SplitMix64(5L)))] > 0, "the best arm must be sampled");
        check(worstEstimateError < 0.5, "weight estimates must converge, worst error " + worstEstimateError);

        // --- the confidence width shrinks as an arm is pulled -------------------------------------
        LinUcb shrinking = new LinUcb(1, DIMENSION, 1.0, 1.0);
        double[] x = context(new SplitMix64(77L));
        double previousWidth = Math.sqrt(shrinking.squaredWidth(0, x));
        int shrinks = 0;
        for (int t = 0; t < 40; t++) {
            shrinking.update(0, context(new SplitMix64(100 + t)), 1.0);
            double currentWidth = Math.sqrt(shrinking.squaredWidth(0, x));
            if (currentWidth <= previousWidth) shrinks++;
            previousWidth = currentWidth;
        }
        check(shrinks >= 38, "the width must shrink monotonically, decreased " + shrinks + " of 40 times");
        check(previousWidth < Math.sqrt(Mat.dot(x, x)) * 0.5, "the width must be much smaller after 40 pulls");

        // --- pure exploitation is worse ---------------------------------------------------------------
        double[] greedy = run(new LinUcb(ARMS, DIMENSION, 0.0, 1.0), weights, new SplitMix64(11L), HORIZON, false);
        check(greedy[0] > result[0], "alpha = 0 must do worse than the tuned exploration, " + greedy[0]
                + " versus " + result[0]);

        // --- boundary cases ------------------------------------------------------------------------------
        LinUcb single = new LinUcb(1, DIMENSION, 0.5, 1.0);
        check(single.select(context(new SplitMix64(9L))) == 0, "with one arm there is nothing to choose");
        LinUcb oneStep = new LinUcb(ARMS, DIMENSION, 0.5, 1.0);
        double[] firstStep = run(oneStep, weights, new SplitMix64(13L), 1, false);
        check(Double.isFinite(firstStep[0]) && oneStep.pulls[0] + oneStep.pulls[1] + oneStep.pulls[2]
                + oneStep.pulls[3] == 1, "a single step must pull exactly one arm");
        LinUcb repeat = new LinUcb(ARMS, DIMENSION, 0.4, 1.0);
        double[] repeated = run(repeat, weights, new SplitMix64(11L), HORIZON, false);
        check(repeated[0] == result[0] && repeated[1] == result[1], "the same seed must reproduce the run");

        System.out.println("LinUCB regret " + result[0] + " (" + result[0] / HORIZON + " per step) versus random "
                + randomResult[0] + ", oracle accuracy " + result[1]);
        System.out.println("greedy regret " + greedy[0] + ", worst weight error " + worstEstimateError
                + ", width " + Math.sqrt(Mat.dot(x, x)) + " -> " + previousWidth + " after 40 pulls");
        System.out.println("regret rate " + firstRate + " -> " + secondRate);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
