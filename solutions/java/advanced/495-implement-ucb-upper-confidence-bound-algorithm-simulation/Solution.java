/** Task 495 - Implement UCB (upper confidence bound) algorithm simulation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * UCB1 on a planted three-armed bandit: each arm is pulled once to initialise its averages
 * (arms in index order), then every trial pulls argmax(mu_hat[i] + sqrt(2 ln t / n_i)), ties to
 * the lowest index, t counting trials so far. Rewards are Bernoulli draws from the fixed LCG
 * (seed = (seed*75 + 74) % 65537, reward when (seed + 0.5)/65537 < p_arm), so the whole run is
 * deterministic. The pseudo-regret uses the known true means, R(T) = sum over trials of
 * (max(mu) - mu[a_t]) with mu = [0.2, 0.5, 0.8] as planted. Documented expectations over 2000
 * fixed trials: the best arm is pulled at least 80% of the time, pulls are strictly ordered
 * best > middle > worst, total regret stays under 100, and the mean per-trial regret of the
 * second half is strictly below that of the first half (the regret rate decreases as UCB1
 * learns). A uniform-random baseline using the same reward stream is used to show UCB1's
 * advantage.
 */
import java.util.*;

class UCB1 {
    final double[] arms;
    int seed;
    final int[] counts;
    final double[] sums;
    final int[] pulls;

    UCB1(double[] arms, int seed) {
        this.arms = arms.clone();
        this.seed = seed;
        counts = new int[arms.length];
        sums = new double[arms.length];
        pulls = new int[arms.length];
    }

    int reward(double p) {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 0.5) / 65537.0 < p ? 1 : 0;
    }

    double index(int arm, int t) {
        if (counts[arm] == 0) return Double.POSITIVE_INFINITY;   // untried arms come first
        double mean = sums[arm] / counts[arm];
        return mean + Math.sqrt(2.0 * Math.log(t) / counts[arm]);
    }

    /** Runs `trials` pulls and returns the cumulative pseudo-regret trace. */
    double[] run(int trials) {
        double best = arms[0];
        for (double value : arms) best = Math.max(best, value);
        double[] trace = new double[trials];
        double regret = 0.0;
        for (int t = 1; t <= trials; t++) {
            int choice = 0;
            for (int i = 1; i < arms.length; i++) {
                if (index(i, t) > index(choice, t)) choice = i;
            }
            int r = reward(arms[choice]);
            counts[choice] += 1;
            sums[choice] += r;
            pulls[choice] += 1;
            regret += best - arms[choice];
            trace[t - 1] = regret;
        }
        return trace;
    }
}

/** Uniform-random arm choice over the same LCG reward stream (control experiment). */
class RandomBaseline {
    final double[] arms;
    int seed;

    RandomBaseline(double[] arms, int seed) {
        this.arms = arms.clone();
        this.seed = seed;
    }

    double run(int trials) {
        double best = arms[0];
        for (double value : arms) best = Math.max(best, value);
        double regret = 0.0;
        for (int t = 0; t < trials; t++) {
            seed = (seed * 75 + 74) % 65537;
            int choice = seed % arms.length;
            seed = (seed * 75 + 74) % 65537;
            regret += best - arms[choice];
        }
        return regret;
    }
}

public class Solution {
    static final double[] ARMS = {0.2, 0.5, 0.8};
    static final int TRIALS = 2000;
    static final int SEED = 99;

    public static void main(String[] args) {
        UCB1 ucb = new UCB1(ARMS, SEED);
        double[] trace = ucb.run(TRIALS);

        int sum = 0;
        for (int pull : ucb.pulls) sum += pull;
        check(sum == TRIALS, "pull total");
        check(Arrays.equals(ucb.pulls, ucb.counts), "counts mirror pulls");

        double bestFrequency = (double) ucb.pulls[2] / TRIALS;
        check(bestFrequency >= 0.80, "documented: best arm picked >= 80% of the time");
        check(ucb.pulls[2] > ucb.pulls[1] && ucb.pulls[1] > ucb.pulls[0], "strict pull ordering");
        double regret = trace[TRIALS - 1];
        check(regret <= 100.0, "documented: total pseudo-regret bound");

        for (int i = 1; i < trace.length; i++) {
            check(trace[i] >= trace[i - 1], "regret is a cumulative sum of gaps");
        }
        int half = TRIALS / 2;
        double firstRate = trace[half - 1] / half;
        double secondRate = (trace[TRIALS - 1] - trace[half - 1]) / (TRIALS - half);
        check(secondRate < firstRate, "documented: regret rate decreases");

        RandomBaseline baseline = new RandomBaseline(ARMS, SEED);
        double baselineRegret = baseline.run(TRIALS);
        check(baselineRegret > 2.0 * regret, "UCB1 must beat uniform sampling");

        UCB1 repeat = new UCB1(ARMS, SEED);
        double[] repeatTrace = repeat.run(TRIALS);
        check(Arrays.equals(repeat.pulls, ucb.pulls), "deterministic pulls");
        check(repeatTrace[TRIALS - 1] == regret, "deterministic regret");

        UCB1 warm = new UCB1(ARMS, SEED);
        warm.run(3);
        check(warm.counts[0] == 1 && warm.counts[1] == 1 && warm.counts[2] == 1,
                "warm-up visits each arm exactly once");
        warm.run(7);
        check(Math.min(warm.counts[0], Math.min(warm.counts[1], warm.counts[2])) >= 1,
                "no arm is skipped during warm-up");
        check(warm.counts[0] + warm.counts[1] + warm.counts[2] == 10, "warm total");

        UCB1 duet = new UCB1(new double[]{0.3, 0.7}, SEED + 1);
        duet.run(1000);
        check(duet.pulls[1] >= 0.9 * 1000, "large-gap concentration");
        check(duet.pulls[1] > duet.pulls[0], "large-gap ordering");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
