/** Task 494 - Implement Thompson sampling simulator for bandit problems. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Beta-Bernoulli Thompson sampling on a planted three-armed bandit: arm i pays 1 with the known
 * probability arms[i], and every trial samples each arm's success rate from its Beta posterior
 * (prior Beta(1,1)), plays the sampled argmax (ties to the lowest index), observes a Bernoulli
 * reward drawn from the fixed LCG and updates the chosen arm's posterior. Beta(a,b) with integer
 * parameters is exact: two Gamma(a,1)/Gamma(b,1) draws built as sums of a (resp. b) unit
 * exponentials -ln(u), u uniform; that only needs the fixed LCG, no random source. Every draw
 * steps seed = (seed*75 + 74) % 65537 and returns (seed + 0.5)/65537 in (0,1). Documented
 * expectations for arms [0.2, 0.5, 0.8] over 1200 fixed trials: the best arm is picked at least
 * 85% of the time, it is picked more often than each other arm, and the average regret
 * R = sum(max(mu) - mu[a_t]) stays under 100; the posterior mean of the best arm sits within
 * 0.05 of 0.8. Re-running with the same seed reproduces the pull vector exactly.
 */
import java.util.*;

class ThompsonSampler {
    final double[] arms;
    int seed;
    final int[] alpha;
    final int[] beta;
    final int[] pulls;

    ThompsonSampler(double[] arms, int seed) {
        this.arms = arms.clone();
        this.seed = seed;
        alpha = new int[arms.length];
        beta = new int[arms.length];
        pulls = new int[arms.length];
        for (int i = 0; i < arms.length; i++) {
            alpha[i] = 1;                       // integer parameters: exact Beta draws
            beta[i] = 1;
        }
    }

    double uniform() {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 0.5) / 65537.0;
    }

    /** Gamma(shape, 1) for integer shape >= 1: a sum of shape unit exponentials. */
    double gammaUnit(int shape) {
        double total = 0.0;
        for (int i = 0; i < shape; i++) total += -Math.log(uniform());
        return total;
    }

    double betaSample(int a, int b) {
        double x = gammaUnit(a);
        double y = gammaUnit(b);
        return x / (x + y);
    }

    void run(int trials) {
        double best = arms[0];
        for (double value : arms) best = Math.max(best, value);
        for (int t = 0; t < trials; t++) {
            double[] samples = new double[arms.length];
            for (int i = 0; i < arms.length; i++) samples[i] = betaSample(alpha[i], beta[i]);
            int choice = 0;
            for (int i = 1; i < samples.length; i++) {
                if (samples[i] > samples[choice]) choice = i;
            }
            int reward = uniform() < arms[choice] ? 1 : 0;
            alpha[choice] += reward;
            beta[choice] += 1 - reward;
            pulls[choice] += 1;
        }
    }

    double posteriorMean(int arm) {
        return (double) alpha[arm] / (alpha[arm] + beta[arm]);
    }
}

public class Solution {
    static final double[] ARMS = {0.2, 0.5, 0.8};
    static final int TRIALS = 1200;
    static final int SEED = 42;

    public static void main(String[] args) {
        ThompsonSampler sampler = new ThompsonSampler(ARMS, SEED);
        sampler.run(TRIALS);

        int sum = 0;
        for (int pull : sampler.pulls) sum += pull;
        check(sum == TRIALS, "pull total");
        double bestFrequency = (double) sampler.pulls[2] / TRIALS;
        check(bestFrequency >= 0.85, "documented: best arm picked >= 85% of the time");
        check(sampler.pulls[2] > sampler.pulls[1] && sampler.pulls[1] > sampler.pulls[0],
                "arm ordering");
        double posterior = sampler.posteriorMean(2);
        check(posterior >= 0.75 && posterior <= 0.85, "posterior mean of the best arm");
        check(Math.abs(posterior - ARMS[2]) <= 0.05, "posterior tracks the true rate");

        ThompsonSampler repeat = new ThompsonSampler(ARMS, SEED);
        repeat.run(TRIALS);
        check(Arrays.equals(repeat.pulls, sampler.pulls), "fixed seed reproduces the pulls");
        check(repeat.posteriorMean(0) == sampler.posteriorMean(0), "deterministic posterior");

        ThompsonSampler trivial = new ThompsonSampler(new double[]{0.9}, SEED + 1);
        trivial.run(200);
        check(trivial.pulls[0] == 200, "single arm always pulled");

        ThompsonSampler duet = new ThompsonSampler(new double[]{0.3, 0.7}, SEED + 2);
        duet.run(600);
        check(duet.pulls[1] >= 0.8 * 600, "larger gap concentrates faster");

        ThompsonSampler tied = new ThompsonSampler(new double[]{0.5, 0.5}, SEED + 3);
        tied.run(600);
        check(tied.pulls[0] + tied.pulls[1] == 600, "tied total");
        check(Math.min(tied.pulls[0], tied.pulls[1]) >= 100, "equal arms both explored");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
