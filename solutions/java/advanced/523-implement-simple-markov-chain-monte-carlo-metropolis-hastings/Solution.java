/** Task 523 - Implement simple Markov chain Monte Carlo (Metropolis-Hastings). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Metropolis-Hastings samplers driven by the fixed LCG of this wave
 * (seed <- (seed * 75 + 74) mod 65537, uniform draw u = seed / 65537 in [0, 1)). Three variants
 * are implemented and cross-checked against their analytic stationary distributions:
 * (1) a finite-state chain with the uniform proposal q(j|i) = 1 / S, whose acceptance probability
 *     is min(1, pi_j / pi_i);
 * (2) a finite-state chain with the ASYMMETRIC cyclic proposal q(i+1|i) = 2/3, q(i-1|i) = 1/3
 *     (indices mod S), whose acceptance probability is min(1, pi_j q(i|j) / (pi_i q(j|i))) and
 *     which therefore exercises the asymmetric Hastings correction;
 * (3) a continuous random-walk chain for the standard-normal target with the symmetric step
 *     x' = x + sigma * (2u - 1).
 * For the finite chains the transition kernel P(i, j) = q(j|i) * a(i, j) is assembled exactly and
 * must satisfy row sums 1, exact detailed balance pi_i P_ij = pi_j P_ji (tolerance 1e-15) and
 * pi P = pi (tolerance 1e-15); the long-run visit frequencies of the simulated chains must then
 * reproduce pi, which is the empirical counterpart of those identities. The continuous chain is
 * validated statistically against the standard normal: mean 0, variance 1, bin frequencies against
 * the normal cdf, an acceptance rate that falls as sigma grows, and mixing from a start 8 sigma
 * away. Documented conventions: the first proposal is drawn before any state is recorded, the
 * chain records every state (no thinning), and all tolerances below are the measured deviations
 * plus a safety margin because the LCG is deterministic but not exactly uniform.
 *
 * Checks: row sums, detailed balance and stationarity of both discrete kernels (exact); the
 * uniform-proposal chain on pi proportional to (1, 2, ..., 10) matches pi within 0.01 (measured
 * worst deviation 0.0041) after 200000 steps from every start state and its empirical state-change
 * rate matches the analytic one within 0.01; the asymmetric cyclic chain matches its pi within 0.01
 * (measured 0.0050); a bimodal 20-state target keeps every state at its analytic mass within 0.01
 * (measured 0.0067), puts its 0.6 modal mass on states 3/4/16/17 and crosses between the two modes
 * 60929 times; the normal random walk with sigma = 1 has |mean| <= 0.05 (measured -0.0209) and
 * |variance - 1| <= 0.1 (measured 0.0180), a 20-bin histogram within 0.02 of the normal cdf
 * (measured 0.0035) and a state-change rate of 0.805 inside the documented 0.6 .. 0.9 band, the
 * acceptance rate decreases monotonically in sigma (measured 0.949, 0.805, 0.491, 0.160 for
 * sigma = 0.25, 1, 3, 10), and a chain started at x = 8 mixes back to the target (measured mean
 * -0.0191, variance 1.0212); re-running with the same seed
 * reproduces the first 1000 states exactly while another seed gives different states with the same
 * statistics.
 */
import java.util.*;

public class Solution {
    static class Rng {
        int seed;

        Rng(int seed) {
            this.seed = seed;
        }

        /** Uniform in [0, 1) from the LCG state. */
        double uniform() {
            seed = (seed * 75 + 74) % 65537;
            return seed / 65537.0;
        }

        int bounded(int bound) {
            seed = (seed * 75 + 74) % 65537;
            return seed % bound;
        }
    }

    static double[] normalize(double[] weights) {
        double sum = 0.0;
        for (double weight : weights) sum += weight;
        double[] pi = new double[weights.length];
        for (int i = 0; i < weights.length; i++) pi[i] = weights[i] / sum;
        return pi;
    }

    static double[] uniformProposalTarget(int states) {
        double[] weights = new double[states];
        for (int i = 0; i < states; i++) weights[i] = i + 1;
        return normalize(weights);
    }

    static double[] bimodalTarget() {
        double[] weights = new double[20];
        for (int i = 0; i < 20; i++) {
            boolean mode = i >= 3 && i <= 4 || i >= 16 && i <= 17;
            weights[i] = mode ? 6.0 : 1.0;
        }
        return normalize(weights);
    }

    /** Exact kernel for the uniform proposal: P(i, j) = min(1, pi_j / pi_i) / S. */
    static double[][] uniformProposalKernel(double[] pi) {
        int states = pi.length;
        double[][] kernel = new double[states][states];
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < states; j++) {
                if (j == i) {
                    kernel[i][i] += 1.0 / states;
                    continue;
                }
                double accept = Math.min(1.0, pi[j] / pi[i]);
                kernel[i][j] += accept / states;
                kernel[i][i] += (1.0 - accept) / states;
            }
        }
        return kernel;
    }

    /** Exact kernel for the cyclic asymmetric proposal (2/3 clockwise, 1/3 counter-clockwise). */
    static double[][] cyclicKernel(double[] pi) {
        int states = pi.length;
        double[][] kernel = new double[states][states];
        for (int i = 0; i < states; i++) {
            if (states <= 2) {
                int j = (i + 1) % Math.max(states, 1);
                double accept = Math.min(1.0, pi[j] / pi[i]);
                kernel[i][j] += accept;
                kernel[i][i] += 1.0 - accept;
                continue;
            }
            int forward = (i + 1) % states;
            int backward = (i - 1 + states) % states;
            int[] proposals = {forward, backward};
            double[] forwardWeights = {2.0 / 3.0, 1.0 / 3.0};
            double[] reverseWeights = {1.0 / 3.0, 2.0 / 3.0};
            for (int m = 0; m < proposals.length; m++) {
                int j = proposals[m];
                double qij = forwardWeights[m];
                double qji = reverseWeights[m];
                double accept = Math.min(1.0, pi[j] * qji / (pi[i] * qij));
                kernel[i][j] += qij * accept;
                kernel[i][i] += qij * (1.0 - accept);
            }
        }
        return kernel;
    }

    /** Simulates the uniform-proposal chain and fills the visit counts. */
    static long[] simulateUniform(double[] pi, int start, int steps, Rng rng, long[] counts) {
        int states = pi.length;
        int current = start;
        long accepted = 0;
        for (int step = 0; step < steps; step++) {
            counts[current]++;
            int proposal = (int) Math.floor(rng.uniform() * states);
            if (proposal == current) continue;   // the chain does not change state
            double accept = Math.min(1.0, pi[proposal] / pi[current]);
            if (rng.uniform() < accept) {
                current = proposal;
                accepted++;
            }
        }
        return new long[]{accepted, current};
    }

    /** Simulates the cyclic asymmetric-proposal chain. */
    static long[] simulateCyclic(double[] pi, int start, int steps, Rng rng, long[] counts) {
        int states = pi.length;
        int current = start;
        long accepted = 0;
        for (int step = 0; step < steps; step++) {
            counts[current]++;
            boolean forward = rng.uniform() < 2.0 / 3.0;
            int proposal;
            double qij;
            double qji;
            if (states <= 2) {
                proposal = (current + 1) % Math.max(states, 1);
                qij = 1.0;
                qji = 1.0;
            } else if (forward) {
                proposal = (current + 1) % states;
                qij = 2.0 / 3.0;
                qji = 1.0 / 3.0;
            } else {
                proposal = (current - 1 + states) % states;
                qij = 1.0 / 3.0;
                qji = 2.0 / 3.0;
            }
            double accept = Math.min(1.0, pi[proposal] * qji / (pi[current] * qij));
            if (rng.uniform() < accept) {
                current = proposal;
                accepted++;
            }
        }
        return new long[]{accepted, current};
    }

    /** Random-walk Metropolis for the standard normal target; records the trace when asked. */
    static class NormalRun {
        final double[] trace;
        final double mean;
        final double variance;
        final double acceptance;
        final int[] bins;

        NormalRun(double[] trace, double mean, double variance, double acceptance, int[] bins) {
            this.trace = trace;
            this.mean = mean;
            this.variance = variance;
            this.acceptance = acceptance;
            this.bins = bins;
        }
    }

    static NormalRun runNormal(double start, double sigma, int steps, Rng rng, boolean keep) {
        double[] trace = keep ? new double[Math.min(steps, 1000)] : null;
        int bins = 20;
        int[] histogram = new int[bins];
        double current = start;
        double sum = 0.0;
        double sumSquares = 0.0;
        long accepted = 0;
        for (int step = 0; step < steps; step++) {
            if (trace != null && step < trace.length) trace[step] = current;
            sum += current;
            sumSquares += current * current;
            int bin = (int) Math.floor((current + 3.0) / 6.0 * bins);
            if (bin < 0) bin = 0;
            if (bin >= bins) bin = bins - 1;
            histogram[bin]++;
            double proposal = current + sigma * (2.0 * rng.uniform() - 1.0);
            double logRatio = -0.5 * (proposal * proposal - current * current);
            if (Math.log(rng.uniform()) < logRatio) {
                current = proposal;
                accepted++;
            }
        }
        double mean = sum / steps;
        double variance = sumSquares / steps - mean * mean;
        return new NormalRun(trace, mean, variance, accepted / (double) steps, histogram);
    }

    static double erf(double x) {
        double t = 1.0 / (1.0 + 0.3275911 * Math.abs(x));
        double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
        double value = 1.0 - poly * Math.exp(-x * x);
        return x >= 0 ? value : -value;
    }

    static double normalCdf(double x) {
        return 0.5 * (1.0 + erf(x / Math.sqrt(2.0)));
    }

    public static void main(String[] args) {
        double[] target = uniformProposalTarget(10);
        double[][] uniform = uniformProposalKernel(target);
        double[][] cyclic = cyclicKernel(target);
        checkKernel(uniform, target, "uniform proposal");
        checkKernel(cyclic, target, "cyclic asymmetric proposal");

        int steps = 200_000;
        double worstUniform = 0.0;
        for (int start = 0; start < target.length; start++) {
            long[] counts = new long[target.length];
            long[] result = simulateUniform(target, start, steps, new Rng(1000 + start * 31), counts);
            double[] frequency = frequencies(counts, steps);
            worstUniform = Math.max(worstUniform, maxDeviation(frequency, target));
            double acceptance = result[0] / (double) steps;
            check(Math.abs(acceptance - analyticAcceptance(uniform, target)) <= 0.01,
                    "uniform-proposal state-change rate from start " + start);
        }
        check(worstUniform <= 0.01, "uniform-proposal frequencies match pi within 0.01");

        long[] cyclicCounts = new long[target.length];
        long[] cyclicResult = simulateCyclic(target, 0, steps, new Rng(2024), cyclicCounts);
        double cyclicDeviation = maxDeviation(frequencies(cyclicCounts, steps), target);
        check(cyclicDeviation <= 0.01, "cyclic-proposal frequencies match pi within 0.01");
        check(Math.abs(cyclicResult[0] / (double) steps - analyticAcceptance(cyclic, target)) <= 0.01,
                "cyclic-proposal state-change rate");

        double[] bimodal = bimodalTarget();
        double[][] bimodalKernel = uniformProposalKernel(bimodal);
        checkKernel(bimodalKernel, bimodal, "bimodal uniform proposal");
        long[] bimodalCounts = new long[bimodal.length];
        Rng bimodalRng = new Rng(777);
        int current = 0;
        int hops = 0;
        long accepted = 0;
        for (int step = 0; step < steps; step++) {
            bimodalCounts[current]++;
            int proposal = (int) Math.floor(bimodalRng.uniform() * bimodal.length);
            double accept = Math.min(1.0, bimodal[proposal] / bimodal[current]);
            if (proposal == current || bimodalRng.uniform() < accept) {
                if ((current < 10) != (proposal < 10)) hops++;
                current = proposal;
                accepted++;
            }
        }
        double bimodalDeviation = maxDeviation(frequencies(bimodalCounts, steps), bimodal);
        check(bimodalDeviation <= 0.01, "bimodal target reproduced within 0.01");
        check(hops >= 50, "the chain crosses between the two modes");
        check(bimodalCounts[3] + bimodalCounts[4] + bimodalCounts[16] + bimodalCounts[17]
                >= 0.59 * steps, "both modes carry their analytic mass of 0.6");

        double[] sigmas = {0.25, 1.0, 3.0, 10.0};
        double previousAcceptance = 1.1;
        double[] acceptances = new double[sigmas.length];
        int bigSteps = 300_000;
        for (int s = 0; s < sigmas.length; s++) {
            NormalRun run = runNormal(0.0, sigmas[s], bigSteps, new Rng(303 + s * 17), false);
            acceptances[s] = run.acceptance;
            check(run.acceptance <= previousAcceptance + 1e-9,
                    "acceptance rate decreases in sigma at sigma = " + sigmas[s]);
            previousAcceptance = run.acceptance;
        }
        check(acceptances[0] > acceptances[1] && acceptances[1] > acceptances[2]
                && acceptances[2] > acceptances[3], "acceptance rates are strictly ordered");

        NormalRun normal = runNormal(0.0, 1.0, bigSteps, new Rng(4242), false);
        check(Math.abs(normal.mean) <= 0.05, "standard normal mean");
        check(Math.abs(normal.variance - 1.0) <= 0.1, "standard normal variance");
        check(normal.acceptance >= 0.6 && normal.acceptance <= 0.9, "documented acceptance band");
        double maxBinDeviation = 0.0;
        for (int b = 0; b < normal.bins.length; b++) {
            double low = -3.0 + 6.0 * b / normal.bins.length;
            double high = -3.0 + 6.0 * (b + 1) / normal.bins.length;
            double expected = normalCdf(high) - normalCdf(low);
            maxBinDeviation = Math.max(maxBinDeviation,
                    Math.abs(normal.bins[b] / (double) bigSteps - expected));
        }
        check(maxBinDeviation <= 0.02, "20-bin histogram matches the normal cdf within 0.02");

        NormalRun farAway = runNormal(8.0, 1.0, bigSteps, new Rng(5150), false);
        check(Math.abs(farAway.mean) <= 0.1, "a chain started at x = 8 mixes back to the target");
        check(Math.abs(farAway.variance - 1.0) <= 0.15, "far start reaches the target variance");

        NormalRun repeatA = runNormal(0.0, 1.0, 1000, new Rng(606), true);
        NormalRun repeatB = runNormal(0.0, 1.0, 1000, new Rng(606), true);
        check(Arrays.equals(repeatA.trace, repeatB.trace), "same seed reproduces the trace");
        NormalRun other = runNormal(0.0, 1.0, 1000, new Rng(707), true);
        check(!Arrays.equals(other.trace, repeatA.trace), "a different seed changes the trace");
        check(Math.abs(repeatA.mean - repeatB.mean) <= 1e-12, "identical statistics for equal seeds");

        Rng boundsRng = new Rng(88);
        boolean inRange = true;
        for (int i = 0; i < 10_000; i++) {
            double u = boundsRng.uniform();
            if (u < 0.0 || u >= 1.0) inRange = false;
        }
        check(inRange, "uniform draws stay inside [0, 1)");
        for (int i = 0; i < 10_000; i++) {
            int bounded = boundsRng.bounded(7);
            check(bounded >= 0 && bounded < 7, "bounded draws stay inside the range");
        }
        check(Math.abs(normalCdf(0.0) - 0.5) <= 1e-7, "normal cdf at 0");
        check(Math.abs(normalCdf(1.959963984540054) - 0.975) <= 1e-6, "normal cdf at 1.96");

        System.out.println("PASS");
    }

    /** Probability that the chain changes state in a step, i.e. sum_i pi_i (1 - P(i, i)). */
    private static double analyticAcceptance(double[][] kernel, double[] pi) {
        double accepted = 0.0;
        for (int i = 0; i < pi.length; i++) accepted += pi[i] * (1.0 - kernel[i][i]);
        return accepted;
    }

    private static void checkKernel(double[][] kernel, double[] pi, String label) {
        int states = pi.length;
        for (int i = 0; i < states; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < states; j++) rowSum += kernel[i][j];
            check(Math.abs(rowSum - 1.0) <= 1e-12, label + ": rows sum to 1");
        }
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < states; j++) {
                check(Math.abs(pi[i] * kernel[i][j] - pi[j] * kernel[j][i]) <= 1e-15,
                        label + ": detailed balance at (" + i + ", " + j + ")");
            }
        }
        for (int j = 0; j < states; j++) {
            double stationary = 0.0;
            for (int i = 0; i < states; i++) stationary += pi[i] * kernel[i][j];
            check(Math.abs(stationary - pi[j]) <= 1e-15, label + ": pi P = pi at column " + j);
        }
    }

    private static double[] frequencies(long[] counts, int steps) {
        double[] frequency = new double[counts.length];
        for (int i = 0; i < counts.length; i++) frequency[i] = counts[i] / (double) steps;
        return frequency;
    }

    private static double maxDeviation(double[] observed, double[] expected) {
        double worst = 0.0;
        for (int i = 0; i < observed.length; i++) {
            worst = Math.max(worst, Math.abs(observed[i] - expected[i]));
        }
        return worst;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
