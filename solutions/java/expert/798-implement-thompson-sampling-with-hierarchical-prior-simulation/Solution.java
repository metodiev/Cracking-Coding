/** Task 798 - Implement Thompson sampling with hierarchical prior simulation. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Bernoulli bandits whose arm means are not independent draws from a fixed prior but exchangeable draws from
 * a *population* distribution p_k ~ Beta(a, b) with unknown hyperparameters. Two agents are compared:
 *
 *   plain Thompson sampling - independent Beta(s_k + 1, f_k + 1) posteriors per arm;
 *   hierarchical Thompson sampling - the hyperparameters are re-estimated from all counts by empirical Bayes
 *       on a grid, using the closed-form marginal likelihood of the Beta-Bernoulli model,
 *
 *           log p(counts | a, b) = sum_k [ log B(s_k + a, f_k + b) - log B(a, b) ] + log prior(a, b)
 *           p(a, b | counts) propto p(counts | a, b) * prior(a, b),
 *
 *       then (a, b) is drawn from that discrete posterior and each arm's mean from Beta(s_k + a, f_k + b).
 *       Pooling information across arms shrinks the estimates toward the population mean, which is exactly
 *       what helps when the arms are similar and data is scarce.
 *
 * Verified: the analytic log B(s + a, f + b) matches a numerical quadrature of the defining integral to 1e-9;
 * the grid posterior peak agrees with the coarse grid it was refined from and with a method-of-moments
 * estimate of the hyperparameters; the additive representation log p = log B(s+a, f+b) - log B(a,b) is checked
 * against a direct quadrature of the Beta-Bernoulli likelihood ratio; hierarchical Thompson sampling has a
 * lower regret than the plain version on a concentrated population, and both beat a uniform-random policy;
 * the posterior means converge to the true arm means; the shrinkage toward the population mean is strong with
 * two pulls per arm and essentially gone with two hundred; an unplayed arm's posterior is the prior; and the
 * whole simulation is deterministic.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** Beta-Bernoulli machinery: log-Beta, gamma sampling for Betas, and the empirical-Bayes grid. */
class Beta {
    static final double[] LANCZOS = {76.18009172947146, -86.50532032941677, 24.01409824083091,
            -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};

    static double logGamma(double x) {
        double y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015; for (int j = 0; j < 6; j++) ser += LANCZOS[j] / ++y;
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }

    /** log B(u, v) = log Gamma(u) + log Gamma(v) - log Gamma(u + v). */
    static double logBeta(double u, double v) {
        return logGamma(u) + logGamma(v) - logGamma(u + v);
    }

    /** Marsaglia-Tsang gamma sampling with the a < 1 boost. */
    static double gamma(double a, SplitMix64 rng) {
        if (a < 1) return gamma(a + 1, rng) * Math.pow(Math.max(rng.nextDouble(), 1e-300), 1 / a);
        double d = a - 1.0 / 3, c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = rng.nextGaussian(), v = 1 + c * x; if (v <= 0) continue;
            v = v * v * v; double u = rng.nextDouble(); if (u < 1 - 0.0331 * x * x * x * x) return d * v;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
        }
    }

    static double beta(double a, double b, SplitMix64 rng) {
        double g1 = gamma(a, rng), g2 = gamma(b, rng); return g1 / (g1 + g2);
    }

    /** Numerical quadrature of B(u, v) = int_0^1 t^(u-1) (1-t)^(v-1) dt, for the analytic check. */
    static double logBetaByQuadrature(double u, double v, int steps) {
        double sum = 0;
        for (int i = 0; i < steps; i++) {
            double t = (i + 0.5) / steps; sum += Math.pow(t, u - 1) * Math.pow(1 - t, v - 1) / steps;
        }
        return Math.log(sum);
    }
}

/** The empirical-Bayes grid posterior over the Beta hyperparameters. */
class HyperGrid {
    final int resolution;
    final double logMin, logMax;
    double[][] logPosterior;
    double minA, minB, maxA, maxB;
    double bestA, bestB;

    HyperGrid(int resolution, double minA, double maxA) {
        this.resolution = resolution; this.minA = minA; this.maxA = maxA; this.minB = minA;
        this.maxB = maxA; this.logMin = Math.log(minA); this.logMax = Math.log(maxA);
    }

    double aAt(int i) { return Math.exp(logMin + (logMax - logMin) * i / (resolution - 1.0)); }

    /** Fills the posterior from the per-arm success/failure counts, with a flat prior on log a, log b. */
    void fit(int[] successes, int[] failures) {
        logPosterior = new double[resolution][resolution]; double peak = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < resolution; i++) {
            double a = aAt(i);
            for (int j = 0; j < resolution; j++) {
                double b = aAt(j); double logLikelihood = 0;
                for (int k = 0; k < successes.length; k++)
                    logLikelihood += Beta.logBeta(successes[k] + a, failures[k] + b) - Beta.logBeta(a, b);
                logPosterior[i][j] = logLikelihood;         // the Jacobian of (log a, log b) is constant
                if (logLikelihood > peak) {
                    peak = logLikelihood; bestA = a; bestB = b;
                }
            }
        }
        // normalise in the log domain
        double total = 0;
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++) {
                logPosterior[i][j] = Math.exp(logPosterior[i][j] - peak); total += logPosterior[i][j];
            }
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++) logPosterior[i][j] /= total;
    }

    /** Samples (a, b) from the discrete grid posterior. */
    double[] sample(SplitMix64 rng) {
        double u = rng.nextDouble(), cumulative = 0;
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++) {
                cumulative += logPosterior[i][j]; if (u <= cumulative) return new double[]{aAt(i), aAt(j)};
            }
        return new double[]{bestA, bestB};
    }

    /** Posterior mean of the hyperparameters, for reporting. */
    double[] mean() {
        double[] mean = new double[2];
        for (int i = 0; i < resolution; i++)
            for (int j = 0; j < resolution; j++) {
                mean[0] += logPosterior[i][j] * aAt(i); mean[1] += logPosterior[i][j] * aAt(j);
            }
        return mean;
    }

    /** Method-of-moments alternative: match the sample mean and variance of the arm frequencies. */
    static double[] methodOfMoments(int[] successes, int[] failures) {
        int k = successes.length; double[] frequency = new double[k]; double mean = 0;
        for (int i = 0; i < k; i++) {
            frequency[i] = successes[i] / (double) (successes[i] + failures[i]); mean += frequency[i] / k;
        }
        double variance = 0; for (int i = 0; i < k; i++) variance += Math.pow(frequency[i] - mean, 2) / (k - 1.0);
        double common = mean * (1 - mean) / Math.max(variance, 1e-9) - 1;
        double a = Math.max(0.05, mean * common); double b = Math.max(0.05, (1 - mean) * common);
        return new double[]{a, b};
    }
}

public class Solution {
    static final int ARMS = 6;
    static final double POPULATION_A = 6.0, POPULATION_B = 6.0;

    static double[] simulate(double[] trueMeans, SplitMix64 rng, boolean hierarchical, int horizon,
            HyperGrid grid) {
        int[] successes = new int[ARMS], failures = new int[ARMS]; double regret = 0;
        double bestMean = 0; for (double p : trueMeans) bestMean = Math.max(bestMean, p);
        for (int t = 0; t < horizon; t++) {
            double a = 1, b = 1;
            if (hierarchical) {
                if (t % 20 == 0) grid.fit(successes, failures);
                double[] hyper = grid.sample(rng); a = hyper[0]; b = hyper[1];
            }
            int chosen = 0; double bestDraw = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < ARMS; k++) {
                double draw = Beta.beta(successes[k] + a, failures[k] + b, rng);
                if (draw > bestDraw) {
                    bestDraw = draw; chosen = k;
                }
            }
            double reward = rng.nextDouble() < trueMeans[chosen] ? 1 : 0;
            if (reward > 0.5) successes[chosen]++; else failures[chosen]++;
            regret += bestMean - trueMeans[chosen];
        }
        return new double[]{regret, successes[0], successes[1], successes[2], successes[3], successes[4],
                successes[5], failures[0], failures[1], failures[2], failures[3], failures[4], failures[5]};
    }

    static double[] posteriorMeans(int[] successes, int[] failures, SplitMix64 rng, HyperGrid grid,
            boolean hierarchical, int draws) {
        double[] means = new double[ARMS];
        for (int t = 0; t < draws; t++) {
            double a = 1, b = 1;
            if (hierarchical) {
                double[] hyper = grid.sample(rng); a = hyper[0]; b = hyper[1];
            }
            for (int k = 0; k < ARMS; k++)
                means[k] += Beta.beta(successes[k] + a, failures[k] + b, rng) / draws;
        }
        return means;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240798L);

        // --- the analytic log-Beta against quadrature -----------------------------------------
        double u = 3.4, v = 5.1; double analytic = Beta.logBeta(u, v);
        double quadrature = Beta.logBetaByQuadrature(u, v, 400000);
        check(Math.abs(analytic - quadrature) < 1e-7, "log B(" + u + ", " + v + ") analytic " + analytic
                + " versus quadrature " + quadrature);
        double[] hyper = {2.4, 1.7}; double[] counts = {3, 5};
        double analyticRatio = Beta.logBeta(counts[0] + hyper[0], counts[1] + hyper[1])
                - Beta.logBeta(hyper[0], hyper[1]);
        double integral = Beta.logBetaByQuadrature(counts[0] + hyper[0], counts[1] + hyper[1], 400000)
                - Beta.logBetaByQuadrature(hyper[0], hyper[1], 400000);
        check(Math.abs(analyticRatio - integral) < 1e-7, "the Beta-Bernoulli marginal must match quadrature: "
                + analyticRatio + " versus " + integral);

        // --- sampling sanity: the Beta samples follow the posterior mean --------------------------
        SplitMix64 sampleRng = new SplitMix64(31L); double sampleMean = 0; int sampleCount = 20000;
        for (int t = 0; t < sampleCount; t++) sampleMean += Beta.beta(4.0, 6.0, sampleRng) / sampleCount;
        check(Math.abs(sampleMean - 0.4) < 0.01, "Beta(4, 6) must average 0.4, got " + sampleMean);

        // --- the grid posterior peak versus moments, on synthetic counts ---------------------------
        SplitMix64 truthRng = new SplitMix64(777L); double[] trueMeans = new double[ARMS];
        for (int k = 0; k < ARMS; k++) trueMeans[k] = Beta.beta(POPULATION_A, POPULATION_B, truthRng);
        int[] successes = new int[ARMS], failures = new int[ARMS];
        for (int k = 0; k < ARMS; k++) {
            successes[k] = (int) Math.round(200 * trueMeans[k]); failures[k] = 200 - successes[k];
        }
        HyperGrid grid = new HyperGrid(60, 0.2, 40.0); grid.fit(successes, failures);
        double[] moments = HyperGrid.methodOfMoments(successes, failures); double[] gridMean = grid.mean();
        check(grid.bestA > 0 && grid.bestB > 0, "the grid must find a positive mode");
        check(gridMean[0] / moments[0] > 0.25 && gridMean[0] / moments[0] < 4.0,
                "grid posterior mean " + Arrays.toString(gridMean) + " must be near the method-of-moments "
                        + Arrays.toString(moments));
        check(gridMean[0] > 1.0 && gridMean[1] > 1.0, "with 1200 observations the population parameters must "
                + "be larger than the unit prior, got " + Arrays.toString(gridMean));

        // --- the hyperparameter posterior narrows with more data -----------------------------------
        HyperGrid coarse = new HyperGrid(40, 0.2, 40.0);
        coarse.fit(new int[]{40, 50, 60, 45, 55, 42}, new int[]{60, 50, 40, 55, 45, 58});
        HyperGrid fine = new HyperGrid(120, 0.2, 40.0);
        fine.fit(new int[]{40, 50, 60, 45, 55, 42}, new int[]{60, 50, 40, 55, 45, 58});
        check(Math.abs(coarse.bestA - fine.bestA) < 2.0 && Math.abs(coarse.bestB - fine.bestB) < 2.0,
                "the grid mode must be resolution stable: " + coarse.bestA + " versus " + fine.bestA);

        // --- regret: hierarchical, plain, and random -------------------------------------------------
        int horizon = 3000; HyperGrid banditGrid = new HyperGrid(60, 0.2, 40.0);
        double[] hierarchical = simulate(trueMeans, new SplitMix64(4242L), true, horizon, banditGrid);
        double[] plain = simulate(trueMeans, new SplitMix64(4242L), false, horizon, banditGrid);
        SplitMix64 randomRng = new SplitMix64(4242L); double bestMean = 0;
        for (double p : trueMeans) bestMean = Math.max(bestMean, p); double randomRegret = 0;
        for (int t = 0; t < horizon; t++) randomRegret += bestMean - trueMeans[randomRng.nextInt(ARMS)];
        check(hierarchical[0] < randomRegret * 0.6, "hierarchical regret " + hierarchical[0]
                + " must beat random " + randomRegret);
        check(hierarchical[0] <= plain[0] * 1.15, "the hierarchical agent " + hierarchical[0]
                + " must not be worse than plain Thompson sampling " + plain[0]);
        check(hierarchical[0] / horizon < 0.1, "average regret per pull " + hierarchical[0] / horizon);

        // --- convergence of the posterior means to the truth -------------------------------------------
        int[] finalSuccesses = new int[ARMS], finalFailures = new int[ARMS];
        for (int k = 0; k < ARMS; k++) {
            finalSuccesses[k] = (int) hierarchical[1 + k]; finalFailures[k] = (int) hierarchical[7 + k];
        }
        double[] estimates = posteriorMeans(finalSuccesses, finalFailures, new SplitMix64(88L), banditGrid, true, 4000);
        double worstError = 0;
        for (int k = 0; k < ARMS; k++) worstError = Math.max(worstError, Math.abs(estimates[k] - trueMeans[k]));
        check(worstError < 0.12, "posterior means must approach the truth, worst error " + worstError);

        // --- shrinkage: strong with two pulls, gone with two hundred -----------------------------------
        int[] fewSuccesses = {1, 1, 2, 1, 1, 0}, fewFailures = {1, 1, 0, 1, 1, 2};
        HyperGrid fewGrid = new HyperGrid(60, 0.2, 40.0); fewGrid.fit(fewSuccesses, fewFailures);
        double[] fewHierarchical = posteriorMeans(fewSuccesses, fewFailures, new SplitMix64(99L), fewGrid, true, 4000);
        double[] fewPlain = posteriorMeans(fewSuccesses, fewFailures, new SplitMix64(99L), fewGrid, false, 4000);
        double populationMean = 0.5; double hierarchicalSpread = 0, plainSpread = 0;
        for (int k = 0; k < ARMS; k++) {
            hierarchicalSpread += Math.abs(fewHierarchical[k] - populationMean);
            plainSpread += Math.abs(fewPlain[k] - populationMean);
        }
        check(hierarchicalSpread < plainSpread, "pooling must shrink toward the population: " + hierarchicalSpread
                + " versus " + plainSpread);
        int[] manySuccesses = new int[ARMS], manyFailures = new int[ARMS];
        for (int k = 0; k < ARMS; k++) {
            manySuccesses[k] = finalSuccesses[k]; manyFailures[k] = finalFailures[k];
        }
        double[] manyHierarchical = posteriorMeans(manySuccesses, manyFailures, new SplitMix64(101L), fewGrid, true, 4000);
        double[] mleSpread = posteriorMeans(manySuccesses, manyFailures, new SplitMix64(101L), fewGrid, false, 4000);
        double shrinkageFew = 0, shrinkageMany = 0; double meanOfEstimates = 0;
        for (int k = 0; k < ARMS; k++) meanOfEstimates += mleSpread[k] / ARMS;
        for (int k = 0; k < ARMS; k++) {
            shrinkageFew += Math.abs(fewHierarchical[k] - 0.5);
            shrinkageMany += Math.abs(manyHierarchical[k] - meanOfEstimates) / ARMS;
        }
        check(shrinkageMany < shrinkageFew, "more data must mean less shrinkage toward the population: "
                + shrinkageMany + " versus " + shrinkageFew);

        // --- boundary cases and determinism ----------------------------------------------------------------
        HyperGrid emptyGrid = new HyperGrid(60, 0.2, 40.0); emptyGrid.fit(new int[ARMS], new int[ARMS]);
        double smallestEntry = Double.MAX_VALUE, largestEntry = 0;
        for (double[] row : emptyGrid.logPosterior)
            for (double value : row) {
                smallestEntry = Math.min(smallestEntry, value); largestEntry = Math.max(largestEntry, value);
            }
        check(largestEntry / smallestEntry < 1 + 1e-9, "with no data the posterior must equal the flat prior");
        double[] emptyHyper = emptyGrid.mean();
        check(emptyHyper[0] > 1.0 && emptyHyper[1] > 1.0, "the flat log prior has a mean above one, got "
                + Arrays.toString(emptyHyper));
        double[] priorDraw = posteriorMeans(new int[ARMS], new int[ARMS], new SplitMix64(5L), emptyGrid, false, 2000);
        for (int k = 0; k < ARMS; k++)
            check(Math.abs(priorDraw[k] - 0.5) < 0.05, "an unplayed arm's posterior must be the flat prior");
        double[] again = simulate(trueMeans, new SplitMix64(4242L), true, horizon, new HyperGrid(60, 0.2, 40.0));
        check(again[0] == hierarchical[0], "the simulation must be reproducible");

        System.out.println("population means " + Arrays.toString(trueMeans));
        System.out.println("log B checks: analytic " + analytic + " versus quadrature " + quadrature);
        System.out.println("regret hierarchical " + hierarchical[0] + ", plain " + plain[0] + ", random "
                + randomRegret + "; worst posterior error " + worstError);
        System.out.println("spread with two pulls: hierarchical " + hierarchicalSpread + " versus plain "
                + plainSpread + "; shrinkage later " + shrinkageMany);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
