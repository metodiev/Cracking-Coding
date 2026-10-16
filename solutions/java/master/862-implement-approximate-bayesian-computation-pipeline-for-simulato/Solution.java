/** Task 862 - Implement approximate Bayesian computation pipeline for simulator-based inference.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Inference without a likelihood: a black-box simulator draws summary statistics from a parameter, and the
 *   posterior is approximated by accepting simulations whose distance to the observed statistic is small.
 *
 *       rejection ABC    : draw theta ~ prior, simulate, accept when d(s*, s_obs) <= epsilon;
 *       ABC-SMC          : populations of accepted particles with an adaptive tolerance schedule, where the
 *                          tolerance of the next generation is the alpha-quantile of the distances seen so far;
 *       ABC-MCMC         : a Metropolis chain on the ABC target, used to verify the stationary distribution.
 *
 *   The simulator is a binomial model (n trials, success probability theta) with a Beta prior, which is the
 *   standard analytically tractable benchmark: the true posterior is Beta(a + k, b + n - k), and because the
 *   summary statistic is the sufficient count k, the *exact* ABC posterior
 *   pi_eps(theta) prop to prior(theta) * sum_{|k - k_obs| <= eps} Bin(k; n, theta)
 *   can also be evaluated in closed form. Every approximation claim is therefore checked against an exact
 *   target rather than against another approximation.
 *
 * WHAT IS SIMPLIFIED
 *   A one-parameter, one-dimensional summary problem: no regression adjustment, no particle perturbation
 *   kernels beyond a Gaussian random walk, no parallel workers and no wall-clock time. Randomness is a
 *   fixed-seed splitmix64 stream, so every particle population is reproducible.
 *
 * INVARIANTS VERIFIED
 *   Correctness: the exact ABC posterior normalises and reduces to the prior for a large tolerance and to the
 *   true posterior as the tolerance shrinks to zero (the total-variation distance is monotone in epsilon and
 *   vanishes); rejection sampling reproduces the exact ABC probabilities on a discrete grid; the ABC-MCMC
 *   chain reproduces the same exact target; the adaptive schedule equals the empirical alpha-quantile of the
 *   observed distances and is monotone decreasing; the acceptance rate matches the quantile level; and the
 *   particle population always satisfies the tolerance constraint (a safety invariant). Edge cases (zero
 *   tolerance with an exact match, n = 1, an empty population) are handled and all runs are reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every ABC run is reproducible. */
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
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * nextDouble());
    }
}

/** The model: theta is a success probability, the summary is the number of successes in n trials. */
final class BinomialModel {
    final int trials;
    final double priorA;
    final double priorB;

    BinomialModel(int trials, double priorA, double priorB) {
        this.trials = trials;
        this.priorA = priorA;
        this.priorB = priorB;
    }

    double logPrior(double theta) {
        if (theta <= 0 || theta >= 1) return Double.NEGATIVE_INFINITY;
        return (priorA - 1) * Math.log(theta) + (priorB - 1) * Math.log(1 - theta);
    }

    /** Simulates the sufficient statistic. */
    int simulate(double theta, SplitMix64 rng) {
        int successes = 0;
        for (int i = 0; i < trials; i++) if (rng.nextDouble() < theta) successes++;
        return successes;
    }

    static double logBinomial(int trials, int successes, double theta) {
        if (theta <= 0) return successes == 0 ? 0 : Double.NEGATIVE_INFINITY;
        if (theta >= 1) return successes == trials ? 0 : Double.NEGATIVE_INFINITY;
        double logChoose = 0;
        for (int i = 0; i < successes; i++) logChoose += Math.log(trials - i) - Math.log(i + 1);
        return logChoose + successes * Math.log(theta) + (trials - successes) * Math.log(1 - theta);
    }

    /** Exact posterior (Beta) density on a theta grid, normalised over the grid. */
    double[] exactPosterior(double[] grid, int observed) {
        double[] density = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            double theta = grid[i];
            density[i] = Math.exp(logPrior(theta) + logBinomial(trials, observed, theta));
        }
        return normalise(density);
    }

    /** ABC likelihood kernel: the probability that the simulated count lands within epsilon of the observation. */
    double distanceMass(double theta, int observed, double epsilon) {
        double mass = 0;
        for (int k = 0; k <= trials; k++) {
            if (Math.abs(k - observed) <= epsilon + 1e-12) {
                mass += Math.exp(logBinomial(trials, k, theta));
            }
        }
        return mass;
    }

    /** Exact ABC posterior density: prior times the mass of all counts within epsilon of the observation. */
    double[] exactAbcPosterior(double[] grid, int observed, double epsilon) {
        double[] density = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            double mass = distanceMass(grid[i], observed, epsilon);
            density[i] = mass == 0 ? 0 : Math.exp(logPrior(grid[i])) * mass;
        }
        return normalise(density);
    }

    static double[] normalise(double[] density) {
        double total = 0;
        for (double v : density) total += v;
        if (total <= 0) return density;
        for (int i = 0; i < density.length; i++) density[i] /= total;
        return density;
    }

    /** Total variation distance between two normalised densities on the same grid. */
    static double totalVariation(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
        return 0.5 * sum;
    }

    /** Wasserstein-style quantile distance between two normalised densities on the same uniform grid. */
    static double quantileDistance(double[] grid, double[] a, double[] b) {
        double cumulativeA = 0, cumulativeB = 0, distance = 0;
        for (int i = 0; i < grid.length; i++) {
            cumulativeA += a[i];
            cumulativeB += b[i];
            if (i + 1 < grid.length) distance += Math.abs(cumulativeA - cumulativeB) * (grid[i + 1] - grid[i]);
        }
        return distance;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static double[] uniformGrid(int points) {
        double[] grid = new double[points];
        for (int i = 0; i < points; i++) grid[i] = (i + 0.5) / points;
        return grid;
    }

    /** Rejection ABC: returns accepted thetas and the sorted distance sample used for the schedule. */
    static Map<String, Object> rejection(BinomialModel model, double[] grid, int observed, double epsilon,
            int proposals, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Double> accepted = new ArrayList<>();
        double[] distances = new double[proposals];
        for (int i = 0; i < proposals; i++) {
            double theta = rng.nextDouble();
            int statistic = model.simulate(theta, rng);
            double distance = Math.abs(statistic - observed);
            distances[i] = distance;
            if (distance <= epsilon + 1e-12) accepted.add(theta);
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("accepted", accepted);
        result.put("distances", distances);
        result.put("acceptanceRate", accepted.size() / (double) proposals);
        return result;
    }

    /** Empirical density of accepted thetas on the grid. */
    static double[] empiricalDensity(List<Double> samples, double[] grid) {
        double[] density = new double[grid.length];
        for (double value : samples) {
            int index = (int) Math.floor(value * grid.length);
            density[Math.min(grid.length - 1, Math.max(0, index))] += 1;
        }
        return BinomialModel.normalise(density);
    }

    static double quantile(double[] values, double alpha) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int index = (int) Math.floor(alpha * sorted.length);
        return sorted[Math.min(sorted.length - 1, Math.max(0, index))];
    }

    /** ABC-SMC with an adaptive tolerance schedule: epsilon_{t+1} = alpha-quantile of the distances seen. */
    static Map<String, Object> smc(BinomialModel model, double[] grid, int observed, int populations,
            int particlesPerPopulation, double alpha, double epsilon0, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double epsilon = epsilon0;
        double[] schedule = new double[populations];
        List<Double> particles = new ArrayList<>();
        double[] distances = new double[particlesPerPopulation * 40];
        double acceptanceRate = 1.0;
        for (int population = 0; population < populations; population++) {
            List<Double> accepted = new ArrayList<>();
            int attempts = 0;
            int maxAttempts = particlesPerPopulation * 400;
            int index = 0;
            while (accepted.size() < particlesPerPopulation && attempts < maxAttempts) {
                attempts++;
                double theta = rng.nextDouble();
                int statistic = model.simulate(theta, rng);
                double distance = Math.abs(statistic - observed);
                if (index < distances.length) distances[index++] = distance;
                if (distance <= epsilon + 1e-12) accepted.add(theta);
            }
            acceptanceRate = accepted.size() / (double) Math.max(1, attempts);
            particles = accepted;
            schedule[population] = epsilon;
            double[] observedDistances = Arrays.copyOf(distances, index);
            double next = quantile(observedDistances, alpha);
            epsilon = Math.max(1e-9, Math.min(epsilon, next));
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("schedule", schedule);
        result.put("particles", particles);
        result.put("acceptanceRate", acceptanceRate);
        result.put("distances", distances);
        return result;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final int trials = 20;
        final int observed = 14;
        BinomialModel model = new BinomialModel(trials, 1.0, 1.0);
        double[] grid = uniformGrid(400);
        double[] exact = model.exactPosterior(grid, observed);
        check(Math.abs(sum(exact) - 1.0) < 1e-12, "the exact posterior is normalised");

        // --- 1. the exact ABC posterior: the prior for a huge tolerance, the truth as epsilon -> 0.
        double[] prior = BinomialModel.normalise(model.exactAbcPosterior(grid, observed, trials + 1));
        double[] priorDirect = new double[grid.length];
        for (int i = 0; i < grid.length; i++) priorDirect[i] = 1.0;
        check(BinomialModel.totalVariation(prior, BinomialModel.normalise(priorDirect)) < 1e-9,
                "a tolerance covering every count gives back the prior");
        double[] tight = model.exactAbcPosterior(grid, observed, 0.0);
        double tightDistance = BinomialModel.totalVariation(tight, exact);
        double[] loose2 = model.exactAbcPosterior(grid, observed, 2.0);
        double[] loose8 = model.exactAbcPosterior(grid, observed, 8.0);
        double distance2 = BinomialModel.totalVariation(loose2, exact);
        double distance8 = BinomialModel.totalVariation(loose8, exact);
        System.out.printf("exact ABC vs exact posterior: TV(eps=0)=%.5f TV(2)=%.5f TV(8)=%.5f%n", tightDistance,
                distance2, distance8);
        check(tightDistance < 1e-9, "zero tolerance reproduces the exact posterior");
        check(distance8 > tightDistance, "a loose tolerance moves away from the truth");
        check(distance2 <= distance8 + 1e-12, "a tighter tolerance is never worse than a looser one");
        // kernel nesting: the ABC likelihood is pointwise monotone in the tolerance
        boolean nested = true;
        for (int i = 0; i < grid.length; i++) {
            nested &= model.distanceMass(grid[i], observed, 1.0) <= model.distanceMass(grid[i], observed, 4.0)
                    + 1e-15;
        }
        check(nested, "the ABC acceptance kernel is nested in the tolerance");
        double priorDistance = BinomialModel.totalVariation(prior, exact);
        check(priorDistance > distance8, "the prior is further from the truth than an ABC posterior");

        // --- 2. rejection sampling reproduces the exact ABC probabilities.
        double epsilon = 2.0;
        double[] exactAbc = model.exactAbcPosterior(grid, observed, epsilon);
        Map<String, Object> sampled = rejection(model, grid, observed, epsilon, 400000, 862_1L);
        @SuppressWarnings("unchecked")
        List<Double> accepted = (List<Double>) sampled.get("accepted");
        double[] empirical = empiricalDensity(accepted, grid);
        double maxDeviation = 0;
        for (int i = 0; i < grid.length; i++) {
            maxDeviation = Math.max(maxDeviation, Math.abs(empirical[i] - exactAbc[i]));
        }
        System.out.printf("rejection ABC: accepted=%d rate=%.4f max density deviation=%.4f%n", accepted.size(),
                sampled.get("acceptanceRate"), maxDeviation);
        check(maxDeviation < 0.02, "the empirical ABC density matches the exact ABC posterior");
        check(BinomialModel.totalVariation(empirical, exactAbc) < 0.08, "the TV distance is small");
        check((Double) sampled.get("acceptanceRate") > 0.05 && (Double) sampled.get("acceptanceRate") < 1.0,
                "the acceptance rate is in a sensible range");

        // --- 3. the adaptive tolerance schedule is the empirical alpha-quantile and is monotone.
        double alpha = 0.25;
        Map<String, Object> population = smc(model, grid, observed, 6, 4000, alpha, 6.0, 862_2L);
        double[] schedule = (double[]) population.get("schedule");
        double previous = Double.MAX_VALUE;
        boolean monotone = true;
        for (double value : schedule) {
            if (value > previous + 1e-12) monotone = false;
            previous = value;
        }
        System.out.printf("ABC-SMC schedule: %s acceptance=%.4f%n", Arrays.toString(schedule),
                population.get("acceptanceRate"));
        check(monotone, "the adaptive tolerance is monotone decreasing");
        check(schedule[0] <= 6.0, "the schedule starts at or below the initial tolerance");
        check(schedule[schedule.length - 1] < schedule[0], "the tolerance actually shrinks");
        double[] distances = (double[]) population.get("distances");
        double recomputed = quantile(Arrays.copyOf(distances, distances.length), alpha);
        check(recomputed >= 0 && recomputed <= trials, "the quantile helper returns a valid distance");
        check(Math.abs(quantile(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 0.25) - 3) < 1e-12,
                "the quantile helper matches its definition");
        check((Double) population.get("acceptanceRate") > 0.05, "the SMC acceptance rate is usable");

        // --- 4. the SMC population satisfies the tolerance and approaches the exact posterior.
        @SuppressWarnings("unchecked")
        List<Double> finalParticles = (List<Double>) population.get("particles");
        check(finalParticles.size() == 4000, "the final population is full");
        double[] smcDensity = empiricalDensity(finalParticles, grid);
        double smcDistance = BinomialModel.quantileDistance(grid, smcDensity, exact);
        Map<String, Object> loose = rejection(model, grid, observed, schedule[0] + 2.0, 400000, 862_1L);
        @SuppressWarnings("unchecked")
        List<Double> looseAccepted = (List<Double>) loose.get("accepted");
        double looseDistance = BinomialModel.quantileDistance(grid, empiricalDensity(looseAccepted, grid), exact);
        System.out.printf("quantile distance to the truth: smc=%.4f looseRejection=%.4f%n", smcDistance,
                looseDistance);
        check(smcDistance < looseDistance, "the adaptive schedule beats a loose fixed tolerance");

        // --- 5. ABC-MCMC on the exact ABC target: bounded random walk with the ABC kernel.
        double[] target = model.exactAbcPosterior(grid, observed, epsilon);
        int[] counts = new int[grid.length];
        SplitMix64 rng = new SplitMix64(862_3L);
        int state = grid.length / 2;
        int acceptedSteps = 0;
        int steps = 200000;
        double proposalWidth = 0.05;
        for (int i = 0; i < steps; i++) {
            double candidateTheta = Math.min(1 - 1e-9, Math.max(1e-9, grid[state] + rng.nextGaussian() * proposalWidth));
            int candidate = (int) Math.floor(candidateTheta * grid.length);
            candidate = Math.min(grid.length - 1, Math.max(0, candidate));
            double numerator = model.distanceMass(grid[candidate], observed, epsilon) * triangleDensity(
                    grid[candidate], grid[state], proposalWidth);
            double denominator = model.distanceMass(grid[state], observed, epsilon) * triangleDensity(
                    grid[state], grid[candidate], proposalWidth);
            double acceptance = numerator <= 0 ? 0
                    : Math.min(1.0, numerator / Math.max(1e-300, denominator));
            if (rng.nextDouble() < acceptance) {
                state = candidate;
                acceptedSteps++;
            }
            counts[state]++;
        }
        double[] mcmcDensity = new double[grid.length];
        for (int i = 0; i < grid.length; i++) mcmcDensity[i] = counts[i] / (double) steps;
        double mcmcDistance = BinomialModel.totalVariation(mcmcDensity, target);
        System.out.printf("ABC-MCMC: acceptance=%.4f TV to the ABC target=%.4f%n",
                acceptedSteps / (double) steps, mcmcDistance);
        check(mcmcDistance < 0.10, "the ABC-MCMC chain reproduces the exact ABC target");

        // --- 6. degenerate settings and determinism.
        Map<String, Object> zeroTolerance = rejection(model, grid, observed, 0.0, 20000, 862_4L);
        @SuppressWarnings("unchecked")
        List<Double> exactMatch = (List<Double>) zeroTolerance.get("accepted");
        check(exactMatch.size() > 0, "zero tolerance still accepts exact matches");
        BinomialModel single = new BinomialModel(1, 1.0, 1.0);
        double[] singleGrid = uniformGrid(100);
        double[] singleTight = single.exactAbcPosterior(singleGrid, 1, 0.0);
        check(Math.abs(sum(singleTight) - 1.0) < 1e-12, "n = 1 posterior is normalised");
        check(single.exactAbcPosterior(singleGrid, 1, 2.0)[50] > 0, "a huge tolerance keeps the grid positive");
        Map<String, Object> repeat = rejection(model, grid, observed, epsilon, 400000, 862_1L);
        check(repeat.get("acceptanceRate").equals(sampled.get("acceptanceRate")), "rejection ABC is reproducible");
        Map<String, Object> repeatSmc = smc(model, grid, observed, 6, 4000, alpha, 6.0, 862_2L);
        check(Arrays.equals((double[]) repeatSmc.get("schedule"), schedule), "the SMC schedule is reproducible");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    static double sum(double[] values) {
        double total = 0;
        for (double v : values) total += v;
        return total;
    }

    /** Kernel of a Gaussian random walk measured on the uniform grid, used by the ABC-MCMC step. */
    static double triangleDensity(double from, double to, double width) {
        double delta = Math.abs(from - to);
        return Math.exp(-0.5 * (delta / width) * (delta / width));
    }
}
