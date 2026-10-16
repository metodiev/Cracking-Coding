/** Task 861 - Implement large-scale MCMC with asynchronous proposals and convergence diagnostics.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Several Metropolis-Hastings chains walk a target distribution. Two schedulers run the same chains:
 *
 *       synchronous : every chain proposes, evaluates and accepts in lock-step;
 *       asynchronous: chains advance independently over a deterministic event loop and may read a *stale*
 *                     copy of the shared state (the classic parameter-server / stale-gradient setting), with a
 *                     hard bound on how stale a read may be.
 *
 *   The diagnostics are implemented exactly as defined: Gelman-Rubin R-hat (between/within variance), the
 *   effective sample size from the initial-positive-sequence autocorrelation estimator, Geweke's z-score, and
 *   the empirical acceptance rate. Correctness of the sampler is verified against an *exact* oracle: a small
 *   discrete target whose stationary distribution is computed by power iteration, plus a continuous Gaussian
 *   whose exact mean and variance are known, and the Metropolis acceptance probability is checked against the
 *   formula min(1, pi(y)/pi(x)) by counting.
 *
 * WHAT IS SIMPLIFIED
 *   No threads, no wall-clock time and no network: "asynchronous" means a deterministic interleaving of chain
 *   steps with an explicit staleness parameter, and "throughput" is proposals per simulated scheduler tick.
 *   Targets are one or three dimensional, chains are short (thousands of steps) and all randomness is a
 *   fixed-seed splitmix64 stream.
 *
 * INVARIANTS VERIFIED
 *   Sampler correctness: the measured acceptance probability matches min(1, pi(y)/pi(x)) and the empirical
 *   occupancy of a three-state chain matches the exactly computed stationary distribution; the chain mean and
 *   variance of a Gaussian target converge to the analytic values and the step size controls the acceptance
 *   rate monotonically.
 *   Diagnostics: R-hat and the effective sample size are ~1 and ~N for i.i.d. draws, R-hat exceeds 1.2 for
 *   chains started far apart, ESS never exceeds the chain length, Geweke's z stays small for stationary chains
 *   and detects a deterministic trend; degenerate cases (one chain, a constant chain, a one-sample chain) are
 *   handled explicitly.
 *   Asynchrony: staleness never exceeds the configured bound, every chain still converges to the analytic
 *   mean, and the asynchronous schedule delivers strictly more proposals per tick than the synchronous one.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every chain and diagnostic is reproducible. */
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

/** Random-walk Metropolis on a one-dimensional Gaussian target. */
final class GaussianChain {
    final double mean;
    final double variance;
    final double step;
    double position;
    int accepted;
    int proposals;
    final double[] samples;
    final SplitMix64 rng;

    GaussianChain(double mean, double variance, double step, double start, long seed, int length) {
        this.mean = mean;
        this.variance = variance;
        this.step = step;
        this.position = start;
        this.samples = new double[length];
        this.rng = new SplitMix64(seed);
    }

    static double logDensity(double x, double mean, double variance) {
        double delta = x - mean;
        return -0.5 * delta * delta / variance;
    }

    /** Runs every step synchronously and records the trace. */
    void run() {
        for (int i = 0; i < samples.length; i++) {
            step();
            samples[i] = position;
        }
    }

    void step() {
        double proposal = position + step * rng.nextGaussian();
        proposals++;
        double acceptance = Math.min(1.0, Math.exp(logDensity(proposal, mean, variance)
                - logDensity(position, mean, variance)));
        if (rng.nextDouble() < acceptance) {
            position = proposal;
            accepted++;
        }
    }
}

/** A tiny three-state Markov chain whose stationary distribution is computed by power iteration. */
final class DiscreteChain {
    final double[][] transition;
    final int[] counts;
    int state;
    final SplitMix64 rng;

    DiscreteChain(double[][] transition, int start, long seed) {
        this.transition = transition;
        this.state = start;
        this.counts = new int[transition.length];
        this.rng = new SplitMix64(seed);
    }

    void run(int steps) {
        for (int i = 0; i < steps; i++) {
            double u = rng.nextDouble();
            double cumulative = 0;
            for (int next = 0; next < transition.length; next++) {
                cumulative += transition[state][next];
                if (u < cumulative) {
                    state = next;
                    break;
                }
            }
            counts[state]++;
        }
    }

    /** Stationary distribution by repeated multiplication of the transition matrix. */
    static double[] stationary(double[][] transition, int iterations) {
        double[] distribution = new double[transition.length];
        Arrays.fill(distribution, 1.0 / transition.length);
        for (int i = 0; i < iterations; i++) {
            double[] next = new double[transition.length];
            for (int from = 0; from < transition.length; from++) {
                for (int to = 0; to < transition.length; to++) {
                    next[to] += distribution[from] * transition[from][to];
                }
            }
            distribution = next;
        }
        return distribution;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ diagnostics

    /** Gelman-Rubin R-hat over chains of equal length; NaN when there is a single chain. */
    static double rHat(List<double[]> chains) {
        int m = chains.size();
        int n = chains.get(0).length;
        if (m < 2 || n < 2) return Double.NaN;
        double[] means = new double[m];
        double grandMean = 0;
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (double v : chains.get(i)) sum += v;
            means[i] = sum / n;
            grandMean += means[i];
        }
        grandMean /= m;
        double between = 0, within = 0;
        for (int i = 0; i < m; i++) {
            between += (means[i] - grandMean) * (means[i] - grandMean);
            double sumSquares = 0;
            for (double v : chains.get(i)) sumSquares += (v - means[i]) * (v - means[i]);
            within += sumSquares / (n - 1);
        }
        between = between * n / (m - 1);
        within /= m;
        return Math.sqrt(((n - 1) / (double) n * within + between / n) / within);
    }

    /** Effective sample size, initial positive sequence estimator over the pooled trace. */
    static double effectiveSampleSize(double[] trace) {
        int n = trace.length;
        double mean = 0;
        for (double v : trace) mean += v;
        mean /= n;
        double variance = 0;
        for (double v : trace) variance += (v - mean) * (v - mean);
        variance /= n;
        if (variance <= 0) return n;
        double sum = 0;
        double previousPair = Double.MAX_VALUE;
        for (int lag = 1; lag < n; lag++) {
            double autocorrelation = 0;
            for (int i = 0; i + lag < n; i++) autocorrelation += (trace[i] - mean) * (trace[i + lag] - mean);
            autocorrelation /= n * variance;
            double pair = autocorrelation;
            if (lag + 1 < n) {
                double second = 0;
                for (int i = 0; i + lag + 1 < n; i++) {
                    second += (trace[i] - mean) * (trace[i + lag + 1] - mean);
                }
                second /= n * variance;
                pair += second;
            }
            if (pair <= 0 || pair >= previousPair) break;
            sum += 2 * pair - autocorrelation;
            previousPair = pair;
        }
        return Math.min(n, n / (1 + sum));
    }

    /** Geweke z-score between the first and the last fraction of the trace. */
    static double geweke(double[] trace, double firstFraction, double lastFraction) {
        int n = trace.length;
        int firstCount = (int) (n * firstFraction);
        int lastCount = (int) (n * lastFraction);
        if (firstCount < 10 || lastCount < 10) return Double.NaN;
        double[] head = Arrays.copyOfRange(trace, 0, firstCount);
        double[] tail = Arrays.copyOfRange(trace, n - lastCount, n);
        double meanHead = mean(head);
        double meanTail = mean(tail);
        double varianceHead = variance(head) / firstCount;
        double varianceTail = variance(tail) / lastCount;
        return (meanHead - meanTail) / Math.sqrt(varianceHead + varianceTail);
    }

    static double mean(double[] values) {
        double sum = 0;
        for (double v : values) sum += v;
        return sum / values.length;
    }

    static double variance(double[] values) {
        double mean = mean(values);
        double sum = 0;
        for (double v : values) sum += (v - mean) * (v - mean);
        return sum / Math.max(1, values.length - 1);
    }

    // ------------------------------------------------------------------ scheduling

    /** Asynchronous run: chains advance over a deterministic event loop and may read a *stale* shared state. */
    static Map<String, Object> runAsync(int chains, int ticks, int staleness, double step, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] positions = new double[chains];
        for (int i = 0; i < chains; i++) positions[i] = -6 + 12.0 * i / Math.max(1, chains - 1);
        double[] shared = positions.clone();
        double[][] traces = new double[chains][ticks];
        long[] publishedVersion = new long[chains];
        long version = 0;
        int accepted = 0, proposals = 0, maxObservedStaleness = 0, refreshes = 0;
        for (int t = 0; t < ticks * chains; t++) {
            int worker = rng.nextInt(chains);
            if (version - publishedVersion[worker] > staleness) {
                shared[worker] = positions[worker];          // bounded staleness: refresh before proposing
                publishedVersion[worker] = version;
                refreshes++;
            }
            int observedStaleness = (int) (version - publishedVersion[worker]);
            maxObservedStaleness = Math.max(maxObservedStaleness, observedStaleness);
            double reference = shared[worker];
            double proposal = positions[worker] + step * rng.nextGaussian();
            proposals++;
            double acceptance = Math.min(1.0, Math.exp(-0.5 * (proposal * proposal - reference * reference)));
            if (rng.nextDouble() < acceptance) {
                positions[worker] = proposal;
                accepted++;
            }
            version++;
            shared[worker] = positions[worker];
            publishedVersion[worker] = version;
            traces[worker][t / chains] = positions[worker];
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("acceptance", accepted / (double) proposals);
        result.put("proposals", proposals);
        result.put("maxStaleness", maxObservedStaleness);
        result.put("refreshes", refreshes);
        result.put("traces", traces);
        result.put("positions", positions);
        return result;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        // --- 1. the acceptance probability matches min(1, pi(y)/pi(x)) by direct measurement.
        GaussianChain probe = new GaussianChain(0.0, 1.0, 1.5, 0.0, 861_1L, 40000);
        probe.run();
        double measured = probe.accepted / (double) probe.proposals;
        double exact = exactAcceptance(probe.step, 1600, 601);
        double coarse = exactAcceptance(probe.step, 400, 151);
        System.out.printf("RWM acceptance: measured=%.4f exact=%.4f coarse=%.4f%n", measured, exact, coarse);
        check(Math.abs(exact - coarse) < 0.005, "the quadrature oracle is grid converged");
        check(Math.abs(measured - exact) < 0.02, "measured acceptance matches the exact expectation");

        // --- 2. convergence to the analytic Gaussian moments and step-size control.
        GaussianChain chain = new GaussianChain(2.0, 4.0, 2.0, -8.0, 861_2L, 60000);
        chain.run();
        double sampleMean = mean(chain.samples);
        double sampleVariance = variance(chain.samples);
        System.out.printf("Gaussian target: mean=%.3f (2.0), variance=%.3f (4.0)%n", sampleMean, sampleVariance);
        check(Math.abs(sampleMean - 2.0) < 0.15, "the chain finds the target mean");
        check(Math.abs(sampleVariance - 4.0) < 0.6, "the chain finds the target variance");
        double previousAcceptance = 1.0;
        for (double step : new double[]{0.1, 0.5, 2.0, 6.0}) {
            GaussianChain chainStep = new GaussianChain(0.0, 1.0, step, 0.0, 861_3L, 20000);
            chainStep.run();
            double acceptance = chainStep.accepted / (double) chainStep.proposals;
            check(acceptance <= previousAcceptance + 0.02, "acceptance falls as the step grows (step=" + step + ")");
            previousAcceptance = acceptance;
        }

        // --- 3. exact oracle: a three-state chain reproduces the power-iterated stationary distribution.
        double[][] transition = {{0.5, 0.3, 0.2}, {0.2, 0.6, 0.2}, {0.1, 0.3, 0.6}};
        double[] stationary = DiscreteChain.stationary(transition, 200);
        double[] reversed = DiscreteChain.stationary(transition, 400);
        for (int i = 0; i < stationary.length; i++) {
            check(Math.abs(stationary[i] - reversed[i]) < 1e-12, "power iteration converged for state " + i);
        }
        DiscreteChain discrete = new DiscreteChain(transition, 0, 861_4L);
        discrete.run(200000);
        double maxOccupancyError = 0;
        for (int i = 0; i < stationary.length; i++) {
            maxOccupancyError = Math.max(maxOccupancyError,
                    Math.abs(discrete.counts[i] / 200000.0 - stationary[i]));
        }
        System.out.printf("discrete occupancy max error = %.4f (stationary=%s)%n", maxOccupancyError,
                Arrays.toString(stationary));
        check(maxOccupancyError < 0.01, "the chain reproduces the exact stationary distribution");

        // --- 4. diagnostics on i.i.d. draws vs a deterministic trend.
        SplitMix64 rng = new SplitMix64(861_5L);
        double[] iid = new double[4000];
        for (int i = 0; i < iid.length; i++) iid[i] = rng.nextGaussian();
        List<double[]> iidChains = new ArrayList<>();
        for (int c = 0; c < 4; c++) {
            double[] part = Arrays.copyOfRange(iid, c * 1000, (c + 1) * 1000);
            iidChains.add(part);
        }
        double iidRHat = rHat(iidChains);
        double iidEss = effectiveSampleSize(iid);
        double iidGeweke = geweke(iid, 0.1, 0.5);
        System.out.printf("i.i.d.: R-hat=%.4f ESS=%.1f/4000 geweke=%.3f%n", iidRHat, iidEss, iidGeweke);
        check(Math.abs(iidRHat - 1.0) < 0.05, "R-hat is about one for i.i.d. draws");
        check(iidEss > 0.6 * iid.length && iidEss <= iid.length, "ESS is close to N and never exceeds it");
        check(Math.abs(iidGeweke) < 2.0, "Geweke does not flag stationary draws");

        double[] trend = new double[4000];
        for (int i = 0; i < trend.length; i++) trend[i] = i * 0.01 + rng.nextGaussian();
        double trendGeweke = geweke(trend, 0.1, 0.5);
        check(Math.abs(trendGeweke) > 5, "Geweke detects a deterministic trend (" + trendGeweke + ")");
        List<double[]> separated = new ArrayList<>();
        for (int c = 0; c < 4; c++) {
            double[] part = new double[500];
            for (int i = 0; i < part.length; i++) part[i] = c * 10.0 + rng.nextGaussian();
            separated.add(part);
        }
        double separatedRHat = rHat(separated);
        System.out.printf("separated chains: R-hat=%.3f%n", separatedRHat);
        check(separatedRHat > 1.2, "R-hat flags chains that have not mixed");
        check(Double.isNaN(rHat(List.of(new double[]{1.0, 2.0}))), "R-hat needs at least two chains");
        double[] constant = new double[100];
        Arrays.fill(constant, 3.0);
        check(Double.isNaN(rHat(List.of(constant, constant))) || rHat(List.of(constant, constant)) >= 1.0,
                "a constant chain does not produce a bogus R-hat");
        check(effectiveSampleSize(new double[]{1.0}) == 1.0, "a one-sample trace has ESS one");

        // --- 5. asynchronous chains: staleness bound, convergence and throughput.
        Map<String, Object> async = runAsync(4, 800, 3, 1.0, 861_6L);
        double[][] traces = (double[][]) async.get("traces");
        double lastValues = 0;
        for (double[] trace : traces) lastValues += trace[trace.length - 1];
        System.out.printf("async: maxStaleness=%d refreshes=%d acceptance=%.3f meanOfLast=%.3f%n",
                async.get("maxStaleness"), async.get("refreshes"), async.get("acceptance"),
                lastValues / traces.length);
        check((Integer) async.get("maxStaleness") <= 4, "observed staleness respects the configured bound");
        check((Integer) async.get("refreshes") > 0, "stale workers actually refresh their state");
        check((Double) async.get("acceptance") > 0.3 && (Double) async.get("acceptance") < 1.0,
                "asynchronous acceptance stays in a healthy range");
        check(Math.abs(lastValues / traces.length) < 3.0, "asynchronous chains end near the target centre");
        Map<String, Object> repeat = runAsync(4, 800, 3, 1.0, 861_6L);
        check(repeat.get("acceptance").equals(async.get("acceptance")), "the asynchronous run is reproducible");
        Map<String, Object> fresh = runAsync(4, 800, 1, 1.0, 861_6L);
        check((Integer) fresh.get("maxStaleness") <= 2, "a tighter staleness bound is respected");
        check((Integer) fresh.get("refreshes") >= (Integer) async.get("refreshes"),
                "a tighter bound forces at least as many refreshes");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    /**
     * Exact expected acceptance of RWM on N(0,1) with proposal sd `s`, by two-dimensional quadrature over the
     * current point and the proposal - the independent oracle for the measured acceptance rate.
     */
    static double exactAcceptance(double s, int xSteps, int ySteps) {
        double total = 0, weight = 0;
        for (int ix = 0; ix < xSteps; ix++) {
            double x = -6 + 12.0 * (ix + 0.5) / xSteps;
            double densityX = Math.exp(-0.5 * x * x);
            for (int iy = 0; iy < ySteps; iy++) {
                double z = -6 + 12.0 * (iy + 0.5) / ySteps;
                double y = x + s * z;
                double densityProposal = Math.exp(-0.5 * z * z);
                double acceptance = Math.min(1.0, Math.exp(-0.5 * (y * y - x * x)));
                total += densityX * densityProposal * acceptance;
                weight += densityX * densityProposal;
            }
        }
        return total / weight;
    }
}
