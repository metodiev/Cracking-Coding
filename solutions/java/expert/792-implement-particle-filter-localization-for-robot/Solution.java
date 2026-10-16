/** Task 792 - Implement particle filter localization for robot. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A robot moves along a line of known beacons and measures its distance to each of them. The belief is
 * represented by N weighted particles:
 *
 *     motion:      x_i <- x_i + u + sigma_u N(0,1)
 *     weight:      w_i <- prod_b N(z_b - |x_i - b|; sigma_z)      then normalise
 *     resample:    systematic resampling whenever the effective sample size falls below N/2
 *
 * Systematic resampling draws a single uniform u in [0, 1/N) and takes the particles at the cumulative
 * quantiles u + k/N, which is the standard low-variance alternative to multinomial resampling, and it makes
 * the weights exactly uniform afterwards.
 *
 * The independent oracle is a histogram (grid) Bayes filter on the *same* one-dimensional state space with
 * the *same* motion and sensor models: propagate the grid density by convolving it with the Gaussian motion
 * kernel, multiply by the likelihood, renormalise. That recursion is exact (up to grid discretisation) and
 * therefore gives the posterior mean and standard deviation the particle cloud must reproduce.
 *
 * Simplified: one-dimensional pose (a line segment) rather than a plane, known correspondences and no
 * data association, no beam model (a Gaussian range model), and a fixed particle count with the simplest
 * ESS-based resampling trigger. The filter itself, the systematic resampler and the grid oracle are real.
 *
 * Verified: the particle posterior tracks the exact grid posterior in mean and standard deviation; the
 * estimate converges to the true trajectory and stays inside a small error band; global localisation from a
 * uniform cloud also converges (a boundary case that rules out a filter that merely follows the truth);
 * weights stay normalised to 1e-12 and the particle count is conserved; systematic resampling equalises the
 * weights and drives the effective sample size back to N while the pre-resampling ESS falls; the resampled
 * empirical mean reproduces the weighted mean within Monte-Carlo error; a huge sensor noise leaves the prior
 * almost untouched while a tiny one collapses the belief; a single-particle and a no-beacon run still work;
 * and repeated runs with the same seed are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** A particle cloud on a line, with systematic resampling. */
class ParticleFilter {
    final double[] beacons;
    final double sigmaU, sigmaZ;
    final double length;
    double[] x, w;
    final SplitMix64 rng;
    double preResampleEss = 0;

    ParticleFilter(int count, double[] beacons, double length, double sigmaU, double sigmaZ, long seed) {
        this.beacons = beacons; this.length = length; this.sigmaU = sigmaU; this.sigmaZ = sigmaZ;
        this.rng = new SplitMix64(seed); x = new double[count]; w = new double[count]; Arrays.fill(w, 1.0 / count);
    }

    void uniformInit() {
        for (int i = 0; i < x.length; i++) x[i] = rng.nextDouble() * length; Arrays.fill(w, 1.0 / x.length);
    }

    void gaussianInit(double mean, double sd) {
        for (int i = 0; i < x.length; i++) x[i] = mean + sd * rng.nextGaussian(); Arrays.fill(w, 1.0 / x.length);
    }

    void predict(double u) {
        for (int i = 0; i < x.length; i++) x[i] += u + sigmaU * rng.nextGaussian();
    }

    /** Multiplies in the Gaussian range likelihood of every beacon and renormalises. */
    void update(double[] ranges) {
        double total = 0; double twoSigmaSquared = 2 * sigmaZ * sigmaZ;
        for (int i = 0; i < x.length; i++) {
            double likelihood = 1;
            for (int b = 0; b < beacons.length; b++) {
                double residual = ranges[b] - Math.abs(x[i] - beacons[b]);
                likelihood *= Math.exp(-residual * residual / twoSigmaSquared);
            }
            w[i] *= likelihood + 1e-300; total += w[i];
        }
        for (int i = 0; i < x.length; i++) w[i] /= total;
    }

    double effectiveSampleSize() {
        double sum = 0; for (double v : w) sum += v * v; return 1.0 / sum;
    }

    double mean() {
        double sum = 0; for (int i = 0; i < x.length; i++) sum += w[i] * x[i]; return sum;
    }

    double standardDeviation() {
        double m = mean(), sum = 0; for (int i = 0; i < x.length; i++) sum += w[i] * (x[i] - m) * (x[i] - m);
        return Math.sqrt(sum);
    }

    /** Systematic resampling; returns the number of draws per particle slot. */
    int[] resample() {
        preResampleEss = effectiveSampleSize(); double[] next = new double[x.length];
        int[] counts = new int[x.length]; double step = 1.0 / x.length;
        double u = rng.nextDouble() * step; double cumulative = w[0]; int index = 0;
        for (int i = 0; i < x.length; i++) {
            while (u > cumulative && index < x.length - 1) {
                index++; cumulative += w[index];
            }
            next[i] = x[index]; counts[index]++; u += step;
        }
        x = next; Arrays.fill(w, 1.0 / x.length); return counts;
    }

    void step(double u, double[] ranges) {
        predict(u); update(ranges); if (effectiveSampleSize() < x.length / 2.0) resample();
    }
}

/** Exact histogram Bayes filter on the same state space: the oracle. */
class GridBayes {
    final double[] beacons;
    final double sigmaU, sigmaZ;
    final double[] grid, density;

    GridBayes(int cells, double length, double[] beacons, double sigmaU, double sigmaZ) {
        this.beacons = beacons; this.sigmaU = sigmaU; this.sigmaZ = sigmaZ; grid = new double[cells];
        density = new double[cells];
        for (int i = 0; i < cells; i++) {
            grid[i] = length * i / (cells - 1.0); density[i] = 1.0 / cells;
        }
        normalise();
    }

    void normalise() {
        double sum = 0; for (double v : density) sum += v;
        for (int i = 0; i < density.length; i++) density[i] /= sum;
    }

    /** Same prior as the particle cloud: a Gaussian truncation to the segment. */
    void gaussianInit(double mean, double sd) {
        for (int i = 0; i < density.length; i++) {
            double d = grid[i] - mean; density[i] = Math.exp(-d * d / (2 * sd * sd));
        }
        normalise();
    }

    /** Convolution with the Gaussian motion kernel, truncated at four standard deviations. */
    void predict(double u) {
        double spacing = grid[1] - grid[0]; int radius = Math.max(1, (int) Math.ceil(4 * sigmaU / spacing));
        double[] next = new double[density.length];
        for (int i = 0; i < density.length; i++) {
            if (density[i] == 0) continue;
            for (int d = -radius; d <= radius; d++) {
                int j = (int) Math.round((grid[i] + u - grid[0]) / spacing) + d;
                if (j < 0 || j >= density.length) continue; double offset = grid[i] + u - grid[j];
                next[j] += density[i] * Math.exp(-offset * offset / (2 * sigmaU * sigmaU));
            }
        }
        System.arraycopy(next, 0, density, 0, density.length); normalise();
    }

    void update(double[] ranges) {
        for (int i = 0; i < density.length; i++) {
            double likelihood = 1;
            for (int b = 0; b < beacons.length; b++) {
                double residual = ranges[b] - Math.abs(grid[i] - beacons[b]);
                likelihood *= Math.exp(-residual * residual / (2 * sigmaZ * sigmaZ));
            }
            density[i] *= likelihood + 1e-300;
        }
        normalise();
    }

    void step(double u, double[] ranges) {
        predict(u); update(ranges);
    }

    double mean() {
        double sum = 0; for (int i = 0; i < density.length; i++) sum += density[i] * grid[i]; return sum;
    }

    double standardDeviation() {
        double m = mean(), sum = 0;
        for (int i = 0; i < density.length; i++) sum += density[i] * (grid[i] - m) * (grid[i] - m);
        return Math.sqrt(sum);
    }
}

public class Solution {
    static final double LENGTH = 20.0;
    static final double[] BEACONS = {2.0, 7.0, 12.0, 17.0};
    static final double SIGMA_U = 0.3, SIGMA_Z = 0.3, STEP = 0.5;

    static double[] trueRanges(double position, SplitMix64 rng, boolean noisy) {
        double[] z = new double[BEACONS.length];
        for (int b = 0; b < BEACONS.length; b++)
            z[b] = Math.abs(position - BEACONS[b]) + (noisy ? SIGMA_Z * rng.nextGaussian() : 0);
        return z;
    }

    public static void main(String[] args) {
        // --- the particle posterior against the exact grid posterior ------------------------
        SplitMix64 rng = new SplitMix64(20240792L);
        ParticleFilter pf = new ParticleFilter(500, BEACONS, LENGTH, SIGMA_U, SIGMA_Z, 777L);
        GridBayes grid = new GridBayes(801, LENGTH, BEACONS, SIGMA_U, SIGMA_Z);
        pf.gaussianInit(3.0, 0.5); grid.gaussianInit(3.0, 0.5);
        double position = 3.0; double worstMeanGap = 0, worstStdGap = 0;
        for (int step = 0; step < 25; step++) {
            position += STEP;                       // the robot moves first, then measures
            double[] ranges = trueRanges(position, rng, true); pf.step(STEP, ranges); grid.step(STEP, ranges);
            check(Math.abs(sum(pf.w) - 1.0) < 1e-12, "weights must stay normalised at step " + step);
            worstMeanGap = Math.max(worstMeanGap, Math.abs(pf.mean() - grid.mean()));
            worstStdGap = Math.max(worstStdGap, Math.abs(pf.standardDeviation() - grid.standardDeviation())
                    / Math.max(grid.standardDeviation(), 1e-3));
        }
        check(worstMeanGap < 0.15, "particle and grid means must agree, worst gap " + worstMeanGap);
        check(worstStdGap < 0.4, "particle and grid spreads must agree, worst relative gap " + worstStdGap);
        check(Math.abs(pf.mean() - position) < 0.25, "the estimate must track the truth, error "
                + Math.abs(pf.mean() - position));
        check(pf.standardDeviation() < 0.5, "the belief must be tight, sd " + pf.standardDeviation());

        // --- global localisation from a uniform cloud ----------------------------------------
        ParticleFilter global = new ParticleFilter(500, BEACONS, LENGTH, SIGMA_U, SIGMA_Z, 4242L);
        global.uniformInit(); double globalPosition = 15.0; boolean spreadEarly = global.standardDeviation() > 4.0;
        check(spreadEarly, "a uniform cloud starts with a large spread, sd "
                + global.standardDeviation());
        for (int step = 0; step < 30; step++) {
            globalPosition += STEP; double[] ranges = trueRanges(globalPosition, rng, true);
            global.step(STEP, ranges);
        }
        check(Math.abs(global.mean() - globalPosition) < 0.3, "global localisation error "
                + Math.abs(global.mean() - globalPosition));
        check(global.standardDeviation() < 0.6, "the global run must also concentrate, sd "
                + global.standardDeviation());

        // --- resampling: ESS, uniform weights, conserved count, first moment ------------------
        ParticleFilter resampler = new ParticleFilter(200, BEACONS, LENGTH, SIGMA_U, SIGMA_Z, 99L);
        resampler.gaussianInit(10.0, 2.0); resampler.predict(0.0);
        double[] skewRanges = trueRanges(10.0, rng, true); resampler.update(skewRanges);
        double essBefore = resampler.effectiveSampleSize(); double weightedMean = resampler.mean();
        int[] counts = resampler.resample(); double essAfter = resampler.effectiveSampleSize();
        int total = 0; for (int c : counts) total += c;
        check(total == 200, "resampling must conserve the particle count, got " + total);
        check(Math.abs(essAfter - 200) < 1e-9, "weights must be uniform after resampling, ESS " + essAfter);
        check(essBefore <= 200 + 1e-9, "the ESS can never exceed the particle count");
        check(Math.abs(resampler.mean() - weightedMean) < 0.5,
                "the resampled mean must reproduce the weighted mean, " + resampler.mean() + " vs "
                        + weightedMean);
        check(resampler.preResampleEss == essBefore, "the recorded pre-resampling ESS must match");

        // --- noise levels drive concentration --------------------------------------------------
        SplitMix64 quiet = new SplitMix64(31L);
        ParticleFilter tight = new ParticleFilter(400, BEACONS, LENGTH, SIGMA_U, 0.02, 515L);
        ParticleFilter loose = new ParticleFilter(400, BEACONS, LENGTH, SIGMA_U, 50.0, 515L);
        tight.gaussianInit(8.0, 1.0); loose.gaussianInit(8.0, 1.0); double truePosition = 8.0;
        for (int step = 0; step < 4; step++) {
            truePosition += 0.0; double[] ranges = trueRanges(truePosition, quiet, true);
            tight.step(0.0, ranges); loose.step(0.0, ranges);
        }
        check(tight.standardDeviation() < 0.2, "a tiny sensor noise must collapse the belief, sd "
                + tight.standardDeviation());
        check(loose.standardDeviation() > 0.8, "a huge sensor noise must leave the prior intact, sd "
                + loose.standardDeviation());
        check(loose.standardDeviation() > 4 * tight.standardDeviation(), "the two regimes must differ clearly");

        // --- boundary cases and determinism ----------------------------------------------------
        ParticleFilter single = new ParticleFilter(1, BEACONS, LENGTH, 0.0, SIGMA_Z, 1L);
        single.gaussianInit(5.0, 0.0); single.step(STEP, trueRanges(5.0, rng, true));
        check(Math.abs(sum(single.w) - 1.0) < 1e-12 && Math.abs(single.mean() - 5.5) < 1e-12,
                "a single particle still follows the motion model");
        ParticleFilter blind = new ParticleFilter(50, new double[0], LENGTH, SIGMA_U, SIGMA_Z, 5L);
        blind.gaussianInit(2.0, 0.4); blind.step(STEP, new double[0]);
        check(blind.standardDeviation() > 0.3, "with no beacons the belief only diffuses");
        ParticleFilter repeatA = new ParticleFilter(300, BEACONS, LENGTH, SIGMA_U, SIGMA_Z, 2024L);
        ParticleFilter repeatB = new ParticleFilter(300, BEACONS, LENGTH, SIGMA_U, SIGMA_Z, 2024L);
        repeatA.gaussianInit(6.0, 1.0); repeatB.gaussianInit(6.0, 1.0); double p = 6.0;
        for (int step = 0; step < 10; step++) {
            double[] ranges = trueRanges(p, new SplitMix64(1000L + step), true);
            repeatA.step(STEP, ranges); repeatB.step(STEP, ranges); p += STEP;
        }
        check(repeatA.mean() == repeatB.mean() && repeatA.standardDeviation() == repeatB.standardDeviation(),
                "the same seed must reproduce the cloud exactly");

        System.out.println("particle vs grid: worst mean gap " + worstMeanGap + ", worst relative sd gap "
                + worstStdGap);
        System.out.println("tracking error " + Math.abs(pf.mean() - position) + ", sd " + pf.standardDeviation()
                + ", global localisation error " + Math.abs(global.mean() - globalPosition));
        System.out.println("ESS before resampling " + essBefore + " -> after " + essAfter);
        System.out.println("PASS");
    }

    static double sum(double[] v) {
        double total = 0; for (double x : v) total += x; return total;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
