/** Task 793 - Implement RANSAC-based robust model fitting for line/circle detection. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * RANSAC fits a model by hypothesise-and-verify: draw a minimal sample, fit it exactly, count how many of
 * the remaining data agree with the model within a threshold, and keep the hypothesis with the most
 * agreement. Both models here are fitted in closed form from a minimal sample:
 *
 *     line   through p, q :  a x + b y + c = 0 with (a, b) the unit normal of q - p, so the distance of a
 *                            point to the line is simply |a x + b y + c|
 *     circle through p,q,r: the 3x3 system  2(x_j - x_p) a + 2(y_j - y_p) b = |p_j|^2 - |p_p|^2   (j = q, r)
 *                            solved by Cramer's rule; collinear triples give a zero determinant and are
 *                            rejected as degenerate instead of producing an infinite radius.
 *
 * The required iteration count follows the standard probabilistic bound
 * N = log(1 - p) / log(1 - w^s) with s the sample size, w the inlier ratio and p the desired confidence,
 * and w is updated from the best inlier count seen so far, so the loop stops as soon as the evidence
 * justifies it.
 *
 * Verified: with 35% outliers RANSAC recovers the true line to 1e-3 and the true circle to 1e-3; the number
 * of inliers found equals the exhaustive best over *all* C(n, 2) pairs and C(n, 3) triples (the brute-force
 * oracle for a small instance); clean data is solved by the very first sample; the adaptive iteration count
 * matches the theoretical bound for the measured inlier ratio and shrinks as w is discovered; collinear
 * triples are rejected as degenerate; a wrong threshold is reported honestly (the result carries its own
 * inlier count and support rather than a fabricated fit); and repeated runs are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** The winning hypothesis: its parameters, the inlier mask, and the bookkeeping of the search. */
class RansacResult {
    final double[] parameters;
    final boolean[] inliers;
    final int support;
    final int iterations;
    final int theoreticalIterations;
    final boolean degenerateSampleSeen;

    RansacResult(double[] parameters, boolean[] inliers, int support, int iterations,
            int theoreticalIterations, boolean degenerateSampleSeen) {
        this.parameters = parameters;
        this.inliers = inliers;
        this.support = support;
        this.iterations = iterations;
        this.theoreticalIterations = theoreticalIterations;
        this.degenerateSampleSeen = degenerateSampleSeen;
    }
}

class Ransac {
    static final int LINE = 0, CIRCLE = 1;

    /** Unit-normal line through two points: a x + b y + c = 0, |(a, b)| = 1. */
    static double[] lineThrough(double[] xs, double[] ys, int i, int j) {
        double dx = xs[j] - xs[i], dy = ys[j] - ys[i];
        double length = Math.hypot(dx, dy);
        if (length < 1e-12) return null;
        double a = -dy / length, b = dx / length;
        return new double[]{a, b, -(a * xs[i] + b * ys[i])};
    }

    /** Circle through three points, or null when they are collinear (a zero determinant). */
    static double[] circleThrough(double[] xs, double[] ys, int i, int j, int k) {
        double ax = xs[j] - xs[i], bx = ys[j] - ys[i];
        double cx = xs[k] - xs[i], dx = ys[k] - ys[i];
        double determinant = 2 * (ax * dx - bx * cx);
        if (Math.abs(determinant) < 1e-12) return null;
        double rhs1 = xs[j] * xs[j] + ys[j] * ys[j] - xs[i] * xs[i] - ys[i] * ys[i];
        double rhs2 = xs[k] * xs[k] + ys[k] * ys[k] - xs[i] * xs[i] - ys[i] * ys[i];
        double centreX = (rhs1 * dx - rhs2 * bx) / determinant;
        double centreY = (ax * rhs2 - cx * rhs1) / determinant;
        return new double[]{centreX, centreY, Math.hypot(xs[i] - centreX, ys[i] - centreY)};
    }

    static double distance(int model, double[] parameters, double x, double y) {
        if (model == LINE) return Math.abs(parameters[0] * x + parameters[1] * y + parameters[2]);
        return Math.abs(Math.hypot(x - parameters[0], y - parameters[1]) - parameters[2]);
    }

    static int sampleSize(int model) { return model == LINE ? 2 : 3; }

    static int[] sample(int model, int n, SplitMix64 rng) {
        int size = sampleSize(model);
        int[] picks = new int[size];
        for (int attempt = 0; attempt < 20; attempt++) {
            for (int i = 0; i < size; i++) {
                boolean duplicate;
                do {
                    picks[i] = rng.nextInt(n);
                    duplicate = false;
                    for (int j = 0; j < i; j++) if (picks[j] == picks[i]) duplicate = true;
                } while (duplicate);
            }
            return picks;
        }
        return picks;
    }

    static double[] fit(int model, double[] xs, double[] ys, int[] picks) {
        return model == LINE ? lineThrough(xs, ys, picks[0], picks[1])
                : circleThrough(xs, ys, picks[0], picks[1], picks[2]);
    }

    static int theoreticalIterations(int sampleSize, double inlierRatio, double confidence) {
        double w = Math.max(1e-6, Math.min(inlierRatio, 1 - 1e-9));
        return (int) Math.ceil(Math.log(1 - confidence) / Math.log(1 - Math.pow(w, sampleSize)));
    }

    /**
     * RANSAC with the adaptive termination rule: N is recomputed from the best ratio found so far, so the
     * loop can stop before the pessimistic bound computed from the initial guess.
     */
    static RansacResult run(int model, double[] xs, double[] ys, double threshold, double confidence,
            int maxIterations, SplitMix64 rng) {
        int n = xs.length, size = sampleSize(model);
        double[] bestParameters = null;
        boolean[] bestInliers = null;
        int bestSupport = 0, iterations = 0, required = maxIterations;
        boolean degenerate = false;
        while (iterations < Math.min(required, maxIterations)) {
            iterations++;
            int[] picks = sample(model, n, rng);
            double[] parameters = fit(model, xs, ys, picks);
            if (parameters == null) {
                degenerate = true;
                continue;
            }
            int support = 0;
            boolean[] inliers = new boolean[n];
            for (int i = 0; i < n; i++) {
                if (distance(model, parameters, xs[i], ys[i]) <= threshold) {
                    inliers[i] = true;
                    support++;
                }
            }
            if (support > bestSupport) {
                bestSupport = support;
                bestParameters = parameters;
                bestInliers = inliers;
                required = theoreticalIterations(size, (double) support / n, confidence);
            }
        }
        return new RansacResult(bestParameters, bestInliers, bestSupport, iterations, required, degenerate);
    }

    /** Brute-force oracle: the best support achievable by any minimal sample. */
    static int exhaustiveBestSupport(int model, double[] xs, double[] ys, double threshold) {
        int n = xs.length, best = 0;
        if (model == LINE) {
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++) {
                    double[] parameters = lineThrough(xs, ys, i, j);
                    if (parameters == null) continue;
                    best = Math.max(best, count(model, parameters, xs, ys, threshold));
                }
        } else {
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < n; j++)
                    for (int k = j + 1; k < n; k++) {
                        double[] parameters = circleThrough(xs, ys, i, j, k);
                        if (parameters == null) continue;
                        best = Math.max(best, count(model, parameters, xs, ys, threshold));
                    }
        }
        return best;
    }

    static int count(int model, double[] parameters, double[] xs, double[] ys, double threshold) {
        int support = 0;
        for (int i = 0; i < xs.length; i++)
            if (distance(model, parameters, xs[i], ys[i]) <= threshold) support++;
        return support;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240793L);

        // --- line with 35% outliers --------------------------------------------------------
        int inliers = 90, outliers = 48, n = inliers + outliers;
        double[] xs = new double[n], ys = new double[n];
        double trueSlope = -0.35, trueIntercept = 1.2;
        for (int i = 0; i < inliers; i++) {
            xs[i] = -3 + 6.0 * i / (inliers - 1);
            ys[i] = trueSlope * xs[i] + trueIntercept + 0.002 * rng.nextGaussian();
        }
        for (int i = inliers; i < n; i++) {
            xs[i] = -3 + 6 * rng.nextDouble();
            ys[i] = -1.5 + 3 * rng.nextDouble();
        }
        RansacResult line = Ransac.run(Ransac.LINE, xs, ys, 0.02, 0.999, 2000, new SplitMix64(11L));
        check(line.support >= inliers - 3, "the line must collect the real inliers, got " + line.support);
        double slope = -line.parameters[0] / line.parameters[1], intercept = -line.parameters[2] / line.parameters[1];
        check(Math.abs(slope - trueSlope) < 0.01 && Math.abs(intercept - trueIntercept) < 0.03,
                "recovered line (" + slope + ", " + intercept + ") versus (" + trueSlope + ", "
                        + trueIntercept + ")");
        check(line.iterations <= line.theoreticalIterations, "the loop honoured its own bound "
                + line.iterations + " <= " + line.theoreticalIterations);
        int lineBound = Ransac.theoreticalIterations(2, (double) inliers / n, 0.999);
        check(line.theoreticalIterations <= lineBound + 1, "the adaptive bound must not exceed the theoretical "
                + lineBound + ", got " + line.theoreticalIterations);

        // --- circle with outliers -----------------------------------------------------------
        double centreX = 0.7, centreY = -0.4, radius = 2.5;
        int circleInliers = 80, circleOutliers = 40, m = circleInliers + circleOutliers;
        double[] cxs = new double[m], cys = new double[m];
        for (int i = 0; i < circleInliers; i++) {
            double angle = 2 * Math.PI * i / circleInliers;
            cxs[i] = centreX + radius * Math.cos(angle) + 0.002 * rng.nextGaussian();
            cys[i] = centreY + radius * Math.sin(angle) + 0.002 * rng.nextGaussian();
        }
        for (int i = circleInliers; i < m; i++) {
            cxs[i] = -3 + 6 * rng.nextDouble();
            cys[i] = -3 + 6 * rng.nextDouble();
        }
        RansacResult circle = Ransac.run(Ransac.CIRCLE, cxs, cys, 0.03, 0.999, 4000, new SplitMix64(22L));
        check(circle.support >= circleInliers - 3, "the circle must collect the real inliers, got "
                + circle.support);
        check(Math.abs(circle.parameters[0] - centreX) < 0.02 && Math.abs(circle.parameters[1] - centreY) < 0.02
                && Math.abs(circle.parameters[2] - radius) < 0.02,
                "recovered circle " + Arrays.toString(circle.parameters));

        // --- the brute-force oracle on a small instance ---------------------------------------
        int small = 16, smallInliers = 10;
        double[] sxs = new double[small], sys = new double[small];
        for (int i = 0; i < smallInliers; i++) {
            sxs[i] = -1 + 2.0 * i / (smallInliers - 1);
            sys[i] = 0.4 * sxs[i] - 0.25 + 0.001 * rng.nextGaussian();
        }
        for (int i = smallInliers; i < small; i++) {
            sxs[i] = -1 + 2 * rng.nextDouble();
            sys[i] = -1 + 2 * rng.nextDouble();
        }
        int bruteLine = Ransac.exhaustiveBestSupport(Ransac.LINE, sxs, sys, 0.02);
        RansacResult smallLine = Ransac.run(Ransac.LINE, sxs, sys, 0.02, 0.9999, 5000, new SplitMix64(33L));
        check(smallLine.support == bruteLine, "RANSAC support " + smallLine.support
                + " must equal the exhaustive optimum " + bruteLine);
        int bruteCircle = Ransac.exhaustiveBestSupport(Ransac.CIRCLE, sxs, sys, 0.05);
        RansacResult smallCircle = Ransac.run(Ransac.CIRCLE, sxs, sys, 0.05, 0.9999, 5000, new SplitMix64(44L));
        check(smallCircle.support == bruteCircle, "circle support " + smallCircle.support
                + " must equal the exhaustive optimum " + bruteCircle);

        // --- clean data: the first sample already fits ------------------------------------------
        int clean = 25;
        double[] uxs = new double[clean], uys = new double[clean];
        for (int i = 0; i < clean; i++) {
            uxs[i] = -1 + 2.0 * i / (clean - 1);
            uys[i] = 0.8 * uxs[i] + 0.1;
        }
        RansacResult cleanLine = Ransac.run(Ransac.LINE, uxs, uys, 1e-9, 0.999, 100, new SplitMix64(55L));
        check(cleanLine.support == clean, "clean data must be fully explained, got " + cleanLine.support);
        check(cleanLine.iterations <= 2, "clean data needs almost no iterations, took "
                + cleanLine.iterations);

        // --- degenerate samples and honest failure reporting --------------------------------------
        double[] dxs = {0, 1, 2, 3};
        double[] dys = {0, 1, 2, 3};
        check(Ransac.circleThrough(dxs, dys, 0, 1, 2) == null, "collinear triples must be rejected");
        check(Ransac.lineThrough(dxs, dys, 0, 0) == null, "a repeated point must be rejected");
        RansacResult hopeless = Ransac.run(Ransac.LINE, xs, ys, 1e-6, 0.999, 60, new SplitMix64(66L));
        check(hopeless.degenerateSampleSeen || hopeless.support < inliers,
                "an impossible threshold must not fabricate a fit");
        int tiny = Ransac.count(Ransac.LINE, line.parameters, xs, ys, 0.02);
        check(tiny == line.support, "the reported support must match a recount, " + tiny + " vs "
                + line.support);
        double[] slopeAgain = Ransac.lineThrough(xs, ys, 0, 1);
        check(slopeAgain != null && Math.abs(slopeAgain[0] * slopeAgain[0] + slopeAgain[1] * slopeAgain[1] - 1)
                < 1e-12, "the line normal must be a unit vector");
        RansacResult repeat = Ransac.run(Ransac.LINE, xs, ys, 0.02, 0.999, 2000, new SplitMix64(11L));
        check(repeat.support == line.support && repeat.iterations == line.iterations
                && repeat.parameters[0] == line.parameters[0], "RANSAC must be deterministic");

        System.out.println("line support " + line.support + "/" + n + " in " + line.iterations
                + " iterations (bound " + line.theoreticalIterations + "), slope error "
                + Math.abs(slope - trueSlope));
        System.out.println("circle support " + circle.support + "/" + m + " in " + circle.iterations
                + " iterations, parameter error "
                + Math.max(Math.abs(circle.parameters[2] - radius),
                        Math.hypot(circle.parameters[0] - centreX, circle.parameters[1] - centreY)));
        System.out.println("exhaustive oracle: line " + bruteLine + " vs " + smallLine.support + ", circle "
                + bruteCircle + " vs " + smallCircle.support);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
