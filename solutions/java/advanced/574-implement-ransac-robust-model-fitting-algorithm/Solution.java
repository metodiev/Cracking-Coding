/** Task 574 - Implement RANSAC robust model fitting algorithm. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * RANSAC: repeatedly draw the minimal sample (2 points for a line, 3 for a circle), fit the model to
 * that sample, count the points within the distance threshold, and keep the largest consensus set;
 * then refit on the consensus set (total least squares for the line — the principal direction of the
 * inlier covariance — and an algebraic Kasa least-squares circle) and recompute the inliers. The
 * iteration count for success probability p at inlier ratio w is N = log(1-p)/log(1-w^s); the
 * adaptive variant recomputes that bound from the best consensus ratio found so far and stops early.
 * Iterations and model evaluations are counted, so the complexity claim is measured.
 *
 * Documented checks. (1) The formula reproduces log(0.01)/log(0.75) = 16.0078 for p = 0.99, w = 0.5,
 * s = 2 (ceiling 17), grows with p and with s and falls as w grows, and is asserted against the closed
 * form recomputed in the test for three settings. (2) The probabilistic claim itself is tested, not
 * just the code: with 50% outliers and 17 iterations the predicted success probability is 0.99248 and
 * the observed rate over 500 fixed seeds is 0.992; with 70% outliers and 49 iterations the prediction
 * is 0.99016 and the observation 0.986. (3) The budget matters: the same 70%-outlier data with only 5
 * iterations predicts 1 - 0.91^5 = 0.376 and observes 0.400, asserted inside a wide interval so that
 * disagreement with the probabilistic prediction would be caught. (4) Accuracy: with 40% outliers
 * RANSAC recovers the true line to 4.3e-15 in the normalised (a, b, c) metric, its consensus set is
 * exactly the 36 planted inliers (Jaccard 1.0 against the ground-truth mask), one model evaluation per
 * iteration (200 of 200), while ordinary total least squares on the same contaminated data is 6.71
 * away, i.e. more than 20 times worse. (5) Circles: 70 iterations for p = 0.99, s = 3 computed with
 * the conservative w = 0.4 although 30 of the 50 points really are inliers; the recovery rate within
 * 0.01 of centre and radius over 500 trials is 494/500 with the consensus refit and 423/500 without
 * it, so the refit is a measured improvement rather than a claim, and it is never worse. A detailed
 * fit recovers centre (5.000000000000001, -1.5000000000000016) and radius 2.5 with 30/30 inliers and a
 * Jaccard index of 1.0. (6) Adaptive stopping: on data with a 51/60 inlier ratio the adaptive variant
 * stops after 4 iterations against a 200-iteration budget with the same 1.4e-15 accuracy. (7) Threshold
 * sensitivity, reported honestly: a threshold 100 times tighter than the inlier noise keeps only 5 of
 * the 36 planted inliers (14% of the data) yet still lands 5.7e-4 from the true line on this instance,
 * which is documented rather than asserted — the failure of a too-tight threshold is the discarded
 * data, not necessarily the parameter error; the correct threshold recovers all 36 inliers with error
 * 0.0036, and a threshold of 50 admits all 60 points and produces a fit 4.70 away.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: the repo's 16-bit LCG has short
 * cycles and poor low-bit structure while this file draws thousands of independent index pairs and
 * triples), and the sampling loop always draws distinct indices; every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

        double nextGaussian() {
            double u1 = nextDouble();
            if (u1 < 1e-300) u1 = 1e-300;
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * nextDouble());
        }

        /** s distinct indices in [0, n). */
        int[] distinct(int n, int s) {
            int[] out = new int[s];
            int count = 0;
            while (count < s) {
                int candidate = nextInt(n);
                boolean duplicate = false;
                for (int i = 0; i < count; i++) if (out[i] == candidate) duplicate = true;
                if (!duplicate) out[count++] = candidate;
            }
            return out;
        }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static double iterationsFor(double success, double w, int s) {
        return Math.log(1.0 - success) / Math.log(1.0 - Math.pow(w, s));
    }

    /** A line as (a, b, c) with a x + b y = c and a^2 + b^2 = 1, so distances are direct. */
    static double[] fitLine(double[][] points, int[] sample) {
        double x1 = points[sample[0]][0];
        double y1 = points[sample[0]][1];
        double x2 = points[sample[1]][0];
        double y2 = points[sample[1]][1];
        double dx = x2 - x1;
        double dy = y2 - y1;
        double norm = Math.sqrt(dx * dx + dy * dy);
        if (norm < 1e-12) return null;
        double a = -dy / norm;
        double b = dx / norm;
        return new double[]{a, b, a * x1 + b * y1};
    }

    static double lineDistance(double[] line, double x, double y) {
        return Math.abs(line[0] * x + line[1] * y - line[2]);
    }

    /** A circle as (cx, cy, r). */
    static double[] fitCircle(double[][] points, int[] sample) {
        double[][] p = new double[3][2];
        for (int i = 0; i < 3; i++) {
            p[i][0] = points[sample[i]][0];
            p[i][1] = points[sample[i]][1];
        }
        double ax = p[0][0];
        double ay = p[0][1];
        double bx = p[1][0];
        double by = p[1][1];
        double cx = p[2][0];
        double cy = p[2][1];
        double d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if (Math.abs(d) < 1e-12) return null;
        double a2 = ax * ax + ay * ay;
        double b2 = bx * bx + by * by;
        double c2 = cx * cx + cy * cy;
        double ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        double uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        double r = Math.sqrt((ax - ux) * (ax - ux) + (ay - uy) * (ay - uy));
        return new double[]{ux, uy, r};
    }

    static double circleDistance(double[] circle, double x, double y) {
        double dx = x - circle[0];
        double dy = y - circle[1];
        return Math.abs(Math.sqrt(dx * dx + dy * dy) - circle[2]);
    }

    /**
     * Algebraic least-squares circle through a point set (Kasa): minimises sum (x^2 + y^2 - a x -
     * b y - c)^2 by solving the 3x3 normal equations, then converts to centre and radius.
     */
    static double[] refineCircle(double[][] points, boolean[] inliers, double[] fallback) {
        double sxx = 0, sxy = 0, sx = 0, syy = 0, sy = 0, s1 = 0, sxz = 0, syz = 0, sz = 0;
        int n = 0;
        for (int i = 0; i < points.length; i++) {
            if (!inliers[i]) continue;
            double x = points[i][0];
            double y = points[i][1];
            double z = x * x + y * y;
            sxx += x * x;
            sxy += x * y;
            sx += x;
            syy += y * y;
            sy += y;
            s1 += 1;
            sxz += x * z;
            syz += y * z;
            sz += z;
            n++;
        }
        if (n < 3) return fallback;
        double[][] m = {{sxx, sxy, sx, sxz}, {sxy, syy, sy, syz}, {sx, sy, s1, sz}};
        for (int col = 0; col < 3; col++) {
            int pivot = -1;
            for (int i = col; i < 3; i++) if (Math.abs(m[i][col]) > 1e-12) { pivot = i; break; }
            if (pivot == -1) return fallback;
            double[] tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
            double p = m[col][col];
            for (int j = col; j <= 3; j++) m[col][j] /= p;
            for (int i = 0; i < 3; i++) {
                if (i == col) continue;
                double factor = m[i][col];
                if (factor == 0.0) continue;
                for (int j = col; j <= 3; j++) m[i][j] -= factor * m[col][j];
            }
        }
        double a = m[0][3];
        double b = m[1][3];
        double c = m[2][3];
        double radiusSquared = c + 0.25 * (a * a + b * b);
        if (radiusSquared <= 0) return fallback;
        return new double[]{a / 2.0, b / 2.0, Math.sqrt(radiusSquared)};
    }

    static final class Fit {
        final double[] model;
        final boolean[] inliers;
        final int inlierCount;
        final int iterations;
        final long evaluations;

        Fit(double[] model, boolean[] inliers, int inlierCount, int iterations, long evaluations) {
            this.model = model;
            this.inliers = inliers;
            this.inlierCount = inlierCount;
            this.iterations = iterations;
            this.evaluations = evaluations;
        }
    }

    /** RANSAC for a line; adaptive when {@code maxIterations} is negative. */
    static Fit ransacLine(double[][] points, double threshold, double success, int maxIterations, Rng rng) {
        int n = points.length;
        double[] best = null;
        boolean[] bestInliers = null;
        int bestCount = -1;
        int iterations = 0;
        long evaluations = 0;
        double w = 0.5;
        while (true) {
            if (maxIterations > 0 && iterations >= maxIterations) break;
            if (maxIterations < 0) {
                double needed = iterationsFor(success, Math.max(w, 1e-9), 2);
                if (needed <= 0 || iterations >= Math.ceil(needed)) break;
            }
            int[] sample = rng.distinct(n, 2);
            double[] candidate = fitLine(points, sample);
            iterations++;
            if (candidate == null) continue;
            evaluations++;
            boolean[] inliers = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (lineDistance(candidate, points[i][0], points[i][1]) <= threshold) {
                    inliers[i] = true;
                    count++;
                }
            }
            if (count > bestCount) {
                bestCount = count;
                best = candidate;
                bestInliers = inliers;
                w = count / (double) n;
            }
        }
        if (best == null) return new Fit(null, new boolean[n], 0, iterations, evaluations);
        // refit on the consensus set by least squares
        double sumX = 0, sumY = 0, sumXX = 0, sumXY = 0, sumYY = 0;
        int m = 0;
        for (int i = 0; i < n; i++) {
            if (!bestInliers[i]) continue;
            double x = points[i][0];
            double y = points[i][1];
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumXY += x * y;
            sumYY += y * y;
            m++;
        }
        double meanX = sumX / m;
        double meanY = sumY / m;
        double sxx = sumXX / m - meanX * meanX;
        double sxy = sumXY / m - meanX * meanY;
        double syy = sumYY / m - meanY * meanY;
        double theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        double a = -Math.sin(theta);
        double b = Math.cos(theta);
        double[] refined = new double[]{a, b, a * meanX + b * meanY};
        boolean[] refinedInliers = new boolean[n];
        int refinedCount = 0;
        for (int i = 0; i < n; i++) {
            if (lineDistance(refined, points[i][0], points[i][1]) <= threshold) {
                refinedInliers[i] = true;
                refinedCount++;
            }
        }
        return new Fit(refined, refinedInliers, refinedCount, iterations, evaluations);
    }

    static Fit ransacCircle(double[][] points, double threshold, double success, int maxIterations, Rng rng) {
        return ransacCircle(points, threshold, success, maxIterations, rng, true);
    }

    static Fit ransacCircle(double[][] points, double threshold, double success, int maxIterations, Rng rng,
                            boolean refine) {
        int n = points.length;
        double[] best = null;
        boolean[] bestInliers = null;
        int bestCount = -1;
        int iterations = 0;
        long evaluations = 0;
        double w = 0.3;
        while (true) {
            if (maxIterations > 0 && iterations >= maxIterations) break;
            if (maxIterations < 0) {
                double needed = iterationsFor(success, Math.max(w, 1e-9), 3);
                if (needed <= 0 || iterations >= Math.ceil(needed)) break;
            }
            int[] sample = rng.distinct(n, 3);
            double[] candidate = fitCircle(points, sample);
            iterations++;
            if (candidate == null || candidate[2] > 1e6) continue;
            evaluations++;
            boolean[] inliers = new boolean[n];
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (circleDistance(candidate, points[i][0], points[i][1]) <= threshold) {
                    inliers[i] = true;
                    count++;
                }
            }
            if (count > bestCount) {
                bestCount = count;
                best = candidate;
                bestInliers = inliers;
                w = count / (double) n;
            }
        }
        if (best == null) return new Fit(null, new boolean[n], 0, iterations, evaluations);
        if (!refine) return new Fit(best, bestInliers, bestCount, iterations, evaluations);
        // The minimal samples are noisy, so refit on the consensus set; keep the refined model when
        // it does not lose inliers (a documented improvement step, measured in the self-test).
        double[] refined = refineCircle(points, bestInliers, best);
        boolean[] refinedInliers = new boolean[n];
        int refinedCount = 0;
        for (int i = 0; i < n; i++) {
            if (circleDistance(refined, points[i][0], points[i][1]) <= threshold) {
                refinedInliers[i] = true;
                refinedCount++;
            }
        }
        if (refinedCount >= bestCount) return new Fit(refined, refinedInliers, refinedCount, iterations, evaluations);
        return new Fit(best, bestInliers, bestCount, iterations, evaluations);
    }

    /** Total least squares line through all points (the fragile baseline). */
    static double[] leastSquaresLine(double[][] points) {
        int n = points.length;
        double sumX = 0, sumY = 0, sumXX = 0, sumXY = 0, sumYY = 0;
        for (double[] p : points) {
            sumX += p[0];
            sumY += p[1];
            sumXX += p[0] * p[0];
            sumXY += p[0] * p[1];
            sumYY += p[1] * p[1];
        }
        double meanX = sumX / n;
        double meanY = sumY / n;
        double sxx = sumXX / n - meanX * meanX;
        double sxy = sumXY / n - meanX * meanY;
        double syy = sumYY / n - meanY * meanY;
        double theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
        double a = -Math.sin(theta);
        double b = Math.cos(theta);
        return new double[]{a, b, a * meanX + b * meanY};
    }

    static double lineDifference(double[] first, double[] second) {
        return Math.abs(first[0] - second[0]) + Math.abs(first[1] - second[1])
                + Math.abs(first[2] - second[2]);
    }

    static double jaccard(boolean[] a, boolean[] b) {
        int intersection = 0;
        int union = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] && b[i]) intersection++;
            if (a[i] || b[i]) union++;
        }
        return union == 0 ? 1.0 : intersection / (double) union;
    }

    /**
     * Data with the first {@code inliers} points on the line y = 0.5 x + 3 and the rest scattered
     * uniformly; {@code noise}, when given, scales the perpendicular jitter of the inliers by 0.01.
     */
    static double[][] lineData(double[] noise, int n, int inliers, double spread, Rng rng, boolean[] mask) {
        double[][] points = new double[n][2];
        for (int i = 0; i < n; i++) {
            if (i < inliers) {
                points[i][0] = i * 1.0;
                points[i][1] = 0.5 * points[i][0] + 3.0;
                if (noise != null) points[i][1] += noise[i] * 0.01;
                mask[i] = true;
            } else {
                points[i][0] = rng.nextDouble() * n;
                points[i][1] = 3.0 + rng.nextDouble() * spread - spread / 2;
                mask[i] = false;
            }
        }
        return points;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(574_2025L);

        // ---- (1) the iteration formula ---------------------------------------------------------
        double classic = iterationsFor(0.99, 0.5, 2);
        System.out.println("N(0.99, w=0.5, s=2) = " + classic + " (ceiling " + (int) Math.ceil(classic) + ")");
        check(Math.abs(classic - 16.0078) <= 1e-3, "the textbook value 16.0078 is reproduced");
        check((int) Math.ceil(classic) == 17, "the ceiling of the formula is 17 iterations");
        check(iterationsFor(0.999, 0.5, 2) > classic, "N increases with the requested success probability");
        check(iterationsFor(0.99, 0.3, 2) > classic, "N increases as the inlier ratio falls");
        check(iterationsFor(0.99, 0.5, 3) > classic, "N increases with the sample size");
        double[] settings = {0.99, 0.99, 0.95};
        double[] ratios = {0.5, 0.3, 0.6};
        int[] sizes = {2, 2, 3};
        for (int i = 0; i < settings.length; i++) {
            double expected = Math.log(1 - settings[i]) / Math.log(1 - Math.pow(ratios[i], sizes[i]));
            check(Math.abs(iterationsFor(settings[i], ratios[i], sizes[i]) - expected) <= 1e-12,
                    "the closed form is implemented exactly");
        }

        // ---- (2) success probability at 50% and 70% outliers ------------------------------------
        int trials = 500;
        int successes = 0;
        int[] thresholds = {20};
        for (int trial = 0; trial < trials; trial++) {
            Rng local = new Rng(10_000L + trial);
            boolean[] mask = new boolean[40];
            double[][] points = lineData(null, 40, 20, 8.0, local, mask);
            Fit fit = ransacLine(points, 0.05, 0.99, (int) Math.ceil(classic), local);
            // the truth line y = 0.5x + 3, normalised to a x + b y = c
            double norm = Math.sqrt(0.5 * 0.5 + 1.0);
            double[] truthLine = {-0.5 / norm, 1.0 / norm, 3.0 / norm};
            if (fit.model != null && lineDifference(fit.model, truthLine) <= 0.02) successes++;
        }
        double observed = successes / (double) trials;
        double predicted = 1 - Math.pow(0.75, Math.ceil(classic));
        System.out.println("50% outliers, " + (int) Math.ceil(classic) + " iterations: predicted " + predicted
                + ", observed " + observed);
        check(observed >= 0.98, "the observed success rate matches the predicted 0.99261");

        double seventy = Math.ceil(iterationsFor(0.99, 0.3, 2));
        successes = 0;
        for (int trial = 0; trial < trials; trial++) {
            Rng local = new Rng(20_000L + trial);
            boolean[] mask = new boolean[40];
            double[][] points = lineData(null, 40, 12, 8.0, local, mask);
            Fit fit = ransacLine(points, 0.05, 0.99, (int) seventy, local);
            double norm = Math.sqrt(0.5 * 0.5 + 1.0);
            double[] truthLine = {-0.5 / norm, 1.0 / norm, 3.0 / norm};
            if (fit.model != null && lineDifference(fit.model, truthLine) <= 0.02) successes++;
        }
        observed = successes / (double) trials;
        predicted = 1 - Math.pow(0.91, seventy);
        System.out.println("70% outliers, " + (int) seventy + " iterations: predicted " + predicted
                + ", observed " + observed);
        check(observed >= 0.96, "the observed success rate matches the predicted 0.99024");

        // ---- (3) an insufficient budget must fail often ------------------------------------------
        successes = 0;
        for (int trial = 0; trial < trials; trial++) {
            Rng local = new Rng(30_000L + trial);
            boolean[] mask = new boolean[40];
            double[][] points = lineData(null, 40, 12, 8.0, local, mask);
            Fit fit = ransacLine(points, 0.05, 0.99, 5, local);
            double norm = Math.sqrt(0.5 * 0.5 + 1.0);
            double[] truthLine = {-0.5 / norm, 1.0 / norm, 3.0 / norm};
            if (fit.model != null && lineDifference(fit.model, truthLine) <= 0.02) successes++;
        }
        observed = successes / (double) trials;
        System.out.println("70% outliers, only 5 iterations: observed " + observed + " (predicted "
                + (1 - Math.pow(0.91, 5)) + ")");
        check(observed <= 0.6, "a five-iteration budget leaves the fit unreliable");
        check(observed >= 0.2, "but occasionally a lucky sample of five still works");

        // ---- (4) accuracy and the least-squares baseline -----------------------------------------
        Rng local = new Rng(40_000L);
        boolean[] mask = new boolean[60];
        double[][] points = lineData(null, 60, 36, 12.0, local, mask);
        Fit fit = ransacLine(points, 0.05, 0.99, 200, local);
        double norm = Math.sqrt(0.5 * 0.5 + 1.0);
        double[] truthLine = {-0.5 / norm, 1.0 / norm, 3.0 / norm};
        double ransacError = lineDifference(fit.model, truthLine);
        double[] plainLeastSquares = leastSquaresLine(points);
        double lsError = lineDifference(plainLeastSquares, truthLine);
        System.out.println("accuracy: RANSAC error " + ransacError + ", least-squares error " + lsError
                + ", inliers " + fit.inlierCount + "/60, evaluations " + fit.evaluations
                + " in " + fit.iterations + " iterations");
        check(ransacError <= 0.02, "RANSAC recovers the true line to 0.02");
        check(lsError >= 20 * Math.max(ransacError, 1e-6), "least squares on contaminated data is far off");
        check(fit.inlierCount == 36, "the consensus set has exactly the 36 planted inliers");
        check(jaccard(fit.inliers, mask) == 1.0, "the inlier mask matches the ground truth exactly");
        check(fit.evaluations == fit.iterations, "one model evaluation per iteration");
        check(fit.iterations <= 200, "the fixed budget is respected");

        // ---- (5) circles -------------------------------------------------------------------------
        double circleIterations = iterationsFor(0.99, 0.4, 3);
        int circleSuccesses = 0;
        int circleSuccessesUnrefined = 0;
        for (int trial = 0; trial < trials; trial++) {
            Rng c = new Rng(50_000L + trial);
            double[][] data = new double[50][2];
            for (int i = 0; i < 30; i++) {
                double angle = c.nextDouble() * Math.PI * 2;
                data[i][0] = 5.0 + 2.5 * Math.cos(angle) + c.nextGaussian() * 0.002;
                data[i][1] = -1.5 + 2.5 * Math.sin(angle) + c.nextGaussian() * 0.002;
            }
            for (int i = 30; i < 50; i++) {
                data[i][0] = c.nextDouble() * 20 - 5;
                data[i][1] = c.nextDouble() * 20 - 5;
            }
            // The assumed inlier ratio 0.4 is conservative: 30 of the 50 points really are inliers.
            Rng second = new Rng(50_000L + trial);
            Fit circle = ransacCircle(data, 0.05, 0.99, (int) Math.ceil(circleIterations), c);
            if (circle.model != null && Math.abs(circle.model[0] - 5.0) <= 0.01
                    && Math.abs(circle.model[1] + 1.5) <= 0.01 && Math.abs(circle.model[2] - 2.5) <= 0.01) {
                circleSuccesses++;
            }
            Fit plain = ransacCircle(data, 0.05, 0.99, (int) Math.ceil(circleIterations), second, false);
            if (plain.model != null && Math.abs(plain.model[0] - 5.0) <= 0.01
                    && Math.abs(plain.model[1] + 1.5) <= 0.01 && Math.abs(plain.model[2] - 2.5) <= 0.01) {
                circleSuccessesUnrefined++;
            }
        }
        System.out.println("circles (40% outliers, " + (int) Math.ceil(circleIterations) + " iterations): "
                + circleSuccesses + "/500 recovered with the consensus refit, "
                + circleSuccessesUnrefined + "/500 without it");
        check(circleSuccesses >= 480, "the circle is recovered in at least 480 of 500 trials");
        check(circleSuccesses >= circleSuccessesUnrefined,
                "the consensus refit never hurts the recovery rate on this family");

        // a single circle fit in detail, including the inlier mask
        Rng details = new Rng(60_000L);
        double[][] circleData = new double[50][2];
        boolean[] circleMask = new boolean[50];
        for (int i = 0; i < 30; i++) {
            double angle = details.nextDouble() * Math.PI * 2;
            circleData[i][0] = 5.0 + 2.5 * Math.cos(angle);
            circleData[i][1] = -1.5 + 2.5 * Math.sin(angle);
            circleMask[i] = true;
        }
        for (int i = 30; i < 50; i++) {
            circleData[i][0] = details.nextDouble() * 20 - 5;
            circleData[i][1] = details.nextDouble() * 20 - 5;
            circleMask[i] = false;
        }
        Fit circleFit = ransacCircle(circleData, 0.05, 0.99, 500, details);
        System.out.println("detailed circle fit: centre (" + circleFit.model[0] + ", " + circleFit.model[1]
                + ") radius " + circleFit.model[2] + ", inliers " + circleFit.inlierCount
                + "/30, Jaccard " + jaccard(circleFit.inliers, circleMask));
        check(Math.abs(circleFit.model[0] - 5.0) <= 0.01, "circle centre x recovered");
        check(Math.abs(circleFit.model[1] + 1.5) <= 0.01, "circle centre y recovered");
        check(Math.abs(circleFit.model[2] - 2.5) <= 0.01, "circle radius recovered");
        check(jaccard(circleFit.inliers, circleMask) >= 0.9, "the circle inlier mask matches the truth");
        // documented comparison: the best minimal-sample circle can be several noise units off,
        // because three nearby noisy points define an ill-conditioned circumcircle
        Fit unrefined = ransacCircle(circleData, 0.05, 0.99, 500, new Rng(61_000L), false);
        System.out.println("  without the consensus refit: centre (" + unrefined.model[0] + ", "
                + unrefined.model[1] + ") radius " + unrefined.model[2]
                + ", inliers " + unrefined.inlierCount);
        check(jaccard(unrefined.inliers, circleMask) >= 0.9, "the unrefined circle still finds the inliers");

        // ---- (6) adaptive stopping ---------------------------------------------------------------
        Rng adaptive = new Rng(70_000L);
        boolean[] adaptiveMask = new boolean[60];
        double[][] adaptiveData = lineData(null, 60, 51, 12.0, adaptive, adaptiveMask);
        Fit adaptiveFit = ransacLine(adaptiveData, 0.05, 0.99, -1, adaptive);
        Fit fixedFit = ransacLine(adaptiveData, 0.05, 0.99, 200, new Rng(70_000L));
        double[] truthAdaptive = {-0.5 / norm, 1.0 / norm, 3.0 / norm};
        System.out.println("adaptive stopping: iterations " + adaptiveFit.iterations + " versus fixed "
                + fixedFit.iterations + ", error " + lineDifference(adaptiveFit.model, truthAdaptive));
        check(adaptiveFit.iterations < fixedFit.iterations, "adaptive stopping saves iterations here");
        check(lineDifference(adaptiveFit.model, truthAdaptive) <= 0.02, "the adaptive fit is still correct");
        check(adaptiveFit.iterations <= 200, "the adaptive run never exceeds the budget");

        // ---- (7) threshold sensitivity -----------------------------------------------------------
        Rng sensitive = new Rng(80_000L);
        boolean[] sensitiveMask = new boolean[60];
        double[] inlierNoise = new double[60];
        for (int i = 0; i < 60; i++) inlierNoise[i] = sensitive.nextGaussian();
        double[][] sensitiveData = lineData(inlierNoise, 60, 36, 12.0, sensitive, sensitiveMask);
        Fit tooTight = ransacLine(sensitiveData, 0.0005, 0.99, 500, new Rng(81_000L));
        Fit correct = ransacLine(sensitiveData, 0.05, 0.99, 500, new Rng(83_000L));
        Fit tooLoose = ransacLine(sensitiveData, 50.0, 0.99, 500, new Rng(82_000L));
        System.out.println("threshold sensitivity: tight threshold inliers " + tooTight.inlierCount
                + " error " + lineDifference(tooTight.model, truthLine)
                + "; correct threshold inliers " + correct.inlierCount
                + " error " + lineDifference(correct.model, truthLine)
                + "; loose threshold inliers " + tooLoose.inlierCount
                + " error " + lineDifference(tooLoose.model, truthLine));
        // A threshold 100 times tighter than the inlier noise keeps only the handful of samples that
        // happen to be nearly on one line: 5 of the 36 planted inliers survive, so the estimate is
        // built from 14% of the data. On this instance its numerical error is still small (5.7e-4,
        // the extreme-low-noise points are close to the true line), which is documented rather than
        // asserted: throwing away 86% of the inliers is the failure, not the parameter error.
        check(tooTight.inlierCount <= 5, "a threshold 100 times tighter than the noise finds few inliers");
        check(correct.inlierCount == 36, "the correct threshold recovers all 36 planted inliers");
        check(lineDifference(correct.model, truthLine) <= 0.05, "and the estimate is accurate");
        check(tooLoose.inlierCount == 60, "a threshold of 50 admits every point");
        check(lineDifference(tooLoose.model, truthLine) > 0.02, "and the unpolluted least-squares fit is wrong");

        System.out.println("PASS");
    }
}
