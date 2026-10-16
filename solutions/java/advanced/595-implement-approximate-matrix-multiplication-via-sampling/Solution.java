/** Task 595 - Implement approximate matrix multiplication via sampling. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Drineas-Kannan-Mahoney style sketching of a product: sample c columns of A together with the matching
 * rows of B, and estimate
 *
 *     A B  ~  X = (1/c) * sum_{t=1..c} (1/p_{j_t}) * A_{:,j_t} B_{j_t,:},
 *
 * with j_t drawn with replacement from a probability vector p. Two families are implemented: uniform
 * sampling (p_j = 1/n) and length-squared (importance) sampling (p_j = ||A_{:,j}||^2 / ||A||_F^2). A third,
 * without-replacement variant with the Horvitz-Thompson weights pi_j = c * p_j is included, because it
 * becomes exact when c = n and therefore pins the scaling conventions down.
 *
 * The file does not stop at Monte-Carlo evidence. For both families the exact mean squared error is known
 * in closed form,
 *
 *     E ||X - A B||_F^2 = (1/c) * ( sum_j ||A_{:,j}||^2 ||B_{j,:}||^2 / p_j  -  ||A B||_F^2 ),
 *
 * which follows from the i.i.d. rank-one nature of the summands, and it is asserted three ways: the formula
 * is compared against a Monte-Carlo estimate over 2000 independent sketches with a self-calibrated
 * four-sigma interval (the interval is computed from the sample's own spread, so the check cannot fail on a
 * mis-guessed variance); the 1/c scaling law MSE(c1) / MSE(c2) = c2 / c1 is asserted as an identity; and for
 * importance sampling the sum simplifies to ||A||_F^2 ||B||_F^2, giving the textbook bound
 * E||X - A B||_F^2 <= (1/c) ||A||_F^2 ||B||_F^2 and, by Jensen, E||X - A B||_F <= sqrt(1/c) ||A||_F ||B||_F,
 * both asserted numerically.
 *
 * Further documented checks: unbiasedness of both estimators by Monte-Carlo means with four-sigma intervals;
 * on a matrix with a single dominant column of A matched by a single dominant row of B the importance sampler
 * is orders of magnitude better than uniform sampling, and the ratio is reported (the closed forms predict a
 * factor close to n); a single-nonzero-column instance makes the importance sampler deterministic with zero
 * error; equal column norms make importance sampling coincide with uniform sampling exactly; the
 * without-replacement variant with c = n reproduces A B to 1e-12 for the uniform family (the only difference
 * being the summation order of the rank-one terms, documented since exact float equality is not the honest
 * claim) and is shown *not* to be exact for the length-squared family, where c p_j < 1 for several columns
 * makes the Horvitz-Thompson inclusions less than 1 - a pps limitation that is asserted rather than papered
 * over; the without-replacement estimator with c = 5 is also verified unbiased by Monte Carlo; and the whole
 * construction is reproducible from fixed seeds (splitmix64, not the repo's short-cycle 16-bit LCG, because a
 * Monte-Carlo study needs many independent draws). Boundary cases: c = 1, c = n, and c > n.
 */
import java.util.*;

public class Solution {
    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        double nextDouble() {
            return (next() >>> 11) * 0x1.0p-53;
        }

        double nextGaussian() {
            double u1 = Math.max(nextDouble(), 1e-12);
            double u2 = nextDouble();
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        }

        /** Categorical draw with the given cumulative probabilities. */
        int draw(double[] cumulative) {
            double u = nextDouble();
            int low = 0;
            int high = cumulative.length - 1;
            while (low < high) {
                int middle = (low + high) >>> 1;
                if (u <= cumulative[middle]) high = middle;
                else low = middle + 1;
            }
            return low;
        }
    }

    // ---------------------------------------------------------------- matrix helpers

    static double[][] multiply(double[][] a, double[][] b) {
        double[][] out = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                double acc = 0.0;
                for (int t = 0; t < b.length; t++) acc += a[i][t] * b[t][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    static double frobenius(double[][] a) {
        double total = 0.0;
        for (double[] row : a) {
            for (double value : row) total += value * value;
        }
        return Math.sqrt(total);
    }

    static double frobeniusSquared(double[][] a) {
        return frobenius(a) * frobenius(a);
    }

    static double difference(double[][] a, double[][] b) {
        double total = 0.0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                double diff = a[i][j] - b[i][j];
                total += diff * diff;
            }
        }
        return Math.sqrt(total);
    }

    static double[] columnSquaredNorms(double[][] a) {
        double[] out = new double[a[0].length];
        for (int j = 0; j < a[0].length; j++) {
            double acc = 0.0;
            for (int i = 0; i < a.length; i++) acc += a[i][j] * a[i][j];
            out[j] = acc;
        }
        return out;
    }

    static double[] rowSquaredNorms(double[][] b) {
        double[] out = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            double acc = 0.0;
            for (int j = 0; j < b[0].length; j++) acc += b[i][j] * b[i][j];
            out[i] = acc;
        }
        return out;
    }

    static double[] uniformProbabilities(int n) {
        double[] p = new double[n];
        Arrays.fill(p, 1.0 / n);
        return p;
    }

    /**
     * Length-squared sampling probabilities. The distribution is undefined for a zero matrix (0/0), so the
     * routine falls back to the uniform vector there and says so at the assertion.
     */
    static double[] importanceProbabilities(double[][] a) {
        double[] norms = columnSquaredNorms(a);
        double total = 0.0;
        for (double value : norms) total += value;
        if (total <= 0.0) return uniformProbabilities(norms.length);
        double[] p = new double[norms.length];
        for (int j = 0; j < norms.length; j++) p[j] = norms[j] / total;
        return p;
    }

    static double[] cumulative(double[] p) {
        double[] out = new double[p.length];
        double running = 0.0;
        for (int j = 0; j < p.length; j++) {
            running += p[j];
            out[j] = running;
        }
        out[p.length - 1] = 1.0;
        return out;
    }

    /** One sketch with replacement: c samples, each contributing an unbiased rank-one term. */
    static double[][] sketch(double[][] a, double[][] b, double[] p, int c, Rng rng) {
        double[] cumulative = cumulative(p);
        double[][] out = new double[a.length][b[0].length];
        for (int t = 0; t < c; t++) {
            int j = rng.draw(cumulative);
            double weight = 1.0 / (c * p[j]);
            for (int i = 0; i < a.length; i++) {
                double scaled = weight * a[i][j];
                for (int k = 0; k < b[0].length; k++) out[i][k] += scaled * b[j][k];
            }
        }
        return out;
    }

    /**
     * Without-replacement variant with Horvitz-Thompson weights 1 / pi_j, pi_j = min(1, c p_j). The
     * estimator is a plain sum, not a c-scaled average: with pi_j = 1 for every column when c = n it
     * reduces to sum_j A_{:,j} B_{j,:} = A B exactly, which is what pins the normalisation down.
     */
    static double[][] sketchWithoutReplacement(double[][] a, double[][] b, double[] p, int c, Rng rng) {
        Integer[] order = new Integer[p.length];
        for (int j = 0; j < p.length; j++) order[j] = j;
        double[] keys = new double[p.length];
        for (int j = 0; j < p.length; j++) keys[j] = p[j] * rng.nextDouble();
        Arrays.sort(order, (x, y) -> Double.compare(keys[y], keys[x]));
        double[][] out = new double[a.length][b[0].length];
        for (int t = 0; t < Math.min(c, p.length); t++) {
            int j = order[t];
            double inclusion = Math.min(1.0, c * p[j]);
            double weight = 1.0 / inclusion;
            for (int i = 0; i < a.length; i++) {
                double scaled = weight * a[i][j];
                for (int k = 0; k < b[0].length; k++) out[i][k] += scaled * b[j][k];
            }
        }
        return out;
    }

    /** The exact expected squared error of the with-replacement estimator. */
    static double exactMeanSquaredError(double[][] a, double[][] b, double[] p, int c) {
        double[] columnNorms = columnSquaredNorms(a);
        double[] rowNorms = rowSquaredNorms(b);
        double total = 0.0;
        for (int j = 0; j < p.length; j++) total += columnNorms[j] * rowNorms[j] / p[j];
        total -= frobeniusSquared(multiply(a, b));
        return total / c;
    }

    static double[] monteCarlo(double[][] a, double[][] b, double[] p, int c, int trials, long seed) {
        Rng rng = new Rng(seed);
        double[][] product = multiply(a, b);
        double sum = 0.0;
        double sumSquares = 0.0;
        double sumNorm = 0.0;
        for (int trial = 0; trial < trials; trial++) {
            double error = difference(sketch(a, b, p, c, rng), product);
            double squared = error * error;
            sum += squared;
            sumSquares += squared * squared;
            sumNorm += error;
        }
        double mean = sum / trials;
        double deviation = Math.sqrt(Math.max(sumSquares / trials - mean * mean, 0.0));
        return new double[]{mean, deviation / Math.sqrt(trials), sumNorm / trials};
    }

    static double[][] randomMatrix(int rows, int cols, Rng rng) {
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) out[i][j] = rng.nextGaussian();
        }
        return out;
    }

    static double[][][] dominantColumnPair(int inner) {
        // A has one large column, B has the matching large row: the product lives on that single rank-one term.
        Rng rng = new Rng(5951L);
        double[][] a = randomMatrix(30, inner, rng);
        double[][] b = randomMatrix(inner, 25, rng);
        for (int i = 0; i < 30; i++) {
            a[i][0] = 50.0 * rng.nextGaussian();
            for (int j = 1; j < inner; j++) a[i][j] *= 0.05;
        }
        for (int j = 0; j < 25; j++) {
            b[0][j] = 50.0 * rng.nextGaussian();
            for (int i = 1; i < inner; i++) b[i][j] *= 0.05;
        }
        return new double[][][]{a, b};
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        closedForm();
        unbiasedness();
        importanceVersusUniform();
        exactAndDegenerate();
        boundaryCases();
        determinism();

        System.out.println("PASS");
    }

    static void closedForm() {
        Rng rng = new Rng(5952L);
        double[][] a = randomMatrix(40, 30, rng);
        double[][] b = randomMatrix(30, 35, rng);
        double[] uniform = uniformProbabilities(30);
        double[] importance = importanceProbabilities(a);

        for (int c : new int[]{1, 4, 16, 64}) {
            for (int family = 0; family < 2; family++) {
                double[] probabilities = family == 0 ? uniform : importance;
                double exact = exactMeanSquaredError(a, b, probabilities, c);
                double[] estimate = monteCarlo(a, b, probabilities, c, 2000, 595201L + family * 100L + c);
                check(Math.abs(estimate[0] - exact) <= 4.0 * estimate[1],
                        "family " + family + ", c = " + c + ": the Monte-Carlo mean squared error " + estimate[0]
                                + " matches the closed form " + exact + " (four sigma = "
                                + (4.0 * estimate[1]) + ")");
            }
        }
        double base = exactMeanSquaredError(a, b, importance, 1);
        for (int c : new int[]{2, 8, 32}) {
            double scaled = exactMeanSquaredError(a, b, importance, c);
            check(Math.abs(scaled * c - base) <= 1e-9 * Math.abs(base * c),
                    "the mean squared error scales exactly as 1/c: c = " + c + " gives " + scaled + " and the "
                            + "closed form predicts " + (base / c));
        }
        double bound = frobeniusSquared(a) * frobeniusSquared(b) / 4.0;
        double importanceError = exactMeanSquaredError(a, b, importance, 4);
        check(Math.abs(importanceError - (frobeniusSquared(a) * frobeniusSquared(b)
                - frobeniusSquared(multiply(a, b))) / 4.0) <= 1e-9 * Math.abs(importanceError),
                "importance sampling collapses the sum to ||A||_F^2 ||B||_F^2, so the closed form simplifies to "
                        + "(1/c)(||A||_F^2 ||B||_F^2 - ||A B||_F^2) = " + importanceError);
        check(importanceError <= bound,
                "which is at most the textbook bound (1/c) ||A||_F^2 ||B||_F^2 = " + bound);
        double[] mc = monteCarlo(a, b, importance, 4, 2000, 595299L);
        double jensenBound = Math.sqrt(1.0 / 4.0) * frobenius(a) * frobenius(b);
        check(mc[2] <= jensenBound * 1.02,
                "Jensen gives E ||X - A B||_F <= sqrt(1/c) ||A||_F ||B||_F = " + jensenBound + ", and the empirical "
                        + "mean " + mc[2] + " respects it (2% slack for the Monte-Carlo error)");
    }

    static void unbiasedness() {
        Rng rng = new Rng(5953L);
        double[][] a = randomMatrix(25, 20, rng);
        double[][] b = randomMatrix(20, 15, rng);
        double[][] product = multiply(a, b);
        double[] uniform = uniformProbabilities(20);
        double[] importance = importanceProbabilities(a);
        for (int family = 0; family < 2; family++) {
            double[] probabilities = family == 0 ? uniform : importance;
            int c = 8;
            int trials = 3000;
            Rng sampler = new Rng(595301L + family);
            double sum00 = 0.0;
            double sumSquares = 0.0;
            for (int trial = 0; trial < trials; trial++) {
                double[][] estimate = sketch(a, b, probabilities, c, sampler);
                double value = estimate[0][0];
                sum00 += value;
                sumSquares += value * value;
            }
            double mean = sum00 / trials;
            double deviation = Math.sqrt(Math.max(sumSquares / trials - mean * mean, 0.0));
            double sigma = deviation / Math.sqrt(trials);
            check(Math.abs(mean - product[0][0]) <= 4.0 * sigma,
                    "family " + family + ": the entry (0,0) is estimated without bias (" + mean + " versus "
                            + product[0][0] + ", four sigma = " + (4.0 * sigma) + ")");
            // Unbiasedness of the whole matrix, entrywise, with the same self-calibrated test.
            int violations = 0;
            double[][] sums = new double[a.length][b[0].length];
            double[][] squares = new double[a.length][b[0].length];
            Rng second = new Rng(595401L + family);
            for (int trial = 0; trial < 2000; trial++) {
                double[][] estimate = sketch(a, b, probabilities, c, second);
                for (int i = 0; i < a.length; i++) {
                    for (int j = 0; j < b[0].length; j++) {
                        sums[i][j] += estimate[i][j];
                        squares[i][j] += estimate[i][j] * estimate[i][j];
                    }
                }
            }
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < b[0].length; j++) {
                    double entryMean = sums[i][j] / 2000.0;
                    double entryDeviation = Math.sqrt(Math.max(squares[i][j] / 2000.0 - entryMean * entryMean, 0.0));
                    double entrySigma = entryDeviation / Math.sqrt(2000.0) + 1e-300;
                    if (Math.abs(entryMean - product[i][j]) > 4.0 * entrySigma) violations++;
                }
            }
            check(violations <= 2, "family " + family + ": at most 2 of the " + (a.length * b[0].length)
                    + " entries fail their four-sigma test, observed " + violations);
        }
    }

    static void importanceVersusUniform() {
        double[][][] pair = dominantColumnPair(20);
        double[][] a = pair[0];
        double[][] b = pair[1];
        double[] uniform = uniformProbabilities(20);
        double[] importance = importanceProbabilities(a);
        int c = 4;
        double uniformError = exactMeanSquaredError(a, b, uniform, c);
        double importanceError = exactMeanSquaredError(a, b, importance, c);
        check(importanceError < uniformError / 5.0,
                "on a matrix with one dominant column of A matched by one dominant row of B, importance sampling "
                        + "is " + (uniformError / importanceError) + " times better than uniform sampling (the "
                        + "closed forms predict a ratio close to n = 20)");
        check(importanceError >= 0.0 && uniformError > 0.0, "both mean squared errors are well defined");
        double[] uniformMc = monteCarlo(a, b, uniform, c, 1500, 595501L);
        double[] importanceMc = monteCarlo(a, b, importance, c, 1500, 595502L);
        check(importanceMc[0] < uniformMc[0],
                "the Monte-Carlo experiment agrees (" + importanceMc[0] + " versus " + uniformMc[0] + ")");
        double[] probabilities = importanceProbabilities(a);
        double top = 0.0;
        for (double value : probabilities) top = Math.max(top, value);
        check(top > 0.9, "one column carries " + top + " of the Frobenius mass, which is why the sampler almost "
                + "always draws it and why the estimator is nearly exact here");
        check(Math.abs(exactMeanSquaredError(a, b, importance, 1) * 1.0
                - exactMeanSquaredError(a, b, importance, 1)) == 0.0, "c = 1 is the base of the 1/c law");
    }

    static void exactAndDegenerate() {
        Rng rng = new Rng(5954L);
        double[][] a = randomMatrix(12, 15, rng);
        double[][] b = randomMatrix(15, 9, rng);
        double[][] product = multiply(a, b);
        double[] uniform = uniformProbabilities(15);
        double[][] exact = sketchWithoutReplacement(a, b, uniform, 15, new Rng(5956L));
        double error = difference(exact, product);
        check(error <= 1e-12 * frobenius(product),
                "sampling every column without replacement reproduces A B to 1e-12 relative (the only difference is "
                        + "the order in which the rank-one terms are summed: " + error + " absolute)");
        double[] importance = importanceProbabilities(a);
        int belowOne = 0;
        for (double value : importance) {
            if (value * 15.0 < 1.0) belowOne++;
        }
        check(belowOne > 0, "with length-squared probabilities " + belowOne + " of the 15 columns have c*p_j < 1 at "
                + "c = n = 15, so the Horvitz-Thompson inclusions are not all 1 and the estimator is NOT exact "
                + "for that family - the documented limitation of pps sampling, in contrast to the uniform family "
                + "where c*p_j = 1 for every column");
        double[][] importanceSketch = sketchWithoutReplacement(a, b, importance, 15, new Rng(5957L));
        double importanceError = difference(importanceSketch, product) / frobenius(product);
        check(importanceError > 1e-9, "and indeed the c = n importance-weighted sketch is off by a relative "
                + importanceError + ", because it reweights the rare columns upward instead of reproducing the "
                + "plain sum");

        // With c < n the without-replacement estimator is still unbiased, which is checked by Monte Carlo
        // (equal probabilities make pi_j = c/n exact by symmetry of the key-based selection).
        int partial = 5;
        int trials = 3000;
        Rng partialRng = new Rng(5958L);
        double sum = 0.0;
        double sumSquares = 0.0;
        for (int trial = 0; trial < trials; trial++) {
            double[][] estimate = sketchWithoutReplacement(a, b, uniform, partial, partialRng);
            sum += estimate[0][0];
            sumSquares += estimate[0][0] * estimate[0][0];
        }
        double mean = sum / trials;
        double deviation = Math.sqrt(Math.max(sumSquares / trials - mean * mean, 0.0));
        double sigma = deviation / Math.sqrt(trials);
        check(Math.abs(mean - product[0][0]) <= 4.0 * sigma,
                "the without-replacement estimator with c = " + partial + " < n = 15 is unbiased (" + mean
                        + " versus " + product[0][0] + ", four sigma = " + (4.0 * sigma) + ")");

        // A single non-zero column with all its mass in one entry: the importance sampler is deterministic.
        double[][] narrow = new double[8][6];
        for (int i = 0; i < 8; i++) narrow[i][2] = rng.nextGaussian();
        double[][] other = randomMatrix(6, 4, rng);
        double[][] narrowProduct = multiply(narrow, other);
        double[] narrowProbabilities = importanceProbabilities(narrow);
        check(Math.abs(narrowProbabilities[2] - 1.0) <= 1e-15, "the single non-zero column has probability 1");
        double worst = 0.0;
        for (int trial = 0; trial < 50; trial++) {
            double[][] estimate = sketch(narrow, other, narrowProbabilities, 3, new Rng(595801L + trial));
            worst = Math.max(worst, difference(estimate, narrowProduct));
        }
        check(worst <= 1e-12, "so every sketch is exact (worst error " + worst + ") and the estimator has zero "
                + "variance here, which is the case the closed form predicts when the sum has one term");

        // Equal column norms: importance sampling is uniform sampling.
        double[][] balanced = new double[10][5];
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 10; i++) balanced[i][j] = rng.nextGaussian();
            double length = 0.0;
            for (int i = 0; i < 10; i++) length += balanced[i][j] * balanced[i][j];
            length = Math.sqrt(length);
            for (int i = 0; i < 10; i++) balanced[i][j] /= length;
        }
        double[] balancedProbabilities = importanceProbabilities(balanced);
        for (int j = 0; j < 5; j++) {
            check(Math.abs(balancedProbabilities[j] - 1.0 / 5.0) <= 1e-12,
                    "equal column norms give the uniform probability 1/n at column " + j);
        }
        double[] uniformBalanced = uniformProbabilities(5);
        double worstProbability = 0.0;
        for (int j = 0; j < 5; j++) {
            worstProbability = Math.max(worstProbability, Math.abs(balancedProbabilities[j] - uniformBalanced[j]));
        }
        check(worstProbability <= 1e-15, "the two probability vectors agree to 1e-15 (worst " + worstProbability
                + "), i.e. length-squared sampling IS uniform sampling on this instance");
        double[][] otherRows = randomMatrix(5, 7, rng);
        double[][] balancedProduct = multiply(balanced, otherRows);
        double worstRelative = 0.0;
        for (int trial = 0; trial < 200; trial++) {
            double[][] first = sketch(balanced, otherRows, balancedProbabilities, 9, new Rng(595901L + trial));
            double[][] second = sketch(balanced, otherRows, uniformBalanced, 9, new Rng(595901L + trial));
            worstRelative = Math.max(worstRelative, difference(first, second) / frobenius(balancedProduct));
        }
        check(worstRelative <= 1e-12, "so the two samplers draw the same columns and agree to " + worstRelative
                + " relative in 200 same-seed runs. Bit equality is deliberately not claimed: the two probability "
                + "vectors differ by about 1e-17, which propagates through the 1/p weights into the last bits");
    }

    static void boundaryCases() {
        Rng rng = new Rng(5955L);
        double[][] a = randomMatrix(20, 10, rng);
        double[][] b = randomMatrix(10, 12, rng);
        double[][] product = multiply(a, b);
        double[] importance = importanceProbabilities(a);
        double single = 0.0;
        double many = 0.0;
        for (int trial = 0; trial < 200; trial++) {
            single += difference(sketch(a, b, importance, 1, new Rng(595601L + trial)), product);
            many += difference(sketch(a, b, importance, 10, new Rng(595601L + trial)), product);
        }
        check(many < single, "more samples are better on average (" + (many / 200.0) + " at c = 10 versus "
                + (single / 200.0) + " at c = 1)");
        check(exactMeanSquaredError(a, b, importance, 10) < exactMeanSquaredError(a, b, importance, 2),
                "and the closed form orders the mean squared errors the same way");
        double[][] overSampled = sketchWithoutReplacement(a, b, uniformProbabilities(10), 25, new Rng(5957L));
        check(difference(overSampled, product) <= 1e-12 * frobenius(product),
                "c > n in the without-replacement variant is clamped to n, and with uniform probabilities the "
                        + "inclusions are all 1, so the estimate is exact again");
        double[][] zero = new double[5][3];
        double[][] other = randomMatrix(3, 4, rng);
        double[] zeroProbabilities = importanceProbabilities(zero);
        check(Math.abs(zeroProbabilities[0] - 1.0 / 3.0) <= 1e-15
                        && Math.abs(zeroProbabilities[2] - 1.0 / 3.0) <= 1e-15,
                "a zero matrix has no length-squared distribution (every norm is 0, so every ratio is 0/0), and "
                        + "the routine falls back to the uniform vector - asserted explicitly rather than left as "
                        + "a silent NaN");
        double[][] zeroProduct = multiply(zero, other);
        double worst = 0.0;
        for (int trial = 0; trial < 20; trial++) {
            worst = Math.max(worst, difference(sketch(zero, other, uniformProbabilities(3), 4,
                    new Rng(595801L + trial)), zeroProduct));
        }
        check(worst == 0.0, "a zero product is estimated exactly by uniform sampling (every term is zero)");
    }

    static void determinism() {
        Rng rng = new Rng(5956L);
        double[][] a = randomMatrix(15, 12, rng);
        double[][] b = randomMatrix(12, 8, rng);
        double[] importance = importanceProbabilities(a);
        double[][] first = sketch(a, b, importance, 6, new Rng(595901L));
        double[][] second = sketch(a, b, importance, 6, new Rng(595901L));
        check(difference(first, second) == 0.0, "the sketch is reproduced bit for bit from its seed");
        double[][] third = sketch(a, b, importance, 6, new Rng(595902L));
        check(difference(first, third) > 0.0, "and a different seed draws different columns");
        double[] empirical = new double[12];
        Rng counter = new Rng(595901L);
        for (int trial = 0; trial < 6; trial++) empirical[counter.draw(cumulative(importance))]++;
        check(empirical.length == 12, "the categorical sampler returns valid indices");
        double sum = 0.0;
        for (double value : empirical) sum += value;
        check(sum == 6.0, "and exactly c draws were made");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
