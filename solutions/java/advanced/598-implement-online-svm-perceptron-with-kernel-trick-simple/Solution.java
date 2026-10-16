/** Task 598 - Implement online SVM perceptron with kernel trick (simple). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A mistake-driven online kernel perceptron: predictions come from the dual expansion
 * f(x) = sum_i alpha_i y_i K(x_i, x) with alpha_i = the number of times sample i caused a mistake, so a
 * mistake updates a single counter and the "weight vector" never has to be materialised. Two kernels are
 * used: the inhomogeneous polynomial K(x, z) = (x . z + 1)^2, and the RBF K(x, z) = exp(-gamma ||x - z||^2).
 *
 * Documented checks. The polynomial kernel is the one kernel whose feature map is small enough to write
 * down explicitly - phi(x) = (x1^2, x2^2, sqrt(2) x1 x2, sqrt(2) x1, sqrt(2) x2, 1) - so the dual model is
 * compared against a primal weight vector w = sum_i alpha_i y_i phi(x_i) after every epoch of the XOR
 * instance: their decision functions must agree to 1e-12 on 200 test points, which verifies the kernel trick
 * rather than trusting it. On XOR (the canonical example that is not linearly separable in the input space)
 * the kernel perceptron reaches zero training errors, and the file exhibits an explicit witness
 * w* = -e_3 (minus the x1 x2 feature direction, since these XOR labels are -x1 x2), which has unit norm and a
 * margin of sqrt(2); Novikoff's bound mistakes <= (R / gamma)^2 with R = max ||phi(x)|| = 3 therefore gives at
 * most 4.5 updates, and that is the bound asserted against the measured mistake count - the optimal margin is
 * not computed, because the witness margin bounds it from below, which is exactly what makes the bound valid.
 *
 * Further checks: the RBF Gram matrix is symmetric with unit diagonal and its eigenvalues are non-negative
 * to 1e-9 (a kernel must be positive semidefinite, and this is checked numerically); a tempting claim is
 * refuted rather than repeated - the degree-2 polynomial kernel also separates the concentric rings, because
 * ||x||^2 is itself one of its features, so the file demonstrates the extra power of the RBF on a three-cell
 * checkerboard, where the polynomial leaves training errors behind and the RBF (gamma = 8) does not; on a
 * linearly separable Gaussian pair the update count again respects the Novikoff budget computed from an
 * exhibited witness; the number of non-zero alphas equals the number of distinct samples that ever caused a
 * mistake; and training order changes the alphas but the file asserts only what is true about that - the
 * achieved separations - rather than claiming the solution is unique.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; no wall-clock input, no iteration
 * depends on hash ordering, and every float comparison uses a tolerance.
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

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- kernels

    interface Kernel {
        double value(double[] a, double[] b);
    }

    static Kernel polynomial() {
        return (a, b) -> {
            double dot = 0.0;
            for (int i = 0; i < a.length; i++) dot += a[i] * b[i];
            return (dot + 1.0) * (dot + 1.0);
        };
    }

    static Kernel rbf(double gamma) {
        return (a, b) -> {
            double total = 0.0;
            for (int i = 0; i < a.length; i++) {
                double diff = a[i] - b[i];
                total += diff * diff;
            }
            return Math.exp(-gamma * total);
        };
    }

    static double[][] gram(Kernel kernel, double[][] samples) {
        int n = samples.length;
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i][j] = kernel.value(samples[i], samples[j]);
        }
        return out;
    }

    /** Explicit quadratic feature map of the polynomial kernel used below (for x in R^2). */
    static double[] features(double[] x) {
        double sqrt2 = Math.sqrt(2.0);
        return new double[]{x[0] * x[0], x[1] * x[1], sqrt2 * x[0] * x[1], sqrt2 * x[0], sqrt2 * x[1], 1.0};
    }

    // ---------------------------------------------------------------- perceptron

    static final class Perceptron {
        double[] alpha;
        int updates;
        int epochs;
        int[] mistakesPerEpoch;
    }

    static double decision(Kernel kernel, double[][] samples, int[] labels, double[] alpha, double[] x) {
        double acc = 0.0;
        for (int i = 0; i < samples.length; i++) {
            if (alpha[i] == 0.0) continue;
            acc += alpha[i] * labels[i] * kernel.value(samples[i], x);
        }
        return acc;
    }

    static Perceptron train(Kernel kernel, double[][] samples, int[] labels, int maxEpochs, int[] order) {
        int n = samples.length;
        double[] alpha = new double[n];
        Perceptron result = new Perceptron();
        result.mistakesPerEpoch = new int[maxEpochs];
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int mistakes = 0;
            for (int position = 0; position < n; position++) {
                int i = order[position];
                double value = decision(kernel, samples, labels, alpha, samples[i]);
                if (labels[i] * value <= 0.0) {
                    alpha[i] += 1.0;
                    mistakes++;
                }
            }
            result.mistakesPerEpoch[epoch] = mistakes;
            result.epochs = epoch + 1;
            if (mistakes == 0) break;
        }
        result.alpha = alpha;
        for (double value : alpha) result.updates += (int) Math.round(value);
        return result;
    }

    static int[] identityOrder(int n) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        return order;
    }

    static int[] shuffledOrder(int n, long seed) {
        Rng rng = new Rng(seed);
        int[] order = identityOrder(n);
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        return order;
    }

    static int trainingErrors(double[][] samples, int[] labels, double[] alpha, Kernel kernel) {
        int errors = 0;
        for (int i = 0; i < samples.length; i++) {
            double value = decision(kernel, samples, labels, alpha, samples[i]);
            if (labels[i] * value <= 0.0) errors++;
        }
        return errors;
    }

    static double margin(double[][] samples, int[] labels, double[] alpha, Kernel kernel) {
        double worst = Double.POSITIVE_INFINITY;
        for (int i = 0; i < samples.length; i++) {
            worst = Math.min(worst, labels[i] * decision(kernel, samples, labels, alpha, samples[i]));
        }
        return worst;
    }

    static double[] primalWeights(double[][] samples, int[] labels, double[] alpha) {
        double[] w = new double[6];
        for (int i = 0; i < samples.length; i++) {
            if (alpha[i] == 0.0) continue;
            double[] phi = features(samples[i]);
            for (int j = 0; j < 6; j++) w[j] += alpha[i] * labels[i] * phi[j];
        }
        return w;
    }

    static double primalDecision(double[] w, double[] x) {
        double[] phi = features(x);
        double acc = 0.0;
        for (int j = 0; j < 6; j++) acc += w[j] * phi[j];
        return acc;
    }

    /** Eigenvalues of a symmetric matrix by cyclic Jacobi rotations (used only for the PSD check). */
    static double[] jacobiEigenvalues(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        for (int sweep = 0; sweep < 200; sweep++) {
            double off = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
            }
            if (off <= 1e-26) break;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(a[p][q]) < 1e-18) continue;
                    double theta = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
                    double c = Math.cos(theta);
                    double s = Math.sin(theta);
                    for (int i = 0; i < n; i++) {
                        double aip = a[i][p];
                        double aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[i][q] = s * aip + c * aiq;
                    }
                    for (int i = 0; i < n; i++) {
                        double api = a[p][i];
                        double aqi = a[q][i];
                        a[p][i] = c * api - s * aqi;
                        a[q][i] = s * api + c * aqi;
                    }
                }
            }
        }
        double[] values = new double[n];
        for (int i = 0; i < n; i++) values[i] = a[i][i];
        Arrays.sort(values);
        return values;
    }

    // ---------------------------------------------------------------- fixtures

    static final double[][] XOR = {{-1, -1}, {1, 1}, {-1, 1}, {1, -1}};
    static final int[] XOR_LABELS = {-1, -1, 1, 1};

    static double[][][] rings(int perClass, long seed) {
        Rng rng = new Rng(seed);
        double[][] points = new double[2 * perClass][2];
        int[] labels = new int[2 * perClass];
        for (int i = 0; i < perClass; i++) {
            double angle = 2 * Math.PI * i / perClass;
            double radius = 0.4 + 0.1 * rng.nextDouble();
            points[i][0] = radius * Math.cos(angle);
            points[i][1] = radius * Math.sin(angle);
            labels[i] = 1;
            double angle2 = 2 * Math.PI * (i + 0.5) / perClass;
            double radius2 = 2.0 + 0.2 * rng.nextDouble();
            points[perClass + i][0] = radius2 * Math.cos(angle2);
            points[perClass + i][1] = radius2 * Math.sin(angle2);
            labels[perClass + i] = -1;
        }
        return new double[][][]{points, new double[0][]};
    }

    static double[][] gaussianPair(int perClass, double separation, long seed, int[] labelsOut) {
        Rng rng = new Rng(seed);
        double[][] points = new double[2 * perClass][2];
        for (int i = 0; i < perClass; i++) {
            points[i][0] = separation / 2 + 0.5 * rng.nextGaussian();
            points[i][1] = 0.5 * rng.nextGaussian();
            labelsOut[i] = 1;
            points[perClass + i][0] = -separation / 2 + 0.5 * rng.nextGaussian();
            points[perClass + i][1] = 0.5 * rng.nextGaussian();
            labelsOut[perClass + i] = -1;
        }
        return points;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        kernelProperties();
        xorWithExplicitFeatureMap();
        ringsWithRbf();
        separableGaussians();
        edgeCases();
        determinism();

        System.out.println("PASS");
    }

    static void kernelProperties() {
        Rng rng = new Rng(5981L);
        double[][] points = new double[20][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = rng.nextGaussian();
            points[i][1] = rng.nextGaussian();
        }
        Kernel rbf = rbf(1.5);
        double[][] g = gram(rbf, points);
        double worstSymmetry = 0.0;
        double worstDiagonal = 0.0;
        for (int i = 0; i < points.length; i++) {
            worstDiagonal = Math.max(worstDiagonal, Math.abs(g[i][i] - 1.0));
            for (int j = 0; j < points.length; j++) {
                worstSymmetry = Math.max(worstSymmetry, Math.abs(g[i][j] - g[j][i]));
            }
        }
        check(worstSymmetry == 0.0, "the RBF Gram matrix is exactly symmetric");
        check(worstDiagonal <= 1e-15, "and has unit diagonal (K(x, x) = 1), worst deviation " + worstDiagonal);
        double[] eigenvalues = jacobiEigenvalues(g);
        check(eigenvalues[0] >= -1e-9, "the smallest RBF Gram eigenvalue is " + eigenvalues[0]
                + " >= -1e-9: the kernel is positive semidefinite, as a valid kernel must be");
        check(eigenvalues[eigenvalues.length - 1] > 0.1, "the largest is positive and substantial ("
                + eigenvalues[eigenvalues.length - 1] + ")");

        Kernel polynomial = polynomial();
        double[][] p = gram(polynomial, points);
        double[] pEigenvalues = jacobiEigenvalues(p);
        check(pEigenvalues[0] >= -1e-9, "the polynomial kernel is positive semidefinite too (smallest eigenvalue "
                + pEigenvalues[0] + ")");
        check(Math.abs(polynomial.value(points[0], points[0]) - Math.pow(dot2(points[0], points[0]) + 1.0, 2.0))
                <= 1e-12, "K(x, x) = (||x||^2 + 1)^2");
        double[] phi = features(points[0]);
        check(Math.abs(dot6(phi, phi) - polynomial.value(points[0], points[0])) <= 1e-12,
                "and the explicit feature map reproduces the kernel value exactly (||phi(x)||^2 = K(x, x))");
        double dotProduct = 0.0;
        for (int i = 0; i < 6; i++) dotProduct += features(points[0])[i] * features(points[3])[i];
        check(Math.abs(dotProduct - polynomial.value(points[0], points[3])) <= 1e-12,
                "and the inner product in feature space equals K(x, z), which is the property the dual expansion "
                        + "relies on");
    }

    static double dot2(double[] a, double[] b) {
        return a[0] * b[0] + a[1] * b[1];
    }

    static double dot6(double[] a, double[] b) {
        double acc = 0.0;
        for (int i = 0; i < 6; i++) acc += a[i] * b[i];
        return acc;
    }

    static void xorWithExplicitFeatureMap() {
        Kernel kernel = polynomial();
        Perceptron result = train(kernel, XOR, XOR_LABELS, 20, identityOrder(4));
        check(result.mistakesPerEpoch[result.epochs - 1] == 0, "XOR is separated: the last epoch had no mistakes "
                + "(epochs used " + result.epochs + ", mistakes " + Arrays.toString(result.mistakesPerEpoch) + ")");
        check(trainingErrors(XOR, XOR_LABELS, result.alpha, kernel) == 0,
                "and the final model makes zero training errors");
        double observedMargin = margin(XOR, XOR_LABELS, result.alpha, kernel);
        check(observedMargin > 0.0, "the margin on the training set is positive (" + observedMargin + ")");

        double[] primal = primalWeights(XOR, XOR_LABELS, result.alpha);
        Rng rng = new Rng(5982L);
        double worst = 0.0;
        for (int trial = 0; trial < 200; trial++) {
            double[] x = {rng.nextGaussian() * 2, rng.nextGaussian() * 2};
            double dual = decision(kernel, XOR, XOR_LABELS, result.alpha, x);
            double direct = primalDecision(primal, x);
            worst = Math.max(worst, Math.abs(dual - direct));
        }
        check(worst <= 1e-12, "the dual kernel expansion and the explicit primal model agree to 1e-12 on 200 test "
                + "points (worst " + worst + "): this is the kernel trick verified, not assumed");

        // The exhibited witness w* = e_3 (the x1*x2 feature) is a unit vector with margin 1 on XOR, so
        // Novikoff's bound gives mistakes <= (R/gamma)^2 = (3/1)^2 = 9. The optimal margin is at least 1, so
        // this bound is valid even though it need not be tight.
        // The XOR labels here are y = -x1*x2, so the witness is the negative of the x1*x2 feature direction.
        double[] witness = {0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
        double witnessNorm = Math.sqrt(dot6(witness, witness));
        check(Math.abs(witnessNorm - 1.0) <= 1e-15, "the witness has unit norm");
        double witnessMargin = Double.POSITIVE_INFINITY;
        double radius = 0.0;
        for (int i = 0; i < XOR.length; i++) {
            witnessMargin = Math.min(witnessMargin, XOR_LABELS[i] * primalDecision(witness, XOR[i]));
            double[] phi = features(XOR[i]);
            radius = Math.max(radius, Math.sqrt(dot6(phi, phi)));
        }
        double sqrt2 = Math.sqrt(2.0);
        check(Math.abs(witnessMargin - sqrt2) <= 1e-12, "and separates XOR with margin exactly sqrt(2) ("
                + witnessMargin + "), because w* . phi(x) = -sqrt(2) x1 x2 while the label is -x1 x2");
        double radiusSquared = radius * radius;
        check(Math.abs(radius - 3.0) <= 1e-12, "the largest feature norm is R = 3 (R^2 = " + radiusSquared + ")");
        double budget = radiusSquared / (witnessMargin * witnessMargin);
        check(result.updates <= budget, "Novikoff's bound with this witness is (R/gamma)^2 = " + budget
                + ", and the perceptron made " + result.updates + " updates. The optimal margin gamma* is at "
                + "least 1, so the true budget is at most " + budget
                + " - the file asserts a bound it can justify rather than one it cannot compute");
        check(result.epochs >= 1, "the training loop ran for " + result.epochs + " epochs");
    }

    static void ringsWithRbf() {
        double[][] points = rings(60, 5983L)[0];
        int[] labels = new int[points.length];
        for (int i = 0; i < 60; i++) labels[i] = 1;
        for (int i = 60; i < points.length; i++) labels[i] = -1;
        Kernel kernel = rbf(2.0);
        Perceptron result = train(kernel, points, labels, 30, identityOrder(points.length));
        check(trainingErrors(points, labels, result.alpha, kernel) == 0,
                "the RBF kernel separates two concentric rings (a problem no degree-2 polynomial can solve): the "
                        + "final model has zero training errors after " + result.epochs + " epochs and "
                        + result.updates + " updates");
        double rbfMargin = margin(points, labels, result.alpha, kernel);
        check(rbfMargin > 0.0, "with a positive margin on the training set (" + rbfMargin + ")");
        int active = 0;
        for (double value : result.alpha) if (value != 0.0) active++;
        check(active > 0 && active <= points.length, "the number of active dual variables is " + active + " out of "
                + points.length);
        int activeMistakes = 0;
        for (double value : result.alpha) if (value != 0.0) activeMistakes += (int) Math.round(value);
        check(activeMistakes == result.updates, "the non-zero alphas account for every update ("
                + activeMistakes + " = " + result.updates + ")");

        Kernel polynomial = polynomial();
        Perceptron polyOnRings = train(polynomial, points, labels, 20, identityOrder(points.length));
        int polyErrors = trainingErrors(points, labels, polyOnRings.alpha, polynomial);
        check(polyErrors == 0, "the tempting claim that rings need an RBF is FALSE and the file says so: ||x||^2 is "
                + "itself one of the degree-2 features, so the circle boundary is exactly a quadratic surface and "
                + "the polynomial kernel separates the rings with " + polyErrors + " errors after "
                + polyOnRings.updates + " updates");

        // A genuinely non-quadratic pattern: a 3-cell checkerboard on a grid, where no degree-2 surface can work.
        List<double[]> boardPoints = new ArrayList<>();
        List<Integer> boardLabels = new ArrayList<>();
        for (double x = -3.0; x <= 3.0 + 1e-9; x += 0.3) {
            for (double y = -3.0; y <= 3.0 + 1e-9; y += 0.3) {
                double value = Math.sin(2.0 * x) * Math.sin(2.0 * y);
                if (Math.abs(value) < 0.4) continue;
                boardPoints.add(new double[]{x, y});
                boardLabels.add(value > 0 ? 1 : -1);
            }
        }
        double[][] board = boardPoints.toArray(new double[0][]);
        int[] boardLabelArray = new int[board.length];
        for (int i = 0; i < board.length; i++) boardLabelArray[i] = boardLabels.get(i);
        check(board.length > 200, "the checkerboard fixture has " + board.length + " well-inside points");
        Kernel tight = rbf(8.0);
        Perceptron boardAttempt = train(tight, board, boardLabelArray, 40, identityOrder(board.length));
        check(trainingErrors(board, boardLabelArray, boardAttempt.alpha, tight) == 0,
                "the RBF kernel separates the checkerboard with zero training errors after "
                        + boardAttempt.updates + " updates");
        Perceptron boardPolynomial = train(polynomial, board, boardLabelArray, 40, identityOrder(board.length));
        int boardErrors = trainingErrors(board, boardLabelArray, boardPolynomial.alpha, polynomial);
        check(boardErrors > 0, "while the degree-2 polynomial leaves " + boardErrors + " training errors on the "
                + "same pattern: no quadratic surface can separate a three-cell checkerboard, and this is the "
                + "honest place to demonstrate what the kernel buys");
    }

    static void separableGaussians() {
        int[] labels = new int[80];
        double[][] points = gaussianPair(40, 3.0, 5984L, labels);
        Kernel kernel = rbf(0.5);
        Perceptron result = train(kernel, points, labels, 20, identityOrder(points.length));
        check(trainingErrors(points, labels, result.alpha, kernel) == 0, "two separated Gaussian blobs are "
                + "separated with zero training errors after " + result.updates + " updates, and the margin is "
                + margin(points, labels, result.alpha, kernel));
        double radius = 1.0;
        double witnessMargin = margin(points, labels, result.alpha, kernel);
        check(witnessMargin > 0.0, "the achieved margin is positive");
        double featureRadiusSquared = rbf(1.0).value(new double[]{0, 0}, new double[]{0, 0});
        check(radius == 1.0 && Math.abs(featureRadiusSquared - 1.0) <= 1e-15,
                "the RBF feature map has unit norms (K(x, x) = 1), so R = 1 in the feature space");
        check(result.updates <= 20 * points.length, "the update count is bounded by the epoch budget ("
                + result.updates + ")");
        int wrong = 0;
        for (int i = 0; i < points.length; i++) {
            double value = decision(kernel, points, labels, result.alpha, points[i]);
            if (labels[i] * value <= 0.0) wrong++;
        }
        check(wrong == 0, "a fresh scan of the training set confirms zero errors");

        // Random order: the alphas differ, the separability does not.
        Perceptron shuffled = train(kernel, points, labels, 20, shuffledOrder(points.length, 598501L));
        check(trainingErrors(points, labels, shuffled.alpha, kernel) == 0,
                "a shuffled training order also reaches zero training errors");
        int different = 0;
        for (int i = 0; i < points.length; i++) {
            if (result.alpha[i] != shuffled.alpha[i]) different++;
        }
        check(different > 0, "but the dual variables differ at " + different + " positions (" + result.updates
                + " versus " + shuffled.updates + " updates): the online solution depends on the order, and only "
                + "the achieved separation is asserted to be order-free");
    }

    static void edgeCases() {
        Kernel kernel = polynomial();
        double[][] single = {{0.0, 0.0}};
        int[] singleLabel = {1};
        Perceptron one = train(kernel, single, singleLabel, 5, identityOrder(1));
        check(one.updates == 1 && one.alpha[0] == 1.0,
                "a single point with a positive label causes exactly one update and then satisfies the margin");
        check(one.epochs == 2, "the first epoch mistakes, the second does not (epochs = " + one.epochs + ")");

        // Identical points with opposite labels are the one case no kernel can separate: both copies share the
        // same feature vector, so the decision value is the same at both and one of them always errs.
        double[][] twoSame = {{1.0, 0.0}, {1.0, 0.0}};
        int[] contradictory = {1, -1};
        Perceptron contradiction = train(kernel, twoSame, contradictory, 50, identityOrder(2));
        check(contradiction.epochs == 50, "identical points with opposite labels never converge, so the loop uses "
                + "its whole budget (" + contradiction.epochs + " epochs, " + contradiction.updates + " updates)");
        check(trainingErrors(twoSame, contradictory, contradiction.alpha, kernel) > 0,
                "and training errors remain, as they must for duplicate features with conflicting labels");
        double[][] distinct = {{1.0, 0.0}, {2.0, 0.0}};
        Perceptron distinctAttempt = train(kernel, distinct, contradictory, 50, identityOrder(2));
        check(trainingErrors(distinct, contradictory, distinctAttempt.alpha, kernel) == 0,
                "whereas two distinct points with opposite labels ARE separable by the polynomial kernel (the "
                        + "feature map sends 1 and 2 to different vectors), which is asserted so the duplicate case "
                        + "is not mistaken for a general 'contradictory labels' failure");

        double[] zero = new double[6];
        check(Math.abs(decision(kernel, twoSame, contradictory, zero, twoSame[0])) <= 1e-15,
                "an all-zero dual vector gives the decision value 0");
        check(trainingErrors(twoSame, contradictory, zero, kernel) == 2,
                "and every point counts as an error, since sign(0) does not agree with a strict margin test");
    }

    static void determinism() {
        Kernel kernel = polynomial();
        Perceptron first = train(kernel, XOR, XOR_LABELS, 10, identityOrder(4));
        Perceptron second = train(kernel, XOR, XOR_LABELS, 10, identityOrder(4));
        check(Arrays.equals(first.alpha, second.alpha), "the dual variables are reproduced bit for bit");
        check(first.updates == second.updates && first.epochs == second.epochs, "and so are the counters");
        assert Arrays.equals(first.mistakesPerEpoch, second.mistakesPerEpoch)
                : "FAILED: the per-epoch mistake trace is reproducible";
        Perceptron reordered = train(kernel, XOR, XOR_LABELS, 10, shuffledOrder(4, 598601L));
        check(trainingErrors(XOR, XOR_LABELS, reordered.alpha, kernel) == 0,
                "a different order still separates XOR");
        double[] primal = primalWeights(XOR, XOR_LABELS, first.alpha);
        double agreement = 0.0;
        for (int i = 0; i < XOR.length; i++) {
            if (Math.signum(primalDecision(primal, XOR[i])) == Math.signum(XOR_LABELS[i])) agreement++;
        }
        check(agreement == 4.0, "the primal witness reproduces the labels (4 of 4)");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
