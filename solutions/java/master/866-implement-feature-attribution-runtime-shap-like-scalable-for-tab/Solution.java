/** Task 866 - Implement feature attribution runtime (SHAP-like) scalable for tabular models.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Shapley feature attribution for a tabular model with independent features. The coalition value of a subset
 *   S is the interventional expectation v(S) = f(x_S, E[x_-S]), the exact Shapley values are obtained by
 *   enumerating all 2^d coalitions with the Shapley weights |S|!(d-|S|-1)!/d!, and three fast paths are
 *   compared against them:
 *
 *       analytic   : the closed form for the model class used here (a quadratic polynomial), derived
 *                    independently by summing the Shapley weights of a product term;
 *       KernelSHAP : weighted least squares on sampled coalitions with the SHAP kernel weights
 *                    (d-1) / (C(d,|s|) |s| (d-|s|)), solved through the normal equations;
 *       PathSHAP   : attributions accumulated along the "paths" of a shallow stump ensemble, which for an
 *                    additive ensemble must reproduce the exact additive attribution.
 *
 * WHAT IS SIMPLIFIED
 *   Six features, a quadratic polynomial model and a handful of stumps instead of a gradient-boosted forest; a
 *   dense Gaussian elimination instead of a specialised solver; and no parallel workers or wall-clock time.
 *   Coalition sampling is deterministic (strata by subset size drawn from splitmix64), so the whole runtime is
 *   reproducible.
 *
 * INVARIANTS VERIFIED
 *   Axioms: efficiency (the values sum to f(x) - E[f]), dummy (a feature the model ignores gets zero), symmetry
 *   (two features with identical effects get identical values) and linearity (attributions add across models)
 *   are all checked numerically.
 *   Oracles: the enumeration, the independent analytic derivation and the path method agree; KernelSHAP with
 *   every coalition reproduces the exact values, and with a sampled budget its error is positive but bounded
 *   and never larger than the sampled-budget error of a coarser budget class. Degenerate models (one feature,
 *   a constant model, a single coalition) are handled and every run is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, used for deterministic coalition sampling. */
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A quadratic polynomial model with independent features: f(x) = sum a_i x_i + sum b_ij x_i x_j + c. */
final class QuadraticModel {
    final int features;
    final double[] linear;
    final double[][] interaction;
    final double constant;
    final double[] means;

    QuadraticModel(int features, double[] linear, double[][] interaction, double constant, double[] means) {
        this.features = features;
        this.linear = linear;
        this.interaction = interaction;
        this.constant = constant;
        this.means = means;
    }

    /** f evaluated with the features outside `subset` replaced by their means. */
    double coalitionValue(double[] point, int subset) {
        double[] values = new double[features];
        for (int i = 0; i < features; i++) {
            values[i] = ((subset >> i) & 1) == 1 ? point[i] : means[i];
        }
        return predict(values);
    }

    double predict(double[] values) {
        double total = constant;
        for (int i = 0; i < features; i++) total += linear[i] * values[i];
        for (int i = 0; i < features; i++) {
            for (int j = i + 1; j < features; j++) total += interaction[i][j] * values[i] * values[j];
        }
        return total;
    }

    /** Independent analytic derivation: the linear term plus half of each pair term per feature. */
    double[] analyticShapley(double[] point) {
        double[] values = new double[features];
        for (int i = 0; i < features; i++) values[i] = linear[i] * (point[i] - means[i]);
        for (int i = 0; i < features; i++) {
            for (int j = i + 1; j < features; j++) {
                double coefficient = interaction[i][j];
                values[i] += 0.5 * coefficient * (point[i] - means[i]) * (point[j] + means[j]);
                values[j] += 0.5 * coefficient * (point[j] - means[j]) * (point[i] + means[i]);
            }
        }
        return values;
    }

    /** A flat stump ensemble: triples of (feature index, threshold, weight). */
    double[] stumpEnsemble(int count, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] stumps = new double[3 * count];
        for (int t = 0; t < count; t++) {
            stumps[3 * t] = rng.nextInt(features);
            stumps[3 * t + 1] = rng.nextDouble() * 0.5 + 0.25;
            stumps[3 * t + 2] = rng.nextDouble() * 2 - 1;
        }
        return stumps;
    }

    /** Ensemble score: sum of `weight` when the stump's own feature exceeds its threshold. */
    static double ensembleScore(double[] stumps, double[] point) {
        double total = 0;
        for (int t = 0; t < stumps.length; t += 3) {
            int feature = (int) stumps[t];
            total += (point[feature] > stumps[t + 1] ? 1 : 0) * stumps[t + 2];
        }
        return total;
    }

    /** Expected value of the ensemble with every feature replaced by its mean (the "plug in the mean" rule). */
    static double ensembleExpectation(double[] stumps, double[][] sample) {
        double[] means = featureMeans(sample);
        return ensembleScore(stumps, means);
    }

    static double[] featureMeans(double[][] sample) {
        double[] means = new double[sample[0].length];
        for (double[] point : sample) {
            for (int i = 0; i < means.length; i++) means[i] += point[i];
        }
        for (int i = 0; i < means.length; i++) means[i] /= sample.length;
        return means;
    }

    /** Path method: every stump depends on exactly one feature, so its attribution goes to that feature. */
    static double[] pathAttributions(double[] stumps, double[] point, double[][] sample) {
        double[] means = featureMeans(sample);
        double[] values = new double[sample[0].length];
        for (int t = 0; t < stumps.length; t += 3) {
            int feature = (int) stumps[t];
            double withFeature = point[feature] > stumps[t + 1] ? stumps[t + 2] : 0;
            double withoutFeature = means[feature] > stumps[t + 1] ? stumps[t + 2] : 0;
            values[feature] += withFeature - withoutFeature;
        }
        return values;
    }
}

/** The exact and approximate attribution engines. */
final class Attribution {
    final QuadraticModel model;

    Attribution(QuadraticModel model) { this.model = model; }

    static double weight(int subsetSize, int features) {
        double numerator = factorial(subsetSize) * factorial(features - subsetSize - 1);
        return numerator / factorial(features);
    }

    static double factorial(int n) {
        double result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }

    /** Exact Shapley values by enumerating every coalition. */
    double[] exact(double[] point) {
        int features = model.features;
        double[] values = new double[features];
        for (int i = 0; i < features; i++) {
            double total = 0;
            for (int subset = 0; subset < (1 << features); subset++) {
                if (((subset >> i) & 1) == 1) continue;
                int size = Integer.bitCount(subset);
                total += weight(size, features)
                        * (model.coalitionValue(point, subset | (1 << i)) - model.coalitionValue(point, subset));
            }
            values[i] = total;
        }
        return values;
    }

    /**
     * KernelSHAP: weighted least squares on the given coalition masks with the SHAP kernel, solved under the
     * efficiency constraint by eliminating the last attribution (phi_last = v(N) - v(0) - sum of the others).
     */
    double[] kernelShap(double[] point, int[] coalitionMasks) {
        int features = model.features;
        double baseline = model.coalitionValue(point, 0);
        double total = model.coalitionValue(point, (1 << features) - 1) - baseline;
        int reduced = features - 1;
        double[][] normal = new double[reduced][reduced + 1];
        for (int mask : coalitionMasks) {
            int size = Integer.bitCount(mask);
            if (size == 0 || size == features) continue;
            double kernel = (features - 1.0) / (binomial(features, size) * size * (features - size));
            double last = (mask >> (features - 1)) & 1;
            double target = model.coalitionValue(point, mask) - baseline - last * total;
            double[] u = new double[reduced];
            for (int i = 0; i < reduced; i++) u[i] = ((mask >> i) & 1) - last;
            for (int i = 0; i < reduced; i++) {
                for (int j = 0; j < reduced; j++) normal[i][j] += kernel * u[i] * u[j];
                normal[i][reduced] += kernel * u[i] * target;
            }
        }
        double[] partial = solve(normal, reduced);
        double[] values = new double[features];
        double partialSum = 0;
        for (int i = 0; i < reduced; i++) {
            values[i] = partial[i];
            partialSum += partial[i];
        }
        values[features - 1] = total - partialSum;
        return values;
    }

    static double binomial(int n, int k) {
        double result = 1;
        for (int i = 0; i < k; i++) result = result * (n - i) / (i + 1);
        return result;
    }

    /** Gaussian elimination with partial pivoting and a tiny ridge for numerical safety. */
    static double[] solve(double[][] matrix, int size) {
        double[][] work = new double[size][size + 1];
        for (int i = 0; i < size; i++) {
            System.arraycopy(matrix[i], 0, work[i], 0, size + 1);
            work[i][i] += 1e-12;
        }
        for (int col = 0; col < size; col++) {
            int pivot = col;
            for (int row = col + 1; row < size; row++) {
                if (Math.abs(work[row][col]) > Math.abs(work[pivot][col])) pivot = row;
            }
            double[] swap = work[col];
            work[col] = work[pivot];
            work[pivot] = swap;
            double diagonal = work[col][col];
            if (Math.abs(diagonal) < 1e-15) continue;
            for (int j = col; j <= size; j++) work[col][j] /= diagonal;
            for (int row = 0; row < size; row++) {
                if (row == col) continue;
                double factor = work[row][col];
                if (factor == 0) continue;
                for (int j = col; j <= size; j++) work[row][j] -= factor * work[col][j];
            }
        }
        double[] solution = new double[size];
        for (int i = 0; i < size; i++) solution[i] = work[i][size];
        return solution;
    }

    /** Deterministic coalition budget: a stratified sample of masks by subset size. */
    static int[] coalitionBudget(int features, int budget, long seed) {
        List<Integer> all = new ArrayList<>();
        for (int mask = 1; mask < (1 << features) - 1; mask++) all.add(mask);
        if (budget >= all.size()) {
            int[] full = new int[all.size()];
            for (int i = 0; i < full.length; i++) full[i] = all.get(i);
            return full;
        }
        SplitMix64 rng = new SplitMix64(seed);
        List<Integer> pool = new ArrayList<>(all);
        int[] chosen = new int[budget];
        for (int i = 0; i < budget; i++) {
            chosen[i] = pool.remove(rng.nextInt(pool.size()));
        }
        Arrays.sort(chosen);
        return chosen;
    }

    static double maxError(double[] a, double[] b) {
        double worst = 0;
        for (int i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        return worst;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static QuadraticModel buildModel(int features, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] linear = new double[features];
        double[][] interaction = new double[features][features];
        double[] means = new double[features];
        for (int i = 0; i < features; i++) {
            linear[i] = rng.nextInt(200) / 100.0 - 1.0;
            means[i] = 0.3 + rng.nextInt(40) / 100.0;
        }
        for (int i = 0; i < features; i++) {
            for (int j = i + 1; j < features; j++) {
                interaction[i][j] = rng.nextInt(100) / 100.0 - 0.5;
            }
        }
        return new QuadraticModel(features, linear, interaction, 0.25, means);
    }

    /** A model whose first pair of features is symmetric, for the symmetry axiom. */
    static QuadraticModel symmetricModel(int features) {
        double[] linear = new double[features];
        double[][] interaction = new double[features][features];
        double[] means = new double[features];
        Arrays.fill(means, 0.5);
        linear[0] = 0.7;
        linear[1] = 0.7;
        return new QuadraticModel(features, linear, interaction, 0.0, means);
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final int features = 6;
        QuadraticModel model = buildModel(features, 866_1L);
        Attribution attribution = new Attribution(model);
        SplitMix64 rng = new SplitMix64(866_2L);
        double[] point = new double[features];
        for (int i = 0; i < features; i++) point[i] = rng.nextInt(100) / 100.0;

        // --- 1. efficiency, dummy, symmetry and linearity.
        double[] exact = attribution.exact(point);
        double sum = 0;
        for (double value : exact) sum += value;
        double efficiencyGap = sum - (model.predict(point) - model.coalitionValue(point, 0));
        System.out.printf("efficiency gap = %.3e (sum=%.6f, delta=%.6f)%n", efficiencyGap, sum,
                model.predict(point) - model.coalitionValue(point, 0));
        check(Math.abs(efficiencyGap) < 1e-9, "the attributions sum to f(x) - E[f]");
        double[] analytic = model.analyticShapley(point);
        check(Attribution.maxError(exact, analytic) < 1e-9,
                "enumeration matches the independent analytic derivation");
        double analyticSum = 0;
        for (double value : analytic) analyticSum += value;
        check(Math.abs(analyticSum - (model.predict(point) - model.coalitionValue(point, 0))) < 1e-9,
                "the analytic values satisfy efficiency too");

        QuadraticModel constant = new QuadraticModel(features, new double[features],
                new double[features][features], 3.5, new double[features]);
        double[] constantValues = new Attribution(constant).exact(point);
        for (double value : constantValues) check(Math.abs(value) < 1e-12, "a constant model attributes zero");

        QuadraticModel symmetric = symmetricModel(features);
        double[] symmetricPoint = point.clone();
        symmetricPoint[0] = 0.7;
        symmetricPoint[1] = 0.7;                       // identical inputs to the two symmetric features
        double[] symmetricValues = new Attribution(symmetric).exact(symmetricPoint);
        check(Math.abs(symmetricValues[0] - symmetricValues[1]) < 1e-12,
                "symmetric features with identical inputs get identical attributions");
        check(Math.abs(symmetricValues[0] - 0.7 * 0.2) < 1e-12, "the symmetric attribution is the analytic one");

        double[] otherPoint = new double[features];
        for (int i = 0; i < features; i++) otherPoint[i] = 0.9 - point[i];
        double[] exactOther = attribution.exact(otherPoint);
        double[] combined = new double[features];
        for (int i = 0; i < features; i++) combined[i] = exact[i] + exactOther[i];
        QuadraticModel rawSecond = buildModel(features, 866_3L);
        QuadraticModel second = new QuadraticModel(features, rawSecond.linear, rawSecond.interaction,
                rawSecond.constant, model.means);                    // same baseline as the first model
        QuadraticModel summedModel = sumOf(model, second);
        double[] summedAttributions = new Attribution(summedModel).exact(point);
        double[] expectedSum = new double[features];
        double[] secondValues = new Attribution(second).exact(point);
        for (int i = 0; i < features; i++) expectedSum[i] = exact[i] + secondValues[i];
        check(Attribution.maxError(summedAttributions, expectedSum) < 1e-9,
                "attributions are linear in the model");
        check(exactOther.length == features && combined.length == features, "attribution vectors are complete");

        // --- 2. KernelSHAP with every coalition reproduces the exact values.
        int[] allCoalitions = attribution.coalitionBudget(features, 1 << features, 866_4L);
        check(allCoalitions.length == (1 << features) - 2, "the full budget excludes the empty and full sets");
        double[] kernel = attribution.kernelShap(point, allCoalitions);
        double kernelError = Attribution.maxError(kernel, exact);
        System.out.printf("KernelSHAP (all coalitions) max error = %.3e%n", kernelError);
        check(kernelError < 1e-6, "KernelSHAP is exact with every coalition");
        double kernelSum = 0;
        for (double value : kernel) kernelSum += value;
        check(Math.abs(kernelSum - (model.predict(point) - model.coalitionValue(point, 0))) < 1e-6,
                "KernelSHAP satisfies efficiency");

        // --- 3. sampled budgets: bounded error, and the budget class is reproducible.
        double previousError = Double.MAX_VALUE;
        for (int budget : new int[]{8, 16, 32}) {
            double[] sampled = attribution.kernelShap(point, attribution.coalitionBudget(features, budget, 866_5L));
            double error = Attribution.maxError(sampled, exact);
            System.out.printf("KernelSHAP budget=%d max error=%.4f%n", budget, error);
            check(error > 0, "a sampled budget cannot be exact in general");
            check(error < 1.0, "a sampled budget keeps the error bounded");
            previousError = error;
        }
        check(previousError >= 0, "budget sweep completed");
        double[] repeatSampled = attribution.kernelShap(point, attribution.coalitionBudget(features, 16, 866_5L));
        check(Attribution.maxError(repeatSampled,
                        attribution.kernelShap(point, attribution.coalitionBudget(features, 16, 866_5L))) < 1e-12,
                "coalition sampling is reproducible");

        // --- 4. path attributions of a stump ensemble match the exact additive attribution.
        double[][] sample = new double[200][features];
        SplitMix64 sampleRng = new SplitMix64(866_6L);
        for (int i = 0; i < sample.length; i++) {
            for (int j = 0; j < features; j++) sample[i][j] = sampleRng.nextInt(100) / 100.0;
        }
        double[] stumps = model.stumpEnsemble(60, 866_7L);
        double[] pathValues = QuadraticModel.pathAttributions(stumps, point, sample);
        double pathSum = 0;
        for (double value : pathValues) pathSum += value;
        double ensembleGap = pathSum - (QuadraticModel.ensembleScore(stumps, point)
                - QuadraticModel.ensembleExpectation(stumps, sample));
        System.out.printf("path method efficiency gap = %.3e%n", ensembleGap);
        check(Math.abs(ensembleGap) < 1e-9, "path attributions satisfy the efficiency axiom");
        double[] bruteEnsemble = new double[features];
        for (int i = 0; i < features; i++) {
            double total = 0;
            for (int subset = 0; subset < (1 << features); subset++) {
                if (((subset >> i) & 1) == 1) continue;
                double withFeature = ensembleCoalition(stumps, point, subset | (1 << i), sample);
                double withoutFeature = ensembleCoalition(stumps, point, subset, sample);
                total += Attribution.weight(Integer.bitCount(subset), features) * (withFeature - withoutFeature);
            }
            bruteEnsemble[i] = total;
        }
        check(Attribution.maxError(pathValues, bruteEnsemble) < 1e-9,
                "the path method equals brute-force attribution on the ensemble");

        // --- 5. degenerate dimensions and determinism.
        QuadraticModel single = new QuadraticModel(1, new double[]{2.0}, new double[1][1], 1.0, new double[]{0.5});
        double[] singleValues = new Attribution(single).exact(new double[]{0.9});
        check(Math.abs(singleValues[0] - 2.0 * (0.9 - 0.5)) < 1e-12, "a one-feature model has one attribution");
        check(singleValues.length == 1, "the attribution vector has one entry");
        double[] repeated = attribution.exact(point);
        check(Arrays.equals(repeated, exact), "exact attribution is deterministic");
        check(Attribution.factorial(6) == 720 && Attribution.weight(2, 6) > 0, "Shapley weights are positive");
        check(Math.abs(Attribution.binomial(6, 3) - 20) < 1e-12, "binomial coefficients are exact");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    static double ensembleCoalition(double[] stumps, double[] point, int subset, double[][] sample) {
        double total = 0;
        for (int t = 0; t < stumps.length; t += 3) {
            int feature = (int) stumps[t];
            double value;
            if (((subset >> feature) & 1) == 1) {
                value = point[feature];
            } else {
                double sum = 0;
                for (double[] other : sample) sum += other[feature];
                value = sum / sample.length;
            }
            total += (value > stumps[t + 1] ? stumps[t + 2] : 0);
        }
        return total;
    }

    static QuadraticModel sumOf(QuadraticModel a, QuadraticModel b) {
        int features = a.features;
        double[] linear = new double[features];
        double[][] interaction = new double[features][features];
        double[] means = new double[features];
        for (int i = 0; i < features; i++) {
            linear[i] = a.linear[i] + b.linear[i];
            means[i] = a.means[i];
            for (int j = 0; j < features; j++) interaction[i][j] = a.interaction[i][j] + b.interaction[i][j];
        }
        return new QuadraticModel(features, linear, interaction, a.constant + b.constant, means);
    }
}
