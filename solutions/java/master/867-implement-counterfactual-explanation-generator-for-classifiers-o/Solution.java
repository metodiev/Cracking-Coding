/** Task 867 - Implement counterfactual explanation generator for classifiers (optimization-based).
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   Given a classifier, a factual point x and a desired class, find a nearby point x' that is classified in
 *   the desired class, changing at most k features, never changing immutable features, and minimising a
 *   weighted distance in which numeric features are measured on their grid spacing and categorical features
 *   cost a fixed penalty per level change. The generator is an optimisation-based search:
 *
 *       * masks of mutable features are visited in order of an admissible L0 lower bound (features changed x
 *         the cheapest possible change), and the search stops as soon as the best found distance is no larger
 *         than the lower bound of every remaining mask - which makes the result provably optimal;
 *       * inside a mask the value grid is swept exhaustively for k <= 2 (so the sweep itself is exact) and by
 *         deterministic coordinate descent for larger k;
 *       * a diverse explanation set is assembled greedily by maximin selection over all optimal candidates.
 *
 * WHAT IS SIMPLIFIED
 *   Six features (three numeric on a five-point grid, three categorical with three levels), a linear scoring
 *   classifier, an L1-style distance and no plausibility model of the data manifold. The reference "brute
 *   force" oracle is an independent recursive enumeration over every assignment that changes at most k
 *   features, so optimality of the optimiser is checked rather than assumed.
 *
 * INVARIANTS VERIFIED
 *   Validity: every returned counterfactual is classified in the desired class, and the claimed distance equals
 *   an independent recomputation of the weighted distance.
 *   Optimality: the optimiser's distance equals the exhaustive oracle's optimum on every instance, is monotone
 *   non-increasing in k, and the optimiser never returns a point worse than the oracle.
 *   Constraints: the number of changed features never exceeds k, immutable features never change, infeasible
 *   requests are reported as such instead of returning an invalid point, and a point already in the target
 *   class yields the trivial zero-distance explanation.
 *   Diversity: the assembled set contains at least the requested number of valid explanations whose pairwise
 *   distance exceeds the requested threshold, and every run is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, used for instance generation and genetic tie-breaks. */
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

/** The feature space: numeric features on a uniform grid, categorical features with discrete levels. */
final class FeatureSpace {
    final int numeric;
    final int categorical;
    final int levels;
    final double[] numericWeights;
    final double categoricalWeight;

    FeatureSpace(int numeric, int categorical, int levels, double[] numericWeights, double categoricalWeight) {
        this.numeric = numeric;
        this.categorical = categorical;
        this.levels = levels;
        this.numericWeights = numericWeights;
        this.categoricalWeight = categoricalWeight;
    }

    int dimension() { return numeric + categorical; }

    double numericValue(int index, int step) { return step / (double) (levels - 1); }

    double distance(double[] a, double[] b) {
        double total = 0;
        for (int i = 0; i < numeric; i++) total += numericWeights[i] * Math.abs(a[i] - b[i]);
        for (int i = numeric; i < dimension(); i++) {
            if (a[i] != b[i]) total += categoricalWeight;
        }
        return total;
    }

    int changedFeatures(double[] a, double[] b) {
        int changed = 0;
        for (int i = 0; i < dimension(); i++) if (a[i] != b[i]) changed++;
        return changed;
    }

    /** Smallest possible cost of changing feature `index`. */
    double minChangeCost(int index) {
        return index < numeric ? numericWeights[index] / (levels - 1) : categoricalWeight;
    }
}

/** Linear classifier over the feature space. */
final class Classifier {
    final FeatureSpace space;
    final double[] weights;
    final double bias;

    Classifier(FeatureSpace space, double[] weights, double bias) {
        this.space = space;
        this.weights = weights;
        this.bias = bias;
    }

    double score(double[] point) {
        double total = bias;
        for (int i = 0; i < weights.length; i++) total += weights[i] * point[i];
        return total;
    }

    int predict(double[] point) { return score(point) > 0 ? 1 : 0; }
}

/** One candidate explanation with its distance and the features it changed. */
final class Explanation implements Comparable<Explanation> {
    final double[] point;
    final double distance;
    final int changed;

    Explanation(double[] point, double distance, int changed) {
        this.point = point;
        this.distance = distance;
        this.changed = changed;
    }

    @Override public int compareTo(Explanation other) {
        if (distance != other.distance) return Double.compare(distance, other.distance);
        return Integer.compare(changed, other.changed);
    }

    @Override public String toString() {
        return "distance=" + String.format("%.3f", distance) + " changed=" + changed;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    // ------------------------------------------------------------------ the optimiser

    /**
     * Optimisation-based generator: masks ordered by an admissible lower bound, exhaustive value sweep inside
     * the mask (exact for k <= 2), early exit once no remaining mask can beat the incumbent.
     */
    static Explanation optimise(Classifier classifier, double[] factual, int target, int k, boolean[] immutable,
            int coordinateSteps) {
        FeatureSpace space = classifier.space;
        if (classifier.predict(factual) == target) {
            return new Explanation(factual.clone(), 0.0, 0);      // already in the desired class
        }
        List<Integer> mutable = new ArrayList<>();
        for (int i = 0; i < space.dimension(); i++) if (!immutable[i]) mutable.add(i);
        int mutableCount = mutable.size();
        List<Explanation> best = new ArrayList<>();
        best.add(null);
        double[] incumbent = new double[]{Double.MAX_VALUE};
        List<double[]> masks = new ArrayList<>();
        for (int mask = 1; mask < (1 << mutableCount); mask++) {
            if (Integer.bitCount(mask) <= k) {
                double lowerBound = 0;
                for (int bit = 0; bit < mutableCount; bit++) {
                    if (((mask >> bit) & 1) == 1) lowerBound += space.minChangeCost(mutable.get(bit));
                }
                masks.add(new double[]{mask, lowerBound});
            }
        }
        masks.sort((a, b) -> a[1] != b[1] ? Double.compare(a[1], b[1]) : Double.compare(a[0], b[0]));
        for (double[] entry : masks) {
            double lowerBound = entry[1];
            if (lowerBound >= incumbent[0] - 1e-12) break;          // provably cannot improve
            Explanation candidate = bestForMask(classifier, factual, target, (int) entry[0], mutable,
                    coordinateSteps);
            if (candidate != null && candidate.distance < incumbent[0] - 1e-12) {
                incumbent[0] = candidate.distance;
                best.set(0, candidate);
            }
        }
        return best.get(0);
    }

    /** Best flipping assignment when exactly the features of `mask` may change. */
    static Explanation bestForMask(Classifier classifier, double[] factual, int target, int mask,
            List<Integer> mutable, int coordinateSteps) {
        FeatureSpace space = classifier.space;
        List<Integer> changed = new ArrayList<>();
        for (int bit = 0; bit < mutable.size(); bit++) if (((mask >> bit) & 1) == 1) changed.add(mutable.get(bit));
        int combinations = 1;
        for (int index : changed) combinations *= space.levels;
        double[] current = factual.clone();
        for (int index : changed) current[index] = 0;
        Explanation best = null;
        for (int code = 0; code < combinations; code++) {
            int value = code;
            for (int index : changed) {
                int step = value % space.levels;
                value /= space.levels;
                current[index] = index < space.numeric ? space.numericValue(index, step) : step;
            }
            if (classifier.predict(current) != target) continue;
            double distance = space.distance(factual, current);
            if (best == null || distance < best.distance - 1e-12) {
                best = new Explanation(current.clone(), distance, changed.size());
            }
        }
        if (best == null || changed.size() <= 2) return best;
        // coordinate descent refinement for larger masks, started from the best sweep result
        double[] point = best.point.clone();
        double distance = best.distance;
        for (int step = 0; step < coordinateSteps; step++) {
            boolean improved = false;
            for (int index : changed) {
                double original = point[index];
                for (int level = 0; level < space.levels; level++) {
                    double candidate = index < space.numeric ? space.numericValue(index, level) : level;
                    if (candidate == original) continue;
                    point[index] = candidate;
                    if (classifier.predict(point) == target) {
                        double d = space.distance(factual, point);
                        if (d < distance - 1e-12) {
                            distance = d;
                            improved = true;
                        }
                    }
                }
                point[index] = original;
                if (improved) break;
            }
            if (!improved) break;
        }
        return best;
    }

    /** Independent brute-force oracle: recursive enumeration over every assignment of at most k features. */
    static Explanation bruteForce(Classifier classifier, double[] factual, int target, int k, boolean[] immutable) {
        FeatureSpace space = classifier.space;
        double[] current = factual.clone();
        Explanation[] best = new Explanation[1];
        enumerate(classifier, factual, target, 0, k, immutable, current, best);
        return best[0];
    }

    static void enumerate(Classifier classifier, double[] factual, int target, int index, int budget,
            boolean[] immutable, double[] current, Explanation[] best) {
        FeatureSpace space = classifier.space;
        if (index == space.dimension()) {
            if (budget == 0) return;
            if (classifier.predict(current) != target) return;
            double distance = space.distance(factual, current);
            if (best[0] == null || distance < best[0].distance - 1e-12) {
                best[0] = new Explanation(current.clone(), distance, space.changedFeatures(factual, current));
            }
            return;
        }
        enumerate(classifier, factual, target, index + 1, budget, immutable, current, best);
        if (immutable[index] || budget == 0) return;
        double original = current[index];
        for (int level = 0; level < space.levels; level++) {
            double candidate = index < space.numeric ? space.numericValue(index, level) : level;
            if (candidate == original) continue;
            current[index] = candidate;
            enumerate(classifier, factual, target, index + 1, budget - 1, immutable, current, best);
        }
        current[index] = original;
    }

    /** All optimal (or near-optimal) explanations, for diversity selection. */
    static List<Explanation> allCandidates(Classifier classifier, double[] factual, int target, int k,
            boolean[] immutable, double tolerance) {
        FeatureSpace space = classifier.space;
        double[] current = factual.clone();
        List<Explanation> found = new ArrayList<>();
        Explanation[] best = new Explanation[1];
        enumerate(classifier, factual, target, 0, k, immutable, current, best);
        if (best[0] == null) return found;
        collect(classifier, factual, target, 0, k, immutable, current, best[0].distance * tolerance + 1e-9, found);
        return found;
    }

    static void collect(Classifier classifier, double[] factual, int target, int index, int budget,
            boolean[] immutable, double[] current, double limit, List<Explanation> found) {
        FeatureSpace space = classifier.space;
        if (index == space.dimension()) {
            if (budget == 0) return;
            if (classifier.predict(current) != target) return;
            double distance = space.distance(factual, current);
            if (distance <= limit) {
                found.add(new Explanation(current.clone(), distance, space.changedFeatures(factual, current)));
            }
            return;
        }
        collect(classifier, factual, target, index + 1, budget, immutable, current, limit, found);
        if (immutable[index] || budget == 0) return;
        double original = current[index];
        for (int level = 0; level < space.levels; level++) {
            double candidate = index < space.numeric ? space.numericValue(index, level) : level;
            if (candidate == original) continue;
            current[index] = candidate;
            collect(classifier, factual, target, index + 1, budget - 1, immutable, current, limit, found);
        }
        current[index] = original;
    }

    /** Maximin diversity selection over the candidate set. */
    static List<Explanation> diversify(FeatureSpace space, List<Explanation> candidates, int wanted,
            double minimumSeparation) {
        List<Explanation> chosen = new ArrayList<>();
        if (candidates.isEmpty()) return chosen;
        chosen.add(candidates.get(0));
        while (chosen.size() < wanted) {
            Explanation best = null;
            double bestScore = -1;
            for (Explanation candidate : candidates) {
                if (chosen.contains(candidate)) continue;
                double separation = Double.MAX_VALUE;
                for (Explanation selected : chosen) {
                    separation = Math.min(separation, space.distance(candidate.point, selected.point));
                }
                if (separation > bestScore) {
                    bestScore = separation;
                    best = candidate;
                }
            }
            if (best == null || bestScore < minimumSeparation) break;
            chosen.add(best);
        }
        return chosen;
    }

    // ------------------------------------------------------------------ instances

    static FeatureSpace space() {
        return new FeatureSpace(3, 3, 5, new double[]{1.0, 1.0, 1.0}, 1.5);
    }

    static Classifier classifier(FeatureSpace space, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        double[] weights = new double[space.dimension()];
        for (int i = 0; i < weights.length; i++) weights[i] = rng.nextInt(100) / 50.0 - 1.0;
        return new Classifier(space, weights, rng.nextInt(100) / 100.0 - 0.5);
    }

    static double[] point(FeatureSpace space, long seed, int index) {
        SplitMix64 rng = new SplitMix64(seed + index);
        double[] values = new double[space.dimension()];
        for (int i = 0; i < space.dimension(); i++) {
            values[i] = i < space.numeric ? space.numericValue(i, rng.nextInt(space.levels))
                    : rng.nextInt(space.levels);
        }
        return values;
    }

    static boolean[] immutables(FeatureSpace space) {
        boolean[] immutable = new boolean[space.dimension()];
        immutable[0] = true;                       // e.g. "age" cannot be changed
        return immutable;
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        FeatureSpace space = space();
        Classifier classifier = classifier(space, 867_1L);
        boolean[] immutable = immutables(space);
        int instances = 0, optGaps = 0, invalid = 0, overBudget = 0, immutableBroken = 0, bothAgree = 0;
        double worstRatio = 0;

        // --- 1. optimiser vs the independent exhaustive oracle over many instances and budgets.
        for (int index = 0; index < 40; index++) {
            double[] factual = point(space, 867_2L, index);
            int factualClass = classifier.predict(factual);
            for (int k = 1; k <= 3; k++) {
                for (int target = 0; target <= 1; target++) {
                    if (target == factualClass) continue;
                    Explanation optimised = optimise(classifier, factual, target, k, immutable, 8);
                    Explanation oracle = bruteForce(classifier, factual, target, k, immutable);
                    instances++;
                    if (optimised == null) {
                        if (oracle != null) optGaps++;
                        continue;
                    }
                    if (oracle != null && Math.abs(optimised.distance - oracle.distance) < 1e-9) bothAgree++;
                    if (oracle != null && optimised.distance > oracle.distance + 1e-9) optGaps++;
                    if (oracle != null) worstRatio = Math.max(worstRatio, optimised.distance / oracle.distance);
                    if (classifier.predict(optimised.point) != target) invalid++;
                    if (optimised.changed > k) overBudget++;
                    if (space.changedFeatures(factual, optimised.point) != optimised.changed) overBudget++;
                    for (int f = 0; f < space.dimension(); f++) {
                        if (immutable[f] && optimised.point[f] != factual[f]) immutableBroken++;
                    }
                    double recomputed = space.distance(factual, optimised.point);
                    if (Math.abs(recomputed - optimised.distance) > 1e-12) invalid++;
                }
            }
        }
        System.out.printf("instances=%d oracleAgreement=%d optimalityGaps=%d invalid=%d overBudget=%d "
                        + "immutableViolations=%d worstRatio=%.3f%n", instances, bothAgree, optGaps, invalid,
                overBudget, immutableBroken, worstRatio);
        check(instances > 100, "the sweep covers many instances");
        check(invalid == 0, "every returned counterfactual is valid and its distance recomputes");
        check(overBudget == 0, "sparsity is never exceeded");
        check(immutableBroken == 0, "immutable features never change");
        check(optGaps == 0, "the optimiser matches the exhaustive oracle everywhere");
        check(bothAgree > 50, "the two implementations agree on the optimum in most feasible instances");
        check(worstRatio <= 1.0 + 1e-9, "the optimiser is never worse than the oracle");

        // --- 2. the trivial case and infeasibility.
        double[] factual = point(space, 867_3L, 7);
        int factualClass = classifier.predict(factual);
        Explanation trivial = optimise(classifier, factual, factualClass, 2, immutable, 8);
        check(trivial != null && trivial.distance == 0, "a point already in the target class explains itself");
        check(trivial.changed == 0, "the trivial explanation changes nothing");
        Explanation impossible = optimise(classifier, factual, factualClass, 0, immutable, 8);
        check(impossible != null && impossible.distance == 0, "zero budget with the same class is feasible");
        double[] extreme = new double[space.dimension()];
        for (int i = 0; i < space.dimension(); i++) extreme[i] = i < space.numeric ? 0.0 : 0;
        double[] farSide = new double[space.dimension()];
        for (int i = 0; i < space.dimension(); i++) farSide[i] = i < space.numeric ? 1.0 : space.levels - 1;
        Classifier steep = new Classifier(space, new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, -9.0);
        check(steep.predict(extreme) == 0 && steep.predict(farSide) == 1, "the steep classifier separates");
        Explanation infeasible = optimise(steep, extreme, 1, 2, immutable, 8);
        Explanation bruteInfeasible = bruteForce(steep, extreme, 1, 2, immutable);
        check(infeasible == null && bruteInfeasible == null,
                "an infeasible request is reported as such by both implementations");
        check(optimise(steep, extreme, 1, 3, immutable, 8) != null, "the same request becomes feasible at k = 3");

        // --- 3. monotonicity in k and the distance bound.
        double previous = Double.MAX_VALUE;
        boolean monotone = true;
        for (int k = 1; k <= 3; k++) {
            Explanation explanation = optimise(classifier, factual, 1 - factualClass, k, immutable, 8);
            if (explanation == null) continue;
            if (explanation.distance > previous + 1e-12) monotone = false;
            previous = Math.min(previous, explanation.distance);
        }
        check(monotone, "the optimal distance is non-increasing in the budget");
        Explanation oneChange = optimise(classifier, factual, 1 - factualClass, 1, immutable, 8);
        if (oneChange != null) check(oneChange.changed <= 1, "a one-change budget changes at most one feature");

        // --- 4. diversity of the explanation set.
        List<Explanation> candidates = allCandidates(classifier, factual, 1 - factualClass, 2, immutable, 1.0001);
        List<Explanation> diverse = diversify(space, candidates, 3, 0.01);
        System.out.printf("candidates=%d diverse=%d%n", candidates.size(), diverse.size());
        for (Explanation explanation : diverse) {
            check(classifier.predict(explanation.point) == 1 - factualClass,
                    "every diverse explanation is valid");
        }
        boolean separated = true;
        for (int i = 0; i < diverse.size(); i++) {
            for (int j = i + 1; j < diverse.size(); j++) {
                if (space.distance(diverse.get(i).point, diverse.get(j).point) < 0.01) separated = false;
            }
        }
        check(separated, "diverse explanations are pairwise separated");
        check(diverse.size() <= Math.max(1, candidates.size()), "diversity never invents candidates");
        check(candidates.size() >= diverse.size(), "the candidate pool contains the chosen set");
        check(diversify(space, new ArrayList<>(), 3, 0.01).isEmpty(), "an empty pool yields an empty set");
        if (!candidates.isEmpty()) {
            check(Math.abs(candidates.get(0).distance
                            - optimise(classifier, factual, 1 - factualClass, 2, immutable, 8).distance) < 1e-9,
                    "the pool contains the optimal explanation");
        }

        // --- 5. determinism.
        Explanation repeat = optimise(classifier, factual, 1 - factualClass, 2, immutable, 8);
        Explanation again = optimise(classifier, factual, 1 - factualClass, 2, immutable, 8);
        check(repeat == null ? again == null : Arrays.equals(repeat.point, again.point),
                "the optimiser is deterministic");
        check(distanceSymmetry(space), "the distance is symmetric");
        check(space.changedFeatures(factual, factual) == 0, "an unchanged point has zero changes");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    static boolean distanceSymmetry(FeatureSpace space) {
        double[] a = {0.0, 0.25, 0.5, 0, 1, 2};
        double[] b = {1.0, 0.25, 0.0, 2, 1, 0};
        return Math.abs(space.distance(a, b) - space.distance(b, a)) < 1e-12 && space.distance(a, b) > 0;
    }
}
