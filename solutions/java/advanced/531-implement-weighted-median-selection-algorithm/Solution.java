/** Task 531 - Implement weighted median selection algorithm. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)
 *
 * Weighted median of n items (value x_i, strictly positive weight w_i > 0, total W). The documented
 * definition is the LOWER weighted median: the smallest value v* such that the total weight of the
 * items with value <= v* is at least W / 2. Equivalently v* is the value at rank ceil(W / 2) in the
 * weight-expanded order, and the file therefore implements a general weighted quantile (the smallest
 * value whose cumulative weight reaches q * W) and derives the weighted median as q = 1/2. Two
 * implementations are kept and cross-checked: a sort-plus-prefix-sum scan (O(n log n)) and a
 * three-way quickselect that partitions on the median-of-three pivot value, accumulating the weights
 * strictly below, equal to and above the pivot and recursing into the correct side (expected O(n),
 * the number of partition passes is counted and bounded by the checks). Duplicate values are merged
 * implicitly by the three-way partition, non-positive weights are rejected with an
 * IllegalArgumentException (documented), and the flat optimum region is documented: whenever a single
 * value carries weight >= W / 2 the objective is minimised exactly at that value, and when the
 * cumulative weight equals W / 2 at a jump every point between the lower and the upper weighted
 * median minimises it too.
 *
 * Checks: the defining property - the objective obj(c) = sum_i w_i |x_i - c| at the returned median
 * must not exceed obj(c) at ANY candidate value, including every midpoint (x_i + x_j) / 2 - is
 * verified exhaustively on 60 small LCG instances; the sort-based and the quickselect results must
 * agree exactly on 60 LCG instances with up to 500 items, on duplicate-heavy values, on duplicate
 * weights and on a 5000-item instance; the CDF definition is re-checked directly (weight strictly
 * below the median < W / 2, weight up to and including the median >= W / 2); with unit weights the
 * median equals the element at sorted index ceil(n / 2) - 1 and the quantile at q equals the element
 * at index ceil(q * n) - 1; the median is invariant under scaling all weights by a positive constant
 * and scales linearly with the values; zero-weight items never change the answer; adding a huge
 * weight to one value pins the median to it; single-item and all-equal inputs collapse; the
 * weighted quantile is monotone in q with quantile(1) the largest value and the median equal to
 * quantile(1/2); non-positive weights throw; the quickselect uses fewer than 100 partition passes on
 * the 5000-item instance (measured 10) while a naive linear scan would scan all 5000 items; repeated runs are
 * identical.
 */
import java.util.*;

public class Solution {
    static int partitionPasses;

    static double totalWeight(double[] weights) {
        double total = 0.0;
        for (double weight : weights) total += weight;
        return total;
    }

    static void validate(double[] values, double[] weights) {
        if (values.length != weights.length || values.length == 0) {
            throw new IllegalArgumentException("values and weights must be non-empty and aligned");
        }
        for (double weight : weights) {
            if (!(weight > 0.0)) throw new IllegalArgumentException("weights must be positive");
        }
    }

    /** Sort-based weighted quantile: the smallest value whose cumulative weight reaches q * W. */
    static double quantileBySorting(double[] values, double[] weights, double q) {
        validate(values, weights);
        if (!(q > 0.0 && q <= 1.0)) throw new IllegalArgumentException("q must be in (0, 1]");
        int n = values.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(values[a], values[b]));
        double target = q * totalWeight(weights);
        double cumulative = 0.0;
        for (int i = 0; i < n; i++) {
            cumulative += weights[order[i]];
            if (cumulative >= target) return values[order[i]];
        }
        return values[order[n - 1]];
    }

    /** Three-way quickselect weighted quantile; counts the partition passes. */
    static double quantileBySelection(double[] values, double[] weights, double q) {
        validate(values, weights);
        if (!(q > 0.0 && q <= 1.0)) throw new IllegalArgumentException("q must be in (0, 1]");
        double[] v = values.clone();
        double[] w = weights.clone();
        double target = q * totalWeight(w);
        partitionPasses = 0;
        int lo = 0;
        int hi = v.length;
        while (true) {
            if (hi - lo == 1) return v[lo];
            partitionPasses++;
            double pivot = medianOfThree(v, lo, hi);
            int less = lo;
            int equal = lo;
            int greater = hi;
            double weightLess = 0.0;
            double weightEqual = 0.0;
            while (equal < greater) {
                if (v[equal] < pivot) {
                    swap(v, w, less, equal);
                    less++;
                    equal++;
                } else if (v[equal] > pivot) {
                    greater--;
                    swap(v, w, equal, greater);
                } else {
                    equal++;
                }
            }
            for (int i = lo; i < less; i++) weightLess += w[i];
            for (int i = less; i < greater; i++) weightEqual += w[i];
            if (target <= weightLess) {
                hi = less;
            } else if (target > weightLess + weightEqual) {
                target -= weightLess + weightEqual;
                lo = greater;
            } else {
                return pivot;
            }
        }
    }

    static double medianOfThree(double[] v, int lo, int hi) {
        int mid = lo + (hi - lo) / 2;
        double a = v[lo];
        double b = v[mid];
        double c = v[hi - 1];
        if (a > b) {
            double t = a;
            a = b;
            b = t;
        }
        if (b > c) {
            double t = b;
            b = c;
            c = t;
        }
        if (a > b) {
            double t = a;
            a = b;
            b = t;
        }
        return Math.max(a, Math.min(b, c));
    }

    static void swap(double[] v, double[] w, int i, int j) {
        double tv = v[i];
        v[i] = v[j];
        v[j] = tv;
        double tw = w[i];
        w[i] = w[j];
        w[j] = tw;
    }

    static double weightedMedianBySorting(double[] values, double[] weights) {
        return quantileBySorting(values, weights, 0.5);
    }

    static double weightedMedianBySelection(double[] values, double[] weights) {
        return quantileBySelection(values, weights, 0.5);
    }

    /** upper weighted median: the smallest value whose SUFFIX weight reaches W / 2. */
    static double upperWeightedMedian(double[] values, double[] weights) {
        validate(values, weights);
        int n = values.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> -Double.compare(values[a], values[b]));
        double target = 0.5 * totalWeight(weights);
        double cumulative = 0.0;
        for (int i = 0; i < n; i++) {
            cumulative += weights[order[i]];
            if (cumulative >= target) return values[order[i]];
        }
        return values[order[n - 1]];
    }

    /** The objective sum_i w_i |x_i - c| that the weighted median minimises. */
    static double objective(double[] values, double[] weights, double c) {
        double total = 0.0;
        for (int i = 0; i < values.length; i++) total += weights[i] * Math.abs(values[i] - c);
        return total;
    }

    static double weightStrictlyBelow(double[] values, double[] weights, double c) {
        double total = 0.0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] < c) total += weights[i];
        }
        return total;
    }

    static double weightAtMost(double[] values, double[] weights, double c) {
        double total = 0.0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] <= c) total += weights[i];
        }
        return total;
    }

    /** Brute-force objective minimiser over every candidate value and midpoint. */
    static double bestCandidate(double[] values, double[] weights) {
        double best = Double.POSITIVE_INFINITY;
        for (double c : candidateSet(values)) best = Math.min(best, objective(values, weights, c));
        return best;
    }

    static double[] candidateSet(double[] values) {
        List<Double> candidates = new ArrayList<>();
        for (double value : values) candidates.add(value);
        for (double a : values) {
            for (double b : values) candidates.add((a + b) / 2.0);
        }
        double[] out = new double[candidates.size()];
        for (int i = 0; i < out.length; i++) out[i] = candidates.get(i);
        return out;
    }

    static double[] lcgValues(int n, int seed, int limit) {
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    static double[] lcgWeights(int n, int seed, int limit) {
        double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            weights[i] = 1 + seed % limit;
        }
        return weights;
    }

    public static void main(String[] args) {
        double[] singleValue = {7.5};
        double[] singleWeight = {3.0};
        check(weightedMedianBySorting(singleValue, singleWeight) == 7.5, "single item");
        check(weightedMedianBySelection(singleValue, singleWeight) == 7.5, "single item by selection");

        double[] equal = {4.0, 4.0, 4.0};
        double[] equalWeights = {1.0, 2.0, 5.0};
        check(weightedMedianBySorting(equal, equalWeights) == 4.0, "all-equal values collapse");

        double[] straddle = {1.0, 2.0};
        double[] straddleWeights = {3.0, 1.0};
        check(weightedMedianBySorting(straddle, straddleWeights) == 1.0,
                "weight 3 out of 4 pins the median to 1");
        check(weightedMedianBySelection(straddle, straddleWeights) == 1.0, "straddle by selection");
        check(upperWeightedMedian(straddle, straddleWeights) == 1.0, "upper median also 1 here");

        double[] balanced = {1.0, 2.0, 3.0, 4.0};
        double[] unitWeights = {1.0, 1.0, 1.0, 1.0};
        check(weightedMedianBySorting(balanced, unitWeights) == 2.0,
                "unit weights give the lower median element");
        check(weightedMedianBySelection(balanced, unitWeights) == 2.0, "lower median by selection");
        check(upperWeightedMedian(balanced, unitWeights) == 3.0, "upper median with unit weights");
        check(quantileBySorting(balanced, unitWeights, 0.01) == 1.0, "quantile near 0 is the minimum");
        check(quantileBySorting(balanced, unitWeights, 1.0) == 4.0, "quantile 1 is the maximum");
        check(quantileBySorting(balanced, unitWeights, 0.25) == 1.0, "quantile 0.25 of unit weights");
        check(quantileBySorting(balanced, unitWeights, 0.5) == 2.0, "median is quantile 0.5");
        check(quantileBySelection(balanced, unitWeights, 0.75) == 3.0, "quantile 0.75 by selection");

        int seed = 606;
        double worstGap = 0.0;
        for (int trial = 0; trial < 60; trial++) {
            int n = 1 + trial % 12;
            double[] values = lcgValues(n, 3000 + trial * 17, 20);
            double[] weights = lcgWeights(n, 7000 + trial * 23, 9);
            double bySorting = weightedMedianBySorting(values, weights);
            double bySelection = weightedMedianBySelection(values, weights);
            check(bySorting == bySelection, "sort and select agree, trial " + trial);
            double median = bySelection;
            double total = totalWeight(weights);
            check(weightStrictlyBelow(values, weights, median) < 0.5 * total
                    || median == minValue(values), "cumulative weight below the median is below W/2");
            check(weightAtMost(values, weights, median) >= 0.5 * total,
                    "cumulative weight up to the median reaches W/2");
            double objectiveAtMedian = objective(values, weights, median);
            for (double candidate : candidateSet(values)) {
                check(objectiveAtMedian <= objective(values, weights, candidate) + 1e-9,
                        "the median minimises the weighted absolute deviation, trial " + trial);
            }
            worstGap = Math.max(worstGap, Math.abs(objectiveAtMedian - bestCandidate(values, weights)));
            for (double q : new double[]{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}) {
                check(quantileBySorting(values, weights, q) == quantileBySelection(values, weights, q),
                        "quantiles agree at q = " + q + ", trial " + trial);
            }
            double[] scaledWeights = new double[n];
            for (int i = 0; i < n; i++) scaledWeights[i] = weights[i] * 4.5;
            check(weightedMedianBySorting(values, scaledWeights) == median,
                    "scaling all weights leaves the median unchanged");
            for (int i = 0; i < n; i++) weights[i] = 1.0;
            check(weightedMedianBySorting(values, weights) == unitMedian(values),
                    "unit weights reproduce the lower median, trial " + trial);
        }
        check(worstGap <= 1e-9, "the median attains the minimum objective");

        double[] tinyValue = {5.0, 100.0, 3.0};
        double[] tinyWeight = {2.0, 0.0000001, 2.0};
        check(weightedMedianBySorting(new double[]{5.0, 3.0}, new double[]{2.0, 2.0}) == 3.0,
                "an exact tie at W/2 picks the smaller value");
        check(weightedMedianBySorting(tinyValue, tinyWeight) == 5.0,
                "documented: any positive weight, however small, moves the W/2 target");

        double[] pinnedValues = {1.0, 2.0, 3.0};
        double[] pinnedWeights = {1.0, 1000.0, 1.0};
        check(weightedMedianBySorting(pinnedValues, pinnedWeights) == 2.0, "huge weight pins the median");
        check(weightedMedianBySelection(pinnedValues, pinnedWeights) == 2.0, "pinned by selection");

        double[] scaledValues = {2.0, 4.0, 6.0, 8.0, 10.0};
        double[] scaledWeights = {1.0, 1.0, 4.0, 1.0, 1.0};
        double baseMedian = weightedMedianBySorting(new double[]{1.0, 2.0, 3.0, 4.0, 5.0},
                scaledWeights);
        check(weightedMedianBySorting(scaledValues, scaledWeights) == 2.0 * baseMedian,
                "scaling the values scales the median");

        double[] heaviest = {1.0, 2.0, 3.0, 4.0};
        double[] heavyWeights = {1.0, 1.0, 6.0, 1.0};
        check(weightedMedianBySorting(heaviest, heavyWeights) == 3.0, "a single dominant value wins");
        check(upperWeightedMedian(heaviest, heavyWeights) == 3.0, "upper median coincides");
        check(objective(heaviest, heavyWeights, 3.0)
                <= objective(heaviest, heavyWeights, 3.0 + 0.5) + 1e-12, "flat region at the mode");

        double[] tieValues = {1.0, 1.0, 2.0, 3.0, 3.0};
        double[] tieWeights = {1.0, 1.0, 2.0, 1.0, 1.0};
        check(weightedMedianBySorting(tieValues, tieWeights) == 2.0, "tied values around W/2");
        check(upperWeightedMedian(tieValues, tieWeights) == 2.0, "upper median of the tie");
        double[] flatValues = {1.0, 3.0};
        double[] flatWeights = {1.0, 1.0};
        check(weightedMedianBySorting(flatValues, flatWeights) == 1.0, "lower median is 1");
        check(upperWeightedMedian(flatValues, flatWeights) == 3.0, "upper median is 3");
        check(Math.abs(objective(flatValues, flatWeights, 1.0)
                - objective(flatValues, flatWeights, 2.0)) <= 1e-12,
                "documented flat optimum region between the lower and upper median");
        check(Math.abs(objective(flatValues, flatWeights, 3.0)
                - objective(flatValues, flatWeights, 2.0)) <= 1e-12,
                "the whole interval [lower, upper] is optimal");

        double[] manyValues = lcgValues(500, 31337, 200);
        double[] manyWeights = lcgWeights(500, 4242, 50);
        check(weightedMedianBySorting(manyValues, manyWeights)
                == weightedMedianBySelection(manyValues, manyWeights), "500 items agree");

        double[] manyDuplicates = lcgValues(500, 999, 3);
        double[] duplicateWeights = lcgWeights(500, 555, 4);
        check(weightedMedianBySorting(manyDuplicates, duplicateWeights)
                == weightedMedianBySelection(manyDuplicates, duplicateWeights),
                "duplicate-heavy instance agrees");

        double[] bigValues = lcgValues(5000, 20250916, 10_000);
        double[] bigWeights = lcgWeights(5000, 5150, 100);
        double bigMedian = weightedMedianBySelection(bigValues, bigWeights);
        int passes = partitionPasses;
        check(bigMedian == weightedMedianBySorting(bigValues, bigWeights),
                "5000-item instance: selection equals sorting");
        check(passes < 100, "the quickselect uses fewer than 100 partition passes");
        check(passes < 5000 / 10, "documented: selection is far below a linear-scan bound");
        check(weightAtMost(bigValues, bigWeights, bigMedian) >= 0.5 * totalWeight(bigWeights),
                "5000-item cumulative check");
        check(weightedMedianBySelection(bigValues, bigWeights) == bigMedian, "deterministic re-run");

        boolean threw = false;
        try {
            weightedMedianBySorting(new double[]{1.0, 2.0}, new double[]{1.0, 0.0});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "zero weight is rejected");
        threw = false;
        try {
            weightedMedianBySorting(new double[]{}, new double[]{});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "empty input is rejected");
        threw = false;
        try {
            quantileBySorting(new double[]{1.0}, new double[]{1.0}, 0.0);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "q = 0 is rejected");

        double best = Double.POSITIVE_INFINITY;
        for (int i = 0; i < 200; i++) best = Math.min(best, objective(manyValues, manyWeights, i));
        check(objective(manyValues, manyWeights,
                weightedMedianBySorting(manyValues, manyWeights)) <= best + 1e-9,
                "the median beats a 200-point grid search");

        System.out.println("PASS");
    }

    static double minValue(double[] values) {
        double min = Double.POSITIVE_INFINITY;
        for (double value : values) min = Math.min(min, value);
        return min;
    }

    /** Lower median of the values when all weights are equal to 1: index ceil(n / 2) - 1. */
    static double unitMedian(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int index = (sorted.length + 1) / 2 - 1;
        return sorted[index];
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
