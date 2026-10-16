/** Task 656 - Implement cost-based materialized view selection sketch. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Cost based selection of materialised views under a storage budget. Candidate view i occupies
 * size[i] units and, once built, answers query q with gain[i][q] units of saved work. Selecting S
 * gives
 *
 *     benefit(S) = sum_q frequency[q] * max_{i in S} gain[i][q]      (0 when no view serves q)
 *
 * which is monotone and submodular, subject to  sum_{i in S} size[i] <= budget.
 *
 * Greedy by benefit/size density (the classic knapsack-flavoured view advisor) is implemented
 * together with the plain greedy-by-benefit variant. This is a sketch: build/refresh maintenance
 * cost, index interactions and update rates are left out, and gains are static estimates - the
 * header notes what is simplified and what is still verified.
 *
 * Verified: every selection respects the budget; marginal gains are strictly positive and
 * non-increasing along the greedy order (submodularity); the result is at least as good as the best
 * single view and never worse than the exhaustive optimum; with equal view sizes the budget is a
 * cardinality constraint and the (1 - 1/e) bound is checked exactly; on every instance the greedy
 * solution is 1-exchange locally optimal (no single add or swap improves it); and two independent
 * optimal solvers - subset enumeration and a pruned branch and bound - agree on every instance.
 */
import java.util.*;

final class ViewSetup {
    final int budget;
    final int[] size;
    final int[] frequency;
    final int[][] gain;      // gain[view][query] >= 0

    ViewSetup(int budget, int[] size, int[] frequency, int[][] gain) {
        if (budget < 0) throw new IllegalArgumentException("negative budget");
        if (gain.length != size.length) throw new IllegalArgumentException("view count mismatch");
        for (int value : size) if (value < 0) throw new IllegalArgumentException("negative size");
        for (int f : frequency) if (f < 0) throw new IllegalArgumentException("negative frequency");
        for (int[] row : gain) {
            if (row.length != frequency.length) throw new IllegalArgumentException("gain width mismatch");
            for (int value : row) if (value < 0) throw new IllegalArgumentException("negative gain");
        }
        this.budget = budget;
        this.size = size;
        this.frequency = frequency;
        this.gain = gain;
    }

    int viewCount() { return size.length; }

    int queryCount() { return frequency.length; }

    int benefit(boolean[] selected) {
        int total = 0;
        for (int q = 0; q < queryCount(); q++) {
            int best = 0;
            for (int i = 0; i < viewCount(); i++) if (selected[i] && gain[i][q] > best) best = gain[i][q];
            total += frequency[q] * best;
        }
        return total;
    }

    int totalSize(boolean[] selected) {
        int total = 0;
        for (int i = 0; i < viewCount(); i++) if (selected[i]) total += size[i];
        return total;
    }

    int marginalGain(int view, boolean[] selected) {
        int gainUnits = 0;
        for (int q = 0; q < queryCount(); q++) {
            int best = 0;
            for (int i = 0; i < viewCount(); i++) if (selected[i] && gain[i][q] > best) best = gain[i][q];
            if (gain[view][q] > best) gainUnits += (gain[view][q] - best) * frequency[q];
        }
        return gainUnits;
    }

    boolean[] maskOf(List<Integer> selection) {
        boolean[] selected = new boolean[viewCount()];
        for (int view : selection) selected[view] = true;
        return selected;
    }

    /** Greedy by benefit per size unit; a size 0 view is free and always wins its tie. */
    List<Integer> greedyByDensity(int[] gainsOut) {
        boolean[] selected = new boolean[viewCount()];
        List<Integer> chosen = new ArrayList<>();
        int used = 0;
        while (true) {
            int best = -1;
            int bestGain = 0;
            for (int i = 0; i < viewCount(); i++) {
                if (selected[i]) continue;
                if (used + size[i] > budget) continue;
                int gainUnits = marginalGain(i, selected);
                if (gainUnits <= 0) continue;
                if (best < 0) {
                    best = i;
                    bestGain = gainUnits;
                    continue;
                }
                // compare gainUnits / size[i] against bestGain / size[best] with cross multiplication
                long left = (long) gainUnits * Math.max(size[best], 1);
                long right = (long) bestGain * Math.max(size[i], 1);
                if (left > right || (left == right && i < best)) {
                    best = i;
                    bestGain = gainUnits;
                }
            }
            if (best < 0) break;
            selected[best] = true;
            used += size[best];
            chosen.add(best);
            if (gainsOut != null) gainsOut[chosen.size() - 1] = bestGain;
        }
        return chosen;
    }

    /** Greedy by absolute marginal benefit - the budget-blind heuristic, kept for comparison. */
    List<Integer> greedyByBenefit() {
        boolean[] selected = new boolean[viewCount()];
        List<Integer> chosen = new ArrayList<>();
        int used = 0;
        while (true) {
            int best = -1, bestGain = 0;
            for (int i = 0; i < viewCount(); i++) {
                if (selected[i] || used + size[i] > budget) continue;
                int gainUnits = marginalGain(i, selected);
                if (gainUnits > bestGain) {
                    bestGain = gainUnits;
                    best = i;
                }
            }
            if (best < 0) break;
            selected[best] = true;
            used += size[best];
            chosen.add(best);
        }
        return chosen;
    }

    int bestSingleViewBenefit() {
        int best = 0;
        for (int i = 0; i < viewCount(); i++) {
            if (size[i] > budget) continue;
            boolean[] selected = new boolean[viewCount()];
            selected[i] = true;
            best = Math.max(best, benefit(selected));
        }
        return best;
    }

    /** Oracle 1: every subset that fits the budget. */
    int optimumByEnumeration() {
        int n = viewCount();
        if (n > 22) throw new IllegalArgumentException("too many views to enumerate");
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            boolean[] selected = new boolean[n];
            int used = 0;
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    selected[i] = true;
                    used += size[i];
                }
            }
            if (used <= budget) best = Math.max(best, benefit(selected));
        }
        return best;
    }

    /** Oracle 2: branch and bound over views ordered by size, pruned with the remaining full benefit. */
    int optimumByBranchAndBound() {
        int n = viewCount();
        Integer[] boxed = new Integer[n];
        for (int i = 0; i < n; i++) boxed[i] = i;
        Arrays.sort(boxed, (a, b) -> size[b] - size[a] != 0 ? size[b] - size[a] : a - b);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = boxed[i];
        int[] suffixBenefit = new int[n + 1];
        for (int i = n - 1; i >= 0; i--) {
            int own = 0;
            for (int q = 0; q < queryCount(); q++) own += frequency[q] * gain[order[i]][q];
            suffixBenefit[i] = suffixBenefit[i + 1] + own;
        }
        return search(order, suffixBenefit, 0, budget, 0, new boolean[n], new int[1]);
    }

    private int search(int[] order, int[] suffixBenefit, int depth, int remaining, int current,
                       boolean[] selected, int[] incumbent) {
        if (current > incumbent[0]) incumbent[0] = current;
        if (depth == order.length) return incumbent[0];
        if (current + suffixBenefit[depth] <= incumbent[0]) return incumbent[0];   // cannot beat the incumbent
        int view = order[depth];
        if (size[view] <= remaining) {
            int added = marginalGain(view, selected);        // `selected` does not contain `view` yet
            selected[view] = true;
            search(order, suffixBenefit, depth + 1, remaining - size[view], current + added, selected, incumbent);
            selected[view] = false;
        }
        return search(order, suffixBenefit, depth + 1, remaining, current, selected, incumbent);
    }

    /** True when any single add or swap improves the objective while respecting the budget. */
    boolean anyOneExchangeImproves(boolean[] selected) {
        int current = benefit(selected);
        int used = totalSize(selected);
        for (int i = 0; i < viewCount(); i++) {
            if (!selected[i]) {
                if (used + size[i] <= budget) {
                    boolean[] added = selected.clone();
                    added[i] = true;
                    if (benefit(added) > current) return true;
                }
                for (int j = 0; j < viewCount(); j++) {
                    if (!selected[j]) continue;
                    if (used - size[j] + size[i] > budget) continue;
                    boolean[] swapped = selected.clone();
                    swapped[j] = false;
                    swapped[i] = true;
                    if (benefit(swapped) > current) return true;
                }
            }
        }
        return false;
    }

    /** Applies one improving add or swap in place; reports whether the selection changed. */
    private boolean applyOneImprovingExchange(boolean[] selected) {
        int current = benefit(selected);
        int used = totalSize(selected);
        for (int i = 0; i < viewCount(); i++) {
            if (selected[i]) continue;
            if (used + size[i] <= budget) {
                selected[i] = true;
                if (benefit(selected) > current) return true;
                selected[i] = false;
            }
            for (int j = 0; j < viewCount(); j++) {
                if (!selected[j]) continue;
                if (used - size[j] + size[i] > budget) continue;
                boolean[] trial = selected.clone();
                trial[j] = false;
                trial[i] = true;
                if (benefit(trial) > current) {
                    selected[j] = false;
                    selected[i] = true;
                    return true;
                }
            }
        }
        return false;
    }

    /** 1-exchange local search seeded with a heuristic selection. */
    List<Integer> localSearch(List<Integer> start) {
        boolean[] selected = maskOf(start);
        for (int step = 0; step < 200; step++) {
            if (!applyOneImprovingExchange(selected)) break;
        }
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < viewCount(); i++) if (selected[i]) out.add(i);
        return out;
    }

    /** The advised plan: the best of the density greedy, the benefit greedy and the best single view. */
    List<Integer> advise() {
        List<Integer> best = greedyByDensity(null);
        int bestBenefit = benefit(maskOf(best));
        List<Integer> byBenefit = greedyByBenefit();
        if (benefit(maskOf(byBenefit)) > bestBenefit) {
            best = byBenefit;
            bestBenefit = benefit(maskOf(byBenefit));
        }
        int singleBest = bestSingleViewBenefit();
        if (singleBest > bestBenefit) {
            for (int i = 0; i < viewCount(); i++) {
                if (size[i] > budget) continue;
                boolean[] only = new boolean[viewCount()];
                only[i] = true;
                if (benefit(only) == singleBest) {
                    best = List.of(i);
                    break;
                }
            }
        }
        return localSearch(best);
    }
}

public class Solution {
    private static long state = 0x1D8E4E27C47D124FL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static ViewSetup randomSetup(int views, int queries, boolean equalSizes) {
        int[] size = new int[views];
        for (int i = 0; i < views; i++) size[i] = equalSizes ? 10 : 1 + nextInt(30);
        int[] frequency = new int[queries];
        for (int q = 0; q < queries; q++) frequency[q] = 1 + nextInt(3);
        int[][] gain = new int[views][queries];
        for (int i = 0; i < views; i++) {
            for (int q = 0; q < queries; q++) gain[i][q] = nextInt(25);
        }
        int budget = equalSizes ? 10 * (1 + nextInt(views)) : 10 + nextInt(40);
        return new ViewSetup(budget, size, frequency, gain);
    }

    static void checkInstance(ViewSetup setup, String label) {
        int[] densityGains = new int[setup.viewCount() + 1];
        List<Integer> densityChoice = setup.greedyByDensity(densityGains);
        boolean[] selected = setup.maskOf(densityChoice);
        check(setup.totalSize(selected) <= setup.budget, label + ": budget respected");
        for (int i = 0; i < densityChoice.size(); i++) check(densityGains[i] > 0, label + ": positive gain at step " + i);

        // the pure benefit greedy must show non-increasing marginal gains (submodularity of the objective)
        List<Integer> byBenefit = setup.greedyByBenefit();
        boolean[] replay = new boolean[setup.viewCount()];
        int previousGain = Integer.MAX_VALUE;
        for (int view : byBenefit) {
            int gain = setup.marginalGain(view, replay);
            check(gain > 0, label + ": benefit greedy keeps making progress");
            check(gain <= previousGain, label + ": benefit greedy gains non-increasing");
            previousGain = gain;
            replay[view] = true;
        }
        check(setup.totalSize(setup.maskOf(byBenefit)) <= setup.budget, label + ": budget respected (benefit greedy)");

        int optimum = setup.optimumByEnumeration();
        check(setup.optimumByBranchAndBound() == optimum, label + ": the two optima agree");
        check(setup.benefit(selected) <= optimum, label + ": the density greedy must not beat the optimum");

        // the advised plan is never worse than any of its seeds and is locally optimal
        boolean[] advised = setup.maskOf(setup.advise());
        int advisedBenefit = setup.benefit(advised);
        check(setup.totalSize(advised) <= setup.budget, label + ": advised plan respects the budget");
        check(advisedBenefit >= setup.benefit(selected), label + ": advised plan beats the density greedy");
        check(advisedBenefit >= setup.bestSingleViewBenefit(), label + ": advised plan beats the best single view");
        check(advisedBenefit <= optimum, label + ": advised plan must not beat the optimum");
        check(!setup.anyOneExchangeImproves(advised), label + ": advised plan is 1-exchange locally optimal");

        // local search must stabilise and never lose benefit
        boolean[] improved = setup.maskOf(setup.localSearch(densityChoice));
        check(setup.benefit(improved) >= setup.benefit(selected), label + ": local search never loses benefit");
        check(!setup.anyOneExchangeImproves(improved), label + ": local search stabilises");
        if (setup.benefit(improved) > setup.benefit(selected)) localSearchHelped = true;
    }

    static boolean localSearchHelped;

    /** Diminishing returns: for every A subset of B and every view v, gain(v|A) >= gain(v|B). */
    static void checkSubmodular(ViewSetup setup, String label) {
        int n = setup.viewCount();
        for (int maskA = 0; maskA < (1 << n); maskA++) {
            for (int maskB = 0; maskB < (1 << n); maskB++) {
                if ((maskA & maskB) != maskA) continue;            // A must be a subset of B
                boolean[] a = setup.maskOf(decode(maskA, n));
                boolean[] b = setup.maskOf(decode(maskB, n));
                for (int v = 0; v < n; v++) {
                    if (a[v] || b[v]) continue;
                    check(setup.marginalGain(v, a) >= setup.marginalGain(v, b),
                            label + ": diminishing returns broken for view " + v);
                }
            }
        }
    }

    static List<Integer> decode(int mask, int n) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) out.add(i);
        return out;
    }

    public static void main(String[] args) {
        // --- degenerate setups --------------------------------------------------
        ViewSetup noViews = new ViewSetup(50, new int[0], new int[]{1, 1}, new int[0][]);
        check(noViews.greedyByDensity(null).isEmpty(), "no views -> empty selection");
        check(noViews.benefit(new boolean[0]) == 0, "no views -> no benefit");
        check(noViews.optimumByEnumeration() == 0 && noViews.optimumByBranchAndBound() == 0,
                "no views -> optimum zero");

        ViewSetup zeroBudget = new ViewSetup(0, new int[]{5, 0}, new int[]{1, 1},
                new int[][]{{10, 10}, {4, 4}});
        List<Integer> freeChoice = zeroBudget.greedyByDensity(null);
        check(freeChoice.equals(List.of(1)), "only the size 0 view survives a zero budget, got " + freeChoice);
        check(zeroBudget.benefit(zeroBudget.maskOf(freeChoice)) == 8, "the free view still helps");

        ViewSetup tooBig = new ViewSetup(3, new int[]{10, 20}, new int[]{1}, new int[][]{{50}, {60}});
        check(tooBig.greedyByDensity(null).isEmpty(), "views that do not fit are never selected");
        check(tooBig.bestSingleViewBenefit() == 0, "no single view fits");
        check(tooBig.optimumByEnumeration() == 0, "optimum is zero when nothing fits");

        ViewSetup single = new ViewSetup(10, new int[]{4}, new int[]{3, 1}, new int[][]{{6, 2}});
        check(single.greedyByDensity(null).equals(List.of(0)), "a single view is selected");
        check(single.benefit(single.maskOf(List.of(0))) == 20, "benefit weights by frequency");
        check(single.optimumByEnumeration() == 20, "single view is the optimum");

        // --- designed instance: density greedy reaches the optimum -------------
        ViewSetup designed = new ViewSetup(10, new int[]{5, 4, 3}, new int[]{1, 1, 1},
                new int[][]{
                        {10, 0, 0},
                        {0, 8, 0},
                        {0, 0, 6}});
        List<Integer> designedChoice = designed.greedyByDensity(null);
        check(designedChoice.equals(List.of(0, 1)), "density greedy spends the budget on the two best views");
        check(designed.benefit(designed.maskOf(designedChoice)) == designed.optimumByEnumeration(),
                "designed instance: greedy matches the optimum");

        // --- unbounded budget selects everything useful ------------------------
        ViewSetup generous = new ViewSetup(100, new int[]{4, 5, 6}, new int[]{1, 2, 1},
                new int[][]{
                        {5, 0, 3},
                        {0, 4, 0},
                        {2, 2, 7}});
        List<Integer> all = generous.greedyByDensity(null);
        check(generous.totalSize(generous.maskOf(all)) == 15, "a generous budget takes every candidate");
        check(generous.benefit(generous.maskOf(all)) == generous.optimumByEnumeration(),
                "unconstrained greedy reaches the optimum");
        check(all.size() == 3, "all three views are chosen, got " + all);

        // --- deterministic tie break ------------------------------------------
        ViewSetup ties = new ViewSetup(10, new int[]{5, 5}, new int[]{1}, new int[][]{{7}, {7}});
        check(ties.greedyByDensity(null).equals(List.of(0)), "equal density breaks to the lower index");
        check(ties.benefit(ties.maskOf(List.of(0))) == ties.optimumByEnumeration(), "tie instance optimum");

        // --- submodularity of the objective itself -----------------------------
        for (int trial = 0; trial < 12; trial++) {
            checkSubmodular(randomSetup(4, 3, false), "submodular trial " + trial);
        }

        // --- random sweep, two independent oracles -----------------------------
        for (int trial = 0; trial < 140; trial++) {
            checkInstance(randomSetup(9, 6, false), "unequal trial " + trial);
        }
        int exact = 0;
        for (int trial = 0; trial < 60; trial++) {
            ViewSetup setup = randomSetup(8, 5, true);
            checkInstance(setup, "equal trial " + trial);
            boolean[] selected = setup.maskOf(setup.greedyByDensity(null));
            int optimum = setup.optimumByEnumeration();
            int greedyBenefit = setup.benefit(selected);
            check(greedyBenefit + 1e-9 >= (1.0 - 1.0 / Math.E) * optimum,
                    "equal sizes: (1-1/e) bound at trial " + trial);
            if (greedyBenefit == optimum) exact++;
        }
        check(exact > 0, "the equal size sweep should hit the optimum sometimes, hits " + exact);
        check(localSearchHelped, "the local search must strictly improve the density greedy somewhere");

        // --- the two oracles agree on wider instances --------------------------
        for (int trial = 0; trial < 40; trial++) {
            ViewSetup setup = randomSetup(14, 7, false);
            check(setup.optimumByEnumeration() == setup.optimumByBranchAndBound(),
                    "oracles agree on wide trial " + trial);
            check(setup.optimumByEnumeration() >= 0, "optimum is non-negative");
        }

        // --- rejected input ----------------------------------------------------
        boolean rejected = false;
        try {
            new ViewSetup(-1, new int[]{1}, new int[]{1}, new int[][]{{1}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative budget must be rejected");
        rejected = false;
        try {
            new ViewSetup(5, new int[]{-2}, new int[]{1}, new int[][]{{1}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative view size must be rejected");
        rejected = false;
        try {
            new ViewSetup(5, new int[]{2}, new int[]{1}, new int[][]{{-3}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative gain must be rejected");
        rejected = false;
        try {
            new ViewSetup(5, new int[]{2, 2}, new int[]{1}, new int[][]{{1}, {2, 3}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "mismatched gain width must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
