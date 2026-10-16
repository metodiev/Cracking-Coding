/** Task 769 - Implement knapsack approximation PTAS idea demonstration. (Tier: Expert, CTCI Ch 8 - Recursion and Dynamic Programming)
 *
 * 0/1 knapsack is weakly NP-hard, so no exact pseudo-polynomial algorithm is acceptable when the
 * capacity is huge; the PTAS idea is to *round the profits down* and then run the pseudo-polynomial
 * DP on the rounded profits, which keeps the DP table small at the price of a (1 - eps) factor.
 *
 * Two approximations are implemented next to the exact DP oracle:
 *
 *   fptasScaling   Lawler/Ibarra-Kim: K = max(1, floor(eps * Pmax / n)), p'_i = floor(p_i / K), then
 *                  an exact DP on the p'_i. Because the scaled optimum loses at most n values of "K",
 *                  the reconstructed set has true value >= OPT - K*n >= (1 - eps)*OPT. When
 *                  K == 1 (eps <= n/Pmax) the rounding is the identity, so the answer is exactly OPT.
 *   ptasGuessHeavy Sahni's "guess the heavy items" variant: the k = ceil(1/eps) largest-profit items
 *                  are guessed exhaustively (2^k subsets) and the remaining items are solved with the
 *                  scaling DP on the residual capacity. The branch in which the guess equals OPT's
 *                  intersection with the heavy set is feasible for the residual sub-problem, so the
 *                  result is again >= (1 - eps)*OPT - the error is confined to the light items.
 *
 * Verified: the exact DP equals a subset-enumeration brute force; both approximations return a
 * genuinely feasible subset whose true value (recomputed from the item list) equals the reported
 * value; the (1 - eps) guarantee and the OPT upper bound hold on every random instance; eps = 0 is
 * exact; the Ibarra-Kim Pareto-state DP (the state-trimming step the real schemes use) agrees with
 * the exact DP and its surviving states form a strict antichain; the ratio-greedy and the fractional
 * relaxation bracket OPT; and the degenerate instances (no items, zero capacity, zero-weight item,
 * everything heavier than the capacity) are correct.
 *
 * Simplified: instances are small (<= 48 items, capacity <= 400) so that the exact oracle and the
 * 2^k enumeration stay cheap; the PTAS is demonstrated on this scale rather than at production size.
 */
import java.util.*;

public class Solution {
    private static long state = 0x2545F4914F6CDD1DL;

    /** splitmix64: deterministic pseudo randomness, no Math.random anywhere. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static class Item {
        final int weight;
        final int value;
        Item(int weight, int value) { this.weight = weight; this.value = value; }
        public String toString() { return "(" + weight + "," + value + ")"; }
    }

    /** A candidate solution: the chosen index set plus the true value/weight of that set. */
    static class Plan {
        final boolean[] take;
        final long value;
        final long weight;
        Plan(boolean[] take, long value, long weight) { this.take = take; this.value = value; this.weight = weight; }
    }

    /** Exact 0/1 DP over the capacity; also the reconstruction the approximations reuse. */
    static Plan exact(List<Item> items, int capacity) {
        int n = items.size();
        long[][] dp = new long[n + 1][capacity + 1];
        for (int i = 1; i <= n; i++) {
            Item item = items.get(i - 1);
            for (int w = 0; w <= capacity; w++) {
                dp[i][w] = dp[i - 1][w];
                if (item.weight <= w) {
                    dp[i][w] = Math.max(dp[i][w], dp[i - 1][w - item.weight] + item.value);
                }
            }
        }
        boolean[] take = new boolean[n];
        int w = capacity;
        for (int i = n; i >= 1; i--) {
            if (dp[i][w] != dp[i - 1][w]) {
                take[i - 1] = true;
                w -= items.get(i - 1).weight;
            }
        }
        return materialise(items, take);
    }

    /** Subset enumeration: the independent oracle used to validate the DP. */
    static long bruteForceOptimum(List<Item> items, int capacity) {
        int n = items.size();
        long best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            long weight = 0, value = 0;
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    weight += items.get(i).weight;
                    value += items.get(i).value;
                }
            }
            if (weight <= capacity) best = Math.max(best, value);
        }
        return best;
    }

    /** Recomputes value and weight from the flags, so no check ever trusts a number we printed. */
    static Plan materialise(List<Item> items, boolean[] take) {
        long value = 0, weight = 0;
        for (int i = 0; i < items.size(); i++) {
            if (take[i]) {
                value += items.get(i).value;
                weight += items.get(i).weight;
            }
        }
        return new Plan(take.clone(), value, weight);
    }

    /** Largest profit among the items that can actually enter a solution of this capacity. */
    static int maxFittingValue(List<Item> items, int capacity) {
        int max = 0;
        for (Item item : items) {
            if (item.weight <= capacity) max = Math.max(max, item.value);
        }
        return max;
    }

    /**
     * The scaling factor K = max(1, floor(eps * Pmax / n)). Pmax is taken over the items that fit,
     * which is what makes Pmax <= OPT and therefore K*n <= eps*OPT for OPT > 0. The error bound the
     * analysis certifies is the stronger value >= OPT - K*n.
     */
    static long scalingK(List<Item> items, int capacity, double eps) {
        int n = items.size();
        if (n == 0) return 1L;
        return Math.max(1L, (long) (eps * maxFittingValue(items, capacity) / n));
    }

    /** Scaling PTAS: round profits down to multiples of K and solve that instance exactly. */
    static Plan fptasScaling(List<Item> items, int capacity, double eps) {
        int n = items.size();
        if (n == 0 || capacity <= 0) return new Plan(new boolean[n], 0, 0);
        long k = scalingK(items, capacity, eps);
        List<Item> rounded = new ArrayList<>();
        for (Item item : items) rounded.add(new Item(item.weight, (int) (item.value / k)));
        Plan scaled = exact(rounded, capacity);       // optimal for the rounded profits
        return materialise(items, scaled.take);       // ... but reported at the true profits
    }

    /** Sahni-style PTAS: guess which of the k heaviest items OPT keeps, then scale the rest. */
    static Plan ptasGuessHeavy(List<Item> items, int capacity, double eps) {
        int n = items.size();
        // The (1 - eps) guarantee holds for *any* k (the branch that guesses OPT's heavy items is
        // always enumerated and the residual is solved with the scaling guarantee), so cap k to keep
        // 2^k small; eps <= 0 needs no guessing at all because the residual DP is then exact.
        int k = eps <= 0 ? 0 : Math.min(n, Math.min(10, (int) Math.ceil(1.0 / eps)));
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n; i++) order.add(i);
        order.sort((a, b) -> Integer.compare(items.get(b).value, items.get(a).value));
        List<Integer> heavy = new ArrayList<>(order.subList(0, k));

        Plan best = new Plan(new boolean[n], 0, 0);
        for (int mask = 0; mask < (1 << k); mask++) {
            boolean[] take = new boolean[n];
            int used = 0;
            long guessedValue = 0;
            for (int b = 0; b < k; b++) {
                if ((mask & (1 << b)) != 0) {
                    int index = heavy.get(b);
                    take[index] = true;
                    used += items.get(index).weight;
                    guessedValue += items.get(index).value;
                }
            }
            if (used > capacity) continue;                // the guess alone does not fit
            List<Item> rest = new ArrayList<>();
            List<Integer> restIndex = new ArrayList<>();
            Set<Integer> heavySet = new HashSet<>(heavy);
            for (int i = 0; i < n; i++) {
                if (!heavySet.contains(i)) {
                    rest.add(items.get(i));
                    restIndex.add(i);
                }
            }
            Plan tail = fptasScaling(rest, capacity - used, eps);
            for (int i = 0; i < rest.size(); i++) {
                if (tail.take[i]) take[restIndex.get(i)] = true;
            }
            Plan candidate = materialise(items, take);
            if (candidate.value > best.value) best = candidate;
        }
        return best;
    }

    static List<Item> randomItems(int count, int maxWeight, int maxValue) {
        List<Item> items = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            items.add(new Item(1 + nextInt(maxWeight), 1 + nextInt(maxValue)));
        }
        return items;
    }

    /** Fractional knapsack (the LP relaxation): an upper bound on OPT, computed by ratio greedy. */
    static double fractionalUpperBound(List<Item> items, int capacity) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((a, b) -> Double.compare((double) b.value / b.weight, (double) a.value / a.weight));
        double total = 0;
        int remaining = capacity;
        for (Item item : sorted) {
            if (item.weight <= remaining) {
                remaining -= item.weight;
                total += item.value;
            } else {
                total += (double) item.value * remaining / item.weight;
                break;
            }
        }
        return total;
    }

    /** Ratio greedy without fractions: a feasible 0/1 solution, so a lower bound on OPT. */
    static long greedyLowerBound(List<Item> items, int capacity) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((a, b) -> Double.compare((double) b.value / b.weight, (double) a.value / a.weight));
        long total = 0;
        int remaining = capacity;
        for (Item item : sorted) {
            if (item.weight <= remaining) {
                remaining -= item.weight;
                total += item.value;
            }
        }
        return total;
    }

    static int paretoStates;
    static int paretoTrimmed;
    static long[] paretoFinalWeight = new long[0];
    static long[] paretoFinalValue = new long[0];

    /**
     * Ibarra-Kim style DP over Pareto-optimal (weight, value) states. A state that another state
     * matches or beats in both weight (no more) and value (no less) can never lead to a better
     * solution, so it is trimmed away; that trimming is what keeps the table small even when the
     * capacities are huge. Verified against the exact DP, which is a completely different algorithm.
     */
    static long paretoDp(List<Item> items, int capacity) {
        List<long[]> states = new ArrayList<>();
        states.add(new long[]{0, 0});
        int trimmed = 0;
        for (Item item : items) {
            List<long[]> next = new ArrayList<>(states);
            for (long[] state : states) {
                if (state[0] + item.weight <= capacity) {
                    next.add(new long[]{state[0] + item.weight, state[1] + item.value});
                }
            }
            next.sort((a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(b[1], a[1]));
            List<long[]> kept = new ArrayList<>();
            long bestValueSeen = -1;
            for (long[] state : next) {
                if (state[1] > bestValueSeen) {       // lighter states with >= value dominate this one
                    kept.add(state);
                    bestValueSeen = state[1];
                } else {
                    trimmed++;
                }
            }
            states = kept;
        }
        long best = 0;
        long previousWeight = -1;
        for (long[] state : states) {
            best = Math.max(best, state[1]);
            if (state[0] <= previousWeight) throw new IllegalStateException("states must be weight sorted");
            previousWeight = state[0];
        }
        paretoStates = states.size();
        paretoTrimmed = trimmed;
        paretoFinalWeight = new long[states.size()];
        paretoFinalValue = new long[states.size()];
        for (int i = 0; i < states.size(); i++) {
            paretoFinalWeight[i] = states.get(i)[0];
            paretoFinalValue[i] = states.get(i)[1];
        }
        return best;
    }

    public static void main(String[] args) {
        // --- degenerate instances ---------------------------------------------
        check(exact(new ArrayList<>(), 10).value == 0, "no items -> value 0");
        check(fptasScaling(new ArrayList<>(), 10, 0.1).value == 0, "no items -> FPTAS 0");
        check(ptasGuessHeavy(new ArrayList<>(), 10, 0.5).value == 0, "no items -> PTAS 0");

        List<Item> heavy = new ArrayList<>(List.of(new Item(11, 90), new Item(12, 80)));
        check(exact(heavy, 10).value == 0, "everything heavier than the capacity");
        check(fptasScaling(heavy, 10, 0.2).value == 0, "FPTAS respects the capacity");
        check(ptasGuessHeavy(heavy, 10, 0.5).value == 0, "PTAS respects the capacity");

        check(exact(heavy, 0).value == 0, "zero capacity -> value 0");

        List<Item> freebie = new ArrayList<>(List.of(new Item(0, 7), new Item(3, 5)));
        Plan freePlan = fptasScaling(freebie, 3, 0.1);
        check(freePlan.value == 12, "zero-weight item is free, got " + freePlan.value);
        check(freePlan.weight == 3, "zero-weight item adds no weight");

        List<Item> identical = new ArrayList<>();
        for (int i = 0; i < 7; i++) identical.add(new Item(2, 3));
        check(exact(identical, 6).value == 9, "three of seven identical items fit");
        check(fptasScaling(identical, 6, 0.5).value == 9, "FPTAS on identical items");
        check(ptasGuessHeavy(identical, 6, 0.5).value == 9, "PTAS on identical items");

        // --- the exact DP against the subset-enumeration oracle ----------------
        for (int trial = 0; trial < 12; trial++) {
            List<Item> items = randomItems(10 + trial % 5, 24, 50);
            int capacity = 40 + 13 * trial;
            check(exact(items, capacity).value == bruteForceOptimum(items, capacity),
                    "DP vs brute force, trial " + trial);
        }

        // --- eps = 0 is exact (the rounding becomes the identity) --------------
        List<Item> medium = randomItems(24, 30, 200);
        int capacity = 150;
        long optimum = exact(medium, capacity).value;
        check(optimum > 0, "the test instance must have a non-trivial optimum");
        check(fptasScaling(medium, capacity, 0.0).value == optimum, "eps = 0 gives the optimum");
        check(ptasGuessHeavy(medium, capacity, 0.0).value == optimum, "eps = 0 PTAS gives the optimum");

        // --- the (1 - eps) guarantee on random instances -----------------------
        double[] epsilons = {0.5, 0.25, 0.1};
        int guaranteesChecked = 0;
        int scalingHitOptimum = 0;
        for (int trial = 0; trial < 8; trial++) {
            List<Item> items = randomItems(18 + trial, 40, 500);
            int cap = 60 + 17 * trial;
            long exactValue = exact(items, cap).value;
            long k = scalingK(items, cap, 0.0);
            check(k == 1, "eps = 0 must reduce to the identity rounding");
            for (double eps : epsilons) {
                long factor = scalingK(items, cap, eps);
                Plan scaled = fptasScaling(items, cap, eps);
                check(scaled.value <= exactValue, "an approximation may never beat OPT (scaling, eps=" + eps + ")");
                check(scaled.value >= (1 - eps) * exactValue - 1e-9,
                        "scaling guarantee eps=" + eps + " got " + scaled.value + " vs OPT " + exactValue);
                check(scaled.value >= exactValue - factor * items.size(),
                        "sharper error bound OPT - K*n, eps=" + eps + " K=" + factor);
                check(scaled.weight <= cap, "scaling plan must be feasible");
                check(materialise(items, scaled.take).value == scaled.value,
                        "scaling plan value must be recomputable from its own flags");

                Plan guessed = ptasGuessHeavy(items, cap, eps);
                check(guessed.value <= exactValue, "an approximation may never beat OPT (PTAS, eps=" + eps + ")");
                check(guessed.value >= (1 - eps) * exactValue - 1e-9,
                        "PTAS guarantee eps=" + eps + " got " + guessed.value + " vs OPT " + exactValue);
                check(guessed.weight <= cap, "PTAS plan must be feasible");
                check(materialise(items, guessed.take).value == guessed.value,
                        "PTAS plan value must be recomputable from its own flags");
                guaranteesChecked += 2;
            }
            if (fptasScaling(items, cap, 0.05).value == exactValue) scalingHitOptimum++;
        }
        check(guaranteesChecked == 48, "all guarantee checks ran, got " + guaranteesChecked);
        check(scalingHitOptimum >= 4, "with fine rounding the PTAS should often be exact, hits=" + scalingHitOptimum);

        // --- Pareto-state DP against the exact DP, and the trimming it performs
        for (int trial = 0; trial < 10; trial++) {
            List<Item> items = randomItems(8 + 2 * trial, 26, 120);
            int cap = 30 + 9 * trial;
            long pareto = paretoDp(items, cap);
            check(pareto == exact(items, cap).value, "Pareto-state DP vs exact DP, trial " + trial);
            // the surviving states are an antichain: weight and value both strictly increase
            boolean antichain = true;
            for (int i = 1; i < paretoFinalWeight.length; i++) {
                if (paretoFinalWeight[i] <= paretoFinalWeight[i - 1]
                        || paretoFinalValue[i] <= paretoFinalValue[i - 1]) antichain = false;
            }
            check(antichain, "the trimmed state list is an antichain, trial " + trial);
        }
        check(paretoStates > 0, "the state list is never empty");
        check(paretoTrimmed > 0, "the trim step dropped states over the random trials");

        // a hand-made instance: (2,1) is dominated as a state by (1,100) and must be trimmed
        List<Item> dominated = new ArrayList<>(List.of(new Item(1, 100), new Item(2, 1)));
        check(paretoDp(dominated, 3) == 101, "both items fit and form the optimum, got " + paretoDp(dominated, 3));
        check(paretoTrimmed == 1, "exactly the dominated state is dropped, got " + paretoTrimmed);
        check(paretoDp(dominated, 1) == 100, "capacity 1 keeps only the cheap item");

        // the same DP on the rounded profits: zero-valued items must not break the trim
        List<Item> zeros = new ArrayList<>(List.of(
                new Item(1, 1), new Item(1, 1), new Item(1, 1), new Item(2, 9), new Item(3, 1)));
        List<Item> roundedZeros = new ArrayList<>();
        for (Item item : zeros) roundedZeros.add(new Item(item.weight, item.value / 3));
        check(paretoDp(roundedZeros, 3) == 3, "rounded profits with zeros, states kept");
        check(exact(roundedZeros, 3).value == 3, "rounded profits with zeros, exact DP agrees");

        // --- independent bounds: ratio greedy <= OPT <= fractional relaxation ----
        int boundChecks = 0;
        for (int trial = 0; trial < 6; trial++) {
            List<Item> items = randomItems(14 + 3 * trial, 30, 400);
            int cap = 45 + 12 * trial;
            long opt = exact(items, cap).value;
            double fractional = fractionalUpperBound(items, cap);
            double ratioGreedy = greedyLowerBound(items, cap);
            check(ratioGreedy <= opt + 1e-9, "ratio greedy cannot beat OPT, trial " + trial);
            check(fractional >= opt - 1e-9, "fractional LP must upper bound OPT, trial " + trial);
            for (double eps : new double[]{0.5, 0.2, 0.1}) {
                check(fptasScaling(items, cap, eps).value <= fractional + 1e-9,
                        "a feasible plan cannot beat the LP bound, trial " + trial + " eps=" + eps);
                boundChecks++;
            }
        }
        check(boundChecks == 18, "all LP-bound checks ran, got " + boundChecks);

        // --- an instance where rounding is clearly not exact --------------------
        List<Item> ties = new ArrayList<>(List.of(
                new Item(2, 100), new Item(1, 51), new Item(1, 51), new Item(3, 151)));
        check(exact(ties, 2).value == 102, "exact optimum of the ties instance");
        Plan roundedPlan = fptasScaling(ties, 2, 0.5);
        check(roundedPlan.value >= 51, "a coarse rounding still returns a good feasible plan");
        check(roundedPlan.value <= 102, "rounded plan cannot beat the exact optimum");
        check(roundedPlan.value >= 102 - 2L * scalingK(ties, 2, 0.5), "error bound on the ties instance");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
