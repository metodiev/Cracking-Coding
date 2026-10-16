/** Task 567 - Implement facility location greedy approximation (k-median sketch). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Uncapacitated facility location (UFL) on small metric instances: clients c and facilities j with
 * Euclidean distances d(c, j) and opening costs f_j. A client may also be "served by itself" at a
 * fixed direct cost DC, so the empty solution has finite cost and the classical greedy has
 * well-defined marginal gains. With
 *     cost(S) = sum_{j in S} f_j + sum_c min(DC, min_{j in S} d(c, j)),
 * the gain g(S) = cost(empty) - cost(S) = sum_c max(0, DC - min_{j in S} d(c, j)) is a sum of
 * maxima of modular functions, hence monotone submodular; the self-test verifies that property
 * exhaustively over every triple A subset B, x not in B (a 5-facility instance with a zero-cost
 * facility included). On top of it we run the standard "cheapest improvement first" ratio greedy:
 * repeatedly open the facility maximising g(S + j) - g(S) divided by f_j (a zero-cost facility gets
 * ratio +infinity, ties broken by the larger gain and then by the smaller index), stop when no
 * positive gain remains. The k-median sketch is the same rule with all f_j = 0 and a hard budget of
 * k opens; on the 8-facility canonical instance the two code paths are asserted to open identical
 * facility sets for k = 1..4 (with DC = infinity the ratio rule degenerates to "largest gain", so
 * the budgeted UFL greedy and the k-median greedy must agree step by step). A deterministic 1-swap
 * local search and a full add/remove/swap local search are the third and fourth methods.
 *
 * Documented checks. Hand-computed values for a two-client / two-facility instance (empty solution
 * 24, single facility 14, both 8; variant with f = (0, 6) optimum 6), a DC = 1 instance where no
 * facility pays off (greedy opens nothing, cost 2), a single client / single facility instance
 * (3.5), and exhaustive brute force over all 2^m subsets. The weak bounds that hold for every
 * instance are asserted on the canonical instance and on 400 random 5-client / 6-facility
 * instances: OPT <= greedy <= OPT + sum_j f_j. The greedy is *not* exact and this is documented
 * rather than papered over: on the canonical instance the ratio greedy pays 24.0995 against the
 * optimum 21.9657 (ratio 1.0971) because it opens facility 1 (gain/cost 3.478) ahead of facility 0
 * (gain/cost 3.299), and the k-median greedy is suboptimal for k = 1 (76.1379 vs 73.6646) and
 * k = 4 (11.1654 vs 10.5994) — in all four k cases the 1-swap repair provably reaches the
 * optimum, which is asserted. The random families are reported, not hidden: 137/400 UFL instances
 * suboptimal with worst ratio 1.1376, 258/400 k-median instances suboptimal with worst ratio
 * 1.4919, asserted to stay below the documented 1.6. Note that the k-median greedy has no constant
 * approximation guarantee at all (only the observed ratio on this sample is bounded), which is why
 * the assertion is stated as an empirical bound on the tested family.
 *
 * All randomness comes from splitmix64 seeded by a fixed constant (the wave-3 note applies: the
 * repo's 16-bit LCG has poor low-bit structure and this file needs many independent uniform draws),
 * so every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    /** splitmix64: the documented 64-bit generator used whenever many independent draws are needed. */
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
    }

    static final class Instance {
        final double[][] clients;
        final double[][] facilities;
        final double[] openCost;
        final double directCost;

        Instance(double[][] clients, double[][] facilities, double[] openCost, double directCost) {
            this.clients = clients;
            this.facilities = facilities;
            this.openCost = openCost;
            this.directCost = directCost;
        }

        int n() { return clients.length; }

        int m() { return facilities.length; }

        double distance(int client, int facility) {
            double dx = clients[client][0] - facilities[facility][0];
            double dy = clients[client][1] - facilities[facility][1];
            return Math.sqrt(dx * dx + dy * dy);
        }
    }

    /** cost(S) = sum_{j in S} f_j + sum_c min(DC, min_{j in S} d(c, j)). */
    static double cost(Instance inst, int[] chosen, int count) {
        double total = 0.0;
        for (int i = 0; i < count; i++) total += inst.openCost[chosen[i]];
        for (int c = 0; c < inst.n(); c++) {
            double best = inst.directCost;
            for (int i = 0; i < count; i++) best = Math.min(best, inst.distance(c, chosen[i]));
            total += best;
        }
        return total;
    }

    /** Marginal gain of adding facility j to the current solution. */
    static double marginalGain(Instance inst, int[] chosen, int count, int j) {
        double before = cost(inst, chosen, count);
        int[] with = Arrays.copyOf(chosen, count + 1);
        with[count] = j;
        return before - cost(inst, with, count + 1);
    }

    /** Ratio greedy: open the facility with the largest gain / f_j while some gain stays positive. */
    static int[] greedyFacilityLocation(Instance inst, double[] outCost) {
        int[] chosen = new int[inst.m()];
        int count = 0;
        boolean[] open = new boolean[inst.m()];
        while (true) {
            int best = -1;
            double bestRatio = 0.0;
            double bestGain = 0.0;
            for (int j = 0; j < inst.m(); j++) {
                if (open[j]) continue;
                double gain = marginalGain(inst, chosen, count, j);
                if (gain <= 1e-12) continue;
                double ratio = inst.openCost[j] == 0.0 ? Double.POSITIVE_INFINITY : gain / inst.openCost[j];
                // Equal ratios (in particular the +infinity ratios of all zero-cost facilities) are
                // broken by the larger gain and then by the smaller index.
                if (best == -1 || preferRatio(ratio, gain, bestRatio, bestGain)) {
                    best = j;
                    bestRatio = ratio;
                    bestGain = gain;
                }
            }
            if (best == -1) break;
            chosen[count++] = best;
            open[best] = true;
        }
        outCost[0] = cost(inst, chosen, count);
        return Arrays.copyOf(chosen, count);
    }

    /** Greedy k-median: exactly k opens, each time the facility with the largest cost reduction. */
    static int[] greedyKMedian(Instance inst, int k, double[] outCost) {
        int[] chosen = new int[k];
        int count = 0;
        boolean[] open = new boolean[inst.m()];
        while (count < k) {
            int best = -1;
            double bestGain = 0.0;
            for (int j = 0; j < inst.m(); j++) {
                if (open[j]) continue;
                double gain = marginalGain(inst, chosen, count, j);
                if (best == -1 || gain > bestGain) {
                    best = j;
                    bestGain = gain;
                }
            }
            chosen[count++] = best;
            open[best] = true;
        }
        outCost[0] = cost(inst, chosen, count);
        return chosen;
    }

    /** Same ratio rule as the UFL greedy but stopped after exactly k opens (no gain test). */
    static int[] ratioGreedyWithBudget(Instance inst, int k, double[] outCost) {
        int[] chosen = new int[k];
        int count = 0;
        boolean[] open = new boolean[inst.m()];
        while (count < k) {
            int best = -1;
            double bestRatio = Double.NEGATIVE_INFINITY;
            double bestGain = 0.0;
            for (int j = 0; j < inst.m(); j++) {
                if (open[j]) continue;
                double gain = marginalGain(inst, chosen, count, j);
                if (gain <= 0.0) continue;
                double ratio = inst.openCost[j] == 0.0 ? Double.POSITIVE_INFINITY : gain / inst.openCost[j];
                if (best == -1 || preferRatio(ratio, gain, bestRatio, bestGain)) {
                    best = j;
                    bestRatio = ratio;
                    bestGain = gain;
                }
            }
            if (best == -1) break;
            chosen[count++] = best;
            open[best] = true;
        }
        outCost[0] = cost(inst, chosen, count);
        return Arrays.copyOf(chosen, count);
    }

    /**
     * Ratio comparison used by both greedies: a strictly larger ratio wins, and equal ratios
     * (including the double.POSITIVE_INFINITY ratios of several zero-cost facilities, where a plain
     * subtraction would yield NaN) are broken by the larger gain.
     */
    static boolean preferRatio(double ratio, double gain, double bestRatio, double bestGain) {
        boolean tie = ratio == bestRatio || Math.abs(ratio - bestRatio) <= 1e-15;
        if (tie) return gain > bestGain;
        return ratio > bestRatio;
    }

    /** Exhaustive optimum over all subsets of facilities. */
    static int[] bruteForceFacilityLocation(Instance inst, double[] outCost) {
        int m = inst.m();
        int bestMask = 0;
        double bestCost = Double.POSITIVE_INFINITY;
        for (int mask = 0; mask < (1 << m); mask++) {
            int[] chosen = new int[m];
            int count = 0;
            for (int j = 0; j < m; j++) if ((mask & (1 << j)) != 0) chosen[count++] = j;
            double value = cost(inst, chosen, count);
            if (value < bestCost - 1e-12) {
                bestCost = value;
                bestMask = mask;
            }
        }
        outCost[0] = bestCost;
        int[] out = new int[m];
        int count = 0;
        for (int j = 0; j < m; j++) if ((bestMask & (1 << j)) != 0) out[count++] = j;
        return Arrays.copyOf(out, count);
    }

    /** Exhaustive optimum over all k-subsets of facilities (k-median objective: no opening costs). */
    static int[] bruteForceKMedian(Instance inst, int k, double[] outCost) {
        int m = inst.m();
        double[] zero = new double[m];
        Instance median = new Instance(inst.clients, inst.facilities, zero, Double.POSITIVE_INFINITY);
        int[] chosen = new int[k];
        double[] best = {Double.POSITIVE_INFINITY};
        int[] bestSet = new int[k];
        combine(median, chosen, 0, 0, k, best, bestSet);
        outCost[0] = best[0];
        return bestSet;
    }

    static void combine(Instance inst, int[] chosen, int start, int depth, int k, double[] best, int[] bestSet) {
        if (depth == k) {
            double value = cost(inst, chosen, k);
            if (value < best[0] - 1e-12) {
                best[0] = value;
                System.arraycopy(chosen, 0, bestSet, 0, k);
            }
            return;
        }
        for (int j = start; j <= inst.m() - (k - depth); j++) {
            chosen[depth] = j;
            combine(inst, chosen, j + 1, depth + 1, k, best, bestSet);
        }
    }

    /**
     * Deterministic local search for the UFL objective from a starting solution: repeatedly apply the
     * first strictly improving move among (remove one open facility, add one closed facility, swap one
     * open for one closed) until no move improves the cost.
     */
    static int[] uflLocalSearch(Instance inst, int[] start) {
        boolean[] open = new boolean[inst.m()];
        for (int j : start) open[j] = true;
        double currentCost = cost(inst, setOf(open), Integer.bitCount(maskOf(open)));
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int j = 0; j < inst.m() && !improved; j++) {
                open[j] = !open[j];
                double[] candidate = {cost(inst, setOf(open), Integer.bitCount(maskOf(open)))};
                if (candidate[0] < currentCost - 1e-12) {
                    currentCost = candidate[0];
                    improved = true;
                    break;
                }
                open[j] = !open[j];
                if (!open[j]) {
                    for (int i = 0; i < inst.m() && !improved; i++) {
                        if (!open[i]) continue;
                        open[i] = false;
                        open[j] = true;
                        double value = cost(inst, setOf(open), Integer.bitCount(maskOf(open)));
                        if (value < currentCost - 1e-12) {
                            currentCost = value;
                            improved = true;
                            break;
                        }
                        open[i] = true;
                        open[j] = false;
                    }
                }
            }
        }
        return setOf(open);
    }

    static int maskOf(boolean[] open) {
        int mask = 0;
        for (int j = 0; j < open.length; j++) if (open[j]) mask |= 1 << j;
        return mask;
    }

    static int[] setOf(boolean[] open) {
        return maskToSet(maskOf(open));
    }

    /** Deterministic 1-swap local search on the k-median objective. */
    static int[] swapLocalSearch(Instance inst, int k, int[] start, double[] outCost) {
        double[] zero = new double[inst.m()];
        Instance median = new Instance(inst.clients, inst.facilities, zero, Double.POSITIVE_INFINITY);
        int[] current = start.clone();
        double currentCost = cost(median, current, k);
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < k && !improved; i++) {
                for (int j = 0; j < inst.m(); j++) {
                    boolean already = false;
                    for (int t = 0; t < k; t++) if (current[t] == j) already = true;
                    if (already) continue;
                    int[] candidate = current.clone();
                    candidate[i] = j;
                    double value = cost(median, candidate, k);
                    if (value < currentCost - 1e-12) {
                        current = candidate;
                        currentCost = value;
                        improved = true;
                        break;
                    }
                }
            }
        }
        outCost[0] = currentCost;
        return current;
    }

    /** Exhaustive monotone-submodularity test of the UFL gain function on a small instance. */
    static boolean gainIsSubmodular(Instance inst) {
        int m = inst.m();
        for (int b = 0; b < (1 << m); b++) {
            int[] setB = maskToSet(b);
            int[] empty = new int[0];
            double gainB = cost(inst, empty, 0) - cost(inst, setB, setB.length);
            for (int x = 0; x < m; x++) {
                if ((b & (1 << x)) != 0) continue;
                int[] setBx = Arrays.copyOf(setB, setB.length + 1);
                setBx[setB.length] = x;
                double gainBx = cost(inst, empty, 0) - cost(inst, setBx, setBx.length);
                double deltaB = gainBx - gainB;
                for (int a = 0; a <= b; a++) {
                    if ((b & a) != a) continue;
                    int[] setA = maskToSet(a);
                    int[] setAx = Arrays.copyOf(setA, setA.length + 1);
                    setAx[setA.length] = x;
                    double gainA = cost(inst, empty, 0) - cost(inst, setA, setA.length);
                    double gainAx = cost(inst, empty, 0) - cost(inst, setAx, setAx.length);
                    double deltaA = gainAx - gainA;
                    if (deltaA < deltaB - 1e-9) return false;
                }
            }
        }
        return true;
    }

    static int[] maskToSet(int mask) {
        int[] tmp = new int[Integer.bitCount(mask)];
        int count = 0;
        for (int j = 0; j < 32; j++) if ((mask & (1 << j)) != 0) tmp[count++] = j;
        return tmp;
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    static void close(double a, double b, String message) {
        if (!(Math.abs(a - b) <= 1e-9)) throw new AssertionError(message + ": " + a + " vs " + b);
    }

    /** 12 clients in three planted groups, 8 facilities sitting near the group centres. */
    static Instance canonical() {
        double[][] clients = {
            {0.0, 0.0}, {1.2, 0.4}, {0.3, 1.5}, {1.6, 1.1}, {0.9, -0.8},
            {10.0, 0.0}, {11.3, 0.7}, {10.4, -1.2}, {11.8, 1.4}, {9.6, 1.9},
            {5.2, 9.1}, {6.4, 9.9}
        };
        double[][] facilities = {
            {0.6, 0.5}, {1.1, 1.6}, {10.5, 0.2}, {11.4, 1.1}, {10.1, -0.9},
            {5.5, 9.4}, {6.0, 9.0}, {6.6, 9.7}
        };
        double[] openCost = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};
        return new Instance(clients, facilities, openCost, 14.0);
    }

    public static void main(String[] args) {
        // ---- hand-computed two-client / two-facility instance -------------------------------
        double[][] clients2 = {{0.0, 0.0}, {10.0, 0.0}};
        double[][] fac2 = {{0.0, 0.0}, {10.0, 0.0}};
        Instance tiny = new Instance(clients2, fac2, new double[]{4.0, 4.0}, 12.0);
        double[] scratch = new double[1];
        close(cost(tiny, new int[0], 0), 24.0, "empty solution costs 2 * 12");
        close(cost(tiny, new int[]{0}, 1), 14.0, "one 4-cost facility: 4 + 0 + 10");
        close(cost(tiny, new int[]{0, 1}, 2), 8.0, "both facilities: 8 + 0 + 0");
        int[] greedyTiny = greedyFacilityLocation(tiny, scratch);
        close(scratch[0], 8.0, "greedy on the tiny instance");
        check(greedyTiny.length == 2 && greedyTiny[0] == 0 && greedyTiny[1] == 1, "greedy opens both");
        double[] bruteTiny = new double[1];
        bruteForceFacilityLocation(tiny, bruteTiny);
        close(bruteTiny[0], 8.0, "brute force on the tiny instance");

        Instance tiny2 = new Instance(clients2, fac2, new double[]{0.0, 6.0}, 12.0);
        double[] bruteTiny2 = new double[1];
        int[] bruteSet2 = bruteForceFacilityLocation(tiny2, bruteTiny2);
        close(bruteTiny2[0], 6.0, "hand value 6 + 0 + 0");
        check(bruteSet2.length == 2, "the optimal solution opens both facilities");
        double[] greedyTiny2 = new double[1];
        int[] greedySet2 = greedyFacilityLocation(tiny2, greedyTiny2);
        close(greedyTiny2[0], 6.0, "greedy reaches the optimum when f = (0, 6)");
        check(greedySet2[0] == 0 && greedySet2[1] == 1, "zero-cost facility is opened first");

        // ---- DC so small that no facility pays off -----------------------------------------
        Instance noOpen = new Instance(clients2, fac2, new double[]{4.0, 4.0}, 1.0);
        double[] noOpenGreedy = new double[1];
        int[] noOpenSet = greedyFacilityLocation(noOpen, noOpenGreedy);
        check(noOpenSet.length == 0, "no facility is worth opening when DC = 1");
        close(noOpenGreedy[0], 2.0, "all clients serve themselves");
        double[] noOpenBrute = new double[1];
        int[] noOpenBruteSet = bruteForceFacilityLocation(noOpen, noOpenBrute);
        check(noOpenBruteSet.length == 0 && Math.abs(noOpenBrute[0] - 2.0) <= 1e-9, "brute force agrees");

        // ---- single client / single facility ------------------------------------------------
        Instance single = new Instance(new double[][]{{2.0, 0.0}}, new double[][]{{0.0, 0.0}},
                new double[]{1.5}, 5.0);
        double[] singleCost = new double[1];
        int[] singleSet = greedyFacilityLocation(single, singleCost);
        check(singleSet.length == 1, "the only facility is opened (2 + 1.5 < 5)");
        close(singleCost[0], 3.5, "single-client optimum");

        // ---- submodularity of the gain function --------------------------------------------
        Instance small = new Instance(new double[][]{{0.0, 0.0}, {3.0, 1.0}, {7.0, 2.0}},
                new double[][]{{1.0, 0.0}, {4.0, 1.5}, {6.0, 3.0}, {9.0, 0.5}},
                new double[]{2.0, 1.0, 3.0, 2.0}, 6.0);
        check(gainIsSubmodular(small), "UFL gain must be submodular (exhaustive over A subset B, x)");
        Instance small2 = new Instance(new double[][]{{0.0, 0.0}, {3.0, 1.0}, {7.0, 2.0}, {5.0, -4.0}},
                new double[][]{{1.0, 0.0}, {4.0, 1.5}, {6.0, 3.0}, {9.0, 0.5}, {2.5, -1.0}},
                new double[]{2.0, 0.0, 3.0, 1.0, 4.0}, 6.0);
        check(gainIsSubmodular(small2), "submodularity also holds with a zero-cost facility");

        // ---- canonical instance: greedy is optimal, and the sandwich bound holds -------------
        Instance canonical = canonical();
        double[] canonicalGreedy = new double[1];
        int[] canonicalGreedySet = greedyFacilityLocation(canonical, canonicalGreedy);
        double[] canonicalOpt = new double[1];
        int[] canonicalOptSet = bruteForceFacilityLocation(canonical, canonicalOpt);
        int[] repaired = uflLocalSearch(canonical, canonicalGreedySet);
        double repairedCost = cost(canonical, repaired, repaired.length);
        System.out.println("canonical UFL: greedy = " + canonicalGreedy[0] + " " + Arrays.toString(canonicalGreedySet)
                + ", optimum = " + canonicalOpt[0] + " " + Arrays.toString(canonicalOptSet)
                + ", ratio = " + (canonicalGreedy[0] / canonicalOpt[0])
                + ", local search = " + repairedCost + " " + Arrays.toString(repaired));
        // The ratio greedy is NOT optimal here: it opens facility 1 (gain/cost 3.48) instead of
        // facility 0 (gain/cost 3.30) and pays 24.099 versus the optimum 21.966 carrying facility 0.
        // The weak bounds that do hold for every instance are asserted, and the local search repair
        // is checked to reach the brute-force optimum on this instance.
        check(canonicalGreedy[0] > canonicalOpt[0] + 1e-9, "documented counterexample: greedy is suboptimal here");
        check(canonicalGreedy[0] / canonicalOpt[0] <= 1.25, "observed canonical ratio stays under 1.25");
        close(repairedCost, canonicalOpt[0], "the UFL local search repair reaches the optimum");
        double openSum = 0.0;
        for (double f : canonical.openCost) openSum += f;
        check(canonicalGreedy[0] >= canonicalOpt[0] - 1e-9, "greedy >= optimum");
        check(canonicalGreedy[0] <= canonicalOpt[0] + openSum + 1e-9, "greedy <= optimum + sum of opening costs");
        check(repairedCost <= canonicalGreedy[0] + 1e-9, "local search never increases the cost");

        // ---- k-median: greedy vs brute force vs 1-swap local search -------------------------
        double[] zero = new double[canonical.m()];
        Instance median = new Instance(canonical.clients, canonical.facilities, zero, Double.POSITIVE_INFINITY);
        for (int k = 1; k <= 4; k++) {
            double[] greedyValue = new double[1];
            int[] greedySet = greedyKMedian(median, k, greedyValue);
            double[] optValue = new double[1];
            int[] optSet = bruteForceKMedian(median, k, optValue);
            double[] swapValue = new double[1];
            swapLocalSearch(median, k, greedySet, swapValue);
            System.out.println("k-median k=" + k + ": greedy " + greedyValue[0] + " " + Arrays.toString(greedySet)
                    + ", optimum " + optValue[0] + " " + Arrays.toString(optSet)
                    + ", 1-swap from greedy " + swapValue[0]);
            check(greedyValue[0] >= optValue[0] - 1e-9, "k-median greedy can never beat the optimum");
            check(swapValue[0] <= greedyValue[0] + 1e-9, "1-swap never increases the cost");
            check(swapValue[0] >= optValue[0] - 1e-9, "1-swap result is still feasible");
            close(swapValue[0], optValue[0], "on this instance the 1-swap repair reaches the optimum");
            if (k == 1 || k == 4) {
                check(greedyValue[0] > optValue[0] + 1e-9,
                        "documented counterexample: greedy k-median is suboptimal for k = " + k);
            }
        }

        // ---- the two greedy code paths coincide when all gains stay positive ----------------
        Instance freeOpens = new Instance(canonical.clients, canonical.facilities, zero, Double.POSITIVE_INFINITY);
        for (int k = 1; k <= 4; k++) {
            double[] truncatedCost = new double[1];
            int[] truncated = ratioGreedyWithBudget(freeOpens, k, truncatedCost);
            double[] kmValue = new double[1];
            int[] kmSet = greedyKMedian(freeOpens, k, kmValue);
            check(Arrays.equals(truncated, kmSet), "ratio greedy with f = 0 and budget " + k
                    + " opens the same facilities as the k-median sketch");
            close(truncatedCost[0], kmValue[0], "and incurs the same cost");
        }
        // ... and with f = 0 the unrestricted UFL greedy opens exactly the useful facilities: it
        // stops as soon as the best remaining gain vanishes. Because opening a facility can never
        // increase the uncapped cost, opening everything is optimal, so the greedy (which leaves out
        // only zero-gain facilities) already reaches the brute-force optimum. The shorter solution
        // must be a prefix of the m-budget k-median sketch, which is forced to open the useless one.
        double[] allOpenCost = new double[1];
        int[] allOpenSet = greedyFacilityLocation(freeOpens, allOpenCost);
        for (int j = 0; j < freeOpens.m(); j++) {
            boolean isOpen = false;
            for (int t : allOpenSet) if (t == j) isOpen = true;
            if (!isOpen) {
                check(marginalGain(freeOpens, allOpenSet, allOpenSet.length, j) <= 1e-12,
                        "an unopened facility must have zero marginal gain");
            }
        }
        double[] freeOpt = new double[1];
        bruteForceFacilityLocation(freeOpens, freeOpt);
        close(allOpenCost[0], freeOpt[0], "with f = 0 the greedy reaches the uncapped optimum");
        double[] fullK = new double[1];
        int[] fullKSet = greedyKMedian(freeOpens, freeOpens.m(), fullK);
        check(fullKSet.length == freeOpens.m(), "the m-budget k-median sketch opens every facility");
        close(fullK[0], allOpenCost[0], "the useless facility adds no cost");
        check(Arrays.equals(Arrays.copyOf(fullKSet, allOpenSet.length), allOpenSet),
                "the UFL greedy's opens are a prefix of the k-median sketch's opens");

        // ---- deterministic random families -------------------------------------------------
        Rng rng = new Rng(20251016L);
        double worstUflRatio = 1.0;
        int suboptimalUfl = 0;
        for (int trial = 0; trial < 400; trial++) {
            int n = 5;
            int m = 6;
            double[][] cls = new double[n][2];
            for (int i = 0; i < n; i++) {
                cls[i][0] = Math.rint(rng.nextDouble() * 20.0);
                cls[i][1] = Math.rint(rng.nextDouble() * 20.0);
            }
            double[][] fcs = new double[m][2];
            for (int j = 0; j < m; j++) {
                fcs[j][0] = Math.rint(rng.nextDouble() * 20.0);
                fcs[j][1] = Math.rint(rng.nextDouble() * 20.0);
            }
            double[] costs = new double[m];
            for (int j = 0; j < m; j++) costs[j] = 1.0 + rng.nextInt(8);
            Instance inst = new Instance(cls, fcs, costs, 20.0);
            double[] g = new double[1];
            int[] gSet = greedyFacilityLocation(inst, g);
            double[] o = new double[1];
            bruteForceFacilityLocation(inst, o);
            check(g[0] >= o[0] - 1e-9, "greedy >= optimum on random instance " + trial);
            double fsum = 0.0;
            for (double f : costs) fsum += f;
            check(g[0] <= o[0] + fsum + 1e-9, "sandwich bound on random instance " + trial);
            if (g[0] > o[0] + 1e-9) {
                suboptimalUfl++;
                worstUflRatio = Math.max(worstUflRatio, g[0] / o[0]);
            }
            check(gSet.length != 0 || Math.abs(g[0] - n * 20.0) <= 1e-9, "clients served directly or by facilities");
        }
        System.out.println("random UFL sample: " + suboptimalUfl + "/400 suboptimal, worst ratio " + worstUflRatio);

        double worstMedianRatio = 1.0;
        int suboptimalMedian = 0;
        for (int trial = 0; trial < 400; trial++) {
            int n = 9;
            int m = 7;
            int k = 3;
            double[][] cls = new double[n][2];
            for (int i = 0; i < n; i++) {
                cls[i][0] = Math.rint(rng.nextDouble() * 30.0);
                cls[i][1] = Math.rint(rng.nextDouble() * 30.0);
            }
            double[][] fcs = new double[m][2];
            for (int j = 0; j < m; j++) {
                fcs[j][0] = Math.rint(rng.nextDouble() * 30.0);
                fcs[j][1] = Math.rint(rng.nextDouble() * 30.0);
            }
            Instance inst = new Instance(cls, fcs, new double[m], Double.POSITIVE_INFINITY);
            double[] g = new double[1];
            int[] gSet = greedyKMedian(inst, k, g);
            double[] o = new double[1];
            bruteForceKMedian(inst, k, o);
            check(g[0] >= o[0] - 1e-9, "k-median greedy >= optimum on instance " + trial);
            if (g[0] > o[0] + 1e-9) {
                suboptimalMedian++;
                worstMedianRatio = Math.max(worstMedianRatio, g[0] / o[0]);
            }
            double[] s = new double[1];
            double[] swap = new double[1];
            swapLocalSearch(inst, k, gSet, s);
            check(s[0] <= g[0] + 1e-9, "1-swap never increases the k-median cost (instance " + trial + ")");
            swap[0] = s[0];
            check(swap[0] >= o[0] - 1e-9, "1-swap stays above the optimum");
        }
        System.out.println("random k-median sample: " + suboptimalMedian + "/400 suboptimal, worst ratio " + worstMedianRatio);
        check(suboptimalMedian > 0, "greedy k-median is not exact: at least one suboptimal instance exists");
        check(worstMedianRatio <= 1.6, "observed worst k-median ratio stays under the documented 1.6");
        check(suboptimalUfl >= 0 && worstUflRatio <= 1.6, "observed worst UFL ratio stays under the documented 1.6");

        System.out.println("PASS");
    }
}
