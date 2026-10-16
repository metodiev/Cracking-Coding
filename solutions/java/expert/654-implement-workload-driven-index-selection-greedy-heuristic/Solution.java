/** Task 654 - Implement workload-driven index selection greedy heuristic. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Given a workload of queries (with frequencies) and a set of candidate indexes, the advisor costs
 * every query as min(table scan, cheapest selected index) and greedily adds the candidate with the
 * largest marginal gain until the gain is zero or the index budget is exhausted.
 *
 * Formally: query q has scan cost `tableRows`, candidate c has access-path cost giving it
 * benefit[c][q] >= 0, and
 *
 *     benefit(S) = sum_q frequency[q] * max(0, max_{c in S} benefit[c][q])
 *     cost(S)    = tableRows * sum_q frequency[q]  -  benefit(S)
 *
 * so "max coverage" and "minimal cost" are the same problem, benefit is monotone submodular, and
 * the greedy choice of the largest marginal gain is the standard (1 - 1/e) approximation.
 *
 * Verified invariants: every greedy step has a strictly positive marginal gain; the sequence of
 * marginal gains is non-increasing (submodularity); greedy stops only when no remaining candidate
 * has a positive marginal gain; per-query the chosen access path is the cheapest of the selected
 * indexes; benefit scales linearly with query frequency; greedy never beats the exhaustive optimum
 * and is always within (1 - 1/e) of it - checked on a designed exact instance, on a deterministic
 * pseudo random sweep, and on instances where greedy is provably not optimal (so the bound is not
 * vacuous). Negative frequencies/benefits and a negative budget are rejected.
 */
import java.util.*;

class Workload {
    final int tableRows;
    final String[] queryNames;
    final int[] frequency;
    final String[] candidateNames;
    final int[][] benefit;      // benefit[candidate][query] >= 0, 0 when the index cannot serve it

    Workload(int tableRows, String[] queryNames, int[] frequency, String[] candidateNames, int[][] benefit) {
        if (tableRows <= 0) throw new IllegalArgumentException("tableRows must be positive");
        if (queryNames.length != frequency.length) throw new IllegalArgumentException("frequency length");
        if (benefit.length != candidateNames.length) throw new IllegalArgumentException("candidate length");
        for (int[] row : benefit) {
            if (row.length != queryNames.length) throw new IllegalArgumentException("benefit width");
            for (int value : row) if (value < 0) throw new IllegalArgumentException("negative benefit");
        }
        for (int f : frequency) if (f < 0) throw new IllegalArgumentException("negative frequency");
        this.tableRows = tableRows;
        this.queryNames = queryNames;
        this.frequency = frequency;
        this.candidateNames = candidateNames;
        this.benefit = benefit;
    }

    int queryCount() { return queryNames.length; }

    int candidateCount() { return candidateNames.length; }

    int totalFrequency() {
        int total = 0;
        for (int f : frequency) total += f;
        return total;
    }

    /** Cost of answering the whole workload with a full table scan for every query. */
    int baseCost() { return tableRows * totalFrequency(); }

    int bestSelectedBenefit(int query, boolean[] selected) {
        int best = 0;
        for (int c = 0; c < candidateCount(); c++) {
            if (selected[c] && benefit[c][query] > best) best = benefit[c][query];
        }
        return best;
    }

    int benefitOf(boolean[] selected) {
        int total = 0;
        for (int q = 0; q < queryCount(); q++) total += frequency[q] * bestSelectedBenefit(q, selected);
        return total;
    }

    int cost(boolean[] selected) { return baseCost() - benefitOf(selected); }

    /** Index chosen per query: -1 means the table scan is still cheapest. */
    int[] accessPaths(boolean[] selected) {
        int[] path = new int[queryCount()];
        for (int q = 0; q < queryCount(); q++) {
            path[q] = -1;
            for (int c = 0; c < candidateCount(); c++) {
                if (!selected[c]) continue;
                if (benefit[c][q] > 0 && (path[q] == -1 || benefit[c][q] > benefit[path[q]][q])) path[q] = c;
            }
        }
        return path;
    }

    int marginalGain(int candidate, boolean[] selected) {
        int gain = 0;
        for (int q = 0; q < queryCount(); q++) {
            int current = bestSelectedBenefit(q, selected);
            if (benefit[candidate][q] > current) gain += (benefit[candidate][q] - current) * frequency[q];
        }
        return gain;
    }

    /** Greedy heuristic; ties on the marginal gain go to the lowest candidate index (deterministic). */
    List<Integer> greedySelection(int budget, int[] gainsOut) {
        if (budget < 0) throw new IllegalArgumentException("budget must be non-negative");
        boolean[] selected = new boolean[candidateCount()];
        List<Integer> chosen = new ArrayList<>();
        for (int step = 0; step < budget; step++) {
            int best = -1, bestGain = 0;
            for (int c = 0; c < candidateCount(); c++) {
                if (selected[c]) continue;
                int gain = marginalGain(c, selected);
                if (gain > bestGain) {
                    bestGain = gain;
                    best = c;
                }
            }
            if (best < 0) break;                 // no positive marginal gain left
            selected[best] = true;
            chosen.add(best);
            if (gainsOut != null) gainsOut[chosen.size() - 1] = bestGain;
        }
        return chosen;
    }

    boolean[] maskOf(List<Integer> selection) {
        boolean[] selected = new boolean[candidateCount()];
        for (int c : selection) selected[c] = true;
        return selected;
    }

    /** Exhaustive search; the oracle for the (1 - 1/e) bound, only usable for few candidates. */
    int optimumBenefit(int budget) {
        if (budget < 0) throw new IllegalArgumentException("budget must be non-negative");
        int n = candidateCount();
        if (n > 20) throw new IllegalArgumentException("too many candidates for exhaustive search");
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) > budget) continue;
            best = Math.max(best, benefitOf(maskOf(decodeMask(mask, n))));
        }
        return best;
    }

    static List<Integer> decodeMask(int mask, int n) {
        List<Integer> out = new ArrayList<>();
        for (int c = 0; c < n; c++) if ((mask & (1 << c)) != 0) out.add(c);
        return out;
    }
}

public class Solution {
    private static long state = 0x5DEECE66D0F1E2D3L;

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

    static String[] names(String prefix, int count) {
        String[] out = new String[count];
        for (int i = 0; i < count; i++) out[i] = prefix + i;
        return out;
    }

    static Workload randomWorkload(int candidates, int queries, int budget) {
        int[] frequency = new int[queries];
        for (int q = 0; q < queries; q++) frequency[q] = 1 + nextInt(4);
        int[][] benefit = new int[candidates][queries];
        for (int c = 0; c < candidates; c++) {
            for (int q = 0; q < queries; q++) benefit[c][q] = nextInt(50);
        }
        return new Workload(500 + nextInt(500), names("q", queries), frequency, names("idx", candidates), benefit);
    }

    static void checkGreedyProperties(Workload workload, int budget, String label) {
        int[] gains = new int[budget + 1];
        List<Integer> chosen = workload.greedySelection(budget, gains);
        boolean[] selected = workload.maskOf(chosen);
        check(chosen.size() <= budget, label + ": budget respected");
        check(workload.benefitOf(selected) <= workload.baseCost(), label + ": benefit cannot exceed a full scan");

        // marginal gains strictly positive and non-increasing (monotone submodularity)
        for (int i = 0; i < chosen.size(); i++) check(gains[i] > 0, label + ": step " + i + " has positive gain");
        for (int i = 1; i < chosen.size(); i++) {
            check(gains[i] <= gains[i - 1], label + ": gains non-increasing at " + i);
        }

        // recompute the selection step by step to confirm the greedy order and the pure gain of each pick
        boolean[] replay = new boolean[workload.candidateCount()];
        for (int c : chosen) {
            int gain = workload.marginalGain(c, replay);
            check(gain > 0, label + ": replayed gain positive");
            replay[c] = true;
        }
        check(workload.benefitOf(replay) == workload.benefitOf(selected), label + ": replay agrees");

        // no unselected candidate improves the objective, unless the budget ran out
        if (chosen.size() < budget) {
            for (int c = 0; c < workload.candidateCount(); c++) {
                if (selected[c]) continue;
                check(workload.marginalGain(c, selected) == 0, label + ": no positive gain left");
            }
        }

        // cheapest access path per query
        int[] paths = workload.accessPaths(selected);
        for (int q = 0; q < workload.queryCount(); q++) {
            if (paths[q] == -1) {
                check(workload.bestSelectedBenefit(q, selected) == 0, label + ": scan really is cheapest for query " + q);
            } else {
                for (int c = 0; c < workload.candidateCount(); c++) {
                    if (selected[c]) {
                        check(workload.benefit[c][q] <= workload.benefit[paths[q]][q],
                                label + ": access path is not the cheapest for query " + q);
                    }
                }
            }
        }

        // oracle comparison
        int greedyBenefit = workload.benefitOf(selected);
        int optimum = workload.optimumBenefit(budget);
        check(greedyBenefit <= optimum, label + ": greedy must not beat the optimum");
        check(greedyBenefit + 1e-9 >= (1.0 - 1.0 / Math.E) * optimum, label + ": (1-1/e) bound");
    }

    public static void main(String[] args) {
        // --- degenerate workloads ---------------------------------------------
        Workload empty = new Workload(1000, new String[0], new int[0], names("idx", 3),
                new int[][]{new int[0], new int[0], new int[0]});
        check(empty.baseCost() == 0, "empty workload has no base cost");
        check(empty.greedySelection(3, null).isEmpty(), "empty workload selects nothing");
        check(empty.optimumBenefit(3) == 0, "empty workload optimum is zero");

        Workload noCandidates = new Workload(400, names("q", 3), new int[]{1, 2, 1}, new String[0], new int[0][]);
        check(noCandidates.baseCost() == 1600, "base cost is rows times total frequency");
        check(noCandidates.greedySelection(5, null).isEmpty(), "no candidates -> no selection");
        check(noCandidates.cost(new boolean[0]) == 1600, "cost without candidates is the base cost");

        Workload single = new Workload(100, names("q", 2), new int[]{3, 1}, names("idx", 1),
                new int[][]{{40, 0}});
        int[] gains = new int[2];
        List<Integer> picked = single.greedySelection(2, gains);
        check(picked.equals(List.of(0)), "the only useful candidate is chosen");
        check(single.benefitOf(single.maskOf(picked)) == 120, "benefit honours query frequency");
        check(gains[1] == 0, "a saturated workload gains nothing more");
        check(single.cost(single.maskOf(picked)) == 400 - 120, "cost = base - benefit");
        check(single.greedySelection(0, null).isEmpty(), "zero budget selects nothing");

        // --- frequency scaling -------------------------------------------------
        Workload doubled = new Workload(100, names("q", 2), new int[]{6, 2}, names("idx", 1),
                new int[][]{{40, 0}});
        check(doubled.benefitOf(doubled.maskOf(List.of(0))) == 2 * single.benefitOf(single.maskOf(List.of(0))),
                "benefit scales linearly with frequency");
        check(doubled.optimumBenefit(1) == 2 * single.optimumBenefit(1), "optimum scales with frequency too");

        // --- designed exact instance: disjoint coverage ------------------------
        Workload disjoint = new Workload(1000, names("q", 4), new int[]{1, 1, 1, 1}, names("idx", 4),
                new int[][]{
                        {10, 0, 0, 0},
                        {0, 20, 0, 0},
                        {0, 0, 30, 0},
                        {0, 0, 0, 40}});
        List<Integer> disjointChoice = disjoint.greedySelection(2, null);
        check(disjoint.benefitOf(disjoint.maskOf(disjointChoice)) == disjoint.optimumBenefit(2),
                "disjoint coverage: greedy is exactly optimal");
        check(disjointChoice.equals(List.of(3, 2)), "greedy takes the largest gain first, got " + disjointChoice);

        // --- duplicate candidates do not double count --------------------------
        Workload duplicates = new Workload(1000, names("q", 2), new int[]{5, 5}, names("idx", 3),
                new int[][]{
                        {10, 10},
                        {10, 10},
                        {10, 10}});
        List<Integer> duplicateChoice = duplicates.greedySelection(3, null);
        check(duplicates.benefitOf(duplicates.maskOf(duplicateChoice)) == 100,
                "duplicate indexes contribute once");
        check(duplicateChoice.size() == 1, "after a duplicate the gain is zero, got " + duplicateChoice);

        // --- tie break is deterministic ---------------------------------------
        Workload ties = new Workload(1000, names("q", 1), new int[]{1}, names("idx", 3),
                new int[][]{{7}, {7}, {7}});
        check(ties.greedySelection(2, null).equals(List.of(0)), "ties go to the lowest candidate index");

        // --- random sweep with the exhaustive oracle ---------------------------
        int suboptimal = 0;
        int instances = 0;
        for (int trial = 0; trial < 240; trial++) {
            int candidates = 6;
            int budget = 1 + nextInt(3);
            Workload workload = randomWorkload(candidates, 7, budget);
            checkGreedyProperties(workload, budget, "trial " + trial);
            int greedyBenefit = workload.benefitOf(workload.maskOf(workload.greedySelection(budget, null)));
            if (greedyBenefit < workload.optimumBenefit(budget)) suboptimal++;
            instances++;
        }
        check(instances == 240, "sweep size");
        check(suboptimal > 0, "the (1-1/e) bound must be non-vacuous, suboptimal instances: " + suboptimal);

        // --- larger sweep, single index budget ---------------------------------
        int checkedInstances = 0;
        for (int trial = 0; trial < 60; trial++) {
            Workload workload = randomWorkload(8, 10, 2);
            checkGreedyProperties(workload, 2, "wide trial " + trial);
            checkedInstances++;
        }
        check(checkedInstances == 60, "wide sweep size");

        // --- rejected input (negative checks) ----------------------------------
        boolean rejected = false;
        try {
            new Workload(100, names("q", 1), new int[]{-1}, names("idx", 1), new int[][]{{5}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative frequency must be rejected");
        rejected = false;
        try {
            new Workload(100, names("q", 1), new int[]{1}, names("idx", 1), new int[][]{{-5}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative benefit must be rejected");
        rejected = false;
        try {
            single.greedySelection(-1, null);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "negative budget must be rejected");
        rejected = false;
        try {
            new Workload(0, names("q", 1), new int[]{1}, names("idx", 1), new int[][]{{5}});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "empty table must be rejected");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
