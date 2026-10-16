/** Task 901 - Implement provable bounds for randomized algorithms used in your projects, with tests.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A randomized algorithm is only usable if its failure/tail bounds are *provable* and *tested*. This file
 * pairs four classic algorithms with the theorem each one relies on, and then measures the theorem:
 *
 *   min cut (Karger)      P[success] >= 2/(n(n-1))          -- product of (1 - 2/(remaining size)) terms
 *   quickselect           E[comparisons] <= 4n              -- sum over recursive levels of the split cost
 *   Bernoulli estimator   P[|mean-p| >= eps] <= 2e^{-2Ne^2}  -- Hoeffding, and Chebyshev with z sigma
 *   event union           P[union of bad events] <= sum P    -- union bound, checked by enumeration
 *   Markov                P[X >= t E[X]] <= 1/t             -- exact, computed from the pmf
 *
 * What is simplified: the graph sizes are tiny (n <= 10) so the exact oracle is exhaustive, trial counts are
 * a few thousand so the whole program stays under a second, and the "randomness" is a fixed-seed splitmix64.
 * Invariants that are still genuinely verified: every empirical rate must sit inside the proved bound, the
 * empirical min cut must equal the exhaustive min cut, quickselect must equal the sorted-array oracle, and
 * an amplification routine that violates its precondition (p <= 1/2) must be rejected.
 */
import java.util.*;

/** splitmix64 - the only source of randomness in this file. */
final class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Uniform in [0, bound); floorMod keeps it non-negative even for negative mixed output. */
    int nextBelow(int bound) {
        if (bound <= 0) throw new IllegalArgumentException("bound must be positive");
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    /** Uniform double in [0,1): 53 bits of mantissa, exactly the IEEE-754 spacing. */
    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** Undirected multigraph as a symmetric count matrix; contraction merges two rows into one. */
final class MultiGraph {
    final int n;
    final int[][] weight;

    MultiGraph(int n) { this.n = n; this.weight = new int[n][n]; }

    void addEdge(int u, int v, int w) {
        weight[u][v] += w;
        weight[v][u] += w;
    }

    int degree(int v) {
        int d = 0;
        for (int u = 0; u < n; u++) d += weight[v][u];
        return d;
    }

    int edges() {
        int total = 0;
        for (int v = 0; v < n; v++) total += degree(v);
        return total / 2;
    }

    /** True when u and v are already the same supernode (self loops are ignored, not contracted). */
    boolean adjacent(int u, int v) { return u != v && weight[u][v] > 0; }
}

/** One Karger trial: contract a uniformly random edge until two supernodes remain. */
final class KargerMinCut {
    static int trial(MultiGraph graph, SplitMix64 rng) {
        int n = graph.n;
        if (n < 2) return 0;
        int[][] w = new int[n][n];
        for (int i = 0; i < n; i++) w[i] = graph.weight[i].clone();
        // Vertex i keeps its own slot; merging v into u zeroes v's row and column out of the live set.
        boolean[] live = new boolean[n];
        Arrays.fill(live, true);
        int liveCount = n;
        while (liveCount > 2) {
            int total = 0;
            for (int u = 0; u < n; u++) if (live[u]) for (int v = u + 1; v < n; v++) if (live[v]) total += w[u][v];
            if (total == 0) break;
            int pick = rng.nextBelow(total);
            int a = -1;
            int b = -1;
            outer:
            for (int u = 0; u < n; u++) {
                if (!live[u]) continue;
                for (int v = u + 1; v < n; v++) {
                    if (!live[v]) continue;
                    pick -= w[u][v];
                    if (pick < 0) { a = u; b = v; break outer; }
                }
            }
            for (int k = 0; k < n; k++) if (live[k] && k != b) { w[a][k] += w[b][k]; w[k][a] += w[b][k]; }
            live[b] = false;
            liveCount--;
        }
        int cut = 0;
        List<Integer> supernodes = new ArrayList<>();
        for (int v = 0; v < n; v++) if (live[v]) supernodes.add(v);
        if (supernodes.size() == 2) cut = w[supernodes.get(0)][supernodes.get(1)];
        return cut;
    }
}

public class Solution {
    public static void main(String[] args) {
        // ---- 1. exact oracle: exhaustive min cut over all subsets containing vertex 0 ------------------
        MultiGraph cycleWithChords = new MultiGraph(6);
        for (int i = 0; i < 6; i++) cycleWithChords.addEdge(i, (i + 1) % 6, 1);
        cycleWithChords.addEdge(0, 2, 1);
        cycleWithChords.addEdge(3, 5, 1);
        int exact = exhaustiveMinCut(cycleWithChords);
        check(exact == 2, "exhaustive min cut of the 6-cycle with two chords must be 2, got " + exact);
        check(cycleWithChords.edges() == 8, "the fixture has 8 edges, got " + cycleWithChords.edges());

        // ---- 2. Karger's bound: P[success] >= 2/(n(n-1)) ---------------------------------------------
        int n = cycleWithChords.n;
        double provedBound = 2.0 / (n * (n - 1));      // 2/30 = 1/15 here
        int trials = 2000;
        SplitMix64 rng = new SplitMix64(0x5EED0001L);
        int successes = 0;
        int minimumSeen = Integer.MAX_VALUE;
        for (int t = 0; t < trials; t++) {
            int cut = KargerMinCut.trial(cycleWithChords, rng);
            minimumSeen = Math.min(minimumSeen, cut);
            if (cut == exact) successes++;
        }
        double rate = successes / (double) trials;
        check(rate >= provedBound, "empirical success rate " + rate + " must meet the proved bound " + provedBound);
        check(minimumSeen >= exact, "no trial may find a cut below the exact min cut " + exact);
        check(successes < trials, "the algorithm is randomized, so some trials should miss");

        // ---- 3. amplification: best of K independent runs --------------------------------------------
        int amplified = Integer.MAX_VALUE;
        for (int k = 0; k < 200; k++) amplified = Math.min(amplified, KargerMinCut.trial(cycleWithChords, rng));
        check(amplified == exact, "best-of-200 Karger runs must hit the exact min cut, got " + amplified);
        check(Math.pow(1 - provedBound, 200) < 1e-4,
                "200 repetitions must push the failure probability below 1e-4");

        // ---- 4. determinism of the randomized core ---------------------------------------------------
        SplitMix64 repeatA = new SplitMix64(0x5EED0001L);
        SplitMix64 repeatB = new SplitMix64(0x5EED0001L);
        int identical = 0;
        for (int t = 0; t < 50; t++) {
            if (KargerMinCut.trial(cycleWithChords, repeatA) == KargerMinCut.trial(cycleWithChords, repeatB)) {
                identical++;
            }
        }
        check(identical == 50, "the same seed must replay the same 50 trials, matched " + identical);

        // ---- 5. quickselect: E[comparisons] <= 4n, and the answer matches the sorted oracle ----------
        int[] array = new int[64];
        SplitMix64 fill = new SplitMix64(0x5EED0002L);
        for (int i = 0; i < array.length; i++) array[i] = fill.nextBelow(1000);
        int[] sorted = array.clone();
        Arrays.sort(sorted);
        long totalComparisons = 0;
        int runs = 200;
        SplitMix64 sel = new SplitMix64(0x5EED0003L);
        for (int r = 0; r < runs; r++) {
            int k = r % array.length;
            Counter counter = new Counter();
            int got = quickselect(array.clone(), k, sel, counter);
            check(got == sorted[k], "quickselect must return the sorted oracle value at rank " + k);
            totalComparisons += counter.comparisons;
        }
        double meanComparisons = totalComparisons / (double) runs;
        check(meanComparisons <= 4.0 * array.length,
                "mean comparisons " + meanComparisons + " must stay under the proved 4n bound");

        // ---- 6. Hoeffding and Chebyshev, measured ----------------------------------------------------
        double p = 0.3;
        int sampleSize = 30;
        double epsilon = 0.15;
        double hoeffding = 2 * Math.exp(-2.0 * sampleSize * epsilon * epsilon);
        SplitMix64 est = new SplitMix64(0x5EED0004L);
        int hoeffdingViolations = 0;
        int chebyshevViolations = 0;
        int estimatorTrials = 2000;
        double z = 4.0;
        double sigma = Math.sqrt(p * (1 - p) / sampleSize);
        for (int t = 0; t < estimatorTrials; t++) {
            int heads = 0;
            for (int i = 0; i < sampleSize; i++) if (est.nextDouble() < p) heads++;
            double mean = heads / (double) sampleSize;
            if (Math.abs(mean - p) >= epsilon) hoeffdingViolations++;
            if (Math.abs(mean - p) >= z * sigma) chebyshevViolations++;
        }
        double hoeffdingRate = hoeffdingViolations / (double) estimatorTrials;
        double chebyshevRate = chebyshevViolations / (double) estimatorTrials;
        check(hoeffdingRate <= hoeffding, "Hoeffding rate " + hoeffdingRate + " exceeds 2e^{-2Ne^2}=" + hoeffding);
        check(chebyshevRate <= 1 / (z * z), "Chebyshev rate " + chebyshevRate + " exceeds 1/z^2");
        check(hoeffding < 1, "the Hoeffding bound must be informative at N=" + sampleSize);

        // ---- 7. union bound and Markov, exact --------------------------------------------------------
        int space = 200;
        List<boolean[]> bad = new ArrayList<>();
        int residueStep = 4;
        for (int e = 0; e < 6; e++) {
            boolean[] set = new boolean[space];
            // e = 0 and e = 5 share a residue class, so the six events genuinely overlap.
            for (int x = 0; x < space; x++) set[x] = x % 20 == (e % 5) * residueStep;
            bad.add(set);
        }
        int unionSize = 0;
        double sum = 0;
        for (int x = 0; x < space; x++) {
            boolean in = false;
            for (boolean[] set : bad) in |= set[x];
            if (in) unionSize++;
        }
        for (boolean[] set : bad) {
            int size = 0;
            for (boolean b : set) if (b) size++;
            sum += size / (double) space;
        }
        double unionProb = unionSize / (double) space;
        check(sum == 6 * (space / 20.0) / space, "each bad event must have probability 1/20, sum=" + sum);
        check(unionProb <= sum, "union bound violated: " + unionProb + " > " + sum);
        check(unionProb < sum - 1e-12, "the union bound must be strict on this overlapping family");

        double[] pmf = new double[10];
        Arrays.fill(pmf, 0.1);
        double mean = 0;
        for (int i = 0; i < 10; i++) mean += (i + 1) * pmf[i];
        for (int t = 1; t <= 10; t++) {
            double tail = 0;
            for (int i = 0; i < 10; i++) if (i + 1 >= t * mean) tail += pmf[i];
            check(tail <= 1.0 / t + 1e-12, "Markov inequality violated at t=" + t + ": " + tail + " > " + 1.0 / t);
        }
        check(Math.abs(mean - 5.5) < 1e-12, "the uniform pmf on 1..10 has mean 5.5, got " + mean);

        // ---- 8. boundary cases: n=2 (bound is 1) and an isolated vertex ------------------------------
        MultiGraph pair = new MultiGraph(2);
        pair.addEdge(0, 1, 1);
        SplitMix64 two = new SplitMix64(0x5EED0005L);
        boolean always = true;
        for (int t = 0; t < 100; t++) always &= KargerMinCut.trial(pair, two) == 1;
        check(always, "with n=2 the min cut is found with probability 1 (the 2/(n(n-1)) bound is exactly 1)");
        MultiGraph lonely = new MultiGraph(3);
        lonely.addEdge(0, 1, 1);
        check(KargerMinCut.trial(lonely, two) == 0, "a graph with an isolated vertex has min cut 0");
        check(exhaustiveMinCut(lonely) == 0, "the oracle agrees the min cut of the split graph is 0");

        // ---- 9. precondition / negative check --------------------------------------------------------
        check(amplifyFailure(0.25, 3) < 0, "amplification must reject a subroutine that is not better than random");
        check(amplifyFailure(0.75, 0) < 0, "amplification must reject a zero-round schedule");
        double threeRounds = amplifyFailure(0.75, 3);
        check(Math.abs(threeRounds - 0.015625) < 1e-12, "three rounds of p=0.75 leave 0.015625 failure");
        check(threeRounds < 1 - 0.75 && amplifyFailure(0.75, 5) < threeRounds,
                "more rounds must strictly lower the failure probability");
        check(amplifyFailure(0.5, 9) < 0, "p = 1/2 is the boundary the precondition must reject");

        // ---- 10. an artificial seed set: the bound holds for every seed, not just a lucky one --------
        for (long seed = 1; seed <= 5; seed++) {
            SplitMix64 seeded = new SplitMix64(seed);
            int hits = 0;
            for (int t = 0; t < 400; t++) if (KargerMinCut.trial(cycleWithChords, seeded) == exact) hits++;
            check(hits / 400.0 >= provedBound, "seed " + seed + " produced a rate below the proved bound");
        }

        System.out.println("Karger success rate " + rate + " vs proved bound " + provedBound
                + " over " + trials + " trials");
        System.out.println("quickselect mean comparisons " + meanComparisons + " vs bound "
                + 4.0 * array.length + " (n=" + array.length + ")");
        System.out.println("Hoeffding " + hoeffdingRate + " <= " + hoeffding + "; Chebyshev " + chebyshevRate
                + " <= " + 1 / (z * z) + "; union " + unionProb + " <= " + sum);
        System.out.println("PASS");
    }

    /** Exhaustive min cut over every proper non-empty subset that contains vertex 0. */
    static int exhaustiveMinCut(MultiGraph graph) {
        int n = graph.n;
        int best = Integer.MAX_VALUE;
        for (int mask = 1; mask < (1 << n) - 1; mask++) {
            if ((mask & 1) == 0) continue;
            int cut = 0;
            for (int u = 0; u < n; u++) {
                if ((mask & (1 << u)) == 0) continue;
                for (int v = 0; v < n; v++) if ((mask & (1 << v)) == 0) cut += graph.weight[u][v];
            }
            best = Math.min(best, cut);
        }
        return best;
    }

    /** Quickselect with a comparison counter; rank is 0-based, the array is destroyed in place. */
    static int quickselect(int[] values, int rank, SplitMix64 rng, Counter counter) {
        int lo = 0;
        int hi = values.length - 1;
        while (true) {
            if (lo == hi) return values[lo];
            int pivot = lo + rng.nextBelow(hi - lo + 1);
            int tmp = values[pivot];
            values[pivot] = values[hi];
            values[hi] = tmp;
            int store = lo;
            for (int i = lo; i < hi; i++) {
                counter.comparisons++;
                if (values[i] < values[hi]) {
                    int t = values[i];
                    values[i] = values[store];
                    values[store] = t;
                    store++;
                }
            }
            counter.comparisons++;
            int t = values[hi];
            values[hi] = values[store];
            values[store] = t;
            if (store == rank) return values[store];
            if (rank < store) hi = store - 1;
            else lo = store + 1;
        }
    }

    /**
     * Amplification failure probability for a subroutine with success probability p run for `rounds`
     * independent repetitions: (1-p)^rounds, the provable bound. Returns -1 when the precondition
     * p > 1/2 fails, because then repetition cannot beat a coin flip exponentially.
     */
    static double amplifyFailure(double p, int rounds) {
        if (p <= 0.5 || p >= 1.0 || rounds < 1) return -1;
        return Math.pow(1 - p, rounds);
    }

    static final class Counter {
        long comparisons;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
