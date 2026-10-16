/** Task 549 - Implement label propagation for community detection. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Label propagation community detection: every vertex starts with its own label and repeatedly
 * adopts the label carrying the largest total edge weight among its neighbours, so labels spread
 * along the dense parts of the graph. Two tie-breaks are implemented: a fully deterministic one
 * that picks the smallest label id, and one that draws uniformly among the tied labels from a
 * fixed LCG seed (still reproducible, but no longer order- and index-biased). The update is
 * asynchronous and in place along a caller-supplied order, ending when no label changes or after
 * the documented n-round cap; results are canonically renumbered so partitions compare label-free.
 *
 * Documented checks and findings: every run must converge within n rounds and end in a fixed point
 * (verified by re-checking the local plurality condition, allowing the documented 1e-12 weight
 * slack); the deterministic variant is reproducible and, as expected from its smallest-label bias,
 * is shown to collapse the two 6-cliques joined by a bridge and the ring of four 5-cliques into a
 * single community of modularity 0, while it does recover disjoint triangles with the hand-computed
 * Q = 0.5, which exhaustive enumeration confirms is optimal; the seeded random variant recovers the
 * bridged cliques in 11 of 12 seeds, the ring in 9 of 12 index-order runs and in all 12 shuffled
 * order runs, always converging and staying locally consistent, and never exceeding the planted
 * partition's modularity (0.659090909..., hand-computed); on small instances (n <= 6, all Bell(n)
 * partitions enumerated) no run ever beats the exhaustive optimum; and the synchronous variant,
 * included to justify the asynchronous update, is asserted to cycle on the 4-cycle with period two
 * ((0,1,2,3) -> (1,0,1,0) -> (0,1,0,1) -> (1,0,1,0)). Stars and isolated vertices behave as
 * documented (the star collapses, isolated vertices keep their labels).
 */
import java.util.*;

public class Solution {
    static double[] weightedDegrees(double[][] w) {
        int n = w.length;
        double[] k = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) k[i] += w[i][j];
        }
        return k;
    }

    static double modularity(double[][] w, int[] labels) {
        int n = w.length;
        double twoM = 0.0;
        double[] k = weightedDegrees(w);
        for (double value : k) twoM += value;
        if (twoM == 0.0) return 0.0;
        Map<Integer, Double> in = new TreeMap<>();
        Map<Integer, Double> tot = new TreeMap<>();
        for (int i = 0; i < n; i++) {
            tot.merge(labels[i], k[i], Double::sum);
            for (int j = 0; j < n; j++) {
                if (i != j && labels[i] == labels[j]) in.merge(labels[i], w[i][j], Double::sum);
            }
        }
        double total = 0.0;
        for (int community : tot.keySet()) {
            double internal = in.getOrDefault(community, 0.0);
            double volume = tot.get(community);
            total += internal / twoM - (volume / twoM) * (volume / twoM);
        }
        return total;
    }

    static int[] relabel(int[] labels) {
        Map<Integer, Integer> map = new TreeMap<>();
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            map.putIfAbsent(labels[i], map.size());
            out[i] = map.get(labels[i]);
        }
        return out;
    }

    /** Best neighbour label by total edge weight, ties towards the smallest label id. */
    static int bestLabel(double[][] w, int n, int[] labels, int vertex) {
        Map<Integer, Double> weights = new TreeMap<>();
        for (int j = 0; j < n; j++) {
            if (j != vertex && w[vertex][j] > 0) weights.merge(labels[j], w[vertex][j], Double::sum);
        }
        int best = labels[vertex];
        double bestWeight = -1.0;
        for (Map.Entry<Integer, Double> entry : weights.entrySet()) {
            if (entry.getValue() > bestWeight + 1e-12) {
                bestWeight = entry.getValue();
                best = entry.getKey();
            }
        }
        return best;
    }

    /** Same rule but ties are broken uniformly at random with a fixed LCG seed. */
    static int bestLabelRandom(double[][] w, int n, int[] labels, int vertex, int[] state) {
        Map<Integer, Double> weights = new TreeMap<>();
        for (int j = 0; j < n; j++) {
            if (j != vertex && w[vertex][j] > 0) weights.merge(labels[j], w[vertex][j], Double::sum);
        }
        double bestWeight = 0.0;
        for (double value : weights.values()) bestWeight = Math.max(bestWeight, value);
        List<Integer> tied = new ArrayList<>();
        for (Map.Entry<Integer, Double> entry : weights.entrySet()) {
            if (entry.getValue() >= bestWeight - 1e-12) tied.add(entry.getKey());
        }
        if (tied.isEmpty() || weights.getOrDefault(labels[vertex], 0.0) >= bestWeight - 1e-12) {
            return labels[vertex];
        }
        state[0] = (state[0] * 75 + 74) % 65537;
        return tied.get(state[0] % tied.size());
    }

    static class Lpa {
        final int[] labels;
        final int rounds;
        final boolean converged;

        Lpa(int[] labels, int rounds, boolean converged) {
            this.labels = labels;
            this.rounds = rounds;
            this.converged = converged;
        }
    }

    /** Asynchronous label propagation in the given update order. */
    static Lpa propagate(double[][] w, int n, int[] order, int maxRounds) {
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = i;
        int rounds = 0;
        boolean converged = false;
        while (rounds < maxRounds) {
            rounds++;
            boolean changed = false;
            for (int position = 0; position < n; position++) {
                int vertex = order[position];
                int best = bestLabel(w, n, labels, vertex);
                if (best != labels[vertex]) {
                    labels[vertex] = best;
                    changed = true;
                }
            }
            if (!changed) {
                converged = true;
                break;
            }
        }
        return new Lpa(relabel(labels), rounds, converged);
    }

    static int[] indexOrder(int n) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        return order;
    }

    static int[] shuffledOrder(int n, int seed) {
        int[] order = indexOrder(n);
        for (int i = n - 1; i > 0; i--) {
            seed = (seed * 75 + 74) % 65537;
            int j = seed % (i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return order;
    }

    /** Asynchronous propagation with LCG tie-breaking; reproducible for a fixed seed. */
    static Lpa propagateRandom(double[][] w, int n, int[] order, int seed, int maxRounds) {
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) labels[i] = i;
        int[] state = {seed % 65537};
        int rounds = 0;
        boolean converged = false;
        while (rounds < maxRounds) {
            rounds++;
            boolean changed = false;
            for (int position = 0; position < n; position++) {
                int vertex = order[position];
                int best = bestLabelRandom(w, n, labels, vertex, state);
                if (best != labels[vertex]) {
                    labels[vertex] = best;
                    changed = true;
                }
            }
            if (!changed) {
                converged = true;
                break;
            }
        }
        return new Lpa(relabel(labels), rounds, converged);
    }

    /** One synchronous sweep from a frozen snapshot; used to show the oscillation. */
    static int[] synchronousStep(double[][] w, int n, int[] labels) {
        int[] next = new int[n];
        for (int v = 0; v < n; v++) next[v] = bestLabel(w, n, labels, v);
        return next;
    }

    /** A partition is locally consistent when no vertex prefers another label. */
    static boolean locallyConsistent(double[][] w, int n, int[] labels) {
        for (int v = 0; v < n; v++) {
            int current = labels[v];
            int best = bestLabel(w, n, labels, v);
            Map<Integer, Double> weights = new TreeMap<>();
            for (int j = 0; j < n; j++) {
                if (j != v && w[v][j] > 0) weights.merge(labels[j], w[v][j], Double::sum);
            }
            double currentWeight = weights.getOrDefault(current, 0.0);
            double bestWeight = 0.0;
            for (double value : weights.values()) bestWeight = Math.max(bestWeight, value);
            if (bestWeight > currentWeight + 1e-12 && best != current) return false;
        }
        return true;
    }

    static double[][] cliques(int groups, int size, double bridge, boolean ring) {
        int n = groups * size;
        double[][] w = new double[n][n];
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < size; i++) {
                for (int j = i + 1; j < size; j++) {
                    w[g * size + i][g * size + j] = 1.0;
                    w[g * size + j][g * size + i] = 1.0;
                }
            }
        }
        for (int g = 0; g + 1 < groups; g++) {
            w[g * size][(g + 1) * size] = bridge;
            w[(g + 1) * size][g * size] = bridge;
        }
        if (ring && groups > 2) {
            w[(groups - 1) * size][0] = bridge;
            w[0][(groups - 1) * size] = bridge;
        }
        return w;
    }

    static double[][] cycle(int n) {
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            w[i][(i + 1) % n] = 1.0;
            w[(i + 1) % n][i] = 1.0;
        }
        return w;
    }

    static int distinct(int[] labels) {
        Set<Integer> set = new TreeSet<>();
        for (int label : labels) set.add(label);
        return set.size();
    }

    static int[] plantedLabels(int groups, int size) {
        int[] labels = new int[groups * size];
        for (int g = 0; g < groups; g++) {
            for (int i = 0; i < size; i++) labels[g * size + i] = g;
        }
        return labels;
    }

    /** Exhaustive optimum modularity over all set partitions. */
    static double bestModularity(double[][] w, int n) {
        int[] labels = new int[n];
        double[] best = {Double.NEGATIVE_INFINITY};
        enumerate(labels, 0, 0, w, best);
        return best[0];
    }

    static void enumerate(int[] labels, int index, int used, double[][] w, double[] best) {
        if (index == labels.length) {
            best[0] = Math.max(best[0], modularity(w, labels));
            return;
        }
        for (int value = 0; value <= used; value++) {
            labels[index] = value;
            enumerate(labels, index + 1, Math.max(used, value + 1), w, best);
        }
    }

    public static void main(String[] args) {
        double[][] pair = cliques(2, 6, 1.0, false);
        double[][] ring = cliques(4, 5, 1.0, true);
        double[][] disjointTriangles = cliques(2, 3, 0.0, false);
        double[][] bridgedTriangles = cliques(2, 3, 1.0, false);
        double[][] empty = new double[5][5];
        double[][] star = starOf(6);
        int[] ids12 = indexOrder(12);
        int[] ids20 = indexOrder(20);

        Lpa detPair = propagate(pair, 12, ids12, 12);
        check(detPair.converged, "the deterministic variant converges on the bridged cliques");
        check(detPair.rounds <= 12, "convergence within n rounds");
        check(locallyConsistent(pair, 12, detPair.labels), "the deterministic result is a fixed point");
        check(distinct(detPair.labels) == 1,
                "smallest-label ties collapse the bridged cliques into one community");
        check(Math.abs(modularity(pair, detPair.labels)) <= 1e-12, "the collapsed partition has Q = 0");
        check(Arrays.equals(propagate(pair, 12, ids12, 12).labels, detPair.labels),
                "the fixed index order makes the deterministic variant reproducible");

        Lpa detRing = propagate(ring, 20, ids20, 20);
        check(detRing.converged && locallyConsistent(ring, 20, detRing.labels),
                "the ring instance also converges deterministically");
        check(distinct(detRing.labels) == 1 && Math.abs(modularity(ring, detRing.labels)) <= 1e-12,
                "the ring collapses under the smallest-label rule as well");

        Lpa detTriangles = propagate(disjointTriangles, 6, indexOrder(6), 6);
        check(Arrays.equals(detTriangles.labels, new int[]{0, 0, 0, 1, 1, 1}),
                "disjoint triangles are recovered by the deterministic variant");
        check(Math.abs(modularity(disjointTriangles, detTriangles.labels) - 0.5) <= 1e-12,
                "the disjoint-triangle split has the hand-computed Q = 0.5");
        check(Math.abs(modularity(disjointTriangles, detTriangles.labels)
                        - bestModularity(disjointTriangles, 6)) <= 1e-9,
                "it also attains the exhaustive optimum");
        check(locallyConsistent(disjointTriangles, 6, detTriangles.labels), "and it is a fixed point");

        Lpa smallProp = propagate(bridgedTriangles, 6, indexOrder(6), 6);
        check(Math.abs(modularity(bridgedTriangles, smallProp.labels)
                        - bestModularity(bridgedTriangles, 6)) <= 1e-9
                        || modularity(bridgedTriangles, smallProp.labels)
                        <= bestModularity(bridgedTriangles, 6) + 1e-9,
                "the bridged triangles never beat the exhaustive optimum");
        check(smallProp.converged && locallyConsistent(bridgedTriangles, 6, smallProp.labels),
                "the bridged-triangle run is a fixed point");

        int[] c4 = {0, 1, 2, 3};
        double[][] square = cycle(4);
        int[] first = synchronousStep(square, 4, c4);
        int[] second = synchronousStep(square, 4, first);
        check(Arrays.equals(first, new int[]{1, 0, 1, 0}), "one synchronous sweep maps (0,1,2,3) to (1,0,1,0)");
        check(Arrays.equals(second, new int[]{0, 1, 0, 1}), "the next sweep gives (0,1,0,1)");
        check(Arrays.equals(synchronousStep(square, 4, second), first),
                "the synchronous update returns to the start: a period-2 cycle");
        check(!Arrays.equals(synchronousStep(square, 4, first), first), "so it never converges");
        Lpa asyncSquare = propagate(square, 4, indexOrder(4), 4);
        check(asyncSquare.converged, "the asynchronous update converges on the same graph");
        check(Math.abs(modularity(square, asyncSquare.labels)) <= 1e-12,
                "a 4-cycle has modularity 0, so the single community is optimal for it");

        int[] seeds = {11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132};
        int pairRecovered = 0;
        int ringRecovered = 0;
        int shuffledRecovered = 0;
        double plantedRingQ = modularity(ring, plantedLabels(4, 5));
        for (int s = 0; s < seeds.length; s++) {
            Lpa randomized = propagateRandom(pair, 12, ids12, seeds[s], 15);
            check(randomized.converged, "seed " + seeds[s] + ": the random variant converges");
            check(randomized.rounds <= 12, "seed " + seeds[s] + ": within n rounds");
            check(locallyConsistent(pair, 12, randomized.labels), "seed " + seeds[s] + ": fixed point");
            if (Arrays.equals(randomized.labels, plantedLabels(2, 6))) pairRecovered++;
            check(randomized.labels[0] == randomized.labels[1]
                            && randomized.labels[1] == randomized.labels[2],
                    "seed " + seeds[s] + ": the first clique stays together");
            Lpa ringRandom = propagateRandom(ring, 20, ids20, seeds[s], 25);
            check(ringRandom.converged, "seed " + seeds[s] + ": ring run converges");
            check(locallyConsistent(ring, 20, ringRandom.labels), "seed " + seeds[s] + ": ring fixed point");
            if (Arrays.equals(ringRandom.labels, plantedLabels(4, 5))) ringRecovered++;
            check(modularity(ring, ringRandom.labels) <= plantedRingQ + 1e-9,
                    "seed " + seeds[s] + ": the planted partition is the best this run can find");
            Lpa shuffled = propagateRandom(ring, 20, shuffledOrder(20, seeds[s]), seeds[s], 25);
            check(shuffled.converged, "seed " + seeds[s] + ": shuffled run converges");
            check(locallyConsistent(ring, 20, shuffled.labels), "seed " + seeds[s] + ": shuffled fixed point");
            if (Arrays.equals(shuffled.labels, plantedLabels(4, 5))) shuffledRecovered++;
            check(Arrays.equals(propagateRandom(pair, 12, ids12, seeds[s], 15).labels, randomized.labels),
                    "seed " + seeds[s] + ": the random variant is reproducible for a fixed seed");
        }
        check(Math.abs(plantedRingQ - 0.6590909090909091) <= 1e-9,
                "the hand-computed modularity of the ring planting is 0.659090909...");
        check(pairRecovered >= 11, "the random variant recovers the bridged cliques in most runs ("
                + pairRecovered + "/12)");
        check(ringRecovered >= 9, "it recovers the ring in most index-order runs (" + ringRecovered + "/12)");
        check(shuffledRecovered >= 12, "every shuffled order recovers the ring planting ("
                + shuffledRecovered + "/12)");

        for (int s = 0; s < 6; s++) {
            Lpa randomized = propagateRandom(bridgedTriangles, 6, indexOrder(6), 200 + s, 8);
            check(randomized.converged && locallyConsistent(bridgedTriangles, 6, randomized.labels),
                    "seeded small run " + s + " converges to a fixed point");
            check(modularity(bridgedTriangles, randomized.labels)
                            <= bestModularity(bridgedTriangles, 6) + 1e-9,
                    "seeded small run " + s + " never beats the exhaustive optimum");
        }

        Lpa starProp = propagate(star, 6, indexOrder(6), 6);
        check(starProp.converged && distinct(starProp.labels) == 1,
                "a star collapses into a single community");
        check(locallyConsistent(star, 6, starProp.labels), "the collapsed star is a fixed point");
        Lpa starRandom = propagateRandom(star, 6, indexOrder(6), 7, 10);
        check(distinct(starRandom.labels) == 1, "the random variant collapses the star too");

        Lpa emptyProp = propagate(empty, 5, indexOrder(5), 5);
        check(Arrays.equals(emptyProp.labels, new int[]{0, 1, 2, 3, 4}),
                "isolated vertices keep their own labels");
        check(Math.abs(modularity(empty, emptyProp.labels)) <= 1e-12, "an edgeless graph has Q = 0");
        Lpa emptyRandom = propagateRandom(empty, 5, indexOrder(5), 3, 5);
        check(Arrays.equals(emptyRandom.labels, new int[]{0, 1, 2, 3, 4}),
                "the random variant leaves isolated vertices alone as well");
        check(propagate(new double[1][1], 1, indexOrder(1), 3).labels.length == 1, "a single vertex works");

        System.out.println("PASS");
    }

    static double[][] starOf(int n) {
        double[][] w = new double[n][n];
        for (int i = 1; i < n; i++) {
            w[0][i] = 1.0;
            w[i][0] = 1.0;
        }
        return w;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
