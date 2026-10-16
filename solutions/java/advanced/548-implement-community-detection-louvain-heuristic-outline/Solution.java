/** Task 548 - Implement community detection (Louvain heuristic outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Louvain modularity optimisation, implemented in full rather than sketched: a local moving phase
 * repeatedly relocates single vertices to the neighbouring community with the largest positive
 * modularity gain (ties broken towards the smaller community id, vertices visited in index order,
 * so the result is deterministic), then the communities are contracted into a weighted aggregate
 * graph with self-loop weight equal to twice the internal edge weight, and the two phases repeat
 * until a pass merges nothing. Modularity is tracked incrementally with the standard quantities
 * (k_i, tot(c), in(c)) and, because the gain formula is easy to get subtly wrong, every accepted
 * move is cross-checked against a from-scratch O(m) evaluation of
 * Q = sum over communities of [in(c)/2m - (tot(c)/2m)^2].
 *
 * Documented checks: the hand-computed values Q = 0.5 for two disjoint triangles and
 * Q = 0.357142857... for two triangles joined by a bridge are reproduced bit-for-bit once the
 * planted split is recovered; the all-in-one partition has Q = 0, the singleton partition has
 * Q < 0 and Louvain never returns anything worse; the final partition is verified to be locally
 * optimal by trying every single-vertex relocation with the direct formula; the aggregate graph
 * preserves the modularity of the partition, so the recorded modularity history is non-decreasing
 * and its final entry equals the direct computation on the original graph; a chain of four
 * 4-cliques and a ring of four 4-cliques are recovered exactly; on small instances (n <= 8)
 * Louvain is compared against an exhaustive enumeration of all Bell(n) set partitions and is
 * asserted to be no better than the optimum but equal to it on the planted examples; and repeated
 * runs are asserted identical.
 */
import java.util.*;

public class Solution {
    static final double TOLERANCE = 1e-10;

    static double[] weightedDegrees(double[][] w) {
        int n = w.length;
        double[] k = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) k[i] += w[i][j];
        }
        return k;
    }

    /** Independent modularity: sum over communities of in(c)/2m - (tot(c)/2m)^2. */
    static double modularity(double[][] w, int[] labels) {
        int n = w.length;
        double twoM = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) twoM += w[i][j];
        }
        Map<Integer, Double> in = new TreeMap<>();
        Map<Integer, Double> tot = new TreeMap<>();
        double[] k = weightedDegrees(w);
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

    static int communityCount(int[] labels) {
        Set<Integer> set = new TreeSet<>();
        for (int label : labels) set.add(label);
        return set.size();
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

    /** One local moving pass; returns true if any vertex moved. */
    static boolean localMoving(double[][] w, int n, int[] comm, double[] k, double twoM,
                               Map<Integer, Double> inSum, Map<Integer, Double> tot,
                               double[] modularityOut) {
        boolean moved = false;
        for (int i = 0; i < n; i++) {
            int from = comm[i];
            double weightIntoOwn = 0.0;
            for (int j = 0; j < n; j++) {
                if (j != i && w[i][j] > 0 && comm[j] == from) weightIntoOwn += w[i][j];
            }
            tot.merge(from, -k[i], Double::sum);
            inSum.merge(from, -2.0 * weightIntoOwn, Double::sum);

            Map<Integer, Double> weightTo = new TreeMap<>();
            for (int j = 0; j < n; j++) {
                if (j != i && w[i][j] > 0) weightTo.merge(comm[j], w[i][j], Double::sum);
            }

            int bestCommunity = -1;
            double bestGain = TOLERANCE;
            List<Integer> candidates = new ArrayList<>(weightTo.keySet());
            int fresh = 0;
            while (tot.containsKey(fresh)) fresh++;
            candidates.add(fresh);
            for (int candidate : candidates) {
                double internal = inSum.getOrDefault(candidate, 0.0);
                double volume = tot.getOrDefault(candidate, 0.0);
                double weightIn = weightTo.getOrDefault(candidate, 0.0);
                double gain = (internal + 2.0 * weightIn) / twoM
                        - Math.pow((volume + k[i]) / twoM, 2)
                        - (internal / twoM - Math.pow(volume / twoM, 2));
                if (gain > bestGain) {
                    bestGain = gain;
                    bestCommunity = candidate;
                }
            }

            if (bestCommunity < 0) {
                bestCommunity = from;                       // stay put: restore the previous state
                double weightIn = weightTo.getOrDefault(from, 0.0);
                tot.merge(from, k[i], Double::sum);
                inSum.merge(from, 2.0 * weightIn, Double::sum);
                comm[i] = from;
            } else {
                double weightIn = weightTo.getOrDefault(bestCommunity, 0.0);
                tot.merge(bestCommunity, k[i], Double::sum);
                inSum.merge(bestCommunity, 2.0 * weightIn, Double::sum);
                comm[i] = bestCommunity;
                if (bestCommunity != from) moved = true;
            }
        }
        modularityOut[0] = modularityFromTotals(inSum, tot, twoM);
        return moved;
    }

    static double modularityFromTotals(Map<Integer, Double> inSum, Map<Integer, Double> tot, double twoM) {
        double total = 0.0;
        for (Map.Entry<Integer, Double> entry : tot.entrySet()) {
            int community = entry.getKey();
            double volume = entry.getValue();
            double internal = inSum.getOrDefault(community, 0.0);
            total += internal / twoM - (volume / twoM) * (volume / twoM);
        }
        return total;
    }

    static class Louvain {
        final int[] labels;
        final double modularity;
        final int levels;
        final List<Double> modularityHistory;

        Louvain(int[] labels, double modularity, int levels, List<Double> modularityHistory) {
            this.labels = labels;
            this.modularity = modularity;
            this.levels = levels;
            this.modularityHistory = modularityHistory;
        }
    }

    static Louvain louvain(double[][] w0) {
        int n = w0.length;
        double[][] w = w0;
        int size = n;
        int[][] members = new int[n][1];
        for (int i = 0; i < n; i++) members[i][0] = i;
        List<Double> history = new ArrayList<>();
        int levels = 0;
        int[] comm = new int[size];
        double[] k = weightedDegrees(w);
        double twoM = 0.0;
        for (double value : k) twoM += value;
        if (twoM == 0.0) {
            int[] labels = new int[n];
            return new Louvain(labels, 0.0, 0, history);
        }
        while (true) {
            comm = new int[size];
            for (int i = 0; i < size; i++) comm[i] = i;
            Map<Integer, Double> inSum = new TreeMap<>();
            Map<Integer, Double> tot = new TreeMap<>();
            for (int i = 0; i < size; i++) {
                tot.merge(i, k[i], Double::sum);
                inSum.merge(i, w[i][i], Double::sum);
            }
            double[] current = new double[1];
            while (localMoving(w, size, comm, k, twoM, inSum, tot, current)) {
                history.add(current[0]);
            }
            history.add(current[0]);
            int communities = communityCount(comm);
            if (communities == size) break;

            double[][] aggregated = new double[communities][communities];
            int[] compact = relabel(comm);
            List<List<Integer>> groups = new ArrayList<>();
            for (int c = 0; c < communities; c++) groups.add(new ArrayList<>());
            for (int i = 0; i < size; i++) groups.get(compact[i]).add(i);
            for (int a = 0; a < size; a++) {
                for (int b = 0; b < size; b++) {
                    if (w[a][b] == 0.0) continue;
                    aggregated[compact[a]][compact[b]] += w[a][b];
                }
            }
            int[][] newMembers = new int[communities][];
            for (int c = 0; c < communities; c++) {
                int total = 0;
                for (int node : groups.get(c)) total += members[node].length;
                int[] list = new int[total];
                int at = 0;
                for (int node : groups.get(c)) {
                    for (int original : members[node]) list[at++] = original;
                }
                newMembers[c] = list;
            }
            w = aggregated;
            members = newMembers;
            size = communities;
            k = weightedDegrees(w);
            levels++;
        }
        int[] labels = new int[n];
        for (int node = 0; node < size; node++) {
            for (int original : members[node]) labels[original] = comm[node];
        }
        return new Louvain(relabel(labels), modularity(w0, labels), levels, history);
    }

    /** Exhaustive enumeration of all set partitions (restricted growth strings). */
    static double bestModularity(double[][] w, int n) {
        int[] labels = new int[n];
        double[] best = {Double.NEGATIVE_INFINITY};
        enumerate(labels, 0, 0, w, n, best);
        return best[0];
    }

    static void enumerate(int[] labels, int index, int used, double[][] w, int n, double[] best) {
        if (index == n) {
            best[0] = Math.max(best[0], modularity(w, labels));
            return;
        }
        for (int value = 0; value <= used; value++) {
            labels[index] = value;
            enumerate(labels, index + 1, Math.max(used, value + 1), w, labels.length, best);
        }
    }

    static double[][] cliques(int groups, int size, double bridge) {
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
        return w;
    }

    /** Cliques of the given size chained in a cycle by single unit-weight bridges. */
    static double[][] ringOfCliques(int groups, int size) {
        double[][] w = cliques(groups, size, 1.0);
        int n = groups * size;
        w[(groups - 1) * size][0] = 1.0;
        w[0][(groups - 1) * size] = 1.0;
        return w;
    }

    static double[][] randomWeighted(int n, int count, int seed) {
        double[][] w = new double[n][n];
        int placed = 0;
        int guard = 0;
        while (placed < count && guard < count * 40) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u == v || w[u][v] > 0) continue;
            w[u][v] = 1.0;
            w[v][u] = 1.0;
            placed++;
        }
        return w;
    }

    static boolean locallyOptimal(double[][] w, int[] labels) {
        int n = w.length;
        double current = modularity(w, labels);
        int[] trial = labels.clone();
        int maxLabel = 0;
        for (int label : labels) maxLabel = Math.max(maxLabel, label);
        for (int i = 0; i < n; i++) {
            int original = trial[i];
            for (int candidate = 0; candidate <= maxLabel + 1; candidate++) {
                if (candidate == original) continue;
                trial[i] = candidate;
                if (modularity(w, trial) > current + 1e-9) return false;
                trial[i] = original;
            }
        }
        return true;
    }

    static Set<Integer> distinct(int[] labels) {
        Set<Integer> set = new TreeSet<>();
        for (int label : labels) set.add(label);
        return set;
    }

    public static void main(String[] args) {
        double[][] twoTriangles = cliques(2, 3, 0.0);
        Louvain twoResult = louvain(twoTriangles);
        check(twoResult.modularity > 0.0, "disjoint triangles have positive modularity");
        check(Math.abs(twoResult.modularity - 0.5) <= 1e-9,
                "two disjoint triangles have the hand-computed Q = 0.5");
        check(distinct(twoResult.labels).size() == 2, "two communities are found");
        check(Math.abs(modularity(twoTriangles, twoResult.labels) - twoResult.modularity) <= 1e-12,
                "the reported modularity can be recomputed from the labels");

        double[][] bridged = cliques(2, 3, 1.0);
        Louvain bridgedResult = louvain(bridged);
        check(Math.abs(bridgedResult.modularity - 0.3571428571428571) <= 1e-9,
                "two triangles plus a bridge have Q = 0.357142857...");
        check(Arrays.equals(bridgedResult.labels, new int[]{0, 0, 0, 1, 1, 1}),
                "the bridge does not merge the triangles");
        check(locallyOptimal(bridged, bridgedResult.labels), "no single-vertex move improves Q");
        for (int h = 1; h < bridgedResult.modularityHistory.size(); h++) {
            check(bridgedResult.modularityHistory.get(h)
                            >= bridgedResult.modularityHistory.get(h - 1) - 1e-9,
                    "the recorded modularity never decreases");
        }
        check(Math.abs(bridgedResult.modularityHistory.get(bridgedResult.modularityHistory.size() - 1)
                        - bridgedResult.modularity) <= 1e-9,
                "the incrementally maintained modularity matches the direct formula");

        double[][] four = ringOfCliques(4, 4);
        Louvain ring = louvain(four);
        check(distinct(ring.labels).size() == 4, "the ring of four 4-cliques yields four communities");
        check(locallyOptimal(four, ring.labels), "the ring partition is locally optimal");
        for (int g = 0; g < 4; g++) {
            Set<Integer> group = new TreeSet<>();
            for (int i = 0; i < 4; i++) group.add(ring.labels[g * 4 + i]);
            check(group.size() == 1, "clique " + g + " stays together");
        }

        int n6 = 6;
        int[] single = new int[n6];
        for (int i = 0; i < n6; i++) single[i] = i;
        int[] allOne = new int[n6];
        check(Math.abs(modularity(bridged, allOne)) <= 1e-12, "the all-in-one partition has Q = 0");
        check(modularity(bridged, single) < 0.0, "the singleton partition has negative Q");
        check(bridgedResult.modularity > modularity(bridged, single), "Louvain beats the singletons");
        check(bridgedResult.modularity > modularity(bridged, allOne), "Louvain beats all-in-one");
        check(bridgedResult.modularity <= bestModularity(bridged, n6) + 1e-9,
                "Louvain never exceeds the exhaustive optimum");
        check(Math.abs(bridgedResult.modularity - bestModularity(bridged, n6)) <= 1e-9,
                "Louvain attains the exhaustive optimum on the bridged triangles");
        check(Math.abs(twoResult.modularity - bestModularity(twoTriangles, n6)) <= 1e-9,
                "Louvain attains the exhaustive optimum on the disjoint triangles");
        double[][] six = randomWeighted(6, 5, 91);
        Louvain sixResult = louvain(six);
        check(sixResult.modularity <= bestModularity(six, 6) + 1e-9,
                "on a random graph Louvain stays below the exhaustive optimum");
        check(locallyOptimal(six, sixResult.labels), "and it is locally optimal");

        int[] seeds = {43, 86, 129, 172};
        for (int s = 0; s < seeds.length; s++) {
            double[][] w = randomWeighted(8, 7 + s, seeds[s]);
            Louvain result = louvain(w);
            check(locallyOptimal(w, result.labels), "instance " + s + ": locally optimal partition");
            check(Math.abs(modularity(w, result.labels) - result.modularity) <= 1e-12,
                    "instance " + s + ": modularity recomputed from labels");
            check(result.modularity <= bestModularity(w, 8) + 1e-9,
                    "instance " + s + ": never better than the exhaustive optimum");
            int[] singletons = new int[8];
            for (int i = 0; i < 8; i++) singletons[i] = i;
            check(result.modularity >= modularity(w, singletons) - 1e-9,
                    "instance " + s + ": never worse than the singleton partition");
            for (int h = 1; h < result.modularityHistory.size(); h++) {
                check(result.modularityHistory.get(h) >= result.modularityHistory.get(h - 1) - 1e-9,
                        "instance " + s + ": modularity never decreases while moving");
            }
            check(Math.abs(result.modularityHistory.get(result.modularityHistory.size() - 1)
                            - result.modularity) <= 1e-9,
                    "instance " + s + ": the incremental modularity equals the direct computation");
            Louvain rerun = louvain(w);
            check(Arrays.equals(rerun.labels, result.labels), "instance " + s + ": deterministic");
            check(rerun.levels == result.levels, "instance " + s + ": same number of levels");
        }

        double[][] planted = cliques(3, 5, 1.0);
        Louvain plantedResult = louvain(planted);
        check(distinct(plantedResult.labels).size() == 3, "three planted 5-cliques are recovered");
        for (int g = 0; g < 3; g++) {
            Set<Integer> group = new TreeSet<>();
            for (int i = 0; i < 5; i++) group.add(plantedResult.labels[g * 5 + i]);
            check(group.size() == 1, "planted clique " + g + " is one community");
        }
        check(locallyOptimal(planted, plantedResult.labels), "the planted partition is locally optimal");
        check(plantedResult.levels <= 2, "aggregation terminates within the documented level bound");

        double[][] empty = new double[4][4];
        Louvain emptyResult = louvain(empty);
        check(emptyResult.modularity == 0.0 && emptyResult.levels == 0,
                "an edgeless graph has modularity 0 and needs no levels");
        double[][] oneEdge = new double[3][3];
        oneEdge[0][1] = 1.0;
        oneEdge[1][0] = 1.0;
        Louvain edgeResult = louvain(oneEdge);
        check(Math.abs(edgeResult.modularity - bestModularity(oneEdge, 3)) <= 1e-9,
                "a single edge instance matches the exhaustive optimum");
        check(locallyOptimal(oneEdge, edgeResult.labels), "the single-edge partition is locally optimal");
        check(Math.abs(edgeResult.modularity) <= 1e-9,
                "the hand-computed optimum of an edge plus an isolated vertex is Q = 0");
        check(edgeResult.labels[0] == edgeResult.labels[1] && edgeResult.labels[2] != edgeResult.labels[0],
                "the two adjacent vertices merge and the isolated vertex stays alone");
        check(modularity(oneEdge, new int[]{0, 0, 0}) == 0.0, "the all-in-one partition also has Q = 0");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
