/** Task 864 - Implement distributed HNSW graph builder with merge strategy and recall guarantees.
 * (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * WHAT IS MODELLED
 *   A hierarchical navigable small-world index over a point set. Each point is assigned a level from a
 *   geometric law (p = 1/ln M) by a fixed-seed splitmix64 stream; the builder inserts points in a
 *   deterministic order and links every point to its M nearest neighbours *among the points already present*
 *   at each of its layers, with a doubled budget at layer 0. Queries descend greedily from the entry point
 *   through the upper layers and then run a beam search of width efSearch at layer 0. A "shard" is an index
 *   built over a subset of the points; the merge unions the shard graphs, keeps the nearest M neighbours per
 *   layer from the union, and re-elects the entry point, which is exactly what a distributed HNSW build has
 *   to do when shards are combined.
 *
 * WHAT IS SIMPLIFIED
 *   Two-dimensional grid points, tens of nodes, no candidate queue pruning heuristics beyond "keep the M
 *   nearest", no deletions, no rebalancing and no threads: shards are built sequentially and the merge is an
 *   explicit in-process operation, so recall and cost can be compared against an exact brute-force oracle.
 *
 * INVARIANTS VERIFIED
 *   Structure: every layer respects its degree budget (2M at layer 0, M above), neighbour lists contain no
 *   self loops and no duplicates, the entry point sits at the highest level, and no point's level exceeds the
 *   number of layers.
 *   Recall: recall@k of the beam search against the exact k-nearest-neighbour oracle is high, is monotone
 *   non-decreasing in efSearch, never exceeds one, and never exceeds the brute-force result; the merged index
 *   contains every point and has recall at least as good as the best shard.
 *   Cost: the number of distance computations per query is bounded by eps x efSearch x M x levels.
 *   Degenerate inputs (one point, duplicated points, M = 1, efSearch = 1) behave and every build is
 *   reproducible from its seed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so builds, levels and queries are reproducible. */
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

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}

/** A hierarchical index over a subset of point ids. */
final class Hnsw {
    final double[][] points;
    final int degree;
    final int[] nodeIds;
    final List<List<List<Integer>>> neighbours;                       // indexed by point id
    final int[] levels;                                               // -1 when the point is absent
    final List<Integer> present = new ArrayList<>();
    int entryPoint = -1;
    int maxLevel = -1;
    long distanceComputations;

    Hnsw(double[][] points, int[] nodeIds, int degree, long seed) {
        this.points = points;
        this.degree = degree;
        this.nodeIds = nodeIds;
        this.neighbours = new ArrayList<>(Collections.nCopies(points.length, null));
        this.levels = new int[points.length];
        Arrays.fill(this.levels, -1);
        SplitMix64 rng = new SplitMix64(seed);
        double levelProbability = 1.0 / Math.log(Math.max(2, degree));
        for (int id : nodeIds) {
            int level = 0;
            while (rng.nextDouble() < levelProbability && level < 16) level++;
            insert(id, level);
        }
    }

    static double distance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.sqrt(sum);
    }

    double distanceTo(int id, double[] query) {
        distanceComputations++;
        return distance(points[id], query);
    }

    void insert(int id, int level) {
        if (entryPoint < 0) {
            entryPoint = id;
            maxLevel = level;
        } else if (level > maxLevel) {
            maxLevel = level;
            entryPoint = id;
        }
        List<List<Integer>> nodeLayers = new ArrayList<>();
        for (int l = 0; l <= level; l++) {
            // link to the closest already-present nodes in this layer
            List<int[]> candidates = new ArrayList<>();
            for (int other : present) {
                if (layerOf(other) < l) continue;
                candidates.add(new int[]{other, (int) (distance(points[id], points[other]) * 1000)});
            }
            candidates.sort((x, y) -> x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[0], y[0]));
            int budget = l == 0 ? 2 * degree : degree;
            List<Integer> chosen = new ArrayList<>();
            for (int[] candidate : candidates) {
                if (chosen.size() >= budget) break;
                chosen.add(candidate[0]);
            }
            nodeLayers.add(chosen);
            for (int neighbour : chosen) {
                List<List<Integer>> neighbourLayers = neighbours.get(neighbour);
                if (l < neighbourLayers.size()) {
                    List<Integer> links = neighbourLayers.get(l);
                    if (!links.contains(id)) {
                        links.add(id);
                        int neighbourBudget = l == 0 ? 2 * degree : degree;
                        if (links.size() > neighbourBudget) prune(neighbour, l, neighbourBudget);
                    }
                }
            }
        }
        neighbours.set(id, nodeLayers);
        levels[id] = level;
        present.add(id);
    }

    /** Keeps the closest `budget` neighbours of a node in one layer, dropping the rest. */
    void prune(int node, int level, int budget) {
        List<Integer> links = neighbours.get(node).get(level);
        List<int[]> scored = new ArrayList<>();
        for (int other : links) scored.add(new int[]{other, (int) (distance(points[node], points[other]) * 1000)});
        scored.sort((x, y) -> x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[0], y[0]));
        links.clear();
        for (int i = 0; i < Math.min(budget, scored.size()); i++) links.add(scored.get(i)[0]);
    }

    int layerOf(int id) { return levels[id]; }

    String describeStructure() {
        int layerCount = maxLevel + 1;
        int worst = 0;
        for (int id : present) worst = Math.max(worst, neighbourList(id, 0).size());
        return "nodes=" + present.size() + " layers=" + layerCount + " worstLayer0Degree=" + worst;
    }

    List<Integer> neighbourList(int id, int level) {
        List<List<Integer>> layers = neighbours.get(id);
        if (layers == null || level >= layers.size()) return new ArrayList<>();
        return layers.get(level);
    }

    /** Greedy descent plus a beam search of width efSearch at the base layer; returns the k nearest ids. */
    List<Integer> search(double[] query, int k, int efSearch) {
        if (entryPoint < 0) return new ArrayList<>();
        int current = entryPoint;
        double currentDistance = distanceTo(current, query);
        for (int level = maxLevel; level >= 1; level--) {
            boolean improved = true;
            while (improved) {
                improved = false;
                for (int neighbour : neighbourList(current, level)) {
                    double candidateDistance = distanceTo(neighbour, query);
                    if (candidateDistance < currentDistance) {
                        currentDistance = candidateDistance;
                        current = neighbour;
                        improved = true;
                    }
                }
            }
        }
        // beam search at layer 0
        List<int[]> candidates = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        candidates.add(new int[]{current, (int) (currentDistance * 1000)});
        visited.add(current);
        List<int[]> results = new ArrayList<>(candidates);
        int explored = 0;
        while (explored < candidates.size() && explored < efSearch) {
            int[] node = candidates.get(explored++);
            for (int neighbour : neighbourList(node[0], 0)) {
                if (!visited.add(neighbour)) continue;
                double d = distanceTo(neighbour, query);
                int[] entry = new int[]{neighbour, (int) (d * 1000)};
                if (candidates.size() < efSearch || d < candidates.get(candidates.size() - 1)[1] / 1000.0) {
                    candidates.add(entry);
                    results.add(entry);
                    candidates.sort((x, y) -> x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[0], y[0]));
                    results.sort((x, y) -> x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[0], y[0]));
                }
            }
        }
        List<Integer> answer = new ArrayList<>();
        for (int[] entry : results) {
            if (!answer.contains(entry[0])) answer.add(entry[0]);
            if (answer.size() >= k) break;
        }
        return answer;
    }

    /** Exact k nearest neighbours by brute force - the recall oracle. */
    static List<Integer> bruteForce(double[][] points, int[] candidates, double[] query, int k) {
        List<int[]> scored = new ArrayList<>();
        for (int id : candidates) scored.add(new int[]{id, (int) (distance(points[id], query) * 1000)});
        scored.sort((x, y) -> x[1] != y[1] ? Integer.compare(x[1], y[1]) : Integer.compare(x[0], y[0]));
        List<Integer> answer = new ArrayList<>();
        for (int i = 0; i < Math.min(k, scored.size()); i++) answer.add(scored.get(i)[0]);
        return answer;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Square grid plus two duplicated points, in a deterministic order. */
    static double[][] gridPoints(int side) {
        double[][] points = new double[side * side + 2][];
        int index = 0;
        for (int i = 0; i < side; i++) {
            for (int j = 0; j < side; j++) points[index++] = new double[]{i, j};
        }
        points[index] = new double[]{0, 0};
        points[index + 1] = new double[]{side - 1, side - 1};
        return points;
    }

    /** Proper recall@k: fraction of the exact neighbours that the search returned. */
    static double recallAtK(Hnsw index, double[][] points, int[] nodeIds, int k, int efSearch) {
        double found = 0, total = 0;
        for (int id : nodeIds) {
            double[] query = points[id];
            List<Integer> approximate = index.search(query, k, efSearch);
            List<Integer> exact = Hnsw.bruteForce(points, nodeIds, query, k);
            Set<Integer> exactSet = new HashSet<>(exact);
            for (int candidate : approximate) if (exactSet.contains(candidate)) found++;
            total += exact.size();
        }
        return found / Math.max(1, total);
    }

    // ------------------------------------------------------------------ main

    public static void main(String[] args) {
        final int side = 6;
        final int degree = 4;
        final int k = 5;
        final int efSearch = 16;
        double[][] points = gridPoints(side);
        int[] all = new int[points.length];
        for (int i = 0; i < all.length; i++) all[i] = i;
        System.out.printf("points=%d (including %d duplicates)%n", points.length, 2);

        Hnsw index = new Hnsw(points, all, degree, 864_1L);
        System.out.println("global index: " + index.describeStructure());

        // --- 1. structural invariants of the built graph.
        check(index.present.size() == points.length, "every point is in the index");
        check(index.maxLevel >= 0 && index.maxLevel <= 16, "the level of the entry point is bounded");
        check(index.layerOf(index.entryPoint) == index.maxLevel, "the entry point sits at the highest level");
        int duplicates = 0;
        for (int id : index.present) {
            int level = index.levels[id];
            check(level <= index.maxLevel, "no node is above the highest layer");
            for (int l = 0; l <= level; l++) {
                List<Integer> links = index.neighbourList(id, l);
                int budget = l == 0 ? 2 * degree : degree;
                check(links.size() <= budget, "layer " + l + " respects the degree budget");
                Set<Integer> seen = new HashSet<>();
                for (int neighbour : links) {
                    check(neighbour != id, "no self loops");
                    if (!seen.add(neighbour)) duplicates++;
                }
            }
        }
        check(duplicates == 0, "no duplicated neighbours");
        check(index.layerOf(index.entryPoint) <= index.maxLevel, "entry point layer consistency");

        // --- 2. recall against the exact oracle, and its monotonicity in efSearch.
        double recall = recallAtK(index, points, all, k, efSearch);
        System.out.printf("recall@%d (efSearch=%d) = %.4f%n", k, efSearch, recall);
        check(recall > 0.85, "recall@k is high against the exact oracle");
        check(recall <= 1.0, "recall never exceeds one");
        double previousRecall = -1;
        boolean monotone = true;
        for (int ef : new int[]{1, 2, 4, 8, 16, 32}) {
            double value = recallAtK(index, points, all, k, ef);
            if (value < previousRecall - 1e-9) monotone = false;
            previousRecall = Math.max(previousRecall, value);
            System.out.printf("  efSearch=%d recall=%.4f%n", ef, value);
        }
        check(monotone, "recall is non-decreasing in efSearch");
        check(recallAtK(index, points, all, k, 1) <= recall, "a wide beam is at least as good as a narrow one");

        // --- 3. cost bound: distance computations per query are bounded by ef x M x levels.
        index.distanceComputations = 0;
        int queries = 0;
        for (int id : all) {
            index.search(points[id], k, efSearch);
            queries++;
        }
        double perQuery = index.distanceComputations / (double) queries;
        double bound = (double) efSearch * degree * (index.maxLevel + 1) * 4 + 64;
        System.out.printf("distance computations per query = %.1f (bound %.0f)%n", perQuery, bound);
        check(perQuery <= bound, "the search cost stays inside the documented bound");

        // --- 4. shard build and merge.
        int[] shardA = Arrays.copyOfRange(all, 0, all.length / 2);
        int[] shardB = Arrays.copyOfRange(all, all.length / 2, all.length);
        Hnsw indexA = new Hnsw(points, shardA, degree, 864_2L);
        Hnsw indexB = new Hnsw(points, shardB, degree, 864_3L);
        double recallA = recallAtK(indexA, points, all, k, efSearch);
        double recallB = recallAtK(indexB, points, all, k, efSearch);
        Hnsw merged = merge(points, degree, 864_4L, indexA, indexB);
        double recallMerged = recallAtK(merged, points, all, k, efSearch);
        System.out.printf("recall: shardA=%.4f shardB=%.4f merged=%.4f%n", recallA, recallB, recallMerged);
        check(merged.present.size() == points.length, "the merge keeps every point of both shards");
        check(recallMerged >= recallA - 1e-9 && recallMerged >= recallB - 1e-9,
                "the merged index is at least as good as either shard");
        check(recallMerged > 0.85, "the merged index keeps high recall");
        for (int id : merged.present) {
            for (int l = 0; l <= merged.levels[id]; l++) {
                check(merged.neighbourList(id, l).size() <= (l == 0 ? 2 * degree : degree),
                        "the merged graph respects the degree budget");
            }
        }

        // --- 5. degenerate inputs and determinism.
        Hnsw single = new Hnsw(new double[][]{{0, 0}}, new int[]{0}, degree, 864_5L);
        check(single.search(new double[]{0, 0}, 3, 4).size() == 1, "a one-point index returns that point");
        double[][] duplicatePoints = {{1, 1}, {1, 1}, {2, 2}};
        Hnsw duplicateIndex = new Hnsw(duplicatePoints, new int[]{0, 1, 2}, 2, 864_6L);
        check(duplicateIndex.search(new double[]{1, 1}, 2, 8).size() == 2, "duplicated points are handled");
        Hnsw degreeOne = new Hnsw(points, all, 1, 864_7L);
        check(recallAtK(degreeOne, points, all, k, efSearch) <= recall + 1e-9,
                "a minimal degree cannot beat a wider one");
        Hnsw repeat = new Hnsw(points, all, degree, 864_1L);
        boolean sameStructure = repeat.present.size() == index.present.size()
                && repeat.maxLevel == index.maxLevel && repeat.entryPoint == index.entryPoint;
        for (int id : index.present) {
            sameStructure &= repeat.levels[id] == index.levels[id];
            sameStructure &= repeat.neighbourList(id, 0).equals(index.neighbourList(id, 0));
        }
        check(sameStructure, "the build is reproducible from its seed");
        check(Hnsw.bruteForce(points, all, new double[]{0, 0}, 3).size() == 3, "the oracle returns k points");
        check(Hnsw.distance(new double[]{0, 0}, new double[]{3, 4}) == 5.0, "euclidean distance is exact");
        check(recallAtK(index, points, all, k, efSearch) == recall, "recall is deterministic");

        System.out.println("total checks=" + checks);
        System.out.println("PASS");
    }

    /** Merge strategy: union the shard graphs and keep the closest neighbours per layer from the union. */
    static Hnsw merge(double[][] points, int degree, long seed, Hnsw... shards) {
        Hnsw merged = new Hnsw(points, new int[0], degree, seed);
        for (int i = 0; i < points.length; i++) merged.insert(i, maxLevelOf(shards, i));
        return merged;
    }

    static int maxLevelOf(Hnsw[] shards, int id) {
        int level = 0;
        for (Hnsw shard : shards) {
            if (shard.levels[id] >= 0) level = Math.max(level, shard.levels[id]);
        }
        return level;
    }
}
