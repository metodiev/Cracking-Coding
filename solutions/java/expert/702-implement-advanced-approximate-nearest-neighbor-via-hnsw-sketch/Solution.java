/** Task 702 - Implement advanced approximate nearest neighbor via HNSW sketch.
 * (Tier: Expert, CTCI Beyond -- Advanced Data Structures & Algorithms)
 *
 * HNSW (Malkov & Yashunin) keeps the points in a hierarchy of proximity graphs. Every inserted point
 * draws a level from a geometric distribution with P(level >= l) = M^-l, so layer 0 holds all n points
 * while each higher layer holds a ~1/M thinning of it. Insertion starts at the top layer and greedily
 * descends to the point's own level; on every layer from there down to 0 it runs an efConstruction
 * beam search (a best-first traversal over the layer, bounded by a max-heap of the ef closest points
 * seen so far), keeps the M closest results as the new point's out-links and adds the reciprocal
 * links, pruning the neighbour's list with the same rule. Search repeats the descent with a beam of 1
 * above layer 0 and an ef-wide beam at layer 0, then reports the k closest points of the result heap.
 *
 * Simplified: neighbour selection keeps the M closest candidates (selectNeighborsSimple) instead of
 * the paper's diversity heuristic (which re-inserts discarded candidates), and the beam search uses
 * "stop as soon as the closest candidate is worse than the current worst" - the same termination rule
 * the paper describes - without the additional heuristic pruning of layer-0 links. Levels come from a
 * deterministic splitmix64 stream instead of random(), so runs are reproducible.
 *
 * Invariants verified: every node is reachable from the entry point on layer 0 (checked by an explicit
 * BFS over the built graph), degrees respect maxDegreeLayer0 = 2M on layer 0 and M elsewhere, the
 * distance sequence returned by a search is sorted, ids are distinct and in range, a search with
 * ef = n returns exactly the brute-force top-k, and self-queries return the query point itself with
 * distance 0. Utility verified: recall@10 against brute force reaches a high level at ef = 32 while
 * the number of point-to-point distances evaluated stays far below the n per query brute force needs,
 * and two builds with the same seed produce identical graphs and identical answers.
 */
import java.util.*;

class Hnsw {
    static final class Neighbor {
        final int id;
        final double distance;

        Neighbor(int id, double distance) {
            this.id = id;
            this.distance = distance;
        }

        @Override
        public String toString() {
            return id + "@" + String.format(Locale.ROOT, "%.4f", distance);
        }
    }

    private final int dim;
    private final int maxDegree;
    private final int maxDegreeLayer0;
    private final int efConstruction;
    private final double levelMultiplier;
    private final long seed;
    private final List<double[]> points = new ArrayList<>();
    private final List<Integer> levels = new ArrayList<>();
    private final List<List<List<Integer>>> links = new ArrayList<>();   // [node][layer] -> ids
    private int entryPoint = -1;
    private int topLevel = -1;
    private long distanceComputations;

    Hnsw(int dim, int maxDegree, int efConstruction, long seed) {
        if (dim <= 0) throw new IllegalArgumentException("dimension must be positive: " + dim);
        if (maxDegree < 2) throw new IllegalArgumentException("degree must be at least 2: " + maxDegree);
        this.dim = dim;
        this.maxDegree = maxDegree;
        this.maxDegreeLayer0 = 2 * maxDegree;
        this.efConstruction = Math.max(efConstruction, maxDegree);
        this.levelMultiplier = 1.0 / Math.log(maxDegree);
        this.seed = seed;
    }

    int size() {
        return points.size();
    }

    int topLevel() {
        return topLevel;
    }

    int levelOf(int node) {
        return levels.get(node);
    }

    double[] point(int node) {
        return points.get(node).clone();
    }

    List<Integer> neighbors(int node, int layer) {
        return Collections.unmodifiableList(links.get(node).get(layer));
    }

    long distanceComputations() {
        return distanceComputations;
    }

    void add(double[] vector) {
        if (vector.length != dim) throw new IllegalArgumentException("expected dimension " + dim);
        int id = points.size();
        double[] copy = vector.clone();
        points.add(copy);
        int level = levelFor(id);
        levels.add(level);
        List<List<Integer>> node = new ArrayList<>();
        for (int l = 0; l <= level; l++) node.add(new ArrayList<>());
        links.add(node);
        if (entryPoint < 0) {
            entryPoint = id;
            topLevel = level;
            return;
        }
        int current = entryPoint;
        for (int l = topLevel; l > level; l--) current = greedyClosest(copy, current, l);
        for (int l = Math.min(topLevel, level); l >= 0; l--) {
            List<Neighbor> found = searchLayer(copy, current, efConstruction, l);
            int capacity = l == 0 ? maxDegreeLayer0 : maxDegree;
            List<Integer> chosen = new ArrayList<>();
            for (int i = 0; i < found.size() && chosen.size() < capacity; i++) chosen.add(found.get(i).id);
            links.get(id).set(l, chosen);
            for (int other : chosen) {
                List<Integer> back = links.get(other).get(l);
                back.add(id);
                if (back.size() > capacity) prune(other, back, capacity);
            }
            if (!found.isEmpty()) current = found.get(0).id;
        }
        if (level > topLevel) {
            topLevel = level;
            entryPoint = id;
        }
    }

    /** Layer 0 search with a beam of ef; returns the ef closest points found, sorted by distance. */
    List<Neighbor> search(double[] query, int k, int ef) {
        if (query.length != dim) throw new IllegalArgumentException("expected dimension " + dim);
        List<Neighbor> result = new ArrayList<>();
        if (entryPoint < 0 || k <= 0) return result;
        int current = entryPoint;
        for (int l = topLevel; l >= 1; l--) current = greedyClosest(query, current, l);
        List<Neighbor> candidates = searchLayer(query, current, Math.max(ef, k), 0);
        for (int i = 0; i < Math.min(k, candidates.size()); i++) result.add(candidates.get(i));
        return result;
    }

    /** Exhaustive distance scan - the oracle the recall numbers are measured against. */
    List<Neighbor> bruteForce(double[] query, int k) {
        List<Neighbor> all = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) all.add(new Neighbor(i, distance(query, points.get(i))));
        all.sort(Comparator.comparingDouble(neighbor -> neighbor.distance));
        return new ArrayList<>(all.subList(0, Math.min(k, all.size())));
    }

    private int greedyClosest(double[] query, int entry, int layer) {
        int current = entry;
        double best = distance(query, points.get(current));
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int candidate : layerLinks(current, layer)) {
                double d = distance(query, points.get(candidate));
                if (d < best) {
                    best = d;
                    current = candidate;
                    improved = true;
                }
            }
        }
        return current;
    }

    private List<Neighbor> searchLayer(double[] query, int entry, int ef, int layer) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Neighbor> candidates = new PriorityQueue<>(Comparator.comparingDouble(n -> n.distance));
        PriorityQueue<Neighbor> closest = new PriorityQueue<>((a, b) -> Double.compare(b.distance, a.distance));
        double first = distance(query, points.get(entry));
        visited.add(entry);
        candidates.add(new Neighbor(entry, first));
        closest.add(new Neighbor(entry, first));
        while (!candidates.isEmpty()) {
            Neighbor current = candidates.poll();
            if (closest.size() >= ef && current.distance > closest.peek().distance) break;
            for (int candidate : layerLinks(current.id, layer)) {
                if (!visited.add(candidate)) continue;
                double d = distance(query, points.get(candidate));
                if (closest.size() < ef || d < closest.peek().distance) {
                    candidates.add(new Neighbor(candidate, d));
                    closest.add(new Neighbor(candidate, d));
                    if (closest.size() > ef) closest.poll();
                }
            }
        }
        List<Neighbor> result = new ArrayList<>(closest);
        result.sort(Comparator.comparingDouble(neighbor -> neighbor.distance));
        return result;
    }

    private List<Integer> layerLinks(int node, int layer) {
        List<List<Integer>> nodeLinks = links.get(node);
        return layer < nodeLinks.size() ? nodeLinks.get(layer) : Collections.<Integer>emptyList();
    }

    private void prune(int owner, List<Integer> list, int capacity) {
        List<Neighbor> ranked = new ArrayList<>();
        for (int other : list) ranked.add(new Neighbor(other, distance(points.get(owner), points.get(other))));
        ranked.sort(Comparator.comparingDouble(neighbor -> neighbor.distance));
        list.clear();
        for (int i = 0; i < capacity; i++) list.add(ranked.get(i).id);
    }

    private int levelFor(int id) {
        double uniform = uniformFromHash(id);
        return (int) Math.floor(-Math.log(uniform) * levelMultiplier);
    }

    /** Deterministic uniform in (0, 1) derived from the seed and the node id. */
    private double uniformFromHash(int id) {
        long z = seed + 0x9E3779B97F4A7C15L * (id + 1);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z ^= z >>> 31;
        double uniform = (z >>> 11) * 0x1.0p-53;
        return uniform == 0.0 ? Double.MIN_VALUE : uniform;
    }

    private double distance(double[] a, double[] b) {
        distanceComputations++;
        double sum = 0;
        for (int i = 0; i < dim; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}

public class Solution {
    private static long state = 0xC0FFEE1234567L;

    /** splitmix64 - deterministic PRNG, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    static double[] randomPoint(int dim) {
        double[] point = new double[dim];
        for (int i = 0; i < dim; i++) point[i] = nextDouble() * 2.0 - 1.0;
        return point;
    }

    static Set<Integer> ids(List<Hnsw.Neighbor> neighbors) {
        Set<Integer> out = new TreeSet<>();
        for (Hnsw.Neighbor neighbor : neighbors) out.add(neighbor.id);
        return out;
    }

    public static void main(String[] args) {
        // --- degenerate indexes ------------------------------------------------
        Hnsw empty = new Hnsw(4, 8, 16, 7);
        check(empty.size() == 0 && empty.search(new double[]{0, 0, 0, 0}, 5, 16).isEmpty(),
                "empty index returns no neighbours");
        check(empty.topLevel() == -1, "empty index has no top level");

        Hnsw single = new Hnsw(3, 4, 8, 11);
        single.add(new double[]{0.5, -0.5, 0.25});
        List<Hnsw.Neighbor> singleResult = single.search(new double[]{0.5, -0.5, 0.25}, 5, 8);
        check(singleResult.size() == 1 && singleResult.get(0).id == 0, "single point is its own neighbour");
        check(singleResult.get(0).distance == 0.0, "single point distance exactly zero");
        check(single.neighbors(0, 0).isEmpty(), "single point has no layer 0 links");

        Hnsw duplicates = new Hnsw(2, 4, 8, 13);
        for (int i = 0; i < 3; i++) duplicates.add(new double[]{1.0, 2.0});
        List<Hnsw.Neighbor> duplicateResult = duplicates.search(new double[]{1.0, 2.0}, 3, 8);
        check(duplicateResult.size() == 3, "duplicate points all returned, got " + duplicateResult);
        for (Hnsw.Neighbor neighbor : duplicateResult) {
            check(neighbor.distance == 0.0, "identical points are at distance zero");
        }

        boolean rejected = false;
        try {
            single.add(new double[]{0.0, 0.0});
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "wrong dimension must be rejected");
        rejected = false;
        try {
            new Hnsw(2, 1, 8, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "degree below two must be rejected");

        // --- the real sketch on a random data set ------------------------------
        int dimension = 8;
        int cardinality = 600;
        int maxDegree = 8;
        Hnsw index = new Hnsw(dimension, maxDegree, 32, 0xABCDEF);
        double[][] data = new double[cardinality][];
        for (int i = 0; i < cardinality; i++) {
            data[i] = randomPoint(dimension);
            index.add(data[i]);
        }
        check(index.size() == cardinality, "every point was inserted");
        check(index.topLevel() >= 1, "with n = " + cardinality + " some node must rise above layer 0");

        int[] levelHistogram = new int[index.topLevel() + 2];
        for (int i = 0; i < cardinality; i++) levelHistogram[index.levelOf(i)]++;
        check(levelHistogram[0] < cardinality && levelHistogram[0] > cardinality / 2,
                "layer 0 holds most but not all points: " + levelHistogram[0]);
        int raised = cardinality - levelHistogram[0];
        check(raised >= cardinality / 40 && raised <= cardinality / 3,
                "level >= 1 population looks geometric: " + raised);

        // --- graph invariants ---------------------------------------------------
        boolean[] reachable = new boolean[cardinality];
        Deque<Integer> queue = new ArrayDeque<>();
        int entry = 0;
        for (int i = 0; i < cardinality; i++) {
            if (index.levelOf(i) == index.topLevel()) {
                entry = i;
                break;
            }
        }
        reachable[entry] = true;
        queue.add(entry);
        int visitedCount = 1;
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int other : index.neighbors(node, 0)) {
                if (!reachable[other]) {
                    reachable[other] = true;
                    visitedCount++;
                    queue.add(other);
                }
            }
        }
        check(visitedCount == cardinality, "layer 0 must be connected: reached " + visitedCount);
        for (int node = 0; node < cardinality; node++) {
            for (int layer = 0; layer <= index.levelOf(node); layer++) {
                int cap = layer == 0 ? 2 * maxDegree : maxDegree;
                check(index.neighbors(node, layer).size() <= cap,
                        "node " + node + " layer " + layer + " degree above " + cap);
            }
        }

        // --- self queries: the nearest neighbour of a point is the point -------
        for (int node = 0; node < cardinality; node += 7) {
            List<Hnsw.Neighbor> self = index.search(data[node], 1, 16);
            check(!self.isEmpty() && self.get(0).id == node && self.get(0).distance == 0.0,
                    "self query for node " + node + " returned " + self);
        }

        // --- exactness: a beam as wide as the data set reproduces brute force ---
        for (int query = 0; query < 6; query++) {
            double[] probe = randomPoint(dimension);
            List<Hnsw.Neighbor> exact = index.search(probe, 10, cardinality);
            List<Hnsw.Neighbor> brute = index.bruteForce(probe, 10);
            check(ids(exact).equals(ids(brute)), "ef = n must be exact for query " + query
                    + ": " + exact + " vs " + brute);
        }

        // --- recall and the cost of approximation -------------------------------
        int queries = 40;
        int recalled = 0;
        int total = 0;
        double worstRecall = 1.0;
        long searchComparisons = 0;
        long slowComparisons = 0;
        for (int query = 0; query < queries; query++) {
            double[] probe = randomPoint(dimension);
            List<Hnsw.Neighbor> brute = index.bruteForce(probe, 10);
            long before = index.distanceComputations();
            List<Hnsw.Neighbor> approximate = index.search(probe, 10, 32);
            searchComparisons += index.distanceComputations() - before;
            check(approximate.size() == 10, "recall probe must return ten neighbours");
            for (int i = 1; i < approximate.size(); i++) {
                check(approximate.get(i - 1).distance <= approximate.get(i).distance,
                        "results must be sorted by distance");
            }
            Set<Integer> approximateIds = ids(approximate);
            check(approximateIds.size() == approximate.size(), "ids must be distinct");
            Set<Integer> bruteIds = ids(brute);
            bruteIds.retainAll(approximateIds);
            recalled += bruteIds.size();
            total += brute.size();
            worstRecall = Math.min(worstRecall, bruteIds.size() / 10.0);
            slowComparisons += cardinality;
        }
        double recall = recalled / (double) total;
        check(recall >= 0.90, "recall@10 at ef = 32 must be high, measured " + recall);
        check(worstRecall >= 0.5, "no query may collapse completely: " + worstRecall);
        check(searchComparisons < slowComparisons / 2, "approximate search must beat a linear scan: "
                + searchComparisons + " vs " + slowComparisons);

        // --- k larger than the data set, and determinism -----------------------
        double[] probe = randomPoint(dimension);
        check(index.search(probe, cardinality + 50, 64).size() == cardinality,
                "k above n returns every point");
        check(index.search(probe, 0, 16).isEmpty(), "k = 0 returns nothing");

        Hnsw rebuilt = new Hnsw(dimension, maxDegree, 32, 0xABCDEF);
        for (int i = 0; i < cardinality; i++) rebuilt.add(data[i]);
        for (int node = 0; node < cardinality; node++) {
            for (int layer = 0; layer <= Math.min(rebuilt.levelOf(node), 1); layer++) {
                check(rebuilt.neighbors(node, layer).equals(index.neighbors(node, layer)),
                        "the same seed must build the same graph at node " + node + " layer " + layer);
            }
        }
        List<Hnsw.Neighbor> firstRun = index.search(probe, 10, 32);
        List<Hnsw.Neighbor> secondRun = rebuilt.search(probe, 10, 32);
        check(firstRun.toString().equals(secondRun.toString()), "identical search results across builds");

        System.out.println("recall@10 (ef=32) = " + String.format(Locale.ROOT, "%.3f", recall)
                + ", distances evaluated = " + searchComparisons + " vs linear scan " + slowComparisons);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
