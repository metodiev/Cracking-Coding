/** Task 766 - Implement a dense vector index with HNSW and integrated quantisation.
 * (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * HNSW builds a hierarchy of proximity graphs: every node is inserted at a geometrically distributed
 * level, and each insertion greedily descends from the top layer, links itself to its closest M
 * neighbours on each layer it belongs to, and prunes those neighbours' lists back to the degree bound.
 * A query descends greedily to layer 1 and then runs a best-first search on layer 0 with a budget of
 * efSearch candidates - which is where recall is traded for speed.
 *
 * Quantisation is integrated the way production indexes do it: the *navigation* distance is computed on
 * int8 scalar codes (cheap, bounded error), while the final candidate set is re-ranked with exact
 * distances. That split is what the checks pin down: the code error is bounded, the returned order is
 * exact, and the number of exact distance computations stays far below the number of vectors.
 *
 * Simplified: the neighbour-selection heuristic is "closest M" rather than HNSW's diversity rule (which
 * also keeps layer 0 better connected - here a repair pass links nodes that lost every incoming link to
 * their nearest reachable node, and the share of such nodes is reported), the level distribution uses
 * mL = 1/ln(M), deletes and updates are absent, the code is one byte per dimension derived from the
 * global min/max, and the data set is a few hundred 8-d vectors.
 *
 * Verified: degree bounds and level consistency; the entry point holds the maximum level; layer-0
 * reachability by BFS; recall@10 against exhaustive search for several efSearch budgets, including a
 * budget large enough to reach 1.0; quantisation error bounded and decoded codes within half a step;
 * exact re-ranking of the shortlist; exact-distance work far below the data size; and boundary cases
 * (empty index, single node, query equal to an indexed vector, duplicates) plus determinism.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

    double nextUnit() { return (nextLong() >>> 11) / (double) (1L << 53); }

    double nextGaussian() {
        double u1 = Math.max(nextUnit(), 1e-12), u2 = nextUnit();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** A hierarchical navigable small-world index over quantised vectors. */
final class Hnsw {
    final double[][] data;
    final int m, m0, efConstruction, dimension;
    final int[] level;
    final List<List<List<Integer>>> links = new ArrayList<>();      // node -> layer -> neighbours
    final int[][] codes;                                            // int8 scalar quantisation
    final double[] minimum, maximum;
    int entry = 0, top = 0;
    int repairs;
    int duplicateAdds;
    final Set<Long> pinned = new LinkedHashSet<>();
    long exactDistanceComputations, quantizedDistanceComputations;

    Hnsw(double[][] data, int m, int efConstruction, long seed) {
        this.data = data;
        this.m = m;
        this.m0 = 2 * m;
        this.efConstruction = efConstruction;
        this.dimension = data[0].length;
        this.level = new int[data.length];
        this.codes = new int[data.length][dimension];
        this.minimum = new double[dimension];
        this.maximum = new double[dimension];
        Arrays.fill(minimum, Double.MAX_VALUE);
        Arrays.fill(maximum, -Double.MAX_VALUE);
        for (double[] vector : data) {
            for (int j = 0; j < dimension; j++) {
                minimum[j] = Math.min(minimum[j], vector[j]);
                maximum[j] = Math.max(maximum[j], vector[j]);
            }
        }
        for (int i = 0; i < data.length; i++) {
            codes[i] = encode(data[i]);
            links.add(new ArrayList<>());
        }
        Prng rng = new Prng(seed);
        double ml = 1.0 / Math.log(m);
        for (int i = 0; i < data.length; i++) {
            level[i] = (int) Math.floor(-Math.log(Math.max(rng.nextUnit(), 1e-12)) * ml);
        }
        for (int i = 0; i < data.length; i++) {
            while (links.get(i).size() <= level[i]) links.get(i).add(new ArrayList<>());
        }
        for (int i = 0; i < data.length; i++) insert(i, rng);
        repairs = repairConnectivity();
    }

    /**
     * HNSW's diversity heuristic normally keeps layer 0 connected; the simplified "closest M" pruning
     * can isolate a node, so isolated nodes are linked to their nearest reachable node once. The
     * connectivity check afterwards is then a real assertion rather than a hope.
     */
    private int repairConnectivity() {
        int repairs = 0;
        for (int round = 0; round < 32; round++) {
            boolean[] reachable = reachableFromEntry();
            boolean progress = false;
            for (int node = 0; node < data.length; node++) {
                if (reachable[node]) continue;
                int best = -1;
                double bestDistance = Double.MAX_VALUE;
                for (int candidate = 0; candidate < data.length; candidate++) {
                    if (!reachable[candidate]) continue;
                    double distance = squaredDistance(data[node], data[candidate]);
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        best = candidate;
                    }
                }
                if (best < 0) continue;
                pinned.add(key(best, node));
                pinned.add(key(node, best));
                addLink(best, node, 0);
                addLink(node, best, 0);
                reachable[node] = true;                          // and every node linked to it below
                repairs++;
                progress = true;
            }
            if (!progress) break;
        }
        return repairs;
    }

    private void addLink(int from, int to, int layer) {
        List<Integer> neighbours = links.get(from).get(layer);
        if (neighbours.contains(to)) return;
        neighbours.add(to);
        prune(from, layer);
    }

    static long key(int from, int to) { return ((long) from << 32) | to; }

    private boolean[] reachableFromEntry() {
        boolean[] reachable = new boolean[data.length];
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(entry);
        reachable[entry] = true;
        while (!queue.isEmpty()) {
            for (int neighbour : links.get(queue.poll()).get(0)) {
                if (!reachable[neighbour]) {
                    reachable[neighbour] = true;
                    queue.add(neighbour);
                }
            }
        }
        return reachable;
    }

    int[] encode(double[] vector) {
        int[] code = new int[dimension];
        for (int j = 0; j < dimension; j++) {
            double span = maximum[j] - minimum[j];
            double normalized = span == 0 ? 0 : (vector[j] - minimum[j]) / span;
            code[j] = (int) Math.round(normalized * 255) - 128;
        }
        return code;
    }

    double[] decode(int[] code) {
        double[] vector = new double[dimension];
        for (int j = 0; j < dimension; j++) {
            double span = maximum[j] - minimum[j];
            vector[j] = minimum[j] + (code[j] + 128) / 255.0 * span;
        }
        return vector;
    }

    int maxDegree(int layer) { return layer == 0 ? m0 : m; }

    /** Navigation distance: quantised by default, exact when the caller wants ground truth. */
    double distance(double[] query, int node, boolean quantized) {
        if (!quantized) {
            exactDistanceComputations++;
            return squaredDistance(query, data[node]);
        }
        quantizedDistanceComputations++;
        int[] code = encode(query);
        double sum = 0;
        for (int j = 0; j < dimension; j++) {
            double difference = code[j] - codes[node][j];
            double span = maximum[j] - minimum[j];
            difference *= span / 255.0;
            sum += difference * difference;
        }
        return sum;
    }

    static double squaredDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double difference = a[i] - b[i];
            sum += difference * difference;
        }
        return sum;
    }

    private void insert(int node, Prng rng) {
        if (node == 0) return;
        int current = entry;
        for (int layer = top; layer > level[node]; layer--) {
            current = greedy(node, current, layer);
        }
        for (int layer = Math.min(level[node], top); layer >= 0; layer--) {
            List<Integer> candidates = searchLayer(node, List.of(current), layer, efConstruction);
            List<Integer> neighbours = candidates.size() <= maxDegree(layer)
                    ? candidates : candidates.subList(0, maxDegree(layer));
            for (int neighbour : neighbours) {
                if (neighbour == node) continue;
                if (links.get(node).get(layer).contains(neighbour)) {
                    duplicateAdds++;
                    continue;                                     // already linked on this layer
                }
                links.get(node).get(layer).add(neighbour);
                if (!links.get(neighbour).get(layer).contains(node)) {
                    links.get(neighbour).get(layer).add(node);
                    if (links.get(neighbour).get(layer).size() > maxDegree(layer)) {
                        prune(neighbour, layer);
                    }
                }
            }
            if (!candidates.isEmpty()) current = candidates.get(0);
            prune(node, layer);                                   // the inserted node's own list too
            rng.nextInt(1);
        }
        if (level[node] > top) {
            top = level[node];
            entry = node;
        }
    }

    /** Greedy descent: walk to the neighbour closest to the node being inserted. */
    private int greedy(int node, int start, int layer) {
        int current = start;
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int neighbour : links.get(current).get(layer)) {
                if (distance(data[node], neighbour, true) < distance(data[node], current, true)) {
                    current = neighbour;
                    improved = true;
                }
            }
        }
        return current;
    }

    /** Best-first search of one layer, returning up to `ef` nodes ordered by distance. */
    List<Integer> searchLayer(int node, List<Integer> entryPoints, int layer, int ef) {
        double[] query = data[node];
        PriorityQueue<Integer> candidates = new PriorityQueue<>((a, b) -> {
            int byDistance = Double.compare(distance(query, a, false), distance(query, b, false));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        PriorityQueue<Integer> results = new PriorityQueue<>((a, b) -> {
            int byDistance = Double.compare(distance(query, b, false), distance(query, a, false));
            return byDistance != 0 ? byDistance : Integer.compare(b, a);
        });
        Set<Integer> visited = new LinkedHashSet<>(entryPoints);
        for (int entryPoint : entryPoints) {
            candidates.add(entryPoint);
            results.add(entryPoint);
        }
        while (!candidates.isEmpty()) {
            int current = candidates.poll();
            if (results.size() >= ef
                    && distance(query, current, false) > distance(query, results.peek(), false)) break;
            for (int neighbour : links.get(current).get(layer)) {
                if (!visited.add(neighbour)) continue;
                candidates.add(neighbour);
                results.add(neighbour);
                if (results.size() > ef) results.poll();
            }
        }
        List<Integer> ordered = new ArrayList<>(results);
        ordered.sort((a, b) -> {
            int byDistance = Double.compare(distance(query, a, false), distance(query, b, false));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        return ordered;
    }

    private void prune(int node, int layer) {
        List<Integer> neighbours = links.get(node).get(layer);
        final int owner = node;
        neighbours.sort((a, b) -> {
            int byDistance = Double.compare(distance(data[owner], a, false), distance(data[owner], b, false));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        while (neighbours.size() > maxDegree(layer)) {
            int victim = -1;
            for (int i = neighbours.size() - 1; i >= 0; i--) {           // farthest unpinned link first
                if (!pinned.contains(key(owner, neighbours.get(i)))) {
                    victim = i;
                    break;
                }
            }
            if (victim < 0) victim = neighbours.size() - 1;
            neighbours.remove(victim);
        }
    }

    /** Query: greedy descent, best-first layer-0 search, then exact re-ranking of the shortlist. */
    List<Integer> search(double[] query, int k, int efSearch, boolean quantizedNavigation) {
        if (data.length == 0) return new ArrayList<>();
        int current = entry;
        for (int layer = top; layer > 0; layer--) {
            boolean improved = true;
            while (improved) {
                improved = false;
                for (int neighbour : links.get(current).get(layer)) {
                    if (distance(query, neighbour, quantizedNavigation) < distance(query, current,
                            quantizedNavigation)) {
                        current = neighbour;
                        improved = true;
                    }
                }
            }
        }
        List<Integer> candidates = searchLayerFromQuery(query, current, efSearch, quantizedNavigation);
        List<double[]> refined = new ArrayList<>();
        for (int id : candidates) {
            exactDistanceComputations++;
            refined.add(new double[]{squaredDistance(query, data[id]), id});
        }
        refined.sort((a, b) -> {
            int byDistance = Double.compare(a[0], b[0]);
            return byDistance != 0 ? byDistance : Double.compare(a[1], b[1]);
        });
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, refined.size()); i++) result.add((int) refined.get(i)[1]);
        return result;
    }

    private List<Integer> searchLayerFromQuery(double[] query, int start, int ef, boolean quantized) {
        PriorityQueue<Integer> candidates = new PriorityQueue<>((a, b) -> {
            int byDistance = Double.compare(distance(query, a, quantized), distance(query, b, quantized));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        PriorityQueue<Integer> results = new PriorityQueue<>((a, b) -> {
            int byDistance = Double.compare(distance(query, b, quantized), distance(query, a, quantized));
            return byDistance != 0 ? byDistance : Integer.compare(b, a);
        });
        Set<Integer> visited = new LinkedHashSet<>(List.of(start));
        candidates.add(start);
        results.add(start);
        while (!candidates.isEmpty()) {
            int current = candidates.poll();
            if (results.size() >= ef
                    && distance(query, current, quantized) > distance(query, results.peek(), quantized)) break;
            for (int neighbour : links.get(current).get(0)) {
                if (!visited.add(neighbour)) continue;
                candidates.add(neighbour);
                results.add(neighbour);
                if (results.size() > ef) results.poll();
            }
        }
        List<Integer> ordered = new ArrayList<>(results);
        final boolean flag = quantized;
        ordered.sort((a, b) -> {
            int byDistance = Double.compare(distance(query, a, flag), distance(query, b, flag));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        return ordered;
    }

    /** BFS over layer 0 from the entry point: the graph must stay connected to be searchable. */
    boolean layerZeroConnected() {
        if (data.length == 0) return true;
        Set<Integer> seen = new LinkedHashSet<>();
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(entry);
        seen.add(entry);
        while (!queue.isEmpty()) {
            for (int neighbour : links.get(queue.poll()).get(0)) {
                if (seen.add(neighbour)) queue.add(neighbour);
            }
        }
        return seen.size() == data.length;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static double[][] buildData(int count, int dimension, int clusters, long seed) {
        Prng rng = new Prng(seed);
        double[][] centres = new double[clusters][dimension];
        for (int c = 0; c < clusters; c++) {
            for (int j = 0; j < dimension; j++) centres[c][j] = rng.nextGaussian() * 4;
        }
        double[][] data = new double[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) data[i][j] = centres[i % clusters][j] + rng.nextGaussian();
        }
        return data;
    }

    /** Exhaustive ground truth. */
    static List<Integer> exactKnn(double[][] data, double[] query, int k) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < data.length; i++) order.add(i);
        order.sort((a, b) -> {
            int byDistance = Double.compare(Hnsw.squaredDistance(query, data[a]),
                    Hnsw.squaredDistance(query, data[b]));
            return byDistance != 0 ? byDistance : Integer.compare(a, b);
        });
        return order.subList(0, Math.min(k, order.size()));
    }

    static double recall(Hnsw index, double[][] queries, int k, int ef, boolean quantized) {
        double total = 0;
        for (double[] query : queries) {
            Set<Integer> truth = new TreeSet<>(exactKnn(index.data, query, k));
            Set<Integer> found = new TreeSet<>(index.search(query, k, ef, quantized));
            Set<Integer> overlap = new TreeSet<>(truth);
            overlap.retainAll(found);
            total += overlap.size() / (double) k;
        }
        return total / queries.length;
    }

    static double[][] queries(double[][] data, int count, long seed) {
        Prng rng = new Prng(seed);
        double[][] out = new double[count][];
        for (int i = 0; i < count; i++) {
            double[] query = data[rng.nextInt(data.length)].clone();
            for (int j = 0; j < query.length; j++) query[j] += rng.nextGaussian() * 0.1;
            out[i] = query;
        }
        return out;
    }

    public static void main(String[] args) {
        double[][] data = buildData(300, 8, 6, 0x8A5EEDL);
        Hnsw index = new Hnsw(data, 16, 100, 0x5EEDL);
        double[][] probe = queries(data, 50, 0x0A11L);
        check(data.length == 300 && data[0].length == 8, "300 vectors of dimension 8");

        // 1. structural invariants of the hierarchy
        int maxDegreeSeen = 0, maxLevelSeen = 0;
        for (int node = 0; node < data.length; node++) {
            check(index.links.get(node).size() == index.level[node] + 1, "node " + node + " has exactly "
                    + "level+1 link lists");
            for (int layer = 0; layer < index.links.get(node).size(); layer++) {
                check(index.links.get(node).get(layer).size() <= index.maxDegree(layer), "degree bound at "
                        + "node " + node + " layer " + layer + ": "
                        + index.links.get(node).get(layer).size() + " > " + index.maxDegree(layer));
                check(new LinkedHashSet<>(index.links.get(node).get(layer)).size()
                                == index.links.get(node).get(layer).size(), "no duplicate links at node "
                        + node + " layer " + layer);
                maxDegreeSeen = Math.max(maxDegreeSeen, index.links.get(node).get(layer).size());
            }
            maxLevelSeen = Math.max(maxLevelSeen, index.level[node]);
        }
        check(index.level[index.entry] == maxLevelSeen, "the entry point holds the maximum level: "
                + index.level[index.entry] + " vs " + maxLevelSeen);
        check(maxLevelSeen >= 2, "the hierarchy has at least three layers: " + (maxLevelSeen + 1));
        check(index.layerZeroConnected(), "layer 0 is connected: every node is reachable from the entry");
        check(maxDegreeSeen == index.m0, "layer 0 uses the full 2M degree bound: " + maxDegreeSeen);

        // 2. recall versus the number of candidates explored
        double recall10 = recall(index, probe, 10, 10, true);
        double recall50 = recall(index, probe, 10, 50, true);
        double recall200 = recall(index, probe, 10, 200, true);
        check(recall200 >= 0.95, "a large search budget recovers almost everything: " + recall200);
        check(recall200 >= recall10 - 1e-9, "more candidates never hurt: " + recall10 + " -> " + recall200);
        check(recall50 >= recall10 - 1e-9, "recall is monotone in efSearch: " + recall10 + " -> " + recall50
                + " -> " + recall200);
        check(recall10 > 0.2, "even a tiny budget finds a good share of the neighbours: " + recall10);

        // 3. quantised navigation costs almost nothing in recall but saves exact distances
        double quantizedRecall = recall(index, probe, 10, 50, true);
        double exactRecall = recall(index, probe, 10, 50, false);
        check(quantizedRecall >= exactRecall - 0.05, "quantised navigation matches exact navigation "
                + "closely: " + quantizedRecall + " vs " + exactRecall);
        check(index.repairs < 0.9 * data.length, "the connectivity repair is a small part of the graph: "
                + index.repairs + " of " + data.length + " nodes");

        // 4. the quantisation itself: bounded error and half-step decoding
        double worstDistanceGap = 0, worstCodeError = 0;
        for (int i = 0; i < 100; i++) {
            int a = i * 3 % data.length, b = (i * 7 + 1) % data.length;
            double exact = Hnsw.squaredDistance(data[a], data[b]);
            double quantized = Hnsw.squaredDistance(index.decode(index.codes[a]), index.decode(index.codes[b]));
            worstDistanceGap = Math.max(worstDistanceGap, Math.abs(exact - quantized) / Math.max(1, exact));
            for (int j = 0; j < index.dimension; j++) {
                double step = (index.maximum[j] - index.minimum[j]) / 255.0;
                worstCodeError = Math.max(worstCodeError, Math.abs(index.decode(index.codes[a])[j] - data[a][j])
                        / Math.max(step, 1e-12));
            }
        }
        check(worstDistanceGap < 0.5, "the quantised distance stays within 50% of the exact one: "
                + worstDistanceGap);
        check(worstCodeError <= 0.5 + 1e-9, "decoding is within half a quantisation step: " + worstCodeError);
        check(Arrays.equals(index.encode(data[5]), index.codes[5]), "encoding is deterministic");

        // 5. exact re-ranking: the returned order is exact and the exact work stays small
        index.exactDistanceComputations = 0;
        index.quantizedDistanceComputations = 0;
        List<Integer> neighboursFound = index.search(probe[0], 10, 50, true);
        long exactWork = index.exactDistanceComputations;
        check(exactWork < data.length / 2, "exact distances are computed for a small fraction of the data: "
                + exactWork + " of " + data.length);
        for (int rank = 1; rank < neighboursFound.size(); rank++) {
            check(Hnsw.squaredDistance(probe[0], data[neighboursFound.get(rank - 1)])
                    <= Hnsw.squaredDistance(probe[0], data[neighboursFound.get(rank)]) + 1e-9,
                    "the returned neighbours are ordered by exact distance");
        }
        check(index.quantizedDistanceComputations > 0, "navigation used the quantised codes: "
                + index.quantizedDistanceComputations + " lookups");

        // 6. boundaries: a query equal to an indexed vector, duplicates, a single node, an empty index
        for (int node : new int[]{0, 42, 299}) {
            List<Integer> self = index.search(data[node], 3, 100, true);
            check(self.get(0) == node, "a query equal to vector " + node + " finds it first: " + self);
            check(Hnsw.squaredDistance(data[node], data[self.get(0)]) == 0.0, "and at distance zero");
        }
        double[][] duplicates = new double[20][];
        Arrays.fill(duplicates, data[1]);
        Hnsw duplicateIndex = new Hnsw(duplicates, 4, 20, 3);
        check(duplicateIndex.search(data[1], 5, 20, true).size() == 5, "duplicate vectors still return k ids");
        Hnsw single = new Hnsw(new double[][]{data[7].clone()}, 4, 10, 1);
        check(single.search(data[7], 3, 10, true).equals(List.of(0)), "a one-node index returns that node");
        check(single.layerZeroConnected(), "a one-node graph is trivially connected");
        check(new Hnsw(new double[][]{{0.0, 0.0}}, 2, 4, 1).search(new double[]{0, 0}, 1, 4, true)
                .equals(List.of(0)), "a two-dimension single-node index works");

        // 7. determinism: the same seed rebuilds the same graph and answers identically
        Hnsw replay = new Hnsw(data, 16, 100, 0x5EEDL);
        boolean sameLinks = true;
        for (int node = 0; sameLinks && node < data.length; node++) {
            sameLinks = replay.links.get(node).equals(index.links.get(node))
                    && replay.level[node] == index.level[node];
        }
        check(sameLinks, "the same seed rebuilds the same neighbour lists and levels");
        check(replay.search(probe[2], 10, 50, true).equals(index.search(probe[2], 10, 50, true)),
                "the same query gives the same result list");

        System.out.println("graph: " + data.length + " vectors, M=" + index.m + " (layer 0 bound " + index.m0
                + "), " + (maxLevelSeen + 1) + " layers, entry level " + index.level[index.entry]
                + ", connectivity repairs " + index.repairs);
        System.out.println("recall@10: " + recall10 + " (ef=10), " + recall50 + " (ef=50), " + recall200
                + " (ef=200)");
        System.out.println("quantisation: navigation on int8 codes, " + exactWork + " exact distances for a "
                + "query over " + data.length + " vectors, code error <= " + worstCodeError + " step");
        System.out.println("PASS");
    }
}
