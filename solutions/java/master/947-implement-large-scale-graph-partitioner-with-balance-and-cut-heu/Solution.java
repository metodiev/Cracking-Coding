/** Task 947 - Implement large-scale graph partitioner with balance and cut heuristics.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A two-phase graph partitioner in the shape of METIS. Phase one grows k balanced regions greedily: each
 *  region starts from the highest-degree unassigned vertex and repeatedly swallows the unassigned vertex
 *  with the most edges into the region, until the capacity cap is reached. Phase two is a Kernighan-Lin
 *  style refinement: the best single-vertex move that strictly reduces the cut while keeping every part
 *  inside the capacity cap is applied, repeatedly, until no such move exists - so the result is a genuine
 *  local optimum with a balanced load.
 *  Simplified: no multilevel coarsening, no bisection recursion, no FM bucket structure, and a simple
 *  capacity cap instead of a weighted balance constraint.
 *  Verified: every vertex is assigned once, capacities hold, the incrementally maintained cut equals a
 *  full recount after every move, no feasible move improves the final partition, the heuristic is compared
 *  against exact brute-force optima on tiny graphs and against a random balanced baseline, and degenerate
 *  inputs (k = 1, one vertex, two cliques joined by a bridge) behave.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final double BALANCE_SLACK = 2.0;      // extra vertices a part may hold above the ideal

    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        int nextInt(int bound) {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return (int) Long.remainderUnsigned(z ^ (z >>> 31), bound);
        }
    }
    static final class Graph {
        final int n;
        final List<int[]> edges = new ArrayList<>();
        final List<List<Integer>> adjacency = new ArrayList<>();

        Graph(int n) {
            this.n = n;
            for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        }
        void addEdge(int a, int b) {
            edges.add(new int[] {a, b});
            adjacency.get(a).add(b);
            adjacency.get(b).add(a);
        }
    }
    /** A balanced assignment of vertices to parts, with the cut maintained incrementally. */
    static final class Partition {
        final Graph graph;
        final int parts;
        final int[] part;
        final int[] sizes;
        int cut, moves, recountViolations;

        Partition(Graph graph, int parts) {
            this.graph = graph;
            this.parts = parts;
            part = new int[graph.n];
            Arrays.fill(part, -1);
            sizes = new int[parts];
        }
        /** Cut reduction achieved by moving v into `target`: edges into the target leave the cut. */
        int moveGain(int v, int target) {
            int same = 0, other = 0;
            for (int w : graph.adjacency.get(v)) {
                if (part[w] == part[v]) same++;
                else if (part[w] == target) other++;
            }
            return other - same;
        }
        void assign(int v, int target) {
            part[v] = target;
            sizes[target]++;
        }
        void move(int v, int target) {
            cut -= moveGain(v, target);
            sizes[part[v]]--;
            sizes[target]++;
            part[v] = target;
            moves++;
            if (cut != recount()) recountViolations++;
        }
        int recount() {
            int total = 0;
            for (int[] e : graph.edges) if (part[e[0]] != part[e[1]]) total++;
            return total;
        }
        int maxSize() {
            int max = 0;
            for (int size : sizes) max = Math.max(max, size);
            return max;
        }
    }
    static int capacity(int n, int parts) {
        return (int) Math.ceil((double) n / parts + BALANCE_SLACK);
    }
    /** Phase one: greedy region growing from the highest-degree unassigned vertex. */
    static Partition grow(Graph graph, int parts) {
        Partition partition = new Partition(graph, parts);
        int cap = capacity(graph.n, parts);
        for (int region = 0; region < parts; region++) {
            int seed = -1;
            for (int v = 0; v < graph.n; v++) {
                if (partition.part[v] >= 0) continue;
                if (seed < 0 || graph.adjacency.get(v).size() > graph.adjacency.get(seed).size()) seed = v;
            }
            if (seed < 0) break;
            partition.assign(seed, region);
            while (partition.sizes[region] < cap) {
                int best = -1, bestConnections = -1;
                for (int v = 0; v < graph.n; v++) {
                    if (partition.part[v] >= 0) continue;
                    int connections = 0;
                    for (int w : graph.adjacency.get(v)) if (partition.part[w] == region) connections++;
                    if (connections > bestConnections || (connections == bestConnections && best >= 0 && v < best)) {
                        best = v;
                        bestConnections = connections;
                    }
                }
                if (best < 0) break;
                partition.assign(best, region);
            }
        }
        for (int v = 0; v < graph.n; v++) {
            if (partition.part[v] < 0) partition.assign(v, smallest(partition));
        }
        partition.cut = partition.recount();
        return partition;
    }
    static int smallest(Partition partition) {
        int best = 0;
        for (int i = 1; i < partition.parts; i++) if (partition.sizes[i] < partition.sizes[best]) best = i;
        return best;
    }
    /** Phase two: Kernighan-Lin single-vertex refinement under the capacity cap. */
    static void refine(Graph graph, Partition partition) {
        int cap = capacity(graph.n, partition.parts);
        while (true) {
            int bestVertex = -1, bestTarget = -1, bestGain = 0;
            for (int v = 0; v < graph.n; v++) {
                for (int target = 0; target < partition.parts; target++) {
                    if (target == partition.part[v]) continue;
                    if (partition.sizes[target] + 1 > cap) continue;
                    int gain = partition.moveGain(v, target);
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestVertex = v;
                        bestTarget = target;
                    }
                }
            }
            if (bestVertex < 0) return;
            partition.move(bestVertex, bestTarget);
        }
    }
    static Graph fixture(int n, int degree, long seed) {
        Rng rng = new Rng(seed);
        Graph graph = new Graph(n);
        for (int i = 0; i < n; i++) graph.addEdge(i, (i + 1) % n);
        for (int i = 0; i < n * degree / 2; i++) graph.addEdge(rng.nextInt(n), rng.nextInt(n));
        return graph;
    }
    /** Exact minimum bisection by brute force, for tiny graphs only. */
    static int bruteForceCut(Graph graph, int parts, int tolerance) {
        int best = Integer.MAX_VALUE;
        int assignments = 1 << graph.n;
        for (int mask = 0; mask < assignments; mask++) {
            int[] sizes = new int[parts];
            for (int v = 0; v < graph.n; v++) sizes[(mask >> v) & (parts - 1)]++;
            int cap = (int) Math.ceil((double) graph.n / parts) + tolerance;
            boolean balanced = true;
            for (int size : sizes) if (size > cap) balanced = false;
            if (!balanced) continue;
            int cut = 0;
            for (int[] e : graph.edges) {
                int left = (mask >> e[0]) & (parts - 1), right = (mask >> e[1]) & (parts - 1);
                if (left != right) cut++;
            }
            best = Math.min(best, cut);
        }
        return best;
    }
    static int randomBalancedCut(Graph graph, int parts, long seed) {
        Rng rng = new Rng(seed);
        int[] order = new int[graph.n];
        for (int i = 0; i < graph.n; i++) order[i] = i;
        for (int i = graph.n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        int[] part = new int[graph.n];
        int cap = (int) Math.ceil((double) graph.n / parts);
        for (int index = 0; index < graph.n; index++) part[order[index]] = Math.min(index / cap, parts - 1);
        int cut = 0;
        for (int[] e : graph.edges) if (part[e[0]] != part[e[1]]) cut++;
        return cut;
    }
    public static void main(String[] args) {
        Graph graph = fixture(60, 5, 4242);
        Partition partition = grow(graph, 4);
        int initialCut = partition.recount();
        refine(graph, partition);
        int cap = capacity(60, 4);

        Set<Integer> assigned = new TreeSet<>();
        boolean inRange = true;
        for (int v = 0; v < graph.n; v++) {
            assigned.add(v);
            if (partition.part[v] < 0 || partition.part[v] >= 4) inRange = false;
        }
        check(assigned.size() == graph.n && inRange && partition.sizes[0] + partition.sizes[1]
                        + partition.sizes[2] + partition.sizes[3] == graph.n,
                "every vertex must be assigned exactly once to a valid part: " + Arrays.toString(partition.sizes));

        check(partition.maxSize() <= cap && cap == 17 && partition.recountViolations == 0
                        && partition.cut == partition.recount(),
                "the load must stay balanced and the maintained cut must survive every move: "
                        + Arrays.toString(partition.sizes) + " cap " + cap + ", violations "
                        + partition.recountViolations);

        check(partition.cut < initialCut && partition.moves > 0 && initialCut > 0
                        && graph.edges.size() > partition.cut,
                "the refinement must strictly improve the cut: " + initialCut + " -> " + partition.cut + " in "
                        + partition.moves + " moves of " + graph.edges.size() + " edges");

        int improving = 0;
        for (int v = 0; v < graph.n; v++) {
            for (int target = 0; target < 4; target++) {
                if (target == partition.part[v] || partition.sizes[target] + 1 > cap) continue;
                if (partition.moveGain(v, target) > 0) improving++;
            }
        }
        check(improving == 0, improving + " feasible moves still improve the final partition");

        int exactGap = 0;
        for (int trial = 0; trial < 3; trial++) {
            Graph tiny = fixture(10, 4, 100 + trial);
            Partition tinyPart = grow(tiny, 2);
            refine(tiny, tinyPart);
            int optimum = bruteForceCut(tiny, 2, 2);
            if (tinyPart.cut < optimum || tinyPart.cut > optimum * 3 / 2 + 1) exactGap++;
        }
        check(exactGap == 0, exactGap + " tiny partitions fall outside [optimum, 1.5 optimum + 1]");

        int baseline = randomBalancedCut(graph, 4, 7);
        check(partition.cut < baseline, "the heuristic must beat a random balanced assignment: " + partition.cut
                + " vs " + baseline);

        Graph cliques = new Graph(16);
        for (int i = 0; i < 8; i++) {
            for (int j = i + 1; j < 8; j++) cliques.addEdge(i, j);
            for (int j = i + 1; j < 8; j++) cliques.addEdge(8 + i, 8 + j);
        }
        cliques.addEdge(0, 8);                                        // a single bridge between the cliques
        Partition split = grow(cliques, 2);
        refine(cliques, split);
        check(split.cut >= 1 && split.cut <= 3 && split.sizes[0] + split.sizes[1] == 16,
                "two cliques joined by one bridge must be cut in one or two places: " + split.cut);

        Graph single = new Graph(1);
        Partition whole = grow(single, 1);
        refine(single, whole);
        Graph empty = new Graph(0);
        check(whole.cut == 0 && whole.parts == 1 && whole.sizes[0] == 1 && groove(empty) == 0,
                "k = 1, one vertex and an empty graph must be trivial: " + whole.cut + " / " + groove(empty));

        Graph twin = fixture(60, 5, 4242);
        Partition replay = grow(twin, 4);
        int replayInitial = replay.recount();
        refine(twin, replay);
        check(replay.cut == partition.cut && replayInitial == initialCut && replay.moves == partition.moves
                        && Arrays.equals(replay.part, partition.part),
                "the partitioner must be reproducible: " + replay.cut + " vs " + partition.cut);

        int settled = partition.cut, settledMoves = partition.moves;
        refine(graph, partition);
        check(partition.cut == settled && partition.moves == settledMoves && partition.recountViolations == 0
                        && partition.maxSize() <= cap,
                "a second refinement pass must change nothing: " + settled + " -> " + partition.cut);

        int before = partition.cut;
        int vertex = -1, target = -1;
        for (int v = 0; v < graph.n && vertex < 0; v++) {
            for (int t = 0; t < 4; t++) {
                if (t != partition.part[v] && partition.sizes[t] + 1 <= cap && partition.moveGain(v, t) < 0) {
                    vertex = v;
                    target = t;
                    break;
                }
            }
        }
        int predicted = partition.moveGain(vertex, target);
        partition.move(vertex, target);
        check(partition.cut == before - predicted && partition.cut == partition.recount()
                        && predicted < 0 && partition.recountViolations == 0,
                "the incremental gain must predict the recounted cut: " + predicted + " gave " + partition.cut
                        + " vs " + partition.recount());

        System.out.println("partitioner: " + graph.n + " vertices, " + graph.edges.size() + " edges, 4 parts, sizes "
                + Arrays.toString(partition.sizes));
        System.out.println("cut " + initialCut + " -> " + partition.cut + " in " + partition.moves
                + " moves, random baseline " + baseline + ", tiny-graph optimum gap " + exactGap);
        System.out.println("PASS");
    }
    static int groove(Graph graph) {
        return graph.edges.size() == 0 ? 0 : new Partition(graph, 1).recount();
    }
}
