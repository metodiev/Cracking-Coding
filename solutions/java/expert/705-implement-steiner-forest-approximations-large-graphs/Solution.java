/** Task 705 - Implement Steiner forest approximations for large graphs.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * A Steiner forest is the cheapest set of edges that connects every demand pair (a Steiner tree is the
 * special case of one demand). The implementation is the shortest-distance (Takahashi-Matsuyama
 * style) heuristic in forest form: while some demand is still unsatisfied, compute for *every*
 * unsatisfied demand the cheapest way to join its two components and buy the globally cheapest of
 * those paths. The distance used is not the plain graph distance but the *contracted* distance: edges
 * whose endpoints already lie in the same component are free, because that component is already
 * connected by bought edges. That single trick is what makes the heuristic share purchased structure
 * between demands instead of paying for a disjoint path per demand. A Dijkstra run from the whole
 * component of u with per-edge cost 0 inside components yields exactly that path.
 *
 * After buying, a reverse-delete pass walks the bought edges in decreasing cost order and drops an edge
 * whenever the remaining set still connects every demand, so the result is inclusion minimal.
 *
 * Simplified: moat growing (the Agrawal-Klein-Ravi primal-dual 2-approximation) is *not* implemented,
 * so no proven approximation ratio is claimed for the forest case; instead the check battery measures
 * the gap against the exact optimum on many small instances and bounds it empirically. What is really
 * verified is what the algorithm itself guarantees: every solution is feasible (an independent DSU
 * confirms each demand), it is inclusion minimal (removing any bought edge breaks feasibility), it can
 * never be cheaper than the true optimum, and for graphs small enough to enumerate all 2^m edge
 * subsets the exact optimum is computed by brute force and compared. Infeasible instances (a demand
 * split across two graph components) are detected and reported, and invalid input is rejected.
 */
import java.util.*;

class WeightedGraph {
    final int n;
    private final List<int[]> edges = new ArrayList<>();
    private final List<List<int[]>> adjacency = new ArrayList<>();

    WeightedGraph(int n) {
        if (n < 1) throw new IllegalArgumentException("need at least one vertex: " + n);
        this.n = n;
        for (int i = 0; i <= n; i++) adjacency.add(new ArrayList<>());
    }

    int edgeCount() {
        return edges.size();
    }

    int[] edge(int id) {
        return edges.get(id).clone();
    }

    int cost(int id) {
        return edges.get(id)[2];
    }

    List<int[]> neighbours(int vertex) {
        return adjacency.get(vertex);
    }

    void addEdge(int u, int v, int cost) {
        if (u < 1 || u > n || v < 1 || v > n) throw new IllegalArgumentException("vertex out of range");
        if (u == v) throw new IllegalArgumentException("self loops cannot help a Steiner forest");
        if (cost <= 0) throw new IllegalArgumentException("costs must be positive: " + cost);
        int id = edges.size();
        edges.add(new int[]{u, v, cost});
        adjacency.get(u).add(new int[]{v, cost, id});
        adjacency.get(v).add(new int[]{u, cost, id});
    }
}

class SteinerForestSolver {
    static final class Result {
        final boolean feasible;
        final List<Integer> edgeIds;
        final int cost;
        final int prunedEdges;

        Result(boolean feasible, List<Integer> edgeIds, int cost, int prunedEdges) {
            this.feasible = feasible;
            this.edgeIds = Collections.unmodifiableList(edgeIds);
            this.cost = cost;
            this.prunedEdges = prunedEdges;
        }
    }

    private final WeightedGraph graph;
    private int[] parent;
    private int[] size;
    private long dijkstraRuns;

    SteinerForestSolver(WeightedGraph graph) {
        this.graph = graph;
    }

    long dijkstraRuns() {
        return dijkstraRuns;
    }

    Result solve(List<int[]> demands) {
        for (int[] demand : demands) {
            if (demand.length != 2 || demand[0] < 1 || demand[0] > graph.n
                    || demand[1] < 1 || demand[1] > graph.n) {
                throw new IllegalArgumentException("bad demand: " + Arrays.toString(demand));
            }
        }
        reset();
        List<Integer> bought = new ArrayList<>();
        while (true) {
            int chosenDemand = -1;
            long chosenCost = Long.MAX_VALUE;
            List<Integer> chosenPath = null;
            for (int index = 0; index < demands.size(); index++) {
                int u = demands.get(index)[0];
                int v = demands.get(index)[1];
                if (find(u) == find(v)) continue;
                Path path = shortestConnection(u, v);
                if (path == null) return new Result(false, new ArrayList<>(), -1, 0);
                if (path.cost < chosenCost) {
                    chosenCost = path.cost;
                    chosenDemand = index;
                    chosenPath = path.edgeIds;
                }
            }
            if (chosenDemand < 0) break;
            for (int edgeId : chosenPath) {
                if (!bought.contains(edgeId)) bought.add(edgeId);
                int[] edge = graph.edge(edgeId);
                union(edge[0], edge[1]);
            }
        }

        List<Integer> ordered = new ArrayList<>(bought);
        ordered.sort((left, right) -> Integer.compare(graph.cost(right), graph.cost(left)));
        List<Integer> kept = new ArrayList<>(bought);
        int pruned = 0;
        for (int edgeId : ordered) {
            kept.remove(Integer.valueOf(edgeId));
            if (connectsAll(kept, demands)) {
                pruned++;
            } else {
                kept.add(edgeId);
            }
        }
        int cost = totalCost(kept);
        return new Result(true, kept, cost, pruned);
    }

    private static final class Path {
        final long cost;
        final List<Integer> edgeIds;

        Path(long cost, List<Integer> edgeIds) {
            this.cost = cost;
            this.edgeIds = edgeIds;
        }
    }

    /**
     * Multi-source Dijkstra from every vertex of u's component to the first settled vertex of v's
     * component, with edges inside a component costing 0 because they are already bought.
     */
    private Path shortestConnection(int u, int v) {
        dijkstraRuns++;
        long[] distance = new long[graph.n + 1];
        int[] previousVertex = new int[graph.n + 1];
        int[] previousEdge = new int[graph.n + 1];
        Arrays.fill(distance, Long.MAX_VALUE);
        Arrays.fill(previousVertex, -1);
        Arrays.fill(previousEdge, -1);
        PriorityQueue<long[]> queue = new PriorityQueue<>(Comparator.comparingLong(entry -> entry[0]));
        int uRoot = find(u);
        int vRoot = find(v);
        for (int vertex = 1; vertex <= graph.n; vertex++) {
            if (find(vertex) == uRoot) {
                distance[vertex] = 0;
                queue.add(new long[]{0, vertex});
            }
        }
        int target = -1;
        while (!queue.isEmpty()) {
            long[] entry = queue.poll();
            int vertex = (int) entry[1];
            if (entry[0] > distance[vertex]) continue;
            if (find(vertex) == vRoot) {
                target = vertex;
                break;
            }
            for (int[] arc : graph.neighbours(vertex)) {
                int other = arc[0];
                long step = find(vertex) == find(other) ? 0 : arc[1];
                long candidate = distance[vertex] + step;
                if (candidate < distance[other]) {
                    distance[other] = candidate;
                    previousVertex[other] = vertex;
                    previousEdge[other] = arc[2];
                    queue.add(new long[]{candidate, other});
                }
            }
        }
        if (target < 0) return null;
        List<Integer> path = new ArrayList<>();
        int cursor = target;
        while (previousEdge[cursor] >= 0) {
            path.add(previousEdge[cursor]);
            cursor = previousVertex[cursor];
        }
        Collections.reverse(path);
        return new Path(distance[target], path);
    }

    private boolean connectsAll(List<Integer> edgeIds, List<int[]> demands) {
        int[] savedParent = parent;
        int[] savedSize = size;
        reset();
        for (int edgeId : edgeIds) {
            int[] edge = graph.edge(edgeId);
            union(edge[0], edge[1]);
        }
        boolean ok = true;
        for (int[] demand : demands) {
            if (find(demand[0]) != find(demand[1])) {
                ok = false;
                break;
            }
        }
        parent = savedParent;
        size = savedSize;
        return ok;
    }

    /** Independent feasibility test used by the checks. */
    static boolean connectsAll(WeightedGraph graph, List<Integer> edgeIds, List<int[]> demands) {
        SteinerForestSolver helper = new SteinerForestSolver(graph);
        helper.reset();
        for (int edgeId : edgeIds) {
            int[] edge = graph.edge(edgeId);
            helper.union(edge[0], edge[1]);
        }
        for (int[] demand : demands) {
            if (helper.find(demand[0]) != helper.find(demand[1])) return false;
        }
        return true;
    }

    /** Exact optimum by enumerating every edge subset; -1 when no subset can connect the demands. */
    static int bruteForceOptimum(WeightedGraph graph, List<int[]> demands) {
        int m = graph.edgeCount();
        if (m > 16) throw new IllegalStateException("brute force is capped at 16 edges, got " + m);
        int best = Integer.MAX_VALUE;
        for (int mask = 0; mask < (1 << m); mask++) {
            int cost = 0;
            for (int edgeId = 0; edgeId < m; edgeId++) {
                if ((mask & (1 << edgeId)) != 0) cost += graph.cost(edgeId);
            }
            if (cost >= best) continue;
            List<Integer> subset = new ArrayList<>();
            for (int edgeId = 0; edgeId < m; edgeId++) {
                if ((mask & (1 << edgeId)) != 0) subset.add(edgeId);
            }
            if (connectsAll(graph, subset, demands)) best = cost;
        }
        return best == Integer.MAX_VALUE ? -1 : best;
    }

    static int totalCost(WeightedGraph graph, List<Integer> edgeIds) {
        int cost = 0;
        for (int edgeId : edgeIds) cost += graph.cost(edgeId);
        return cost;
    }

    private int totalCost(List<Integer> edgeIds) {
        return totalCost(graph, edgeIds);
    }

    private void reset() {
        parent = new int[graph.n + 1];
        size = new int[graph.n + 1];
        for (int i = 1; i <= graph.n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    private int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private void union(int a, int b) {
        int rootA = find(a);
        int rootB = find(b);
        if (rootA == rootB) return;
        if (size[rootA] < size[rootB]) {
            int swap = rootA;
            rootA = rootB;
            rootB = swap;
        }
        parent[rootB] = rootA;
        size[rootA] += size[rootB];
    }
}

public class Solution {
    private static long state = 0x5EED5EED5EEDL;

    /** splitmix64 - deterministic PRNG, no Math.random. */
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

    /** True when no single bought edge can be dropped without breaking some demand. */
    static boolean isInclusionMinimal(WeightedGraph graph, List<Integer> solution, List<int[]> demands) {
        for (int edgeId : solution) {
            List<Integer> without = new ArrayList<>(solution);
            without.remove(Integer.valueOf(edgeId));
            if (SteinerForestSolver.connectsAll(graph, without, demands)) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        // --- a single demand is a shortest path -------------------------------
        WeightedGraph line = new WeightedGraph(3);
        line.addEdge(1, 2, 4);
        line.addEdge(2, 3, 5);
        line.addEdge(1, 3, 20);
        List<int[]> lineDemand = List.of(new int[]{1, 3});
        SteinerForestSolver.Result lineResult = new SteinerForestSolver(line).solve(lineDemand);
        check(lineResult.feasible, "the line instance is feasible");
        check(lineResult.cost == 9, "cheapest connection of 1-3 costs 9, got " + lineResult.cost);
        check(lineResult.edgeIds.size() == 2, "two edges are needed");
        check(SteinerForestSolver.connectsAll(line, lineResult.edgeIds, lineDemand), "line is connected");

        // --- shared structure is bought once ----------------------------------
        WeightedGraph star = new WeightedGraph(4);
        star.addEdge(1, 2, 1);
        star.addEdge(1, 3, 1);
        star.addEdge(1, 4, 1);
        List<int[]> starDemands = List.of(new int[]{2, 3}, new int[]{3, 4});
        SteinerForestSolver.Result starResult = new SteinerForestSolver(star).solve(starDemands);
        check(starResult.feasible, "star instance feasible");
        check(starResult.cost == 3, "the centre edge is shared: expected 3, got " + starResult.cost);
        check(starResult.edgeIds.size() == 3, "three edges form the forest");
        check(SteinerForestSolver.connectsAll(star, starResult.edgeIds, starDemands), "star connected");

        WeightedGraph shared = new WeightedGraph(5);
        shared.addEdge(1, 2, 3);
        shared.addEdge(2, 3, 1);
        shared.addEdge(2, 4, 1);
        shared.addEdge(4, 5, 1);
        List<int[]> sharedDemands = List.of(new int[]{1, 3}, new int[]{3, 5});
        SteinerForestSolver.Result sharedResult = new SteinerForestSolver(shared).solve(sharedDemands);
        check(sharedResult.feasible, "shared-path instance feasible");
        check(SteinerForestSolver.connectsAll(shared, sharedResult.edgeIds, sharedDemands),
                "shared-path forest connected");
        check(sharedResult.cost == SteinerForestSolver.bruteForceOptimum(shared, sharedDemands),
                "greedy finds the optimum here: " + sharedResult.cost);
        check(isInclusionMinimal(shared, sharedResult.edgeIds, sharedDemands), "shared-path minimality");

        // --- degenerate demands -------------------------------------------------
        WeightedGraph loop = new WeightedGraph(3);
        loop.addEdge(1, 2, 7);
        loop.addEdge(2, 3, 7);
        SteinerForestSolver.Result empty = new SteinerForestSolver(loop).solve(new ArrayList<>());
        check(empty.feasible && empty.cost == 0 && empty.edgeIds.isEmpty(),
                "no demands cost nothing");
        List<int[]> selfDemands = List.of(new int[]{2, 2});
        SteinerForestSolver.Result selfResult = new SteinerForestSolver(loop).solve(selfDemands);
        check(selfResult.feasible && selfResult.cost == 0 && selfResult.edgeIds.isEmpty(),
                "a demand of a vertex with itself needs no edge");
        SteinerForestSolver.Result single = new SteinerForestSolver(new WeightedGraph(2)).solve(
                List.of(new int[]{2, 1}));
        check(!single.feasible, "a graph with no edges cannot satisfy a demand");

        // --- infeasible instance and rejected input ---------------------------
        WeightedGraph split = new WeightedGraph(4);
        split.addEdge(1, 2, 1);
        split.addEdge(3, 4, 1);
        SteinerForestSolver.Result infeasible = new SteinerForestSolver(split).solve(
                List.of(new int[]{1, 3}));
        check(!infeasible.feasible, "demands across two components are infeasible");
        check(SteinerForestSolver.bruteForceOptimum(split, List.of(new int[]{1, 3})) == -1,
                "brute force agrees the instance is infeasible");
        check(rejects(() -> new WeightedGraph(3).addEdge(1, 1, 1)), "self loop rejected");
        check(rejects(() -> new WeightedGraph(3).addEdge(1, 4, 1)), "vertex out of range rejected");
        check(rejects(() -> new WeightedGraph(3).addEdge(1, 2, 0)), "non-positive cost rejected");
        check(rejects(() -> new SteinerForestSolver(loop).solve(List.of(new int[]{0, 1}))),
                "demand with a bad vertex rejected");

        // --- random instances against the exact optimum -----------------------
        double worstRatio = 0.0;
        int feasibleInstances = 0;
        for (int trial = 0; trial < 40; trial++) {
            int vertices = 4 + nextInt(4);
            int edgeBudget = 5 + nextInt(8);
            WeightedGraph graph = new WeightedGraph(vertices);
            for (int edge = 0; edge < edgeBudget; edge++) {
                int u = 1 + nextInt(vertices);
                int v = 1 + nextInt(vertices);
                if (u == v) continue;
                graph.addEdge(u, v, 1 + nextInt(9));
            }
            int demandCount = 1 + nextInt(3);
            List<int[]> demands = new ArrayList<>();
            for (int d = 0; d < demandCount; d++) {
                demands.add(new int[]{1 + nextInt(vertices), 1 + nextInt(vertices)});
            }
            SteinerForestSolver.Result result = new SteinerForestSolver(graph).solve(demands);
            int optimum = SteinerForestSolver.bruteForceOptimum(graph, demands);
            check(result.feasible == (optimum >= 0),
                    "trial " + trial + " feasibility disagrees with brute force");
            if (!result.feasible) continue;
            feasibleInstances++;
            check(SteinerForestSolver.connectsAll(graph, result.edgeIds, demands),
                    "trial " + trial + " solution must connect every demand");
            check(isInclusionMinimal(graph, result.edgeIds, demands),
                    "trial " + trial + " solution must be inclusion minimal");
            check(result.cost == SteinerForestSolver.totalCost(graph, result.edgeIds),
                    "trial " + trial + " reported cost matches the edge list");
            check(new HashSet<>(result.edgeIds).size() == result.edgeIds.size(),
                    "trial " + trial + " duplicate edge in the solution");
            check(result.cost >= optimum, "trial " + trial + " beat the optimum: "
                    + result.cost + " < " + optimum);
            worstRatio = Math.max(worstRatio, optimum == 0 ? 1.0 : result.cost / (double) optimum);
        }
        check(feasibleInstances >= 15, "most random instances are feasible: " + feasibleInstances);
        check(worstRatio <= 2.0, "empirical gap must stay within a factor of two: " + worstRatio);

        // --- larger graph: the solver does not fall over -----------------------
        WeightedGraph grid = new WeightedGraph(24);
        for (int row = 0; row < 6; row++) {
            for (int column = 0; column < 4; column++) {
                int vertex = row * 4 + column + 1;
                if (column + 1 < 4) grid.addEdge(vertex, vertex + 1, 1 + nextInt(5));
                if (row + 1 < 6) grid.addEdge(vertex, vertex + 4, 1 + nextInt(5));
            }
        }
        List<int[]> gridDemands = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            gridDemands.add(new int[]{1 + nextInt(6) * 4, 1 + nextInt(6) * 4});
        }
        long before = System.nanoTime();
        SteinerForestSolver.Result gridResult = new SteinerForestSolver(grid).solve(gridDemands);
        long elapsedMillis = (System.nanoTime() - before) / 1_000_000;
        check(gridResult.feasible, "the grid instance is feasible");
        check(SteinerForestSolver.connectsAll(grid, gridResult.edgeIds, gridDemands),
                "grid forest connects every demand");
        check(isInclusionMinimal(grid, gridResult.edgeIds, gridDemands), "grid forest is minimal");
        check(gridResult.cost <= SteinerForestSolver.totalCost(grid, allEdges(grid)),
                "the forest cannot cost more than the whole graph");
        check(elapsedMillis < 2000, "solving the grid must be fast: " + elapsedMillis + " ms");

        // --- determinism --------------------------------------------------------
        SteinerForestSolver.Result again = new SteinerForestSolver(grid).solve(gridDemands);
        check(again.cost == gridResult.cost && again.edgeIds.equals(gridResult.edgeIds),
                "the same input must produce the same forest");

        System.out.println("PASS");
    }

    static List<Integer> allEdges(WeightedGraph graph) {
        List<Integer> edges = new ArrayList<>();
        for (int edgeId = 0; edgeId < graph.edgeCount(); edgeId++) edges.add(edgeId);
        return edges;
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
