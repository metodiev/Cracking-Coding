/** Task 707 - Implement multi-commodity flow approximate algorithm outline.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * Multi-commodity flow asks for the largest lambda such that every commodity i can ship lambda * d_i
 * units from s_i to t_i while every arc stays within its capacity - a linear program whose optimum is
 * NP-hard to reach by combinatorial means, so the classic approximation is Lagrangian: give every arc
 * a *length* and route each commodity along its shortest path instead of solving a joint flow problem.
 *
 * This is the Garg-Koenemann / Plotkin-Shmoys-Tardos scheme in its outline form implemented here:
 *   - lengths start at 1 / capacity, so a loaded arc becomes expensive quickly;
 *   - each of the T iterations routes the full demand d_i of every commodity along the current
 *     shortest path (Dijkstra over the lengths), accumulating load[e];
 *   - the lengths are then updated multiplicatively by the flow just routed,
 *     length[e] *= 1 + epsilon * delta[e] / capacity[e], and renormalised (sum capacity * length = 1, a
 *     scaling that cannot change any shortest path but keeps the values bounded);
 *   - after every iteration the accumulated flow is scaled down by rho = max_e load[e] / capacity[e] to
 *     a feasible point, giving the candidate lambda = t / rho, and the best candidate over all t wins.
 * The returned solution is uniform across commodities by construction: each commodity rotated t * d_i
 * units, so after scaling by 1/rho every commodity carries exactly lambda * d_i.
 *
 * Simplified (this is explicitly the outline): one shortest path per commodity per iteration (no
 * min-cost-flow re-routing), accumulated flow rather than a per-iteration convex combination, and no
 * binary search for lambda - the scaling step plays that role. What is *not* approximated is the
 * verification: for small instances the LP optimum is computed exactly by enumerating all vertices of
 * the path-based formulation (all simple paths per commodity, one variable per path plus lambda, one
 * capacity constraint per arc and one demand constraint per commodity, Gaussian elimination on every
 * subset of tight constraints of full size), and the approximate lambda is required to be feasible
 * (arc loads within capacities), to be at most that optimum, and to be close to it.
 */
import java.util.*;

class MultiCommodityFlow {
    static final double TOLERANCE = 1e-9;

    static final class Result {
        final double lambda;
        final double[] arcLoad;
        final double[] commodityFlow;

        Result(double lambda, double[] arcLoad, double[] commodityFlow) {
            this.lambda = lambda;
            this.arcLoad = arcLoad;
            this.commodityFlow = commodityFlow;
        }
    }

    private final int n;
    private final int[] tail;
    private final int[] head;
    private final double[] capacity;
    private final List<List<int[]>> outgoing;
    private int arcs;
    private int commodityCount;
    private final int[] source;
    private final int[] sink;
    private final double[] demand;

    MultiCommodityFlow(int nodeCount, int maxArcs, int maxCommodities) {
        if (nodeCount < 2) throw new IllegalArgumentException("need at least two nodes");
        n = nodeCount;
        tail = new int[maxArcs];
        head = new int[maxArcs];
        capacity = new double[maxArcs];
        outgoing = new ArrayList<>();
        for (int i = 0; i <= nodeCount; i++) outgoing.add(new ArrayList<>());
        source = new int[maxCommodities];
        sink = new int[maxCommodities];
        demand = new double[maxCommodities];
    }

    int nodeCount() {
        return n;
    }

    int arcCount() {
        return arcs;
    }

    int commodityCount() {
        return commodityCount;
    }

    int tailOf(int id) {
        return tail[id];
    }

    int headOf(int id) {
        return head[id];
    }

    double capacityOf(int id) {
        return capacity[id];
    }

    int sourceOf(int id) {
        return source[id];
    }

    int sinkOf(int id) {
        return sink[id];
    }

    double demandOf(int id) {
        return demand[id];
    }

    void addArc(int u, int v, double arcCapacity) {
        if (u < 1 || u > n || v < 1 || v > n) throw new IllegalArgumentException("node out of range");
        if (u == v) throw new IllegalArgumentException("self loops carry no commodity flow");
        if (!(arcCapacity > 0)) throw new IllegalArgumentException("capacity must be positive");
        tail[arcs] = u;
        head[arcs] = v;
        capacity[arcs] = arcCapacity;
        outgoing.get(u).add(new int[]{v, arcs});
        arcs++;
    }

    void addCommodity(int s, int t, double units) {
        if (s < 1 || s > n || t < 1 || t > n) throw new IllegalArgumentException("node out of range");
        if (s == t) throw new IllegalArgumentException("a commodity needs two distinct endpoints");
        if (!(units > 0)) throw new IllegalArgumentException("demand must be positive");
        source[commodityCount] = s;
        sink[commodityCount] = t;
        demand[commodityCount] = units;
        commodityCount++;
    }

    Result solve(int iterations, double epsilon) {
        if (iterations < 1) throw new IllegalArgumentException("need at least one iteration");
        if (!(epsilon > 0 && epsilon < 1)) throw new IllegalArgumentException("epsilon must be in (0,1)");
        double[] length = new double[arcs];
        double[] load = new double[arcs];
        double[] routed = new double[commodityCount];
        for (int e = 0; e < arcs; e++) length[e] = 1.0 / capacity[e];
        normalize(length);
        double bestLambda = 0;
        int bestIteration = 0;
        for (int iteration = 1; iteration <= iterations; iteration++) {
            double[] delta = new double[arcs];
            for (int commodity = 0; commodity < commodityCount; commodity++) {
                int[] path = shortestPath(commodity, length);
                if (path == null) {
                    return new Result(0, new double[arcs], new double[commodityCount]);
                }
                for (int arc : path) {
                    load[arc] += demand[commodity];
                    delta[arc] += demand[commodity];
                }
                routed[commodity] += demand[commodity];
            }
            double rho = 0;
            for (int e = 0; e < arcs; e++) rho = Math.max(rho, load[e] / capacity[e]);
            double candidate = rho == 0 ? 0 : iteration / rho;
            if (candidate > bestLambda) {
                bestLambda = candidate;
                bestIteration = iteration;
            }
            // Multiplicative update driven by *this* iteration's flow, then renormalise: dividing every
            // length by the same constant cannot change any shortest path but keeps the values bounded.
            for (int e = 0; e < arcs; e++) {
                length[e] *= 1.0 + epsilon * (delta[e] / capacity[e]);
            }
            normalize(length);
        }
        // Re-run the first bestIteration accumulates by rebuilding the loads for that prefix.
        double[] bestLoad = replay(bestIteration, epsilon);
        double rho = 0;
        for (int e = 0; e < arcs; e++) rho = Math.max(rho, bestLoad[e] / capacity[e]);
        double scale = rho == 0 ? 1.0 : 1.0 / rho;
        double[] scaledLoad = new double[arcs];
        for (int e = 0; e < arcs; e++) scaledLoad[e] = bestLoad[e] * scale;
        double[] commodityFlow = new double[commodityCount];
        for (int commodity = 0; commodity < commodityCount; commodity++) {
            commodityFlow[commodity] = routedUnits(bestIteration, commodity) * scale;
        }
        return new Result(bestLambda, scaledLoad, commodityFlow);
    }

    /** Recomputes the accumulated load of the first `iterations` steps (deterministic replay). */
    private double[] replay(int iterations, double epsilon) {
        double[] length = new double[arcs];
        double[] load = new double[arcs];
        for (int e = 0; e < arcs; e++) length[e] = 1.0 / capacity[e];
        normalize(length);
        for (int iteration = 0; iteration < iterations; iteration++) {
            double[] delta = new double[arcs];
            for (int commodity = 0; commodity < commodityCount; commodity++) {
                int[] path = shortestPath(commodity, length);
                for (int arc : path) {
                    load[arc] += demand[commodity];
                    delta[arc] += demand[commodity];
                }
            }
            for (int e = 0; e < arcs; e++) length[e] *= 1.0 + epsilon * (delta[e] / capacity[e]);
            normalize(length);
        }
        return load;
    }

    /** Scales the lengths so that sum(capacity * length) == 1; shortest paths are unaffected. */
    private void normalize(double[] length) {
        double total = 0;
        for (int e = 0; e < arcs; e++) total += capacity[e] * length[e];
        if (!(total > 0)) return;
        for (int e = 0; e < arcs; e++) length[e] /= total;
    }

    private double routedUnits(int iterations, int commodity) {
        return iterations * demand[commodity];
    }

    /** Dijkstra over the current lengths; returns the arc ids of the shortest s-t path. */
    private int[] shortestPath(int commodity, double[] length) {
        double[] distance = new double[n + 1];
        int[] previousArc = new int[n + 1];
        Arrays.fill(distance, Double.POSITIVE_INFINITY);
        Arrays.fill(previousArc, -1);
        distance[source[commodity]] = 0;
        PriorityQueue<double[]> queue = new PriorityQueue<>(Comparator.comparingDouble(entry -> entry[0]));
        queue.add(new double[]{0, source[commodity]});
        while (!queue.isEmpty()) {
            double[] entry = queue.poll();
            int node = (int) entry[1];
            if (entry[0] > distance[node] + TOLERANCE) continue;
            for (int[] arcInfo : outgoing.get(node)) {
                int other = arcInfo[0];
                int arc = arcInfo[1];
                double candidate = distance[node] + length[arc];
                if (candidate < distance[other] - TOLERANCE) {
                    distance[other] = candidate;
                    previousArc[other] = arc;
                    queue.add(new double[]{candidate, other});
                }
            }
        }
        if (Double.isInfinite(distance[sink[commodity]])) return null;
        List<Integer> path = new ArrayList<>();
        int cursor = sink[commodity];
        while (previousArc[cursor] >= 0) {
            path.add(previousArc[cursor]);
            cursor = tail[previousArc[cursor]];
        }
        int[] result = new int[path.size()];
        for (int i = 0; i < path.size(); i++) result[i] = path.get(path.size() - 1 - i);
        return result;
    }
}

public class Solution {
    private static long state = 0x7B0FFACE1234L;

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

    static double nextDouble() {
        return (nextLong() >>> 11) * 0x1.0p-53;
    }

    /** All simple paths (as arc-id lists) between two nodes, used by the exact LP oracle. */
    static List<int[]> simplePaths(int n, int[][] arcs, int from, int to) {
        List<int[]> paths = new ArrayList<>();
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i <= n; i++) adjacency.add(new ArrayList<>());
        for (int e = 0; e < arcs.length; e++) adjacency.get(arcs[e][0]).add(e);
        Deque<Integer> stack = new ArrayDeque<>();
        boolean[] onPath = new boolean[n + 1];
        collect(n, arcs, adjacency, from, to, stack, onPath, paths);
        return paths;
    }

    private static void collect(int n, int[][] arcs, List<List<Integer>> adjacency, int node, int to,
                                Deque<Integer> stack, boolean[] onPath, List<int[]> paths) {
        onPath[node] = true;
        if (node == to) {
            int[] copy = new int[stack.size()];
            int index = 0;
            for (int arc : stack) copy[index++] = arc;
            paths.add(copy);
        } else {
            for (int arc : adjacency.get(node)) {
                int other = arcs[arc][1];
                if (onPath[other]) continue;
                stack.addLast(arc);
                collect(n, arcs, adjacency, other, to, stack, onPath, paths);
                stack.removeLast();
            }
        }
        onPath[node] = false;
    }

    /** Gaussian elimination; null when the system is singular or inconsistent. */
    static double[] solveLinear(double[][] matrix, double[] rhs) {
        int size = rhs.length;
        double[][] augmented = new double[size][size + 1];
        for (int i = 0; i < size; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, size);
            augmented[i][size] = rhs[i];
        }
        for (int column = 0; column < size; column++) {
            int pivot = column;
            for (int row = column + 1; row < size; row++) {
                if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
            }
            if (Math.abs(augmented[pivot][column]) < 1e-9) return null;
            double[] swap = augmented[column];
            augmented[column] = augmented[pivot];
            augmented[pivot] = swap;
            for (int row = 0; row < size; row++) {
                if (row == column) continue;
                double factor = augmented[row][column] / augmented[column][column];
                if (factor == 0) continue;
                for (int k = column; k <= size; k++) augmented[row][k] -= factor * augmented[column][k];
            }
        }
        double[] solution = new double[size];
        for (int i = 0; i < size; i++) solution[i] = augmented[i][size] / augmented[i][i];
        return solution;
    }

    /**
     * Exact optimum of the path-based multi-commodity LP by vertex enumeration. Variables are
     * (lambda, x_path...); constraints are one demand row per commodity, one capacity row per arc and
     * the non-negativity rows. Every subset of constraints of full size is forced to equality and the
     * feasible solutions are ranked by lambda.
     */
    static double exactConcurrentFlow(int n, int[][] arcs, double[] capacities,
                                      int[][] commodities, double[] demands) {
        List<List<int[]>> pathsPerCommodity = new ArrayList<>();
        int variables = 1;
        for (int[] commodity : commodities) {
            List<int[]> paths = simplePaths(n, arcs, commodity[0], commodity[1]);
            pathsPerCommodity.add(paths);
            variables += paths.size();
        }
        List<double[]> rows = new ArrayList<>();
        List<Double> bounds = new ArrayList<>();
        for (int i = 0; i < commodities.length; i++) {
            double[] row = new double[variables];
            row[0] = demands[i];
            for (int[] path : pathsPerCommodity.get(i)) {
                int index = indexOfPath(pathsPerCommodity, i, path);
                row[index] = -1;
            }
            rows.add(row);
            bounds.add(0.0);
        }
        for (int e = 0; e < arcs.length; e++) {
            double[] row = new double[variables];
            for (int i = 0; i < commodities.length; i++) {
                for (int[] path : pathsPerCommodity.get(i)) {
                    int index = indexOfPath(pathsPerCommodity, i, path);
                    for (int arc : path) if (arc == e) row[index] += 1;
                }
            }
            rows.add(row);
            bounds.add(capacities[e]);
        }
        for (int v = 0; v < variables; v++) {
            double[] row = new double[variables];
            row[v] = -1;
            rows.add(row);
            bounds.add(0.0);
        }
        double best = 0;
        int[] chosen = new int[variables];
        best = enumerateVertices(rows, bounds, chosen, 0, 0, variables, best);
        return best;
    }

    private static int indexOfPath(List<List<int[]>> pathsPerCommodity, int commodity, int[] path) {
        int offset = 1;
        for (int i = 0; i < commodity; i++) offset += pathsPerCommodity.get(i).size();
        List<int[]> paths = pathsPerCommodity.get(commodity);
        for (int j = 0; j < paths.size(); j++) {
            if (Arrays.equals(paths.get(j), path)) return offset + j;
        }
        throw new IllegalStateException("path not found");
    }

    private static double enumerateVertices(List<double[]> rows, List<Double> bounds, int[] chosen,
                                            int start, int depth, int variables, double best) {
        if (depth == variables) {
            double[][] matrix = new double[variables][variables];
            double[] rhs = new double[variables];
            for (int i = 0; i < variables; i++) {
                System.arraycopy(rows.get(chosen[i]), 0, matrix[i], 0, variables);
                rhs[i] = bounds.get(chosen[i]);
            }
            double[] solution = solveLinear(matrix, rhs);
            if (solution == null) return best;
            for (int i = 0; i < variables; i++) {
                if (solution[i] < -1e-7) return best;
            }
            for (int r = 0; r < rows.size(); r++) {
                double value = 0;
                double[] row = rows.get(r);
                for (int v = 0; v < variables; v++) value += row[v] * solution[v];
                if (value > bounds.get(r) + 1e-7) return best;
            }
            return Math.max(best, solution[0]);
        }
        for (int index = start; index < rows.size(); index++) {
            chosen[depth] = index;
            best = enumerateVertices(rows, bounds, chosen, index + 1, depth + 1, variables, best);
        }
        return best;
    }

    /** Upper bound: no commodity can push more than the capacity leaving its source. */
    static double cutUpperBound(MultiCommodityFlow flow) {
        double bound = Double.MAX_VALUE;
        for (int commodity = 0; commodity < flow.commodityCount(); commodity++) {
            double leaving = 0;
            double entering = 0;
            for (int e = 0; e < flow.arcCount(); e++) {
                if (flow.tailOf(e) == flow.sourceOf(commodity)) leaving += flow.capacityOf(e);
                if (flow.headOf(e) == flow.sinkOf(commodity)) entering += flow.capacityOf(e);
            }
            bound = Math.min(bound, Math.min(leaving, entering) / flow.demandOf(commodity));
        }
        return bound;
    }

    public static void main(String[] args) {
        // --- single commodity: the LP optimum equals the min cut -----------------
        MultiCommodityFlow single = new MultiCommodityFlow(4, 12, 1);
        single.addArc(1, 2, 3);
        single.addArc(1, 3, 2);
        single.addArc(2, 4, 2);
        single.addArc(3, 4, 3);
        single.addCommodity(1, 4, 2);
        int[][] singleArcs = {{1, 2}, {1, 3}, {2, 4}, {3, 4}};
        double[] singleCapacities = {3, 2, 2, 3};
        int[][] singleCommodities = {{1, 4}};
        double[] singleDemands = {2};
        double exact = exactConcurrentFlow(4, singleArcs, singleCapacities, singleCommodities, singleDemands);
        check(Math.abs(exact - 2.0) < 1e-6, "single commodity optimum should be 4/2, got " + exact);
        MultiCommodityFlow.Result singleResult = single.solve(200, 0.1);
        check(singleResult.lambda > 0, "the single commodity is routable");
        check(singleResult.lambda <= exact + 1e-6,
                "approximation must not beat the optimum: " + singleResult.lambda + " vs " + exact);
        check(singleResult.lambda >= 0.9 * exact,
                "approximation should be close to the optimum: " + singleResult.lambda + " vs " + exact);

        // --- two commodities sharing a bottleneck --------------------------------
        MultiCommodityFlow shared = new MultiCommodityFlow(4, 12, 2);
        shared.addArc(1, 2, 4);
        shared.addArc(3, 4, 4);
        shared.addArc(2, 3, 2);
        shared.addArc(1, 4, 1);
        shared.addCommodity(1, 3, 2);
        shared.addCommodity(2, 4, 2);
        int[][] sharedArcs = {{1, 2}, {3, 4}, {2, 3}, {1, 4}};
        double[] sharedCapacities = {4, 4, 2, 1};
        int[][] sharedCommodities = {{1, 3}, {2, 4}};
        double[] sharedDemands = {2, 2};
        double sharedExact = exactConcurrentFlow(4, sharedArcs, sharedCapacities,
                sharedCommodities, sharedDemands);
        MultiCommodityFlow.Result sharedResult = shared.solve(300, 0.05);
        check(sharedResult.lambda > 0, "the shared bottleneck instance is routable");
        check(sharedResult.lambda <= sharedExact + 1e-6,
                "shared bottleneck value above the optimum: " + sharedResult.lambda);
        check(sharedResult.lambda >= 0.9 * sharedExact,
                "shared bottleneck approximation: " + sharedResult.lambda + " vs " + sharedExact);
        double upper = cutUpperBound(shared);
        check(sharedExact <= upper + 1e-9, "the exact optimum must respect the cut bound");
        check(sharedResult.lambda <= upper + 1e-9, "the approximation must respect the cut bound");

        // --- feasibility invariants and the uniform lambda ----------------------
        checkFeasible(shared, sharedResult, "shared bottleneck");
        for (int commodity = 0; commodity < shared.commodityCount(); commodity++) {
            check(Math.abs(sharedResult.commodityFlow[commodity]
                            - sharedResult.lambda * shared.demandOf(commodity)) < 1e-6,
                    "commodity " + commodity + " must receive lambda * demand");
        }
        check(sharedResult.lambda * 2 <= 2 + 1e-9, "the bottleneck arc limits the concurrent flow");

        // --- a commodity that cannot be routed at all ---------------------------
        MultiCommodityFlow unreachable = new MultiCommodityFlow(3, 4, 1);
        unreachable.addArc(1, 2, 1);
        unreachable.addCommodity(1, 3, 1);
        check(unreachable.solve(20, 0.1).lambda == 0, "an unroutable commodity yields lambda = 0");

        // --- rejected input ------------------------------------------------------
        check(rejects(() -> new MultiCommodityFlow(3, 4, 1).addArc(1, 1, 1)), "self loop rejected");
        check(rejects(() -> new MultiCommodityFlow(3, 4, 1).addArc(1, 4, 1)), "bad node rejected");
        check(rejects(() -> new MultiCommodityFlow(3, 4, 1).addArc(1, 2, 0)), "zero capacity rejected");
        check(rejects(() -> new MultiCommodityFlow(3, 4, 1).addCommodity(1, 1, 1)),
                "commodity with equal endpoints rejected");
        check(rejects(() -> new MultiCommodityFlow(3, 4, 1).addCommodity(1, 2, 0)),
                "zero demand rejected");
        check(rejects(() -> single.solve(0, 0.1)), "zero iterations rejected");
        check(rejects(() -> single.solve(10, 1.5)), "epsilon outside (0,1) rejected");

        // --- random small instances against the exact LP ------------------------
        double worstRatio = 1.0;
        int compared = 0;
        for (int trial = 0; trial < 25; trial++) {
            int nodes = 4;
            int arcBudget = 6 + nextInt(5);
            MultiCommodityFlow instance = new MultiCommodityFlow(nodes, 2 * arcBudget + 4, 4);
            List<int[]> arcList = new ArrayList<>();
            for (int e = 0; e < arcBudget; e++) {
                int u = 1 + nextInt(nodes);
                int v = 1 + nextInt(nodes);
                if (u == v) continue;
                double arcCapacity = 1 + nextInt(4);
                instance.addArc(u, v, arcCapacity);
                arcList.add(new int[]{u, v});
            }
            List<int[]> commodityList = new ArrayList<>();
            List<Double> demandList = new ArrayList<>();
            for (int c = 0; c < 2; c++) {
                int s = 0;
                List<Integer> reachable = Collections.emptyList();
                for (int attempt = 0; attempt < 8 && reachable.isEmpty(); attempt++) {
                    s = 1 + nextInt(nodes);
                    reachable = reachableFrom(nodes, arcList, s);
                }
                if (reachable.isEmpty()) break;
                int t = reachable.get(nextInt(reachable.size()));
                double units = 1 + nextInt(3);
                instance.addCommodity(s, t, units);
                commodityList.add(new int[]{s, t});
                demandList.add(units);
            }
            if (commodityList.isEmpty()) continue;
            double[] capacities = new double[arcList.size()];
            for (int arc = 0; arc < instance.arcCount(); arc++) capacities[arc] = instance.capacityOf(arc);
            double[] demandArray = new double[demandList.size()];
            for (int i = 0; i < demandArray.length; i++) demandArray[i] = demandList.get(i);
            double reference = exactConcurrentFlow(nodes, arcList.toArray(new int[0][]), capacities,
                    commodityList.toArray(new int[0][]), demandArray);
            MultiCommodityFlow.Result result = instance.solve(400, 0.05);
            checkFeasible(instance, result, "trial " + trial);
            check(result.lambda <= reference + 1e-6, "trial " + trial + " lambda " + result.lambda
                    + " above the exact optimum " + reference);
            if (reference > 1e-9) {
                compared++;
                worstRatio = Math.min(worstRatio, result.lambda / reference);
            }
        }
        check(compared >= 8, "enough random instances had a positive optimum: " + compared);
        check(worstRatio >= 0.9, "the approximation stayed close to the optimum: " + worstRatio);

        // --- larger instance: feasibility and the cut bound ---------------------
        int nodes = 30;
        int arcBudget = 90;
        MultiCommodityFlow large = new MultiCommodityFlow(nodes, 2 * arcBudget + 4, 6);
        for (int e = 0; e < arcBudget; e++) {
            int u = 1 + nextInt(nodes);
            int v = 1 + nextInt(nodes);
            if (u == v) continue;
            large.addArc(u, v, 1 + nextInt(9));
        }
        for (int e = 0; e < arcBudget / 2; e++) {
            int u = 1 + nextInt(nodes);
            int v = 1 + nextInt(nodes);
            if (u == v || u == v) continue;
            large.addArc(u, v, 1 + nextInt(9));
        }
        for (int c = 0; c < 5; c++) {
            int s = 1 + nextInt(5);
            int t = nodes - nextInt(5);
            if (s == t) t = nodes;
            large.addCommodity(s, t, 1 + nextInt(3));
        }
        long started = System.nanoTime();
        MultiCommodityFlow.Result largeResult = large.solve(600, 0.05);
        long elapsedMillis = (System.nanoTime() - started) / 1_000_000;
        checkFeasible(large, largeResult, "large instance");
        check(largeResult.lambda <= cutUpperBound(large) + 1e-6,
                "large instance must respect the cut bound: " + largeResult.lambda);
        check(elapsedMillis < 2000, "solving 600 iterations must be fast: " + elapsedMillis + " ms");
        MultiCommodityFlow.Result repeated = large.solve(600, 0.05);
        check(Math.abs(repeated.lambda - largeResult.lambda) < 1e-12, "the solver must be deterministic");

        System.out.println("PASS");
    }

    /** Directed reachability helper for the random instance generator. */
    static List<Integer> reachableFrom(int n, List<int[]> arcs, int from) {
        boolean[] seen = new boolean[n + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        seen[from] = true;
        queue.add(from);
        List<Integer> reached = new ArrayList<>();
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int[] arc : arcs) {
                if (arc[0] == node && !seen[arc[1]]) {
                    seen[arc[1]] = true;
                    queue.add(arc[1]);
                    reached.add(arc[1]);
                }
            }
        }
        return reached;
    }

    /** Arc loads must respect capacities and every commodity must receive lambda * demand. */
    static void checkFeasible(MultiCommodityFlow flow, MultiCommodityFlow.Result result, String label) {
        for (int e = 0; e < flow.arcCount(); e++) {
            check(result.arcLoad[e] <= flow.capacityOf(e) + 1e-6,
                    label + ": arc " + e + " overloaded " + result.arcLoad[e]);
            check(result.arcLoad[e] >= -1e-12, label + ": negative arc load");
        }
        for (int commodity = 0; commodity < flow.commodityCount(); commodity++) {
            check(result.commodityFlow[commodity] >= -1e-9, label + ": negative commodity flow");
            if (result.lambda > 0) {
                check(Math.abs(result.commodityFlow[commodity]
                                - result.lambda * flow.demandOf(commodity)) <= 1e-6 * (1 + result.lambda),
                        label + ": commodity flow " + result.commodityFlow[commodity]
                                + " is not lambda * demand");
            }
        }
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
