/** Task 706 - Implement max flow with lower/upper bounds and circulation feasibility.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * Capacities come as a pair (lower, upper). The standard two-phase construction turns both questions
 * into ordinary max flow:
 *
 * Phase 1 (circulation). Push the mandatory lower bound of every arc, lo(u,v) units, and record the
 * resulting imbalance balance[v] = sum of lo on incoming arcs - sum of lo on outgoing arcs. What is
 * left is a residual problem with capacities hi - lo in which every node must end balanced. Nodes with
 * balance > 0 have surplus inflow and get an arc v -> T, nodes with balance < 0 have a deficit and get
 * S -> v of capacity -balance; a max flow of S -> T saturating every arc out of S exists exactly when
 * a feasible circulation exists. The flow on the residual arcs of the *original* arcs is recovered as
 * initial - remaining = hi - lo - residual, so flow = lo + that.
 *
 * Phase 2 (max s-t flow). Add the return arc t -> s with capacity infinity, run Phase 1, and the flow
 * on that arc is the value of a feasible s-t flow. Then maximize inside the residual graph of that flow
 * (forward residual hi - f, backward residual f - lo, because a lower bound lets flow be pushed back
 * down to lo but no further) with s as source and t as sink, and add the augmentation to the value.
 * Maximum flow with lower bounds is the value of the min cut where forward arcs contribute hi and
 * backward arcs *subtract* lo, a formula the checks compare against by enumerating all 2^n cuts.
 *
 * Verified: every reported flow satisfies lo <= f <= hi and conservation at every node (except the
 * source/sink pair of an s-t flow, where the net outflow is the value), no augmenting path is left in
 * the residual graph, the value equals the brute-force optimum obtained by enumerating every integral
 * flow of tiny instances, the value equals the min-cut bound on larger instances, feasibility of a
 * circulation agrees with the same enumeration, and malformed input (lo > hi, negative lo, s = t,
 * unknown node) is rejected.
 */
import java.util.*;

class Dinic {
    private final int nodeCount;
    private final int[] head;
    private final int[] next;
    private final int[] to;
    private final long[] capacity;
    private final int[] level;
    private final int[] iterator;
    private int arcs;

    Dinic(int nodeCount, int maxArcs) {
        this.nodeCount = nodeCount;
        head = new int[nodeCount];
        Arrays.fill(head, -1);
        next = new int[maxArcs];
        to = new int[maxArcs];
        capacity = new long[maxArcs];
        level = new int[nodeCount];
        iterator = new int[nodeCount];
    }

    /** Adds an arc plus its residual twin in the same block of two; returns the forward index. */
    int addArc(int u, int v, long cap) {
        int index = arcs;
        to[index] = v;
        capacity[index] = cap;
        next[index] = head[u];
        head[u] = index;
        to[index + 1] = u;
        capacity[index + 1] = 0;
        next[index + 1] = head[v];
        head[v] = index + 1;
        arcs += 2;
        return index;
    }

    long residualOf(int arc) {
        return capacity[arc];
    }

    long maxFlow(int source, int sink) {
        long total = 0;
        while (buildLevels(source, sink)) {
            System.arraycopy(head, 0, iterator, 0, nodeCount);
            long pushed;
            while ((pushed = augment(source, sink, Long.MAX_VALUE / 8)) > 0) total += pushed;
        }
        return total;
    }

    private boolean buildLevels(int source, int sink) {
        Arrays.fill(level, -1);
        Deque<Integer> queue = new ArrayDeque<>();
        level[source] = 0;
        queue.add(source);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int arc = head[node]; arc != -1; arc = next[arc]) {
                if (capacity[arc] > 0 && level[to[arc]] < 0) {
                    level[to[arc]] = level[node] + 1;
                    queue.add(to[arc]);
                }
            }
        }
        return level[sink] >= 0;
    }

    private long augment(int node, int sink, long limit) {
        if (node == sink) return limit;
        for (; iterator[node] != -1; iterator[node] = next[iterator[node]]) {
            int arc = iterator[node];
            int other = to[arc];
            if (capacity[arc] <= 0 || level[other] != level[node] + 1) continue;
            long pushed = augment(other, sink, Math.min(limit, capacity[arc]));
            if (pushed > 0) {
                capacity[arc] -= pushed;
                capacity[arc ^ 1] += pushed;
                return pushed;
            }
        }
        return 0;
    }
}

class BoundedFlowNetwork {
    static final long INFINITE = Long.MAX_VALUE / 8;
    static final long INFEASIBLE = Long.MIN_VALUE;

    private final int n;
    private final int[] tail;
    private final int[] head;
    private final long[] lower;
    private final long[] upper;
    private final long[] flow;
    private int arcs;

    BoundedFlowNetwork(int nodeCount, int maxArcs) {
        if (nodeCount < 1) throw new IllegalArgumentException("need at least one node");
        n = nodeCount;
        tail = new int[maxArcs];
        head = new int[maxArcs];
        lower = new long[maxArcs];
        upper = new long[maxArcs];
        flow = new long[maxArcs];
    }

    int nodeCount() {
        return n;
    }

    int edgeCount() {
        return arcs;
    }

    int tailOf(int id) {
        return tail[id];
    }

    int headOf(int id) {
        return head[id];
    }

    long lowerBound(int id) {
        return lower[id];
    }

    long upperBound(int id) {
        return upper[id];
    }

    long flowOn(int id) {
        return flow[id];
    }

    void addEdge(int u, int v, long lowerBound, long upperBound) {
        if (u < 1 || u > n || v < 1 || v > n) throw new IllegalArgumentException("node out of range");
        if (lowerBound < 0) throw new IllegalArgumentException("negative lower bound: " + lowerBound);
        if (upperBound < lowerBound) throw new IllegalArgumentException("upper below lower: "
                + upperBound + " < " + lowerBound);
        tail[arcs] = u;
        head[arcs] = v;
        lower[arcs] = lowerBound;
        upper[arcs] = upperBound;
        arcs++;
    }

    boolean feasibleCirculation() {
        return runPhaseOne(-1, -1) != INFEASIBLE;
    }

    /** Maximum s-t flow; INFEASIBLE when no flow satisfies the lower bounds. */
    long maxFlow(int s, int t) {
        if (s < 1 || s > n || t < 1 || t > n) throw new IllegalArgumentException("node out of range");
        if (s == t) throw new IllegalArgumentException("source and sink must differ");
        if (runPhaseOne(s, t) == INFEASIBLE) return INFEASIBLE;
        int[] forward = new int[arcs];
        int[] backward = new int[arcs];
        long[] forwardInitial = new long[arcs];
        long[] backwardInitial = new long[arcs];
        Dinic residual = new Dinic(n + 1, 2 * (2 * arcs + 2));
        for (int e = 0; e < arcs; e++) {
            forwardInitial[e] = upper[e] - flow[e];
            forward[e] = residual.addArc(tail[e], head[e], forwardInitial[e]);
            backwardInitial[e] = flow[e] - lower[e];
            backward[e] = residual.addArc(head[e], tail[e], backwardInitial[e]);
        }
        residual.maxFlow(s, t);
        for (int e = 0; e < arcs; e++) {
            long usedForward = forwardInitial[e] - residual.residualOf(forward[e]);
            long usedBackward = backwardInitial[e] - residual.residualOf(backward[e]);
            flow[e] += usedForward - usedBackward;
        }
        return netOutflow(s);
    }

    private long runPhaseOne(int s, int t) {
        long[] balance = new long[n + 1];
        for (int e = 0; e < arcs; e++) {
            balance[tail[e]] -= lower[e];
            balance[head[e]] += lower[e];
        }
        int superSource = n + 1;
        int superSink = n + 2;
        Dinic dinic = new Dinic(n + 3, 2 * (arcs + n + 4));
        int[] original = new int[arcs];
        for (int e = 0; e < arcs; e++) original[e] = dinic.addArc(tail[e], head[e], upper[e] - lower[e]);
        int dummy = -1;
        if (s > 0) dummy = dinic.addArc(t, s, INFINITE);
        long required = 0;
        for (int v = 1; v <= n; v++) {
            if (balance[v] > 0) {
                // surplus inflow from the lower bounds: v must push that much extra out
                dinic.addArc(superSource, v, balance[v]);
                required += balance[v];
            } else if (balance[v] < 0) {
                dinic.addArc(v, superSink, -balance[v]);
            }
        }
        long pushed = dinic.maxFlow(superSource, superSink);
        if (pushed != required) return INFEASIBLE;
        for (int e = 0; e < arcs; e++) {
            flow[e] = lower[e] + (upper[e] - lower[e] - dinic.residualOf(original[e]));
        }
        return s > 0 ? INFINITE - dinic.residualOf(dummy) : 0;
    }

    /** True when some residual path could still carry flow from s to t: false means optimal. */
    boolean hasAugmentingPath(int s, int t) {
        boolean[] seen = new boolean[n + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        seen[s] = true;
        queue.add(s);
        while (!queue.isEmpty()) {
            int node = queue.poll();
            if (node == t) return true;
            for (int e = 0; e < arcs; e++) {
                int candidate = 0;
                if (tail[e] == node && upper[e] - flow[e] > 0) candidate = head[e];
                else if (head[e] == node && flow[e] - lower[e] > 0) candidate = tail[e];
                if (candidate != 0 && !seen[candidate]) {
                    seen[candidate] = true;
                    queue.add(candidate);
                }
            }
        }
        return false;
    }

    long netOutflow(int node) {
        long out = 0;
        for (int e = 0; e < arcs; e++) {
            if (tail[e] == node) out += flow[e];
            else if (head[e] == node) out -= flow[e];
        }
        return out;
    }

    /** Null when the flow is a valid circulation, otherwise the first violation found. */
    String verifyCirculation() {
        String bounds = verifyBounds();
        if (bounds != null) return bounds;
        for (int v = 1; v <= n; v++) {
            if (netOutflow(v) != 0) return "node " + v + " is not balanced: " + netOutflow(v);
        }
        return null;
    }

    /** Null when the flow is a valid s-t flow of the given value, otherwise the first violation. */
    String verifyFlow(int s, int t, long value) {
        String bounds = verifyBounds();
        if (bounds != null) return bounds;
        for (int v = 1; v <= n; v++) {
            long net = netOutflow(v);
            if (v == s) {
                if (net != value) return "source outflow " + net + " != value " + value;
            } else if (v == t) {
                if (net != -value) return "sink inflow does not match value " + value;
            } else if (net != 0) {
                return "node " + v + " is not balanced: " + net;
            }
        }
        return null;
    }

    private String verifyBounds() {
        for (int e = 0; e < arcs; e++) {
            if (flow[e] < lower[e] || flow[e] > upper[e]) {
                return "arc " + tail[e] + "->" + head[e] + " carries " + flow[e]
                        + " outside [" + lower[e] + "," + upper[e] + "]";
            }
        }
        return null;
    }
}

public class Solution {
    private static long state = 0xB0DE9CAFEL;

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

    /** Min cut capacity with lower bounds: forward arcs add hi, backward arcs subtract lo. */
    static long minCutBound(int n, int[][] edges, int s, int t) {
        long best = Long.MAX_VALUE;
        for (int mask = 0; mask < (1 << n); mask++) {
            if ((mask & (1 << (s - 1))) == 0) continue;
            if ((mask & (1 << (t - 1))) != 0) continue;
            long capacity = 0;
            for (int[] edge : edges) {
                boolean tailIn = (mask & (1 << (edge[0] - 1))) != 0;
                boolean headIn = (mask & (1 << (edge[1] - 1))) != 0;
                if (tailIn && !headIn) capacity += edge[3];
                else if (!tailIn && headIn) capacity -= edge[2];
            }
            best = Math.min(best, capacity);
        }
        return best;
    }

    /**
     * Enumeration of every integral flow of a tiny instance. With s > 0 it looks for s-t flows of
     * non-negative value (the same convention the two-phase construction uses: the return arc t -> s
     * has lower bound 0) and records the largest value; with s < 0 it only asks whether some
     * circulation exists. Returns false when no assignment qualifies.
     */
    static boolean enumerateFeasible(int n, int[][] edges, long[] chosen, int index,
                                     int s, int t, long[] bestValue) {
        if (index == edges.length) {
            long[] balance = new long[n + 1];
            for (int e = 0; e < edges.length; e++) {
                balance[edges[e][0]] -= chosen[e];
                balance[edges[e][1]] += chosen[e];
            }
            if (s < 0) {
                for (int v = 1; v <= n; v++) {
                    if (balance[v] != 0) return false;
                }
                return true;
            }
            for (int v = 1; v <= n; v++) {
                if (v != s && v != t && balance[v] != 0) return false;
            }
            if (balance[s] + balance[t] != 0) return false;
            long value = -balance[s];
            if (value < 0) return false;               // a flow from t to s is not an s-t flow here
            bestValue[0] = Math.max(bestValue[0], value);
            return true;
        }
        boolean any = false;
        for (long value = edges[index][2]; value <= edges[index][3]; value++) {
            chosen[index] = value;
            any |= enumerateFeasible(n, edges, chosen, index + 1, s, t, bestValue);
        }
        return any;
    }

    /** -1 when no integral s-t flow satisfies the bounds. */
    static long bruteForceMaxFlow(int n, int[][] edges, int s, int t) {
        long[] best = {Long.MIN_VALUE};
        boolean feasible = enumerateFeasible(n, edges, new long[edges.length], 0, s, t, best);
        return feasible ? best[0] : -1;
    }

    static boolean bruteForceCirculationFeasible(int n, int[][] edges) {
        return enumerateFeasible(n, edges, new long[edges.length], 0, -1, -1, new long[]{0});
    }

    public static void main(String[] args) {
        // --- a feasible circulation -------------------------------------------
        BoundedFlowNetwork circulation = new BoundedFlowNetwork(3, 8);
        circulation.addEdge(1, 2, 2, 5);
        circulation.addEdge(2, 3, 1, 2);
        circulation.addEdge(3, 1, 1, 3);
        check(circulation.feasibleCirculation(), "triangle circulation is feasible");
        check(circulation.verifyCirculation() == null,
                "triangle circulation invariant: " + circulation.verifyCirculation());
        check(circulation.flowOn(0) >= 2 && circulation.flowOn(0) <= 5, "lower bound respected");

        // --- an infeasible circulation ----------------------------------------
        BoundedFlowNetwork blocked = new BoundedFlowNetwork(2, 4);
        blocked.addEdge(1, 2, 3, 3);
        blocked.addEdge(2, 1, 0, 1);
        check(!blocked.feasibleCirculation(), "3 units cannot circulate through a 1-unit return arc");

        BoundedFlowNetwork trivial = new BoundedFlowNetwork(4, 4);
        trivial.addEdge(1, 2, 0, 0);
        check(trivial.feasibleCirculation(), "the zero circulation is always feasible");
        check(trivial.flowOn(0) == 0, "zero capacity carries zero flow");

        // --- max flow with lower bounds, hand checked -------------------------
        BoundedFlowNetwork bounded = new BoundedFlowNetwork(3, 6);
        bounded.addEdge(1, 2, 1, 3);
        bounded.addEdge(2, 3, 0, 2);
        bounded.addEdge(1, 3, 0, 1);
        long value = bounded.maxFlow(1, 3);
        check(value == 3, "expected max flow 3, got " + value);
        check(bounded.verifyFlow(1, 3, value) == null,
                "bounded flow invariant: " + bounded.verifyFlow(1, 3, value));
        check(!bounded.hasAugmentingPath(1, 3), "no augmenting path may remain after the maximum");
        check(bounded.flowOn(0) >= 1, "the mandatory lower bound is honoured");
        int[][] boundedEdges = {{1, 2, 1, 3}, {2, 3, 0, 2}, {1, 3, 0, 1}};
        check(value == minCutBound(3, boundedEdges, 1, 3),
                "max flow must equal the lower-bounded min cut: " + value + " vs "
                        + minCutBound(3, boundedEdges, 1, 3));
        check(value == bruteForceMaxFlow(3, boundedEdges, 1, 3),
                "max flow must equal the enumeration optimum");

        // --- forced flows (lo == hi) ------------------------------------------
        BoundedFlowNetwork forced = new BoundedFlowNetwork(3, 4);
        forced.addEdge(1, 2, 2, 2);
        forced.addEdge(2, 3, 2, 2);
        check(forced.maxFlow(1, 3) == 2, "a forced chain carries exactly two units");
        check(forced.verifyFlow(1, 3, 2) == null, "forced chain invariant");
        BoundedFlowNetwork forcedDetour = new BoundedFlowNetwork(3, 6);
        forcedDetour.addEdge(1, 2, 2, 2);
        forcedDetour.addEdge(2, 1, 0, 0);
        check(forcedDetour.maxFlow(1, 3) == BoundedFlowNetwork.INFEASIBLE,
                "a forced arc with no way to return is infeasible");

        // --- rejected input ----------------------------------------------------
        BoundedFlowNetwork rejection = new BoundedFlowNetwork(3, 8);
        check(rejects(() -> rejection.addEdge(1, 2, 4, 3)), "lo > hi rejected");
        check(rejects(() -> rejection.addEdge(1, 2, -1, 3)), "negative lo rejected");
        check(rejects(() -> rejection.addEdge(1, 9, 0, 1)), "unknown node rejected");
        check(rejects(() -> rejection.maxFlow(1, 1)), "s == t rejected");
        check(rejects(() -> rejection.maxFlow(0, 3)), "node 0 rejected");

        // --- random tiny instances against the enumeration --------------------
        int feasibleCount = 0;
        for (int trial = 0; trial < 60; trial++) {
            int n = 4;
            int m = 2 + nextInt(3);
            int[][] edges = new int[m][];
            BoundedFlowNetwork network = new BoundedFlowNetwork(n, 2 * m + 4);
            for (int e = 0; e < m; e++) {
                int u = 1 + nextInt(n);
                int v = 1 + nextInt(n);
                if (u == v) v = u % n + 1;
                int low = nextInt(3);
                int high = low + nextInt(3);
                edges[e] = new int[]{u, v, low, high};
                network.addEdge(u, v, low, high);
            }
            long computed = network.maxFlow(1, n);
            long expected = bruteForceMaxFlow(n, edges, 1, n);
            check((computed == BoundedFlowNetwork.INFEASIBLE) == (expected == -1),
                    "trial " + trial + " feasibility disagreed with the enumeration: "
                            + computed + " vs " + expected);
            if (expected == -1) continue;
            feasibleCount++;
            check(computed == expected, "trial " + trial + " value " + computed + " != " + expected);
            check(network.verifyFlow(1, n, computed) == null,
                    "trial " + trial + " invariant: " + network.verifyFlow(1, n, computed));
            check(!network.hasAugmentingPath(1, n), "trial " + trial + " augmenting path left");
            check(computed == minCutBound(n, edges, 1, n),
                    "trial " + trial + " min cut " + minCutBound(n, edges, 1, n) + " != " + computed);
        }
        check(feasibleCount >= 10, "enough random instances were feasible: " + feasibleCount);

        // --- random circulations against the enumeration ----------------------
        int circulationFeasible = 0;
        for (int trial = 0; trial < 40; trial++) {
            int n = 3;
            int m = 2 + nextInt(3);
            int[][] edges = new int[m][];
            BoundedFlowNetwork network = new BoundedFlowNetwork(n, 2 * m + 4);
            for (int e = 0; e < m; e++) {
                int u = 1 + nextInt(n);
                int v = 1 + nextInt(n);
                if (u == v) v = u % n + 1;
                int low = nextInt(3);
                int high = low + nextInt(3);
                edges[e] = new int[]{u, v, low, high};
                network.addEdge(u, v, low, high);
            }
            boolean feasible = network.feasibleCirculation();
            check(feasible == bruteForceCirculationFeasible(n, edges),
                    "trial " + trial + " circulation feasibility disagreed with the enumeration");
            if (feasible) {
                circulationFeasible++;
                check(network.verifyCirculation() == null,
                        "trial " + trial + " circulation invariant: " + network.verifyCirculation());
            }
        }
        check(circulationFeasible > 3, "some random circulations are feasible: " + circulationFeasible);

        // --- larger instance: value against the min cut -----------------------
        int n = 14;
        int edgeBudget = 45;
        List<int[]> arcs = new ArrayList<>();
        BoundedFlowNetwork large = new BoundedFlowNetwork(n, 2 * edgeBudget + 4);
        for (int e = 0; e < edgeBudget; e++) {
            int u = 1 + nextInt(n);
            int v = 1 + nextInt(n);
            if (u == v || (u == 1 && v == n)) continue;
            int high = 1 + nextInt(9);
            arcs.add(new int[]{u, v, 0, high});
            large.addEdge(u, v, 0, high);
        }
        int[][] largeArcs = arcs.toArray(new int[0][]);
        long largeValue = large.maxFlow(1, n);
        check(largeValue != BoundedFlowNetwork.INFEASIBLE, "the plain network is always feasible");
        check(large.verifyFlow(1, n, largeValue) == null, "large instance invariant");
        check(!large.hasAugmentingPath(1, n), "large instance is maximum");
        check(largeValue == minCutBound(n, largeArcs, 1, n),
                "large instance value " + largeValue + " vs min cut "
                        + minCutBound(n, largeArcs, 1, n));

        // --- lower bounds derived from that flow stay feasible ----------------
        BoundedFlowNetwork tightened = new BoundedFlowNetwork(n, 2 * edgeBudget + 4);
        List<int[]> tightenedArcs = new ArrayList<>();
        for (int e = 0; e < large.edgeCount(); e++) {
            long lowerBound = Math.min(large.flowOn(e), large.upperBound(e));
            tightened.addEdge(large.tailOf(e), large.headOf(e), lowerBound, large.upperBound(e));
            tightenedArcs.add(new int[]{large.tailOf(e), large.headOf(e),
                    (int) lowerBound, (int) large.upperBound(e)});
        }
        long tightenedValue = tightened.maxFlow(1, n);
        check(tightenedValue != BoundedFlowNetwork.INFEASIBLE,
                "the previously used flow proves feasibility");
        check(tightenedValue == largeValue, "raising the lower bounds to a used flow changes nothing");
        check(tightened.verifyFlow(1, n, tightenedValue) == null, "tightened instance invariant");
        check(tightenedValue == minCutBound(n, tightenedArcs.toArray(new int[0][]), 1, n),
                "tightened instance matches its min cut");
        for (int e = 0; e < tightened.edgeCount(); e++) {
            check(tightened.flowOn(e) >= tightened.lowerBound(e), "tightened lower bound respected");
        }

        // --- determinism and speed --------------------------------------------
        long started = System.nanoTime();
        BoundedFlowNetwork repeat = new BoundedFlowNetwork(n, 2 * edgeBudget + 4);
        for (int e = 0; e < large.edgeCount(); e++) {
            repeat.addEdge(large.tailOf(e), large.headOf(e), 0, large.upperBound(e));
        }
        check(repeat.maxFlow(1, n) == largeValue, "the same instance gives the same value");
        long elapsedMillis = (System.nanoTime() - started) / 1_000_000;
        check(elapsedMillis < 2000, "solving must be fast: " + elapsedMillis + " ms");

        System.out.println("PASS");
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
