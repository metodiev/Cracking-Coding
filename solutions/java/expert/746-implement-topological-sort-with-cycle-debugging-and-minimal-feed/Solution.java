/** Task 746 - Implement topological sort with cycle debugging and minimal feedback arc set heuristics.
 * (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * The task has three parts, all on one small digraph class:
 *   1. topological sort - Kahn's algorithm. When it stalls, the nodes still holding residual
 *      in-degree are exactly the ones that lie on or downstream of a cycle, which is the diagnostic
 *      that makes the failure actionable;
 *   2. cycle debugging - Tarjan's SCC decomposition plus a *minimal* witness cycle per nontrivial
 *      component: for every arc (u,w) inside the component a BFS finds the shortest path w -> u, and
 *      the shortest such round trip is reported as a closed, simple cycle;
 *   3. feedback arc set heuristics - the Eades-Lin-Smyth greedy (peel sinks to the right, sources to
 *      the left, otherwise remove a vertex of maximum out-degree minus in-degree to the left) and a
 *      second heuristic that orders by in-degree minus out-degree. Both turn the digraph into a DAG
 *      by deleting the arcs that run backwards in their order; the number of deleted arcs is
 *      compared against the exhaustive optimum.
 *
 * Simplification (documented). The minimum feedback arc set is NP-hard, so both heuristics are
 * compared against an *exhaustive* optimum (all n! orderings) which is only feasible for the small
 * instances used here; SCC extraction is the plain recursive Tarjan (fine for a handful of nodes)
 * and the graph is a dense adjacency matrix, so nothing here scales beyond a few dozen nodes.
 *
 * Verified. Every acyclic instance yields a permutation in which *all* arcs point forward (checked
 * arc by arc); every reported cycle is a closed walk along real arcs with pairwise distinct vertices
 * and lies inside one nontrivial SCC; the SCC partition is cross-checked against an independent
 * mutual-reachability oracle; deleting each heuristic's backward arcs leaves an acyclic graph
 * (re-verified with Kahn) and the deleted count equals the number of backward arcs in its order;
 * each heuristic is bounded below by the exhaustive optimum, and on the tested instances stays
 * within one arc of it; self-loops, empty graphs and duplicate arcs are handled explicitly.
 */
import java.util.*;

/** Small dense digraph with topological sort, SCC/cycle diagnostics and FAS heuristics. */
final class DiGraph {
    final int n;
    final boolean[][] adj;
    final List<int[]> arcs = new ArrayList<>();     // deterministic iteration order

    DiGraph(int n) {
        this.n = n;
        this.adj = new boolean[n][n];
    }

    void addArc(int u, int v) {
        if (u < 0 || v < 0 || u >= n || v >= n) throw new IllegalArgumentException("bad arc " + u + "->" + v);
        if (adj[u][v]) return;                       // duplicates are ignored
        adj[u][v] = true;
        arcs.add(new int[]{u, v});
    }

    int outDegree(int u) {
        int count = 0;
        for (int v = 0; v < n; v++) if (adj[u][v]) count++;
        return count;
    }

    int inDegree(int u) {
        int count = 0;
        for (int v = 0; v < n; v++) if (adj[v][u]) count++;
        return count;
    }

    /** Kahn's algorithm; returns null when the graph has a cycle. */
    int[] topologicalOrder() {
        int[] indegree = new int[n];
        for (int[] arc : arcs) indegree[arc[1]]++;
        Deque<Integer> ready = new ArrayDeque<>();
        for (int v = 0; v < n; v++) if (indegree[v] == 0) ready.add(v);
        int[] order = new int[n];
        int filled = 0;
        while (!ready.isEmpty()) {
            int u = ready.poll();
            order[filled++] = u;
            for (int v = 0; v < n; v++) {
                if (adj[u][v] && --indegree[v] == 0) ready.add(v);
            }
        }
        return filled == n ? order : null;
    }

    /** Nodes left with residual in-degree when Kahn's algorithm stalls (cycle members or their heirs). */
    List<Integer> blockedNodes() {
        int[] indegree = new int[n];
        for (int[] arc : arcs) indegree[arc[1]]++;
        Deque<Integer> ready = new ArrayDeque<>();
        for (int v = 0; v < n; v++) if (indegree[v] == 0) ready.add(v);
        while (!ready.isEmpty()) {
            int u = ready.poll();
            for (int v = 0; v < n; v++) if (adj[u][v] && --indegree[v] == 0) ready.add(v);
        }
        List<Integer> blocked = new ArrayList<>();
        for (int v = 0; v < n; v++) if (indegree[v] > 0) blocked.add(v);
        return blocked;
    }

    /** Tarjan SCC decomposition; the components come out in deterministic (reverse finish) order. */
    List<List<Integer>> stronglyConnectedComponents() {
        int[] index = new int[n];
        int[] low = new int[n];
        boolean[] onStack = new boolean[n];
        Arrays.fill(index, -1);
        Deque<Integer> stack = new ArrayDeque<>();
        List<List<Integer>> components = new ArrayList<>();
        int[] counter = {0};
        for (int v = 0; v < n; v++) {
            if (index[v] < 0) tarjan(v, index, low, onStack, stack, components, counter);
        }
        components.sort((a, b) -> Integer.compare(a.get(0), b.get(0)));
        return components;
    }

    private void tarjan(int v, int[] index, int[] low, boolean[] onStack, Deque<Integer> stack,
                        List<List<Integer>> components, int[] counter) {
        index[v] = low[v] = counter[0]++;
        stack.push(v);
        onStack[v] = true;
        for (int w = 0; w < n; w++) {
            if (!adj[v][w]) continue;
            if (index[w] < 0) {
                tarjan(w, index, low, onStack, stack, components, counter);
                low[v] = Math.min(low[v], low[w]);
            } else if (onStack[w]) {
                low[v] = Math.min(low[v], index[w]);
            }
        }
        if (low[v] == index[v]) {
            List<Integer> component = new ArrayList<>();
            int w;
            do {
                w = stack.pop();
                onStack[w] = false;
                component.add(w);
            } while (w != v);
            Collections.sort(component);
            components.add(component);
        }
    }

    /** Independent SCC oracle: two nodes belong together iff each reaches the other. */
    List<List<Integer>> sccByReachability() {
        boolean[][] reach = new boolean[n][n];
        for (int v = 0; v < n; v++) {
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(v);
            reach[v][v] = true;
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int w = 0; w < n; w++) {
                    if (adj[u][w] && !reach[v][w]) {
                        reach[v][w] = true;
                        queue.add(w);
                    }
                }
            }
        }
        List<List<Integer>> components = new ArrayList<>();
        boolean[] used = new boolean[n];
        for (int v = 0; v < n; v++) {
            if (used[v]) continue;
            List<Integer> component = new ArrayList<>();
            for (int w = 0; w < n; w++) {
                if (reach[v][w] && reach[w][v]) {
                    component.add(w);
                    used[w] = true;
                }
            }
            components.add(component);
        }
        components.sort((a, b) -> Integer.compare(a.get(0), b.get(0)));
        return components;
    }

    /** Shortest closed cycle through the arcs of one strongly connected component, or null. */
    List<Integer> minimalCycleIn(List<Integer> component) {
        List<Integer> best = null;
        for (int u : component) {
            if (adj[u][u]) return List.of(u, u);          // a self loop is the shortest cycle possible
        }
        for (int u : component) {
            for (int w = 0; w < n; w++) {
                if (!adj[u][w] || w == u) continue;
                int[] parent = shortestPathTree(w, u);
                if (parent == null) continue;
                LinkedList<Integer> path = new LinkedList<>();
                for (int v = u; v != w; v = parent[v]) path.addFirst(v);
                // path = [v1, ..., vk, u] where v1 is adjacent to w, so u -> w -> v1 -> ... -> u closes
                List<Integer> cycle = new ArrayList<>();
                cycle.add(u);
                cycle.add(w);
                cycle.addAll(path.subList(0, path.size() - 1));
                cycle.add(u);
                if (best == null || cycle.size() < best.size()) best = cycle;
            }
        }
        return best;
    }

    /** BFS from `from`; returns parent pointers for a path to `to`, or null when unreachable. */
    private int[] shortestPathTree(int from, int to) {
        int[] parent = new int[n];
        Arrays.fill(parent, -2);
        Deque<Integer> queue = new ArrayDeque<>();
        parent[from] = -1;
        queue.add(from);
        while (!queue.isEmpty()) {
            int u = queue.poll();
            if (u == to) return parent;
            for (int v = 0; v < n; v++) {
                if (adj[u][v] && parent[v] == -2) {
                    parent[v] = u;
                    queue.add(v);
                }
            }
        }
        return parent[to] == -2 ? null : parent;
    }

    /** Eades-Lin-Smyth ordering: sinks right, sources left, otherwise the best pivot to the left. */
    int[] eadesLinSmythOrder() {
        boolean[] removed = new boolean[n];
        int[] left = new int[n];
        int[] right = new int[n];
        int leftCount = 0;
        int rightCount = 0;
        for (int step = 0; step < n; step++) {
            int sink = -1;
            for (int v = 0; v < n; v++) {
                if (removed[v]) continue;
                int out = 0;
                for (int w = 0; w < n; w++) if (!removed[w] && adj[v][w]) out++;
                if (out == 0) {
                    sink = v;
                    break;
                }
            }
            if (sink >= 0) {
                right[rightCount++] = sink;
                removed[sink] = true;
                continue;
            }
            int source = -1;
            for (int v = 0; v < n; v++) {
                if (removed[v]) continue;
                int in = 0;
                for (int w = 0; w < n; w++) if (!removed[w] && adj[w][v]) in++;
                if (in == 0) {
                    source = v;
                    break;
                }
            }
            if (source >= 0) {
                left[leftCount++] = source;
                removed[source] = true;
                continue;
            }
            int pivot = -1;
            int bestDelta = Integer.MIN_VALUE;
            for (int v = 0; v < n; v++) {
                if (removed[v]) continue;
                int out = 0;
                int in = 0;
                for (int w = 0; w < n; w++) {
                    if (removed[w]) continue;
                    if (adj[v][w]) out++;
                    if (adj[w][v]) in++;
                }
                if (out - in > bestDelta) {
                    bestDelta = out - in;
                    pivot = v;
                }
            }
            left[leftCount++] = pivot;
            removed[pivot] = true;
        }
        int[] order = new int[n];
        for (int i = 0; i < leftCount; i++) order[i] = left[i];
        for (int i = 0; i < rightCount; i++) order[leftCount + i] = right[rightCount - 1 - i];
        return order;
    }

    /** Simple heuristic: order vertices by (in-degree - out-degree) descending. */
    int[] inMinusOutOrder() {
        Integer[] ids = new Integer[n];
        for (int i = 0; i < n; i++) ids[i] = i;
        Arrays.sort(ids, (a, b) -> {
            int da = inDegree(a) - outDegree(a);
            int db = inDegree(b) - outDegree(b);
            return da != db ? Integer.compare(da, db) : Integer.compare(a, b);
        });
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = ids[i];
        return order;
    }

    /** Arcs that run backwards in the given order; a self loop is always a feedback arc. */
    List<int[]> feedbackArcs(int[] order) {
        int[] position = new int[n];
        for (int i = 0; i < n; i++) position[order[i]] = i;
        List<int[]> back = new ArrayList<>();
        for (int[] arc : arcs) {
            if (arc[0] == arc[1] || position[arc[0]] > position[arc[1]]) back.add(arc);
        }
        return back;
    }

    DiGraph withoutArcs(List<int[]> removed) {
        DiGraph copy = new DiGraph(n);
        for (int[] arc : arcs) {
            boolean drop = false;
            for (int[] kill : removed) {
                if (kill[0] == arc[0] && kill[1] == arc[1]) drop = true;
            }
            if (!drop) copy.addArc(arc[0], arc[1]);
        }
        return copy;
    }

    /** Exhaustive minimum feedback arc set over all n! orderings. */
    int exhaustiveMinFeedbackArcSet() {
        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        int[] best = {Integer.MAX_VALUE};
        permute(permutation, 0, best);
        return best[0];
    }

    private void permute(int[] permutation, int position, int[] best) {
        if (position == n) {
            best[0] = Math.min(best[0], feedbackArcs(permutation).size());
            return;
        }
        for (int i = position; i < n; i++) {
            int swap = permutation[position];
            permutation[position] = permutation[i];
            permutation[i] = swap;
            permute(permutation, position + 1, best);
            swap = permutation[position];
            permutation[position] = permutation[i];
            permutation[i] = swap;
        }
    }
}

public class Solution {
    private static long seed = 0xBE5466CF34E90C6CL;
    static int checksRun;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static void assertTopological(DiGraph graph, int[] order) {
        check(order != null, "expected an acyclic digraph");
        boolean[] seen = new boolean[graph.n];
        for (int v : order) {
            check(!seen[v], "a node appears twice in the order");
            seen[v] = true;
        }
        int[] position = new int[graph.n];
        for (int i = 0; i < order.length; i++) position[order[i]] = i;
        for (int[] arc : graph.arcs) {
            check(position[arc[0]] < position[arc[1]],
                    "arc " + arc[0] + "->" + arc[1] + " runs backwards in the topological order");
        }
    }

    static void assertRealCycle(DiGraph graph, List<Integer> cycle) {
        check(cycle != null && cycle.size() >= 2, "a cycle witness was produced");
        check(cycle.get(0).equals(cycle.get(cycle.size() - 1)), "the witness is closed");
        Set<Integer> interior = new HashSet<>(cycle.subList(0, cycle.size() - 1));
        check(interior.size() == cycle.size() - 1, "the witness is a simple cycle");
        for (int i = 1; i < cycle.size(); i++) {
            check(graph.adj[cycle.get(i - 1)][cycle.get(i)],
                    "the witness uses a non-existing arc " + cycle.get(i - 1) + "->" + cycle.get(i));
        }
    }

    static String draw(List<Integer> cycle) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cycle.size(); i++) {
            if (i > 0) sb.append(" -> ");
            sb.append(cycle.get(i));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // --- boundaries: empty graph, single node, self loop ---------------------
        DiGraph empty = new DiGraph(0);
        int[] emptyOrder = empty.topologicalOrder();
        check(emptyOrder != null && emptyOrder.length == 0, "empty graph has an empty order");
        check(empty.stronglyConnectedComponents().isEmpty(), "empty graph has no components");

        DiGraph single = new DiGraph(1);
        assertTopological(single, single.topologicalOrder());
        check(single.topologicalOrder()[0] == 0, "single node order is the node itself");

        DiGraph selfLoop = new DiGraph(1);
        selfLoop.addArc(0, 0);
        check(selfLoop.topologicalOrder() == null, "a self loop blocks topological sorting");
        check(selfLoop.blockedNodes().equals(List.of(0)), "the self loop node is reported blocked");
        List<Integer> selfCycle = selfLoop.minimalCycleIn(selfLoop.stronglyConnectedComponents().get(0));
        assertRealCycle(selfLoop, selfCycle);
        check(selfCycle.size() == 2, "the shortest self loop witness has two entries");
        check(selfLoop.exhaustiveMinFeedbackArcSet() == 1, "a self loop needs one arc removed");

        // --- rejected input (negative check) -------------------------------------
        boolean rejected = false;
        try {
            single.addArc(0, 3);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an arc to a non existing node is rejected");
        single.addArc(0, 0);
        single.addArc(0, 0);
        check(single.arcs.size() == 1, "duplicate arcs are collapsed");

        // --- acyclic instance ----------------------------------------------------
        DiGraph dag = new DiGraph(8);
        int[][] layers = {{0, 1}, {1, 2}, {1, 3}, {2, 4}, {3, 4}, {4, 5}, {5, 6}, {5, 7}, {0, 7}};
        for (int[] arc : layers) dag.addArc(arc[0], arc[1]);
        int[] dagOrder = dag.topologicalOrder();
        assertTopological(dag, dagOrder);
        check(dag.blockedNodes().isEmpty(), "an acyclic graph blocks nothing");
        for (List<Integer> component : dag.stronglyConnectedComponents()) {
            check(component.size() == 1, "an acyclic graph has singleton components");
        }
        check(dag.exhaustiveMinFeedbackArcSet() == 0, "an acyclic graph needs no arc removed");
        check(dag.feedbackArcs(dag.eadesLinSmythOrder()).isEmpty(), "ELS removes nothing on a DAG");

        // --- a plain cycle -------------------------------------------------------
        DiGraph cycle = new DiGraph(4);
        cycle.addArc(0, 1);
        cycle.addArc(1, 2);
        cycle.addArc(2, 3);
        cycle.addArc(3, 0);
        check(cycle.topologicalOrder() == null, "a cycle blocks topological sorting");
        check(cycle.blockedNodes().size() == 4, "every cycle node is blocked");
        List<Integer> witness = cycle.minimalCycleIn(cycle.stronglyConnectedComponents().get(0));
        assertRealCycle(cycle, witness);
        check(witness.size() == 5, "the four cycle witness is reported in full, got " + draw(witness));
        check(cycle.exhaustiveMinFeedbackArcSet() == 1, "one arc breaks the four cycle");
        check(cycle.feedbackArcs(cycle.eadesLinSmythOrder()).size() == 1, "ELS finds one arc here");

        // --- mixed instance: a DAG with two cycles and a chord -------------------
        DiGraph mixed = new DiGraph(7);
        int[][] mixedArcs = {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 5}, {5, 3}, {5, 6}, {0, 6}};
        for (int[] arc : mixedArcs) mixed.addArc(arc[0], arc[1]);
        check(mixed.topologicalOrder() == null, "the mixed instance is cyclic");
        List<List<Integer>> components = mixed.stronglyConnectedComponents();
        List<List<Integer>> oracle = mixed.sccByReachability();
        check(components.size() == oracle.size(), "SCC count matches the reachability oracle");
        for (int i = 0; i < components.size(); i++) {
            Set<Integer> a = new HashSet<>(components.get(i));
            Set<Integer> b = new HashSet<>(oracle.get(i));
            check(a.equals(b), "SCC " + i + " differs from the reachability oracle");
        }
        int nontrivial = 0;
        for (List<Integer> component : components) {
            if (component.size() == 1 && !mixed.adj[component.get(0)][component.get(0)]) continue;
            nontrivial++;
            List<Integer> witnessCycle = mixed.minimalCycleIn(component);
            assertRealCycle(mixed, witnessCycle);
            for (int v : witnessCycle) {
                check(component.contains(v), "the witness leaves its own SCC");
            }
        }
        check(nontrivial == 2, "the mixed instance has two cyclic components, got " + nontrivial);
        check(mixed.blockedNodes().size() >= 6, "blocked nodes cover both cycles");

        // --- heuristics versus exhaustive optimum --------------------------------
        int[][] graphSpecs = {
                {5, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0},
                {5, 0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 4, 1},
                {6, 0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3, 2, 5},
                {4, 0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2},
                {4, 0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3},
        };
        int worstGap = 0;
        for (int[] spec : graphSpecs) {
            DiGraph graph = new DiGraph(spec[0]);
            for (int i = 1; i + 1 < spec.length; i += 2) graph.addArc(spec[i], spec[i + 1]);
            int optimum = graph.exhaustiveMinFeedbackArcSet();
            int[] els = graph.eadesLinSmythOrder();
            int[] inMinusOut = graph.inMinusOutOrder();
            List<int[]> elsBack = graph.feedbackArcs(els);
            List<int[]> altBack = graph.feedbackArcs(inMinusOut);
            check(elsBack.size() >= optimum, "ELS cannot beat the exhaustive optimum");
            check(altBack.size() >= optimum, "the second heuristic cannot beat the optimum");
            check(graph.withoutArcs(elsBack).topologicalOrder() != null,
                    "removing the ELS feedback arcs must leave an acyclic graph");
            check(graph.withoutArcs(altBack).topologicalOrder() != null,
                    "removing the second heuristic's arcs must leave an acyclic graph");
            check(eachArcIsBackward(graph, els, elsBack), "ELS output arcs are exactly the backward ones");
            worstGap = Math.max(worstGap, Math.min(elsBack.size(), altBack.size()) - optimum);
        }
        check(worstGap <= 1, "the better heuristic stays within one arc of the optimum, gap " + worstGap);

        // --- randomised small graphs: invariants must hold -----------------------
        int tested = 0;
        int cyclic = 0;
        for (int trial = 0; trial < 40; trial++) {
            int n = 3 + nextInt(3);
            DiGraph graph = new DiGraph(n);
            int arcCount = nextInt(n * n + 1);
            for (int i = 0; i < arcCount; i++) graph.addArc(nextInt(n), nextInt(n));
            int[] order = graph.topologicalOrder();
            if (order != null) {
                assertTopological(graph, order);
                check(graph.exhaustiveMinFeedbackArcSet() == 0, "an acyclic graph has FAS 0");
            } else {
                cyclic++;
                List<List<Integer>> trialComponents = graph.stronglyConnectedComponents();
                check(samePartition(trialComponents, graph.sccByReachability()),
                        "SCC partition mismatch in trial " + trial);
                for (List<Integer> component : trialComponents) {
                    if (component.size() == 1 && !graph.adj[component.get(0)][component.get(0)]) continue;
                    List<Integer> witnessCycle = graph.minimalCycleIn(component);
                    assertRealCycle(graph, witnessCycle);
                }
                int optimum = graph.exhaustiveMinFeedbackArcSet();
                check(optimum > 0, "a cyclic graph needs at least one arc removed");
                List<int[]> elsBack = graph.feedbackArcs(graph.eadesLinSmythOrder());
                check(elsBack.size() >= optimum, "heuristic below the optimum in trial " + trial);
                check(graph.withoutArcs(elsBack).topologicalOrder() != null,
                        "heuristic removal left a cycle in trial " + trial);
            }
            tested++;
        }
        check(tested == 40, "every randomised trial ran");
        check(cyclic > 5, "enough cyclic instances were exercised, got " + cyclic);

        // --- determinism ----------------------------------------------------------
        DiGraph repeat = new DiGraph(7);
        for (int[] arc : mixedArcs) repeat.addArc(arc[0], arc[1]);
        check(Arrays.equals(mixed.eadesLinSmythOrder(), repeat.eadesLinSmythOrder()),
                "the heuristic order is deterministic");
        check(mixed.minimalCycleIn(mixed.stronglyConnectedComponents().get(0)).equals(
                        repeat.minimalCycleIn(repeat.stronglyConnectedComponents().get(0))),
                "the cycle witness is deterministic");

        System.out.println("mixed instance: blocked = " + mixed.blockedNodes().size() + ", cyclic SCCs = "
                + nontrivial + ", witness = " + draw(mixed.minimalCycleIn(
                mixed.stronglyConnectedComponents().get(0))));
        System.out.println("heuristics vs exhaustive optimum: worst gap = " + worstGap
                + " over " + graphSpecs.length + " graphs, " + cyclic + " random cyclic graphs, "
                + checksRun + " checks");
        System.out.println("PASS");
    }

    static boolean eachArcIsBackward(DiGraph graph, int[] order, List<int[]> back) {
        if (graph.feedbackArcs(order).size() != back.size()) return false;
        for (int[] arc : back) {
            boolean found = false;
            for (int[] other : graph.feedbackArcs(order)) {
                if (other[0] == arc[0] && other[1] == arc[1]) found = true;
            }
            if (!found) return false;
        }
        return true;
    }

    static boolean samePartition(List<List<Integer>> a, List<List<Integer>> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (!new HashSet<>(a.get(i)).equals(new HashSet<>(b.get(i)))) return false;
        }
        return true;
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
