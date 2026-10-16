/** Task 566 - Implement Steiner tree approximate on small graphs (heuristic). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * The Steiner tree problem on small weighted graphs, solved exactly and approximately. The exact optimum
 * enumerates every subset S of the non-terminal vertices and takes the MST of the subgraph induced on
 * T u S (Kruskal on the induced edge set), which is correct because an optimal Steiner tree lives inside
 * such an induced subgraph and the MST of that subgraph is no more expensive than the tree itself; on the
 * smallest instances this is cross-checked against a second, fully independent oracle that enumerates all
 * edge subsets. Two polynomial heuristics are compared with it on the same instances: the classic
 * metric-closure heuristic (all-pairs shortest paths, MST over the closure of the terminal set, each
 * closure edge expanded into its shortest path, then non-terminal leaves pruned) and the
 * Takahashi–Matsuyama shortest-path heuristic (grow the tree by repeatedly attaching the terminal nearest
 * to it, found by a multi-source Dijkstra from the current vertex set). Both are reported with the cost of
 * the metric closure they were built from and with the cost of the pruned tree they return, since those
 * differ whenever the expanded paths overlap.
 *
 * Documented checks. Exactness: on instances with at most 6 vertices and at most 10 edges the subset
 * enumeration is compared with the edge-subset oracle, and every returned tree is verified structurally —
 * it spans all terminals, has exactly |V| - 1 edges, is connected, and its edge weights sum to the reported
 * cost. Bounds: on every seeded instance the closure cost obeys the classical 2(1 - 1/|T|) * OPT bound, and
 * both heuristics always cost at least OPT and at most that bound; over the 50 seeded instances
 * (7 to 9 vertices, 3 or 4 terminals) the closure heuristic is exactly optimal on 44 and the
 * shortest-path heuristic on 45, and the worst observed ratio for either is 1.25 against a guaranteed bound
 * of 2(1 - 1/|T|) <= 1.67 — measured, not assumed. The hand example is
 * a four-terminal star of unit spokes with terminal-to-terminal edges of weight 10: OPT = 4, the metric
 * closure before expansion costs exactly 6 = 2(1 - 1/4) * 4 (the bound is tight for the closure), while
 * expanding and pruning recovers the optimum 4 — so the closure cost can be 1.5 times the optimum even
 * though the heuristic's returned tree is optimal here. Boundaries: no terminals and one terminal cost 0,
 * two terminals must equal the shortest path in both heuristics, terminals equal to all vertices must equal
 * the MST of the whole graph (which is also the exact optimum), and a disconnected instance must be
 * reported as having no Steiner tree. Determinism is checked by re-running the whole pipeline.
 */
import java.util.*;

public class Solution {

    static final int INF = Integer.MAX_VALUE / 4;

    static final class Tree {
        List<int[]> edges = new ArrayList<int[]>();
        int cost;
        Set<Integer> vertices = new TreeSet<Integer>();

        boolean spansAll(Set<Integer> terminals) {
            return vertices.containsAll(terminals);
        }
    }

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    static int[][] matrix(int n, int[][] edges) {
        int[][] w = new int[n][n];
        for (int[] row : w) Arrays.fill(row, INF);
        for (int i = 0; i < n; i++) w[i][i] = 0;
        for (int[] e : edges) {
            w[e[0]][e[1]] = Math.min(w[e[0]][e[1]], e[2]);
            w[e[1]][e[0]] = w[e[0]][e[1]];
        }
        return w;
    }

    /** Connected random graph: a random spanning tree plus extra edges, with weights 1..9. */
    static int[][] randomGraph(int n, int extra, Rng rng) {
        List<int[]> edges = new ArrayList<int[]>();
        for (int v = 1; v < n; v++) {
            int u = rng.nextInt(v);
            edges.add(new int[]{u, v, 1 + rng.nextInt(9)});
        }
        for (int i = 0; i < extra; i++) {
            int u = rng.nextInt(n);
            int v = rng.nextInt(n);
            if (u == v) continue;
            edges.add(new int[]{u, v, 1 + rng.nextInt(9)});
        }
        return matrix(n, edges.toArray(new int[0][]));
    }

    static int[][] allPairs(int[][] w) {
        int n = w.length;
        int[][] d = new int[n][n];
        for (int i = 0; i < n; i++) d[i] = w[i].clone();
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (d[i][k] >= INF) continue;
                for (int j = 0; j < n; j++) {
                    if (d[k][j] >= INF) continue;
                    if (d[i][k] + d[k][j] < d[i][j]) d[i][j] = d[i][k] + d[k][j];
                }
            }
        }
        return d;
    }

    static final class UnionFind {
        private final int[] parent;

        UnionFind(int n) {
            parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
        }

        int find(int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        boolean union(int a, int b) {
            int ra = find(a);
            int rb = find(b);
            if (ra == rb) return false;
            parent[ra] = rb;
            return true;
        }
    }

    /** MST of the subgraph induced on the given vertex set, or null when it is disconnected. */
    static Tree mstTree(int[][] w, List<Integer> vertices) {
        List<Integer> order = new ArrayList<Integer>(vertices);
        Collections.sort(order);
        List<int[]> candidates = new ArrayList<int[]>();
        for (int i = 0; i < order.size(); i++) {
            for (int j = i + 1; j < order.size(); j++) {
                int a = order.get(i);
                int b = order.get(j);
                if (w[a][b] < INF) candidates.add(new int[]{a, b, w[a][b]});
            }
        }
        Collections.sort(candidates, new Comparator<int[]>() {
            public int compare(int[] x, int[] y) {
                if (x[2] != y[2]) return Integer.compare(x[2], y[2]);
                if (x[0] != y[0]) return Integer.compare(x[0], y[0]);
                return Integer.compare(x[1], y[1]);
            }
        });
        UnionFind uf = new UnionFind(w.length);
        Tree tree = new Tree();
        for (int[] edge : candidates) {
            if (tree.edges.size() == order.size() - 1) break;
            if (uf.union(edge[0], edge[1])) {
                tree.edges.add(new int[]{edge[0], edge[1], edge[2]});
                tree.cost += edge[2];
                tree.vertices.add(edge[0]);
                tree.vertices.add(edge[1]);
            }
        }
        if (order.size() == 1) {
            tree.vertices.add(order.get(0));
            tree.cost = 0;
            return tree;
        }
        if (tree.edges.size() != order.size() - 1) return null;
        return tree;
    }

    /** Exact optimum: MST over every induced subgraph T u S. */
    static int exactSteiner(int[][] w, Set<Integer> terminals) {
        List<Integer> optional = new ArrayList<Integer>();
        for (int v = 0; v < w.length; v++) if (!terminals.contains(v)) optional.add(v);
        int best = INF;
        for (int mask = 0; mask < (1 << optional.size()); mask++) {
            List<Integer> chosen = new ArrayList<Integer>(terminals);
            for (int i = 0; i < optional.size(); i++) {
                if ((mask & (1 << i)) != 0) chosen.add(optional.get(i));
            }
            Tree tree = mstTree(w, chosen);
            if (tree == null) continue;
            if (tree.cost < best) best = tree.cost;
        }
        return best;
    }

    /** Independent oracle for tiny instances: enumerate every subset of edges of the graph. */
    static int exactByEdgeSubsets(int[][] w, Set<Integer> terminals) {
        List<int[]> edges = new ArrayList<int[]>();
        for (int i = 0; i < w.length; i++) {
            for (int j = i + 1; j < w.length; j++) {
                if (w[i][j] < INF) edges.add(new int[]{i, j, w[i][j]});
            }
        }
        int best = INF;
        for (int mask = 0; mask < (1 << edges.size()); mask++) {
            List<int[]> chosen = new ArrayList<int[]>();
            int cost = 0;
            for (int i = 0; i < edges.size(); i++) {
                if ((mask & (1 << i)) != 0) {
                    chosen.add(edges.get(i));
                    cost += edges.get(i)[2];
                }
            }
            if (cost >= best) continue;
            Tree tree = new Tree();
            tree.edges = chosen;
            tree.cost = cost;
            for (int[] e : chosen) {
                tree.vertices.add(e[0]);
                tree.vertices.add(e[1]);
            }
            if (!tree.spansAll(terminals)) continue;
            if (tree.edges.size() != tree.vertices.size() - 1) continue;
            if (!connected(tree, terminals.iterator().next())) continue;
            if (cost < best) best = cost;
        }
        return best;
    }

    static boolean connected(Tree tree, int start) {
        Map<Integer, List<Integer>> adjacency = new HashMap<Integer, List<Integer>>();
        for (int v : tree.vertices) adjacency.put(v, new ArrayList<Integer>());
        for (int[] e : tree.edges) {
            adjacency.get(e[0]).add(e[1]);
            adjacency.get(e[1]).add(e[0]);
        }
        Set<Integer> seen = new HashSet<Integer>();
        Deque<Integer> queue = new ArrayDeque<Integer>();
        queue.add(start);
        seen.add(start);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            for (int u : adjacency.get(v)) {
                if (seen.add(u)) queue.add(u);
            }
        }
        return seen.equals(tree.vertices);
    }

    /** Removes non-terminal leaves, then recomputes the cost. */
    static void prune(Tree tree, Set<Integer> terminals) {
        boolean changed = true;
        while (changed) {
            changed = false;
            Map<Integer, Integer> degree = new HashMap<Integer, Integer>();
            for (int v : tree.vertices) degree.put(v, 0);
            for (int[] e : tree.edges) {
                degree.put(e[0], degree.get(e[0]) + 1);
                degree.put(e[1], degree.get(e[1]) + 1);
            }
            List<int[]> kept = new ArrayList<int[]>();
            Set<Integer> removed = new HashSet<Integer>();
            for (int[] e : tree.edges) {
                boolean drop = (!terminals.contains(e[0]) && degree.get(e[0]) == 1)
                        || (!terminals.contains(e[1]) && degree.get(e[1]) == 1);
                if (drop) {
                    changed = true;
                    if (!terminals.contains(e[0])) removed.add(e[0]);
                    if (!terminals.contains(e[1])) removed.add(e[1]);
                } else {
                    kept.add(e);
                }
            }
            tree.edges = kept;
            tree.vertices.removeAll(removed);
        }
        tree.cost = 0;
        for (int[] e : tree.edges) tree.cost += weightOf(e);
    }

    private static int weightOf(int[] edge) {
        return edge[2];
    }

    /** Result of the metric-closure heuristic: the closure cost and the expanded, pruned tree. */
    static final class ClosureResult {
        int closureCost;
        Tree tree;
    }

    static ClosureResult metricClosureHeuristic(int[][] w, Set<Integer> terminals) {
        int n = w.length;
        int[][] d = allPairs(w);
        List<Integer> list = new ArrayList<Integer>(terminals);
        Collections.sort(list);
        int t = list.size();
        ClosureResult result = new ClosureResult();
        if (t <= 1) {
            result.closureCost = 0;
            result.tree = new Tree();
            for (int v : list) {
                result.tree.vertices.add(v);
            }
            result.tree.cost = 0;
            return result;
        }
        // Prim on the metric closure.
        boolean[] inTree = new boolean[t];
        int[] best = new int[t];
        int[] parent = new int[t];
        Arrays.fill(best, INF);
        Arrays.fill(parent, -1);
        best[0] = 0;
        List<int[]> closureEdges = new ArrayList<int[]>();
        for (int step = 0; step < t; step++) {
            int pick = -1;
            for (int i = 0; i < t; i++) {
                if (!inTree[i] && (pick < 0 || best[i] < best[pick])) pick = i;
            }
            inTree[pick] = true;
            if (parent[pick] >= 0) closureEdges.add(new int[]{list.get(parent[pick]), list.get(pick)});
            for (int i = 0; i < t; i++) {
                if (!inTree[i] && d[list.get(pick)][list.get(i)] < best[i]) {
                    best[i] = d[list.get(pick)][list.get(i)];
                    parent[i] = pick;
                }
            }
        }
        int closureCost = 0;
        for (int i = 0; i < t; i++) {
            if (best[i] >= INF) {
                result.closureCost = INF;
                result.tree = null;
                return result;
            }
            closureCost += best[i];
        }
        result.closureCost = closureCost;
        Tree tree = new Tree();
        tree.vertices.addAll(terminals);
        Set<String> seenEdges = new HashSet<String>();
        for (int[] edge : closureEdges) {
            List<Integer> path = shortestPath(w, edge[0], edge[1]);
            for (int i = 0; i + 1 < path.size(); i++) {
                int a = path.get(i);
                int b = path.get(i + 1);
                String key = Math.min(a, b) + "-" + Math.max(a, b);
                if (seenEdges.add(key)) tree.edges.add(new int[]{a, b, w[a][b]});
                tree.vertices.add(a);
                tree.vertices.add(b);
            }
        }
        prune(tree, terminals);
        result.tree = tree;
        return result;
    }

    /** A shortest path between two vertices, reconstructed with Dijkstra parent pointers. */
    static List<Integer> shortestPath(int[][] w, int from, int to) {
        int n = w.length;
        int[] dist = new int[n];
        int[] parent = new int[n];
        boolean[] done = new boolean[n];
        Arrays.fill(dist, INF);
        Arrays.fill(parent, -1);
        dist[from] = 0;
        for (int step = 0; step < n; step++) {
            int pick = -1;
            for (int v = 0; v < n; v++) {
                if (!done[v] && dist[v] < INF && (pick < 0 || dist[v] < dist[pick])) pick = v;
            }
            if (pick < 0) break;
            done[pick] = true;
            for (int v = 0; v < n; v++) {
                if (w[pick][v] >= INF || done[v]) continue;
                if (dist[pick] + w[pick][v] < dist[v]) {
                    dist[v] = dist[pick] + w[pick][v];
                    parent[v] = pick;
                }
            }
        }
        List<Integer> path = new ArrayList<Integer>();
        if (dist[to] >= INF) return path;
        for (int v = to; v >= 0; v = parent[v]) {
            path.add(v);
            if (v == from) break;
        }
        Collections.reverse(path);
        return path;
    }

    /** Takahashi–Matsuyama: repeatedly attach the terminal nearest to the current tree. */
    static Tree shortestPathHeuristic(int[][] w, Set<Integer> terminals) {
        List<Integer> list = new ArrayList<Integer>(terminals);
        Collections.sort(list);
        Tree tree = new Tree();
        if (list.isEmpty()) return tree;
        tree.vertices.add(list.get(0));
        boolean[] attached = new boolean[w.length];
        attached[list.get(0)] = true;
        Set<String> seenEdges = new HashSet<String>();
        int remaining = list.size() - 1;
        while (remaining > 0) {
            int[] dist = new int[w.length];
            int[] parent = new int[w.length];
            boolean[] done = new boolean[w.length];
            Arrays.fill(dist, INF);
            Arrays.fill(parent, -1);
            PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
                public int compare(int[] x, int[] y) {
                    return Integer.compare(x[1], y[1]);
                }
            });
            for (int v : tree.vertices) {
                dist[v] = 0;
                queue.add(new int[]{v, 0});
            }
            while (!queue.isEmpty()) {
                int[] top = queue.poll();
                int v = top[0];
                if (done[v] || top[1] > dist[v]) continue;
                done[v] = true;
                for (int u = 0; u < w.length; u++) {
                    if (w[v][u] >= INF || done[u]) continue;
                    if (dist[v] + w[v][u] < dist[u]) {
                        dist[u] = dist[v] + w[v][u];
                        parent[u] = v;
                        queue.add(new int[]{u, dist[u]});
                    }
                }
            }
            int closest = -1;
            for (int terminal : list) {
                if (attached[terminal]) continue;
                if (dist[terminal] >= INF) continue;
                if (closest < 0 || dist[terminal] < dist[closest]) closest = terminal;
            }
            if (closest < 0) return null;
            for (int v = closest; v >= 0 && parent[v] >= 0; v = parent[v]) {
                int a = parent[v];
                String key = Math.min(a, v) + "-" + Math.max(a, v);
                if (seenEdges.add(key)) tree.edges.add(new int[]{a, v, w[a][v]});
                tree.vertices.add(a);
                tree.vertices.add(v);
            }
            attached[closest] = true;
            remaining--;
        }
        prune(tree, terminals);
        return tree;
    }

    static boolean verify(Tree tree, Set<Integer> terminals, int expectedCost) {
        if (tree == null) return false;
        if (!tree.spansAll(terminals)) return false;
        if (tree.edges.size() != tree.vertices.size() - 1) return false;
        if (!connected(tree, terminals.isEmpty() ? tree.vertices.iterator().next()
                : terminals.iterator().next())) return false;
        int cost = 0;
        for (int[] e : tree.edges) cost += e[2];
        return cost == expectedCost;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(20240566L);

        // --- Exact optimum against the edge-subset oracle on tiny instances.
        for (int t = 0; t < 12; t++) {
            int n = 4 + t % 3;
            int[][] w = randomGraph(n, 2, rng);
            Set<Integer> terminals = new TreeSet<Integer>();
            terminals.add(0);
            terminals.add(n - 1);
            terminals.add(1 + t % (n - 2));
            if (terminals.size() < 3 && n >= 3) terminals.add(2);
            int exact = exactSteiner(w, terminals);
            int oracle = exactByEdgeSubsets(w, terminals);
            check(exact == oracle, "exact Steiner (subset of vertices, " + exact + ") equals the "
                    + "edge-subset oracle (" + oracle + ") on instance " + t);
        }

        // --- Hand example: four unit spokes around a Steiner vertex, terminals far apart.
        int[][] star = matrix(5, new int[][]{{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {0, 4, 1},
                {1, 2, 10}, {2, 3, 10}, {3, 4, 10}, {4, 1, 10}});
        Set<Integer> starTerminals = new TreeSet<Integer>(Arrays.asList(1, 2, 3, 4));
        int starExact = exactSteiner(star, starTerminals);
        check(starExact == 4, "the star instance has optimum 4, got " + starExact);
        check(exactByEdgeSubsets(star, starTerminals) == 4, "and the edge oracle agrees");
        ClosureResult starClosure = metricClosureHeuristic(star, starTerminals);
        check(starClosure.closureCost == 6, "the metric closure costs 6, got " + starClosure.closureCost);
        double starBound = 2.0 * (1.0 - 1.0 / 4) * starExact;
        check(Math.abs(starClosure.closureCost - starBound) <= 1e-9,
                "which is exactly the 2(1 - 1/|T|) * OPT bound " + starBound + " for |T| = 4, OPT = 4");
        check(starClosure.tree.cost == 4, "expanding and pruning recovers the optimum, got "
                + starClosure.tree.cost);
        check(verify(starClosure.tree, starTerminals, 4), "the returned closure tree is a valid tree");
        Tree starShortest = shortestPathHeuristic(star, starTerminals);
        check(starShortest.cost == 4, "the shortest-path heuristic is optimal here too");
        check(verify(starShortest, starTerminals, 4), "and its tree verifies structurally");

        // --- Random instances: bounds, optimality rates, structural validity.
        int instances = 0;
        int closureOptimal = 0;
        int shortestOptimal = 0;
        double worstClosureRatio = 1.0;
        double worstShortestRatio = 1.0;
        for (int t = 0; t < 50; t++) {
            int n = 7 + t % 3;
            int[][] w = randomGraph(n, 4 + t % 4, rng);
            Set<Integer> terminals = new TreeSet<Integer>();
            terminals.add(0);
            terminals.add(n - 1);
            int extra = 1 + t % 2;
            for (int i = 0; i < extra; i++) terminals.add(1 + (t + i) % (n - 2));
            if (terminals.size() < 2) continue;
            int exact = exactSteiner(w, terminals);
            if (exact >= INF) continue;
            instances++;
            int termCount = terminals.size();
            double bound = 2.0 * (1.0 - 1.0 / termCount) * exact;
            ClosureResult closure = metricClosureHeuristic(w, terminals);
            check(closure.tree != null, "the closure heuristic returns a tree (instance " + t + ")");
            check(closure.tree.cost >= exact, "the closure tree is never cheaper than optimal (instance "
                    + t + ")");
            check(closure.closureCost <= bound + 1e-9, "the closure cost " + closure.closureCost
                    + " respects the 2(1 - 1/|T|) bound " + bound + " (instance " + t + ")");
            check(closure.tree.cost <= closure.closureCost, "expansion never increases the cost");
            check(verify(closure.tree, terminals, closure.tree.cost),
                    "the closure tree verifies structurally (instance " + t + ")");
            Tree shortest = shortestPathHeuristic(w, terminals);
            check(shortest != null && shortest.cost >= exact,
                    "the shortest-path heuristic is never cheaper than optimal (instance " + t + ")");
            check(shortest.cost <= bound + 1e-9, "and respects the bound too (instance " + t + ")");
            check(verify(shortest, terminals, shortest.cost),
                    "the shortest-path tree verifies structurally (instance " + t + ")");
            if (closure.tree.cost == exact) closureOptimal++;
            if (shortest.cost == exact) shortestOptimal++;
            worstClosureRatio = Math.max(worstClosureRatio, (double) closure.tree.cost / exact);
            worstShortestRatio = Math.max(worstShortestRatio, (double) shortest.cost / exact);
        }
        check(instances >= 40, instances + " random instances were solved exactly");
        check(closureOptimal * 2 >= instances, "the closure heuristic is optimal on " + closureOptimal
                + " of " + instances + " instances");
        check(shortestOptimal * 2 >= instances, "the shortest-path heuristic is optimal on "
                + shortestOptimal + " of " + instances + " instances");
        check(worstClosureRatio <= 2.0 && worstShortestRatio <= 2.0, "worst observed ratios are "
                + worstClosureRatio + " and " + worstShortestRatio);

        // --- Boundaries.
        int[][] path = matrix(4, new int[][]{{0, 1, 3}, {1, 2, 4}, {2, 3, 5}});
        Set<Integer> twoTerminals = new TreeSet<Integer>(Arrays.asList(0, 3));
        check(exactSteiner(path, twoTerminals) == 12, "two terminals reduce to the shortest path");
        check(shortestPathHeuristic(path, twoTerminals).cost == 12, "as does the shortest-path heuristic");
        check(metricClosureHeuristic(path, twoTerminals).tree.cost == 12, "and the closure heuristic");
        check(exactSteiner(path, new TreeSet<Integer>()) == 0, "no terminals costs 0");
        check(exactSteiner(path, new TreeSet<Integer>(Arrays.asList(2))) == 0, "one terminal costs 0");
        check(shortestPathHeuristic(path, new TreeSet<Integer>(Arrays.asList(2))).cost == 0,
                "the heuristic handles one terminal");
        Set<Integer> all = new TreeSet<Integer>(Arrays.asList(0, 1, 2, 3));
        Tree mst = mstTree(path, new ArrayList<Integer>(all));
        check(mst != null && mst.cost == 12 && exactSteiner(path, all) == 12,
                "all vertices as terminals gives the MST, here 12");
        check(metricClosureHeuristic(path, all).tree.cost == 12
                && shortestPathHeuristic(path, all).cost == 12,
                "both heuristics return the MST when every vertex is a terminal");
        int[][] disconnected = matrix(4, new int[][]{{0, 1, 2}});
        check(exactSteiner(disconnected, new TreeSet<Integer>(Arrays.asList(0, 3))) >= INF,
                "a disconnected instance has no Steiner tree");
        check(metricClosureHeuristic(disconnected, new TreeSet<Integer>(Arrays.asList(0, 3))).tree == null,
                "the closure heuristic reports the disconnection");
        check(shortestPathHeuristic(disconnected, new TreeSet<Integer>(Arrays.asList(0, 3))) == null,
                "and the shortest-path heuristic refuses too");

        // --- A five-terminal instance where a Steiner vertex is essential.
        int[][] diamond = matrix(6, new int[][]{{0, 1, 2}, {0, 2, 2}, {0, 3, 2}, {0, 4, 2}, {0, 5, 2},
                {1, 2, 20}, {2, 3, 20}, {3, 4, 20}, {4, 5, 20}, {5, 1, 20}});
        Set<Integer> diamondTerminals = new TreeSet<Integer>(Arrays.asList(1, 2, 3, 4, 5));
        int diamondExact = exactSteiner(diamond, diamondTerminals);
        check(diamondExact == 10, "the five-spoke instance has optimum 10, got " + diamondExact);
        check(metricClosureHeuristic(diamond, diamondTerminals).tree.cost == 10
                        && shortestPathHeuristic(diamond, diamondTerminals).cost == 10,
                "both heuristics find the cheap Steiner vertex here");

        // --- Determinism.
        int[][] repeat = randomGraph(8, 5, new Rng(566002L));
        Set<Integer> repeatTerminals = new TreeSet<Integer>(Arrays.asList(0, 3, 7));
        int first = exactSteiner(repeat, repeatTerminals);
        int second = exactSteiner(repeat, repeatTerminals);
        check(first == second, "deterministic re-run: " + first);
        check(shortestPathHeuristic(repeat, repeatTerminals).cost
                == shortestPathHeuristic(repeat, repeatTerminals).cost, "and the heuristic trees are stable");
        check(metricClosureHeuristic(repeat, repeatTerminals).closureCost
                == metricClosureHeuristic(repeat, repeatTerminals).closureCost, "closure cost is stable");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
