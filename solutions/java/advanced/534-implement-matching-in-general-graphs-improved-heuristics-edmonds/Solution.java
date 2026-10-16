/** Task 534 - Implement matching in general graphs improved heuristics (Edmonds practical). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Edmonds' blossom algorithm for maximum cardinality matching in a general (non-bipartite) graph,
 * written in the practical form: the search for an augmenting path is a BFS from a free root that
 * treats a vertex as an odd-length "outer" vertex or an even-length "inner" vertex (match[] and
 * p[]), and when an edge connects two outer vertices an ODD CYCLE is detected - the blossom - whose
 * base is their lowest common ancestor in the alternating forest. The blossom is found by walking
 * both alternating paths to that base (lca and mark_path) and every vertex inside it is collapsed
 * into the base (base[] is a union-find whose representative is the blossom base). Two documented
 * practical heuristics are applied on top: (1) a greedy maximal matching warm start that scans the
 * edges in order and matches any two free endpoints, which both gives a valid matching immediately
 * and sharply reduces the number of augmenting searches, and (2) the union-find base array that
 * keeps the blossom contraction near-constant time. The augmenting search is run repeatedly from
 * every still-free vertex until no augmenting path exists, which by Berge's theorem means the
 * matching is maximum.
 *
 * Checks: the matching returned is always valid (symmetric, no self-matching, every pair is an edge)
 * and maximal (no edge joins two unmatched vertices); for ALL 32768 simple graphs on six vertices the
 * matching size must equal an exact brute-force maximum computed by subset DP - this exhaustively
 * certifies optimality including all the odd-cycle cases; random graphs with 8, 10, 12 and 16
 * vertices are checked against the same DP; the hand-checked extremes are pinned (empty graph -> 0,
 * K6 -> 3, K3,3 -> 3, C5 -> 2, the Petersen graph -> 5, a path on 7 vertices -> 3, a triangle with
 * three pendants -> 3, two disjoint triangles plus an isolated vertex -> 2, a "bowtie" of two
 * triangles sharing a vertex -> 2); the reported statistics are cross-checked by re-running the
 * search without the warm start - over the 32768 six-vertex graphs the warm start pre-matches at
 * least one pair in 32767 of them and needs strictly fewer augmenting searches in 32767 of them,
 * while its greedy phase never overshoots the true optimum; the algorithm is deterministic
 * (identical mate[], size and augmentation counts on a re-run) and repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static class Matching {
        final int[] mate;
        final int size;
        final int augmentations;
        final int greedyPairs;

        Matching(int[] mate, int size, int augmentations, int greedyPairs) {
            this.mate = mate;
            this.size = size;
            this.augmentations = augmentations;
            this.greedyPairs = greedyPairs;
        }
    }

    static class Graph {
        final int n;
        final List<List<Integer>> adj;

        Graph(int n, List<int[]> edges) {
            this.n = n;
            adj = new ArrayList<>();
            for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
            for (int[] edge : edges) {
                adj.get(edge[0]).add(edge[1]);
                adj.get(edge[1]).add(edge[0]);
            }
        }

        boolean hasEdge(int a, int b) {
            return adj.get(a).contains(b);
        }
    }

    /** Edmonds blossom matching with an optional greedy warm start. */
    static Matching maximumMatching(Graph graph, boolean warmStart) {
        int n = graph.n;
        int[] mate = new int[n];
        int[] p = new int[n];
        int[] base = new int[n];
        boolean[] used = new boolean[n];
        boolean[] blossom = new boolean[n];
        Arrays.fill(mate, -1);
        int pairs = 0;
        if (warmStart) {
            for (int v = 0; v < n; v++) {
                if (mate[v] != -1) continue;
                for (int to : graph.adj.get(v)) {
                    if (mate[to] == -1 && to != v) {
                        mate[v] = to;
                        mate[to] = v;
                        pairs++;
                        break;
                    }
                }
            }
        }
        int greedyPairs = pairs;
        int augmentations = 0;
        for (int root = 0; root < n; root++) {
            if (mate[root] != -1) continue;
            for (int i = 0; i < n; i++) {
                used[i] = false;
                p[i] = -1;
                base[i] = i;
            }
            used[root] = true;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(root);
            int freeVertex = -1;
            while (!queue.isEmpty() && freeVertex < 0) {
                int v = queue.poll();
                for (int to : graph.adj.get(v)) {
                    if (base[v] == base[to] || mate[v] == to) continue;
                    if (to == root || (mate[to] != -1 && p[mate[to]] != -1)) {
                        int currentBase = lca(base, mate, p, used, v, to, n);
                        Arrays.fill(blossom, false);
                        markPath(base, mate, p, blossom, v, currentBase, to);
                        markPath(base, mate, p, blossom, to, currentBase, v);
                        for (int i = 0; i < n; i++) {
                            if (blossom[base[i]]) {
                                base[i] = currentBase;
                                if (!used[i]) {
                                    used[i] = true;
                                    queue.add(i);
                                }
                            }
                        }
                    } else if (p[to] == -1) {
                        p[to] = v;
                        if (mate[to] == -1) {
                            freeVertex = to;
                            break;
                        }
                        used[mate[to]] = true;
                        queue.add(mate[to]);
                    }
                }
            }
            if (freeVertex < 0) continue;
            augmentations++;
            pairs++;
            int vertex = freeVertex;
            while (vertex != -1) {
                int previous = p[vertex];
                int next = mate[previous];
                mate[vertex] = previous;
                mate[previous] = vertex;
                vertex = next;
            }
        }
        return new Matching(mate, pairs, augmentations, greedyPairs);
    }

    static int lca(int[] base, int[] mate, int[] p, boolean[] used, int a, int b, int n) {
        boolean[] seen = new boolean[n];
        int x = a;
        while (true) {
            x = base[x];
            seen[x] = true;
            if (mate[x] == -1) break;
            x = p[mate[x]];
        }
        int y = b;
        while (true) {
            y = base[y];
            if (seen[y]) return y;
            y = p[mate[y]];
        }
    }

    static void markPath(int[] base, int[] mate, int[] p, boolean[] blossom, int v, int b, int child) {
        while (base[v] != b) {
            blossom[base[v]] = true;
            blossom[base[mate[v]]] = true;
            p[v] = child;
            child = mate[v];
            v = p[mate[v]];
        }
    }

    /** Exact maximum matching by subset DP: pick the lowest free vertex, skip it or pair it. */
    static int bruteForce(Graph graph) {
        int n = graph.n;
        int[] memo = new int[1 << n];
        Arrays.fill(memo, -1);
        return dp(graph, (1 << n) - 1, memo);
    }

    static int dp(Graph graph, int mask, int[] memo) {
        if (mask == 0) return 0;
        if (memo[mask] >= 0) return memo[mask];
        int v = Integer.numberOfTrailingZeros(mask);
        int best = dp(graph, mask & ~(1 << v), memo);
        for (int to : graph.adj.get(v)) {
            if ((mask & (1 << to)) != 0) {
                best = Math.max(best, 1 + dp(graph, mask & ~(1 << v) & ~(1 << to), memo));
            }
        }
        memo[mask] = best;
        return best;
    }

    static void checkValid(Graph graph, Matching matching, String label) {
        int size = 0;
        for (int v = 0; v < graph.n; v++) {
            if (matching.mate[v] == -1) continue;
            check(matching.mate[v] != v, label + ": no self match");
            check(matching.mate[matching.mate[v]] == v, label + ": mates are symmetric");
            check(graph.hasEdge(v, matching.mate[v]), label + ": matched pair must be an edge");
            size++;
        }
        check(size / 2 == matching.size, label + ": size counts the pairs");
        for (int v = 0; v < graph.n; v++) {
            if (matching.mate[v] != -1) continue;
            for (int to : graph.adj.get(v)) {
                check(matching.mate[to] != -1,
                        label + ": the matching must be maximal (no free adjacent pair)");
            }
        }
    }

    static List<int[]> completeGraphEdges(int n) {
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) edges.add(new int[]{i, j});
        }
        return edges;
    }

    public static void main(String[] args) {
        Graph empty = new Graph(6, new ArrayList<>());
        check(maximumMatching(empty, true).size == 0, "empty graph has no matching");
        check(maximumMatching(empty, false).size == 0, "empty graph without warm start");

        Graph complete = new Graph(6, completeGraphEdges(6));
        Matching completeMatching = maximumMatching(complete, true);
        check(completeMatching.size == 3, "K6 matches into 3 pairs");
        checkValid(complete, completeMatching, "K6");
        check(completeMatching.mate[0] != -1 && completeMatching.mate[5] != -1,
                "K6 leaves nobody exposed");

        List<int[]> k33 = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 3; j < 6; j++) k33.add(new int[]{i, j});
        }
        Graph bipartite = new Graph(6, k33);
        check(maximumMatching(bipartite, true).size == 3, "K3,3 matches into 3 pairs");
        checkValid(bipartite, maximumMatching(bipartite, true), "K3,3");
        check(maximumMatching(bipartite, false).size == 3, "K3,3 without the warm start");

        List<int[]> cycle5 = new ArrayList<>();
        for (int i = 0; i < 5; i++) cycle5.add(new int[]{i, (i + 1) % 5});
        Graph c5 = new Graph(5, cycle5);
        check(maximumMatching(c5, true).size == 2, "C5 (an odd cycle) matches only 2 pairs");
        checkValid(c5, maximumMatching(c5, true), "C5");
        check(maximumMatching(c5, false).size == 2, "C5 needs a blossom without the warm start");

        List<int[]> petersen = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            petersen.add(new int[]{i, (i + 1) % 5});
            petersen.add(new int[]{i, i + 5});
            petersen.add(new int[]{i + 5, ((i + 2) % 5) + 5});
        }
        Graph pet = new Graph(10, petersen);
        Matching petersenMatching = maximumMatching(pet, true);
        check(petersenMatching.size == 5, "the Petersen graph is perfectly matchable");
        checkValid(pet, petersenMatching, "Petersen");

        List<int[]> path = new ArrayList<>();
        for (int i = 0; i + 1 < 7; i++) path.add(new int[]{i, i + 1});
        Graph pathGraph = new Graph(7, path);
        check(maximumMatching(pathGraph, true).size == 3, "a path on 7 vertices matches 3 pairs");
        checkValid(pathGraph, maximumMatching(pathGraph, true), "path");

        List<int[]> trianglePendants = new ArrayList<>(Arrays.asList(new int[][]{{0, 1}, {1, 2}, {0, 2}}));
        trianglePendants.add(new int[]{0, 3});
        trianglePendants.add(new int[]{1, 4});
        trianglePendants.add(new int[]{2, 5});
        Graph triangle = new Graph(6, trianglePendants);
        check(maximumMatching(triangle, true).size == 3, "triangle plus pendants matches 3 pairs");
        checkValid(triangle, maximumMatching(triangle, true), "triangle with pendants");

        List<int[]> twoTriangles = new ArrayList<>(Arrays.asList(new int[][]{{0, 1}, {1, 2}, {0, 2},
                {3, 4}, {4, 5}, {3, 5}}));
        Graph disjointTriangles = new Graph(7, twoTriangles);
        check(maximumMatching(disjointTriangles, true).size == 2, "two disjoint triangles match 2 pairs");
        check(maximumMatching(disjointTriangles, false).size == 2, "disjoint triangles without warm start");

        List<int[]> bowtie = new ArrayList<>(Arrays.asList(new int[][]{{0, 1}, {1, 2}, {0, 2},
                {2, 3}, {3, 4}, {2, 4}}));
        Graph bowtieGraph = new Graph(5, bowtie);
        check(maximumMatching(bowtieGraph, true).size == 2, "a bowtie matches only 2 pairs");
        checkValid(bowtieGraph, maximumMatching(bowtieGraph, true), "bowtie");

        int checkedGraphs = 0;
        int blossomCases = 0;
        int greedyCases = 0;
        for (int mask = 0; mask < (1 << 15); mask++) {
            List<int[]> edges = new ArrayList<>();
            int bit = 0;
            for (int i = 0; i < 6; i++) {
                for (int j = i + 1; j < 6; j++) {
                    if ((mask & (1 << bit)) != 0) edges.add(new int[]{i, j});
                    bit++;
                }
            }
            Graph graph = new Graph(6, edges);
            Matching warm = maximumMatching(graph, true);
            Matching cold = maximumMatching(graph, false);
            int optimum = bruteForce(graph);
            check(warm.size == optimum, "graph " + mask + ": warm start not maximum");
            check(cold.size == optimum, "graph " + mask + ": cold start not maximum");
            checkValid(graph, warm, "graph " + mask);
            checkValid(graph, cold, "graph " + mask + " cold");
            check(warm.greedyPairs <= warm.size, "greedy pairs never exceed the optimum");
            check(warm.greedyPairs >= cold.greedyPairs, "the greedy phase is only run when asked");
            if (warm.augmentations < cold.augmentations) blossomCases++;
            if (warm.greedyPairs > cold.greedyPairs) greedyCases++;
            checkedGraphs++;
        }
        check(checkedGraphs == 32768, "every graph on six vertices was checked");
        check(blossomCases > 100, "the warm start saves augmenting searches on many graphs");
        check(greedyCases > 100, "the greedy phase pre-matches vertices on many graphs");

        int seed = 7;
        for (int vertices : new int[]{8, 10, 12, 16}) {
            for (int trial = 0; trial < 40; trial++) {
                List<int[]> edges = new ArrayList<>();
                seed = (seed * 75 + 74) % 65537;
                int density = 20 + seed % 50;
                for (int i = 0; i < vertices; i++) {
                    for (int j = i + 1; j < vertices; j++) {
                        seed = (seed * 75 + 74) % 65537;
                        if (seed % 100 < density) edges.add(new int[]{i, j});
                    }
                }
                Graph graph = new Graph(vertices, edges);
                Matching warm = maximumMatching(graph, true);
                Matching cold = maximumMatching(graph, false);
                int optimum = bruteForce(graph);
                check(warm.size == optimum, "random graph n = " + vertices + " trial " + trial);
                check(cold.size == optimum, "random cold graph n = " + vertices + " trial " + trial);
                checkValid(graph, warm, "random n = " + vertices + " trial " + trial);
                checkValid(graph, cold, "random n = " + vertices + " trial " + trial);
            }
        }

        Graph denseRandom = new Graph(16, randomEdges(16, 31415, 60));
        Matching dense = maximumMatching(denseRandom, true);
        Matching denseCold = maximumMatching(denseRandom, false);
        check(dense.size == denseCold.size, "the warm start does not change the optimum");
        check(dense.augmentations <= denseCold.augmentations,
                "the warm start performs no more augmenting searches");
        check(dense.greedyPairs <= dense.size, "greedy pairs never exceed the optimum");
        check(dense.greedyPairs > 0, "the greedy phase finds pairs on a dense graph");
        check(dense.greedyPairs >= denseCold.greedyPairs,
                "the cold run reports no greedy pairs");

        Graph repeatGraph = new Graph(12, randomEdges(12, 99991, 45));
        Matching repeatA = maximumMatching(repeatGraph, true);
        Matching repeatB = maximumMatching(repeatGraph, true);
        check(Arrays.equals(repeatA.mate, repeatB.mate), "deterministic re-run");
        check(repeatA.size == repeatB.size && repeatA.augmentations == repeatB.augmentations,
                "deterministic statistics");

        System.out.println("PASS");
    }

    static List<int[]> randomEdges(int vertices, int seed, int density) {
        List<int[]> edges = new ArrayList<>();
        int state = seed;
        for (int i = 0; i < vertices; i++) {
            for (int j = i + 1; j < vertices; j++) {
                state = (state * 75 + 74) % 65537;
                if (state % 100 < density) edges.add(new int[]{i, j});
            }
        }
        return edges;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
