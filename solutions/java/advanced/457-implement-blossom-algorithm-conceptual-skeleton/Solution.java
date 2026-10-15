/** Task 457 - Implement blossom algorithm (conceptual/skeleton). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static List<List<Integer>> buildAdj(int n, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) {
            adj.get(e[0]).add(e[1]);
            adj.get(e[1]).add(e[0]);
        }
        return adj;
    }

    /** Asserts validity and returns the matching size. */
    static int checkMatching(int n, List<List<Integer>> adj, int[] mate) {
        int size = 0;
        for (int v = 0; v < n; v++) {
            if (mate[v] != -1) {
                check(mate[mate[v]] == v, "symmetric mates");
                check(mate[v] != v, "no self match");
                check(adj.get(v).contains(mate[v]), "matched pair must be an edge");
                size++;
            }
        }
        check(size % 2 == 0, "even matched count");
        return size / 2;
    }

    /** Edmonds' algorithm with labelled BFS + blossom contraction. */
    static class Blossom {
        final int n;
        final List<List<Integer>> adj;
        final int[] mate, base, p;
        final boolean[] used, blossom;
        int contractions = 0;

        Blossom(int n, List<List<Integer>> adj) {
            this.n = n;
            this.adj = adj;
            mate = new int[n];
            base = new int[n];
            p = new int[n];
            used = new boolean[n];
            blossom = new boolean[n];
            Arrays.fill(mate, -1);
        }

        int lca(int a, int b) {
            boolean[] seen = new boolean[n];
            while (true) {
                a = base[a];
                seen[a] = true;
                if (mate[a] == -1) break;
                a = p[mate[a]];
            }
            while (true) {
                b = base[b];
                if (seen[b]) return b;
                b = p[mate[b]];
            }
        }

        void markPath(int v, int b, int child) {
            while (base[v] != b) {
                blossom[base[v]] = true;
                blossom[base[mate[v]]] = true;
                p[v] = child;
                child = mate[v];
                v = p[mate[v]];
            }
        }

        int findPath(int root) {
            for (int i = 0; i < n; i++) {
                used[i] = false;
                p[i] = -1;
                base[i] = i;
            }
            used[root] = true;
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                int v = queue.poll();
                for (int to : adj.get(v)) {
                    if (base[v] == base[to] || mate[v] == to) continue;
                    if (to == root || (mate[to] != -1 && p[mate[to]] != -1)) {
                        contractions++;
                        int curbase = lca(v, to);
                        Arrays.fill(blossom, false);
                        markPath(v, curbase, to);
                        markPath(to, curbase, v);
                        for (int i = 0; i < n; i++) {
                            if (blossom[base[i]]) {
                                base[i] = curbase;
                                if (!used[i]) {
                                    used[i] = true;
                                    queue.add(i);
                                }
                            }
                        }
                    } else if (p[to] == -1) {
                        p[to] = v;
                        if (mate[to] == -1) return to;
                        int nxt = mate[to];
                        used[nxt] = true;
                        queue.add(nxt);
                    }
                }
            }
            return -1;
        }

        /** Runs the algorithm; returns {mate, {contractions}}. */
        int[][] run() {
            for (int i = 0; i < n; i++) {
                if (mate[i] == -1) {
                    int v = findPath(i);
                    while (v != -1) {
                        int pv = p[v];
                        int ppv = mate[pv];
                        mate[v] = pv;
                        mate[pv] = v;
                        v = ppv;
                    }
                }
            }
            return new int[][]{mate, new int[]{contractions}};
        }
    }

    /** Exact maximum matching by exhaustive backtracking (first-free-vertex branching). */
    static int bruteMaxMatching(int n, List<List<Integer>> adj) {
        int[] mate = new int[n];
        Arrays.fill(mate, -1);
        return rec(0, n, adj, mate);
    }

    static int rec(int i, int n, List<List<Integer>> adj, int[] mate) {
        while (i < n && mate[i] != -1) i++;
        if (i == n) return 0;
        int best = rec(i + 1, n, adj, mate);
        for (int v : adj.get(i)) {
            if (v > i && mate[v] == -1) {
                mate[i] = v;
                mate[v] = i;
                best = Math.max(best, 1 + rec(i + 1, n, adj, mate));
                mate[i] = -1;
                mate[v] = -1;
            }
        }
        return best;
    }

    static class Instance {
        final int n;
        final int[][] edges;

        Instance(int n, int[][] edges) {
            this.n = n;
            this.edges = edges;
        }
    }

    static List<Instance> lcgGeneralGraphs(int count, int seed) {
        List<Instance> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 2 + seed % 7;
            List<int[]> edges = new ArrayList<>();
            for (int node = 1; node < n; node++) {
                seed = (seed * 75 + 74) % 65537;
                edges.add(new int[]{node, seed % node});
            }
            for (int k = 0; k < n + 1; k++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                if (u != v && !hasEdge(edges, u, v)) edges.add(new int[]{u, v});
            }
            graphs.add(new Instance(n, edges.toArray(new int[0][])));
        }
        return graphs;
    }

    static boolean hasEdge(List<int[]> edges, int u, int v) {
        for (int[] e : edges) {
            if ((e[0] == u && e[1] == v) || (e[0] == v && e[1] == u)) return true;
        }
        return false;
    }

    public static void main(String[] args) {
        List<List<Integer>> c5adj = buildAdj(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});
        Blossom c5 = new Blossom(5, c5adj);
        int[][] c5result = c5.run();
        check(checkMatching(5, c5adj, c5result[0]) == 2, "C5 optimum");
        check(c5result[1][0] >= 1, "C5 contractions");
        check(bruteMaxMatching(5, c5adj) == 2, "C5 brute");

        Object[][] handGraphs = {
                {"triangle", 3, new int[][]{{0, 1}, {1, 2}, {0, 2}}, 1},
                {"C5", 5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}, 2},
                {"K4", 4, new int[][]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}, 2},
                {"star", 5, new int[][]{{0, 1}, {0, 2}, {0, 3}, {0, 4}}, 1},
                {"C6", 6, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}}, 3},
                {"C7", 7, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 0}}, 3},
                {"K3,3", 6, new int[][]{{0, 3}, {0, 4}, {0, 5}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5}}, 3},
                {"two triangles", 6, new int[][]{{0, 1}, {1, 2}, {0, 2}, {3, 4}, {4, 5}, {3, 5}}, 2},
                {"diamond", 4, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}}, 2},
                {"P4", 4, new int[][]{{0, 1}, {1, 2}, {2, 3}}, 2},
                {"empty", 5, new int[][]{}, 0},
                {"single", 2, new int[][]{{0, 1}}, 1},
        };
        for (Object[] graph : handGraphs) {
            String label = (String) graph[0];
            int n = (Integer) graph[1];
            int[][] edges = (int[][]) graph[2];
            int want = (Integer) graph[3];
            List<List<Integer>> adj = buildAdj(n, edges);
            check(bruteMaxMatching(n, adj) == want, label + " brute");
            Blossom blossom = new Blossom(n, adj);
            int[][] result = blossom.run();
            check(checkMatching(n, adj, result[0]) == want, label + " optimum");
        }

        int checked = 0, totalContractions = 0;
        for (int seed : new int[]{4571, 4572, 4573, 4574}) {
            for (Instance inst : lcgGeneralGraphs(6, seed)) {
                List<List<Integer>> adj = buildAdj(inst.n, inst.edges);
                Blossom blossom = new Blossom(inst.n, adj);
                int[][] result = blossom.run();
                totalContractions += result[1][0];
                int size = checkMatching(inst.n, adj, result[0]);
                check(size == bruteMaxMatching(inst.n, adj), "blossom equals brute");
                checked++;
            }
        }
        check(checked == 24, "graph count");
        check(totalContractions >= 1, "contractions exercised");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
