/** Task 456 - Implement matchings in general graph (Edmonds blossom outline). (Tier: Advanced, CTCI Ch 4) */
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

    static int[] greedyMatching(int n, List<List<Integer>> adj) {
        int[] mate = new int[n];
        Arrays.fill(mate, -1);
        for (int u = 0; u < n; u++) {
            if (mate[u] == -1) {
                for (int v : adj.get(u)) {
                    if (v > u && mate[v] == -1) {
                        mate[u] = v;
                        mate[v] = u;
                        break;
                    }
                }
            }
        }
        return mate;
    }

    /** DFS over simple alternating paths from a free vertex; returns the edge list or null. */
    static List<int[]> findAugmentingPath(int n, List<List<Integer>> adj, int[] mate, int root) {
        boolean[] onPath = new boolean[n];
        onPath[root] = true;
        List<int[]> path = new ArrayList<>();
        if (dfs(root, adj, mate, onPath, path)) return path;
        return null;
    }

    static boolean dfs(int u, List<List<Integer>> adj, int[] mate, boolean[] onPath, List<int[]> path) {
        for (int v : adj.get(u)) {
            if (mate[u] == v || onPath[v]) continue;
            if (mate[v] == -1) {
                path.add(new int[]{u, v});
                return true;
            }
            int w = mate[v];
            if (onPath[w]) continue;
            onPath[v] = true;
            onPath[w] = true;
            path.add(new int[]{u, v});
            path.add(new int[]{v, w});
            if (dfs(w, adj, mate, onPath, path)) return true;
            path.remove(path.size() - 1);
            path.remove(path.size() - 1);
            onPath[v] = false;
            onPath[w] = false;
        }
        return false;
    }

    static void applyAugmenting(int[] mate, List<int[]> path) {
        for (int i = 0; i < path.size(); i++) {
            if (i % 2 == 1) {
                int[] e = path.get(i);
                check(mate[e[0]] == e[1] && mate[e[1]] == e[0], "path alternates");
                mate[e[0]] = -1;
                mate[e[1]] = -1;
            }
        }
        for (int i = 0; i < path.size(); i++) {
            if (i % 2 == 0) {
                int[] e = path.get(i);
                mate[e[0]] = e[1];
                mate[e[1]] = e[0];
            }
        }
    }

    /** Greedy + augmenting paths; returns {mate, augments}. */
    static int[][] maxMatchingOutline(int n, List<List<Integer>> adj) {
        int[] mate = greedyMatching(n, adj);
        int augmentations = 0;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int u = 0; u < n; u++) {
                if (mate[u] == -1) {
                    List<int[]> path = findAugmentingPath(n, adj, mate, u);
                    if (path != null) {
                        applyAugmenting(mate, path);
                        augmentations++;
                        changed = true;
                    }
                }
            }
        }
        return new int[][]{mate, new int[]{augmentations}};
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
        Object[][] handGraphs = {
                {"triangle", 3, new int[][]{{0, 1}, {1, 2}, {0, 2}}, 1},
                {"C5", 5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}}, 2},
                {"K4", 4, new int[][]{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}, 2},
                {"star", 5, new int[][]{{0, 1}, {0, 2}, {0, 3}, {0, 4}}, 1},
                {"C6", 6, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}}, 3},
                {"C7", 7, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 0}}, 3},
                {"two triangles", 6, new int[][]{{0, 1}, {1, 2}, {0, 2}, {3, 4}, {4, 5}, {3, 5}}, 2},
                {"K3,3", 6, new int[][]{{0, 3}, {0, 4}, {0, 5}, {1, 3}, {1, 4}, {1, 5}, {2, 3}, {2, 4}, {2, 5}}, 3},
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
            int[][] result = maxMatchingOutline(n, adj);
            check(checkMatching(n, adj, result[0]) == want, label + " outline");
        }

        int improvedGraphs = 0, checked = 0, totalAugmentations = 0;
        for (int seed : new int[]{4561, 4562, 4563, 4564}) {
            for (Instance inst : lcgGeneralGraphs(6, seed)) {
                List<List<Integer>> adj = buildAdj(inst.n, inst.edges);
                int[] greedy = greedyMatching(inst.n, adj);
                int greedySize = checkMatching(inst.n, adj, greedy);
                int[][] result = maxMatchingOutline(inst.n, adj);
                int size = checkMatching(inst.n, adj, result[0]);
                check(size >= greedySize, "outline at least greedy");
                if (size > greedySize) improvedGraphs++;
                check(size == bruteMaxMatching(inst.n, adj), "outline equals brute");
                totalAugmentations += result[1][0];
                checked++;
            }
        }
        check(checked == 24, "graph count");
        check(improvedGraphs >= 1, "augmenting improved something");
        check(totalAugmentations >= improvedGraphs, "augmentation count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
