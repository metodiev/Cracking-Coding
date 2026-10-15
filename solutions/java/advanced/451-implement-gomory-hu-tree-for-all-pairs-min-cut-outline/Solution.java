/** Task 451 - Implement Gomory-Hu tree for all-pairs min-cut (outline). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final int INF = Integer.MAX_VALUE / 4;

    static List<int[]> lcgGraphs(int count, int seed, int minN, int maxN) {
        List<int[]> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = minN + seed % (maxN - minN + 1);
            List<int[]> edges = new ArrayList<>();
            for (int node = 1; node < n; node++) {
                seed = (seed * 75 + 74) % 65537;
                int parent = seed % node;
                seed = (seed * 75 + 74) % 65537;
                edges.add(new int[]{node, parent, 1 + seed % 9});
            }
            for (int k = 0; k < n; k++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.add(new int[]{u, v, 1 + seed % 9});
            }
            int[] packed = new int[1 + edges.size() * 3];
            packed[0] = n;
            for (int i = 0; i < edges.size(); i++) {
                packed[1 + 3 * i] = edges.get(i)[0];
                packed[2 + 3 * i] = edges.get(i)[1];
                packed[3 + 3 * i] = edges.get(i)[2];
            }
            graphs.add(packed);
        }
        return graphs;
    }

    static int[][] capacityMatrix(int n, int[] packedEdges) {
        int[][] cap = new int[n][n];
        for (int i = 1; i < packedEdges.length; i += 3) {
            int u = packedEdges[i], v = packedEdges[i + 1], w = packedEdges[i + 2];
            if (u != v) {
                cap[u][v] += w;
                cap[v][u] += w;
            }
        }
        return cap;
    }

    /** Edmonds-Karp on the directed model of an undirected graph. Returns {value, sideBitmask}. */
    static int[] maxFlowUndirected(int n, int[] packedEdges, int s, int t) {
        int[][] cap = capacityMatrix(n, packedEdges);
        int flow = 0;
        while (true) {
            int[] parent = new int[n];
            Arrays.fill(parent, -1);
            parent[s] = s;
            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(s);
            while (!q.isEmpty()) {
                int u = q.poll();
                for (int v = 0; v < n; v++) {
                    if (parent[v] == -1 && cap[u][v] > 0) {
                        parent[v] = u;
                        q.add(v);
                    }
                }
            }
            int side = 0;
            for (int i = 0; i < n; i++) if (parent[i] != -1) side |= 1 << i;
            if (parent[t] == -1) return new int[]{flow, side};
            int aug = INF;
            for (int v = t; v != s; v = parent[v]) aug = Math.min(aug, cap[parent[v]][v]);
            for (int v = t; v != s; v = parent[v]) {
                cap[parent[v]][v] -= aug;
                cap[v][parent[v]] += aug;
            }
            flow += aug;
        }
    }

    /** Gusfield simplification: n-1 max-flow runs. Returns {parent[], weight[], runs}. */
    static int[][] gomoryHu(int n, int[] packedEdges) {
        int[] parent = new int[n];
        int[] weight = new int[n];
        int runs = 0;
        for (int s = 1; s < n; s++) {
            int t = parent[s];
            int[] result = maxFlowUndirected(n, packedEdges, s, t);
            runs++;
            weight[s] = result[0];
            for (int i = s + 1; i < n; i++) {
                if (parent[i] == t && ((result[1] >> i) & 1) == 1) parent[i] = s;
            }
        }
        return new int[][]{parent, weight, new int[]{runs}};
    }

    /** Minimum edge weight on the tree path u..v. */
    static int treeMinCut(int n, int[] parent, int[] weight, int u, int v) {
        List<List<int[]>> tree = new ArrayList<>();
        for (int i = 0; i < n; i++) tree.add(new ArrayList<>());
        for (int i = 1; i < n; i++) {
            tree.get(i).add(new int[]{parent[i], weight[i]});
            tree.get(parent[i]).add(new int[]{i, weight[i]});
        }
        boolean[] seen = new boolean[n];
        seen[u] = true;
        ArrayDeque<int[]> q = new ArrayDeque<>();
        q.add(new int[]{u, INF});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            if (cur[0] == v) return cur[1];
            for (int[] nxt : tree.get(cur[0])) {
                if (!seen[nxt[0]]) {
                    seen[nxt[0]] = true;
                    q.add(new int[]{nxt[0], Math.min(cur[1], nxt[1])});
                }
            }
        }
        return -1;
    }

    /** Exact min u-v cut: minimum crossing weight over all vertex partitions. */
    static int brutePairMinCut(int n, int[] packedEdges, int u, int v) {
        int best = INF;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (((mask >> u) & 1) == ((mask >> v) & 1)) continue;
            int cut = 0;
            for (int i = 1; i < packedEdges.length; i += 3) {
                if (((mask >> packedEdges[i]) & 1) != ((mask >> packedEdges[i + 1]) & 1))
                    cut += packedEdges[i + 2];
            }
            best = Math.min(best, cut);
        }
        return best;
    }

    public static void main(String[] args) {
        int[] triangle = {3, 0, 1, 1, 1, 2, 1, 0, 2, 1};
        int[][] gh = gomoryHu(3, triangle);
        check(gh[2][0] == 2, "triangle flow runs");
        for (int u = 0; u < 3; u++) {
            for (int v = u + 1; v < 3; v++) {
                check(treeMinCut(3, gh[0], gh[1], u, v) == 2, "triangle tree cut");
                check(treeMinCut(3, gh[0], gh[1], u, v) == treeMinCut(3, gh[0], gh[1], v, u), "symmetry");
                check(brutePairMinCut(3, triangle, u, v) == 2, "triangle brute");
            }
        }

        int[] path = {4, 0, 1, 5, 1, 2, 3, 2, 3, 4};
        int[] pU = {0, 0, 0, 1, 1, 2};
        int[] pV = {1, 2, 3, 2, 3, 3};
        int[] pW = {5, 3, 3, 3, 3, 4};
        gh = gomoryHu(4, path);
        check(gh[2][0] == 3, "path flow runs");
        for (int k = 0; k < 6; k++) {
            check(treeMinCut(4, gh[0], gh[1], pU[k], pV[k]) == pW[k], "path tree cut");
            check(brutePairMinCut(4, path, pU[k], pV[k]) == pW[k], "path brute");
        }

        int[] k4 = {4};
        List<Integer> k4e = new ArrayList<>();
        for (int u = 0; u < 4; u++)
            for (int v = u + 1; v < 4; v++) {
                k4e.add(u);
                k4e.add(v);
                k4e.add(1);
            }
        int[] k4packed = new int[1 + k4e.size()];
        k4packed[0] = 4;
        for (int i = 0; i < k4e.size(); i++) k4packed[1 + i] = k4e.get(i);
        gh = gomoryHu(4, k4packed);
        for (int u = 0; u < 4; u++) {
            for (int v = u + 1; v < 4; v++) {
                check(treeMinCut(4, gh[0], gh[1], u, v) == 3, "K4 tree cut");
                check(brutePairMinCut(4, k4packed, u, v) == 3, "K4 brute");
            }
        }

        int[] parallel = {2, 0, 1, 3, 1, 0, 4};
        gh = gomoryHu(2, parallel);
        check(treeMinCut(2, gh[0], gh[1], 0, 1) == 7, "parallel edges tree");
        check(brutePairMinCut(2, parallel, 0, 1) == 7, "parallel edges brute");

        int pairChecks = 0, graphCount = 0;
        int[] seeds = {4511, 4512, 4513, 4514};
        for (int seed : seeds) {
            for (int[] packed : lcgGraphs(4, seed, 4, 7)) {
                int n = packed[0];
                gh = gomoryHu(n, packed);
                check(gh[2][0] == n - 1, "LCG flow runs");
                for (int u = 0; u < n; u++) {
                    for (int v = u + 1; v < n; v++) {
                        int got = treeMinCut(n, gh[0], gh[1], u, v);
                        int want = brutePairMinCut(n, packed, u, v);
                        check(got == want, "LCG pair " + u + "," + v);
                        check(treeMinCut(n, gh[0], gh[1], v, u) == got, "LCG symmetry");
                        pairChecks++;
                    }
                }
                int[] direct = maxFlowUndirected(n, packed, 0, n - 1);
                check(direct[0] == brutePairMinCut(n, packed, 0, n - 1), "LCG direct oracle");
                graphCount++;
            }
        }
        check(graphCount == 16, "graph count");
        check(pairChecks >= 16 * 6, "pair count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
