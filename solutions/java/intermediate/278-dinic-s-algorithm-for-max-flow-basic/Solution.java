/** Task 278 - Dinic's algorithm for max flow (basic). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static int[][] residual;
    private static int[] level, pointer;
    private static int sinkNode;

    /** capacity[i][j] is the capacity of i -> j; returns the max flow value. */
    static int dinic(int n, int[][] capacity, int source, int sink) {
        residual = new int[n][];
        for (int i = 0; i < n; i++) residual[i] = capacity[i].clone();
        sinkNode = sink;
        int total = 0;
        while (true) {
            level = new int[n];
            Arrays.fill(level, -1);
            level[source] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(source);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v = 0; v < n; v++) {
                    if (level[v] == -1 && residual[u][v] > 0) {
                        level[v] = level[u] + 1;
                        queue.add(v);
                    }
                }
            }
            if (level[sink] == -1) return total;

            pointer = new int[n];
            while (true) {
                int pushed = dfs(source, Integer.MAX_VALUE);
                if (pushed == 0) break;
                total += pushed;
            }
        }
    }

    private static int dfs(int u, int pushed) {
        if (u == sinkNode) return pushed;
        while (pointer[u] < residual.length) {
            int v = pointer[u];
            if (residual[u][v] > 0 && level[v] == level[u] + 1) {
                int here = dfs(v, Math.min(pushed, residual[u][v]));
                if (here > 0) {
                    residual[u][v] -= here;
                    residual[v][u] += here;
                    return here;
                }
            }
            pointer[u]++;
        }
        return 0;
    }

    /** BFS augmenting-path reference implementation used to cross-check Dinic. */
    static int edmondsKarp(int n, int[][] capacity, int source, int sink) {
        int[][] res = new int[n][];
        for (int i = 0; i < n; i++) res[i] = capacity[i].clone();
        int total = 0;
        while (true) {
            int[] parent = new int[n];
            Arrays.fill(parent, -1);
            parent[source] = source;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(source);
            while (!queue.isEmpty() && parent[sink] == -1) {
                int u = queue.poll();
                for (int v = 0; v < n; v++) {
                    if (parent[v] == -1 && res[u][v] > 0) { parent[v] = u; queue.add(v); }
                }
            }
            if (parent[sink] == -1) return total;
            int bottleneck = Integer.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) bottleneck = Math.min(bottleneck, res[parent[v]][v]);
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                res[u][v] -= bottleneck;
                res[v][u] += bottleneck;
            }
            total += bottleneck;
        }
    }

    static int[][] matrixFromEdges(int n, int[][] edges) {
        int[][] capacity = new int[n][n];
        for (int[] edge : edges) capacity[edge[0]][edge[1]] += edge[2];
        return capacity;
    }

    private static int seed = 97531;

    private static int[][] lcgCapacities(int n) {
        int[][] capacity = new int[n][n];
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 4 == 0) capacity[u][v] = seed % 10;
            }
        }
        return capacity;
    }

    /** Independent oracle: enumerate every s-t cut (max-flow = min-cut). */
    static int minCutBruteForce(int n, int[][] capacity, int source, int sink) {
        int[] others = new int[n - 2];
        int idx = 0;
        for (int v = 0; v < n; v++) if (v != source && v != sink) others[idx++] = v;
        int best = Integer.MAX_VALUE;
        for (int mask = 0; mask < (1 << others.length); mask++) {
            boolean[] side = new boolean[n];
            side[source] = true;
            for (int i = 0; i < others.length; i++) if ((mask >> i & 1) == 1) side[others[i]] = true;
            int cut = 0;
            for (int u = 0; u < n; u++) {
                if (!side[u]) continue;
                for (int v = 0; v < n; v++) if (!side[v]) cut += capacity[u][v];
            }
            best = Math.min(best, cut);
        }
        return best;
    }

    public static void main(String[] args) {
        int[][] clrs = matrixFromEdges(6, new int[][]{
            {0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},{3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}});
        check(dinic(6, clrs, 0, 5) == 23, "CLRS 23");
        check(edmondsKarp(6, clrs, 0, 5) == 23, "EK agrees on CLRS");

        check(dinic(4, matrixFromEdges(4, new int[][]{{0,1,3},{0,2,2},{1,3,2},{2,3,3}}), 0, 3) == 4, "diamond");
        int[][] crossing = matrixFromEdges(4, new int[][]{{0,1,1},{0,2,1},{1,2,1},{1,3,1},{2,3,1}});
        check(dinic(4, crossing, 0, 3) == 2, "back-edge augmentation");
        check(dinic(3, matrixFromEdges(3, new int[][]{{0,1,1}}), 0, 2) == 0, "unreachable sink");

        for (int trial = 0; trial < 150; trial++) {
            int[][] capacity = lcgCapacities(6);
            check(dinic(6, capacity, 0, 5) == edmondsKarp(6, capacity, 0, 5), "random agreement " + trial);
        }

        int[][] layered = matrixFromEdges(8, new int[][]{
            {0,1,10},{0,2,10},{1,3,5},{1,4,6},{2,5,6},{2,6,5},{3,7,4},{4,7,7},{5,7,5},{6,7,4}});
        check(dinic(8, layered, 0, 7) == edmondsKarp(8, layered, 0, 7), "layered agrees with EK");
        check(dinic(8, layered, 0, 7) == minCutBruteForce(8, layered, 0, 7), "layered equals min cut");
        check(dinic(6, clrs, 0, 5) == minCutBruteForce(6, clrs, 0, 5), "CLRS equals min cut");
        seed = 1;
        for (int trial = 0; trial < 40; trial++) {
            int[][] capacity = lcgCapacities(6);
            check(dinic(6, capacity, 0, 5) == minCutBruteForce(6, capacity, 0, 5), "random min cut " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
