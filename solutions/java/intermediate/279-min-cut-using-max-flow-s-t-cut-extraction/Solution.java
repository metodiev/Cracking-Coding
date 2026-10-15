/** Task 279 - Min cut using max flow (s-t cut extraction). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Edmonds-Karp; returns {value, then the residual matrix flattened row by row}. */
    static int[][] maxFlow(int n, int[][] capacity, int source, int sink) {
        int[][] residual = new int[n][];
        for (int i = 0; i < n; i++) residual[i] = capacity[i].clone();
        int[][] result = new int[n + 1][];
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
                    if (parent[v] == -1 && residual[u][v] > 0) { parent[v] = u; queue.add(v); }
                }
            }
            if (parent[sink] == -1) break;
            int bottleneck = Integer.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) bottleneck = Math.min(bottleneck, residual[parent[v]][v]);
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= bottleneck;
                residual[v][u] += bottleneck;
            }
            total += bottleneck;
        }
        result[0] = new int[]{total};
        for (int i = 0; i < n; i++) result[i + 1] = residual[i];
        return result;
    }

    /** Returns {value, sCount... } packed: result[0][0] is the value, S as a boolean string in result[1]. */
    static String describeMinCut(int n, int[][] capacity, int source, int sink) {
        int[][] flowResult = maxFlow(n, capacity, source, sink);
        int value = flowResult[0][0];
        int[][] residual = Arrays.copyOfRange(flowResult, 1, n + 1);
        boolean[] side = new boolean[n];
        side[source] = true;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v = 0; v < n; v++) {
                if (!side[v] && residual[u][v] > 0) { side[v] = true; queue.add(v); }
            }
        }
        int cutCapacity = 0;
        for (int u = 0; u < n; u++) {
            if (!side[u]) continue;
            for (int v = 0; v < n; v++) if (!side[v]) cutCapacity += capacity[u][v];
        }
        StringBuilder sb = new StringBuilder();
        sb.append(value).append('|').append(cutCapacity).append('|');
        for (int v = 0; v < n; v++) sb.append(side[v] ? '1' : '0');
        return sb.toString();
    }

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

    static int[][] matrixFromEdges(int n, int[][] edges) {
        int[][] capacity = new int[n][n];
        for (int[] edge : edges) capacity[edge[0]][edge[1]] += edge[2];
        return capacity;
    }

    private static int seed = 24680;

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

    public static void main(String[] args) {
        int[][] clrs = matrixFromEdges(6, new int[][]{
            {0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},{3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}});
        String[] parts = describeMinCut(6, clrs, 0, 5).split("\\|");
        check(parts[0].equals("23"), "CLRS flow 23");
        check(parts[1].equals(parts[0]), "cut capacity equals flow");
        check(parts[2].startsWith("1"), "source in S");
        check(parts[2].endsWith("0"), "sink in T");
        check(minCutBruteForce(6, clrs, 0, 5) == 23, "brute force min cut");

        int[][] diamond = matrixFromEdges(4, new int[][]{{0,1,3},{0,2,2},{1,3,2},{2,3,3}});
        parts = describeMinCut(4, diamond, 0, 3).split("\\|");
        check(parts[0].equals("4") && parts[1].equals("4"), "diamond cut");
        check(minCutBruteForce(4, diamond, 0, 3) == 4, "diamond brute force");

        int[][] stranded = matrixFromEdges(3, new int[][]{{0,1,7}});
        parts = describeMinCut(3, stranded, 0, 2).split("\\|");
        check(parts[0].equals("0") && parts[1].equals("0") && parts[2].equals("110"), "stranded sink");

        for (int trial = 0; trial < 60; trial++) {
            int[][] capacity = lcgCapacities(6);
            String[] p = describeMinCut(6, capacity, 0, 5).split("\\|");
            check(p[0].equals(p[1]), "cut equals flow " + trial);
            check(Integer.parseInt(p[0]) == minCutBruteForce(6, capacity, 0, 5), "brute force " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
