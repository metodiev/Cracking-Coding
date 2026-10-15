/** Task 277 - Maximum flow (Edmonds-Karp). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** capacity[i][j] is the capacity of the edge i -> j (0 when there is none). */
    static int edmondsKarp(int n, int[][] capacity, int source, int sink) {
        int[][] residual = new int[n][n];
        for (int i = 0; i < n; i++) residual[i] = capacity[i].clone();
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
                    if (parent[v] == -1 && residual[u][v] > 0) {
                        parent[v] = u;
                        queue.add(v);
                    }
                }
            }
            if (parent[sink] == -1) return total;

            int bottleneck = Integer.MAX_VALUE;
            for (int v = sink; v != source; v = parent[v]) {
                bottleneck = Math.min(bottleneck, residual[parent[v]][v]);
            }
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= bottleneck;
                residual[v][u] += bottleneck;
            }
            total += bottleneck;
        }
    }

    static int[][] matrixFromEdges(int n, int[][] edges) {
        int[][] capacity = new int[n][n];
        for (int[] edge : edges) capacity[edge[0]][edge[1]] += edge[2];
        return capacity;
    }

    public static void main(String[] args) {
        int[][] clrs = matrixFromEdges(6, new int[][]{
            {0,1,16},{0,2,13},{1,2,10},{2,1,4},{1,3,12},{3,2,9},{2,4,14},{4,3,7},{3,5,20},{4,5,4}});
        check(edmondsKarp(6, clrs, 0, 5) == 23, "CLRS 23");

        check(edmondsKarp(4, matrixFromEdges(4, new int[][]{{0,1,3},{0,2,2},{1,3,2},{2,3,3}}), 0, 3) == 4, "diamond");
        check(edmondsKarp(3, matrixFromEdges(3, new int[][]{{0,1,3},{1,2,5}}), 0, 2) == 3, "bottleneck path");
        check(edmondsKarp(3, matrixFromEdges(3, new int[][]{{0,1,1}}), 0, 2) == 0, "unreachable sink");
        check(edmondsKarp(2, matrixFromEdges(2, new int[][]{}), 0, 1) == 0, "no edges");

        int[][] crossing = matrixFromEdges(4, new int[][]{{0,1,1},{0,2,1},{1,2,1},{1,3,1},{2,3,1}});
        check(edmondsKarp(4, crossing, 0, 3) == 2, "back-edge augmentation");

        check(edmondsKarp(3, matrixFromEdges(3, new int[][]{{0,1,2},{0,1,3},{1,2,4}}), 0, 2) == 4, "parallel edges add up");

        int totalOut = 0;
        for (int v = 0; v < 6; v++) totalOut += clrs[0][v];
        check(edmondsKarp(6, clrs, 0, 5) <= totalOut, "bounded by source capacity");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
