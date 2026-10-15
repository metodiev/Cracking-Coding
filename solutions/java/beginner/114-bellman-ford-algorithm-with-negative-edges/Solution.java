/** Task 114 - Bellman-Ford algorithm (with negative edges). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    /** edges rows: {u, v, w}. Returns distances or null when a negative cycle is reachable. */
    static long[] bellmanFord(int n, int[][] edges, int src) {
        long[] dist = new long[n];
        Arrays.fill(dist, INF);
        dist[src] = 0;
        for (int pass = 0; pass < n - 1; pass++) {
            boolean changed = false;
            for (int[] e : edges) {
                if (dist[e[0]] != INF && dist[e[0]] + e[2] < dist[e[1]]) {
                    dist[e[1]] = dist[e[0]] + e[2];
                    changed = true;
                }
            }
            if (!changed) break;
        }
        for (int[] e : edges)
            if (dist[e[0]] != INF && dist[e[0]] + e[2] < dist[e[1]]) return null;
        return dist;
    }

    public static void main(String[] args) {
        int[][] edges = {{0, 1, 4}, {0, 2, 5}, {1, 2, -3}, {2, 3, 2}};
        long[] dist = bellmanFord(4, edges, 0);
        check(dist != null && dist[0] == 0 && dist[1] == 4 && dist[2] == 1 && dist[3] == 3, "negative edge");

        int[][] nonNeg = {{0, 1, 1}, {1, 2, 2}, {0, 2, 4}};
        long[] d2 = bellmanFord(3, nonNeg, 0);
        check(d2[1] == 1 && d2[2] == 3, "matches Dijkstra");

        int[][] cycle = {{0, 1, 1}, {1, 2, -2}, {2, 1, -2}};
        check(bellmanFord(3, cycle, 0) == null, "negative cycle");

        int[][] far = {{1, 2, 1}};
        check(bellmanFord(3, far, 0)[2] == INF, "unreachable");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
