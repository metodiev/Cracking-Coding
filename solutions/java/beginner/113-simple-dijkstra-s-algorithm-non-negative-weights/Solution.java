/** Task 113 - Simple Dijkstra's algorithm (non-negative weights). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final long INF = Long.MAX_VALUE / 4;

    /** adjacency[u] = list of {v, weight}; returns distances, INF = unreachable. */
    static long[] dijkstra(int n, int[][][] adjacency, int src) {
        long[] dist = new long[n];
        Arrays.fill(dist, INF);
        dist[src] = 0;
        PriorityQueue<long[]> heap = new PriorityQueue<>(Comparator.comparingLong(e -> e[0]));
        heap.add(new long[]{0, src});
        while (!heap.isEmpty()) {
            long[] top = heap.poll();
            long d = top[0];
            int u = (int) top[1];
            if (d > dist[u]) continue;
            for (int[] edge : adjacency[u]) {
                long nd = d + edge[1];
                if (nd < dist[edge[0]]) {
                    dist[edge[0]] = nd;
                    heap.add(new long[]{nd, edge[0]});
                }
            }
        }
        return dist;
    }

    public static void main(String[] args) {
        int[][][] adjacency = {
            {{1, 1}, {2, 4}},
            {{2, 2}, {3, 6}},
            {{3, 3}},
            {},
        };
        long[] dist = dijkstra(4, adjacency, 0);
        check(dist[0] == 0 && dist[1] == 1 && dist[2] == 3 && dist[3] == 6, "classic");

        int[][][] adjacency2 = {{{1, 5}}, {{0, 5}}, {}};
        long[] dist2 = dijkstra(3, adjacency2, 0);
        check(dist2[0] == 0 && dist2[1] == 5 && dist2[2] == INF, "unreachable is INF");

        int[][][] adjacency3 = {{{1, 2}, {2, 5}}, {{2, 1}, {3, 4}}, {{3, 1}}, {}};
        long[] d3 = dijkstra(4, adjacency3, 0);
        check(d3[0] == 0 && d3[1] == 2 && d3[2] == 3 && d3[3] == 4, "DAG");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
