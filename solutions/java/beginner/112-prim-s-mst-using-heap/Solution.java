/** Task 112 - Prim's MST using heap. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** adjacency: adjacency[u] = list of {v, weight}. Returns {total, edgeCount} or null. */
    static long[] prim(int n, int[][][] adjacency) {
        boolean[] visited = new boolean[n];
        long total = 0;
        int count = 0;
        // heap entries: {weight, node, parent}
        PriorityQueue<int[]> heap = new PriorityQueue<>(Comparator.comparingInt(e -> e[0]));
        heap.add(new int[]{0, 0, -1});
        while (!heap.isEmpty() && count < n) {
            int[] e = heap.poll();
            int weight = e[0], u = e[1];
            if (visited[u]) continue;
            visited[u] = true;
            total += weight;
            if (e[2] != -1) count++;
            for (int[] edge : adjacency[u])
                if (!visited[edge[0]]) heap.add(new int[]{edge[1], edge[0], u});
        }
        return count == Math.max(0, n - 1) ? new long[]{total, count} : null;
    }

    public static void main(String[] args) {
        int[][][] adjacency = {
            {{1, 4}, {2, 3}},
            {{0, 4}, {2, 1}, {3, 2}},
            {{0, 3}, {1, 1}, {3, 4}, {4, 5}},
            {{1, 2}, {2, 4}, {4, 2}},
            {{3, 2}, {2, 5}},
        };
        long[] result = prim(5, adjacency);
        check(result != null && result[0] == 8 && result[1] == 4, "classic graph");

        int[][][] single = { {} };          // one vertex, no edges
        check(prim(1, single)[0] == 0, "single vertex");

        int[][][] disconnected = {{{1, 1}}, {{0, 1}}, {{}}};
        check(prim(3, disconnected) == null, "disconnected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
