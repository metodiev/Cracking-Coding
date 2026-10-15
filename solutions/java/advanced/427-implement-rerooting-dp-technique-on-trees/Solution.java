/** Task 427 - Implement rerooting DP technique on trees. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    static long[] sumOfDistancesFromEachNode(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        parent[0] = 0;
        List<Integer> order = new ArrayList<>();
        order.add(0);
        for (int i = 0; i < order.size(); i++) {
            int u = order.get(i);
            for (int v : adjacency.get(u)) {
                if (v != parent[u]) {
                    parent[v] = u;
                    order.add(v);
                }
            }
        }
        long[] size = new long[n];
        long[] down = new long[n];                  // distances from u into its subtree
        Arrays.fill(size, 1);
        for (int index = order.size() - 1; index >= 0; index--) {
            int u = order.get(index);
            for (int v : adjacency.get(u)) {
                if (v != parent[u]) {
                    size[u] += size[v];
                    down[u] += down[v] + size[v];
                }
            }
        }
        long[] result = new long[n];
        result[0] = down[0];
        for (int u : order) {
            for (int v : adjacency.get(u)) {
                if (v != parent[u]) {
                    result[v] = result[u] + n - 2 * size[v];
                }
            }
        }
        return result;
    }

    static long[] bruteForce(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        long[] totals = new long[n];
        for (int source = 0; source < n; source++) {
            int[] distance = new int[n];
            Arrays.fill(distance, -1);
            distance[source] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(source);
            long total = 0;
            while (!queue.isEmpty()) {
                int u = queue.poll();
                total += distance[u];
                for (int v : adjacency.get(u)) {
                    if (distance[v] == -1) {
                        distance[v] = distance[u] + 1;
                        queue.add(v);
                    }
                }
            }
            totals[source] = total;
        }
        return totals;
    }

    private static int seed = 427;

    private static Object[] lcgTree(int maxN) {
        int n = 2 + seed % (maxN - 1);
        int[][] edges = new int[n - 1][];
        for (int node = 1; node < n; node++) {
            seed = (seed * 75 + 74) % 65537;
            edges[node - 1] = new int[]{seed % node, node};
        }
        return new Object[]{n, edges};
    }

    public static void main(String[] args) {
        check(Arrays.equals(sumOfDistancesFromEachNode(1, new int[][]{}), new long[]{0}), "single node");
        check(Arrays.equals(sumOfDistancesFromEachNode(2, new int[][]{{0, 1}}), new long[]{1, 1}), "edge");
        check(Arrays.equals(sumOfDistancesFromEachNode(3, new int[][]{{0, 1}, {1, 2}}), new long[]{3, 2, 3}), "path3");
        check(Arrays.equals(sumOfDistancesFromEachNode(4, new int[][]{{0, 1}, {1, 2}, {2, 3}}),
                            new long[]{6, 4, 4, 6}), "path4");
        check(Arrays.equals(sumOfDistancesFromEachNode(4, new int[][]{{0, 1}, {0, 2}, {0, 3}}),
                            new long[]{3, 5, 5, 5}), "star");
        check(Arrays.equals(sumOfDistancesFromEachNode(4, new int[][]{{0, 1}, {0, 2}, {1, 3}}),
                            new long[]{4, 4, 6, 6}), "balanced");

        for (int trial = 0; trial < 30; trial++) {
            Object[] tree = lcgTree(60);
            int n = (Integer) tree[0];
            int[][] edges = (int[][]) tree[1];
            long[] got = sumOfDistancesFromEachNode(n, edges);
            long[] expected = bruteForce(n, edges);
            check(Arrays.equals(got, expected), "lcg trial " + trial + " n=" + n);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
