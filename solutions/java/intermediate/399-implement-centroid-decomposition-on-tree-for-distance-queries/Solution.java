/** Task 399 - Implement centroid decomposition for distance queries. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static List<List<Integer>> adjacency;
    private static boolean[] removed;
    private static int[] size;
    private static long total;

    static long countPairsAtDistance(int n, int[][] edges, int k) {
        adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        removed = new boolean[n];
        size = new int[n];
        total = 0;
        if (n > 0) decompose(0, k);
        return total;
    }

    private static void sizes(int u, int parent) {
        size[u] = 1;
        for (int v : adjacency.get(u)) {
            if (v != parent && !removed[v]) {
                sizes(v, u);
                size[u] += size[v];
            }
        }
    }

    private static int findCentroid(int u, int parent, int componentSize) {
        for (int v : adjacency.get(u)) {
            if (v != parent && !removed[v] && size[v] > componentSize / 2) {
                return findCentroid(v, u, componentSize);
            }
        }
        return u;
    }

    private static void collect(int u, int parent, int distance, List<Integer> out) {
        out.add(distance);
        for (int v : adjacency.get(u)) {
            if (v != parent && !removed[v]) collect(v, u, distance + 1, out);
        }
    }

    private static void decompose(int start, int k) {
        sizes(start, -1);
        int centroid = findCentroid(start, -1, size[start]);
        removed[centroid] = true;
        Map<Integer, Integer> seen = new HashMap<>();
        seen.put(0, 1);                        // the centroid itself is at distance 0
        for (int v : adjacency.get(centroid)) {
            if (removed[v]) continue;
            List<Integer> subtree = new ArrayList<>();
            collect(v, centroid, 1, subtree);
            for (int d : subtree) total += seen.getOrDefault(k - d, 0);
            for (int d : subtree) seen.merge(d, 1, Integer::sum);
        }
        for (int v : adjacency.get(centroid)) {
            if (!removed[v]) decompose(v, k);
        }
    }

    static long bruteForce(int n, int[][] edges, int k) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        long count = 0;
        for (int source = 0; source < n; source++) {
            int[] distance = new int[n];
            Arrays.fill(distance, -1);
            distance[source] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(source);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v : graph.get(u)) {
                    if (distance[v] == -1) {
                        distance[v] = distance[u] + 1;
                        queue.add(v);
                    }
                }
            }
            for (int target = source + 1; target < n; target++) {
                if (distance[target] == k) count++;
            }
        }
        return count;
    }

    private static int seed = 600;

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
        check(countPairsAtDistance(3, new int[][]{{0,1},{1,2}}, 1) == 2, "path3 k1");
        check(countPairsAtDistance(3, new int[][]{{0,1},{1,2}}, 2) == 1, "path3 k2");
        check(countPairsAtDistance(3, new int[][]{{0,1},{1,2}}, 3) == 0, "path3 k3");
        check(countPairsAtDistance(3, new int[][]{{0,1},{0,2}}, 2) == 1, "star k2");
        check(countPairsAtDistance(3, new int[][]{{0,1},{0,2}}, 1) == 2, "star k1");
        check(countPairsAtDistance(1, new int[][]{}, 0) == 0, "single node");
        check(countPairsAtDistance(2, new int[][]{{0,1}}, 1) == 1, "two nodes");
        check(countPairsAtDistance(2, new int[][]{{0,1}}, 2) == 0, "two nodes too far");

        int[][] path6 = new int[5][];
        for (int i = 0; i < 5; i++) path6[i] = new int[]{i, i + 1};
        for (int k = 0; k <= 6; k++) {
            check(countPairsAtDistance(6, path6, k) == bruteForce(6, path6, k), "path6 k=" + k);
        }

        for (int trial = 0; trial < 30; trial++) {
            Object[] tree = lcgTree(60);
            int n = (Integer) tree[0];
            int[][] edges = (int[][]) tree[1];
            for (int k = 0; k < 6; k++) {
                long got = countPairsAtDistance(n, edges, k);
                long expected = bruteForce(n, edges, k);
                check(got == expected, "random n=" + n + " k=" + k + " got=" + got + " want=" + expected);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
