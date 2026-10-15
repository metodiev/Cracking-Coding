/** Task 425 - Implement centroid decomposition for k-distance pairs. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    private static List<List<Integer>> adjacency;
    private static boolean[] removed;
    private static int[] size;
    private static long total;
    private static int target;

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
        target = k;
        if (n > 0) decompose(0);
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

    private static int findCentroid(int u, int parent, int component) {
        for (int v : adjacency.get(u)) {
            if (v != parent && !removed[v] && size[v] > component / 2) {
                return findCentroid(v, u, component);
            }
        }
        return u;
    }

    private static void collect(int u, int parent, int distance, List<Integer> distances) {
        distances.add(distance);
        for (int v : adjacency.get(u)) {
            if (v != parent && !removed[v]) collect(v, u, distance + 1, distances);
        }
    }

    private static long pairsWithSum(List<Integer> distances) {
        Map<Integer, Integer> seen = new HashMap<>();
        long pairs = 0;
        for (int d : distances) {
            pairs += seen.getOrDefault(target - d, 0);
            seen.merge(d, 1, Integer::sum);
        }
        return pairs;
    }

    private static void decompose(int start) {
        sizes(start, -1);
        int centroid = findCentroid(start, -1, size[start]);
        removed[centroid] = true;
        List<Integer> allDistances = new ArrayList<>();
        allDistances.add(0);                        // the centroid itself
        List<List<Integer>> branches = new ArrayList<>();
        for (int v : adjacency.get(centroid)) {
            if (removed[v]) continue;
            List<Integer> distances = new ArrayList<>();
            collect(v, centroid, 1, distances);
            branches.add(distances);
            allDistances.addAll(distances);
        }
        total += pairsWithSum(allDistances);
        for (List<Integer> distances : branches) {  // remove pairs inside one branch
            total -= pairsWithSum(distances);
        }
        for (int v : adjacency.get(centroid)) {
            if (!removed[v]) decompose(v);
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
            for (int other = source + 1; other < n; other++) {
                if (distance[other] == k) count++;
            }
        }
        return count;
    }

    private static int seed = 425;

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
        check(countPairsAtDistance(0, new int[][]{}, 0) == 0, "empty tree");
        check(countPairsAtDistance(1, new int[][]{}, 0) == 0, "single node k0");
        check(countPairsAtDistance(1, new int[][]{}, 1) == 0, "single node k1");
        check(countPairsAtDistance(2, new int[][]{{0, 1}}, 1) == 1, "two nodes k1");
        check(countPairsAtDistance(2, new int[][]{{0, 1}}, 0) == 0, "two nodes k0");
        check(countPairsAtDistance(2, new int[][]{{0, 1}}, 2) == 0, "two nodes k2");

        int[][] path3 = {{0, 1}, {1, 2}};
        check(countPairsAtDistance(3, path3, 0) == 0, "path3 k0");
        check(countPairsAtDistance(3, path3, 1) == 2, "path3 k1");
        check(countPairsAtDistance(3, path3, 2) == 1, "path3 k2");
        check(countPairsAtDistance(3, path3, 3) == 0, "path3 k3");

        int[][] star = {{0, 1}, {0, 2}, {0, 3}};
        check(countPairsAtDistance(4, star, 1) == 3, "star k1");
        check(countPairsAtDistance(4, star, 2) == 3, "star k2");
        check(countPairsAtDistance(4, star, 3) == 0, "star k3");

        int[][] path4 = {{0, 1}, {1, 2}, {2, 3}};
        check(countPairsAtDistance(4, path4, 1) == 3, "path4 k1");
        check(countPairsAtDistance(4, path4, 2) == 2, "path4 k2");
        check(countPairsAtDistance(4, path4, 3) == 1, "path4 k3");
        check(countPairsAtDistance(4, path4, 4) == 0, "path4 k4");

        int[][] path6 = new int[5][];
        for (int i = 0; i < 5; i++) path6[i] = new int[]{i, i + 1};
        int[] expectedPath6 = {0, 5, 4, 3, 2, 1, 0};
        for (int k = 0; k < expectedPath6.length; k++) {
            check(countPairsAtDistance(6, path6, k) == expectedPath6[k], "path6 k=" + k);
        }

        for (int trial = 0; trial < 30; trial++) {
            Object[] tree = lcgTree(60);
            int n = (Integer) tree[0];
            int[][] edges = (int[][]) tree[1];
            for (int k = 0; k < 13; k++) {
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
