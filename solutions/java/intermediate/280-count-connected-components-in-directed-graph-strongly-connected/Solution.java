/** Task 280 - Count strongly connected components of a directed graph. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static List<List<Integer>> adjacency, reversedAdjacency;
    private static boolean[] visited;
    private static List<Integer> order;

    /** Kosaraju: returns components as sorted vertex lists. */
    static List<List<Integer>> stronglyConnectedComponents(int n, int[][] edges) {
        adjacency = new ArrayList<>();
        reversedAdjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) { adjacency.add(new ArrayList<>()); reversedAdjacency.add(new ArrayList<>()); }
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            reversedAdjacency.get(edge[1]).add(edge[0]);
        }

        visited = new boolean[n];
        order = new ArrayList<>();
        for (int v = 0; v < n; v++) if (!visited[v]) dfsFinish(v);

        int[] component = new int[n];
        Arrays.fill(component, -1);
        List<List<Integer>> components = new ArrayList<>();
        for (int i = order.size() - 1; i >= 0; i--) {
            int u = order.get(i);
            if (component[u] != -1) continue;
            int index = components.size();
            component[u] = index;
            List<Integer> group = new ArrayList<>();
            group.add(u);
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(u);
            while (!stack.isEmpty()) {
                int x = stack.pop();
                for (int y : reversedAdjacency.get(x)) {
                    if (component[y] == -1) {
                        component[y] = index;
                        group.add(y);
                        stack.push(y);
                    }
                }
            }
            Collections.sort(group);
            components.add(group);
        }
        return components;
    }

    private static void dfsFinish(int u) {
        visited[u] = true;
        for (int v : adjacency.get(u)) if (!visited[v]) dfsFinish(v);
        order.add(u);
    }

    static int countComponents(int n, int[][] edges) {
        return stronglyConnectedComponents(n, edges).size();
    }

    /** Transitive closure oracle: u ~ v iff each reaches the other. */
    static List<List<Integer>> bruteForceComponents(int n, int[][] edges) {
        boolean[][] reach = new boolean[n][n];
        for (int i = 0; i < n; i++) reach[i][i] = true;
        for (int[] edge : edges) reach[edge[0]][edge[1]] = true;
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i][k])
                    for (int j = 0; j < n; j++)
                        if (reach[k][j]) reach[i][j] = true;

        Map<String, List<Integer>> groups = new TreeMap<>();
        for (int v = 0; v < n; v++) {
            StringBuilder key = new StringBuilder();
            for (int i = 0; i < n; i++) if (reach[v][i] && reach[i][v]) key.append(i).append(',');
            groups.computeIfAbsent(key.toString(), k -> new ArrayList<>()).add(v);
        }
        List<List<Integer>> result = new ArrayList<>(groups.values());
        for (List<Integer> group : result) Collections.sort(group);
        result.sort((a, b) -> a.toString().compareTo(b.toString()));
        return result;
    }

    private static int seed = 13579;

    private static int[][] randomEdges(int n) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 3 == 0) edges.add(new int[]{u, v});
            }
        }
        return edges.toArray(new int[0][]);
    }

    static String join(List<List<Integer>> components) {
        List<String> parts = new ArrayList<>();
        for (List<Integer> group : components) parts.add(group.toString());
        Collections.sort(parts);
        return String.join("|", parts);
    }

    public static void main(String[] args) {
        check(countComponents(3, new int[][]{{0,1},{1,2},{2,0}}) == 1, "cycle");
        check(countComponents(4, new int[][]{{0,1},{1,2},{2,0},{3,3}}) == 2, "cycle plus self loop");
        check(countComponents(3, new int[][]{{0,1},{1,2}}) == 3, "DAG");
        check(countComponents(3, new int[][]{}) == 3, "no edges");
        check(countComponents(3, new int[][]{{0,1},{1,0},{1,2}}) == 2, "mutual pair plus tail");
        check(countComponents(2, new int[][]{{0,1}}) == 2, "one-way edge");

        int[][] classic = {{0,1},{1,2},{2,0},{3,4},{4,3},{4,5}};
        check(join(stronglyConnectedComponents(6, classic)).equals("[0, 1, 2]|[3, 4]|[5]"), "classic components");
        check(countComponents(6, classic) == 3, "classic count");

        for (int trial = 0; trial < 120; trial++) {
            int[][] edges = randomEdges(6);
            check(join(stronglyConnectedComponents(6, edges)).equals(join(bruteForceComponents(6, edges))), "random " + trial);
            check(countComponents(6, edges) <= 6, "bounded " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
