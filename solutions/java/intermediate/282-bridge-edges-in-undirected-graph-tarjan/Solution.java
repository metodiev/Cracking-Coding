/** Task 282 - Bridge edges via Tarjan's disc/low values. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static List<List<int[]>> adjacency;
    private static int[] disc;
    private static int[] low;
    private static int timer;
    private static List<int[]> bridges;

    /** Returns bridges as canonical sorted (min, max) pairs. */
    static int[][] findBridges(int n, int[][] edges) {
        adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int eid = 0; eid < edges.length; eid++) {
            int a = edges[eid][0], b = edges[eid][1];
            if (a == b) continue;                              // self loops are never bridges
            adjacency.get(a).add(new int[]{b, eid});
            adjacency.get(b).add(new int[]{a, eid});
        }

        disc = new int[n];
        Arrays.fill(disc, -1);
        low = new int[n];
        timer = 0;
        bridges = new ArrayList<>();
        for (int v = 0; v < n; v++) if (disc[v] == -1) dfs(v, -1);

        bridges.sort((p, q) -> p[0] != q[0] ? Integer.compare(p[0], q[0]) : Integer.compare(p[1], q[1]));
        return bridges.toArray(new int[0][]);
    }

    private static void dfs(int u, int parentEdge) {
        disc[u] = low[u] = timer++;
        for (int[] step : adjacency.get(u)) {
            int v = step[0], eid = step[1];
            if (eid == parentEdge) continue;                   // skip only the arrival edge itself
            if (disc[v] != -1) {
                low[u] = Math.min(low[u], disc[v]);            // back edge to an ancestor
            } else {
                dfs(v, eid);
                low[u] = Math.min(low[u], low[v]);
                if (low[v] > disc[u]) {                        // v's subtree cannot escape above u
                    bridges.add(new int[]{Math.min(u, v), Math.max(u, v)});
                }
            }
        }
    }

    static int countComponents(int n, int[][] edges) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            if (edge[0] == edge[1]) continue;
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        boolean[] seen = new boolean[n];
        int total = 0;
        for (int s = 0; s < n; s++) {
            if (seen[s]) continue;
            total++;
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(s);
            seen[s] = true;
            while (!stack.isEmpty()) {
                int x = stack.pop();
                for (int y : graph.get(x)) {
                    if (!seen[y]) { seen[y] = true; stack.push(y); }
                }
            }
        }
        return total;
    }

    /** Oracle: deleting a bridge increases the number of connected components. */
    static int[][] bruteForceBridges(int n, int[][] edges) {
        int base = countComponents(n, edges);
        TreeSet<String> found = new TreeSet<>();
        for (int i = 0; i < edges.length; i++) {
            int[][] without = new int[edges.length - 1][];
            int k = 0;
            for (int j = 0; j < edges.length; j++) if (j != i) without[k++] = edges[j];
            if (countComponents(n, without) > base) {
                int a = Math.min(edges[i][0], edges[i][1]), b = Math.max(edges[i][0], edges[i][1]);
                found.add(a + "-" + b);
            }
        }
        int[][] result = new int[found.size()][];
        int k = 0;
        for (String key : found) {
            String[] parts = key.split("-");
            result[k++] = new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};
        }
        return result;
    }

    /** Simple graphs: each unordered pair {u, v} considered once. */
    static int[][] randomUndirectedGraph(int n, int[] seed) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % 3 == 0) edges.add(new int[]{u, v});
            }
        }
        return edges.toArray(new int[0][]);
    }

    /** Multigraphs: (u, v) and (v, u) may both appear as parallel edges. */
    static int[][] randomMultigraph(int n, int[] seed) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % 3 == 0) edges.add(new int[]{u, v});
            }
        }
        return edges.toArray(new int[0][]);
    }

    static String key(int[][] pairs) {
        List<String> parts = new ArrayList<>();
        for (int[] pair : pairs) parts.add(pair[0] + "-" + pair[1]);
        Collections.sort(parts);
        return String.join("|", parts);
    }

    public static void main(String[] args) {
        check(key(findBridges(3, new int[][]{{0, 1}, {1, 2}})).equals("0-1|1-2"), "path has 2 bridges");
        check(key(findBridges(3, new int[][]{{0, 1}, {1, 2}, {2, 0}})).equals(""), "triangle has none");
        check(key(findBridges(4, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}})).equals(""), "square has none");
        check(key(findBridges(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {2, 4}})).equals("2-4"), "square plus tail");
        check(key(findBridges(5, new int[][]{{0, 1}, {1, 2}, {3, 4}})).equals("0-1|1-2|3-4"), "disconnected");
        check(key(findBridges(3, new int[][]{{0, 1}, {0, 1}, {1, 2}})).equals("1-2"), "parallel edges");
        check(key(findBridges(2, new int[][]{{0, 0}, {0, 1}})).equals("0-1"), "self loop ignored");
        check(key(findBridges(3, new int[][]{{0, 1}})).equals("0-1"), "isolated vertex");

        int[] simpleSeed = {13579};
        int[] multiSeed = {13579};
        boolean simpleWithBridge = false, simpleWithoutBridge = false;
        for (int trial = 0; trial < 120; trial++) {
            int[][] edges = randomUndirectedGraph(6, simpleSeed);
            int[][] bridgesFound = findBridges(6, edges);
            check(key(bridgesFound).equals(key(bruteForceBridges(6, edges))), "simple " + trial);
            if (bridgesFound.length > 0) simpleWithBridge = true; else simpleWithoutBridge = true;
        }
        check(simpleWithBridge && simpleWithoutBridge, "simple coverage");

        boolean multiWithBridge = false, multiWithoutBridge = false;
        for (int trial = 0; trial < 120; trial++) {
            int[][] edges = randomMultigraph(6, multiSeed);
            int[][] bridgesFound = findBridges(6, edges);
            check(key(bridgesFound).equals(key(bruteForceBridges(6, edges))), "multi " + trial);
            if (bridgesFound.length > 0) multiWithBridge = true; else multiWithoutBridge = true;
        }
        check(multiWithBridge && multiWithoutBridge, "multigraph coverage");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
