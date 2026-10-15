/** Task 283 - Articulation points (cut vertices). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static List<List<Integer>> adjacency;
    private static int[] disc;
    private static int[] low;
    private static int timer;
    private static TreeSet<Integer> points;

    /** Returns cut vertices in increasing order. */
    static int[] articulationPoints(int n, int[][] edges) {
        adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }

        disc = new int[n];
        Arrays.fill(disc, -1);
        low = new int[n];
        timer = 0;
        points = new TreeSet<>();
        for (int v = 0; v < n; v++) if (disc[v] == -1) dfs(v, -1);

        int[] result = new int[points.size()];
        int k = 0;
        for (int p : points) result[k++] = p;
        return result;
    }

    private static void dfs(int u, int parent) {
        disc[u] = low[u] = timer++;
        int children = 0;
        for (int v : adjacency.get(u)) {
            if (disc[v] == -1) {                       // tree edge
                children++;
                dfs(v, u);
                low[u] = Math.min(low[u], low[v]);
                if (parent != -1 && low[v] >= disc[u]) points.add(u); // v's subtree cannot escape
            } else if (v != parent) {                  // back edge to an ancestor
                low[u] = Math.min(low[u], disc[v]);
            }
        }
        if (parent == -1 && children > 1) points.add(u); // root special case
    }

    static int countComponents(int n, int[][] edges, int skipped) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) graph.add(new ArrayList<>());
        for (int[] edge : edges) {
            if (edge[0] == skipped || edge[1] == skipped) continue;
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        boolean[] seen = new boolean[n];
        int total = 0;
        for (int s = 0; s < n; s++) {
            if (s == skipped || seen[s]) continue;
            total++;
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(s);
            seen[s] = true;
            while (!stack.isEmpty()) {
                int x = stack.pop();
                for (int y : graph.get(x)) if (!seen[y]) { seen[y] = true; stack.push(y); }
            }
        }
        return total;
    }

    /** Oracle: remove each vertex and count connected components. */
    static int[] bruteForceArticulationPoints(int n, int[][] edges) {
        int base = countComponents(n, edges, -1);
        List<Integer> result = new ArrayList<>();
        for (int v = 0; v < n; v++) if (countComponents(n, edges, v) > base) result.add(v);
        int[] array = new int[result.size()];
        for (int i = 0; i < array.length; i++) array[i] = result.get(i);
        return array;
    }

    /** Each unordered pair {u, v} appears at most once; edge when seed % divisor == 0. */
    static int[][] randomSimpleGraph(int n, int divisor, int[] seed) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = u + 1; v < n; v++) {
                seed[0] = (seed[0] * 75 + 74) % 65537;
                if (seed[0] % divisor == 0) edges.add(new int[]{u, v});
            }
        }
        return edges.toArray(new int[0][]);
    }

    static String key(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int value : values) builder.append(value).append(',');
        return builder.toString();
    }

    public static void main(String[] args) {
        check(key(articulationPoints(4, new int[][]{{0, 1}, {1, 2}, {2, 3}})).equals("1,2,"), "path middles");
        check(key(articulationPoints(4, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 0}})).equals(""), "cycle has none");
        check(key(articulationPoints(5, new int[][]{{0, 1}, {0, 2}, {0, 3}, {0, 4}})).equals("0,"), "star center");
        check(key(articulationPoints(3, new int[][]{{0, 1}, {0, 2}})).equals("0,"), "root with 2 children");
        check(key(articulationPoints(6, new int[][]{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}})).equals(""), "two triangles");
        check(key(articulationPoints(4, new int[][]{{0, 1}, {1, 2}, {2, 0}, {2, 3}})).equals("2,"), "triangle plus tail");
        check(key(articulationPoints(2, new int[][]{{0, 1}})).equals(""), "single edge");
        check(key(articulationPoints(3, new int[][]{})).equals(""), "isolated vertices");
        check(key(articulationPoints(3, new int[][]{{0, 1}, {0, 1}, {1, 2}})).equals("1,"), "parallel edges");
        check(key(articulationPoints(7, new int[][]{{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {2, 6}})).equals("2,"), "two cycles linked");

        for (int divisor : new int[]{3, 5}) {
            int[] seed = {13579};
            boolean seenCut = false, seenNone = false;
            for (int trial = 0; trial < 120; trial++) {
                int[][] edges = randomSimpleGraph(6, divisor, seed);
                int[] found = articulationPoints(6, edges);
                check(key(found).equals(key(bruteForceArticulationPoints(6, edges))), "random " + divisor + " " + trial);
                if (found.length > 0) seenCut = true; else seenNone = true;
            }
            check(seenCut && seenNone, "coverage " + divisor);
        }

        int[][] pathEdges = new int[9][];
        for (int i = 0; i < 9; i++) pathEdges[i] = new int[]{i, i + 1};
        StringBuilder expected = new StringBuilder();
        for (int i = 1; i <= 8; i++) expected.append(i).append(',');
        check(key(articulationPoints(10, pathEdges)).equals(expected.toString()), "long path interior");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
