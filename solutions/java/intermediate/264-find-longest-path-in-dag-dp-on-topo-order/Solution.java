/** Task 264 - Find longest path in DAG (DP on topo order). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * Longest path in an UNWEIGHTED DAG measured in EDGES (documented; no weighted
     * variant). Kahn's topological order feeds the DP: when u is popped, dist[u] is
     * final because all its predecessors are done; every edge u->v relaxes
     * dist[v] = max(dist[v], dist[u] + 1). Isolated vertices stay at 0.
     * A cyclic input raises IllegalArgumentException (a longest path in a cyclic graph
     * is ill-defined): Kahn's queue drains before every vertex is processed.
     * O(V + E) time and space.
     */
    static int longestPath(int n, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        int[] indeg = new int[n];
        for (int[] e : edges) {
            adj.get(e[0]).add(e[1]);
            indeg[e[1]]++;
        }
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (indeg[i] == 0) q.add(i);
        int[] dist = new int[n];
        int processed = 0;
        while (!q.isEmpty()) {
            int u = q.poll();
            processed++;
            for (int v : adj.get(u)) {
                if (dist[u] + 1 > dist[v]) dist[v] = dist[u] + 1;
                if (--indeg[v] == 0) q.add(v);
            }
        }
        if (processed != n) throw new IllegalArgumentException("graph contains a cycle");
        int best = 0;
        for (int d : dist) best = Math.max(best, d);
        return best;
    }

    /** Memoized DFS reference (valid on acyclic inputs); used only for cross-checks. */
    static int longestPathDfs(int n, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(e[1]);
        int[] memo = new int[n];
        Arrays.fill(memo, -1);
        int best = 0;
        for (int i = 0; i < n; i++) best = Math.max(best, depth(adj, memo, i));
        return best;
    }

    private static int depth(List<List<Integer>> adj, int[] memo, int u) {
        if (memo[u] >= 0) return memo[u];
        int best = 0;
        for (int v : adj.get(u)) best = Math.max(best, 1 + depth(adj, memo, v));
        memo[u] = best;
        return best;
    }

    public static void main(String[] args) {
        int[][] dag = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {2, 5}};
        check(longestPath(6, dag) == 3, "hand-verified DAG (0->2->3->4)");
        check(longestPath(8, dag) == 3, "disconnected vertices 6, 7");
        check(longestPath(5, new int[][]{{0, 1}, {1, 2}, {2, 3}, {3, 4}}) == 4, "linear chain");
        check(longestPath(1, new int[0][]) == 0, "single vertex");
        check(longestPath(0, new int[0][]) == 0, "no vertices");
        check(longestPath(5, new int[][]{{0, 1}, {1, 2}, {3, 4}}) == 2, "disconnected chains");
        check(longestPath(4, new int[][]{{0, 1}, {0, 2}, {1, 3}, {2, 3}}) == 2, "diamond");
        check(longestPath(3, new int[][]{{0, 2}, {2, 1}}) == 2, "topo order != index order");
        check(longestPath(4, new int[][]{{3, 2}, {2, 1}, {1, 0}}) == 3, "reversed chain");

        int[][][] cyclicCases = {
            {{0, 1}, {1, 2}, {2, 0}},
            {{0, 1}, {1, 2}, {2, 0}, {2, 3}},
        };
        for (int[][] bad : cyclicCases) {
            boolean threw = false;
            try { longestPath(bad.length == 3 ? 3 : 4, bad); }
            catch (IllegalArgumentException e) { threw = true; }
            check(threw, "cycle rejected");
        }

        Random rng = new Random(264);
        for (int test = 0; test < 300; test++) {        // cross-check vs the memoized-DFS oracle
            int n = rng.nextInt(13);
            List<int[]> edgesList = new ArrayList<>();
            for (int u = 0; u < n; u++)
                for (int v = u + 1; v < n; v++)
                    if (rng.nextDouble() < 0.25) edgesList.add(new int[]{u, v});
            int[][] edges = edgesList.toArray(new int[0][]);
            check(longestPath(n, edges) == longestPathDfs(n, edges), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
