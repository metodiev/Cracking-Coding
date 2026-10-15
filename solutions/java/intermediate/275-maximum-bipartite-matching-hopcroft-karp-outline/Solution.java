/** Task 275 - Maximum bipartite matching (Hopcroft-Karp). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static int[] pairU, pairV, dist;
    private static List<List<Integer>> adjacency;

    /** Returns {matching size, pairs...} flattened: [0] is the size, then u,v pairs. */
    static int[] hopcroftKarp(int nl, List<List<Integer>> adj, int nr) {
        adjacency = adj;
        pairU = new int[nl];
        pairV = new int[nr];
        dist = new int[nl];
        Arrays.fill(pairU, -1);
        Arrays.fill(pairV, -1);

        int matching = 0;
        while (bfs(nl)) {
            for (int u = 0; u < nl; u++) {
                if (pairU[u] == -1 && dfs(u)) matching++;
            }
        }
        int count = 0;
        for (int u = 0; u < nl; u++) if (pairU[u] != -1) count++;
        int[] result = new int[1 + 2 * count];
        result[0] = matching;
        int idx = 1;
        for (int u = 0; u < nl; u++) {
            if (pairU[u] != -1) {
                result[idx++] = u;
                result[idx++] = pairU[u];
            }
        }
        return result;
    }

    private static boolean bfs(int nl) {
        Deque<Integer> queue = new ArrayDeque<>();
        for (int u = 0; u < nl; u++) {
            if (pairU[u] == -1) { dist[u] = 0; queue.add(u); }
            else dist[u] = -1;
        }
        boolean reachable = false;
        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int v : adjacency.get(u)) {
                int w = pairV[v];
                if (w == -1) reachable = true;
                else if (dist[w] == -1) { dist[w] = dist[u] + 1; queue.add(w); }
            }
        }
        return reachable;
    }

    private static boolean dfs(int u) {
        for (int v : adjacency.get(u)) {
            int w = pairV[v];
            if (w == -1 || (dist[w] == dist[u] + 1 && dfs(w))) {
                pairU[u] = v;
                pairV[v] = u;
                return true;
            }
        }
        dist[u] = -1;
        return false;
    }

    /** Simple O(V*E) augmenting-path oracle. */
    static int kuhn(int nl, List<List<Integer>> adj, int nr) {
        int[] matchV = new int[nr];
        Arrays.fill(matchV, -1);
        int count = 0;
        for (int u = 0; u < nl; u++) {
            if (tryKuhn(u, adj, matchV, new boolean[nr])) count++;
        }
        return count;
    }

    private static boolean tryKuhn(int u, List<List<Integer>> adj, int[] matchV, boolean[] visited) {
        for (int v : adj.get(u)) {
            if (!visited[v]) {
                visited[v] = true;
                if (matchV[v] == -1 || tryKuhn(matchV[v], adj, matchV, visited)) {
                    matchV[v] = u;
                    return true;
                }
            }
        }
        return false;
    }

    static boolean isValidMatching(int[] result, List<List<Integer>> adj, int nl, int nr) {
        Set<Integer> lefts = new HashSet<>();
        Set<Integer> rights = new HashSet<>();
        for (int i = 1; i + 1 < result.length; i += 2) {
            int u = result[i], v = result[i + 1];
            if (u < 0 || u >= nl || v < 0 || v >= nr) return false;
            if (!lefts.add(u) || !rights.add(v)) return false;
            if (!adj.get(u).contains(v)) return false;
        }
        return (result.length - 1) / 2 == result[0];
    }

    static List<List<Integer>> edges(int nl, int[][] pairs) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < nl; i++) adj.add(new ArrayList<>());
        for (int[] pair : pairs) adj.get(pair[0]).add(pair[1]);
        return adj;
    }

    public static void main(String[] args) {
        check(hopcroftKarp(3, edges(3, new int[][]{{0,0},{0,1},{0,2},{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}}), 3)[0] == 3, "K3,3");
        check(hopcroftKarp(3, edges(3, new int[][]{}), 3)[0] == 0, "no edges");
        check(hopcroftKarp(3, edges(3, new int[][]{{0,0},{0,1},{1,0}}), 2)[0] == 2, "three lefts two rights");
        check(hopcroftKarp(3, edges(3, new int[][]{{0,0},{1,0}}), 1)[0] == 1, "contention for one right");
        check(hopcroftKarp(3, edges(3, new int[][]{{0,0},{1,0},{1,1},{2,1}}), 2)[0] == 2, "chain");
        check(hopcroftKarp(3, edges(3, new int[][]{{0,0},{0,1},{1,1},{2,2}}), 3)[0] == 3, "distinct rights");
        check(hopcroftKarp(2, edges(2, new int[][]{{0,0},{0,1},{1,0}}), 2)[0] == 2, "greedy trap");
        check(hopcroftKarp(2, edges(2, new int[][]{{0,0},{0,1},{0,2},{0,3},{0,4},{1,0}}), 5)[0] == 2, "unbalanced");

        List<List<Integer>> k33 = edges(3, new int[][]{{0,0},{0,1},{0,2},{1,0},{1,1},{1,2},{2,0},{2,1},{2,2}});
        int[] result = hopcroftKarp(3, k33, 3);
        check(isValidMatching(result, k33, 3, 3), "matching validity");

        int seed = 12345;
        for (int trial = 0; trial < 200; trial++) {
            List<List<Integer>> adj = new ArrayList<>();
            for (int u = 0; u < 6; u++) adj.add(new ArrayList<>());
            for (int u = 0; u < 6; u++) {
                for (int v = 0; v < 6; v++) {
                    seed = (seed * 75 + 74) % 65537;
                    if (seed % 3 == 0) adj.get(u).add(v);
                }
            }
            int[] hk = hopcroftKarp(6, adj, 6);
            check(isValidMatching(hk, adj, 6, 6), "random matching validity " + trial);
            check(hk[0] == kuhn(6, adj, 6), "random agreement " + trial);
            check(hk[0] <= 6, "bounded " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
