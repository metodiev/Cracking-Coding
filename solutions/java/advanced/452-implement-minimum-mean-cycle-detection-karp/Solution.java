/** Task 452 - Implement minimum mean cycle detection (Karp). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final int INF = 1_000_000_000;

    static int gcd(int a, int b) {
        while (b != 0) {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    static int[] reduceFraction(int num, int den) {
        int g = gcd(Math.abs(num), den);
        return new int[]{num / g, den / g};
    }

    static boolean fracLess(int[] a, int[] b) {
        return a[0] * b[1] < b[0] * a[1];
    }

    static boolean fracGreater(int[] a, int[] b) {
        return a[0] * b[1] > b[0] * a[1];
    }

    /** Returns the min mean cycle weight as a reduced {num, den}, or null when acyclic. */
    static int[] karpMinMeanCycle(int n, int[][] edges) {
        int[][] d = new int[n + 1][n];
        for (int k = 0; k <= n; k++) Arrays.fill(d[k], INF);
        for (int v = 0; v < n; v++) d[0][v] = 0;
        for (int k = 1; k <= n; k++) {
            for (int[] e : edges) {
                int u = e[0], v = e[1], w = e[2];
                if (d[k - 1][u] < INF && d[k - 1][u] + w < d[k][v]) d[k][v] = d[k - 1][u] + w;
            }
        }
        int[] best = null;
        for (int v = 0; v < n; v++) {
            if (d[n][v] >= INF) continue;
            int[] local = null;
            for (int k = 0; k < n; k++) {
                if (d[k][v] >= INF) continue;
                int[] cand = reduceFraction(d[n][v] - d[k][v], n - k);
                if (local == null || fracGreater(cand, local)) local = cand;
            }
            if (best == null || fracLess(local, best)) best = local;
        }
        return best;
    }

    /** Minimum mean over all simple cycles by enumeration (each cycle from its min vertex). */
    static int[] bruteMinMeanCycle(int n, int[]... edges) {
        List<List<int[]>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(new int[]{e[1], e[2]});
        boolean[] hasBest = {false};
        int[] best = new int[2];
        for (int s = 0; s < n; s++) {
            List<Integer> path = new ArrayList<>();
            path.add(s);
            walk(adj, path, 0, s, hasBest, best);
        }
        return hasBest[0] ? best : null;
    }

    static void walk(List<List<int[]>> adj, List<Integer> path, int total, int start,
                     boolean[] hasBest, int[] best) {
        int cur = path.get(path.size() - 1);
        for (int[] e : adj.get(cur)) {
            int v = e[0], w = e[1];
            if (v == start) {
                int[] cand = reduceFraction(total + w, path.size());
                if (!hasBest[0] || fracLess(cand, best)) {
                    best[0] = cand[0];
                    best[1] = cand[1];
                    hasBest[0] = true;
                }
            } else if (v > start && !path.contains(v)) {
                path.add(v);
                walk(adj, path, total + w, start, hasBest, best);
                path.remove(path.size() - 1);
            }
        }
    }

    static class Graph {
        final int n;
        final int[][] edges;

        Graph(int n, int[][] edges) {
            this.n = n;
            this.edges = edges;
        }
    }

    static List<Graph> lcgDigraphs(int count, int seed, int minN, int maxN) {
        List<Graph> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = minN + seed % (maxN - minN + 1);
            List<int[]> edges = new ArrayList<>();
            for (int node = 0; node < n - 1; node++) {
                seed = (seed * 75 + 74) % 65537;
                edges.add(new int[]{node, node + 1, seed % 15 - 5});
            }
            seed = (seed * 75 + 74) % 65537;
            edges.add(new int[]{n - 1, 0, seed % 15 - 5});
            for (int k = 0; k < 2 * n; k++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                if (seed % 100 < 20) {
                    seed = (seed * 75 + 74) % 65537;
                    edges.add(new int[]{u, u, seed % 9 - 4});
                    continue;
                }
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.add(new int[]{u, v, seed % 15 - 5});
            }
            graphs.add(new Graph(n, edges.toArray(new int[0][])));
        }
        return graphs;
    }

    public static void main(String[] args) {
        int[] fourCycle = {1, 1};
        check(Arrays.equals(karpMinMeanCycle(4, new int[][]{{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}, {2, 0, 5}}), fourCycle),
              "4-cycle mean 1 beats shortcut");
        check(Arrays.equals(bruteMinMeanCycle(4, new int[][]{{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}, {2, 0, 5}}), fourCycle),
              "4-cycle brute");

        int[] negative = {-2, 1};
        check(Arrays.equals(karpMinMeanCycle(3, new int[][]{{0, 1, -2}, {1, 2, -1}, {2, 0, -3}}), negative), "negative cycle");
        check(Arrays.equals(bruteMinMeanCycle(3, new int[][]{{0, 1, -2}, {1, 2, -1}, {2, 0, -3}}), negative), "negative brute");

        int[] selfLoop = {7, 1};
        check(Arrays.equals(karpMinMeanCycle(1, new int[][]{{0, 0, 7}}), selfLoop), "self loop");
        check(Arrays.equals(bruteMinMeanCycle(1, new int[][]{{0, 0, 7}}), selfLoop), "self loop brute");

        int[] twoCycles = {-1, 1};
        check(Arrays.equals(karpMinMeanCycle(3, new int[][]{{0, 1, 2}, {1, 0, 2}, {1, 2, -1}, {2, 1, -1}}), twoCycles), "two cycles");
        check(Arrays.equals(bruteMinMeanCycle(3, new int[][]{{0, 1, 2}, {1, 0, 2}, {1, 2, -1}, {2, 1, -1}}), twoCycles), "two cycles brute");

        int[] half = {1, 2};
        check(Arrays.equals(karpMinMeanCycle(2, new int[][]{{0, 1, 1}, {1, 0, 0}}), half), "fraction 1/2");
        check(Arrays.equals(bruteMinMeanCycle(2, new int[][]{{0, 1, 1}, {1, 0, 0}}), half), "fraction brute");

        check(karpMinMeanCycle(3, new int[][]{{0, 1, 1}, {1, 2, 2}}) == null, "DAG karp");
        check(bruteMinMeanCycle(3, new int[][]{{0, 1, 1}, {1, 2, 2}}) == null, "DAG brute");

        int checked = 0;
        for (int seed : new int[]{4521, 4522, 4523, 4524}) {
            for (Graph graph : lcgDigraphs(5, seed, 3, 6)) {
                int n = graph.n;
                int[][] edges = graph.edges;
                int[] expected = bruteMinMeanCycle(n, edges);
                int[] got = karpMinMeanCycle(n, edges);
                check(expected != null, "LCG has cycle");
                check(Arrays.equals(got, expected), "LCG min mean " + Arrays.toString(got) + " vs " + Arrays.toString(expected));
                checked++;
            }
        }
        check(checked == 20, "graph count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
