/** Task 450 - Implement stoer-wagner minimum cut algorithm. (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

class MinCutInfo {
    final int value;
    final int[] side;

    MinCutInfo(int value, int[] side) {
        this.value = value;
        this.side = side;
    }
}

class StoerWagner {
    static MinCutInfo minCut(int n, int[][] edges) {
        long[][] weight = new long[n][n];
        for (int[] e : edges) {
            if (e[0] == e[1]) continue;
            weight[e[0]][e[1]] += e[2];
            weight[e[1]][e[0]] += e[2];
        }
        List<Integer> vertices = new ArrayList<>();
        for (int i = 0; i < n; i++) vertices.add(i);
        List<List<Integer>> groups = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> group = new ArrayList<>();
            group.add(i);
            groups.add(group);
        }
        long best = Long.MAX_VALUE;
        List<Integer> bestSide = null;
        while (vertices.size() > 1) {
            Set<Integer> remaining = new LinkedHashSet<>(vertices);
            Map<Integer, Long> connection = new HashMap<>();
            for (int v : vertices) connection.put(v, 0L);
            List<Integer> order = new ArrayList<>();
            while (!remaining.isEmpty()) {
                int pick = -1;
                for (int v : remaining) {
                    if (pick == -1) pick = v;
                    else if (connection.get(v) > connection.get(pick)) pick = v;
                }
                remaining.remove(pick);
                order.add(pick);
                for (int other : remaining) connection.put(other, connection.get(other) + weight[pick][other]);
            }
            int t = order.get(order.size() - 1);
            int s = order.get(order.size() - 2);
            long cut = connection.get(t);
            if (cut < best) {
                best = cut;
                bestSide = new ArrayList<>(groups.get(t));
            }
            for (int v : vertices) {
                if (v != s && v != t) {
                    weight[s][v] += weight[t][v];
                    weight[v][s] = weight[s][v];
                }
            }
            for (int v : vertices) {
                weight[t][v] = 0;
                weight[v][t] = 0;
            }
            vertices.remove(Integer.valueOf(t));
            groups.get(s).addAll(groups.get(t));
        }
        int[] side = new int[bestSide.size()];
        for (int i = 0; i < side.length; i++) side[i] = bestSide.get(i);
        Arrays.sort(side);
        return new MinCutInfo((int) best, side);
    }
}

public class Solution {
    static int bruteMinCut(int n, int[][] edges) {
        int best = Integer.MAX_VALUE;
        for (int mask = 1; mask < (1 << (n - 1)); mask++) {
            int cut = 0;
            for (int[] e : edges) {
                int bu = e[0] > 0 ? (mask >> (e[0] - 1)) & 1 : 0;
                int bv = e[1] > 0 ? (mask >> (e[1] - 1)) & 1 : 0;
                if (bu != bv) cut += e[2];
            }
            best = Math.min(best, cut);
        }
        return best;
    }

    static int crossingWeight(int[][] edges, int[] side) {
        boolean[] onSide = new boolean[64];
        for (int v : side) onSide[v] = true;
        int total = 0;
        for (int[] e : edges) if (onSide[e[0]] != onSide[e[1]]) total += e[2];
        return total;
    }

    private static int seed;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<Object[]> lcgGraphs(int count, int maxN, boolean connected, int baseSeed) {
        seed = baseSeed;
        List<Object[]> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int n = 2 + seed % (maxN - 1);
            List<int[]> edges = new ArrayList<>();
            int start = connected ? 1 : 0;
            for (int node = start; node < n; node++) {
                int parent = node > 0 ? nextSeed() % node : 0;
                int w = 1 + nextSeed() % 20;
                edges.add(new int[]{node, parent, w});
            }
            for (int e = 0; e < n + 2; e++) {
                int u = nextSeed() % n;
                int v = nextSeed() % n;
                int w = 1 + nextSeed() % 20;
                if (u != v) edges.add(new int[]{u, v, w});
            }
            graphs.add(new Object[]{n, edges.toArray(new int[0][])});
        }
        return graphs;
    }

    public static void main(String[] args) {
        int[][] triangle = {{0, 1, 1}, {1, 2, 1}, {0, 2, 1}};
        int[][] k4 = new int[6][];
        int idx = 0;
        for (int u = 0; u < 4; u++) {
            for (int v = u + 1; v < 4; v++) k4[idx++] = new int[]{u, v, 1};
        }
        check(bruteMinCut(3, triangle) == 2, "triangle brute");
        check(bruteMinCut(4, k4) == 3, "K4 brute");
        MinCutInfo info = StoerWagner.minCut(3, triangle);
        check(info.value == 2, "triangle value");
        check(crossingWeight(triangle, info.side) == 2, "triangle witness");
        info = StoerWagner.minCut(4, k4);
        check(info.value == 3, "K4 value");
        check(crossingWeight(k4, info.side) == 3, "K4 witness");

        int[][] cycle4 = {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}};
        check(StoerWagner.minCut(4, cycle4).value == 2, "cycle value");
        check(bruteMinCut(4, cycle4) == 2, "cycle brute");

        int[][] disjoint = {{0, 1, 5}, {1, 2, 5}, {0, 2, 5}, {3, 4, 7}, {4, 5, 7}, {3, 5, 7}};
        check(bruteMinCut(6, disjoint) == 0, "disjoint brute");
        info = StoerWagner.minCut(6, disjoint);
        check(info.value == 0, "disjoint value");
        check(crossingWeight(disjoint, info.side) == 0, "disjoint witness");

        int[][] parallel = {{0, 1, 3}, {1, 0, 4}, {0, 1, 5}};
        check(StoerWagner.minCut(2, parallel).value == 12, "parallel value");
        int[][] weighted = {{0, 1, 5}, {1, 2, 5}, {0, 2, 5}};
        check(StoerWagner.minCut(3, weighted).value == 10, "weighted triangle");

        int checked = 0;
        for (boolean connected : new boolean[]{true, false}) {
            for (Object[] graph : lcgGraphs(8, 8, connected, connected ? 450 : 451)) {
                int n = (Integer) graph[0];
                int[][] edges = (int[][]) graph[1];
                int expected = bruteMinCut(n, edges);
                MinCutInfo result = StoerWagner.minCut(n, edges);
                check(crossingWeight(edges, result.side) == result.value, "witness value");
                check(result.value == expected, "LCG value n=" + n);
                checked++;
            }
        }
        for (int extra : new int[]{452, 453, 454, 455, 456, 457}) {
            for (Object[] graph : lcgGraphs(3, 7, true, extra)) {
                int n = (Integer) graph[0];
                int[][] edges = (int[][]) graph[1];
                check(StoerWagner.minCut(n, edges).value == bruteMinCut(n, edges), "extra seed " + extra);
                checked++;
            }
        }
        check(checked == 34, "coverage count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
