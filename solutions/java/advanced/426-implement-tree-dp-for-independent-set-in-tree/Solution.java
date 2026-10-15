/** Task 426 - Implement tree DP for independent set in tree. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class IndependentSet {
    /** Returns (best weight, chosen flag array). Ties prefer the untaken option. */
    static Object[] solve(int n, int[][] edges, int[] weights) {
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
        List<List<Integer>> children = new ArrayList<>();
        for (int i = 0; i < n; i++) children.add(new ArrayList<>());
        for (int index = 1; index < order.size(); index++) {
            int u = order.get(index);
            children.get(parent[u]).add(u);
        }
        long[] take = new long[n];
        long[] skip = new long[n];
        for (int index = order.size() - 1; index >= 0; index--) {
            int u = order.get(index);
            long chosen = weights[u];
            long notChosen = 0;
            for (int v : children.get(u)) {
                chosen += skip[v];
                notChosen += Math.max(take[v], skip[v]);
            }
            take[u] = chosen;
            skip[u] = notChosen;
        }
        boolean[] picked = new boolean[n];
        Deque<Object[]> stack = new ArrayDeque<>();
        stack.push(new Object[]{0, take[0] > skip[0]});
        while (!stack.isEmpty()) {
            Object[] frame = stack.pop();
            int u = (Integer) frame[0];
            boolean taking = (Boolean) frame[1];
            picked[u] = taking;
            for (int v : children.get(u)) {
                stack.push(new Object[]{v, Boolean.valueOf(taking ? false : take[v] > skip[v])});
            }
        }
        long total = 0;
        for (int u = 0; u < n; u++) {
            if (picked[u]) total += weights[u];
        }
        return new Object[]{total, picked};
    }

    static boolean valid(int n, int[][] edges, boolean[] picked) {
        for (int[] edge : edges) {
            if (picked[edge[0]] && picked[edge[1]]) return false;
        }
        return true;
    }

    static long bruteForce(int n, int[][] edges, int[] weights) {
        long best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            long weight = 0;
            for (int i = 0; i < n; i++) {
                if ((mask >> i & 1) != 0) weight += weights[i];
            }
            if (weight <= best) continue;
            boolean ok = true;
            for (int[] edge : edges) {
                if ((mask >> edge[0] & 1) != 0 && (mask >> edge[1] & 1) != 0) {
                    ok = false;
                    break;
                }
            }
            if (ok) best = weight;
        }
        return best;
    }
}

public class Solution {
    public static void main(String[] args) {
        Object[] result = IndependentSet.solve(3, new int[][]{{0, 1}, {1, 2}}, new int[]{10, 1, 10});
        check((Long) result[0] == 20 && Arrays.equals((boolean[]) result[1], new boolean[]{true, false, true}), "path3");

        result = IndependentSet.solve(3, new int[][]{{0, 1}, {0, 2}}, new int[]{5, 4, 4});
        check((Long) result[0] == 8 && Arrays.equals((boolean[]) result[1], new boolean[]{false, true, true}), "star");

        result = IndependentSet.solve(2, new int[][]{{0, 1}}, new int[]{-1, -2});
        check((Long) result[0] == 0 && Arrays.equals((boolean[]) result[1], new boolean[]{false, false}), "negative pair");

        result = IndependentSet.solve(1, new int[][]{}, new int[]{7});
        check((Long) result[0] == 7 && Arrays.equals((boolean[]) result[1], new boolean[]{true}), "single positive");
        result = IndependentSet.solve(1, new int[][]{}, new int[]{-7});
        check((Long) result[0] == 0 && Arrays.equals((boolean[]) result[1], new boolean[]{false}), "single negative");

        result = IndependentSet.solve(4, new int[][]{{0, 1}, {1, 2}, {2, 3}}, new int[]{3, 10, 3, 10});
        check((Long) result[0] == 20, "path4 weight");
        check(Arrays.equals((boolean[]) result[1], new boolean[]{false, true, false, true}), "path4 choice");

        int local = 426426;
        for (int trial = 0; trial < 15; trial++) {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 15;
            int[][] edges = new int[n - 1][];
            int[] weights = new int[n];
            for (int node = 0; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                weights[node] = local % 21 - 10;
                if (node > 0) {
                    local = (local * 75 + 74) % 65537;
                    edges[node - 1] = new int[]{local % node, node};
                }
            }
            Object[] got = IndependentSet.solve(n, edges, weights);
            long expected = IndependentSet.bruteForce(n, edges, weights);
            check((Long) got[0] == expected, "small brute trial " + trial);
            check(IndependentSet.valid(n, edges, (boolean[]) got[1]), "small validity " + trial);
            long weight = 0;
            boolean[] picked = (boolean[]) got[1];
            for (int u = 0; u < n; u++) {
                if (picked[u]) weight += weights[u];
            }
            check(weight == (Long) got[0], "small reconstruction weight " + trial);
        }

        local = 99999;
        for (int trial = 0; trial < 10; trial++) {
            local = (local * 75 + 74) % 65537;
            int n = 20 + local % 41;
            int[][] edges = new int[n - 1][];
            int[] weights = new int[n];
            for (int node = 0; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                weights[node] = local % 41 - 20;
                if (node > 0) {
                    local = (local * 75 + 74) % 65537;
                    edges[node - 1] = new int[]{local % node, node};
                }
            }
            Object[] got = IndependentSet.solve(n, edges, weights);
            check(IndependentSet.valid(n, edges, (boolean[]) got[1]), "large validity " + trial);
            long weight = 0;
            boolean[] picked = (boolean[]) got[1];
            for (int u = 0; u < n; u++) {
                if (picked[u]) weight += weights[u];
            }
            check(weight == (Long) got[0], "large reconstruction weight " + trial);
            check((Long) got[0] >= 0, "large non-negative " + trial);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
