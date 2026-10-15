/** Task 424 - Implement small-to-large merging on trees for frequency queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class FrequencySummary {
    static List<List<Integer>> childrenOf(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        List<List<Integer>> children = new ArrayList<>();
        for (int i = 0; i < n; i++) children.add(new ArrayList<>());
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
                    children.get(u).add(v);
                    order.add(v);
                }
            }
        }
        return children;
    }

    static int[][] compute(int n, int[][] edges, int[] colors) {
        List<List<Integer>> children = childrenOf(n, edges);
        List<Map<Integer, Integer>> counters = new ArrayList<>();
        for (int i = 0; i < n; i++) counters.add(null);
        int[] distinct = new int[n];
        int[] top = new int[n];
        int[] parent = new int[n];
        // rebuild the parent/preorder to process bottom-up
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
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
        for (int index = order.size() - 1; index >= 0; index--) {
            int u = order.get(index);
            int biggest = -1;
            for (int v : children.get(u)) {
                if (biggest == -1 || counters.get(v).size() > counters.get(biggest).size()) biggest = v;
            }
            Map<Integer, Integer> table;
            int best;
            if (biggest == -1) {
                table = new HashMap<>();
                table.put(colors[u], 1);
                best = 1;
            } else {
                table = counters.get(biggest);              // reuse the largest counter
                best = top[biggest];
                int amount = table.merge(colors[u], 1, Integer::sum);
                if (amount > best) best = amount;
                for (int v : children.get(u)) {
                    if (v == biggest) continue;
                    for (Map.Entry<Integer, Integer> entry : counters.get(v).entrySet()) {
                        int merged = table.merge(entry.getKey(), entry.getValue(), Integer::sum);
                        if (merged > best) best = merged;
                    }
                }
            }
            counters.set(u, table);
            distinct[u] = table.size();
            top[u] = best;
        }
        return new int[][]{distinct, top};
    }

    static int[][] naive(int n, int[][] edges, int[] colors) {
        List<List<Integer>> children = childrenOf(n, edges);
        int[] distinct = new int[n];
        int[] top = new int[n];
        for (int u = 0; u < n; u++) {
            Map<Integer, Integer> counts = new HashMap<>();
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(u);
            while (!stack.isEmpty()) {
                int node = stack.pop();
                counts.merge(colors[node], 1, Integer::sum);
                for (int v : children.get(node)) stack.push(v);
            }
            distinct[u] = counts.size();
            int best = 0;
            for (int amount : counts.values()) best = Math.max(best, amount);
            top[u] = best;
        }
        return new int[][]{distinct, top};
    }
}

public class Solution {
    public static void main(String[] args) {
        int[][] tree = {{0, 1}, {0, 2}, {1, 3}};
        int[] colors = {0, 1, 0, 1};
        int[][] summary = FrequencySummary.compute(4, tree, colors);
        check(Arrays.equals(summary[0], new int[]{2, 1, 1, 1}), "distinct counts");
        check(Arrays.equals(summary[1], new int[]{2, 2, 1, 1}), "top frequencies");
        int[][] expected = FrequencySummary.naive(4, tree, colors);
        check(Arrays.equals(summary[0], expected[0]) && Arrays.equals(summary[1], expected[1]), "small tree vs naive");

        int[][] path = new int[4][];
        for (int i = 0; i < 4; i++) path[i] = new int[]{i, i + 1};
        summary = FrequencySummary.compute(5, path, new int[]{1, 2, 2, 3, 3});
        check(Arrays.equals(summary[0], new int[]{3, 2, 2, 1, 1}), "path distinct");
        check(Arrays.equals(summary[1], new int[]{2, 2, 2, 2, 1}), "path top");
        expected = FrequencySummary.naive(5, path, new int[]{1, 2, 2, 3, 3});
        check(Arrays.equals(summary[0], expected[0]) && Arrays.equals(summary[1], expected[1]), "path vs naive");

        summary = FrequencySummary.compute(1, new int[][]{}, new int[]{7});
        check(Arrays.equals(summary[0], new int[]{1}) && Arrays.equals(summary[1], new int[]{1}), "single node");

        int[][] star = new int[4][];
        for (int i = 0; i < 4; i++) star[i] = new int[]{0, i + 1};
        summary = FrequencySummary.compute(5, star, new int[]{3, 3, 3, 3, 3});
        check(Arrays.equals(summary[0], new int[]{1, 1, 1, 1, 1}), "star distinct");
        check(Arrays.equals(summary[1], new int[]{5, 1, 1, 1, 1}), "star top");

        for (int seed : new int[]{424, 2024, 77777}) {
            int local = seed;
            for (int trial = 0; trial < 20; trial++) {
                local = (local * 75 + 74) % 65537;
                int n = 2 + local % 59;
                int[][] edges = new int[n - 1][];
                for (int node = 1; node < n; node++) {
                    local = (local * 75 + 74) % 65537;
                    edges[node - 1] = new int[]{local % node, node};
                }
                int[] palette = new int[n];
                for (int node = 0; node < n; node++) {
                    local = (local * 75 + 74) % 65537;
                    palette[node] = local % (2 + seed % 5);
                }
                int[][] got = FrequencySummary.compute(n, edges, palette);
                int[][] want = FrequencySummary.naive(n, edges, palette);
                check(Arrays.equals(got[0], want[0]) && Arrays.equals(got[1], want[1]),
                      "lcg trial " + trial + " n=" + n);
            }
        }

        int local = 909090;
        for (int trial = 0; trial < 10; trial++) {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 59;
            int[][] edges = new int[n - 1][];
            for (int node = 1; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                edges[node - 1] = new int[]{local % node, node};
            }
            int[] palette = new int[n];
            for (int node = 0; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                palette[node] = local % 1000;
            }
            int[][] got = FrequencySummary.compute(n, edges, palette);
            int[][] want = FrequencySummary.naive(n, edges, palette);
            check(Arrays.equals(got[0], want[0]) && Arrays.equals(got[1], want[1]), "wide palette " + n);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
