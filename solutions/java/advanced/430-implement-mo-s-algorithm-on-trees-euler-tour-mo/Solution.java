/** Task 430 - Implement Mo's algorithm on trees (Euler tour + Mo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class EulerTour {
    final int[] tin;
    final int[] tout;
    final int[] euler;
    final List<List<Integer>> children;

    EulerTour(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        children = new ArrayList<>();
        for (int i = 0; i < n; i++) children.add(new ArrayList<>());
        tin = new int[n];
        tout = new int[n];
        euler = new int[n];
        dfs(0, -1, adjacency, new int[]{0});
    }

    private void dfs(int u, int parent, List<List<Integer>> adjacency, int[] timer) {
        tin[u] = timer[0];
        euler[timer[0]] = u;
        timer[0]++;
        for (int v : adjacency.get(u)) {
            if (v != parent) {
                children.get(u).add(v);
                dfs(v, u, adjacency, timer);
            }
        }
        tout[u] = timer[0] - 1;
    }
}

class MoSubtreeQueries {
    static int[] subtreeDistinctCounts(int n, int[][] edges, int[] colors, int[] queries) {
        EulerTour tour = new EulerTour(n, edges);
        int[] tin = tour.tin;
        int[] tout = tour.tout;
        int[] euler = tour.euler;
        int block = Math.max(1, (int) Math.sqrt(n));
        Integer[] order = new Integer[queries.length];
        for (int i = 0; i < queries.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int slotA = tin[queries[a]] / block;
            int slotB = tin[queries[b]] / block;
            if (slotA != slotB) return slotA - slotB;
            int rightA = slotA % 2 == 0 ? tout[queries[a]] : -tout[queries[a]];
            int rightB = slotB % 2 == 0 ? tout[queries[b]] : -tout[queries[b]];
            return rightA - rightB;
        });
        Map<Integer, Integer> counts = new HashMap<>();
        int distinct = 0;
        int currentLeft = 0;
        int currentRight = -1;
        int[] answers = new int[queries.length];
        for (int index : order) {
            int left = tin[queries[index]];
            int right = tout[queries[index]];
            while (currentLeft > left) {
                currentLeft--;
                int color = colors[euler[currentLeft]];
                int amount = counts.merge(color, 1, Integer::sum);
                if (amount == 1) distinct++;
            }
            while (currentRight < right) {
                currentRight++;
                int color = colors[euler[currentRight]];
                int amount = counts.merge(color, 1, Integer::sum);
                if (amount == 1) distinct++;
            }
            while (currentLeft < left) {
                int color = colors[euler[currentLeft]];
                int amount = counts.merge(color, -1, Integer::sum);
                if (amount == 0) distinct--;
                currentLeft++;
            }
            while (currentRight > right) {
                int color = colors[euler[currentRight]];
                int amount = counts.merge(color, -1, Integer::sum);
                if (amount == 0) distinct--;
                currentRight--;
            }
            answers[index] = distinct;
        }
        return answers;
    }

    static int[] naiveSubtreeDistinct(int n, int[][] edges, int[] colors, int[] queries) {
        EulerTour tour = new EulerTour(n, edges);
        int[] answers = new int[queries.length];
        for (int i = 0; i < queries.length; i++) {
            Set<Integer> seen = new HashSet<>();
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(queries[i]);
            while (!stack.isEmpty()) {
                int node = stack.pop();
                seen.add(colors[node]);
                for (int v : tour.children.get(node)) stack.push(v);
            }
            answers[i] = seen.size();
        }
        return answers;
    }
}

public class Solution {
    public static void main(String[] args) {
        int[][] edges = {{0, 1}, {0, 2}, {1, 3}};
        int[] colors = {0, 1, 0, 1};
        check(Arrays.equals(MoSubtreeQueries.subtreeDistinctCounts(4, edges, colors, new int[]{0, 1, 2, 3}),
                            new int[]{2, 1, 1, 1}), "ascending queries");
        check(Arrays.equals(MoSubtreeQueries.subtreeDistinctCounts(4, edges, colors, new int[]{3, 2, 1, 0}),
                            new int[]{1, 1, 1, 2}), "descending queries");
        check(Arrays.equals(MoSubtreeQueries.subtreeDistinctCounts(4, edges, colors, new int[]{2, 0, 2, 0}),
                            new int[]{1, 2, 1, 2}), "repeated queries");
        check(Arrays.equals(MoSubtreeQueries.subtreeDistinctCounts(4, edges, new int[]{5, 5, 5, 5}, new int[]{0, 1, 2, 3}),
                            new int[]{1, 1, 1, 1}), "uniform colours");
        check(Arrays.equals(MoSubtreeQueries.subtreeDistinctCounts(1, new int[][]{}, new int[]{9}, new int[]{0}),
                            new int[]{1}), "single node");

        for (int seed : new int[]{430, 24680, 13579}) {
            int local = seed;
            for (int trial = 0; trial < 15; trial++) {
                local = (local * 75 + 74) % 65537;
                int n = 2 + local % 49;
                int[][] treeEdges = new int[n - 1][];
                for (int node = 1; node < n; node++) {
                    local = (local * 75 + 74) % 65537;
                    treeEdges[node - 1] = new int[]{local % node, node};
                }
                int[] palette = new int[n];
                for (int node = 0; node < n; node++) {
                    local = (local * 75 + 74) % 65537;
                    palette[node] = local % (2 + seed % 6);
                }
                int[] queries = new int[n + 12];
                for (int i = 0; i < n; i++) queries[i] = i;
                for (int i = n; i < queries.length; i++) {
                    local = (local * 75 + 74) % 65537;
                    queries[i] = local % n;
                }
                int[] got = MoSubtreeQueries.subtreeDistinctCounts(n, treeEdges, palette, queries);
                int[] want = MoSubtreeQueries.naiveSubtreeDistinct(n, treeEdges, palette, queries);
                check(Arrays.equals(got, want), "lcg trial " + trial + " n=" + n);
            }
        }

        int local = 123123;
        for (int trial = 0; trial < 10; trial++) {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 49;
            int[][] treeEdges = new int[n - 1][];
            for (int node = 1; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                treeEdges[node - 1] = new int[]{local % node, node};
            }
            int[] palette = new int[n];
            for (int node = 0; node < n; node++) {
                local = (local * 75 + 74) % 65537;
                palette[node] = local % 1000;
            }
            int[] queries = new int[n];
            for (int i = 0; i < n; i++) queries[i] = i;
            int[] got = MoSubtreeQueries.subtreeDistinctCounts(n, treeEdges, palette, queries);
            int[] want = MoSubtreeQueries.naiveSubtreeDistinct(n, treeEdges, palette, queries);
            check(Arrays.equals(got, want), "wide palette " + n);
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
