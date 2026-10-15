/** Task 447 - Implement dynamic MST under edge insertions (conceptual). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

class DynamicMST {
    private final List<int[]> edges = new ArrayList<>();
    private final List<int[]> treeEdges = new ArrayList<>();
    int swaps;

    void addEdge(int u, int v, int w) {
        edges.add(new int[]{u, v, w});
        int[] heaviest = heaviestOnPath(u, v);
        if (heaviest == null) {
            treeEdges.add(new int[]{u, v, w});
        } else if (heaviest[2] > w) {
            treeEdges.remove(heaviest);
            treeEdges.add(new int[]{u, v, w});
            swaps++;
        }
    }

    private int[] heaviestOnPath(int u, int v) {
        if (u == v) return null;
        Map<Integer, List<int[]>> adjacency = new HashMap<>();
        for (int i = 0; i < treeEdges.size(); i++) {
            int[] e = treeEdges.get(i);
            adjacency.computeIfAbsent(e[0], key -> new ArrayList<>()).add(new int[]{e[1], i});
            adjacency.computeIfAbsent(e[1], key -> new ArrayList<>()).add(new int[]{e[0], i});
        }
        if (!adjacency.containsKey(u)) return null;
        Map<Integer, int[]> parent = new HashMap<>();
        parent.put(u, new int[]{-1, -1});
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(u);
        while (!stack.isEmpty()) {
            int x = stack.pop();
            for (int[] pair : adjacency.getOrDefault(x, Collections.emptyList())) {
                if (!parent.containsKey(pair[0])) {
                    parent.put(pair[0], new int[]{x, pair[1]});
                    stack.push(pair[0]);
                }
            }
        }
        if (!parent.containsKey(v)) return null;
        int[] heaviest = null;
        int node = v;
        while (parent.get(node)[0] != -1) {
            int[] info = parent.get(node);
            int[] edge = treeEdges.get(info[1]);
            if (heaviest == null || edge[2] > heaviest[2]) heaviest = edge;
            node = info[0];
        }
        return heaviest;
    }

    long mstWeight() {
        long total = 0;
        for (int[] e : treeEdges) total += e[2];
        return total;
    }
}

public class Solution {
    private static int seed;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static long kruskalWeight(int n, List<int[]> edges) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        List<int[]> sorted = new ArrayList<>(edges);
        sorted.sort(Comparator.comparingInt(e -> e[2]));
        long total = 0;
        for (int[] e : sorted) {
            int ru = find(parent, e[0]);
            int rv = find(parent, e[1]);
            if (ru != rv) {
                parent[ru] = rv;
                total += e[2];
            }
        }
        return total;
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static List<int[]> lcgEdgeStream(int count, int n, int baseSeed) {
        seed = baseSeed;
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int u = nextSeed() % n;
            int v = nextSeed() % n;
            if (u == v) v = (v + 1) % n;
            int w = 1 + nextSeed() % 30;
            edges.add(new int[]{u, v, w});
        }
        return edges;
    }

    public static void main(String[] args) {
        DynamicMST dynamic = new DynamicMST();
        dynamic.addEdge(0, 1, 5);
        check(dynamic.mstWeight() == 5, "first edge");
        dynamic.addEdge(1, 2, 3);
        check(dynamic.mstWeight() == 8, "second edge");
        dynamic.addEdge(2, 3, 7);
        check(dynamic.mstWeight() == 15, "third edge");
        dynamic.addEdge(0, 3, 1);
        check(dynamic.mstWeight() == 9, "swap in lighter edge");
        check(dynamic.swaps == 1, "swap counted");
        dynamic.addEdge(0, 2, 9);
        check(dynamic.mstWeight() == 9, "heavier edge ignored");

        int swapsSeen = 0;
        for (int trial = 0; trial < 25; trial++) {
            int n = 4 + trial % 5;
            List<int[]> stream = lcgEdgeStream(2 * n, n, 900 + trial * 37);
            dynamic = new DynamicMST();
            List<int[]> inserted = new ArrayList<>();
            for (int[] edge : stream) {
                dynamic.addEdge(edge[0], edge[1], edge[2]);
                inserted.add(edge);
                check(dynamic.mstWeight() == kruskalWeight(n, inserted), "trial " + trial);
            }
            swapsSeen += dynamic.swaps;
        }
        check(swapsSeen > 0, "swap rule exercised");

        int n = 6;
        List<int[]> stream = lcgEdgeStream(3 * n, n, 321);
        dynamic = new DynamicMST();
        for (int[] edge : stream) dynamic.addEdge(edge[0], edge[1], edge[2]);
        check(dynamic.mstWeight() == kruskalWeight(n, stream), "final weight");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
