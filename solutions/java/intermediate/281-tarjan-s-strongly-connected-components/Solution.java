/** Task 281 - Tarjan's strongly connected components in one DFS pass. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static List<List<Integer>> adjacency;
    private static int[] index;
    private static int[] low;
    private static boolean[] onStack;
    private static Deque<Integer> stack;
    private static List<List<Integer>> components;
    private static int counter;

    /** Single-pass Tarjan; returns components sorted by smallest vertex. */
    static List<List<Integer>> tarjanScc(int n, int[][] edges) {
        adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) adjacency.get(edge[0]).add(edge[1]);

        index = new int[n];
        Arrays.fill(index, -1);
        low = new int[n];
        onStack = new boolean[n];
        stack = new ArrayDeque<>();
        components = new ArrayList<>();
        counter = 0;

        for (int v = 0; v < n; v++) if (index[v] == -1) strongconnect(v);
        components.sort((a, b) -> Integer.compare(a.get(0), b.get(0)));
        return components;
    }

    private static void strongconnect(int u) {
        index[u] = low[u] = counter++;
        stack.push(u);
        onStack[u] = true;
        for (int v : adjacency.get(u)) {
            if (index[v] == -1) {                 // tree edge: descend
                strongconnect(v);
                low[u] = Math.min(low[u], low[v]);
            } else if (onStack[v]) {              // back/cross edge to an open SCC
                low[u] = Math.min(low[u], index[v]);
            }
        }
        if (low[u] == index[u]) {                 // u closes an SCC: pop until u
            List<Integer> group = new ArrayList<>();
            while (true) {
                int w = stack.pop();
                onStack[w] = false;
                group.add(w);
                if (w == u) break;
            }
            Collections.sort(group);
            components.add(group);
        }
    }

    private static List<List<Integer>> dfsAdjacency;
    private static boolean[] dfsVisited;
    private static List<Integer> dfsOrder;

    /** Task 280's two-pass algorithm, used here only as an independent oracle. */
    static List<List<Integer>> kosarajuScc(int n, int[][] edges) {
        List<List<Integer>> backward = new ArrayList<>();
        dfsAdjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) { dfsAdjacency.add(new ArrayList<>()); backward.add(new ArrayList<>()); }
        for (int[] edge : edges) {
            dfsAdjacency.get(edge[0]).add(edge[1]);
            backward.get(edge[1]).add(edge[0]);
        }

        dfsVisited = new boolean[n];
        dfsOrder = new ArrayList<>();
        for (int v = 0; v < n; v++) if (!dfsVisited[v]) dfsFinish(v);

        int[] component = new int[n];
        Arrays.fill(component, -1);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = dfsOrder.size() - 1; i >= 0; i--) {
            int u = dfsOrder.get(i);
            if (component[u] != -1) continue;
            int indexOf = result.size();
            component[u] = indexOf;
            List<Integer> group = new ArrayList<>();
            group.add(u);
            Deque<Integer> pending = new ArrayDeque<>();
            pending.push(u);
            while (!pending.isEmpty()) {
                int x = pending.pop();
                for (int y : backward.get(x)) {
                    if (component[y] == -1) {
                        component[y] = indexOf;
                        group.add(y);
                        pending.push(y);
                    }
                }
            }
            Collections.sort(group);
            result.add(group);
        }
        return result;
    }

    private static void dfsFinish(int u) {
        dfsVisited[u] = true;
        for (int v : dfsAdjacency.get(u)) if (!dfsVisited[v]) dfsFinish(v);
        dfsOrder.add(u);
    }

    private static int seed = 13579;

    private static int[][] randomEdges(int n) {
        List<int[]> edges = new ArrayList<>();
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                seed = (seed * 75 + 74) % 65537;
                if (seed % 3 == 0) edges.add(new int[]{u, v});
            }
        }
        return edges.toArray(new int[0][]);
    }

    static List<List<Integer>> canonical(List<List<Integer>> groups) {
        List<List<Integer>> copy = new ArrayList<>();
        for (List<Integer> group : groups) {
            List<Integer> sorted = new ArrayList<>(group);
            Collections.sort(sorted);
            copy.add(sorted);
        }
        copy.sort((a, b) -> Integer.compare(a.get(0), b.get(0)));
        return copy;
    }

    static String join(List<List<Integer>> groups) {
        List<String> parts = new ArrayList<>();
        for (List<Integer> group : groups) parts.add(group.toString());
        Collections.sort(parts);
        return String.join("|", parts);
    }

    static boolean isPartition(List<List<Integer>> groups, int n) {
        List<Integer> flat = new ArrayList<>();
        for (List<Integer> group : groups) flat.addAll(group);
        Collections.sort(flat);
        List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < n; i++) expected.add(i);
        return flat.equals(expected);
    }

    public static void main(String[] args) {
        int[][] classic = {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 3}, {4, 5}};
        check(tarjanScc(6, classic).equals(List.of(List.of(0, 1, 2), List.of(3, 4), List.of(5))), "classic");

        check(tarjanScc(3, new int[][]{{0, 1}, {1, 2}, {2, 0}}).equals(List.of(List.of(0, 1, 2))), "cycle");
        check(tarjanScc(3, new int[][]{{0, 1}, {1, 2}}).equals(List.of(List.of(0), List.of(1), List.of(2))), "DAG");
        check(tarjanScc(3, new int[][]{}).equals(List.of(List.of(0), List.of(1), List.of(2))), "no edges");
        check(tarjanScc(3, new int[][]{{0, 1}, {1, 0}, {1, 2}}).equals(List.of(List.of(0, 1), List.of(2))), "mutual pair plus tail");
        check(tarjanScc(2, new int[][]{{0, 1}}).equals(List.of(List.of(0), List.of(1))), "one-way edge");
        check(tarjanScc(1, new int[][]{{0, 0}}).equals(List.of(List.of(0))), "self loop");

        boolean nontrivialSeen = false;
        for (int trial = 0; trial < 120; trial++) {
            int[][] edges = randomEdges(6);
            List<List<Integer>> components = tarjanScc(6, edges);
            List<List<Integer>> kosaraju = kosarajuScc(6, edges);
            check(join(components).equals(join(kosaraju)), "kosaraju match " + trial);
            check(isPartition(components, 6), "partition " + trial);
            for (List<Integer> group : components) if (group.size() > 1) nontrivialSeen = true;
        }
        check(nontrivialSeen, "LCG graphs should contain real SCCs");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
