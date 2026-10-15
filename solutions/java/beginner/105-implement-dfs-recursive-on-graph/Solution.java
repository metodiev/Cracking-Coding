/** Task 105 - Implement DFS (recursive) on graph. (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static List<Integer> dfsOrder(Map<Integer, List<Integer>> graph, int source) {
        List<Integer> order = new ArrayList<>();
        if (!graph.containsKey(source)) return order;
        Set<Integer> visited = new HashSet<>();
        dfs(graph, source, visited, order);
        return order;
    }

    static List<Integer> dfsAll(Map<Integer, List<Integer>> graph) {
        List<Integer> order = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        for (int v : new TreeSet<>(graph.keySet())) {
            if (!visited.contains(v)) dfs(graph, v, visited, order);
        }
        return order;
    }

    static void dfs(Map<Integer, List<Integer>> graph, int v, Set<Integer> visited, List<Integer> order) {
        visited.add(v);
        order.add(v);
        for (int w : graph.getOrDefault(v, List.<Integer>of())) {
            if (!visited.contains(w)) dfs(graph, w, visited, order);
        }
    }

    static Map<Integer, List<Integer>> makeGraph(int[][] edges) {
        Map<Integer, List<Integer>> g = new LinkedHashMap<>();
        for (int[] e : edges) {
            g.computeIfAbsent(e[0], k -> new ArrayList<>()).add(e[1]);
        }
        return g;
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> cyclic = makeGraph(new int[][] {{0, 1}, {0, 2}, {1, 2}, {2, 0}, {2, 3}, {3, 3}});
        check(dfsOrder(cyclic, 0).equals(List.of(0, 1, 2, 3)), "cycle guard");

        Map<Integer, List<Integer>> twoCycles = makeGraph(new int[][] {{0, 1}, {1, 0}, {2, 3}, {3, 2}});
        check(dfsOrder(twoCycles, 0).equals(List.of(0, 1)), "single component only");
        check(dfsAll(twoCycles).equals(List.of(0, 1, 2, 3)), "all components");

        Map<Integer, List<Integer>> single = makeGraph(new int[][] {});
        single.put(5, new ArrayList<>());
        check(dfsOrder(single, 5).equals(List.of(5)), "single vertex");
        check(dfsOrder(new LinkedHashMap<>(), 0).isEmpty(), "empty graph");
        check(dfsOrder(cyclic, 99).isEmpty(), "source not in graph");
        check(dfsAll(new LinkedHashMap<>()).isEmpty(), "dfs_all empty");
        Map<Integer, List<Integer>> lone = makeGraph(new int[][] {});
        lone.put(7, new ArrayList<>());
        check(dfsAll(lone).equals(List.of(7)), "dfs_all single");

        Map<Integer, List<Integer>> chain = new LinkedHashMap<>();
        for (int i = 0; i <= 50; i++) {
            List<Integer> next = new ArrayList<>();
            if (i < 50) next.add(i + 1);
            chain.put(i, next);
        }
        check(dfsOrder(chain, 0).size() == 51, "deep chain");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
