/** Task 104 - Implement BFS on adjacency list graph. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static List<Integer> bfsOrder(Map<Integer, List<Integer>> graph, int source) {
        List<Integer> order = new ArrayList<>();
        if (!graph.containsKey(source)) return order;
        Set<Integer> seen = new HashSet<>();
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        seen.add(source);
        queue.add(source);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            order.add(v);
            for (int w : graph.getOrDefault(v, List.<Integer>of())) {
                if (seen.add(w)) queue.add(w);
            }
        }
        return order;
    }

    static Map<Integer, Integer> bfsDistances(Map<Integer, List<Integer>> graph, int source) {
        Map<Integer, Integer> dist = new LinkedHashMap<>();
        if (!graph.containsKey(source)) return dist;
        dist.put(source, 0);
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        queue.add(source);
        while (!queue.isEmpty()) {
            int v = queue.poll();
            for (int w : graph.getOrDefault(v, List.<Integer>of())) {
                if (!dist.containsKey(w)) {
                    dist.put(w, dist.get(v) + 1);
                    queue.add(w);
                }
            }
        }
        return dist;
    }

    static Map<Integer, List<Integer>> makeGraph(int[][] edges) {
        Map<Integer, List<Integer>> g = new LinkedHashMap<>();
        for (int[] e : edges) {
            g.computeIfAbsent(e[0], k -> new ArrayList<>()).add(e[1]);
        }
        return g;
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> g = makeGraph(new int[][] {{0, 1}, {0, 2}, {1, 2}, {2, 0}, {2, 3}, {3, 3}});
        check(bfsOrder(g, 0).equals(List.of(0, 1, 2, 3)), "order with cycle");
        check(bfsDistances(g, 0).equals(Map.of(0, 0, 1, 1, 2, 1, 3, 2)), "distances");

        Map<Integer, List<Integer>> parts = makeGraph(new int[][] {{0, 1}, {1, 0}, {2, 3}, {3, 2}});
        parts.putIfAbsent(4, new ArrayList<>());
        check(bfsOrder(parts, 0).equals(List.of(0, 1)), "disconnected order");
        check(bfsDistances(parts, 0).equals(Map.of(0, 0, 1, 1)), "disconnected distances");
        check(bfsOrder(parts, 4).equals(List.of(4)), "loner vertex");
        check(bfsDistances(parts, 4).equals(Map.of(4, 0)), "loner distance");
        check(bfsOrder(parts, 99).isEmpty(), "source not in graph");
        check(bfsDistances(parts, 99).isEmpty(), "distances for missing source");
        check(bfsOrder(new LinkedHashMap<>(), 0).isEmpty(), "empty graph");
        check(bfsDistances(new LinkedHashMap<>(), 0).isEmpty(), "empty graph distances");
        Map<Integer, List<Integer>> twoCycles = makeGraph(new int[][] {{5, 6}, {6, 5}, {6, 7}, {7, 5}});
        check(bfsOrder(twoCycles, 5).equals(List.of(5, 6, 7)), "two-node cycle");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
