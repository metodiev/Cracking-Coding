/** Task 107 - Detect cycle in directed graph (DFS coloring). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static final int WHITE = 0, GRAY = 1, BLACK = 2;

    static boolean hasCycle(Map<Integer, List<Integer>> graph) {
        Map<Integer, Integer> color = new HashMap<>();
        for (int v : graph.keySet()) color.put(v, WHITE);
        for (int v : graph.keySet()) {
            if (color.get(v) == WHITE && visit(graph, color, v)) return true;
        }
        return false;
    }

    static boolean visit(Map<Integer, List<Integer>> graph, Map<Integer, Integer> color, int v) {
        color.put(v, GRAY);
        for (int w : graph.getOrDefault(v, List.<Integer>of())) {
            Integer c = color.get(w);
            if (c == null) {
                c = WHITE;
                color.put(w, WHITE);
            }
            if (c == GRAY) return true;
            if (c == WHITE && visit(graph, color, w)) return true;
        }
        color.put(v, BLACK);
        return false;
    }

    static Map<Integer, List<Integer>> makeGraph(int[][] edges) {
        Map<Integer, List<Integer>> g = new LinkedHashMap<>();
        for (int[] e : edges) {
            g.computeIfAbsent(e[0], k -> new ArrayList<>()).add(e[1]);
        }
        return g;
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> dag = makeGraph(new int[][] {{0, 1}, {0, 2}, {1, 3}, {2, 3}});
        check(!hasCycle(dag), "DAG");
        check(!hasCycle(new LinkedHashMap<>()), "empty graph");
        Map<Integer, List<Integer>> single = new LinkedHashMap<>();
        single.put(7, new ArrayList<>());
        check(!hasCycle(single), "single vertex");

        check(hasCycle(makeGraph(new int[][] {{0, 1}, {1, 2}, {2, 0}})), "simple cycle");
        Map<Integer, List<Integer>> selfLoop = new LinkedHashMap<>();
        selfLoop.put(0, new ArrayList<>(List.of(0)));
        check(hasCycle(selfLoop), "self-loop");
        check(hasCycle(makeGraph(new int[][] {{0, 1}, {2, 3}, {3, 2}})), "cycle in one component");
        check(!hasCycle(makeGraph(new int[][] {{0, 1}, {2, 3}})), "disconnected acyclic");
        check(hasCycle(makeGraph(new int[][] {{0, 1}, {1, 2}, {2, 1}})), "back edge into gray");
        check(!hasCycle(makeGraph(new int[][] {{0, 1}, {0, 2}, {1, 3}, {2, 3}})), "diamond is acyclic");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
