/** Task 106 - Topological sort using DFS (DAG). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static final int WHITE = 0, GRAY = 1, BLACK = 2;

    static List<Integer> topoSort(Map<Integer, List<Integer>> graph) {
        Map<Integer, Integer> color = new HashMap<>();
        for (int v : graph.keySet()) color.put(v, WHITE);
        List<Integer> order = new ArrayList<>();
        for (int v : graph.keySet()) {
            if (color.get(v) == WHITE && !visit(graph, color, order, v)) return null;
        }
        Collections.reverse(order);
        return order;
    }

    static boolean visit(Map<Integer, List<Integer>> graph, Map<Integer, Integer> color, List<Integer> order, int v) {
        color.put(v, GRAY);
        for (int w : graph.getOrDefault(v, List.<Integer>of())) {
            Integer c = color.get(w);
            if (c == null) {
                c = WHITE;
                color.put(w, WHITE);
            }
            if (c == GRAY) return false;
            if (c == WHITE && !visit(graph, color, order, w)) return false;
        }
        color.put(v, BLACK);
        order.add(v);
        return true;
    }

    static boolean isValidTopo(Map<Integer, List<Integer>> graph, List<Integer> order) {
        if (order == null || order.size() != graph.size()) return false;
        Map<Integer, Integer> index = new HashMap<>();
        for (int i = 0; i < order.size(); i++) index.put(order.get(i), i);
        for (Map.Entry<Integer, List<Integer>> e : graph.entrySet()) {
            Integer ui = index.get(e.getKey());
            if (ui == null) return false;
            for (int w : e.getValue()) {
                Integer wi = index.get(w);
                if (wi == null || ui >= wi) return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        Map<Integer, List<Integer>> courses = new LinkedHashMap<>();
        courses.put(0, new ArrayList<>(List.of(1, 2)));
        courses.put(1, new ArrayList<>(List.of(3)));
        courses.put(2, new ArrayList<>(List.of(3)));
        courses.put(3, new ArrayList<>());
        List<Integer> order = topoSort(courses);
        check(order != null && order.equals(List.of(0, 2, 1, 3)), "course prerequisites");
        check(isValidTopo(courses, order), "order respects every edge");

        Map<Integer, List<Integer>> cycle = new LinkedHashMap<>();
        cycle.put(0, new ArrayList<>(List.of(1)));
        cycle.put(1, new ArrayList<>(List.of(2)));
        cycle.put(2, new ArrayList<>(List.of(0)));
        check(topoSort(cycle) == null, "cycle detected");

        Map<Integer, List<Integer>> selfLoop = new LinkedHashMap<>();
        selfLoop.put(0, new ArrayList<>(List.of(0)));
        check(topoSort(selfLoop) == null, "self-loop detected");

        check(topoSort(new LinkedHashMap<>()).equals(List.of()), "empty graph");
        Map<Integer, List<Integer>> single = new LinkedHashMap<>();
        single.put(9, new ArrayList<>());
        check(topoSort(single).equals(List.of(9)), "single vertex");

        Map<Integer, List<Integer>> forest = new LinkedHashMap<>();
        forest.put(0, new ArrayList<>(List.of(1)));
        forest.put(1, new ArrayList<>());
        forest.put(2, new ArrayList<>(List.of(3)));
        forest.put(3, new ArrayList<>());
        List<Integer> forestOrder = topoSort(forest);
        check(isValidTopo(forest, forestOrder) && forestOrder.size() == 4, "disconnected DAG");

        Map<Integer, List<Integer>> chained = new LinkedHashMap<>();
        chained.put(0, new ArrayList<>());
        chained.put(1, new ArrayList<>(List.of(0)));
        chained.put(2, new ArrayList<>(List.of(1)));
        chained.put(3, new ArrayList<>(List.of(2)));
        List<Integer> chainOrder = topoSort(chained);
        check(chainOrder.equals(List.of(3, 2, 1, 0)) && isValidTopo(chained, chainOrder), "chain");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
