/** Task 273 - Find an Eulerian trail in a directed graph (Hierholzer). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Returns the vertex sequence, an empty list for no edges, or null if no trail exists. */
    static List<Integer> findEulerianTrail(int n, int[][] edges, Integer start) {
        if (edges.length == 0) return new ArrayList<>();
        int[] outDegree = new int[n];
        int[] inDegree = new int[n];
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            outDegree[edge[0]]++;
            inDegree[edge[1]]++;
        }
        for (List<Integer> neighbours : adjacency) Collections.sort(neighbours);

        List<Integer> starts = new ArrayList<>();
        List<Integer> ends = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            if (outDegree[v] - inDegree[v] == 1) starts.add(v);
            if (inDegree[v] - outDegree[v] == 1) ends.add(v);
        }
        if (starts.size() > 1 || ends.size() > 1) return null;
        for (int v = 0; v < n; v++) {
            if (outDegree[v] != inDegree[v] && !starts.contains(v) && !ends.contains(v)) return null;
        }

        int from;
        if (start == null) {
            if (!starts.isEmpty()) from = starts.get(0);
            else {
                from = -1;
                for (int v = 0; v < n; v++) if (outDegree[v] > 0) { from = v; break; }
            }
        } else {
            from = start;
            if (!starts.isEmpty() && from != starts.get(0)) return null;
            int diff = outDegree[from] - inDegree[from];
            if (diff != 0 && diff != 1) return null;
        }

        Set<Integer> active = new HashSet<>();
        for (int v = 0; v < n; v++) if (outDegree[v] > 0 || inDegree[v] > 0) active.add(v);
        if (!active.isEmpty()) {
            List<List<Integer>> undirected = new ArrayList<>();
            for (int i = 0; i < n; i++) undirected.add(new ArrayList<>());
            for (int[] edge : edges) {
                undirected.get(edge[0]).add(edge[1]);
                undirected.get(edge[1]).add(edge[0]);
            }
            Set<Integer> seen = new HashSet<>();
            Deque<Integer> stack = new ArrayDeque<>();
            seen.add(from);
            stack.push(from);
            while (!stack.isEmpty()) {
                int u = stack.pop();
                for (int v : undirected.get(u)) {
                    if (seen.add(v)) stack.push(v);
                }
            }
            if (!seen.containsAll(active)) return null;
        }

        int[] pointer = new int[n];
        List<Integer> trail = new ArrayList<>();
        Deque<Integer> walk = new ArrayDeque<>();
        walk.push(from);
        while (!walk.isEmpty()) {
            int u = walk.peek();
            if (pointer[u] < adjacency.get(u).size()) {
                walk.push(adjacency.get(u).get(pointer[u]++));
            } else {
                trail.add(walk.pop());
            }
        }
        Collections.reverse(trail);
        return trail.size() == edges.length + 1 ? trail : null;
    }

    static boolean usesAllEdges(List<Integer> trail, int[][] edges) {
        if (trail == null) return false;
        Map<String, Integer> used = new TreeMap<>();
        for (int i = 0; i + 1 < trail.size(); i++)
            used.merge(trail.get(i) + "->" + trail.get(i + 1), 1, Integer::sum);
        Map<String, Integer> wanted = new TreeMap<>();
        for (int[] edge : edges) wanted.merge(edge[0] + "->" + edge[1], 1, Integer::sum);
        return used.equals(wanted);
    }

    static String join(List<Integer> values) {
        return values == null ? "null" : values.toString();
    }

    public static void main(String[] args) {
        int[][] cycle = {{0,1},{1,2},{2,0}};
        check(join(findEulerianTrail(3, cycle, null)).equals("[0, 1, 2, 0]"), "cycle");
        check(usesAllEdges(findEulerianTrail(3, cycle, null), cycle), "cycle uses every edge");
        check(join(findEulerianTrail(3, cycle, 1)).equals("[1, 2, 0, 1]"), "balanced graph: any start");

        check(join(findEulerianTrail(3, new int[][]{{0,1},{1,2}}, null)).equals("[0, 1, 2]"), "open path");
        int[][] mixed = {{0,1},{1,2},{2,0},{0,3}};
        List<Integer> trail = findEulerianTrail(4, mixed, null);
        check(join(trail).equals("[0, 1, 2, 0, 3]"), "circuit plus spur");
        check(usesAllEdges(trail, mixed), "spur uses every edge");

        check(findEulerianTrail(4, new int[][]{{0,1},{0,2},{2,0},{0,3}}, null) == null, "out-in == 2");
        check(findEulerianTrail(4, new int[][]{{0,1},{2,3}}, null) == null, "disconnected");
        check(join(findEulerianTrail(3, new int[][]{}, null)).equals("[]"), "no edges");
        check(join(findEulerianTrail(3, cycle, 2)).equals("[2, 0, 1, 2]"), "balanced graph: other start");
        check(join(findEulerianTrail(3, new int[][]{{1,2},{2,0}}, 1)).equals("[1, 2, 0]"), "explicit valid start");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
