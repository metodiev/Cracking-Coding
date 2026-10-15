/** Task 274 - Check if a graph is bipartite (BFS 2-coloring). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Returns the colour array (0/1) or null when the graph is not bipartite. */
    static int[] twoColor(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        int[] color = new int[n];
        Arrays.fill(color, -1);
        for (int start = 0; start < n; start++) {
            if (color[start] != -1) continue;
            color[start] = 0;
            Deque<Integer> queue = new ArrayDeque<>();
            queue.add(start);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v : adjacency.get(u)) {
                    if (color[v] == -1) {
                        color[v] = color[u] ^ 1;
                        queue.add(v);
                    } else if (color[v] == color[u]) {
                        return null;
                    }
                }
            }
        }
        return color;
    }

    static boolean isBipartite(int n, int[][] edges) {
        return twoColor(n, edges) != null;
    }

    public static void main(String[] args) {
        check(isBipartite(4, new int[][]{{0,1},{1,2},{2,3},{3,0}}), "even cycle");
        check(!isBipartite(3, new int[][]{{0,1},{1,2},{2,0}}), "triangle");
        check(isBipartite(3, new int[][]{{0,1},{1,2}}), "path");
        check(!isBipartite(1, new int[][]{{0,0}}), "self-loop");
        check(isBipartite(5, new int[][]{}), "no edges");
        check(!isBipartite(7, new int[][]{{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,4}}), "one odd component");
        check(isBipartite(6, new int[][]{{0,1},{1,2},{2,3},{3,0},{4,5}}), "even plus edge");

        int[][] k33 = new int[9][2];
        int idx = 0;
        for (int i = 0; i < 3; i++) for (int j = 3; j < 6; j++) k33[idx++] = new int[]{i, j};
        check(isBipartite(6, k33), "K3,3");

        int[][] k4 = new int[6][2];
        idx = 0;
        for (int i = 0; i < 4; i++) for (int j = i + 1; j < 4; j++) k4[idx++] = new int[]{i, j};
        check(!isBipartite(4, k4), "K4");

        int[] colors = twoColor(4, new int[][]{{0,1},{1,2},{2,3},{3,0}});
        check(colors != null && colors[0] != colors[1] && colors[1] != colors[2], "colours alternate");
        check(twoColor(3, new int[][]{{0,1},{1,2},{2,0}}) == null, "no colouring for triangle");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
