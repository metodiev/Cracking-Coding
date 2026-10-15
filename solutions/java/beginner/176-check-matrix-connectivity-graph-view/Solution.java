/** Task 176 - Check matrix connectivity (graph view). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static boolean isConnected(int[][] adj) {
        int n = adj.length;
        if (n <= 1) return true;
        boolean[] seen = new boolean[n];
        seen[0] = true;
        Deque<Integer> q = new ArrayDeque<>();
        q.add(0);
        int reached = 1;
        while (!q.isEmpty()) {
            int u = q.poll();
            for (int v = 0; v < n; v++)
                if (adj[u][v] == 1 && !seen[v]) {
                    seen[v] = true;
                    reached++;
                    q.add(v);
                }
        }
        return reached == n;
    }

    static int connectedComponents(int[][] adj) {
        int n = adj.length;
        boolean[] seen = new boolean[n];
        int components = 0;
        for (int start = 0; start < n; start++) {
            if (seen[start]) continue;
            components++;
            seen[start] = true;
            Deque<Integer> q = new ArrayDeque<>();
            q.add(start);
            while (!q.isEmpty()) {
                int u = q.poll();
                for (int v = 0; v < n; v++)
                    if (adj[u][v] == 1 && !seen[v]) {
                        seen[v] = true;
                        q.add(v);
                    }
            }
        }
        return components;
    }

    public static void main(String[] args) {
        int[][] path = {{0, 1, 0, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}, {0, 0, 1, 0}};
        check(isConnected(path) && connectedComponents(path) == 1, "path graph");

        int[][] split = {{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        check(!isConnected(split) && connectedComponents(split) == 2, "two pairs");

        int[][] lonely = {{0, 1, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}};
        check(!isConnected(lonely) && connectedComponents(lonely) == 2, "isolated vertex");

        check(isConnected(new int[0][]) && isConnected(new int[][]{{0}}), "trivial graphs");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
