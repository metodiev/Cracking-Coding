/** Task 178 - Compute transitive closure (Floyd-Warshall naive). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static boolean[][] transitiveClosure(int[][] adj) {
        int n = adj.length;
        boolean[][] reach = new boolean[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                reach[i][j] = adj[i][j] != 0 || i == j;   // identity diagonal
        for (int k = 0; k < n; k++)
            for (int i = 0; i < n; i++)
                if (reach[i][k])
                    for (int j = 0; j < n; j++)
                        if (reach[k][j]) reach[i][j] = true;
        return reach;
    }

    public static void main(String[] args) {
        int[][] chain = {{0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
        boolean[][] reach = transitiveClosure(chain);
        check(reach[0][0] && reach[0][1] && reach[0][2], "0 reaches all");
        check(!reach[1][0] && reach[1][1] && reach[1][2], "1 reaches 1,2");
        check(!reach[2][0] && !reach[2][1] && reach[2][2], "2 only itself");

        int[][] cycle = {{0, 1, 0}, {1, 0, 0}, {0, 0, 0}};
        boolean[][] reach2 = transitiveClosure(cycle);
        check(reach2[0][1] && reach2[1][0] && !reach2[0][2], "cycle 0<->1");

        check(transitiveClosure(new int[0][]).length == 0, "empty graph");
        check(transitiveClosure(new int[][]{{0}})[0][0], "single node reaches itself");

        // cross-check against BFS reachability on a fixed pseudo-random graph
        Random rng = new Random(3);
        int n = 8;
        int[][] adj = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j && rng.nextDouble() < 0.2) adj[i][j] = 1;
        boolean[][] reach3 = transitiveClosure(adj);
        for (int src = 0; src < n; src++) {
            boolean[] seen = new boolean[n];
            Deque<Integer> stack = new ArrayDeque<>();
            seen[src] = true;
            stack.push(src);
            while (!stack.isEmpty()) {
                int u = stack.pop();
                for (int v = 0; v < n; v++)
                    if (adj[u][v] == 1 && !seen[v]) {
                        seen[v] = true;
                        stack.push(v);
                    }
            }
            for (int v = 0; v < n; v++) check(reach3[src][v] == seen[v], "bfs agrees src=" + src + " v=" + v);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
