/** Task 111 - Kruskal's MST using union-find. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class Dsu {
    private final int[] parent, rank;

    Dsu(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    boolean union(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) { int t = ra; ra = rb; rb = t; }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        return true;
    }
}

public class Solution {
    /** edges rows: {u, v, w}. Returns {totalWeight, edgeCount} or null when disconnected. */
    static long[] kruskal(int n, int[][] edges) {
        Arrays.sort(edges, Comparator.comparingInt(e -> e[2]));
        Dsu dsu = new Dsu(n);
        long total = 0;
        int count = 0;
        for (int[] e : edges) {
            if (dsu.union(e[0], e[1])) {
                total += e[2];
                if (++count == Math.max(0, n - 1)) break;
            }
        }
        return count == Math.max(0, n - 1) ? new long[]{total, count} : null;
    }

    public static void main(String[] args) {
        int[][] edges = {
            {0, 1, 4}, {0, 2, 3}, {1, 2, 1},
            {1, 3, 2}, {2, 3, 4}, {3, 4, 2}, {2, 4, 5},
        };
        long[] result = kruskal(5, edges);
        check(result != null && result[0] == 8 && result[1] == 4, "classic graph");
        check(kruskal(1, new int[0][])[0] == 0, "single vertex");
        check(kruskal(3, new int[][]{{0, 1, 1}}) == null, "disconnected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
