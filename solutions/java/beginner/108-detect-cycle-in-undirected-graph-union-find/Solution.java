/** Task 108 - Detect cycle in undirected graph (union-find). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class Dsu {
    private final int[] parent;
    private final int[] rank;

    Dsu(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];   // path halving
            x = parent[x];
        }
        return x;
    }

    boolean union(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) {
            int t = ra;
            ra = rb;
            rb = t;
        }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        return true;
    }
}

public class Solution {
    static boolean hasCycleUndirected(int n, int[][] edges) {
        Dsu dsu = new Dsu(n);
        for (int[] e : edges) {
            int u = e[0], v = e[1];
            if (u < 0 || u >= n || v < 0 || v >= n)
                throw new IllegalArgumentException("edge endpoint out of range");
            if (!dsu.union(u, v)) return true;
        }
        return false;
    }

    public static void main(String[] args) {
        check(!hasCycleUndirected(5, new int[][] {{0, 1}, {0, 2}, {1, 3}, {1, 4}}), "tree");
        check(hasCycleUndirected(3, new int[][] {{0, 1}, {1, 2}, {2, 0}}), "triangle");
        check(!hasCycleUndirected(6, new int[][] {{0, 1}, {2, 3}, {3, 4}, {4, 5}}), "forest");
        check(hasCycleUndirected(6, new int[][] {{0, 1}, {2, 3}, {3, 4}, {4, 5}, {5, 2}}), "cycle in one tree");
        check(!hasCycleUndirected(1, new int[][] {}), "single vertex");
        check(hasCycleUndirected(2, new int[][] {{0, 0}}), "self-loop");
        check(hasCycleUndirected(2, new int[][] {{0, 1}, {1, 0}}), "parallel edges");

        boolean threw = false;
        try {
            hasCycleUndirected(2, new int[][] {{0, 5}});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "out-of-range edge rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
