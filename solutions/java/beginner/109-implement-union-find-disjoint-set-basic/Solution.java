/** Task 109 - Implement union-find / disjoint set (basic). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class Dsu {
    private final int[] parent;
    private final int[] rank;
    private int components;

    Dsu(int n) {
        parent = new int[n];
        rank = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        components = n;
    }

    int find(int x) {                     // bare pointer walk (no compression yet)
        while (parent[x] != x) x = parent[x];
        return x;
    }

    boolean union(int a, int b) {         // plain linking
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        parent[ra] = rb;
        components--;
        return true;
    }

    boolean unionByRank(int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        if (rank[ra] < rank[rb]) {
            int t = ra;
            ra = rb;
            rb = t;
        }
        parent[rb] = ra;
        if (rank[ra] == rank[rb]) rank[ra]++;
        components--;
        return true;
    }

    boolean connected(int a, int b) {
        return find(a) == find(b);
    }

    int count() {
        return components;
    }

    int depth(int x) {
        int d = 0;
        while (parent[x] != x) {
            x = parent[x];
            d++;
        }
        return d;
    }
}

public class Solution {
    public static void main(String[] args) {
        Dsu d = new Dsu(6);
        check(d.count() == 6, "fresh components");
        check(d.find(3) == 3, "each element its own root");
        check(d.connected(0, 0), "element connected to itself");
        check(d.union(0, 1), "merge 0-1");
        check(d.union(1, 2), "merge 1-2");
        check(d.count() == 4, "count after two merges");
        check(d.connected(0, 2), "0 and 2 connected");
        check(d.depth(0) == 2, "plain linking leaves a chain");
        check(!d.union(0, 2), "redundant union is a no-op");
        check(d.count() == 4, "redundant union keeps count");
        check(!d.union(5, 5), "self union is a no-op");
        check(d.unionByRank(3, 4), "rank merge 3-4");
        check(d.unionByRank(4, 5), "rank merge 4-5");
        check(d.connected(3, 5), "3 and 5 connected");
        check(!d.connected(2, 3), "different sets stay apart");
        check(d.count() == 2, "final component count");

        Dsu r = new Dsu(8);
        for (int i = 0; i < 7; i++) check(r.unionByRank(i, i + 1), "rank merge " + i);
        check(r.count() == 1, "rank merges connect everything");
        int maxDepth = 0;
        for (int i = 0; i < 8; i++) maxDepth = Math.max(maxDepth, r.depth(i));
        check(maxDepth <= 1, "union-by-rank keeps the tree shallow");

        Dsu one = new Dsu(1);
        check(one.count() == 1 && one.find(0) == 0 && one.depth(0) == 0, "single element");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
