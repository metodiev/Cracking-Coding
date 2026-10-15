/** Task 110 - Path compression in union-find. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class Dsu {
    private final int[] parent;
    private int components;

    Dsu(int n) {
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        components = n;
    }

    int find(int x) {
        int root = x;
        while (parent[root] != root) root = parent[root];
        while (parent[x] != root) {          // second pass flattens the path
            int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    boolean union(int a, int b) {            // plain linking on purpose
        int ra = find(a), rb = find(b);
        if (ra == rb) return false;
        parent[ra] = rb;
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
        Dsu d = new Dsu(9);
        check(d.count() == 9, "fresh components");
        for (int i = 0; i < 7; i++) check(d.union(i, i + 1), "chain link " + i);
        check(d.count() == 2, "chain plus loner");
        check(d.depth(0) == 7, "chain is deep before compression");
        check(d.find(0) == 7, "root of the chain");
        check(d.depth(0) == 1, "find flattened the chain");
        for (int i = 1; i <= 6; i++) check(d.depth(i) == 1, "node " + i + " points at root");
        check(d.depth(7) == 0, "root has depth 0");
        check(d.connected(0, 6), "chain members connected");
        check(d.find(0) == d.find(0), "find is idempotent");
        check(d.depth(0) == 1, "second find stays flat");
        check(d.union(7, 8), "attach loner");
        check(d.count() == 1, "everything connected");
        check(d.find(0) == 8, "new root");
        check(d.depth(0) <= 2, "depth stays near constant");
        check(d.find(0) == 8 && d.depth(0) == 1, "compressed onto new root");
        int maxDepth = 0;
        for (int i = 0; i < 9; i++) maxDepth = Math.max(maxDepth, d.depth(i));
        check(maxDepth <= 2, "bounded depth overall");

        Dsu one = new Dsu(1);
        check(one.count() == 1 && one.find(0) == 0 && one.depth(0) == 0, "single element");

        Dsu flat = new Dsu(3);
        check(flat.union(0, 1) && flat.union(1, 2), "flat unions");
        check(flat.depth(0) == 2, "chain of three");
        check(flat.find(0) == 2 && flat.depth(0) == 1 && flat.depth(1) == 1, "compression on chain of three");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
