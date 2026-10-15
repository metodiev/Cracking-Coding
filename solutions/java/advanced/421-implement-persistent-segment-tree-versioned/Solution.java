/** Task 421 - Implement persistent segment tree (versioned). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class PersistentSegTree {
    final int n;
    final List<Integer> left = new ArrayList<>();
    final List<Integer> right = new ArrayList<>();
    final List<Long> total = new ArrayList<>();
    final List<Integer> roots = new ArrayList<>();

    PersistentSegTree(int[] values) {
        n = values.length;
        left.add(0);                       // node 0 acts as the null node
        right.add(0);
        total.add(0L);
        roots.add(build(values, 0, n - 1));
    }

    private int makeNode(int l, int r, long t) {
        left.add(l);
        right.add(r);
        total.add(t);
        return total.size() - 1;
    }

    private int build(int[] values, int lo, int hi) {
        if (lo == hi) return makeNode(0, 0, values[lo]);
        int mid = (lo + hi) / 2;
        int l = build(values, lo, mid);
        int r = build(values, mid + 1, hi);
        return makeNode(l, r, total.get(l) + total.get(r));
    }

    int update(int version, int pos, long value) {
        roots.add(updateNode(roots.get(version), 0, n - 1, pos, value));
        return roots.size() - 1;
    }

    private int updateNode(int node, int lo, int hi, int pos, long value) {
        if (lo == hi) return makeNode(0, 0, value);
        int mid = (lo + hi) / 2;
        if (pos <= mid) {
            int l = updateNode(left.get(node), lo, mid, pos, value);
            int r = right.get(node);
            return makeNode(l, r, total.get(l) + total.get(r));
        }
        int l = left.get(node);
        int r = updateNode(right.get(node), mid + 1, hi, pos, value);
        return makeNode(l, r, total.get(l) + total.get(r));
    }

    long rangeSum(int version, int lo, int hi) {
        return rangeSumNode(roots.get(version), 0, n - 1, lo, hi);
    }

    private long rangeSumNode(int node, int lo, int hi, int ql, int qr) {
        if (node == 0 || qr < lo || hi < ql) return 0;
        if (ql <= lo && hi <= qr) return total.get(node);
        int mid = (lo + hi) / 2;
        return rangeSumNode(left.get(node), lo, mid, ql, qr)
             + rangeSumNode(right.get(node), mid + 1, hi, ql, qr);
    }
}

public class Solution {
    private static int seed;

    private static int[] lcgArray(int n) {
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 41 - 20;
        }
        return values;
    }

    public static void main(String[] args) {
        PersistentSegTree st = new PersistentSegTree(new int[]{1, 2, 3, 4, 5});
        check(st.roots.size() == 1, "one initial version");
        check(st.rangeSum(0, 0, 4) == 15, "v0 whole");
        check(st.rangeSum(0, 1, 3) == 9, "v0 middle");
        check(st.rangeSum(0, 2, 2) == 3, "v0 single");

        int v1 = st.update(0, 2, 10);                 // [1, 2, 10, 4, 5]
        check(v1 == 1, "v1 index");
        check(st.rangeSum(1, 0, 4) == 22, "v1 whole");
        check(st.rangeSum(1, 2, 2) == 10, "v1 changed cell");
        check(st.rangeSum(1, 1, 3) == 16, "v1 window");
        check(st.rangeSum(0, 0, 4) == 15, "v0 untouched");
        check(st.rangeSum(0, 2, 2) == 3, "v0 cell untouched");

        int v2 = st.update(1, 0, -1);                 // [-1, 2, 10, 4, 5]
        check(v2 == 2, "v2 index");
        check(st.rangeSum(2, 0, 4) == 20, "v2 whole");
        check(st.rangeSum(2, 0, 2) == 11, "v2 prefix");
        check(st.rangeSum(1, 0, 2) == 13, "v1 prefix unchanged");

        int v3 = st.update(0, 4, 7);                  // branch off OLD version: [1, 2, 3, 4, 7]
        check(v3 == 3, "v3 index");
        check(st.rangeSum(3, 0, 4) == 17, "v3 whole");
        check(st.rangeSum(3, 3, 4) == 11, "v3 suffix");
        check(st.rangeSum(3, 2, 2) == 3, "v3 old cell");
        check(st.rangeSum(2, 0, 4) == 20, "other branch unchanged");
        check(st.rangeSum(1, 0, 4) == 22, "v1 still intact");
        check(st.rangeSum(0, 0, 4) == 15, "v0 still intact");

        for (int config : new int[]{11, 421, 9001}) {
            seed = config;
            int n = 1 + seed % 29;
            int[] values = lcgArray(n);
            PersistentSegTree tree = new PersistentSegTree(values);
            List<int[]> snapshots = new ArrayList<>();
            snapshots.add(values.clone());
            int local = seed;
            for (int step = 0; step < 60; step++) {
                local = (local * 75 + 74) % 65537;
                if (local % 3 == 0) {
                    int version = local % snapshots.size();
                    int left = local % n;
                    int right = (local / n) % n;
                    if (left > right) { int tmp = left; left = right; right = tmp; }
                    long expected = 0;
                    for (int i = left; i <= right; i++) expected += snapshots.get(version)[i];
                    check(tree.rangeSum(version, left, right) == expected, "random query");
                } else {
                    int pos = local % n;
                    local = (local * 75 + 74) % 65537;
                    int value = local % 201 - 100;
                    int version = tree.update(tree.roots.size() - 1, pos, value);
                    int[] state = snapshots.get(snapshots.size() - 1).clone();
                    state[pos] = value;
                    snapshots.add(state);
                    check(version == snapshots.size() - 1, "version chain");
                    long whole = 0;
                    for (int v : state) whole += v;
                    check(tree.rangeSum(version, 0, n - 1) == whole, "random total");
                }
            }
        }

        PersistentSegTree small = new PersistentSegTree(new int[]{42});
        check(small.rangeSum(0, 0, 0) == 42, "single value");
        small.update(0, 0, -8);
        check(small.rangeSum(1, 0, 0) == -8, "single value updated");
        check(small.rangeSum(0, 0, 0) == 42, "single value versioned");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
