/** Task 441 - Implement range add + range max with lazy propagation. (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

class RangeAddRangeMax {
    private final int n;
    private final long[] tree, lazy;

    RangeAddRangeMax(long[] values) {
        n = values.length;
        tree = new long[4 * Math.max(1, n)];
        lazy = new long[4 * Math.max(1, n)];
        if (n > 0) build(1, 0, n - 1, values);
    }

    private void build(int node, int lo, int hi, long[] values) {
        if (lo == hi) {
            tree[node] = values[lo];
            return;
        }
        int mid = (lo + hi) / 2;
        build(2 * node, lo, mid, values);
        build(2 * node + 1, mid + 1, hi, values);
        tree[node] = Math.max(tree[2 * node], tree[2 * node + 1]);
    }

    private void push(int node) {
        long delta = lazy[node];
        if (delta != 0) {
            for (int child : new int[]{2 * node, 2 * node + 1}) {
                tree[child] += delta;
                lazy[child] += delta;
            }
            lazy[node] = 0;
        }
    }

    void add(int left, int right, long value) {
        add(1, 0, n - 1, left, right, value);
    }

    private void add(int node, int lo, int hi, int left, int right, long value) {
        if (right < lo || hi < left) return;
        if (left <= lo && hi <= right) {
            tree[node] += value;
            lazy[node] += value;
            return;
        }
        push(node);
        int mid = (lo + hi) / 2;
        add(2 * node, lo, mid, left, right, value);
        add(2 * node + 1, mid + 1, hi, left, right, value);
        tree[node] = Math.max(tree[2 * node], tree[2 * node + 1]);
    }

    long max(int left, int right) {
        return max(1, 0, n - 1, left, right);
    }

    private long max(int node, int lo, int hi, int left, int right) {
        if (left <= lo && hi <= right) return tree[node];
        push(node);
        int mid = (lo + hi) / 2;
        long best = Long.MIN_VALUE;
        if (left <= mid) best = Math.max(best, max(2 * node, lo, mid, left, right));
        if (right > mid) best = Math.max(best, max(2 * node + 1, mid + 1, hi, left, right));
        return best;
    }
}

public class Solution {
    private static int seed = 777;

    private static int[][] lcgOps(int count, int n) {
        int[][] ops = new int[count][];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int lo = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int hi = lo + seed % (n - lo);
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int value = seed % 21 - 10;
            ops[k] = new int[]{kind, lo, hi, value};
        }
        return ops;
    }

    public static void main(String[] args) {
        RangeAddRangeMax tree = new RangeAddRangeMax(new long[]{1, 3, 2, 5, 4});
        check(tree.max(0, 4) == 5, "initial max");
        check(tree.max(3, 3) == 5, "single point");
        tree.add(1, 3, 2);
        check(tree.max(0, 4) == 7, "after add");
        check(tree.max(0, 0) == 1, "untouched left");
        check(tree.max(1, 1) == 5, "shifted point");
        check(tree.max(2, 3) == 7, "window max");
        tree.add(0, 4, -10);
        check(tree.max(0, 4) == -3, "negative add");
        tree.add(4, 4, 100);
        check(tree.max(0, 4) == 94, "point boost");

        int n = 16;
        seed = 1234;
        long[] base = new long[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            base[i] = seed % 50 - 25;
        }
        tree = new RangeAddRangeMax(base);
        long[] brute = base.clone();
        for (int[] op : lcgOps(200, n)) {
            int kind = op[0], lo = op[1], hi = op[2], value = op[3];
            if (kind == 0) {
                tree.add(lo, hi, value);
                for (int i = lo; i <= hi; i++) brute[i] += value;
            } else {
                long best = Long.MIN_VALUE;
                for (int i = lo; i <= hi; i++) best = Math.max(best, brute[i]);
                check(tree.max(lo, hi) == best, "random query " + lo + ".." + hi);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
