/** Task 442 - Implement segment tree beats for complex range operations (conceptual). (Tier: Advanced, CTCI Ch 4) */
class SegmentTreeBeats {
    private final int n;
    private final long[] nodeMax;
    private final long[] nodeSum;

    SegmentTreeBeats(long[] values) {
        n = values.length;
        nodeMax = new long[4 * Math.max(1, n)];
        nodeSum = new long[4 * Math.max(1, n)];
        if (n > 0) build(1, 0, n - 1, values);
    }

    private void build(int node, int lo, int hi, long[] values) {
        if (lo == hi) {
            nodeMax[node] = values[lo];
            nodeSum[node] = values[lo];
            return;
        }
        int mid = (lo + hi) >> 1;
        build(2 * node, lo, mid, values);
        build(2 * node + 1, mid + 1, hi, values);
        nodeMax[node] = Math.max(nodeMax[2 * node], nodeMax[2 * node + 1]);
        nodeSum[node] = nodeSum[2 * node] + nodeSum[2 * node + 1];
    }

    void chmin(int left, int right, long cap) {
        chmin(1, 0, n - 1, left, right, cap);
    }

    private void chmin(int node, int lo, int hi, int left, int right, long cap) {
        if (right < lo || hi < left || nodeMax[node] <= cap) return;
        if (lo == hi) {
            nodeMax[node] = cap;
            nodeSum[node] = cap;
            return;
        }
        int mid = (lo + hi) >> 1;
        chmin(2 * node, lo, mid, left, right, cap);
        chmin(2 * node + 1, mid + 1, hi, left, right, cap);
        nodeMax[node] = Math.max(nodeMax[2 * node], nodeMax[2 * node + 1]);
        nodeSum[node] = nodeSum[2 * node] + nodeSum[2 * node + 1];
    }

    long rangeMax(int left, int right) {
        return rangeMax(1, 0, n - 1, left, right);
    }

    private long rangeMax(int node, int lo, int hi, int left, int right) {
        if (left <= lo && hi <= right) return nodeMax[node];
        int mid = (lo + hi) >> 1;
        long best = Long.MIN_VALUE;
        if (left <= mid) best = Math.max(best, rangeMax(2 * node, lo, mid, left, right));
        if (right > mid) best = Math.max(best, rangeMax(2 * node + 1, mid + 1, hi, left, right));
        return best;
    }

    long rangeSum(int left, int right) {
        return rangeSum(1, 0, n - 1, left, right);
    }

    private long rangeSum(int node, int lo, int hi, int left, int right) {
        if (left <= lo && hi <= right) return nodeSum[node];
        int mid = (lo + hi) >> 1;
        long total = 0;
        if (left <= mid) total += rangeSum(2 * node, lo, mid, left, right);
        if (right > mid) total += rangeSum(2 * node + 1, mid + 1, hi, left, right);
        return total;
    }
}

public class Solution {
    private static int seed;

    private static int[][] lcgOps(int count, int n) {
        int[][] ops = new int[count][];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int lo = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int hi = lo + seed % (n - lo);
            seed = (seed * 75 + 74) % 65537;
            int kind = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int cap = seed % 41 - 20;
            ops[k] = new int[]{kind, lo, hi, cap};
        }
        return ops;
    }

    public static void main(String[] args) {
        SegmentTreeBeats tree = new SegmentTreeBeats(new long[]{1, 5, 3, 4, 2});
        check(tree.rangeMax(0, 4) == 5, "initial max");
        check(tree.rangeSum(0, 4) == 15, "initial sum");
        tree.chmin(1, 3, 3);
        check(tree.rangeMax(0, 4) == 3, "capped max");
        check(tree.rangeMax(1, 1) == 3, "capped point");
        check(tree.rangeSum(0, 4) == 12, "capped sum");
        check(tree.rangeSum(2, 3) == 6, "window sum");
        tree.chmin(0, 4, -1);
        check(tree.rangeMax(0, 4) == -1, "global cap max");
        check(tree.rangeSum(0, 4) == -5, "global cap sum");
        tree.chmin(2, 2, 10);
        check(tree.rangeMax(0, 4) == -1, "no-op cap max");
        check(tree.rangeSum(0, 4) == -5, "no-op cap sum");

        int n = 16;
        seed = 1234;
        long[] base = new long[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            base[i] = seed % 50 - 25;
        }
        tree = new SegmentTreeBeats(base);
        long[] brute = base.clone();
        for (int[] op : lcgOps(300, n)) {
            int kind = op[0], lo = op[1], hi = op[2], cap = op[3];
            if (kind == 0) {
                tree.chmin(lo, hi, cap);
                for (int i = lo; i <= hi; i++) brute[i] = Math.min(brute[i], cap);
            } else if (kind == 1) {
                long best = Long.MIN_VALUE;
                for (int i = lo; i <= hi; i++) best = Math.max(best, brute[i]);
                check(tree.rangeMax(lo, hi) == best, "random max " + lo + ".." + hi);
            } else {
                long total = 0;
                for (int i = lo; i <= hi; i++) total += brute[i];
                check(tree.rangeSum(lo, hi) == total, "random sum " + lo + ".." + hi);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
