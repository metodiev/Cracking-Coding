/** Task 182 - Implement segment tree for range sum (build/query/update). (Tier: Beginner, CTCI Ch 4) */
public class Solution {
    /**
     * Recursive segment tree in a flat array of size 4n; node 1 is the root and covers
     * [0, n - 1], the children of node i are 2i and 2i + 1. Build O(n), point update and
     * inclusive range sum O(log n).
     */
    static class SegmentTree {
        private final int n;
        private final long[] tree;

        SegmentTree(int[] values) {
            n = values.length;
            tree = new long[4 * Math.max(n, 1)];
            if (n > 0) build(1, 0, n - 1, values);
        }

        private void build(int node, int lo, int hi, int[] values) {
            if (lo == hi) {
                tree[node] = values[lo];
                return;
            }
            int mid = (lo + hi) / 2;
            build(2 * node, lo, mid, values);
            build(2 * node + 1, mid + 1, hi, values);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }

        void update(int index, long value) {
            if (index < 0 || index >= n) throw new IndexOutOfBoundsException("index out of range");
            update(1, 0, n - 1, index, value);
        }

        private void update(int node, int lo, int hi, int index, long value) {
            if (lo == hi) {
                tree[node] = value;
                return;
            }
            int mid = (lo + hi) / 2;
            if (index <= mid) update(2 * node, lo, mid, index, value);
            else update(2 * node + 1, mid + 1, hi, index, value);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }

        long rangeSum(int lo, int hi) {
            if (lo > hi) return 0;
            if (lo < 0 || hi >= n) throw new IndexOutOfBoundsException("range out of bounds");
            return query(1, 0, n - 1, lo, hi);
        }

        private long query(int node, int lo, int hi, int qlo, int qhi) {
            if (qlo <= lo && hi <= qhi) return tree[node];
            int mid = (lo + hi) / 2;
            long total = 0;
            if (qlo <= mid) total += query(2 * node, lo, mid, qlo, qhi);
            if (qhi > mid) total += query(2 * node + 1, mid + 1, hi, qlo, qhi);
            return total;
        }
    }

    public static void main(String[] args) {
        int[] values = {1, 3, 5, 7, 9, 11};
        SegmentTree st = new SegmentTree(values);
        check(st.rangeSum(0, 5) == 36, "whole array");
        check(st.rangeSum(0, 0) == 1, "first element");
        check(st.rangeSum(5, 5) == 11, "last element");
        check(st.rangeSum(2, 3) == 12, "middle range");
        check(st.rangeSum(3, 2) == 0, "empty range");

        st.update(2, 10);
        values[2] = 10;
        check(st.rangeSum(1, 3) == 20, "after update");
        st.update(0, -4);
        values[0] = -4;
        check(st.rangeSum(0, 5) == 36, "sum unchanged after -5/+5 swap");
        check(st.rangeSum(0, 0) == -4, "updated single value");

        SegmentTree single = new SegmentTree(new int[] {7});
        check(single.rangeSum(0, 0) == 7, "single element");
        single.update(0, 2);
        check(single.rangeSum(0, 0) == 2, "single element updated");

        SegmentTree empty = new SegmentTree(new int[0]);
        check(empty.rangeSum(0, -1) == 0, "empty tree");

        boolean threw = false;
        try { st.rangeSum(-1, 2); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "negative lo rejected");
        threw = false;
        try { st.update(6, 0); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "bad update index rejected");

        java.util.Random rng = new java.util.Random(21);
        int[] data = new int[17];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextInt(41) - 20;
        SegmentTree st2 = new SegmentTree(data);
        for (int t = 0; t < 40; t++) {
            int i = rng.nextInt(data.length);
            int value = rng.nextInt(41) - 20;
            data[i] = value;
            st2.update(i, value);
            for (int lo = 0; lo < data.length; lo++) {
                long expected = 0;
                for (int hi = lo; hi < data.length; hi++) {
                    expected += data[hi];
                    check(st2.rangeSum(lo, hi) == expected, "cross-check range " + lo + ".." + hi);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
