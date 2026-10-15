/** Task 183 - Implement iterative segment tree. (Tier: Beginner, CTCI Ch 4) */
public class Solution {
    /**
     * Bottom-up segment tree in a flat array of size 2n: leaves live at [n, 2n) and
     * parents at [1, n) with tree[i] = tree[2i] + tree[2i + 1]. Everything is a loop, so
     * there is no recursion overhead; compared to the recursive tree of task 182 it needs
     * 2n slots instead of 4n and never recurses, while producing the same answers.
     */
    static class IterativeSegmentTree {
        private final int n;
        private final long[] tree;

        IterativeSegmentTree(int[] values) {
            n = values.length;
            tree = new long[2 * n];
            for (int i = 0; i < n; i++) tree[n + i] = values[i];
            for (int i = n - 1; i >= 1; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];
        }

        void update(int index, long value) {
            if (index < 0 || index >= n) throw new IndexOutOfBoundsException("index out of range");
            int i = n + index;
            tree[i] = value;
            for (i /= 2; i >= 1; i /= 2) tree[i] = tree[2 * i] + tree[2 * i + 1];
        }

        long rangeSum(int lo, int hi) {
            if (lo > hi) return 0;
            if (lo < 0 || hi >= n) throw new IndexOutOfBoundsException("range out of bounds");
            long total = 0;
            int left = lo + n;
            int right = hi + 1 + n;               // classic half-open [left, right)
            while (left < right) {
                if ((left & 1) == 1) total += tree[left++];
                if ((right & 1) == 1) total += tree[--right];
                left /= 2;
                right /= 2;
            }
            return total;
        }
    }

    public static void main(String[] args) {
        int[] values = {1, 3, 5, 7, 9, 11};
        IterativeSegmentTree st = new IterativeSegmentTree(values);
        check(st.rangeSum(0, 5) == 36, "whole array");
        check(st.rangeSum(0, 0) == 1, "first element");
        check(st.rangeSum(5, 5) == 11, "last element");
        check(st.rangeSum(2, 3) == 12, "middle range");
        check(st.rangeSum(1, 4) == 24, "inner range");
        check(st.rangeSum(3, 2) == 0, "empty range");

        st.update(2, 10);
        values[2] = 10;
        check(st.rangeSum(0, 5) == 41, "after update");
        check(st.rangeSum(2, 2) == 10, "updated point");
        st.update(0, -4);
        values[0] = -4;
        check(st.rangeSum(0, 5) == 36, "after second update");

        IterativeSegmentTree single = new IterativeSegmentTree(new int[] {7});
        check(single.rangeSum(0, 0) == 7, "single element");
        single.update(0, 2);
        check(single.rangeSum(0, 0) == 2, "single element updated");

        IterativeSegmentTree empty = new IterativeSegmentTree(new int[0]);
        check(empty.rangeSum(0, -1) == 0, "empty tree");

        boolean threw = false;
        try { st.rangeSum(-1, 2); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "negative lo rejected");
        threw = false;
        try { st.update(6, 0); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "bad update index rejected");

        java.util.Random rng = new java.util.Random(33);
        int[] data = new int[24];
        for (int i = 0; i < data.length; i++) data[i] = rng.nextInt(41) - 20;
        IterativeSegmentTree st2 = new IterativeSegmentTree(data);
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
