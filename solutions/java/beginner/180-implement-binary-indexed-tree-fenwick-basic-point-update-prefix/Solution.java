/** Task 180 - Implement binary indexed tree (Fenwick) basic. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static class Fenwick {
        private final int n;
        private final long[] tree;

        Fenwick(int[] values) {
            n = values.length;
            tree = new long[n + 1];
            for (int i = 0; i < n; i++) updateIndex(i + 1, values[i]);
        }

        private void updateIndex(int i, long delta) {
            while (i <= n) {
                tree[i] += delta;
                i += i & -i;
            }
        }

        void update(int index, long delta) {
            if (index < 0 || index >= n) throw new IndexOutOfBoundsException("index out of range");
            updateIndex(index + 1, delta);
        }

        long prefixSum(int index) {
            int i = index + 1;
            if (i < 0 || i > n) throw new IndexOutOfBoundsException("index out of range");
            long total = 0;
            while (i > 0) {
                total += tree[i];
                i -= i & -i;
            }
            return total;
        }

        long rangeSum(int lo, int hi) {
            if (lo > hi) return 0;
            return prefixSum(hi) - prefixSum(lo - 1);
        }
    }

    public static void main(String[] args) {
        int[] values = {1, 3, 5, 7, 9, 11};
        Fenwick fw = new Fenwick(values);
        check(fw.prefixSum(0) == 1, "prefix 0");
        check(fw.prefixSum(5) == 36, "prefix 5");
        check(fw.rangeSum(1, 3) == 15, "range 1..3");
        check(fw.rangeSum(3, 2) == 0, "empty range");

        fw.update(2, 5);
        values[2] += 5;
        check(fw.prefixSum(5) == 41, "after +5");
        fw.update(0, -1);
        values[0] -= 1;
        check(fw.prefixSum(0) == 0, "after -1");

        Random rng = new Random(9);
        Fenwick fw2 = new Fenwick(values);
        for (int t = 0; t < 50; t++) {
            int i = rng.nextInt(values.length);
            int delta = rng.nextInt(11) - 5;
            values[i] += delta;
            fw2.update(i, delta);
            for (int end = 0; end < values.length; end++) {
                long expected = 0;
                for (int k = 0; k <= end; k++) expected += values[k];
                check(fw2.prefixSum(end) == expected, "cross-check prefix " + end);
            }
            int lo = rng.nextInt(values.length);
            int hi = lo + rng.nextInt(values.length - lo);
            long expectedRange = 0;
            for (int k = lo; k <= hi; k++) expectedRange += values[k];
            check(fw2.rangeSum(lo, hi) == expectedRange, "cross-check range");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
