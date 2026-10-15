/** Task 082 - Range sum query naive using prefix sums. (Tier: Beginner, CTCI Ch 1)
 *
 * Build the prefix sums once in O(n); then every inclusive rangeSum(lo, hi) query
 * is O(1) via prefix[hi + 1] - prefix[lo]. Invalid ranges (lo < 0, hi >= n,
 * lo > hi) throw IllegalArgumentException.
 */
public class Solution {
    static class RangeSumQuery {
        private final int[] prefix;

        RangeSumQuery(int[] a) {
            prefix = new int[a.length + 1];
            for (int i = 0; i < a.length; i++) prefix[i + 1] = prefix[i] + a[i];
        }

        int rangeSum(int lo, int hi) {
            int n = prefix.length - 1;
            if (lo < 0 || hi >= n || lo > hi) {
                throw new IllegalArgumentException("invalid range [" + lo + ", " + hi + "] for n=" + n);
            }
            return prefix[hi + 1] - prefix[lo];
        }
    }

    public static void main(String[] args) {
        RangeSumQuery rs = new RangeSumQuery(new int[]{3, 1, 4, 1, 5});
        check(rs.rangeSum(0, 4) == 14, "full range");
        check(rs.rangeSum(1, 3) == 6, "inner range");
        check(rs.rangeSum(2, 2) == 4, "single element");
        check(rs.rangeSum(0, 0) == 3, "first element");
        check(rs.rangeSum(4, 4) == 5, "last element");
        check(new RangeSumQuery(new int[]{7}).rangeSum(0, 0) == 7, "single element array");
        checkThrows(() -> rs.rangeSum(-1, 2), "lo < 0");
        checkThrows(() -> rs.rangeSum(2, 1), "lo > hi");
        checkThrows(() -> rs.rangeSum(1, 5), "hi >= n");
        checkThrows(() -> new RangeSumQuery(new int[]{}).rangeSum(0, 0), "empty array");
        System.out.println("PASS");
    }

    static void checkThrows(Runnable r, String label) {
        try {
            r.run();
        } catch (IllegalArgumentException e) {
            return;
        }
        throw new AssertionError("FAILED: expected IllegalArgumentException: " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
