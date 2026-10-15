/** Task 022 - Rotate array in-place by k positions. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int[] rotateInPlace(int[] items, int k) {
        int n = items.length;
        if (n == 0) return items;
        k %= n;
        reverse(items, 0, n - 1);
        reverse(items, 0, k - 1);
        reverse(items, k, n - 1);
        return items;
    }

    private static void reverse(int[] a, int lo, int hi) {
        while (lo < hi) { int t = a[lo]; a[lo++] = a[hi]; a[hi--] = t; }
    }

    public static void main(String[] args) {
        check(java.util.Arrays.equals(rotateInPlace(new int[]{1, 2, 3, 4, 5}, 2), new int[]{4, 5, 1, 2, 3}), "k=2");
        check(java.util.Arrays.equals(rotateInPlace(new int[]{1, 2}, 0), new int[]{1, 2}), "k=0");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
