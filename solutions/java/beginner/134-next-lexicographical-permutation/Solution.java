/** Task 134 - Next lexicographical permutation. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    /** Mutates in place; returns false (and wraps to sorted order) at the last permutation. */
    static boolean nextPermutation(int[] a) {
        int n = a.length;
        int pivot = n - 2;
        while (pivot >= 0 && a[pivot] >= a[pivot + 1]) pivot--;
        if (pivot < 0) {
            reverse(a, 0, n - 1);
            return false;
        }
        int successor = n - 1;
        while (a[successor] <= a[pivot]) successor--;
        swap(a, pivot, successor);
        reverse(a, pivot + 1, n - 1);
        return true;
    }

    private static void reverse(int[] a, int lo, int hi) {
        while (lo < hi) swap(a, lo++, hi--);
    }

    private static void swap(int[] a, int i, int j) {
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }

    public static void main(String[] args) {
        int[] a = {1, 2, 3};
        check(nextPermutation(a) && a[0] == 1 && a[1] == 3 && a[2] == 2, "123 -> 132");
        check(nextPermutation(a) && a[0] == 2 && a[1] == 1 && a[2] == 3, "132 -> 213");

        int[] last = {3, 2, 1};
        check(!nextPermutation(last) && last[0] == 1 && last[1] == 2 && last[2] == 3, "wrap");

        int[] single = {5};
        check(!nextPermutation(single) && single[0] == 5, "single");

        int[] b = {1, 2, 3, 4};
        int seen = 1;
        while (nextPermutation(b)) seen++;
        check(seen == 24 && b[0] == 1 && b[3] == 4, "24 permutations cycle");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
