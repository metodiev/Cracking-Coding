/** Task 076 - Binary search recursive. (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int binarySearchRecursive(int[] a, int target) {
        return search(a, target, 0, a.length - 1);
    }

    private static int search(int[] a, int target, int lo, int hi) {
        if (lo > hi) return -1;
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == target) return mid;
        if (a[mid] < target) return search(a, target, mid + 1, hi);
        return search(a, target, lo, mid - 1);
    }

    public static void main(String[] args) {
        int[] a = {2, 4, 6, 8, 10};
        check(binarySearchRecursive(a, 8) == 3, "found");
        check(binarySearchRecursive(a, 2) == 0, "first");
        check(binarySearchRecursive(a, 5) == -1, "absent");
        check(binarySearchRecursive(new int[]{}, 1) == -1, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
