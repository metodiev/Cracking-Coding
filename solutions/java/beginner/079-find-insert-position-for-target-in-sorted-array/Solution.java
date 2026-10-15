/** Task 079 - Find insert position for target in sorted array. (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int searchInsert(int[] a, int target) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void main(String[] args) {
        int[] a = {1, 3, 5, 6};
        check(searchInsert(a, 5) == 2, "existing");
        check(searchInsert(a, 2) == 1, "between");
        check(searchInsert(a, 7) == 4, "after end");
        check(searchInsert(a, 0) == 0, "before start");
        check(searchInsert(new int[]{}, 42) == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
