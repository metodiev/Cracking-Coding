/** Task 075 - Binary search iterative on sorted array. (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int binarySearch(int[] a, int target) {
        int lo = 0, hi = a.length - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (a[mid] == target) return mid;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] a = {1, 3, 5, 7, 9, 11};
        check(binarySearch(a, 7) == 3, "middle");
        check(binarySearch(a, 1) == 0, "first");
        check(binarySearch(a, 11) == 5, "last");
        check(binarySearch(a, 4) == -1, "absent");
        check(binarySearch(new int[]{}, 5) == -1, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
