/** Task 077 - Find first occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int firstOccurrence(int[] a, int target) {
        int lo = 0, hi = a.length;          // [lo, hi)
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (a[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return (lo < a.length && a[lo] == target) ? lo : -1;
    }

    public static void main(String[] args) {
        int[] a = {1, 2, 2, 2, 3, 4, 4};
        check(firstOccurrence(a, 2) == 1, "duplicated middle");
        check(firstOccurrence(a, 4) == 5, "duplicated tail");
        check(firstOccurrence(a, 1) == 0, "single head");
        check(firstOccurrence(a, 9) == -1, "absent");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
