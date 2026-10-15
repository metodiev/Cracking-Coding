/** Task 078 - Find last occurrence of target in sorted array with duplicates. (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int lastOccurrence(int[] a, int target) {
        int lo = 0, hi = a.length;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (a[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        int idx = lo - 1;
        return (idx >= 0 && a[idx] == target) ? idx : -1;
    }

    public static void main(String[] args) {
        int[] a = {1, 2, 2, 2, 3, 4, 4};
        check(lastOccurrence(a, 2) == 3, "duplicated middle");
        check(lastOccurrence(a, 4) == 6, "duplicated tail");
        check(lastOccurrence(a, 1) == 0, "single head");
        check(lastOccurrence(a, 9) == -1, "absent");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
