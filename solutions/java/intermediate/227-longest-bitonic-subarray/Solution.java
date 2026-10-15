/** Task 227 - Longest bitonic subarray. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Longest contiguous subarray that strictly rises then strictly falls. */
    static int longestBitonicSubarray(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        int[] inc = new int[n], dec = new int[n];
        inc[0] = 1;
        for (int i = 1; i < n; i++) inc[i] = nums[i] > nums[i - 1] ? inc[i - 1] + 1 : 1;
        dec[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) dec[i] = nums[i] > nums[i + 1] ? dec[i + 1] + 1 : 1;
        int best = 0;
        for (int i = 0; i < n; i++) best = Math.max(best, inc[i] + dec[i] - 1);
        return best;
    }

    public static void main(String[] args) {
        check(longestBitonicSubarray(new int[]{2, 1, 4, 5, 6, 3, 2}) == 6, "[1,4,5,6,3,2]");
        check(longestBitonicSubarray(new int[]{1, 5, 1}) == 3, "single peak");
        check(longestBitonicSubarray(new int[]{1, 2, 3, 4}) == 4, "monotone up");
        check(longestBitonicSubarray(new int[]{4, 3, 2, 1}) == 4, "monotone down");
        check(longestBitonicSubarray(new int[]{7}) == 1, "single");
        check(longestBitonicSubarray(new int[]{}) == 0, "empty");
        check(longestBitonicSubarray(new int[]{1, 1}) == 1, "not strict");
        check(longestBitonicSubarray(new int[]{12, 4, 78, 90, 45, 23, 100}) == 5, "[4,78,90,45,23]");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
