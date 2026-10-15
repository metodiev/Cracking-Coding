/** Task 221 - Circular array maximum subarray (Kadane variant). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    /** Best subarray sum when the array may wrap past the end. */
    static int maxCircularSubarray(int[] nums) {
        if (nums.length == 0) throw new IllegalArgumentException("empty array has no subarray");
        int best = nums[0], cur = nums[0];       // Kadane: maximum subarray
        int worst = nums[0], low = nums[0];      // Kadane: minimum subarray
        int total = 0;
        for (int x : nums) total += x;
        for (int i = 1; i < nums.length; i++) {
            cur = Math.max(nums[i], cur + nums[i]);
            best = Math.max(best, cur);
            low = Math.min(nums[i], low + nums[i]);
            worst = Math.min(worst, low);
        }
        if (best < 0) return best;               // all negative: wrapping picks nothing
        return Math.max(best, total - worst);
    }

    public static void main(String[] args) {
        check(maxCircularSubarray(new int[]{5, -3, 5}) == 10, "wrap [5] + [5]");
        check(maxCircularSubarray(new int[]{-3, -2, -3}) == -2, "all negative");
        check(maxCircularSubarray(new int[]{1, -2, 3, -2}) == 3, "linear beats wrap");
        check(maxCircularSubarray(new int[]{1, 2, 3}) == 6, "all positive");
        check(maxCircularSubarray(new int[]{8, -8, 9}) == 17, "wrap [9] + [8]");
        check(maxCircularSubarray(new int[]{11, -1, 11}) == 22, "wrap skips the dip");
        check(maxCircularSubarray(new int[]{5}) == 5, "single");
        check(maxCircularSubarray(new int[]{-5}) == -5, "single negative");
        check(maxCircularSubarray(new int[]{0}) == 0, "single zero");

        boolean threw = false;
        try { maxCircularSubarray(new int[]{}); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "empty rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
