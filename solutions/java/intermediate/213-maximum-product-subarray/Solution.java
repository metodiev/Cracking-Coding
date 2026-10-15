/** Task 213 - Maximum product subarray. (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    static long maxProduct(int[] nums) {
        if (nums.length == 0) throw new IllegalArgumentException("empty array has no subarray");
        long best = nums[0], curMax = nums[0], curMin = nums[0];
        for (int i = 1; i < nums.length; i++) {
            long x = nums[i];
            long a = x, b = curMax * x, c = curMin * x;
            curMax = Math.max(a, Math.max(b, c));
            curMin = Math.min(a, Math.min(b, c));
            best = Math.max(best, curMax);
        }
        return best;
    }

    public static void main(String[] args) {
        check(maxProduct(new int[]{2, 3, -2, 4}) == 6, "classic");
        check(maxProduct(new int[]{-2, 0, -1}) == 0, "zero split");
        check(maxProduct(new int[]{-2, 3, -4}) == 24, "double negative");
        check(maxProduct(new int[]{0, 2}) == 2, "leading zero");
        check(maxProduct(new int[]{-2}) == -2, "single negative");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
