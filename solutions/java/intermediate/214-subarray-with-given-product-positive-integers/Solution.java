/** Task 214 - Subarray with given product (positive integers). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    /** Returns {start, end} inclusive, or null. */
    static int[] subarrayWithProduct(int[] nums, int target) {
        for (int x : nums) if (x <= 0) throw new IllegalArgumentException("requires positive values");
        if (target < 1) return null;
        int left = 0;
        long product = 1;
        for (int right = 0; right < nums.length; right++) {
            product *= nums[right];
            while (left <= right && product > target) {
                product /= nums[left];
                left++;
            }
            if (left <= right && product == target) return new int[]{left, right};
        }
        return null;
    }

    public static void main(String[] args) {
        check(java.util.Arrays.equals(subarrayWithProduct(new int[]{2, 3, 1, 2, 4}, 6), new int[]{0, 1}), "first match");
        check(java.util.Arrays.equals(subarrayWithProduct(new int[]{1, 2, 3, 4}, 6), new int[]{0, 2}), "later match");
        check(java.util.Arrays.equals(subarrayWithProduct(new int[]{10}, 10), new int[]{0, 0}), "single");
        check(subarrayWithProduct(new int[]{3, 1, 1}, 2) == null, "none with ones");
        check(subarrayWithProduct(new int[]{2, 4}, 6) == null, "none");
        check(java.util.Arrays.equals(subarrayWithProduct(new int[]{1, 1, 1}, 1), new int[]{0, 0}), "ones");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
