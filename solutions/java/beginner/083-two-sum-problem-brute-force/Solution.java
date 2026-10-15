/** Task 083 - Two-sum problem (brute force). (Tier: Beginner, CTCI Ch 1)
 *
 * O(n^2) double loop. Returns the indices of the first pair found (smallest i,
 * then smallest j > i) whose values add to target, or {-1, -1} when no pair
 * exists.
 */
import java.util.*;

public class Solution {
    static int[] twoSumBrute(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) return new int[]{i, j};
            }
        }
        return new int[]{-1, -1};
    }

    public static void main(String[] args) {
        check(Arrays.equals(twoSumBrute(new int[]{2, 7, 11, 15}, 9), new int[]{0, 1}), "classic");
        check(Arrays.equals(twoSumBrute(new int[]{3, 3}, 6), new int[]{0, 1}), "equal values");
        check(Arrays.equals(twoSumBrute(new int[]{3, 2, 4}, 6), new int[]{1, 2}), "later pair");
        check(Arrays.equals(twoSumBrute(new int[]{-3, 4, 3, 90}, 0), new int[]{0, 2}), "negatives");
        check(Arrays.equals(twoSumBrute(new int[]{1, 5, 3, 4}, 9), new int[]{1, 3}), "first pair with smallest i");
        check(Arrays.equals(twoSumBrute(new int[]{2, 2, 2, 2}, 4), new int[]{0, 1}), "duplicates");
        check(Arrays.equals(twoSumBrute(new int[]{1, 2, 3}, 7), new int[]{-1, -1}), "none");
        check(Arrays.equals(twoSumBrute(new int[]{}, 0), new int[]{-1, -1}), "empty");
        check(Arrays.equals(twoSumBrute(new int[]{5}, 5), new int[]{-1, -1}), "single");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
