/** Task 084 - Two-sum using hashmap. (Tier: Beginner, CTCI Ch 1)
 *
 * One pass, O(n): keep a map value -> earliest index seen so far and, for each
 * new index j, look up target - nums[j]. A single pass suffices because when
 * index j is reached, every earlier index has already been stored, so any pair
 * (i < j) is detected exactly when its later element arrives; each element is
 * visited once. Returns {i, j} with i < j, or {-1, -1} when no pair exists.
 */
import java.util.*;

public class Solution {
    static int[] twoSumHash(int[] nums, int target) {
        Map<Integer, Integer> seen = new HashMap<>();
        for (int j = 0; j < nums.length; j++) {
            int complement = target - nums[j];
            if (seen.containsKey(complement)) {
                return new int[]{seen.get(complement), j};
            }
            if (!seen.containsKey(nums[j])) {
                seen.put(nums[j], j);
            }
        }
        return new int[]{-1, -1};
    }

    static int[] twoSumBrute(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) return new int[]{i, j};
            }
        }
        return new int[]{-1, -1};
    }

    public static void main(String[] args) {
        check(Arrays.equals(twoSumHash(new int[]{2, 7, 11, 15}, 9), new int[]{0, 1}), "classic");
        check(Arrays.equals(twoSumHash(new int[]{3, 2, 4}, 6), new int[]{1, 2}), "later pair");
        check(Arrays.equals(twoSumHash(new int[]{3, 3}, 6), new int[]{0, 1}), "equal values");
        check(Arrays.equals(twoSumHash(new int[]{-3, 4, 3, 90}, 0), new int[]{0, 2}), "negatives");
        check(Arrays.equals(twoSumHash(new int[]{1, 2, 3}, 7), new int[]{-1, -1}), "none");
        check(Arrays.equals(twoSumHash(new int[]{}, 0), new int[]{-1, -1}), "empty");
        check(Arrays.equals(twoSumHash(new int[]{5}, 5), new int[]{-1, -1}), "single");

        int[][] cases = {{2, 7, 11, 15}, {3, 2, 4}, {3, 3}, {0, 0, 1}, {-3, 4, 3, 90},
                         {1, 2, 3}, {}, {5}, {1, 5, 3, 4}};
        int[] targets = {9, 6, 6, 0, 0, 7, 0, 5, 9};
        for (int c = 0; c < cases.length; c++) {
            int[] got = twoSumHash(cases[c], targets[c]);
            boolean exists = twoSumBrute(cases[c], targets[c])[0] != -1;
            check((got[0] != -1) == exists, "agrees with brute force on case " + c);
            if (got[0] != -1) {
                check(0 <= got[0] && got[0] < got[1] && got[1] < cases[c].length, "valid index order, case " + c);
                check(cases[c][got[0]] + cases[c][got[1]] == targets[c], "valid sum, case " + c);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
