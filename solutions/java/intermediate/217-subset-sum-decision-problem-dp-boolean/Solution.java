/** Task 217 - Subset sum decision problem (DP boolean). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static boolean subsetSum(int[] nums, int target) {
        if (target < 0) return false;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        for (int x : nums)
            for (int s = target; s >= x; s--)
                if (dp[s - x]) dp[s] = true;
        return dp[target];
    }

    /** Exponential memoized variant, cross-checked against the DP on small inputs. */
    static boolean subsetSumRecursive(int[] nums, int target) {
        return go(nums, 0, target, new HashMap<>());
    }

    private static boolean go(int[] nums, int i, int remaining, Map<Long, Boolean> memo) {
        if (remaining == 0) return true;
        if (i == nums.length || remaining < 0) return false;
        long key = (long) i * 1_000_003 + remaining;
        Boolean hit = memo.get(key);
        if (hit != null) return hit;
        boolean value = go(nums, i + 1, remaining, memo) || go(nums, i + 1, remaining - nums[i], memo);
        memo.put(key, value);
        return value;
    }

    public static void main(String[] args) {
        check(subsetSum(new int[]{3, 34, 4, 12, 5, 2}, 9), "classic true");
        check(!subsetSum(new int[]{3, 34, 4, 12, 5, 2}, 30), "classic false");
        check(subsetSum(new int[]{}, 0), "empty target 0");
        check(!subsetSum(new int[]{}, 5), "empty target 5");
        check(subsetSum(new int[]{5}, 5), "single hit");
        check(!subsetSum(new int[]{5}, 3), "single miss");

        Random rng = new Random(2);
        for (int t = 0; t < 200; t++) {
            int n = rng.nextInt(9);
            int[] nums = new int[n];
            for (int i = 0; i < n; i++) nums[i] = 1 + rng.nextInt(14);
            int target = rng.nextInt(40);
            check(subsetSum(nums, target) == subsetSumRecursive(nums, target), "cross-check " + t);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
