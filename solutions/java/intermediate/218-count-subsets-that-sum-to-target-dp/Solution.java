/** Task 218 - Count subsets that sum to target (DP). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    static long countSubsets(int[] nums, int target) {
        if (target < 0) return 0;
        long[] dp = new long[target + 1];
        dp[0] = 1;
        for (int x : nums)
            for (int s = target; s >= x; s--)
                dp[s] += dp[s - x];
        return dp[target];
    }

    public static void main(String[] args) {
        check(countSubsets(new int[]{1, 1, 1, 1}, 2) == 6, "four ones");
        check(countSubsets(new int[]{1, 2, 3}, 3) == 2, "two subsets");
        check(countSubsets(new int[]{}, 0) == 1, "empty subset");
        check(countSubsets(new int[]{}, 5) == 0, "impossible");
        check(countSubsets(new int[]{0}, 0) == 2, "zero doubles");
        check(countSubsets(new int[]{2, 3, 5}, 5) == 2, "two ways");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
