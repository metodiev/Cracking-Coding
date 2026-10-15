/** Task 215 - Partition array into equal-sum subsets (subset sum). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    static boolean canPartition(int[] nums) {
        int total = 0;
        for (int x : nums) total += x;
        if (total % 2 != 0) return false;
        int half = total / 2;
        boolean[] dp = new boolean[half + 1];
        dp[0] = true;
        for (int x : nums)
            for (int s = half; s >= x; s--)
                if (dp[s - x]) dp[s] = true;
        return dp[half];
    }

    public static void main(String[] args) {
        check(canPartition(new int[]{1, 5, 11, 5}), "classic true");
        check(!canPartition(new int[]{1, 2, 3, 5}), "odd total");
        check(canPartition(new int[]{2, 2}), "pair");
        check(canPartition(new int[]{1, 1}), "two ones");
        check(!canPartition(new int[]{1}), "single");
        check(canPartition(new int[]{}), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
