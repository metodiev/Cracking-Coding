/** Task 125 - Climbing stairs DP (count ways). (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static long climbWays(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        long a = 1, b = 1;
        for (int i = 0; i < n; i++) {
            long next = a + b;
            a = b;
            b = next;
        }
        return a;
    }

    static long climbWaysGeneral(int n, int[] steps) {
        long[] dp = new long[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++)
            for (int s : steps)
                if (s <= i) dp[i] += dp[i - s];
        return dp[n];
    }

    public static void main(String[] args) {
        long[] expected = {1, 1, 2, 3, 5, 8};
        for (int n = 0; n < expected.length; n++)
            check(climbWays(n) == expected[n], "n=" + n);
        check(climbWays(35) == 14930352L, "n=35");
        check(climbWaysGeneral(4, new int[]{1, 2, 3}) == 7, "steps [1,2,3]");
        check(climbWaysGeneral(0, new int[]{1, 2}) == 1, "zero stairs");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
