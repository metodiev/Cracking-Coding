/** Task 127 - Decode ways (num decodings) DP. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static int numDecodings(String s) {
        if (s.isEmpty()) return 0;
        int n = s.length();
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i <= n; i++) {
            int one = s.charAt(i - 1) - '0';
            int two = Integer.parseInt(s.substring(i - 2, i));
            if (one >= 1 && one <= 9) dp[i] += dp[i - 1];
            if (two >= 10 && two <= 26) dp[i] += dp[i - 2];
        }
        return dp[n];
    }

    public static void main(String[] args) {
        check(numDecodings("12") == 2, "12");
        check(numDecodings("226") == 3, "226");
        check(numDecodings("06") == 0, "leading zero");
        check(numDecodings("10") == 1, "10");
        check(numDecodings("27") == 1, "27");
        check(numDecodings("0") == 0, "single zero");
        check(numDecodings("11106") == 2, "11106");
        check(numDecodings("") == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
