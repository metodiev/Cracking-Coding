/** Task 122 - Longest common substring (DP). (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    /** Returns {"length", "substring"} — substring is taken from a. */
    static Object[] longestCommonSubstring(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        int bestLen = 0, bestEnd = 0;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                if (a.charAt(i - 1) == b.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                    if (dp[i][j] > bestLen) {
                        bestLen = dp[i][j];
                        bestEnd = i;
                    }
                }
        return new Object[]{bestLen, a.substring(bestEnd - bestLen, bestEnd)};
    }

    public static void main(String[] args) {
        Object[] r = longestCommonSubstring("abcdef", "zbcdf");
        check((int) r[0] == 3 && r[1].equals("bcd"), "classic");
        Object[] same = longestCommonSubstring("abc", "abc");
        check((int) same[0] == 3 && same[1].equals("abc"), "identical");
        check((int) longestCommonSubstring("abc", "xyz")[0] == 0, "no match");
        check((int) longestCommonSubstring("", "abc")[0] == 0, "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
