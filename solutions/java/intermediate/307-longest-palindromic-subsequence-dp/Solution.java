/** Task 307 - Longest palindromic subsequence (DP). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static int longestPalindromicSubsequence(String s) {
        int n = s.length();
        if (n == 0) return 0;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) dp[i][i] = 1;
        for (int length = 2; length <= n; length++) {
            for (int i = 0; i + length - 1 < n; i++) {
                int j = i + length - 1;
                if (s.charAt(i) == s.charAt(j)) dp[i][j] = length == 2 ? 2 : dp[i + 1][j - 1] + 2;
                else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
        return dp[0][n - 1];
    }

    static int bruteForce(String s) {
        int best = 0;
        for (int mask = 0; mask < (1 << s.length()); mask++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); i++) if ((mask >> i & 1) == 1) sb.append(s.charAt(i));
            String candidate = sb.toString();
            if (candidate.length() > best && candidate.equals(sb.reverse().toString())) best = candidate.length();
        }
        return best;
    }

    private static int seed = 9753;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("abc".charAt(seed % 3));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(longestPalindromicSubsequence("bbbab") == 4, "bbbab");
        check(longestPalindromicSubsequence("cbbd") == 2, "cbbd");
        check(longestPalindromicSubsequence("abcde") == 1, "abcde");
        check(longestPalindromicSubsequence("racecar") == 7, "racecar");
        check(longestPalindromicSubsequence("") == 0, "empty");
        check(longestPalindromicSubsequence("a") == 1, "single");

        for (int trial = 0; trial < 60; trial++) {
            String s = lcgString(10);
            check(longestPalindromicSubsequence(s) == bruteForce(s), "random " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
