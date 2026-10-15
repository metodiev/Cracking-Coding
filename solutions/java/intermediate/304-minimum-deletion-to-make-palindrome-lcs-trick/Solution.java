/** Task 304 - Minimum deletions to make a palindrome (LCS trick). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static int lcsLength(String a, String b) {
        int[] previous = new int[b.length() + 1];
        for (int i = 1; i <= a.length(); i++) {
            int[] current = new int[b.length() + 1];
            for (int j = 1; j <= b.length(); j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) current[j] = previous[j - 1] + 1;
                else current[j] = Math.max(previous[j], current[j - 1]);
            }
            previous = current;
        }
        return previous[b.length()];
    }

    /** Direct interval DP, independent of the LCS formulation. */
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

    static int minimumDeletions(String s) {
        return s.length() - longestPalindromicSubsequence(s);
    }

    private static int seed = 3579;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("abcd".charAt(seed % 4));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(minimumDeletions("aebcbda") == 2, "aebcbda");
        check(minimumDeletions("abc") == 2, "abc");
        check(minimumDeletions("abcd") == 3, "abcd");
        check(minimumDeletions("aa") == 0, "aa");
        check(minimumDeletions("racecar") == 0, "racecar");
        check(minimumDeletions("") == 0, "empty");
        check(minimumDeletions("a") == 0, "single");

        for (String s : new String[]{"aebcbda", "geeksforgeeks", "abca", "abbc", ""}) {
            StringBuilder reversed = new StringBuilder(s).reverse();
            check(longestPalindromicSubsequence(s) == lcsLength(s, reversed.toString()), "lcs trick " + s);
        }

        for (int trial = 0; trial < 60; trial++) {
            String s = lcgString(10);
            StringBuilder reversed = new StringBuilder(s).reverse();
            int lps = longestPalindromicSubsequence(s);
            check(lps == lcsLength(s, reversed.toString()), "random " + s);
            check(s.length() - lps >= 0 && s.length() - lps <= s.length(), "bounds " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
