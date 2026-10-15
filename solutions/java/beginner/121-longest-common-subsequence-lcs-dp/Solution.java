/** Task 121 - Longest common subsequence (LCS) DP. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static int lcsLength(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = a.charAt(i - 1) == b.charAt(j - 1)
                        ? dp[i - 1][j - 1] + 1
                        : Math.max(dp[i - 1][j], dp[i][j - 1]);
        return dp[n][m];
    }

    static String lcsString(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = a.charAt(i - 1) == b.charAt(j - 1)
                        ? dp[i - 1][j - 1] + 1
                        : Math.max(dp[i - 1][j], dp[i][j - 1]);
        StringBuilder out = new StringBuilder();
        int x = n, y = m;
        while (x > 0 && y > 0) {
            if (a.charAt(x - 1) == b.charAt(y - 1)) {
                out.append(a.charAt(x - 1));
                x--; y--;
            } else if (dp[x - 1][y] >= dp[x][y - 1]) {
                x--;
            } else {
                y--;
            }
        }
        return out.reverse().toString();
    }

    static boolean isSubsequence(String s, String t) {
        int j = 0;
        for (int i = 0; i < t.length() && j < s.length(); i++)
            if (t.charAt(i) == s.charAt(j)) j++;
        return j == s.length();
    }

    public static void main(String[] args) {
        check(lcsLength("ABCBDAB", "BDCABA") == 4, "classic");
        check(lcsLength("", "abc") == 0, "empty");
        check(lcsLength("abc", "abc") == 3, "identical");
        String seq = lcsString("ABCBDAB", "BDCABA");
        check(seq.length() == 4 && isSubsequence(seq, "ABCBDAB") && isSubsequence(seq, "BDCABA"),
                "reconstruction is a common subsequence");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
