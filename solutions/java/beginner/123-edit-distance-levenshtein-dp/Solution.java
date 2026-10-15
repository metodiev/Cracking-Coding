/** Task 123 - Edit distance (Levenshtein) DP. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static int editDistance(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
                dp[i][j] = a.charAt(i - 1) == b.charAt(j - 1)
                        ? dp[i - 1][j - 1]
                        : 1 + Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1]));
        return dp[n][m];
    }

    static int editDistanceRollingRow(String a, String b) {
        int[] prev = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            int[] cur = new int[b.length() + 1];
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++)
                cur[j] = a.charAt(i - 1) == b.charAt(j - 1)
                        ? prev[j - 1]
                        : 1 + Math.min(prev[j], Math.min(cur[j - 1], prev[j - 1]));
            prev = cur;
        }
        return prev[b.length()];
    }

    public static void main(String[] args) {
        check(editDistance("kitten", "sitting") == 3, "kitten/sitting");
        check(editDistance("", "abc") == 3, "empty->abc");
        check(editDistance("abc", "") == 3, "abc->empty");
        check(editDistance("same", "same") == 0, "identical");
        check(editDistance("flaw", "lawn") == 2, "flaw/lawn");
        check(editDistance("abc", "yabd") == editDistanceRollingRow("abc", "yabd"), "rolling row agrees");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
