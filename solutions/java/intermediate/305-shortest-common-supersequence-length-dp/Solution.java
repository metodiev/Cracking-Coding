/** Task 305 - Shortest common supersequence length (DP). (Tier: Intermediate, CTCI Ch 1) */
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

    /** Direct DP: shortest string containing both a and b as subsequences. */
    static int scsLength(String a, String b) {
        int n = a.length(), m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 0; i <= n; i++) dp[i][0] = i;
        for (int j = 0; j <= m; j++) dp[0][j] = j;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1)) dp[i][j] = dp[i - 1][j - 1] + 1;
                else dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
            }
        }
        return dp[n][m];
    }

    private static int seed = 4682;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("abc".charAt(seed % 3));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(scsLength("AGGTAB", "GXTXAYB") == 9, "AGGTAB/GXTXAYB");
        check(scsLength("ab", "b") == 2, "ab/b");
        check(scsLength("", "abc") == 3, "empty left");
        check(scsLength("abc", "") == 3, "empty right");
        check(scsLength("", "") == 0, "both empty");
        check(scsLength("abc", "abc") == 3, "identical");
        check(scsLength("geek", "eke") == 5, "geek/eke");

        String[][] pairs = {{"AGGTAB","GXTXAYB"},{"geek","eke"},{"ab","b"}};
        for (String[] pair : pairs) {
            check(scsLength(pair[0], pair[1]) == pair[0].length() + pair[1].length() - lcsLength(pair[0], pair[1]),
                  "lcs formula " + pair[0]);
        }

        String[] strings = new String[40];
        for (int i = 0; i < strings.length; i++) strings[i] = lcgString(6);
        for (int i = 0; i + 1 < strings.length; i += 2) {
            String a = strings[i], b = strings[i + 1];
            check(scsLength(a, b) == a.length() + b.length() - lcsLength(a, b), "random " + a + "/" + b);
            check(scsLength(a, b) >= Math.max(a.length(), b.length()), "lower bound " + a);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
