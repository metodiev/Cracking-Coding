/** Task 306 - Reconstruct the shortest common supersequence. (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static String shortestCommonSupersequence(String a, String b) {
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

        StringBuilder reversed = new StringBuilder();
        int i = n, j = m;
        while (i > 0 && j > 0) {
            if (a.charAt(i - 1) == b.charAt(j - 1)) {
                reversed.append(a.charAt(i - 1));
                i--; j--;
            } else if (dp[i - 1][j] <= dp[i][j - 1]) {   // tie prefers a's character
                reversed.append(a.charAt(i - 1));
                i--;
            } else {
                reversed.append(b.charAt(j - 1));
                j--;
            }
        }
        while (i > 0) reversed.append(a.charAt(--i));
        while (j > 0) reversed.append(b.charAt(--j));
        return reversed.reverse().toString();
    }

    static boolean isSubsequence(String candidate, String text) {
        int idx = 0;
        for (int i = 0; i < text.length() && idx < candidate.length(); i++) {
            if (text.charAt(i) == candidate.charAt(idx)) idx++;
        }
        return idx == candidate.length();
    }

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

    private static int seed = 3141;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("ab".charAt(seed % 2));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(shortestCommonSupersequence("ab", "b").equals("ab"), "ab/b");
        check(shortestCommonSupersequence("abc", "abc").equals("abc"), "identical");
        check(shortestCommonSupersequence("", "abc").equals("abc"), "empty left");
        check(shortestCommonSupersequence("abc", "").equals("abc"), "empty right");
        check(shortestCommonSupersequence("", "").isEmpty(), "both empty");

        String a = "AGGTAB", b = "GXTXAYB";
        String scs = shortestCommonSupersequence(a, b);
        check(scs.length() == 9 && scs.length() == a.length() + b.length() - lcsLength(a, b), "classic length");
        check(isSubsequence(a, scs) && isSubsequence(b, scs), "classic contents");

        String[] strings = new String[40];
        for (int i = 0; i < strings.length; i++) strings[i] = lcgString(5);
        for (int i = 0; i + 1 < strings.length; i += 2) {
            String x = strings[i], y = strings[i + 1];
            String candidate = shortestCommonSupersequence(x, y);
            check(candidate.length() == x.length() + y.length() - lcsLength(x, y), "random length " + x + "/" + y);
            check(isSubsequence(x, candidate) && isSubsequence(y, candidate), "random contents " + x + "/" + y);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
