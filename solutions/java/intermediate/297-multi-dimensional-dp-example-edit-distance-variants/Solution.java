/** Task 297 - Edit distance variants (multi-dimensional DP). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /** Minimum cost to turn a into b using insert/delete/replace operations. */
    static int editDistance(String a, String b, int insertCost, int deleteCost, int replaceCost) {
        int n = a.length();
        int m = b.length();
        int[][] dp = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) dp[i][0] = i * deleteCost;
        for (int j = 1; j <= m; j++) dp[0][j] = j * insertCost;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                int substitution = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : replaceCost;
                dp[i][j] = Math.min(
                        dp[i - 1][j - 1] + substitution,
                        Math.min(dp[i - 1][j] + deleteCost, dp[i][j - 1] + insertCost));
            }
        }
        return dp[n][m];
    }

    /** Independent memoized recursion straight from the operation definition. */
    static int bruteEdit(String a, String b, int insertCost, int deleteCost, int replaceCost) {
        int[][] memo = new int[a.length() + 1][b.length() + 1];
        for (int[] row : memo) Arrays.fill(row, -1);
        return rec(a, b, a.length(), b.length(), insertCost, deleteCost, replaceCost, memo);
    }

    private static int rec(String a, String b, int i, int j, int ins, int del, int rep, int[][] memo) {
        if (i == 0) return j * ins;
        if (j == 0) return i * del;
        if (memo[i][j] >= 0) return memo[i][j];
        int substitution = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : rep;
        int best = Math.min(
                rec(a, b, i - 1, j - 1, ins, del, rep, memo) + substitution,
                Math.min(rec(a, b, i - 1, j, ins, del, rep, memo) + del,
                        rec(a, b, i, j - 1, ins, del, rep, memo) + ins));
        memo[i][j] = best;
        return best;
    }

    private static int seed = 31337;

    private static String lcgString(int maxLength) {
        seed = (seed * 75 + 74) % 65537;
        int length = seed % (maxLength + 1);
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append("abc".charAt(seed % 3));
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        check(editDistance("kitten", "sitting", 1, 1, 1) == 3, "kitten -> sitting");
        check(editDistance("flaw", "lawn", 1, 1, 1) == 2, "flaw -> lawn");
        check(editDistance("abc", "abc", 1, 1, 1) == 0, "equal strings");
        check(editDistance("", "abcd", 1, 1, 1) == 4, "empty -> non-empty");
        check(editDistance("abcd", "", 1, 1, 1) == 4, "non-empty -> empty");
        check(editDistance("", "", 1, 1, 1) == 0, "both empty");

        check(editDistance("ab", "ba", 1, 1, 2) == 2, "replace 2 still loses to delete+insert");
        check(editDistance("abc", "axc", 1, 1, 3) == 2, "two edits beat one replacement");
        check(editDistance("abc", "axc", 1, 1, 1) == 1, "one replacement with unit costs");

        check(editDistance("abc", "abcd", 1, 5, 1) == 1, "cheap insertion");
        check(editDistance("abcd", "abc", 1, 5, 1) == 5, "expensive deletion");
        check(editDistance("abc", "abcd", 7, 1, 1) == 7, "expensive insertion");

        for (int trial = 0; trial < 200; trial++) {
            String a = lcgString(6);
            String b = lcgString(6);
            seed = (seed * 75 + 74) % 65537;
            int insertCost = 1 + seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int deleteCost = 1 + seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int replaceCost = 1 + seed % 3;
            check(editDistance(a, b, insertCost, deleteCost, replaceCost)
                    == bruteEdit(a, b, insertCost, deleteCost, replaceCost), "random trial " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
