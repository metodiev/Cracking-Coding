/** Task 308 - Count distinct subsequences. (Tier: Intermediate, CTCI Ch 1) */
import java.math.BigInteger;

public class Solution {
    static BigInteger countDistinctSubsequences(String s, String t) {
        int n = s.length(), m = t.length();
        if (m == 0) return BigInteger.ONE;
        if (n < m) return BigInteger.ZERO;
        BigInteger[][] dp = new BigInteger[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) dp[i][j] = BigInteger.ZERO;
            dp[i][0] = BigInteger.ONE;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                dp[i][j] = dp[i - 1][j];
                if (s.charAt(i - 1) == t.charAt(j - 1)) dp[i][j] = dp[i][j].add(dp[i - 1][j - 1]);
            }
        }
        return dp[n][m];
    }

    static long bruteForce(String s, String t) {
        long count = 0;
        for (int mask = 0; mask < (1 << s.length()); mask++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); i++) if ((mask >> i & 1) == 1) sb.append(s.charAt(i));
            if (sb.toString().equals(t)) count++;
        }
        return count;
    }

    private static int seed = 8642;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("ab".charAt(seed % 2));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(countDistinctSubsequences("rabbbit", "rabbit").equals(BigInteger.valueOf(3)), "rabbbit");
        check(countDistinctSubsequences("babgbag", "bag").equals(BigInteger.valueOf(5)), "babgbag");
        check(countDistinctSubsequences("abc", "").equals(BigInteger.ONE), "empty t");
        check(countDistinctSubsequences("", "a").equals(BigInteger.ZERO), "empty s");
        check(countDistinctSubsequences("aaa", "a").equals(BigInteger.valueOf(3)), "aaa/a");
        check(countDistinctSubsequences("aaa", "aa").equals(BigInteger.valueOf(3)), "aaa/aa");
        check(countDistinctSubsequences("abc", "abcd").equals(BigInteger.ZERO), "t longer than s");

        String[] strings = new String[40];
        for (int i = 0; i < strings.length; i++) strings[i] = lcgString(8);
        for (int i = 0; i + 1 < strings.length; i += 2) {
            String s = strings[i];
            String t = strings[i + 1].substring(0, 4);
            check(countDistinctSubsequences(s, t).equals(BigInteger.valueOf(bruteForce(s, t))), "random " + s + "/" + t);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
