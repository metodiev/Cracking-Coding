/** Task 093 - Implement basic regex-like matching for '.' and '*'. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static boolean isMatch(String s, String p) {
        int[][] memo = new int[s.length() + 1][p.length() + 1];  // 0 unknown, 1 true, 2 false
        return rec(s, p, 0, 0, memo);
    }

    private static boolean rec(String s, String p, int i, int j, int[][] memo) {
        if (memo[i][j] != 0) return memo[i][j] == 1;
        boolean res;
        if (j == p.length()) {
            res = i == s.length();
        } else {
            boolean first = i < s.length() && (p.charAt(j) == s.charAt(i) || p.charAt(j) == '.');
            if (j + 1 < p.length() && p.charAt(j + 1) == '*') {
                res = rec(s, p, i, j + 2, memo) || (first && rec(s, p, i + 1, j, memo));
            } else {
                res = first && rec(s, p, i + 1, j + 1, memo);
            }
        }
        memo[i][j] = res ? 1 : 2;
        return res;
    }

    public static void main(String[] args) {
        check(isMatch("", ""), "empty/empty");
        check(!isMatch("", "a"), "empty/a");
        check(isMatch("", "a*"), "empty/a*");
        check(isMatch("", ".*"), "empty/.*");
        check(!isMatch("a", ""), "a/empty");
        check(isMatch("a", "a"), "a/a");
        check(isMatch("a", "."), "a/.");
        check(!isMatch("a", "b"), "a/b");
        check(!isMatch("aa", "a"), "aa/a");
        check(isMatch("aa", "a*"), "aa/a*");
        check(isMatch("aa", ".*"), "aa/.*");
        check(isMatch("ab", ".*"), "ab/.*");
        check(isMatch("aab", "c*a*b"), "aab/c*a*b");
        check(isMatch("abc", "ab.*c"), "abc/ab.*c");
        check(isMatch("abXYZc", "ab.*c"), "abXYZc/ab.*c");
        check(!isMatch("ab", "ab.*c"), "ab/ab.*c");
        check(isMatch("aaa", "a*a"), "aaa/a*a");
        check(!isMatch("aaa", "ab*a"), "aaa/ab*a");
        check(!isMatch("mississippi", "mis*is*p*."), "mississippi");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
