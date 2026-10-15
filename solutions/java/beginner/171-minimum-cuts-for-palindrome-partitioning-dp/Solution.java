/** Task 171 - Minimum cuts for palindrome partitioning (DP). (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static int minPalindromeCuts(String s) {
        int n = s.length();
        if (n <= 1) return 0;
        boolean[][] isPal = new boolean[n][n];
        int[] cuts = new int[n];
        for (int end = 0; end < n; end++) {
            int best = end;
            for (int start = 0; start <= end; start++) {
                if (s.charAt(start) == s.charAt(end)
                        && (end - start <= 1 || isPal[start + 1][end - 1])) {
                    isPal[start][end] = true;
                    best = start == 0 ? 0 : Math.min(best, cuts[start - 1] + 1);
                }
            }
            cuts[end] = best;
        }
        return cuts[n - 1];
    }

    public static void main(String[] args) {
        check(minPalindromeCuts("aab") == 1, "aab");
        check(minPalindromeCuts("a") == 0, "single");
        check(minPalindromeCuts("") == 0, "empty");
        check(minPalindromeCuts("ab") == 1, "ab");
        check(minPalindromeCuts("aabb") == 1, "aabb");
        check(minPalindromeCuts("abcba") == 0, "already palindrome");
        check(minPalindromeCuts("abcd") == 3, "abcd");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
