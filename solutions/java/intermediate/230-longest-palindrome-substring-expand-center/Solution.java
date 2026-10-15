/** Task 230 - Longest palindrome substring (expand center). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    /**
     * Expands around each of the 2n-1 centers. Centers are visited left to
     * right with the odd center before the even one, and only strictly longer
     * palindromes replace the answer, so the first longest one wins ties.
     */
    static String longestPalindromeSubstring(String s) {
        if (s.isEmpty()) return "";
        int start = 0, length = 1;
        for (int c = 0; c < s.length(); c++) {
            int[] odd = expand(s, c, c);
            if (odd[1] > length) { start = odd[0]; length = odd[1]; }
            int[] even = expand(s, c, c + 1);
            if (even[1] > length) { start = even[0]; length = even[1]; }
        }
        return s.substring(start, start + length);
    }

    /** Returns {start, length} of the longest palindrome around (l, r). */
    static int[] expand(String s, int l, int r) {
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) { l--; r++; }
        return new int[]{l + 1, r - l - 1};
    }

    public static void main(String[] args) {
        check(longestPalindromeSubstring("babad").equals("bab"), "tie goes to the first found");
        check(longestPalindromeSubstring("cbbd").equals("bb"), "even center");
        check(longestPalindromeSubstring("a").equals("a"), "single char");
        check(longestPalindromeSubstring("").equals(""), "empty");
        check(longestPalindromeSubstring("aaaa").equals("aaaa"), "all same");
        check(longestPalindromeSubstring("forgeeksskeegfor").equals("geeksskeeg"), "long even");
        check(longestPalindromeSubstring("abaxyzzyxf").equals("xyzzyx"), "long even 2");
        check(longestPalindromeSubstring("ac").equals("a"), "no palindrome pair");
        check(longestPalindromeSubstring("aac").equals("aa"), "prefix pair");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
