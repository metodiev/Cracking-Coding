/** Task 231 - Manacher's algorithm for longest palindromic substring. (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    /** Manacher: leftmost longest palindromic substring of s; "" for empty input. */
    static String longestPalindrome(String s) {
        if (s.isEmpty()) return "";
        StringBuilder sb = new StringBuilder("^");
        for (int k = 0; k < s.length(); k++) sb.append('#').append(s.charAt(k));
        sb.append("#$");
        String t = sb.toString();
        int n = t.length();
        int[] p = new int[n];
        int centre = 0, right = 0, bestLen = 0, bestStart = 0;
        for (int i = 1; i < n - 1; i++) {
            int mirror = 2 * centre - i;
            if (i < right) p[i] = Math.min(right - i, p[mirror]);
            while (t.charAt(i + 1 + p[i]) == t.charAt(i - 1 - p[i])) p[i]++;
            if (i + p[i] > right) { centre = i; right = i + p[i]; }
            if (p[i] > bestLen) {                 // strict: ties keep the leftmost centre
                bestLen = p[i];
                bestStart = (i - p[i]) / 2;
            }
        }
        return s.substring(bestStart, bestStart + bestLen);
    }

    /** O(n^2) reference helper: length of the longest palindromic substring of s. */
    static int expandAroundCentreLength(String s) {
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            best = Math.max(best, expand(s, i, i));
            best = Math.max(best, expand(s, i, i + 1));
        }
        return best;
    }

    private static int expand(String s, int lo, int hi) {
        while (lo >= 0 && hi < s.length() && s.charAt(lo) == s.charAt(hi)) { lo--; hi++; }
        return hi - lo - 1;
    }

    public static void main(String[] args) {
        check(longestPalindrome("babad").equals("bab"), "leftmost of \"bab\"/\"aba\"");
        check(longestPalindrome("cbbd").equals("bb"), "even palindrome");
        check(longestPalindrome("forgeeksskeegfor").equals("geeksskeeg"), "classic");
        check(longestPalindrome("abacaba").equals("abacaba"), "whole string");
        check(longestPalindrome("bananas").equals("anana"), "bananas");
        check(longestPalindrome("abcde").equals("a"), "no repeats");
        check(longestPalindrome("ccc").equals("ccc"), "all same");
        check(longestPalindrome("").equals(""), "empty");
        check(longestPalindrome("a").equals("a"), "single");

        java.util.Random rng = new java.util.Random(231);
        StringBuilder sb = new StringBuilder();
        for (int t = 0; t < 300; t++) {           // cross-check vs expand-around-centre
            int len = rng.nextInt(41);
            sb.setLength(0);
            for (int i = 0; i < len; i++) sb.append(rng.nextBoolean() ? 'a' : 'b');
            String s = sb.toString();
            String got = longestPalindrome(s);
            check(got.equals(new StringBuilder(got).reverse().toString()), "palindromic result");
            check(got.length() == expandAroundCentreLength(s), "length matches expand-around-centre");
            if (!s.isEmpty()) {
                int first = -1;
                for (int i = 0; i + got.length() <= s.length() && first < 0; i++) {
                    String sub = s.substring(i, i + got.length());
                    if (sub.equals(new StringBuilder(sub).reverse().toString())) first = i;
                }
                check(s.indexOf(got) == first, "leftmost among maximal palindromes");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
