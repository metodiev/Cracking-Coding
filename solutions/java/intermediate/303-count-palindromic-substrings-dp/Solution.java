/** Task 303 - Count palindromic substrings. (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static int countPalindromicSubstrings(String s) {
        int n = s.length();
        int count = 0;
        for (int center = 0; center < n; center++) {
            for (int[] seed : new int[][]{{center, center}, {center, center + 1}}) {
                int left = seed[0], right = seed[1];
                while (left >= 0 && right < n && s.charAt(left) == s.charAt(right)) {
                    count++;
                    left--;
                    right++;
                }
            }
        }
        return count;
    }

    static int bruteForce(String s) {
        int total = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) {
                String part = s.substring(i, j);
                if (part.equals(new StringBuilder(part).reverse().toString())) total++;
            }
        }
        return total;
    }

    private static int seed = 2468;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("abc".charAt(seed % 3));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(countPalindromicSubstrings("abc") == 3, "abc");
        check(countPalindromicSubstrings("aaa") == 6, "aaa");
        check(countPalindromicSubstrings("abba") == 6, "abba");
        check(countPalindromicSubstrings("") == 0, "empty");
        check(countPalindromicSubstrings("a") == 1, "single");

        for (String s : new String[]{"aab", "aba", "abcba", "abccba", "aaaa"})
            check(countPalindromicSubstrings(s) == bruteForce(s), "brute " + s);

        for (int trial = 0; trial < 60; trial++) {
            String s = lcgString(9);
            check(countPalindromicSubstrings(s) == bruteForce(s), "random " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
