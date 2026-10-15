/** Task 046 - Implement substring search (naive). (Tier: Beginner, CTCI Ch 10) */
public class Solution {
    static int substringSearch(String text, String pattern) {
        int n = text.length(), m = pattern.length();
        if (m == 0) return 0;
        for (int i = 0; i + m <= n; i++) {
            int j = 0;
            while (j < m && text.charAt(i + j) == pattern.charAt(j)) j++;
            if (j == m) return i;
        }
        return -1;
    }

    public static void main(String[] args) {
        check(substringSearch("hello world", "world") == 6, "found");
        check(substringSearch("aaaa", "aa") == 0, "first occurrence");
        check(substringSearch("abc", "xyz") == -1, "missing");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
