/** Task 040 - Merge two strings alternately. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static String mergeAlternately(String a, String b) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.max(a.length(), b.length()); i++) {
            if (i < a.length()) sb.append(a.charAt(i));
            if (i < b.length()) sb.append(b.charAt(i));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(mergeAlternately("abc", "123").equals("a1b2c3"), "equal length");
        check(mergeAlternately("ab", "1234").equals("a1b234"), "b longer");
        check(mergeAlternately("", "xyz").equals("xyz"), "a empty");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
