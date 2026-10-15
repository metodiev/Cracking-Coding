/** Task 044 - Trim leading and trailing spaces from string. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static String trim(String s) {
        int lo = 0, hi = s.length();
        while (lo < hi && s.charAt(lo) == ' ') lo++;
        while (hi > lo && s.charAt(hi - 1) == ' ') hi--;
        return s.substring(lo, hi);
    }

    public static void main(String[] args) {
        check(trim("  hello  ").equals("hello"), "both sides");
        check(trim("hello").equals("hello"), "already trimmed");
        check(trim("   ").equals(""), "all spaces");
        check(trim("  a b ").equals("a b"), "inner space kept");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
