/** Task 002 - Check if a string is a palindrome. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static boolean isPalindrome(String s) {
        for (int i = 0, j = s.length() - 1; i < j; i++, j--)
            if (s.charAt(i) != s.charAt(j)) return false;
        return true;
    }

    public static void main(String[] args) {
        check(isPalindrome("racecar"), "racecar");
        check(isPalindrome(""), "empty");
        check(isPalindrome("a"), "single");
        check(!isPalindrome("hello"), "hello");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
