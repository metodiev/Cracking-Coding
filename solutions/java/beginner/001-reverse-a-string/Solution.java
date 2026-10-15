/** Task 001 - Reverse a string. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static String reverseString(String s) {
        return new StringBuilder(s).reverse().toString();
    }

    public static void main(String[] args) {
        check(reverseString("hello").equals("olleh"), "hello");
        check(reverseString("").equals(""), "empty");
        check(reverseString("ab").equals("ba"), "even");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
