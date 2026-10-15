/** Task 031 - Palindrome number check (without string conversion). (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static boolean isPalindromeNumber(int n) {
        if (n < 0) return false;
        int original = n;
        long reversed = 0;
        while (n != 0) { reversed = reversed * 10 + n % 10; n /= 10; }
        return original == reversed;
    }

    public static void main(String[] args) {
        check(isPalindromeNumber(121), "121");
        check(!isPalindromeNumber(-121), "-121");
        check(isPalindromeNumber(0), "0");
        check(!isPalindromeNumber(123), "123");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
