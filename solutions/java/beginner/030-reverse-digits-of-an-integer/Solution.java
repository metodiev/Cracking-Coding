/** Task 030 - Reverse digits of an integer. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static int reverseDigits(int n) {
        long result = 0;
        while (n != 0) {
            result = result * 10 + n % 10;
            n /= 10;
        }
        return (result < Integer.MIN_VALUE || result > Integer.MAX_VALUE) ? 0 : (int) result;
    }

    public static void main(String[] args) {
        check(reverseDigits(123) == 321, "123");
        check(reverseDigits(-123) == -321, "-123");
        check(reverseDigits(0) == 0, "0");
        check(reverseDigits(1534236469) == 0, "overflow clamps to 0");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
