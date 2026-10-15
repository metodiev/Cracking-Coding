/** Task 018 - Convert integer to string (manual itoa). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static String itoa(int n) {
        if (n == 0) return "0";
        boolean negative = n < 0;
        long value = Math.abs((long) n);
        StringBuilder digits = new StringBuilder();
        while (value > 0) {
            digits.append((char) ('0' + value % 10));
            value /= 10;
        }
        digits.reverse();
        return (negative ? "-" : "") + digits;
    }

    public static void main(String[] args) {
        check(itoa(0).equals("0"), "0");
        check(itoa(123).equals("123"), "123");
        check(itoa(-456).equals("-456"), "-456");
        check(itoa(Integer.MIN_VALUE).equals("-2147483648"), "min int");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
