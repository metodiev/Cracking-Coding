/** Task 017 - Convert string to integer (atoi) basic. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int atoi(String s) {
        if (s.isEmpty()) throw new IllegalArgumentException("empty input");
        int i = 0, sign = 1;
        if (s.charAt(0) == '+' || s.charAt(0) == '-') {
            sign = s.charAt(0) == '-' ? -1 : 1;
            i = 1;
        }
        if (i == s.length()) throw new IllegalArgumentException("no digits");
        int value = 0;
        for (; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') throw new IllegalArgumentException("invalid char: " + c);
            value = value * 10 + (c - '0');
        }
        return sign * value;
    }

    public static void main(String[] args) {
        check(atoi("42") == 42, "42");
        check(atoi("-123") == -123, "-123");
        check(atoi("+7") == 7, "+7");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
