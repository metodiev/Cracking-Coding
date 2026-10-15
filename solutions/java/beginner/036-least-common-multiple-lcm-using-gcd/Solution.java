/** Task 036 - Least common multiple (LCM) using GCD. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static long lcm(long a, long b) {
        if (a == 0 || b == 0) return 0;
        long g = gcd(Math.abs(a), Math.abs(b));
        return Math.abs(a / g * b);
    }

    private static long gcd(long a, long b) {
        while (b != 0) { long t = b; b = a % b; a = t; }
        return a;
    }

    public static void main(String[] args) {
        check(lcm(4, 6) == 12, "4/6");
        check(lcm(21, 6) == 42, "21/6");
        check(lcm(0, 5) == 0, "zero");
        check(lcm(-3, 5) == 15, "negative");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
