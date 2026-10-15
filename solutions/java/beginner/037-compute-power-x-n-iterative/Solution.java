/** Task 037 - Compute power(x, n) iterative. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static double power(double x, int n) {
        int exp = Math.abs(n);
        double result = 1.0;
        for (int i = 0; i < exp; i++) result *= x;
        return n >= 0 ? result : 1.0 / result;
    }

    public static void main(String[] args) {
        check(power(2, 10) == 1024.0, "2^10");
        check(power(2, -2) == 0.25, "2^-2");
        check(power(5, 0) == 1.0, "x^0");
        check(power(3, 3) == 27.0, "3^3");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
