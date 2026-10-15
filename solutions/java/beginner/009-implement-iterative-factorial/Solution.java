/** Task 009 - Implement iterative factorial. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static long factorialIterative(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        long result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }

    public static void main(String[] args) {
        check(factorialIterative(0) == 1, "0");
        check(factorialIterative(1) == 1, "1");
        check(factorialIterative(5) == 120, "5");
        check(factorialIterative(20) == 2432902008176640000L, "20");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
