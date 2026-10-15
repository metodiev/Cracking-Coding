/** Task 010 - Implement recursive factorial. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static long factorialRecursive(int n) {
        if (n < 0) throw new IllegalArgumentException("n must be >= 0");
        return n <= 1 ? 1 : n * factorialRecursive(n - 1);
    }

    public static void main(String[] args) {
        check(factorialRecursive(0) == 1, "0");
        check(factorialRecursive(5) == 120, "5");
        check(factorialRecursive(20) == 2432902008176640000L, "20");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
