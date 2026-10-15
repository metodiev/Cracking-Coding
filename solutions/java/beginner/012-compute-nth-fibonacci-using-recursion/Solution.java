/** Task 012 - Compute nth Fibonacci using recursion. O(2^n), intentionally naive. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static long fib(int n) {
        return n <= 1 ? n : fib(n - 1) + fib(n - 2);
    }

    public static void main(String[] args) {
        check(fib(0) == 0 && fib(1) == 1, "base cases");
        check(fib(20) == 6765, "fib(20)");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
