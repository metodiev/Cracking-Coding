/** Task 033 - List primes up to N using trial division. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static java.util.List<Integer> primesUpto(int n) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int x = 2; x <= n; x++) if (isPrime(x)) out.add(x);
        return out;
    }

    private static boolean isPrime(int n) {
        for (long i = 2; i * i <= n; i++) if (n % i == 0) return false;
        return true;
    }

    public static void main(String[] args) {
        check(primesUpto(1).isEmpty(), "<=1");
        check(primesUpto(10).toString().equals("[2, 3, 5, 7]"), "10");
        check(primesUpto(20).toString().equals("[2, 3, 5, 7, 11, 13, 17, 19]"), "20");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
