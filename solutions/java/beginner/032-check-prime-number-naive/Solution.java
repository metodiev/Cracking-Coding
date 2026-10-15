/** Task 032 - Check prime number naive. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static boolean isPrime(int n) {
        if (n < 2) return false;
        for (long i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }

    public static void main(String[] args) {
        check(!isPrime(0) && !isPrime(1), "below 2");
        check(isPrime(2) && isPrime(97), "primes");
        check(!isPrime(100), "100");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
