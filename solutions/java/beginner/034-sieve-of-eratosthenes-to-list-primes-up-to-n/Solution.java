/** Task 034 - Sieve of Eratosthenes to list primes up to N. (Tier: Beginner, CTCI Ch 6) */
public class Solution {
    static java.util.List<Integer> sieve(int n) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        if (n < 2) return out;
        boolean[] isPrime = new boolean[n + 1];
        java.util.Arrays.fill(isPrime, true);
        isPrime[0] = isPrime[1] = false;
        for (int p = 2; (long) p * p <= n; p++)
            if (isPrime[p])
                for (int m = p * p; m <= n; m += p) isPrime[m] = false;
        for (int x = 2; x <= n; x++) if (isPrime[x]) out.add(x);
        return out;
    }

    public static void main(String[] args) {
        check(sieve(1).isEmpty(), "1");
        check(sieve(30).toString().equals("[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]"), "30");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
