/** Task 158 - Implement simple memoized prime checker cache. (Tier: Beginner, CTCI Ch 6) */
import java.util.*;

/**
 * Trial division up to sqrt(n) with a verdict cache; hits/misses counters prove
 * that repeated queries for the same number are served from the cache.
 */
class PrimeChecker {
    private final Map<Integer, Boolean> cache = new HashMap<>();
    int hits;
    int misses;

    /** Memoized primality test: n < 2 is never prime. */
    boolean isPrime(int n) {
        Boolean cached = cache.get(n);
        if (cached != null) {
            hits++;
            return cached;
        }
        misses++;
        boolean verdict = check(n);
        cache.put(n, verdict);
        return verdict;
    }

    int size() {
        return cache.size();
    }

    /** Trial division up to sqrt(n): O(sqrt(n)) time, O(1) space. */
    private static boolean check(int n) {
        if (n < 2) return false;
        if (n % 2 == 0) return n == 2;
        for (int d = 3; (long) d * d <= n; d += 2) {
            if (n % d == 0) return false;
        }
        return true;
    }
}

public class Solution {
    public static void main(String[] args) {
        PrimeChecker checker = new PrimeChecker();
        check(checker.isPrime(97), "97 is prime");
        check(checker.hits == 0 && checker.misses == 1 && checker.size() == 1, "one miss, nothing cached before");

        // repeated queries for the same number hit the cache
        check(checker.isPrime(97) && checker.isPrime(97), "repeat hits");
        check(checker.hits == 2 && checker.misses == 1 && checker.size() == 1, "two hits, one miss");

        int[][] cases = {{0, 0}, {1, 0}, {2, 1}, {100, 0}, {7919, 1}, {7917, 0}};
        for (int[] c : cases) {
            check(checker.isPrime(c[0]) == (c[1] == 1), "case " + c[0]);
        }
        check(checker.hits == 2 && checker.misses == 7 && checker.size() == 7, "counters after first sweep");

        // repeat the whole set: every answer now comes from the cache
        int[][] again = {{0, 0}, {1, 0}, {2, 1}, {97, 1}, {100, 0}, {7919, 1}, {7917, 0}};
        for (int[] c : again) {
            check(checker.isPrime(c[0]) == (c[1] == 1), "again " + c[0]);
        }
        check(checker.hits == 9 && checker.misses == 7 && checker.size() == 7, "counters after cached sweep");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
