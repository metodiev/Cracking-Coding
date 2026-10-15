/** Task 013 - Compute nth Fibonacci with memoization. O(n). (Tier: Beginner, CTCI Ch 8) */
import java.util.HashMap;
import java.util.Map;

public class Solution {
    static long fibMemo(int n) {
        Map<Integer, Long> cache = new HashMap<>();
        cache.put(0, 0L);
        cache.put(1, 1L);
        return fibMemo(n, cache);
    }

    private static long fibMemo(int n, Map<Integer, Long> cache) {
        Long hit = cache.get(n);
        if (hit != null) return hit;
        long value = fibMemo(n - 1, cache) + fibMemo(n - 2, cache);
        cache.put(n, value);
        return value;
    }

    public static void main(String[] args) {
        check(fibMemo(50) == 12586269025L, "fib(50)");
        check(fibMemo(0) == 0 && fibMemo(1) == 1, "bases");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
