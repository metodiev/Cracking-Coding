/** Task 128 - Implement basic memoization wrapper. (Tier: Beginner, CTCI Ch 8) */
import java.util.*;
import java.util.function.Function;

public class Solution {
    /** Generic memoizer keyed by the argument list (deep-ish via List.toString). */
    static <T, R> Function<T, R> memoize(Function<T, R> fn, Map<T, R> cache) {
        return arg -> cache.computeIfAbsent(arg, fn);
    }

    static Map<Integer, Long> fibCache = new HashMap<>();
    static int fibCalls = 0;

    static long fib(int n) {
        if (fibCache.containsKey(n)) return fibCache.get(n);
        fibCalls++;
        long value = n <= 1 ? n : fib(n - 1) + fib(n - 2);
        fibCache.put(n, value);
        return value;
    }

    public static void main(String[] args) {
        check(fib(30) == 832040L, "fib(30)");
        check(fibCalls == 31, "distinct subproblems only (0..30), got " + fibCalls);
        check(fibCache.get(10) == 55L, "cache inspectable");

        // generic wrapper on a pure function
        Map<Integer, Long> squares = new HashMap<>();
        Function<Integer, Long> memoSquare = memoize(x -> (long) x * x, squares);
        check(memoSquare.apply(9) == 81L, "square");
        check(memoSquare.apply(9) == 81L && squares.size() == 1, "wrapped result cached once");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
