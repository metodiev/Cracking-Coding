/** Task 129 - Implement tail recursion (if language supports). (Tier: Beginner, CTCI Ch 8) */
import java.math.BigInteger;

public class Solution {
    /** Marker for "call me again later" (a thunk). */
    interface Thunk {
        Object run();
    }

    /** Tail-recursive shape (Java has no TCO; small n only). */
    static long factorialTail(int n, long acc) {
        return n <= 1 ? acc : factorialTail(n - 1, n * acc);
    }

    static Object factStep(int remaining, BigInteger acc) {
        if (remaining <= 1) return acc;
        int next = remaining - 1;
        BigInteger nextAcc = acc.multiply(BigInteger.valueOf(remaining));
        return (Thunk) () -> factStep(next, nextAcc);
    }

    static Object trampoline(Thunk start) {
        Object result = start;
        while (result instanceof Thunk) result = ((Thunk) result).run();
        return result;
    }

    /** Stack-safe: constant stack depth regardless of n. */
    static BigInteger factorialTrampolined(int n) {
        return (BigInteger) trampoline(() -> factStep(n, BigInteger.ONE));
    }

    public static void main(String[] args) {
        check(factorialTail(5, 1) == 120, "tail 5");
        check(factorialTail(20, 1) == 2432902008176640000L, "tail 20");

        // iterative reference for the deep comparison (5000! exceeds long)
        BigInteger expected = BigInteger.ONE;
        for (int i = 2; i <= 5000; i++) expected = expected.multiply(BigInteger.valueOf(i));
        check(factorialTrampolined(5000).equals(expected), "trampoline matches 5000!");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
