/** Task 300 - Implement convolution (naive) for polynomials. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing) */
import java.util.Arrays;

public class Solution {
    /** Naive O(n*m) convolution: result[k] = sum of a[i] * b[k - i]. */
    static long[] convolve(long[] a, long[] b) {
        if (a.length == 0 || b.length == 0) return new long[0];
        long[] result = new long[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i + j] += a[i] * b[j];
            }
        }
        return result;
    }

    /** Horner evaluation used as an independent oracle for the LCG cross-checks. */
    static long evaluate(long[] polynomial, long x) {
        long value = 0;
        for (int i = polynomial.length - 1; i >= 0; i--) value = value * x + polynomial[i];
        return value;
    }

    private static int seed = 70707;

    private static long[] lcgVector(int size) {
        long[] values = new long[size];
        for (int i = 0; i < size; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % 10;
        }
        return values;
    }

    public static void main(String[] args) {
        check(Arrays.equals(convolve(new long[]{1, 2, 3}, new long[]{0, 1, 1}), new long[]{0, 1, 3, 5, 3}), "sample");
        check(Arrays.equals(convolve(new long[]{5}, new long[]{7}), new long[]{35}), "single terms");
        check(Arrays.equals(convolve(new long[]{5}, new long[]{0}), new long[]{0}), "zero factor");
        check(convolve(new long[]{}, new long[]{1, 2}).length == 0, "empty left");
        check(convolve(new long[]{1, 2}, new long[]{}).length == 0, "empty right");

        long[] a = new long[]{4, 1, 2, 3};
        long[] b = new long[]{2, 0, 5};
        long[] c = new long[]{1, 1, 0, 2};
        check(Arrays.equals(convolve(a, b), convolve(b, a)), "commutativity");
        check(Arrays.equals(convolve(convolve(a, b), c), convolve(a, convolve(b, c))), "associativity");

        check(Arrays.equals(convolve(a, new long[]{1}), a), "identity right");
        check(Arrays.equals(convolve(new long[]{1}, a), a), "identity left");
        check(Arrays.equals(convolve(new long[]{0, 0, 1}, new long[]{1, 2, 3}), new long[]{0, 0, 1, 2, 3}), "delta shift");
        check(Arrays.equals(convolve(new long[]{1, 0, 0}, new long[]{3, 4}), new long[]{3, 4, 0, 0}), "trailing zeros");

        for (int trial = 0; trial < 120; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 6;
            seed = (seed * 75 + 74) % 65537;
            int m = 1 + seed % 6;
            long[] left = lcgVector(n);
            long[] right = lcgVector(m);
            long[] product = convolve(left, right);
            check(product.length == n + m - 1, "length " + trial);
            check(Arrays.equals(product, convolve(right, left)), "commutativity " + trial);
            for (long x : new long[]{2, 3, 5}) {
                check(evaluate(product, x) == evaluate(left, x) * evaluate(right, x), "evaluation " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
