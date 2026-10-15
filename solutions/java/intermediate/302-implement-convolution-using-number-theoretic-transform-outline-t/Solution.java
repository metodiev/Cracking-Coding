/** Task 302 - Convolution via number-theoretic transform (outline). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static final long MOD = 998244353L;
    static final long PRIMITIVE_ROOT = 3L;

    static long power(long base, long exponent, long modulus) {
        long result = 1;
        base %= modulus;
        while (exponent > 0) {
            if ((exponent & 1) == 1) result = result * base % modulus;
            base = base * base % modulus;
            exponent >>= 1;
        }
        return result;
    }

    /** In-place iterative NTT; `values` holds coefficients modulo MOD. */
    static void ntt(long[] values, boolean invert) {
        int n = values.length;
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j |= bit;
            if (i < j) { long tmp = values[i]; values[i] = values[j]; values[j] = tmp; }
        }

        for (int length = 2; length <= n; length <<= 1) {
            long root = power(PRIMITIVE_ROOT, (MOD - 1) / length, MOD);
            if (invert) root = power(root, MOD - 2, MOD);
            int half = length >> 1;
            for (int start = 0; start < n; start += length) {
                long w = 1;
                for (int k = 0; k < half; k++) {
                    long u = values[start + k];
                    long v = values[start + k + half] * w % MOD;
                    values[start + k] = (u + v) % MOD;
                    values[start + k + half] = ((u - v) % MOD + MOD) % MOD;
                    w = w * root % MOD;
                }
            }
        }

        if (invert) {
            long inverseN = power(n, MOD - 2, MOD);
            for (int i = 0; i < n; i++) values[i] = values[i] * inverseN % MOD;
        }
    }

    static long[] convolutionNtt(long[] a, long[] b) {
        if (a.length == 0 || b.length == 0) return new long[0];
        int size = 1;
        while (size < a.length + b.length - 1) size <<= 1;
        long[] left = new long[size];
        long[] right = new long[size];
        for (int i = 0; i < a.length; i++) left[i] = a[i] % MOD;
        for (int i = 0; i < b.length; i++) right[i] = b[i] % MOD;
        ntt(left, false);
        ntt(right, false);
        for (int i = 0; i < size; i++) left[i] = left[i] * right[i] % MOD;
        ntt(left, true);
        return Arrays.copyOf(left, a.length + b.length - 1);
    }

    static long[] convolutionNaiveMod(long[] a, long[] b) {
        if (a.length == 0 || b.length == 0) return new long[0];
        long[] out = new long[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b.length; j++)
                out[i + j] = (out[i + j] + a[i] * b[j]) % MOD;
        return out;
    }

    private static int seed = 7788;

    private static long[] lcgVector(int size) {
        long[] vector = new long[size];
        for (int i = 0; i < size; i++) {
            seed = (seed * 75 + 74) % 65537;
            vector[i] = seed % 1000;
        }
        return vector;
    }

    public static void main(String[] args) {
        check(power(PRIMITIVE_ROOT, MOD - 1, MOD) == 1, "Fermat root of unity");
        check((MOD - 1) % (1 << 23) == 0 && (MOD - 1) / (1 << 23) == 119, "modulus shape");

        check(Arrays.equals(convolutionNtt(new long[]{1,2,3}, new long[]{0,1,1}), new long[]{0,1,3,5,3}), "classic");
        check(Arrays.equals(convolutionNtt(new long[]{1,1}, new long[]{1,1}), new long[]{1,2,1}), "square");
        check(Arrays.equals(convolutionNtt(new long[]{7}, new long[]{9}), new long[]{63}), "scalars");
        check(convolutionNtt(new long[]{}, new long[]{1}).length == 0, "empty operand");
        check(Arrays.equals(convolutionNtt(new long[]{MOD - 1}, new long[]{2}), new long[]{MOD - 2}), "negation mod p");

        long[] vector = {3, 1, 4, 1, 5, 9, 2, 6};
        long[] copy = vector.clone();
        ntt(copy, false);
        check(!Arrays.equals(copy, vector), "forward transform changes the vector");
        ntt(copy, true);
        check(Arrays.equals(copy, vector), "inverse restores the vector");

        for (int trial = 0; trial < 40; trial += 2) {
            int sizeA = 1 + (trial + seed) % 7;
            long[] a = lcgVector(sizeA);
            int sizeB = 1 + seed % 7;
            long[] b = lcgVector(sizeB);
            check(Arrays.equals(convolutionNtt(a, b), convolutionNaiveMod(a, b)), "random forward " + trial);
            check(Arrays.equals(convolutionNtt(b, a), convolutionNaiveMod(a, b)), "random swapped " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
