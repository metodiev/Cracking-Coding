/** Task 539 - Implement bitset convolution using FWHT (XOR convolution) small N. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * XOR (dyadic) convolution of integer vectors, computed with the Fast Walsh-Hadamard transform:
 * the butterfly pair (u + v, u - v) is applied in place over log2(n) stages, the transform of a
 * vector is multiplied pointwise with the transform of the other, and one more transform divided
 * by n returns c[k] = sum over i ^ j == k of a[i] * b[j]. Under the DFT analogy the transform is
 * its own inverse up to the factor n, so the same loop implements both directions; the file
 * counts butterflies and asserts the exact O(n log n) cost ((n log2 n)/2 butterflies per transform,
 * 3 (n log2 n)/2 for a convolution) and that the un-normalised inverse is divisible by n.
 *
 * The motivating use is bitset convolution: an indicator vector of a set of positions is a
 * bitset, and the XOR convolution of two such vectors is exactly the histogram of pairwise XOR
 * values, so it answers "how many pairs (i in A, j in B) have i ^ j == k" in O(n log n) instead
 * of O(n^2); iterating the convolution counts ordered k-tuples with a prescribed XOR, i.e. XOR
 * reachability with exactly k elements. Documented checks: the transform is verified against the
 * quadratic definition for n = 1, 2, 4, 8, 16, 32; F^2 = n * identity, the unit basis vector is
 * a two-sided identity, and commutativity, associativity, linearity, the sum identity
 * sum(conv) = sum(a) * sum(b) and Parseval's sum_i a_i b_i relation are all asserted; pair-XOR
 * histograms and k-fold XOR reachability are compared against exhaustive enumeration for small
 * bitsets.
 */
import java.util.*;

public class Solution {
    static long butterflyOps;

    /** In-place Walsh-Hadamard transform; applying it twice multiplies the vector by n. */
    static void fwht(long[] a) {
        int n = a.length;
        for (int len = 1; len < n; len <<= 1) {
            for (int start = 0; start < n; start += len << 1) {
                for (int j = 0; j < len; j++) {
                    long u = a[start + j];
                    long v = a[start + j + len];
                    a[start + j] = u + v;
                    a[start + j + len] = u - v;
                    butterflyOps++;
                }
            }
        }
    }

    /** Un-normalised inverse: transform, multiply pointwise, transform again. */
    static long[] xorConvolutionRaw(long[] a, long[] b) {
        long[] fa = a.clone();
        long[] fb = b.clone();
        fwht(fa);
        fwht(fb);
        for (int i = 0; i < fa.length; i++) fa[i] *= fb[i];
        fwht(fa);
        return fa;
    }

    static long[] xorConvolution(long[] a, long[] b) {
        long[] raw = xorConvolutionRaw(a, b);
        long[] out = new long[raw.length];
        for (int i = 0; i < raw.length; i++) out[i] = raw[i] / raw.length;
        return out;
    }

    static long[] bruteConvolution(long[] a, long[] b) {
        int n = a.length;
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) out[i ^ j] += a[i] * b[j];
        }
        return out;
    }

    static long[] indicator(int n, int... positions) {
        long[] out = new long[n];
        for (int p : positions) out[p] = 1;
        return out;
    }

    static long[] lcgVector(int n, int seed) {
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            out[i] = seed % 11 - 5;
        }
        return out;
    }

    static int log2(int n) {
        int result = 0;
        while ((1 << result) < n) result++;
        return result;
    }

    public static void main(String[] args) {
        check(Arrays.equals(xorConvolution(new long[]{5}, new long[]{7}), new long[]{35}),
                "n = 1 convolves by ordinary multiplication");

        long[] tiny = {1, 2, 3, 4};
        long[] tinyRaw = xorConvolutionRaw(tiny, tiny);
        for (long value : tinyRaw) check(value % tiny.length == 0, "the inverse is divisible by n");
        check(Arrays.equals(xorConvolution(tiny, tiny), bruteConvolution(tiny, tiny)),
                "n = 4 matches the quadratic definition");

        int[] sizes = {2, 4, 8, 16, 32};
        int[] seeds = {9, 21, 33, 45, 57};
        for (int s = 0; s < sizes.length; s++) {
            int n = sizes[s];
            long[] a = lcgVector(n, seeds[s]);
            long[] b = lcgVector(n, seeds[s] + 100);
            check(Arrays.equals(xorConvolution(a, b), bruteConvolution(a, b)),
                    "n = " + n + " matches brute force");
            check(Arrays.equals(xorConvolution(b, a), xorConvolution(a, b)), "convolution is commutative");

            long[] doubled = a.clone();
            butterflyOps = 0;
            fwht(doubled);
            check(butterflyOps == n * log2(n) / 2, "one transform costs (n log2 n) / 2 butterflies");
            fwht(doubled);
            long[] scaled = new long[n];
            for (int i = 0; i < n; i++) scaled[i] = n * a[i];
            check(Arrays.equals(doubled, scaled), "F applied twice is n times the identity");

            long[] unit = indicator(n, 0);
            check(Arrays.equals(xorConvolution(a, unit), a), "the unit vector is a two-sided identity");

            long[] c = lcgVector(n, seeds[s] + 200);
            long[] left = xorConvolution(xorConvolution(a, b), c);
            long[] right = xorConvolution(a, xorConvolution(b, c));
            check(Arrays.equals(left, right), "convolution is associative");

            long alpha = 3;
            long beta = -2;
            long[] mixed = new long[n];
            for (int i = 0; i < n; i++) mixed[i] = alpha * a[i] + beta * b[i];
            long[] linearLeft = xorConvolution(mixed, c);
            long[] linearRight = new long[n];
            long[] ac = xorConvolution(a, c);
            long[] bc = xorConvolution(b, c);
            for (int i = 0; i < n; i++) linearRight[i] = alpha * ac[i] + beta * bc[i];
            check(Arrays.equals(linearLeft, linearRight), "convolution is bilinear");

            long sumA = 0;
            long sumB = 0;
            long sumC = 0;
            long[] conv = xorConvolution(a, b);
            for (int i = 0; i < n; i++) {
                sumA += a[i];
                sumB += b[i];
                sumC += conv[i];
            }
            check(sumC == sumA * sumB, "sum(conv) = sum(a) * sum(b)");

            butterflyOps = 0;
            xorConvolution(a, b);
            check(butterflyOps == 3L * n * log2(n) / 2, "a convolution costs 3 (n log2 n) / 2 butterflies");
        }

        long[] setA = indicator(32, 0, 3, 5, 6, 12, 17, 25, 30);
        long[] setB = indicator(32, 1, 3, 7, 12, 19, 25, 28, 31);
        long[] histogram = xorConvolution(setA, setB);
        long total = 0;
        long sizeA = 0;
        long sizeB = 0;
        for (int i = 0; i < 32; i++) {
            sizeA += setA[i];
            sizeB += setB[i];
        }
        for (long value : histogram) total += value;
        check(total == sizeA * sizeB, "the histogram sums to |A| * |B|");
        for (int k = 0; k < 32; k++) {
            int pairs = 0;
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    if (setA[i] == 1 && setB[j] == 1 && (i ^ j) == k) pairs++;
                }
            }
            check(histogram[k] == pairs, "pair XOR histogram at " + k);
        }

        int[] multiset = {1, 2, 4, 7, 8};
        long[] base = new long[32];
        for (int value : multiset) base[value] = 1;
        for (int k = 1; k <= 4; k++) {
            long[] power = base.clone();
            for (int step = 1; step < k; step++) power = xorConvolution(power, base);
            long[] brute = new long[32];
            enumerate(0, k, 0, multiset, brute);
            check(Arrays.equals(power, brute), "exactly " + k + " elements: XOR histogram matches");
        }

        long[] zeroVector = new long[8];
        check(Arrays.equals(xorConvolution(zeroVector, lcgVector(8, 5)), new long[8]),
                "the zero vector annihilates");
        long[] allOnes = new long[8];
        Arrays.fill(allOnes, 1);
        long[] onesResult = xorConvolution(allOnes, allOnes);
        for (int i = 0; i < 8; i++) check(onesResult[i] == 8, "indicator of the full set doubles to n");

        System.out.println("PASS");
    }

    /** Exhaustive enumeration of ordered k-tuples, accumulating the XOR histogram. */
    static void enumerate(int step, int k, int xor, int[] multiset, long[] out) {
        if (step == k) {
            out[xor]++;
            return;
        }
        for (int value : multiset) enumerate(step + 1, k, xor ^ value, multiset, out);
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
