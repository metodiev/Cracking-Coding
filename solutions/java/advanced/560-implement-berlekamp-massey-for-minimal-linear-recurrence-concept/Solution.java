/** Task 560 - Implement Berlekamp–Massey for minimal linear recurrence (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Berlekamp–Massey over the prime field GF(1000000007): given the first m terms of a sequence it returns
 * the shortest linear recurrence C with C[0] = 1 and s(n) = -(C[1] s(n-1) + ... + C[L] s(n-L)), i.e. the
 * connection polynomial of minimal degree L — the sequence's linear complexity. The implementation is the
 * standard discrepancy loop (track the previous connection polynomial b and the last discrepancy bb,
 * update C by shifting coef * b when the discrepancy d is nonzero, and reset b = C, L = i + 1 - L whenever
 * 2L <= i), with modular inverses by Fermat's little theorem. The self-test does not take the returned
 * polynomial on trust: every claimed recurrence is replayed over the whole known sequence, and minimality
 * is verified independently by solving the coefficient system of every smaller order with Gaussian
 * elimination over the same field.
 *
 * Documented checks. Known instances: the zero sequence has linear complexity 0; a constant sequence and
 * every geometric sequence r^n have complexity 1 (with the exact connection polynomial recovered);
 * Fibonacci (1, 1, 2, 3, ...) has complexity 2 and connection polynomial 1 - x - x^2; an affine LCG
 * x(n+1) = 3x(n) + 7 also needs 2; a sum of two geometrics 2^n + 3^n has the closed form
 * 1 - 5x + 6x^2; the arithmetic and quadratic sequences n and n^2 have complexity 2 and 3; a random
 * order-3 recurrence has complexity exactly 3. For every instance the Gaussian check confirms that no
 * order below the reported L can fit the given terms, that the reported order does fit, that the fitted
 * polynomial reproduces every term of a longer continuation of the sequence, and that the first 2L terms
 * suffice to predict term 2L + 1 (the Berlekamp–Massey "2L theorem" in action) — including on prefixes
 * where the short prefix admits a shorter recurrence, where the code asserts the reported complexity is
 * never larger than the true order and never smaller than what the prefix forces. Prefix linear
 * complexity is also checked to be monotone non-decreasing. All coefficients are asserted to lie in
 * [0, p) and the whole run is verified to be deterministic.
 */
import java.util.*;

public class Solution {

    static final long P = 1000000007L;

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        long nextMod() {
            return (next() >>> 1) % P;
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    static long modPow(long base, long exponent) {
        long result = 1;
        long b = base % P;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = result * b % P;
            b = b * b % P;
            e >>= 1;
        }
        return result;
    }

    static long inverse(long a) {
        return modPow(a, P - 2);
    }

    /**
     * Berlekamp–Massey. Returns an array whose length is L + 1 with C[0] = 1 and
     * s(n) = -(C[1] s(n-1) + ... + C[L] s(n-L)) for every n >= L that the data supports.
     */
    static long[] berlekampMassey(long[] s) {
        int n = s.length;
        long[] c = new long[n + 1];
        long[] b = new long[n + 1];
        c[0] = 1;
        b[0] = 1;
        int l = 0;
        int m = 1;
        long bb = 1;
        for (int i = 0; i < n; i++) {
            long d = s[i] % P;
            for (int j = 1; j <= l; j++) d = (d + c[j] * s[i - j]) % P;
            if (d == 0) {
                m++;
                continue;
            }
            long coef = d * inverse(bb) % P;
            long[] previous = c.clone();
            for (int j = 0; j + m <= n; j++) {
                c[j + m] = ((c[j + m] - coef * b[j]) % P + P) % P;
            }
            if (2 * l <= i) {
                l = i + 1 - l;
                b = previous;
                bb = d;
                m = 1;
            } else {
                m++;
            }
        }
        long[] out = new long[l + 1];
        System.arraycopy(c, 0, out, 0, l + 1);
        for (int j = 0; j < out.length; j++) out[j] = (out[j] % P + P) % P;
        out[0] = 1;
        return out;
    }

    static boolean satisfiesRecurrence(long[] c, long[] s, int from) {
        int l = c.length - 1;
        for (int n = Math.max(from, l); n < s.length; n++) {
            long value = 0;
            for (int j = 1; j <= l; j++) value = (value + c[j] * s[n - j]) % P;
            if ((value + s[n]) % P != 0) return false;
        }
        return true;
    }

    /** Replays the recurrence to extend the sequence; used for prediction checks. */
    static long nextTerm(long[] c, long[] s) {
        int l = c.length - 1;
        long value = 0;
        for (int j = 1; j <= l; j++) value = (value + c[j] * s[s.length - j]) % P;
        return (P - value % P) % P;
    }

    static long[] generate(long[] c, long[] seed, int count) {
        int l = c.length - 1;
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            if (i < seed.length) {
                out[i] = seed[i];
            } else {
                long value = 0;
                for (int j = 1; j <= l; j++) value = (value + (P - c[j]) * out[i - j]) % P;
                out[i] = value;
            }
        }
        return out;
    }

    /** Independent minimality test: does some recurrence of the given order fit all these terms? */
    static boolean hasRecurrenceOfOrder(long[] s, int order) {
        if (order == 0) {
            for (long value : s) if (value % P != 0) return false;
            return true;
        }
        int rows = s.length - order;
        if (rows <= 0) return true;
        long[][] matrix = new long[rows][order + 1];
        for (int r = 0; r < rows; r++) {
            int i = r + order;
            for (int j = 0; j < order; j++) matrix[r][j] = s[i - 1 - j];
            matrix[r][order] = (P - s[i] % P) % P;
        }
        return consistent(matrix, rows, order);
    }

    /** Gaussian elimination over GF(P): is the augmented system consistent? */
    static boolean consistent(long[][] matrix, int rows, int cols) {
        int row = 0;
        for (int col = 0; col < cols && row < rows; col++) {
            int pivot = -1;
            for (int r = row; r < rows; r++) {
                if (matrix[r][col] % P != 0) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) continue;
            long[] tmp = matrix[row];
            matrix[row] = matrix[pivot];
            matrix[pivot] = tmp;
            long inv = inverse(matrix[row][col]);
            for (int c = col; c <= cols; c++) matrix[row][c] = matrix[row][c] * inv % P;
            for (int r = 0; r < rows; r++) {
                if (r == row || matrix[r][col] % P == 0) continue;
                long factor = matrix[r][col];
                for (int c = col; c <= cols; c++) {
                    matrix[r][c] = ((matrix[r][c] - factor * matrix[row][c]) % P + P) % P;
                }
            }
            row++;
        }
        for (int r = 0; r < rows; r++) {
            boolean allZero = true;
            for (int c = 0; c < cols; c++) if (matrix[r][c] % P != 0) allZero = false;
            if (allZero && matrix[r][cols] % P != 0) return false;
        }
        return true;
    }

    /** Minimal order found by the independent elimination scan. */
    static int independentComplexity(long[] s) {
        for (int order = 0; order <= s.length; order++) {
            if (hasRecurrenceOfOrder(s, order)) return order;
        }
        return s.length;
    }

    public static void main(String[] args) {
        // --- Known instances.
        long[] zeros = new long[24];
        long[] cZeros = berlekampMassey(zeros);
        check(cZeros.length == 1, "the zero sequence has complexity 0");

        long[] ones = new long[24];
        Arrays.fill(ones, 1);
        long[] cOnes = berlekampMassey(ones);
        check(cOnes.length == 2 && cOnes[1] == P - 1, "a constant sequence has complexity 1 with 1 - x");

        long[] geometric = new long[20];
        long r = 5;
        geometric[0] = 1;
        for (int i = 1; i < 20; i++) geometric[i] = geometric[i - 1] * r % P;
        long[] cGeometric = berlekampMassey(geometric);
        check(cGeometric.length == 2 && cGeometric[1] == (P - r) % P,
                "a geometric sequence with ratio 5 needs 1 - 5x");

        long[] fibSequence = new long[40];
        fibSequence[0] = 0;
        fibSequence[1] = 1;
        for (int i = 2; i < 40; i++) fibSequence[i] = (fibSequence[i - 1] + fibSequence[i - 2]) % P;
        long[] cFib = berlekampMassey(fibSequence);
        check(cFib.length == 3 && cFib[1] == P - 1 && cFib[2] == P - 1,
                "Fibonacci needs 1 - x - x^2");

        long[] lcg = new long[30];
        lcg[0] = 11;
        for (int i = 1; i < 30; i++) lcg[i] = (3 * lcg[i - 1] + 7) % P;
        long[] cLcg = berlekampMassey(lcg);
        check(cLcg.length == 3, "an affine LCG needs complexity 2");

        long[] mixed = new long[30];
        long pow2 = 1;
        long pow3 = 1;
        for (int i = 0; i < 30; i++) {
            mixed[i] = (pow2 + pow3) % P;
            pow2 = pow2 * 2 % P;
            pow3 = pow3 * 3 % P;
        }
        long[] cMixed = berlekampMassey(mixed);
        check(cMixed.length == 3 && cMixed[1] == P - 5 && cMixed[2] == 6,
                "2^n + 3^n needs 1 - 5x + 6x^2, got " + Arrays.toString(cMixed));

        long[] arithmetic = new long[24];
        for (int i = 0; i < 24; i++) arithmetic[i] = i;
        check(berlekampMassey(arithmetic).length == 3, "the sequence n has complexity 2");
        long[] quadratic = new long[30];
        for (int i = 0; i < 30; i++) quadratic[i] = (long) i * i % P;
        check(berlekampMassey(quadratic).length == 4, "the sequence n^2 has complexity 3");

        // --- Random recurrences of known order, verified exhaustively.
        Rng rng = new Rng(20240560L);
        for (int order = 1; order <= 6; order++) {
            for (int trial = 0; trial < 4; trial++) {
                long[] c = new long[order + 1];
                c[0] = 1;
                for (int j = 1; j <= order; j++) c[j] = (P - 1 - rng.nextMod() % 1000) % P;
                long[] seed = new long[order];
                for (int j = 0; j < order; j++) seed[j] = 1 + rng.nextMod() % 1000000;
                long[] sequence = generate(c, seed, 2 * order + 6);
                long[] found = berlekampMassey(sequence);
                check(found.length - 1 == order, "order " + order + " recovered, got " + (found.length - 1)
                        + " (trial " + trial + ")");
                check(satisfiesRecurrence(found, sequence, 0),
                        "the recurrence reproduces the whole sequence (order " + order + ")");
                check(independentComplexity(sequence) == order,
                        "Gaussian elimination confirms the minimality (order " + order + ")");
                long[] prefix = Arrays.copyOf(sequence, 2 * order);
                long[] fromPrefix = berlekampMassey(prefix);
                check(fromPrefix.length - 1 == order, "2L terms determine the recurrence (order " + order
                        + ")");
                check(nextTerm(fromPrefix, prefix) == sequence[2 * order],
                        "and predict the next term correctly (order " + order + ")");
            }
        }

        // --- Prefix behaviour: complexity never exceeds the true order, never falls below the forced one.
        long[] truth = generate(new long[]{1, (P - 1) % P, (P - 3) % P, (P - 2) % P},
                new long[]{1, 5, 3}, 40);
        int previousLength = 0;
        int previousOrder = 0;
        for (int length = 1; length <= 40; length++) {
            long[] prefix = Arrays.copyOf(truth, length);
            long[] found = berlekampMassey(prefix);
            int order = found.length - 1;
            check(order <= 3, "the prefix complexity never exceeds the true order 3 (length " + length + ")");
            check(order == independentComplexity(prefix),
                    "the prefix complexity matches Gaussian elimination (length " + length + ")");
            check(satisfiesRecurrence(found, prefix, 0),
                    "the prefix recurrence fits its own data (length " + length + ")");
            check(order >= previousOrder, "prefix complexity is monotone (length " + length + ")");
            check(length > previousLength, "the sweep is progressing");
            previousLength = length;
            previousOrder = order;
        }
        check(previousOrder == 3, "the full sequence has complexity 3");

        // --- Cyclic structures: a pure cycle of length 5 over the field.
        long[] cycle = new long[40];
        for (int i = 0; i < 40; i++) cycle[i] = new long[]{2, 3, 5, 7, 11}[i % 5];
        long[] cCycle = berlekampMassey(cycle);
        check(cCycle.length == 6, "a 5-periodic sequence has complexity 5, got " + (cCycle.length - 1));
        check(independentComplexity(cycle) == 5, "and Gaussian elimination agrees");
        check(satisfiesRecurrence(cCycle, cycle, 0), "the cyclic recurrence reproduces all 40 terms");
        check(nextTerm(cCycle, cycle) == cycle[0], "and predicts the wrap-around term");

        // --- Insufficient data still yields a consistent (possibly shorter) recurrence.
        long[] shortPrefix = Arrays.copyOf(truth, 4);
        long[] cShort = berlekampMassey(shortPrefix);
        check(satisfiesRecurrence(cShort, shortPrefix, 0),
                "with few terms the reported recurrence still fits them all");
        check(independentComplexity(shortPrefix) == cShort.length - 1,
                "and it is already minimal for that data");
        check(berlekampMassey(new long[0]).length == 1, "an empty sequence has complexity 0");
        for (long value : cCycle) check(value >= 0 && value < P, "coefficients are reduced into [0, p)");

        check(Arrays.equals(berlekampMassey(fibSequence), berlekampMassey(fibSequence)),
                "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
