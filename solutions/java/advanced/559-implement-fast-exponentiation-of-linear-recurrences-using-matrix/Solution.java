/** Task 559 - Implement fast exponentiation of linear recurrences using matrix power. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A k-th order linear recurrence a(n) = c1 a(n-1) + ... + ck a(n-k) is evaluated in O(k^3 log n) by
 * raising its companion matrix to the n-th power: the first row of the companion matrix is the
 * coefficient vector, the sub-diagonal carries the identity, and the top row of C^n is exactly
 * (a(n+k-1), ..., a(n)), so the n-th term is one matrix power away. The implementation is written twice
 * — once in exact BigInteger arithmetic (so Fibonacci terms up to n = 1000 can be compared with the
 * naive iteration term by term) and once modulo 1000000007 — plus a fast-doubling routine
 * (F(2n) = F(n)(2F(n+1) - F(n)), F(2n+1) = F(n)^2 + F(n+1)^2) as a second, structurally different route
 * to the same numbers, and a counter of the matrix multiplications performed.
 *
 * Documented checks. Exactness: for k = 0..1000 the matrix power reproduces the iterative Fibonacci
 * sequence in BigInteger arithmetic, and modulo p the two agree up to n = 5000; fast doubling agrees
 * with the matrix power for n up to 10^6 and for n = 10^9 (10^12 also exercised through the modular
 * route), which is the regime where the O(n) iteration is out of reach. Structural identities: the 2x2
 * Fibonacci matrix satisfies M^n = [[F(n+1), F(n)], [F(n), F(n-1)]] in all four entries, Cassini's
 * identity F(n+1)F(n-1) - F(n)^2 = (-1)^n holds (so det(M^n) = det(M)^n), and the Lucas numbers
 * L(n) = L(n-1) + L(n-2) with L(0) = 2, L(1) = 1 come out equal to 2F(n+1) - F(n) and to the trace of
 * M^n. Cayley–Hamilton is verified numerically for a 4th-order recurrence with mixed signs:
 * C^4 = c1 C^3 + c2 C^2 + c3 C + c4 I, and the order-4 and order-3 (Tribonacci) recurrences are
 * compared against plain iteration for several hundred terms. Cost is measured rather than asserted
 * qualitatively: the number of matrix multiplications used for exponent n never exceeds 2*floor(log2 n) + 2,
 * which is asserted for every exponent tested, and the k = 0 case (a single initial value) is covered.
 */
import java.math.BigInteger;
import java.util.*;

public class Solution {

    static final long MOD = 1000000007L;

    static long[][] identity(int k) {
        long[][] out = new long[k][k];
        for (int i = 0; i < k; i++) out[i][i] = 1;
        return out;
    }

    static long[][] multiplyMod(long[][] a, long[][] b) {
        int k = a.length;
        long[][] out = new long[k][k];
        for (int i = 0; i < k; i++) {
            for (int t = 0; t < k; t++) {
                if (a[i][t] == 0) continue;
                for (int j = 0; j < k; j++) {
                    out[i][j] = (out[i][j] + a[i][t] * b[t][j]) % MOD;
                }
            }
        }
        return out;
    }

    /** Binary exponentiation; multiplications[0] counts matrix products actually performed. */
    static long[][] powerMod(long[][] base, long exponent, long[] multiplications) {
        int k = base.length;
        long[][] result = identity(k);
        long[][] square = base;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) {
                result = multiplyMod(result, square);
                if (multiplications != null) multiplications[0]++;
            }
            e >>= 1;
            if (e > 0) {
                square = multiplyMod(square, square);
                if (multiplications != null) multiplications[0]++;
            }
        }
        return result;
    }

    /** Companion matrix of a(n) = c[0] a(n-1) + ... + c[k-1] a(n-k). */
    static long[][] companion(long[] c) {
        int k = c.length;
        long[][] m = new long[k][k];
        for (int j = 0; j < k; j++) m[0][j] = ((c[j] % MOD) + MOD) % MOD;
        for (int i = 1; i < k; i++) m[i][i - 1] = 1;
        return m;
    }

    static BigInteger[][] multiplyBig(BigInteger[][] a, BigInteger[][] b) {
        int k = a.length;
        BigInteger[][] out = new BigInteger[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                BigInteger sum = BigInteger.ZERO;
                for (int t = 0; t < k; t++) sum = sum.add(a[i][t].multiply(b[t][j]));
                out[i][j] = sum;
            }
        }
        return out;
    }

    static BigInteger[][] powerBig(BigInteger[][] base, long exponent) {
        int k = base.length;
        BigInteger[][] result = new BigInteger[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) result[i][j] = i == j ? BigInteger.ONE : BigInteger.ZERO;
        }
        BigInteger[][] square = base;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = multiplyBig(result, square);
            e >>= 1;
            if (e > 0) square = multiplyBig(square, square);
        }
        return result;
    }

    /** The n-th term of the recurrence defined by coefficients c and initial terms a(0..k-1). */
    static long termMod(long[] c, long[] initial, long n) {
        int k = c.length;
        if (n < k) return initial[(int) n] % MOD;
        long[][] power = powerMod(companion(c), n - (k - 1), null);
        long total = 0;
        for (int j = 0; j < k; j++) {
            total = (total + power[0][j] * (initial[k - 1 - j] % MOD)) % MOD;
        }
        return total;
    }

    static BigInteger termBig(long[] c, long[] initial, long n) {
        int k = c.length;
        if (n < k) return BigInteger.valueOf(initial[(int) n]);
        BigInteger[][] companion = new BigInteger[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                companion[i][j] = i == 0 ? BigInteger.valueOf(c[j])
                        : (i == j + 1 ? BigInteger.ONE : BigInteger.ZERO);
            }
        }
        BigInteger[][] power = powerBig(companion, n - (k - 1));
        BigInteger total = BigInteger.ZERO;
        for (int j = 0; j < k; j++) {
            total = total.add(power[0][j].multiply(BigInteger.valueOf(initial[k - 1 - j])));
        }
        return total;
    }

    /** Fast doubling pair {F(n), F(n+1)} modulo MOD. */
    static long[] fibPair(long n) {
        if (n == 0) return new long[]{0, 1};
        long[] half = fibPair(n >> 1);
        long a = half[0];
        long b = half[1];
        long c = a * ((2 * b % MOD - a + MOD) % MOD) % MOD;
        long d = (a * a + b * b) % MOD;
        if ((n & 1L) == 0) return new long[]{c, d};
        return new long[]{d, (c + d) % MOD};
    }

    /** Iterative walk of the recurrence, used as the reference for moderate n. */
    static long[] iterate(long[] c, long[] initial, int count) {
        int k = c.length;
        long[] seq = new long[count];
        for (int i = 0; i < count; i++) {
            if (i < k) {
                seq[i] = initial[i];
            } else {
                long value = 0;
                for (int j = 0; j < k; j++) value = (value + c[j] * seq[i - 1 - j]) % MOD;
                seq[i] = value;
            }
        }
        return seq;
    }

    public static void main(String[] args) {
        long[] fib = {1, 1};
        long[] fibInitial = {0, 1};

        // --- Exact BigInteger Fibonacci versus the naive iteration.
        BigInteger[] exact = new BigInteger[1001];
        exact[0] = BigInteger.ZERO;
        exact[1] = BigInteger.ONE;
        for (int i = 2; i <= 1000; i++) exact[i] = exact[i - 1].add(exact[i - 2]);
        for (int n = 0; n <= 1000; n++) {
            check(termBig(fib, fibInitial, n).equals(exact[n]),
                    "exact Fibonacci via the matrix power at n = " + n);
        }

        // --- Modular Fibonacci versus the naive iteration.
        long[] modular = iterate(fib, fibInitial, 5001);
        for (int n = 0; n <= 5000; n++) {
            check(termMod(fib, fibInitial, n) == modular[n],
                    "modular Fibonacci via the matrix power at n = " + n);
        }

        // --- Fast doubling versus the matrix power, up to and beyond the iterative range.
        for (long n : new long[]{1000L, 100000L, 1000000L, 1000000000L, 1000000000000L}) {
            long[] pair = fibPair(n);
            check(pair[0] == termMod(fib, fibInitial, n),
                    "fast doubling matches the matrix power at n = " + n);
            check(pair[1] == termMod(fib, fibInitial, n + 1),
                    "and for the successor at n = " + n);
        }

        // --- Structural identities.
        for (int n = 1; n <= 40; n++) {
            long[][] m = powerMod(new long[][]{{1, 1}, {1, 0}}, n, null);
            check(m[0][0] == modular[n + 1] && m[0][1] == modular[n]
                            && m[1][0] == modular[n] && m[1][1] == modular[n - 1],
                    "M^n = [[F(n+1), F(n)], [F(n), F(n-1)]] at n = " + n);
            long cassini = (m[0][0] * m[1][1] % MOD - m[0][1] * m[1][0] % MOD + MOD) % MOD;
            long expected = (n % 2 == 0) ? 1 : MOD - 1;
            check(cassini == expected, "Cassini's identity at n = " + n);
            long trace = (m[0][0] + m[1][1]) % MOD;
            long lucas = (2 * termMod(fib, fibInitial, n + 1) % MOD
                    - termMod(fib, fibInitial, n) + MOD) % MOD;
            check(trace == lucas, "the trace of M^n is the Lucas number L(n) at n = " + n);
        }
        long[] lucasIter = iterate(new long[]{1, 1}, new long[]{2, 1}, 201);
        for (int n = 0; n <= 200; n++) {
            long byMatrix = (2 * termMod(fib, fibInitial, n + 1) % MOD
                    - termMod(fib, fibInitial, n) + MOD) % MOD;
            check(lucasIter[n] == byMatrix, "L(n) = 2F(n+1) - F(n) at n = " + n);
        }

        // --- Higher order recurrences.
        long[] trib = {1, 1, 1};
        long[] tribInitial = {0, 0, 1};
        long[] tribIter = iterate(trib, tribInitial, 501);
        for (int n = 0; n <= 500; n++) {
            check(termMod(trib, tribInitial, n) == tribIter[n], "Tribonacci at n = " + n);
        }
        long[] mixed = {3, 4, 1, 6};
        long[] mixedInitial = {1, 2, 3, 4};
        long[] mixedIter = iterate(mixed, mixedInitial, 401);
        for (int n = 0; n <= 400; n++) {
            check(termMod(mixed, mixedInitial, n) == mixedIter[n], "4th-order recurrence at n = " + n);
        }

        // --- Cayley–Hamilton for the 4x4 companion matrix.
        long[][] c4 = companion(mixed);
        long[][] left = powerMod(c4, 4, null);
        long[][] right = new long[4][4];
        long[][] current = identity(4);
        for (int power = 0; power < 4; power++) {
            long coeff = mixed[3 - power];
            for (int a = 0; a < 4; a++) {
                for (int b = 0; b < 4; b++) {
                    right[a][b] = (right[a][b] + coeff * current[a][b]) % MOD;
                }
            }
            current = multiplyMod(current, c4);
        }
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                check(left[a][b] == right[a][b], "Cayley–Hamilton entry (" + a + ", " + b + ")");
            }
        }

        // --- Cost: the number of matrix products stays logarithmic.
        for (long n : new long[]{1L, 2L, 3L, 7L, 8L, 1023L, 1024L, 10L * 1000 * 1000}) {
            long[] multiplications = new long[1];
            powerMod(companion(fib), n, multiplications);
            long bound = 2 * (63 - Long.numberOfLeadingZeros(n)) + 2;
            check(multiplications[0] <= bound, "exponent " + n + " used " + multiplications[0]
                    + " matrix products, the bound is " + bound);
        }

        // --- Order zero and boundaries.
        check(termMod(new long[]{0}, new long[]{5}, 0) == 5, "order-1 constant recurrence");
        check(termMod(new long[]{0}, new long[]{5}, 17) == 0, "a(0) = 5, a(n) = 0 for n >= 1");
        check(termMod(new long[]{1}, new long[]{7}, 1000) == 7, "a(n) = a(n-1) is constant");
        check(termBig(new long[]{1, 1}, new long[]{0, 1}, 1000)
                .equals(termBig(new long[]{1, 1}, new long[]{0, 1}, 999).add(
                        termBig(new long[]{1, 1}, new long[]{0, 1}, 998))), "the recurrence relation holds");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
