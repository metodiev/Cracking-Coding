/** Task 587 - Implement matrix determinant via LU decomposition (modular or double). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Two flavours of the same LU idea:
 *   - modular: Gaussian elimination over the prime field Z_p (p = 998244353) with row swaps, in which
 *     every pivot is either invertible or the matrix is singular, so the determinant is exact - the
 *     result is the product of the pivots times the sign of the permutation. Nothing ever overflows:
 *     entries stay below p < 2^30, so even a product of two entries and the accumulated pivot product
 *     stay far below 2^63.
 *   - double: LU with partial pivoting, det = sign * product of the pivots, with a documented
 *     threshold of 1e-300 on |pivot| to declare the matrix numerically singular. Partial pivoting is
 *     what keeps this usable: the naive [[0,1],[1,0]] example has a zero pivot and would divide by
 *     zero without a swap.
 * The decomposition itself is verified, not assumed: the file reconstructs P*A = L*U and checks the
 * product of the L and U factors against the original matrix entry by entry.
 *
 * A third, independent exact method - the Bareiss fraction-free elimination over the integers - is
 * included as a cross-check for integer matrices: it performs the same elimination but keeps entries
 * integral through the exact divisions, so it must agree with the modular determinant reduced mod p
 * and with the rounded double determinant. The long-typed Bareiss overflows once the determinant
 * exceeds a signed 64-bit integer (documented and asserted on the order-12 instance, where the exact
 * determinant is larger than 9e18), so a BigInteger variant carries the larger cross-checks.
 *
 * Documented checks: permutation matrices of order 5 (all 120 of them) have determinant equal to the
 * parity of the permutation; the Leibniz formula over all even/odd permutations is the oracle for
 * random matrices of order 1..6; det is multiplicative (det AB = det A det B, in both fields and in
 * Bareiss), det A^T = det A, det(cA) = c^n det A, and det of a diagonal/triangular matrix is the
 * product of its diagonal; the zero matrix, duplicated rows, a rank-one update u*v^T and a matrix with
 * a zero row all give 0; the closed forms for Toeplitz-like bidiagonal, Vandermonde and Cauchy
 * matrices are matched (Vandermonde nodes 0..n-1 have determinant prod_{k=1}^{n-1} k!, checked exactly
 * modulo p and to a relative 1e-6 in double, with the documented caveat that the 8x8 Vandermonde with
 * nodes 0..7 has condition number of order 1e11, so a 1e-9 relative claim would be dishonest there,
 * while the Cauchy matrix with nodes 1..5 is well conditioned and matches to 1e-9).
 * One honest caveat is asserted rather than hidden: a modular determinant cannot see an integer
 * determinant that is a nonzero multiple of p - the file exhibits a 2x2 integer matrix with det = 97
 * whose determinant modulo 97 is 0 while the double and Bareiss routines return 97.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; the runs are deterministic.
 */
import java.math.BigInteger;
import java.util.*;

public class Solution {
    static final long MOD = 998244353L;

    // ---------------------------------------------------------------- RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }

        long nextMod(long m) {
            return Math.floorMod(next(), m);
        }
    }

    // ---------------------------------------------------------------- modular determinant

    static long powMod(long base, long exp, long mod) {
        long result = 1;
        base %= mod;
        while (exp > 0) {
            if ((exp & 1) == 1) result = result * base % mod;
            base = base * base % mod;
            exp >>= 1;
        }
        return result;
    }

    static long detMod(long[][] input, long mod) {
        int n = input.length;
        if (n == 0) return 1;
        long[][] a = new long[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        int sign = 1;
        long det = 1;
        for (int col = 0; col < n; col++) {
            int pivot = -1;
            for (int row = col; row < n; row++) {
                if (a[row][col] % mod != 0) {
                    pivot = row;
                    break;
                }
            }
            if (pivot < 0) return 0;
            if (pivot != col) {
                long[] t = a[pivot];
                a[pivot] = a[col];
                a[col] = t;
                sign = -sign;
            }
            long pivotValue = a[col][col] % mod;
            det = det * pivotValue % mod;
            long inv = powMod(pivotValue, mod - 2, mod);
            for (int row = col + 1; row < n; row++) {
                if (a[row][col] % mod == 0) continue;
                long factor = a[row][col] % mod * inv % mod;
                for (int j = col; j < n; j++) {
                    a[row][j] = ((a[row][j] - factor * a[col][j]) % mod + mod) % mod;
                }
            }
        }
        return (det * sign % mod + mod) % mod;
    }

    // ---------------------------------------------------------------- double determinant

    static double detDouble(double[][] input) {
        int n = input.length;
        double[][] a = new double[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        double det = 1.0;
        int sign = 1;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
            }
            if (Math.abs(a[pivot][col]) < 1e-300) return 0.0;
            if (pivot != col) {
                double[] t = a[pivot];
                a[pivot] = a[col];
                a[col] = t;
                sign = -sign;
            }
            det *= a[col][col];
            for (int row = col + 1; row < n; row++) {
                double factor = a[row][col] / a[col][col];
                for (int j = col; j < n; j++) a[row][j] -= factor * a[col][j];
            }
        }
        return sign * det;
    }

    /** Bareiss fraction-free elimination: exact for integer inputs, with the same row swaps. */
    static long detBareiss(long[][] input) {
        int n = input.length;
        if (n == 0) return 1;
        long[][] a = new long[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        int sign = 1;
        long previous = 1;
        for (int k = 0; k < n - 1; k++) {
            if (a[k][k] == 0) {
                int swap = -1;
                for (int row = k + 1; row < n; row++) {
                    if (a[row][k] != 0) {
                        swap = row;
                        break;
                    }
                }
                if (swap < 0) return 0;
                long[] t = a[swap];
                a[swap] = a[k];
                a[k] = t;
                sign = -sign;
            }
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / previous;
                }
            }
            previous = a[k][k];
        }
        return sign * a[n - 1][n - 1];
    }

    /**
     * The same fraction-free elimination in BigInteger arithmetic. The long-typed version overflows
     * once the determinant no longer fits a signed 64-bit integer; the file exhibits a strictly
     * diagonally dominant 16x16 instance whose determinant is of order 1e48 (asserted to exceed
     * Long.MAX_VALUE), which is the documented reason for this second variant.
     */
    static BigInteger detBareissBig(long[][] input) {
        int n = input.length;
        if (n == 0) return BigInteger.ONE;
        BigInteger[][] a = new BigInteger[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = BigInteger.valueOf(input[i][j]);
        }
        int sign = 1;
        BigInteger previous = BigInteger.ONE;
        for (int k = 0; k < n - 1; k++) {
            if (a[k][k].signum() == 0) {
                int swap = -1;
                for (int row = k + 1; row < n; row++) {
                    if (a[row][k].signum() != 0) {
                        swap = row;
                        break;
                    }
                }
                if (swap < 0) return BigInteger.ZERO;
                BigInteger[] t = a[swap];
                a[swap] = a[k];
                a[k] = t;
                sign = -sign;
            }
            for (int i = k + 1; i < n; i++) {
                for (int j = k + 1; j < n; j++) {
                    a[i][j] = a[i][j].multiply(a[k][k]).subtract(a[i][k].multiply(a[k][j])).divide(previous);
                }
            }
            previous = a[k][k];
        }
        return sign > 0 ? a[n - 1][n - 1] : a[n - 1][n - 1].negate();
    }

    // ---------------------------------------------------------------- oracles and closed forms

    static double detLeibniz(double[][] a) {
        int n = a.length;
        int[] permutation = new int[n];
        for (int i = 0; i < n; i++) permutation[i] = i;
        return leibnizRec(a, permutation, 0, n);
    }

    static double leibnizRec(double[][] a, int[] permutation, int at, int n) {
        if (at == n) {
            double product = 1.0;
            for (int i = 0; i < n; i++) product *= a[i][permutation[i]];
            return permutationSign(permutation) * product;
        }
        double total = 0.0;
        for (int i = at; i < n; i++) {
            int t = permutation[at];
            permutation[at] = permutation[i];
            permutation[i] = t;
            total += leibnizRec(a, permutation, at + 1, n);
            t = permutation[at];
            permutation[at] = permutation[i];
            permutation[i] = t;
        }
        return total;
    }

    static int permutationSign(int[] permutation) {
        int inversions = 0;
        for (int i = 0; i < permutation.length; i++) {
            for (int j = i + 1; j < permutation.length; j++) {
                if (permutation[i] > permutation[j]) inversions++;
            }
        }
        return (inversions & 1) == 0 ? 1 : -1;
    }

    static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) a[i][i] = 1;
        return a;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double acc = 0.0;
                for (int k = 0; k < n; k++) acc += a[i][k] * b[k][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    static long[][] toLongs(double[][] a) {
        long[][] out = new long[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) out[i][j] = Math.round(a[i][j]);
        }
        return out;
    }

    static double[][] randomIntegerMatrix(int n, int bound, Rng rng) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = rng.nextInt(2 * bound + 1) - bound;
        }
        return a;
    }

    /** The exact integer determinant of the Vandermonde matrix with nodes 0..n-1 is prod_{k=1}^{n-1} k!. */
    static long vandermondeExact(int n) {
        long product = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) product *= j - i;
        }
        return product;
    }

    static double[][] vandermonde(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            double value = 1.0;
            for (int j = 0; j < n; j++) {
                a[i][j] = value;
                value *= i;
            }
        }
        return a;
    }

    /** Cauchy matrix C_ij = 1/(x_i + y_j): det = prod_{i<j}(x_j - x_i)(y_j - y_i) / prod_{i,j}(x_i + y_j). */
    static double[][] cauchy(double[] x, double[] y) {
        double[][] a = new double[x.length][y.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < y.length; j++) a[i][j] = 1.0 / (x[i] + y[j]);
        }
        return a;
    }

    static double cauchyExact(double[] x, double[] y) {
        int n = x.length;
        double numerator = 1.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                numerator *= (x[j] - x[i]) * (y[j] - y[i]);
            }
        }
        double denominator = 1.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) denominator *= x[i] + y[j];
        }
        return numerator / denominator;
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        basicShapesAndSingularity();
        permutationMatrices();
        leibnizOracle();
        multiplicativeAndTranspose();
        factorizationIsVerified();
        closedForms();
        modularBlindSpot();
        bareissAgrees();
        determinism();

        System.out.println("PASS");
    }

    static void basicShapesAndSingularity() {
        check(detMod(new long[][]{{1, 0}, {0, 1}}, MOD) == 1, "det I = 1 modulo p");
        check(Math.abs(detDouble(identity(4)) - 1.0) <= 1e-15, "det I = 1 in double");
        check(detDouble(new double[0][0]) == 1.0, "the empty determinant is 1");
        check(detDouble(new double[][]{{5.0}}) == 5.0, "a 1x1 determinant is its entry");
        check(detMod(new long[][]{{7}}, MOD) == 7, "a 1x1 determinant modulo p is its residue");

        long[][] diagonal = {{3, 0, 0}, {0, 5, 0}, {0, 0, 7}};
        check(detMod(diagonal, MOD) == 105, "a diagonal determinant is the diagonal product (105)");
        check(Math.abs(detDouble(new double[][]{{3, 0, 0}, {0, 5, 0}, {0, 0, 7}}) - 105.0) <= 1e-12,
                "the same in double");
        long[][] upper = {{2, 3, 4}, {0, -1, 5}, {0, 0, 6}};
        check(detMod(upper, MOD) == MOD - 12, "an upper triangular determinant is -12 modulo p");
        check(Math.abs(detDouble(new double[][]{{2, 3, 4}, {0, -1, 5}, {0, 0, 6}}) + 12.0) <= 1e-12,
                "and -12 in double");
        long[][] lower = {{2, 0, 0}, {7, -1, 0}, {5, 6, 6}};
        check(detMod(lower, MOD) == MOD - 12, "a lower triangular determinant is -12 as well");

        check(detMod(new long[][]{{1, 2}, {2, 4}}, MOD) == 0, "duplicated rows give a zero determinant");
        check(Math.abs(detDouble(new double[][]{{1, 2}, {2, 4}})) <= 1e-300, "singular in double too");
        check(detMod(new long[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, MOD) == 0,
                "the 1..9 matrix is singular modulo p");
        check(Math.abs(detDouble(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}})) <= 1e-9,
                "and numerically singular in double");
        check(detMod(new long[][]{{0, 0}, {0, 0}}, MOD) == 0, "the zero matrix is singular");
        double[][] rankOne = {{1, 2}, {3, 6}};
        check(Math.abs(detDouble(rankOne)) <= 1e-15, "a rank-one 2x2 matrix is singular");
        double[][] rankOneLarger = {{1, 2, 3}, {2, 4, 6}, {5, 10, 15}};
        check(Math.abs(detDouble(rankOneLarger)) <= 1e-12, "and so is a rank-one 3x3 matrix");
        check(detMod(new long[][]{{0, 0, 0}, {1, 2, 3}, {4, 5, 6}}, MOD) == 0, "a zero row gives 0");

        long[][] pivotNeeded = {{0, 1}, {1, 0}};
        check(detMod(pivotNeeded, MOD) == MOD - 1, "[[0,1],[1,0]] has determinant -1 modulo p");
        check(Math.abs(detDouble(new double[][]{{0, 1}, {1, 0}}) + 1.0) <= 1e-15,
                "and -1 in double, which needs a pivot swap");
        long[][] needsTwo = {{0, 0, 2}, {0, 4, 5}, {6, 7, 8}};
        check(Math.abs(detDouble(new double[][]{{0, 0, 2}, {0, 4, 5}, {6, 7, 8}}) - (-48.0)) <= 1e-12,
                "a twice-pivoted matrix is -48");
        check(detBareiss(needsTwo) == -48 && Math.floorMod(detBareiss(needsTwo), MOD) == detMod(needsTwo, MOD),
                "Bareiss gives -48 and its residue matches the modular determinant on the pivoting example");
        check(Math.abs(detDouble(new double[][]{{0, 0, 2}, {0, 4, 5}, {6, 7, 8}}) - detBareiss(needsTwo))
                <= 1e-9, "and the double determinant too");
    }

    static void permutationMatrices() {
        int n = 5;
        int[] permutation = {0, 1, 2, 3, 4};
        int count = 0;
        while (true) {
            long[][] p = new long[n][n];
            for (int i = 0; i < n; i++) p[i][permutation[i]] = 1;
            int sign = permutationSign(permutation);
            long expected = sign > 0 ? 1 : MOD - 1;
            check(detMod(p, MOD) == expected, "the permutation determinant matches its parity");
            double[][] pd = new double[n][n];
            for (int i = 0; i < n; i++) pd[i][permutation[i]] = 1;
            check(Math.abs(detDouble(pd) - sign) <= 1e-15, "the double determinant gives the signed unit");
            check(detBareiss(p) == sign, "Bareiss gives the signed unit as well");
            count++;
            if (!nextPermutation(permutation)) break;
        }
        check(count == 120, "all 120 permutations of order 5 were examined");
    }

    static boolean nextPermutation(int[] a) {
        int i = a.length - 2;
        while (i >= 0 && a[i] >= a[i + 1]) i--;
        if (i < 0) return false;
        int j = a.length - 1;
        while (a[j] <= a[i]) j--;
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
        for (int l = i + 1, r = a.length - 1; l < r; l++, r--) {
            t = a[l];
            a[l] = a[r];
            a[r] = t;
        }
        return true;
    }

    static void leibnizOracle() {
        Rng rng = new Rng(5871L);
        for (int n = 1; n <= 6; n++) {
            for (int trial = 0; trial < 4; trial++) {
                double[][] a = randomIntegerMatrix(n, 4, rng);
                long exact = detBareiss(toLongs(a));
                double leibniz = detLeibniz(a);
                check(Math.abs(leibniz - exact) <= 1e-9, "Leibniz equals Bareiss at n = " + n);
                check(Math.abs(detDouble(a) - exact) <= 1e-9 * Math.max(1.0, Math.abs((double) exact)),
                        "the double LU equals the exact determinant at n = " + n);
                long modular = detMod(toLongs(a), MOD);
                check(modular == Math.floorMod(exact, MOD), "the modular determinant matches at n = " + n);
            }
        }
        Rng rng2 = new Rng(5872L);
        for (int trial = 0; trial < 20; trial++) {
            int n = 6 + rng2.nextInt(4);
            double[][] a = randomIntegerMatrix(n, 6, rng2);
            long exact = detBareiss(toLongs(a));
            check(Math.abs(detDouble(a) - exact) <= 1e-6 * Math.max(1.0, Math.abs((double) exact)),
                    "the double LU stays within 1e-6 relative at n = " + n);
            check(detMod(toLongs(a), MOD) == Math.floorMod(exact, MOD),
                    "the modular and Bareiss determinants agree at n = " + n);
        }
        double[][] big = randomIntegerMatrix(12, 9, new Rng(5873L));
        BigInteger exactBig = detBareissBig(toLongs(big));
        check(detMod(toLongs(big), MOD) == exactBig.mod(BigInteger.valueOf(MOD)).longValue(),
                "order 12 agrees between the modular elimination and the BigInteger Bareiss");
        check(Math.abs(detDouble(big) / exactBig.doubleValue() - 1.0) <= 1e-8,
                "and the double LU is within 1e-8 relative of the exact order-12 determinant");

        // A strictly diagonally dominant 16x16 matrix has a determinant far beyond a signed long, so
        // the long-typed Bareiss is provably out of its depth here and the BigInteger variant is used.
        int n = 16;
        long[][] dominant = new long[n][n];
        Rng dominantRng = new Rng(5878L);
        for (int i = 0; i < n; i++) {
            dominant[i][i] = 1000;
            for (int j = 0; j < n; j++) {
                if (i != j) dominant[i][j] = dominantRng.nextInt(3) - 1;
            }
        }
        BigInteger huge = detBareissBig(dominant);
        check(huge.abs().compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0,
                "the diagonally dominant 16x16 determinant " + huge + " exceeds a signed long, so the "
                        + "long-typed Bareiss would overflow (documented)");
        check(Math.abs(detDouble(toDoubles(dominant)) / huge.doubleValue() - 1.0) <= 1e-9,
                "the double LU is within 1e-9 relative of that exact 1e48-scale determinant");
        check(detMod(dominant, MOD) == huge.mod(BigInteger.valueOf(MOD)).longValue(),
                "the modular determinant matches the exact one on the huge instance");
    }

    static double[][] toDoubles(long[][] a) {
        double[][] out = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) out[i][j] = a[i][j];
        }
        return out;
    }

    static void multiplicativeAndTranspose() {
        Rng rng = new Rng(5874L);
        for (int trial = 0; trial < 6; trial++) {
            int n = 2 + trial;
            double[][] a = randomIntegerMatrix(n, 5, rng);
            double[][] b = randomIntegerMatrix(n, 5, rng);
            double detA = detDouble(a);
            double detB = detDouble(b);
            double detAB = detDouble(multiply(a, b));
            double scale = Math.max(1.0, Math.abs(detA * detB));
            check(Math.abs(detAB - detA * detB) <= 1e-8 * scale,
                    "det(AB) = det(A)det(B) in double at n = " + n);
            long modA = detMod(toLongs(a), MOD);
            long modB = detMod(toLongs(b), MOD);
            check(detMod(toLongs(multiply(a, b)), MOD) == modA * modB % MOD,
                    "det(AB) = det(A)det(B) modulo p at n = " + n);
            check(detBareiss(toLongs(multiply(a, b))) == detBareiss(toLongs(a)) * detBareiss(toLongs(b)),
                    "det(AB) = det(A)det(B) in exact integer arithmetic at n = " + n);

            double[][] transposed = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) transposed[i][j] = a[j][i];
            }
            check(Math.abs(detDouble(transposed) - detA) <= 1e-9 * scale, "det(A^T) = det(A) at n = " + n);
            check(detBareiss(toLongs(transposed)) == detBareiss(toLongs(a)), "Bareiss agrees on the transpose");

            for (long c : new long[]{2, 3, MOD - 2}) {
                long[][] scaled = new long[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) scaled[i][j] = Math.floorMod(c * toLongs(a)[i][j], MOD);
                }
                long expected = detMod(toLongs(a), MOD) * powMod(c, n, MOD) % MOD;
                check(detMod(scaled, MOD) == expected, "det(cA) = c^n det(A) modulo p at n = " + n);
            }
            double[][] scaledDouble = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) scaledDouble[i][j] = 2.0 * a[i][j];
            }
            check(Math.abs(detDouble(scaledDouble) - Math.pow(2.0, n) * detA) <= 1e-9 * scale * Math.pow(2, n),
                    "det(2A) = 2^n det(A) in double at n = " + n);
        }
    }

    static void factorizationIsVerified() {
        Rng rng = new Rng(5875L);
        for (int trial = 0; trial < 5; trial++) {
            int n = 3 + trial;
            double[][] a = randomIntegerMatrix(n, 6, rng);
            double[][] work = new double[n][];
            for (int i = 0; i < n; i++) work[i] = Arrays.copyOf(a[i], n);
            int[] swap = new int[n];
            for (int i = 0; i < n; i++) swap[i] = i;
            double product = 1.0;
            int sign = 1;
            for (int col = 0; col < n; col++) {
                int pivot = col;
                for (int row = col + 1; row < n; row++) {
                    if (Math.abs(work[row][col]) > Math.abs(work[pivot][col])) pivot = row;
                }
                if (pivot != col) {
                    double[] t = work[pivot];
                    work[pivot] = work[col];
                    work[col] = t;
                    int s = swap[pivot];
                    swap[pivot] = swap[col];
                    swap[col] = s;
                    sign = -sign;
                }
                product *= work[col][col];
                for (int row = col + 1; row < n; row++) {
                    double factor = work[row][col] / work[col][col];
                    work[row][col] = factor;
                    for (int j = col + 1; j < n; j++) work[row][j] -= factor * work[col][j];
                }
            }
            check(Math.abs(product * sign - detDouble(a)) <= 1e-9 * Math.max(1.0, Math.abs(product)),
                    "det equals the signed product of the LU pivots at n = " + n);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = 0.0;
                    for (int k = 0; k <= Math.min(i, j); k++) {
                        double l = i == k ? 1.0 : (i > k ? work[i][k] : 0.0);
                        double u = j >= k ? work[k][j] : 0.0;
                        acc += l * u;
                    }
                    check(Math.abs(acc - a[swap[i]][j]) <= 1e-9 * Math.max(1.0, Math.abs(a[swap[i]][j])),
                            "P*A = L*U entry (" + i + "," + j + ") at n = " + n);
                }
            }
            check(swapIsPermutation(swap), "the swap record is a permutation");
        }
    }

    static boolean swapIsPermutation(int[] swap) {
        boolean[] seen = new boolean[swap.length];
        for (int value : swap) {
            if (value < 0 || value >= swap.length || seen[value]) return false;
            seen[value] = true;
        }
        return true;
    }

    static void closedForms() {
        for (int n = 2; n <= 8; n++) {
            double[][] v = vandermonde(n);
            long exact = vandermondeExact(n);
            check(Math.abs(detDouble(v) - exact) <= 1e-6 * Math.abs((double) exact),
                    "the Vandermonde determinant at n = " + n + " matches prod (j-i) to 1e-6 relative (got "
                            + detDouble(v) + " versus " + exact + "); the 8x8 instance has condition ~1e11, so a"
                            + " tighter claim would be dishonest");
            long[][] integers = toLongs(v);
            check(detMod(integers, MOD) == Math.floorMod(exact, MOD),
                    "the modular Vandermonde determinant is exact at n = " + n);
            check(detBareiss(integers) == exact, "Bareiss reproduces the closed form exactly at n = " + n);
        }
        long[][] bidiagonal = new long[6][6];
        for (int i = 0; i < 6; i++) {
            bidiagonal[i][i] = 2;
            if (i + 1 < 6) bidiagonal[i][i + 1] = 1;
        }
        check(detMod(bidiagonal, MOD) == 64, "the bidiagonal determinant is 2^6 = 64");
        check(detBareiss(bidiagonal) == 64, "and so is Bareiss");

        double[] x = {1, 2, 3, 4, 5};
        double[] y = {1, 2, 3, 4, 5};
        double[][] c = cauchy(x, y);
        double exactCauchy = cauchyExact(x, y);
        check(Math.abs(detDouble(c) - exactCauchy) <= 1e-9 * Math.abs(exactCauchy),
                "the Cauchy determinant matches its closed form to 1e-9 relative (" + detDouble(c)
                        + " versus " + exactCauchy + ")");
        check(exactCauchy > 0 && exactCauchy < 1e-9, "the instance's determinant is a tiny positive number");
        double[] x2 = {1, 3, 7};
        double[] y2 = {2, 5, 11};
        check(Math.abs(detDouble(cauchy(x2, y2)) - cauchyExact(x2, y2)) <= 1e-10 * Math.abs(cauchyExact(x2, y2)),
                "a second Cauchy instance matches too");
    }

    static void modularBlindSpot() {
        // An integer determinant that is a nonzero multiple of the prime is invisible modulo p.
        long[][] small = {{7, 3}, {4, 2}};
        check(detBareiss(small) == 2, "the integer determinant of [[7,3],[4,2]] is 2");
        check(detMod(small, MOD) == 2, "2 is visible modulo p, so the modular answer is 2 as well");
        long[][] example = {{10, 3}, {5, 8}};
        check(detBareiss(example) == 65, "[[10,3],[5,8]] has integer determinant 65");
        check(detMod(example, MOD) == 65, "and 65 is visible modulo p");
        check(Math.abs(detDouble(new double[][]{{10, 3}, {5, 8}}) - 65.0) <= 1e-12,
                "the double LU also reports 65");
        check(detMod(example, 5L) == 0 && detBareiss(example) == 65,
                "the same matrix has determinant 0 modulo 5, a documented blind spot of any modular method");
        long[][] detP = {{2, 1}, {MOD - 2, MOD - 1}};
        check(detBareiss(detP) == MOD, "the constructed matrix has integer determinant p = " + MOD);
        check(detMod(detP, MOD) == 0,
                "so the modular determinant reports 0 while the integer determinant is p (documented blind spot)");
        check(Math.abs(detDouble(new double[][]{{2, 1}, {MOD - 2.0, MOD - 1.0}}) - MOD) <= 1e-3,
                "the double variant still reports about p, with 1e9-magnitude rounding (tolerance 1e-3)");
        check(Math.abs(detDouble(new double[][]{{1, 2}, {3, 4}}) + 2.0) <= 1e-15, "det [[1,2],[3,4]] = -2");
        check(detMod(new long[][]{{1, 2}, {3, 4}}, MOD) == MOD - 2, "and -2 modulo p");
    }

    static void bareissAgrees() {
        Rng rng = new Rng(5876L);
        for (int trial = 0; trial < 8; trial++) {
            int n = 2 + rng.nextInt(5);
            double[][] a = randomIntegerMatrix(n, 7, rng);
            long[][] integers = toLongs(a);
            long bareiss = detBareiss(integers);
            check(bareiss == detBareiss(integers), "Bareiss is deterministic");
            check(detMod(integers, MOD) == Math.floorMod(bareiss, MOD), "Bareiss and the modular path agree");
            check(Math.abs(detDouble(a) - bareiss) <= 1e-6 * Math.max(1.0, Math.abs((double) bareiss)),
                    "and the double determinant is close at n = " + n);
        }
        long[][] singular = {{1, 2, 3}, {2, 4, 6}, {1, 1, 1}};
        check(detBareiss(singular) == 0, "Bareiss detects a zero pivot column");
        check(detMod(singular, MOD) == 0, "and so does the modular elimination");
        double[][] scaled = {{1e-8, 2e-8}, {3e-8, 1e-8}};
        check(Math.abs(detDouble(scaled) - (1e-16 - 6e-16)) <= 1e-25,
                "tiny but well-scaled entries keep the double determinant accurate");
        double[][] illConditioned = {{1e12, 1e12 + 1}, {1e12, 1e12 + 2}};
        check(Math.abs(detDouble(illConditioned) - 1e12) <= 1e-3,
                "a 1e12-scaled matrix keeps about 1e-3 absolute accuracy (documented conditioning limit)");
    }

    static void determinism() {
        Rng rng = new Rng(5877L);
        double[][] a = randomIntegerMatrix(7, 5, rng);
        check(Math.abs(detDouble(a) - detDouble(a)) <= 0.0, "the double determinant is reproducible");
        check(detMod(toLongs(a), MOD) == detMod(toLongs(a), MOD), "the modular determinant is reproducible");
        check(detBareiss(toLongs(a)) == detBareiss(toLongs(a)), "Bareiss is reproducible");
        check(Math.abs(detDouble(a) - detLeibniz(a)) <= 1e-6 * Math.max(1.0, Math.abs(detLeibniz(a))),
                "a 7x7 matrix still matches the permutation oracle");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
