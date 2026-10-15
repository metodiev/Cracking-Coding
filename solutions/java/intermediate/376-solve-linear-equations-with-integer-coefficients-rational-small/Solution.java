/** Task 376 - Exact linear systems over rationals (integer coefficients). (Tier: Intermediate, CTCI Ch 6) */

/**
 * The same Gaussian elimination as task 375, but every entry is an exact rational
 * number: a small Fraction type on top of gcd (implemented here per language) keeps
 * num/den normalised with den > 0.  All arithmetic is exact, so:
 *   * the systems of task 375 return exact rational solutions - every assertion is an
 *     EQUALITY of rationals, NOT a tolerance check (contrast with 375);
 *   * no pivoting strategy is required: the first non-zero pivot can be taken because
 *     exact arithmetic never introduces rounding, so a zero pivot really is zero
 *     (documented; 375's partial pivoting exists to limit rounding, which cannot
 *     happen here);
 *   * sentinels: status "unique"/"infinite"/"inconsistent", x = null unless unique.
 *
 * Cross-check with floats: a compact double-precision Gaussian implemented HERE (same
 * file, duplicating 375's algorithm) must agree with the exact result to 1e-9 - this is
 * the documented link between the two tasks; with small integer inputs the doubles
 * round the rationals faithfully.
 *
 * Inputs and expected solutions include genuine fractions (1/3, 1/6) whose float images
 * cannot be represented exactly - those are asserted exactly on the rationals.
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

/** Exact fraction with normalised sign and gcd-reduced terms. */
class Rational {
    final long num;
    final long den;

    Rational(long num) {
        this(num, 1);
    }

    Rational(long num, long den) {
        if (den == 0) throw new ArithmeticException("rational with zero denominator");
        if (den < 0) {
            num = -num;
            den = -den;
        }
        long g = gcd(Math.abs(num), den);
        if (g == 0) {
            num = 0;
            den = 1;
        } else {
            num /= g;
            den /= g;
        }
        this.num = num;
        this.den = den;
    }

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    Rational add(Rational o) {
        return new Rational(num * o.den + o.num * den, den * o.den);
    }

    Rational sub(Rational o) {
        return new Rational(num * o.den - o.num * den, den * o.den);
    }

    Rational mul(Rational o) {
        return new Rational(num * o.num, den * o.den);
    }

    Rational div(Rational o) {
        if (o.num == 0) throw new ArithmeticException("division by zero rational");
        return new Rational(num * o.den, den * o.num);
    }

    Rational neg() {
        return new Rational(-num, den);
    }

    boolean isZero() {
        return num == 0;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Rational)) return false;
        Rational o = (Rational) other;
        return num == o.num && den == o.den;
    }

    @Override
    public int hashCode() {
        return Long.hashCode(num) * 31 + Long.hashCode(den);
    }

    @Override
    public String toString() {
        return den == 1 ? Long.toString(num) : num + "/" + den;
    }
}

/** status in {"unique", "infinite", "inconsistent"}; x is null unless unique. */
class RationalResult {
    final String status;
    final Rational[] x;

    RationalResult(String status, Rational[] x) {
        this.status = status;
        this.x = x;
    }
}

public class Solution {
    static RationalResult solveRational(int[][] matrix, long[] rhs) {
        int n = matrix.length;
        Rational[][] augmented = new Rational[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) augmented[i][j] = new Rational(matrix[i][j]);
            augmented[i][n] = new Rational(rhs[i]);
        }

        int rank = 0;
        for (int col = 0; col < n; col++) {
            int pivotRow = -1;
            for (int row = rank; row < n; row++) {
                if (!augmented[row][col].isZero()) {
                    pivotRow = row;
                    break;
                }
            }
            if (pivotRow == -1) continue;                         // genuinely free column
            Rational[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++) {
                if (augmented[row][col].isZero()) continue;
                Rational factor = augmented[row][col].div(augmented[rank][col]);
                for (int k = col; k <= n; k++) {
                    augmented[row][k] = augmented[row][k].sub(factor.mul(augmented[rank][k]));
                }
            }
            rank++;
        }

        for (int row = rank; row < n; row++) {
            if (!augmented[row][n].isZero()) return new RationalResult("inconsistent", null);
        }
        if (rank < n) return new RationalResult("infinite", null);

        Rational[] x = new Rational[n];
        for (int i = 0; i < n; i++) x[i] = new Rational(0);
        for (int row = n - 1; row >= 0; row--) {
            Rational total = augmented[row][n];
            for (int col = row + 1; col < n; col++) {
                total = total.sub(augmented[row][col].mul(x[col]));
            }
            x[row] = total.div(augmented[row][row]);
        }
        return new RationalResult("unique", x);
    }

    /** Compact double-precision copy of 375's elimination, used only as a cross-check. */
    static double[] solveDouble(int[][] matrix, long[] rhs) {
        double eps = 1e-12;
        int n = matrix.length;
        double[][] augmented = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) augmented[i][j] = matrix[i][j];
            augmented[i][n] = rhs[i];
        }
        int rank = 0;
        for (int col = 0; col < n; col++) {
            int pivotRow = -1;
            double pivotSize = 0.0;
            for (int row = rank; row < n; row++) {
                if (Math.abs(augmented[row][col]) > pivotSize) {
                    pivotSize = Math.abs(augmented[row][col]);
                    pivotRow = row;
                }
            }
            if (pivotRow == -1 || pivotSize <= eps) continue;
            double[] swap = augmented[rank];
            augmented[rank] = augmented[pivotRow];
            augmented[pivotRow] = swap;
            for (int row = rank + 1; row < n; row++) {
                double factor = augmented[row][col] / augmented[rank][col];
                for (int k = col; k <= n; k++) augmented[row][k] -= factor * augmented[rank][k];
            }
            rank++;
        }
        if (rank < n) return null;
        double[] x = new double[n];
        for (int row = n - 1; row >= 0; row--) {
            double total = augmented[row][n];
            for (int col = row + 1; col < n; col++) total -= augmented[row][col] * x[col];
            x[row] = total / augmented[row][row];
        }
        return x;
    }

    static boolean sameRationals(Rational[] got, long[] expectedNum, long[] expectedDen) {
        if (got.length != expectedNum.length) return false;
        for (int i = 0; i < got.length; i++) {
            if (!got[i].equals(new Rational(expectedNum[i], expectedDen[i]))) return false;
        }
        return true;
    }

    static Object[] lcgSystem(Lcg lcg) {
        int size = 2 + lcg.next(3);
        int[][] matrix = new int[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) matrix[i][j] = lcg.next(11) - 5;
        }
        int[] planted = new int[size];
        for (int i = 0; i < size; i++) planted[i] = lcg.next(11) - 5;
        long[] rhs = new long[size];
        for (int i = 0; i < size; i++) {
            long total = 0;
            for (int j = 0; j < size; j++) total += (long) matrix[i][j] * planted[j];
            rhs[i] = total;
        }
        return new Object[]{matrix, rhs, planted};
    }

    public static void main(String[] args) {
        // 2x2 integer system: x = 2, y = 1, exactly
        RationalResult result = solveRational(new int[][]{{1, 1}, {1, -1}}, new long[]{3, 1});
        check(result.status.equals("unique"), "2x2 unique");
        check(sameRationals(result.x, new long[]{2, 1}, new long[]{1, 1}), "2x2 exact (2, 1)");

        // pivot swap: leading zero, x = (3, 2)
        result = solveRational(new int[][]{{0, 1}, {1, 0}}, new long[]{2, 3});
        check(result.status.equals("unique"), "pivot swap unique");
        check(sameRationals(result.x, new long[]{3, 2}, new long[]{1, 1}), "pivot swap (3, 2)");

        // classic 3x3, solution (5, 3, -2)
        result = solveRational(new int[][]{{1, 1, 1}, {0, 2, 5}, {2, 5, -1}}, new long[]{6, -4, 27});
        check(result.status.equals("unique"), "3x3 unique");
        check(sameRationals(result.x, new long[]{5, 3, -2}, new long[]{1, 1, 1}), "3x3 exact (5, 3, -2)");

        // 4x4 diagonally dominant, solution (1, 2, 3, 4)
        result = solveRational(new int[][]{{2, 1, 1, 1}, {1, 2, 1, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}},
                new long[]{11, 12, 13, 14});
        check(result.status.equals("unique"), "4x4 unique");
        check(sameRationals(result.x, new long[]{1, 2, 3, 4}, new long[]{1, 1, 1, 1}), "4x4 exact");

        // genuine fractions: 3x = 1 -> x = 1/3
        result = solveRational(new int[][]{{3}}, new long[]{1});
        check(result.status.equals("unique"), "1x1 unique");
        check(sameRationals(result.x, new long[]{1}, new long[]{3}), "1x1 exact 1/3");

        // 2x + 4y = 1, x - y = 0 -> x = y = 1/6
        result = solveRational(new int[][]{{2, 4}, {1, -1}}, new long[]{1, 0});
        check(result.status.equals("unique"), "1/6 system unique");
        check(sameRationals(result.x, new long[]{1, 1}, new long[]{6, 6}), "1/6 exact");

        // 5x + 2y = 1, x + y = 0 -> x = 1/3, y = -1/3
        result = solveRational(new int[][]{{5, 2}, {1, 1}}, new long[]{1, 0});
        check(result.status.equals("unique"), "third system unique");
        check(sameRationals(result.x, new long[]{1, -1}, new long[]{3, 3}), "1/3, -1/3 exact");

        // sentinels behave exactly like 375's
        result = solveRational(new int[][]{{1, 1}, {2, 2}}, new long[]{1, 2});
        check(result.status.equals("infinite") && result.x == null, "infinite sentinel");
        result = solveRational(new int[][]{{1, 1}, {2, 2}}, new long[]{1, 5});
        check(result.status.equals("inconsistent") && result.x == null, "inconsistent sentinel");
        result = solveRational(new int[][]{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, new long[]{1, 2, 3});
        check(result.status.equals("infinite") && result.x == null, "3x3 dependent infinite");
        result = solveRational(new int[][]{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, new long[]{1, 2, 4});
        check(result.status.equals("inconsistent") && result.x == null, "3x3 dependent inconsistent");

        // --- float cross-check: the double solver agrees to 1e-9 ---------------
        int[][][] matrices = {
                {{1, 1}, {1, -1}}, {{0, 1}, {1, 0}}, {{1, 1, 1}, {0, 2, 5}, {2, 5, -1}},
                {{3}}, {{2, 4}, {1, -1}}, {{5, 2}, {1, 1}}};
        long[][] rhss = {{3, 1}, {2, 3}, {6, -4, 27}, {1}, {1, 0}, {1, 0}};
        long[][] expectedNums = {{2, 1}, {3, 2}, {5, 3, -2}, {1}, {1, 1}, {1, -1}};
        long[][] expectedDens = {{1, 1}, {1, 1}, {1, 1, 1}, {3}, {6, 6}, {3, 3}};
        for (int s = 0; s < matrices.length; s++) {
            double[] approx = solveDouble(matrices[s], rhss[s]);
            check(approx != null, "float cross-check solvable " + s);
            for (int i = 0; i < approx.length; i++) {
                double exact = (double) expectedNums[s][i] / expectedDens[s][i];
                check(Math.abs(approx[i] - exact) < 1e-9, "float cross-check " + s + " index " + i);
            }
        }

        // --- LCG cross-checks: exact recovery of the planted solution ----------
        Lcg lcg = new Lcg(37613);
        int uniqueCount = 0;
        for (int trial = 0; trial < 80; trial++) {
            Object[] system = lcgSystem(lcg);
            int[][] matrix = (int[][]) system[0];
            long[] rhs = (long[]) system[1];
            int[] planted = (int[]) system[2];
            result = solveRational(matrix, rhs);
            if (!result.status.equals("unique")) continue;         // singular; skipped
            uniqueCount++;
            long[] plantedNum = new long[planted.length];
            long[] plantedDen = new long[planted.length];
            for (int i = 0; i < planted.length; i++) {
                plantedNum[i] = planted[i];
                plantedDen[i] = 1;
            }
            check(sameRationals(result.x, plantedNum, plantedDen), "planted solution " + trial);
            // exact substitution A x == b, no tolerance involved
            for (int i = 0; i < matrix.length; i++) {
                Rational total = new Rational(0);
                for (int j = 0; j < matrix[i].length; j++) {
                    total = total.add(new Rational(matrix[i][j]).mul(result.x[j]));
                }
                check(total.equals(new Rational(rhs[i])), "exact substitution " + trial);
            }
            double[] approx = solveDouble(matrix, rhs);
            check(approx != null, "float path solvable " + trial);
            for (int i = 0; i < planted.length; i++) {
                check(Math.abs(approx[i] - planted[i]) < 1e-9, "float agrees " + trial);
            }
        }
        check(uniqueCount >= 40, "enough unique systems, got " + uniqueCount);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
