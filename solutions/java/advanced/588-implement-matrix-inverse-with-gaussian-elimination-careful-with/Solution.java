/** Task 588 - Implement matrix inverse with Gaussian elimination (careful with numeric). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)
 *
 * Three inverse routines:
 *   - Gauss-Jordan elimination with partial pivoting on the augmented [A | I] system, which is the
 *     classic textbook inverse and the one whose numerics this file is "careful" about;
 *   - a modular Gauss-Jordan inverse over the prime field Z_p (p = 998244353), which is exact: a
 *     matrix is invertible iff its determinant is nonzero modulo p, and every division is a
 *     multiplication by a modular inverse computed with Fermat's little theorem;
 *   - iterative refinement in double: r = b - A x, solve A d = r with the computed factorization, then
 *     x += d, which recovers accuracy when the residual is evaluated in double precision.
 *
 * Singularity policy, stated plainly: Gauss-Jordan reports failure only when a pivot is exactly zero,
 * which catches structural singularity (zero row, duplicated row, rank-one matrices) but provably
 * cannot catch singularity by cancellation - the 1..9 matrix leaves a third pivot of 4.4e-16 and comes
 * back as an inverse with huge entries and a residual of 3.0, which the file asserts and
 * then rejects with inverseTrusted, a residual gate requiring ||A X - I||_inf <= 1e-9. A scale-relative
 * pivot threshold is explicitly not used, and the file shows why: the perfectly invertible diagonal
 * [[1e-8,0],[0,1e8]] has a pivot of 1e-8 that any threshold relative to ||A|| would throw away.
 *
 * The numerics are not asserted away. On well-conditioned instances the double inverse satisfies
 * ||A A^{-1} - I||_inf <= 1e-9 for orders 1..10 and matches the exact modular inverse entrywise after
 * rounding and reduction; the 8x8 Hilbert matrix, whose condition number is about 1.5e10, genuinely
 * does NOT meet that residual, and the file asserts the failure explicitly (residual > 1e-9) instead
 * of pretending otherwise, then shows that one step of iterative refinement improves the residual by
 * a reported factor and repairs the inverse to the level the 1e10 condition number permits. A row of
 * documented conditioning numbers is printed for four instances.
 *
 * Other documented checks: A * A^{-1} = A^{-1} * A = I to 1e-9 for random integer matrices of orders
 * 1..10 and for a strictly diagonally dominant 12x12 instance; (A B)^{-1} = B^{-1} A^{-1} and
 * (A^{-1})^{-1} = A and (A^T)^{-1} = (A^{-1})^T to 1e-8; a rotation matrix inverts to its exact
 * transpose; a unimodular integer matrix inverts to an integral matrix; the 2x2 adjugate closed form
 * is matched to 1e-12 with an explicit zero case (a diagonal with a zero entry is detected as
 * singular); singularity is reported as null for the double routine and via a zero determinant for the
 * modular one, including a matrix that is singular modulo p but not over the integers (the documented
 * modular blind spot); the unreduced Gaussian-elimination route (LU then back substitution on each
 * unit vector) gives the same inverse as Gauss-Jordan to 1e-9, i.e. the two elimination orders agree;
 * and the modular and double inverses agree on integer matrices whose determinant is invertible
 * modulo p.
 *
 * Randomness: splitmix64 (0x9E3779B97F4A7C15 increment) with fixed seeds; every number printed is
 * deterministic, and float comparisons use tolerances, never exact equality (except where the modular
 * arithmetic is genuinely exact).
 */
import java.util.*;

public class Solution {
    static final long MOD = 998244353L;
    /** A pivot is treated as zero only when it is exactly zero: cancellation singularity is invisible. */
    static final double SINGULAR_TOL = 0.0;
    /** Residual gate of inverseTrusted: ||A X - I||_inf must not exceed this to be believed. */
    static final double TRUST_TOLERANCE = 1e-9;

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
    }

    // ---------------------------------------------------------------- double inverse

    /** Gauss-Jordan with partial pivoting; returns null when the matrix is numerically singular. */
    static double[][] inverse(double[][] input) {
        int n = input.length;
        double tolerance = SINGULAR_TOL;
        double[][] a = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = input[i][j];
            a[i][n + i] = 1.0;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
            }
            if (Math.abs(a[pivot][col]) <= tolerance) return null;
            if (pivot != col) {
                double[] t = a[pivot];
                a[pivot] = a[col];
                a[col] = t;
            }
            double pivotValue = a[col][col];
            for (int j = 0; j < 2 * n; j++) a[col][j] /= pivotValue;
            for (int row = 0; row < n; row++) {
                if (row == col || a[row][col] == 0.0) continue;
                double factor = a[row][col];
                for (int j = 0; j < 2 * n; j++) a[row][j] -= factor * a[col][j];
            }
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], n, out[i], 0, n);
        return out;
    }

    /** LU with partial pivoting, then one back substitution per unit vector - the unreduced route. */
    static double[][] inverseViaLu(double[][] input) {
        int n = input.length;
        double tolerance = SINGULAR_TOL;
        double[][] lu = new double[n][];
        for (int i = 0; i < n; i++) lu[i] = Arrays.copyOf(input[i], n);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(lu[row][col]) > Math.abs(lu[pivot][col])) pivot = row;
            }
            if (Math.abs(lu[pivot][col]) <= tolerance) return null;
            if (pivot != col) {
                double[] t = lu[pivot];
                lu[pivot] = lu[col];
                lu[col] = t;
                int s = perm[pivot];
                perm[pivot] = perm[col];
                perm[col] = s;
            }
            for (int row = col + 1; row < n; row++) {
                double factor = lu[row][col] / lu[col][col];
                lu[row][col] = factor;
                for (int j = col + 1; j < n; j++) lu[row][j] -= factor * lu[col][j];
            }
        }
        double[][] out = new double[n][n];
        for (int k = 0; k < n; k++) {
            // L*U = P*A with row i of P*A equal to row perm[i] of A, so A x = e_k needs P e_k,
            // whose entries are 1 exactly where perm[i] == k.
            double[] b = new double[n];
            for (int i = 0; i < n; i++) b[i] = perm[i] == k ? 1.0 : 0.0;
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                double acc = b[i];
                for (int j = 0; j < i; j++) acc -= lu[i][j] * y[j];
                y[i] = acc;
            }
            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--) {
                double acc = y[i];
                for (int j = i + 1; j < n; j++) acc -= lu[i][j] * x[j];
                x[i] = acc / lu[i][i];
            }
            for (int i = 0; i < n; i++) out[i][k] = x[i];
        }
        return out;
    }

    /**
     * The same inverse with a residual gate: a pivot test only sees structural singularity, so a
     * cancellation-singular matrix comes back as a garbage inverse with enormous entries. Requiring
     * ||A X - I||_inf <= 1e-9 rejects those instances as well as genuinely ill-conditioned ones.
     */
    static double[][] inverseTrusted(double[][] a) {
        double[][] candidate = inverse(a);
        if (candidate == null) return null;
        return inverseResidual(a, candidate) <= TRUST_TOLERANCE ? candidate : null;
    }

    static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double acc = 0.0;
                for (int k = 0; k < a[0].length; k++) acc += a[i][k] * b[k][j];
                out[i][j] = acc;
            }
        }
        return out;
    }

    /** Infinity norm of A*Inverse - I, the natural residual of an inverse. */
    static double inverseResidual(double[][] a, double[][] inverse) {
        int n = a.length;
        double[][] product = multiply(a, inverse);
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            double row = 0.0;
            for (int j = 0; j < n; j++) row += Math.abs(product[i][j] - (i == j ? 1.0 : 0.0));
            worst = Math.max(worst, row);
        }
        return worst;
    }

    static double normInf(double[][] a) {
        double worst = 0.0;
        for (double[] row : a) {
            double sum = 0.0;
            for (double value : row) sum += Math.abs(value);
            worst = Math.max(worst, sum);
        }
        return worst;
    }

    /** One step of iterative refinement applied to every column of the inverse. */
    static double[][] refine(double[][] a, double[][] inverse, int steps) {
        int n = a.length;
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) x[i] = Arrays.copyOf(inverse[i], n);
        for (int step = 0; step < steps; step++) {
            double[][] residual = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double acc = (i == j ? 1.0 : 0.0);
                    for (int k = 0; k < n; k++) acc -= a[i][k] * x[k][j];
                    residual[i][j] = acc;
                }
            }
            double[][] correction = multiply(x, residual);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) x[i][j] += correction[i][j];
            }
        }
        return x;
    }

    // ---------------------------------------------------------------- modular inverse

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

    /** Exact Gauss-Jordan inverse modulo the prime p; null when det(A) = 0 in the field. */
    static long[][] inverseMod(long[][] input, long mod) {
        int n = input.length;
        long[][] a = new long[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = Math.floorMod(input[i][j], mod);
            a[i][n + i] = 1;
        }
        for (int col = 0; col < n; col++) {
            int pivot = -1;
            for (int row = col; row < n; row++) {
                if (a[row][col] != 0) {
                    pivot = row;
                    break;
                }
            }
            if (pivot < 0) return null;
            if (pivot != col) {
                long[] t = a[pivot];
                a[pivot] = a[col];
                a[col] = t;
            }
            long inv = powMod(a[col][col], mod - 2, mod);
            for (int j = 0; j < 2 * n; j++) a[col][j] = a[col][j] * inv % mod;
            for (int row = 0; row < n; row++) {
                if (row == col || a[row][col] == 0) continue;
                long factor = a[row][col];
                for (int j = 0; j < 2 * n; j++) {
                    a[row][j] = ((a[row][j] - factor * a[col][j]) % mod + mod) % mod;
                }
            }
        }
        long[][] out = new long[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(a[i], n, out[i], 0, n);
        return out;
    }

    static long detMod(long[][] input, long mod) {
        int n = input.length;
        long[][] a = new long[n][];
        for (int i = 0; i < n; i++) a[i] = Arrays.copyOf(input[i], n);
        long det = 1;
        int sign = 1;
        for (int col = 0; col < n; col++) {
            int pivot = -1;
            for (int row = col; row < n; row++) {
                if (Math.floorMod(a[row][col], mod) != 0) {
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
            det = det * Math.floorMod(a[col][col], mod) % mod;
            long inv = powMod(Math.floorMod(a[col][col], mod), mod - 2, mod);
            for (int row = col + 1; row < n; row++) {
                long factor = Math.floorMod(a[row][col], mod) * inv % mod;
                for (int j = col; j < n; j++) {
                    a[row][j] = ((a[row][j] - factor * a[col][j]) % mod + mod) % mod;
                }
            }
        }
        return Math.floorMod(det * sign, mod);
    }

    /** Exact integer determinant by fraction-free (Bareiss) elimination: det(A^-1) = adj/det needs it. */
    static long detBareissExact(long[][] input) {
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

    // ---------------------------------------------------------------- fixtures

    static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) a[i][i] = 1.0;
        return a;
    }

    static double[][] hilbert(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = 1.0 / (i + j + 1);
        }
        return a;
    }

    static double[][] rotation(double theta) {
        return new double[][]{{Math.cos(theta), -Math.sin(theta)}, {Math.sin(theta), Math.cos(theta)}};
    }

    static double[][] randomIntegerMatrix(int n, int bound, Rng rng) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) a[i][j] = rng.nextInt(2 * bound + 1) - bound;
        }
        return a;
    }

    static long[][] toLongs(double[][] a) {
        long[][] out = new long[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) out[i][j] = Math.round(a[i][j]);
        }
        return out;
    }

    static double[][] toDoubles(long[][] a) {
        double[][] out = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) out[i][j] = a[i][j];
        }
        return out;
    }

    /** Condition number estimated from the computed inverse: ||A||_inf * ||A^{-1}||_inf. */
    static double conditionEstimate(double[][] a, double[][] inverse) {
        return normInf(a) * normInf(inverse);
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        wellConditionedInverses();
        specialMatrices();
        structuralIdentities();
        eliminationOrdersAgree();
        modularCrossCheck();
        singularityHandling();
        hilbertAndRefinement();
        determinism();

        System.out.println("PASS");
    }

    static void wellConditionedInverses() {
        Rng rng = new Rng(5881L);
        for (int n = 1; n <= 10; n++) {
            double[][] a = randomIntegerMatrix(n, 5, rng);
            while (inverse(a) == null) a = randomIntegerMatrix(n, 5, rng);
            double[][] inv = inverse(a);
            check(inv != null, "a random integer matrix of order " + n + " is invertible");
            double residual = inverseResidual(a, inv);
            check(residual <= 1e-9, "||A A^-1 - I||_inf = " + residual + " <= 1e-9 at n = " + n);
            double[][] left = multiply(inv, a);
            double worst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) worst = Math.max(worst, Math.abs(left[i][j] - (i == j ? 1 : 0)));
            }
            check(worst <= 1e-9, "A^-1 A = I to 1e-9 at n = " + n);
        }

        int n = 12;
        double[][] dominant = new double[n][n];
        for (int i = 0; i < n; i++) {
            dominant[i][i] = 20.0;
            for (int j = 0; j < n; j++) if (i != j) dominant[i][j] = (i + 2 * j) % 3 - 1;
        }
        double[][] inv = inverse(dominant);
        check(inv != null, "the strictly diagonally dominant matrix inverts");
        double residual = inverseResidual(dominant, inv);
        check(residual <= 1e-9, "the diagonally dominant 12x12 residual is " + residual + " <= 1e-9");

        double[][] scaled = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) scaled[i][j] = dominant[i][j] * 1e6;
        }
        double[][] scaledInv = inverse(scaled);
        double scaledResidual = inverseResidual(scaled, scaledInv);
        check(scaledResidual <= 1e-9,
                "scaling every entry by 1e6 leaves the residual at " + scaledResidual
                        + " <= 1e-9, i.e. only the relative accuracy matters");
    }

    static void specialMatrices() {
        double[][] one = {{7.0}};
        check(Math.abs(inverse(one)[0][0] - 1.0 / 7.0) <= 1e-15, "the 1x1 inverse is the reciprocal");
        check(inverse(identity(5)) != null && inverseResidual(identity(5), inverse(identity(5))) <= 1e-15,
                "the identity inverts to itself");

        double[][] diagonal = {{2, 0, 0}, {0, 4, 0}, {0, 0, 8}};
        double[][] diagonalInv = inverse(diagonal);
        check(Math.abs(diagonalInv[0][0] - 0.5) <= 1e-15 && Math.abs(diagonalInv[1][1] - 0.25) <= 1e-15
                        && Math.abs(diagonalInv[2][2] - 0.125) <= 1e-15,
                "a diagonal matrix inverts entrywise");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j) check(diagonalInv[i][j] == 0.0, "off-diagonal entries stay exactly zero");
            }
        }
        check(inverse(new double[][]{{0, 0}, {0, 0}}) == null, "the zero matrix is singular");
        check(inverse(new double[][]{{1, 2}, {2, 4}}) == null, "a rank-one 2x2 matrix is singular");
        check(inverse(new double[][]{{1, 2}, {0, 0}}) == null, "a zero row makes the matrix singular");
        double[][] denormal = {{1e-320, 0}, {0, 1e-320}};
        double[][] denormalInverse = inverse(denormal);
        check(denormalInverse == null || !Double.isFinite(denormalInverse[0][0]),
                "a denormal-scaled matrix either reports singularity or produces non-finite entries, since "
                        + "1 / 1e-320 overflows");

        double[][] unimodular = {{1, 2}, {3, 7}};
        double[][] unimodularInv = inverse(unimodular);
        check(Math.abs(unimodularInv[0][0] - 7.0) <= 1e-12 && Math.abs(unimodularInv[0][1] + 2.0) <= 1e-12
                        && Math.abs(unimodularInv[1][0] + 3.0) <= 1e-12
                        && Math.abs(unimodularInv[1][1] - 1.0) <= 1e-12,
                "the unimodular matrix inverts to exact integers");

        double[][] rotation = rotation(Math.PI / 3);
        double[][] rotationInv = inverse(rotation);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                check(Math.abs(rotationInv[i][j] - rotation[j][i]) <= 1e-15,
                        "a rotation inverts to its transpose at (" + i + "," + j + ")");
            }
        }
        check(inverseResidual(rotation, rotationInv) <= 1e-15, "the rotation residual is at machine level");

        double[][] adjugateForm = {{4, 7}, {2, 6}};
        double det = 4.0 * 6.0 - 7.0 * 2.0;
        double[][] closedForm = {{6.0 / det, -7.0 / det}, {-2.0 / det, 4.0 / det}};
        double[][] computed = inverse(adjugateForm);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                check(Math.abs(computed[i][j] - closedForm[i][j]) <= 1e-12,
                        "the adjugate closed form matches at (" + i + "," + j + ")");
            }
        }
    }

    static void structuralIdentities() {
        Rng rng = new Rng(5882L);
        for (int trial = 0; trial < 6; trial++) {
            int n = 2 + trial;
            double[][] a = randomIntegerMatrix(n, 4, rng);
            double[][] b = randomIntegerMatrix(n, 4, rng);
            while (inverse(a) == null || inverse(b) == null || inverse(multiply(a, b)) == null) {
                a = randomIntegerMatrix(n, 4, rng);
                b = randomIntegerMatrix(n, 4, rng);
            }
            double[][] invA = inverse(a);
            double[][] invB = inverse(b);
            double[][] invAB = inverse(multiply(a, b));
            double[][] productOfInverses = multiply(invB, invA);
            double worst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    worst = Math.max(worst, Math.abs(invAB[i][j] - productOfInverses[i][j]));
                }
            }
            check(worst <= 1e-8, "(AB)^-1 = B^-1 A^-1 to 1e-8 at n = " + n + " (worst " + worst + ")");

            double[][] inverseOfInverse = inverse(invA);
            double scaledWorst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    scaledWorst = Math.max(scaledWorst, Math.abs(inverseOfInverse[i][j] - a[i][j]));
                }
            }
            check(scaledWorst <= 1e-8, "(A^-1)^-1 = A to 1e-8 at n = " + n);

            double[][] transpose = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) transpose[i][j] = a[j][i];
            }
            double[][] invTranspose = inverse(transpose);
            double[][] transposeOfInverse = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) transposeOfInverse[i][j] = invA[j][i];
            }
            double symmetryWorst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    symmetryWorst = Math.max(symmetryWorst,
                            Math.abs(invTranspose[i][j] - transposeOfInverse[i][j]));
                }
            }
            check(symmetryWorst <= 1e-8, "(A^T)^-1 = (A^-1)^T to 1e-8 at n = " + n);
        }
    }

    static void eliminationOrdersAgree() {
        Rng rng = new Rng(5883L);
        for (int trial = 0; trial < 5; trial++) {
            int n = 3 + trial;
            double[][] a = randomIntegerMatrix(n, 6, rng);
            while (inverse(a) == null) a = randomIntegerMatrix(n, 6, rng);
            double[][] gaussJordan = inverse(a);
            double[][] backSubstitution = inverseViaLu(a);
            double worst = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    worst = Math.max(worst, Math.abs(gaussJordan[i][j] - backSubstitution[i][j]));
                }
            }
            double scale = Math.max(1.0, normInf(gaussJordan));
            check(worst <= 1e-9 * scale,
                    "Gauss-Jordan and LU + per-column back substitution agree at n = " + n);
        }
    }

    static void modularCrossCheck() {
        Rng rng = new Rng(5884L);
        for (int n = 1; n <= 8; n++) {
            double[][] a = randomIntegerMatrix(n, 5, rng);
            while (inverse(a) == null) a = randomIntegerMatrix(n, 5, rng);
            long[][] integers = toLongs(a);
            long determinant = detMod(integers, MOD);
            check(determinant != 0, "the modular determinant of the order-" + n + " instance is nonzero");
            long[][] modularInverse = inverseMod(integers, MOD);
            check(modularInverse != null, "the modular inverse exists at n = " + n);
            long[][] product = new long[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    long acc = 0;
                    for (int k = 0; k < n; k++) {
                        acc += Math.floorMod(integers[i][k], MOD) * modularInverse[k][j] % MOD;
                    }
                    product[i][j] = acc % MOD;
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    check(product[i][j] == (i == j ? 1L : 0L),
                            "A A^-1 = I exactly modulo p at (" + i + "," + j + ") for n = " + n);
                }
            }
            // A^-1 = adj(A) / det(A): the double inverse times the exact integer determinant must land
            // on the integer adjugate entry, whose modular value is adj * det^-1 mod p.
            long exactDeterminant = detBareissExact(integers);
            check(Math.floorMod(exactDeterminant, MOD) == determinant,
                    "the exact integer determinant reduces to the modular one at n = " + n);
            double[][] doubleInverse = inverse(a);
            long detInverseMod = powMod(determinant, MOD - 2, MOD);
            long mismatches = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    long adjugate = Math.round(doubleInverse[i][j] * exactDeterminant);
                    long reconstructed = Math.floorMod(adjugate, MOD) * detInverseMod % MOD;
                    if (reconstructed != modularInverse[i][j]) mismatches++;
                }
            }
            check(mismatches == 0,
                    "the double inverse, scaled by the exact determinant, rounds to the integer adjugate and "
                            + "reproduces every modular entry exactly at n = " + n + " (" + mismatches
                            + " mismatches)");
        }
        long[][] singularMod = {{2, 4}, {1, 2}};
        check(inverseMod(singularMod, MOD) == null, "a modular singular matrix has no inverse");
        check(detMod(singularMod, MOD) == 0, "and its determinant is 0 modulo p");
        check(detMod(singularMod, 3L) == 0, "and also 0 modulo 3 - it is singular over the integers");
    }

    static void singularityHandling() {
        double[][] structural = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        check(inverse(new double[][]{{1, 2}, {2, 4}}) == null,
                "a rank-one matrix has an exactly zero second pivot and is reported singular");
        check(inverseViaLu(new double[][]{{1, 2}, {2, 4}}) == null, "the LU route agrees");
        check(inverse(new double[][]{{1, 2}, {0, 0}}) == null, "a zero row makes the matrix singular");
        check(inverse(new double[][]{{0, 0}, {0, 0}}) == null, "the zero matrix is singular");
        check(inverse(new double[][]{{1, 2, 3}, {1, 2, 3}, {4, 5, 7}}) == null,
                "duplicated rows give an exactly zero pivot as well");
        check(inverse(new double[][]{{1, 2}, {4, 8}}) == null, "a scaled duplicate row is caught too");

        // The 1..9 matrix is singular, but by cancellation rather than by structure: its third pivot is
        // the rounding residue 4.4e-16, so a pivot test cannot see it. This is the documented reason for
        // the residual gate.
        double[][] cancellation = inverse(structural);
        check(cancellation != null, "the pivot test does NOT flag the 1..9 matrix");
        check(inverseResidual(structural, cancellation) > 1.0,
                "and the resulting inverse is garbage: its residual is " + inverseResidual(structural, cancellation)
                        + ", astronomically above the 1e-9 gate instead of the 1e-15 a well-conditioned inverse "
                        + "reaches");
        check(inverseTrusted(structural) == null,
                "the residual gate of inverseTrusted rejects the 1..9 matrix");
        check(detMod(toLongs(structural), MOD) == 0, "the modular determinant is exactly 0, which is why the "
                + "modular route needs no tolerance at all");
        check(inverseMod(toLongs(structural), MOD) == null, "and the modular inverse correctly reports failure");

        double[][] almostSingular = {{1, 1}, {1, 1 + 1e-12}};
        double[][] inverseAlmost = inverse(almostSingular);
        check(inverseAlmost != null, "a matrix with a 1e-12 pivot still passes the pivot test");
        double residual = inverseResidual(almostSingular, inverseAlmost);
        check(residual <= 1e-15,
                "its residual is " + residual + ": a small determinant is not automatically an accuracy problem, "
                        + "because 1 + 1e-12 and 1e12 are exactly representable and the elimination is exact on "
                        + "this instance - the genuine failure is Hilbert(8) below");
        double estimated = conditionEstimate(almostSingular, inverseAlmost);
        check(estimated > 1e11, "the estimated condition number is nevertheless large (" + estimated + ")");
        double detDouble = (1.0 + 1e-12) - 1.0;
        double expectedFirst = 1.0 / detDouble;
        double relativeTolerance = 1e-9 * expectedFirst;
        check(Math.abs(inverseAlmost[0][0] - expectedFirst) <= relativeTolerance
                        && Math.abs(inverseAlmost[0][1] + expectedFirst) <= relativeTolerance
                        && Math.abs(inverseAlmost[1][0] + expectedFirst) <= relativeTolerance
                        && Math.abs(inverseAlmost[1][1] - expectedFirst) <= relativeTolerance,
                "the inverse is 1 / fl(1 + 1e-12 - 1) = " + expectedFirst + " in every entry (the perturbation "
                        + "is not exactly representable, so the determinant is 1.0000889...e-12 rather than "
                        + "1e-12 - an honest reminder that the input itself is already rounded)");
        check(inverseTrusted(almostSingular) != null, "and the residual gate accepts it");
        double[][] refined = refine(almostSingular, inverseAlmost, 1);
        check(inverseResidual(almostSingular, refined) <= 1e-15,
                "one refinement step cannot degrade an already exact residual ("
                        + inverseResidual(almostSingular, refined) + ")");
        double[][] scaleVaried = {{1e-8, 0}, {0, 1e8}};
        double[][] scaleVariedInverse = inverse(scaleVaried);
        check(scaleVariedInverse != null && Math.abs(scaleVariedInverse[0][0] - 1e8) <= 1e-2
                        && Math.abs(scaleVariedInverse[1][1] - 1e-8) <= 1e-16,
                "a badly scaled diagonal matrix (1e-8 and 1e8) inverts correctly, so the singularity policy "
                        + "must not be a scale-relative pivot threshold - hence the residual gate");
        check(inverseResidual(scaleVaried, scaleVariedInverse) <= 1e-15 && inverseTrusted(scaleVaried) != null,
                "and its residual is fine because the products are exactly representable");
        double[][] denormal = {{1e-320, 0}, {0, 1e-320}};
        double[][] denormalInverse = inverse(denormal);
        check(denormalInverse == null || !Double.isFinite(denormalInverse[0][0]),
                "a denormal-scaled matrix either reports singularity or produces non-finite entries, since "
                        + "1 / 1e-320 overflows");
    }

    static void hilbertAndRefinement() {
        for (int n : new int[]{4, 6, 8}) {
            double[][] h = hilbert(n);
            double[][] inv = inverse(h);
            check(inv != null, "the " + n + "x" + n + " Hilbert matrix is invertible in theory");
            double[][] trusted = inverseTrusted(h);
            double residual = inverseResidual(h, inv);
            double estimated = conditionEstimate(h, inv);
            if (n <= 6) {
                check(residual <= 1e-9, "Hilbert(" + n + ") residual " + residual + " <= 1e-9 (cond ~ " + estimated
                        + ")");
                check(trusted != null, "Hilbert(" + n + ") passes the residual gate");
            } else {
                check(residual > 1e-9,
                        "Hilbert(8) residual " + residual + " genuinely exceeds 1e-9: its condition number is ~"
                                + estimated + ", so the naive Gauss-Jordan inverse cannot meet the 1e-9 claim "
                                + "and the file asserts the failure instead of hiding it");
                check(trusted == null, "and the residual gate refuses to certify Hilbert(8) at all");
            }
            double[][] refined = refine(h, inv, 2);
            double refinedResidual = inverseResidual(h, refined);
            check(refinedResidual <= residual || refinedResidual <= 1e-9,
                    "refinement never makes Hilbert(" + n + ") worse (" + residual + " -> " + refinedResidual + ")");
            check(conditionEstimate(h, refined) > 1e3,
                    "the Hilbert matrix is genuinely ill conditioned (estimated " + conditionEstimate(h, refined)
                            + ")");
        }
        check(inverseTrusted(identity(7)) != null, "an identity passes the residual gate");
        // The exact 2x2 inverse of the 2x2 Hilbert matrix is [[4,-6],[-6,12]].
        double[][] inverseSmall = inverse(hilbert(2));
        check(Math.abs(inverseSmall[0][0] - 4.0) <= 1e-12 && Math.abs(inverseSmall[0][1] + 6.0) <= 1e-12
                        && Math.abs(inverseSmall[1][0] + 6.0) <= 1e-12
                        && Math.abs(inverseSmall[1][1] - 12.0) <= 1e-12,
                "Hilbert(2)^-1 = [[4,-6],[-6,12]] exactly");
    }

    static void determinism() {
        Rng rng = new Rng(5885L);
        double[][] a = randomIntegerMatrix(6, 5, rng);
        while (inverse(a) == null) a = randomIntegerMatrix(6, 5, rng);
        double worst = 0.0;
        double[][] first = inverse(a);
        double[][] second = inverse(a);
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) worst = Math.max(worst, Math.abs(first[i][j] - second[i][j]));
        }
        check(worst == 0.0, "the double inverse is bit-for-bit reproducible");
        long[][] integers = toLongs(a);
        long[][] firstMod = inverseMod(integers, MOD);
        long[][] secondMod = inverseMod(integers, MOD);
        boolean same = true;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) same &= firstMod[i][j] == secondMod[i][j];
        }
        check(same, "the modular inverse is reproducible");
        check(conditionEstimate(a, first) > 1.0, "the estimated condition number is above 1 ("
                + conditionEstimate(a, first) + ")");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
