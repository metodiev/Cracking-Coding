/** Task 708 - Implement fractional packing/covering using multiplicative weights update.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * Both linear programs are relaxations that appear everywhere in combinatorial optimisation:
 *
 *   packing:  max { 1^T x : A x <= 1, x >= 0 }        covering:  min { 1^T x : A x >= 1, x >= 0 }
 *
 * with a non-negative matrix A. They are solved here with the multiplicative-weights / Lagrangian
 * framework (Plotkin-Shmoys-Tardos, Arora-Hazan-Kale) whose only oracle is "pick the single most
 * promising variable":
 *
 *   - keep a weight w_i > 0 per constraint row, all starting at 1;
 *   - packing: pick the column minimising the weighted usage w . A_j (the best profit per weighted
 *     resource) and take the largest step that column allows, step = 1 / max_i A_ij, so its binding
 *     row hits exactly 1; then multiply every weight by 1 + epsilon * A_ij * step, i.e. the rows that
 *     the column consumed become expensive;
 *   - covering: pick the column maximising the weighted coverage w . A_j and take the same maximal
 *     step; then multiply every weight by 1 + epsilon * (1 - A_ij * step), which makes the rows the
 *     column covered *badly* expensive, so the next pick has to serve them;
 *   - accumulate X over the iterations and finally rescale it to the boundary of the feasible region
 *     (down to max_i (A X)_i = 1 for packing, up to min_i (A X)_i = 1 for covering).
 *
 * The primal solution is the rescaled accumulation; the weight vector doubles as the dual solution,
 * which is why the same loop solves both programs. Verified: on many small random instances both
 * rescaled solutions are checked for feasibility, compared against the exact optimum computed by
 * enumerating every vertex of the polytope (Gaussian elimination on all full-size subsets of tight
 * constraints), the packing value is shown to be at most the optimum, the covering cost at least the
 * optimum, and the measured gaps are asserted to stay within a few percent. Strong duality of the two
 * forms (min covering = max packing of A transposed) is verified as well, which cross-checks the exact
 * oracle itself, and degenerate input (an all-zero row makes covering infeasible, a zero column can
 * never help) is rejected or reported.
 */
import java.util.*;

class MultiplicativeWeights {
    private static void validate(int iterations, double epsilon) {
        if (iterations < 1) throw new IllegalArgumentException("need at least one iteration");
        if (!(epsilon > 0 && epsilon < 1)) throw new IllegalArgumentException("epsilon must be in (0,1)");
    }

    /** max { 1^T x : A x <= 1, x >= 0 }; returns a feasible, rescaled solution. */
    static double[] packing(double[][] a, int iterations, double epsilon) {
        validate(iterations, epsilon);
        int rows = a.length;
        int columns = a[0].length;
        double[] weight = new double[rows];
        Arrays.fill(weight, 1.0);
        double[] x = new double[columns];
        for (int iteration = 0; iteration < iterations; iteration++) {
            int chosen = -1;
            double bestUsage = Double.MAX_VALUE;
            for (int column = 0; column < columns; column++) {
                double usage = 0;
                for (int row = 0; row < rows; row++) usage += weight[row] * a[row][column];
                if (usage < bestUsage - 1e-15) {
                    bestUsage = usage;
                    chosen = column;
                }
            }
            double step = largestStep(a, chosen);
            x[chosen] += step;
            for (int row = 0; row < rows; row++) {
                weight[row] *= 1.0 + epsilon * a[row][chosen] * step;
            }
        }
        double worst = 0;
        for (int row = 0; row < rows; row++) worst = Math.max(worst, usage(a, row, x));
        if (worst > 0) {
            // scale the accumulated point out to the boundary: max_i (A x)_i == 1 exactly
            for (int column = 0; column < columns; column++) x[column] /= worst;
        }
        return x;
    }

    /** min { 1^T x : A x >= 1, x >= 0 }; null when a row of A is all zeros (infeasible). */
    static double[] covering(double[][] a, int iterations, double epsilon) {
        validate(iterations, epsilon);
        int rows = a.length;
        int columns = a[0].length;
        for (int row = 0; row < rows; row++) {
            double total = 0;
            for (int column = 0; column < columns; column++) total += a[row][column];
            if (!(total > 0)) return null;
        }
        double[] weight = new double[rows];
        Arrays.fill(weight, 1.0);
        double[] x = new double[columns];
        for (int iteration = 0; iteration < iterations; iteration++) {
            int chosen = -1;
            double bestCoverage = -1;
            for (int column = 0; column < columns; column++) {
                double coverage = 0;
                for (int row = 0; row < rows; row++) coverage += weight[row] * a[row][column];
                if (coverage > bestCoverage + 1e-15) {
                    bestCoverage = coverage;
                    chosen = column;
                }
            }
            double step = largestStep(a, chosen);
            x[chosen] += step;
            for (int row = 0; row < rows; row++) {
                weight[row] *= 1.0 + epsilon * (1.0 - Math.min(1.0, a[row][chosen] * step));
            }
        }
        double least = Double.MAX_VALUE;
        for (int row = 0; row < rows; row++) least = Math.min(least, usage(a, row, x));
        if (!(least > 0)) return null;
        // scale in to the boundary: min_i (A x)_i == 1 exactly, which is the cheapest feasible point
        // on the ray through the accumulated solution
        for (int column = 0; column < columns; column++) x[column] /= least;
        return x;
    }

    /** Largest value a single column can take while keeping its binding row at exactly one. */
    private static double largestStep(double[][] a, int column) {
        double coefficient = 0;
        for (int row = 0; row < a.length; row++) coefficient = Math.max(coefficient, a[row][column]);
        return coefficient <= 0 ? 0 : 1.0 / coefficient;
    }

    static double usage(double[][] a, int row, double[] x) {
        double total = 0;
        for (int column = 0; column < x.length; column++) total += a[row][column] * x[column];
        return total;
    }

    static double objective(double[] x) {
        double total = 0;
        for (double value : x) total += value;
        return total;
    }

    static double[][] transpose(double[][] a) {
        double[][] transposed = new double[a[0].length][a.length];
        for (int row = 0; row < a.length; row++) {
            for (int column = 0; column < a[row].length; column++) transposed[column][row] = a[row][column];
        }
        return transposed;
    }
}

public class Solution {
    private static long state = 0x708ABCDFEDL;

    /** splitmix64 - deterministic PRNG, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    /** Gaussian elimination; null when the system is singular or inconsistent. */
    static double[] solveLinear(double[][] matrix, double[] rhs) {
        int size = rhs.length;
        double[][] augmented = new double[size][size + 1];
        for (int i = 0; i < size; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, size);
            augmented[i][size] = rhs[i];
        }
        for (int column = 0; column < size; column++) {
            int pivot = column;
            for (int row = column + 1; row < size; row++) {
                if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
            }
            if (Math.abs(augmented[pivot][column]) < 1e-9) return null;
            double[] swap = augmented[column];
            augmented[column] = augmented[pivot];
            augmented[pivot] = swap;
            for (int row = 0; row < size; row++) {
                if (row == column) continue;
                double factor = augmented[row][column] / augmented[column][column];
                if (factor == 0) continue;
                for (int k = column; k <= size; k++) augmented[row][k] -= factor * augmented[column][k];
            }
        }
        double[] solution = new double[size];
        for (int i = 0; i < size; i++) solution[i] = augmented[i][size] / augmented[i][i];
        return solution;
    }

    /**
     * Exact optimum of max/min { c.x : coefficients . x <= bounds } by enumerating all vertices: every
     * full-size subset of constraints is forced to equality, solved and, if feasible, ranked.
     */
    static double optimize(double[][] coefficients, double[] bounds, double[] objective,
                           boolean maximize) {
        int variables = objective.length;
        int constraints = coefficients.length;
        double best = maximize ? 0.0 : Double.POSITIVE_INFINITY;
        int[] chosen = new int[variables];
        for (int mask = 0; mask < (1 << constraints); mask++) {
            if (Integer.bitCount(mask) != variables) continue;
            int depth = 0;
            for (int index = 0; index < constraints; index++) {
                if ((mask & (1 << index)) != 0) chosen[depth++] = index;
            }
            double[][] matrix = new double[variables][variables];
            double[] rhs = new double[variables];
            for (int i = 0; i < variables; i++) {
                System.arraycopy(coefficients[chosen[i]], 0, matrix[i], 0, variables);
                rhs[i] = bounds[chosen[i]];
            }
            double[] solution = solveLinear(matrix, rhs);
            if (solution == null) continue;
            boolean feasible = true;
            for (int index = 0; index < constraints && feasible; index++) {
                double value = 0;
                for (int v = 0; v < variables; v++) value += coefficients[index][v] * solution[v];
                if (value > bounds[index] + 1e-7) feasible = false;
            }
            if (!feasible) continue;
            double value = 0;
            for (int v = 0; v < variables; v++) value += objective[v] * solution[v];
            best = maximize ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    /** max { 1^T x : A x <= 1, x >= 0 }. */
    static double exactPacking(double[][] a) {
        int rows = a.length;
        int columns = a[0].length;
        double[][] coefficients = new double[rows + columns][columns];
        double[] bounds = new double[rows + columns];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(a[row], 0, coefficients[row], 0, columns);
            bounds[row] = 1.0;
        }
        for (int column = 0; column < columns; column++) {
            coefficients[rows + column][column] = -1.0;
            bounds[rows + column] = 0.0;
        }
        double[] objective = new double[columns];
        Arrays.fill(objective, 1.0);
        return optimize(coefficients, bounds, objective, true);
    }

    /** min { 1^T x : A x >= 1, x >= 0 }, or +infinity when infeasible. */
    static double exactCovering(double[][] a) {
        int rows = a.length;
        int columns = a[0].length;
        double[][] coefficients = new double[rows + columns][columns];
        double[] bounds = new double[rows + columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) coefficients[row][column] = -a[row][column];
            bounds[row] = -1.0;
        }
        for (int column = 0; column < columns; column++) {
            coefficients[rows + column][column] = -1.0;
            bounds[rows + column] = 0.0;
        }
        double[] objective = new double[columns];
        Arrays.fill(objective, 1.0);
        return optimize(coefficients, bounds, objective, false);
    }

    public static void main(String[] args) {
        int iterations = 2000;
        double epsilon = 0.05;

        // --- hand-checked two by two instance ---------------------------------
        double[][] twoByTwo = {{2, 1}, {1, 2}};
        check(Math.abs(exactPacking(twoByTwo) - 2.0 / 3.0) < 1e-9,
                "packing optimum of 2/3, got " + exactPacking(twoByTwo));
        check(Math.abs(exactCovering(twoByTwo) - 2.0 / 3.0) < 1e-9,
                "covering optimum of 2/3, got " + exactCovering(twoByTwo));
        double[] packingSolution = MultiplicativeWeights.packing(twoByTwo, iterations, epsilon);
        double[] coveringSolution = MultiplicativeWeights.covering(twoByTwo, iterations, epsilon);
        checkFeasiblePacking(twoByTwo, packingSolution, "two by two");
        checkFeasibleCovering(twoByTwo, coveringSolution, "two by two");
        check(MultiplicativeWeights.objective(packingSolution) >= 0.9 * exactPacking(twoByTwo),
                "packing value " + MultiplicativeWeights.objective(packingSolution));
        check(MultiplicativeWeights.objective(coveringSolution) <= 1.1 * exactCovering(twoByTwo),
                "covering cost " + MultiplicativeWeights.objective(coveringSolution));

        // --- one variable, one row: exact in a single step ---------------------
        double[][] single = {{3}};
        double[] onePacking = MultiplicativeWeights.packing(single, 10, epsilon);
        double[] oneCovering = MultiplicativeWeights.covering(single, 10, epsilon);
        check(Math.abs(MultiplicativeWeights.objective(onePacking) - 1.0 / 3.0) < 1e-9,
                "single variable packing is 1/3, got " + MultiplicativeWeights.objective(onePacking));
        check(Math.abs(MultiplicativeWeights.objective(oneCovering) - 1.0 / 3.0) < 1e-9,
                "single variable covering is 1/3, got " + MultiplicativeWeights.objective(oneCovering));

        // --- a column that never helps stays out of the covering solution ------
        double[][] withZeroColumn = {{1, 0, 2}, {1, 0, 1}};
        double[] zeroCovering = MultiplicativeWeights.covering(withZeroColumn, 200, epsilon);
        check(zeroCovering[1] == 0.0, "a column with no coverage must stay at zero");
        checkFeasibleCovering(withZeroColumn, zeroCovering, "zero column covering");
        check(Math.abs(MultiplicativeWeights.objective(zeroCovering) - exactCovering(withZeroColumn)) < 1e-6,
                "covering with a useless column should hit the optimum: "
                        + MultiplicativeWeights.objective(zeroCovering));

        // --- identity matrix: pack and cover both cost exactly the dimension ---
        double[][] identity = {{1, 0}, {0, 1}};
        check(Math.abs(exactPacking(identity) - 2.0) < 1e-9, "identity packs to two");
        check(Math.abs(exactCovering(identity) - 2.0) < 1e-9, "identity covers with cost two");
        double[] identityPacking = MultiplicativeWeights.packing(identity, 200, epsilon);
        double[] identityCovering = MultiplicativeWeights.covering(identity, 200, epsilon);
        checkFeasiblePacking(identity, identityPacking, "identity");
        checkFeasibleCovering(identity, identityCovering, "identity");
        check(Math.abs(MultiplicativeWeights.objective(identityPacking) - 2.0) < 1e-6,
                "identity packing value " + MultiplicativeWeights.objective(identityPacking));
        check(Math.abs(MultiplicativeWeights.objective(identityCovering) - 2.0) < 1e-6,
                "identity covering cost " + MultiplicativeWeights.objective(identityCovering));

        // --- infeasible covering and rejected parameters ------------------------
        double[][] zeroRow = {{1, 0}, {0, 0}};
        check(MultiplicativeWeights.covering(zeroRow, 50, epsilon) == null,
                "a zero row makes covering infeasible");
        check(exactCovering(zeroRow) == Double.POSITIVE_INFINITY,
                "the exact oracle agrees covering is infeasible");
        check(rejects(() -> MultiplicativeWeights.packing(twoByTwo, 0, epsilon)),
                "zero iterations rejected");
        check(rejects(() -> MultiplicativeWeights.packing(twoByTwo, 10, 0.0)),
                "epsilon zero rejected");
        check(rejects(() -> MultiplicativeWeights.packing(twoByTwo, 10, 1.5)),
                "epsilon above one rejected");
        check(rejects(() -> MultiplicativeWeights.covering(twoByTwo, -3, epsilon)),
                "negative iterations rejected");

        // --- strong duality cross-check ---------------------------------------
        check(Math.abs(exactCovering(twoByTwo)
                        - exactPacking(MultiplicativeWeights.transpose(twoByTwo))) < 1e-9,
                "covering(A) must equal packing(A^T)");

        // --- random instances against the exact optimum ------------------------
        double worstPackingRatio = 1.0;
        double worstCoveringRatio = 1.0;
        int compared = 0;
        for (int trial = 0; trial < 30; trial++) {
            int rows = 2 + nextInt(4);
            int columns = 2 + nextInt(3);
            double[][] a = new double[rows][columns];
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) a[row][column] = nextInt(5);
            }
            // keep every column bounded and every row coverable so both LPs are well posed
            for (int column = 0; column < columns; column++) {
                boolean any = false;
                for (int row = 0; row < rows; row++) any |= a[row][column] > 0;
                if (!any) a[nextInt(rows)][column] = 1;
            }
            for (int row = 0; row < rows; row++) {
                boolean any = false;
                for (int column = 0; column < columns; column++) any |= a[row][column] > 0;
                if (!any) a[row][nextInt(columns)] = 1;
            }
            double packingOptimum = exactPacking(a);
            double coveringOptimum = exactCovering(a);
            check(!(coveringOptimum < Double.POSITIVE_INFINITY)
                            || Math.abs(coveringOptimum - exactPacking(MultiplicativeWeights.transpose(a))) < 1e-7,
                    "trial " + trial + " duality failed");
            if (!(coveringOptimum < Double.POSITIVE_INFINITY)) continue;
            compared++;
            double[] packed = MultiplicativeWeights.packing(a, iterations, epsilon);
            double[] covered = MultiplicativeWeights.covering(a, iterations, epsilon);
            checkFeasiblePacking(a, packed, "trial " + trial);
            checkFeasibleCovering(a, covered, "trial " + trial);
            double packingValue = MultiplicativeWeights.objective(packed);
            double coveringCost = MultiplicativeWeights.objective(covered);
            check(packingValue <= packingOptimum + 1e-7, "trial " + trial + " packing value "
                    + packingValue + " above the optimum " + packingOptimum);
            check(coveringCost >= coveringOptimum - 1e-7, "trial " + trial + " covering cost "
                    + coveringCost + " below the optimum " + coveringOptimum);
            if (packingOptimum > 1e-9) worstPackingRatio = Math.min(worstPackingRatio,
                    packingValue / packingOptimum);
            if (coveringOptimum > 1e-9) worstCoveringRatio = Math.min(worstCoveringRatio,
                    coveringOptimum / coveringCost);
        }
        check(compared >= 20, "enough feasible random instances: " + compared);
        check(worstPackingRatio >= 0.9, "packing stayed close to the optimum: " + worstPackingRatio);
        check(worstCoveringRatio >= 0.9, "covering stayed close to the optimum: " + worstCoveringRatio);

        // --- larger instance: feasibility and a simple upper bound -------------
        int rows = 60;
        int columns = 40;
        double[][] large = new double[rows][columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) large[row][column] = nextInt(5);
        }
        for (int column = 0; column < columns; column++) {
            boolean any = false;
            for (int row = 0; row < rows; row++) any |= large[row][column] > 0;
            if (!any) large[0][column] = 1;
        }
        long started = System.nanoTime();
        double[] largePacking = MultiplicativeWeights.packing(large, 400, epsilon);
        double[] largeCovering = MultiplicativeWeights.covering(large, 400, epsilon);
        long elapsedMillis = (System.nanoTime() - started) / 1_000_000;
        checkFeasiblePacking(large, largePacking, "large instance");
        checkFeasibleCovering(large, largeCovering, "large instance");
        double trivialBound = 0;
        for (int column = 0; column < columns; column++) {
            double best = Double.MAX_VALUE;
            for (int row = 0; row < rows; row++) {
                if (large[row][column] > 0) best = Math.min(best, 1.0 / large[row][column]);
            }
            if (best < Double.MAX_VALUE) trivialBound += best;
        }
        check(MultiplicativeWeights.objective(largePacking) <= trivialBound + 1e-6,
                "packing value must respect the per-column bound: "
                        + MultiplicativeWeights.objective(largePacking) + " vs " + trivialBound);
        check(elapsedMillis < 2000, "800 iterations on a 60x40 matrix must be fast: "
                + elapsedMillis + " ms");
        double[] repeated = MultiplicativeWeights.packing(twoByTwo, iterations, epsilon);
        check(Arrays.equals(repeated, packingSolution), "the same instance must give the same answer");

        System.out.println("PASS");
    }

    static void checkFeasiblePacking(double[][] a, double[] x, String label) {
        for (double value : x) check(value >= -1e-9, label + ": negative variable");
        for (int row = 0; row < a.length; row++) {
            check(MultiplicativeWeights.usage(a, row, x) <= 1.0 + 1e-7,
                    label + ": row " + row + " exceeded its budget");
        }
    }

    static void checkFeasibleCovering(double[][] a, double[] x, String label) {
        if (x == null) throw new AssertionError("FAILED: " + label + ": no covering solution");
        for (double value : x) check(value >= -1e-9, label + ": negative variable");
        for (int row = 0; row < a.length; row++) {
            check(MultiplicativeWeights.usage(a, row, x) >= 1.0 - 1e-7,
                    label + ": row " + row + " is not covered");
        }
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
