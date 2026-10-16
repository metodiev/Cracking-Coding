/** Task 562 - Implement mixed integer programming small model using branch-and-bound outline. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A working branch-and-bound solver for small all-integer models, built on a real LP relaxation rather
 * than a heuristic bound. The relaxation is a dense two-phase simplex tableau for max{c^T x : A x <= b,
 * x >= 0}: rows with negative right-hand side are negated and receive an artificial variable, phase 1
 * minimises the sum of the artificials (which both finds the feasible basis and certifies infeasibility),
 * and phase 2 optimises the original objective; both phases use Bland's rule (smallest eligible entering
 * index, smallest basic index among tied minimum ratios), which rules out cycling. Branch and bound then
 * solves the relaxation at every node, prunes when the bound cannot beat the incumbent (within 1e-9),
 * updates the incumbent when the relaxation is integral, and otherwise branches on the most fractional
 * variable into x_j <= floor(v) and x_j >= ceil(v) (the latter written as -x_j <= -ceil(v)). The
 * implementation is deterministic: node order is fixed, ties are broken by index, and no wall-clock
 * quantity is used.
 *
 * Documented checks. The LP relaxation is verified against a completely independent method — enumerate
 * every vertex of the feasible region by solving all C(m+n, n) subsystems of the constraint rows plus the
 * non-negativity rows with Gaussian elimination — on hand instances and on seeded random small models; the
 * hand instance max 2x1 + 3x2 s.t. x1 + x2 <= 4, x1 + 3x2 <= 6 is asserted to give 9 at the interior
 * vertex (3,1) and the same constraints with objective 3x1 + 2x2 to give 12 at (4,0), and a knapsack
 * relaxation is asserted to equal the classical fractional-greedy optimum. The integer optima on
 * five seeded 12-item knapsacks are compared with two independent methods, brute force over all 2^12
 * subsets and a 0/1 knapsack DP over the capacity, and the returned solution is re-checked for integrality
 * and feasibility; objective values are compared with a 1e-6 tolerance because a simplex tableau works in
 * floating point, so an optimum such as 31 can come back as 30.999999999999996. The relaxation bound is
 * asserted to dominate the integer optimum on every instance, and
 * a documented case where the two strictly differ (LP bound 13.5 against integer optimum 12, root solution
 * fractional, more than one node explored) is asserted explicitly, together with the general fact that the
 * node count exceeds one whenever the root relaxation is fractional. An infeasible model (x <= 0.5 and
 * x >= 1) must be detected by phase 1, and an empty/integer-relaxation model must terminate at the root
 * with exactly one node. Everything is re-run to confirm determinism. Only bounded models are exercised,
 * so unboundedness is detected in the simplex but is not part of the claimed coverage.
 */
import java.util.*;

public class Solution {

    static final double EPS = 1e-9;

    static final class Lp {
        boolean feasible;
        boolean unbounded;
        double objective;
        double[] x;
        int iterations;
    }

    /** max c^T x subject to A x <= b, x >= 0, with b of arbitrary sign (two-phase simplex). */
    static Lp solveLp(double[][] a, double[] b, double[] c) {
        int m = a.length;
        int n = c.length;
        int slackStart = n;
        int artStart = n + m;
        int cols = n + 2 * m;
        double[][] t = new double[m + 1][cols + 1];
        int[] basis = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) t[i][j] = a[i][j];
            if (b[i] >= 0.0) {
                t[i][slackStart + i] = 1.0;
                t[i][cols] = b[i];
                basis[i] = slackStart + i;
            } else {
                for (int j = 0; j < n; j++) t[i][j] = -t[i][j];
                t[i][slackStart + i] = -1.0;
                t[i][artStart + i] = 1.0;
                t[i][cols] = -b[i];
                basis[i] = artStart + i;
            }
        }
        int zRow = m;
        Lp result = new Lp();
        // Phase 1: minimise the sum of artificial variables.
        for (int j = 0; j <= cols; j++) t[zRow][j] = 0.0;
        for (int i = 0; i < m; i++) {
            if (basis[i] < artStart) continue;
            for (int j = 0; j <= cols; j++) t[zRow][j] -= t[i][j];
        }
        result.iterations += simplex(t, basis, cols, m, artStart, false);
        double artificialSum = 0.0;
        for (int i = 0; i < m; i++) if (basis[i] >= artStart) artificialSum += t[i][cols];
        if (artificialSum > 1e-7) {
            result.feasible = false;
            return result;
        }
        // Drive artificials out of the basis where possible; rows that cannot be cleaned are redundant.
        for (int i = 0; i < m; i++) {
            if (basis[i] < artStart) continue;
            for (int j = 0; j < artStart; j++) {
                if (Math.abs(t[i][j]) > 1e-9) {
                    pivot(t, i, j, m, cols);
                    basis[i] = j;
                    break;
                }
            }
        }
        // Phase 2: the original objective.
        for (int j = 0; j <= cols; j++) t[zRow][j] = j < n ? c[j] : 0.0;
        for (int i = 0; i < m; i++) {
            double cost = basis[i] < n ? c[basis[i]] : 0.0;
            if (cost == 0.0) continue;
            for (int j = 0; j <= cols; j++) t[zRow][j] -= cost * t[i][j];
        }
        int phase2 = simplex(t, basis, cols, m, artStart, true);
        result.iterations += phase2;
        result.feasible = true;
        result.unbounded = phase2 < 0;
        result.x = new double[n];
        for (int i = 0; i < m; i++) {
            if (basis[i] < n) result.x[basis[i]] = t[i][cols];
        }
        // The tableau keeps the reduced-cost row with the negated objective in its right-hand column.
        result.objective = -t[zRow][cols];
        return result;
    }

    /** Returns the number of pivots, or -1 when the problem is unbounded. */
    private static int simplex(double[][] t, int[] basis, int cols, int rows, int artStart,
                              boolean maximize) {
        int zRow = rows;
        int pivots = 0;
        int limit = 20000;
        while (true) {
            int entering = -1;
            for (int j = 0; j < artStart; j++) {
                double rc = t[zRow][j];
                if (maximize ? rc > EPS : rc < -EPS) {
                    entering = j;
                    break;
                }
            }
            if (entering < 0) return pivots;
            int leaving = -1;
            double bestRatio = Double.POSITIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                if (t[i][entering] <= EPS) continue;
                double ratio = t[i][cols] / t[i][entering];
                if (ratio < bestRatio - 1e-12
                        || (Math.abs(ratio - bestRatio) <= 1e-12
                        && (leaving < 0 || basis[i] < basis[leaving]))) {
                    bestRatio = ratio;
                    leaving = i;
                }
            }
            if (leaving < 0) return -1;
            pivot(t, leaving, entering, rows, cols);
            basis[leaving] = entering;
            if (++pivots > limit) throw new RuntimeException("FAILED: simplex pivot limit exceeded");
        }
    }

    private static void pivot(double[][] t, int row, int col, int rows, int cols) {
        double p = t[row][col];
        for (int j = 0; j <= cols; j++) t[row][j] /= p;
        for (int i = 0; i <= rows; i++) {
            if (i == row) continue;
            double factor = t[i][col];
            if (factor == 0.0) continue;
            for (int j = 0; j <= cols; j++) t[i][j] -= factor * t[row][j];
        }
    }

    static final class Milp {
        boolean feasible;
        double objective;
        double[] x;
        int nodes;
        double rootBound;
        boolean rootFractional;
        boolean rootFeasible;
    }

    /** Branch and bound over x integer, with explicit upper bounds ub[j] >= 0 for every variable. */
    static Milp branchAndBound(double[][] a, double[] b, double[] c, double[] ub) {
        Milp result = new Milp();
        int n = c.length;
        Deque<List<double[]>> stack = new ArrayDeque<List<double[]>>();
        stack.push(new ArrayList<double[]>());
        double incumbent = Double.NEGATIVE_INFINITY;
        double[] incumbentX = null;
        boolean first = true;
        while (!stack.isEmpty()) {
            List<double[]> extra = stack.pop();
            int m = a.length + n + extra.size();
            double[][] rows = new double[m][];
            double[] rhs = new double[m];
            int r = 0;
            for (int i = 0; i < a.length; i++) {
                rows[r] = a[i].clone();
                rhs[r] = b[i];
                r++;
            }
            for (int j = 0; j < n; j++) {
                double[] row = new double[n];
                row[j] = 1.0;
                rows[r] = row;
                rhs[r] = ub[j];
                r++;
            }
            for (double[] row : extra) {
                rows[r] = row.clone();
                rhs[r] = row[n];
                r++;
            }
            Lp lp = solveLp(rows, rhs, c);
            result.nodes++;
            if (first) {
                first = false;
                result.rootFeasible = lp.feasible;
                result.rootBound = lp.feasible ? lp.objective : Double.POSITIVE_INFINITY;
            }
            if (!lp.feasible) continue;
            if (lp.objective <= incumbent + 1e-9) continue;
            int branch = -1;
            double worstFraction = 1e-6;
            for (int j = 0; j < n; j++) {
                double value = lp.x[j];
                double fraction = Math.abs(value - Math.rint(value));
                if (fraction > worstFraction) {
                    worstFraction = fraction;
                    branch = j;
                }
            }
            if (branch < 0) {
                incumbent = lp.objective;
                incumbentX = lp.x.clone();
                continue;
            }
            if (result.nodes == 1) result.rootFractional = true;
            double floorValue = Math.floor(lp.x[branch]);
            double ceilValue = Math.ceil(lp.x[branch]);
            double[] down = new double[n + 1];
            down[branch] = 1.0;
            down[n] = floorValue;
            double[] up = new double[n + 1];
            up[branch] = -1.0;
            up[n] = -ceilValue;
            List<double[]> upChild = new ArrayList<double[]>(extra);
            upChild.add(up);
            List<double[]> downChild = new ArrayList<double[]>(extra);
            downChild.add(down);
            stack.push(upChild);
            stack.push(downChild);
        }
        if (incumbentX == null) {
            result.feasible = false;
            return result;
        }
        result.feasible = true;
        result.objective = incumbent;
        result.x = incumbentX;
        return result;
    }

    /** Brute force over every integral point in the box [0, ub]. */
    static double bruteMilp(double[][] a, double[] b, double[] c, double[] ub) {
        int n = c.length;
        int[] point = new int[n];
        double best = Double.NEGATIVE_INFINITY;
        double[] current = new double[n];
        int[] limits = new int[n];
        for (int j = 0; j < n; j++) limits[j] = (int) Math.floor(ub[j] + 1e-9);
        long combinations = 1;
        for (int limit : limits) combinations *= (limit + 1);
        for (long code = 0; code < combinations; code++) {
            long rest = code;
            for (int j = 0; j < n; j++) {
                point[j] = (int) (rest % (limits[j] + 1));
                rest /= (limits[j] + 1);
                current[j] = point[j];
            }
            boolean feasible = true;
            for (int i = 0; i < a.length && feasible; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += a[i][j] * current[j];
                if (sum > b[i] + 1e-9) feasible = false;
            }
            if (!feasible) continue;
            double value = 0.0;
            for (int j = 0; j < n; j++) value += c[j] * current[j];
            if (value > best) best = value;
        }
        return best;
    }

    /** 0/1 knapsack by dynamic programming: an independent check on the branch-and-bound optimum. */
    static int knapsackDp(int[] weights, int[] values, int capacity) {
        int[] best = new int[capacity + 1];
        for (int i = 0; i < weights.length; i++) {
            for (int w = capacity; w >= weights[i]; w--) {
                best[w] = Math.max(best[w], best[w - weights[i]] + values[i]);
            }
        }
        return best[capacity];
    }

    static long knapsackBruteForce(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        long best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int weight = 0;
            int value = 0;
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    weight += weights[i];
                    value += values[i];
                }
            }
            if (weight <= capacity) best = Math.max(best, value);
        }
        return best;
    }

    /** Fractional-greedy optimum of the knapsack relaxation, sorted by value/weight ratio. */
    static double fractionalGreedy(int[] weights, int[] values, int capacity) {
        Integer[] order = new Integer[weights.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, new Comparator<Integer>() {
            public int compare(Integer x, Integer y) {
                double rx = (double) values[x] / weights[x];
                double ry = (double) values[y] / weights[y];
                return Double.compare(ry, rx);
            }
        });
        double total = 0.0;
        int remaining = capacity;
        for (int index : order) {
            if (remaining <= 0) break;
            int take = Math.min(remaining, weights[index]);
            total += (double) take * values[index] / weights[index];
            remaining -= take;
        }
        return total;
    }

    /** Best vertex of the LP by enumerating all subsets of n constraints; independent of the simplex. */
    static double bestVertexValue(double[][] a, double[] b, double[] c) {
        int m = a.length;
        int n = c.length;
        double[][] all = new double[m + n][];
        double[] rhs = new double[m + n];
        for (int i = 0; i < m; i++) {
            all[i] = a[i].clone();
            rhs[i] = b[i];
        }
        for (int j = 0; j < n; j++) {
            double[] row = new double[n];
            row[j] = -1.0;
            all[m + j] = row;
            rhs[m + j] = 0.0;
        }
        int[] pick = new int[n];
        return bestVertexRecursive(all, rhs, c, 0, 0, pick);
    }

    private static double bestVertexRecursive(double[][] all, double[] rhs, double[] c, int start,
                                              int depth, int[] pick) {
        int n = c.length;
        if (depth == n) {
            double[][] system = new double[n][];
            double[] targets = new double[n];
            for (int i = 0; i < n; i++) {
                system[i] = all[pick[i]];
                targets[i] = rhs[pick[i]];
            }
            double[] solution = solveSystem(system, targets);
            if (solution == null) return Double.NEGATIVE_INFINITY;
            for (int r = 0; r < all.length; r++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += all[r][j] * solution[j];
                if (sum > rhs[r] + 1e-7) return Double.NEGATIVE_INFINITY;
            }
            double value = 0.0;
            for (int j = 0; j < n; j++) value += c[j] * solution[j];
            return value;
        }
        double best = Double.NEGATIVE_INFINITY;
        for (int i = start; i < all.length; i++) {
            pick[depth] = i;
            best = Math.max(best, bestVertexRecursive(all, rhs, c, i + 1, depth + 1, pick));
        }
        return best;
    }

    static double[] solveSystem(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] work = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            work[i] = Arrays.copyOf(matrix[i], n + 1);
            work[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = -1;
            for (int r = col; r < n; r++) {
                if (Math.abs(work[r][col]) > 1e-10) {
                    pivot = r;
                    break;
                }
            }
            if (pivot < 0) return null;
            double[] tmp = work[col];
            work[col] = work[pivot];
            work[pivot] = tmp;
            double pivotValue = work[col][col];
            for (int j = col; j <= n; j++) work[col][j] /= pivotValue;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double factor = work[r][col];
                for (int j = col; j <= n; j++) work[r][j] -= factor * work[col][j];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = work[i][n];
        return out;
    }

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

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    public static void main(String[] args) {
        // --- The hand LP and the vertex-enumeration oracle.
        double[][] hand = {{1, 1}, {1, 3}};
        double[] handRhs = {4, 6};
        double[] handCost = {2, 3};
        Lp handLp = solveLp(hand, handRhs, handCost);
        check(handLp.feasible && !handLp.unbounded, "the hand LP is solvable");
        check(Math.abs(handLp.objective - 9.0) <= 1e-9, "the hand LP optimum is 9, got "
                + handLp.objective);
        check(Math.abs(handLp.x[0] - 3.0) <= 1e-9 && Math.abs(handLp.x[1] - 1.0) <= 1e-9,
                "attained at (3, 1)");
        check(Math.abs(bestVertexValue(hand, handRhs, handCost) - 9.0) <= 1e-7,
                "the vertex enumeration agrees");
        Lp corner = solveLp(hand, handRhs, new double[]{3, 2});
        check(Math.abs(corner.objective - 12.0) <= 1e-9 && Math.abs(corner.x[0] - 4.0) <= 1e-9,
                "maximising 3x1 + 2x2 instead puts the optimum at the vertex (4, 0) with value 12");

        // --- Random small LPs against the vertex oracle.
        Rng rng = new Rng(20240562L);
        for (int t = 0; t < 40; t++) {
            int n = 2 + (t % 2);
            int m = 3 + (t % 2);
            double[][] a = new double[m][n];
            double[] b = new double[m];
            double[] c = new double[n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) a[i][j] = 1 + rng.nextInt(9);
                b[i] = 5 + rng.nextInt(30);
            }
            for (int j = 0; j < n; j++) c[j] = 1 + rng.nextInt(12);
            Lp lp = solveLp(a, b, c);
            double oracle = bestVertexValue(a, b, c);
            check(lp.feasible && !lp.unbounded, "random LP " + t + " is solvable");
            check(Math.abs(lp.objective - oracle) <= 1e-6, "simplex " + lp.objective
                    + " equals the best vertex " + oracle + " on LP " + t);
            for (int i = 0; i < m; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += a[i][j] * lp.x[j];
                check(sum <= b[i] + 1e-7, "the returned point satisfies constraint " + i);
            }
            for (int j = 0; j < n; j++) check(lp.x[j] >= -1e-9, "the returned point is non-negative");
        }

        // --- Knapsack: branch and bound against brute force and DP.
        for (int instance = 0; instance < 5; instance++) {
            int n = 12;
            int[] weights = new int[n];
            int[] values = new int[n];
            for (int i = 0; i < n; i++) {
                weights[i] = 1 + rng.nextInt(20);
                values[i] = 1 + rng.nextInt(30);
            }
            int capacity = 30 + rng.nextInt(20);
            double[][] a = new double[1][n];
            double[] b = new double[1];
            for (int j = 0; j < n; j++) a[0][j] = weights[j];
            b[0] = capacity;
            double[] c = new double[n];
            double[] ub = new double[n];
            for (int j = 0; j < n; j++) {
                c[j] = values[j];
                ub[j] = 1.0;
            }
            Milp solution = branchAndBound(a, b, c, ub);
            long oracle = knapsackBruteForce(weights, values, capacity);
            check(solution.feasible, "knapsack " + instance + " is feasible");
            check(Math.abs(solution.objective - oracle) <= 1e-6, "knapsack " + instance + " optimum "
                    + solution.objective + " equals brute force " + oracle);
            check(knapsackDp(weights, values, capacity) == oracle,
                    "the DP agrees with brute force on knapsack " + instance);
            check(solution.rootBound >= solution.objective - 1e-9,
                    "the root relaxation bound dominates the integer optimum (knapsack " + instance + ")");
            check(Math.abs(solution.rootBound - fractionalGreedy(weights, values, capacity)) <= 1e-7,
                    "the root bound equals the fractional-greedy value (knapsack " + instance + ")");
            double weightUsed = 0.0;
            for (int j = 0; j < n; j++) {
                check(Math.abs(solution.x[j] - Math.rint(solution.x[j])) <= 1e-9,
                        "the reported solution is integral (knapsack " + instance + ")");
                check(solution.x[j] >= -1e-9 && solution.x[j] <= 1 + 1e-9,
                        "and respects the 0/1 bounds (knapsack " + instance + ")");
                weightUsed += weights[j] * Math.rint(solution.x[j]);
            }
            check(weightUsed <= capacity + 1e-9, "and respects the capacity (knapsack " + instance + ")");
            if (solution.rootFractional) {
                check(solution.nodes > 1, "a fractional root forces branching (knapsack " + instance
                        + " explored " + solution.nodes + " nodes)");
            }
            check(solution.nodes <= 4096, "the search stays small: " + solution.nodes + " nodes");
        }

        // --- A documented LP-versus-IP gap.
        double[][] gap = {{1, 1}, {1, 3}};
        double[] gapRhs = {4.5, 6};
        double[] gapCost = {3, 2};
        Milp gapSolution = branchAndBound(gap, gapRhs, gapCost, new double[]{10, 10});
        Lp gapLp = solveLp(gap, gapRhs, gapCost);
        check(Math.abs(gapLp.objective - 13.5) <= 1e-9, "the fractional optimum is 13.5, got "
                + gapLp.objective);
        check(Math.abs(gapSolution.objective - 12.0) <= 1e-9, "the integer optimum is 12, got "
                + gapSolution.objective);
        check(gapSolution.rootFractional, "the root relaxation is fractional");
        check(gapSolution.nodes > 1, "so branch and bound explores " + gapSolution.nodes + " nodes");
        check(Math.abs(bruteMilp(gap, gapRhs, gapCost, new double[]{10, 10}) - 12.0) <= 1e-9,
                "the integral brute force agrees");

        // --- Infeasibility is detected by phase 1.
        double[][] infeasible = {{1}, {-1}};
        double[] infeasibleRhs = {0.5, -1};
        Lp bad = solveLp(infeasible, infeasibleRhs, new double[]{1});
        check(!bad.feasible, "x <= 0.5 and x >= 1 is detected as infeasible");
        Milp badMilp = branchAndBound(infeasible, infeasibleRhs, new double[]{1}, new double[]{5});
        check(!badMilp.feasible && badMilp.nodes == 1, "the MILP reports infeasible after one node");

        // --- The pure-integer boundary: no branching when the relaxation is integral.
        double[][] integral = {{1, 1}};
        Milp integralSolution = branchAndBound(integral, new double[]{2}, handCost, new double[]{2, 2});
        check(integralSolution.feasible && Math.abs(integralSolution.objective - 6.0) <= 1e-9,
                "x1 + x2 <= 2 maximising 3x1 + 2x2 gives 6");
        check(integralSolution.nodes == 1, "an integral relaxation needs exactly one node");
        check(!integralSolution.rootFractional, "and the root is not fractional");

        // --- Determinism.
        int[] weights = {3, 4, 5, 8, 9, 2, 7, 6};
        int[] values = {4, 5, 8, 9, 10, 3, 7, 6};
        double[][] a = new double[1][8];
        double[] b = new double[]{25};
        double[] c = new double[8];
        for (int j = 0; j < 8; j++) {
            a[0][j] = weights[j];
            c[j] = values[j];
        }
        double[] ub = new double[8];
        Arrays.fill(ub, 1.0);
        Milp first = branchAndBound(a, b, c, ub);
        Milp second = branchAndBound(a, b, c, ub);
        check(first.feasible && second.feasible && first.objective == second.objective
                && first.nodes == second.nodes, "deterministic re-run");
        check(Math.abs(first.objective - knapsackBruteForce(weights, values, 25)) <= 1e-6,
                "and the optimum matches the brute force on the last instance (" + first.objective + ")");
        check(Math.abs(first.objective - Math.rint(first.objective)) <= 1e-6,
                "the objective is integral up to floating-point rounding: " + first.objective);

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
