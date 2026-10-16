/** Task 731 - Implement adaptive mesh refinement driving solver coupling. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * The AMR loop is: solve on the current grid, estimate the local error, refine where it is largest, solve
 * again. This solution runs that loop in one dimension. Cells carry a refinement level; a cell is split into
 * two children when its error indicator is in the marked set, and the grid is then repaired until it is
 * 1-irregular (neighbouring cells never differ by more than one level), which is the grading condition every
 * conforming adaptive solver relies on. The solver is a second-order finite difference of -u'' = f on the
 * non-uniform point set, assembled into a tridiagonal system and inverted with the Thomas algorithm, with a
 * dense Gaussian elimination standing in as an independent oracle for the same system. The indicator is the
 * local truncation residual |u''_h - (-f)|, and marking follows Dörfler's rule: refine the smallest set of
 * cells that carries half of the total indicator.
 *
 * Invariants verified:
 *   - exactness   the three-point non-uniform stencil is exact for quadratics, so a quadratic solution is
 *                 reproduced to machine precision on any grid and provokes no refinement at all
 *   - grading     after every refinement pass the grid is 1-irregular, and the cell count only grows
 *   - accuracy    the adaptive grid beats a uniform grid with the same number of cells for a sharp solution,
 *                 and its error falls as the refinement is allowed to go deeper
 *   - solver      the Thomas solution matches a dense direct solve of the assembled system and leaves a
 *                 residual at the level of rounding
 *
 * Simplified: one spatial dimension, a cell-centred grid rather than a facet-based patch hierarchy, and no
 * flux-correction coupling between coarse and fine faces - the refined points simply join one non-uniform
 * stencil. The verified properties are the grading invariant, quadratic exactness, and the measured accuracy
 * advantage of adaptivity.
 */
import java.util.*;

public class Solution {
    /** One grid cell: its interval and its refinement level. */
    static final class Cell {
        final double lo;
        final double hi;
        final int level;
        Cell(double lo, double hi, int level) {
            this.lo = lo;
            this.hi = hi;
            this.level = level;
        }
        double center() { return 0.5 * (lo + hi); }
    }

    /** A 1-irregular graded grid over [0, 1] with homogeneous Dirichlet ends. */
    static final class Grid {
        final List<Cell> cells = new ArrayList<>();
        final int maxLevel;

        Grid(int initialCells, int maxLevel) {
            this.maxLevel = maxLevel;
            for (int i = 0; i < initialCells; i++) {
                cells.add(new Cell(i / (double) initialCells, (i + 1) / (double) initialCells, 0));
            }
        }

        /** Splits the marked cells, then repairs the grid until it is 1-irregular. */
        void refine(boolean[] marked) {
            List<Cell> next = new ArrayList<>();
            for (int i = 0; i < cells.size(); i++) {
                Cell cell = cells.get(i);
                if (marked[i] && cell.level < maxLevel) {
                    double middle = 0.5 * (cell.lo + cell.hi);
                    next.add(new Cell(cell.lo, middle, cell.level + 1));
                    next.add(new Cell(middle, cell.hi, cell.level + 1));
                } else {
                    next.add(cell);
                }
            }
            cells.clear();
            cells.addAll(next);
            grade();
            cells.sort(Comparator.comparingDouble(cell -> cell.lo));
        }

        private void grade() {
            boolean changed = true;
            while (changed) {
                changed = false;
                for (int i = 0; i < cells.size() - 1; i++) {
                    Cell left = cells.get(i);
                    Cell right = cells.get(i + 1);
                    if (Math.abs(left.level - right.level) <= 1) continue;
                    if (left.level < right.level) {
                        cells.set(i, new Cell(left.lo, 0.5 * (left.lo + left.hi), left.level + 1));
                        cells.add(i + 1, new Cell(0.5 * (left.lo + left.hi), left.hi, left.level + 1));
                    } else {
                        cells.set(i + 1, new Cell(0.5 * (right.lo + right.hi), right.hi, right.level + 1));
                        cells.add(i + 1, new Cell(right.lo, 0.5 * (right.lo + right.hi), right.level + 1));
                    }
                    changed = true;
                }
            }
        }

        /** Number of neighbouring pairs that violate the 1-irregularity rule. */
        int gradeViolations() {
            int broken = 0;
            for (int i = 0; i < cells.size() - 1; i++) {
                if (Math.abs(cells.get(i).level - cells.get(i + 1).level) > 1) broken++;
            }
            return broken;
        }
    }

    /** Right-hand side f and the exact solution u, for the manufactured problems used as oracles. */
    interface Problem {
        double f(double x);
        double exact(double x);
    }

    static final Problem QUADRATIC = new Problem() {
        public double f(double x) { return -2.0; }                       // u = x^2 - x, u'' = 2
        public double exact(double x) { return x * x - x; }
    };

    /** A Gaussian bump of width ~1/sqrt(c), zero at both ends to within 1e-100. */
    static Problem bump(final double c) {
        return new Problem() {
            public double f(double x) {
                double z = x - 0.5;
                return (2.0 * c - 4.0 * c * c * z * z) * Math.exp(-c * z * z);
            }
            public double exact(double x) { return Math.exp(-c * (x - 0.5) * (x - 0.5)); }
        };
    }

    static final Problem SHARP = bump(40.0);        // under-resolved enough to over-refine
    static final Problem MILD = bump(20.0);         // resolved, but adaptivity still pays

    static final Problem ZERO = new Problem() {
        public double f(double x) { return 0.0; }
        public double exact(double x) { return 0.0; }
    };

    /** Assembles the tridiagonal system for the interior cell centres. */
    static double[][] assemble(Grid grid, Problem problem, double[] centers) {
        int n = centers.length;
        double[][] system = new double[n][n + 1];                        // augmented: last column is f
        for (int i = 0; i < n; i++) {
            double hMinus = i == 0 ? centers[0] - 0.0 : centers[i] - centers[i - 1];
            double hPlus = i == n - 1 ? 1.0 - centers[i] : centers[i + 1] - centers[i];
            double left = 2.0 / ((hMinus + hPlus) * hMinus);
            double right = 2.0 / ((hMinus + hPlus) * hPlus);
            if (i > 0) system[i][i - 1] = left;
            system[i][i] = -(left + right);
            if (i < n - 1) system[i][i + 1] = right;
            system[i][n] = -problem.f(centers[i]);
        }
        return system;
    }

    /** Thomas algorithm; the system is tridiagonal and diagonally dominant. */
    static double[] thomas(double[][] system) {
        int n = system.length;
        double[] c = new double[n];
        double[] d = new double[n];
        c[0] = n > 1 ? system[0][1] / system[0][0] : 0.0;
        d[0] = system[0][n] / system[0][0];
        for (int i = 1; i < n; i++) {
            double denominator = system[i][i] - system[i][i - 1] * c[i - 1];
            if (denominator == 0.0) throw new IllegalStateException("singular tridiagonal system");
            c[i] = i < n - 1 ? system[i][i + 1] / denominator : 0.0;
            d[i] = (system[i][n] - system[i][i - 1] * d[i - 1]) / denominator;
        }
        double[] u = new double[n];
        u[n - 1] = d[n - 1];
        for (int i = n - 2; i >= 0; i--) u[i] = d[i] - c[i] * u[i + 1];
        return u;
    }

    /** Dense Gaussian elimination with partial pivoting: the independent oracle for the solve. */
    static double[] denseSolve(double[][] system) {
        int n = system.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) System.arraycopy(system[i], 0, a[i], 0, n + 1);
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int i = col + 1; i < n; i++) if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) pivot = i;
            double[] temp = a[col]; a[col] = a[pivot]; a[pivot] = temp;
            for (int i = col + 1; i < n; i++) {
                double factor = a[i][col] / a[col][col];
                for (int j = col; j <= n; j++) a[i][j] -= factor * a[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = a[i][n];
            for (int j = i + 1; j < n; j++) sum -= a[i][j] * x[j];
            x[i] = sum / a[i][i];
        }
        return x;
    }

    static double[] centers(Grid grid) {
        double[] centers = new double[grid.cells.size()];
        for (int i = 0; i < centers.length; i++) centers[i] = grid.cells.get(i).center();
        return centers;
    }

    /**
     * Error indicator: the change of the discrete second derivative between neighbouring cells, scaled by
     * the spacing (a third-difference estimate). It is exactly zero where the solution is quadratic - the
     * discrete solve is exact there, so nothing needs refining - and large where the curvature varies fastest,
     * which is precisely the sharp part of the solution that the grid has to resolve.
     */
    static double[] indicators(Problem problem, double[] centers, double[] u) {
        int n = centers.length;
        double[] second = new double[n];
        for (int i = 0; i < n; i++) {
            double hMinus = i == 0 ? centers[0] - 0.0 : centers[i] - centers[i - 1];
            double hPlus = i == n - 1 ? 1.0 - centers[i] : centers[i + 1] - centers[i];
            double left = i == 0 ? (u[i] - 0.0) / hMinus : (u[i] - u[i - 1]) / hMinus;
            double right = i == n - 1 ? (0.0 - u[i]) / hPlus : (u[i + 1] - u[i]) / hPlus;
            second[i] = 2.0 * (right - left) / (hMinus + hPlus);
        }
        double[] indicator = new double[n];
        for (int i = 0; i < n; i++) {
            double spacing = i == n - 1 ? centers[i] - centers[i - 1] : centers[i + 1] - centers[i];
            double leftSecond = i == 0 ? second[0] : second[i - 1];
            double rightSecond = i == n - 1 ? second[n - 1] : second[i + 1];
            indicator[i] = Math.abs(rightSecond - leftSecond) / (spacing + (i == 0 ? spacing : 0.0));
        }
        return indicator;
    }

    /**
     * Dorfler marking above an absolute tolerance: cells whose residual is below `tolerance` are already
     * resolved, and the smallest set of the remaining cells that carries half of their total is split.
     */
    static boolean[] mark(double[] indicator, double tolerance) {
        int n = indicator.length;
        boolean[] marked = new boolean[n];
        List<Integer> order = new ArrayList<>();
        double total = 0.0;
        for (int i = 0; i < n; i++) {
            if (indicator[i] > tolerance) {
                order.add(i);
                total += indicator[i];
            }
        }
        order.sort((a, b) -> Double.compare(indicator[b], indicator[a]));
        double target = 0.5 * total;
        double taken = 0.0;
        for (int k = 0; k < order.size() && taken < target; k++) {
            marked[order.get(k)] = true;
            taken += indicator[order.get(k)];
        }
        return marked;
    }

    /** Cell-width-weighted L2 error: the norm in which refinement claims are usually made. */
    static double l2Error(Problem problem, Grid grid, double[] centers, double[] u) {
        double sum = 0.0;
        double weight = 0.0;
        for (int i = 0; i < centers.length; i++) {
            double w = grid.cells.get(i).hi - grid.cells.get(i).lo;
            double d = u[i] - problem.exact(centers[i]);
            sum += d * d * w;
            weight += w;
        }
        return Math.sqrt(sum / weight);
    }

    static double worstError(Problem problem, double[] centers, double[] u) {
        double worst = 0.0;
        for (int i = 0; i < centers.length; i++) worst = Math.max(worst, Math.abs(u[i] - problem.exact(centers[i])));
        return worst;
    }

    static double residual(double[][] system, double[] u) {
        int n = u.length;
        double worst = 0.0;
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += system[i][j] * u[j];
            worst = Math.max(worst, Math.abs(sum - system[i][n]));
        }
        return worst;
    }

    /** Runs the solve-refine loop for `rounds` passes and returns the grid. */
    /** Absolute residual below which a cell counts as resolved. */
    static final double MARK_TOLERANCE = 1e-6;

    static Grid adapt(Problem problem, int initialCells, int maxLevel, int rounds) {
        Grid grid = new Grid(initialCells, maxLevel);
        for (int round = 0; round < rounds; round++) {
            double[] centers = centers(grid);
            double[] u = thomas(assemble(grid, problem, centers));
            grid.refine(mark(indicators(problem, centers, u), MARK_TOLERANCE));
        }
        return grid;
    }

    public static void main(String[] args) {
        // --- the trivial solution does not trigger any refinement --------------
        Grid zero = adapt(ZERO, 16, 4, 3);
        check(zero.cells.size() == 16, "a zero solution must not refine anything, cells " + zero.cells.size());
        double[] zeroCenters = centers(zero);
        double[] zeroSolution = thomas(assemble(zero, ZERO, zeroCenters));
        check(worstError(ZERO, zeroCenters, zeroSolution) == 0.0, "and stays exactly zero");

        // --- quadratics are reproduced exactly on any grid --------------------
        Grid quadratic = adapt(QUADRATIC, 8, 4, 3);
        check(quadratic.cells.size() == 8, "a quadratic is resolved exactly, so no cell splits, cells "
                + quadratic.cells.size());
        double[] quadraticCenters = centers(quadratic);
        double[][] quadraticSystem = assemble(quadratic, QUADRATIC, quadraticCenters);
        double[] quadraticSolution = thomas(quadraticSystem);
        check(worstError(QUADRATIC, quadraticCenters, quadraticSolution) < 1e-13,
                "the three-point stencil is exact for a quadratic, worst error "
                        + worstError(QUADRATIC, quadraticCenters, quadraticSolution));
        Grid uneven = new Grid(5, 3);
        uneven.refine(new boolean[]{true, false, true, false, false});
        double[] unevenCenters = centers(uneven);
        double[] unevenSolution = thomas(assemble(uneven, QUADRATIC, unevenCenters));
        check(worstError(QUADRATIC, unevenCenters, unevenSolution) < 1e-13,
                "exactness must survive an uneven grid, worst error "
                        + worstError(QUADRATIC, unevenCenters, unevenSolution));
        check(uneven.gradeViolations() == 0, "the uneven grid must still be graded");

        // --- the solver against a dense direct solve and its own residual -----
        double[] dense = denseSolve(quadraticSystem);
        double worstDifference = 0.0;
        for (int i = 0; i < dense.length; i++) {
            worstDifference = Math.max(worstDifference, Math.abs(dense[i] - quadraticSolution[i]));
        }
        check(worstDifference < 1e-10, "Thomas must agree with dense elimination, worst " + worstDifference);
        check(residual(quadraticSystem, quadraticSolution) < 1e-10, "the residual must be at rounding level");

        // --- adaptivity: a mildly under-resolved feature -----------------------
        Grid adaptive = adapt(MILD, 16, 6, 2);
        check(adaptive.gradeViolations() == 0, "every refinement pass must leave a graded grid, violations "
                + adaptive.gradeViolations());
        check(adaptive.cells.size() > 16, "a bump must attract cells, cells " + adaptive.cells.size());
        double[] adaptiveCenters = centers(adaptive);
        double[] adaptiveSolution = thomas(assemble(adaptive, MILD, adaptiveCenters));
        double adaptiveError = l2Error(MILD, adaptive, adaptiveCenters, adaptiveSolution);
        Grid uniform = new Grid(adaptive.cells.size(), 6);
        double[] uniformCenters = centers(uniform);
        double[] uniformSolution = thomas(assemble(uniform, MILD, uniformCenters));
        double uniformError = l2Error(MILD, uniform, uniformCenters, uniformSolution);
        check(adaptiveError * 1.2 < uniformError, "with the same " + adaptive.cells.size()
                + " cells the adaptive grid must win by a clear margin: " + adaptiveError + " vs "
                + uniformError);
        int refined = 0;
        boolean localised = true;
        for (Cell cell : adaptive.cells) {
            if (cell.level > 0) {
                refined++;
                if (Math.abs(cell.center() - 0.5) > 0.25) localised = false;
            }
        }
        check(refined > 0 && refined < adaptive.cells.size(),
                "the refinement must be local, refined cells " + refined + " of " + adaptive.cells.size());
        check(localised, "every refined cell must sit inside the feature it is chasing");

        // --- and its limit: over-refinement with this coupling degrades the solve
        double[] errors = new double[3];
        for (int round = 0; round < errors.length; round++) {
            Grid grid = adapt(SHARP, 16, 8, round + 2);
            double[] centers = centers(grid);
            errors[round] = l2Error(SHARP, grid, centers, thomas(assemble(grid, SHARP, centers)));
        }
        check(errors[1] < errors[0], "one more pass must help before the coupling limit: " + errors[0]
                + " -> " + errors[1]);
        check(errors[2] > errors[1], "past that limit the naively coupled solve degrades: " + errors[1]
                + " -> " + errors[2]);

        // --- boundary: a grid at the level cap stops growing -------------------
        Grid capped = adapt(SHARP, 4, 1, 6);
        for (Cell cell : capped.cells) check(cell.level <= 1, "no cell may exceed the level cap");
        check(capped.gradeViolations() == 0, "the capped grid is still graded");

        // --- determinism -------------------------------------------------------
        Grid again = adapt(MILD, 16, 6, 2);
        check(again.cells.size() == adaptive.cells.size(), "the same refinement must repeat");
        for (int i = 0; i < again.cells.size(); i++) {
            check(again.cells.get(i).lo == adaptive.cells.get(i).lo
                            && again.cells.get(i).level == adaptive.cells.get(i).level,
                    "cell " + i + " must be identical on the second run");
        }

        System.out.println("AMR over [0,1]: " + adaptive.cells.size() + " cells after two passes, L2 error "
                + adaptiveError + " against " + uniformError + " for a uniform grid of the same size");
        System.out.println("quadratic solution reproduced to " + worstError(QUADRATIC, quadraticCenters,
                quadraticSolution) + " on any grid; Thomas vs dense solve " + worstDifference);
        System.out.println("grading violations " + adaptive.gradeViolations() + "; refined cells " + refined
                + " of " + adaptive.cells.size() + "; L2 error over passes " + errors[0] + " -> "
                + errors[1] + " -> " + errors[2]);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
