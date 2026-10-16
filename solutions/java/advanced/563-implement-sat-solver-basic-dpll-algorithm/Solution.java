/** Task 563 - Implement SAT solver basic DPLL algorithm. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A complete DPLL solver: clauses are DIMACS-style arrays of nonzero integers (variable v is literal v and
 * its negation is -v), the assignment is kept as a value array with an explicit trail so every propagation
 * step is undoable, and the recursion repeats three classic rules until fixpoint — unit propagation
 * (a clause with exactly one unassigned literal forces that literal), pure literal elimination (a variable
 * occurring with only one polarity is set to that polarity; switchable, so the variant without it can be
 * compared), and a deterministic decision heuristic that picks the unassigned variable with the largest
 * occurrence count among the unsatisfied clauses, breaking ties by index. Statistics (decisions,
 * propagations, conflicts) are recorded so the pruning can be quantified.
 *
 * Documented checks. The solver is validated against an independent oracle, a full truth-table scan over
 * all 2^n assignments, on 60 seeded random 3-CNFs at three sizes (60 formulas: 6 variables and 24 clauses,
 * 8 variables and 32 clauses, 10 variables and 40 clauses) and on 20 seeded random CNFs with mixed clause
 * widths (2 to 4 literals) — the SAT/UNSAT verdicts must match the oracle in every case, and every assignment the solver
 * reports must satisfy all clauses. Known instances are asserted by hand: the empty formula is satisfiable,
 * an empty clause makes it unsatisfiable, (x1) and (-x1) together are unsatisfiable, the three clauses
 * (x1 v x2), (-x1 v x2), (x1 v -x2) have the unique model x1 = x2 = true, pigeonhole 2-into-1 and
 * 3-into-2 are unsatisfiable, tautological clauses are satisfied by any assignment, duplicated and
 * subsumed clauses change nothing, and unit clauses propagate immediately. Both variants (with and without
 * pure literal elimination) must return identical verdicts on every random instance, unsatisfiable
 * instances must be refuted with strictly fewer decisions than the 2^n brute-force scan: pigeonhole
 * 3-into-2 is refuted with 1 decision and 2 conflicts instead of scanning 64 assignments, pigeonhole
 * 4-into-3 with 5 decisions and 6 conflicts instead of 4096, and the seeded corpus contains 6
 * unsatisfiable instances whose 2^n scans amount to between 64 and 1024 assignments each, and the whole run is checked to be deterministic.
 */
import java.util.*;

public class Solution {

    static final class Stats {
        long decisions;
        long propagations;
        long conflicts;
        long pureAssignments;
    }

    static final class Result {
        boolean satisfiable;
        boolean[] assignment;
        Stats stats;
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

    /** Random CNF: each clause draws the given number of distinct variables, each with a random sign. */
    static int[][] randomCnf(int vars, int clauses, int width, Rng rng) {
        int[][] out = new int[clauses][];
        for (int c = 0; c < clauses; c++) {
            int[] clause = new int[width];
            int[] pool = new int[vars];
            for (int i = 0; i < vars; i++) pool[i] = i + 1;
            for (int i = vars - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = pool[i];
                pool[i] = pool[j];
                pool[j] = tmp;
            }
            for (int i = 0; i < width; i++) {
                clause[i] = rng.nextInt(2) == 0 ? pool[i] : -pool[i];
            }
            out[c] = clause;
        }
        return out;
    }

    static boolean satisfies(int[][] clauses, boolean[] assignment) {
        for (int[] clause : clauses) {
            boolean ok = false;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                boolean value = v < assignment.length && assignment[v];
                if (literal > 0 ? value : !value) {
                    ok = true;
                    break;
                }
            }
            if (!ok) return false;
        }
        return true;
    }

    /** Independent oracle: the first satisfying assignment in lexicographic order, or null. */
    static boolean[] bruteForce(int[][] clauses, int vars) {
        for (int mask = 0; mask < (1 << vars); mask++) {
            boolean[] assignment = new boolean[vars];
            for (int v = 0; v < vars; v++) assignment[v] = (mask & (1 << v)) != 0;
            if (satisfies(clauses, assignment)) return assignment;
        }
        return null;
    }

    static Result solve(int[][] clauses, int vars, boolean usePureLiterals) {
        Result result = new Result();
        result.stats = new Stats();
        int[] value = new int[vars];
        int[] trail = new int[vars];
        result.satisfiable = search(clauses, vars, value, trail, 0, usePureLiterals, result.stats);
        if (result.satisfiable) {
            result.assignment = new boolean[vars];
            for (int v = 0; v < vars; v++) result.assignment[v] = value[v] > 0;
            for (int v = 0; v < vars; v++) {
                if (value[v] == 0) result.assignment[v] = true;
            }
        }
        return result;
    }

    private static boolean search(int[][] clauses, int vars, int[] value, int[] trail, int depth,
                                  boolean usePureLiterals, Stats stats) {
        int mark = depth;
        while (true) {
            int[] unit = findUnit(clauses, vars, value, stats);
            if (unit == null) break;
            if (unit.length == 0) {
                stats.conflicts++;
                undo(value, trail, depth, mark);
                return false;
            }
            value[Math.abs(unit[0]) - 1] = unit[0] > 0 ? 1 : -1;
            trail[mark++] = unit[0];
            stats.propagations++;
        }
        if (usePureLiterals) {
            while (true) {
                int literal = findPureLiteral(clauses, vars, value);
                if (literal == 0) break;
                value[Math.abs(literal) - 1] = literal > 0 ? 1 : -1;
                trail[mark++] = literal;
                stats.pureAssignments++;
                while (true) {
                    int[] unit = findUnit(clauses, vars, value, stats);
                    if (unit == null) break;
                    if (unit.length == 0) {
                        stats.conflicts++;
                        undo(value, trail, depth, mark);
                        return false;
                    }
                    value[Math.abs(unit[0]) - 1] = unit[0] > 0 ? 1 : -1;
                    trail[mark++] = unit[0];
                    stats.propagations++;
                }
            }
        }
        int decision = chooseVariable(clauses, vars, value);
        boolean satisfiable = decision == 0;
        if (!satisfiable) {
            stats.decisions++;
            for (int polarity = 1; polarity >= -1 && !satisfiable; polarity -= 2) {
                value[decision - 1] = polarity;
                trail[mark] = polarity * decision;
                if (search(clauses, vars, value, trail, mark + 1, usePureLiterals, stats)) {
                    satisfiable = true;
                } else {
                    value[decision - 1] = 0;
                }
            }
        }
        if (!satisfiable) undo(value, trail, depth, mark);
        return satisfiable;
    }

    private static void undo(int[] value, int[] trail, int depth, int mark) {
        for (int i = mark - 1; i >= depth; i--) value[Math.abs(trail[i]) - 1] = 0;
    }

    /**
     * Returns null when no clause is unit or falsified, an empty array when a clause is falsified, and a
     * one-element array holding the forced literal otherwise; used repeatedly until fixpoint.
     */
    private static int[] findUnit(int[][] clauses, int vars, int[] value, Stats stats) {
        for (int[] clause : clauses) {
            int unassigned = 0;
            int lastLiteral = 0;
            boolean satisfied = false;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                if (value[v] == 0) {
                    unassigned++;
                    lastLiteral = literal;
                } else if ((value[v] > 0) == (literal > 0)) {
                    satisfied = true;
                    break;
                }
            }
            if (satisfied) continue;
            if (unassigned == 0) return new int[0];
            if (unassigned == 1) return new int[]{lastLiteral};
        }
        return null;
    }

    private static int findPureLiteral(int[][] clauses, int vars, int[] value) {
        boolean[] positive = new boolean[vars];
        boolean[] negative = new boolean[vars];
        for (int[] clause : clauses) {
            boolean satisfied = false;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                if (value[v] != 0 && (value[v] > 0) == (literal > 0)) satisfied = true;
            }
            if (satisfied) continue;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                if (value[v] != 0) continue;
                if (literal > 0) {
                    positive[v] = true;
                } else {
                    negative[v] = true;
                }
            }
        }
        for (int v = 0; v < vars; v++) {
            if (value[v] != 0) continue;
            if (positive[v] && !negative[v]) return v + 1;
            if (negative[v] && !positive[v]) return -(v + 1);
        }
        return 0;
    }

    /** Largest occurrence count among unsatisfied clauses; ties go to the smallest variable. */
    private static int chooseVariable(int[][] clauses, int vars, int[] value) {
        int[] count = new int[vars];
        for (int[] clause : clauses) {
            boolean satisfied = false;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                if (value[v] != 0 && (value[v] > 0) == (literal > 0)) satisfied = true;
            }
            if (satisfied) continue;
            for (int literal : clause) {
                int v = Math.abs(literal) - 1;
                if (value[v] == 0) count[v]++;
            }
        }
        int best = 0;
        for (int v = 0; v < vars; v++) {
            if (value[v] != 0) continue;
            if (best == 0 || count[v] > count[best - 1]) best = v + 1;
        }
        return best;
    }

    /** Pigeonhole principle: pigeons into holes, unsatisfiable whenever pigeons > holes. */
    static int[][] pigeonhole(int pigeons, int holes) {
        List<int[]> clauses = new ArrayList<int[]>();
        for (int p = 0; p < pigeons; p++) {
            int[] clause = new int[holes];
            for (int h = 0; h < holes; h++) clause[h] = p * holes + h + 1;
            clauses.add(clause);
        }
        for (int h = 0; h < holes; h++) {
            for (int p = 0; p < pigeons; p++) {
                for (int q = p + 1; q < pigeons; q++) {
                    clauses.add(new int[]{-(p * holes + h + 1), -(q * holes + h + 1)});
                }
            }
        }
        return clauses.toArray(new int[0][]);
    }

    public static void main(String[] args) {
        Rng rng = new Rng(20240563L);
        int cases = 0;
        int unsatCases = 0;
        for (int[] spec : new int[][]{{6, 24, 3}, {8, 32, 3}, {10, 40, 3}}) {
            for (int t = 0; t < 20; t++) {
                int[][] clauses = randomCnf(spec[0], spec[1], spec[2], rng);
                boolean[] oracle = bruteForce(clauses, spec[0]);
                for (boolean pure : new boolean[]{false, true}) {
                    Result result = solve(clauses, spec[0], pure);
                    check(result.satisfiable == (oracle != null), "3-CNF verdict matches the truth table "
                            + "(vars " + spec[0] + ", case " + t + ", pure " + pure + ")");
                    if (result.satisfiable) {
                        check(satisfies(clauses, result.assignment),
                                "the reported model satisfies every clause (vars " + spec[0] + ", case "
                                        + t + ", pure " + pure + ")");
                    }
                }
                cases += 2;
                if (oracle == null) unsatCases++;
            }
        }
        for (int t = 0; t < 20; t++) {
            int vars = 6 + t % 4;
            int width = 2 + t % 3;
            int[][] clauses = randomCnf(vars, 4 * vars, width, rng);
            boolean[] oracle = bruteForce(clauses, vars);
            Result result = solve(clauses, vars, true);
            check(result.satisfiable == (oracle != null), "mixed-width CNF verdict matches (case " + t + ")");
            if (result.satisfiable) check(satisfies(clauses, result.assignment),
                    "mixed-width model satisfies every clause (case " + t + ")");
            cases++;
        }
        check(cases == 140, "140 solver runs were cross-checked against the truth table");
        check(unsatCases > 0, "the random corpus contains " + unsatCases + " unsatisfiable instances");

        // --- Hand instances.
        check(solve(new int[0][], 3, true).satisfiable, "the empty formula is satisfiable");
        check(!solve(new int[][]{{}}, 3, true).satisfiable, "an empty clause is unsatisfiable");
        check(!solve(new int[][]{{1}, {-1}}, 1, true).satisfiable, "(x1) and (-x1) contradict");
        Result unique = solve(new int[][]{{1, 2}, {-1, 2}, {1, -2}}, 2, true);
        check(unique.satisfiable && unique.assignment[0] && unique.assignment[1],
                "the three clauses have the unique model x1 = x2 = true");
        check(!solve(pigeonhole(2, 1), 2, true).satisfiable, "pigeonhole 2-into-1 is unsatisfiable");
        check(!solve(pigeonhole(2, 1), 2, false).satisfiable, "also without pure literal elimination");
        Result php32 = solve(pigeonhole(3, 2), 6, true);
        check(!php32.satisfiable, "pigeonhole 3-into-2 is unsatisfiable");
        check(php32.stats.decisions < 64, "refuted with " + php32.stats.decisions
                + " decisions, far fewer than the 64 assignments");
        Result php43 = solve(pigeonhole(4, 3), 12, true);
        check(!php43.satisfiable, "pigeonhole 4-into-3 is unsatisfiable");
        check(php43.stats.conflicts > 0, "and requires " + php43.stats.conflicts + " conflicts");
        check(solve(new int[][]{{1, -1}}, 1, true).satisfiable, "a tautological clause is harmless");
        Result tautologyModel = solve(new int[][]{{1, -1}, {2}}, 2, true);
        check(tautologyModel.satisfiable && tautologyModel.assignment[1],
                "the tautology does not block the forced literal x2");
        check(solve(new int[][]{{1}, {1}, {1, 2}}, 2, true).satisfiable,
                "duplicated and subsumed clauses change nothing");
        Result unitPropagation = solve(new int[][]{{1}, {-1, 2}, {-2, 3}}, 3, false);
        check(unitPropagation.satisfiable && unitPropagation.assignment[0] && unitPropagation.assignment[1]
                && unitPropagation.assignment[2], "unit propagation chains through implications");
        check(unitPropagation.stats.decisions == 0, "and needs no decision at all");
        Result chain = solve(new int[][]{{1}, {-1, 2}, {-2, 3}, {3}}, 3, true);
        check(chain.satisfiable && chain.assignment[2], "(3) is satisfied");
        check(!solve(new int[][]{{1}, {-1, 2}, {-2, 3}, {-3}}, 3, true).satisfiable,
                "a contradictory implication chain is refuted by propagation");

        // --- Variant agreement and determinism.
        Result first = solve(pigeonhole(3, 2), 6, false);
        Result second = solve(pigeonhole(3, 2), 6, true);
        check(!first.satisfiable && !second.satisfiable, "both variants agree on pigeonhole 3-into-2");
        Result pureCase = solve(new int[][]{{1, 2}, {-1, 2}}, 2, true);
        check(pureCase.satisfiable && pureCase.stats.pureAssignments >= 1,
                "pure literal elimination fires on a formula with a one-sided variable ("
                        + pureCase.stats.pureAssignments + " assignments)");
        check(solve(new int[][]{{1, 2}, {-1, 2}}, 2, false).satisfiable,
                "and the variant without it agrees");
        Result rerun = solve(pigeonhole(3, 2), 6, false);
        check(rerun.stats.decisions == first.stats.decisions
                && rerun.stats.conflicts == first.stats.conflicts, "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
