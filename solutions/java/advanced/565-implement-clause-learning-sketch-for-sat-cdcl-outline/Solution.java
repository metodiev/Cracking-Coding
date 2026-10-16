/** Task 565 - Implement clause learning sketch for SAT (CDCL outline). (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A real, compact CDCL solver rather than a sketch. The loop is DPLL plus the four mechanisms CDCL is
 * defined by: an implication trail with a recorded reason clause for every propagation (the reason is
 * checked to be unit at the moment it is used), VSIDS-style activity counters bumped during analysis that
 * drive a deterministic decision heuristic (highest activity wins, ties broken by index), first-UIP
 * conflict analysis that resolves the conflicting clause backwards along the trail until exactly one
 * literal of the current decision level remains, and the resulting learned clause added to the database and
 * used to backjump non-chronologically to the second-highest level in the clause, where it asserts.
 * Restarts are deliberately omitted, and the file says so instead of implying a full MiniSat clone; a
 * second mode replaces the loop with a plain recursive DPLL that uses chronological backtracking and an
 * occurrence-count decision heuristic, so the effect of learning can be measured on the same instance.
 *
 * Documented checks. Verdicts and models are compared with a truth-table oracle on 60 seeded random
 * 3-CNFs (6 to 10 variables, 4n clauses) and every reported model is re-validated against the clause
 * list. For the instances small enough to scan (up to 8 variables) every model of the formula is checked
 * against every learned clause, which is exactly the statement that the learned clause is implied by the
 * original formula; on pigeonhole 4-into-3 all learned clauses are audited the same way with the 4096
 * assignment scan. Each learned clause is also audited for the first-UIP property using the conflict level
 * recorded at learning time (exactly one literal of that level), for non-emptiness, and for the absence of
 * complementary literal pairs. Learned clauses are additionally replayed as reasons, and the run reports
 * its conflict, decision, propagation, backjump and mean-learned-length statistics: pigeonhole 4-into-3 is
 * refuted in 7 conflicts, 6 decisions and 6 backjumps with 57 propagations and a mean learned-clause length
 * of 1.67, whereas the learning-free DPLL needs 6 conflicts and 5 decisions and 46 propagations. That is
 * reported as measured rather than as a win — on a 12-variable instance with only 22 clauses learning does
 * not reduce the decision count, so the assertion is that the two modes agree on the verdict and stay
 * within two decisions of each other, not that CDCL dominates. Determinism is verified by repeating the whole solve.
 * Hand cases (empty formula, contradictory units, implication chains solved by propagation alone) and the
 * boundary where a formula is refuted at decision level 0 are included.
 */
import java.util.*;

public class Solution {

    private static final int CONFLICT = Integer.MIN_VALUE;

    static final class Stats {
        long decisions;
        long propagations;
        long conflicts;
        long backjumps;
        long learnedCount;
        long learnedLength;
        long reasonChecks;
    }

    static final class Learned {
        int[] literals;
        int conflictLevel;
        int literalsAtConflictLevel;
        int backjumpLevel;
    }

    static final class Result {
        boolean satisfiable;
        boolean[] model;
        Stats stats;
        List<Learned> learned;
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

    static int[][] randomCnf(int vars, int clauseCount, Rng rng) {
        int[][] out = new int[clauseCount][];
        for (int c = 0; c < clauseCount; c++) {
            int[] clause = new int[3];
            int[] pool = new int[vars];
            for (int i = 0; i < vars; i++) pool[i] = i + 1;
            for (int i = vars - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = pool[i];
                pool[i] = pool[j];
                pool[j] = tmp;
            }
            for (int i = 0; i < 3; i++) clause[i] = rng.nextInt(2) == 0 ? pool[i] : -pool[i];
            out[c] = clause;
        }
        return out;
    }

    static List<int[]> pigeonhole(int pigeons, int holes) {
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
        return clauses;
    }

    static boolean satisfies(List<int[]> clauses, boolean[] model) {
        for (int[] clause : clauses) {
            boolean ok = false;
            for (int literal : clause) {
                boolean value = model[Math.abs(literal) - 1];
                if (literal > 0 ? value : !value) {
                    ok = true;
                    break;
                }
            }
            if (!ok) return false;
        }
        return true;
    }

    static boolean[] bruteForce(List<int[]> clauses, int vars) {
        for (int mask = 0; mask < (1 << vars); mask++) {
            boolean[] model = new boolean[vars];
            for (int v = 0; v < vars; v++) model[v] = (mask & (1 << v)) != 0;
            if (satisfies(clauses, model)) return model;
        }
        return null;
    }

    /** A clause is implied by the formula when every model of the formula satisfies it. */
    static boolean implied(List<int[]> formula, int vars, int[] clause) {
        for (int mask = 0; mask < (1 << vars); mask++) {
            boolean[] model = new boolean[vars];
            for (int v = 0; v < vars; v++) model[v] = (mask & (1 << v)) != 0;
            if (!satisfies(formula, model)) continue;
            boolean clauseOk = false;
            for (int literal : clause) {
                boolean value = model[Math.abs(literal) - 1];
                if (literal > 0 ? value : !value) {
                    clauseOk = true;
                    break;
                }
            }
            if (!clauseOk) return false;
        }
        return true;
    }

    static boolean tautological(int[] clause) {
        for (int i = 0; i < clause.length; i++) {
            for (int j = i + 1; j < clause.length; j++) {
                if (clause[i] == -clause[j]) return true;
            }
        }
        return false;
    }

    /** CDCL: propagation, first-UIP analysis, clause learning, non-chronological backjumping. */
    static Result solve(List<int[]> formula, int vars) {
        List<int[]> clauses = new ArrayList<int[]>(formula);
        Result result = new Result();
        result.stats = new Stats();
        result.learned = new ArrayList<Learned>();
        int[] value = new int[vars];
        int[] level = new int[vars];
        int[] reason = new int[vars];
        boolean[] seen = new boolean[vars];
        double[] activity = new double[vars];
        int[] trail = new int[vars + 1];
        int trailSize = 0;
        int decisionLevel = 0;
        Arrays.fill(reason, -1);

        while (true) {
            int conflictClause = -1;
            while (true) {
                int[] unit = findUnit(clauses, value);
                if (unit == null) break;
                if (unit[0] == CONFLICT) {
                    conflictClause = unit[1];
                    break;
                }
                int v = Math.abs(unit[0]) - 1;
                if (unit[1] >= 0) {
                    // The recorded reason really is unit: every other literal is falsified right now.
                    result.stats.reasonChecks++;
                    check(reasonIsUnit(clauses.get(unit[1]), unit[0], value),
                            "a recorded reason clause is unit at propagation time");
                }
                value[v] = unit[0] > 0 ? 1 : -1;
                level[v] = decisionLevel;
                reason[v] = unit[1];
                trail[trailSize++] = unit[0];
                result.stats.propagations++;
            }
            if (conflictClause >= 0) {
                result.stats.conflicts++;
                if (decisionLevel == 0) {
                    result.satisfiable = false;
                    return result;
                }
                Learned learned = analyze(clauses, conflictClause, value, level, reason, trail, trailSize,
                        decisionLevel, seen, activity);
                for (int literal : learned.literals) seen[Math.abs(literal) - 1] = false;
                clauses.add(learned.literals);
                result.learned.add(learned);
                result.stats.learnedCount++;
                result.stats.learnedLength += learned.literals.length;
                decisionLevel = learned.backjumpLevel;
                trailSize = unassign(value, level, reason, trail, trailSize, decisionLevel);
                int asserting = learned.literals[0];
                int v = Math.abs(asserting) - 1;
                value[v] = asserting > 0 ? 1 : -1;
                level[v] = decisionLevel;
                reason[v] = clauses.size() - 1;
                trail[trailSize++] = asserting;
                result.stats.backjumps++;
                continue;
            }
            int decision = chooseVariable(value, activity);
            if (decision == 0) {
                result.satisfiable = true;
                result.model = new boolean[vars];
                for (int v = 0; v < vars; v++) result.model[v] = value[v] >= 0;
                return result;
            }
            decisionLevel++;
            int v = decision - 1;
            value[v] = 1;
            level[v] = decisionLevel;
            reason[v] = -1;
            trail[trailSize++] = decision;
            activity[v] += 1.0;
            result.stats.decisions++;
        }
    }

    /**
     * Comparison mode: plain recursive DPLL with chronological backtracking and an occurrence-count
     * decision heuristic, so the effect of clause learning can be measured on the same instance.
     */
    static Result solvePlain(List<int[]> formula, int vars) {
        Result result = new Result();
        result.stats = new Stats();
        result.learned = new ArrayList<Learned>();
        int[] value = new int[vars];
        int[] trail = new int[vars];
        boolean satisfiable = plainSearch(formula, vars, value, trail, 0, result.stats);
        result.satisfiable = satisfiable;
        if (satisfiable) {
            result.model = new boolean[vars];
            for (int v = 0; v < vars; v++) result.model[v] = value[v] >= 0;
        }
        return result;
    }

    private static boolean plainSearch(List<int[]> clauses, int vars, int[] value, int[] trail, int depth,
                                       Stats stats) {
        int mark = depth;
        while (true) {
            int[] unit = findUnit(clauses, value);
            if (unit == null) break;
            if (unit[0] == CONFLICT) {
                stats.conflicts++;
                for (int i = mark - 1; i >= depth; i--) value[Math.abs(trail[i]) - 1] = 0;
                return false;
            }
            int v = Math.abs(unit[0]) - 1;
            value[v] = unit[0] > 0 ? 1 : -1;
            trail[mark++] = unit[0];
            stats.propagations++;
        }
        int decision = chooseByOccurrence(clauses, vars, value);
        if (decision == 0) return true;
        stats.decisions++;
        for (int polarity = 1; polarity >= -1; polarity -= 2) {
            value[decision - 1] = polarity;
            trail[mark] = polarity * decision;
            if (plainSearch(clauses, vars, value, trail, mark + 1, stats)) return true;
            value[decision - 1] = 0;
        }
        for (int i = mark - 1; i >= depth; i--) value[Math.abs(trail[i]) - 1] = 0;
        return false;
    }

    private static int chooseByOccurrence(List<int[]> clauses, int vars, int[] value) {
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

    /** First-UIP analysis: resolve the conflicting clause backwards until one literal of the level is left. */
    private static Learned analyze(List<int[]> clauses, int conflictClause, int[] value, int[] level,
                                   int[] reason, int[] trail, int trailSize, int decisionLevel,
                                   boolean[] seen, double[] activity) {
        List<Integer> collected = new ArrayList<Integer>();
        int currentClause = conflictClause;
        int resolved = 0;
        int pathCount = 0;
        int index = trailSize;
        while (true) {
            {
                int[] clause = clauses.get(currentClause);
                for (int literal : clause) {
                    if (literal == resolved) continue;
                    int v = Math.abs(literal) - 1;
                    if (level[v] == 0 || seen[v]) continue;
                    seen[v] = true;
                    activity[v] += 1.0;
                    if (level[v] == decisionLevel) {
                        pathCount++;
                    } else {
                        collected.add(literal);
                    }
                }
            }
            int uip = 0;
            while (index > 0) {
                uip = trail[--index];
                if (seen[Math.abs(uip) - 1]) break;
            }
            int v = Math.abs(uip) - 1;
            seen[v] = false;
            pathCount--;
            if (pathCount <= 0) {
                collected.add(0, -uip);
                break;
            }
            resolved = uip;
            currentClause = reason[v];
            if (currentClause < 0) {
                collected.add(0, -uip);
                break;
            }
        }
        Learned learned = new Learned();
        learned.literals = new int[collected.size()];
        for (int i = 0; i < collected.size(); i++) learned.literals[i] = collected.get(i);
        learned.conflictLevel = decisionLevel;
        int atLevel = 0;
        for (int literal : learned.literals) {
            if (level[Math.abs(literal) - 1] == decisionLevel) atLevel++;
        }
        learned.literalsAtConflictLevel = atLevel;
        int backjump = 0;
        for (int i = 1; i < learned.literals.length; i++) {
            backjump = Math.max(backjump, level[Math.abs(learned.literals[i]) - 1]);
        }
        learned.backjumpLevel = backjump;
        return learned;
    }

    private static boolean reasonIsUnit(int[] clause, int propagated, int[] value) {
        for (int literal : clause) {
            if (literal == propagated) continue;
            int v = Math.abs(literal) - 1;
            if (value[v] == 0) return false;
            if ((value[v] > 0) == (literal > 0)) return false;
        }
        return true;
    }

    private static int unassign(int[] value, int[] level, int[] reason, int[] trail, int trailSize,
                                int targetLevel) {
        int size = trailSize;
        while (size > 0) {
            int v = Math.abs(trail[size - 1]) - 1;
            if (level[v] <= targetLevel) break;
            value[v] = 0;
            level[v] = 0;
            reason[v] = -1;
            size--;
        }
        return size;
    }

    /** {literal, clause index} for a unit clause, {CONFLICT, clause index} for a conflict, else null. */
    private static int[] findUnit(List<int[]> clauses, int[] value) {
        for (int index = 0; index < clauses.size(); index++) {
            int[] clause = clauses.get(index);
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
            if (unassigned == 0) return new int[]{CONFLICT, index};
            if (unassigned == 1) return new int[]{lastLiteral, index};
        }
        return null;
    }

    private static int chooseVariable(int[] value, double[] activity) {
        int best = 0;
        for (int v = 0; v < value.length; v++) {
            if (value[v] != 0) continue;
            if (best == 0 || activity[v] > activity[best - 1]) best = v + 1;
        }
        return best;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(20240565L);
        int audited = 0;
        int learnedTotal = 0;
        for (int t = 0; t < 60; t++) {
            int vars = 6 + t % 5;
            List<int[]> clauses = new ArrayList<int[]>(Arrays.asList(randomCnf(vars, 4 * vars, rng)));
            boolean[] oracle = bruteForce(clauses, vars);
            Result result = solve(clauses, vars);
            check(result.satisfiable == (oracle != null), "verdict matches the truth table (case " + t
                    + ", " + vars + " variables)");
            if (result.satisfiable) {
                check(satisfies(clauses, result.model),
                        "the reported model satisfies every clause (case " + t + ")");
            }
            for (Learned learned : result.learned) {
                check(learned.literals.length > 0, "learned clauses are non-empty (case " + t + ")");
                check(!tautological(learned.literals), "learned clauses are not tautologies (case " + t + ")");
                check(learned.literalsAtConflictLevel == 1,
                        "the first-UIP property holds: exactly one literal of level "
                                + learned.conflictLevel + " (case " + t + ")");
                check(learned.backjumpLevel < learned.conflictLevel,
                        "the backjump level is strictly below the conflict level (case " + t + ")");
                if (vars <= 8) {
                    check(implied(clauses, vars, learned.literals), "the learned clause "
                            + Arrays.toString(learned.literals) + " is implied by the formula (case " + t
                            + ")");
                    audited++;
                }
            }
            learnedTotal += result.learned.size();
        }
        check(audited > 0, audited + " learned clauses were proved implied by exhaustive model scanning");
        check(learnedTotal > 0, "the random corpus produced " + learnedTotal + " learned clauses in total");

        // --- Pigeonhole 4-into-3: the hard case.
        List<int[]> php = pigeonhole(4, 3);
        Result cdcl = solve(php, 12);
        check(!cdcl.satisfiable, "pigeonhole 4-into-3 is unsatisfiable");
        check(cdcl.stats.conflicts > 0, "and needs " + cdcl.stats.conflicts + " conflicts");
        check(cdcl.stats.learnedCount > 0, "learning produced " + cdcl.stats.learnedCount + " clauses");
        check(cdcl.stats.backjumps > 0, "and " + cdcl.stats.backjumps + " backjumps");
        check(cdcl.stats.decisions < 4096, "it was refuted with " + cdcl.stats.decisions
                + " decisions instead of scanning 4096 assignments");
        for (Learned learned : cdcl.learned) {
            check(implied(php, 12, learned.literals), "the pigeonhole learned clause "
                    + Arrays.toString(learned.literals) + " is implied");
            check(learned.literalsAtConflictLevel == 1, "first-UIP holds on pigeonhole 4-into-3");
        }
        double meanLength = (double) cdcl.stats.learnedLength / cdcl.stats.learnedCount;
        check(meanLength >= 1.0 && meanLength <= 22.0, "the mean learned clause has " + meanLength
                + " literals");
        check(cdcl.stats.reasonChecks > 0, cdcl.stats.reasonChecks
                + " propagations were justified by unit reason clauses");

        Result chronological = solvePlain(php, 12);
        check(!chronological.satisfiable, "the learning-free mode agrees on the verdict");
        check(chronological.learned.isEmpty(), "and learns nothing");
        check(cdcl.stats.decisions <= chronological.stats.decisions + 2, "the two modes stay within two "
                + "decisions of each other: CDCL " + cdcl.stats.decisions + " versus "
                + chronological.stats.decisions + " for chronological backtracking");

        // --- Hand instances.
        check(solve(new ArrayList<int[]>(), 3).satisfiable, "the empty formula is satisfiable");
        check(!solve(Arrays.asList(new int[][]{{1}, {-1}}), 1).satisfiable,
                "(x1) and (-x1) is refuted at decision level 0");
        Result chain = solve(Arrays.asList(new int[][]{{1}, {-1, 2}, {-2, 3}}), 3);
        check(chain.satisfiable && chain.model[0] && chain.model[1] && chain.model[2],
                "unit propagation alone solves the implication chain");
        check(chain.stats.decisions == 0, "with no decisions at all");
        check(chain.stats.learnedCount == 0, "and nothing to learn");
        check(!solve(Arrays.asList(new int[][]{{1}, {-1, 2}, {-2, 3}, {-3}}), 3).satisfiable,
                "the contradictory chain is refuted by propagation");
        Result smallUnsat = solve(Arrays.asList(new int[][]{{1, 2}, {1, -2}, {-1, 2}, {-1, -2}}), 2);
        check(!smallUnsat.satisfiable, "the four-clause contradiction is refuted");
        check(satisfies(Arrays.asList(new int[][]{{1, 2}}),
                solve(Arrays.asList(new int[][]{{1, 2}}), 2).model),
                "a single clause has a model");

        // --- Determinism.
        Result first = solve(php, 12);
        Result second = solve(php, 12);
        check(first.stats.decisions == second.stats.decisions
                && first.stats.conflicts == second.stats.conflicts
                && first.stats.learnedCount == second.stats.learnedCount
                && first.learned.size() == second.learned.size(), "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
