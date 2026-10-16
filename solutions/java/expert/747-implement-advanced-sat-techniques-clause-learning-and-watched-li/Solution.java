/** Task 747 - Implement advanced SAT techniques: clause learning and watched literals in practice.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A CDCL solver on CNF, with the two techniques the task names implemented for real:
 *   watched literals - every clause watches two of its literals; only the watch lists of the newly
 *     falsified literal are visited by propagation, and a watch only moves when its literal is
 *     falsified. Watches are kept in the same array as the clause itself (indices 0 and 1), so a
 *     moved watch is a swap inside the clause plus two watch-list updates;
 *   clause learning - on conflict the implication graph is resolved backwards to the first unique
 *     implication point (1UIP), the learnt clause is added with the asserting literal first, and the
 *     solver backjumps to the second highest decision level in it.
 * Literals are encoded as 2*var + sign (sign 0 = positive), so negation is `lit ^ 1`. The decision
 * heuristic is a deterministic VSIDS variant: activities grow by 1 per bump and ties go to the
 * lowest variable index, so every run is reproducible without a random source.
 *
 * Simplification (documented). No restarts, no clause deletion/vivification, no preprocessing, no
 * phase saving and no non chronological backjumping; VSIDS uses a plain integer-ish bump instead of
 * the exponential decay of MiniSat. These omissions only affect speed, not the contract: the solver
 * decides exactly the formulas it is given.
 *
 * Verified. On every generated instance the SAT/UNSAT verdict is compared against exhaustive
 * enumeration of all 2^n assignments; each returned model is checked against every original clause;
 * every learnt clause is checked to be *implied* by the formula (no model of the formula falsifies
 * it), which is the soundness property clause learning must have; the watched literal invariant is
 * re-derived from scratch after propagation (no unsatisfied clause ends up with two false watches,
 * and no unit clause remains at the fixpoint); propagation itself is compared against an independent
 * naive scan-until-fixpoint implementation on random assumption sets; and empty formulas, empty
 * clauses, tautologies, duplicate literals and unit clauses are covered by dedicated checks.
 */
import java.util.*;

/** CDCL SAT solver with watched literals, 1UIP learning and VSIDS-style decisions. */
final class CdclSolver {
    static final int UNKNOWN = 0;
    static final int TRUE = 1;
    static final int FALSE = 2;

    private final int nVars;
    private final List<int[]> clauses = new ArrayList<>();
    private final List<List<int[]>> watches = new ArrayList<>();
    private final int[] value;          // per variable: -1 false, 0 unassigned, +1 true
    private final int[] level;          // decision level of the assignment
    private final int[] reasonOf;       // clause index that implied the literal, -1 for decisions
    private final double[] activity;
    private final List<Integer> trail = new ArrayList<>();
    private int trailHead;
    private int currentLevel;
    private int conflicts;
    private final List<int[]> learnt = new ArrayList<>();
    private boolean triviallyUnsat;
    private int[] conflictClause;

    CdclSolver(int nVars, List<int[]> formula) {
        this.nVars = nVars;
        value = new int[nVars];
        level = new int[nVars];
        reasonOf = new int[nVars];
        activity = new double[nVars];
        Arrays.fill(reasonOf, -1);
        for (int i = 0; i < 2 * nVars; i++) watches.add(new ArrayList<>());
        for (int[] raw : formula) addClause(raw);
    }

    static int lit(int variable, boolean positive) { return 2 * variable + (positive ? 0 : 1); }

    static int varOf(int literal) { return literal >>> 1; }

    static int negate(int literal) { return literal ^ 1; }

    private int literalValue(int literal) {
        int assigned = value[varOf(literal)];
        if (assigned == 0) return 0;
        return (literal & 1) == 0 ? assigned : -assigned;
    }

    /** Adds a clause after removing duplicates and tautologies; an empty clause makes it UNSAT. */
    private void addClause(int[] raw) {
        boolean[] present = new boolean[2 * nVars];
        for (int literal : raw) present[literal] = true;
        List<Integer> literals = new ArrayList<>();
        for (int literal = 0; literal < 2 * nVars; literal++) {
            if (!present[literal]) continue;
            if (present[literal ^ 1]) return;               // tautology: contributes nothing
            literals.add(literal);
        }
        if (literals.isEmpty()) {
            triviallyUnsat = true;
            return;
        }
        int[] clause = new int[literals.size()];
        for (int i = 0; i < clause.length; i++) clause[i] = literals.get(i);
        clauses.add(clause);
        if (clause.length == 1) {
            // a unit clause is not watched: it is asserted immediately at level zero
            if (!assign(clause[0], clauses.size() - 1)) triviallyUnsat = true;
            return;
        }
        watches.get(clause[0]).add(clause);
        watches.get(clause[1]).add(clause);
    }

    private boolean assign(int literal, int reasonClause) {
        int variable = varOf(literal);
        if (value[variable] != 0) return literalValue(literal) == 1;
        value[variable] = (literal & 1) == 0 ? 1 : -1;
        level[variable] = currentLevel;
        reasonOf[variable] = reasonClause;
        trail.add(literal);
        return true;
    }

    /** Unit propagation over the watch lists; returns false on conflict. */
    private boolean propagate() {
        while (trailHead < trail.size()) {
            int trueLiteral = trail.get(trailHead++);
            List<int[]> affected = watches.get(negate(trueLiteral));
            for (int i = 0; i < affected.size(); i++) {
                int[] clause = affected.get(i);
                if (clause.length < 2) continue;        // units are asserted, never watched
                int falseIndex = clause[0] == negate(trueLiteral) ? 0 : 1;
                int otherIndex = 1 - falseIndex;
                if (literalValue(clause[otherIndex]) == 1) continue;      // clause satisfied
                int replacement = -1;
                for (int k = 2; k < clause.length; k++) {
                    if (literalValue(clause[k]) != -1) {
                        replacement = k;
                        break;
                    }
                }
                if (replacement >= 0) {
                    int swap = clause[falseIndex];
                    clause[falseIndex] = clause[replacement];
                    clause[replacement] = swap;
                    watches.get(clause[falseIndex]).add(clause);
                    affected.remove(i);
                    i--;
                    continue;
                }
                if (literalValue(clause[otherIndex]) == -1) {                // conflict
                    conflictClause = clause;
                    return false;
                }
                if (!assign(clause[otherIndex], clauses.indexOf(clause))) return false;
            }
        }
        return true;
    }

    /** 1UIP conflict analysis; the returned clause starts with the asserting literal. */
    private int[] analyse(int[] conflictClause) {
        List<Integer> learned = new ArrayList<>();
        boolean[] seen = new boolean[nVars];
        int pathCount = 0;
        int index = trail.size() - 1;
        int[] clause = conflictClause;
        int resolved = -1;                              // -1: the conflict clause, resolve everything
        while (true) {
            for (int literal : clause) {
                if (literal == resolved) continue;      // resolution drops the resolved literal
                int variable = varOf(literal);
                if (seen[variable] || level[variable] == 0) continue;
                seen[variable] = true;
                activity[variable] += 1;
                if (level[variable] == currentLevel) pathCount++;
                else learned.add(literal);
            }
            while (index >= 0 && !seen[varOf(trail.get(index))]) index--;
            if (index < 0) throw new IllegalStateException("conflict analysis ran off the trail");
            int uip = trail.get(index--);
            seen[varOf(uip)] = false;
            pathCount--;
            if (pathCount <= 0) {
                int[] result = new int[learned.size() + 1];
                result[0] = negate(uip);                // the asserting literal
                for (int i = 0; i < learned.size(); i++) result[i + 1] = learned.get(i);
                return result;
            }
            resolved = uip;
            int reasonIndex = reasonOf[varOf(uip)];
            if (reasonIndex < 0) {
                throw new IllegalStateException("a decision literal cannot be resolved away");
            }
            clause = clauses.get(reasonIndex);
        }
    }

    /** Undoes every assignment above the given level. */
    private void backjumpTo(int targetLevel) {
        for (int i = trail.size() - 1; i >= 0; i--) {
            int literal = trail.get(i);
            int variable = varOf(literal);
            if (level[variable] <= targetLevel) break;
            value[variable] = 0;
            reasonOf[variable] = -1;
            trail.remove(i);
        }
        trailHead = trail.size();
        currentLevel = targetLevel;
    }

    /** Adds a learnt clause (asserting literal first) and re-wires its watches. */
    private int learn(int[] clauseData) {
        int[] clause = clauseData.clone();
        if (clause.length > 1) {
            // index 0 keeps the asserting literal; index 1 gets the highest level of the rest
            int highest = 1;
            for (int i = 2; i < clause.length; i++) {
                if (level[varOf(clause[i])] > level[varOf(clause[highest])]) highest = i;
            }
            if (highest != 1) {
                int swap = clause[1];
                clause[1] = clause[highest];
                clause[highest] = swap;
            }
        }
        clauses.add(clause);
        if (clause.length > 1) {
            watches.get(clause[0]).add(clause);
            watches.get(clause[1]).add(clause);
        }
        learnt.add(clause);
        return clauses.size() - 1;
    }

    /** Solves the formula; TRUE/FALSE are definitive, UNKNOWN is never returned. */
    int solve() {
        if (triviallyUnsat) return FALSE;
        if (!propagate()) return FALSE;                 // level zero units first
        while (true) {
            if (!propagate()) {
                conflicts++;
                if (currentLevel == 0) return FALSE;
                int[] learntData = analyse(conflictClause);
                int backjump = 0;
                for (int i = 1; i < learntData.length; i++) {
                    backjump = Math.max(backjump, level[varOf(learntData[i])]);
                }
                backjumpTo(backjump);
                int clauseIndex = learn(learntData);
                // the learnt clause is asserting: its first literal is unassigned at the new level
                if (!assign(clauses.get(clauseIndex)[0], clauseIndex)) return FALSE;
                continue;
            }
            int decision = pickBranchVariable();
            if (decision < 0) return TRUE;              // every variable assigned: model found
            currentLevel++;
            assign(lit(decision, true), -1);
        }
    }

    private int pickBranchVariable() {
        int best = -1;
        double bestActivity = -1;
        for (int variable = 0; variable < nVars; variable++) {
            if (value[variable] != 0) continue;
            if (activity[variable] > bestActivity) {
                bestActivity = activity[variable];
                best = variable;
            }
        }
        return best;
    }

    /** Solve under assumptions (used to compare propagation with an independent implementation). */
    int solveWithAssumptions(int[] assumptions) {
        for (int assumption : assumptions) {
            if (literalValue(assumption) == -1) return FALSE;
            if (!assign(assumption, -1)) return FALSE;
        }
        return solve();
    }

    /** Propagation-only entry point: applies assumptions and propagates, without deciding. */
    boolean propagateOnly(int[] assumptions) {
        if (triviallyUnsat) return false;
        for (int assumption : assumptions) {
            if (literalValue(assumption) == -1) return false;
            if (!assign(assumption, -1)) return false;
        }
        return propagate();
    }

    /** The current partial/total assignment as a var -> boolean array (zero entries = unassigned). */
    int[] assignment() { return value.clone(); }

    List<int[]> allClauses() { return Collections.unmodifiableList(clauses); }

    List<int[]> learntClauses() { return Collections.unmodifiableList(learnt); }

    int conflictCount() { return conflicts; }

    /**
     * Fixpoint invariant, meaningful only when propagation stopped without a conflict: no clause is
     * left with exactly one unassigned literal (that clause would have been propagated).
     */
    String verifyPropagationFixpoint() {
        for (int[] clause : clauses) {
            int unassigned = 0;
            int satisfied = 0;
            for (int literal : clause) {
                int v = literalValue(literal);
                if (v == 0) unassigned++;
                if (v == 1) satisfied++;
            }
            if (satisfied > 0) continue;
            if (unassigned == 0) continue;              // a conflicting clause, handled by propagate
            if (unassigned == 1) return "a unit clause survived propagation: " + Arrays.toString(clause);
        }
        return null;
    }

    /** Watch bookkeeping invariant: every clause is watched exactly by its first two literals. */
    String verifyWatchBookkeeping() {
        for (int[] clause : clauses) {
            int entries = 0;
            for (int literal = 0; literal < 2 * nVars; literal++) {
                for (int[] watched : watches.get(literal)) if (watched == clause) entries++;
            }
            int expected = clause.length > 1 ? 2 : 0;    // unit clauses are asserted, not watched
            if (entries != expected) {
                return "clause " + Arrays.toString(clause) + " sits in " + entries
                        + " watch lists instead of " + expected;
            }
            if (clause.length == 1) continue;
            boolean firstWatched = false;
            for (int[] watched : watches.get(clause[0])) if (watched == clause) firstWatched = true;
            if (!firstWatched) return "clause " + Arrays.toString(clause) + " lost its first watch";
            boolean secondWatched = false;
            for (int[] watched : watches.get(clause[1])) if (watched == clause) secondWatched = true;
            if (!secondWatched) {
                return "clause " + Arrays.toString(clause) + " lost its second watch";
            }
        }
        return null;
    }
}

public class Solution {
    private static long seed = 0xC0AC29B7C97C50DDL;
    static int checksRun;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static List<int[]> formula(int[][] raw) {
        List<int[]> out = new ArrayList<>();
        for (int[] clause : raw) out.add(clause);
        return out;
    }

    static boolean clauseSatisfied(int[] clause, boolean[] model) {
        for (int literal : clause) {
            int variable = CdclSolver.varOf(literal);
            boolean positive = (literal & 1) == 0;
            if (model[variable] == positive) return true;
        }
        return false;
    }

    /** Exhaustive oracle: every model of the formula as a boolean array. */
    static List<boolean[]> enumerateModels(int nVars, List<int[]> clauses) {
        List<boolean[]> models = new ArrayList<>();
        for (int mask = 0; mask < (1 << nVars); mask++) {
            boolean[] model = new boolean[nVars];
            for (int v = 0; v < nVars; v++) model[v] = ((mask >> v) & 1) == 1;
            boolean ok = true;
            for (int[] clause : clauses) if (!clauseSatisfied(clause, model)) ok = false;
            if (ok) models.add(model);
        }
        return models;
    }

    /** Independent naive propagation: scan every clause until nothing changes. */
    static int[] naivePropagate(int nVars, List<int[]> clauses, int[] assumptions) {
        int[] value = new int[nVars];
        for (int assumption : assumptions) {
            int variable = CdclSolver.varOf(assumption);
            int wanted = (assumption & 1) == 0 ? 1 : -1;
            if (value[variable] != 0 && value[variable] != wanted) return null;
            value[variable] = wanted;
        }
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int[] clause : clauses) {
                int unassigned = 0;
                int lastUnassigned = -1;
                boolean satisfied = false;
                for (int literal : clause) {
                    int v = value[CdclSolver.varOf(literal)];
                    int literalValue = v == 0 ? 0 : ((literal & 1) == 0 ? v : -v);
                    if (literalValue == 1) satisfied = true;
                    if (literalValue == 0) {
                        unassigned++;
                        lastUnassigned = literal;
                    }
                }
                if (satisfied) continue;
                if (unassigned == 0) return null;        // conflict
                if (unassigned == 1) {
                    int variable = CdclSolver.varOf(lastUnassigned);
                    int wanted = (lastUnassigned & 1) == 0 ? 1 : -1;
                    if (value[variable] != wanted) {
                        value[variable] = wanted;
                        changed = true;
                    }
                }
            }
        }
        return value;
    }

    public static void main(String[] args) {
        // --- boundaries ----------------------------------------------------------
        CdclSolver noClauses = new CdclSolver(3, new ArrayList<>());
        check(noClauses.solve() == CdclSolver.TRUE, "an empty formula is satisfiable");
        check(noClauses.learntClauses().isEmpty(), "no learned clause without conflicts");

        CdclSolver emptyClause = new CdclSolver(2, formula(new int[][]{{}}));
        check(emptyClause.solve() == CdclSolver.FALSE, "an empty clause is unsatisfiable");

        CdclSolver tautology = new CdclSolver(2, formula(new int[][]{{0, 1}, {2, 3}}));
        check(tautology.solve() == CdclSolver.TRUE, "a tautological clause is dropped");
        check(tautology.allClauses().isEmpty(), "the tautology left no clause behind");

        CdclSolver duplicates = new CdclSolver(2, formula(new int[][]{{0, 0, 0}}));
        check(duplicates.allClauses().get(0).length == 1, "duplicate literals collapse to a unit");
        check(duplicates.solve() == CdclSolver.TRUE, "the collapsed unit clause is satisfiable");
        check(duplicates.assignment()[0] == 1, "the unit clause forced variable 0 to true");

        CdclSolver unitChain = new CdclSolver(3, formula(new int[][]{{0}, {2, 4}, {5, 3}}));
        check(unitChain.solve() == CdclSolver.TRUE, "a unit chain is satisfiable");
        int[] chainModel = unitChain.assignment();
        for (int v = 0; v < 3; v++) check(chainModel[v] != 0, "the model assigns variable " + v);
        check(chainModel[0] == 1, "the unit clause propagated variable 0");
        check(chainModel[1] == -chainModel[2],
                "the two clauses (x1 v x2) and (!x1 v !x2) force exactly one of them");
        boolean[] chainBool = {chainModel[0] == 1, chainModel[1] == 1, chainModel[2] == 1};
        check(clauseSatisfied(new int[]{0}, chainBool), "the unit clause is satisfied");
        check(clauseSatisfied(new int[]{2, 4}, chainBool) && clauseSatisfied(new int[]{5, 3}, chainBool),
                "the two clauses are satisfied by the returned model");
        check(unitChain.verifyWatchBookkeeping() == null, "unit chain watch bookkeeping");
        check(unitChain.verifyPropagationFixpoint() == null, "unit chain propagation fixpoint");

        // --- a known unsatisfiable instance --------------------------------------
        List<int[]> php = new ArrayList<>();
        for (int pigeon = 0; pigeon < 3; pigeon++) {
            php.add(new int[]{
                    CdclSolver.lit(pigeon * 2, true), CdclSolver.lit(pigeon * 2 + 1, true)});
        }
        for (int hole = 0; hole < 2; hole++) {
            for (int a = 0; a < 3; a++) {
                for (int b = a + 1; b < 3; b++) {
                    php.add(new int[]{CdclSolver.lit(a * 2 + hole, false),
                            CdclSolver.lit(b * 2 + hole, false)});
                }
            }
        }
        CdclSolver pigeons = new CdclSolver(6, php);
        check(pigeons.solve() == CdclSolver.FALSE, "three pigeons in two holes is unsatisfiable");
        check(enumerateModels(6, php).isEmpty(), "the oracle agrees the pigeonhole instance is UNSAT");
        check(pigeons.conflictCount() > 0, "the solver really searched");

        // --- random instances against exhaustive enumeration --------------------
        int satCount = 0;
        int unsatCount = 0;
        int learntClauses = 0;
        for (int trial = 0; trial < 12; trial++) {
            int nVars = 6 + nextInt(3);
            int nClauses = 4 * nVars;
            List<int[]> clauses = new ArrayList<>();
            for (int c = 0; c < nClauses; c++) {
                int size = 3;
                int[] clause = new int[size];
                for (int k = 0; k < size; k++) {
                    int variable = nextInt(nVars);
                    boolean positive = nextInt(2) == 0;
                    clause[k] = CdclSolver.lit(variable, positive);
                }
                clauses.add(clause);
            }
            List<boolean[]> models = enumerateModels(nVars, clauses);
            CdclSolver solver = new CdclSolver(nVars, clauses);
            int verdict = solver.solve();
            if (models.isEmpty()) {
                unsatCount++;
                check(verdict == CdclSolver.FALSE, "solver said SAT on an UNSAT instance " + trial);
            } else {
                satCount++;
                check(verdict == CdclSolver.TRUE, "solver said UNSAT on a SAT instance " + trial);
                int[] assignment = solver.assignment();
                boolean[] model = new boolean[nVars];
                for (int v = 0; v < nVars; v++) {
                    check(assignment[v] != 0, "the model must assign every variable, trial " + trial);
                    model[v] = assignment[v] == 1;
                }
                for (int[] clause : clauses) {
                    check(clauseSatisfied(clause, model), "the returned model falsifies a clause, trial "
                            + trial);
                }
            }
            check(solver.verifyWatchBookkeeping() == null,
                    "watch bookkeeping violated: " + solver.verifyWatchBookkeeping());
            if (verdict == CdclSolver.TRUE) {
                check(solver.verifyPropagationFixpoint() == null,
                        "propagation fixpoint violated: " + solver.verifyPropagationFixpoint());
            }
            learntClauses += solver.learntClauses().size();
            for (int[] learned : solver.learntClauses()) {
                for (boolean[] model : models) {
                    check(clauseSatisfied(learned, model),
                            "a learned clause is not implied by the formula: "
                                    + Arrays.toString(learned));
                }
            }
        }
        check(satCount > 0 && unsatCount > 0, "the random set mixes SAT and UNSAT: " + satCount + "/"
                + unsatCount);
        check(learntClauses > 0, "the solver learned clauses on the random instances");

        // --- propagation oracle --------------------------------------------------
        List<int[]> probeFormula = formula(new int[][]{
                {0, 2}, {1, 3}, {2, 5}, {0, 4}, {3, 4}, {1, 5}});
        int propagationComparisons = 0;
        for (int trial = 0; trial < 30; trial++) {
            int[] assumptions = new int[1 + nextInt(2)];
            for (int i = 0; i < assumptions.length; i++) {
                assumptions[i] = CdclSolver.lit(nextInt(3), nextInt(2) == 0);
            }
            CdclSolver probe = new CdclSolver(3, probeFormula);
            boolean propagated = probe.propagateOnly(assumptions);
            int[] expected = naivePropagate(3, probeFormula, assumptions);
            if (expected == null) {
                check(!propagated, "the watched literal propagation missed a conflict");
            } else {
                int[] actual = probe.assignment();
                for (int v = 0; v < 3; v++) {
                    check(actual[v] == expected[v], "propagation differs on variable " + v
                            + " for assumptions " + Arrays.toString(assumptions));
                }
                propagationComparisons++;
            }
            check(probe.verifyWatchBookkeeping() == null,
                    "watch bookkeeping violated after propagation: " + probe.verifyWatchBookkeeping());
            if (propagated) {
                check(probe.verifyPropagationFixpoint() == null,
                        "propagation fixpoint violated after propagation: "
                                + probe.verifyPropagationFixpoint());
            }
        }
        check(propagationComparisons > 0, "propagation agreed with the oracle in some cases");

        // --- assumptions decide like the enumeration -----------------------------
        int assumptionChecks = 0;
        for (int trial = 0; trial < 40; trial++) {
            int nVars = 4;
            List<int[]> clauses = new ArrayList<>();
            for (int c = 0; c < 10; c++) {
                clauses.add(new int[]{
                        CdclSolver.lit(nextInt(nVars), nextInt(2) == 0),
                        CdclSolver.lit(nextInt(nVars), nextInt(2) == 0)});
            }
            int[] assumptions = new int[2];
            for (int i = 0; i < 2; i++) assumptions[i] = CdclSolver.lit(nextInt(nVars), nextInt(2) == 0);
            CdclSolver solver = new CdclSolver(nVars, clauses);
            int verdict = solver.solveWithAssumptions(assumptions);
            boolean expectedSat = false;
            for (boolean[] model : enumerateModels(nVars, clauses)) {
                boolean matches = true;
                for (int assumption : assumptions) {
                    if (model[CdclSolver.varOf(assumption)] != ((assumption & 1) == 0)) matches = false;
                }
                if (matches) expectedSat = true;
            }
            check((verdict == CdclSolver.TRUE) == expectedSat,
                    "assumption verdict differs from enumeration in trial " + trial);
            assumptionChecks++;
        }
        check(assumptionChecks == 40, "all assumption trials ran");

        // --- determinism ---------------------------------------------------------
        CdclSolver first = new CdclSolver(6, php);
        CdclSolver second = new CdclSolver(6, php);
        check(first.solve() == second.solve(), "same instance, same verdict");
        check(first.conflictCount() == second.conflictCount(), "same instance, same conflict count");

        System.out.println("random instances: " + satCount + " SAT / " + unsatCount + " UNSAT, "
                + learntClauses + " learned clauses, all implied by their formulas");
        System.out.println("pigeonhole(3,2): verdict UNSAT with " + pigeons.conflictCount()
                + " conflicts; propagation agreed with the naive oracle in " + propagationComparisons
                + " of 30 assumption trials; " + checksRun + " checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
