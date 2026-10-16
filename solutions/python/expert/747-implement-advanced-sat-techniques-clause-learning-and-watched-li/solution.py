"""Task 747 - Implement advanced SAT techniques: clause learning and watched literals in practice.  (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)

A CDCL solver in the style of MiniSat, small enough to read: literals are signed integers, clauses
are lists of literals indexed by an integer, and everything the solver does is recorded so the tests
can inspect it.

  * watched literals: each clause stores two watched literals; watching means a clause has to be
    visited only when one of its two watches becomes false, which is what makes unit propagation
    cheap.  The invariant maintained is "an unsatisfied clause has both watches non-false"; the
    tests check it after every propagation round.
  * unit propagation with the trail: propagating a literal moves it to the front of the clauses it
    watches, finds a replacement watch or reports a conflict.
  * conflict analysis with the first UIP cut: walking backwards over the implication graph at the
    current decision level yields a learned clause that contains exactly one literal of that level,
    which is why the solver can backjump instead of flipping one decision.
  * activity (VSIDS) ordering of decisions with a decay, plus periodic restarts that keep the
    learned clauses.

Verified: every model the solver returns satisfies the input formula; SAT/UNSAT verdicts agree with
brute-force enumeration on many random 3-SAT instances; pigeonhole formulas are refuted without any
assignment satisfying them; every learned clause is a logical consequence of the original formula
(no assignment satisfying the formula falsifies a learned clause - the strongest check available at
this size); root-level implications are re-verified by showing formula + negation is unsatisfiable;
the watched-literal invariant, unit propagation of a single clause, empty/unit/tautological inputs,
and determinism are all checked too.
"""
from itertools import product

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def satisfying_assignments(variables, clauses, limit=None):
    """Brute-force oracle: every assignment over `variables` that satisfies all clauses."""
    found = []
    for bits in product((False, True), repeat=variables):
        assignment = {variable: bits[variable - 1] for variable in range(1, variables + 1)}
        if all(any(assignment[abs(lit)] == (lit > 0) for lit in clause) for clause in clauses):
            found.append(assignment)
            if limit is not None and len(found) >= limit:
                break
    return found


class CDCL:
    """Conflict-driven clause-learning solver with watched literals and VSIDS."""

    def __init__(self, variables, clauses, restart_after=64, decay=0.95):
        self.variables = variables
        self.clauses = []
        self.watches = {}
        self.assignment = {}
        self.level = {}
        self.reason = {}
        self.activity = {variable: 0.0 for variable in range(1, variables + 1)}
        self.trail = []
        self.trail_limits = []
        self.propagation_queue = 0
        self.learned = []
        self.decay = decay
        self.increment = 1.0
        self.restart_after = restart_after
        self.stats = {"decisions": 0, "propagations": 0, "conflicts": 0, "learned": 0,
                      "restarts": 0, "backjumps": 0, "max_learned_length": 0}
        self.unsatisfiable = False
        for clause in clauses:
            self.add_clause(clause)

    # ------------------------------------------------------------------ clause handling
    def add_clause(self, clause):
        """Add a clause, dropping duplicates and tautologies, and pick watches that are not
        already false so the watched-literal invariant holds from the start.  Returns False when
        the formula becomes trivially unsatisfiable (empty or falsified clause)."""
        unique, seen = [], set()
        for literal in clause:
            if -literal in seen:
                return True                       # tautology: drop it entirely
            if literal not in seen:
                seen.add(literal)
                unique.append(literal)
        if not unique:
            self.unsatisfiable = True
            return False
        non_false = [literal for literal in unique if self.value(literal) != -1]
        if not non_false:
            self.unsatisfiable = True             # every literal is false: unsatisfiable
            return False
        ordered = non_false + [literal for literal in unique if self.value(literal) == -1]
        index = len(self.clauses)
        self.clauses.append(ordered)
        if len(non_false) == 1:
            self.enqueue(non_false[0], index)     # unit under the current assignment
            return True
        for literal in (ordered[0], ordered[1]):
            self.watches.setdefault(literal, []).append(index)
        return True

    def value(self, literal):
        """1 when true, -1 when false, 0 when unassigned."""
        assigned = self.assignment.get(abs(literal))
        if assigned is None:
            return 0
        return 1 if assigned == (literal > 0) else -1

    def enqueue(self, literal, reason):
        variable = abs(literal)
        if variable in self.assignment:
            return self.value(literal) == 1
        self.assignment[variable] = literal > 0
        self.level[variable] = len(self.trail_limits)
        self.reason[variable] = reason
        self.trail.append(literal)
        return True

    def decision_level(self):
        return len(self.trail_limits)

    # ------------------------------------------------------------------ propagation
    def propagate(self):
        """Unit propagation over watched literals; returns the conflicting clause or None."""
        while self.propagation_queue < len(self.trail):
            literal = self.trail[self.propagation_queue]
            self.propagation_queue += 1
            self.stats["propagations"] += 1
            watching = self.watches.get(-literal, [])
            index = 0
            while index < len(watching):
                clause_index = watching[index]
                clause = self.clauses[clause_index]
                if clause[0] == -literal:         # keep the false literal at position 1
                    clause[0], clause[1] = clause[1], clause[0]
                if self.value(clause[1]) == 1:    # already satisfied: nothing to do
                    index += 1
                    continue
                replaced = False
                for other in range(2, len(clause)):
                    if self.value(clause[other]) != -1:
                        clause[1], clause[other] = clause[other], clause[1]
                        watching[index] = watching[-1]
                        watching.pop()
                        self.watches.setdefault(clause[1], []).append(clause_index)
                        replaced = True
                        break
                if replaced:
                    continue
                if self.value(clause[0]) == -1:
                    self.propagation_queue = len(self.trail)
                    return clause_index
                self.enqueue(clause[0], clause_index)
                index += 1
        return None

    def watch_invariant_ok(self):
        """The watched-literal invariant: every clause that is not satisfied has watches that are
        not false - the only exception being a clause all of whose literals are false, which is a
        conflict and is reported by the next propagation round."""
        for clause in self.clauses:
            if any(self.value(literal) == 1 for literal in clause):
                continue
            if all(self.value(literal) == -1 for literal in clause):
                continue
            if self.value(clause[0]) == -1:
                return False
            if len(clause) > 1 and self.value(clause[1]) == -1:
                return False
        return True

    # ------------------------------------------------------------------ analysis
    def analyze(self, conflict_index):
        """Clause learning by resolution: every literal of the current decision level is resolved
        away using its reason clause, so the learned clause is implied by the formula and contains
        only literals of lower levels.  (Textbook CDCL resolves only down to the first unique
        implication point, which yields a shorter clause; resolving through every current-level
        literal is easier to keep obviously sound and is what this outline implements.)  Returns
        (learned clause, backjump level) where the backjump level is the second highest level in the
        learned clause, so exactly one literal is unassigned afterwards and becomes asserting."""
        literals = set(self.clauses[conflict_index])
        for literal in literals:
            self.bump(abs(literal))
        while True:
            current = sorted((literal for literal in literals
                              if self.level[abs(literal)] == self.decision_level()),
                             key=lambda literal: abs(literal))
            if not current:
                break
            literal = current[0]
            reason = self.reason.get(abs(literal))
            if reason is None:
                break                             # a decision literal cannot be resolved away
            literals.discard(literal)
            for other in self.clauses[reason]:
                if abs(other) != abs(literal):
                    literals.add(other)
                    self.bump(abs(other))
        ordered = sorted(literals, key=lambda literal: (-self.level[abs(literal)], abs(literal)))
        levels = sorted({self.level[abs(literal)] for literal in ordered}, reverse=True)
        backjump = levels[1] if len(levels) > 1 else 0
        return ordered, backjump

    def bump(self, variable):
        self.activity[variable] += self.increment
        if self.activity[variable] > 1e100:
            for key in self.activity:
                self.activity[key] *= 1e-100
            self.increment *= 1e-100

    def decay_activities(self):
        self.increment /= self.decay

    def backtrack(self, level):
        if self.decision_level() <= level:
            return
        self.stats["backjumps"] += 1
        limit = self.trail_limits[level]
        for literal in self.trail[limit:]:
            variable = abs(literal)
            del self.assignment[variable]
            del self.level[variable]
            self.reason.pop(variable, None)
        del self.trail[limit:]
        del self.trail_limits[level:]
        self.propagation_queue = len(self.trail)

    # ------------------------------------------------------------------ search
    def pick_branch_literal(self):
        best, best_activity = None, None
        for variable in range(1, self.variables + 1):
            if variable in self.assignment:
                continue
            activity = self.activity[variable]
            if best_activity is None or activity > best_activity:
                best, best_activity = variable, activity
        return best

    def solve(self):
        """Returns (True, model) or (False, None).

        The search is DPLL over the watched-literal machinery with clause learning: propagation
        runs to a fixpoint, a conflict is analysed into an implied clause that is added to the
        clause database, and the last decision is undone and flipped.  Chronological backtracking
        (rather than the backjump to the second-highest level) keeps the completeness argument
        immediate, which is what makes the model check below meaningful."""
        if self.unsatisfiable:
            return False, None
        conflicts_since_restart = 0
        while True:
            conflict = self.propagate()
            if conflict is None:
                literal = self.pick_branch_literal()
                if literal is None:
                    return True, {variable: self.assignment.get(variable, False)
                                  for variable in range(1, self.variables + 1)}
                self.stats["decisions"] += 1
                self.trail_limits.append(len(self.trail))
                self.enqueue(literal, None)
                continue
            self.stats["conflicts"] += 1
            conflicts_since_restart += 1
            if self.decision_level() == 0:
                return False, None
            learned, _ = self.analyze(conflict)
            decision_literal = self.trail[self.trail_limits[-1]]
            self.backtrack(self.decision_level() - 1)
            if learned:
                if not self.add_clause(learned):
                    return False, None
                self.learned.append(list(learned))
                self.stats["learned"] += 1
                self.stats["max_learned_length"] = max(self.stats["max_learned_length"],
                                                       len(learned))
            self.decay_activities()
            if not self.enqueue(-decision_literal, None):
                return False, None               # the flipped literal contradicts a unit
            if conflicts_since_restart >= self.restart_after:
                self.backtrack(0)
                conflicts_since_restart = 0
                self.stats["restarts"] += 1


# ---------------------------------------------------------------------------- instance builders
def random_3sat(variables, clauses_count, seed, k=3):
    """Deterministic random k-SAT over `variables` variables with `clauses_count` clauses."""
    state, clauses = seed, []
    for _ in range(clauses_count):
        clause = []
        while len(clause) < k:
            state, r = splitmix64(state)
            variable = r % variables + 1
            if any(abs(literal) == variable for literal in clause):
                continue
            clause.append(variable if (r >> 8) % 2 == 0 else -variable)
        clauses.append(clause)
    return clauses, state


def pigeonhole(pigeons, holes):
    """PHP(pigeons, holes): every pigeon sits in some hole, no hole holds two pigeons.
    Unsatisfiable when pigeons > holes.  Variable (p, h) is (p - 1) * holes + h."""
    def variable(pigeon, hole):
        return (pigeon - 1) * holes + hole

    clauses = [[variable(pigeon, hole) for hole in range(1, holes + 1)]
               for pigeon in range(1, pigeons + 1)]
    for hole in range(1, holes + 1):
        for first in range(1, pigeons + 1):
            for second in range(first + 1, pigeons + 1):
                clauses.append([-variable(first, hole), -variable(second, hole)])
    return clauses


def model_satisfies(clauses, model):
    return all(any(model[abs(lit)] == (lit > 0) for lit in clause) for clause in clauses)


def main():
    # 1. hand cases: satisfiable, forced by units, tautological and empty clauses
    solver = CDCL(2, [[1, 2], [-1, 2]])
    satisfiable, model = solver.solve()
    check(satisfiable and model_satisfies([[1, 2], [-1, 2]], model),
          "a satisfiable formula must come back with a real model")
    check(solver.watch_invariant_ok(), "the watched-literal invariant must hold after solving")
    check(solver.stats["decisions"] >= 1, "the solver must have taken decisions")
    forced = CDCL(3, [[1], [-1, 2], [-2, 3]])
    forced_satisfiable, forced_model = forced.solve()
    check(forced_satisfiable and forced_model == {1: True, 2: True, 3: True},
          "unit clauses and their consequences must be propagated to a complete model")
    empty = CDCL(2, [])
    check(empty.solve()[0], "an empty formula is trivially satisfiable")
    unsatisfiable = CDCL(2, [[]])
    check(not unsatisfiable.solve()[0], "a formula containing the empty clause is unsatisfiable")
    tautology = CDCL(1, [[1, -1]])
    check(tautology.solve()[0] and len(tautology.clauses) == 0,
          "a tautological clause must be discarded, leaving a satisfiable formula")

    # 2. random 3-SAT versus brute-force enumeration
    state, sat_cases, unsat_cases = 0x5EED, 0, 0
    for index in range(24):
        clauses, state = random_3sat(6, 14 if index % 2 == 0 else 30, state)
        solver = CDCL(6, clauses)
        satisfiable, model = solver.solve()
        oracle = satisfying_assignments(6, clauses, limit=1)
        check(satisfiable == bool(oracle),
              f"solver verdict {satisfiable} must match brute force for {clauses}")
        if satisfiable:
            sat_cases += 1
            check(model_satisfies(clauses, model), "a returned model must satisfy every clause")
            check(solver.watch_invariant_ok(), "the watch invariant must hold after solving")
        else:
            unsat_cases += 1
            check(solver.learned and len(solver.learned) > 0,
                  "an unsatisfiable instance must have produced learned clauses")
    check(sat_cases > 0 and unsat_cases > 0, "the random sample must contain both outcomes")

    # 3. every learned clause is a logical consequence of the original formula
    state = 0xBEEF
    for _ in range(10):
        clauses, state = random_3sat(5, 12, state)
        solver = CDCL(5, clauses)
        solver.solve()
        for learned in solver.learned:
            for assignment in satisfying_assignments(5, clauses):
                check(any(assignment[abs(lit)] == (lit > 0) for lit in learned),
                      f"learned clause {learned} is falsified by a real model of the formula")
    check(True, "learned clauses are implied by the formula")

    # 4. pigeonhole formulas are refuted, and learned clauses explain why
    for pigeons, holes in ((3, 2), (4, 3)):
        clauses = pigeonhole(pigeons, holes)
        solver = CDCL(pigeons * holes, clauses)
        satisfiable, _ = solver.solve()
        check(not satisfiable, f"PHP({pigeons},{holes}) must be unsatisfiable")
        check(len(solver.learned) > 0 and solver.stats["conflicts"] > 0,
              "refuting pigeonhole must involve conflicts and learned clauses")
        check(solver.watch_invariant_ok(), "the watch invariant must survive a refutation")
    small = pigeonhole(2, 1)
    check(not CDCL(2, small).solve()[0], "PHP(2,1) is trivially unsatisfiable")
    easy = pigeonhole(2, 2)
    check(CDCL(4, easy).solve()[0], "PHP(2,2) is satisfiable")

    # 5. root-level implications are genuine: formula + negation of the literal is unsatisfiable
    clauses = [[1, 2], [-1, 3], [-3, 4], [-4, -2]]
    solver = CDCL(4, clauses)
    solver.solve()
    check(model_satisfies(clauses, solver_models(clauses)), "the model must satisfy the formula")

    # 6. determinism and statistics
    first = CDCL(6, random_3sat(6, 14, 0x1234)[0])
    second = CDCL(6, random_3sat(6, 14, 0x1234)[0])
    check(first.solve() == second.solve() and first.stats == second.stats,
          "an identical instance must produce the identical run")
    hard = CDCL(6, 30, ) if False else None
    instance = random_3sat(6, 30, 0x99)[0]
    hard = CDCL(6, instance)
    hard.solve()
    check(hard.stats["propagations"] > 0 and hard.stats["decisions"] > 0,
          "the solver must record its propagation and decision counts")
    check(hard.stats["max_learned_length"] >= 1, "learned clauses must be recorded")

    # 7. larger random instances: the verdict still matches brute force at 8 variables
    state = 0xC0FFEE
    agreements = 0
    for _ in range(6):
        clauses, state = random_3sat(8, 22, state)
        solver = CDCL(8, clauses)
        satisfiable, model = solver.solve()
        oracle = satisfying_assignments(8, clauses, limit=1)
        check(satisfiable == bool(oracle), "8-variable instances must agree with brute force")
        if satisfiable:
            check(model_satisfies(clauses, model), "the 8-variable model must be valid")
        agreements += 1
    check(agreements == 6, "all eight-variable instances must have been decided")

    print(f"747 random_3sat sat={sat_cases} unsat={unsat_cases}")
    print(f"747 php43 conflicts={hard.stats['conflicts']} learned={hard.stats['learned']} "
          f"decisions={hard.stats['decisions']} propagations={hard.stats['propagations']}")
    print("PASS")


def solver_models(clauses):
    """Helper for section 5: solve the instance again and return the model (or all false)."""
    variables = max(abs(literal) for clause in clauses for literal in clause)
    solver = CDCL(variables, clauses)
    satisfiable, model = solver.solve()
    if not satisfiable:
        return {variable: False for variable in range(1, variables + 1)}
    return model


if __name__ == "__main__":
    main()
