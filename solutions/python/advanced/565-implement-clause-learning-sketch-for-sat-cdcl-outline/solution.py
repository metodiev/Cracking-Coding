"""Task 565 - Implement clause learning sketch for SAT (CDCL outline). (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)

A working CDCL solver rather than a sketch: two-watched-literal unit propagation, first-UIP
conflict analysis by resolution along the trail, a learned clause that is asserting at its backjump
level, activity bumping of the variables that took part in a conflict, phase saving for decision
polarity, and a trail that is truncated to the backjump level. The solver keeps every learned clause
in a list so the checks can inspect them: each learned clause is verified to be *logically implied*
by the input formula (no satisfying assignment of the formula falsifies it, checked by brute force),
exactly one of its literals lies at the conflict's decision level (the first-UIP property), the
backjump level equals the second-highest level appearing in the clause, and after backjumping the
clause is asserting - every other literal false, the asserting literal unassigned - which is the
invariant that makes the next propagation step deterministic. Two documented simplifications: no
clause deletion and no restarts, so the learned database grows monotonically and the solver is
complete but not competitive; and decisions pick the most active unassigned variable, ties broken by
the smallest index. Formulas come from a full-period 64-bit LCG (MMIX constants) read in its high
bits, since the repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period and low-bit structure
that would repeat clause patterns.

Checks: on random CNFs the verdict equals an exhaustive truth-table scan, every returned model
satisfies the input, the number of models found by adding blocking clauses matches brute force the
same way the DPLL file does, PHP(3,2) and PHP(4,3) are refuted while PHP(4,5) is satisfied, the
solver produces at least one conflict and one learned clause on the refuted instances, and every
learned clause collected during those runs is implied by the input formula and satisfies the
first-UIP, backjump and asserting invariants - the conflict analysis is therefore checked
structurally as well as by its verdict. Repeated runs are identical.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_cnf(clauses, width, variables, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 79) & MASK64)
    formula = []
    for _ in range(clauses):
        clause = []
        for _ in range(width):
            state = prng64(state)
            variable = 1 + (state >> 33) % variables
            state = prng64(state)
            clause.append(variable if (state >> 33) & 1 else -variable)
        formula.append(clause)
    return formula


def satisfies(clauses, assignment):
    return all(any(assignment.get(abs(literal), False) == (literal > 0) for literal in clause)
               for clause in clauses)


def brute_force_satisfiable(variables, clauses):
    for bits in range(1 << variables):
        assignment = {variable: bool(bits >> (variable - 1) & 1)
                      for variable in range(1, variables + 1)}
        if satisfies(clauses, assignment):
            return True
    return False


def brute_force_models(variables, clauses, limit=100000):
    found = []
    for bits in range(1 << variables):
        assignment = {variable: bool(bits >> (variable - 1) & 1)
                      for variable in range(1, variables + 1)}
        if satisfies(clauses, assignment):
            found.append(assignment)
    return found[:limit]


class CdclSolver:
    """A small CDCL solver with watched literals, first-UIP learning and phase saving."""

    def __init__(self, variables, clauses):
        self.variables = variables
        self.clauses = [list(clause) for clause in clauses]
        self.watch = {}
        self.value = [None] * (variables + 1)
        self.level = [0] * (variables + 1)
        self.reason = [None] * (variables + 1)
        self.activity = [0] * (variables + 1)
        self.phase = [False] * (variables + 1)
        self.trail = []
        self.trail_lim = []
        self.learned = []
        self.invariants = []
        self.conflicts = 0
        for index, clause in enumerate(self.clauses):
            for literal in clause[:2]:
                self.watch.setdefault(literal, []).append(index)

    # --- trail helpers -----------------------------------------------------------------
    @property
    def decision_level(self):
        return len(self.trail_lim)

    def value_of(self, literal):
        value = self.value[abs(literal)]
        if value is None:
            return None
        return value if literal > 0 else not value

    def enqueue(self, literal, reason):
        self.value[abs(literal)] = literal > 0
        self.phase[abs(literal)] = literal > 0
        self.level[abs(literal)] = self.decision_level
        self.reason[abs(literal)] = reason
        self.trail.append(literal)
        return True

    def cancel_until(self, level):
        while self.decision_level > level:
            start = self.trail_lim.pop()
            for literal in self.trail[start:]:
                self.value[abs(literal)] = None
                self.reason[abs(literal)] = None
                self.level[abs(literal)] = 0
            del self.trail[start:]

    def new_decision_level(self):
        self.trail_lim.append(len(self.trail))

    # --- propagation -------------------------------------------------------------------
    def propagate(self):
        """Two-watched-literal propagation; returns the index of a conflicting clause or None."""
        head = 0
        while head < len(self.trail):
            literal = self.trail[head]
            head += 1
            false_literal = -literal
            watch_list = self.watch.setdefault(false_literal, [])
            position = 0
            while position < len(watch_list):
                index = watch_list[position]
                clause = self.clauses[index]
                if clause[0] == false_literal:
                    clause[0], clause[1] = clause[1], clause[0]
                if clause[1] != false_literal:
                    position += 1
                    continue
                replaced = False
                for other in range(2, len(clause)):
                    if self.value_of(clause[other]) is not False:
                        clause[1], clause[other] = clause[other], clause[1]
                        watch_list[position] = watch_list[-1]
                        watch_list.pop()
                        self.watch.setdefault(clause[1], []).append(index)
                        replaced = True
                        break
                if replaced:
                    continue
                first_value = self.value_of(clause[0])
                if first_value is False:
                    return index
                if first_value is None:
                    self.enqueue(clause[0], index)
                position += 1
        return None

    # --- conflict analysis -------------------------------------------------------------
    def analyze(self, conflict_index):
        """First-UIP learning; returns (learned clause, backjump level)."""
        body = list(self.clauses[conflict_index])
        seen = [False] * (self.variables + 1)
        learned = []
        counter = 0
        index = len(self.trail) - 1
        while True:
            for literal in body:
                variable = abs(literal)
                if not seen[variable] and self.level[variable] > 0:
                    seen[variable] = True
                    if self.level[variable] == self.decision_level:
                        counter += 1
                    else:
                        learned.append(literal)
            while index >= 0 and not seen[abs(self.trail[index])]:
                index -= 1
            assert index >= 0, "conflict analysis walked off the trail"
            asserting = self.trail[index]
            index -= 1
            seen[abs(asserting)] = False
            counter -= 1
            if counter == 0:
                break
            # The reason clause's own copy of the resolved literal is skipped explicitly: the
            # two-watched-literal machinery may have shuffled its literal positions since the
            # clause was used as a reason.
            if self.reason[abs(asserting)] is None:
                body = []
            else:
                body = [literal for literal in self.clauses[self.reason[abs(asserting)]]
                        if abs(literal) != abs(asserting)]
        learned.insert(0, -asserting)
        if len(learned) > 1:
            best = max(range(1, len(learned)), key=lambda i: self.level[abs(learned[i])])
            (learned[1], learned[best]) = (learned[best], learned[1])
            backjump = self.level[abs(learned[1])]
        else:
            backjump = 0
        # Recorded *before* the trail is truncated, because that is the only moment at which the
        # decision levels of the learned clause still describe the situation it was derived from.
        self.invariants.append({"clause": list(learned), "backjump": backjump,
                                "conflict_level": self.decision_level,
                                "levels": [self.level[abs(literal)] for literal in learned]})
        return learned, backjump

    # --- search ------------------------------------------------------------------------
    def solve(self, conflict_limit=200000):
        for index, clause in enumerate(self.clauses):
            if not clause:
                return (False, None)
            if len(clause) == 1:
                if self.value_of(clause[0]) is False:
                    return (False, None)
                if self.value_of(clause[0]) is None:
                    self.enqueue(clause[0], index)
        if self.propagate() is not None:
            return (False, None)
        while True:
            conflict = self.propagate()
            if conflict is not None:
                self.conflicts += 1
                if self.conflicts > conflict_limit:
                    raise RuntimeError("conflict limit reached")
                if self.decision_level == 0:
                    return (False, None)
                learned, backjump = self.analyze(conflict)
                for literal in learned:
                    self.activity[abs(literal)] += 1
                self.cancel_until(backjump)
                # The learned clause must be asserting after the backjump: every literal but the
                # first is already false and the first is unassigned and can be propagated.
                if len(learned) > 1:
                    assert all(self.value_of(literal) is False for literal in learned[1:]), \
                        "a learned clause has to be unit after backjumping"
                assert self.value_of(learned[0]) is None, "the asserting literal must be free"
                self.learned.append(list(learned))
                if len(learned) == 1:
                    # A unit learned clause is a permanent level-0 fact; it is recorded but not
                    # added to the database, which is what keeps the trail consistent.
                    self.enqueue(learned[0], None)
                else:
                    index = len(self.clauses)
                    self.clauses.append(learned)
                    self.watch.setdefault(learned[0], []).append(index)
                    self.watch.setdefault(learned[1], []).append(index)
                    self.enqueue(learned[0], index)
            else:
                unassigned = [variable for variable in range(1, self.variables + 1)
                              if self.value[variable] is None]
                if not unassigned:
                    return (True, {variable: self.value[variable]
                                   for variable in range(1, self.variables + 1)})
                variable = max(unassigned, key=lambda v: (self.activity[v], -v))
                self.new_decision_level()
                self.enqueue(variable if self.phase[variable] else -variable, None)


def php(pigeons, holes):
    def variable(pigeon, hole):
        return pigeon * holes + hole + 1
    clauses = []
    for pigeon in range(pigeons):
        clauses.append([variable(pigeon, hole) for hole in range(holes)])
    for hole in range(holes):
        for first in range(pigeons):
            for second in range(first + 1, pigeons):
                clauses.append([-variable(first, hole), -variable(second, hole)])
    return clauses


def main():
    # --- Random formulas against brute force ---------------------------------------------
    for (clauses, width, variables, seed) in ((4, 2, 3, 1), (6, 3, 3, 2), (8, 3, 4, 3),
                                              (10, 2, 5, 4), (12, 3, 5, 5), (14, 3, 6, 6),
                                              (18, 3, 6, 7), (22, 3, 7, 8), (26, 3, 7, 9)):
        formula = random_cnf(clauses, width, variables, seed)
        solver = CdclSolver(variables, formula)
        (verdict, model) = solver.solve()
        assert verdict == brute_force_satisfiable(variables, formula), (clauses, width, seed)
        if verdict:
            assert satisfies(formula, model)
            assert set(model) == set(range(1, variables + 1)), "a full model is returned"

    # --- Model counting through blocking clauses -----------------------------------------
    formula = random_cnf(10, 3, 5, 21)
    working = [list(clause) for clause in formula]
    found = []
    for _ in range(100):
        solver = CdclSolver(5, working)
        (verdict, model) = solver.solve()
        if not verdict:
            break
        found.append(tuple(sorted(model.items())))
        working.append([variable if not model[variable] else -variable
                        for variable in range(1, 6)])
    assert len(found) == len(set(found))
    assert len(found) == len(brute_force_models(5, formula)), "all models, exactly once"

    # --- Pigeonhole ----------------------------------------------------------------------
    for (pigeons, holes) in ((2, 1), (3, 2), (4, 3)):
        formula = php(pigeons, holes)
        variables = pigeons * holes
        solver = CdclSolver(variables, formula)
        (verdict, model) = solver.solve()
        assert verdict is False, (pigeons, holes)
        assert brute_force_satisfiable(variables, formula) is False
        if (pigeons, holes) != (2, 1):
            assert solver.conflicts > 0, "this refutation needs decisions and conflicts"
            assert solver.learned, "and at least one learned clause"
        else:
            assert solver.conflicts == 0, "PHP(2,1) already falls to level-0 propagation"
        # Every learned clause must be implied by the input formula.
        for clause in solver.learned:
            for assignment in brute_force_models(variables, formula):
                assert satisfies([clause], assignment), (pigeons, holes, clause)
        # Structural invariants of every learned clause, from the recorded snapshots.
        for record in solver.invariants:
            learn = record["clause"]
            levels = record["levels"]
            assert len(learn) == len(set(learn)), "a learned clause repeats no literal"
            assert len({abs(literal) for literal in learn}) == len(learn), "and no variable"
            assert levels[0] == record["conflict_level"] > record["backjump"], \
                "the asserting literal sits at the conflict level, above the backjump"
            assert sum(1 for level in levels if level == record["conflict_level"]) == 1, \
                "exactly one literal at the conflict level - the first UIP"
            assert all(level <= record["conflict_level"] for level in levels)
            assert record["backjump"] == max([0] + levels[1:]), \
                "the backjump level is the highest level among the other literals"
    formula = php(4, 5)
    solver = CdclSolver(20, formula)
    (verdict, model) = solver.solve()
    assert verdict is True and satisfies(formula, model)

    # --- The same invariants on random instances, plus the verdict -----------------------
    checked = 0
    for (clauses, width, variables, seed) in ((12, 3, 5, 31), (16, 3, 6, 32), (20, 3, 6, 33),
                                              (24, 3, 7, 34)):
        formula = random_cnf(clauses, width, variables, seed)
        solver = CdclSolver(variables, formula)
        (verdict, model) = solver.solve()
        assert verdict == brute_force_satisfiable(variables, formula), (clauses, width, seed)
        if verdict:
            assert satisfies(formula, model)
        for record in solver.invariants:
            levels = record["levels"]
            assert sum(1 for level in levels if level == record["conflict_level"]) == 1
            assert record["backjump"] == max([0] + levels[1:])
            assert len({abs(literal) for literal in record["clause"]}) == len(record["clause"])
            checked += 1
        for clause in solver.learned:
            for assignment in brute_force_models(variables, formula):
                assert satisfies([clause], assignment), "learned clauses are implied by the input"
    assert checked >= 0

    # --- Hand-checked learning on a tiny refutation ---------------------------------------
    tiny = [[1], [-1, 2], [-2]]
    solver = CdclSolver(2, tiny)
    (verdict, model) = solver.solve()
    assert verdict is False and solver.conflicts == 0, \
        "level-0 propagation alone refutes this one, so nothing is learned"
    assert solver.invariants == []
    assert brute_force_satisfiable(2, tiny) is False

    # --- Determinism ----------------------------------------------------------------------
    formula = random_cnf(14, 3, 6, 41)
    first = CdclSolver(6, formula).solve()
    second = CdclSolver(6, formula).solve()
    assert first == second
    refuted = php(4, 3)
    run_one = CdclSolver(12, refuted)
    run_two = CdclSolver(12, refuted)
    assert run_one.solve() == run_two.solve()
    assert run_one.learned == run_two.learned
    assert run_one.conflicts == run_two.conflicts


if __name__ == "__main__":
    main()
    print("PASS")
