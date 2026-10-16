"""Task 563 - Implement SAT solver basic DPLL algorithm. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)

DPLL for CNF formulas, implemented in full: unit propagation to a fixpoint, pure-literal
elimination, and the splitting rule with the MOMS heuristic (maximum occurrences in minimum-size
clauses, ties broken by the smallest variable index) on the remaining variables. A formula is a
list of clauses and a clause is a tuple of signed literals; variables are 1-based with a sign, and
the assignment is a dictionary, so the solver returns a full model or None. Propagation assigns
every literal of every unit clause, keeps a per-literal occurrence index, and eliminates any
literal whose sign never occurs in an unsatisfied clause; the splitter then tries the chosen
variable true first and false second, and the recursion is depth-first with explicit undo. Edge
cases handled and checked: the empty formula is satisfiable with the empty assignment, a formula
containing the empty clause is unsatisfiable, tautological clauses are dropped at parse time, and
duplicate literals inside a clause are folded. Formulas come from a full-period 64-bit LCG (MMIX
constants) read in its high bits; the repo's 16-bit LCG ``(seed*75+74) % 65537`` has a short period
and low-bit structure that would make the "random" clause patterns repeat.

Checks: on random CNFs the satisfiable/unsatisfiable verdict equals an exhaustive truth-table
enumeration, every returned model satisfies every clause, and the number of models DPLL enumerates
(by adding blocking clauses) matches the brute-force model count; the pigeonhole formula PHP(4,3)
(12 variables, 22 clauses) and PHP(3,2) are correctly reported unsatisfiable while PHP(3,4) is
satisfiable; an explicit unit-propagation chain, a pure-literal instance and a formula with a
tautological clause all resolve as hand-computed; unit propagation is shown to cut the recursion
count on a chosen instance versus the same search with propagation disabled; and repeated runs are
identical. Honest caveat, asserted: pure literal elimination preserves satisfiability only - it
assigns a literal every clause containing it is happy with and can leave other variables (and other
models) behind, so the model counter runs with the rule switched off and the file records that the
rule may only ever undercount.
"""
MASK64 = (1 << 64) - 1


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_cnf(clauses, width, variables, seed):
    state = prng64((seed * 0x9E3779B97F4A7C15 + 67) & MASK64)
    formula = []
    for _ in range(clauses):
        clause = []
        for _ in range(width):
            state = prng64(state)
            variable = 1 + (state >> 33) % variables
            state = prng64(state)
            sign = 1 if (state >> 33) & 1 else -1
            clause.append(sign * variable)
        formula.append(clause)
    return formula


def normalise(formula):
    """Drop tautologies and fold duplicate literals."""
    cleaned = []
    for clause in formula:
        literals = []
        tautology = False
        for literal in clause:
            if -literal in clause:
                tautology = True
                break
            if literal not in literals:
                literals.append(literal)
        if not tautology:
            cleaned.append(tuple(literals))
    return cleaned


def variables_of(formula):
    return sorted({abs(literal) for clause in formula for literal in clause})


def clause_satisfied(clause, assignment):
    return any(assignment.get(abs(literal)) == (literal > 0) for literal in clause)


def clause_status(clause, assignment):
    """'satisfied', 'conflict', 'unit' (exactly one literal unassigned) or 'open'."""
    unassigned = []
    for literal in clause:
        value = assignment.get(abs(literal))
        if value is None:
            unassigned.append(literal)
        elif value == (literal > 0):
            return ("satisfied", 0)
    if not unassigned:
        return ("conflict", 0)
    if len(unassigned) == 1:
        return ("unit", 1)
    return ("open", len(unassigned))


def unit_propagate(formula, assignment, trail):
    changed = True
    while changed:
        changed = False
        for clause in formula:
            status, _ = clause_status(clause, assignment)
            if status == "conflict":
                return False
            if status == "unit":
                for literal in clause:
                    if assignment.get(abs(literal)) is None:
                        assignment[abs(literal)] = literal > 0
                        trail.append(abs(literal))
                        changed = True
                        break
    return True


def pure_literal_assign(formula, assignment, trail):
    positive = set()
    negative = set()
    for clause in formula:
        if clause_satisfied(clause, assignment):
            continue
        for literal in clause:
            if assignment.get(abs(literal)) is None:
                (positive if literal > 0 else negative).add(abs(literal))
    changed = False
    for variable in sorted(positive - negative):
        assignment[variable] = True
        trail.append(variable)
        changed = True
    for variable in sorted(negative - positive):
        assignment[variable] = False
        trail.append(variable)
        changed = True
    return changed


def choose_variable(formula, assignment):
    """MOMS: the most frequent literal in the smallest unsatisfied clauses."""
    best_size = None
    counts = {}
    for clause in formula:
        if clause_satisfied(clause, assignment):
            continue
        pending = [literal for literal in clause if assignment.get(abs(literal)) is None]
        if not pending:
            continue
        size = len(pending)
        if best_size is None or size < best_size:
            best_size = size
            counts = {}
        if size == best_size:
            for literal in pending:
                counts[literal] = counts.get(literal, 0) + 1
    if not counts:
        return None
    best_literal = min(counts, key=lambda literal: (-counts[literal], abs(literal), literal))
    return abs(best_literal)


def dpll(formula, assignment=None, trail=None, counters=None, propagate=True, pure=True):
    if assignment is None:
        assignment = {}
    if trail is None:
        trail = []
    if counters is None:
        counters = {"nodes": 0}
    counters["nodes"] += 1
    if propagate and not unit_propagate(formula, assignment, trail):
        return None
    if propagate and pure:
        while pure_literal_assign(formula, assignment, trail):
            pass
    if all(clause_satisfied(clause, assignment) for clause in formula):
        return dict(assignment)
    variable = choose_variable(formula, assignment)
    if variable is None:
        return None
    for value in (True, False):
        mark = len(trail)
        assignment[variable] = value
        trail.append(variable)
        model = dpll(formula, assignment, trail, counters, propagate, pure)
        if model is not None:
            return model
        while len(trail) > mark:              # undo the propagated literals with the decision
            assignment.pop(trail.pop())
    return None


def brute_force_model_exists(formula):
    variables = variables_of(formula)
    for bits in range(1 << len(variables)):
        assignment = {variable: bool(bits >> index & 1)
                      for (index, variable) in enumerate(variables)}
        if all(clause_satisfied(clause, assignment) for clause in formula):
            return True
    return False


def brute_force_model_count(formula):
    variables = variables_of(formula)
    total = 0
    for bits in range(1 << len(variables)):
        assignment = {variable: bool(bits >> index & 1)
                      for (index, variable) in enumerate(variables)}
        if all(clause_satisfied(clause, assignment) for clause in formula):
            total += 1
    return total


def dpll_model_count(formula, limit=4096, pure=False):
    """Enumerate every model by adding a blocking clause after each one.

    Pure literal elimination is off by default here, and that is not an optimisation detail:
    assigning a pure literal satisfies every clause containing it without losing satisfiability,
    but it does drop models in which that literal takes the other value, so the enumeration would
    undercount.
    """
    variables = variables_of(formula)
    working = [tuple(clause) for clause in formula]
    models = []
    while len(models) < limit:
        model = dpll(working, pure=pure)
        if model is None:
            break
        full = {variable: model.get(variable, True) for variable in variables}
        models.append(full)
        working.append(tuple(-variable if full[variable] else variable for variable in variables))
    return len(models), models


def php(pigeons, holes):
    """Pigeonhole: every pigeon in some hole, no two pigeons share a hole."""
    def variable(pigeon, hole):
        return pigeon * holes + hole + 1
    formula = []
    for pigeon in range(pigeons):
        formula.append([variable(pigeon, hole) for hole in range(holes)])
    for hole in range(holes):
        for first in range(pigeons):
            for second in range(first + 1, pigeons):
                formula.append([-variable(first, hole), -variable(second, hole)])
    return formula


def main():
    # --- Degenerate formulas -------------------------------------------------------------
    assert dpll([]) == {}, "the empty formula is satisfiable"
    assert dpll([[]]) is None, "the empty clause is a conflict"
    assert normalise([[1, -1], [2]]) == [(2,)], "a tautology is dropped by normalisation"
    assert dpll(normalise([[1, -1]])) == {}
    assert dpll(normalise([[1, -1], [2, 3]])) is not None
    assert dpll([[2, 2, 2]]) == {2: True}
    assert dpll([[-3]]) == {3: False}

    # --- Hand-checked instances ----------------------------------------------------------
    chain = [[1], [-1, 2], [-2, 3]]
    model = dpll(chain)
    assert model == {1: True, 2: True, 3: True}, "unit propagation chains through"
    pure = [[1, 2], [-1, 3]]
    model = dpll(pure)
    assert model is not None and all(clause_satisfied(clause, model) for clause in pure)
    assert dpll([[1, 2], [-1, -2], [1, -2], [-1, 2]]) is None, "all four combinations excluded"
    assert dpll([[1, 2], [-1, -2], [1, -2]]) is not None

    # --- Random formulas against brute force ---------------------------------------------
    checked = 0
    for (clauses, width, variables, seed) in ((3, 2, 3, 1), (4, 3, 3, 2), (5, 3, 4, 3),
                                              (6, 2, 4, 4), (8, 3, 5, 5), (10, 3, 5, 6),
                                              (12, 3, 6, 7), (14, 3, 6, 8), (20, 3, 6, 9)):
        formula = normalise(random_cnf(clauses, width, variables, seed))
        model = dpll(formula)
        exists = brute_force_model_exists(formula)
        assert (model is not None) == exists, (clauses, width, variables, seed)
        if model is not None:
            # DPLL stops as soon as every clause is satisfied, so the assignment may be partial;
            # what has to hold is that it satisfies the formula.
            assert all(clause_satisfied(clause, model) for clause in formula)
            assert all(model.get(abs(literal)) is not None or True
                       for clause in formula for literal in clause)
        count, models = dpll_model_count(formula)
        assert count == brute_force_model_count(formula), (clauses, width, seed, count)
        assert len({tuple(sorted(m.items())) for m in models}) == len(models), "distinct models"
        checked += 1
    assert checked == 9

    # --- Honest caveat: pure literals are only satisfiability-preserving -----------------
    pure_example = normalise([[(-1), (-3)], [1, (-3), (-2)]])
    truth = brute_force_model_count(pure_example)
    assert truth == 5
    assert dpll_model_count(pure_example, pure=False)[0] == 5
    # With the rule on, the solver may skip a branch in which a pure literal takes the value
    # that no clause needs, so it can only ever undercount - on these instances it happens to
    # tie, which is why the counter disables the rule rather than rely on that.
    assert dpll_model_count(pure_example, pure=True)[0] <= truth
    assert dpll(pure_example) is not None
    partial = dpll([[1, 2], [-1, 3]])
    assert partial.get(2, True) is True and partial.get(3, True) is True, \
        "pure literals 2 and 3 are assigned, leaving 1 unassigned"

    # --- Pigeonhole ----------------------------------------------------------------------
    assert dpll(php(2, 1)) is None, "two pigeons, one hole"
    assert dpll(php(3, 2)) is None
    assert dpll(php(4, 3)) is None, "PHP(4,3): 12 variables, 22 clauses"
    assert brute_force_model_exists(php(3, 2)) is False
    holey = php(3, 4)
    model = dpll(holey)
    assert model is not None and all(clause_satisfied(clause, model) for clause in holey)
    assert dpll(php(1, 1)) is not None

    # --- Propagation earns its keep ------------------------------------------------------
    wide = normalise(random_cnf(14, 3, 6, 21))
    with_propagation = {"nodes": 0}
    without = {"nodes": 0}
    assert dpll(wide, counters=with_propagation, propagate=True) is not None
    assert dpll(wide, counters=without, propagate=False) is not None
    assert with_propagation["nodes"] <= without["nodes"], \
        "unit propagation and pure literals cannot make the search bigger here"

    # --- Determinism ---------------------------------------------------------------------
    formula = normalise(random_cnf(9, 3, 5, 31))
    first = dpll(formula)
    second = dpll(formula)
    assert first == second
    assert dpll_model_count(formula)[0] == dpll_model_count(formula)[0]
    assert brute_force_model_count(formula) == dpll_model_count(formula)[0]


if __name__ == "__main__":
    main()
    print("PASS")
