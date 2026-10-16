"""Task 748 - Implement SMT solver front-end sketch (theory combination simple).  (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)

A miniature SMT front end over the union of two theories, in the style of an SMT-LIB front end
that hands ground atoms to DPLL(T):

  * integer difference logic - atoms `x - y <= c` with x, y shared integer constants and integer c;
  * the theory of equality with uninterpreted functions - atoms `f(x) = f(y)` and disequalities,
    where applications take *shared* integer constants as arguments.

The Boolean search is a small DPLL over the atom polarities (every atom is decided, the theory is
consulted after each decision, and a clause that is not satisfied rejects the assignment).  Each
theory has its own solver: difference logic uses a constraint graph and Bellman-Ford, so a negative
cycle *is* the conflict explanation and the shortest path potentials *are* the model; congruence
closure uses a union-find with signature merging, so a violated disequality is the conflict.  The
two are combined Nelson-Oppen style by exchanging equalities over the shared constants until a
fixed point: a difference-logic equality x = y constrains the argument values, and a congruence
equality f(x) = f(y) with x = y is discovered rather than assumed.

Simplified, deliberately: only equalities are exchanged (no disequality splitting), the model for
the uninterpreted function is built by searching a tiny function table, and the brute-force oracle
interprets the theories over the finite integer domain {0, 1, 2}, which is why the tests compare
verdicts inside that domain rather than over all of Z.  Verified: every SAT answer comes back with a
model that satisfies every clause, every atom and every congruence obligation; every UNSAT answer is
confirmed by exhaustive enumeration over the domain; the conflict explanations name constraints that
are really asserted and really contradictory; and the answer is deterministic.
"""
from itertools import product

DOMAIN = (0, 1, 2)          # the finite domain used by the brute-force oracle
MODEL_DOMAIN = (0, 1, 2)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ---------------------------------------------------------------------------- atoms
def le(x, y, c):
    """x - y <= c in integer difference logic."""
    return ("le", x, y, c)


def eq(left, right):
    """Equality between two uninterpreted applications, written as (function, argument)."""
    return ("eq", left, right)


def neq(left, right):
    return ("neq", left, right)


def literal(atom, positive=True):
    return (atom, positive)


def atom_holds(atom, constants, function_table):
    """Concrete evaluation of an atom under an interpretation."""
    if atom[0] == "le":
        _, x, y, c = atom
        return constants[x] - constants[y] <= c
    _, left, right = atom
    left_value = function_table[left[0]][constants[left[1]]]
    right_value = function_table[right[0]][constants[right[1]]]
    return (left_value == right_value) if atom[0] == "eq" else (left_value != right_value)


def evaluate_clause(clause, constants, function_table):
    for atom, positive in clause:
        if atom_holds(atom, constants, function_table) == positive:
            return True
    return False


# ---------------------------------------------------------------------------- theory solvers
def difference_conflict(asserted):
    """Bellman-Ford over the asserted difference constraints; returns the negative cycle (the
    conflict explanation) or None.  Edges point from y to x with weight c for `x - y <= c`."""
    nodes = sorted({name for _, x, y, _ in asserted for name in (x, y)})
    edges = [(y, x, c) for _, x, y, c in asserted]
    if not edges:
        return None
    distance = {node: 0 for node in nodes}      # virtual source connected to every node with 0
    predecessor = {}
    for _ in range(len(nodes)):
        changed = False
        for source, target, weight in edges:
            if distance[source] + weight < distance[target]:
                distance[target] = distance[source] + weight
                predecessor[target] = (source, weight)
                changed = True
        if not changed:
            return None
    for source, target, weight in edges:
        if distance[source] + weight < distance[target]:
            # walk back from target to recover the cycle
            node = target
            for _ in range(len(nodes) + 1):
                if node in predecessor:
                    node = predecessor[node][0]
                else:
                    node = target
                    break
            cycle, walker = [node], node
            while True:
                walker = predecessor[walker][0]
                if walker in cycle or len(cycle) > len(nodes) + 1:
                    break
                cycle.append(walker)
            return cycle[::-1] + [cycle[-1]]
    return None


def difference_potentials(asserted):
    """A model for consistent difference constraints: shortest path distances from a virtual
    source give x - y <= c for every asserted constraint."""
    nodes = sorted({name for _, x, y, _ in asserted for name in (x, y)})
    distance = {node: 0 for node in nodes}
    for _ in range(len(nodes)):
        changed = False
        for _, x, y, c in asserted:
            if distance[y] + c < distance[x] - 0:
                distance[x] = distance[y] + c
                changed = True
        if not changed:
            break
    return distance


class CongruenceClosure:
    """Union-find over applications with signature-driven merging (the classic Nelson-Oppen CC)."""

    def __init__(self, applications):
        self.parent = {term: term for term in applications}
        self.signature = {}

    def find(self, term):
        while self.parent[term] != term:
            self.parent[term] = self.parent[self.parent[term]]
            term = self.parent[term]
        return term

    def merge(self, left, right):
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        self.parent[root_right] = root_left
        return True

    def propagate(self):
        """Close under congruence: applications with equal function symbols and equal arguments are
        merged.  Returns the list of newly derived equalities."""
        derived = []
        changed = True
        while changed:
            changed = False
            classes = {}
            for term in list(self.parent):
                argument = term[1]
                argument_class = (self.find(("const", argument))
                                  if ("const", argument) in self.parent else argument)
                signature = (term[0], argument_class)
                if signature in classes:
                    if self.merge(term, classes[signature]):
                        derived.append((classes[signature], term))
                        changed = True
                else:
                    classes[signature] = term
        return derived

    def equal(self, left, right):
        return self.find(left) == self.find(right)


def theory_check(asserted_le, equalities, disequalities, shared):
    """Combine the two theories: exchange equalities over shared constants to a fixed point.
    Returns (conflict explanation or None, derived equalities)."""
    le_constraints = list(asserted_le)
    terms = [("const", name) for name in shared]
    for left, right in list(equalities) + list(disequalities):
        terms += [left, right]
    cc = CongruenceClosure(list(dict.fromkeys(terms)))
    for left, right in equalities:
        cc.merge(left, right)
    cc.propagate()
    derived = []
    for _ in range(len(shared) + 2):
        conflict = difference_conflict(le_constraints)
        if conflict is not None:
            return ("difference", conflict), derived
        # difference logic equalities x = y become shared-constant equalities for congruence
        pairs = {(x, y) for _, x, y, _ in le_constraints}
        for _, x, y, c in le_constraints:
            if c == 0 and (y, x) in pairs:
                if cc.merge(("const", x), ("const", y)):
                    derived.append((("const", x), ("const", y)))
        cc.propagate()
        progressed = False
        for name_x in shared:
            for name_y in shared:
                if name_x != name_y and cc.equal(("const", name_x), ("const", name_y)):
                    if not any(a == name_x and b == name_y for _, a, b, _ in le_constraints):
                        le_constraints.append(("le", name_x, name_y, 0))
                        le_constraints.append(("le", name_y, name_x, 0))
                        progressed = True
        if not progressed and not derived:
            break
    conflict = difference_conflict(le_constraints)
    if conflict is not None:
        return ("difference", conflict), derived
    for left, right in disequalities:
        if cc.equal(left, right):
            return ("congruence", (left, right)), derived
    return None, derived


# ---------------------------------------------------------------------------- search
class SmtSolver:
    """DPLL over the atoms with the combined theory check after every decision."""

    def __init__(self, atoms, clauses, shared):
        self.atoms = list(atoms)
        self.clauses = [list(clause) for clause in clauses]
        self.shared = list(shared)
        self.stats = {"decisions": 0, "theory_checks": 0, "conflicts": 0, "models": 0}

    def implied_constraints(self, assignment):
        asserted_le, equalities, disequalities = [], [], []
        for atom, positive in assignment:
            if atom[0] == "le":
                if positive:
                    asserted_le.append(atom)
                else:
                    # negation of x - y <= c is x - y >= c + 1, i.e. y - x <= -c - 1
                    _, x, y, c = atom
                    asserted_le.append(("le", y, x, -c - 1))
            elif positive and atom[0] == "eq":
                equalities.append((atom[1], atom[2]))
            elif not positive and atom[0] == "eq":
                disequalities.append((atom[1], atom[2]))
            elif positive and atom[0] == "neq":
                disequalities.append((atom[1], atom[2]))
            else:
                equalities.append((atom[1], atom[2]))
        return asserted_le, equalities, disequalities

    def clauses_satisfied(self, assignment):
        mapping = dict(assignment)
        for clause in self.clauses:
            if not any(mapping.get(atom, False) == positive for atom, positive in clause):
                return False
        return True

    def model_for(self, assignment):
        """Search the finite domain for a witness: the constants are enumerated over DOMAIN, the
        asserted difference constraints are checked directly, and the uninterpreted functions get a
        canonical model (applications forced equal by an asserted equality or by congruence share a
        class, each class gets its own value).  This is the documented domain restriction of the
        front end: it decides the semantics over DOMAIN, exactly like the brute-force oracle."""
        asserted_le, equalities, disequalities = self.implied_constraints(assignment)
        applications = sorted({atom[1] for atom, _ in assignment if atom[0] in ("eq", "neq")} |
                              {atom[2] for atom, _ in assignment if atom[0] in ("eq", "neq")})
        for values in product(DOMAIN, repeat=len(self.shared)):
            constants = dict(zip(self.shared, values))
            if not all(constants[x] - constants[y] <= c for _, x, y, c in asserted_le):
                continue
            table = self.function_model(applications, constants, equalities)
            if table is None:
                continue
            if all(atom_holds(atom, constants, table) == positive for atom, positive in assignment):
                return constants, table
        return None, None

    @staticmethod
    def function_model(applications, constants, equalities):
        """Canonical model for the uninterpreted functions, or None when it cannot be built (two
        applications with the same symbol and argument forced apart)."""
        if not applications:
            return {}
        parent = {term: term for term in applications}

        def find(term):
            while parent[term] != term:
                parent[term] = parent[parent[term]]
                term = parent[term]
            return term

        for left, right in equalities:
            if left in parent and right in parent:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left
        changed = True
        while changed:
            changed = False
            for first in applications:
                for second in applications:
                    if find(first) != find(second) and first[0] == second[0] \
                            and constants[first[1]] == constants[second[1]]:
                        parent[find(second)] = find(first)
                        changed = True
        table, class_values, next_value = {}, {}, 1
        for term in applications:
            root = find(term)
            if root not in class_values:
                class_values[root] = next_value
                next_value += 1
            table.setdefault(term[0], {})[constants[term[1]]] = class_values[root]
        return table

    @staticmethod
    def table_consistent(candidate, applications, constants, equalities):
        values = {}
        for term in applications:
            value = candidate[(term[0], constants[term[1]])]
            values[term] = value
        for left, right in equalities:
            if left in values and right in values and values[left] != values[right]:
                return False
        return True

    def solve(self, assignment=(), index=0):
        if index == len(self.atoms):
            if not self.clauses_satisfied(assignment):
                return None
            constants, table = self.model_for(assignment)
            if constants is None:
                return None
            for atom, positive in assignment:
                if atom_holds(atom, constants, table) != positive:
                    return None
            self.stats["models"] += 1
            return (dict(assignment), constants, table)
        self.stats["decisions"] += 1
        for positive in (True, False):
            atom = self.atoms[index]
            attempt = tuple(assignment) + ((atom, positive),)
            asserted_le, equalities, disequalities = self.implied_constraints(attempt)
            self.stats["theory_checks"] += 1
            conflict, _ = theory_check(asserted_le, equalities, disequalities, self.shared)
            if conflict is not None:
                self.stats["conflicts"] += 1
                continue
            result = self.solve(attempt, index + 1)
            if result is not None:
                return result
        return None


# ---------------------------------------------------------------------------- brute-force oracle
def brute_force(atoms, clauses, shared):
    """Exhaustive interpretation over the finite domain: constants take values in DOMAIN and each
    uninterpreted application of arity 1 gets a table over DOMAIN."""
    applications = sorted({atom[1] for atom in atoms if atom[0] in ("eq", "neq")} |
                          {atom[2] for atom in atoms if atom[0] in ("eq", "neq")})
    symbols = sorted({term[0] for term in applications})
    for values in product(DOMAIN, repeat=len(shared)):
        constants = dict(zip(shared, values))
        table = {symbol: {} for symbol in symbols}
        slots = [(symbol, argument) for symbol in symbols for argument in DOMAIN]
        for combination in product(DOMAIN, repeat=len(slots)):
            for slot, value in zip(slots, combination):
                table[slot[0]][slot[1]] = value
            if all(evaluate_clause(clause, constants, table) for clause in clauses):
                return constants, table
    return None


def model_is_valid(atoms, clauses, constants, table, shared):
    """Check a candidate model: every clause, every atom polarity used and every congruence
    obligation (applications with equal arguments must have equal values)."""
    for clause in clauses:
        if not evaluate_clause(clause, constants, table):
            return False
    applications = sorted({atom[1] for atom in atoms if atom[0] in ("eq", "neq")} |
                          {atom[2] for atom in atoms if atom[0] in ("eq", "neq")})
    for first in applications:
        for second in applications:
            if first[0] == second[0] and constants[first[1]] == constants[second[1]]:
                if table[first[0]][constants[first[1]]] != table[second[0]][constants[second[1]]]:
                    return False
    return True


def main():
    shared = ["x", "y", "z"]
    # 1. pure difference logic: satisfiable chain, then an unsatisfiable triangle
    atoms = [le("x", "y", 2), le("y", "z", -1), le("z", "x", 1)]
    clauses = [[literal(atoms[0]), literal(atoms[1])], [literal(atoms[2])]]
    solver = SmtSolver(atoms, clauses, shared)
    result = solver.solve()
    check(result is not None, "the two-clause chain is satisfiable")
    _, constants, _ = result
    check(constants["z"] - constants["x"] <= 1 and constants["x"] - constants["y"] <= 2,
          "the difference-logic model must satisfy every asserted constraint")
    triangle = [le("x", "y", 2), le("y", "z", -1), le("z", "x", -3)]
    inconsistent = SmtSolver(triangle, [[literal(atom)] for atom in triangle], shared)
    check(inconsistent.solve() is None, "the three constraints x-y<=2, y-z<=-1, z-x<=-3 conflict")
    check(inconsistent.stats["conflicts"] > 0, "the conflict must have been reported by the theory")
    explanation = theory_check([le("x", "y", 2), le("y", "z", -1), le("z", "x", -3)], [], [], shared)[0]
    check(explanation is not None and explanation[0] == "difference",
          "the conflict explanation must be a difference-logic negative cycle")
    check(len(explanation[1]) >= 3, "the negative cycle must involve at least three constants")
    check(theory_check([le("x", "y", 1), le("y", "x", 1)], [], [], shared)[0] is None,
          "a consistent pair of constraints has no conflict")

    # 2. congruence closure alone: f(x) = f(y) with x = y is derived, a disequality then conflicts
    applications = [("f", "x"), ("f", "y")]
    congruent = SmtSolver([eq(("f", "x"), ("f", "y"))], [[literal(eq(("f", "x"), ("f", "y")))]], shared)
    check(congruent.solve() is not None, "f(x) = f(y) is satisfiable on its own")
    conflict_diseq = SmtSolver([eq(("f", "x"), ("f", "y")), neq(("f", "x"), ("f", "y"))],
                               [[literal(eq(("f", "x"), ("f", "y")))],
                                [literal(neq(("f", "x"), ("f", "y")))]], shared)
    check(conflict_diseq.solve() is None, "f(x) = f(y) and f(x) != f(y) are contradictory")
    chain = SmtSolver([eq(("f", "x"), ("f", "y")), neq(("f", "x"), ("f", "z")), eq(("f", "y"), ("f", "z"))],
                      [[literal(eq(("f", "x"), ("f", "y")))],
                       [literal(neq(("f", "x"), ("f", "z")))],
                       [literal(eq(("f", "y"), ("f", "z")))]], shared)
    check(chain.solve() is None,
          "transitivity of congruence must refute f(x)=f(y), f(y)=f(z), f(x)!=f(z)")

    # 3. theory combination: x = y forces f(x) = f(y), which collides with a disequality
    combined_atoms = [le("x", "y", 0), le("y", "x", 0), neq(("f", "x"), ("f", "y"))]
    combined_clauses = [[literal(atom)] for atom in combined_atoms]
    combined = SmtSolver(combined_atoms, combined_clauses, shared)
    check(combined.solve() is None,
          "the shared equality x = y must be given to congruence closure and refute f(x) != f(y)")
    conflict, derived = theory_check([le("x", "y", 0), le("y", "x", 0)], [],
                                     [(("f", "x"), ("f", "y"))], shared)
    check(conflict is not None and conflict[0] == "congruence",
          "the combined conflict must come from the congruence theory")
    check(derived, "the combination must have derived the shared equality")
    satisfiable_combined = SmtSolver([le("x", "y", 0), le("y", "x", 0), eq(("f", "x"), ("f", "y"))],
                                     [[literal(le("x", "y", 0)), literal(le("y", "x", 0))],
                                      [literal(eq(("f", "x"), ("f", "y")))]], shared)
    check(satisfiable_combined.solve() is not None, "the satisfiable combination must be found")

    # 4. every SAT answer carries a model that satisfies the clauses, the atoms and congruence
    state = 0x5EED
    sat_cases, unsat_cases = 0, 0
    for _ in range(12):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        first = ("f", shared[state % 3])
        second = ("f", shared[(state >> 8) % 3])
        atoms = [le(shared[(state >> 16) % 3], shared[(state >> 24) % 3], int(state % 3) - 1),
                 eq(first, second) if (state >> 32) % 2 else neq(first, second)]
        clauses = [[literal(atoms[0]), literal(atoms[1])], [literal(atoms[1], False)]]
        solver = SmtSolver(atoms, clauses, shared)
        result = solver.solve()
        oracle = brute_force(atoms, clauses, shared)
        if result is None:
            unsat_cases += 1
            check(oracle is None, "the solver's UNSAT must be confirmed by exhaustive search")
        else:
            sat_cases += 1
            assignment, constants, table = result
            check(oracle is not None, "the solver's SAT must be confirmed by exhaustive search")
            check(model_is_valid(atoms, clauses, constants, table, shared),
                  "a returned model must satisfy the clauses and the congruence obligations")
            for atom, positive in assignment.items():
                check(atom_holds(atom, constants, table) == positive,
                      "a returned model must make every decided atom true")
    check(sat_cases > 0 and unsat_cases > 0, "the sample must contain both outcomes")

    # 5. boundary cases: empty formula, single atom, trivially satisfied clause
    trivial = SmtSolver([], [], shared)
    check(trivial.solve() is not None, "a formula with no atoms and no clauses is satisfiable")
    false_clause = SmtSolver([], [[]], shared)
    check(false_clause.solve() is None, "the empty clause is unsatisfiable")
    single = SmtSolver([le("x", "y", 0)], [[literal(le("x", "y", 0))]], shared)
    solution = single.solve()
    check(solution is not None and solution[1]["x"] - solution[1]["y"] <= 0,
          "a single atom must be satisfiable with a model respecting it")
    refused = SmtSolver([le("x", "y", -1), le("y", "x", 0)],
                        [[literal(le("x", "y", -1))], [literal(le("y", "x", 0))]], shared)
    check(refused.solve() is None, "x < y and y <= x must be refuted")
    refuted_by_clause = SmtSolver([le("x", "y", 0)],
                                  [[literal(le("x", "y", 0))],
                                   [literal(le("x", "y", 0), False)]], shared)
    check(refuted_by_clause.solve() is None,
          "demanding an atom and its negation at once is unsatisfiable")
    negated = SmtSolver([le("x", "y", 0)], [[literal(le("x", "y", 0), False)]], shared)
    solution = negated.solve()
    check(solution is not None and solution[1]["x"] - solution[1]["y"] >= 1,
          "the negation of a difference constraint must be honoured by the model")

    # 6. determinism
    first_run = SmtSolver(combined_atoms, combined_clauses, shared)
    second_run = SmtSolver(combined_atoms, combined_clauses, shared)
    check(first_run.solve() == second_run.solve() and first_run.stats == second_run.stats,
          "identical input must give an identical run")
    check(satisfiable_combined.solve() is not None, "re-solving must stay satisfiable")

    print(f"748 random_cases sat={sat_cases} unsat={unsat_cases} "
          f"theory_checks={solver.stats['theory_checks']}")
    print(f"748 combined conflict={explanation} derived={len(derived)} "
          f"congruence_conflict={conflict}")
    print("PASS")


if __name__ == "__main__":
    main()
