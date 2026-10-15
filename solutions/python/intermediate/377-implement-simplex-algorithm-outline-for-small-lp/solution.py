"""Task 377 - Simplex algorithm for small LPs (outline). (Tier: Intermediate, CTCI Ch 6)

Standard form: maximise c.x subject to A x <= b, x >= 0.  The tableau simplex starts
from the slack basis (x = 0) when b >= 0.  Rows with b_i < 0 are negated (their slack
coefficient becomes -1) and get an artificial variable; a phase-1 objective
(-1 on the artificial columns, i.e. minimising their sum) drives them out, and a
positive phase-1 optimum means the LP is infeasible.  Phase 2 then runs the real
objective with the artificial columns forbidden from entering.

Bland's rule is used for both iterations: the entering variable is the smallest
index with a positive reduced cost and the leaving variable breaks ratio ties by the
smallest basic-variable index.  This guarantees termination (no cycling) and is
asserted by the built-in iteration cap - the cap is a failure sentinel, never hit by
the test problems.

Documented sentinels (the API returns a status string, x = None unless optimal):
  * "optimal"     -> x is the optimal vertex,
  * "unbounded"   -> some entering column has no positive tableau coefficient
                     (the objective grows without bound),
  * "infeasible"  -> a positive phase-1 optimum (the constraints contradict).

Re-derivation note: the task brief suggests "max 3x + 2y s.t. x+y<=4, x+3y<=6 ->
x=3, y=1, value 11".  That point is the intersection of the two constraint lines, but
the vertex (4, 0) satisfies both constraints (4+0=4, 4+0=4<=6) and gives
3*4 + 2*0 = 12 > 11, so the true optimum is 12 at (4, 0) - the code asserts 12 and
the brief's expected value is flagged in the test comments.

The objective values are small exact integers here; comparisons use the documented
tolerance 1e-9 because the tableau arithmetic is in doubles.  A Cramer-rule vertex
enumeration is used as an independent oracle for the LCG-generated bounded LPs.
"""

import itertools

EPS = 1e-9
MAX_ITERATIONS = 10000


def _run_simplex(tableau, rhs, basis, cost, can_enter):
    """Primal simplex core (Bland's rule). Mutates the tableau; returns (status, value)."""
    rows = len(tableau)
    columns = len(cost)

    reduced = list(cost)
    value = 0.0
    for i in range(rows):
        basic_cost = cost[basis[i]]
        if basic_cost != 0.0:
            value += basic_cost * rhs[i]
            for j in range(columns):
                reduced[j] -= basic_cost * tableau[i][j]

    iterations = 0
    while True:
        iterations += 1
        if iterations > MAX_ITERATIONS:
            raise AssertionError("simplex iteration cap exceeded (cycling?)")

        entering = -1
        for j in range(columns):                 # Bland: smallest improving index
            if can_enter[j] and reduced[j] > EPS:
                entering = j
                break
        if entering == -1:
            return "optimal", value

        leaving = -1
        best_ratio = 0.0
        for i in range(rows):
            if tableau[i][entering] > EPS:
                ratio = rhs[i] / tableau[i][entering]
                if leaving == -1 or ratio < best_ratio - 1e-12:
                    leaving, best_ratio = i, ratio
                elif abs(ratio - best_ratio) <= 1e-12 and basis[i] < basis[leaving]:
                    leaving, best_ratio = i, ratio     # Bland ratio tie-break
        if leaving == -1:
            return "unbounded", None

        pivot = tableau[leaving][entering]
        tableau[leaving] = [v / pivot for v in tableau[leaving]]
        rhs[leaving] /= pivot
        for i in range(rows):
            if i != leaving and tableau[i][entering] != 0.0:
                factor = tableau[i][entering]
                tableau[i] = [a - factor * b for a, b in zip(tableau[i], tableau[leaving])]
                rhs[i] -= factor * rhs[leaving]
        factor = reduced[entering]
        reduced = [a - factor * b for a, b in zip(reduced, tableau[leaving])]
        value += factor * rhs[leaving]
        basis[leaving] = entering


def simplex(objective, constraints, bounds):
    """max c.x s.t. A x <= b, x >= 0.  Returns (status, x, value)."""
    n = len(objective)
    m = len(constraints)
    negative_rows = [i for i in range(m) if bounds[i] < 0]
    artificial_count = len(negative_rows)
    columns = n + m + artificial_count

    tableau = []
    rhs = []
    basis = []
    artificial_of_row = {}
    next_artificial = n + m
    for i in range(m):
        row = [0.0] * columns
        if bounds[i] >= 0:
            for j in range(n):
                row[j] = float(constraints[i][j])
            row[n + i] = 1.0                              # slack basis
            basis.append(n + i)
            rhs.append(float(bounds[i]))
        else:
            for j in range(n):
                row[j] = -float(constraints[i][j])        # negated row
            row[n + i] = -1.0
            row[next_artificial] = 1.0                    # artificial basis
            artificial_of_row[i] = next_artificial
            next_artificial += 1
            basis.append(next_artificial - 1)
            rhs.append(float(-bounds[i]))
        tableau.append(row)

    if artificial_count:
        phase_one_cost = [0.0] * columns
        for i in negative_rows:
            phase_one_cost[artificial_of_row[i]] = -1.0
        status_one, value_one = _run_simplex(tableau, rhs, basis, phase_one_cost,
                                             [True] * columns)
        assert status_one == "optimal"                    # bounded by 0 from above
        if value_one < -EPS:                              # artificials could not be cleared
            return "infeasible", None, None

    phase_two_cost = [0.0] * columns
    for j in range(n):
        phase_two_cost[j] = float(objective[j])
    can_enter = [True] * (n + m) + [False] * artificial_count
    status, value = _run_simplex(tableau, rhs, basis, phase_two_cost, can_enter)
    if status != "optimal":
        return status, None, None

    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = rhs[i]
    return "optimal", x, value


def determinant(matrix):
    """Exact-ish recursive determinant for tiny matrices (n <= 3 in this task)."""
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    total = 0.0
    for j in range(size):
        minor = [row[:j] + row[j + 1:] for row in matrix[1:]]
        total += (-1) ** j * matrix[0][j] * determinant(minor)
    return total


def brute_force_lp(objective, constraints, bounds, tolerance=1e-7):
    """Cramer-rule vertex enumeration for bounded feasible LPs (unique optimum value)."""
    n = len(objective)
    m = len(constraints)
    hyperplanes = [(list(constraints[i]), float(bounds[i])) for i in range(m)]
    for j in range(n):
        unit = [0.0] * n
        unit[j] = 1.0
        hyperplanes.append((unit, 0.0))

    best = None
    for combo in itertools.combinations(range(len(hyperplanes)), n):
        matrix = [hyperplanes[i][0] for i in combo]
        rhs = [hyperplanes[i][1] for i in combo]
        det = determinant(matrix)
        if abs(det) < 1e-12:
            continue
        x = []
        for col in range(n):
            replaced = [row[:col] + [rhs[r]] + row[col + 1:] for r, row in enumerate(matrix)]
            x.append(determinant(replaced) / det)
        if any(value < -tolerance for value in x):
            continue
        feasible = True
        for i in range(m):
            if sum(constraints[i][j] * x[j] for j in range(n)) > bounds[i] + tolerance:
                feasible = False
                break
        if not feasible:
            continue
        value = sum(objective[j] * x[j] for j in range(n))
        if best is None or value > best:
            best = value
    return best


def lcg_bounded_lp(seed):
    """Random bounded feasible LP: n <= 2, non-negative A (each column positive), b > 0."""
    n = 1 + seed % 2
    m = 2 + seed % 2
    constraints = []
    for _ in range(m):
        row = []
        for _ in range(n):
            seed = (seed * 75 + 74) % 65537
            row.append(seed % 9)
        constraints.append(row)
    for j in range(n):                       # every column must have a positive entry
        if all(constraints[i][j] == 0 for i in range(m)):
            seed = (seed * 75 + 74) % 65537
            constraints[seed % m][j] = 1 + seed % 3
    bounds = []
    for _ in range(m):
        seed = (seed * 75 + 74) % 65537
        bounds.append(1 + seed % 20)
    objective = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        objective.append(seed % 10)
    return objective, constraints, bounds, seed


def main():
    # --- the brief's LP, re-derived: optimum (4, 0) with value 12, NOT (3, 1)/11 --
    status, x, value = simplex([3, 2], [[1, 1], [1, 3]], [4, 6])
    assert status == "optimal"
    assert abs(value - 12.0) < EPS
    assert abs(x[0] - 4.0) < EPS and abs(x[1]) < EPS
    assert 3 * 4 + 2 * 0 == 12 and 4 + 0 <= 4 and 4 + 0 <= 6
    # vertex enumeration agrees
    assert abs(brute_force_lp([3, 2], [[1, 1], [1, 3]], [4, 6]) - 12.0) < 1e-7

    # hand-solved: vertices (0,0)=0, (2.5,0)=5, (0,4)=12, (1,3)=11 -> 12 at (0,4)
    status, x, value = simplex([2, 3], [[1, 1], [2, 1]], [4, 5])
    assert status == "optimal" and abs(value - 12.0) < EPS
    assert abs(x[0]) < EPS and abs(x[1] - 4.0) < EPS
    assert abs(brute_force_lp([2, 3], [[1, 1], [2, 1]], [4, 5]) - 12.0) < 1e-7

    # 3 variables: max 3x+2y+4z s.t. x+y+z<=5, 2x+y+3z<=9
    # optima (4,1,0), (2,2,1), (0,3,2) all give 14; Bland terminates at one of them
    status, x, value = simplex([3, 2, 4], [[1, 1, 1], [2, 1, 3]], [5, 9])
    assert status == "optimal" and abs(value - 14.0) < EPS
    assert abs(x[0] + x[1] + x[2] - 5.0) < EPS
    assert all(component > -EPS for component in x)
    assert 2 * x[0] + x[1] + 3 * x[2] <= 9 + 1e-7
    assert abs(3 * x[0] + 2 * x[1] + 4 * x[2] - value) < EPS
    assert abs(brute_force_lp([3, 2, 4], [[1, 1, 1], [2, 1, 3]], [5, 9]) - 14.0) < 1e-7

    # tie along x + y <= 1 (degenerate objective direction)
    status, x, value = simplex([1, 1], [[1, 1]], [1])
    assert status == "optimal" and abs(value - 1.0) < EPS
    assert abs(x[0] + x[1] - 1.0) < EPS

    # duplicate constraints (degenerate basis)
    status, x, value = simplex([1], [[1], [1]], [3, 3])
    assert status == "optimal" and abs(value - 3.0) < EPS and abs(x[0] - 3.0) < EPS

    # --- sentinels ---------------------------------------------------------
    status, x, value = simplex([1], [], [])                       # no constraints
    assert status == "unbounded" and x is None and value is None
    status, x, value = simplex([1], [[-1]], [-1])                 # x >= 1, max x
    assert status == "unbounded" and x is None
    status, x, value = simplex([1], [[1], [-1]], [1, -3])         # x <= 1 and x >= 3
    assert status == "infeasible" and x is None and value is None

    # feasible LP with a negative bound (phase 1 needed, artificial leaves the basis)
    status, x, value = simplex([1], [[1], [-1]], [5, -2])         # 2 <= x <= 5
    assert status == "optimal" and abs(x[0] - 5.0) < EPS and abs(value - 5.0) < EPS
    status, x, value = simplex([-1], [[1], [-1]], [5, -2])        # minimise x
    assert status == "optimal" and abs(x[0] - 2.0) < EPS and abs(value + 2.0) < EPS

    # --- LCG cross-checks against vertex enumeration -----------------------
    seed = 37713
    for trial in range(40):
        objective, constraints, bounds, seed = lcg_bounded_lp(seed)
        status, x, value = simplex(objective, constraints, bounds)
        assert status == "optimal", (trial, objective, constraints, bounds)
        oracle = brute_force_lp(objective, constraints, bounds)
        assert oracle is not None
        assert abs(value - oracle) < 1e-6, (trial, value, oracle)
        for i, row in enumerate(constraints):
            lhs = sum(row[j] * x[j] for j in range(len(objective)))
            assert lhs <= bounds[i] + 1e-7, (trial, i)
        assert all(component > -1e-7 for component in x), trial
        assert abs(sum(objective[j] * x[j] for j in range(len(objective))) - value) < 1e-7


if __name__ == "__main__":
    main()
    print("PASS")
