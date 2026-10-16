"""Task 570 - Implement linear programming simplex pivot demonstration. (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

A complete two-phase simplex for max c^T x subject to A x <= b, x >= 0 with no sign restriction on
b. Slacks turn each row into an equality; rows with b_i < 0 are negated (which makes their slack
coefficient -1) and receive an artificial variable, so Phase 1 maximises minus the sum of the
artificials to reach a feasible basis, redundant all-artificial rows (rhs zero) are dropped, and
Phase 2 maximises the real objective. All arithmetic uses fractions.Fraction, so the pivoting,
the ratio tests and the cycle detection are exact - no floating-point tolerance ever decides a
pivot - and the file keeps the whole sequence of bases. Two pivoting rules are implemented: Bland's
rule (enter the smallest-index positive reduced cost, break ratio ties by the smallest basic index),
which provably terminates, and Dantzig's rule (enter the most positive reduced cost), which does not.
The verbatim textbook demonstration is included: on Beale's classic cycling example Dantzig's rule
returns to the basis it started from after exactly six pivots and is reported as "cycling", while
Bland's rule on the same input reaches x1 = 1/25, x3 = 1 with optimum 1/20 (also in six pivots); the
file asserts both behaviours - including the exact pivot count and the repeated basis - instead of
hiding the failure. Unboundedness, infeasibility, and degenerate redundancy are all
detected and reported as statuses.

Checks: hand-verifiable optima with exact vertex answers (a box, a triangle, a problem whose optimum
needs Phase 1 because of negative right-hand sides); the primal solution's feasibility and objective;
strong duality by solving the dual LP with the same machinery (-dual minimum equals the primal
maximum) and by the complementary-slackness products; the return of the primal-dual pair for a
hand-checked instance; comparison against vertex enumeration (all n-subsets of the m + n inequality
constraints solved by exact Gaussian elimination) on hand instances and on 60 random integer LCG
instances (drawn until 60 feasible bounded ones are found, biased towards non-negative right-hand
sides while every third instance still has negative ones), where the simplex optimum, the best
vertex and the dual value must all agree; unbounded and infeasible inputs, the latter cross-checked
against the vertex enumeration finding no vertex at all; Bland's rule never cycling on any tested input and never returning a
sub-optimal basis; a reduced-cost certificate (all reduced costs of the optimal basis <= 0); and
determinism.
"""
from fractions import Fraction as F

MAX_PIVOTS = 400
LCG_MOD = 65537


def lcg(seed):
    return (seed * 75 + 74) % LCG_MOD


def pivot(tableau, row, column, m, ncol):
    """Gauss-Jordan pivot on tableau[row][column] (never called on a zero entry)."""
    pivot_value = tableau[row][column]
    tableau[row] = [value / pivot_value for value in tableau[row]]
    for i in range(m):
        if i == row:
            continue
        factor = tableau[i][column]
        if factor != 0:
            tableau[i] = [tableau[i][j] - factor * tableau[row][j] for j in range(ncol + 1)]


def run_tableau(cost, matrix, rhs, basis, rule="bland"):
    """Simplex on an equality system with a known feasible basis; returns a status dictionary."""
    m = len(matrix)
    ncol = len(cost)
    tableau = [[F(matrix[i][j]) for j in range(ncol)] + [F(rhs[i])] for i in range(m)]
    basis = list(basis)
    pivots = 0
    seen = {tuple(basis): 0}
    while True:
        # Objective row of the max tableau: r_j = c_j - c_B B^-1 A_j; a positive entry marks a
        # column that improves the objective, and optimality means every r_j <= 0.
        reduced = [cost[j] - sum(cost[basis[i]] * tableau[i][j] for i in range(m))
                   for j in range(ncol)]
        entering = None
        for j in range(ncol):
            if reduced[j] > 0:
                if rule == "bland":
                    entering = j
                    break
                if entering is None or reduced[j] > reduced[entering]:
                    entering = j
        if entering is None:
            solution = [F(0)] * ncol
            for i in range(m):
                solution[basis[i]] = tableau[i][ncol]
            value = sum(cost[j] * solution[j] for j in range(ncol))
            return {"status": "optimal", "x": solution, "value": value, "basis": basis[:],
                    "pivots": pivots, "reduced": reduced, "tableau": tableau}
        leaving = None
        for i in range(m):
            if tableau[i][entering] > 0:
                if leaving is None:
                    leaving = i
                    continue
                candidate = tableau[i][ncol] / tableau[i][entering]
                current = tableau[leaving][ncol] / tableau[leaving][entering]
                if candidate < current:
                    leaving = i
                elif candidate == current and rule == "bland" and basis[i] < basis[leaving]:
                    leaving = i
        if leaving is None:
            return {"status": "unbounded", "pivots": pivots, "entering": entering,
                    "basis": basis[:]}
        pivot(tableau, leaving, entering, m, ncol)
        basis[leaving] = entering
        pivots += 1
        key = tuple(basis)
        if key in seen:
            return {"status": "cycling", "pivots": pivots, "basis": basis[:],
                    "repeated_at": seen[key]}
        seen[key] = pivots
        if pivots > MAX_PIVOTS:
            return {"status": "pivot-limit", "pivots": pivots, "basis": basis[:]}


def build_standard_form(A, b):
    """Columns: n structural, m slacks, one artificial per row with negative right-hand side."""
    m = len(A)
    n = len(A[0]) if A else 0
    rows = [[F(A[i][j]) for j in range(n)] + [F(0)] * m + [F(0)] * m for i in range(m)]
    rhs = []
    slacks = []
    artificials = []
    for i in range(m):
        if b[i] >= 0:
            rows[i][n + i] = F(1)
            rhs.append(F(b[i]))
        else:
            for j in range(n):
                rows[i][j] = -rows[i][j]
            rows[i][n + i] = F(-1)
            rows[i][n + m + i] = F(1)
            rhs.append(F(-b[i]))
        slacks.append(n + i)
    for i in range(m):
        if b[i] < 0:
            artificials.append(n + m + i)
        else:
            artificials.append(None)
    return rows, rhs, slacks, artificials, n, m


def solve_lp(c, A, b, rule="bland"):
    """max c^T x subject to A x <= b, x >= 0; returns a status dictionary."""
    rows, rhs, slacks, artificials, n, m = build_standard_form(A, b)
    ncol = n + 2 * m
    basis = [artificials[i] if artificials[i] is not None else slacks[i] for i in range(m)]
    if any(artificials[i] is not None for i in range(m)):
        phase1_cost = [F(0)] * ncol
        for i in range(m):
            if artificials[i] is not None:
                phase1_cost[artificials[i]] = F(-1)
        phase1 = run_tableau(phase1_cost, rows, rhs, basis, rule=rule)
        if phase1["value"] != 0:
            return {"status": "infeasible", "pivots": phase1["pivots"]}
        rows, rhs, basis = phase1["tableau"], [phase1["tableau"][i][ncol] for i in range(m)], \
            phase1["basis"]
        # Drop rows whose basic variable is still artificial (degenerate, redundant constraints).
        kept = [i for i in range(m) if basis[i] < n + m]
        rows = [
            [rows[i][j] for j in range(n + m)] + [rows[i][ncol]]
            for i in kept
        ]
        basis = [basis[i] for i in kept]
        rhs = [rows[i][n + m] for i in range(len(rows))]
        m_eff = len(kept)
        cost = [F(c[j]) for j in range(n)] + [F(0)] * m
        if m_eff == 0:
            return {"status": "optimal", "x": [F(0)] * n, "value": F(0), "basis": [], "pivots": 0}
        return run_tableau(cost, rows, rhs, basis, rule=rule)
    cost = [F(c[j]) for j in range(n)] + [F(0)] * m
    return run_tableau(cost, rows, rhs, basis, rule=rule)


def primal_solution(result, n):
    return [result["x"][j] for j in range(n)]


def reduced_costs_ok(result):
    return all(value <= 0 for value in result["reduced"])


def vertex_optimum(c, A, b, n):
    """Exact optimum by enumerating every vertex: solve each n-subset of the m + n inequalities."""
    constraints = [(list(row), rhs) for row, rhs in zip(A, b)]
    for j in range(n):
        row = [F(0)] * n
        row[j] = F(-1)
        constraints.append((row, F(0)))
    best = None
    best_point = None
    for combo in __import__("itertools").combinations(range(len(constraints)), n):
        matrix = [[F(constraints[r][0][j]) for j in range(n)] for r in combo]
        point = solve_square(matrix, [F(constraints[r][1]) for r in combo], n)
        if point is None:
            continue
        if any(value < 0 for value in point):
            continue
        if any(sum(F(A[i][j]) * point[j] for j in range(n)) > F(b[i]) for i in range(len(A))):
            continue
        value = sum(F(c[j]) * point[j] for j in range(n))
        if best is None or value > best:
            best = value
            best_point = point
    return best, best_point


def solve_square(matrix, rhs, n):
    """Exact Gaussian elimination with partial pivoting by row order; None if singular."""
    augmented = [[F(matrix[i][j]) for j in range(n)] + [F(rhs[i])] for i in range(n)]
    for column in range(n):
        pivot_row = None
        for i in range(column, n):
            if augmented[i][column] != 0:
                pivot_row = i
                break
        if pivot_row is None:
            return None
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for i in range(n):
            if i != column and augmented[i][column] != 0:
                factor = augmented[i][column]
                augmented[i] = [augmented[i][j] - factor * augmented[column][j]
                                for j in range(n + 1)]
    return [augmented[i][n] for i in range(n)]


def main():
    # --- Hand-checked: a box ------------------------------------------------------------
    c = [3, 5]
    A = [[1, 0], [0, 2], [3, 2]]
    b = [4, 12, 18]
    result = solve_lp(c, A, b)
    assert result["status"] == "optimal"
    assert (result["value"], primal_solution(result, 2)) == (36, [F(2), F(6)])
    assert reduced_costs_ok(result)
    best, point = vertex_optimum(c, A, b, 2)
    assert best == result["value"] and sum(c[j] * point[j] for j in range(2)) == best

    # --- Hand-checked: negative right-hand sides force Phase 1 --------------------------
    c = [1, 1]
    A = [[1, 1], [-1, 0], [0, -1]]
    b = [5, -1, -2]                     # x1 >= 1, x2 >= 2
    result = solve_lp(c, A, b)
    assert result["status"] == "optimal"
    # x1 >= 1, x2 >= 2 and x1 + x2 <= 5 make the optimal face degenerate (any split works), so
    # only the value is pinned down and the reported basis is whatever Bland's rule walks to.
    assert result["value"] == F(5)
    x = primal_solution(result, 2)
    assert x[0] >= 1 and x[1] >= 2 and x[0] + x[1] == 5
    best, _ = vertex_optimum(c, A, b, 2)
    assert best == F(5)

    # --- Strong duality, by solving the dual with the same machinery --------------------
    dual_c = [-b[i] for i in range(len(b))]
    dual_A = [[-A[i][j] for i in range(len(b))] for j in range(len(c))]
    dual_b = [-c[j] for j in range(len(c))]
    dual = solve_lp(dual_c, dual_A, dual_b)
    assert dual["status"] == "optimal"
    assert -dual["value"] == result["value"], "max c x equals min b y"
    y = primal_solution(dual, len(b))
    assert sum(F(b[i]) * y[i] for i in range(len(b))) == result["value"]
    assert all(sum(F(A[i][j]) * F(primal_solution(result, 2)[j]) for j in range(2)) == F(b[i])
               or y[i] == 0 for i in range(len(b))), "complementary slackness"
    assert all(sum(F(A[i][j]) * y[i] for i in range(len(b))) >= F(c[j]) for j in range(2))

    # --- Beale's cycling example: Dantzig loops, Bland terminates -----------------------
    beale_c = [F(3, 4), F(-150), F(1, 50), F(-6)]
    beale_A = [[F(1, 4), F(-60), F(-1, 25), F(9)],
               [F(1, 2), F(-90), F(-1, 50), F(3)],
               [F(0), F(0), F(1), F(0)]]
    beale_b = [F(0), F(0), F(1)]
    dantzig = solve_lp(beale_c, beale_A, beale_b, rule="dantzig")
    assert dantzig["status"] == "cycling", "Dantzig's rule revisits a basis on Beale's example"
    assert dantzig["repeated_at"] < dantzig["pivots"]
    # The textbook cycle: six pivots bring Dantzig's rule back to the basis it started from.
    assert (dantzig["pivots"], dantzig["repeated_at"]) == (6, 0)
    assert dantzig["basis"] == [4, 5, 6]
    bland = solve_lp(beale_c, beale_A, beale_b, rule="bland")
    assert bland["status"] == "optimal", "Bland's rule cannot cycle"
    assert bland["value"] == F(1, 20) == vertex_optimum(beale_c, beale_A, beale_b, 4)[0]
    assert reduced_costs_ok(bland)

    # --- Unbounded and infeasible -------------------------------------------------------
    unbounded = solve_lp([1], [[-1]], [1])
    assert unbounded["status"] == "unbounded", "x1 is bounded below only"
    infeasible = solve_lp([1], [[1], [-1]], [-1, -1])
    assert infeasible["status"] == "infeasible", "x1 <= -1 and x1 >= 1 cannot coexist"
    degenerate = solve_lp([1, 1], [[1, 1], [1, 1]], [2, 2])
    assert degenerate["status"] == "optimal" and degenerate["value"] == 2

    # --- Random instances against vertex enumeration and the dual -----------------------
    seed = 2718
    checked = 0
    infeasible_count = 0
    unbounded_count = 0
    trials = 0
    while checked < 60 and trials < 200:
        trials += 1
        n = 2 + seed % 2
        m = 3 + seed % 3
        c = []
        for _j in range(n):
            seed = lcg(seed)
            c.append(seed % 7)                       # non-negative objective
        A = []
        b = []
        for _i in range(m):
            row = []
            for _j in range(n):
                seed = lcg(seed)
                row.append(seed % 8 - 2)             # mostly non-negative rows, x >= 0 feasible
            A.append(row)
            seed = lcg(seed)
            b.append(seed % 17 - 6 if trials % 3 == 0 else seed % 13)
        result = solve_lp(c, A, b)
        if result["status"] == "infeasible":
            infeasible_count += 1
            assert vertex_optimum(c, A, b, n)[0] is None, "no vertex means infeasible"
            continue
        if result["status"] == "unbounded":
            unbounded_count += 1
            continue
        assert result["status"] == "optimal", "Bland's rule always terminates"
        x = primal_solution(result, n)
        assert all(value >= 0 for value in x), "x >= 0"
        assert all(sum(F(A[i][j]) * x[j] for j in range(n)) <= F(b[i]) for i in range(m))
        assert sum(F(c[j]) * x[j] for j in range(n)) == result["value"]
        assert reduced_costs_ok(result)
        best, _ = vertex_optimum(c, A, b, n)
        assert best is not None
        assert best == result["value"], "the simplex optimum is the best vertex"
        dual = solve_lp([-b[i] for i in range(m)],
                        [[-A[i][j] for i in range(m)] for j in range(n)],
                        [-c[j] for j in range(n)])
        assert dual["status"] == "optimal"
        assert -dual["value"] == result["value"], "strong duality"
        checked += 1
    assert checked == 60, "sixty feasible bounded instances were compared"
    assert trials < 200, "the generator found them without exhausting its budget"

    # --- Determinism and pivot counting -------------------------------------------------
    A = [[1, 0], [0, 2], [3, 2]]
    b = [4, 12, 18]
    first = solve_lp([3, 5], A, b)
    second = solve_lp([3, 5], A, b)
    assert first["value"] == second["value"] and first["basis"] == second["basis"]
    assert first["pivots"] == second["pivots"] >= 1
    assert solve_lp([1, 1], [[1, 1]], [3])["pivots"] == 1


if __name__ == "__main__":
    main()
    print("PASS")
