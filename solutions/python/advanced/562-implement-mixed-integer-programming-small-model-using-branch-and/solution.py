"""Task 562 - Implement mixed integer programming small model using branch-and-bound outline. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)

A working branch-and-bound integer program solver, not an outline. The LP relaxation is solved
*exactly* in rational arithmetic (`fractions.Fraction`) by enumerating basic solutions: with m
constraint rows and n variables the file adds m slack variables, picks every size-m subset of the
n + m variables, solves that m x m subsystem by Gaussian elimination, keeps the solutions that are
nonnegative and satisfy every row, and takes the best objective. That is exponential in the number
of rows and is only used on tiny models, but it needs no tolerance, no pivoting rule and no
numerical luck - which is exactly why the checks can compare rational values for equality. Branch
and bound then follows the standard scheme: relax, prune a node when its bound cannot beat the
incumbent, otherwise branch on the most fractional variable by adding x_j <= floor(v) or
x_j >= ceil(v) as a new row, and stop when the stack empties. The incumbent is reported with the
model and its objective is compared against the exact integer optimum from brute force over the
bounded box and against the dynamic-programming knapsack optimum. Models are capped to bounded,
mostly-binary shapes: pure-integer variables with explicit upper-bound rows, since the LP solver
enumerates vertices and would stall on an unbounded region - the header says so rather than
pretending otherwise.

Checks: on 0/1 knapsack instances the branch-and-bound optimum equals an independent bitmask
brute force and the textbook O(nW) knapsack DP, the LP relaxation value equals the greedy
fractional-knapsack bound computed as an exact rational, and the relaxation bound is at least the
integer optimum on every model; the classic 2x + 2y <= 3 example gives LP value 3/2 (the whole edge x + y = 3/2 is
optimal, and the vertex enumeration returns (3/2, 0)) against an integer optimum of 1; a model needing real branching has node count greater than one and still finds
the true optimum; every reported solution is feasible and its objective matches the reported value;
an infeasible model is reported as infeasible by both the node search and the raw LP; and the
reported optimum is never beaten by any point in the brute-force box. Determinism is checked by
re-running.
"""
import itertools
from fractions import Fraction

MASK64 = (1 << 64) - 1
INF = float("inf")


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_knapsack(n, seed):
    """Weights and values for a small 0/1 knapsack."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 61) & MASK64)
    weights = []
    values = []
    for _ in range(n):
        state = prng64(state)
        weights.append(1 + (state >> 33) % 12)
        state = prng64(state)
        values.append(1 + (state >> 33) % 20)
    return weights, values


def solve_exact_system(matrix, rhs):
    """Solve an m x m rational system by Gauss-Jordan; returns None when singular."""
    m = len(matrix)
    rows = [[Fraction(value) for value in matrix[i]] + [Fraction(rhs[i])] for i in range(m)]
    for column in range(m):
        pivot = None
        for r in range(column, m):
            if rows[r][column] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        (rows[column], rows[pivot]) = (rows[pivot], rows[column])
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for r in range(m):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column]
                rows[r] = [rows[r][c] - factor * rows[column][c] for c in range(m + 1)]
    return [rows[i][m] for i in range(m)]


def lp_relaxation(objective, rows, rhs):
    """max objective . x over {x >= 0 : rows x <= rhs}, exactly, by basic-solution enumeration."""
    n = len(objective)
    m = len(rows)
    if m == 0:
        if any(value > 0 for value in objective):
            return (INF, None)
        return (Fraction(0), [Fraction(0)] * n)
    best = None
    best_point = None
    for basic in itertools.combinations(range(n + m), m):
        matrix = []
        for row_index in range(m):
            matrix.append([Fraction(rows[row_index][variable]) if variable < n
                           else Fraction(1 if variable - n == row_index else 0)
                           for variable in basic])
        values = solve_exact_system(matrix, rhs)
        if values is None or any(value < 0 for value in values):
            continue
        point = [Fraction(0)] * n
        for position, variable in enumerate(basic):
            if variable < n:
                point[variable] = values[position]
        feasible = True
        for row_index in range(m):
            total = sum(Fraction(rows[row_index][j]) * point[j] for j in range(n))
            if total > Fraction(rhs[row_index]):
                feasible = False
                break
        if not feasible:
            continue
        value = sum(Fraction(objective[j]) * point[j] for j in range(n))
        if best is None or value > best:
            best = value
            best_point = point
    return (best, best_point)


def branch_and_bound(objective, rows, rhs, node_cap=20000):
    """maximize objective . x over integers x >= 0 with rows x <= rhs (bounded models)."""
    nodes = 0
    best_value = None
    best_point = None
    stack = [(list(rows), list(rhs))]
    while stack:
        (node_rows, node_rhs) = stack.pop()
        nodes += 1
        assert nodes <= node_cap, "the search should stay tiny on these models"
        (value, point) = lp_relaxation(objective, node_rows, node_rhs)
        if value == INF:
            return (INF, None, nodes + 1)               # an unbounded relaxation
        if point is None:
            continue                                    # this node is infeasible
        if best_value is not None and value.__floor__() <= best_value:
            # Every integer point has an integer objective, so the relaxation can be rounded
            # down before the comparison - a valid and much stronger incumbent test.
            continue
        fractional = [j for j in range(len(point)) if point[j].denominator != 1]
        if not fractional:
            if best_value is None or value > best_value:
                best_value = value
                best_point = list(point)
            continue
        index = max(fractional, key=lambda j: min(point[j] - point[j].__floor__(),
                                                  point[j].__ceil__() - point[j]))
        floor_value = point[index].__floor__()
        ceil_value = point[index].__ceil__()
        lower = [0] * len(objective)
        lower[index] = -1
        stack.append((node_rows + [lower], node_rhs + [Fraction(-ceil_value)]))
        upper = [0] * len(objective)
        upper[index] = 1
        stack.append((node_rows + [upper], node_rhs + [Fraction(floor_value)]))
    return (best_value, best_point, nodes)


def knapsack_model(weights, values, capacity):
    """Rows for max sum values_i x_i with sum weights_i x_i <= capacity and x_i <= 1."""
    n = len(weights)
    rows = [[weights[j] for j in range(n)]]
    rhs = [Fraction(capacity)]
    for j in range(n):
        row = [0] * n
        row[j] = 1
        rows.append(row)
        rhs.append(Fraction(1))
    return (values, rows, rhs)


def fractional_knapsack_bound(weights, values, capacity):
    """The LP relaxation optimum of the knapsack, by the greedy ratio argument, exactly."""
    items = sorted(range(len(weights)), key=lambda j: Fraction(-values[j], weights[j]))
    remaining = Fraction(capacity)
    total = Fraction(0)
    for j in items:
        if remaining <= 0:
            break
        take = min(Fraction(1), remaining / Fraction(weights[j]))
        total += take * values[j]
        remaining -= take * weights[j]
    return total


def knapsack_dp(weights, values, capacity):
    """Textbook O(n W) 0/1 knapsack optimum."""
    best = [0] * (capacity + 1)
    for j in range(len(weights)):
        for room in range(capacity, weights[j] - 1, -1):
            candidate = best[room - weights[j]] + values[j]
            if candidate > best[room]:
                best[room] = candidate
    return best[capacity]


def brute_force_integer(objective, rows, rhs, upper_bounds):
    """Best integer point in the box, for tiny models."""
    ranges = [range(0, bound + 1) for bound in upper_bounds]
    best_value = None
    best_point = None
    for point in itertools.product(*ranges):
        feasible = all(sum(rows[i][j] * point[j] for j in range(len(objective))) <= rhs[i]
                       for i in range(len(rows)))
        if not feasible:
            continue
        value = sum(objective[j] * point[j] for j in range(len(objective)))
        if best_value is None or value > best_value:
            best_value = value
            best_point = point
    return best_value, best_point


def is_feasible(point, rows, rhs):
    return point is not None and all(
        sum(rows[i][j] * point[j] for j in range(len(point))) <= rhs[i]
        for i in range(len(rows)))


def main():
    # --- Knapsack instances against brute force and the DP -------------------------------
    for (n, capacity, seed) in ((4, 12, 2), (4, 15, 1), (4, 14, 6)):
        weights, values = random_knapsack(n, seed)
        objective, rows, rhs = knapsack_model(weights, values, capacity)
        (bounded, point, nodes) = branch_and_bound(objective, rows, rhs)
        upper = [1] * n
        (reference, reference_point) = brute_force_integer(objective, rows, rhs, upper)
        assert bounded == reference == knapsack_dp(weights, values, capacity), (n, capacity, seed)
        assert is_feasible(point, rows, rhs), "the reported solution must be feasible"
        assert sum(objective[j] * point[j] for j in range(n)) == bounded
        assert all(point[j].denominator == 1 for j in range(n)), "and integral"
        assert nodes >= 1
        (relaxation, relaxation_point) = lp_relaxation(objective, rows, rhs)
        assert relaxation == fractional_knapsack_bound(weights, values, capacity), (n, seed)
        assert relaxation >= bounded, "the relaxation is an upper bound"
        assert is_feasible(relaxation_point, rows, rhs)
        assert sum(objective[j] * relaxation_point[j] for j in range(n)) == relaxation

    # --- The textbook fractional example -------------------------------------------------
    objective = [1, 1]
    rows = [[2, 2]]
    rhs = [Fraction(3)]
    (relaxation, point) = lp_relaxation(objective, rows, rhs)
    assert relaxation == Fraction(3, 2), "2x + 2y <= 3 admits x + y = 3/2"
    assert sum(point) == Fraction(3, 2)
    assert is_feasible(point, rows, rhs)
    (bounded, bounded_point, nodes) = branch_and_bound(objective, rows, rhs)
    assert bounded == 1, bounded
    assert nodes > 1, "this model genuinely needs branching"
    assert brute_force_integer(objective, rows, rhs, [1, 1])[0] == 1

    # --- A hand-computed LP whose optimum happens to be integral -------------------------
    objective = [3, 2]
    rows = [[1, 1], [1, 3]]
    rhs = [Fraction(4), Fraction(6)]
    (relaxation, point) = lp_relaxation(objective, rows, rhs)
    assert (relaxation, point) == (Fraction(12), [Fraction(4), Fraction(0)]), (relaxation, point)
    (bounded, bounded_point, nodes) = branch_and_bound(objective, rows, rhs)
    assert bounded == 12
    assert brute_force_integer(objective, rows, rhs, [4, 4])[0] == 12
    assert is_feasible(bounded_point, rows, rhs)

    # --- Infeasible and degenerate models ------------------------------------------------
    objective = [1]
    rows = [[1], [-1]]
    rhs = [Fraction(2), Fraction(-3)]
    assert lp_relaxation(objective, rows, rhs) == (None, None)
    assert branch_and_bound(objective, rows, rhs)[0] is None
    assert branch_and_bound([1], [[1]], [Fraction(0)])[0] == 0
    assert lp_relaxation([-1], [[1]], [Fraction(5)]) == (Fraction(0), [Fraction(0)]), \
        "a maximisation with a nonpositive objective sits at the origin"
    assert branch_and_bound([1], [], [])[0] == INF, "an empty model is unbounded"

    # --- No integer point beats the reported optimum -------------------------------------
    weights, values = random_knapsack(4, 11)
    objective, rows, rhs = knapsack_model(weights, values, 12)
    (bounded, point, nodes) = branch_and_bound(objective, rows, rhs)
    assert is_feasible(point, rows, rhs)
    assert sum(objective[j] * point[j] for j in range(4)) == bounded
    for mask in range(1 << 4):
        weight = sum(weights[j] for j in range(4) if mask >> j & 1)
        if weight <= 12:
            assert sum(values[j] for j in range(4) if mask >> j & 1) <= bounded
    for sample in breadth_sample():
        candidate = [Fraction(value, denominator) for (value, denominator) in sample]
        if is_feasible(candidate, rows, rhs):
            assert sum(objective[j] * candidate[j] for j in range(4)) <= bounded
    # --- Determinism ---------------------------------------------------------------------
    weights, values = random_knapsack(4, 21)
    objective, rows, rhs = knapsack_model(weights, values, 16)
    first = branch_and_bound(objective, rows, rhs)
    second = branch_and_bound(objective, rows, rhs)
    assert first == second
    assert lp_relaxation(objective, rows, rhs) == lp_relaxation(objective, rows, rhs)


def breadth_sample():
    """Rational points (numerator, denominator) pairs used to probe for a better feasible point."""
    return [((a, 2), (b, 2), ((a + b) % 3, 2), ((a * b) % 3, 2))
            for a in range(0, 8) for b in range(0, 8)]


if __name__ == "__main__":
    main()
    print("PASS")
