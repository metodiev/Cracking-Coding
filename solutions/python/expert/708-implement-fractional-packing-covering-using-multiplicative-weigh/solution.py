"""Task 708 - Implement fractional packing/covering using multiplicative weights update.  (Tier: Expert, CTCI Ch 4 - Trees and Graphs)

The packing LP is  max 1^T x  s.t.  A x <= 1, x >= 0  (A >= 0) and its dual, the covering LP, is
min 1^T y  s.t.  A^T y >= 1, y >= 0.  Weak duality says 1^T x <= 1^T y for every feasible pair,
so a single run of the multiplicative-weights method produces *both sides* of a sandwich around
the (unknown) optimum, and the two sides are certified feasible by independent checks:

  * prices -- a probability vector p over the m packing constraints, started uniform;
  * oracle -- the "linear optimisation oracle" is the cheapest column under the current prices,
  j = argmin_j (A^T p)_j, which is where the MWU method plugs in a real solver;
  * usage -- the chosen column is counted once per iteration, and once the accumulated loads are
  congested the counting vector scaled by 1 / max_i (A usage)_i is a *feasible* packing
  solution, giving the lower bound LB = (number of iterations) / congestion;
  * update -- prices of the congested constraints grow multiplicatively,
  p_i <- p_i (1 + eps * load_i / congestion), then renormalise: the oracle then avoids the
  bottleneck, which is exactly the multiplicative-weights dynamics;
  * covering side -- by the averaging argument ("the average of the dual plays is nearly
  optimal") a prefix mean price vector a = (1/t) sum_{s<=t} p^(s) satisfies
  (A^T a)_j <= (1/t) sum_{s<=t} v_s for the oracle values v_s, so y = a / min_j (A^T a)_j is
  dual feasible and its objective UB = 1 / min_j (A^T a)_j upper-bounds the optimum; the best
  of the prefix means is kept.

Both bounds are recomputed from scratch by `packing_residual` / `covering_residual` before they
are believed, so a wrong bound cannot slip through.  Verified: LB <= OPT* <= UB against an exact
rational (Fraction) vertex-enumeration solver for both LPs, the residuals are non-positive, the
sandwich tightens as the iteration count grows, and the degenerate cases (a column no constraint
touches -> unbounded packing, empty matrix, invalid epsilon) are reported rather than hidden.

Simplified versus a production LP solver: the oracle is a linear scan instead of a min-cost-flow
or knapsack oracle (the interesting choice only matters in the fully generalised framework), the
capacities are all 1 (an instance with general b is pre-scaled), and iterations/epsilon are
parameters rather than chosen adaptively.
"""
import sys
from fractions import Fraction

INF = float("inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------- the algorithm
def oracle_pick(A, prices):
    """Linear optimisation oracle: the column with the smallest weighted resource cost.

    In the packing reading, a column consumes A_ij units of resource i, and p_i is the current
    price of that resource, so the oracle returns the cheapest way to gain one unit of objective.
    """
    best_j, best_value = 0, INF
    for j in range(len(A[0])):
        value = 0.0
        for i in range(len(A)):
            value += A[i][j] * prices[i]
        if value < best_value:
            best_j, best_value = j, value
    return best_j, best_value


def packing_loads(A, usage):
    return [sum(A[i][j] * usage[j] for j in range(len(usage))) for i in range(len(A))]


def multiplicative_weights(A, iterations=400, epsilon=0.1):
    """Run the MWU loop and return the certified sandwich (lb, x, ub, y, stats)."""
    check(0.0 < epsilon <= 1.0, "epsilon must lie in (0, 1]")
    check(iterations >= 1, "at least one iteration is required")
    m = len(A)
    n = len(A[0]) if m else 0
    for row in A:
        check(len(row) == n, "the matrix must be rectangular")
        check(all(value >= 0 for value in row), "A must be non-negative")
    if m == 0:
        return INF, None, INF, None, {"reason": "no constraints: the packing is unbounded"}
    if any(all(A[i][j] == 0 for i in range(m)) for j in range(n)):
        return INF, None, INF, None, {"reason": "a column misses every constraint"}

    prices = [1.0 / m] * m
    usage = [0.0] * n
    price_sum = [0.0] * m
    best_lb = 0.0
    best_x = [0.0] * n
    best_ub = INF
    best_y = None
    ub_checks = 0
    checkpoints = sorted({max(1, iterations * (k + 1) // 24) for k in range(24)} | {iterations})

    for step in range(1, iterations + 1):
        for i in range(m):
            price_sum[i] += prices[i]
        j, _ = oracle_pick(A, prices)
        usage[j] += 1.0
        loads = packing_loads(A, usage)

        # scaling certificate: usage / congestion is a feasible packing solution
        congestion = max(loads)
        if congestion > 0:
            lam = sum(usage) / congestion
            if lam > best_lb:
                best_lb = lam
                best_x = [value / congestion for value in usage]

        # multiplicative update: congested constraints become pricier, so the oracle avoids them
        for i in range(m):
            if loads[i] > 0:
                prices[i] *= 1.0 + epsilon * (loads[i] / congestion)
        total = sum(prices)
        prices = [value / total for value in prices]

        if step in checkpoints:
            ub_checks += 1
            average = [value / step for value in price_sum]
            coverage = [sum(A[i][j] * average[i] for i in range(m)) for j in range(n)]
            smallest = min(coverage)
            if smallest > 0 and 1.0 / smallest < best_ub:
                best_ub = 1.0 / smallest
                best_y = [value / smallest for value in average]

    stats = {"iterations": iterations, "oracle_calls": iterations,
             "ub_checks": ub_checks, "lower_bound": best_lb}
    return best_lb, best_x, best_ub, best_y, stats


# --------------------------------------------------------------------------- independent audit
def packing_residual(A, x):
    """max_i (A x)_i - 1: <= 0 means x is a feasible packing solution."""
    return max(sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))) - 1.0


def covering_residual(A, y):
    """1 - min_j (A^T y)_j: <= 0 means y is a feasible covering (dual) solution."""
    m, n = len(A), len(A[0])
    smallest = min(sum(A[i][j] * y[i] for i in range(m)) for j in range(n))
    return 1.0 - smallest


# --------------------------------------------------------------------------- exact LP oracle
def _solve(rows, rhs):
    """Gaussian elimination over Fractions; returns None when the system is singular."""
    size = len(rhs)
    matrix = [[Fraction(value) for value in row] + [Fraction(rhs[index])]
              for index, row in enumerate(rows)]
    for column in range(size):
        pivot = next((r for r in range(column, size) if matrix[r][column] != 0), None)
        if pivot is None:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        factor = matrix[column][column]
        matrix[column] = [value / factor for value in matrix[column]]
        for r in range(size):
            if r != column and matrix[r][column] != 0:
                scale = matrix[r][column]
                matrix[r] = [a - scale * b for a, b in zip(matrix[r], matrix[column])]
    return [matrix[index][size] for index in range(size)]


def _subsets(items, size):
    if size == 0:
        yield []
        return
    for index in range(len(items) - size + 1):
        for rest in _subsets(items[index + 1:], size - 1):
            yield [items[index]] + rest


def exact_packing_optimum(A):
    """max 1^T x s.t. A x <= 1, x >= 0, exactly, via vertex enumeration."""
    m, n = len(A), len(A[0])
    best = Fraction(0)
    for support_size in range(1, min(m, n) + 1):
        for support in _subsets(list(range(n)), support_size):
            for tight in _subsets(list(range(m)), support_size):
                rows = [[A[i][j] for j in support] for i in tight]
                solution = _solve(rows, [1] * support_size)
                if solution is None or any(value <= 0 for value in solution):
                    continue
                feasible = all(sum(A[i][j] * value for j, value in zip(support, solution)) <= 1
                               for i in range(m))
                if feasible:
                    best = max(best, sum(solution))
    return best


def exact_covering_optimum(A):
    """min 1^T y s.t. A^T y >= 1, y >= 0, exactly, via vertex enumeration."""
    m, n = len(A), len(A[0])
    best = None
    for support_size in range(1, min(m, n) + 1):
        for support in _subsets(list(range(m)), support_size):
            for tight in _subsets(list(range(n)), support_size):
                rows = [[A[i][j] for i in support] for j in tight]
                solution = _solve(rows, [1] * support_size)
                if solution is None or any(value <= 0 for value in solution):
                    continue
                feasible = all(sum(A[i][j] * value for i, value in zip(support, solution)) >= 1
                               for j in range(n))
                if feasible:
                    total = sum(solution)
                    if best is None or total < best:
                        best = total
    return best if best is not None else INF


# --------------------------------------------------------------------------- checks
def test_handcrafted_instances():
    A = [[1, 1], [1, 0], [0, 1]]
    exact = exact_packing_optimum(A)
    check(exact == Fraction(1), f"the optimum here is 1, got {exact}")
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=200)
    check(lb <= float(exact) + 1e-9, f"the lower bound {lb} must not exceed the optimum")
    check(ub >= float(exact) - 1e-9, f"the upper bound {ub} must not fall below the optimum")
    check(packing_residual(A, x) <= 1e-9, f"x must be feasible, residual {packing_residual(A, x)}")
    check(covering_residual(A, y) <= 1e-9, f"y must be feasible, residual {covering_residual(A, y)}")
    check(abs(sum(x) - lb) < 1e-9, "the reported lower bound is the objective of x")
    check(abs(sum(y) - ub) < 1e-9, "the reported upper bound is the objective of y")

    # the triangle instance: three constraints, three columns, optimum 3/2
    A = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    exact = exact_packing_optimum(A)
    check(exact == Fraction(3, 2), f"the triangle optimum is 3/2, got {exact}")
    check(exact_covering_optimum(A) == exact, "strong duality: both LPs share the optimum")
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=400)
    check(lb <= 1.5 + 1e-9 <= ub, f"the sandwich [{lb}, {ub}] must contain 1.5")
    check(ub / lb <= 1.35, f"the gap {ub / lb} should be small for this instance")
    check(packing_residual(A, x) <= 1e-9, "certified packing feasibility")
    check(covering_residual(A, y) <= 1e-9, "certified covering feasibility")

    # only the diagonal: every column needs its own constraint
    A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=400)
    check(abs(lb - 3.0) < 1e-9, f"diagonal optimum is 3, got {lb}")
    check(3.0 <= ub <= 3.05, f"the upper bound must close in on 3, got {ub}")
    return exact


def test_sandwich_against_exact_lps():
    mask = (1 << 64) - 1
    seed = 0x0BADC0DE12345678
    worst_gap = 1.0
    tested = 0
    for trial in range(20):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        m = 2 + seed % 3
        n = 2 + (seed >> 7) % 3
        A = []
        for i in range(m):
            row = []
            for j in range(n):
                seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
                row.append((seed >> 11) % 3)
            A.append(row)
        if all(value == 0 for row in A for value in row):
            continue
        if any(all(A[i][j] == 0 for i in range(m)) for j in range(n)):
            continue
        exact = exact_packing_optimum(A)
        if exact == 0:
            continue
        dual = exact_covering_optimum(A)
        check(dual == exact, f"trial {trial}: the two exact LPs disagree ({exact} vs {dual})")
        lb, x, ub, y, _ = multiplicative_weights(A, iterations=600)
        check(lb <= float(exact) + 1e-9,
              f"trial {trial}: lower bound {lb} above the exact optimum {float(exact)}")
        check(ub >= float(exact) - 1e-9,
              f"trial {trial}: upper bound {ub} below the exact optimum {float(exact)}")
        check(packing_residual(A, x) <= 1e-9, f"trial {trial}: x is not feasible")
        check(covering_residual(A, y) <= 1e-9, f"trial {trial}: y is not feasible")
        check(lb <= ub + 1e-9, f"trial {trial}: weak duality broken ({lb} > {ub})")
        worst_gap = max(worst_gap, ub / lb)
        tested += 1
    check(tested >= 10, f"only {tested} random LPs were solved")
    check(worst_gap <= 1.6, f"the worst sandwich gap {worst_gap} is too wide")
    return worst_gap


def test_more_iterations_help():
    A = [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0]]
    exact = float(exact_packing_optimum(A))
    gaps = []
    for iterations in (10, 40, 160, 640):
        lb, x, ub, y, _ = multiplicative_weights(A, iterations=iterations)
        check(lb <= exact + 1e-9 <= ub, f"{iterations} iterations: [{lb}, {ub}] misses {exact}")
        check(packing_residual(A, x) <= 1e-9 and covering_residual(A, y) <= 1e-9,
              "both sides stay certified feasible")
        gaps.append(ub / lb)
    check(gaps[0] >= gaps[-1] - 1e-9,
          f"the sandwich should not widen with more iterations: {gaps}")
    check(gaps[-1] <= 1.1, f"640 iterations should be tight, got gap {gaps[-1]}")
    return gaps


def test_multiplicative_update_actually_moves_prices():
    # the oracle picks the cheapest column under the current prices
    A = [[1, 0], [0, 1]]
    j, value = oracle_pick(A, [0.5, 0.5])
    check(j == 0 and abs(value - 0.5) < 1e-12, "uniform prices break ties by the lowest index")
    j, value = oracle_pick(A, [1.0, 0.5])
    check(j == 1 and abs(value - 0.5) < 1e-12, "the cheaper constraint wins")
    j, value = oracle_pick(A, [0.25, 0.75])
    check(j == 0 and abs(value - 0.25) < 1e-12, "and the oracle follows the prices")

    lb, x, ub, y, stats = multiplicative_weights(A, iterations=50)
    check(stats["oracle_calls"] == 50, "one oracle call per iteration")
    check(abs(lb - 2.0) < 1e-9, f"the diagonal LP has optimum 2, got {lb}")
    check(2.0 <= ub <= 2.02, f"the upper bound must close in on 2, got {ub}")
    check(all(value > 0 for value in x), "both columns must end up in the packing")


def test_degenerate_cases():
    lb, x, ub, y, stats = multiplicative_weights([], iterations=5)
    check(lb == INF and ub == INF, "no constraints means an unbounded packing")
    check("unbounded" in stats["reason"], f"the reason is reported, got {stats}")

    lb, x, ub, y, stats = multiplicative_weights([[1, 0], [0, 0]], iterations=50)
    check(lb == INF, "a column no constraint touches makes the packing unbounded")
    check("misses every constraint" in stats["reason"], f"reason: {stats}")

    A = [[4]]
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=100)
    check(abs(lb - 0.25) < 1e-9 and abs(ub - 0.25) < 1e-9,
          f"a 1x1 instance has optimum 1/4, got {lb},{ub}")
    check(packing_residual(A, x) <= 1e-9 and covering_residual(A, y) <= 1e-9, "certified")

    A = [[1]]
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=3)
    check(abs(lb - 1.0) < 1e-9 and abs(ub - 1.0) < 1e-9, f"a 1x1 unit instance: {lb},{ub}")

    A = [[0, 1], [1, 0]]
    lb, x, ub, y, _ = multiplicative_weights(A, iterations=200)
    check(abs(lb - 2.0) < 1e-9, f"anti-diagonal optimum 2, got {lb}")
    check(2.0 <= ub <= 2.01, f"upper bound close to 2, got {ub}")

    try:
        multiplicative_weights([[1]], iterations=10, epsilon=0.0)
        raise AssertionError("epsilon = 0 must be rejected")
    except AssertionError as error:
        check("epsilon" in str(error), f"unexpected message: {error}")
    try:
        multiplicative_weights([[-1]], iterations=10)
        raise AssertionError("a negative matrix entry must be rejected")
    except AssertionError as error:
        check("non-negative" in str(error), f"unexpected message: {error}")
    try:
        multiplicative_weights([[1, 2], [3]], iterations=10)
        raise AssertionError("a ragged matrix must be rejected")
    except AssertionError as error:
        check("rectangular" in str(error), f"unexpected message: {error}")

    first = multiplicative_weights(A, iterations=64)
    second = multiplicative_weights(A, iterations=64)
    check(first[0] == second[0] and first[1] == second[1] and first[2] == second[2],
          "the method must be deterministic")


def test_bigger_instance():
    n = 30
    mask = (1 << 64) - 1
    seed = 0xFACEFEED0BADF00D
    A = []
    for _ in range(24):
        row = []
        for _ in range(n):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
            row.append((seed >> 17) % 2)
        if all(value == 0 for value in row):
            row[seed % n] = 1
        A.append(row)
    if any(all(A[i][j] == 0 for i in range(len(A))) for j in range(n)):
        A[0][0] = 1
    lb, x, ub, y, stats = multiplicative_weights(A, iterations=3000)
    check(lb > 0, "the lower bound must be positive on a useful instance")
    check(packing_residual(A, x) <= 1e-9, "the packing solution is feasible")
    check(covering_residual(A, y) <= 1e-9, "the dual solution is feasible")
    check(lb <= ub + 1e-9, "weak duality holds on the large instance")
    check(ub / lb <= 1.6, f"the gap {ub / lb} should stay small")
    check(stats["oracle_calls"] == 3000, "every iteration calls the oracle once")
    return len(A), n, lb, ub


def main():
    triangle = test_handcrafted_instances()
    worst_gap = test_sandwich_against_exact_lps()
    gaps = test_more_iterations_help()
    test_multiplicative_update_actually_moves_prices()
    test_degenerate_cases()
    rows, columns, lb, ub = test_bigger_instance()

    print(f"triangle_exact={float(triangle):.4f} worst_random_gap={worst_gap:.4f}")
    print(f"iteration_gaps={[round(gap, 4) for gap in gaps]}")
    print(f"big: rows={rows} columns={columns} lb={lb:.4f} ub={ub:.4f} gap={ub / lb:.4f}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
