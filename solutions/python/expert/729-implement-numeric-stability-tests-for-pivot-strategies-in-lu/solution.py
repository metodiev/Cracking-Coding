"""Task 729 - Implement numeric stability tests for pivot strategies in LU.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

Gaussian elimination factors P A Q = L U.  Which pivot to pick is a numerical-stability
decision, not a cosmetic one: taking the diagonal entry blindly ("no pivoting") can divide by a
tiny number and amplify the remaining entries, partial pivoting (largest entry in the column)
bounds that growth by 2^(n-1), and complete pivoting (largest entry in the whole submatrix)
bounds it even more tightly at the cost of a column permutation and extra comparisons.

The file implements all three strategies and measures the two quantities that decide the
argument.  The *growth factor* is the largest magnitude appearing anywhere during elimination
divided by the largest magnitude in the original matrix - a textbook error-amplification bound.
The *backward error* is ||P A Q - L U||_inf / ||A||_inf, which must stay at rounding level for
any stable factorisation.  Solutions of A x = b obtained by forward/backward substitution are
then judged by the residual ||A x - b||, which forward error cannot be trusted for: on the
Hilbert matrix the computed solution can be wrong in several digits while the residual is still
tiny, and both facts are checked.

Verified invariants: the permuted factorisation equals A to rounding for every strategy and
matrix, L is unit lower triangular with a zero strict upper triangle, U has a zero strict lower
triangle, the permutations really are permutations, the growth of partial pivoting obeys the
2^(n-1) bound while no-pivot growth explodes on a matrix with a tiny leading pivot, solution
residuals are at rounding level for the pivoting strategies, and the Hilbert case shows a tiny
residual together with a large forward error.  Boundary cases: 1x1, identity, diagonal (no
pivoting needed), a singular matrix (rejected instead of returning garbage) and a matrix whose
first pivot is exactly zero.

Simplified: scalar doubles with a flat tolerance, no scaling/equilibration, no blocked or
panel-based factorisation, no iterative refinement, no sparse or banded handling, and growth is
measured as a scalar inf-norm ratio rather than a matrix-norm bound.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_matrix(size, seed, spread=6, boost=0.0):
    """Small integer entries with an optional diagonal boost (to avoid zero no-pivot pivots)."""
    state, grid = seed & MASK64, []
    for _ in range(size):
        row = []
        for _ in range(size):
            state, r = splitmix64(state)
            row.append(float(int(r % spread) - spread // 2))
        grid.append(row)
    for i in range(size):
        grid[i][i] += boost
    return grid, state


def hilbert(size):
    return [[1.0 / (i + j + 1) for j in range(size)] for i in range(size)]


def vandermonde(size):
    """Distinct nodes in [1, 1 + 0.3(n-1)]; ill-conditioned but never singular."""
    nodes = [1.0 + 0.3 * i for i in range(size)]
    return [[node ** j for j in range(size)] for node in nodes]


def tiny_pivot_matrix(size):
    """A tiny leading pivot with a nonzero entry below it: no pivoting must lose ~12 digits."""
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        matrix[i][i] = 1.0
    matrix[0][0] = 1e-12
    matrix[0][1] = 1.0
    matrix[1][0] = 1.0
    return matrix


def inf_norm(matrix):
    return max((max(abs(value) for value in row) for row in matrix), default=0.0)


def lu_factor(matrix, strategy="partial", tiny=1e-14):
    """Return (L, U, row_perm, col_perm, growth) for the requested pivot strategy."""
    if strategy not in ("none", "partial", "complete"):
        raise ValueError("unknown pivot strategy")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("the matrix must be square and non-empty")
    a = [row[:] for row in matrix]
    rows = list(range(n))
    columns = list(range(n))
    scale = inf_norm(matrix)
    peaked = scale                              # largest magnitude seen anywhere, input included
    for step in range(n):
        if strategy == "none":
            pivot_row, pivot_col = step, step
        elif strategy == "partial":
            pivot_row = max(range(step, n), key=lambda r: abs(a[r][step]))
            pivot_col = step
        else:
            best = (0.0, step, step)
            for r in range(step, n):
                for c in range(step, n):
                    if abs(a[r][c]) > best[0]:
                        best = (abs(a[r][c]), r, c)
            _, pivot_row, pivot_col = best
        if abs(a[pivot_row][pivot_col]) <= tiny:
            raise ValueError("singular matrix: no acceptable pivot")
        if pivot_row != step:
            a[step], a[pivot_row] = a[pivot_row], a[step]
            rows[step], rows[pivot_row] = rows[pivot_row], rows[step]
        if pivot_col != step:
            for row in a:
                row[step], row[pivot_col] = row[pivot_col], row[step]
            columns[step], columns[pivot_col] = columns[pivot_col], columns[step]
        for r in range(step + 1, n):
            factor = a[r][step] / a[step][step]
            a[r][step] = factor
            for c in range(step + 1, n):
                a[r][c] -= factor * a[step][c]
                peaked = max(peaked, abs(a[r][c]))
        peaked = max(peaked, abs(a[step][step]))
    lower = [[0.0] * n for _ in range(n)]
    upper = [[0.0] * n for _ in range(n)]
    for i in range(n):
        lower[i][i] = 1.0
        for j in range(i):
            lower[i][j] = a[i][j]
        for j in range(i, n):
            upper[i][j] = a[i][j]
    return lower, upper, rows, columns, peaked / scale if scale else 1.0


def permute(matrix, rows, columns):
    """Return the matrix with rows and columns reordered by the two permutations."""
    n = len(matrix)
    return [[matrix[rows[i]][columns[j]] for j in range(n)] for i in range(n)]


def multiply(left, right):
    n = len(left)
    inner = len(right)
    cols = len(right[0])
    return [[sum(left[i][t] * right[t][j] for t in range(inner)) for j in range(cols)]
            for i in range(n)]


def forward_error(left, right):
    n = len(left)
    return max(abs(left[i][j] - right[i][j]) for i in range(n) for j in range(n))


def lu_solve(matrix, rhs, strategy="partial"):
    """Solve A x = b through the permuted factorisation."""
    lower, upper, rows, columns, growth = lu_factor(matrix, strategy)
    n = len(matrix)
    permuted = [rhs[rows[i]] for i in range(n)]
    y = [0.0] * n
    for i in range(n):
        y[i] = permuted[i] - sum(lower[i][j] * y[j] for j in range(i))
    z = [0.0] * n
    for i in range(n - 1, -1, -1):
        z[i] = (y[i] - sum(upper[i][j] * z[j] for j in range(i + 1, n))) / upper[i][i]
    x = [0.0] * n
    for i in range(n):
        x[columns[i]] = z[i]
    return x, growth


def residual(matrix, rhs, solution):
    n = len(matrix)
    return max(abs(sum(matrix[i][j] * solution[j] for j in range(n)) - rhs[i]) for i in range(n))


def is_unit_lower_triangular(lower):
    n = len(lower)
    for i in range(n):
        if lower[i][i] != 1.0:
            return False
        for j in range(i + 1, n):
            if lower[i][j] != 0.0:
                return False
    return True


def is_upper_triangular(upper):
    n = len(upper)
    return all(upper[i][j] == 0.0 for i in range(n) for j in range(i))


def main():
    seed = 0x7291

    # 1. the factorisation identity for all three strategies on well-behaved matrices, plus the
    #    structural properties of L and U.
    cases = []
    for size, spread in ((4, 6), (6, 5), (9, 3)):
        matrix, seed = random_matrix(size, seed, spread, boost=2.0 * spread)
        cases.append((f"random{size}", matrix))
    cases.append(("hilbert6", hilbert(6)))
    cases.append(("vandermonde6", vandermonde(6)))
    for name, matrix in cases:
        for strategy in ("none", "partial", "complete"):
            lower, upper, rows, columns, growth = lu_factor(matrix, strategy)
            rebuilt = permute(matrix, rows, columns)
            check(forward_error(rebuilt, multiply(lower, upper)) < 1e-10,
                  f"{name}/{strategy}: P A Q must equal L U")
            check(is_unit_lower_triangular(lower), f"{name}/{strategy}: L must be unit lower")
            check(is_upper_triangular(upper), f"{name}/{strategy}: U must be upper triangular")
            check(sorted(rows) == list(range(len(matrix))), "the row permutation must be valid")
            check(sorted(columns) == list(range(len(matrix))), "the column permutation must be valid")
            check(growth >= 1.0 - 1e-12, "the growth factor is at least one by definition")

    # 2. the classic failure mode: a tiny leading pivot.  No pivoting produces a multiplier of
    #    1/tiny and a growth factor of about 1e16, while both pivoting strategies stay at one.
    broken = tiny_pivot_matrix(6)
    _, _, _, _, growth_none = lu_factor(broken, "none")
    _, _, _, _, growth_partial = lu_factor(broken, "partial")
    _, _, _, _, growth_complete = lu_factor(broken, "complete")
    check(growth_none > 1e10, "no pivoting must explode on a tiny leading pivot")
    check(growth_partial <= 2 ** (len(broken) - 1),
          "partial pivoting must respect the 2^(n-1) growth bound")
    check(growth_complete <= 2 ** (len(broken) - 1),
          "complete pivoting must respect the same bound")
    check(growth_complete <= growth_partial * 2.0,
          "complete pivoting must not be dramatically worse than partial")
    check(growth_none > growth_partial * 1e9, "the growth comparison must be decisive")

    # 3. solution quality: the residual of the substituted solution is at rounding level for the
    #    pivoting strategies, and the broken matrix really breaks the unpivoted solve.
    for name, matrix in (("hilbert6", hilbert(6)), ("vandermonde6", vandermonde(6))):
        rhs = [1.0 + 0.25 * i for i in range(len(matrix))]
        for strategy in ("partial", "complete"):
            solution, _ = lu_solve(matrix, rhs, strategy)
            check(residual(matrix, rhs, solution) < 1e-8,
                  f"{name}/{strategy}: the residual must be tiny")
    broken_rhs = [1.0 + 0.5 * i for i in range(6)]
    pivoted, _ = lu_solve(broken, broken_rhs, "partial")
    unpivoted, _ = lu_solve(broken, broken_rhs, "none")
    complete_solution, _ = lu_solve(broken, broken_rhs, "complete")
    drift = max(abs(unpivoted[i] - pivoted[i]) for i in range(6))
    check(residual(broken, broken_rhs, pivoted) < 1e-12,
          "the pivoted solve must be accurate on the broken matrix")
    check(max(abs(complete_solution[i] - pivoted[i]) for i in range(6)) < 1e-12,
          "the two pivoting strategies must agree on the broken matrix")
    check(drift > 1e-6, "the unpivoted solve must drift away from the pivoted solution")
    check(residual(broken, broken_rhs, unpivoted) > 1e-9,
          "the unpivoted residual must be visibly damaged by the elimination")

    # 4. backward-stability arithmetic: on the Hilbert matrix the residual stays at rounding
    #    level even though the *forward* error is large - the two are not interchangeable - and
    #    at n = 12 the matrix is numerically singular, which the factorisation must report.
    size = 10
    hilbert_matrix = hilbert(size)
    coefficients = [1.0] * size                      # the exact solution is all ones
    rhs = [sum(hilbert_matrix[i][j] * coefficients[j] for j in range(size)) for i in range(size)]
    solution, _ = lu_solve(hilbert_matrix, rhs, "partial")
    forward = max(abs(solution[i] - coefficients[i]) for i in range(size))
    rounding = residual(hilbert_matrix, rhs, solution)
    check(rounding < 1e-8, "the Hilbert residual must still be at rounding level")
    check(forward > 1e-4, "the Hilbert forward error must be large (ill-conditioning)")
    check(forward > 1e6 * rounding, "forward and backward error must differ by orders of magnitude")
    expect_value_error(lambda: lu_factor(hilbert(12), "partial"),
                       "Hilbert(12) is numerically singular in double precision")

    # 5. boundary cases: 1x1, identity, diagonal, a zero first pivot that needs a swap, and a
    #    singular matrix that must be refused rather than returning garbage.
    check(lu_factor([[4.0]], "none")[4] == 1.0, "a 1x1 matrix must factor trivially")
    check(lu_factor([[4.0]], "partial")[0] == [[1.0]], "L of a 1x1 matrix is the identity")
    identity = [[float(i == j) for j in range(5)] for i in range(5)]
    lower, upper, rows, columns, growth = lu_factor(identity, "complete")
    check(growth == 1.0 and lower == identity and upper == identity,
          "the identity must factor into itself with unit growth")
    diagonal = [[float(2 * (i + 1)) if i == j else 0.0 for j in range(4)] for i in range(4)]
    for strategy in ("none", "partial"):
        check(lu_factor(diagonal, strategy)[2] == [0, 1, 2, 3],
              "a diagonal matrix must not need a row exchange")
    check(sorted(lu_factor(diagonal, "complete")[2]) == [0, 1, 2, 3],
          "complete pivoting may reorder rows, but only by a permutation")
    swap = [[0.0, 1.0], [1.0, 1.0]]
    solution, _ = lu_solve(swap, [1.0, 2.0], "partial")
    check(residual(swap, [1.0, 2.0], solution) < 1e-12,
          "a zero leading pivot must be handled by a row swap")
    expect_value_error(lambda: lu_factor([[1.0, 2.0], [2.0, 4.0]], "none"), "a singular matrix")
    expect_value_error(lambda: lu_factor([[0.0, 0.0], [0.0, 0.0]], "partial"),
                       "the zero matrix is singular")
    expect_value_error(lambda: lu_factor([], "partial"), "an empty matrix is invalid")
    expect_value_error(lambda: lu_factor([[1.0, 2.0, 3.0]], "partial"), "a non-square matrix")
    expect_value_error(lambda: lu_factor([[1.0]], "magic"), "an unknown strategy")

    # 6. determinism and the summary report.
    probe, seed = random_matrix(7, seed)
    first = lu_factor(probe, "partial")
    second = lu_factor(probe, "partial")
    check(first[2] == second[2] and first[3] == second[3] and first[4] == second[4],
          "the factorisation must be deterministic")
    small = [[2.0, 1.0], [1.0, 3.0]]
    exact, _ = lu_solve(small, [3.0, 4.0], "complete")
    check(residual(small, [3.0, 4.0], exact) < 1e-15,
          "a well-conditioned 2x2 system must be solved to rounding")
    print(f"729 growth none={growth_none:.3e} partial={growth_partial:.3f} "
          f"complete={growth_complete:.3f} bound={2 ** 5}")
    print(f"729 hilbert10 forward={forward:.3e} residual={rounding:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
