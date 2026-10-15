"""Task 375 - Gaussian elimination with partial pivoting (doubles). (Tier: Intermediate, CTCI Ch 6)

Solve A x = b for a small dense double matrix.  Forward elimination with partial
pivoting (swap in the row with the largest |pivot|), then back substitution.
Documented conventions:
  * a pivot column whose remaining entries are all <= EPS in absolute value is
    treated as a free column - the system then has no unique solution;
  * after elimination, a row with an all-zero coefficient part but |rhs| > EPS is
    reported "inconsistent";
  * otherwise a rank-deficient consistent system is reported "infinite";
  * results are computed in doubles, so verification by substitution uses the
    documented tolerance TOL = 1e-9 (absolute) - the small hand-checked systems
    below have exact integer solutions that the elimination reproduces to well
    within that tolerance.
Sentinel: the status string ("unique", "infinite", "inconsistent") plus x = None for
anything that is not "unique".

Includes systems that require a pivot swap (leading zero in the first column) and a
4x4 diagonally dominant system, plus LCG cross-checks: a random integer matrix A and
integer x* give b = A x*; the solver must recover x* whenever the elimination reports
a unique solution.
"""

EPS = 1e-12        # pivot/inconsistency threshold for the elimination itself
TOL = 1e-9         # tolerance for verifying a solution by substitution


def solve(matrix, rhs):
    """Returns (status, x) with status in {"unique", "infinite", "inconsistent"}."""
    n = len(matrix)
    augmented = [list(matrix[i]) + [float(rhs[i])] for i in range(n)]

    rank = 0
    for col in range(n):
        pivot_row = -1
        pivot_size = 0.0
        for row in range(rank, n):
            if abs(augmented[row][col]) > pivot_size:
                pivot_size = abs(augmented[row][col])
                pivot_row = row
        if pivot_row == -1 or pivot_size <= EPS:
            continue                                          # free column
        augmented[rank], augmented[pivot_row] = augmented[pivot_row], augmented[rank]
        for row in range(rank + 1, n):
            factor = augmented[row][col] / augmented[rank][col]
            if factor != 0.0:
                for k in range(col, n + 1):
                    augmented[row][k] -= factor * augmented[rank][k]
        rank += 1

    for row in range(rank, n):
        if abs(augmented[row][n]) > EPS:
            return "inconsistent", None

    if rank < n:
        return "infinite", None

    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n]
        for col in range(row + 1, n):
            total -= augmented[row][col] * x[col]
        x[row] = total / augmented[row][row]
    return "unique", x


def residual(matrix, x, rhs):
    """max |A x - b| - the independent substitution check."""
    worst = 0.0
    for i, row in enumerate(matrix):
        total = sum(row[j] * x[j] for j in range(len(row)))
        worst = max(worst, abs(total - rhs[i]))
    return worst


def lcg_system(seed):
    size = 2 + seed % 3
    matrix = []
    for _ in range(size):
        row = []
        for _ in range(size):
            seed = (seed * 75 + 74) % 65537
            row.append(seed % 11 - 5)
        matrix.append(row)
    solution = []
    for _ in range(size):
        seed = (seed * 75 + 74) % 65537
        solution.append(seed % 11 - 5)
    rhs = [sum(matrix[i][j] * solution[j] for j in range(size)) for i in range(size)]
    return matrix, rhs, solution, seed


def main():
    # 2x2, unique integer solution (2, 1)
    status, x = solve([[1, 1], [1, -1]], [3, 1])
    assert status == "unique" and abs(x[0] - 2) < TOL and abs(x[1] - 1) < TOL

    # needs a pivot swap: first pivot is zero
    status, x = solve([[0, 1], [1, 0]], [2, 3])
    assert status == "unique" and abs(x[0] - 3) < TOL and abs(x[1] - 2) < TOL

    # 3x3 permutation system: x = (1, 2, 3)
    status, x = solve([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [3, 2, 1])
    assert status == "unique" and all(abs(x[i] - (i + 1)) < TOL for i in range(3))

    # classic 3x3 with solution (5, 3, -2): checked by hand
    matrix3 = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
    rhs3 = [6, -4, 27]
    assert 5 + 3 - 2 == 6 and 2 * 3 + 5 * -2 == -4 and 2 * 5 + 5 * 3 + 2 == 27
    status, x = solve(matrix3, rhs3)
    assert status == "unique"
    assert abs(x[0] - 5) < TOL and abs(x[1] - 3) < TOL and abs(x[2] + 2) < TOL
    assert residual(matrix3, x, rhs3) < TOL

    # 4x4 diagonally dominant with solution (1, 2, 3, 4)
    matrix4 = [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
    rhs4 = [11, 12, 13, 14]
    assert 2 * 1 + 2 + 3 + 4 == 11 and 1 + 2 * 2 + 3 + 4 == 12
    assert 1 + 2 + 2 * 3 + 4 == 13 and 1 + 2 + 3 + 2 * 4 == 14
    status, x = solve(matrix4, rhs4)
    assert status == "unique"
    for i in range(4):
        assert abs(x[i] - (i + 1)) < TOL, (i, x[i])
    assert residual(matrix4, x, rhs4) < TOL

    # rank-deficient consistent -> infinite (x + y = 1 doubled)
    status, x = solve([[1, 1], [2, 2]], [1, 2])
    assert status == "infinite" and x is None

    # rank-deficient inconsistent -> inconsistent (x + y = 1 and x + y = 2)
    status, x = solve([[1, 1], [2, 2]], [1, 2.5])
    assert status == "inconsistent" and x is None

    # three-way dependent 3x3: row3 = row1 + row2 (rank 2), consistent -> infinite
    status, x = solve([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 3])
    assert status == "infinite" and x is None

    # same dependent matrix with a mismatched rhs -> inconsistent
    status, x = solve([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 4])
    assert status == "inconsistent" and x is None

    # --- LCG cross-checks: b = A x* must recover x* ------------------------
    seed = 37513
    unique_count = 0
    for trial in range(80):
        matrix, rhs, solution, seed = lcg_system(seed)
        status, x = solve(matrix, rhs)
        if status != "unique":
            continue                     # singular random matrix; skipped, documented
        unique_count += 1
        for i in range(len(solution)):
            assert abs(x[i] - solution[i]) < 1e-6, (trial, solution, x)
        assert residual(matrix, x, rhs) < TOL, trial
    assert unique_count >= 40             # the suite really exercises unique systems


if __name__ == "__main__":
    main()
    print("PASS")
