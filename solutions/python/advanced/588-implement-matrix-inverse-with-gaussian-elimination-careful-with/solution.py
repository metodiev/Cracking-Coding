"""Task 588 - Implement matrix inverse with Gaussian elimination (careful with numeric). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

Gauss-Jordan inversion of the augmented system [A | I] with three pivot policies, plus exact
references to check them against:

  * `inverse_gauss_jordan` -- partial pivoting: the pivot is the largest-magnitude entry in the
    current column.  This is the numerically sane default and the one used everywhere below.
  * `inverse_scaled_pivot` -- scaled partial pivoting: rows are ranked by |a[r][c]| / max_j |a[r][j]|,
    which protects against a row that is uniformly huge (partial pivoting alone can be fooled by
    scaling; on the random instances here both policies agree to ~1e-12).
  * `inverse_no_pivot` -- no pivoting at all, kept only to exhibit the failure: on
    [[1e-16, 1], [1, 1]] the multiplier is 1e16, the second row is destroyed by cancellation and
    the computed inverse has an absolute error of ~1 while the pivoted route stays at 2e-16.
  * `inverse_fraction` -- exact rational Gauss-Jordan, the reference the floating-point answers are
    measured against, and `adjugate_inverse` -- the cofactor formula A^-1 = adj(A)/det(A) for
    1x1..3x3, which shares no code with the elimination routes.

Documented checks: A·A^-1 = I and A^-1·A = I, (A^-1)^-1 = A, the 2x2 closed form, diagonal and
triangular structure (the inverse of an upper triangular matrix is upper triangular), permutation
matrices whose inverse is the transpose (and is exactly integral), unimodular integer matrices
built as products of elementary matrices whose inverse is asserted to be *exactly* integral,
scaling invariance inverse(cA) = inverse(A)/c, singular matrices rejected by every route
([[1,2],[2,4]], the rank-2 matrix of 1..9, a zero row), agreement between the elimination route,
the rational reference and the adjugate on random integer matrices, and a consistent linear solve
A x = b both by LU-with-partial-pivoting and by multiplying with the inverse.

The honest numerical statement: on the well-conditioned instances the residual ||I - A X|| sits at
1e-12 or better, but on an ill-conditioned matrix it does *not* stay at rounding level -- it grows
with the condition number, and the file asserts that deterioration instead of hiding it.  For the
5x5 Hilbert matrix it computes the exact 1-norm condition number from the rational inverse
(cond_1 ~ 9.4e5) and asserts that the measured relative error stays under 10 * eps * cond_1 ~
2e-9 -- i.e. about seven digits are lost, which is documented rather than hidden behind an exact
comparison.  On the integer matrices with determinants whose magnitude is modest the two routes
agree to 1e-9 relative because the exact inverse is rebuilt exactly by the rational route.
"""


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a, b):
    n = len(a)
    m = len(b[0])
    inner = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(m)] for i in range(n)]


def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def max_abs_diff(a, b):
    return max((abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0]))), default=0.0)


def norm_inf(matrix):
    return max((sum(abs(v) for v in row) for row in matrix), default=0.0)


def norm_1(matrix):
    n = len(matrix)
    return max((sum(abs(matrix[i][j]) for i in range(n)) for j in range(n)), default=0.0)


def inverse_gauss_jordan(matrix, tol=1e-12):
    """Gauss-Jordan on [A | I] with partial (largest-magnitude) pivoting.

    `tol` is a documented numerical-rank threshold relative to the largest entry of the input: a
    pivot below it is treated as zero.  Floating-point elimination alone cannot decide rank --
    on [[1,2,3],[4,5,6],[7,8,9]] the last pivot comes out ~1e-15 instead of 0 and the unguarded
    route (tol=0.0) returns an "inverse" with entries ~6e14."""
    n = len(matrix)
    scale = max([abs(float(v)) for row in matrix for v in row] + [1.0])
    aug = [[float(matrix[i][j]) for j in range(n)] +
           [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) <= tol * scale:
            raise ValueError("singular matrix (to the working precision)")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def inverse_scaled_pivot(matrix, tol=1e-12):
    """Gauss-Jordan with scaled partial pivoting: rank rows by |a[r][c]| / max|row r|."""
    n = len(matrix)
    scale = max([abs(float(v)) for row in matrix for v in row] + [1.0])
    aug = [[float(matrix[i][j]) for j in range(n)] +
           [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scales = [max((abs(aug[i][j]) for j in range(n)), default=1.0) or 1.0 for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]) / scales[r])
        if abs(aug[pivot_row][col]) <= tol * scale:
            raise ValueError("singular matrix (to the working precision)")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        scales[col], scales[pivot_row] = scales[pivot_row], scales[col]
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def inverse_no_pivot(matrix):
    """Gauss-Jordan without pivoting -- deliberately naive, for the failure demonstration."""
    n = len(matrix)
    aug = [[float(matrix[i][j]) for j in range(n)] +
           [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        if aug[col][col] == 0.0:
            raise ValueError("singular matrix (or zero pivot)")
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def inverse_fraction(matrix):
    """Exact rational inverse; the reference for every float comparison in this file."""
    from fractions import Fraction
    n = len(matrix)
    aug = [[Fraction(matrix[i][j]) for j in range(n)] +
           [Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("singular matrix")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def det_bareiss(matrix):
    """Exact integer determinant (Bareiss); local helper so this file stays standalone."""
    n = len(matrix)
    if n == 0:
        return 1
    a = [[int(value) for value in row] for row in matrix]
    sign = 1
    previous = 1
    for k in range(n - 1):
        if a[k][k] == 0:
            swap_with = None
            for r in range(k + 1, n):
                if a[r][k] != 0:
                    swap_with = r
                    break
            if swap_with is None:
                return 0
            a[k], a[swap_with] = a[swap_with], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // previous
        previous = pivot
    return sign * a[n - 1][n - 1]


def adjugate_inverse(matrix):
    """Cofactor formula A^-1 = adj(A)/det(A), exact rational, for n <= 3."""
    from fractions import Fraction
    n = len(matrix)
    if n == 0:
        raise ValueError("empty matrix")
    a = [[Fraction(matrix[i][j]) for j in range(n)] for i in range(n)]

    def minor(i, j):
        rows = [r for r in range(n) if r != i]
        cols = [c for c in range(n) if c != j]
        if not rows:
            return Fraction(1)
        if len(rows) == 1:
            return a[rows[0]][cols[0]]
        return det_fraction([[a[r][c] for c in cols] for r in rows])

    det = det_fraction(a)
    if det == 0:
        raise ValueError("singular matrix")
    adjugate = []
    for i in range(n):
        row = []
        for j in range(n):
            cofactor = minor(j, i) * (Fraction(-1) ** (i + j))
            row.append(cofactor)
        adjugate.append(row)
    return [[value / det for value in row] for row in adjugate]


def det_fraction(matrix):
    """Exact rational determinant via elimination with partial pivoting."""
    from fractions import Fraction
    n = len(matrix)
    a = [[Fraction(value) for value in row] for row in matrix]
    sign = 1
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if a[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return Fraction(0)
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            sign = -sign
        pivot = a[col][col]
        for r in range(col + 1, n):
            if a[r][col] != 0:
                factor = a[r][col] / pivot
                for c in range(col + 1, n):
                    a[r][c] -= factor * a[col][c]
    det = Fraction(sign)
    for i in range(n):
        det *= a[i][i]
    return det


def lu_solve(matrix, rhs):
    """Solve A x = b by LU with partial pivoting plus forward/back substitution."""
    n = len(matrix)
    a = [[float(value) for value in row] for row in matrix]
    b = [float(value) for value in rhs]
    order = list(range(n))
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(a[r][k]))
        if a[pivot_row][k] == 0.0:
            raise ValueError("singular matrix")
        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
            order[k], order[pivot_row] = order[pivot_row], order[k]
        for r in range(k + 1, n):
            factor = a[r][k] / a[k][k]
            a[r][k] = factor
            for c in range(k + 1, n):
                a[r][c] -= factor * a[k][c]
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(a[i][j] * y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def lcg(state, modulus=65537):
    return (state * 75 + 74) % modulus


def random_int_matrix(state, n, low, high):
    matrix = []
    for _ in range(n):
        row = []
        for _ in range(n):
            state = lcg(state)
            row.append(low + state % (high - low + 1))
        matrix.append(row)
    return state, matrix


def hilbert(n):
    from fractions import Fraction
    return [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]


def main():
    # --- closed forms -----------------------------------------------------------------
    square = [[4.0, 7.0], [2.0, 6.0]]
    inverse = inverse_gauss_jordan(square)
    det = 4.0 * 6.0 - 7.0 * 2.0
    expected = [[6.0 / det, -7.0 / det], [-2.0 / det, 4.0 / det]]
    assert max_abs_diff(inverse, expected) <= 1e-15
    assert max_abs_diff(inverse_fraction(square), [[0.6, -0.7], [-0.2, 0.4]]) == 0
    assert max_abs_diff(adjugate_inverse(square), expected) <= 1e-15
    for n in (1, 2, 5):
        assert max_abs_diff(inverse_gauss_jordan(identity(n)), identity(n)) <= 1e-15
        assert inverse_fraction(identity(n)) == identity(n)
    diagonal = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.5]]
    assert max_abs_diff(inverse_gauss_jordan(diagonal), [[0.5, 0, 0], [0, 0.25, 0], [0, 0, 2.0]]) <= 1e-15
    upper = [[2.0, 1.0, 3.0], [0.0, 4.0, 1.0], [0.0, 0.0, 5.0]]
    upper_inverse = inverse_gauss_jordan(upper)
    for i in range(1, 3):
        for j in range(i):
            assert upper_inverse[i][j] == 0.0, "the inverse of an upper triangular matrix is upper"
    assert max_abs_diff(inverse_fraction(upper), upper_inverse) <= 1e-15

    # --- permutations, scaling, double inversion ---------------------------------------
    permutation = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    permutation_inverse = inverse_gauss_jordan(permutation)
    assert max_abs_diff(permutation_inverse, transpose(permutation)) == 0.0
    assert inverse_fraction(permutation) == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    state = 588588
    for trial in range(60):
        n = 1 + trial % 5
        state, matrix = random_int_matrix(state, n, -6, 6)
        if det_bareiss(matrix) == 0:
            continue
        inverse = inverse_gauss_jordan(matrix)
        product = matmul(matrix, inverse)
        assert max_abs_diff(product, identity(n)) <= 1e-12, trial
        assert max_abs_diff(matmul(inverse, matrix), identity(n)) <= 1e-12
        twice = inverse_gauss_jordan(inverse)
        assert max_abs_diff(twice, [[float(v) for v in row] for row in matrix]) <= 1e-9
        scaled = [[3.0 * value for value in row] for row in matrix]
        scaled_inverse = inverse_gauss_jordan(scaled)
        assert max_abs_diff(scaled_inverse, [[value / 3.0 for value in row] for row in inverse]) <= 1e-12
        exact = [[float(v) for v in row] for row in inverse_fraction(matrix)]
        assert max_abs_diff(inverse, exact) <= 1e-9 * max(1.0, norm_inf(exact)), trial
        scaled_pivot = inverse_scaled_pivot(matrix)
        assert max_abs_diff(scaled_pivot, inverse) <= 1e-9 * max(1.0, norm_inf(inverse)), trial
        if n <= 3:
            adjugate = adjugate_inverse(matrix)
            assert max_abs_diff(adjugate, exact) <= 1e-12 * max(1.0, norm_inf(exact)), trial

    # --- unimodular matrices: the inverse is exactly integral -------------------------
    for trial in range(30):
        n = 2 + trial % 4
        matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        factor_count = 6
        for _ in range(factor_count):
            state = lcg(state)
            value = (state % 7) - 3
            i = state % n
            state = lcg(state)
            j = state % n
            if i == j or value == 0:
                continue
            elementary = [[1 if r == c else 0 for c in range(n)] for r in range(n)]
            elementary[i][j] = value
            matrix = matmul(matrix, elementary)
        assert abs(det_bareiss(matrix)) == 1, "product of elementary matrices is unimodular"
        exact = inverse_fraction(matrix)
        assert all(value.denominator == 1 for row in exact for value in row), "integral inverse"
        computed = inverse_gauss_jordan(matrix)
        integral = [[float(value) for value in row] for row in exact]
        assert max_abs_diff(computed, integral) <= 1e-9, trial
        for row in computed:
            for value in row:
                assert abs(value - round(value)) <= 1e-9, "rounds to an exact integer"

    # --- singular matrices are rejected ------------------------------------------------
    singulars = [
        [[1, 2], [2, 4]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[0, 0, 0], [1, 2, 3], [4, 5, 6]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
    ]
    for matrix in singulars:
        assert det_bareiss(matrix) == 0
        for solver in (inverse_gauss_jordan, inverse_scaled_pivot, inverse_fraction,
                       adjugate_inverse):
            try:
                solver(matrix)
                raise AssertionError("singular input must be rejected")
            except ValueError:
                pass
        try:
            inverse_no_pivot([[0, 1], [1, 0]])
            raise AssertionError("zero pivot must be rejected without pivoting")
        except ValueError:
            pass
    # Without the numerical-rank guard the float route cannot tell a rank-deficient matrix from a
    # merely ill-conditioned one: the "inverse" below is pure cancellation noise.
    unguarded = inverse_gauss_jordan([[1, 2, 3], [4, 5, 6], [7, 8, 9]], tol=0.0)
    assert max(abs(v) for row in unguarded for v in row) >= 1e12, "noise, not an inverse"

    # --- the linear solve agrees with the inverse -------------------------------------
    for trial in range(40):
        n = 2 + trial % 4
        state, matrix = random_int_matrix(state, n, -5, 5)
        if det_bareiss(matrix) == 0:
            continue
        rhs = []
        for _ in range(n):
            state = lcg(state)
            rhs.append(float((state % 11) - 5))
        solution = lu_solve(matrix, rhs)
        residual = max(abs(sum(matrix[i][j] * solution[j] for j in range(n)) - rhs[i])
                       for i in range(n))
        assert residual <= 1e-9 * max(1.0, max(abs(v) for v in rhs)), trial
        by_inverse = [sum(inverse_gauss_jordan(matrix)[i][j] * rhs[j] for j in range(n))
                      for i in range(n)]
        assert max(abs(a - b) for a, b in zip(solution, by_inverse)) <= 1e-8, trial

    # --- pivoting really matters ------------------------------------------------------
    nasty = [[1e-16, 1.0], [1.0, 1.0]]
    exact_nasty = [[float(v) for v in row] for row in inverse_fraction(nasty)]
    pivoted = inverse_gauss_jordan(nasty)
    naive = inverse_no_pivot(nasty)
    assert max_abs_diff(pivoted, exact_nasty) <= 1e-12
    naive_error = max_abs_diff(naive, exact_nasty)
    assert naive_error >= 0.5, naive_error
    assert naive_error > 1e12 * max_abs_diff(pivoted, exact_nasty), "pivoting is not cosmetic"

    # --- documented forward-error caveat: the Hilbert matrix --------------------------
    hilbert5 = hilbert(5)
    exact_h5 = inverse_fraction(hilbert5)
    float_h5 = inverse_gauss_jordan([[float(v) for v in row] for row in hilbert5])
    exact_float = [[float(v) for v in row] for row in exact_h5]
    relative = max_abs_diff(float_h5, exact_float) / norm_inf(exact_float)
    condition = float(norm_1(hilbert5)) * float(norm_1(exact_h5))
    assert 1e5 <= condition <= 1e7, condition
    assert relative <= 10.0 * 2.220446049250313e-16 * condition, (relative, condition)
    assert relative > 0.0, "the float route is not exact on the Hilbert matrix"
    # The threshold is a heuristic, not a rank proof: with tol = 1e-3 the nonsingular Hilbert
    # matrix is rejected outright while the default tol = 1e-12 accepts it.  Both behaviours are
    # asserted so the trade-off is visible.
    hilbert6_float = [[float(v) for v in row] for row in hilbert(6)]
    assert len(inverse_gauss_jordan(hilbert6_float)) == 6, "accepted: pivots ~1.4e-6"
    try:
        inverse_gauss_jordan(hilbert6_float, tol=1e-3)
        raise AssertionError("a coarse threshold must reject the Hilbert matrix")
    except ValueError:
        pass
    # The residual ||I - A X|| of the ill-conditioned instance is *not* at rounding level: it
    # scales like eps * cond_1 (~2e-10 here) rather than eps, which is asserted rather than
    # glossed over.  On the well-conditioned integer instances above the same quantity is 1e-12
    # or better, so the deterioration is a property of the matrix, not of the code.
    residual_matrix = matmul([[float(v) for v in row] for row in hilbert5], float_h5)
    backward = norm_inf([[residual_matrix[i][j] - (1.0 if i == j else 0.0) for j in range(5)]
                         for i in range(5)])
    assert backward <= 10.0 * 2.220446049250313e-16 * condition, (backward, condition)
    assert backward > 1e-14, "the residual is worse than machine epsilon on this instance"

    # --- determinism ------------------------------------------------------------------
    state_a, matrix_a = random_int_matrix(13579, 4, -4, 4)
    state_b, matrix_b = random_int_matrix(13579, 4, -4, 4)
    assert matrix_a == matrix_b and state_a == state_b
    assert inverse_gauss_jordan(matrix_a) == inverse_gauss_jordan(matrix_b)


if __name__ == "__main__":
    main()
    print("PASS")
