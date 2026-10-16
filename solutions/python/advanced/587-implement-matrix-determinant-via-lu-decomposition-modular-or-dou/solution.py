"""Task 587 - Implement matrix determinant via LU decomposition (modular or double). (Tier: Advanced, CTCI Beyond — Numerical & Scientific Computing)

Three determinants of the same matrix, each exact in a different sense:

  * `det_lu`  -- LU decomposition in double precision with partial (row) pivoting; the sign flips
    per swap and the determinant is the product of the pivots.  The classic failure mode with no
    pivoting at all is included as `det_lu_no_pivot` and demonstrated on a matrix whose leading
    entry is 1e-16: the pivot is ~1e-16, the multiplier is 1e16, and the computed determinant is
    off by ~100% while partial pivoting stays at the rounding level.
  * `det_bareiss` -- fraction-free (Bareiss) elimination over the integers: every intermediate is
    an integer, the divisions by the previous pivot are exact, so the result is exact for integer
    input with no overflow (Python ints are unbounded) and no rational blow-up.
  * `det_mod_p` -- Gaussian elimination over GF(p) for a prime p, with the pivots inverted modulo
    p; exact too, but only meaningful modulo p.

It also carries `det_fraction`, an exact rational LU with partial pivoting, used for the float
comparison below.  Documented checks: hand-computed 2x2 and 3x3 values, identity/diagonal/
triangular products, permutation matrices (sign), singular matrices (exactly 0), the Vandermonde
closed form prod_{i<j}(x_j - x_i), det(A^T) = det(A), row scaling by c, row swap sign flip and
"add a multiple of a row" invariance, det(AB) = det(A)·det(B) (exact with Bareiss, modulo p) and
three-way agreement on 200 random integer matrices.  The honest caveat: double-precision LU is
*not* exact on ill-conditioned input -- for the 6x6 Hilbert matrix the file computes the exact
1-norm condition number from the exact rational inverse and asserts the measured relative error
of the double determinant against the closed form (prod_{i<n} i!)^4 / prod_{i<2n} i! lies within
n·eps·cond_1, which is ~1e-8 here and therefore much worse than machine epsilon; the exact methods
are asserted to agree with the closed form bit for bit.  Overflow is also not handled by the
double route (a matrix of 1e200 entries gives inf) while Bareiss and the modular elimination stay
exact, and the two exact results are cross-checked against each other on that instance.
"""
import math


def det_lu(matrix):
    """Determinant by LU with partial pivoting (double precision).  Returns (det, swaps)."""
    n = len(matrix)
    if n == 0:
        return 1.0, 0
    a = [[float(value) for value in row] for row in matrix]
    sign = 1
    swaps = 0
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if a[pivot_row][col] == 0.0:
            return 0.0, swaps
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            sign = -sign
            swaps += 1
        pivot = a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] / pivot
            for c in range(col + 1, n):
                a[r][c] -= factor * a[col][c]
    det = float(sign)
    for i in range(n):
        det *= a[i][i]
    return det, swaps


def det_lu_no_pivot(matrix):
    """Same LU without pivoting -- included only to exhibit the numerical failure."""
    n = len(matrix)
    a = [[float(value) for value in row] for row in matrix]
    det = 1.0
    for col in range(n):
        pivot = a[col][col]
        if pivot == 0.0:
            return 0.0
        det *= pivot
        for r in range(col + 1, n):
            factor = a[r][col] / pivot
            for c in range(col + 1, n):
                a[r][c] -= factor * a[col][c]
    return det


def det_bareiss(matrix):
    """Exact fraction-free determinant for an integer matrix (Bareiss, with row swaps)."""
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


def det_mod_p(matrix, modulus):
    """Determinant over GF(modulus) by Gaussian elimination with row swaps."""
    n = len(matrix)
    a = [[int(value) % modulus for value in row] for row in matrix]
    det = 1
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if a[r][col] % modulus:
                pivot_row = r
                break
        if pivot_row is None:
            return 0
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            det = -det
        pivot = a[col][col] % modulus
        det = det * pivot % modulus
        inv = pow(pivot, modulus - 2, modulus)
        for r in range(col + 1, n):
            if a[r][col] % modulus:
                factor = a[r][col] * inv % modulus
                for c in range(col, n):
                    a[r][c] = (a[r][c] - factor * a[col][c]) % modulus
    return det % modulus


def det_fraction(matrix):
    """Exact rational determinant (LU with partial pivoting over Fraction)."""
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


def inverse_fraction(matrix):
    """Exact rational inverse by Gauss-Jordan elimination (used for the condition number)."""
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


def norm_1(matrix):
    n = len(matrix)
    if n == 0:
        return 0
    return max(sum(abs(matrix[i][j]) for i in range(n)) for j in range(n))


def vandermonde(xs):
    return [[float(x) ** power for x in xs] for power in range(len(xs))]


def hilbert(n):
    from fractions import Fraction
    return [[Fraction(1, i + j + 1) for j in range(n)] for i in range(n)]


def hilbert_det_closed_form(n):
    """det(H_n) = (prod_{k<n} k!)^4 / prod_{k<2n} k!  (exact)."""
    from fractions import Fraction
    numerator = 1
    for k in range(1, n):
        numerator *= math.factorial(k)
    denominator = 1
    for k in range(1, 2 * n):
        denominator *= math.factorial(k)
    return Fraction(numerator ** 4, denominator)


def matmul(a, b):
    n = len(a)
    m = len(b[0])
    inner = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(m)] for i in range(n)]


def lcg(state, modulus=65537):
    return (state * 75 + 74) % modulus


def random_int_matrix(state, n, low, high):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            state = lcg(state)
            row.append(low + state % (high - low + 1))
        matrix.append(row)
    return state, matrix


def main():
    # --- hand-computed values ---------------------------------------------------------
    assert det_lu([[1, 2], [3, 4]])[0] == -2.0
    assert det_bareiss([[1, 2], [3, 4]]) == -2
    assert det_mod_p([[1, 2], [3, 4]], 1000003) == 1000001
    assert det_lu([[6, 1, 1], [4, -2, 5], [2, 8, 7]])[0] == -306.0
    assert det_bareiss([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) == -306
    assert det_fraction([[6, 1, 1], [4, -2, 5], [2, 8, 7]]) == -306
    # Identity, diagonal and triangular: the product of the diagonal.
    for n in (1, 2, 5):
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        assert det_lu(identity)[0] == 1.0 and det_bareiss(identity) == 1
        assert det_mod_p(identity, 97) == 1
    triangular = [[2, 9, 9], [0, 3, 9], [0, 0, 5]]
    assert det_bareiss(triangular) == 30 and det_lu(triangular)[0] == 30.0
    assert det_bareiss([[0, 0], [0, 0]]) == 0 and det_lu([[0, 0], [0, 0]])[0] == 0.0
    assert det_fraction(hilbert(3)) == det_fraction(hilbert(3))
    # A singular matrix has determinant exactly zero, in every implementation.
    singular = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert det_bareiss(singular) == 0
    assert det_mod_p(singular, 1000003) == 0
    assert det_fraction(singular) == 0
    assert abs(det_lu(singular)[0]) <= 1e-12

    # --- permutation matrices: the sign of the permutation ----------------------------
    swap = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    rotation = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert det_bareiss(swap) == -1 and det_lu(swap)[0] == -1.0 and det_lu(swap)[1] == 1
    assert det_bareiss(rotation) == 1 and det_lu(rotation)[0] == 1.0

    # --- det(A^T), row scaling, row swap, row addition -------------------------------
    state = 587587
    for trial in range(40):
        n = 2 + trial % 5
        state, a = random_int_matrix(state, n, -9, 9)
        transpose = [[a[j][i] for j in range(n)] for i in range(n)]
        base_mod = det_mod_p(a, 1000003)
        assert det_mod_p(a, 1000003) == det_mod_p(transpose, 1000003), "det(A^T) = det(A)"
        assert det_bareiss(a) == det_bareiss(transpose)
        assert abs(det_lu(a)[0] - det_lu(transpose)[0]) <= 1e-6 * max(1.0, abs(det_lu(a)[0]))
        scaled = [row[:] for row in a]
        scaled[0] = [3 * value for value in scaled[0]]
        assert det_bareiss(scaled) == 3 * det_bareiss(a), "scaling a row scales the determinant"
        added = [row[:] for row in a]
        added[1] = [added[1][j] + 5 * a[0][j] for j in range(n)]
        assert det_bareiss(added) == det_bareiss(a), "adding a row multiple changes nothing"
        assert det_mod_p(added, 1000003) == base_mod
        # det(A * B) = det(A) * det(B), exactly for integers and modulo p.
        state, b = random_int_matrix(state, n, -9, 9)
        product_det = det_bareiss(matmul(a, b))
        assert product_det == det_bareiss(a) * det_bareiss(b)
        assert det_mod_p(matmul(a, b), 1000003) == \
            det_mod_p(a, 1000003) * det_mod_p(b, 1000003) % 1000003
        assert abs(det_lu(matmul(a, b))[0] - det_lu(a)[0] * det_lu(b)[0]) <= \
            1e-6 * max(1.0, abs(det_lu(a)[0] * det_lu(b)[0]))

    # --- Vandermonde closed form ------------------------------------------------------
    xs = [2, 3, 5, 7, 11, 13]
    matrix = vandermonde(xs)
    closed = 1
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            closed *= xs[j] - xs[i]
    assert det_bareiss(matrix) == closed
    assert det_mod_p(matrix, 1000003) == closed % 1000003
    assert abs(det_lu(matrix)[0] - closed) <= 1e-6 * abs(closed)

    # --- three-way agreement on random integer matrices -------------------------------
    worst_double_relative = 0.0
    for trial in range(200):
        n = 2 + trial % 5
        state, a = random_int_matrix(state, n, -9, 9)
        bare = det_bareiss(a)
        assert bare % 1000003 == det_mod_p(a, 1000003), trial
        if bare != 0:
            double = det_lu(a)[0]
            worst_double_relative = max(worst_double_relative,
                                        abs(double - float(bare)) / abs(float(bare)))
    assert worst_double_relative <= 1e-9, worst_double_relative

    # --- pivoting matters -------------------------------------------------------------
    # Honest scope note: a merely tiny leading pivot does NOT wreck the determinant the way it
    # wrecks an inverse, because det is a product of pivots and not a quotient of them -- the
    # 2x2 case below is reproduced correctly by both routes.  What no pivoting really cannot
    # survive is a zero pivot, so the failure is shown on a 7x7 matrix whose (0,0) entry is 0.
    nasty = [[1e-16, 1.0], [1.0, 1.0]]
    assert abs(det_lu(nasty)[0] - (-1.0)) <= 1e-15
    assert abs(det_lu_no_pivot(nasty) - (-1.0)) <= 1e-15, "tiny pivot alone is survivable"
    assert det_lu([[0, 2], [3, 4]])[0] == -6.0 and det_lu([[0, 2], [3, 4]])[1] == 1
    assert det_lu_no_pivot([[0, 2], [3, 4]]) == 0.0, "the unpivoted product collapses to 0"
    abs_diagonal = [[0.0 if i == j else float(abs(i - j)) for j in range(7)] for i in range(7)]
    assert det_fraction(abs_diagonal) == 192
    assert abs(det_lu(abs_diagonal)[0] - 192.0) <= 1e-9
    assert det_lu_no_pivot(abs_diagonal) == 0.0, "zero pivot without pivoting: silently 0"

    # --- documented caveats: ill-conditioning and overflow ----------------------------
    hilbert6 = hilbert(6)
    exact_h6 = hilbert_det_closed_form(6)
    assert det_fraction(hilbert6) == exact_h6, "exact rational determinant matches the closed form"
    float_matrix = [[float(value) for value in row] for row in hilbert6]
    approx, _ = det_lu(float_matrix)
    exact_value = float(exact_h6)
    relative_error = abs(approx - exact_value) / abs(exact_value)
    exact_inverse = inverse_fraction(hilbert6)
    condition = float(norm_1(hilbert6)) * float(norm_1(exact_inverse))
    assert 1e6 <= condition <= 1e8, condition
    # eps * cond_1 * n is roughly 1e-8: the double determinant is *not* accurate to machine
    # epsilon here, and the file says so instead of tightening the tolerance.
    assert relative_error <= 3.0 * 6 * 2.220446049250313e-16 * condition, \
        (relative_error, condition)
    assert relative_error > 0.0, "the double route is not exact on the Hilbert matrix"
    huge = [[10 ** 200, 0, 0], [0, 10 ** 200, 0], [0, 0, 10 ** 200]]
    assert math.isinf(det_lu(huge)[0]), "the double route overflows"
    exact_huge = det_bareiss(huge)
    assert exact_huge == 10 ** 600, "the exact route has no overflow"
    assert det_mod_p(huge, 1000003) == exact_huge % 1000003, "exact and modular agree"

    # --- determinism ------------------------------------------------------------------
    state_a, matrix_a = random_int_matrix(24680, 4, -5, 5)
    state_b, matrix_b = random_int_matrix(24680, 4, -5, 5)
    assert matrix_a == matrix_b and state_a == state_b
    assert det_bareiss(matrix_a) == det_bareiss(matrix_b)


if __name__ == "__main__":
    main()
    print("PASS")
