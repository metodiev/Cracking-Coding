"""Task 376 - Exact linear systems over rationals (integer coefficients). (Tier: Intermediate, CTCI Ch 6)

The same Gaussian elimination as task 375, but every entry is an exact rational
number: a small Fraction type on top of gcd (implemented here per language - Python
could use fractions.Fraction, but this mirrors the other languages) keeps num/den
normalised with den > 0.  All arithmetic is exact, so:
  * the systems of task 375 return exact rational solutions - every assertion is an
    EQUALITY of rationals, NOT a tolerance check (contrast with 375);
  * no pivoting strategy is required: the first non-zero pivot can be taken because
    exact arithmetic never introduces rounding, so a zero pivot really is zero
    (documented; 375's partial pivoting exists to limit rounding, which cannot
    happen here);
  * sentinels: status "unique"/"infinite"/"inconsistent", x = None unless unique.

Cross-check with floats: a compact double-precision Gaussian implemented HERE (same
file, duplicating 375's algorithm) must agree with the exact result to 1e-9 - this is
the documented link between the two tasks; with small integer inputs the doubles
round the rationals faithfully.

Inputs and expected solutions include genuine fractions (1/3, 1/6) whose float images
cannot be represented exactly - those are asserted exactly on the rationals.
"""


class Rational:
    """Exact fraction with normalised sign and gcd-reduced terms."""

    __slots__ = ("num", "den")

    def __init__(self, num, den=1):
        if den == 0:
            raise ZeroDivisionError("rational with zero denominator")
        if den < 0:
            num, den = -num, -den
        g = gcd(abs(num), den)
        if g == 0:
            self.num, self.den = 0, 1
        else:
            self.num, self.den = num // g, den // g

    def __add__(self, other):
        return Rational(self.num * other.den + other.num * self.den, self.den * other.den)

    def __sub__(self, other):
        return Rational(self.num * other.den - other.num * self.den, self.den * other.den)

    def __mul__(self, other):
        return Rational(self.num * other.num, self.den * other.den)

    def __truediv__(self, other):
        if other.num == 0:
            raise ZeroDivisionError("division by zero rational")
        return Rational(self.num * other.den, self.den * other.num)

    def __neg__(self):
        return Rational(-self.num, self.den)

    def __eq__(self, other):
        return self.num == other.num and self.den == other.den

    def __hash__(self):
        return hash((self.num, self.den))

    def is_zero(self):
        return self.num == 0

    def __repr__(self):
        return str(self.num) if self.den == 1 else "%d/%d" % (self.num, self.den)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def solve_rational(matrix, rhs):
    """Exact Gaussian elimination; returns (status, x)."""
    n = len(matrix)
    augmented = [[Rational(matrix[i][j]) for j in range(n)] + [Rational(rhs[i])]
                 for i in range(n)]

    rank = 0
    for col in range(n):
        pivot_row = -1
        for row in range(rank, n):
            if not augmented[row][col].is_zero():             # exact zero test
                pivot_row = row
                break
        if pivot_row == -1:
            continue                                          # genuinely free column
        augmented[rank], augmented[pivot_row] = augmented[pivot_row], augmented[rank]
        for row in range(rank + 1, n):
            if augmented[row][col].is_zero():
                continue
            factor = augmented[row][col] / augmented[rank][col]
            for k in range(col, n + 1):
                augmented[row][k] = augmented[row][k] - factor * augmented[rank][k]
        rank += 1

    for row in range(rank, n):
        if not augmented[row][n].is_zero():
            return "inconsistent", None
    if rank < n:
        return "infinite", None

    x = [Rational(0)] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n]
        for col in range(row + 1, n):
            total = total - augmented[row][col] * x[col]
        x[row] = total / augmented[row][row]
    return "unique", x


def solve_double(matrix, rhs):
    """Compact double-precision copy of 375's elimination, used only as a cross-check."""
    eps = 1e-12
    n = len(matrix)
    augmented = [[float(matrix[i][j]) for j in range(n)] + [float(rhs[i])] for i in range(n)]
    rank = 0
    for col in range(n):
        pivot_row = max(range(rank, n), key=lambda r: abs(augmented[r][col]), default=-1)
        if pivot_row == -1 or abs(augmented[pivot_row][col]) <= eps:
            continue
        augmented[rank], augmented[pivot_row] = augmented[pivot_row], augmented[rank]
        for row in range(rank + 1, n):
            factor = augmented[row][col] / augmented[rank][col]
            for k in range(col, n + 1):
                augmented[row][k] -= factor * augmented[rank][k]
        rank += 1
    if rank < n:
        return None
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = augmented[row][n]
        for col in range(row + 1, n):
            total -= augmented[row][col] * x[col]
        x[row] = total / augmented[row][row]
    return x


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
    # 2x2 integer system: x = 2, y = 1, exactly
    status, x = solve_rational([[1, 1], [1, -1]], [3, 1])
    assert status == "unique"
    assert x == [Rational(2), Rational(1)]
    assert x[0].num == 2 and x[0].den == 1

    # pivot swap: leading zero, x = (3, 2)
    status, x = solve_rational([[0, 1], [1, 0]], [2, 3])
    assert status == "unique" and x == [Rational(3), Rational(2)]

    # classic 3x3, solution (5, 3, -2)
    status, x = solve_rational([[1, 1, 1], [0, 2, 5], [2, 5, -1]], [6, -4, 27])
    assert status == "unique" and x == [Rational(5), Rational(3), Rational(-2)]

    # 4x4 diagonally dominant, solution (1, 2, 3, 4)
    status, x = solve_rational([[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]],
                               [11, 12, 13, 14])
    assert status == "unique" and x == [Rational(1), Rational(2), Rational(3), Rational(4)]

    # genuine fractions: 3x = 1 -> x = 1/3
    status, x = solve_rational([[3]], [1])
    assert status == "unique" and x == [Rational(1, 3)]
    assert x[0].num == 1 and x[0].den == 3

    # 2x + 4y = 1, x - y = 0 -> x = y = 1/6 (verified: 2/6 + 4/6 = 1, 1/6 - 1/6 = 0)
    status, x = solve_rational([[2, 4], [1, -1]], [1, 0])
    assert status == "unique" and x == [Rational(1, 6), Rational(1, 6)]

    # 5x + 2y = 1, x + y = 0 -> x = 1/3, y = -1/3
    status, x = solve_rational([[5, 2], [1, 1]], [1, 0])
    assert status == "unique" and x == [Rational(1, 3), Rational(-1, 3)]

    # sentinels behave exactly like 375's
    status, x = solve_rational([[1, 1], [2, 2]], [1, 2])
    assert status == "infinite" and x is None
    status, x = solve_rational([[1, 1], [2, 2]], [1, 5])          # 2(x+y) = 5 vs x+y = 1
    assert status == "inconsistent" and x is None
    status, x = solve_rational([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 3])
    assert status == "infinite" and x is None
    status, x = solve_rational([[1, 0, 0], [0, 1, 0], [1, 1, 0]], [1, 2, 4])
    assert status == "inconsistent" and x is None

    # --- float cross-check: the double solver agrees to 1e-9 ---------------
    double_systems = [
        ([[1, 1], [1, -1]], [3, 1], [Rational(2), Rational(1)]),
        ([[0, 1], [1, 0]], [2, 3], [Rational(3), Rational(2)]),
        ([[1, 1, 1], [0, 2, 5], [2, 5, -1]], [6, -4, 27],
         [Rational(5), Rational(3), Rational(-2)]),
        ([[3]], [1], [Rational(1, 3)]),
        ([[2, 4], [1, -1]], [1, 0], [Rational(1, 6), Rational(1, 6)]),
        ([[5, 2], [1, 1]], [1, 0], [Rational(1, 3), Rational(-1, 3)]),
    ]
    for matrix, rhs, exact in double_systems:
        approx = solve_double(matrix, rhs)
        assert approx is not None
        for i, value in enumerate(exact):
            assert abs(approx[i] - value.num / value.den) < 1e-9, (matrix, rhs, i)

    # --- LCG cross-checks: exact recovery of the planted solution ----------
    seed = 37613
    unique_count = 0
    for trial in range(80):
        matrix, rhs, planted, seed = lcg_system(seed)
        status, x = solve_rational(matrix, rhs)
        if status != "unique":
            continue                       # singular random matrix; skipped, documented
        unique_count += 1
        assert x == [Rational(v) for v in planted], (trial, planted, x)
        # exact substitution A x == b, no tolerance involved
        for i, row in enumerate(matrix):
            total = Rational(0)
            for j, coefficient in enumerate(row):
                total = total + Rational(coefficient) * x[j]
            assert total == Rational(rhs[i]), (trial, i)
        approx = solve_double(matrix, rhs)
        assert approx is not None
        for i, value in enumerate(planted):
            assert abs(approx[i] - value) < 1e-9, (trial, i)
    assert unique_count >= 40


if __name__ == "__main__":
    main()
    print("PASS")
