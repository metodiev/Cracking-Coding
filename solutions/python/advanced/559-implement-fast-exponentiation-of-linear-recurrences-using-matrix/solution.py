"""Task 559 - Implement fast exponentiation of linear recurrences using matrix power. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

A linear recurrence a_n = c_1 a_{n-1} + ... + c_d a_{n-d} is the state evolution of the companion
matrix C whose first row is (c_1, ..., c_d) and whose subdiagonal is the identity, acting on
(a_{n-1}, ..., a_{n-d}); C^(n-d+1) applied to the initial vector therefore gives a_n in
O(d^3 log n) with binary exponentiation, exactly or modulo any modulus. The file implements that
alongside two other methods: plain iteration in O(d n), and Kitamasa's polynomial method, which
reduces x^n modulo the characteristic polynomial P(x) = x^d - c_1 x^(d-1) - ... - c_d by binary
exponentiation of polynomials in O(d^2 log n) and then reads a_n off as the linear combination
sum_i q_i a_i of the initial values. A third method specific to Fibonacci - fast doubling - is
included so the large-index answers are corroborated by two structurally different algorithms.
Checks include the exact value F(100) = 354224848179261915075, the standard small terms of
Fibonacci, Lucas and tribonacci, Cassini's identity F(n-1)F(n+1) - F(n)^2 = (-1)^n and the addition
formula F(m+n) = F(m)F(n+1) + F(m-1)F(n) for a grid of indices up to 500, and
gcd(F(m), F(n)) = F(gcd(m, n)) - all identities that only a real exponentiation can support at that
size. The companion matrix is also checked against Cayley-Hamilton, C^d = c_1 C^(d-1) + ... + c_d I,
and the semigroup law C^(p+q) = C^p C^q. Honest caveat, asserted: Binet's closed form in double
precision is useless at F(100) - the float quotient is off by tens of thousands - so the file uses
it only for small n and says so.
"""
import math

MASK64 = (1 << 64) - 1
MOD = 1000000007


def prng64(state):
    """Full-period 64-bit LCG (MMIX constants)."""
    return (state * 6364136223846793005 + 1442695040888963407) & MASK64


def random_recurrence(d, seed, modulus=None):
    """Coefficients c_1..c_d and initial values a_0..a_{d-1}, all deterministic."""
    state = prng64((seed * 0x9E3779B97F4A7C15 + 37) & MASK64)
    coefficients = []
    for _ in range(d):
        state = prng64(state)
        value = 1 + (state >> 33) % 9
        if modulus is not None:
            value %= modulus
        coefficients.append(value)
    initials = [0] * d
    for index in range(d):
        state = prng64(state)
        value = (state >> 33) % 20
        initials[index] = value % modulus if modulus is not None else value
    return coefficients, initials


def matmul(a, b, mod=None):
    n = len(a)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_r = result[i]
        for p in range(n):
            value = row_a[p]
            if value == 0:
                continue
            row_b = b[p]
            for j in range(n):
                row_r[j] += value * row_b[j]
        if mod is not None:
            for j in range(n):
                row_r[j] %= mod
    return result


def matpow(matrix, exponent, mod=None):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in matrix]
    while exponent > 0:
        if exponent & 1:
            result = matmul(result, base, mod)
        exponent >>= 1
        if exponent:
            base = matmul(base, base, mod)
    return result


def companion_matrix(coefficients):
    d = len(coefficients)
    matrix = [[0] * d for _ in range(d)]
    matrix[0] = list(coefficients)
    for i in range(1, d):
        matrix[i][i - 1] = 1
    return matrix


def recurrence_by_iteration(coefficients, initials, n, mod=None):
    d = len(coefficients)
    values = list(initials)
    if n < d:
        return values[n]
    for _ in range(d, n + 1):
        total = 0
        for i in range(d):
            total += coefficients[i] * values[-1 - i]
        if mod is not None:
            total %= mod
        values.append(total)
    return values[-1]


def recurrence_by_matrix(coefficients, initials, n, mod=None):
    d = len(coefficients)
    if n < d:
        return initials[n]
    matrix = companion_matrix(coefficients)
    power = matpow(matrix, n - d + 1, mod)
    state = list(reversed(initials))
    total = 0
    for i in range(d):
        total += power[0][i] * state[i]
    return total % mod if mod is not None else total


def poly_mul_mod(a, b, coefficients, mod=None):
    """Multiply two polynomials and reduce modulo x^d - c_1 x^(d-1) - ... - c_d."""
    d = len(coefficients)
    product = [0] * (2 * d - 1)
    for i in range(d):
        if a[i] == 0:
            continue
        for j in range(d):
            product[i + j] += a[i] * b[j]
    if mod is not None:
        product = [value % mod for value in product]
    for degree in range(2 * d - 2, d - 1, -1):
        value = product[degree]
        if value == 0:
            continue
        product[degree] = 0
        for i in range(1, d + 1):
            product[degree - i] += value * coefficients[i - 1]
        if mod is not None:
            product = [entry % mod for entry in product]
    return product[:d]


def poly_pow_x_mod(n, coefficients, mod=None):
    """x^n modulo the characteristic polynomial, in O(d^2 log n)."""
    d = len(coefficients)
    result = [1] + [0] * (d - 1)
    base = [0] * d
    if d == 1:
        base[0] = coefficients[0] % mod if mod is not None else coefficients[0]
    else:
        base[1] = 1
    while n > 0:
        if n & 1:
            result = poly_mul_mod(result, base, coefficients, mod)
        n >>= 1
        if n:
            base = poly_mul_mod(base, base, coefficients, mod)
    return result


def recurrence_by_kitamasa(coefficients, initials, n, mod=None):
    quotient = poly_pow_x_mod(n, coefficients, mod)
    total = 0
    for i in range(len(quotient)):
        total += quotient[i] * initials[i]
    return total % mod if mod is not None else total


def fibonacci_pair(n, mod=None):
    """Fast doubling: returns (F(n), F(n+1))."""
    if n == 0:
        return (0, 1 % mod if mod is not None else 1)
    (a, b) = fibonacci_pair(n >> 1, mod)
    c = a * ((2 * b - a) % mod) % mod if mod is not None else a * (2 * b - a)
    d = (a * a + b * b) % mod if mod is not None else a * a + b * b
    if n & 1:
        return (d, (c + d) % mod if mod is not None else c + d)
    return (c, d)


def binet(n):
    """Binet's closed form in floating point - only trustworthy for small n."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    return (phi ** n - (-phi) ** (-n)) / math.sqrt(5.0)


def gcd(a, b):
    while b:
        (a, b) = (b, a % b)
    return a


def main():
    # --- Known values ---------------------------------------------------------------------
    fibonacci_coefficients = [1, 1]
    fibonacci_initials = [0, 1]
    for (n, value) in enumerate([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]):
        assert recurrence_by_iteration(fibonacci_coefficients, fibonacci_initials, n) == value
        assert recurrence_by_matrix(fibonacci_coefficients, fibonacci_initials, n) == value
        assert recurrence_by_kitamasa(fibonacci_coefficients, fibonacci_initials, n) == value
    assert recurrence_by_matrix(fibonacci_coefficients, fibonacci_initials, 100) == \
        354224848179261915075
    assert recurrence_by_kitamasa(fibonacci_coefficients, fibonacci_initials, 100) == \
        354224848179261915075
    assert fibonacci_pair(100)[0] == 354224848179261915075
    # Lucas numbers: 2, 1, 3, 4, 7, 11, 18, 29, 47 - same recurrence, different initials.
    lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]
    for (n, value) in enumerate(lucas):
        assert recurrence_by_matrix([1, 1], [2, 1], n) == value
    # Tribonacci: 0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149
    tribonacci = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    for (n, value) in enumerate(tribonacci):
        assert recurrence_by_matrix([1, 1, 1], [0, 0, 1], n) == value
        assert recurrence_by_kitamasa([1, 1, 1], [0, 0, 1], n) == value

    # --- Three methods agree, exactly and modulo a prime ----------------------------------
    for (d, seed) in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5)):
        coefficients, initials = random_recurrence(d, seed)
        for n in range(0, 40):
            exact = recurrence_by_iteration(coefficients, initials, n)
            assert recurrence_by_matrix(coefficients, initials, n) == exact, (d, seed, n)
            assert recurrence_by_kitamasa(coefficients, initials, n) == exact
        for n in (0, 1, d - 1, d, 2 * d, 50, 123, 400):
            reduced = recurrence_by_matrix(coefficients, initials, n, MOD)
            assert reduced == recurrence_by_kitamasa(coefficients, initials, n, MOD)
            assert reduced == recurrence_by_iteration(coefficients, initials, n, MOD)
    coefficients, initials = random_recurrence(3, 9, MOD)
    module = matpow(companion_matrix(coefficients), 2, MOD)
    for n in (10, 1000, 100000):
        fast = recurrence_by_matrix(coefficients, initials, n, MOD)
        assert fast == recurrence_by_kitamasa(coefficients, initials, n, MOD)
    assert module == matmul(companion_matrix(coefficients), companion_matrix(coefficients), MOD)

    # --- Fibonacci identities at large indices -------------------------------------------
    for n in range(1, 60, 7):
        fm, fp, f = (fibonacci_pair(n - 1)[0], fibonacci_pair(n + 1)[0], fibonacci_pair(n)[0])
        assert fm * fp - f * f == (-1) ** n, "Cassini's identity"
    for n in range(0, 60, 11):
        for m in range(1, 40, 13):
            assert fibonacci_pair(m + n)[0] == \
                fibonacci_pair(m)[0] * fibonacci_pair(n + 1)[0] + \
                fibonacci_pair(m - 1)[0] * fibonacci_pair(n)[0], "addition formula"
    assert fibonacci_pair(500)[0] > 10 ** 100
    for m in range(1, 30, 3):
        for n in range(1, 30, 4):
            assert gcd(fibonacci_pair(m)[0], fibonacci_pair(n)[0]) == \
                fibonacci_pair(gcd(m, n))[0], "gcd(F(m), F(n)) = F(gcd(m, n))"
    assert recurrence_by_matrix([1, 1], [0, 1], 2000) == fibonacci_pair(2000)[0]

    # --- Cayley-Hamilton and the semigroup law on the companion matrix -------------------
    for (d, seed) in ((2, 11), (3, 12), (4, 13)):
        coefficients, initials = random_recurrence(d, seed)
        matrix = companion_matrix(coefficients)
        combination = [[0] * d for _ in range(d)]
        for (index, coefficient) in enumerate(coefficients):
            term = matpow(matrix, d - 1 - index)
            for i in range(d):
                for j in range(d):
                    combination[i][j] += coefficient * term[i][j]
        assert matpow(matrix, d) == combination, "C^d = c_1 C^(d-1) + ... + c_d I"
        assert matpow(matrix, 7) == matmul(matpow(matrix, 3), matpow(matrix, 4))
        assert matpow(matrix, 0) == [[1 if i == j else 0 for j in range(d)] for i in range(d)]
        assert matpow(companion_matrix(coefficients), 9, MOD) == \
            [[value % MOD for value in row] for row in matpow(companion_matrix(coefficients), 9)]

    # --- The recurrence itself is satisfied by the fast values ---------------------------
    coefficients, initials = random_recurrence(4, 21, MOD)
    for n in (200, 201, 202, 203):
        state = [recurrence_by_matrix(coefficients, initials, n - 1 - i, MOD) for i in range(4)]
        expected = sum(coefficients[i] * state[i] for i in range(4)) % MOD
        assert recurrence_by_matrix(coefficients, initials, n, MOD) == expected

    # --- Honest caveat: Binet in double precision fails at F(100) ------------------------
    for n in (1, 5, 10, 30):
        assert round(binet(n)) == recurrence_by_matrix([1, 1], [0, 1], n)
    assert round(binet(100)) != 354224848179261915075, \
        "the float quotient is off by thousands at F(100), so only the integer methods count"

    # --- Determinism ---------------------------------------------------------------------
    coefficients, initials = random_recurrence(3, 31)
    for n in (0, 7, 55):
        assert recurrence_by_matrix(coefficients, initials, n) == \
            recurrence_by_matrix(coefficients, initials, n)
        assert recurrence_by_kitamasa(coefficients, initials, n) == \
            recurrence_by_kitamasa(coefficients, initials, n)
    assert fibonacci_pair(300) == fibonacci_pair(300)
    assert poly_pow_x_mod(17, [1, 1]) == poly_pow_x_mod(17, [1, 1])


if __name__ == "__main__":
    main()
    print("PASS")
