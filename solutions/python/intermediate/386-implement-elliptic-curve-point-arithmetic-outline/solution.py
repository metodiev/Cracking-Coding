"""Task 386 - Implement elliptic curve point arithmetic (outline). (Tier: Intermediate, CTCI Beyond — Security & Privacy)

Short Weierstrass curve E: y^2 = x^3 + a*x + b over the prime field F_p, instantiated as
y^2 = x^3 + 2x + 3 over F_97 (a classic small textbook example; group order 100). A point is
a tuple (x, y), and None represents the point at infinity O. Group law: P + O = P, and for
P != -Q the slope is lambda = (y2 - y1)/(x2 - x1), or lambda = (3x1^2 + a)/(2y1) when
P == Q; then x3 = lambda^2 - x1 - x2 and y3 = lambda(x1 - x3) - y1; P + (-P) = O.
Scalar multiplication uses double-and-add and is cross-checked against naive repeated addition.
"""


def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a, modulus):
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % modulus


def is_on_curve(point, a, b, p):
    if point is None:
        return True                                  # O is on every curve by definition
    x, y = point
    return (y * y - (x ** 3 + a * x + b)) % p == 0


def add(point1, point2, a, p):
    """Group addition on y^2 = x^3 + a*x + b over F_p; None is the identity."""
    if point1 is None:
        return point2
    if point2 is None:
        return point1
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2 and (y1 + y2) % p == 0:
        return None                                  # P + (-P) = O (also covers doubling y = 0)
    if point1 == point2:
        slope = (3 * x1 * x1 + a) * mod_inverse(2 * y1, p) % p
    else:
        slope = (y2 - y1) * mod_inverse(x2 - x1, p) % p
    x3 = (slope * slope - x1 - x2) % p
    y3 = (slope * (x1 - x3) - y1) % p
    return (x3, y3)


def scalar_multiply(k, point, a, p):
    """Double-and-add; k >= 0, k = 0 gives O."""
    result = None
    addend = point
    while k > 0:
        if k & 1:
            result = add(result, addend, a, p)
        addend = add(addend, addend, a, p)
        k >>= 1
    return result


def scalar_multiply_naive(k, point, a, p):
    """Repeated addition; reference oracle for small k."""
    result = None
    for _ in range(k):
        result = add(result, point, a, p)
    return result


def enumerate_points(a, b, p):
    """All affine points as a deterministic sorted list."""
    return [(x, y) for x in range(p) for y in range(p)
            if (y * y - (x ** 3 + a * x + b)) % p == 0]


def curve_order(a, b, p):
    """#E(F_p) = 1 + sum_x (1 + chi(f(x))) with chi the Legendre symbol."""
    count = 1
    for x in range(p):
        rhs = (x ** 3 + a * x + b) % p
        if rhs == 0:
            count += 1
        elif pow(rhs, (p - 1) // 2, p) == 1:
            count += 2
    return count


def main():
    a, b, p = 2, 3, 97
    point = (3, 6)
    assert is_on_curve(point, a, b, p)
    assert (6 * 6 - (3 ** 3 + 2 * 3 + 3)) % 97 == 0

    points = enumerate_points(a, b, p)
    assert len(points) == 99
    assert all(is_on_curve(pt, a, b, p) for pt in points)
    assert (30, 0) in points and (68, 0) in points and (96, 0) in points

    # independent group-order computation via Legendre symbols, and the Hasse bound
    order = curve_order(a, b, p)
    assert order == 100 == len(points) + 1
    assert (order - (p + 1)) ** 2 <= 4 * p           # |#E - (p+1)| <= 2*sqrt(p)

    # identity handling
    assert add(None, point, a, p) == point
    assert add(point, None, a, p) == point
    assert add(None, None, a, p) is None

    # hand-verified group law on P = (3, 6): 2P, 3P, 4P, 5P = O, 6P = P
    p2 = add(point, point, a, p)
    assert p2 == (80, 10)
    p3 = add(p2, point, a, p)
    assert p3 == (80, 87)
    p4 = add(p2, p2, a, p)
    assert p4 == (3, 91)
    assert add(point, (3, 91), a, p) is None         # -P = (3, -6) = (3, 91)
    assert scalar_multiply(5, point, a, p) is None
    assert scalar_multiply(6, point, a, p) == point
    assert is_on_curve(p2, a, b, p) and is_on_curve(p3, a, b, p) and is_on_curve(p4, a, b, p)

    # negation and an order-2 point (doubling must take the y1 + y2 = 0 branch)
    assert add((0, 10), (0, 87), a, p) is None
    assert add((30, 0), (30, 0), a, p) is None

    # commutativity and associativity spot checks on enumerated points
    for i in range(0, 12, 3):
        q = points[i]
        r = points[(i + 4) % len(points)]
        assert add(point, q, a, p) == add(q, point, a, p)
        assert add(add(point, q, a, p), r, a, p) == add(point, add(q, r, a, p), a, p)

    # double-and-add equals naive repeated addition, and results stay on the curve
    for k in range(0, 31):
        assert scalar_multiply(k, point, a, p) == scalar_multiply_naive(k, point, a, p), k
        assert scalar_multiply(k, (80, 10), a, p) == scalar_multiply_naive(k, (80, 10), a, p), k
    for pt in points[:6]:
        for k in range(0, 13):
            got = scalar_multiply(k, pt, a, p)
            assert got == scalar_multiply_naive(k, pt, a, p), (pt, k)
            assert got is None or is_on_curve(got, a, b, p), (pt, k)

    # the group order is 100, so 100 * P = O for every point
    for pt in points:
        assert scalar_multiply(100, pt, a, p) is None, pt

    # LCG-driven keys: double-and-add and naive must agree
    seed = 3131
    for _ in range(40):
        seed = (seed * 75 + 74) % 65537
        k = seed % 50
        seed = (seed * 75 + 74) % 65537
        base = points[seed % len(points)]
        got = scalar_multiply(k, base, a, p)
        assert got == scalar_multiply_naive(k, base, a, p), (k, base)
        assert got is None or is_on_curve(got, a, b, p), (k, base)


if __name__ == "__main__":
    main()
    print("PASS")
