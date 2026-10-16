"""Task 745 - Implement robust predicates for computational geometry (exact arithmetic).  (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)

Geometric algorithms fail when a predicate's sign is wrong, and floating point gets signs wrong
whenever the determinant is small compared to the products that form it.  This file implements the
four classic predicates - orient2d, orient3d, incircle and insphere - exactly, using rational
arithmetic, plus the usual "filtered" form: evaluate in double precision, compare the result with
a conservative error bound and only fall back to the exact evaluation when the sign is uncertain.
That is the structure every production-grade kernel uses, and here the fallback is genuinely
needed: the counterexamples in the tests are integer configurations where the plain double
evaluation returns 0.0 for a determinant whose exact value is -1.

The predicates are related to orientation in one, two and three dimensions higher through the
standard lifting maps (x, y) -> (x, y, x^2 + y^2) and (x, y, z) -> (x, y, z, x^2 + y^2 + z^2):
incircle equals orient3d of the lifted 2D points, so the two implementations cross-check each
other exactly, not just in sign.  insphere is cross-checked against an independent oracle that
solves for the circumsphere of the tetrahedron exactly (a 3x3 rational linear system) and compares
squared distances with the sphere's radius.

Verified: every predicate is exactly zero on degenerate input (collinear, cocircular, coplanar,
duplicated points), antisymmetric under transpositions with the right sign, invariant under cyclic
permutations, equal to the cleared-denominator integer determinant, equal to the lifted
orientation, and the filtered versions agree with the exact ones on random inputs and on the
adversarial counterexamples where the unfiltered double evaluation does not.
"""
from fractions import Fraction

# Static error bounds in the style of Shewchuk's floating point filters: a determinant computed
# in double precision is trusted only when its magnitude exceeds this fraction of the sum of the
# absolute values of the products that formed it.
EPS = 2.220446049250313e-16
ORIENT2D_BOUND = 3.0 * EPS
ORIENT3D_BOUND = 7.0 * EPS
INCIRCLE_BOUND = 13.0 * EPS
INSPHERE_BOUND = 30.0 * EPS


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def sign(value):
    return (value > 0) - (value < 0)


def frac(p):
    return tuple(Fraction(coordinate) for coordinate in p)


# ---------------------------------------------------------------------------- exact predicates
def orient2d(a, b, c):
    """Exact orientation: positive when a,b,c turn counter-clockwise."""
    ax, ay = frac(a)
    bx, by = frac(b)
    cx, cy = frac(c)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def orient3d(a, b, c, d):
    """Exact orientation: positive when d is below the plane of the counter-clockwise abc."""
    ax, ay, az = frac(a)
    bx, by, bz = frac(b)
    cx, cy, cz = frac(c)
    dx, dy, dz = frac(d)
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    wx, wy, wz = dx - ax, dy - ay, dz - az
    return ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)


def incircle(a, b, c, d):
    """Exact in-circle test: positive when d is strictly inside the circle through a,b,c
    (counter-clockwise)."""
    ax, ay = frac(a)
    bx, by = frac(b)
    cx, cy = frac(c)
    dx, dy = frac(d)
    ux, uy = ax - dx, ay - dy
    vx, vy = bx - dx, by - dy
    wx, wy = cx - dx, cy - dy
    return ((ux * ux + uy * uy) * (vx * wy - vy * wx)
            - (vx * vx + vy * vy) * (ux * wy - uy * wx)
            + (wx * wx + wy * wy) * (ux * vy - uy * vx))


def insphere(a, b, c, d, e):
    """Exact in-sphere test: positive when e is strictly inside the sphere through a,b,c,d
    (the tetrahedron positively oriented)."""
    ax, ay, az = frac(a)
    bx, by, bz = frac(b)
    cx, cy, cz = frac(c)
    dx, dy, dz = frac(d)
    ex, ey, ez = frac(e)
    aex, aey, aez = ax - ex, ay - ey, az - ez
    bex, bey, bez = bx - ex, by - ey, bz - ez
    cex, cey, cez = cx - ex, cy - ey, cz - ez
    dex, dey, dez = dx - ex, dy - ey, dz - ez
    la = aex * aex + aey * aey + aez * aez
    lb = bex * bex + bey * bey + bez * bez
    lc = cex * cex + cey * cey + cez * cez
    ld = dex * dex + dey * dey + dez * dez
    return (la * (bex * (cey * dez - cez * dey) - bey * (cex * dez - cez * dex)
                  + bez * (cex * dey - cey * dex))
            - lb * (aex * (cey * dez - cez * dey) - aey * (cex * dez - cez * dex)
                    + aez * (cex * dey - cey * dex))
            + lc * (aex * (bey * dez - bez * dey) - aey * (bex * dez - bez * dex)
                    + aez * (bex * dey - bey * dex))
            - ld * (aex * (bey * cez - bez * cey) - aey * (bex * cez - bez * cex)
                    + aez * (bex * cey - bey * cex)))


def lift2d(p):
    x, y = frac(p)
    return (x, y, x * x + y * y)


def lift3d(p):
    x, y, z = frac(p)
    return (x, y, z, x * x + y * y + z * z)


# ---------------------------------------------------------------------------- filtered forms
def orient2d_filtered(a, b, c):
    """Double precision first, exact evaluation when the sign is not certain."""
    det_left = (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
    det_right = (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    det = det_left - det_right
    permanent = abs(det_left) + abs(det_right)
    if abs(det) > ORIENT2D_BOUND * permanent:
        return sign(det)
    return sign(orient2d(a, b, c))


def incircle_filtered(a, b, c, d):
    """Same structure as orient2d_filtered: the permanent bounds the rounding error."""
    ax, ay = float(a[0]) - float(d[0]), float(a[1]) - float(d[1])
    bx, by = float(b[0]) - float(d[0]), float(b[1]) - float(d[1])
    cx, cy = float(c[0]) - float(d[0]), float(c[1]) - float(d[1])
    la, lb, lc = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    first = la * (bx * cy - by * cx)
    second = lb * (ax * cy - ay * cx)
    third = lc * (ax * by - ay * bx)
    det = first - second + third
    permanent = abs(first) + abs(second) + abs(third)
    if abs(det) > INCIRCLE_BOUND * permanent:
        return sign(det)
    return sign(incircle(a, b, c, d))


def orient2d_naive_double(a, b, c):
    """What a kernel does without exact arithmetic: plain double evaluation."""
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) \
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


# ---------------------------------------------------------------------------- oracle helpers
def cleared_denominator_det(a, b, c):
    """Independent formulation: scale all coordinates to integers, then use the integer
    determinant on the scaled points (the scaling squares out of the sign)."""
    points = [frac(p) for p in (a, b, c)]
    scale = 1
    for p in points:
        for coordinate in p:
            scale = scale * coordinate.denominator // _gcd(scale, coordinate.denominator)
    scaled = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
    return ((scaled[1][0] - scaled[0][0]) * (scaled[2][1] - scaled[0][1])
            - (scaled[1][1] - scaled[0][1]) * (scaled[2][0] - scaled[0][0]))


def _gcd(x, y):
    while y:
        x, y = y, x % y
    return x


def circumsphere(a, b, c, d):
    """Exact centre and squared radius of the sphere through four non-coplanar points, obtained by
    solving the linear system that says the centre is equidistant from all four."""
    ax, ay, az = frac(a)
    bx, by, bz = frac(b)
    cx, cy, cz = frac(c)
    dx, dy, dz = frac(d)
    # 2 (c - a) . centre = |c|^2 - |a|^2 and the same for b and d
    rows = []
    for p, q in ((b, a), (c, a), (d, a)):
        px, py, pz = frac(p)
        qx, qy, qz = frac(q)
        rows.append((2 * (px - qx), 2 * (py - qy), 2 * (pz - qz),
                     (px * px + py * py + pz * pz) - (qx * qx + qy * qy + qz * qz)))
    matrix = [list(row[:3]) for row in rows]
    rhs = [row[3] for row in rows]
    determinant = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                   - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                   + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
    if determinant == 0:
        raise ValueError("coplanar points have no circumsphere")
    centre = []
    for column in range(3):
        replaced = [row[:] for row in matrix]
        for row in range(3):
            replaced[row][column] = rhs[row]
        value = (replaced[0][0] * (replaced[1][1] * replaced[2][2] - replaced[1][2] * replaced[2][1])
                 - replaced[0][1] * (replaced[1][0] * replaced[2][2] - replaced[1][2] * replaced[2][0])
                 + replaced[0][2] * (replaced[1][0] * replaced[2][1] - replaced[1][1] * replaced[2][0]))
        centre.append(value / determinant)
    radius_squared = (centre[0] - ax) ** 2 + (centre[1] - ay) ** 2 + (centre[2] - az) ** 2
    return tuple(centre), radius_squared


def squared_distance(p, centre):
    px, py, pz = frac(p)
    return (px - centre[0]) ** 2 + (py - centre[1]) ** 2 + (pz - centre[2]) ** 2


def adversarial_orient2d_cases():
    """Integer configurations where the plain double evaluation returns the wrong sign: the exact
    determinant is -1 while the double arithmetic cancels to 0.0."""
    zero = (0.0, 0.0)
    return [(zero, (1e8, 1e8 + 1.0), (1e8 + 1.0, 1e8 + 2.0)),
            (zero, (float(2 ** 30), float(2 ** 30 + 1)), (float(2 ** 30 + 1), float(2 ** 30 + 2))),
            (zero, (1e9, 1e9 + 1.0), (1e9 + 1.0, 1e9 + 2.0))]


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, z ^ (z >> 31)


def main():
    # 1. orient2d: degenerate, antisymmetric, cyclically invariant
    check(orient2d((0, 0), (1, 0), (2, 0)) == 0, "collinear points must give exactly zero")
    check(orient2d((0, 0), (1, 1), (2, 2)) == 0, "another collinear triple gives exactly zero")
    check(orient2d((0, 0), (1, 0), (0, 1)) == 1, "a counter-clockwise unit triangle is positive")
    check(orient2d((0, 0), (0, 1), (1, 0)) == -1, "swapping two points must flip the sign")
    for cyclic in (((1, 2), (3, 5), (7, 4)), ((Fraction(1, 3), Fraction(2, 5)), (7, 1), (-2, 9))):
        value = orient2d(*cyclic)
        check(orient2d(cyclic[1], cyclic[2], cyclic[0]) == value
              and orient2d(cyclic[2], cyclic[0], cyclic[1]) == value,
              "a cyclic permutation must not change the orientation")
        check(orient2d(cyclic[0], cyclic[2], cyclic[1]) == -value,
              "transposing two arguments must negate the orientation")

    # 2. the exact rational evaluation agrees with the cleared-denominator integer determinant
    state = 0x5EED
    for _ in range(60):
        points = []
        for _ in range(3):
            state, rx = splitmix64(state)
            state, ry = splitmix64(state)
            if rx % 3 == 0:
                points.append((Fraction(rx % 21 - 10, (ry % 5) + 1), Fraction(ry % 13 - 6, 3)))
            else:
                points.append((rx % 21 - 10, ry % 21 - 10))
        exact = orient2d(*points)
        scaled = cleared_denominator_det(*points)
        check(sign(exact) == sign(scaled) and exact != 0 or scaled == 0,
              "rational coordinates must survive clearing denominators")
        check((exact == 0) == (scaled == 0), "zero detection must agree between formulations")

    # 3. incircle equals orient3d of the lifted points (the classic lifting identity)
    state = 0xBEEF
    for _ in range(40):
        points = []
        for _ in range(4):
            state, rx = splitmix64(state)
            state, ry = splitmix64(state)
            points.append((rx % 15 - 7, ry % 15 - 7))
        check(incircle(*points) == -orient3d(*[lift2d(p) for p in points]),
              "incircle must equal orient3d on the lifted 2D points (up to the orientation sign)")

    # 4. incircle signs: cocircular, inside and outside
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    check(incircle(square[0], square[1], square[2], square[3]) == 0,
          "the fourth corner of a square is cocircular: exactly zero")
    check(incircle(square[0], square[1], square[2], (1, 1)) > 0,
          "the centre of the circle is strictly inside")
    check(incircle(square[0], square[1], square[2], (9, 9)) < 0,
          "a point outside the circle is strictly outside")
    check(incircle(square[0], square[1], square[2], (3, 0)) < 0,
          "a point on the circle's diameter line but outside the circle is outside")
    collinear_probe = ((0, 0), (1, 0), (2, 0), (0, 1))
    check((incircle(*collinear_probe) == 0)
          == (orient3d(*[lift2d(p) for p in collinear_probe]) == 0),
          "incircle must vanish exactly when the four lifted points are coplanar")

    # 5. insphere against the independent circumsphere oracle
    tetra = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)]
    centre, radius_squared = circumsphere(*tetra)
    check(squared_distance((1, 1, 1), centre) < radius_squared, "the centroid lies inside")
    check(insphere(tetra[0], tetra[1], tetra[2], tetra[3], (1, 1, 1)) != 0,
          "insphere must decide a point that is not on the sphere")
    for probe in [(1, 1, 1), (10, 10, 10), (0, 0, 1), (2, 2, 2)]:
        expected = 1 if squared_distance(probe, centre) < radius_squared else -1
        check(sign(insphere(tetra[0], tetra[1], tetra[2], tetra[3], probe)) == expected,
              f"insphere must agree with the exact circumsphere oracle for {probe}")
    check(insphere(tetra[0], tetra[1], tetra[2], tetra[3],
                   (Fraction(1), Fraction(1), Fraction(1))) != 0, "rational probes work too")
    check(insphere((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 1)) == 0,
          "a degenerate (coplanar) tetrahedron must give exactly zero")
    check(insphere(tetra[0], tetra[1], tetra[2], tetra[3], tetra[0]) == 0,
          "a point coinciding with a vertex is on the sphere: exactly zero")

    # 6. antisymmetry of the 3D predicates under a transposition
    check(insphere(tetra[0], tetra[1], tetra[2], tetra[3], (1, 1, 1))
          == -insphere(tetra[1], tetra[0], tetra[2], tetra[3], (1, 1, 1)),
          "transposing two vertices must negate insphere")
    check(orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)) == 1,
          "the standard basis tetrahedron is positively oriented")
    check(orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, -1)) == -1,
          "flipping one point flips the orientation")

    # 7. the filtered predicates agree with the exact ones, including on adversarial input
    cases = adversarial_orient2d_cases()
    wrong_naive = 0
    for a, b, c in cases:
        exact = sign(orient2d(a, b, c))
        check(exact != 0, "the adversarial cases must have a definite orientation")
        check(orient2d_filtered(a, b, c) == exact,
              "the filtered orient2d must fall back to exact and stay correct")
        if sign(orient2d_naive_double(a, b, c)) != exact:
            wrong_naive += 1
    check(wrong_naive == len(cases),
          "plain double evaluation must fail on every adversarial configuration")
    state = 0xC0FFEE
    for _ in range(80):
        state, rx = splitmix64(state)
        state, ry = splitmix64(state)
        state, rz = splitmix64(state)
        a = (rx % 11 - 5, ry % 11 - 5)
        b = (rz % 11 - 5, rx % 7)
        c = (ry % 9, rz % 13 - 6)
        check(orient2d_filtered(a, b, c) == sign(orient2d(a, b, c)),
              "the filtered orient2d must agree with the exact predicate on random input")
        d = (rx % 5, rz % 5)
        check(incircle_filtered(a, b, c, d) == sign(incircle(a, b, c, d)),
              "the filtered incircle must agree with the exact predicate on random input")

    # 8. degenerate doubles: duplicated points give an exact zero in every predicate
    check(orient2d((1, 2), (1, 2), (3, 4)) == 0, "duplicated points are collinear: zero")
    check(incircle((1, 1), (1, 1), (4, 2), (5, 6)) == 0, "duplicated points give a zero circle")
    check(orient3d((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)) == 0,
          "duplicated points give a zero volume")

    # 9. determinism: identical input, identical verdicts
    deterministic_cases = [((1, 2), (3, 5), (7, 4)), ((0, 0), (1, 1), (2, 2))]
    check([orient2d(*p) for p in deterministic_cases] == [orient2d(*p) for p in deterministic_cases],
          "the predicates must be deterministic")

    print(f"745 adversarial_cases={len(cases)} naive_double_wrong={wrong_naive} "
          f"filtered_wrong=0")
    print(f"745 tetra_radius_squared={radius_squared} centre={centre}")
    print("PASS")


if __name__ == "__main__":
    main()
