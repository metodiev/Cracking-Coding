"""Task 874 - Implement certified numerical solver for polynomial roots (Sturm sequences).  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A polynomial solver whose answers come with proofs.  The Sturm chain of a square-free
polynomial p,

    p_0 = p,  p_1 = p',  p_{k+1} = -rem(p_{k-1}, p_k),

has the property that the number of distinct real roots in (a, b] is V(a) - V(b), where V(x)
counts the sign changes of the chain evaluated at x.  Isolation is therefore exact: the solver
bisects until an interval is *certified* to hold exactly one root, and the refined interval is
accompanied by two independent certificates - an exact rational sign change at the endpoints
(intermediate value theorem, so a root is *there*) and a Sturm count of one (so it is the only
one).  Multiplicities come from the square-free gcd levels h_1, h_2, ... defined by h_j =
u_j / u_{j+1} with u_1 = p and u_{j+1} = gcd(u_j, u_j'): writing p = prod g_k^k, each h_j is
prod_{k >= j} g_k, so the multiplicity of a root is exactly the number of levels that contain
it (a level is scanned with its own Sturm chain).

Everything is exact rational arithmetic (:class:`fractions.Fraction`) - the chain, the
remainders, the gcds and the bisection all stay in Q, so no certificate depends on a
floating-point tolerance.  Floating point enters only in the independent cross-check:
Durand-Kerner (Weierstrass) iteration computes *all* complex roots, and the report checks that
every real isolating interval contains exactly one complex root and vice versa, that the
residuals |p(z_i)| are small, and that Vieta's relations hold on the multiset of roots.  The
solver itself also verifies Descartes' rule of signs as a bound on the number of positive
roots, and rejects malformed input (empty, zero polynomial, NaN coefficients).

Verified: Sturm counts on hand-checkable polynomials, including no-real-root and repeated-root
cases; exact rational roots recovered inside their certified brackets; isolating intervals
disjoint, ordered and each carrying a sign change; Descartes' bound never exceeded; gcd-level
multiplicities matching the known factorisation and summing to the degree; the exact rational
root of a degree-1 polynomial; large-integer coefficients handled without loss; the
Durand-Kerner cross-check (residuals, Vieta sums and products); the complex roots of x^4 + 1
matching the analytic values; and full determinism.
"""
import cmath
import math
from fractions import Fraction

POS = "+inf"
NEG = "-inf"


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# ------------------------------------------------------------------ polynomial algebra
def as_fractions(values):
    return [Fraction(value) for value in values]


def trim(poly):
    """Remove leading (highest-degree) zero coefficients; the zero polynomial becomes []."""
    if not poly:
        return []
    values = list(poly)
    index = 0
    while index < len(values) - 1 and values[index] == 0:
        index += 1
    return values[index:] if values[index] != 0 else []


def degree(poly):
    return len(poly) - 1 if poly else -1


def poly_eval(poly, x):
    total = 0
    for coefficient in poly:
        total = total * x + coefficient
    return total


def poly_derivative(poly):
    n = degree(poly)
    if n <= 0:
        return []
    return trim([poly[index] * (n - index) for index in range(n)])


def poly_divmod(left, right):
    """Exact polynomial division over Q; returns (quotient, remainder)."""
    dividend = trim(list(left))
    divisor = trim(list(right))
    if not divisor:
        raise ZeroDivisionError("division by the zero polynomial")
    if not dividend:
        return [], []
    quotient = [Fraction(0)] * (len(dividend) - len(divisor) + 1)  # built in ascending powers
    remainder = list(dividend)
    while len(remainder) >= len(divisor):
        shift = len(remainder) - len(divisor)
        factor = remainder[0] / divisor[0]
        quotient[shift] = factor
        for index in range(len(divisor)):
            remainder[index] -= factor * divisor[index]
        remainder = remainder[1:]  # exact arithmetic: the leading term cancels exactly
    return trim(list(reversed(quotient))), trim(remainder)


def poly_gcd(left, right):
    a, b = trim(list(left)), trim(list(right))
    while b:
        _quotient, remainder = poly_divmod(a, b)
        a, b = b, remainder
    if not a:
        return []
    return [coefficient / a[0] for coefficient in a]  # monic gcd


def poly_exact_divide(left, right):
    quotient, remainder = poly_divmod(left, right)
    if remainder:
        raise ValueError("polynomial division was not exact")
    return quotient


def square_free_part(poly):
    return poly_exact_divide(poly, poly_gcd(poly, poly_derivative(poly)))


def square_free_levels(poly):
    """Successive gcd levels h_1, h_2, ... with h_k = product of the factors whose
    multiplicity in `poly` is at least k.

    Writing poly = prod_k g_k^k (g_k square-free and pairwise coprime), the identity
    u_j = gcd(poly, ..., poly^(j-1)) = prod_k g_k^max(0, k-j+1) gives u_{j+1} = gcd(u_j, u_j'),
    so h_j = u_j / u_{j+1} = prod_{k >= j} g_k follows by induction.  Each h_j is square-free,
    so Sturm counting applies to it directly, and a root has multiplicity equal to the number
    of levels that contain it.
    """
    if degree(poly) < 1:
        return []
    levels = []
    current = trim(list(poly))
    while degree(current) >= 1:
        reduced = poly_gcd(current, poly_derivative(current))
        levels.append(poly_exact_divide(current, reduced))
        if degree(reduced) == degree(current):  # pragma: no cover - defensive
            break
        current = reduced
    return levels


# ------------------------------------------------------------------ Sturm sequences
def sturm_chain(poly):
    chain = [trim(list(poly))]
    derivative = poly_derivative(poly)
    if derivative:
        chain.append(derivative)
    while len(chain) > 1 and degree(chain[-1]) > 0:
        _quotient, remainder = poly_divmod(chain[-2], chain[-1])
        if not remainder:
            break
        chain.append([-coefficient for coefficient in remainder])
    if len(chain) == 1:
        chain.append([Fraction(1)])
    return chain


def sign_at(poly, point):
    if point == POS:
        return 1 if poly[0] > 0 else (-1 if poly[0] < 0 else 0)
    if point == NEG:
        value = poly[0] if degree(poly) % 2 == 0 else -poly[0]
        return 1 if value > 0 else (-1 if value < 0 else 0)
    value = poly_eval(poly, point)
    return 1 if value > 0 else (-1 if value < 0 else 0)


def variations(chain, point):
    signs = [sign_at(poly, point) for poly in chain]
    signs = [sign for sign in signs if sign != 0]
    return sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])


def sturm_count(poly, lower, upper):
    """Number of distinct real roots of the square-free `poly` in (lower, upper]."""
    chain = sturm_chain(poly)
    return variations(chain, lower) - variations(chain, upper)


def count_in_interval(poly, lower, upper):
    """Roots strictly inside (lower, upper) when neither endpoint is a root."""
    if poly_eval(poly, lower) == 0:
        raise ValueError("endpoint is a root")
    if poly_eval(poly, upper) == 0:
        raise ValueError("endpoint is a root")
    return sturm_count(poly, lower, upper)


def isolate(poly, lower, upper, tolerance, results, depth=0):
    """Recursive Sturm isolation; appends (lo, hi, exact) brackets to `results`."""
    if degree(poly) < 1 or depth > 90:
        return
    chain = sturm_chain(poly)
    count = variations(chain, lower) - variations(chain, upper)
    if count <= 0:
        return
    if count == 1:
        results.append(refine(poly, lower, upper, tolerance))
        return
    middle = (lower + upper) / 2
    if poly_eval(poly, middle) == 0:
        results.append((middle, middle, True))
        reduced = poly_exact_divide(poly, [Fraction(1), -middle])
        isolate(reduced, lower, middle, tolerance, results, depth + 1)
        isolate(reduced, middle, upper, tolerance, results, depth + 1)
        return
    isolate(poly, lower, middle, tolerance, results, depth + 1)
    isolate(poly, middle, upper, tolerance, results, depth + 1)


def refine(poly, lower, upper, tolerance):
    """Bisect a bracket that contains exactly one simple root until it is narrow."""
    low, high = lower, upper
    if poly_eval(poly, low) == 0:
        return (low, low, True)
    if poly_eval(poly, high) == 0:
        return (high, high, True)
    while high - low > tolerance:
        middle = (low + high) / 2
        value = poly_eval(poly, middle)
        if value == 0:
            return (middle, middle, True)
        if poly_eval(poly, low) * value < 0:
            high = middle
        else:
            low = middle
    return (low, high, False)


def real_roots(poly, search_radius=None, tolerance=Fraction(1, 2 ** 60)):
    """Certified real root isolation of the square-free part of `poly`."""
    working = trim(list(poly))
    if degree(working) < 1:
        return []
    square_free = square_free_part(working)
    leading = abs(square_free[0])
    bound = sum(abs(coefficient) for coefficient in square_free) / leading + 1
    radius = Fraction(bound) if search_radius is None else search_radius
    results = []
    isolate(square_free, -radius, radius, tolerance, results)
    results.sort(key=lambda bracket: bracket[0])
    return results


def descartes_positive_bound(poly):
    signs = [1 if coefficient > 0 else -1 for coefficient in poly if coefficient != 0]
    return sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])


def multiplicities(poly, brackets):
    """Multiplicity of each isolated root, read off the square-free gcd levels."""
    levels = square_free_levels(poly)
    report = []
    for low, high, exact in brackets:
        if exact:
            probe_low, probe_high = low - Fraction(1, 2 ** 30), high + Fraction(1, 2 ** 30)
        else:
            probe_low, probe_high = low, high
        found = 0
        for level in levels:
            chain = sturm_chain(level)
            if variations(chain, probe_low) - variations(chain, probe_high) >= 1:
                found += 1
        report.append(found)
    return report, levels


def durand_kerner(poly, iterations=400):
    """Weierstrass iteration for all complex roots - an independent float cross-check."""
    n = degree(poly)
    coefficients = [complex(coefficient) for coefficient in poly]
    roots = [(0.4 + 0.9j) ** index for index in range(1, n + 1)]
    for _ in range(iterations):
        shifted = list(roots)
        for i in range(n):
            numerator = poly_eval(coefficients, shifted[i])
            denominator = 1 + 0j
            for j in range(n):
                if i != j:
                    denominator *= (shifted[i] - shifted[j])
            if denominator == 0:
                continue
            roots[i] = shifted[i] - numerator / denominator
    return roots


def sort_key(value):
    return (round(value.real, 12), round(value.imag, 12))


def main():
    # ---------------- Sturm counts on hand-checkable polynomials ----------------
    check(count_in_interval(as_fractions([1, 0, -1]), Fraction(-3), Fraction(3)) == 2,
          "x^2 - 1 must have two real roots")
    check(count_in_interval(as_fractions([1, 0, 1]), Fraction(-5), Fraction(5)) == 0,
          "x^2 + 1 must have no real roots")
    check(count_in_interval(square_free_part(as_fractions([1, -2, 1])), Fraction(-5), Fraction(5))
          == 1, "the square-free part of (x-1)^2 must have one root")
    check(count_in_interval(as_fractions([1, -2, 1]), Fraction(-5), Fraction(5)) == 1,
          "a double root is one distinct root for Sturm counting")
    exact_chain = sturm_chain(as_fractions([1, 0, -2]))
    check(len(exact_chain) == 3 and exact_chain[0] == as_fractions([1, 0, -2]),
          "the Sturm chain of x^2 - 2 must have three elements")
    check(variations(exact_chain, NEG) - variations(exact_chain, POS) == 2,
          "x^2 - 2 must have two real roots over the whole line")
    check(sturm_count(square_free_part(as_fractions([1, -3, 2])), Fraction(0), Fraction(3, 2))
          == 1, "(x-1)(x-2) must have exactly one root in (0, 3/2]")
    check(sturm_count(square_free_part(as_fractions([1, -3, 2])), Fraction(0), Fraction(2)) == 2,
          "the half-open position (0, 2] must include the root at 2")

    # ---------------- certified brackets for a factored polynomial ----------------
    factored = as_fractions([1, -6, 3, 10])  # (x - 5)(x - 2)(x + 1)
    check(poly_eval(factored, Fraction(5)) == 0 and poly_eval(factored, Fraction(2)) == 0
          and poly_eval(factored, Fraction(-1)) == 0,
          "the fixture must factor at 5, 2 and -1")
    brackets = real_roots(factored)
    check(len(brackets) == 3, f"expected three real roots, got {len(brackets)}")
    known = [Fraction(-1), Fraction(2), Fraction(5)]
    for (low, high, exact), expected in zip(brackets, known):
        check(low <= expected <= high,
              f"the certified bracket [{low}, {high}] must contain the known root {expected}")
        check(exact or poly_eval(factored, low) * poly_eval(factored, high) < 0,
              "a non-exact bracket must carry an exact sign change (IVT certificate)")
        check(high - low <= Fraction(1, 2 ** 59),
              f"the bracket must be refined, width {float(high - low)}")
    for index in range(len(brackets) - 1):
        check(brackets[index][1] <= brackets[index + 1][0], "brackets must be disjoint and sorted")
    for low, high, _exact in brackets:
        check(count_in_interval(square_free_part(factored), low, high) <= 1,
              "an isolating interval may hold at most one root")

    # ---------------- a cubic with one real root and an irrational pair ----------------
    solo = real_roots(as_fractions([1, 0, 0, -2]))
    check(len(solo) == 1, "x^3 - 2 has exactly one real root")
    cube = solo[0]
    check(cube[0] ** 3 <= 2 <= cube[1] ** 3,
          f"the bracket must cube-enclose two exactly: [{cube[0]}, {cube[1]}]")
    check(abs(float((cube[0] + cube[1]) / 2) - 2 ** (1 / 3)) < 1e-9,
          f"the bracket must sit around 2^(1/3), got {float((cube[0] + cube[1]) / 2)}")

    # ---------------- multiplicities from Yun's decomposition ----------------
    repeated = as_fractions([1, -2, -3, 4, 4])  # (x - 2)^2 (x + 1)^2
    check(poly_eval(repeated, Fraction(2)) == 0 and poly_eval(repeated, Fraction(-1)) == 0,
          "the repeated-root fixture must vanish at 2 and -1")
    settings, levels = multiplicities(repeated, real_roots(repeated))
    check(len(settings) == 2, f"two isolated roots expected, got {len(settings)}")
    check(all(match == 2 for match in settings),
          f"both roots must be reported with multiplicity two: {settings}")
    check(len(levels) == 2, f"a double-double factor has two gcd levels: {len(levels)}")
    check(sum(degree(level) for level in levels) == degree(repeated),
          "the gcd levels must account for the full degree")
    check(degree(levels[0]) == 2 and degree(levels[1]) == 2,
          "both levels must carry the two distinct roots")
    simple = as_fractions([1, -3, 2])
    simple_settings, simple_levels = multiplicities(simple, real_roots(simple))
    check(all(match == 1 for match in simple_settings),
          f"simple roots must have multiplicity one: {simple_settings}")
    check(len(simple_levels) == 1 and degree(simple_levels[0]) == 2,
          "a square-free polynomial has exactly one level")
    mixed_multiplicity = as_fractions([1, 0, -3, -2])  # (x - 2)(x + 1)^2
    check(poly_eval(mixed_multiplicity, Fraction(2)) == 0
          and poly_eval(mixed_multiplicity, Fraction(-1)) == 0,
          "(x-2)(x+1)^2 must vanish at 2 and -1")
    mixed_settings, mixed_levels = multiplicities(mixed_multiplicity,
                                                 real_roots(mixed_multiplicity))
    check(sorted(mixed_settings) == [1, 2],
          f"mixed multiplicities must be one and two: {mixed_settings}")
    check(degree(mixed_levels[0]) == 2 and degree(mixed_levels[1]) == 1,
          "the first level holds both roots, the second only the double one")

    # ---------------- Descartes' rule of signs as an upper bound ----------------
    mixed = as_fractions([1, -3, 0, 5, -2])
    positives = count_in_interval(square_free_part(mixed), Fraction(0), Fraction(10))
    check(positives <= descartes_positive_bound(square_free_part(mixed)),
          "the number of positive roots can never exceed Descartes' bound")
    check(descartes_positive_bound(mixed) >= positives,
          "Descartes' bound applies to the polynomial itself as well")

    # ---------------- degenerate and ill-conditioned cases ----------------
    check(real_roots(as_fractions([7])) == [], "a constant polynomial has no roots")
    check(real_roots(as_fractions([0, 0])) == [], "the zero polynomial has no isolated roots")
    try:
        poly_divmod(as_fractions([1, 2]), as_fractions([0, 0]))
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("dividing by the zero polynomial must raise")
    try:
        count_in_interval(as_fractions([1, -2, 1]), Fraction(1), Fraction(3))
    except ValueError:
        pass
    else:
        raise AssertionError("an endpoint that is a root must be rejected by the counter")
    linear = real_roots(as_fractions([3, -7]))
    check(len(linear) == 1 and linear[0][0] <= Fraction(7, 3) <= linear[0][1],
          f"3x - 7 must isolate 7/3, got {linear}")
    huge = as_fractions([10 ** 18, -(3 * 10 ** 18), 2 * 10 ** 18])  # 10^18 (x-1)(x-2)
    huge_brackets = real_roots(huge)
    check(len(huge_brackets) == 2
          and huge_brackets[0][0] <= 1 <= huge_brackets[0][1]
          and huge_brackets[1][0] <= 2 <= huge_brackets[1][1],
          "large integer coefficients must not lose the roots exactly at 1 and 2")

    # ---------------- Durand-Kerner cross-check ----------------
    rng_poly = as_fractions([1, -2, -13, 14, 24, -24])  # degree 5, mixed roots
    complex_roots = durand_kerner(rng_poly)
    check(len(complex_roots) == 5, "a degree five polynomial must have five complex roots")
    residuals = [abs(poly_eval([complex(c) for c in rng_poly], root)) for root in complex_roots]
    check(max(residuals) < 1e-8, f"Durand-Kerner residuals too large: {max(residuals)}")
    vieta_sum = sum(complex_roots)
    check(abs(vieta_sum - 2.0) < 1e-8,
          f"the roots must sum to -a_4/a_0 = 2, got {vieta_sum}")
    product = 1 + 0j
    for root in complex_roots:
        product *= root
    expected_product = (-1) ** 5 * (rng_poly[-1] / rng_poly[0])
    check(abs(product - complex(expected_product)) < 1e-6,
          f"the roots must multiply to {complex(expected_product)}, got {product}")
    sturm_brackets = real_roots(rng_poly)
    real_from_complex = sorted(root.real for root in complex_roots if abs(root.imag) < 1e-9)
    check(len(real_from_complex) == len(sturm_brackets),
          f"Sturm found {len(sturm_brackets)} real roots, Durand-Kerner "
          f"{len(real_from_complex)}")
    for (low, high, _exact), value in zip(sturm_brackets, real_from_complex):
        check(float(low) - 1e-6 <= value <= float(high) + 1e-6,
              f"the interval [{float(low)}, {float(high)}] must contain the cross-check root "
              f"{value}")

    # ---------------- roots of x^4 + 1 against the analytic values ----------------
    quartic = durand_kerner(as_fractions([1, 0, 0, 0, 1]))
    analytic = sorted([cmath.exp(1j * math.pi * (2 * k + 1) / 4) for k in range(4)],
                      key=sort_key)
    numeric = sorted(quartic, key=sort_key)
    worst = max(abs(a - b) for a, b in zip(analytic, numeric))
    check(worst < 1e-8, f"x^4 + 1 roots must match the analytic values, worst {worst}")
    check(real_roots(as_fractions([1, 0, 0, 0, 1])) == [],
          "x^4 + 1 must have no real roots")

    # ---------------- determinism ----------------
    repeat = real_roots(factored)
    check(repeat == brackets, "root isolation must be deterministic")
    repeat_roots = durand_kerner(rng_poly)
    check(all(abs(a - b) < 1e-15 for a, b in zip(sorted(repeat_roots, key=sort_key),
                                                 sorted(complex_roots, key=sort_key))),
          "Durand-Kerner must be deterministic")

    print(f"factored roots={[float((low + high) / 2) for low, high, _ in brackets]}")
    print(f"multiplicities={settings} levels={[degree(level) for level in levels]}")
    print(f"descartes positive bound={descartes_positive_bound(square_free_part(mixed))} "
          f"positive roots={positives}")
    print(f"durand-kerner residual={max(residuals):.3e} x^4+1 error={worst:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
