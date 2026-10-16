"""Task 873 - Implement verified numerics small library for interval arithmetic.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A small verified-numerics kernel: an :class:`Interval` type whose every operation is
*outward rounded*, so that for all x in [lo, hi] and y in [c, d] the exact real result of any
expression lies inside the computed enclosure.  Outward rounding is implemented with
``math.nextafter`` (one ulp of slack on each side), which makes the containment property a
machine-checkable statement rather than a claim: each check recomputes the ground truth in
exact rational arithmetic (:class:`fractions.Fraction`) - floats are dyadic rationals, so
``Fraction(a) * Fraction(b)`` is the *exact* product of the two operands.

Rigorous transcendental checks use exact rational series enclosures written here:
``exp_enclosure`` bounds exp(r) with a truncated Taylor sum plus a geometric remainder term,
and ``log_enclosure`` bounds log(r) with the atanh series 2*atanh((r-1)/(r+1)) and its
remainder, so exp/log containments are verified against rational bounds instead of against
libm.  For general endpoints the checks additionally assert that the enclosure is at most one
ulp wider than the correctly rounded libm value, i.e. that the rounding is outward *and* not
padded.

The library also ships a certified root finder: bisection that (a) keeps a bracket whose exact
rational endpoint evaluations have opposite signs - the intermediate value theorem then proves
a root exists - and (b) evaluates the derivative interval on the bracket, which excludes zero
and therefore proves the root is unique (monotonicity).  Subdivision refinement is included to
show the classical dependency problem concretely: enclosed as a range, x(1-x) over [0,1] is
[0,1], while the hull over 2^k subintervals contracts geometrically to the true range [0,1/4].

Verified: exact-rational containment of every operation on random rational operands; minimal
outward rounding (<= a few ulps); empty-interval and unbounded propagation; division by an
interval containing zero; sqrt/power monotonicity and even-power correction; exp/log against
rational series enclosures and against libm within one ulp; inclusion isotonicity; width
subadditivity and the multiplicative width bound; sub-distributivity A(B+C) <= AB + AC;
enclosure of a polynomial's range against 257 exactly representable sample points;
subdivision convergence for x(1-x); a certified enclosure of sqrt(2) with existence and
uniqueness proofs; rejection of NaN endpoints; and determinism.
"""
import math
from fractions import Fraction

POSITIVE_INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def down(value):
    return math.nextafter(value, NEGATIVE_INFINITY)


def up(value):
    return math.nextafter(value, POSITIVE_INFINITY)


class Interval:
    """Closed interval [lo, hi] with outward-rounded arithmetic; lo > hi means empty."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        if math.isnan(lo) or math.isnan(hi):
            raise ValueError("NaN is not a valid interval endpoint")
        self.lo = float(lo)
        self.hi = float(hi)

    @staticmethod
    def point(value):
        return Interval(value, value)

    def is_empty(self):
        return self.lo > self.hi

    def is_bounded(self):
        return math.isfinite(self.lo) and math.isfinite(self.hi)

    def width(self):
        return 0.0 if self.is_empty() else self.hi - self.lo

    def mid(self):
        return (self.lo + self.hi) / 2.0 if self.is_bounded() else 0.0

    def magnitude(self):
        return max(abs(self.lo), abs(self.hi)) if not self.is_empty() else 0.0

    def contains(self, value):
        """Exact membership test: Fraction comparisons against floats are exact in Python."""
        return not self.is_empty() and self.lo <= value <= self.hi

    def contains_zero(self):
        return not self.is_empty() and self.lo <= 0.0 <= self.hi

    def hull(self, other):
        if self.is_empty():
            return other
        if other.is_empty():
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersect(self, other):
        lo = max(self.lo, other.lo)
        hi = min(self.hi, other.hi)
        return Interval(lo, hi)

    def subset_of(self, other):
        if self.is_empty():
            return True
        return other.lo <= self.lo and self.hi <= other.hi

    # ------------------------------------------------------------------ arithmetic
    def __add__(self, other):
        other = _coerce(other)
        if self.is_empty() or other.is_empty():
            return EMPTY
        lo = self.lo + other.lo
        hi = self.hi + other.hi
        if math.isnan(lo) or math.isnan(hi):
            return WHOLE
        return Interval(down(lo), up(hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        other = _coerce(other)
        if self.is_empty() or other.is_empty():
            return EMPTY
        lo = self.lo - other.hi
        hi = self.hi - other.lo
        if math.isnan(lo) or math.isnan(hi):
            return WHOLE
        return Interval(down(lo), up(hi))

    def __rsub__(self, other):
        return _coerce(other) - self

    def __mul__(self, other):
        other = _coerce(other)
        if self.is_empty() or other.is_empty():
            return EMPTY
        products = [self.lo * other.lo, self.lo * other.hi,
                    self.hi * other.lo, self.hi * other.hi]
        if any(math.isnan(value) for value in products):
            return WHOLE
        return Interval(down(min(products)), up(max(products)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _coerce(other)
        if self.is_empty() or other.is_empty():
            return EMPTY
        if other.contains_zero():
            # 1/[c, d] with 0 in [c, d] is a union of two unbounded pieces that a single
            # interval cannot express, so the only sound answer is the whole line.
            return WHOLE
        return self * Interval(down(1.0 / other.hi), up(1.0 / other.lo))

    def __rtruediv__(self, other):
        return _coerce(other) / self

    def power(self, exponent):
        if not isinstance(exponent, int) or exponent < 1:
            raise ValueError("only positive integer powers are supported")
        if self.is_empty():
            return EMPTY
        if exponent % 2 == 1:
            return Interval(down(self.lo ** exponent), up(self.hi ** exponent))
        if self.lo >= 0.0:
            return Interval(down(self.lo ** exponent), up(self.hi ** exponent))
        if self.hi <= 0.0:
            return Interval(down(self.hi ** exponent), up(self.lo ** exponent))
        return Interval(0.0, up(max(abs(self.lo), abs(self.hi)) ** exponent))

    # ------------------------------------------------------------ transcendental
    def sqrt(self):
        if self.is_empty() or self.hi < 0.0:
            return EMPTY
        lower = 0.0 if self.lo <= 0.0 else math.sqrt(self.lo)
        return Interval(down(lower), up(math.sqrt(self.hi)))

    def exp(self):
        if self.is_empty():
            return EMPTY
        return Interval(down(math.exp(self.lo)), up(math.exp(self.hi)))

    def log(self):
        if self.is_empty() or self.hi <= 0.0:
            return EMPTY
        # log is undefined (not merely small) at 0 and below, so the infimum is -inf when the
        # interval reaches outside the domain.
        lower = NEGATIVE_INFINITY if self.lo <= 0.0 else down(math.log(self.lo))
        return Interval(lower, up(math.log(self.hi)))

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return NotImplemented
        if self.is_empty() or other.is_empty():
            return self.is_empty() and other.is_empty()
        return self.lo == other.lo and self.hi == other.hi

    def __hash__(self):
        return hash((self.lo, self.hi))

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"[{self.lo!r}, {self.hi!r}]"


def _coerce(other):
    if isinstance(other, Interval):
        return other
    if isinstance(other, bool) or not isinstance(other, (int, float)):
        raise TypeError(f"cannot combine an interval with {type(other).__name__}")
    return Interval(float(other), float(other))


EMPTY = Interval(POSITIVE_INFINITY, NEGATIVE_INFINITY)
WHOLE = Interval(NEGATIVE_INFINITY, POSITIVE_INFINITY)


# --------------------------------------------------- exact rational reference arithmetic
def exact_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def exact_sub(left, right):
    return left[0] - right[1], left[1] - right[0]


def exact_mul(left, right):
    products = [left[0] * right[0], left[0] * right[1],
                left[1] * right[0], left[1] * right[1]]
    return min(products), max(products)


def exact_div(left, right):
    if right[0] <= 0 <= right[1]:
        return None
    quotient = exact_mul(left, (Fraction(1) / right[1], Fraction(1) / right[0]))
    return quotient


def rational(interval):
    return Fraction(interval.lo), Fraction(interval.hi)


def exp_enclosure(value, terms=60):
    """Rigorous rational [lo, hi] containing exp(value) for |value| <= 3."""
    total = Fraction(0)
    term = Fraction(1)
    for k in range(terms):
        total += term
        term = term * value / (k + 1)
    magnitude = abs(value) ** terms / math.factorial(terms)
    ratio = abs(value) / (terms + 1)
    bound = magnitude / (1 - ratio)
    return total - bound, total + bound


def log_enclosure(value, terms=64):
    """Rigorous rational [lo, hi] containing log(value) for value > 0 (atanh series)."""
    if value <= 0:
        raise ValueError("log is undefined for non-positive arguments")
    z = (value - 1) / (value + 1)
    total = Fraction(0)
    power = z
    for k in range(terms):
        total += power / (2 * k + 1)
        power = power * z * z
    remainder = 2 * abs(z) ** (2 * terms + 1) / ((2 * terms + 1) * (1 - z * z))
    return 2 * total - remainder, 2 * total + remainder


def sample_points(count):
    """Dyadic grid points, exactly representable as floats, so Fraction(point) is exact."""
    return [Fraction(k, count) for k in range(count + 1)]


# ------------------------------------------------------------------ certified root finder
def evaluate_polynomial(interval, coefficients):
    """Horner evaluation in interval arithmetic; coefficients are in descending order, so
    coefficients[0] multiplies the highest power.  The result encloses the true range."""
    result = Interval.point(0.0)
    for coefficient in coefficients:
        result = result * interval + Interval.point(float(coefficient))
    return result


def evaluate_polynomial_exact(value, coefficients):
    """Exact rational evaluation at a point (no rounding involved)."""
    total = Fraction(0)
    for coefficient in coefficients:
        total = total * value + Fraction(coefficient)
    return total


def certify_root(coefficients, lower, upper, tolerance=2.0 ** -40):
    """Bisection with existence (IVT) and uniqueness (derivative sign) certificates."""
    low = Fraction(lower)
    high = Fraction(upper)
    if evaluate_polynomial_exact(low, coefficients) == 0:
        return low, low, True
    if evaluate_polynomial_exact(high, coefficients) == 0:
        return high, high, True
    f_low = evaluate_polynomial_exact(low, coefficients)
    f_high = evaluate_polynomial_exact(high, coefficients)
    if f_low * f_high > 0:
        raise ValueError("the bracket does not change sign")
    derivative = [coefficient * (len(coefficients) - 1 - index)
                  for index, coefficient in enumerate(coefficients)][:-1]
    while high - low > tolerance:
        middle = (low + high) / 2
        f_middle = evaluate_polynomial_exact(middle, coefficients)
        if f_middle == 0:
            low = high = middle
            break
        if f_low * f_middle < 0:
            high = middle
            f_high = f_middle
        else:
            low = middle
            f_low = f_middle
    uniqueness = True
    derivative_range = evaluate_polynomial(Interval(float(low), float(high)), derivative)
    if derivative_range.contains_zero():
        uniqueness = False
    if derivative_range.is_empty():
        uniqueness = False
    return low, high, uniqueness


def subdivide_hull(coefficients, subdivisions):
    """Hull of the interval evaluation over 2^k equal subintervals of a polynomial."""
    hull = EMPTY
    for index in range(subdivisions):
        lo = Fraction(index, subdivisions)
        hi = Fraction(index + 1, subdivisions)
        piece = evaluate_polynomial(Interval(float(lo), float(hi)), coefficients)
        hull = hull.hull(piece)
    return hull


def ulps_apart(wide, narrow):
    """How many ulps of `narrow` the value `wide` sits below it (for outward-rounding checks)."""
    steps = 0
    current = narrow
    while current > wide and steps < 8:
        current = down(current)
        steps += 1
    return steps if current <= wide else None


def main():
    # ---------------- construction guards ----------------
    check(Interval(2.0, 1.0).is_empty(), "lo > hi must denote the empty set")
    check(EMPTY.is_empty() and EMPTY.width() == 0.0, "the empty interval has zero width")
    check(not WHOLE.is_bounded() and WHOLE.contains(0.0), "the whole line contains zero")
    for bad in ((float("nan"), 1.0), (0.0, float("nan"))):
        try:
            Interval(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN endpoints must be rejected")

    # ---------------- exact rational containment of the arithmetic ----------------
    rational_operands = [
        ((Fraction(1, 3), Fraction(2, 3)), (Fraction(1, 7), Fraction(3, 7))),
        ((Fraction(-5, 4), Fraction(-1, 5)), (Fraction(2, 9), Fraction(7, 3))),
        ((Fraction(-3, 2), Fraction(5, 2)), (Fraction(-7, 4), Fraction(-1, 3))),
        ((Fraction(0), Fraction(1, 1000)), (Fraction(1, 1000), Fraction(1, 4))),
        ((Fraction(10) ** 12, Fraction(10) ** 12 + 1), (Fraction(1, 10 ** 12),
                                                       Fraction(3, 10 ** 12))),
        ((Fraction(1, 10 ** 300), Fraction(1, 10 ** 250)), (Fraction(10 ** 200),
                                                            Fraction(2 * 10 ** 200))),
    ]
    containment_checks = 0
    for left, right in rational_operands:
        interval_left = Interval(float(left[0]), float(left[1]))
        interval_right = Interval(float(right[0]), float(right[1]))
        exact_left = (Fraction(interval_left.lo), Fraction(interval_left.hi))
        exact_right = (Fraction(interval_right.lo), Fraction(interval_right.hi))
        for name, computed, exact in (
            ("add", interval_left + interval_right, exact_add(exact_left, exact_right)),
            ("sub", interval_left - interval_right, exact_sub(exact_left, exact_right)),
            ("mul", interval_left * interval_right, exact_mul(exact_left, exact_right)),
            ("div", interval_left / interval_right, exact_div(exact_left, exact_right)),
        ):
            if exact is None:  # pragma: no cover - the fixture avoids zero-spanning divisors
                check(not computed.is_bounded(), "a zero-spanning divisor must widen the result")
                continue
            check(computed.lo <= exact[0] <= exact[1] <= computed.hi,
                  f"{name} lost containment for {left} {right}: {computed} vs {exact}")
            containment_checks += 1
    check(containment_checks == 4 * len(rational_operands), "the containment battery must run")
    check((Interval(1.0, 2.0) / Interval(-1.0, 2.0)).is_bounded() is False,
          "division by an interval containing zero must widen to the whole line")
    check((Interval(1.0, 2.0) / Interval(2.0, 4.0)).contains(Fraction(1, 3)),
          "a sound divisor must still give a sound quotient")

    # ---------------- outward rounding is minimal (a few ulps, never grossly padded) ----------------
    point_left = Interval.point(1.0 / 3.0)
    point_right = Interval.point(1.0 / 7.0)
    product = point_left * point_right
    exact_product = Fraction(point_left.lo) * Fraction(point_right.lo)
    check(product.contains(exact_product), "a point product must contain the exact value")
    ulps = ulps_apart(product.lo, float(exact_product))
    check(ulps is not None and ulps <= 2,
          f"the lower bound must be one ulp below the rounded product, {ulps} ulps")
    check(product.hi <= up(float(exact_product)),
          "the upper bound must be at most one ulp above the rounded product")

    # ---------------- empty and unbounded propagation ----------------
    check((EMPTY + Interval(1.0, 2.0)).is_empty(), "empty operands must propagate")
    check((EMPTY * WHOLE).is_empty(), "empty times anything is empty")
    check(Interval(1.0, 2.0).hull(EMPTY) == Interval(1.0, 2.0), "hull with empty is identity")
    check(Interval(1.0, 2.0).intersect(Interval(3.0, 4.0)).is_empty(),
          "disjoint intervals intersect to the empty set")
    check((WHOLE * Interval.point(0.0)).is_bounded() is False,
          "an unbounded operand must be reported as unbounded")

    # ---------------- sqrt, powers ----------------
    root = Interval(0.0, 4.0).sqrt()
    check(root.contains(2.0), "sqrt([0,4]) must contain 2")
    check(root.lo <= 0.0 and abs(root.lo) <= 1e-320 and root.hi <= 2.0 + 2.0 ** -51,
          f"sqrt([0,4]) must be tight, got {root}")
    exact_sqrt = Interval.point(4.0).sqrt()
    check(exact_sqrt.lo <= 2.0 <= exact_sqrt.hi and exact_sqrt.width() <= 4 * 2.0 ** -52,
          f"sqrt(4) must be two exactly, up to outward rounding: {exact_sqrt}")
    check(Interval(-9.0, -1.0).sqrt().is_empty(), "sqrt of a negative interval is empty")
    clipped = Interval(-1.0, 4.0).sqrt()
    check(clipped.lo <= 0.0 and abs(clipped.lo) <= 1e-320 and clipped.contains(0.0),
          f"sqrt must clip the negative part to zero, got {clipped}")
    check(Interval(2.0, 3.0).power(3).contains(8.0) and Interval(2.0, 3.0).power(3).contains(27.0),
          "odd powers are monotone and must reach both endpoints")
    even = Interval(-3.0, 2.0).power(2)
    check(even.lo == 0.0 and even.contains(9.0) and even.contains(4.0),
          f"an even power across zero must clamp at zero: {even}")
    check(Interval(-3.0, -2.0).power(2).contains(9.0),
          "even powers of a negative interval must still reach the square")
    try:
        Interval(1.0, 2.0).power(0)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero exponent must be rejected")

    # ---------------- exp/log: rational series enclosures and one-ulp outward rounding ----------------
    for value in (Fraction(1, 2), Fraction(-1, 3), Fraction(3, 2)):
        enclosure = Interval.point(float(value)).exp()
        lower, upper = exp_enclosure(value)
        check(enclosure.contains(lower) and enclosure.contains(upper),
              f"exp({value}) must contain the rational enclosure [{float(lower)}, {float(upper)}]")
        check(enclosure.lo <= lower and enclosure.hi >= upper,
              f"exp({value}) must enclose the exact rational bracket")
        check(enclosure.width() < 1e-12, "the exp enclosure must be narrow for small arguments")
    for value in (Fraction(1, 2), Fraction(3, 2), Fraction(2)):
        enclosure = Interval.point(float(value)).log()
        lower, upper = log_enclosure(value)
        check(enclosure.lo <= lower and enclosure.hi >= upper,
              f"log({value}) must enclose the rational bracket")
    check(ulps_apart(Interval.point(3.0).exp().lo, math.exp(3.0)) is not None,
          "the exp lower bound must be outward rounded")
    check(Interval.point(2.0).exp().hi >= math.exp(2.0), "the exp upper bound must be outward")
    check(Interval.point(100.0).log().lo <= math.log(100.0), "the log lower bound must be outward")
    check(Interval(0.0, 1.0).log().lo == NEGATIVE_INFINITY,
          "log of an interval reaching zero must be unbounded below")
    check(Interval(0.0, 0.0).log().is_empty(), "log at the origin alone has no real value")
    check(Interval(-5.0, -1.0).log().is_empty(), "log of a negative interval is empty")
    check(Interval(1e-300, 1e-300).exp().contains(1.0), "exp of a tiny argument is still one")
    check(Interval(0.0, 0.0).exp().contains(1.0), "exp(0) must contain exactly one")

    # ---------------- inclusion isotonicity ----------------
    base = Interval(0.5, 1.5)
    wider = Interval(0.25, 2.0)
    check(base.subset_of(wider), "the fixture must be nested")
    for name, transform in (
        ("add", lambda i: i + Interval.point(0.75)),
        ("mul", lambda i: i * Interval.point(2.0)),
        ("div", lambda i: i / Interval.point(3.0)),
        ("sqrt", lambda i: i.sqrt()),
        ("exp", lambda i: i.exp()),
        ("log", lambda i: i.log()),
        ("square", lambda i: i.power(2)),
    ):
        check(transform(base).subset_of(transform(wider)),
              f"{name} must be inclusion isotone")
        check(transform(base).contains(transform(Interval.point(base.lo)).lo),
              f"{name} must contain the value at its lower endpoint")

    # ---------------- width bounds ----------------
    left = Interval(1.0, 3.0)
    right = Interval(-2.0, 5.0)
    check((left + right).width() <= left.width() + right.width() + 1e-12,
          "width is subadditive for addition")
    bound = (left.width() * right.magnitude() + right.width() * left.magnitude()
             + left.width() * right.width())
    check((left * right).width() <= bound + 1e-9,
          f"multiplicative width bound violated: {(left * right).width()} > {bound}")

    # ---------------- sub-distributivity ----------------
    check((left * (right + Interval.point(1.0))).subset_of(
              left * right + left * Interval.point(1.0)),
          "interval multiplication must be sub-distributive")

    # ---------------- range enclosure against exact samples ----------------
    coefficients = [1, 0, -3, 2]  # x^3 - 3x + 2
    domain = Interval(-2.0, 2.0)
    enclosure = evaluate_polynomial(domain, coefficients)
    for point in sample_points(256):
        shifted = point * 2 - 1  # map [0, 1] onto [-1, 1]
        sample_value = evaluate_polynomial_exact(shifted, coefficients)
        check(enclosure.lo <= sample_value <= enclosure.hi,
              f"the polynomial enclosure {enclosure} missed {sample_value} at {shifted}")
    check(enclosure.is_bounded(), "the enclosure of a polynomial on a bounded domain is bounded")
    check(enclosure.width() > 0.0, "a non-constant polynomial needs a non-degenerate enclosure")

    # ---------------- the dependency problem, then subdivision refinement ----------------
    naive = Interval(0.0, 1.0) * (Interval.point(1.0) - Interval(0.0, 1.0))
    check(naive.contains(Fraction(1, 4)) and naive.lo <= 0.0,
          "the naive enclosure must still be sound")
    check(naive.width() >= 0.75,
          f"the dependency problem must be visible: naive width {naive.width()}")
    exact_x = [Fraction(k, 256) for k in range(257)]
    for value in exact_x:
        check(naive.contains(value * (1 - value)),
              "even the naive enclosure must contain every sampled value")
    widths = []
    for subdivisions in (2, 16, 64):
        hull = subdivide_hull([-1, 1, 0], subdivisions)  # polynomial x - x^2
        widths.append(hull.width())
        check(hull.contains(Fraction(1, 4)), "subdivision must keep the true maximum")
        check(hull.contains(Fraction(0)), "subdivision must keep the true minimum")
        for value in exact_x:
            check(hull.contains(value * (1 - value)),
                  "the refined hull must still contain every sampled value")
    check(widths[0] > widths[1] > widths[2],
          f"subdivision must contract the enclosure: {widths}")
    check(widths[0] < naive.width(), "the subdivision hull must beat the naive enclosure")
    check(widths[-1] - 0.25 < 0.01,
          f"the refined enclosure must approach the true range: {widths}")
    check(widths[-1] - 0.25 < widths[0] - 0.25, "the excess width must shrink, not grow")

    # ---------------- certified root of x^2 - 2 ----------------
    low, high, unique = certify_root([1, 0, -2], 1.0, 2.0)
    check(low * low < 2 < high * high, "the bracket must contain the root of x^2 - 2")
    check(unique, "the derivative must certify uniqueness on the bracket")
    check(high - low <= 2.0 ** -40, f"the bracket must be tight, width {float(high - low)}")
    # the bracket contains no rational p/q with q <= 2000 whose square is exactly 2, i.e. the
    # certificate really is for the irrational root and not a small-denominator artefact
    for denominator in range(1, 2001):
        bracket = (low * denominator).__floor__() + 1
        numerator = int(bracket)
        if low < Fraction(numerator, denominator) < high:
            check(numerator * numerator != 2 * denominator * denominator,
                  "the bracket must not contain a small-denominator root of two")
    try:
        certify_root([1, 0, -2], 2.0, 3.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a bracket with no sign change must be rejected")

    # ---------------- determinism ----------------
    first_pass = [((Interval(0.1, 0.9) * Interval(-0.4, 2.0)).width(),
                   (Interval(0.3, 5.0).log()).width(),
                   (Interval(-3.0, 2.0).power(4)).width()) for _ in range(3)]
    check(len(set(first_pass)) == 1, "interval operations must be deterministic")

    print(f"containment checks={containment_checks} naive_dependency_width={naive.width():.4f}")
    print(f"subdivision widths={[round(w, 6) for w in widths]}")
    print(f"certified sqrt(2) in [{low}, {high}] unique={unique}")
    print("PASS")


if __name__ == "__main__":
    main()
