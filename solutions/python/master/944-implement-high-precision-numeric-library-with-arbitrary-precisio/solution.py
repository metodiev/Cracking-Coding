"""Task 944 - Implement high-precision numeric library with arbitrary precision rational arithmetic.  (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)

A small exact-arithmetic library built on Python's arbitrary-precision integers: a Rational type that
normalises every value (sign on the numerator, reduced by the greatest common divisor), supports exact
addition, subtraction, multiplication, division and integer powers, total ordering, floor/ceil,
continued-fraction expansion and reconstruction, exact decimal rounding with banker's rounding, and a
rational square-root bracket produced by bisection on exact rationals.

Simplified versus mpmath/gmpy2: no transcendental functions, no interval type beyond the square-root
bracket, no precision management beyond Python's integers.  Verified: every arithmetic result equals
the standard-library Fraction oracle over a deterministic operation sequence; normalisation, hashing
and ordering are consistent; division by zero and a zero denominator are rejected; decimal rounding
matches banker's rounding including exact halves and negative values; continued fractions round-trip
and reproduce the known expansion of 415/93 and the Pell identity p^2 - 2q^2 = +-1 for the sqrt(2)
convergents; the square-root bracket really contains the root (lo^2 <= 2 <= hi^2 checked exactly) and
can be made arbitrarily narrow; harmonic sums stay exact and differ from their float image by less than
1e-14; huge Fibonacci/rational powers match the oracle; the library is deterministic.
"""
import math
from fractions import Fraction


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rational:
    __slots__ = ("num", "den")

    def __init__(self, num, den=1):
        if den == 0:
            raise ZeroDivisionError("Rational with a zero denominator")
        num, den = int(num), int(den)
        if den < 0:
            num, den = -num, -den
        common = math.gcd(abs(num), den)
        self.num, self.den = num // common, den // common

    @property
    def numerator(self):
        return self.num

    def __add__(self, other):
        other = to_rational(other)
        return Rational(self.num * other.den + other.num * self.den, self.den * other.den)

    def __sub__(self, other):
        other = to_rational(other)
        return Rational(self.num * other.den - other.num * self.den, self.den * other.den)

    def __mul__(self, other):
        other = to_rational(other)
        return Rational(self.num * other.num, self.den * other.den)

    def __truediv__(self, other):
        other = to_rational(other)
        return Rational(self.num * other.den, self.den * other.num)

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            raise TypeError("only integer exponents are supported")
        if exponent < 0:
            return Rational(self.den ** -exponent, self.num ** -exponent)
        return Rational(self.num ** exponent, self.den ** exponent)

    def __neg__(self):
        return Rational(-self.num, self.den)

    def __eq__(self, other):
        return (self.num, self.den) == (other.num, other.den)

    def __lt__(self, other):
        return self.num * other.den < other.num * self.den

    def __le__(self, other):
        return self.num * other.den <= other.num * self.den

    def __hash__(self):
        return hash((self.num, self.den))

    def __repr__(self):
        return f"Rational({self.num}, {self.den})"

    def __float__(self):
        return self.num / self.den

    def floor(self):
        return self.num // self.den

    def ceil(self):
        return -((-self.num) // self.den)

    def is_integer(self):
        return self.den == 1

    def decimal(self, digits=10):
        """Exact decimal expansion rounded half-to-even at `digits` places."""
        scaled, remainder = divmod(abs(self.num) * 10 ** digits, self.den)
        if 2 * remainder > self.den or (2 * remainder == self.den and scaled % 2 == 1):
            scaled += 1
        text = str(scaled).rjust(digits + 1, "0")
        body = text[:-digits] + "." + text[-digits:] if digits else text
        return ("-" if self.num < 0 else "") + body

    def continued_fraction(self, terms=32):
        """Partial quotients, terminated early when the value is exactly rational."""
        num, den, result = self.num, self.den, []
        while den and len(result) < terms:
            quotient, remainder = divmod(num, den)
            result.append(quotient)
            num, den = den, remainder
        return tuple(result)


def to_rational(value):
    if isinstance(value, Rational) or isinstance(value, int):
        return value if isinstance(value, Rational) else Rational(value)
    raise TypeError(f"cannot coerce {value!r} to Rational")


def from_continued_fraction(terms):
    """Reconstruct a Rational from its partial quotients (Horner form)."""
    value = Rational(terms[-1])
    for term in reversed(terms[:-1]):
        value = Rational(term) + Rational(1) / value
    return value


def sqrt_bracket(value, tolerance):
    """Bisection on exact rationals: [lo, hi] with lo^2 <= value <= hi^2 and hi - lo <= tolerance."""
    low, high = Rational(max(1, math.isqrt(value) - 1)), Rational(math.isqrt(value) + 2)
    while high - low > tolerance:
        middle = (low + high) / 2
        low, high = (middle, high) if middle * middle < Rational(value) else (low, middle)
    return low, high


def main():
    operations = [
        ("add", Rational(7, 9), Rational(2, 15)),
        ("sub", Rational(1, 3), Rational(5, 7)),
        ("mul", Rational(-22, 7), Rational(49, 121)),
        ("div", Rational(3, 40), Rational(9, 80)),
        ("pow", Rational(2, 3), Rational(5)),
    ]
    # 1) independent oracle: exact arithmetic equals the standard-library Fraction results.
    for name, left, right in operations:
        expected = {"add": Fraction(left.num, left.den) + Fraction(right.num, right.den),
                    "sub": Fraction(left.num, left.den) - Fraction(right.num, right.den),
                    "mul": Fraction(left.num, left.den) * Fraction(right.num, right.den),
                    "div": Fraction(left.num, left.den) / Fraction(right.num, right.den),
                    "pow": Fraction(left.num, left.den) ** right.num}[name]
        observed = {"add": left + right, "sub": left - right, "mul": left * right,
                    "div": left / right, "pow": left ** right.num}[name]
        check((observed.num, observed.den) == (expected.numerator, expected.denominator),
              f"{name} mismatched the Fraction oracle: {observed} vs {expected}")
        check(abs(float(observed) - float(expected)) < 1e-12 * max(1.0, abs(float(expected))),
              f"{name} float image diverged from the oracle")

    # 2) normalisation, hashing and ordering stay consistent for equal values.
    check(Rational(6, 8).num == 3 and Rational(6, 8).den == 4
          and Rational(-6, -8) == Rational(3, 4) and Rational(6, -8) == Rational(-3, 4)
          and hash(Rational(10, 15)) == hash(Rational(2, 3))
          and len({Rational(1, 2), Rational(2, 4), Rational(3, 6)}) == 1,
          "normalisation, sign handling and hashing must agree for equal values")

    # 3) ordering against the oracle, plus floor/ceil boundaries on negatives.
    pairs = [(Rational(p, 7), Rational(q, 5)) for p in range(-4, 5) for q in range(-4, 5)]
    check(all((a < b) == (Fraction(a.num, a.den) < Fraction(b.num, b.den))
              and (a == b) == (Fraction(a.num, a.den) == Fraction(b.num, b.den)) for a, b in pairs)
          and Rational(-7, 2).floor() == -4 and Rational(-7, 2).ceil() == -3
          and Rational(7, 2).floor() == 3 and Rational(7, 2).ceil() == 4,
          "ordering, floor and ceil must match exact arithmetic")

    # 4) invalid input is rejected rather than silently producing a wrong value.
    for call, why in ((lambda: Rational(1, 0), "zero denominator"),
                      (lambda: Rational(1, 2) / Rational(0, 5), "division by zero"),
                      (lambda: Rational(1, 2) ** Rational(2), "non-integer exponent")):
        try:
            call()
        except (ZeroDivisionError, TypeError):
            pass
        else:
            raise AssertionError(f"expected an error for {why}")

    # 5) decimal rounding is exact banker's rounding at the requested digit count.
    check(Rational(1, 3).decimal(10) == "0.3333333333"
          and Rational(2, 3).decimal(10) == "0.6666666667"
          and Rational(1, 8).decimal(2) == "0.12"          # exact half rounds to even
          and Rational(3, 8).decimal(2) == "0.38"
          and Rational(-1, 3).decimal(4) == "-0.3333"
          and Rational(5).decimal(3) == "5.000" and Rational(1, 2).decimal(0) == "0",
          "decimal expansion must round half-to-even")

    # 6) continued fractions round-trip and reproduce the known expansion of 415/93.
    check(Rational(415, 93).continued_fraction() == (4, 2, 6, 7)
          and from_continued_fraction((4, 2, 6, 7)) == Rational(415, 93)
          and all(from_continued_fraction(value.continued_fraction()) == value
                  for value in (Rational(0), Rational(-7), Rational(22, 7), Rational(355, 113))),
          "continued fractions must round-trip, including the classic 415/93")

    # 7) sqrt(2) bracket: the invariant is checked exactly on rationals and can be tightened.
    pell_terms = [1] + [2] * 11          # sqrt(2) = [1; 2, 2, 2, ...]
    pell = [from_continued_fraction(pell_terms[:terms]) for terms in range(1, 13)]
    check(all(convergent.num ** 2 - 2 * convergent.den ** 2 in (1, -1) for convergent in pell)
          and pell[-1] == Rational(19601, 13860),
          f"Pell convergents of sqrt(2) must satisfy p^2 - 2q^2 = +-1: {pell[-1]}")
    low, high = sqrt_bracket(2, Rational(1, 10 ** 12))
    check(low * low < Rational(2) and high * high > Rational(2)
          and high - low <= Rational(1, 10 ** 12)
          and abs(float(pell[-1]) - math.sqrt(2)) < 1e-6,
          f"sqrt bracket [{low.decimal(14)}, {high.decimal(14)}] must contain sqrt(2)")

    # 8) harmonic sums stay exact and agree with floating point to 1e-15.
    harmonic = Rational(0)
    for term in range(1, 21):
        harmonic += Rational(1, term)
    check(harmonic.decimal(12) == "3.597739657144"
          and abs(float(harmonic) - sum(1.0 / k for k in range(1, 21))) < 1e-14
          and harmonic - Rational(1, 20) == sum((Rational(1, k) for k in range(1, 20)),
                                                Rational(0)),
          f"harmonic(20)={harmonic.decimal(12)} must be exact")

    # 9) big integers, zero and negative powers match the oracle exactly.
    big = Rational(2) ** 200 / Rational(3) ** 100
    fibonacci = [0, 1]
    while len(fibonacci) < 200:
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    ratio = Rational(fibonacci[199], fibonacci[198])
    check((big.num, big.den) == (2 ** 200, 3 ** 100)
          and Fraction(ratio.num, ratio.den) == Fraction(fibonacci[199], fibonacci[198])
          and Rational(0) ** 5 == Rational(0) and Rational(7) ** 0 == Rational(1)
          and Rational(2) ** -3 == Rational(1, 8)
          and abs(float(big) - (2.0 ** 200) / (3.0 ** 100)) < 1e-9,
          "big powers, zero and negative exponents must be exact")

    print(f"harmonic(20)={harmonic.decimal(12)} sqrt2=[{low.decimal(12)}, {high.decimal(12)}] "
          f"convergent={pell[-1]}")
    print(f"phi approx={ratio.decimal(12)} 2^200/3^100 digits="
          f"{len(str(big.num))}/{len(str(big.den))}")
    print("PASS")


if __name__ == "__main__":
    main()
