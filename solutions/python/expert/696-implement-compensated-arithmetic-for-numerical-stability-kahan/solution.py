"""Task 696 - Implement compensated arithmetic for numerical stability (Kahan).  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

Floating-point addition is not associative, so a running sum quietly drops low-order bits.  Four
summation strategies are implemented and measured against an exact oracle:

* `naive_sum` — the plain loop, whose error can grow linearly with the count;
* `kahan_sum` — the original compensated sum: the lost low-order part of each addition is recovered
  as `c = (t - total) - y` and subtracted from the next addend, so the error stays second order;
* `neumaier_sum` — Kahan-Babuska: adds the *other* recovery term when the new addend is larger than
  the running total in magnitude, which fixes the case Kahan's formula mishandles;
* `pairwise_sum` — the recursive split-the-list-in-half method, with O(log n) error growth, as a third
  independent way to get accuracy without carrying a compensation term.

The oracle is exact: every float is converted to a `fractions.Fraction` (lossless) and summed in
rational arithmetic before being rounded once, and `math.fsum` (correctly rounded) is used as a second
opinion.  The compensated dot product uses a two-product step built on `math.fma`, which returns the
exact rounding error of a product, so the products contribute no error of their own.

Verified: the classic catastrophe [1e16, 1, ..., -1e16] where naive summation answers 0; the pairing
[1, 1e100, 1, -1e100] where Kahan's own formula fails and the Babuska term does not; exact rational
equality on integer-valued inputs whose true sum is exactly representable; that measured errors stay
inside the textbook bounds (n*u*sum|x| for naive, a small multiple of u*sum|x| for compensated); that
pairwise summation beats naive but not compensated; and that a compensated dot product never loses to
the naive one and matches `math.fsum` on the crafted cases.
"""

import math
from fractions import Fraction

UNIT_ROUNDOFF = 2.0 ** -53                       # u for binary64


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def naive_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def kahan_sum(values):
    """Kahan's compensated summation: recover what each addition lost and re-inject it next time."""
    total, compensation = 0.0, 0.0
    for value in values:
        adjusted = value - compensation
        combined = total + adjusted
        compensation = (combined - total) - adjusted
        total = combined
    return total


def neumaier_sum(values):
    """Kahan-Babuska-Neumaier: also compensates when the addend is the larger operand."""
    total, compensation = 0.0, 0.0
    for value in values:
        combined = total + value
        if abs(total) >= abs(value):
            compensation += (total - combined) + value
        else:
            compensation += (value - combined) + total
        total = combined
    return total + compensation


def pairwise_sum(values):
    """Divide and conquer: the error grows with log2(n) instead of n."""
    if not values:
        return 0.0
    if len(values) <= 8:
        total = 0.0
        for value in values:
            total += value
        return total
    middle = len(values) // 2
    return pairwise_sum(values[:middle]) + pairwise_sum(values[middle:])


def exact_sum(values):
    """Lossless oracle: exact rational addition, rounded once at the very end."""
    total = Fraction(0)
    for value in values:
        total += Fraction(value)
    return float(total)


def two_product(left, right):
    """(product, error) with product + error == left*right exactly, via FMA."""
    product = left * right
    return product, math.fma(left, right, -product)


def naive_dot(left, right):
    total = 0.0
    for a, b in zip(left, right):
        total += a * b
    return total


def compensated_dot(left, right):
    """Compensate both the products (through FMA) and the running sum with the Kahan-Babuska rule,
    which unlike plain Kahan also copes when an addend outgrows the running total."""
    total, compensation = 0.0, 0.0
    for a, b in zip(left, right):
        product, product_error = two_product(a, b)
        combined = total + product
        if abs(total) >= abs(product):
            compensation += ((total - combined) + product) + product_error
        else:
            compensation += ((product - combined) + total) + product_error
        total = combined
    return total + compensation


def relative_error(value, reference):
    if reference == 0.0:
        return abs(value - reference)
    return abs(value - reference) / abs(reference)


def exact_integer_series(count, seed=12345):
    """Integers whose values and running sums are exactly representable in binary64."""
    values, state = [], seed
    for _ in range(count):
        state = (state * 1103515245 + 12345) % 2147483648
        values.append(float(state % 1000))
    return values


def main():
    check(UNIT_ROUNDOFF == math.ulp(1.0) / 2.0, "u is half an ulp at one")
    check(naive_sum([1.0, 2.0, 3.0]) == 6.0, "the naive loop adds small integers exactly")
    check(kahan_sum([1.0, 2.0, 3.0]) == 6.0 and neumaier_sum([1.0, 2.0, 3.0]) == 6.0,
          "the compensated loops agree on easy input")
    check(pairwise_sum([4.0, 5.0, 6.0]) == 15.0, "pairwise summation is exact here too")
    check(exact_sum([1.0, 2.0, 3.0]) == 6.0, "so is the rational oracle")

    catastrophe = [1e16] + [1.0] * 10 + [-1e16]
    check(naive_sum(catastrophe) == 0.0, "naive summation loses all ten units to the big addend")
    check(kahan_sum(catastrophe) == 10.0, "Kahan recovers every one of them")
    check(neumaier_sum(catastrophe) == 10.0, "and so does the Babuska variant")
    check(pairwise_sum(catastrophe) == 4.0,
          "pairwise splitting recovers four of the ten units; compensation is what gets all ten")
    check(exact_sum(catastrophe) == 10.0 and math.fsum(catastrophe) == 10.0,
          "the oracles agree on the true value")

    kahan_weakness = [1.0, 1e100, 1.0, -1e100]
    check(naive_sum(kahan_weakness) == 0.0, "naive summation gives zero")
    check(neumaier_sum(kahan_weakness) == 2.0, "Kahan-Babuska gets the exact answer, two")
    check(exact_sum(kahan_weakness) == 2.0 and math.fsum(kahan_weakness) == 2.0,
          "so do both oracles")
    check(relative_error(neumaier_sum(kahan_weakness), 2.0) <
          relative_error(kahan_sum(kahan_weakness), 2.0),
          "Babuska beats plain Kahan exactly on the case Kahan's formula misses")

    values = exact_integer_series(5000)
    exact = exact_sum(values)
    check(exact == float(sum(Fraction(v) for v in values)), "the oracle is deterministic")
    check(naive_sum(values) == exact, "small integers sum exactly even naively")
    check(kahan_sum(values) == exact and neumaier_sum(values) == exact and
          pairwise_sum(values) == exact, "and every strategy is exact on representable input")

    magnitudes = [1e9]
    ladder = [magnitudes[0] / (2.0 ** index) for index in range(1, 60)]
    ladder = [value for value in ladder if value > 0.0]
    for _ in range(3):
        ladder = ladder + [-value for value in ladder[::-1]]
    ladder.append(1.0)
    truth = exact_sum(ladder)
    naive = naive_sum(ladder)
    kahan = kahan_sum(ladder)
    neumaier = neumaier_sum(ladder)
    pairwise = pairwise_sum(ladder)
    check(truth == 1.0, "the geometric ladder cancels back to one, exactly")
    check(relative_error(kahan, truth) <= relative_error(naive, truth),
          "compensation never loses to the naive loop")
    check(relative_error(neumaier, truth) < 1e-15,
          f"Kahan-Babuska stays within a few ulps: {neumaier!r}")
    check(relative_error(pairwise, truth) <= relative_error(naive, truth),
          "pairwise summation is at least as good as naive on the ladder")
    magnitude_sum = sum(abs(value) for value in ladder)
    check(relative_error(naive, truth) <= len(ladder) * UNIT_ROUNDOFF * magnitude_sum,
          "the naive error obeys the textbook n*u*sum|x| bound")
    check(relative_error(neumaier, truth) <= 4.0 * UNIT_ROUNDOFF * magnitude_sum,
          "the compensated error obeys the much smaller u*sum|x| bound")

    check(naive_sum([1e17, 1.0, -1e17]) == 0.0, "at 1e17 the naive loop still loses the one")
    check(neumaier_sum([1e17, 1.0, -1e17]) == 1.0, "Kahan-Babuska keeps it")
    check(kahan_sum([1e17, 1.0, -1e17]) != 1.0,
          "while Kahan's own recurrence does not, because the addend outgrows the running total")

    left = [1e8, 1.0, -1e8, 3.0, 1e-8, -1e-8, 7.0]
    right = [1.0, 1e8, -1.0, 1e8, 1.0, -1.0, 1.0]
    dot_truth = exact_sum([a * b for a, b in zip(left, right)])
    naive_dot_value = naive_dot(left, right)
    compensated_dot_value = compensated_dot(left, right)
    check(compensated_dot_value == math.fsum([a * b for a, b in zip(left, right)]) or
          relative_error(compensated_dot_value, dot_truth) < 1e-15,
          "the compensated dot product matches the correctly rounded value")
    check(relative_error(compensated_dot_value, dot_truth) <=
          relative_error(naive_dot_value, dot_truth),
          "and never loses to the naive dot product")
    hard_left = [1e16, 1.0, -1e16, 1.0]
    hard_right = [1.0, 1.0, 1.0, 1.0]
    check(naive_dot(hard_left, hard_right) == 1.0,
          "the naive dot product loses one of the two units to the giant term")
    check(compensated_dot(hard_left, hard_right) == 2.0,
          "the compensated dot product keeps both units")
    check(exact_sum([a * b for a, b in zip(hard_left, hard_right)]) == 2.0,
          "and the exact oracle confirms the value two")
    for scale in (1.0, 1e-5, 1e5):
        a = [scale * value for value in left]
        b = list(right)
        check(relative_error(compensated_dot(a, b), exact_sum([x * y for x, y in zip(a, b)])) < 1e-14,
              f"the compensated dot product holds up at scale {scale}")

    check(kahan_sum([]) == 0.0 and pairwise_sum([]) == 0.0 and exact_sum([]) == 0.0,
          "the empty sum is zero everywhere")
    single = [math.pi]
    check(kahan_sum(single) == math.pi and pairwise_sum(single) == math.pi,
          "a single element comes back untouched")
    check(kahan_sum([2.0 ** 60, 2.0 ** -60]) == naive_sum([2.0 ** 60, 2.0 ** -60]),
          "both agree when there is nothing to compensate")
    check(kahan_sum([1.0] * 4) == 4.0, "repeated adds are exact")

    print(f"catastrophe [1e16, 1x10, -1e16]: naive {naive_sum(catastrophe)}, "
          f"Kahan {kahan_sum(catastrophe)}, Babuska {neumaier_sum(catastrophe)}, "
          f"pairwise {pairwise_sum(catastrophe)}, exact 10")
    print(f"ladder of {len(ladder)} terms: naive error {relative_error(naive, truth):.3e}, "
          f"Kahan {relative_error(kahan, truth):.3e}, Babuska {relative_error(neumaier, truth):.3e}")
    print(f"dot product: naive error {relative_error(naive_dot_value, dot_truth):.3e}, "
          f"compensated {relative_error(compensated_dot_value, dot_truth):.3e}; "
          f"the 1e16 pattern gives naive {naive_dot(hard_left, hard_right)} vs "
          f"compensated {compensated_dot(hard_left, hard_right)} (true 2)")
    print("PASS")


if __name__ == "__main__":
    main()
