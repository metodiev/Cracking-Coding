"""Task 531 - Implement weighted median selection algorithm. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)

Weighted median of values v_i with positive weights w_i: the smallest value x in the multiset
whose cumulative weight reaches half of the total weight W, i.e. the smallest x with
weight(v <= x) >= W/2 (the lower weighted median, the convention that reduces to the ordinary
lower median when all weights are equal). Two independent algorithms are implemented and must
agree: an O(n log n) sort-and-scan over prefix sums, and a selection-based O(n) expected-time
routine that locates the value through a single pass of weighted quickselect around a
deterministically chosen pivot (median of the first, middle and last entries - no randomness), so
the answer is obtained without sorting. Documented checks: hand-computed examples (equal weights
give the ordinary lower median; [1,2,3] with weights [0.1,0.5,0.4] gives 2; [1,2,3] with weights
[1,2,7] gives 3; a single element; a dominant weight w_i > W/2 forces that value); the exact
postcondition weight(v < x) < W/2 <= weight(v <= x); agreement of the two algorithms on LCG
instances with integer and with float weights, including duplicate values and single-value
inputs; the replication identity - expanding value v_i into w_i copies and taking the element at
index W//2 for odd W and W//2 - 1 for even W reproduces the weighted median exactly; the minimiser
property that x minimises the weighted absolute deviation sum w_i * |v_i - x| over all candidate
values; shift invariance (adding c to every value shifts the median by c) and scale invariance of
the weights; monotonicity of the answer as one weight grows; and the error cases of an empty
input, a non-positive total weight and negative weights.
"""

from fractions import Fraction


def weighted_median_sorted(values, weights):
    """O(n log n): sort by value, scan prefix weights until half of the total is reached."""
    total = Fraction(0)
    for weight in weights:
        if weight < 0:
            raise ValueError("weights must be non-negative")
        total += Fraction(weight)
    if total <= 0:
        raise ValueError("the total weight must be positive")
    half = total / 2
    running = Fraction(0)
    for value, weight in sorted(zip(values, weights)):
        running += Fraction(weight)
        if running >= half:
            return value
    raise AssertionError("unreachable: the total weight was positive")


def weighted_median_select(values, weights):
    """Expected O(n) weighted quickselect with a deterministic median-of-three pivot."""
    if not values:
        raise ValueError("cannot select from an empty sequence")
    for weight in weights:
        if weight < 0:
            raise ValueError("weights must be non-negative")
    total = Fraction(0)
    for weight in weights:
        total += Fraction(weight)
    if total <= 0:
        raise ValueError("the total weight must be positive")
    half = total / 2
    items = list(zip(values, weights))
    while True:
        if len(items) == 1:
            return items[0][0]
        pivot = median_of_three(items)
        below = Fraction(0)
        equal = Fraction(0)
        lower = []
        upper = []
        for value, weight in items:
            if value < pivot:
                below += Fraction(weight)
                lower.append((value, weight))
            elif value > pivot:
                upper.append((value, weight))
            else:
                equal += Fraction(weight)
        if below >= half:
            items = lower
        elif below + equal >= half:
            return pivot
        else:
            half -= below + equal
            items = upper


def median_of_three(items):
    """Deterministic pivot: the median value of the first, middle and last entries."""
    picks = [items[0][0], items[len(items) // 2][0], items[-1][0]]
    return sorted(picks)[1]


def replicate_and_pick(values, weights):
    """Independent reduction: expand each value into w_i copies and take the documented index."""
    expanded = []
    for value, weight in zip(values, weights):
        expanded.extend([value] * int(weight))
    expanded.sort()
    wanted = len(expanded) // 2 if len(expanded) % 2 == 1 else len(expanded) // 2 - 1
    return expanded[wanted]


def deviation_cost(values, weights, x):
    return sum(abs(v - x) * w for v, w in zip(values, weights))


def lcg_pairs(count, seed, value_limit=50, weight_limit=20):
    values = []
    weights = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % value_limit)
        seed = (seed * 75 + 74) % 65537
        weights.append(1 + seed % weight_limit)
    return values, weights


def main():
    # Hand-computed examples.
    assert weighted_median_sorted([4, 1, 3, 2], [1, 1, 1, 1]) == 2, "ordinary lower median"
    assert weighted_median_select([4, 1, 3, 2], [1, 1, 1, 1]) == 2
    assert replicate_and_pick([4, 1, 3, 2], [1, 1, 1, 1]) == 2
    assert weighted_median_sorted([1, 2, 3], [0.1, 0.5, 0.4]) == 2
    assert weighted_median_select([1, 2, 3], [0.1, 0.5, 0.4]) == 2
    assert weighted_median_sorted([1, 2, 3], [1, 2, 7]) == 3
    assert weighted_median_select([1, 2, 3], [1, 2, 7]) == 3
    assert weighted_median_sorted([7], [3]) == 7
    assert weighted_median_select([7], [3]) == 7
    # A dominant weight beats the whole rest of the distribution.
    assert weighted_median_sorted([10, 20], [6, 5]) == 10, "6 > 11/2 forces value 10"
    assert weighted_median_sorted([10, 20], [5, 6]) == 20
    assert weighted_median_select([10, 20], [6, 5]) == 10

    # Postcondition: weight(v < x) < W/2 <= weight(v <= x), exactly, with Fractions.
    for values, weights in (([1, 2, 3], [1, 2, 7]),
                            ([4, 1, 3, 2], [1, 1, 1, 1]),
                            ([5, 5, 2], [1, 3, 4]),
                            ([0, 10], [1, 1]),
                            ([0, 10], [1, 2])):
        x = weighted_median_sorted(values, weights)
        total = Fraction(sum(weights))
        below = Fraction(sum(w for v, w in zip(values, weights) if v < x))
        upto = Fraction(sum(w for v, w in zip(values, weights) if v <= x))
        assert below < total / 2 <= upto, (values, weights, x)

    # LCG instances: the two algorithms agree, the replication identity holds for integer
    # weights, and the median is the minimiser of the weighted absolute deviation.
    for count, seed, value_limit, weight_limit in ((1, 3, 20, 5), (2, 11, 20, 5), (7, 23, 10, 9),
                                                   (25, 37, 50, 20), (60, 53, 5, 4),
                                                   (120, 71, 1000, 3)):
        values, weights = lcg_pairs(count, seed, value_limit, weight_limit)
        sorted_answer = weighted_median_sorted(values, weights)
        select_answer = weighted_median_select(values, weights)
        assert sorted_answer == select_answer, (count, seed, sorted_answer, select_answer)
        assert replicate_and_pick(values, weights) == sorted_answer, (count, seed)
        assert sorted_answer in values
        assert min(values) <= sorted_answer <= max(values)
        # Minimal weighted absolute deviation among the candidate values.
        best_cost = deviation_cost(values, weights, sorted_answer)
        for candidate in values:
            assert best_cost <= deviation_cost(values, weights, candidate) + 1e-9, \
                (count, seed, candidate)
        # Shift invariance and weight-scale invariance.
        shifted = [v + 100 for v in values]
        assert weighted_median_sorted(shifted, weights) == sorted_answer + 100
        assert weighted_median_select(shifted, weights) == sorted_answer + 100
        scaled = [w * 4 for w in weights]
        assert weighted_median_sorted(values, scaled) == sorted_answer
        assert weighted_median_select(values, scaled) == sorted_answer
        # Duplicated entries (same value, two weights) behave like the summed weight.
        doubled_values = values + values
        doubled_weights = weights + weights
        assert weighted_median_sorted(doubled_values, doubled_weights) == sorted_answer
        assert weighted_median_select(doubled_values, doubled_weights) == sorted_answer

    # Float weights: the sort-based and the selection-based algorithms still agree.
    float_weights = [0.5, 1.25, 3.0, 0.75, 2.0, 0.125]
    float_values = [9, 3, 7, 1, 5, 11]
    assert weighted_median_sorted(float_values, float_weights) == \
        weighted_median_select(float_values, float_weights)
    assert weighted_median_sorted(float_values, float_weights) == \
        replicate_and_pick(float_values, [w * 8 for w in float_weights])

    # Monotonicity: growing one weight can only move the answer up (to that value or higher).
    values = [1, 2, 3, 4, 5]
    previous = weighted_median_sorted(values, [1] * 5)
    seen = []
    for boost in range(1, 40):
        answer = weighted_median_sorted(values, [boost if v == 4 else 1 for v in values])
        seen.append(answer)
        assert answer >= previous, (boost, answer, previous)
        assert answer in values
        previous = answer
    assert seen.count(3) >= 1
    assert seen[0] == 3, "equal weights give the ordinary lower median of 1..5"
    assert seen[-1] == 4 and 4 in seen, ("a heavy weight on 4 wins", seen)
    # Growing the *largest* weight of a two-point distribution flips the median.
    assert weighted_median_sorted([0, 10], [3, 2]) == 0
    assert weighted_median_sorted([0, 10], [2, 3]) == 10

    # All weight on one value, and equal weights on duplicated values.
    assert weighted_median_sorted([1, 2, 3], [0, 5, 0]) == 2
    assert weighted_median_select([1, 2, 3], [0, 5, 0]) == 2
    assert weighted_median_sorted([2, 2, 2], [1, 1, 1]) == 2

    # Errors.
    def expect_value_error(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    expect_value_error(lambda: weighted_median_sorted([], []))
    expect_value_error(lambda: weighted_median_sorted([1, 2], [0, 0]))
    expect_value_error(lambda: weighted_median_sorted([1, 2], [1, -1]))
    expect_value_error(lambda: weighted_median_select([], []))
    expect_value_error(lambda: weighted_median_select([1, 2], [0, 0]))
    expect_value_error(lambda: weighted_median_select([1, 2], [-1, 3]))

    # Determinism: the pivot choice does not depend on randomness or input order.
    values, weights = lcg_pairs(40, 97)
    first = weighted_median_select(values, weights)
    assert all(weighted_median_select(values, weights) == first for _ in range(5))
    assert weighted_median_select(list(reversed(values)), list(reversed(weights))) == first


if __name__ == "__main__":
    main()
    print("PASS")
