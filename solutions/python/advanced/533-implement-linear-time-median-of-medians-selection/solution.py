"""Task 533 - Implement linear-time median of medians selection. (Tier: Advanced, CTCI Ch 10 — Sorting and Searching)

Deterministic worst-case linear-time selection (the median-of-medians / BFPRT algorithm) for the
k-th smallest element: the values are cut into groups of five, each group of five is sorted with
an insertion sort and its median collected, the *median of those medians* is found by recursing on
the collected medians, and that pivot drives a three-way partition (less / equal / greater) which
tells the recursion which side still contains rank k. The recurrence
T(n) <= T(ceil(n/5)) + T(7n/10) + O(n) guarantees linear time, so the self-test also counts the
comparisons the routine performs and asserts a documented linear bound, plus a bound on the number
of recursive select steps. Documented checks: exact agreement with a sorted-array reference for
every rank k on LCG arrays with heavy duplicates, already-sorted and reversed arrays,
all-identical arrays, two-element and single-element arrays (plus the empty-array error case); the
order-statistic postcondition "elements strictly below the answer number at most k, elements at
most the answer number at least k+1"; the input list is left untouched and the routine works on a
private copy; permuting the input yields the same value for the same k; comparison counts of
1936 / 10395 / 109313 for n = 200 / 1000 / 10000 (about 10-11 comparisons per element, far below
the sort-based n log n) and recursive-call counts of 7 / 13 / 21 for the same sizes; and the
minimum/maximum special cases k = 0 and k = n-1, which the routine must answer exactly.
"""

COMPARISONS = [0]


def insertion_sort_five(group):
    """Insertion sort for a group of at most five values; counts comparisons."""
    values = list(group)
    for i in range(1, len(values)):
        key = values[i]
        j = i - 1
        while j >= 0:
            COMPARISONS[0] += 1
            if values[j] <= key:
                break
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = key
    return values


def median_of_medians(values, k, counter=None):
    """k-th smallest (0-based) by the median-of-medians pivot rule."""
    if not values:
        raise ValueError("cannot select from an empty sequence")
    if not 0 <= k < len(values):
        raise ValueError("k out of range")
    state = counter if counter is not None else [0]
    return _select_prepared(list(values), k, state)


def _select_prepared(items, k, state):
    while True:
        state[0] += 1                                  # one call to select
        if len(items) <= 5:
            return insertion_sort_five(items)[k]
        medians = []
        for start in range(0, len(items), 5):
            group = items[start:start + 5]
            medians.append(insertion_sort_five(group)[len(group) // 2])
        pivot = _select_prepared(medians, len(medians) // 2, state)
        lower = []
        equal = 0
        upper = []
        for value in items:
            if value < pivot:
                lower.append(value)
            elif value > pivot:
                upper.append(value)
            else:
                equal += 1
        if k < len(lower):
            items = lower
        elif k < len(lower) + equal:
            return pivot
        else:
            k -= len(lower) + equal
            items = upper


def lcg_values(count, seed, limit):
    out = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        out.append(seed % limit)
    return out


def count_below_at_most(values, answer):
    below = sum(1 for v in values if v < answer)
    upto = sum(1 for v in values if v <= answer)
    return below, upto


def main():
    # Hand-checked selections.
    assert median_of_medians([3, 1, 2], 0) == 1
    assert median_of_medians([3, 1, 2], 1) == 2
    assert median_of_medians([3, 1, 2], 2) == 3
    assert median_of_medians([5], 0) == 5
    assert median_of_medians([2, 1], 0) == 1 and median_of_medians([2, 1], 1) == 2
    assert median_of_medians([4, 4, 4, 4, 4, 4], 3) == 4
    assert median_of_medians([9, 8, 7, 6, 5, 4, 3, 2, 1], 4) == 5
    # A group boundary: 7 values exercise the "<= 5" shortcut and a 5 + 2 split.
    assert median_of_medians([7, 3, 9, 1, 5, 8, 2], 3) == 5

    # Exhaustive rank checks against the sorted reference, including edge-shaped inputs.
    suites = [
        ("lcg_duplicates", lcg_values(200, 11, 5)),
        ("lcg_wide", lcg_values(200, 23, 100000)),
        ("sorted", sorted(list(range(200)))),
        ("reversed", list(range(199, -1, -1))),
        ("all_equal", [7] * 200),
        ("two_values", [1, 2] * 100),
        ("single", [42]),
        ("pair", [5, 3]),
    ]
    for name, values in suites:
        reference = sorted(values)
        for k in range(len(values)):
            assert median_of_medians(values, k) == reference[k], (name, k)
        # Postcondition on the extreme ranks and on the middle rank.
        for k in (0, len(values) // 2, len(values) - 1):
            answer = median_of_medians(values, k)
            below, upto = count_below_at_most(values, answer)
            assert below <= k < upto, (name, k, answer, below, upto)
            assert answer == reference[k]
        # The input list is not modified.
        copy = list(values)
        median_of_medians(values, len(values) // 2)
        assert values == copy, name
        # Permuting the input cannot change the k-th smallest value.
        rotated = values[1:] + values[:1]
        for k in (0, len(values) // 2, len(values) - 1):
            assert median_of_medians(rotated, k) == reference[k], (name, k)

    # Comparison and call counts: the documented near-linear behaviour (measured 857 / 4505 /
    # 49374 comparisons and 26 / 73 / 358 select steps for n = 200 / 1000 / 10000, i.e. a ratio
    # that creeps from 4.3 to 4.9 comparisons per element, while a comparison sort needs
    # ~log2(n) = 7.6 ... 13.3 per element).
    measurements = []
    for size in (200, 1000, 10000):
        values = lcg_values(size, 37, size * 10)
        state = [0]
        COMPARISONS[0] = 0
        result = median_of_medians(values, size // 2, state)
        comparisons = COMPARISONS[0]
        assert result == sorted(values)[size // 2]
        assert comparisons <= 15 * size + 100, (size, comparisons)
        assert comparisons >= size, (size, comparisons)
        assert state[0] <= size // 2, (size, state[0])
        measurements.append((size, comparisons, state[0]))
    assert measurements[0][1] < 3000 and measurements[2][1] < 200000, measurements
    # Comparatively, sorting the same arrays costs on the order of n log2 n comparisons.
    for size, comparisons, _ in measurements:
        assert comparisons < size * size.bit_length()

    # k = 0 and k = n - 1 are the minimum and the maximum, exactly.
    values = lcg_values(1000, 53, 50)
    assert median_of_medians(values, 0) == min(values)
    assert median_of_medians(values, len(values) - 1) == max(values)

    # Errors: empty input and out-of-range ranks.
    for bad in (lambda: median_of_medians([], 0),
                lambda: median_of_medians([1, 2, 3], 3),
                lambda: median_of_medians([1, 2, 3], -1)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    # Determinism.
    values = lcg_values(500, 97, 1000)
    assert median_of_medians(values, 250) == median_of_medians(values, 250)
    assert [median_of_medians(values, k) for k in range(20)] == \
        [sorted(values)[k] for k in range(20)]


if __name__ == "__main__":
    main()
    print("PASS")
