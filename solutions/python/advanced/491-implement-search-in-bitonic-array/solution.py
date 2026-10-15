"""Task 491 - Implement search in bitonic array. (Tier: Advanced, CTCI Ch 10)

A bitonic (mountain) array strictly rises to a single peak and then strictly falls; the peak
may sit at either end, so purely increasing and purely decreasing arrays are the degenerate
cases. find_peak is the standard O(log n) binary search that compares values[mid] with its
neighbour; search then binary searches the ascending side [0, peak] and the descending side
(peak, n-1] separately, for O(log n) total. An absent target returns the documented sentinel
-1 (indices are non-negative, so -1 is unambiguous). Verified against a linear-scan oracle
on every element and on absent probes over LCG-built bitonic arrays of many sizes, including
single-element arrays and arrays whose peak is at a boundary.
"""

SENTINEL = -1


def find_peak(values):
    """Index of the unique maximum (the peak) of a strictly bitonic array."""
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < values[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search(values, target):
    """Index of target in a strictly bitonic array, or SENTINEL (-1) when absent."""
    n = len(values)
    if n == 0:
        return SENTINEL
    peak = find_peak(values)

    lo, hi = 0, peak                       # ascending side
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] == target:
            return mid
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1

    lo, hi = peak + 1, n - 1               # descending side
    while lo <= hi:
        mid = (lo + hi) // 2
        if values[mid] == target:
            return mid
        if values[mid] > target:
            lo = mid + 1
        else:
            hi = mid - 1

    return SENTINEL


def linear_search(values, target):
    """Simple oracle: first index holding target, or SENTINEL."""
    for i, value in enumerate(values):
        if value == target:
            return i
    return SENTINEL


def lcg_bitonic(n, seed, peak):
    """Build a strictly bitonic array of n distinct LCG values with its peak at `peak`:
    take n distinct values, sort them, and mirror the tail (s[peak+1:] descending) so the
    maximum s[-1] sits at index `peak`."""
    distinct = []
    seen = set()
    while len(distinct) < n:
        seed = (seed * 75 + 74) % 65537
        value = seed % 1000
        if value not in seen:
            seen.add(value)
            distinct.append(value)
    distinct.sort()
    tail_len = n - 1 - peak                    # values strictly below the peak, descending
    descending = distinct[tail_len - 1::-1] if tail_len > 0 else []
    return distinct[tail_len:] + descending


def main():
    # Hand example: peak 9 at index 3; both sides strict.
    sample = [1, 3, 5, 9, 7, 4, 2]
    assert find_peak(sample) == 3
    for i, value in enumerate(sample):
        assert search(sample, value) == i
    for absent in (0, 6, 8, 10, -5):
        assert search(sample, absent) == SENTINEL

    # Degenerate shapes: single element, purely increasing, purely decreasing.
    assert find_peak([5]) == 0
    assert search([5], 5) == 0 and search([5], 4) == SENTINEL
    assert find_peak([1, 2, 3, 4]) == 3
    assert search([1, 2, 3, 4], 4) == 3 and search([1, 2, 3, 4], 0) == SENTINEL
    assert find_peak([4, 3, 2, 1]) == 0
    assert search([4, 3, 2, 1], 1) == 3 and search([4, 3, 2, 1], 5) == SENTINEL

    # LCG bitonic arrays: every size from 1..17 and every peak position.
    checked_arrays = 0
    for n in range(1, 18):
        for peak in range(n):
            values = lcg_bitonic(n, 1000 + 13 * n + peak, peak)
            checked_arrays += 1
            assert find_peak(values) == peak, (n, peak)
            for i, value in enumerate(values):
                assert search(values, value) == i, (n, peak, i)
                assert search(values, value) == linear_search(values, value)
            for absent in (values[0] - 1, values[peak] + 1, 1000, -1, 65537):
                assert search(values, absent) == SENTINEL, (n, peak, absent)
                assert linear_search(values, absent) == SENTINEL
    assert checked_arrays == 17 * 18 // 2, checked_arrays

    # Larger LCG arrays with interior and boundary peaks: elementwise oracle comparison.
    for n, seed, peak in ((256, 4242, 100), (257, 777, 0), (255, 31337, 254), (300, 5, 150)):
        values = lcg_bitonic(n, seed, peak)
        assert find_peak(values) == peak
        for i in (0, 1, n // 2, peak - 1, peak, peak + 1, n - 2, n - 1):
            if 0 <= i < n:
                assert search(values, values[i]) == i
        assert search(values, values[peak] + 7) == SENTINEL


if __name__ == "__main__":
    main()
    print("PASS")
