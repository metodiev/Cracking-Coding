"""Task 261 - Median of two sorted arrays (log min(n,m) algorithm). (Tier: Intermediate, CTCI Ch 10)

Partition-based binary search on the SHORTER array: pick i = number of a-elements on
the left side; the partner split j = (m+n+1)//2 - i follows. A valid partition satisfies
a[i-1] <= b[j] and b[j-1] <= a[i]; out-of-range sides count as +-inf, which is exactly
what empty arrays and the "one array fully left/right" cases need. The median is then
max(left side) for an odd combined length and the mean of max(left) and min(right) for
an even one. O(log min(m, n)) time, O(1) extra space.

median_oracle (O(m+n) merge) is used only as an independent cross-check. Two empty
arrays are rejected with ValueError (documented boundary).
"""

NEG_INF = float("-inf")
POS_INF = float("inf")


def median_of_two_sorted(a, b):
    if not a and not b:
        raise ValueError("at least one array must be non-empty")
    if len(a) > len(b):
        a, b = b, a                                  # binary search the shorter array
    m, n = len(a), len(b)
    half = (m + n + 1) // 2
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = half - i
        a_left = a[i - 1] if i > 0 else NEG_INF
        a_right = a[i] if i < m else POS_INF
        b_left = b[j - 1] if j > 0 else NEG_INF
        b_right = b[j] if j < n else POS_INF
        if a_left <= b_right and b_left <= a_right:  # valid partition found
            if (m + n) % 2 == 1:
                return float(max(a_left, b_left))
            return (max(a_left, b_left) + min(a_right, b_right)) / 2.0
        if a_left > b_right:
            hi = i - 1                               # a contributes too many left elements
        else:
            lo = i + 1                               # a contributes too few left elements
    raise ValueError("inputs are not sorted")


def merge_sorted(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def median_oracle(a, b):
    """O(m+n) merged-middle reference used only for cross-checks."""
    merged = merge_sorted(a, b)
    n = len(merged)
    return float(merged[n // 2]) if n % 2 else (merged[n // 2 - 1] + merged[n // 2]) / 2.0


def main():
    assert median_of_two_sorted([1, 3], [2]) == 2.0
    assert median_of_two_sorted([1, 2], [3, 4]) == 2.5
    assert median_of_two_sorted([], [1]) == 1.0                  # empty first array
    assert median_of_two_sorted([1, 2, 3, 4, 5], []) == 3.0      # empty second array
    assert median_of_two_sorted([0, 0], [0, 0]) == 0.0
    assert median_of_two_sorted([2], [1, 3]) == 2.0
    assert median_of_two_sorted([1, 3, 8], [7, 9, 10, 11]) == 8.0        # classic mixed case
    assert median_of_two_sorted([1, 2], [3, 4, 5, 6]) == 3.5             # one array fully left
    assert median_of_two_sorted([5, 6, 7, 8], [1, 2, 3, 4]) == 4.5       # one array fully right
    assert median_of_two_sorted([1, 1], [1, 1]) == 1.0
    assert median_of_two_sorted([-5, -3, -1], [-2]) == -2.5
    assert median_of_two_sorted([1, 2, 3], [4, 5, 6]) == 3.5

    try:
        median_of_two_sorted([], [])
        raise AssertionError("both-empty input must raise")
    except ValueError:
        pass

    import random

    rng = random.Random(261)
    for _ in range(300):                            # cross-check vs the merge oracle
        a = sorted(rng.randint(-20, 20) for _ in range(rng.randint(0, 8)))
        b = sorted(rng.randint(-20, 20) for _ in range(rng.randint(0, 8)))
        if not a and not b:
            continue
        assert median_of_two_sorted(a, b) == median_oracle(a, b)


if __name__ == "__main__":
    main()
    print("PASS")
