
"""Task 206 - Pair with sum closest to target in sorted array. (Tier: Intermediate, CTCI Ch 16)

Two pointers on a sorted array; move the pointer that improves the distance. O(n).
Returns (pair, sum) with a documented tie-break: on equal distance the first found wins.
"""


def closest_pair(a, target):
    lo, hi = 0, len(a) - 1
    if lo >= hi:
        raise ValueError("need at least two values")
    best_pair = (a[lo], a[hi])
    best_diff = abs(a[lo] + a[hi] - target)
    while lo < hi:
        total = a[lo] + a[hi]
        diff = abs(total - target)
        if diff < best_diff:
            best_diff = diff
            best_pair = (a[lo], a[hi])
        if total < target:
            lo += 1
        elif total > target:
            hi -= 1
        else:
            break                       # exact match
    return best_pair, best_pair[0] + best_pair[1]


def main():
    a = [1, 3, 4, 5, 7]
    assert closest_pair(a, 10) == ((3, 7), 10)
    assert closest_pair(a, 8) == ((1, 7), 8)
    assert closest_pair(a, 12) == ((5, 7), 12)
    assert closest_pair(a, 0) == ((1, 3), 4)          # closest achievable
    assert closest_pair([2, 4], 5) == ((2, 4), 6)
    try:
        closest_pair([1], 5)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
