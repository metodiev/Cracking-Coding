"""Task 259 - Kth smallest element using order statistics / quickselect. (Tier: Intermediate, CTCI Beyond — Numerical & Scientific Computing)

Quickselect: Lomuto-partition the window around its last element (the documented
deterministic pivot), then continue only into the side that contains position k-1 - the loop
is iterative, so a bad pivot sequence can never grow the call stack. Expected O(n) per query
with O(1) extra space; worst case O(n^2), e.g. already-sorted input. k is 1-based and the
input list is copied, so callers keep their data. Test values come from a Park-Miller LCG so
every language sees identical arrays.
"""

LCG_MOD = 2147483647
LCG_MUL = 48271


def partition(a, lo, hi):
    """Lomuto partition with a[hi] as the pivot; returns the pivot's final index."""
    pivot = a[hi]
    store = lo
    for i in range(lo, hi):
        if a[i] <= pivot:
            a[store], a[i] = a[i], a[store]
            store += 1
    a[store], a[hi] = a[hi], a[store]
    return store


def kth_smallest(values, k):
    """The k-th smallest value (1-based k) of values."""
    a = list(values)
    if k < 1 or k > len(a):
        raise ValueError("k out of range")
    lo, hi, target = 0, len(a) - 1, k - 1
    while True:
        p = partition(a, lo, hi)
        if p == target:
            return a[p]
        if p < target:
            lo = p + 1
        else:
            hi = p - 1


def lcg_values(seed, count, bound):
    """Park-Miller minimal standard: state = state * 48271 mod 2^31-1."""
    state = seed
    out = []
    for _ in range(count):
        state = (state * LCG_MUL) % LCG_MOD
        out.append(state % bound)
    return out


def main():
    assert kth_smallest([7], 1) == 7
    assert kth_smallest([3, 1, 2], 1) == 1
    assert kth_smallest([3, 1, 2], 2) == 2
    assert kth_smallest([3, 1, 2], 3) == 3
    assert kth_smallest([5, 5, 5, 5], 3) == 5
    assert kth_smallest([2, 1, 1, 3], 2) == 1              # sorted: [1, 1, 2, 3]

    original = [9, 4, 7, 1, 8, 2, 6, 3, 5]
    assert kth_smallest(original, 4) == 4
    assert original == [9, 4, 7, 1, 8, 2, 6, 3, 5]         # input untouched

    for bad in (0, -1, 10):
        try:
            kth_smallest([1, 2, 3], bad)
            raise AssertionError("expected ValueError for k=%r" % (bad,))
        except ValueError:
            pass

    # Fixed cases (incl. duplicates) cross-checked against the sorted() oracle for every k.
    cases = [
        lcg_values(259001, 1, 5),
        lcg_values(259002, 2, 5),
        lcg_values(259003, 3, 7),
        lcg_values(259004, 5, 20),
        lcg_values(259005, 8, 3),
        lcg_values(259006, 16, 1000),
        lcg_values(259007, 33, 10),
        lcg_values(259008, 50, 50),
        list(range(64)),
        list(range(63, -1, -1)),
        [7] * 32,
        [5, 5, 4, 4, 3, 3, 2, 2, 1, 1],
        lcg_values(259009, 200, 10000),
    ]
    for case in cases:
        ordered = sorted(case)
        snapshot = list(case)
        for k in range(1, len(case) + 1):
            assert kth_smallest(case, k) == ordered[k - 1]
        assert case == snapshot                             # still untouched


if __name__ == "__main__":
    main()
    print("PASS")
