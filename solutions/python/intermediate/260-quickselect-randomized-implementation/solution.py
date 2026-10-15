"""Task 260 - Quickselect, randomized implementation. (Tier: Intermediate, CTCI Ch 10)

Randomized quickselect with a three-way (Dutch-national-flag) partition. The pivot index is
drawn from a fixed-seed Park-Miller LCG, so every run - and every language - makes the same
choices and the whole run is reproducible; the returned value never depends on the seed,
only on the input. Complexity note: expected O(n), worst case O(n^2) (much better than worst
case on all inputs here); duplicates land in one equal block, so an all-equal array finishes
after exactly one partition, and the search loop is iterative - no stack can blow up. k is
1-based; the input list is copied.
"""

LCG_MOD = 2147483647
LCG_MUL = 48271
SEED = 260260


def partition3(a, lo, hi, pivot):
    """Dutch flag; returns (lt, gt): a[lo:lt] < pivot == a[lt:gt+1], a[gt+1:hi+1] > pivot."""
    lt, i, gt = lo, lo, hi
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif a[i] > pivot:
            a[i], a[gt] = a[gt], a[i]
            gt -= 1
        else:
            i += 1
    return lt, gt


def kth_smallest(values, k, seed=SEED):
    """The k-th smallest value (1-based k); the LCG seed is exposed for testing only."""
    a = list(values)
    if k < 1 or k > len(a):
        raise ValueError("k out of range")
    state = seed % LCG_MOD or 1
    lo, hi, target = 0, len(a) - 1, k - 1
    while True:
        state = (state * LCG_MUL) % LCG_MOD
        pivot = a[lo + state % (hi - lo + 1)]
        lt, gt = partition3(a, lo, hi, pivot)
        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            return a[target]                # target sits inside the equal block


def lcg_values(seed, count, bound):
    state = seed
    out = []
    for _ in range(count):
        state = (state * LCG_MUL) % LCG_MOD
        out.append(state % bound)
    return out


def main():
    assert kth_smallest([7], 1) == 7
    assert kth_smallest([3, 1, 2], 2) == 2
    assert kth_smallest([5, 5, 5, 5], 3) == 5
    assert kth_smallest([2, 1, 1, 3], 2) == 1

    original = [9, 4, 7, 1, 8, 2, 6, 3, 5]
    assert kth_smallest(original, 4) == 4
    assert original == [9, 4, 7, 1, 8, 2, 6, 3, 5]         # input untouched

    for bad in (0, -1, 10):
        try:
            kth_smallest([1, 2, 3], bad)
            raise AssertionError("expected ValueError for k=%r" % (bad,))
        except ValueError:
            pass

    # Agrees with the sorted() oracle on the same fixed cases (all k, duplicates included).
    cases = [
        lcg_values(260001, 1, 5),
        lcg_values(260002, 2, 5),
        lcg_values(260003, 3, 7),
        lcg_values(260004, 5, 20),
        lcg_values(260005, 8, 3),
        lcg_values(260006, 16, 1000),
        lcg_values(260007, 33, 10),
        lcg_values(260008, 50, 50),
        list(range(64)),
        list(range(63, -1, -1)),
        [9] * 17,
        [1, 2] * 200,
        lcg_values(260009, 200, 10000),
    ]
    for case in cases:
        ordered = sorted(case)
        for k in range(1, len(case) + 1):
            assert kth_smallest(case, k) == ordered[k - 1]
            assert kth_smallest(case, k, seed=999) == ordered[k - 1]   # seed-independent

    # Adversarial shapes: one partition is enough for all-equal input, and the iterative
    # loop keeps even 5000 sorted/equal elements safe.
    big_equal = [7] * 5000
    for k in (1, 2, 2500, 4999, 5000):
        assert kth_smallest(big_equal, k) == 7
    ascending = list(range(4000))
    descending = list(range(3999, -1, -1))     # same values, opposite order
    sawtooth = [i % 17 for i in range(1000)]
    for k in (1, 2000, 4000):
        assert kth_smallest(ascending, k) == k - 1
        assert kth_smallest(descending, k) == k - 1
    ordered_saw = sorted(sawtooth)
    for k in (1, 500, 1000):
        assert kth_smallest(sawtooth, k) == ordered_saw[k - 1]


if __name__ == "__main__":
    main()
    print("PASS")
