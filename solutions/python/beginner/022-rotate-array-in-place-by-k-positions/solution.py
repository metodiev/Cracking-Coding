"""Task 022 - Rotate array in-place by k positions. (Tier: Beginner, CTCI Ch 1)"""

def rotate_in_place(items, k):
    """Three-reversal trick: O(n) time, O(1) extra space."""
    n = len(items)
    if n == 0:
        return items
    k %= n

    def reverse(lo, hi):
        while lo < hi:
            items[lo], items[hi] = items[hi], items[lo]
            lo, hi = lo + 1, hi - 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
    return items


def main():
    assert rotate_in_place([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
    assert rotate_in_place([1], 4) == [1]
    assert rotate_in_place([1, 2], 0) == [1, 2]


if __name__ == "__main__":
    main()
    print("PASS")
