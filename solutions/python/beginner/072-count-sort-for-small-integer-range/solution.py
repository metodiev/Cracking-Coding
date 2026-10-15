
"""Task 072 - Count sort for small integer range. (Tier: Beginner, CTCI Ch 10)

O(n + k) where k is the value range. Stable when built with prefix sums.
"""


def counting_sort(a):
    if not a:
        return []
    lo, hi = min(a), max(a)
    counts = [0] * (hi - lo + 1)
    for x in a:
        counts[x - lo] += 1
    out = []
    for offset, count in enumerate(counts):
        out.extend([lo + offset] * count)
    return out


def main():
    assert counting_sort([4, 2, 2, 8, 3, 3, 1]) == [1, 2, 2, 3, 3, 4, 8]
    assert counting_sort([]) == []
    assert counting_sort([7, 7]) == [7, 7]
    assert counting_sort([-3, -1, -3]) == [-3, -3, -1]     # negatives supported


if __name__ == "__main__":
    main()
    print("PASS")
