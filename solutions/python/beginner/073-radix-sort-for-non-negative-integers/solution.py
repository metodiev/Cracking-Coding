
"""Task 073 - Radix sort for non-negative integers. (Tier: Beginner, CTCI Ch 10)

LSD radix sort with counting sort per digit: O(d * (n + 10)).
"""


def radix_sort(a):
    if not a:
        return []
    maximum = max(a)
    exp = 1
    out = list(a)
    while maximum // exp > 0:
        buckets = [[] for _ in range(10)]
        for x in out:
            buckets[(x // exp) % 10].append(x)
        out = [x for bucket in buckets for x in bucket]
        exp *= 10
    return out


def main():
    assert radix_sort([170, 45, 75, 90, 802, 24, 2, 66]) == [2, 24, 45, 66, 75, 90, 170, 802]
    assert radix_sort([]) == []
    assert radix_sort([7]) == [7]
    assert radix_sort([0, 10, 100]) == [0, 10, 100]


if __name__ == "__main__":
    main()
    print("PASS")
