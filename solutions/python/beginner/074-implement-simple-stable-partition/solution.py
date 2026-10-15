
"""Task 074 - Implement simple stable partition. (Tier: Beginner, CTCI Ch 10)

Rearrange so predicate-true elements come first, preserving relative order within
both groups. O(n) time, O(n) extra space (or O(n^2) in place with rotations).
"""


def stable_partition(items, predicate):
    yes = [x for x in items if predicate(x)]
    no = [x for x in items if not predicate(x)]
    return yes + no


def main():
    assert stable_partition([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0) == [2, 4, 6, 1, 3, 5]
    assert stable_partition([], lambda x: True) == []
    assert stable_partition([1, 3, 5], lambda x: x % 2 == 0) == [1, 3, 5]
    # stability: both 4s keep their original relative order
    assert stable_partition([4, 1, 4, 2], lambda x: x == 4) == [4, 4, 1, 2]


if __name__ == "__main__":
    main()
    print("PASS")
