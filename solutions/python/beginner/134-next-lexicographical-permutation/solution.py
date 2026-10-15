
"""Task 134 - Next lexicographical permutation. (Tier: Beginner, CTCI Ch 8)

1) find the rightmost ascent (pivot); 2) swap it with the smallest larger suffix value;
3) reverse the suffix. Returns False (and wraps to the sorted order) at the last permutation.
"""


def next_permutation(a):
    """Mutates the list in place; returns True if a next permutation exists."""
    n = len(a)
    pivot = n - 2
    while pivot >= 0 and a[pivot] >= a[pivot + 1]:
        pivot -= 1
    if pivot < 0:
        a.reverse()                       # was the last permutation: wrap around
        return False
    successor = n - 1
    while a[successor] <= a[pivot]:
        successor -= 1
    a[pivot], a[successor] = a[successor], a[pivot]
    a[pivot + 1:] = reversed(a[pivot + 1:])
    return True


def main():
    a = [1, 2, 3]
    assert next_permutation(a) and a == [1, 3, 2]
    assert next_permutation(a) and a == [2, 1, 3]

    last = [3, 2, 1]
    assert next_permutation(last) is False
    assert last == [1, 2, 3]              # wrapped to the first permutation

    single = [5]
    assert next_permutation(single) is False and single == [5]

    # walking all 24 permutations of [1,2,3,4] returns to the start
    b = [1, 2, 3, 4]
    seen = 1
    while next_permutation(b):
        seen += 1
    assert seen == 24 and b == [1, 2, 3, 4]


if __name__ == "__main__":
    main()
    print("PASS")
