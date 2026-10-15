"""Task 023 - Find second largest element in an array. (Tier: Beginner, CTCI Ch 1)"""

def second_largest(items):
    """O(n) single pass; raises if fewer than two distinct values."""
    best = second = None
    for x in items:
        if best is None or x > best:
            second = best
            best = x
        elif x != best and (second is None or x > second):
            second = x
    if second is None:
        raise ValueError("no second largest")
    return second


def main():
    assert second_largest([3, 7, 2, 9, 4]) == 7
    assert second_largest([5, 5, 4]) == 4
    assert second_largest([-1, -2, -3]) == -2


if __name__ == "__main__":
    main()
    print("PASS")
