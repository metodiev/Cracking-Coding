"""Task 007 - Compute sum and average of array elements.  (Tier: Beginner, CTCI Ch 1)"""


def sum_and_average(items):
    if not items:
        return 0, 0.0
    total = sum(items)
    return total, total / len(items)


def main():
    assert sum_and_average([1, 2, 3, 4]) == (10, 2.5)
    assert sum_and_average([]) == (0, 0.0)
    assert sum_and_average([5]) == (5, 5.0)
    print("PASS")


if __name__ == "__main__":
    main()
