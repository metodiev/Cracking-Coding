"""Task 019 - Check if array is sorted (ascending).  (Tier: Beginner, CTCI Ch 1)"""


def is_sorted(items):
    return all(items[i] <= items[i + 1] for i in range(len(items) - 1))


def main():
    assert is_sorted([])
    assert is_sorted([1])
    assert is_sorted([1, 2, 2, 3])
    assert not is_sorted([1, 3, 2])
    print("PASS")


if __name__ == "__main__":
    main()
