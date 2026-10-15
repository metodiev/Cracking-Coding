"""Task 008 - Find the index of a target in an unsorted array (linear search).  (Tier: Beginner, CTCI Ch 1)"""


def linear_search(items, target):
    for i, x in enumerate(items):
        if x == target:
            return i
    return -1


def main():
    assert linear_search([4, 2, 7, 1], 7) == 2
    assert linear_search([4, 2, 7, 1], 9) == -1
    assert linear_search([], 1) == -1
    assert linear_search([5, 5, 5], 5) == 0
    print("PASS")


if __name__ == "__main__":
    main()
