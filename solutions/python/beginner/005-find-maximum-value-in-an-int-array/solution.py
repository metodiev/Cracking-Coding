"""Task 005 - Find maximum value in an int array.  (Tier: Beginner, CTCI Ch 1)"""


def max_value(items):
    if not items:
        raise ValueError("empty array")
    best = items[0]
    for x in items[1:]:
        if x > best:
            best = x
    return best


def main():
    assert max_value([3, 7, 2, 9, 4]) == 9
    assert max_value([-5, -1, -3]) == -1
    assert max_value([42]) == 42
    print("PASS")


if __name__ == "__main__":
    main()
