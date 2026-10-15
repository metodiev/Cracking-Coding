"""Task 048 - Find most frequent element in array. (Tier: Beginner, CTCI Ch 1)"""

from collections import Counter


def most_frequent(items):
    """Ties resolved by first occurrence."""
    if not items:
        raise ValueError("empty array")
    counts = Counter(items)
    best, best_count = None, 0
    for x in items:
        if counts[x] > best_count:
            best, best_count = x, counts[x]
    return best


def main():
    assert most_frequent([1, 3, 2, 3, 4, 3]) == 3
    assert most_frequent([5]) == 5
    assert most_frequent([1, 2, 1, 2]) == 1     # tie -> first


if __name__ == "__main__":
    main()
    print("PASS")
