"""Task 021 - Rotate array right by k positions (create new array). (Tier: Beginner, CTCI Ch 1)"""

def rotate_right(items, k):
    if not items:
        return []
    k %= len(items)
    return items[-k:] + items[:-k] if k else items[:]


def main():
    assert rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
    assert rotate_right([1, 2], 3) == [2, 1]
    assert rotate_right([], 5) == []


if __name__ == "__main__":
    main()
    print("PASS")
