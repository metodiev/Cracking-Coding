"""Task 024 - Remove element at index from array (in-place). (Tier: Beginner, CTCI Ch 1)"""

def remove_at(items, index):
    """Shifts elements left; returns the new logical length."""
    if index < 0 or index >= len(items):
        raise IndexError("index out of range")
    for i in range(index, len(items) - 1):
        items[i] = items[i + 1]
    return len(items) - 1


def main():
    a = [10, 20, 30, 40]
    n = remove_at(a, 1)
    assert n == 3 and a[:n] == [10, 30, 40]
    b = [1]
    assert remove_at(b, 0) == 0


if __name__ == "__main__":
    main()
    print("PASS")
