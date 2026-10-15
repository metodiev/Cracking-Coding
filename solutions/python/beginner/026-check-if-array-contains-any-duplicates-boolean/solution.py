"""Task 026 - Check if array contains any duplicates (boolean). (Tier: Beginner, CTCI Ch 1)"""

def contains_duplicates(items):
    seen = set()
    for x in items:
        if x in seen:
            return True
        seen.add(x)
    return False


def main():
    assert contains_duplicates([1, 2, 3, 1])
    assert not contains_duplicates([1, 2, 3])
    assert not contains_duplicates([])


if __name__ == "__main__":
    main()
    print("PASS")
