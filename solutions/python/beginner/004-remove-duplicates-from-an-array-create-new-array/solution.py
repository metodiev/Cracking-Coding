"""Task 004 - Remove duplicates from an array (create new array).  (Tier: Beginner, CTCI Ch 1)"""


def remove_duplicates(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]
    assert remove_duplicates([]) == []
    assert remove_duplicates([5, 5, 5]) == [5]
    assert remove_duplicates(["a", "b", "a"]) == ["a", "b"]
    print("PASS")


if __name__ == "__main__":
    main()
