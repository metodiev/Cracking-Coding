"""Task 020 - Merge two sorted arrays into a sorted array.  (Tier: Beginner, CTCI Ch 10)"""


def merge_sorted(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def main():
    assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted([], [1, 2]) == [1, 2]
    assert merge_sorted([2, 2], [2]) == [2, 2, 2]
    assert merge_sorted([], []) == []
    print("PASS")


if __name__ == "__main__":
    main()
