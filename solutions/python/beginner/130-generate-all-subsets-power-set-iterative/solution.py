
"""Task 130 - Generate all subsets (power set) iterative. (Tier: Beginner, CTCI Ch 8)

Two iterative routes: bitmask enumeration and incremental doubling.
Order is deterministic (bitmask order = 0..2^n-1).
"""


def subsets_bitmask(items):
    n = len(items)
    out = []
    for mask in range(1 << n):
        out.append([items[i] for i in range(n) if mask & (1 << i)])
    return out


def subsets_incremental(items):
    """Start with [[]]; each next item doubles the list with item appended."""
    out = [[]]
    for x in items:
        out += [subset + [x] for subset in out]
    return out


def main():
    assert subsets_bitmask([1, 2, 3]) == [
        [], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3],
    ]
    assert subsets_bitmask([]) == [[]]
    assert subsets_incremental([1, 2]) == [[], [1], [2], [1, 2]]

    # both routes produce the same set of subsets (as sorted tuples)
    for items in ([1, 2, 3, 4], ["a", "b", "c"]):
        a = {tuple(sorted(s)) for s in subsets_bitmask(items)}
        b = {tuple(sorted(s)) for s in subsets_incremental(items)}
        assert a == b and len(a) == 2 ** len(items)


if __name__ == "__main__":
    main()
    print("PASS")
