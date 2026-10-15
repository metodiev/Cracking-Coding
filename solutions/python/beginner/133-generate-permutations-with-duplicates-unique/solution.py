
"""Task 133 - Generate permutations with duplicates (unique). (Tier: Beginner, CTCI Ch 8)

Sort first, then skip a duplicate value when its identical predecessor was not used
at this level: that prunes identical branches while keeping every unique ordering.
"""


def unique_permutations(items):
    items = sorted(items)
    used = [False] * len(items)
    out = []

    def backtrack(current):
        if len(current) == len(items):
            out.append(list(current))
            return
        for i, x in enumerate(items):
            if used[i]:
                continue
            if i > 0 and items[i] == items[i - 1] and not used[i - 1]:
                continue                       # identical predecessor unused -> skip duplicate
            used[i] = True
            current.append(x)
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return out


def main():
    assert unique_permutations([1, 1, 2]) == [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    assert unique_permutations([1, 1, 1]) == [[1, 1, 1]]
    assert len(unique_permutations([1, 2, 3])) == 6      # distinct case unchanged
    assert unique_permutations([]) == [[]]
    # brute-force cross-check: set of all orderings via permutations of indices
    from itertools import permutations as brute
    for data in ([1, 1, 2], [1, 2, 2, 3], [2, 2, 2, 2]):
        expected = {p for p in brute(data)}
        got = {tuple(p) for p in unique_permutations(list(data))}
        assert got == expected, data


if __name__ == "__main__":
    main()
    print("PASS")
