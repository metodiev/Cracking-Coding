
"""Task 132 - Generate permutations of distinct elements. (Tier: Beginner, CTCI Ch 8)

Backtracking with a used[] array: n! results, deterministic order.
"""


def permutations(items):
    used = [False] * len(items)
    out = []

    def backtrack(current):
        if len(current) == len(items):
            out.append(list(current))
            return
        for i, x in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            current.append(x)
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return out


def main():
    assert permutations([1, 2, 3]) == [
        [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1],
    ]
    assert permutations([]) == [[]]
    assert permutations([7]) == [[7]]
    assert len(permutations([1, 2, 3, 4])) == 24
    # all results are distinct and each is a reordering of the input
    perms = permutations([1, 2, 3, 4])
    assert len({tuple(p) for p in perms}) == 24
    assert all(sorted(p) == [1, 2, 3, 4] for p in perms)


if __name__ == "__main__":
    main()
    print("PASS")
