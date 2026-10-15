
"""Task 131 - Generate all subsets recursive. (Tier: Beginner, CTCI Ch 8)"""


def subsets_recursive(items):
    """Backtracking: at each index, branch on take/skip. O(2^n) results."""
    out = []

    def backtrack(i, current):
        if i == len(items):
            out.append(list(current))
            return
        backtrack(i + 1, current)                  # skip items[i]
        current.append(items[i])
        backtrack(i + 1, current)                  # take items[i]
        current.pop()

    backtrack(0, [])
    return out


def main():
    assert subsets_recursive([1, 2, 3]) == [
        [], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3],
    ]
    assert subsets_recursive([]) == [[]]
    assert len(subsets_recursive([1, 2, 3, 4])) == 16

    # same set of subsets as the iterative bitmask version (task 130)
    iterative = {tuple(sorted(subset)) for subset in [
        [], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]
    }
    recursive = {tuple(sorted(s)) for s in subsets_recursive([1, 2, 3])}
    assert iterative == recursive


if __name__ == "__main__":
    main()
    print("PASS")
