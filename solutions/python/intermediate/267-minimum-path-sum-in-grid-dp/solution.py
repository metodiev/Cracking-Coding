"""Task 267 - Minimum path sum in grid (DP). (Tier: Intermediate, CTCI Ch 8)

Minimum sum along a monotone right/down path from the top-left to the bottom-right
cell. Rolling 1D DP over rows: dp[c] = min(dp[c] from above, dp[c-1] from the left)
+ grid[r][c]; the first row and first column each have a single predecessor.
O(m*n) time, O(n) space.

Boundaries (documented): an empty grid (or empty first row) raises ValueError; a single
cell returns its value; 1 x n and n x 1 grids have exactly one path, so their sum is the
answer. Negative values are allowed (they only make a path cheaper).
min_path_sum_brute enumerates all paths recursively and is used only for cross-checks.
"""


def min_path_sum(grid):
    if not grid or not grid[0]:
        raise ValueError("grid must be a non-empty rectangle")
    n = len(grid[0])
    dp = [0] * n
    for r, row in enumerate(grid):
        for c in range(n):
            if r == 0 and c == 0:
                dp[c] = row[c]
            elif r == 0:
                dp[c] = dp[c - 1] + row[c]                      # only from the left
            elif c == 0:
                dp[c] += row[c]                                 # only from above
            else:
                dp[c] = min(dp[c], dp[c - 1]) + row[c]
    return dp[n - 1]


def min_path_sum_brute(grid):
    """Recursive enumeration of all paths; used only for cross-checks."""
    m, n = len(grid), len(grid[0])

    def go(r, c):
        if r == m - 1 and c == n - 1:
            return grid[r][c]
        best = None
        if r + 1 < m:
            best = go(r + 1, c)
        if c + 1 < n:
            right = go(r, c + 1)
            best = right if best is None else min(best, right)
        return grid[r][c] + best

    return go(0, 0)


def main():
    assert min_path_sum([[1, 3, 1], [1, 5, 1], [4, 2, 1]]) == 7      # classic: 1+3+1+1+1
    assert min_path_sum([[5]]) == 5                                  # single cell
    assert min_path_sum([[-3]]) == -3
    assert min_path_sum([[1, 2, 3]]) == 6                            # 1 x n: the only path
    assert min_path_sum([[1], [2], [3]]) == 6                        # n x 1: the only path
    assert min_path_sum([[1, 2, 3], [4, 100, 6]]) == 12              # route above the huge cell
    assert min_path_sum([[1, 100], [1, 1]]) == 3                     # go down first, then right
    assert min_path_sum([[1, -3], [1, -5]]) == -7                    # negatives are allowed

    try:
        min_path_sum([])
        raise AssertionError("empty grid must raise")
    except ValueError:
        pass
    try:
        min_path_sum([[]])
        raise AssertionError("empty first row must raise")
    except ValueError:
        pass

    import random

    rng = random.Random(267)
    for _ in range(300):                            # cross-check vs recursive enumeration
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        grid = [[rng.randint(-5, 9) for _ in range(n)] for _ in range(m)]
        assert min_path_sum(grid) == min_path_sum_brute(grid)


if __name__ == "__main__":
    main()
    print("PASS")
