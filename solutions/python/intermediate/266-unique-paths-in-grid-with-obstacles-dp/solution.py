"""Task 266 - Unique paths in grid with obstacles (DP). (Tier: Intermediate, CTCI Ch 8)

Number of monotone right/down paths from the top-left to the bottom-right cell; 1 marks
an obstacle, 0 a free cell. Two equivalent counters are implemented and asserted to
agree on identical grids:
  * unique_paths_with_obstacles_2d - explicit m x n DP,
  * count_ways_265                 - task 265's rolling 1D DP (same recurrence).
Relation (documented): with no obstacles both reduce to the closed form
C(m+n-2, m-1) (unique_paths_no_obstacles) that gives the "unique paths" name; obstacles
only zero out blocked cells and, since paths cannot cross them, they can only lower the
count. A blocked start or end yields 0.

O(m*n) time; the 2D version uses O(m*n) space, the rolling one O(n).
unique_paths_brute enumerates the paths recursively and is used only for cross-checks.
"""

import math


def unique_paths_with_obstacles_2d(grid):
    if not grid or not grid[0]:
        return 0
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    for r in range(m):
        for c in range(n):
            if grid[r][c] == 1:
                dp[r][c] = 0
            elif r == 0 and c == 0:
                dp[r][c] = 1
            else:
                from_above = dp[r - 1][c] if r > 0 else 0
                from_left = dp[r][c - 1] if c > 0 else 0
                dp[r][c] = from_above + from_left
    return dp[m - 1][n - 1]


def count_ways_265(grid):
    """Task 265's rolling 1D DP; identical grids must give identical counts."""
    if not grid or not grid[0]:
        return 0
    n = len(grid[0])
    ways = [0] * n
    ways[0] = 1
    for row in grid:
        for c in range(n):
            if row[c] == 1:
                ways[c] = 0
            elif c > 0:
                ways[c] += ways[c - 1]
    return ways[n - 1]


def unique_paths_no_obstacles(m, n):
    """Closed form for an obstacle-free m x n grid: C(m+n-2, m-1)."""
    if m <= 0 or n <= 0:
        return 0
    return math.comb(m + n - 2, m - 1)


def unique_paths_brute(grid):
    """Recursive enumeration used only for cross-checks."""
    m, n = len(grid), len(grid[0])

    def go(r, c):
        if grid[r][c] == 1:
            return 0
        if r == m - 1 and c == n - 1:
            return 1
        total = 0
        if r + 1 < m:
            total += go(r + 1, c)
        if c + 1 < n:
            total += go(r, c + 1)
        return total

    return go(0, 0)


def main():
    assert unique_paths_with_obstacles_2d([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 6     # classic 3x3
    assert unique_paths_with_obstacles_2d([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2     # classic obstacle variant
    assert unique_paths_with_obstacles_2d([[0, 1], [0, 0]]) == 1
    assert unique_paths_with_obstacles_2d([[0, 0], [0, 1]]) == 0                      # goal blocked
    assert unique_paths_with_obstacles_2d([[1, 0], [0, 0]]) == 0                      # start blocked
    assert unique_paths_with_obstacles_2d([[0, 0], [1, 0]]) == 1                      # detour right
    assert unique_paths_with_obstacles_2d([[0, 0, 0, 0, 0]]) == 1                     # 1 x n
    assert unique_paths_with_obstacles_2d([[0], [0], [0]]) == 1                       # n x 1
    assert unique_paths_with_obstacles_2d([[0, 1, 0]]) == 0                           # 1 x n blocked
    assert unique_paths_with_obstacles_2d([]) == 0
    assert unique_paths_with_obstacles_2d([[]]) == 0

    shared = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [1, 0]],
        [[0, 1, 0], [0, 0, 0]],
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
    ]
    for grid in shared:                       # 265's counter agrees on identical grids
        assert unique_paths_with_obstacles_2d(grid) == count_ways_265(grid)

    for m, n, expected in ((1, 1, 1), (1, 5, 1), (2, 2, 2), (3, 3, 6), (3, 7, 28), (4, 4, 20)):
        free = [[0] * n for _ in range(m)]
        assert unique_paths_with_obstacles_2d(free) == expected == unique_paths_no_obstacles(m, n)

    import random

    rng = random.Random(266)
    for _ in range(300):                      # 2D == 265's rolling DP == recursive enumeration
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        grid = [[1 if rng.random() < 0.3 else 0 for _ in range(n)] for _ in range(m)]
        assert unique_paths_with_obstacles_2d(grid) == count_ways_265(grid) == unique_paths_brute(grid)


if __name__ == "__main__":
    main()
    print("PASS")
