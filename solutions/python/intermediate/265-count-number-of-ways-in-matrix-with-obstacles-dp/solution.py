"""Task 265 - Count number of ways in matrix with obstacles (DP). (Tier: Intermediate, CTCI Ch 8)

Count monotone right/down paths from the top-left to the bottom-right cell of a grid
where 1 marks an obstacle and 0 a free cell. A rolling 1D DP keeps, for the current
row, the number of ways to reach each column:
ways[c] = ways[c] (from above) + ways[c-1] (from the left), zeroed on obstacles.
O(m*n) time, O(n) space.

Boundaries (documented): an empty grid or an empty first row -> 0; an obstacle on the
start or the end cell -> 0 (no path exists); 1 x n and n x 1 grids have exactly one
monotone path, so the answer is 1 iff no cell is blocked.

Relation to task 266 (documented): on identical grids this counter and task 266's
unique-path DP return the same value; on obstacle-free grids both equal the closed form
C(m+n-2, m-1), which is asserted below for several sizes.

count_ways_brute enumerates the paths recursively and is used only for cross-checks.
"""

import math


def count_ways(grid):
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


def count_ways_brute(grid):
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


def unique_paths_no_obstacles(m, n):
    """Task 266's closed form C(m+n-2, m-1), used here to cross-check obstacle-free grids."""
    if m <= 0 or n <= 0:
        return 0
    return math.comb(m + n - 2, m - 1)


def main():
    assert count_ways([[0, 0, 0], [0, 0, 0], [0, 0, 0]]) == 6            # classic 3x3
    assert count_ways([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == 2            # classic obstacle case
    assert count_ways([[0, 1], [0, 0]]) == 1                             # only down, then right
    assert count_ways([[1, 0], [0, 0]]) == 0                             # start blocked
    assert count_ways([[0, 0], [0, 1]]) == 0                             # end blocked
    assert count_ways([[0, 0, 0, 0, 0]]) == 1                            # 1 x n
    assert count_ways([[0], [0], [0]]) == 1                              # n x 1
    assert count_ways([[0, 1, 0]]) == 0                                  # 1 x n blocked
    assert count_ways([[0], [1], [0]]) == 0                              # n x 1 blocked
    assert count_ways([[0, 1, 0], [0, 0, 0]]) == 1
    assert count_ways([[0, 1], [1, 0]]) == 0                             # both moves blocked
    assert count_ways([]) == 0                                           # empty grid (documented)
    assert count_ways([[]]) == 0

    for m, n, expected in ((1, 1, 1), (1, 5, 1), (2, 2, 2), (3, 3, 6), (3, 7, 28), (4, 4, 20)):
        free = [[0] * n for _ in range(m)]
        assert count_ways(free) == expected == unique_paths_no_obstacles(m, n)

    import random

    rng = random.Random(265)
    for _ in range(300):                            # cross-check vs recursive enumeration
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        grid = [[1 if rng.random() < 0.3 else 0 for _ in range(n)] for _ in range(m)]
        assert count_ways(grid) == count_ways_brute(grid)


if __name__ == "__main__":
    main()
    print("PASS")
