"""Task 101 - Count islands in boolean grid (DFS). (Tier: Beginner, CTCI Ch 4)

Flood-fills each 4-connected block of 1s with DFS over an explicit stack and
counts how many fills it takes; the input grid is left untouched.
"""


def count_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [row[:] for row in grid]
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if seen[r][c] == 1:
                islands += 1
                _flood(seen, r, c)
    return islands


def _flood(grid, sr, sc):
    rows, cols = len(grid), len(grid[0])
    stack = [(sr, sc)]
    grid[sr][sc] = 0
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 0
                stack.append((nr, nc))


def main():
    assert count_islands([]) == 0                       # empty grid
    assert count_islands([[]]) == 0                     # empty row
    assert count_islands([[0, 0], [0, 0]]) == 0         # all water
    assert count_islands([[1, 1], [1, 1]]) == 1         # all land
    assert count_islands([[1]]) == 1
    assert count_islands([[0]]) == 0
    assert count_islands([[1, 0], [0, 1]]) == 2         # diagonal is not connected
    mixed = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1],
    ]
    assert count_islands(mixed) == 5                    # multiple islands
    assert mixed[0][0] == 1                             # input not mutated


if __name__ == "__main__":
    main()
    print("PASS")
