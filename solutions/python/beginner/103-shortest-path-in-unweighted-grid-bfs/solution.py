"""Task 103 - Shortest path in unweighted grid (BFS). (Tier: Beginner, CTCI Ch 11)

BFS expands one edge at a time, so the first time the goal is reached its
distance is the minimum number of 4-directional moves.  Blocked or cut-off
cells give -1.
"""
from collections import deque


def shortest_path(grid, start, goal):
    if not grid or not grid[0]:
        return -1
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return -1
    if start == goal:
        return 0
    dist = [[-1] * cols for _ in range(rows)]
    dist[sr][sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                if (nr, nc) == goal:
                    return dist[nr][nc]
                queue.append((nr, nc))
    return -1


def main():
    open3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert shortest_path(open3, (0, 0), (2, 2)) == 4
    assert shortest_path(open3, (0, 0), (0, 0)) == 0          # start == goal
    assert shortest_path([[0]], (0, 0), (0, 0)) == 0          # single cell

    detour = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    assert shortest_path(detour, (0, 0), (2, 0)) == 6          # forced around the wall

    split = [[0, 1], [1, 0]]
    assert shortest_path(split, (0, 0), (1, 1)) == -1          # only reachable diagonally

    blocked = [[0, 1], [0, 0]]
    assert shortest_path(blocked, (0, 0), (0, 1)) == -1        # goal is an obstacle
    assert shortest_path(blocked, (0, 1), (0, 0)) == -1        # start is an obstacle

    assert shortest_path([], (0, 0), (0, 0)) == -1             # empty grid

    direct = [[0] * 4 for _ in range(4)]
    assert shortest_path(direct, (0, 0), (3, 3)) == 6          # Manhattan distance


if __name__ == "__main__":
    main()
    print("PASS")
