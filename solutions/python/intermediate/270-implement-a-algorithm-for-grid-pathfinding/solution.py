"""Task 270 - Implement A* algorithm for grid pathfinding. (Tier: Intermediate, CTCI Ch 4)

A* on a 4-neighbour grid with unit move cost and the Manhattan heuristic, which is
admissible (a single move changes the Manhattan distance to the goal by at most 1, so it
never overestimates the true remaining cost) and consistent. grid[r][c] == 1 is a wall;
the result is the minimum-move path INCLUDING both endpoints, or None when no path
exists (also when start or goal lies on a wall).

Tie-break (documented, deterministic): the open set is ordered by (f, h, row-major cell
id r*cols+c). With equal f the cell closer to the goal (smaller h) wins, then the
smaller row-major index, so the returned path is fully deterministic.

The cost is asserted against a plain BFS oracle (astar_cost == bfs_cost) and the path is
checked to be a valid 4-connected route; bfs_distances_from additionally verifies
admissibility (Manhattan <= exact distance) on every reachable cell.
"""

import heapq
from collections import deque


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    def free(cell):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0

    if not free(start) or not free(goal):
        return None
    if start == goal:
        return [start]
    h_start = manhattan(start, goal)
    open_heap = [(h_start, h_start, start[0] * cols + start[1], start)]   # (f, h, id, cell)
    g = {start: 0}
    parent = {}
    closed = set()
    while open_heap:
        _, _, _, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        closed.add(cell)
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        r, c = cell
        for nxt in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not free(nxt) or nxt in closed:
                continue
            tentative = g[cell] + 1
            if tentative < g.get(nxt, float("inf")):
                g[nxt] = tentative
                parent[nxt] = cell
                hn = manhattan(nxt, goal)
                heapq.heappush(open_heap, (tentative + hn, hn, nxt[0] * cols + nxt[1], nxt))
    return None


def astar_cost(grid, start, goal):
    path = astar(grid, start, goal)
    return None if path is None else len(path) - 1


def bfs_distances_from(grid, goal):
    """Exact distance from every free cell to goal (-1 when unreachable/walled)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    dist = [[-1] * cols for _ in range(rows)]
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols and grid[goal[0]][goal[1]] == 0):
        return dist
    dist[goal[0]][goal[1]] = 0
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist


def bfs_cost(grid, start, goal):
    dist = bfs_distances_from(grid, goal)
    if not dist or not dist[0]:
        return -1
    if not (0 <= start[0] < len(dist) and 0 <= start[1] < len(dist[0])):
        return -1
    return dist[start[0]][start[1]]


def valid_path(grid, path, start, goal):
    if path is None or path[0] != start or path[-1] != goal:
        return False
    for r, c in path:
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0):
            return False
    for i in range(1, len(path)):
        if abs(path[i][0] - path[i - 1][0]) + abs(path[i][1] - path[i - 1][1]) != 1:
            return False
    return True


def main():
    serpentine = [[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]]
    detour = [[0, 0, 0, 0],
              [1, 1, 1, 0],
              [0, 0, 0, 0]]
    blocked = [[0, 1, 0],
               [0, 1, 0],
               [1, 1, 0]]
    open_grid = [[0] * 6 for _ in range(4)]

    assert astar_cost(serpentine, (0, 0), (4, 4)) == 8 == bfs_cost(serpentine, (0, 0), (4, 4))
    assert astar_cost(detour, (0, 0), (2, 0)) == 8 == bfs_cost(detour, (0, 0), (2, 0))   # vs Manhattan 2
    assert astar_cost(detour, (2, 0), (0, 0)) == 8 == bfs_cost(detour, (2, 0), (0, 0))
    assert astar_cost(open_grid, (0, 0), (3, 5)) == 8 == manhattan((0, 0), (3, 5))       # open field
    assert astar_cost(blocked, (0, 0), (0, 2)) is None and bfs_cost(blocked, (0, 0), (0, 2)) == -1

    assert astar(serpentine, (2, 2), (2, 2)) == [(2, 2)]                  # start == goal
    assert astar(serpentine, (1, 1), (0, 0)) is None                      # start on a wall
    assert astar(serpentine, (0, 0), (1, 1)) is None                      # goal on a wall

    for grid, start, goal in ((serpentine, (0, 0), (4, 4)), (detour, (0, 0), (2, 0)),
                              (detour, (2, 0), (0, 0)), (open_grid, (0, 0), (3, 5)),
                              (serpentine, (2, 2), (2, 2))):
        path = astar(grid, start, goal)
        assert valid_path(grid, path, start, goal), (grid, start, goal)
        if grid is open_grid:                      # on an open grid Manhattan is exact
            dist = bfs_distances_from(grid, goal)
            for r in range(len(grid)):
                for c in range(len(grid[0])):
                    assert manhattan((r, c), goal) == dist[r][c]

    for grid, goal in ((serpentine, (4, 4)), (detour, (2, 0)), (blocked, (0, 2))):
        dist = bfs_distances_from(grid, goal)
        for r in range(len(grid)):                 # admissibility on every reachable cell
            for c in range(len(grid[0])):
                if dist[r][c] >= 0:
                    assert manhattan((r, c), goal) <= dist[r][c]

    import random

    rng = random.Random(270)
    for _ in range(300):                           # A* cost == BFS cost + valid path
        rows, cols = rng.randint(2, 6), rng.randint(2, 6)
        grid = [[1 if rng.random() < 0.3 else 0 for _ in range(cols)] for _ in range(rows)]
        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        got = astar_cost(grid, start, goal)
        want = bfs_cost(grid, start, goal)
        assert (got if got is not None else -1) == want, (grid, start, goal)
        if got is not None:
            assert valid_path(grid, astar(grid, start, goal), start, goal)
            if want > 0:
                dist = bfs_distances_from(grid, goal)
                assert manhattan(start, goal) <= dist[start[0]][start[1]]


if __name__ == "__main__":
    main()
    print("PASS")
