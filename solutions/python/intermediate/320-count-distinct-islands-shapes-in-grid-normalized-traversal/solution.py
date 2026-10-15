"""Task 320 - Count distinct island shapes in a grid (normalised traversal). (Tier: Intermediate, CTCI Ch 4)

Flood-fill each 4-connected island of 1s (iterative DFS), then normalise the shape by subtracting
the topmost row and leftmost column, so only translations are identified; the shape key is the
frozenset of relative coordinates. Rotations and reflections are NOT considered equal (only
translation), so a 1x2 bar and a 2x1 bar are two different shapes. The count is cross-checked on
deterministic LCG grids with a second, independent implementation (recursive flood fill, sorted
tuple keys).
"""


def collect_island_shapes(grid):
    """Set of shape keys (frozensets of normalised (dr, dc) cells) for 4-connected islands."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = [[False] * cols for _ in range(rows)]
    shapes = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not seen[r][c]:
                stack = [(r, c)]
                seen[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                min_r = min(rr for rr, _ in cells)
                min_c = min(cc for _, cc in cells)
                shapes.add(frozenset((rr - min_r, cc - min_c) for rr, cc in cells))
    return shapes


def count_distinct_islands(grid):
    return len(collect_island_shapes(grid))


def count_islands(grid):
    """Number of 4-connected islands of 1s (shapes merged or not)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    seen = [[False] * cols for _ in range(rows)]
    total = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not seen[r][c]:
                total += 1
                stack = [(r, c)]
                seen[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1 and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
    return total


def island_shapes_sorted_tuples(grid):
    """Independent oracle: recursive flood fill + sorted tuple shape keys."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    visited = set()

    def flood(r, c, cells):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != 1 or (r, c) in visited:
            return
        visited.add((r, c))
        cells.append((r, c))
        flood(r + 1, c, cells)
        flood(r - 1, c, cells)
        flood(r, c + 1, cells)
        flood(r, c - 1, cells)

    shapes = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in visited:
                cells = []
                flood(r, c, cells)
                min_r = min(rr for rr, _ in cells)
                min_c = min(cc for _, cc in cells)
                shapes.add(tuple(sorted((rr - min_r, cc - min_c) for rr, cc in cells)))
    return shapes


def lcg_grids(rows, cols, count, seed=3200):
    grids = []
    for _ in range(count):
        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                seed = (seed * 75 + 74) % 65537
                row.append(1 if seed % 3 == 0 else 0)
            grid.append(row)
        grids.append(grid)
    return grids


def main():
    classic = [[1, 1, 0, 0, 0],
               [1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1]]
    assert count_distinct_islands(classic) == 3
    assert collect_island_shapes(classic) == {
        frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),            # 2x2 square
        frozenset({(0, 0), (0, 1)}),                            # horizontal bar
        frozenset({(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)}),    # 1x5 line
    }

    # Rotations are NOT identified: a horizontal bar and a vertical bar are different shapes
    rotated = [[1, 1, 0],
               [0, 0, 0],
               [0, 0, 1],
               [0, 0, 1]]
    assert count_distinct_islands(rotated) == 2
    assert collect_island_shapes(rotated) == {frozenset({(0, 0), (0, 1)}), frozenset({(0, 0), (1, 0)})}

    # Identical islands (even far apart) count once
    assert count_distinct_islands([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]) == 1
    # ... but mirrored L-shapes are two shapes
    assert count_distinct_islands([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]]) == 2
    assert count_distinct_islands([[1]]) == 1
    assert count_distinct_islands([[0, 0], [0, 0]]) == 0
    assert count_distinct_islands([]) == 0
    # Diagonal cells are not 4-connected: two separate islands, but both are single cells,
    # so they still share one shape. The 3x3 grid below distinguishes the two situations:
    # 4-connectivity gives {(0,0)} and {(0,0),(0,1)} -> 2 shapes, diagonal merging only 1.
    assert count_islands([[1, 0], [0, 1]]) == 2
    assert count_distinct_islands([[1, 0], [0, 1]]) == 1        # both are single-cell islands
    assert count_distinct_islands([[1, 0, 0], [0, 1, 1], [0, 0, 0]]) == 2
    assert count_distinct_islands([[1, 1, 1, 1]]) == 1
    assert count_distinct_islands([[1], [1], [1]]) == 1

    # Zero padding on the top/left must not change the answer (shapes are translation invariant)
    padded = [[0] * 7 for _ in range(2)] + [[0, 0] + row + [0] for row in classic] + [[0] * 7 for _ in range(2)]
    assert count_distinct_islands(padded) == 3

    for grid in lcg_grids(8, 8, 40) + lcg_grids(6, 11, 20) + lcg_grids(4, 4, 20):
        shapes = collect_island_shapes(grid)
        oracle = {frozenset(shape) for shape in island_shapes_sorted_tuples(grid)}
        assert shapes == oracle, grid
        assert count_distinct_islands(grid) == len(oracle), grid
        for shape in shapes:                                    # normalised keys ...
            assert len(shape) > 0
            assert min(rr for rr, _ in shape) == 0
            assert min(cc for _, cc in shape) == 0
            # ... whose cells form a 4-connected island
            frontier = [next(iter(shape))]
            reached = {frontier[0]}
            while frontier:
                cr, cc = frontier.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (cr + dr, cc + dc) in shape and (cr + dr, cc + dc) not in reached:
                        reached.add((cr + dr, cc + dc))
                        frontier.append((cr + dr, cc + dc))
            assert reached == shape, grid


if __name__ == "__main__":
    main()
    print("PASS")
