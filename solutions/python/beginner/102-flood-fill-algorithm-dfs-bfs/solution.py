"""Task 102 - Flood fill algorithm (DFS/BFS). (Tier: Beginner, CTCI Ch 4)

BFS from the start cell over 4-neighbors.  A fill with the color the cell
already has is a no-op, so the frontier cannot bounce back and forth forever.
"""
from collections import deque


def flood_fill(image, sr, sc, new_color):
    if not image or not image[0]:
        return image
    rows, cols = len(image), len(image[0])
    start_color = image[sr][sc]
    if start_color == new_color:
        return image
    image[sr][sc] = new_color
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and image[nr][nc] == start_color:
                image[nr][nc] = new_color
                queue.append((nr, nc))
    return image


def main():
    classic = [[1, 1, 1], [1, 1, 0], [1, 0, 1]]
    assert flood_fill(classic, 1, 1, 2) == [[2, 2, 2], [2, 2, 0], [2, 0, 1]]

    noop = [[0, 0], [0, 0]]
    assert flood_fill(noop, 0, 0, 0) == [[0, 0], [0, 0]]      # same color is a no-op

    patch = [[1, 1, 0], [0, 1, 0]]
    assert flood_fill(patch, 0, 0, 5) == [[5, 5, 0], [0, 5, 0]]

    single = [[7]]
    assert flood_fill(single, 0, 0, 9) == [[9]]

    assert flood_fill([], 0, 0, 1) == []                      # empty image

    mix = [[1, 1], [1, 0]]
    assert flood_fill(mix, 0, 0, 0) == [[0, 0], [0, 0]]       # zero is a legal color


if __name__ == "__main__":
    main()
    print("PASS")
