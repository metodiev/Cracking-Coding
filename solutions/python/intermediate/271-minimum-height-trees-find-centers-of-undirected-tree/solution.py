"""Task 271 - Minimum height trees (tree centers). (Tier: Intermediate, CTCI Ch 4)

Peel leaves layer by layer; the 1 or 2 nodes left are the centers, i.e. the roots of the
minimum-height trees. O(n) with a degree queue.
"""

from collections import deque


def min_height_trees(n, edges):
    if n == 1:
        return [0]
    adj = [[] for _ in range(n)]
    degree = [0] * n
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        degree[a] += 1
        degree[b] += 1

    leaves = deque(i for i in range(n) if degree[i] == 1)
    remaining = n
    while remaining > 2:
        remaining -= len(leaves)
        for _ in range(len(leaves)):
            leaf = leaves.popleft()
            for neighbour in adj[leaf]:
                degree[neighbour] -= 1
                if degree[neighbour] == 1:
                    leaves.append(neighbour)
    return sorted(leaves)


def brute_force_centers(n, edges):
    """Independent oracle: BFS height from every root, keep the minimizers."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    def height(root):
        seen = {root}
        frontier = [root]
        levels = -1
        while frontier:
            levels += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
        return levels

    heights = [height(r) for r in range(n)]
    best = min(heights)
    return sorted(r for r in range(n) if heights[r] == best), best


def main():
    path6 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert min_height_trees(6, path6) == [2, 3]

    star = [(0, 1), (0, 2), (0, 3)]
    assert min_height_trees(4, star) == [0]

    assert min_height_trees(1, []) == [0]
    assert min_height_trees(2, [(0, 1)]) == [0, 1]

    sample = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6)]
    centers, best = brute_force_centers(7, sample)
    assert min_height_trees(7, sample) == centers
    assert best == 3
    assert len(min_height_trees(7, sample)) <= 2


if __name__ == "__main__":
    main()
    print("PASS")
