"""Task 328 - Skyline problem (heap + sweep line). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Sweep the sorted critical x values (every left and right edge).  A building starts
being active at its left edge and stops at its right edge, so buildings live on
[left, right).  A max-heap keyed by (-height, right) with lazy deletion gives the
current height; a point [x, height] is emitted whenever that height changes.  The
divide-and-conquer skyline is implemented here as well and both must agree.
"""

import heapq


def skyline_heap(buildings):
    """Heap sweep; returns critical points as [x, height]."""
    if not buildings:
        return []
    starts = {}
    xs = set()
    for left, right, height in buildings:
        starts.setdefault(left, []).append((right, height))
        xs.add(left)
        xs.add(right)
    heap = []                                    # (-height, right)
    result = []
    for x in sorted(xs):
        for right, height in starts.get(x, ()):
            heapq.heappush(heap, (-height, right))
        while heap and heap[0][1] <= x:          # lazy deletion of finished buildings
            heapq.heappop(heap)
        height = -heap[0][0] if heap else 0
        if not result or result[-1][1] != height:
            result.append([x, height])
    return result


def skyline_divide_conquer(buildings):
    """Cross-check implementation: recursive split with a linear skyline merge."""
    if not buildings:
        return []
    if len(buildings) == 1:
        left, right, height = buildings[0]
        return [[left, height], [right, 0]]
    mid = len(buildings) // 2
    return merge_skylines(
        skyline_divide_conquer(buildings[:mid]),
        skyline_divide_conquer(buildings[mid:]),
    )


def merge_skylines(a, b):
    """Linear merge of two x-sorted skylines of [x, height] critical points."""
    merged = []
    i = j = 0
    height_a = height_b = 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and a[i][0] < b[j][0]):
            x, height_a = a[i]
            i += 1
        elif i >= len(a) or b[j][0] < a[i][0]:
            x, height_b = b[j]
            j += 1
        else:
            x, height_a = a[i]
            _, height_b = b[j]
            i += 1
            j += 1
        height = max(height_a, height_b)
        if not merged or merged[-1][1] != height:
            merged.append([x, height])
    return merged


def lcg_buildings(count, seed):
    buildings = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        left = seed % 40
        seed = (seed * 75 + 74) % 65537
        width = 1 + seed % 10
        seed = (seed * 75 + 74) % 65537
        height = 1 + seed % 20
        buildings.append([left, left + width, height])
    return buildings, seed


CANONICAL = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
CANONICAL_SKYLINE = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]]


def main():
    assert skyline_heap(CANONICAL) == CANONICAL_SKYLINE
    assert skyline_divide_conquer(CANONICAL) == CANONICAL_SKYLINE

    assert skyline_heap([]) == [] and skyline_divide_conquer([]) == []
    assert skyline_heap([[0, 2, 5], [2, 4, 7]]) == [[0, 5], [2, 7], [4, 0]]
    assert skyline_heap([[0, 2, 5], [5, 8, 3]]) == [[0, 5], [2, 0], [5, 3], [8, 0]]

    seed = 24601
    for trial in range(300):
        count = 1 + seed % 8
        buildings, seed = lcg_buildings(count, seed)
        from_heap = skyline_heap(buildings)
        from_divide_conquer = skyline_divide_conquer(buildings)
        assert from_heap == from_divide_conquer, buildings


if __name__ == "__main__":
    main()
    print("PASS")
