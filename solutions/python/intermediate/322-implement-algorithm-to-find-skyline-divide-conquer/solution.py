"""Task 322 - Implement algorithm to find skyline (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Split the buildings in half, compute each half's skyline recursively, then merge the
two skylines in a single linear pass that walks both point lists and records every
height change.  A building is active on [left, right), so a skyline is a list of
[x, height] critical points where the height changes (height 0 = nothing above x).
Results are cross-checked against an independent event-sweep implementation.
"""


def skyline_divide_conquer(buildings):
    """buildings: list of [left, right, height]; returns critical points as [x, height]."""
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


def skyline_sweep(buildings):
    """Independent oracle: sweep all boundaries keeping a count of active heights."""
    if not buildings:
        return []
    max_height = max(height for _, _, height in buildings)
    starts = {}
    ends = {}
    for left, right, height in buildings:
        starts.setdefault(left, []).append(height)
        ends.setdefault(right, []).append(height)
    counts = [0] * (max_height + 1)
    current = 0
    result = []
    for x in sorted(set(starts) | set(ends)):
        for height in ends.get(x, ()):          # remove first: active on [left, right)
            counts[height] -= 1
        for height in starts.get(x, ()):
            counts[height] += 1
            if height > current:
                current = height
        while current > 0 and counts[current] == 0:
            current -= 1
        if not result or result[-1][1] != current:
            result.append([x, current])
    return result


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
    assert skyline_divide_conquer(CANONICAL) == CANONICAL_SKYLINE
    assert skyline_sweep(CANONICAL) == CANONICAL_SKYLINE

    assert skyline_divide_conquer([]) == []
    assert skyline_sweep([]) == []
    assert skyline_divide_conquer([[1, 3, 4]]) == [[1, 4], [3, 0]]
    assert skyline_divide_conquer([[0, 2, 5], [2, 4, 7]]) == [[0, 5], [2, 7], [4, 0]]
    assert skyline_divide_conquer([[0, 2, 5], [5, 8, 3]]) == [[0, 5], [2, 0], [5, 3], [8, 0]]
    assert skyline_sweep([[0, 2, 5], [5, 8, 3]]) == [[0, 5], [2, 0], [5, 3], [8, 0]]

    seed = 1123
    for trial in range(300):
        count = 1 + seed % 8
        buildings, seed = lcg_buildings(count, seed)
        expected = skyline_sweep(buildings)
        assert skyline_divide_conquer(buildings) == expected, buildings


if __name__ == "__main__":
    main()
    print("PASS")
