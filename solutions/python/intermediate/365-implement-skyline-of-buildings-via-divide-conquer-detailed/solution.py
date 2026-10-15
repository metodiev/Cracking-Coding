"""Task 365 - Skyline of buildings via divide & conquer (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Buildings are [left, right, height] with positive width and height; a building covers
[left, right), so its right edge contributes nothing to its own height.  The detailed
divide & conquer: sort buildings by left edge, split the list in half, recursively
return each half's skyline as a list of critical points [x, height] (strictly increasing
x, every height change recorded, last height 0), then merge the two skylines with a
single two-pointer sweep.  The merge invariant is that h_left / h_right always hold the
heights of the halves at the current x; whenever the running maximum differs from the
last emitted height a new critical point is emitted, which both drops redundant points
and keeps x ordering.  Asserted on the canonical example and cross-checked against an
independent event-sweep oracle on deterministic LCG inputs.
"""


def skyline_divide_conquer(buildings):
    """Returns critical points [x, height]; degenerate buildings are ignored."""
    ordered = sorted(b for b in buildings if b[1] > b[0] and b[2] > 0)
    return _solve(ordered)


def _solve(ordered):
    if not ordered:
        return []
    if len(ordered) == 1:
        left, right, height = ordered[0]
        return [[left, height], [right, 0]]
    mid = len(ordered) // 2
    return _merge(_solve(ordered[:mid]), _solve(ordered[mid:]))


def _merge(sky_a, sky_b):
    """Linear merge of two x-sorted skylines of [x, height] critical points."""
    merged = []
    i = j = 0
    height_a = height_b = 0
    while i < len(sky_a) or j < len(sky_b):
        if j >= len(sky_b) or (i < len(sky_a) and sky_a[i][0] < sky_b[j][0]):
            x, height_a = sky_a[i]
            i += 1
        elif i >= len(sky_a) or sky_b[j][0] < sky_a[i][0]:
            x, height_b = sky_b[j]
            j += 1
        else:
            x = sky_a[i][0]
            height_a = sky_a[i][1]
            height_b = sky_b[j][1]
            i += 1
            j += 1
        height = height_a if height_a > height_b else height_b
        if not merged or merged[-1][1] != height:
            merged.append([x, height])
    return merged


def skyline_sweep(buildings):
    """Independent oracle: sweep every boundary x keeping height counts active on [l, r)."""
    events = {}
    for left, right, height in buildings:
        if right <= left or height <= 0:
            continue
        events.setdefault(left, []).append((height, 1))
        events.setdefault(right, []).append((height, -1))
    counts = {}
    result = []
    for x in sorted(events):
        for height, delta in events[x]:
            counts[height] = counts.get(height, 0) + delta
        current = max((h for h, c in counts.items() if c > 0), default=0)
        if not result or result[-1][1] != current:
            result.append([x, current])
    return result


def lcg_buildings(count, seed):
    buildings = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        left = seed % 50
        seed = (seed * 75 + 74) % 65537
        width = 1 + seed % 12
        seed = (seed * 75 + 74) % 65537
        height = 1 + seed % 15
        buildings.append([left, left + width, height])
    return buildings, seed


CANONICAL = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]
CANONICAL_SKYLINE = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]]


def main():
    # Canonical example: at x=7 the 15-block ends, 12 remains; at x=9 the 10-block ends
    # but 12 still stands, so no point is emitted there.
    assert skyline_divide_conquer(CANONICAL) == CANONICAL_SKYLINE
    assert skyline_sweep(CANONICAL) == CANONICAL_SKYLINE

    assert skyline_divide_conquer([]) == []
    assert skyline_sweep([]) == []
    assert skyline_divide_conquer([[1, 3, 4]]) == [[1, 4], [3, 0]]
    assert skyline_sweep([[1, 3, 4]]) == [[1, 4], [3, 0]]
    assert skyline_divide_conquer([[0, 2, 5], [2, 4, 7]]) == [[0, 5], [2, 7], [4, 0]]
    assert skyline_divide_conquer([[0, 2, 5], [5, 8, 3]]) == [[0, 5], [2, 0], [5, 3], [8, 0]]
    assert skyline_divide_conquer([[0, 10, 5], [2, 4, 3]]) == [[0, 5], [10, 0]]
    assert skyline_divide_conquer([[0, 10, 5], [2, 4, 9]]) == [[0, 5], [2, 9], [4, 5], [10, 0]]
    assert skyline_divide_conquer([[0, 4, 3], [0, 4, 5]]) == [[0, 5], [4, 0]]
    assert skyline_divide_conquer([[0, 4, 5], [0, 4, 5], [1, 2, 1]]) == [[0, 5], [4, 0]]

    # Degenerate buildings (zero width or height) are ignored by both implementations.
    assert skyline_divide_conquer([[3, 3, 9]]) == []
    assert skyline_divide_conquer([[0, 5, 0]]) == []
    assert skyline_divide_conquer([[0, 5, 4], [2, 2, 9]]) == [[0, 4], [5, 0]]
    assert skyline_sweep([[0, 5, 4], [2, 2, 9]]) == [[0, 4], [5, 0]]

    # LCG cross-check with many building sets; both algorithms must agree exactly.
    seed = 8113
    for trial in range(400):
        seed = (seed * 75 + 74) % 65537
        count = 1 + seed % 10
        buildings, seed = lcg_buildings(count, seed)
        expected = skyline_sweep(buildings)
        assert skyline_divide_conquer(buildings) == expected, (trial, buildings)


if __name__ == "__main__":
    main()
    print("PASS")
