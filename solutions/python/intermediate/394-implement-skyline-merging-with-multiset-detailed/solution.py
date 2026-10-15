"""Task 394 - Implement skyline merging with a multiset (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

Two layers of the same sweep. (1) skyline_of_rectangles: x-events add/remove heights from a
counted multiset; after each event x the multiset maximum decides whether the skyline height
changes. (2) merge_skylines: every consecutive point pair of a skyline becomes one segment,
which is exactly the same event shape, so merging k skylines is one more sweep over all
their events. The two functions must agree when a rectangle set is split in two.
"""

def _sweep(events):
    """events: list of (x, delta_height); returns skyline points [[x, h], ...]."""
    counts = {}
    order = sorted({x for x, _ in events})
    by_x = {}
    for x, delta in events:
        by_x.setdefault(x, []).append(delta)

    points = []
    current = 0
    for x in order:
        for delta in by_x[x]:
            if delta >= 0:
                counts[delta] = counts.get(delta, 0) + 1
            else:
                height = -delta
                counts[height] -= 1
                if counts[height] == 0:
                    del counts[height]
        top = max(counts) if counts else 0
        if top != current:
            points.append([x, top])
            current = top
    return points


def skyline_of_rectangles(rectangles):
    events = []
    for x1, x2, height in rectangles:
        if x1 < x2 and height > 0:
            events.append((x1, height))
            events.append((x2, -height))
    return _sweep(events)


def merge_skylines(skylines):
    events = []
    for skyline in skylines:
        for i in range(len(skyline) - 1):
            x1, height = skyline[i]
            x2 = skyline[i + 1][0]
            if height > 0 and x1 < x2:
                events.append((x1, height))
                events.append((x2, -height))
    return _sweep(events)


def lcg_rectangles(count, seed, span=20):
    rectangles = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        x1 = seed % span
        seed = (seed * 75 + 74) % 65537
        length = 1 + seed % 6
        seed = (seed * 75 + 74) % 65537
        height = 1 + seed % 10
        rectangles.append((x1, x1 + length, height))
    return rectangles


def main():
    canonical = [(2, 9, 10), (3, 7, 15), (5, 12, 12), (15, 20, 10), (19, 24, 8)]
    expected = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]]
    assert skyline_of_rectangles(canonical) == expected

    # merging two skylines overlays their segments: A covers [0,4) with h=3,
    # B covers [1,2) h=1 and [2,5) h=4, so the max is 3, then 4 from x=2, then 0
    left = [[0, 3], [4, 0]]
    right = [[1, 1], [2, 4], [5, 0]]
    assert merge_skylines([left, right]) == [[0, 3], [2, 4], [5, 0]]

    # laying a skyline over itself changes nothing
    assert merge_skylines([expected, expected]) == expected

    # touching rectangles: the height drops and rises at the same x, so no point is emitted
    assert skyline_of_rectangles([(0, 2, 3), (2, 4, 3)]) == [[0, 3], [4, 0]]
    # nested rectangles hide the inner one
    assert skyline_of_rectangles([(0, 10, 5), (2, 5, 3)]) == [[0, 5], [10, 0]]
    assert skyline_of_rectangles([]) == []

    # splitting the rectangle set in any way and merging the parts must agree
    seed = 777
    for _ in range(40):
        rectangles = lcg_rectangles(9, seed)
        seed = (seed * 75 + 74) % 65537
        cut = seed % (len(rectangles) + 1)
        first, second = rectangles[:cut], rectangles[cut:]
        combined = skyline_of_rectangles(rectangles)
        merged = merge_skylines([skyline_of_rectangles(first), skyline_of_rectangles(second)])
        assert merged == combined, (rectangles, combined, merged)

    assert skyline_of_rectangles([(3, 1, 5)]) == []          # x1 > x2 is ignored
    assert skyline_of_rectangles([(0, 5, 0)]) == []          # zero height draws nothing


if __name__ == "__main__":
    main()
    print("PASS")
