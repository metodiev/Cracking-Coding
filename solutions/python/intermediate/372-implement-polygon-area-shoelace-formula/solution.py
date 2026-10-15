"""Task 372 - Polygon area via the shoelace formula (+ orientation). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics)

The shoelace (Gauss) formula sums the cross products x_i*y_{i+1} - x_{i+1}*y_i over
the edges of the closed polygon.  Half of that sum is the SIGNED area: positive when
the vertices are listed counter-clockwise, negative when clockwise, and its absolute
value is the area for any *simple* polygon.  Because the vertices are integers the
doubled area is an exact integer and the area is an exact multiple of 0.5, so all
comparisons below are exact (no tolerance) except for the deliberately tolerant
rasterization cross-check of a non-axis-aligned triangle.

Self-intersecting / self-touching (figure-8, "bowtie") polygons: the formula is not
an area at all.  It returns the sum of the lobe areas weighted by their traversal
orientation, so a bowtie whose two lobes cancel reports 0 (or the signed difference
for unequal lobes).  `abs()` does not repair that - only simple polygons are
guaranteed to yield the true area.  Two bowties below document this behaviour.

Independent oracle: `rasterized_area` counts unit cells (or finer) of the bounding
box whose centres lie inside the polygon via ray casting.  For polygons whose edges
are axis-parallel with integer coordinates every cell lies entirely inside or
outside, so the count is EXACT and asserted equal to |shoelace area|.  For a
general triangle the comparison uses a documented tolerance (cells crossed by the
boundary contribute at most ~perimeter/samples_per_unit).
"""


def signed_double_area(points):
    """Twice the signed area (an exact integer for integer vertices)."""
    total = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total


def polygon_area(points):
    """Signed area; exact (a multiple of 0.5) because vertices are integers."""
    return signed_double_area(points) / 2.0


def orientation(points):
    """+1 counter-clockwise, -1 clockwise, 0 degenerate (all collinear)."""
    double_area = signed_double_area(points)
    return (double_area > 0) - (double_area < 0)


def point_in_polygon(x, y, points):
    """Ray casting; sample points used below never lie on an edge."""
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_cross > x:
                inside = not inside
    return inside


def rasterized_area(points, samples_per_unit=1):
    """Count cell centres inside the polygon; exact for integer axis-aligned edges."""
    if len(points) < 3:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = samples_per_unit
    epsilon = 1e-6                        # nudges samples off exact integer/diagonal lines
    inside = 0
    i = min(xs) * scale
    while i < max(xs) * scale:
        x = (i + 0.5) / scale + epsilon
        j = min(ys) * scale
        while j < max(ys) * scale:
            y = (j + 0.5) / scale + epsilon
            if point_in_polygon(x, y, points):
                inside += 1
            j += 1
        i += 1
    return inside / (scale * scale)


def lcg_rectangle(seed):
    seed = (seed * 75 + 74) % 65537
    width = 1 + seed % 6
    seed = (seed * 75 + 74) % 65537
    height = 1 + seed % 6
    return [(0, 0), (width, 0), (width, height), (0, height)], seed


def lcg_l_shape(seed):
    seed = (seed * 75 + 74) % 65537
    a = 2 + seed % 7
    seed = (seed * 75 + 74) % 65537
    b = 1 + seed % 4
    seed = (seed * 75 + 74) % 65537
    c = 1 + seed % (a - 1)
    seed = (seed * 75 + 74) % 65537
    d = b + 1 + seed % 5
    shape = [(0, 0), (a, 0), (a, b), (c, b), (c, d), (0, d)]
    return shape, a * b + c * (d - b), seed


def main():
    # --- hand-checked polygons -------------------------------------------
    square = [(0, 0), (2, 0), (2, 2), (0, 2)]
    assert signed_double_area(square) == 8
    assert polygon_area(square) == 4.0
    assert orientation(square) == 1
    assert rasterized_area(square) == 4.0                          # exact oracle

    small_triangle = [(0, 0), (1, 0), (0, 1)]
    assert polygon_area(small_triangle) == 0.5
    assert orientation(small_triangle) == 1

    big_triangle = [(0, 0), (2, 0), (0, 2)]
    assert polygon_area(big_triangle) == 2.0
    assert orientation(big_triangle) == 1

    clockwise = [(0, 0), (0, 2), (2, 2), (2, 0)]
    assert signed_double_area(clockwise) == -8
    assert polygon_area(clockwise) == -4.0
    assert abs(polygon_area(clockwise)) == 4.0
    assert orientation(clockwise) == -1

    l_shape = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    assert polygon_area(l_shape) == 5.0
    assert rasterized_area(l_shape) == 5.0                         # exact oracle

    # general triangle with diagonal edges: rasterized within the documented tolerance
    general = [(0, 0), (5, 1), (2, 4)]
    assert polygon_area(general) == 9.0
    estimate = rasterized_area(general, 32)
    assert abs(estimate - 9.0) <= 0.5, estimate

    # triplets of collinear vertices do not change the area
    collinear_extra = [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]
    assert polygon_area(collinear_extra) == 4.0
    assert orientation([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]) == 1

    # --- figure-8 (bowtie) documented behaviour --------------------------
    # equal lobes, opposite orientation: lobe areas +1 and -1 cancel to 0
    bowtie = [(0, 0), (2, 2), (2, 0), (0, 2)]
    assert signed_double_area(bowtie) == 0
    assert polygon_area(bowtie) == 0.0
    # unequal lobes: 8/3 (right lobe) - 2/3 (left lobe) = 2, signed -2 here
    skewed_bowtie = [(0, 0), (2, 4), (2, 0), (0, 2)]
    assert signed_double_area(skewed_bowtie) == -4
    assert polygon_area(skewed_bowtie) == -2.0

    # --- LCG cross-checks against the exact rasterization oracle ---------
    seed = 37211
    for _ in range(30):
        rectangle, seed = lcg_rectangle(seed)
        width, height = max(p[0] for p in rectangle), max(p[1] for p in rectangle)
        assert polygon_area(rectangle) == float(width * height)
        assert rasterized_area(rectangle) == float(width * height)
        reversed_rectangle = rectangle[::-1]
        assert polygon_area(reversed_rectangle) == -polygon_area(rectangle)
        assert orientation(reversed_rectangle) == -orientation(rectangle)

    for _ in range(30):
        shape, expected, seed = lcg_l_shape(seed)
        assert signed_double_area(shape) == 2 * expected
        assert polygon_area(shape) == float(expected)
        assert rasterized_area(shape) == float(expected)
        assert orientation(shape) == 1


if __name__ == "__main__":
    main()
    print("PASS")
