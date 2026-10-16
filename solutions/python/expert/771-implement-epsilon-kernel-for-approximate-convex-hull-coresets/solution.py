"""Task 771 - Implement epsilon-kernel for approximate convex hull (coresets).  (Tier: Expert, Beyond - Numerical & Scientific Computing)

An eps-kernel of a point set P (with diameter R) is a small subset K of P whose convex hull
approximates the hull of P in *every* direction:
    for every unit vector u:   h_P(u) - eps * R  <=  h_K(u)  <=  h_P(u),
where h(u) = max_{p in K} <u, p> is the support function.  Equivalently the directed Hausdorff
distance from conv(P) to conv(K) is at most eps * R, and the hull of K can be used in place of
the hull of P wherever only support queries matter.

Implementation (grid/rounding kernel, Agarwal-Yu style but with the simplest grid):
  1. take the axis-aligned bounding box, R = its diagonal (an upper bound on the diameter);
  2. use a grid of side delta = eps * R / (2 * sqrt(d));
  3. keep one representative point per occupied cell.
For the cell of every p, the representative q of that cell satisfies ||q - p||_2 <= delta*sqrt(d)
= eps*R/2, hence <u, q> >= <u, p> - eps*R/2 for unit u.  Applying this to the maximiser p* of a
direction gives the kernel guarantee; the hidden constant is the documented one.

Simplified versus the literature: the cells are aligned to the bounding box (no shifted grids for
the constant-factor improvement), the representative is the first point that landed in the cell,
and the size bound is the naive one, (2*sqrt(d)/eps + 2)^d cells.  What is *not* simplified is the
guarantee: it is checked numerically by enumerating dense direction sets in 1-D, 2-D and 3-D,
against an exact support-function oracle, plus the exact 2-D convex hull (monotone chain) for the
implied Hausdorff and area bound and the size/monotonicity of the coreset.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def random_points(count, dims, seed):
    """Points with coordinates in [0, 1); a deterministic stream of 30-bit fractions."""
    points = []
    for _ in range(count):
        coords = []
        for _ in range(dims):
            seed, r = splitmix64(seed)
            coords.append((r % (1 << 30)) / float(1 << 30))
        points.append(tuple(coords))
    return points, seed


def dot(u, p):
    return sum(a * b for a, b in zip(u, p))


def support(points, direction):
    """Exact support function h_P(u) computed by brute force over all points."""
    if not points:
        raise ValueError("support function of an empty set is undefined")
    return max(dot(direction, p) for p in points)


def bounding_diagonal(points):
    dims = len(points[0])
    extents = [max(p[i] for p in points) - min(p[i] for p in points) for i in range(dims)]
    return math.sqrt(sum(e * e for e in extents))


def grid_kernel(points, eps):
    """Return (kernel_points, chosen_indices); eps <= 0 keeps every point."""
    if len(points) <= 1 or eps <= 0:
        return list(points), list(range(len(points)))
    dims = len(points[0])
    diagonal = bounding_diagonal(points)
    if diagonal == 0:
        return [points[0]], [0]
    side = eps * diagonal / (2.0 * math.sqrt(dims))
    low = [min(p[i] for p in points) for i in range(dims)]
    cells = {}
    for index, p in enumerate(points):
        key = tuple(int((p[i] - low[i]) / side) for i in range(dims))
        if key not in cells:
            cells[key] = index
    indices = sorted(cells.values())
    return [points[i] for i in indices], indices


def unit_directions_2d(count):
    return [(math.cos(2 * math.pi * i / count), math.sin(2 * math.pi * i / count))
            for i in range(count)]


def unit_directions_nd(count, dims, seed):
    """Deterministic pseudo-random unit vectors (normalised 30-bit fractions)."""
    directions = []
    for _ in range(count):
        coords = []
        for _ in range(dims):
            seed, r = splitmix64(seed)
            coords.append((r % (1 << 30)) / float(1 << 30) - 0.5)
        norm = math.sqrt(sum(c * c for c in coords))
        if norm == 0:
            continue
        directions.append(tuple(c / norm for c in coords))
    return directions, seed


def convex_hull_2d(points):
    """Monotone chain; returns hull vertices counter-clockwise without collinear points."""
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def hull_area_2d(points):
    hull = convex_hull_2d(points)
    if len(hull) < 3:
        return 0.0
    total = 0.0
    for i, p in enumerate(hull):
        q = hull[(i + 1) % len(hull)]
        total += p[0] * q[1] - q[0] * p[1]
    return abs(total) / 2.0


def hull_perimeter_2d(points):
    hull = convex_hull_2d(points)
    if len(hull) < 2:
        return 0.0
    if len(hull) == 2:
        return 2.0 * math.dist(hull[0], hull[1])
    total = 0.0
    for i, p in enumerate(hull):
        total += math.dist(p, hull[(i + 1) % len(hull)])
    return total


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def distance_to_hull(p, hull_points):
    """Distance from p to conv(hull_points) for a 2-D set (0 when p is inside)."""
    if not hull_points:
        raise ValueError("no hull points")
    if len(hull_points) == 1:
        return math.dist(p, hull_points[0])
    if len(hull_points) == 2:
        return point_segment_distance(p, hull_points[0], hull_points[1])
    inside = True
    for i, a in enumerate(hull_points):
        b = hull_points[(i + 1) % len(hull_points)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0:
            inside = False
            break
    if inside:
        return 0.0
    return min(point_segment_distance(p, hull_points[i], hull_points[(i + 1) % len(hull_points)])
               for i in range(len(hull_points)))


def main():
    # 1. the exact hull oracle itself
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5), (0.2, 0.8)]
    hull = convex_hull_2d(square)
    check(len(hull) == 4, "the square hull must have four vertices")
    check(abs(hull_area_2d(square) - 1.0) < 1e-12, "the unit square hull area must be one")
    check(abs(hull_perimeter_2d(square) - 4.0) < 1e-12, "the unit square perimeter must be four")
    check(len(convex_hull_2d([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])) == 2, "collinear points give a segment")

    # 2. boundary: empty, single point, identical points, eps <= 0
    check(grid_kernel([], 0.1) == ([], []), "an empty set has an empty kernel")
    single = [(0.3, 0.7)]
    check(grid_kernel(single, 0.1)[1] == [0], "a single point is its own kernel")
    identical = [(0.5, 0.5)] * 6
    check(len(grid_kernel(identical, 0.01)[0]) == 1, "identical points collapse to one representative")
    check(grid_kernel([(0.1, 0.1), (0.9, 0.9)], 0.0)[1] == [0, 1], "eps = 0 keeps every point")

    # 3. the kernel is a subset of the input (a coreset, not a synthetic point set)
    seed = 0x5EED
    points, seed = random_points(120, 2, seed)
    for eps in (0.5, 0.2, 0.1):
        kernel, indices = grid_kernel(points, eps)
        check(len(set(indices)) == len(indices), "kernel indices must be distinct")
        check(all(points[i] == p for i, p in zip(indices, kernel)), "kernel points must come from P")
        check(len(kernel) <= len(points), "a kernel cannot be larger than its input")

    # 4. the guarantee in 2-D: support error <= eps * R over 720 directions
    directions = unit_directions_2d(720)
    worst_ratio = 0.0
    guarantee_checks = 0
    for eps in (0.5, 0.2, 0.1, 0.05):
        for trial in range(3):
            points, seed = random_points(90 + 10 * trial, 2, seed)
            radius = bounding_diagonal(points)
            kernel, _ = grid_kernel(points, eps)
            for u in directions:
                gap = support(points, u) - support(kernel, u)
                check(gap >= -1e-12, "the kernel support must never exceed the exact support")
                check(gap <= eps * radius + 1e-12,
                      f"support gap {gap} exceeds eps*R = {eps * radius}")
                worst_ratio = max(worst_ratio, gap / (eps * radius) if eps * radius else 0.0)
                guarantee_checks += 1

    # 5. the guarantee in 3-D with random unit directions
    for eps in (0.4, 0.15):
        points, seed = random_points(140, 3, seed)
        radius = bounding_diagonal(points)
        kernel, _ = grid_kernel(points, eps)
        random_dirs, seed = unit_directions_nd(150, 3, seed)
        for u in random_dirs:
            gap = support(points, u) - support(kernel, u)
            check(gap <= eps * radius + 1e-12, "the 3-D support gap must respect eps*R")

    # 6. one dimension: the only directions are +1 and -1
    line, seed = random_points(80, 1, seed)
    radius = bounding_diagonal(line)
    kernel, _ = grid_kernel(line, 0.25)
    for u in ((1.0,), (-1.0,)):
        gap = support(line, u) - support(kernel, u)
        check(gap <= 0.25 * radius + 1e-12, "the 1-D support gap must respect eps*R")
    check(min(kernel)[0] - min(line)[0] <= 0.25 * radius + 1e-12, "the 1-D kernel nearly keeps the min")
    check(max(line)[0] - max(kernel)[0] <= 0.25 * radius + 1e-12, "the 1-D kernel nearly keeps the max")

    # 7. degenerate 2-D input: collinear points still yield a valid kernel
    collinear = [(i * 0.01, 2.0 * i * 0.01) for i in range(40)]
    radius = bounding_diagonal(collinear)
    kernel, _ = grid_kernel(collinear, 0.2)
    for u in directions[:180]:
        gap = support(collinear, u) - support(kernel, u)
        check(gap <= 0.2 * radius + 1e-12, "a degenerate input must still respect the bound")
    check(hull_area_2d(collinear) == 0.0 and hull_area_2d(kernel) == 0.0,
          "a collinear input has a zero-area hull, hence so has its kernel")

    # 8. implied Hausdorff bound: every hull vertex of P is within eps*R of conv(K)
    points, seed = random_points(150, 2, seed)
    radius = bounding_diagonal(points)
    eps = 0.1
    kernel, _ = grid_kernel(points, eps)
    kernel_hull = convex_hull_2d(kernel)
    worst_hausdorff = 0.0
    for vertex in convex_hull_2d(points):
        worst_hausdorff = max(worst_hausdorff, distance_to_hull(vertex, kernel_hull))
    check(worst_hausdorff <= eps * radius + 1e-12,
          "the directed Hausdorff distance from conv(P) to conv(K) must respect eps*R")
    check(distance_to_hull((0.0, 0.0), kernel_hull) >= 0.0, "the distance helper must behave")

    # 9. implied area bound: conv(K) loses at most perimeter * t + pi * t^2 of area, t = eps*R/2
    t = eps * radius / 2.0
    area_gap = hull_area_2d(points) - hull_area_2d(kernel)
    area_budget = hull_perimeter_2d(points) * t + math.pi * t * t
    check(area_gap <= area_budget, f"area gap {area_gap} exceeds the budget {area_budget}")

    # 10. size bound and coarsening: doubling eps (a grid refinement of exactly 2) cannot add cells
    for eps in (0.5, 0.2, 0.1):
        kernel, _ = grid_kernel(points, eps)
        bound = (2.0 * math.sqrt(2) / eps + 2.0) ** 2
        check(len(kernel) <= bound, f"kernel size {len(kernel)} exceeds the {bound} cell bound")
        coarse, _ = grid_kernel(points, 2 * eps)
        check(len(coarse) <= len(kernel), "a coarser grid must not increase the kernel size")
    huge, indices = grid_kernel(points, 10.0)
    check(len(huge) == 1, "an eps far above the diameter collapses the kernel to one point")

    # 11. determinism
    first = grid_kernel(points, 0.1)
    second = grid_kernel(points, 0.1)
    check(first == second, "the kernel must be deterministic")

    print(f"771 directions={guarantee_checks} worst_gap/(eps*R)={worst_ratio:.3f} "
          f"hausdorff={(worst_hausdorff / radius):.4f}*R")
    print(f"771 n={len(points)} kernel={len(kernel)} eps={eps} area_gap={area_gap:.5f}")
    print("PASS")


if __name__ == "__main__":
    main()
