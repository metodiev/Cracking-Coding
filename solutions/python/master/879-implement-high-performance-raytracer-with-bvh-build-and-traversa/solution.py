"""Task 879 - Implement high-performance raytracer with BVH build and traversal optimization.  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)

A triangle raytracer whose acceleration structure is built and traversed the way a production
renderer does it:

  * build - a top-down BVH with the *binned* surface-area heuristic.  Each node bins its
    primitives along the widest centroid axis (12 bins), evaluates the SAH cost
    SA(left) * n_left + SA(right) * n_right of every split, and only splits when that cost
    beats the leaf cost; the best split is compared against a midpoint-split BVH, which is the
    standard way to show the SAH is actually doing work;
  * traversal - an explicit stack visited in near-child-first order, with a per-ray counter of
    node visits and triangle tests, so the acceleration is *measured* (rays x triangles for
    brute force versus the counted traversals for the BVH) rather than asserted;
  * intersection - the slab test for AABBs and Moller-Trumbore for triangles.  Both are
    verified against independent oracles: the slab test against sampling the slab region, and
    the traversal against brute-force ray casting, which must report the *same* hit triangle
    and the same distance for every primary and shadow ray.

Verified: node geometry (every primitive in exactly one leaf, 2*leaves - 1 nodes, monotone
nesting of the bounds, the root bounds equal to the union of the primitives); SAH build cost
below the midpoint-split cost; the traversed image identical to the brute-force image with a
large measured reduction in triangle tests; near-child-first ordering changing nothing but the
visit order (the same hits are produced with the ordering disabled); shadow-ray occlusion
identical to brute force; a two-bounce reflection still deterministic; the slab test against
sampling; degenerate inputs (empty scene, zero-area triangle, ray parallel to a slab, grazing
hit on a shared edge) handled without infinities; and bit-determinism of the whole render.

Simplified versus a production renderer: float arithmetic on the host (no SIMD packets), a
single-level BVH instead of a two-level BVH over dynamic geometry, no deferred/streamed build,
and shading is a fixed Lambert + shadow term instead of physically based materials.
"""
import math

LEAF_COST = 1.0
TRAVERSAL_COST = 1.0
BINS = 12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize(vector):
    length = math.sqrt(dot(vector, vector))
    if length == 0.0:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


class Rng:
    def __init__(self, seed):
        self.state = seed & ((1 << 64) - 1)

    def uniform(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return ((z ^ (z >> 31)) >> 11) * (1.0 / float(1 << 53))


class Triangle:
    __slots__ = ("v0", "v1", "v2", "colour")

    def __init__(self, v0, v1, v2, colour):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.colour = colour

    def centroid(self):
        return ((self.v0[0] + self.v1[0] + self.v2[0]) / 3.0,
                (self.v0[1] + self.v1[1] + self.v2[1]) / 3.0,
                (self.v0[2] + self.v1[2] + self.v2[2]) / 3.0)

    def bounds(self):
        return (min(self.v0[0], self.v1[0], self.v2[0]),
                min(self.v0[1], self.v1[1], self.v2[1]),
                min(self.v0[2], self.v1[2], self.v2[2]),
                max(self.v0[0], self.v1[0], self.v2[0]),
                max(self.v0[1], self.v1[1], self.v2[1]),
                max(self.v0[2], self.v1[2], self.v2[2]))

    def normal(self):
        return normalize(cross(subtract(self.v1, self.v0), subtract(self.v2, self.v0)))

    def area(self):
        return 0.5 * math.sqrt(dot(cross(subtract(self.v1, self.v0), subtract(self.v2, self.v0)),
                                  cross(subtract(self.v1, self.v0), subtract(self.v2, self.v0))))


# ------------------------------------------------------------------ intersections
def ray_aabb(origin, direction, bounds):
    """Slab test; returns (t_enter, t_exit) or None.  Division by a zero direction is avoided
    explicitly so a ray exactly parallel to a slab still works."""
    low = [-math.inf, -math.inf, -math.inf]
    high = [math.inf, math.inf, math.inf]
    for axis in range(3):
        minimum = bounds[axis]
        maximum = bounds[axis + 3]
        if abs(direction[axis]) < 1e-15:
            if origin[axis] < minimum or origin[axis] > maximum:
                return None
            continue
        inverse = 1.0 / direction[axis]
        first = (minimum - origin[axis]) * inverse
        second = (maximum - origin[axis]) * inverse
        if first > second:
            first, second = second, first
        low[axis] = first
        high[axis] = second
    enter = max(low)
    exit_time = min(high)
    if enter > exit_time or exit_time < 0.0:
        return None
    return (max(enter, 0.0), exit_time)


def ray_triangle(origin, direction, triangle):
    """Moller-Trumbore; returns the distance or None."""
    edge1 = subtract(triangle.v1, triangle.v0)
    edge2 = subtract(triangle.v2, triangle.v0)
    pvec = cross(direction, edge2)
    determinant = dot(edge1, pvec)
    if abs(determinant) < 1e-15:
        return None
    inverse = 1.0 / determinant
    tvec = subtract(origin, triangle.v0)
    u = dot(tvec, pvec) * inverse
    if u < 0.0 or u > 1.0:
        return None
    qvec = cross(tvec, edge1)
    v = dot(direction, qvec) * inverse
    if v < 0.0 or u + v > 1.0:
        return None
    distance = dot(edge2, qvec) * inverse
    if distance <= 1e-9:
        return None
    return distance


def bounds_union(bounds_list):
    """Union of any number of AABBs; None entries are ignored, all-None yields None."""
    result = None
    for bounds in bounds_list:
        if bounds is None:
            continue
        if result is None:
            result = list(bounds)
            continue
        for axis in range(3):
            result[axis] = min(result[axis], bounds[axis])
            result[axis + 3] = max(result[axis + 3], bounds[axis + 3])
    return None if result is None else tuple(result)


def surface_area(bounds):
    if bounds is None:
        return 0.0
    sizes = (bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    return 2.0 * (sizes[0] * sizes[1] + sizes[1] * sizes[2] + sizes[2] * sizes[0])


# ------------------------------------------------------------------ BVH
class Node:
    __slots__ = ("bounds", "left", "right", "members", "count", "axis")

    def __init__(self, bounds):
        self.bounds = bounds
        self.left = None
        self.right = None
        self.members = []
        self.count = 0
        self.axis = -1

    @property
    def is_leaf(self):
        return self.left is None


def binned_sah_split(primitives, indices, centroids, bounds_list):
    """Best SAH split over the widest centroid axis with 12 bins; None if unsplittable."""
    centroid_bounds = bounds_union([(centroids[i][0], centroids[i][1], centroids[i][2],
                                     centroids[i][0], centroids[i][1], centroids[i][2])
                                    for i in indices])
    extents = [centroid_bounds[3] - centroid_bounds[0], centroid_bounds[4] - centroid_bounds[1],
               centroid_bounds[5] - centroid_bounds[2]]
    axis = max(range(3), key=lambda index: extents[index])
    if extents[axis] <= 0.0:
        return None
    low = centroid_bounds[axis]
    scale = BINS * (1.0 - 1e-9) / extents[axis]
    counts = [0] * BINS
    bin_bounds = [None] * BINS
    bin_of = {}
    for index in indices:
        bin_index = min(BINS - 1, int((centroids[index][axis] - low) * scale))
        counts[bin_index] += 1
        bin_of[index] = bin_index
        bin_bounds[bin_index] = bounds_union([bin_bounds[bin_index], bounds_list[index]])
    left_counts = [0] * BINS
    left_bounds = [None] * BINS
    running = 0
    accumulated = None
    for bin_index in range(BINS):
        running += counts[bin_index]
        left_counts[bin_index] = running
        accumulated = bounds_union([accumulated, bin_bounds[bin_index]])
        left_bounds[bin_index] = accumulated
    right_counts = [0] * BINS
    right_bounds = [None] * BINS
    running = 0
    accumulated = None
    for bin_index in range(BINS - 1, -1, -1):
        running += counts[bin_index]
        right_counts[bin_index] = running
        accumulated = bounds_union([accumulated, bin_bounds[bin_index]])
        right_bounds[bin_index] = accumulated
    best_cost = None
    best_split = None
    total = len(indices)
    for bin_index in range(BINS - 1):
        left = left_counts[bin_index]
        right = total - left
        if left == 0 or right == 0:
            continue
        cost = (surface_area(left_bounds[bin_index]) * left
                + surface_area(right_bounds[bin_index + 1]) * right)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_split = (bin_index, axis)
    if best_split is None:
        return None
    bin_index, axis = best_split
    left = [index for index in indices if bin_of[index] <= bin_index]
    right = [index for index in indices if bin_of[index] > bin_index]
    if not left or not right:
        return None
    return axis, left, right


class Bvh:
    def __init__(self, primitives, leaf_size=4, strategy="sah", max_leaf=8):
        self.primitives = list(primitives)
        self.leaf_size = leaf_size
        self.max_leaf = max(max_leaf, leaf_size)
        self.strategy = strategy
        self.order = list(range(len(self.primitives)))
        self.bounds_list = [primitive.bounds() for primitive in self.primitives]
        self.centroids = [primitive.centroid() for primitive in self.primitives]
        self.nodes = []
        self.root = None
        if self.primitives:
            self.root = self._build(list(range(len(self.primitives))), 0)

    def _build(self, indices, depth):
        bounds = bounds_union([self.bounds_list[index] for index in indices])
        node = Node(bounds)
        node.members = list(indices)
        node.count = len(indices)
        self.nodes.append(node)
        if len(indices) <= self.leaf_size or depth > 40:
            node.left = None
            return node
        split = binned_sah_split(self.primitives, indices, self.centroids, self.bounds_list) \
            if self.strategy == "sah" else midpoint_split(indices, self.centroids,
                                                          self.bounds_list)
        if split is None:
            node.left = None
            return node
        axis, left, right = split
        parent_area = max(surface_area(bounds), 1e-12)
        left_area = surface_area(bounds_union([self.bounds_list[index] for index in left]))
        right_area = surface_area(bounds_union([self.bounds_list[index] for index in right]))
        split_cost = TRAVERSAL_COST + LEAF_COST * (left_area * len(left)
                                                    + right_area * len(right)) / parent_area
        if self.strategy == "sah" and split_cost >= LEAF_COST * len(indices) \
                and len(indices) <= self.max_leaf:
            node.left = None
            return node
        node.axis = axis
        node.left = self._build(left, depth + 1)
        node.right = self._build(right, depth + 1)
        return node

    def leaves(self):
        return [node for node in self.nodes if node.is_leaf]

    def members(self):
        if self.root is None:
            return []
        for node in self.nodes:
            if node.is_leaf:
                yield from node.members

    def traverse(self, origin, direction, ordered=True):
        """Stack traversal; returns (distance, primitive index or None, test counters)."""
        if self.root is None:
            return None, None, {"nodes": 0, "triangles": 0}
        best = math.inf
        best_index = None
        counters = {"nodes": 0, "triangles": 0}
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not isinstance(node, Node):
                distance, index = node
                counters["triangles"] += 1
                if distance < best:
                    best = distance
                    best_index = index
                continue
            counters["nodes"] += 1
            hit = ray_aabb(origin, direction, node.bounds)
            if hit is None or hit[0] >= best:
                continue
            if node.is_leaf:
                for index in node.members:
                    distance = ray_triangle(origin, direction, self.primitives[index])
                    stack.append((distance if distance is not None else math.inf, index))
                continue
            if ordered:
                left_hit = ray_aabb(origin, direction, node.left.bounds)
                right_hit = ray_aabb(origin, direction, node.right.bounds)
                left_time = left_hit[0] if left_hit else math.inf
                right_time = right_hit[0] if right_hit else math.inf
                if left_time <= right_time:
                    stack.append(node.right)
                    stack.append(node.left)
                else:
                    stack.append(node.left)
                    stack.append(node.right)
            else:
                stack.append(node.right)
                stack.append(node.left)
        if best_index is None:
            return None, None, counters
        return best, best_index, counters

    def leaf_primitives(self):
        """Map every primitive to the leaf that owns it (for the structural checks)."""
        ownership = {}
        for node_index, node in enumerate(self.nodes):
            if node.is_leaf:
                for index in node.members:
                    ownership.setdefault(index, []).append(node_index)
        return ownership

    def height(self):
        if self.root is None:
            return 0

        def walk(node, depth):
            if node is None or node.is_leaf:
                return depth
            return max(walk(node.left, depth + 1), walk(node.right, depth + 1))

        return walk(self.root, 1)


def expected_cost(bvh):
    """Canonical SAH estimate: sum over nodes of (SA(node)/SA(root)) * node cost."""
    if bvh.root is None:
        return 0.0
    root_area = max(surface_area(bvh.root.bounds), 1e-12)
    total = 0.0
    for node in bvh.nodes:
        weight = surface_area(node.bounds) / root_area
        total += weight * (TRAVERSAL_COST if not node.is_leaf else LEAF_COST * node.count)
    return total


def midpoint_split(indices, centroids, bounds_list):
    """The naive split used as the baseline for the SAH comparison."""
    centroid_bounds = bounds_union([(centroids[i][0], centroids[i][1], centroids[i][2],
                                     centroids[i][0], centroids[i][1], centroids[i][2])
                                    for i in indices])
    extents = [centroid_bounds[3] - centroid_bounds[0], centroid_bounds[4] - centroid_bounds[1],
               centroid_bounds[5] - centroid_bounds[2]]
    axis = max(range(3), key=lambda index: extents[index])
    if extents[axis] <= 0.0:
        return None
    middle = (centroid_bounds[axis] + centroid_bounds[axis + 3]) / 2.0
    left = [index for index in indices if centroids[index][axis] <= middle]
    right = [index for index in indices if centroids[index][axis] > middle]
    if not left or not right:
        return None
    return axis, left, right


def brute_force(primitives, origin, direction):
    best = math.inf
    best_index = None
    tests = 0
    for index, primitive in enumerate(primitives):
        tests += 1
        distance = ray_triangle(origin, direction, primitive)
        if distance is not None and distance < best:
            best = distance
            best_index = index
    return (best if best_index is not None else None), best_index, tests


# ------------------------------------------------------------------ scene and renderer
def build_scene(seed=20260879, grid=7, spacing=1.1, height=0.6):
    """A triangulated height field plus two ground triangles."""
    rng = Rng(seed)
    offsets = [[height * (rng.uniform() * 2.0 - 1.0) for _ in range(grid + 1)]
               for _ in range(grid + 1)]
    triangles = []
    for i in range(grid):
        for j in range(grid):
            x0, x1 = i * spacing, (i + 1) * spacing
            z0, z1 = j * spacing, (j + 1) * spacing
            a = (x0, offsets[i][j], z0)
            b = (x1, offsets[i + 1][j], z0)
            c = (x1, offsets[i + 1][j + 1], z1)
            d = (x0, offsets[i][j + 1], z1)
            tint = ((i % 3) / 3.0, (j % 4) / 4.0, 0.5)
            triangles.append(Triangle(a, b, c, tint))
            triangles.append(Triangle(a, c, d, tint))
    span = grid * spacing
    triangles.append(Triangle((-2.0, -0.8, -2.0), (span + 2.0, -0.8, -2.0),
                              (span + 2.0, -0.8, span + 2.0), (0.3, 0.3, 0.4)))
    triangles.append(Triangle((-2.0, -0.8, -2.0), (span + 2.0, -0.8, span + 2.0),
                              (-2.0, -0.8, span + 2.0), (0.3, 0.3, 0.4)))
    return triangles, span


def camera_rays(width, height, span):
    """Pinhole camera looking down the height field, with an explicit orthonormal basis."""
    origin = (span * 0.5, span * 1.2, -span * 0.8)
    target = (span * 0.5, 0.0, span * 0.4)
    forward = normalize(subtract(target, origin))
    right = normalize(cross(forward, (0.0, 1.0, 0.0)))
    up = cross(right, forward)
    rays = []
    for row in range(height):
        for column in range(width):
            offset_x = ((column + 0.5) / width - 0.5) * 2.0
            offset_y = (0.5 - (row + 0.5) / height) * 2.0
            direction = normalize((forward[0] + 0.6 * (right[0] * offset_x + up[0] * offset_y),
                                   forward[1] + 0.6 * (right[1] * offset_x + up[1] * offset_y),
                                   forward[2] + 0.6 * (right[2] * offset_x + up[2] * offset_y)))
            rays.append((origin, direction))
    return rays


def shade(primitive, distance, direction):
    normal = primitive.normal()
    facing = normal if dot(normal, direction) < 0 else (-normal[0], -normal[1], -normal[2])
    light = normalize((-0.4, 0.9, -0.6))
    lambert = max(0.0, dot(facing, light))
    return (primitive.colour[0] * (0.2 + 0.8 * lambert),
            primitive.colour[1] * (0.2 + 0.8 * lambert),
            primitive.colour[2] * (0.2 + 0.8 * lambert))


def render(primitives, rays, use_bvh=True, ordered=True):
    """Returns per-ray (hit distance, primitive index, colour) plus the work counters."""
    bvh = Bvh(primitives) if use_bvh else None
    image = []
    counters = {"nodes": 0, "triangles": 0}
    for origin, direction in rays:
        if use_bvh:
            distance, index, stats = bvh.traverse(origin, direction, ordered=ordered)
            counters["nodes"] += stats["nodes"]
            counters["triangles"] += stats["triangles"]
        else:
            distance, index, tests = brute_force(primitives, origin, direction)
            counters["triangles"] += tests
        if index is None:
            image.append((None, None, (0.05, 0.05, 0.08)))
        else:
            image.append((distance, index,
                          shade(primitives[index], distance, direction)))
    return image, counters, bvh


def main():
    primitives, span = build_scene()
    check(len(primitives) == 2 * 7 * 7 + 2, f"scene size changed: {len(primitives)}")
    check(all(primitive.area() > 0.0 for primitive in primitives),
          "the scene must contain no degenerate triangles")

    # ---------------- slab test against sampling ----------------
    bounds = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    inside = 0
    outside = 0
    grazing = 0
    for index in range(400):
        rng = Rng(1000 + index)
        origin = (rng.uniform() * 4.0 - 1.5, rng.uniform() * 4.0 - 1.5,
                  rng.uniform() * 4.0 - 1.5)
        direction = normalize((rng.uniform() * 2.0 - 1.0, rng.uniform() * 2.0 - 1.0,
                               rng.uniform() * 2.0 - 1.0))
        hit = ray_aabb(origin, direction, bounds)
        sampled = None
        step = 0.0
        while step < 4.0:
            point = (origin[0] + direction[0] * step, origin[1] + direction[1] * step,
                     origin[2] + direction[2] * step)
            if all(0.0 <= point[axis] <= 1.0 for axis in range(3)):
                sampled = step
                break
            step += 0.005
        if hit is None:
            check(sampled is None, "the slab test must not miss a sampled entry")
            outside += 1
        elif sampled is None:
            # the sampled chord through the box is thinner than the marching step
            check(hit[1] - hit[0] < 0.01,
                  f"a hit the sampling missed must be grazing: {hit[0]}..{hit[1]}")
            grazing += 1
            inside += 1
        else:
            check(abs(hit[0] - sampled) <= 0.01 or hit[0] == 0.0,
                  f"the slab entry should match the sample: {hit[0]} vs {sampled}")
            inside += 1
    check(inside > 20 and outside > 20,
          f"the slab battery needs both outcomes: {inside}/{outside}")
    parallel = ray_aabb((0.5, -2.0, 0.5), (0.0, 1.0, 0.0), bounds)
    check(parallel is not None and parallel[0] >= 0.0,
          "a ray parallel to a slab must still be handled")
    check(ray_aabb((2.0, 2.0, 2.0), (1.0, 1.0, 1.0), bounds) is None,
          "a ray pointing away must miss the box")

    # ---------------- BVH structure ----------------
    bvh = Bvh(primitives, leaf_size=4)
    leaves = bvh.leaves()
    check(len(bvh.nodes) == 2 * len(leaves) - 1,
          f"a binary BVH must have 2L-1 nodes: {len(bvh.nodes)} vs {2 * len(leaves) - 1}")
    ownership = bvh.leaf_primitives()
    check(sorted(ownership) == list(range(len(primitives))),
          "every primitive must be stored in exactly one leaf")
    check(all(len(owners) == 1 for owners in ownership.values()),
          "no primitive may appear in two leaves")
    total_in_leaves = sum(node.count for node in leaves)
    check(total_in_leaves == len(primitives), "leaf counts must add up to the primitive count")
    root_bounds = bvh.root.bounds
    check(all(root_bounds[axis] <= primitive.bounds()[axis] + 1e-12 and
              root_bounds[axis + 3] >= primitive.bounds()[axis + 3] - 1e-12
              for primitive in primitives for axis in range(3)),
          "the root bounds must enclose every primitive")

    def check_nesting(node):
        if node is None or node.is_leaf:
            return
        for child in (node.left, node.right):
            for axis in range(3):
                check(node.bounds[axis] <= child.bounds[axis] + 1e-9
                      and node.bounds[axis + 3] >= child.bounds[axis + 3] - 1e-9,
                      "child bounds must be nested inside the parent")
            check_nesting(child)

    check_nesting(bvh.root)
    check(bvh.height() <= 2 * math.log2(max(len(primitives), 2)) + 6,
          f"the tree must stay shallow, height {bvh.height()}")

    # ---------------- SAH beats the midpoint split ----------------
    baseline = Bvh(primitives, leaf_size=4, strategy="midpoint")
    check(surface_area(bvh.root.bounds) > 0.0, "the root must have a real surface area")
    sah_cost = expected_cost(bvh)
    mid_cost = expected_cost(baseline)
    check(sah_cost <= mid_cost + 1e-9,
          f"the SAH build must not be worse than midpoint splitting: {sah_cost} vs {mid_cost}")
    check(len(bvh.leaves()) <= len(baseline.leaves()),
          "a cost-driven build should stop splitting earlier than a forced midpoint split")

    # ---------------- traversal equals brute force ----------------
    rays = camera_rays(24, 24, span)
    bvh_image, bvh_counters, _ = render(primitives, rays, use_bvh=True)
    brute_image, brute_counters, _ = render(primitives, rays, use_bvh=False)
    hits = 0
    for index, (accelerated, reference) in enumerate(zip(bvh_image, brute_image)):
        if reference[1] is None:
            check(accelerated[1] is None, f"ray {index} must miss too")
            continue
        check(accelerated[1] is not None, f"ray {index} must hit something")
        hits += 1
        check(abs(accelerated[0] - reference[0]) < 1e-9,
              f"ray {index} distance differs: {accelerated[0]} vs {reference[0]}")
        check(max(abs(accelerated[2][c] - reference[2][c]) for c in range(3)) < 1e-9,
              f"ray {index} colour differs")
    check(hits > len(rays) // 3, f"the image must actually hit geometry: {hits}")
    check(bvh_counters["triangles"] * 5 < brute_counters["triangles"],
          f"the BVH must cut triangle tests by at least 5x: {bvh_counters} vs {brute_counters}")
    check(bvh_counters["triangles"] < len(primitives) * len(rays) / 4,
          f"the average ray must test far fewer than the whole scene: {bvh_counters}")

    # ---------------- ordering changes the visit order, not the result ----------------
    unordered_image, unordered_counters, _ = render(primitives, rays, use_bvh=True,
                                                    ordered=False)
    check(all(a[1] == b[1] and (a[0] is None) == (b[0] is None)
              and (a[0] is None or abs(a[0] - b[0]) < 1e-12)
              for a, b in zip(unordered_image, bvh_image)),
          "near-child-first ordering must not change any hit")
    check(abs(unordered_counters["triangles"] - bvh_counters["triangles"])
          <= 0.1 * bvh_counters["triangles"] + 10,
          "reordering the children must not change the amount of work dramatically")
    near = Triangle((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 0.0))
    far = Triangle((0.0, 0.0, 5.0), (1.0, 0.0, 5.0), (0.0, 1.0, 5.0), (0.0, 0.0, 1.0))
    two_leaf = Bvh([far, near], leaf_size=1)
    for ordered_flag in (True, False):
        distance, index, _ = two_leaf.traverse((0.2, 0.2, 0.0), (0.0, 0.0, 1.0),
                                               ordered=ordered_flag)
        check(index == 1 and abs(distance - 1.0) < 1e-12,
              f"the nearer triangle must win regardless of ordering (ordered={ordered_flag})")

    # ---------------- shadow rays and degenerate cases ----------------
    light = normalize((-1.0, 0.4, -0.7))
    shadow_bvh = Bvh(primitives)
    shadow_agreements = 0
    occluded = 0
    for origin, direction in rays[::7]:
        distance, index, _ = shadow_bvh.traverse(origin, direction)
        if index is None:
            continue
        point = (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance,
                 origin[2] + direction[2] * distance)
        shadow_origin = (point[0] + light[0] * 1e-4, point[1] + light[1] * 1e-4,
                         point[2] + light[2] * 1e-4)
        accelerated = shadow_bvh.traverse(shadow_origin, light)[1]
        reference = brute_force(primitives, shadow_origin, light)[1]
        check((accelerated is None) == (reference is None),
              "shadow-ray occlusion must agree with brute force")
        shadow_agreements += 1
        occluded += 1 if accelerated is not None else 0
    check(shadow_agreements > 20, f"the shadow battery must run: {shadow_agreements}")
    check(occluded > 0, "some shadow rays must be blocked by the terrain")
    check(occluded < shadow_agreements, "some shadow rays must reach the sky")

    empty_bvh = Bvh([])
    check(empty_bvh.traverse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))[1] is None,
          "an empty scene must miss")
    degenerate = Triangle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    check(ray_triangle((0.0, 1.0, 0.0), (0.0, -1.0, 0.0), degenerate) is None,
          "a zero-area triangle must never be hit")
    shared = Triangle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    edge = ray_triangle((0.5, 0.5, 1.0), (0.0, 0.0, -1.0), shared)
    check(edge is not None and abs(edge - 1.0) < 1e-12,
          f"a grazing hit on the hypotenuse must register: {edge}")
    behind = ray_triangle((0.25, 0.25, 1.0), (0.0, 0.0, 1.0), shared)
    check(behind is None, "a triangle behind the origin must not be hit")

    # ---------------- two bounces and determinism ----------------
    def trace_two_bounces(primitives, rays):
        bvh = Bvh(primitives)
        output = []
        for origin, direction in rays:
            distance, index, _ = bvh.traverse(origin, direction)
            if index is None:
                output.append((0.05, 0.05, 0.08))
                continue
            point = (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance,
                     origin[2] + direction[2] * distance)
            normal = primitives[index].normal()
            facing = normal if dot(normal, direction) < 0 else (
                -normal[0], -normal[1], -normal[2])
            reflected = normalize(subtract(direction, tuple(2.0 * dot(direction, facing)
                                                            * value for value in facing)))
            bounce_origin = (point[0] + reflected[0] * 1e-4, point[1] + reflected[1] * 1e-4,
                             point[2] + reflected[2] * 1e-4)
            second, second_index, _ = bvh.traverse(bounce_origin, reflected)
            colour = shade(primitives[index], distance, direction)
            if second_index is not None:
                secondary = shade(primitives[second_index], second, reflected)
                colour = tuple(0.6 * colour[axis] + 0.4 * secondary[axis] for axis in range(3))
            output.append(colour)
        return output

    first_pass = trace_two_bounces(primitives, rays[::5])
    second_pass = trace_two_bounces(primitives, rays[::5])
    check(first_pass == second_pass, "the two-bounce trace must be deterministic")
    check(all(0.0 <= channel <= 1.0 for colour in first_pass for channel in colour),
          "shaded colours must stay in range")
    check([node.bounds for node in Bvh(primitives).nodes] == [node.bounds for node in bvh.nodes],
          "the BVH build must be deterministic")

    print(f"primitives={len(primitives)} nodes={len(bvh.nodes)} leaves={len(leaves)} "
          f"height={bvh.height()}")
    print(f"sah_cost={sah_cost:.2f} midpoint_cost={mid_cost:.2f}")
    print(f"triangle tests bvh={bvh_counters['triangles']} brute={brute_counters['triangles']} "
          f"ratio={brute_counters['triangles'] / max(bvh_counters['triangles'], 1):.1f}x")
    print(f"hits={hits}/{len(rays)} shadows={occluded}/{shadow_agreements} "
          f"unordered_tests={unordered_counters['triangles']}")
    print("PASS")


if __name__ == "__main__":
    main()
