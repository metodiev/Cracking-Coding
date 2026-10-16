"""Task 794 - Implement 3D point cloud voxelization and octree building.  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)

Voxelization maps each point to the integer cell (i, j, k) = (floor(x/s), floor(y/s),
floor(z/s)) of a uniform grid of size s, which gives O(1) neighbourhood lookups, deduplication
and a fixed memory footprint per cell.  The octree is the adaptive version of the same idea:
a node owns an axis-aligned cube and the points inside it, and is split into eight octants
whenever it holds more than max_points points and is still larger than min_size, so dense
regions are refined while empty space costs a single node.  Both structures answer "which
points lie inside this box?" - the voxel grid by walking the integer cell range and the
octree by descending only into the children whose bounds intersect the query box.

Verified: every point is assigned to its floor cell and lies within half a cell of that cell's
centre; the voxel counts and the octree leaf populations both sum to the point count; every
point stored in a node lies inside that node's cube; the eight children of a split node
partition the parent exactly (half the edge length, sum of volumes equal to the parent's);
box queries from the voxel grid and from the octree agree with a brute-force scan for many
random boxes; leaf capacity is respected wherever refinement was still allowed; boundary
assignment is deterministic for points exactly on a cell border and for negative coordinates;
and an empty or single-point cloud degenerates gracefully.

Simplified versus a production octree: the tree is built once in memory (no incremental
insertion or deletion, no Morton-code ordering, no serialisation), the cube is chosen as the
smallest cube containing the cloud, and the range query returns point indices rather than
reorganising memory in cache-friendly leaf order.
"""
import math

MASK64 = (1 << 64) - 1
OCTANTS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state, low, high):
    state, r = splitmix64(state)
    return state, low + (high - low) * r / 18446744073709551616.0


# ------------------------------------------------------------------ voxel grid
def voxel_key(point, size):
    """Integer cell of a point; exact for negative coordinates as well (floor semantics)."""
    return (math.floor(point[0] / size), math.floor(point[1] / size), math.floor(point[2] / size))


def voxel_centre(key, size):
    return ((key[0] + 0.5) * size, (key[1] + 0.5) * size, (key[2] + 0.5) * size)


def voxelize(points, size):
    """Deterministic dict cell -> indices, in point order."""
    if size <= 0.0:
        raise ValueError("voxel size must be positive")
    grid = {}
    for index, point in enumerate(points):
        grid.setdefault(voxel_key(point, size), []).append(index)
    return grid


def voxel_range_query(points, size, grid, box):
    """Brute-force cell walk over the integer range covering the query box."""
    lo = voxel_key((box[0], box[1], box[2]), size)
    hi = voxel_key((box[3], box[4], box[5]), size)
    found = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):
                for index in grid.get((i, j, k), ()):
                    point = points[index]
                    if (box[0] <= point[0] <= box[3] and box[1] <= point[1] <= box[4]
                            and box[2] <= point[2] <= box[5]):
                        found.append(index)
    return sorted(found)


def brute_force_box_query(points, box):
    return sorted(index for index, point in enumerate(points)
                  if box[0] <= point[0] <= box[3] and box[1] <= point[1] <= box[4]
                  and box[2] <= point[2] <= box[5])


# ---------------------------------------------------------------------- octree
class OctreeNode:
    __slots__ = ("bounds", "indices", "children", "depth")

    def __init__(self, bounds, indices, depth):
        self.bounds = bounds
        self.indices = indices
        self.children = None
        self.depth = depth

    def is_leaf(self):
        return self.children is None

    def size(self):
        return self.bounds[3] - self.bounds[0]

    def centre(self):
        return tuple(0.5 * (self.bounds[i] + self.bounds[i + 3]) for i in range(3))


def containing_cube(points, pad=1e-9):
    """Smallest axis-aligned cube containing every point (a tiny pad keeps borders inside)."""
    if not points:
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    edge = max(high[i] - low[i] for i in range(3))
    if edge <= 0.0:
        edge = 1.0
    edge += 2.0 * pad
    centre = [0.5 * (low[i] + high[i]) for i in range(3)]
    return tuple(centre[i] - 0.5 * edge for i in range(3)) + \
        tuple(centre[i] + 0.5 * edge for i in range(3))


def inside_box(point, bounds):
    return (bounds[0] <= point[0] <= bounds[3] and bounds[1] <= point[1] <= bounds[4]
            and bounds[2] <= point[2] <= bounds[5])


def build_octree(points, max_points=4, min_size=0.05, max_depth=12):
    """Recursive build; a node splits while it is too full and refinement is still allowed."""
    if max_points < 1:
        raise ValueError("max_points must be positive")
    root = OctreeNode(containing_cube(points), list(range(len(points))), 0)
    stack = [root]
    while stack:
        node = stack.pop()
        if (len(node.indices) <= max_points or node.size() <= min_size
                or node.depth >= max_depth):
            continue
        children = []
        for dx, dy, dz in OCTANTS:
            lo = node.bounds
            half = 0.5 * node.size()
            base = (lo[0] + dx * half, lo[1] + dy * half, lo[2] + dz * half)
            bounds = (base[0], base[1], base[2], base[0] + half, base[1] + half, base[2] + half)
            children.append(OctreeNode(bounds, [], node.depth + 1))
        for index in node.indices:
            point = points[index]
            for child in children:
                if inside_box(point, child.bounds):
                    child.indices.append(index)
                    break
        node.children = children
        node.indices = []
        stack.extend(children)
    return root


def walk(node):
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        if not current.is_leaf():
            stack.extend(current.children)


def leaves(root):
    return [node for node in walk(root) if node.is_leaf()]


def node_count(root):
    return sum(1 for _node in walk(root))


def max_depth(root):
    return max(node.depth for node in walk(root))


def query_box(root, points, box):
    """Descend only into intersecting children; returns sorted point indices."""
    found = []
    stack = [root]
    while stack:
        node = stack.pop()
        bounds = node.bounds
        if not (bounds[0] <= box[3] and bounds[3] >= box[0]
                and bounds[1] <= box[4] and bounds[4] >= box[1]
                and bounds[2] <= box[5] and bounds[5] >= box[2]):
            continue
        if node.is_leaf():
            found.extend(index for index in node.indices
                         if inside_box(points[index], box))
        else:
            stack.extend(node.children)
    return sorted(found)


def leaf_population(root):
    return [len(node.indices) for node in leaves(root)]


# ------------------------------------------------------------------ data sets
def make_cloud(seed, count=64, scale=1.0):
    points = []
    state = seed
    for _ in range(count):
        coords = []
        for _c in range(3):
            state, value = uniform(state, -scale, scale)
            coords.append(value)
        points.append(tuple(coords))
    return points, state


def main():
    points, _state = make_cloud(794001, count=64, scale=1.2)
    size = 0.25
    grid = voxelize(points, size)

    # --- 1. voxel assignment: exact cell and half-cell bound ----------------
    worst_offset = 0.0
    for index, point in enumerate(points):
        key = voxel_key(point, size)
        check(key in grid and index in grid[key],
              "every point must be found in the cell its floor index names")
        centre = voxel_centre(key, size)
        worst_offset = max(worst_offset,
                           max(abs(point[i] - centre[i]) for i in range(3)))
    check(worst_offset <= 0.5 * size + 1e-12,
          "a point must lie within half a cell of its centre, worst %.6f" % worst_offset)
    check(sum(len(indices) for indices in grid.values()) == len(points),
          "the voxel grid must contain every point exactly once")
    check(all(len(set(indices)) == len(indices) for indices in grid.values()),
          "no cell may list the same index twice")

    # --- 2. octree invariants: conservation, containment, partition ---------
    root = build_octree(points, max_points=4, min_size=0.05)
    check(sum(leaf_population(root)) == len(points),
          "the leaves must contain every point exactly once")
    for node in walk(root):
        for index in node.indices:
            check(inside_box(points[index], node.bounds),
                  "every stored point must lie inside its node cube")
    volume_sum_ok = True
    for node in walk(root):
        if node.is_leaf():
            continue
        child_volume = sum(child.size() ** 3 for child in node.children)
        volume_sum_ok &= abs(child_volume - node.size() ** 3) < 1e-12 * max(1.0, node.size() ** 3)
        volume_sum_ok &= all(child.size() == 0.5 * node.size() for child in node.children)
        volume_sum_ok &= all(node.bounds[0] <= child.bounds[0]
                             and child.bounds[3] <= node.bounds[3] for child in node.children)
    check(volume_sum_ok, "the eight children must partition the parent cube exactly")
    check(len(root.children) == 8, "a split root must have eight children")

    # --- 3. leaf capacity where refinement was still possible ---------------
    capacity_ok = True
    for node in leaves(root):
        if node.size() > 0.05 and node.depth < 12:
            capacity_ok &= len(node.indices) <= 4
    check(capacity_ok, "no leaf above the minimum size may exceed max_points")
    check(len(root.indices) == 0, "a split node must hand its points to its children")

    # --- 4. range queries against brute force -------------------------------
    state = 794002
    mismatches = 0
    for _ in range(12):
        corners = []
        for _c in range(6):
            state, value = uniform(state, -1.4, 1.4)
            corners.append(value)
        box = (min(corners[0], corners[3]), min(corners[1], corners[4]),
               min(corners[2], corners[5]), max(corners[0], corners[3]),
               max(corners[1], corners[4]), max(corners[2], corners[5]))
        expected = brute_force_box_query(points, box)
        mismatches += (query_box(root, points, box) != expected)
        mismatches += (voxel_range_query(points, size, grid, box) != expected)
    check(mismatches == 0,
          "octree and voxel range queries must match brute force, %d mismatches" % mismatches)
    whole = containing_cube(points)
    check(query_box(root, points, whole) == list(range(len(points))),
          "a query covering the root cube must return every point")

    # --- 5. boundary cases --------------------------------------------------
    check(voxel_key((0.0, 0.0, 0.0), 0.5) == (0, 0, 0), "the origin must land in cell (0,0,0)")
    check(voxel_key((0.5, -0.5, -1.0), 0.5) == (1, -1, -2),
          "points exactly on a border must go to the higher cell (floor, not round)")
    check(voxel_key((-0.1, -0.6, -1.1), 0.5) == (-1, -2, -3),
          "negative coordinates must floor towards minus infinity")
    empty_root = build_octree([])
    check(empty_root.is_leaf() and empty_root.indices == [],
          "an empty cloud must give a single empty root leaf")
    check(query_box(empty_root, [], (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)) == [],
          "a query on an empty tree must return nothing")
    single = [(0.3, -0.2, 0.9)]
    single_root = build_octree(single)
    check(single_root.is_leaf() and single_root.indices == [0],
          "a single point must stay in an unsplit root")
    duplicates = [(0.1, 0.2, 0.3)] * 6
    duplicate_grid = voxelize(duplicates, 0.25)
    check(list(duplicate_grid.values()) == [list(range(6))],
          "identical points must share one cell and keep every index")
    duplicate_root = build_octree(duplicates, max_points=2, min_size=0.01)
    check(sum(leaf_population(duplicate_root)) == 6,
          "coincident points must survive the octree split")
    raised = False
    try:
        voxelize(points, 0.0)
    except ValueError:
        raised = True
    check(raised, "a non-positive voxel size must be rejected")
    raised = False
    try:
        build_octree(points, max_points=0)
    except ValueError:
        raised = True
    check(raised, "a non-positive leaf capacity must be rejected")

    # --- 6. determinism and refinement behaviour ----------------------------
    twin = build_octree(points, max_points=4, min_size=0.05)
    check([sorted(node.indices) for node in walk(twin)]
          == [sorted(node.indices) for node in walk(root)],
          "two builds from the same cloud must produce the same structure")
    coarse = build_octree(points, max_points=16, min_size=0.05)
    fine = build_octree(points, max_points=2, min_size=0.05)
    check(node_count(fine) > node_count(coarse),
          "a smaller leaf capacity must produce more nodes: %d vs %d"
          % (node_count(fine), node_count(coarse)))
    check(max_depth(fine) >= max_depth(coarse),
          "a smaller leaf capacity must not reduce the depth")

    print("voxels: %d cells for %d points at size %.2f (worst half-cell offset %.4f)"
          % (len(grid), len(points), size, worst_offset))
    print("octree: %d nodes, %d leaves, max depth %d, largest leaf %d points"
          % (node_count(root), len(leaves(root)), max_depth(root), max(leaf_population(root))))
    print("range queries: %d random boxes, 0 mismatches against brute force" % 12)
    print("refinement: coarse %d nodes vs fine %d nodes"
          % (node_count(coarse), node_count(fine)))
    print("PASS")


if __name__ == "__main__":
    main()
