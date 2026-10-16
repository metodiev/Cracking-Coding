/** Task 794 - Implement 3D point cloud voxelization and octree building. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two classic spatial structures over the same cloud:
 *
 *   voxel grid - a uniform bucket size r. A point p lands in the integer cell (floor(p_x/r), floor(p_y/r),
 *                floor(p_z/r)), so the grid is exact and O(1) per point, and the occupied cell count at
 *                resolution 1/2^d equals the number of leaves an octree reaches at depth d on a dense cloud.
 *   octree     - a root cube split into eight octants, recursively, until a node holds at most maxPoints
 *                points or the depth limit is reached. Every node keeps the indices of the points inside its
 *                cube, so the structure supports radius queries with box pruning.
 *
 * Both are built so that the fundamental invariant holds at every level: the children partition the parent's
 * point set, hence the sum of the leaf counts equals the number of points and every point stays inside the
 * cube of the node that owns it. The radius query walks the tree and skips any node whose cube is further
 * than the radius from the query centre.
 *
 * Simplified: no parallelism, no Morton-code compression, no point deduplication, and the child blocks are
 * recomputed by recursion rather than stored as offsets into a flat array (the memory layout of a production
 * octree). The geometry, partitioning and query pruning are the real algorithms.
 *
 * Verified: the voxel occupancy matches an independent hash-map count and the bucket counts sum to N; every
 * point lies inside its own voxel; the octree leaves partition the cloud exactly (no loss, no duplication)
 * and every point lies inside its leaf cube; node occupancy respects maxPoints unless the depth limit was
 * reached, and the depth never exceeds the limit; on a dense uniform cloud the leaf count at the depth limit
 * equals the occupied voxel count at the matching resolution; radius queries agree with brute force for
 * several centres and radii; degenerate clouds (empty, single point, all points coincident, points at the
 * top boundary, a cloud that fits in one leaf) behave; and rebuilding yields the identical structure.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** Uniform voxel grid over [0, 1)^3. */
class VoxelGrid {
    final double resolution;
    final Map<Long, Integer> buckets = new LinkedHashMap<>();

    VoxelGrid(double resolution) { this.resolution = resolution; }

    long cellKey(double[] p) {
        long i = (long) Math.floor(p[0] / resolution); long j = (long) Math.floor(p[1] / resolution);
        long k = (long) Math.floor(p[2] / resolution); return (i * 1000003L + j) * 1000003L + k;
    }

    void add(double[] p) { buckets.merge(cellKey(p), 1, Integer::sum); }

    int occupiedCells() { return buckets.size(); }

    int points() {
        int total = 0; for (int v : buckets.values()) total += v; return total;
    }

    /** Lower corner of a cell, recovered from the key, to check containment independently. */
    double[] cellLower(long key) {
        long k = Math.floorMod(key, 1000003L); long rest = (key - k) / 1000003L;
        long j = Math.floorMod(rest, 1000003L); long i = Math.floorDiv(rest, 1000003L);
        return new double[]{i * resolution, j * resolution, k * resolution};
    }
}

/** Octree over a cube; leaves hold the indices of the points inside their cube. */
class Octree {
    final double[] min;
    final double size;
    final int[] indices;
    Octree[] children;

    Octree(double[] min, double size, int[] indices) {
        this.min = min; this.size = size; this.indices = indices;
    }

    static boolean inside(double[] p, double[] min, double size) {
        for (int a = 0; a < 3; a++) if (p[a] < min[a] - 1e-12 || p[a] > min[a] + size + 1e-12) return false;
        return true;
    }

    int octantOf(double[] p) {
        int octant = 0; for (int a = 0; a < 3; a++) if (p[a] >= min[a] + size / 2) octant |= 1 << a; return octant;
    }

    static Octree build(double[][] cloud, int[] indices, double[] min, double size, int maxPoints,
            int maxDepth) {
        Octree node = new Octree(min.clone(), size, indices);
        if (indices.length <= maxPoints || maxDepth == 0 || size < 1e-12) return node;
        List<List<Integer>> buckets = new ArrayList<>(); for (int i = 0; i < 8; i++) buckets.add(new ArrayList<>());
        for (int index : indices) buckets.get(node.octantOf(cloud[index])).add(index); node.children = new Octree[8];
        for (int octant = 0; octant < 8; octant++) {
            List<Integer> bucket = buckets.get(octant); int[] childIndices = new int[bucket.size()];
            for (int i = 0; i < childIndices.length; i++) childIndices[i] = bucket.get(i);
            double[] childMin = min.clone();
            for (int a = 0; a < 3; a++) if ((octant & (1 << a)) != 0) childMin[a] += size / 2;
            node.children[octant] = build(cloud, childIndices, childMin, size / 2, maxPoints, maxDepth - 1);
        }
        return node;
    }

    boolean isLeaf() { return children == null; }

    void leaves(List<Octree> out) {
        if (isLeaf()) {
            out.add(this); return;
        }
        for (Octree child : children) child.leaves(out);
    }

    int nodeCount() {
        int total = 1; if (children != null) for (Octree child : children) total += child.nodeCount(); return total;
    }

    int depth() {
        if (children == null) return 0; int deepest = 0;
        for (Octree child : children) deepest = Math.max(deepest, child.depth()); return deepest + 1;
    }

    /** Squared distance from a point to the node's cube (zero inside). */
    double squaredDistanceToBox(double[] centre) {
        double sum = 0;
        for (int a = 0; a < 3; a++) {
            double d = 0; if (centre[a] < min[a]) d = min[a] - centre[a];
            else if (centre[a] > min[a] + size) d = centre[a] - (min[a] + size); sum += d * d;
        }
        return sum;
    }

    void radiusQuery(double[][] cloud, double[] centre, double radius, List<Integer> out) {
        if (squaredDistanceToBox(centre) > radius * radius) return;
        if (isLeaf()) {
            for (int index : indices) {
                double d = Math.hypot(Math.hypot(cloud[index][0] - centre[0], cloud[index][1] - centre[1]),
                        cloud[index][2] - centre[2]);
                if (d <= radius) out.add(index);
            }
            return;
        }
        for (Octree child : children) child.radiusQuery(cloud, centre, radius, out);
    }
}

public class Solution {
    static double[][] clusteredCloud(int n, SplitMix64 rng) {
        double[][] cloud = new double[n][3];
        for (int i = 0; i < n; i++) {
            double cluster = i % 5;
            double cx = 0.15 + 0.2 * cluster, cy = 0.2 + 0.15 * (i % 3), cz = 0.3 + 0.1 * (i % 4);
            for (int a = 0; a < 3; a++) {
                double base = a == 0 ? cx : a == 1 ? cy : cz;
                cloud[i][a] = Math.min(0.999999, Math.max(0.0, base + 0.05 * (rng.nextDouble() - 0.5)));
            }
        }
        return cloud;
    }

    static double[][] uniformCloud(int n, SplitMix64 rng) {
        double[][] cloud = new double[n][3];
        for (int i = 0; i < n; i++)
            for (int a = 0; a < 3; a++) cloud[i][a] = rng.nextDouble() * 0.999;
        return cloud;
    }

    static int[] allIndices(int n) {
        int[] indices = new int[n]; for (int i = 0; i < n; i++) indices[i] = i; return indices;
    }

    static List<Integer> bruteRadius(double[][] cloud, double[] centre, double radius) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < cloud.length; i++) {
            double d = Math.hypot(Math.hypot(cloud[i][0] - centre[0], cloud[i][1] - centre[1]),
                    cloud[i][2] - centre[2]);
            if (d <= radius) out.add(i);
        }
        Collections.sort(out); return out;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240794L); double[][] cloud = clusteredCloud(240, rng);
        double resolution = 0.05;
        Octree tree = Octree.build(cloud, allIndices(cloud.length), new double[]{0, 0, 0}, 1.0, 4, 6);
        boolean[] seen = new boolean[cloud.length]; int depth = tree.depth(), nodes = tree.nodeCount();
        check(depth <= 6, "the octree must respect the depth limit, got " + depth);

        // --- occupancy: leaves partition the cloud exactly --------------------------------
        List<Octree> leaves = new ArrayList<>(); tree.leaves(leaves);
        int total = 0, duplicates = 0, misplaced = 0, oversized = 0;
        for (Octree leaf : leaves) {
            total += leaf.indices.length;
            for (int index : leaf.indices) {
                if (seen[index]) duplicates++; seen[index] = true;
                if (!Octree.inside(cloud[index], leaf.min, leaf.size)) misplaced++;
            }
            if (leaf.indices.length > 4 && leaf.size > 1.0 / 64 + 1e-12) oversized++;
        }
        check(total == cloud.length, "leaf counts must sum to the cloud size, got " + total);
        check(duplicates == 0, "leaves must not duplicate points, found " + duplicates);
        check(misplaced == 0, "every point must lie inside its leaf cube, " + misplaced + " do not");
        check(oversized == 0, "leaves beyond the depth limit must respect maxPoints, " + oversized + " do not");
        check(nodes == 1 + 8 * (nodes - leaves.size()),
                "every internal node must own exactly eight children: " + nodes + " nodes, " + leaves.size()
                        + " leaves");

        // --- voxel grid against an independent count ----------------------------------------
        VoxelGrid grid = new VoxelGrid(resolution); for (double[] p : cloud) grid.add(p);
        check(grid.points() == cloud.length, "every point must land in some bucket");
        Set<Long> independent = new HashSet<>();
        for (double[] p : cloud) independent.add(((long) (p[0] / resolution) * 7919L
                + (long) (p[1] / resolution)) * 7919L + (long) (p[2] / resolution));
        check(independent.size() == grid.occupiedCells(), "independent voxel count " + independent.size()
                + " versus grid " + grid.occupiedCells());
        int contained = 0;
        for (Map.Entry<Long, Integer> entry : grid.buckets.entrySet()) {
            double[] lower = grid.cellLower(entry.getKey());
            for (double[] p : cloud) {
                boolean inCell = true;
                for (int a = 0; a < 3; a++)
                    if (p[a] < lower[a] || p[a] >= lower[a] + resolution) inCell = false;
                if (inCell) {
                    contained++; break;
                }
            }
        }
        check(contained == grid.occupiedCells(), "every bucket must own a point inside its own cube");

        // --- a dense uniform cloud: leaves at the depth limit equal the voxel count ----------
        double[][] dense = uniformCloud(3000, new SplitMix64(5150L));
        Octree denseTree = Octree.build(dense, allIndices(dense.length), new double[]{0, 0, 0}, 1.0, 1, 3);
        List<Octree> denseLeaves = new ArrayList<>(); denseTree.leaves(denseLeaves); int deepLeaves = 0;
        for (Octree leaf : denseLeaves)
            if (leaf.size <= 0.125 + 1e-12 && leaf.indices.length > 0) deepLeaves++;
        VoxelGrid denseGrid = new VoxelGrid(0.125); for (double[] p : dense) denseGrid.add(p);
        check(deepLeaves == denseGrid.occupiedCells(), "octree leaves at depth 3 " + deepLeaves
                + " must equal voxels at resolution 1/8 " + denseGrid.occupiedCells());

        // --- radius queries against brute force ------------------------------------------------
        double[][] centres = {{0.3, 0.3, 0.3}, {0.5, 0.5, 0.5}, {0.99, 0.01, 0.5}, {0.0, 0.0, 0.0}};
        double[] radii = {0.05, 0.12, 0.3};
        for (double[] centre : centres) {
            for (double radius : radii) {
                List<Integer> fromTree = new ArrayList<>();
                tree.radiusQuery(cloud, centre, radius, fromTree); Collections.sort(fromTree);
                List<Integer> brute = bruteRadius(cloud, centre, radius);
                check(fromTree.equals(brute), "radius query at " + Arrays.toString(centre) + " r = " + radius
                        + " returned " + fromTree.size() + ", brute force " + brute.size());
            }
        }

        // --- degenerate clouds --------------------------------------------------------------------
        Octree empty = Octree.build(new double[0][3], new int[0], new double[]{0, 0, 0}, 1.0, 4, 6);
        check(empty.isLeaf() && empty.nodeCount() == 1 && empty.depth() == 0, "an empty cloud is one leaf");
        List<Integer> emptyHits = new ArrayList<>();
        empty.radiusQuery(new double[0][3], new double[]{0.5, 0.5, 0.5}, 1.0, emptyHits);
        check(emptyHits.isEmpty(), "an empty cloud answers nothing");

        double[][] single = {{0.25, 0.25, 0.25}};
        Octree singleTree = Octree.build(single, new int[]{0}, new double[]{0, 0, 0}, 1.0, 4, 6);
        check(singleTree.depth() == 0 && singleTree.indices.length == 1, "one point fits in the root");
        check(Octree.inside(single[0], singleTree.min, singleTree.size), "the single point is inside the root");

        double[][] coincident = new double[50][3];
        for (double[] p : coincident) {
            p[0] = 0.5; p[1] = 0.5; p[2] = 0.5;
        }
        Octree deep = Octree.build(coincident, allIndices(50), new double[]{0, 0, 0}, 1.0, 4, 6);
        check(deep.depth() == 6, "coincident points must be split to the depth limit, got " + deep.depth());
        List<Octree> coincidentLeaves = new ArrayList<>(); deep.leaves(coincidentLeaves); int coincidentTotal = 0;
        for (Octree leaf : coincidentLeaves) coincidentTotal += leaf.indices.length;
        check(coincidentTotal == 50, "coincident points must not be lost, got " + coincidentTotal);

        double[][] boundary = {{0.0, 0.0, 0.0}, {0.999999, 0.999999, 0.999999}, {0.5, 0.5, 0.5}};
        Octree boundaryTree = Octree.build(boundary, allIndices(3), new double[]{0, 0, 0}, 1.0, 1, 4);
        List<Octree> boundaryLeaves = new ArrayList<>(); boundaryTree.leaves(boundaryLeaves); int boundaryTotal = 0;
        for (Octree leaf : boundaryLeaves) {
            boundaryTotal += leaf.indices.length;
            for (int index : leaf.indices)
                check(Octree.inside(boundary[index], leaf.min, leaf.size),
                        "boundary point " + index + " escaped its leaf");
        }
        check(boundaryTotal == 3, "boundary points must all be kept, got " + boundaryTotal);

        // --- idempotence -----------------------------------------------------------------------------
        Octree rebuilt = Octree.build(cloud, allIndices(cloud.length), new double[]{0, 0, 0}, 1.0, 4, 6);
        List<Octree> rebuiltLeaves = new ArrayList<>(); rebuilt.leaves(rebuiltLeaves);
        boolean identical = rebuiltLeaves.size() == leaves.size();
        for (int i = 0; identical && i < leaves.size(); i++) {
            identical = rebuiltLeaves.get(i).indices.length == leaves.get(i).indices.length;
            if (identical)
                for (int a = 0; a < 3; a++)
                    identical &= rebuiltLeaves.get(i).min[a] == leaves.get(i).min[a];
        }
        check(identical, "rebuilding the same cloud must give the same tree");

        System.out.println("octree: " + nodes + " nodes, " + leaves.size() + " leaves, depth " + depth);
        System.out.println("voxels at " + resolution + ": " + grid.occupiedCells() + " occupied of "
                + cloud.length + " points; depth-3 leaves " + deepLeaves + " versus voxels "
                + denseGrid.occupiedCells());
        System.out.println("radius queries agreed with brute force for " + (centres.length * radii.length)
                + " probes");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
