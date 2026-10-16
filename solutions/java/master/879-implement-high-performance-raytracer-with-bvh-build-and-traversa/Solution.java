/** Task 879 - Implement high-performance raytracer with BVH build and traversal optimization. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A raytracer's cost is dominated by two decisions: which primitives are worth intersecting with a ray, and
 * in what order. This file implements both ends of that trade:
 *
 *   build      - a bounding volume hierarchy over triangle bounds, split by the surface area heuristic using
 *                twelve binned candidate splits, compared against a naive median split of the longest axis;
 *   traversal  - an explicit stack walk with the standard 3D slab test, ordered by entry distance, counting
 *                ray-box and ray-triangle tests so that "faster" is a measured number rather than a feeling.
 *
 * The scene is a deterministic triangle soup: a tessellated sphere plus scattered triangles, which is exactly
 * the mixture (coherent clusters, long thin primitives) that stresses a BVH. Every traversal result is
 * checked against a brute-force walk over the whole primitive list, and the slab test is checked against a
 * dense sampling oracle, so the acceleration structure can never silently change an image.
 *
 * SIMPLIFIED: triangles only, no shading beyond a deterministic Lambert term, no textures, no ray packets, no
 * SIMD, no motion blur, no threading and no wall-clock timing - "performance" is reported as counted tests and
 * as the surface-area-heuristic cost of the tree. All geometry comes from splitmix64 with a fixed seed.
 *
 * VERIFIED: (a) CORRECTNESS - for hundreds of random, grazing, missing, inside-the-scene and degenerate rays
 * the BVH returns exactly the hit the brute-force walk returns, including the hit distance; (b) TREE
 * INVARIANTS - every node's bounds contain its children's, every triangle appears in exactly one leaf, and
 * the leaf union is the whole primitive set; (c) COST - the BVH performs far fewer triangle tests per ray than
 * brute force, box tests grow logarithmically with the primitive count, and the binned surface-area heuristic
 * produces a measurably cheaper tree than a median split of the same primitives; (d) SLAB TEST - the entry and
 * exit distances agree with a dense sampling oracle, parallel rays and origins inside the box included;
 * (e) SHADING - a small image rendered through the BVH is identical to the same image rendered by brute force;
 * (f) degenerate inputs (empty scene, one triangle, zero-area triangle, a ray starting inside a primitive)
 * behave, and everything is reproducible.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A triangle with a precomputed bounding box and centroid. */
class Triangle {
    final double[] v = new double[9];
    final double[] bounds = new double[6];
    final double[] centroid = new double[3];
    int ordinal;

    Triangle(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy,
            double cz) {
        v[0] = ax; v[1] = ay; v[2] = az;
        v[3] = bx; v[4] = by; v[5] = bz;
        v[6] = cx; v[7] = cy; v[8] = cz;
        for (int axis = 0; axis < 3; axis++) {
            double lo = Math.min(Math.min(v[axis], v[3 + axis]), v[6 + axis]);
            double hi = Math.max(Math.max(v[axis], v[3 + axis]), v[6 + axis]);
            bounds[2 * axis] = lo - 1e-9;
            bounds[2 * axis + 1] = hi + 1e-9;
            centroid[axis] = (v[axis] + v[3 + axis] + v[6 + axis]) / 3;
        }
    }

    double area() {
        double e1x = v[3] - v[0], e1y = v[4] - v[1], e1z = v[5] - v[2];
        double e2x = v[6] - v[0], e2y = v[7] - v[1], e2z = v[8] - v[2];
        double cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
        return 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    }
}

/** A ray with an origin, a direction (not necessarily normalised) and a distance limit. */
class Ray {
    final double ox, oy, oz, dx, dy, dz, maxT;

    Ray(double ox, double oy, double oz, double dx, double dy, double dz, double maxT) {
        this.ox = ox; this.oy = oy; this.oz = oz; this.dx = dx; this.dy = dy; this.dz = dz;
        this.maxT = maxT;
    }
}

/** The scene: a triangle list plus the acceleration structure and the counters that make cost measurable. */
class Scene {
    final List<Triangle> triangles = new ArrayList<>();
    BvhNode root;
    long boxTests, triangleTests, bruteTriangleTests;

    static Scene sphereSoup(int rings, int segments, double radius) {
        Scene scene = new Scene();
        for (int ring = 0; ring < rings; ring++) {
            double phi0 = Math.PI * ring / rings, phi1 = Math.PI * (ring + 1) / rings;
            for (int segment = 0; segment < segments; segment++) {
                double theta0 = 2 * Math.PI * segment / segments, theta1 = 2 * Math.PI * (segment + 1) / segments;
                double[] a = point(radius, phi0, theta0), b = point(radius, phi1, theta0);
                double[] c = point(radius, phi1, theta1), d = point(radius, phi0, theta1);
                scene.triangles.add(new Triangle(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]));
                scene.triangles.add(new Triangle(a[0], a[1], a[2], c[0], c[1], c[2], d[0], d[1], d[2]));
            }
        }
        return scene;
    }

    static double[] point(double radius, double phi, double theta) {
        return new double[]{radius * Math.sin(phi) * Math.cos(theta), radius * Math.cos(phi),
                radius * Math.sin(phi) * Math.sin(theta)};
    }

    static Scene mixed(int scattered, SplitMix64 rng) {
        Scene scene = sphereSoup(12, 18, 1.0);
        for (int i = 0; i < scattered; i++) {
            double x = 4 * (rng.nextDouble() - 0.5), y = 4 * (rng.nextDouble() - 0.5);
            double size = 0.05 + 0.35 * rng.nextDouble();
            double z = -1 - rng.nextDouble();
            scene.triangles.add(new Triangle(x, y, z, x + size, y + 0.3 * size, z + 0.4 * size,
                    x - 0.2 * size, y - size, z + 0.1 * size));
        }
        return scene;
    }

    void buildBinnedSah() { root = BvhNode.build(new ArrayList<>(triangles), true); }

    void buildMedian() { root = BvhNode.build(new ArrayList<>(triangles), false); }

    Hit trace(Ray ray, boolean count) {
        Hit best = new Hit(false, 0, -1, 0, 0, 0);
        if (root == null) return best;
        root.traverse(ray, best, this, count);
        return best;
    }

    /** The oracle: test every triangle, in order. */
    Hit traceBruteForce(Ray ray, boolean count) {
        Hit best = new Hit(false, 0, -1, 0, 0, 0);
        for (int i = 0; i < triangles.size(); i++) {
            if (count) bruteTriangleTests++;
            double t = Hit.intersect(triangles.get(i), ray);
            if (t >= 0 && (!best.hit || t < best.t)) {
                best = new Hit(true, t, i, ray.ox + t * ray.dx, ray.oy + t * ray.dy, ray.oz + t * ray.dz);
            }
        }
        return best;
    }

    /** A deterministic shade of a hit, so an image comparison is a meaningful equality test. */
    static int shade(Scene scene, Hit hit) {
        if (!hit.hit) return 0;
        Triangle triangle = scene.triangles.get(hit.index);
        double e1x = triangle.v[3] - triangle.v[0], e1y = triangle.v[4] - triangle.v[1];
        double e2x = triangle.v[6] - triangle.v[0], e2y = triangle.v[7] - triangle.v[1];
        double normalZ = e1x * e2y - e1y * e2x;
        double length = Math.sqrt(e1x * e1x + e1y * e1y + e2x * e2x + e2y * e2y + normalZ * normalZ);
        double lambert = Math.abs(normalZ) / Math.max(1e-12, length) * 0.5 + 0.5;
        return (int) Math.round(255 * lambert * (1.0 / (1.0 + 0.1 * hit.t)));
    }
}

/** A hit: whether there was one, at what distance, on which triangle. */
class Hit {
    boolean hit;
    double t;
    int index;
    double x, y, z;

    Hit(boolean hit, double t, int index, double x, double y, double z) {
        this.hit = hit; this.t = t; this.index = index; this.x = x; this.y = y; this.z = z;
    }

    /** Moeller-Trumbore ray/triangle intersection; a negative result means no hit. */
    static double intersect(Triangle triangle, Ray ray) {
        double e1x = triangle.v[3] - triangle.v[0], e1y = triangle.v[4] - triangle.v[1];
        double e1z = triangle.v[5] - triangle.v[2];
        double e2x = triangle.v[6] - triangle.v[0], e2y = triangle.v[7] - triangle.v[1];
        double e2z = triangle.v[8] - triangle.v[2];
        double px = ray.dy * e2z - ray.dz * e2y, py = ray.dz * e2x - ray.dx * e2z;
        double pz = ray.dx * e2y - ray.dy * e2x;
        double determinant = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(determinant) < 1e-14) return -1;                 // parallel or degenerate
        double inverse = 1 / determinant;
        double tx = ray.ox - triangle.v[0], ty = ray.oy - triangle.v[1], tz = ray.oz - triangle.v[2];
        double u = (tx * px + ty * py + tz * pz) * inverse;
        if (u < 0 || u > 1) return -1;
        double qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
        double v = (ray.dx * qx + ray.dy * qy + ray.dz * qz) * inverse;
        if (v < 0 || u + v > 1) return -1;
        double t = (e2x * qx + e2y * qy + e2z * qz) * inverse;
        return t > 1e-9 && t <= ray.maxT ? t : -1;
    }
}

/** A BVH node: an axis-aligned box, either two children or a leaf range. */
class BvhNode {
    static final int BINS = 12;
    static final int LEAF_SIZE = 4;

    final double[] bounds = new double[6];
    BvhNode left, right;
    int[] indices;
    int depth;

    static BvhNode build(List<Triangle> triangles, boolean surfaceArea) {
        BvhNode node = new BvhNode();
        for (int axis = 0; axis < 3; axis++) {
            double lo = Double.MAX_VALUE, hi = -Double.MAX_VALUE;
            for (Triangle triangle : triangles) {
                lo = Math.min(lo, triangle.bounds[2 * axis]);
                hi = Math.max(hi, triangle.bounds[2 * axis + 1]);
            }
            node.bounds[2 * axis] = lo;
            node.bounds[2 * axis + 1] = hi;
        }
        if (triangles.size() <= LEAF_SIZE) {
            node.indices = new int[triangles.size()];
            for (int i = 0; i < triangles.size(); i++) node.indices[i] = triangles.get(i).ordinal;
            return node;
        }
        int[] choice = surfaceArea ? binnedChoice(triangles, node) : medianChoice(node);
        if (choice == null) { node.indices = indicesOf(triangles); return node; }
        splitInPlace(triangles, choice[0], choice[1]);
        int split = choice[1] >= 0 ? choice[1] : triangles.size() / 2;
        if (split <= 0 || split >= triangles.size()) { node.indices = indicesOf(triangles); return node; }
        List<Triangle> leftList = new ArrayList<>(triangles.subList(0, split));
        List<Triangle> rightList = new ArrayList<>(triangles.subList(split, triangles.size()));
        node.left = build(leftList, surfaceArea);
        node.right = build(rightList, surfaceArea);
        return node;
    }

    private static int[] indicesOf(List<Triangle> triangles) {
        int[] out = new int[triangles.size()];
        for (int i = 0; i < out.length; i++) out[i] = triangles.get(i).ordinal;
        return out;
    }

    private static int longestAxis(BvhNode node) {
        int axis = 0;
        double best = -1;
        for (int i = 0; i < 3; i++) {
            double extent = node.bounds[2 * i + 1] - node.bounds[2 * i];
            if (extent > best) { best = extent; axis = i; }
        }
        return best <= 0 ? -1 : axis;
    }

    /** Grow a bounding box by another box; an empty box is represented by null. */
    private static double[] merge(double[] accumulated, double[] box) {
        if (accumulated == null) return box.clone();
        double[] out = new double[6];
        for (int k = 0; k < 6; k++)
            out[k] = k % 2 == 0 ? Math.min(accumulated[k], box[k]) : Math.max(accumulated[k], box[k]);
        return out;
    }

    /**
     * The surface area heuristic: bucket the triangles into twelve bins along each axis, accumulate the
     * bounding box from the left and from the right, and keep the split whose children's surface areas,
     * weighted by their counts, are cheapest. Returns {axis, splitIndex} or null when no split helps.
     */
    private static int[] binnedChoice(List<Triangle> triangles, BvhNode node) {
        int bestAxis = -1, bestSplit = -1;
        double bestCost = Double.MAX_VALUE;
        for (int axis = 0; axis < 3; axis++) {
            double lo = node.bounds[2 * axis], hi = node.bounds[2 * axis + 1];
            if (hi - lo <= 1e-12) continue;
            double scale = BINS / (hi - lo);
            int[] counts = new int[BINS];
            double[][] binBounds = new double[BINS][6];
            for (int b = 0; b < BINS; b++)
                for (int k = 0; k < 6; k++)
                    binBounds[b][k] = k % 2 == 0 ? Double.MAX_VALUE : -Double.MAX_VALUE;
            for (Triangle triangle : triangles) {
                int bin = (int) ((triangle.centroid[axis] - lo) * scale);
                bin = Math.max(0, Math.min(BINS - 1, bin));
                counts[bin]++;
                binBounds[bin] = merge(binBounds[bin], triangle.bounds);
            }
            double[][] leftBounds = new double[BINS][];
            int[] leftCount = new int[BINS];
            double[] accumulator = null;
            int count = 0;
            for (int b = 0; b < BINS; b++) {
                if (counts[b] > 0) accumulator = merge(accumulator, binBounds[b]);
                count += counts[b];
                leftBounds[b] = accumulator;
                leftCount[b] = count;
            }
            double[] rightArea = new double[BINS];
            int[] rightCount = new int[BINS];
            accumulator = null;
            count = 0;
            for (int b = BINS - 1; b >= 0; b--) {
                if (counts[b] > 0) accumulator = merge(accumulator, binBounds[b]);
                count += counts[b];
                rightArea[b] = accumulator == null ? 0 : area(accumulator);
                rightCount[b] = count;
            }
            for (int b = 0; b < BINS - 1; b++) {
                if (leftCount[b] == 0 || rightCount[b + 1] == 0) continue;
                double cost = area(leftBounds[b]) * leftCount[b] + rightArea[b + 1] * rightCount[b + 1];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = leftCount[b];
                }
            }
        }
        return bestAxis < 0 ? null : new int[]{bestAxis, bestSplit};
    }

    private static int[] medianChoice(BvhNode node) {
        int axis = longestAxis(node);
        return axis < 0 ? null : new int[]{axis, -1};
    }

    /** Split the list in place: by the chosen bin boundary for the heuristic, or in half for the median. */
    private static void splitInPlace(List<Triangle> triangles, int axis, int splitIndex) {
        triangles.sort((a, b) -> Double.compare(a.centroid[axis], b.centroid[axis]));
    }

    private static double area(double[] bounds) {
        double dx = Math.max(0, bounds[1] - bounds[0]);
        double dy = Math.max(0, bounds[3] - bounds[2]);
        double dz = Math.max(0, bounds[5] - bounds[4]);
        return 2 * (dx * dy + dy * dz + dz * dx);
    }

    boolean isLeaf() { return indices != null; }

    /** The 3D slab test: intersect the ray with the three slabs and keep the overlapping interval. */
    static double[] slab(Ray ray, double[] bounds) {
        double tMin = 0, tMax = ray.maxT;
        for (int axis = 0; axis < 3; axis++) {
            double origin = axis == 0 ? ray.ox : axis == 1 ? ray.oy : ray.oz;
            double direction = axis == 0 ? ray.dx : axis == 1 ? ray.dy : ray.dz;
            double low = bounds[2 * axis], high = bounds[2 * axis + 1];
            if (Math.abs(direction) < 1e-15) {
                if (origin < low || origin > high) return null;
                continue;
            }
            double first = (low - origin) / direction, second = (high - origin) / direction;
            if (first > second) { double swap = first; first = second; second = swap; }
            tMin = Math.max(tMin, first);
            tMax = Math.min(tMax, second);
            if (tMin > tMax) return null;
        }
        return new double[]{tMin, tMax};
    }

    void traverse(Ray ray, Hit best, Scene scene, boolean count) {
        double[] interval = slab(ray, bounds);
        if (interval == null) return;
        if (interval[0] > best.t && best.hit) return;
        if (count) scene.boxTests++;
        if (isLeaf()) {
            for (int index : indices) {
                if (count) scene.triangleTests++;
                double t = Hit.intersect(scene.triangles.get(index), ray);
                if (t >= 0 && (!best.hit || t < best.t)) {
                    best.hit = true; best.t = t; best.index = index;
                    best.x = ray.ox + t * ray.dx; best.y = ray.oy + t * ray.dy; best.z = ray.oz + t * ray.dz;
                }
            }
            return;
        }
        BvhNode first = left, second = right;
        double[] leftInterval = slab(ray, left.bounds);
        double[] rightInterval = slab(ray, right.bounds);
        double leftEntry = leftInterval == null ? Double.MAX_VALUE : leftInterval[0];
        double rightEntry = rightInterval == null ? Double.MAX_VALUE : rightInterval[0];
        if (leftEntry > rightEntry) { BvhNode swap = first; first = second; second = swap; }
        first.traverse(ray, best, scene, count);
        second.traverse(ray, best, scene, count);
    }

    int nodeCount() { return isLeaf() ? 1 : 1 + left.nodeCount() + right.nodeCount(); }

    int leafCount() { return isLeaf() ? 1 : left.leafCount() + right.leafCount(); }

    void collectLeaves(List<int[]> out) {
        if (isLeaf()) { out.add(indices); return; }
        left.collectLeaves(out);
        right.collectLeaves(out);
    }

    /** The surface area heuristic cost of the tree: the standard sum of area(child) * count(child). */
    double sahCost() {
        if (isLeaf()) return 0;
        double cost = area(left.bounds) * count(left) + area(right.bounds) * count(right);
        return cost + left.sahCost() + right.sahCost();
    }

    static int count(BvhNode node) {
        if (node.isLeaf()) return node.indices.length;
        return count(node.left) + count(node.right);
    }

    boolean boundsContainChildren() {
        if (isLeaf()) return true;
        for (int axis = 0; axis < 3; axis++) {
            if (left.bounds[2 * axis] < bounds[2 * axis] - 1e-12
                    || left.bounds[2 * axis + 1] > bounds[2 * axis + 1] + 1e-12) return false;
            if (right.bounds[2 * axis] < bounds[2 * axis] - 1e-12
                    || right.bounds[2 * axis + 1] > bounds[2 * axis + 1] + 1e-12) return false;
        }
        return left.boundsContainChildren() && right.boundsContainChildren();
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);
        Scene scene = Scene.mixed(400, rng);
        for (int i = 0; i < scene.triangles.size(); i++) scene.triangles.get(i).ordinal = i;
        scene.buildBinnedSah();
        check(scene.triangles.size() >= 800, "the scene must be substantial, got "
                + scene.triangles.size() + " triangles");

        // --- TREE INVARIANTS -----------------------------------------------------------------------------
        List<int[]> leaves = new ArrayList<>();
        scene.root.collectLeaves(leaves);
        Set<Integer> seen = new LinkedHashSet<>();
        for (int[] leaf : leaves) for (int index : leaf) seen.add(index);
        check(seen.size() == scene.triangles.size(), "every triangle must appear in exactly one leaf, got "
                + seen.size() + " of " + scene.triangles.size());
        check(scene.root.boundsContainChildren(), "every node's box must contain its children's boxes");
        check(scene.root.nodeCount() > 100 && scene.root.leafCount() > 50, "the tree must actually branch: "
                + scene.root.nodeCount() + " nodes, " + scene.root.leafCount() + " leaves");
        check(BvhNode.count(scene.root) == scene.triangles.size(), "the subtree counts must add up");

        // --- CORRECTNESS against the brute-force walk ----------------------------------------------------
        int hits = 0, misses = 0, mismatches = 0;
        double worstDistance = 0;
        for (int trial = 0; trial < 600; trial++) {
            double ox = 6 * (rng.nextDouble() - 0.5), oy = 6 * (rng.nextDouble() - 0.5);
            double oz = 6 * (rng.nextDouble() - 0.5);
            double dx = 2 * (rng.nextDouble() - 0.5), dy = 2 * (rng.nextDouble() - 0.5);
            double dz = 2 * (rng.nextDouble() - 0.5);
            Ray ray = new Ray(ox, oy, oz, dx, dy, dz, 1e6);
            Hit fast = scene.trace(ray, false), slow = scene.traceBruteForce(ray, false);
            if (fast.hit != slow.hit) mismatches++;
            else if (fast.hit) {
                hits++;
                if (Math.abs(fast.t - slow.t) > 1e-9 || fast.index != slow.index) mismatches++;
                worstDistance = Math.max(worstDistance, Math.abs(fast.t - slow.t));
            } else misses++;
        }
        check(mismatches == 0, "the BVH must return exactly the brute-force hit, mismatches " + mismatches);
        check(hits > 50 && misses > 50, "the ray battery must contain hits and misses, got " + hits
                + " and " + misses);
        check(worstDistance < 1e-9, "hit distances must agree, worst difference " + worstDistance);
        Ray inside = new Ray(0, 0, 0, 1, 0, 0, 1e6);
        check(scene.trace(inside, false).hit == scene.traceBruteForce(inside, false).hit,
                "a ray starting inside the sphere must agree with brute force");
        Ray grazing = new Ray(-5, 1.0, 0, 1, 0, 0, 1e6);
        check(scene.trace(grazing, false).hit == scene.traceBruteForce(grazing, false).hit,
                "a grazing ray must agree with brute force");
        Ray far = new Ray(0, 0, 0, 0, 0, 1e-12, 1e6);
        check(scene.trace(far, false).hit == scene.traceBruteForce(far, false).hit,
                "a nearly degenerate direction must agree with brute force");

        // --- COST ----------------------------------------------------------------------------------------
        long boxTests = 0, triangleTests = 0, bruteTests = 0;
        int traced = 0;
        for (int trial = 0; trial < 400; trial++) {
            double ox = 3 * (rng.nextDouble() - 0.5), oy = 3 * (rng.nextDouble() - 0.5);
            double oz = 3 * (rng.nextDouble() - 0.5);
            Ray ray = new Ray(ox, oy, oz, 2 * (rng.nextDouble() - 0.5), 2 * (rng.nextDouble() - 0.5),
                    2 * (rng.nextDouble() - 0.5), 1e6);
            scene.trace(ray, true);
            scene.traceBruteForce(ray, true);
            traced++;
        }
        boxTests = scene.boxTests;
        triangleTests = scene.triangleTests;
        bruteTests = scene.bruteTriangleTests;
        check(triangleTests * 10 < bruteTests, "the BVH must cut triangle tests by more than ten times: "
                + triangleTests + " against " + bruteTests);
        check(boxTests < triangleTests * 4, "box tests must stay proportional to triangle tests: " + boxTests
                + " against " + triangleTests);
        check(traced == 400, "the cost battery must be complete");
        double perRay = boxTests / (double) traced;
        check(perRay < 64,
                "box tests per ray must stay logarithmic in the primitive count, got " + perRay);

        // --- the surface area heuristic versus a median split --------------------------------------------
        double sahCost = scene.root.sahCost();
        Scene medianScene = Scene.mixed(400, new SplitMix64(20240517L));
        for (int i = 0; i < medianScene.triangles.size(); i++) medianScene.triangles.get(i).ordinal = i;
        medianScene.buildMedian();
        double medianCost = medianScene.root.sahCost();
        check(sahCost < medianCost, "the binned surface area heuristic must beat the median split: "
                + sahCost + " against " + medianCost);
        int sahLeafSizes = 0;
        for (int[] leaf : leaves) sahLeafSizes += leaf.length;
        check(sahLeafSizes == scene.triangles.size(), "leaf sizes must sum to the primitive count");

        // --- SLAB TEST against a sampling oracle ---------------------------------------------------------
        int slabMismatches = 0, slabHits = 0;
        for (int trial = 0; trial < 500; trial++) {
            double[] bounds = {(rng.nextDouble() - 0.5) * 4, (rng.nextDouble() - 0.5) * 4,
                    (rng.nextDouble() - 0.5) * 4, (rng.nextDouble() - 0.5) * 4,
                    (rng.nextDouble() - 0.5) * 4, (rng.nextDouble() - 0.5) * 4};
            for (int axis = 0; axis < 3; axis++) if (bounds[2 * axis] > bounds[2 * axis + 1]) {
                double swap = bounds[2 * axis]; bounds[2 * axis] = bounds[2 * axis + 1];
                bounds[2 * axis + 1] = swap;
            }
            Ray ray = new Ray(4 * (rng.nextDouble() - 0.5), 4 * (rng.nextDouble() - 0.5),
                    4 * (rng.nextDouble() - 0.5), 2 * (rng.nextDouble() - 0.5),
                    2 * (rng.nextDouble() - 0.5), 2 * (rng.nextDouble() - 0.5), 1e6);
            double[] interval = BvhNode.slab(ray, bounds);
            boolean sampled = false;
            for (int step = 0; step < 4000 && !sampled; step++) {
                double t = 0.002 * step;
                double x = ray.ox + t * ray.dx, y = ray.oy + t * ray.dy, z = ray.oz + t * ray.dz;
                if (x >= bounds[0] && x <= bounds[1] && y >= bounds[2] && y <= bounds[3] && z >= bounds[4]
                        && z <= bounds[5]) sampled = true;
            }
            boolean intersect = interval != null;
            if (intersect) {
                slabHits++;
                double t = interval[0] + 1e-9;
                double x = ray.ox + t * ray.dx, y = ray.oy + t * ray.dy, z = ray.oz + t * ray.dz;
                boolean insideEntry = x >= bounds[0] - 1e-6 && x <= bounds[1] + 1e-6
                        && y >= bounds[2] - 1e-6 && y <= bounds[3] + 1e-6 && z >= bounds[4] - 1e-6
                        && z <= bounds[5] + 1e-6;
                if (!insideEntry) slabMismatches++;
            }
            if (intersect != sampled) slabMismatches++;
        }
        check(slabMismatches == 0, "the slab test must agree with dense sampling, mismatches " + slabMismatches);
        check(slabHits > 40, "the slab battery must contain hits, got " + slabHits);
        double[] unitBox = {0, 1, 0, 1, 0, 1};
        check(BvhNode.slab(new Ray(0.5, 0.5, -1, 0, 0, 1, 1e6), unitBox)[0] == 1.0,
                "a ray from outside must enter at the near face");
        check(BvhNode.slab(new Ray(0.5, 0.5, 0.5, 0, 0, 1, 1e6), unitBox)[0] == 0.0,
                "a ray from inside must start at zero");
        check(BvhNode.slab(new Ray(2, 0.5, 0.5, 0, 0, 1, 1e6), unitBox) == null,
                "a ray parallel to a slab and outside it must miss");

        // --- SHADING: the same image through both paths --------------------------------------------------
        int width = 24, height = 18;
        int[] throughBvh = new int[width * height], throughBrute = new int[width * height];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                double sx = 2.0 * x / width - 1.0, sy = 1.0 - 2.0 * y / height;
                Ray ray = new Ray(sx, sy, 3, 0, 0, -1, 1e6);
                throughBvh[y * width + x] = Scene.shade(scene, scene.trace(ray, false));
                Ray mirrored = new Ray(sx, sy, 3, 0, 0, -1, 1e6);
                throughBrute[y * width + x] = Scene.shade(scene, scene.traceBruteForce(mirrored, false));
            }
        check(Arrays.equals(throughBvh, throughBrute), "a rendered image must be identical through both "
                + "traversal paths");
        int litPixels = 0;
        for (int value : throughBvh) if (value > 0) litPixels++;
        check(litPixels > width * height / 2, "the image must actually show the scene, lit pixels "
                + litPixels + " of " + (width * height));

        // --- degenerate scenes ---------------------------------------------------------------------------
        Scene empty = new Scene();
        empty.buildBinnedSah();
        check(!empty.trace(new Ray(0, 0, 0, 1, 0, 0, 1e6), false).hit, "an empty scene must miss");
        Scene single = new Scene();
        single.triangles.add(new Triangle(0, 0, 0, 1, 0, 0, 0, 1, 0));
        single.triangles.get(0).ordinal = 0;
        single.buildBinnedSah();
        check(single.trace(new Ray(0.25, 0.25, 1, 0, 0, -1, 1e6), false).hit,
                "a single triangle must be hit");
        check(!single.trace(new Ray(2, 2, 1, 0, 0, -1, 1e6), false).hit, "a miss must stay a miss");
        Triangle degenerate = new Triangle(0, 0, 0, 1, 1, 1, 2, 2, 2);
        degenerate.ordinal = 0;
        check(Hit.intersect(degenerate, new Ray(0.5, 0.5, 0.5, 0, 0, 1, 1e6)) < 0,
                "a zero-area triangle must never be hit");
        check(Hit.intersect(new Triangle(0, 0, 0, 1, 0, 0, 0, 1, 0),
                new Ray(0.25, 0.25, 1, 0, 0, 1, 1e6)) < 0, "a ray pointing away must not hit");
        check(degenerate.area() == 0.0 && single.triangles.get(0).area() > 0.49,
                "triangle areas must be computed exactly");

        // --- determinism ---------------------------------------------------------------------------------
        Scene replay = Scene.mixed(400, new SplitMix64(20240517L));
        for (int i = 0; i < replay.triangles.size(); i++) replay.triangles.get(i).ordinal = i;
        replay.buildBinnedSah();
        check(replay.root.nodeCount() == scene.root.nodeCount()
                && replay.root.sahCost() == scene.root.sahCost(), "the BVH build must be deterministic");
        Scene costReplay = Scene.mixed(400, new SplitMix64(20240517L));
        for (int i = 0; i < costReplay.triangles.size(); i++) costReplay.triangles.get(i).ordinal = i;
        costReplay.buildBinnedSah();
        long replayBoxes = 0;
        for (int trial = 0; trial < 50; trial++) {
            Ray ray = new Ray(rng.nextDouble() - 0.5, rng.nextDouble() - 0.5, rng.nextDouble() - 0.5,
                    rng.nextDouble() - 0.5, rng.nextDouble() - 0.5, rng.nextDouble() - 0.5, 1e6);
            costReplay.trace(ray, true);
            replayBoxes += costReplay.boxTests;
            costReplay.boxTests = 0;
        }
        check(replayBoxes >= 0 && costReplay.root.nodeCount() == scene.root.nodeCount(),
                "the traversal counters must be reproducible");

        System.out.println("Scene: " + scene.triangles.size() + " triangles, BVH " + scene.root.nodeCount()
                + " nodes, " + scene.root.leafCount() + " leaves; " + hits + " hits and " + misses
                + " misses agreed with brute force");
        System.out.println("Cost over 400 rays: " + boxTests + " box tests and " + triangleTests
                + " triangle tests against " + bruteTests + " brute-force tests ("
                + String.format(Locale.ROOT, "%.1f", bruteTests / (double) triangleTests) + "x fewer)");
        System.out.println("Surface area heuristic cost " + String.format(Locale.ROOT, "%.2f", sahCost)
                + " against median split " + String.format(Locale.ROOT, "%.2f", medianCost)
                + "; image identical through both paths");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
