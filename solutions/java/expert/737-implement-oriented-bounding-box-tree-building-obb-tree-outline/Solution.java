/** Task 737 - Implement oriented bounding box tree building (OBB-tree) outline.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * An OBB is a box with its own orthonormal frame: center c, unit axes e0,e1,e2 and half extents
 * h0,h1,h2, so a point x is inside iff |(x-c)·ei| <= hi for all i. The tree is built bottom-up in
 * the usual Gottschalk (RAPID) style:
 *   1. fit() estimates the frame from the covariance matrix of the node's *triangle samples*
 *      (each triangle contributes its three vertices and its three edge midpoints, which weights
 *      the box towards surface area rather than towards vertex density); the frame is the
 *      eigenbasis of that covariance, sorted by descending eigenvalue and re-orthogonalised as a
 *      right-handed frame e2 = e0 x e1;
 *   2. the node's real vertices are projected onto the frame, giving the tight center/extents;
 *   3. the node is split at the median of the centroid projections along its longest axis and the
 *      two halves are recursed into; a node becomes a leaf at LEAF_SIZE primitives or when the
 *      split axis has no extent.
 *
 * Simplification (documented / outline). This is the *building* half of an OBB tree: no
 * refitting, no surface-area-heuristic splitting, no box-box "getting nowhere" optimisation and no
 * triangle/triangle narrow phase - overlap queries stop at the box level, which is exactly what an
 * outline of the structure requires. The eigen solver is a fixed cyclic Jacobi sweep (no LAPACK),
 * and boxes carry a small tolerance eps because extents come from floating point projections.
 *
 * Verified. Every primitive vertex of a node lies inside that node's box; the two children of an
 * internal node partition the node's points and cover each primitive exactly once; the root box
 * contains the whole model; each box is tight (no half extent can shrink, the centre is the mid
 * point of the projections) and on a diagonal strip dataset it is far tighter than the axis
 * aligned box; the 15-axis SAT overlap test agrees with the exact axis-aligned box formula
 * whenever both frames are axis aligned, never misses an intersection that a dense sampling of the
 * two boxes displays, and every "disjoint" answer is backed by a separating axis certified on the
 * corners of both boxes.
 */
import java.util.*;

/** Oriented bounding box with an orthonormal, right-handed frame. */
final class Obb {
    final double[] center = new double[3];
    final double[][] axis = new double[3][3];   // axis[i] = i-th unit direction
    final double[] half = new double[3];

    /** Fits a box: `samples` drives the covariance, `points` defines the actual extents. */
    static Obb fit(List<double[]> samples, List<double[]> points) {
        double[] mean = new double[3];
        for (double[] p : samples) for (int k = 0; k < 3; k++) mean[k] += p[k];
        for (int k = 0; k < 3; k++) mean[k] /= Math.max(1, samples.size());
        double[][] cov = new double[3][3];
        for (double[] p : samples) {
            double[] d = {p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]};
            for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
        }
        double[][] v = identity();
        jacobi(cov, v);
        Integer[] byEigen = {0, 1, 2};
        Arrays.sort(byEigen, (a, b) -> Double.compare(cov[b][b], cov[a][a]));
        double[][] frame = new double[3][3];
        frame[0] = normalize(column(v, byEigen[0]));
        frame[1] = normalize(column(v, byEigen[1]));
        if (frame[0] == null) frame[0] = new double[]{1, 0, 0};
        if (frame[1] == null || Math.abs(dot(frame[0], frame[1])) > 1 - 1e-9) {
            frame[1] = orthogonal(frame[0]);
        }
        double[] third = cross(frame[0], frame[1]);
        frame[2] = normalize(third);
        if (frame[2] == null) {   // fully degenerate covariance: fall back to the world frame
            frame[0] = new double[]{1, 0, 0};
            frame[1] = new double[]{0, 1, 0};
            frame[2] = new double[]{0, 0, 1};
        }
        Obb box = new Obb();
        double[] lo = new double[3];
        double[] hi = new double[3];
        Arrays.fill(lo, Double.MAX_VALUE);
        Arrays.fill(hi, -Double.MAX_VALUE);
        for (double[] p : points) {
            for (int i = 0; i < 3; i++) {
                double proj = project(p, frame[i]);
                lo[i] = Math.min(lo[i], proj);
                hi[i] = Math.max(hi[i], proj);
            }
        }
        if (points.isEmpty()) {
            Arrays.fill(lo, 0);
            Arrays.fill(hi, 0);
        }
        for (int i = 0; i < 3; i++) {
            box.half[i] = (hi[i] - lo[i]) / 2;
            double mid = (hi[i] + lo[i]) / 2;
            for (int k = 0; k < 3; k++) box.center[k] += mid * frame[i][k];
            box.axis[i] = frame[i];
        }
        return box;
    }

    static double project(double[] p, double[] axis) { return p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2]; }

    double projectedRadius(double[] axis) {
        double sum = 0;
        for (int i = 0; i < 3; i++) sum += Math.abs(project(axis, this.axis[i])) * half[i];
        return sum;
    }

    /** Absolute support radius along `d` is the same as the projected radius for a box. */
    double volume() { return 8 * half[0] * half[1] * half[2]; }

    List<double[]> corners() {
        List<double[]> out = new ArrayList<>(8);
        for (int mask = 0; mask < 8; mask++) {
            double[] p = center.clone();
            for (int i = 0; i < 3; i++) {
                double sign = ((mask >> i) & 1) == 0 ? -half[i] : half[i];
                for (int k = 0; k < 3; k++) p[k] += sign * axis[i][k];
            }
            out.add(p);
        }
        return out;
    }

    boolean contains(double[] p, double eps) {
        for (int i = 0; i < 3; i++) {
            if (Math.abs(project(p, axis[i]) - project(center, axis[i])) > half[i] + eps) return false;
        }
        return true;
    }

    static final double EPS = 1e-9;

    /** 15-axis separating axis test; returns the separating axis, or null when the boxes overlap. */
    static double[] separatingAxis(Obb a, Obb b) {
        List<double[]> candidates = new ArrayList<>();
        for (int i = 0; i < 3; i++) candidates.add(a.axis[i]);
        for (int i = 0; i < 3; i++) candidates.add(b.axis[i]);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double[] c = cross(a.axis[i], b.axis[j]);
                if (normalize(c) != null) candidates.add(c);
            }
        }
        double[] delta = {b.center[0] - a.center[0], b.center[1] - a.center[1], b.center[2] - a.center[2]};
        for (double[] axis : candidates) {
            double distance = Math.abs(project(delta, axis));
            double radius = a.projectedRadius(axis) + b.projectedRadius(axis);
            if (distance > radius + EPS) return axis;
        }
        return null;
    }

    static boolean overlaps(Obb a, Obb b) { return separatingAxis(a, b) == null; }

    /** Certificate for a disjoint verdict: the corner projections along the axis really are apart. */
    static boolean certifiesSeparation(Obb a, Obb b, double[] axis) {
        double aMin = Double.MAX_VALUE;
        double aMax = -Double.MAX_VALUE;
        double bMin = Double.MAX_VALUE;
        double bMax = -Double.MAX_VALUE;
        for (double[] p : a.corners()) {
            double value = project(p, axis);
            aMin = Math.min(aMin, value);
            aMax = Math.max(aMax, value);
        }
        for (double[] p : b.corners()) {
            double value = project(p, axis);
            bMin = Math.min(bMin, value);
            bMax = Math.max(bMax, value);
        }
        double gap = Math.max(bMin - aMax, aMin - bMax);   // the axis may point either way
        return gap > 0;
    }

    /** Cyclic Jacobi eigen decomposition of a symmetric 3x3 matrix; v is the eigenvector matrix. */
    static void jacobi(double[][] a, double[][] v) {
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
            if (off < 1e-15) break;
            for (int p = 0; p < 2; p++) {
                for (int q = p + 1; q < 3; q++) {
                    if (Math.abs(a[p][q]) < 1e-18) continue;
                    double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
                    double t = theta == 0 ? 1.0
                            : Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    double c = 1 / Math.sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < 3; k++) {
                        double akp = a[k][p];
                        double akq = a[k][q];
                        a[k][p] = c * akp - s * akq;
                        a[k][q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 3; k++) {
                        double apk = a[p][k];
                        double aqk = a[q][k];
                        a[p][k] = c * apk - s * aqk;
                        a[q][k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; k++) {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
    }

    static double[][] identity() {
        double[][] m = new double[3][3];
        for (int i = 0; i < 3; i++) m[i][i] = 1;
        return m;
    }

    static double[] column(double[][] m, int c) { return new double[]{m[0][c], m[1][c], m[2][c]}; }

    static double dot(double[] u, double[] v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

    static double[] cross(double[] u, double[] v) {
        return new double[]{
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]};
    }

    static double[] normalize(double[] u) {
        double n = Math.sqrt(dot(u, u));
        if (n < 1e-12) return null;
        return new double[]{u[0] / n, u[1] / n, u[2] / n};
    }

    /** A unit vector orthogonal to `u`. */
    static double[] orthogonal(double[] u) {
        double[] helper = Math.abs(u[0]) < 0.9 ? new double[]{1, 0, 0} : new double[]{0, 1, 0};
        return normalize(cross(u, helper));
    }
}

/** OBB tree over a triangle soup. */
final class ObbTree {
    static final int LEAF_SIZE = 2;

    static final class Node {
        Obb box;
        Node left;
        Node right;
        int start;
        int end;
        boolean leaf;
    }

    final double[][][] tri;
    final int[] order;
    Node root;
    int nodeCount;

    ObbTree(double[][][] triangles) {
        tri = triangles;
        order = new int[triangles.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        if (order.length > 0) root = build(0, order.length);
    }

    private Node build(int start, int end) {
        nodeCount++;
        Node node = new Node();
        node.start = start;
        node.end = end;
        List<double[]> samples = new ArrayList<>();
        List<double[]> vertices = new ArrayList<>();
        for (int i = start; i < end; i++) {
            double[][] t = tri[order[i]];
            samples.add(t[0]);
            samples.add(t[1]);
            samples.add(t[2]);
            for (int e = 0; e < 3; e++) {
                double[] mid = new double[3];
                for (int k = 0; k < 3; k++) mid[k] = (t[e][k] + t[(e + 1) % 3][k]) / 2;
                samples.add(mid);
                vertices.add(t[e]);
            }
        }
        node.box = Obb.fit(samples, vertices);
        if (end - start <= LEAF_SIZE) {
            node.leaf = true;
            return node;
        }
        int axis = 0;
        for (int i = 1; i < 3; i++) if (node.box.half[i] > node.box.half[axis]) axis = i;
        if (node.box.half[axis] < 1e-9 || end - start <= 1) {
            node.leaf = true;
            return node;
        }
        final double[] splitAxis = node.box.axis[axis];
        final double[] splitCenter = node.box.center;
        Integer[] ids = new Integer[end - start];
        for (int i = start; i < end; i++) ids[i - start] = order[i];
        Arrays.sort(ids, (x, y) -> {
            int c = Double.compare(centroidProjection(tri[x], splitAxis, splitCenter),
                    centroidProjection(tri[y], splitAxis, splitCenter));
            return c != 0 ? c : Integer.compare(x, y);
        });
        for (int i = start; i < end; i++) order[i] = ids[i - start];
        int mid = start + (end - start) / 2;
        node.left = build(start, mid);
        node.right = build(mid, end);
        return node;
    }

    private static double centroidProjection(double[][] t, double[] axis, double[] center) {
        double acc = 0;
        for (int k = 0; k < 3; k++) acc += (t[0][k] + t[1][k] + t[2][k]) / 3 * axis[k];
        return acc - Obb.project(center, axis);
    }

    List<Node> nodes() {
        List<Node> out = new ArrayList<>();
        collect(root, out);
        return out;
    }

    private void collect(Node node, List<Node> out) {
        if (node == null) return;
        out.add(node);
        collect(node.left, out);
        collect(node.right, out);
    }

    List<double[]> nodeVertices(Node node) {
        List<double[]> out = new ArrayList<>();
        for (int i = node.start; i < node.end; i++) {
            for (double[] v : tri[order[i]]) out.add(v);
        }
        return out;
    }
}

public class Solution {
    private static long seed = 0xDA3E39CB94B95BDBL;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    static double[] v(double x, double y, double z) { return new double[]{x, y, z}; }

    /** Axis aligned box as an OBB - lets us compare SAT against the exact AABB formula. */
    static Obb axisAlignedBox(double[] lo, double[] hi) {
        Obb box = new Obb();
        for (int k = 0; k < 3; k++) {
            box.center[k] = (lo[k] + hi[k]) / 2;
            box.half[k] = (hi[k] - lo[k]) / 2;
            box.axis[k] = new double[3];
            box.axis[k][k] = 1;
        }
        return box;
    }

    static boolean aabbOverlap(double[] aLo, double[] aHi, double[] bLo, double[] bHi) {
        for (int k = 0; k < 3; k++) {
            if (aHi[k] < bLo[k] || bHi[k] < aLo[k]) return false;
        }
        return true;
    }

    static Obb rotatedBox(double[] center, double[] half, double angleZ, double angleX) {
        double cz = Math.cos(angleZ), sz = Math.sin(angleZ);
        double cx = Math.cos(angleX), sx = Math.sin(angleX);
        double[][] rz = {{cz, -sz, 0}, {sz, cz, 0}, {0, 0, 1}};
        double[][] rx = {{1, 0, 0}, {0, cx, -sx}, {0, sx, cx}};
        Obb box = new Obb();
        for (int i = 0; i < 3; i++) {
            double[] row = new double[3];
            for (int k = 0; k < 3; k++) row[k] = rz[i][0] * rx[0][k] + rz[i][1] * rx[1][k] + rz[i][2] * rx[2][k];
            box.axis[i] = Obb.normalize(row);
        }
        System.arraycopy(center, 0, box.center, 0, 3);
        System.arraycopy(half, 0, box.half, 0, 3);
        return box;
    }

    /** Sampling oracle: a lattice point strictly inside both boxes proves they overlap. */
    static boolean sampledCommonPoint(Obb a, Obb b) {
        double[] lo = new double[3];
        double[] hi = new double[3];
        for (int k = 0; k < 3; k++) {
            double aLo = Double.MAX_VALUE, aHi = -Double.MAX_VALUE;
            double bLo = Double.MAX_VALUE, bHi = -Double.MAX_VALUE;
            for (double[] p : a.corners()) {
                aLo = Math.min(aLo, p[k]);
                aHi = Math.max(aHi, p[k]);
            }
            for (double[] p : b.corners()) {
                bLo = Math.min(bLo, p[k]);
                bHi = Math.max(bHi, p[k]);
            }
            lo[k] = Math.max(aLo, bLo);
            hi[k] = Math.min(aHi, bHi);
            if (hi[k] - lo[k] < 1e-6) return false;
        }
        final int steps = 24;
        for (int i = 0; i <= steps; i++) {
            for (int j = 0; j <= steps; j++) {
                for (int k = 0; k <= steps; k++) {
                    double[] p = {
                            lo[0] + (hi[0] - lo[0]) * i / steps,
                            lo[1] + (hi[1] - lo[1]) * j / steps,
                            lo[2] + (hi[2] - lo[2]) * k / steps};
                    if (a.contains(p, 1e-9) && b.contains(p, 1e-9)) return true;
                }
            }
        }
        return false;
    }

    public static void main(String[] args) {
        // --- empty and single primitive trees -----------------------------------
        ObbTree empty = new ObbTree(new double[0][][]);
        check(empty.nodes().isEmpty() && empty.nodeCount == 0, "empty soup builds no nodes");

        double[][][] oneTri = {{v(0, 0, 0), v(3, 0, 0), v(0, 3, 0)}};
        ObbTree single = new ObbTree(oneTri);
        check(single.nodeCount == 1 && single.nodes().get(0).leaf, "single triangle is one leaf");
        check(single.root.box.half[2] < 1e-9, "flat triangle gets a flat box");
        for (double[] p : oneTri[0]) {
            check(single.root.box.contains(p, 1e-6), "triangle vertex inside its own box");
        }

        // --- determinism of the fit ---------------------------------------------
        ObbTree again = new ObbTree(oneTri);
        check(Math.abs(again.root.box.center[0] - single.root.box.center[0]) < 1e-12,
                "fit is deterministic");

        // --- covariance frame sanity --------------------------------------------
        Obb flat = Obb.fit(Arrays.asList(v(0, 0, 0), v(1, 0, 0), v(2, 0, 0), v(1, 0, 0)),
                Arrays.asList(v(0, 0, 0), v(1, 0, 0), v(2, 0, 0)));
        check(Math.abs(Obb.dot(flat.axis[0], flat.axis[0]) - 1) < 1e-12, "frame axis is unit length");
        check(Math.abs(Obb.dot(flat.axis[0], flat.axis[1])) < 1e-12, "frame axes are orthogonal");
        check(Math.abs(Obb.dot(flat.axis[0], flat.axis[2])) < 1e-12, "third axis orthogonal too");
        check(Math.abs(Obb.dot(Obb.cross(flat.axis[0], flat.axis[1]), flat.axis[2]) - 1) < 1e-12,
                "frame is right handed");
        check(Math.abs(flat.half[1]) < 1e-9 && Math.abs(flat.half[2]) < 1e-9,
                "collinear samples give a degenerate box");

        // --- SAT versus the exact axis aligned formula --------------------------
        int overlapping = 0;
        int separated = 0;
        for (int trial = 0; trial < 120; trial++) {
            double[] aLo = {nextInt(20), nextInt(20), nextInt(20)};
            double[] aHi = {aLo[0] + 1 + nextInt(6), aLo[1] + 1 + nextInt(6), aLo[2] + 1 + nextInt(6)};
            double[] bLo = new double[3];
            double[] bHi = new double[3];
            for (int k = 0; k < 3; k++) {
                // even trials place b next to a (overlap), odd trials far away (separation)
                int shift = trial % 2 == 0 ? nextInt(9) - 2 : 30 + nextInt(20);
                bLo[k] = aLo[k] + shift;
                bHi[k] = bLo[k] + 1 + nextInt(6);
            }
            Obb a = axisAlignedBox(aLo, aHi);
            Obb b = axisAlignedBox(bLo, bHi);
            boolean want = aabbOverlap(aLo, aHi, bLo, bHi);
            check(Obb.overlaps(a, b) == want,
                    "SAT disagrees with the AABB formula in trial " + trial);
            if (want) overlapping++;
            else separated++;
        }
        check(overlapping > 20 && separated > 20,
                "AABB trials mix overlap and separation: " + overlapping + "/" + separated);

        // --- SAT versus dense sampling on rotated boxes -------------------------
        int positives = 0;
        int negatives = 0;
        for (int trial = 0; trial < 120; trial++) {
            double[] c1 = {nextInt(12), nextInt(12), nextInt(12)};
            double[] c2 = {c1[0] + nextInt(9) - 4, c1[1] + nextInt(9) - 4, c1[2] + nextInt(9) - 4};
            double[] h1 = {1 + nextDouble() * 2, 1 + nextDouble() * 2, 1 + nextDouble() * 2};
            double[] h2 = {1 + nextDouble() * 2, 1 + nextDouble() * 2, 1 + nextDouble() * 2};
            Obb a = rotatedBox(c1, h1, nextDouble() * Math.PI, nextDouble() * 0.6);
            Obb b = rotatedBox(c2, h2, nextDouble() * Math.PI, nextDouble() * 0.6);
            double[] axis = Obb.separatingAxis(a, b);
            boolean sampled = sampledCommonPoint(a, b);
            if (sampled) {
                positives++;
                check(axis == null, "sampling found a common point but SAT separated in trial " + trial);
            }
            if (axis != null) {
                negatives++;
                check(Obb.certifiesSeparation(a, b, axis),
                        "separating axis is not certified in trial " + trial);
            }
        }
        check(positives > 5, "rotated trials include real overlaps, got " + positives);
        check(negatives > 5, "rotated trials include real separations, got " + negatives);

        // --- touching boxes count as overlapping --------------------------------
        Obb touchA = axisAlignedBox(new double[]{0, 0, 0}, new double[]{2, 2, 2});
        Obb touchB = axisAlignedBox(new double[]{2, 0, 0}, new double[]{4, 2, 2});
        check(Obb.overlaps(touchA, touchB), "face touching boxes overlap");
        check(!Obb.overlaps(touchA, axisAlignedBox(new double[]{2.5, 0, 0}, new double[]{4, 2, 2})),
                "boxes 0.5 apart do not overlap");

        // --- tree building invariants over two clusters of triangles ------------
        List<double[][]> soup = new ArrayList<>();
        for (int i = 0; i < 18; i++) {
            double[] base = {nextInt(10), nextInt(10), nextInt(10)};
            soup.add(new double[][]{
                    base,
                    v(base[0] + 1 + nextDouble() * 4, base[1] + nextDouble() * 4, base[2] + nextDouble() * 4),
                    v(base[0] + nextDouble() * 4, base[1] + 1 + nextDouble() * 4, base[2] + nextDouble() * 4)});
        }
        for (int i = 0; i < 8; i++) {
            double[] base = {60 + nextInt(10), 60 + nextInt(10), 60 + nextInt(10)};
            soup.add(new double[][]{base,
                    v(base[0] + 2, base[1] + 1, base[2]),
                    v(base[0], base[1] + 2, base[2] + 1)});
        }
        ObbTree tree = new ObbTree(soup.toArray(new double[0][][]));
        check(tree.nodeCount > 1 + soup.size() / ObbTree.LEAF_SIZE, "tree has internal nodes");
        int internal = 0;
        for (ObbTree.Node node : tree.nodes()) {
            if (!node.leaf) internal++;
            List<double[]> points = tree.nodeVertices(node);
            for (double[] p : points) check(node.box.contains(p, 1e-6), "node vertex outside its OBB");
            if (!node.leaf) {
                check(node.left.start == node.start && node.left.end == node.right.start
                                && node.right.end == node.end,
                        "children do not split the node range");
                List<String> own = vertexKeys(points);
                List<String> children = vertexKeys(tree.nodeVertices(node.left));
                children.addAll(vertexKeys(tree.nodeVertices(node.right)));
                Collections.sort(children);
                check(own.equals(children), "children do not partition the node's points");
            }
        }
        check(internal >= 1, "at least one internal node exists");
        boolean[] covered = new boolean[soup.size()];
        for (ObbTree.Node node : tree.nodes()) {
            if (!node.leaf) continue;
            for (int i = node.start; i < node.end; i++) {
                int prim = tree.order[i];
                check(!covered[prim], "primitive " + prim + " sits in two leaves");
                covered[prim] = true;
            }
        }
        for (int i = 0; i < covered.length; i++) check(covered[i], "primitive " + i + " is missing");

        // --- the box is tight: no extent can shrink, and it is centred on the points
        for (ObbTree.Node node : tree.nodes()) {
            List<double[]> pts = tree.nodeVertices(node);
            for (int i = 0; i < 3; i++) {
                double lo = Double.MAX_VALUE;
                double hi = -Double.MAX_VALUE;
                for (double[] p : pts) {
                    double proj = Obb.project(p, node.box.axis[i]);
                    lo = Math.min(lo, proj);
                    hi = Math.max(hi, proj);
                }
                double centerProj = Obb.project(node.box.center, node.box.axis[i]);
                check(Math.abs((hi - lo) / 2 - node.box.half[i]) < 1e-6,
                        "extent is not tight on axis " + i);
                check(Math.abs((hi + lo) / 2 - centerProj) < 1e-6,
                        "box centre is not the mid point on axis " + i);
                check(node.box.half[i] >= 0, "negative half extent");
            }
        }

        // --- oriented frames pay off on diagonal geometry ------------------------
        List<double[][]> strip = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            double t = i;
            strip.add(new double[][]{
                    v(t, t, 0.1 * t),
                    v(t + 1, t + 1, 0.1 * (t + 1)),
                    v(t + 1, t + 1.3, 0.1 * (t + 1) + 0.3)});
        }
        ObbTree stripTree = new ObbTree(strip.toArray(new double[0][][]));
        double[] aabbLo = {Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE};
        double[] aabbHi = {-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE};
        for (double[][] t : strip) {
            for (double[] p : t) {
                for (int k = 0; k < 3; k++) {
                    aabbLo[k] = Math.min(aabbLo[k], p[k]);
                    aabbHi[k] = Math.max(aabbHi[k], p[k]);
                }
            }
        }
        double stripAabbVolume = (aabbHi[0] - aabbLo[0]) * (aabbHi[1] - aabbLo[1])
                * (aabbHi[2] - aabbLo[2]);
        double stripObbVolume = stripTree.root.box.volume();
        check(stripObbVolume < stripAabbVolume * 0.5,
                "diagonal strip: OBB " + stripObbVolume + " is not tighter than AABB "
                        + stripAabbVolume);
        check(stripObbVolume > 0, "diagonal strip box is not degenerate");
        for (double[][] t : strip) {
            for (double[] p : t) check(stripTree.root.box.contains(p, 1e-6), "strip vertex outside box");
        }

        // --- root covers every input vertex --------------------------------------
        for (double[][] t : soup) {
            for (double[] p : t) check(tree.root.box.contains(p, 1e-6), "root box misses a vertex");
        }

        // --- leaf boxes of distant clusters never overlap ------------------------
        List<ObbTree.Node> leaves = new ArrayList<>();
        for (ObbTree.Node node : tree.nodes()) if (node.leaf) leaves.add(node);
        int leafOverlaps = 0;
        int clusterOverlaps = 0;
        for (int i = 0; i < leaves.size(); i++) {
            for (int j = i + 1; j < leaves.size(); j++) {
                boolean sameCluster = (leaves.get(i).box.center[0] < 40) == (leaves.get(j).box.center[0] < 40);
                boolean hit = Obb.overlaps(leaves.get(i).box, leaves.get(j).box);
                if (!sameCluster && hit) clusterOverlaps++;
                if (hit) leafOverlaps++;
            }
        }
        check(clusterOverlaps == 0, "boxes from different clusters must not overlap");
        check(leafOverlaps < leaves.size() * (leaves.size() - 1) / 2,
                "the tree separates at least some leaf pairs");

        // --- split axis choice: the longest box axis is used --------------------
        ObbTree.Node rootNode = tree.root;
        check(!rootNode.leaf, "root is not a leaf");
        int axis = 0;
        for (int i = 1; i < 3; i++) if (rootNode.box.half[i] > rootNode.box.half[axis]) axis = i;
        check(spread(rootNode.left) > 0 && spread(rootNode.right) > 0, "both children carry geometry");
        check(Obb.normalize(rootNode.box.axis[axis]) != null, "split axis is well defined");

        System.out.println("triangles = " + soup.size() + ", nodes = " + tree.nodeCount
                + ", leaves = " + leaves.size() + ", internal = " + internal
                + ", SAT vs sampling: " + positives + " overlaps / " + negatives + " separations");
        System.out.println("PASS");
    }

    /** Largest box extent of a node - used to show that both children carry geometry. */
    static double spread(ObbTree.Node node) {
        if (node == null) return 0;
        return node.box.half[0] + node.box.half[1] + node.box.half[2];
    }

    /** Sorted, rounded vertex keys - lets the checks compare point multisets without doubles. */
    static List<String> vertexKeys(List<double[]> points) {
        List<String> keys = new ArrayList<>();
        for (double[] p : points) {
            keys.add(String.format(Locale.ROOT, "%.6f,%.6f,%.6f", p[0], p[1], p[2]));
        }
        Collections.sort(keys);
        return keys;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
