/** Task 736 - Implement intersection detection in 3D using bounding volume hierarchies.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Structure. A binary AABB tree over the triangle soup: nodes are built top-down by splitting the
 * primitives at the median of their centroids along the widest centroid extent, so the tree stays
 * balanced and every node box is the tight box of its children. Intersection detection is the
 * classic two-phase scheme:
 *   broad phase  - traverse the two node sets simultaneously; a node pair whose boxes are disjoint
 *                  is *pruned* (nothing inside can intersect), otherwise descend the larger node.
 *                  A leaf/leaf pair is expanded into primitive pairs, again box-filtered first.
 *   narrow phase - exact triangle/triangle intersection test:
 *                    1. signed distances of T2's vertices to T1's plane; all on one strict side
 *                       (and symmetric) rejects immediately;
 *                    2. coplanar case (n1 x n2 == 0): separating-axis test on the 6 edge normals
 *                       of the projected 2D triangles, exact with longs;
 *                    3. otherwise: the planes meet in the line L = n1 x n2; both triangles cut
 *                       that line in an interval, and they intersect iff the intervals overlap.
 * Each interior vertex of the interval is a rational point V_i + t*(V_j - V_i) with
 * t = d_i / (d_i - d_j); intervals are compared as exact BigInteger fractions, so touching
 * contacts, near-coplanar slivers and exactly-cocircular degeneracies are classified without
 * an epsilon.
 *
 * Simplification (documented). Only axis-aligned boxes (task 737 covers OBB trees), one static
 * tree per soup with no refit/rotate during traversal, no triangle stripping, single-threaded,
 * and rotations/deletions of primitives are not supported. Triangle coordinates are bounded small
 * integers so that the long products in the sign tests stay exact (they are; only the interval
 * endpoints need BigIntegers). The invariant verified is *pair exactness*: the set of reported
 * intersecting primitive pairs equals the O(n^2) brute-force result on the same input, and the
 * tree genuinely prunes work (node-pair tests far below n*(n-1)/2 on a clustered input).
 */
import java.util.*;
import java.math.BigInteger;

/** Exact rational value, denominator strictly positive, always reduced. */
final class Frac {
    final BigInteger num;
    final BigInteger den;

    private Frac(BigInteger num, BigInteger den) {
        this.num = num;
        this.den = den;
    }

    static Frac of(long value) {
        return new Frac(BigInteger.valueOf(value), BigInteger.ONE);
    }

    static Frac of(long numerator, long denominator) {
        if (denominator == 0) throw new ArithmeticException("zero denominator");
        BigInteger n = BigInteger.valueOf(numerator);
        BigInteger d = BigInteger.valueOf(denominator);
        if (d.signum() < 0) { n = n.negate(); d = d.negate(); }
        BigInteger g = n.gcd(d);
        if (g.signum() != 0 && !g.equals(BigInteger.ONE)) { n = n.divide(g); d = d.divide(g); }
        return new Frac(n, d);
    }

    Frac add(Frac o) { return of(num.multiply(o.den).add(o.num.multiply(den)), den.multiply(o.den)); }
    Frac sub(Frac o) { return of(num.multiply(o.den).subtract(o.num.multiply(den)), den.multiply(o.den)); }
    Frac mul(Frac o) { return of(num.multiply(o.num), den.multiply(o.den)); }
    int compareTo(Frac o) { return num.multiply(o.den).compareTo(o.num.multiply(den)); }

    private static Frac of(BigInteger n, BigInteger d) {
        if (d.signum() < 0) { n = n.negate(); d = d.negate(); }
        BigInteger g = n.gcd(d);
        if (g.signum() != 0 && !g.equals(BigInteger.ONE)) { n = n.divide(g); d = d.divide(g); }
        return new Frac(n, d);
    }

    @Override public String toString() { return num + "/" + den; }
}

/** AABB tree over triangles, plus exact intersection queries. */
final class Bvh {
    static final int LEAF_SIZE = 4;

    static final class Node {
        long[] lo;
        long[] hi;
        int left = -1;
        int right = -1;
        int start;
        int end;
        boolean leaf;
    }

    private final long[][][] tri;       // {a, b, c} per primitive
    private final long[][] primLo;
    private final long[][] primHi;
    private final long[][] centroid;
    private final int[] order;          // primitive ids, permuted so leaves are contiguous
    private final List<Node> nodes = new ArrayList<>();

    long boxTests;        // node pairs (or primitive pairs) whose boxes were actually tested
    long narrowTests;     // triangle/triangle tests actually executed

    Bvh(long[][][] triangles) {
        this.tri = triangles;
        int n = triangles.length;
        primLo = new long[n][];
        primHi = new long[n][];
        centroid = new long[n][];
        for (int i = 0; i < n; i++) {
            long[] box = Tri.boxOf(triangles[i]);
            primLo[i] = new long[]{box[0], box[1], box[2]};
            primHi[i] = new long[]{box[3], box[4], box[5]};
            centroid[i] = new long[]{
                    (box[0] + box[3]) / 2, (box[1] + box[4]) / 2, (box[2] + box[5]) / 2};
        }
        order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        if (n > 0) build(0, n);
    }

    private int build(int start, int end) {
        Node node = new Node();
        int id = nodes.size();
        nodes.add(node);
        node.start = start;
        node.end = end;
        node.lo = new long[]{Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE};
        node.hi = new long[]{Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE};
        long[] cLo = {Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE};
        long[] cHi = {Long.MIN_VALUE, Long.MIN_VALUE, Long.MIN_VALUE};
        for (int i = start; i < end; i++) {
            int p = order[i];
            for (int k = 0; k < 3; k++) {
                node.lo[k] = Math.min(node.lo[k], primLo[p][k]);
                node.hi[k] = Math.max(node.hi[k], primHi[p][k]);
                cLo[k] = Math.min(cLo[k], centroid[p][k]);
                cHi[k] = Math.max(cHi[k], centroid[p][k]);
            }
        }
        if (end - start <= LEAF_SIZE) {
            node.leaf = true;
            return id;
        }
        int axis = 0;
        for (int k = 1; k < 3; k++) if (cHi[k] - cLo[k] > cHi[axis] - cLo[axis]) axis = k;
        if (cHi[axis] == cLo[axis]) {   // all centroids coincide on every axis: cannot split
            node.leaf = true;
            return id;
        }
        final int keyAxis = axis;
        Integer[] ids = new Integer[end - start];
        for (int i = start; i < end; i++) ids[i - start] = order[i];
        Arrays.sort(ids, (x, y) -> {
            int c = Long.compare(centroid[x][keyAxis], centroid[y][keyAxis]);
            return c != 0 ? c : Integer.compare(x, y);
        });
        for (int i = start; i < end; i++) order[i] = ids[i - start];
        int mid = start + (end - start) / 2;
        node.left = build(start, mid);
        node.right = build(mid, end);
        return id;
    }

    int rootIndex() { return nodes.isEmpty() ? -1 : 0; }
    int nodeCount() { return nodes.size(); }

    /** All intersecting primitive pairs, as sorted {i, j} with i < j. */
    List<long[]> intersectingPairs() {
        List<long[]> out = new ArrayList<>();
        if (!nodes.isEmpty()) collectPairs(0, 0, out);
        out.sort((p, q) -> p[0] != q[0] ? Long.compare(p[0], q[0]) : Long.compare(p[1], q[1]));
        return out;
    }

    private void collectPairs(int a, int b, List<long[]> out) {
        Node na = nodes.get(a);
        Node nb = nodes.get(b);
        boxTests++;
        if (!boxesOverlap(na, nb)) return;              // pruned
        if (na.leaf && nb.leaf) {
            if (a == b) {
                for (int i = na.start; i < na.end; i++) {
                    for (int j = i + 1; j < na.end; j++) testPrimitive(order[i], order[j], out);
                }
            } else {
                for (int i = na.start; i < na.end; i++) {
                    for (int j = nb.start; j < nb.end; j++) testPrimitive(order[i], order[j], out);
                }
            }
            return;
        }
        if (a == b) {
            collectPairs(na.left, na.left, out);
            collectPairs(na.right, na.right, out);
            collectPairs(na.left, na.right, out);
        } else if (nb.leaf || (!na.leaf && na.end - na.start >= nb.end - nb.start)) {
            collectPairs(na.left, b, out);
            collectPairs(na.right, b, out);
        } else {
            collectPairs(a, nb.left, out);
            collectPairs(a, nb.right, out);
        }
    }

    private void testPrimitive(int i, int j, List<long[]> out) {
        boxTests++;
        if (!boxesOverlap(primLo[i], primHi[i], primLo[j], primHi[j])) return;
        narrowTests++;
        if (Tri.intersect(tri[i], tri[j])) out.add(new long[]{Math.min(i, j), Math.max(i, j)});
    }

    /** Primitives whose box overlaps the query box; only boxes on the path are visited. */
    List<Integer> queryBox(long[] qlo, long[] qhi) {
        List<Integer> found = new ArrayList<>();
        if (!nodes.isEmpty()) queryInto(0, qlo, qhi, found);
        Collections.sort(found);
        return found;
    }

    private void queryInto(int id, long[] qlo, long[] qhi, List<Integer> found) {
        Node node = nodes.get(id);
        boxTests++;
        if (!boxesOverlap(node.lo, node.hi, qlo, qhi)) return;
        if (node.leaf) {
            for (int i = node.start; i < node.end; i++) {
                int p = order[i];
                if (boxesOverlap(primLo[p], primHi[p], qlo, qhi)) found.add(p);
            }
            return;
        }
        queryInto(node.left, qlo, qhi, found);
        queryInto(node.right, qlo, qhi, found);
    }

    private static boolean boxesOverlap(Node a, Node b) { return boxesOverlap(a.lo, a.hi, b.lo, b.hi); }

    static boolean boxesOverlap(long[] aLo, long[] aHi, long[] bLo, long[] bHi) {
        for (int k = 0; k < 3; k++) {
            if (aHi[k] < bLo[k] || bHi[k] < aLo[k]) return false;
        }
        return true;
    }
}

/** Exact triangle predicates in 3D over small integer coordinates. */
final class Tri {
    static long[] sub(long[] p, long[] q) {
        return new long[]{p[0] - q[0], p[1] - q[1], p[2] - q[2]};
    }

    static long[] cross(long[] u, long[] v) {
        return new long[]{
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]};
    }

    static long dot(long[] u, long[] v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

    static boolean isZero(long[] v) { return v[0] == 0 && v[1] == 0 && v[2] == 0; }

    /** {minX, minY, minZ, maxX, maxY, maxZ} of a triangle. */
    static long[] boxOf(long[][] t) {
        long[] out = new long[6];
        for (int k = 0; k < 3; k++) {
            out[k] = Math.min(t[0][k], Math.min(t[1][k], t[2][k]));
            out[k + 3] = Math.max(t[0][k], Math.max(t[1][k], t[2][k]));
        }
        return out;
    }

    /** Exact triangle/triangle intersection; degenerate triangles are rejected. */
    static boolean intersect(long[][] t1, long[][] t2) {
        long[] n1 = cross(sub(t1[1], t1[0]), sub(t1[2], t1[0]));
        long[] n2 = cross(sub(t2[1], t2[0]), sub(t2[2], t2[0]));
        if (isZero(n1) || isZero(n2)) throw new IllegalArgumentException("degenerate triangle");
        long d1 = dot(n1, t1[0]);
        long d2 = dot(n2, t2[0]);
        long[] s2 = new long[3];
        long[] s1 = new long[3];
        for (int i = 0; i < 3; i++) {
            s2[i] = dot(n1, t2[i]) - d1;
            s1[i] = dot(n2, t1[i]) - d2;
        }
        if (allStrictSign(s2) || allStrictSign(s1)) return false;
        long[] line = cross(n1, n2);
        if (isZero(line)) {
            // parallel planes: the sign test proves both triangles share one plane
            return coplanarOverlap(t1, t2, n1);
        }
        long[][][] both = {t1, t2};
        Frac[][] intervals = new Frac[2][];
        for (int i = 0; i < 2; i++) {
            long[] distances = i == 0 ? s1 : s2;
            long[] proj = new long[3];
            for (int k = 0; k < 3; k++) proj[k] = dot(line, both[i][k]);
            intervals[i] = intervalOnLine(distances, proj);
            if (intervals[i] == null) return false;
        }
        // closed intervals overlap -> touching contacts count as intersecting
        Frac low = max(intervals[0][0], intervals[1][0]);
        Frac high = min(intervals[0][1], intervals[1][1]);
        return low.compareTo(high) <= 0;
    }

    private static boolean allStrictSign(long[] values) {
        boolean positive = values[0] > 0 && values[1] > 0 && values[2] > 0;
        boolean negative = values[0] < 0 && values[1] < 0 && values[2] < 0;
        return positive || negative;
    }

    /** The set of points where a triangle meets the other plane, as a closed interval on `line`. */
    private static Frac[] intervalOnLine(long[] distances, long[] projections) {
        List<Frac> points = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            if (distances[i] == 0) points.add(Frac.of(projections[i]));
        }
        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3;
            long a = distances[i];
            long b = distances[j];
            if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
                Frac t = Frac.of(a, a - b);                     // V_i + t*(V_j - V_i) meets the plane
                points.add(Frac.of(projections[i])
                        .add(t.mul(Frac.of(projections[j] - projections[i]))));
            }
        }
        if (points.isEmpty()) return null;
        Frac low = points.get(0);
        Frac high = points.get(0);
        for (Frac f : points) {
            if (f.compareTo(low) < 0) low = f;
            if (f.compareTo(high) > 0) high = f;
        }
        return new Frac[]{low, high};
    }

    private static Frac max(Frac a, Frac b) { return a.compareTo(b) >= 0 ? a : b; }
    private static Frac min(Frac a, Frac b) { return a.compareTo(b) <= 0 ? a : b; }

    /** Coplanar case: separating-axis test in the plane, projecting away the dominant normal axis. */
    private static boolean coplanarOverlap(long[][] t1, long[][] t2, long[] n1) {
        int k = 0;
        if (Math.abs(n1[1]) > Math.abs(n1[k])) k = 1;
        if (Math.abs(n1[2]) > Math.abs(n1[k])) k = 2;
        int u = (k + 1) % 3;
        int v = (k + 2) % 3;
        long[][] p = project(t1, u, v);
        long[][] q = project(t2, u, v);
        return !separated(p, q) && !separated(q, p);
    }

    private static long[][] project(long[][] t, int u, int v) {
        long[][] out = new long[3][2];
        for (int i = 0; i < 3; i++) {
            out[i][0] = t[i][u];
            out[i][1] = t[i][v];
        }
        return out;
    }

    /** True when some edge normal of `a` separates a from b (strict, so touching is not separation). */
    private static boolean separated(long[][] a, long[][] b) {
        for (int i = 0; i < 3; i++) {
            long[] e = sub2(a[(i + 1) % 3], a[i]);
            long ax = -e[1];
            long ay = e[0];
            long aMin = Long.MAX_VALUE, aMax = Long.MIN_VALUE;
            long bMin = Long.MAX_VALUE, bMax = Long.MIN_VALUE;
            for (int j = 0; j < 3; j++) {
                long pa = a[j][0] * ax + a[j][1] * ay;
                long pb = b[j][0] * ax + b[j][1] * ay;
                aMin = Math.min(aMin, pa);
                aMax = Math.max(aMax, pa);
                bMin = Math.min(bMin, pb);
                bMax = Math.max(bMax, pb);
            }
            if (aMax < bMin || bMax < aMin) return true;
        }
        return false;
    }

    private static long[] sub2(long[] p, long[] q) {
        return new long[]{p[0] - q[0], p[1] - q[1]};
    }
}

public class Solution {
    private static long seed = 0x853C49E6748FEA9BL;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static long[][] tri(long a, long b, long c, long d, long e, long f, long g, long h, long i) {
        return new long[][]{{a, b, c}, {d, e, f}, {g, h, i}};
    }

    /** Brute force O(n^2) reference: every pair through the exact predicate. */
    static List<long[]> brutePairs(long[][][] tris) {
        List<long[]> out = new ArrayList<>();
        for (int i = 0; i < tris.length; i++) {
            for (int j = i + 1; j < tris.length; j++) {
                if (Tri.intersect(tris[i], tris[j])) out.add(new long[]{i, j});
            }
        }
        return out;
    }

    static String pairKeys(List<long[]> pairs) {
        StringBuilder sb = new StringBuilder();
        for (long[] p : pairs) sb.append(p[0]).append('-').append(p[1]).append(' ');
        return sb.toString().trim();
    }

    public static void main(String[] args) {
        // --- hand crafted configurations ---------------------------------------
        long[][] crossing = tri(0, 0, 0, 4, 0, 0, 0, 4, 0);          // z = 0 plane
        long[][] stabbing = tri(1, 1, -3, 1, 1, 3, 3, 3, 0);        // pierces the first triangle
        check(Tri.intersect(crossing, stabbing), "stabbing triangle must intersect");
        long[][] farAway = tri(50, 50, 50, 54, 50, 50, 50, 54, 50);
        check(!Tri.intersect(crossing, farAway), "distant triangle must not intersect");

        long[][] coplanarOverlapping = tri(1, 1, 0, 3, 1, 0, 1, 3, 0);
        check(Tri.intersect(crossing, coplanarOverlapping), "coplanar overlapping triangles");
        long[][] coplanarDisjoint = tri(10, 10, 0, 12, 10, 0, 10, 12, 0);
        check(!Tri.intersect(crossing, coplanarDisjoint), "coplanar disjoint triangles");

        long[][] sharedVertex = tri(0, 0, 0, -4, 0, 0, 0, -4, 0);
        check(Tri.intersect(crossing, sharedVertex), "touching at a shared vertex counts");
        long[][] sharedEdge = tri(0, 0, 0, 4, 0, 0, 0, 0, 4);
        check(Tri.intersect(crossing, sharedEdge), "touching along a shared edge counts");

        long[][] parallelButApart = tri(0, 0, 5, 4, 0, 5, 0, 4, 5);
        check(!Tri.intersect(crossing, parallelButApart), "parallel, distinct planes");
        long[][] upsideDown = tri(0, 0, -1, 4, 0, -1, 0, 4, -1);
        check(!Tri.intersect(crossing, upsideDown), "mirrored parallel triangle");
        check(Tri.intersect(farAway, tri(52, 52, 50, 56, 52, 50, 52, 56, 50)),
                "coplanar pair far from the origin");

        // --- rejected input (negative check) ------------------------------------
        boolean rejected = false;
        try {
            Tri.intersect(tri(0, 0, 0, 1, 1, 1, 2, 2, 2), crossing);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "collinear (degenerate) triangle must be rejected");

        // --- empty and single primitive trees ------------------------------------
        Bvh emptyTree = new Bvh(new long[0][][]);
        check(emptyTree.intersectingPairs().isEmpty(), "empty soup has no pairs");
        check(emptyTree.nodeCount() == 0, "empty soup has no nodes");
        Bvh oneTree = new Bvh(new long[][][]{crossing});
        check(oneTree.intersectingPairs().isEmpty(), "single triangle has no pairs");
        check(oneTree.queryBox(new long[]{0, 0, 0}, new long[]{4, 4, 0}).equals(List.of(0)),
                "single triangle query hits");

        // --- clustered soup: BVH must agree with brute force and prune -----------
        long[][][] soup = new long[26][][];
        for (int i = 0; i < 20; i++) {
            long[] base = {nextInt(12), nextInt(12), nextInt(12)};
            soup[i] = new long[][]{
                    base,
                    {base[0] + 1 + nextInt(6), base[1] + nextInt(8), base[2] + nextInt(8)},
                    {base[0] + nextInt(8), base[1] + 1 + nextInt(6), base[2] + nextInt(8)}};
        }
        for (int i = 20; i < 26; i++) {   // second cluster, far away
            long[] base = {100 + nextInt(8), 100 + nextInt(8), 100 + nextInt(8)};
            soup[i] = new long[][]{
                    base,
                    {base[0] + 2, base[1] + 1, base[2]},
                    {base[0], base[1] + 2, base[2] + 1}};
        }
        Bvh tree = new Bvh(soup);
        List<long[]> viaBvh = tree.intersectingPairs();
        List<long[]> viaBrute = brutePairs(soup);
        check(pairKeys(viaBvh).equals(pairKeys(viaBrute)),
                "BVH pairs differ from brute force: [" + pairKeys(viaBvh) + "] vs ["
                        + pairKeys(viaBrute) + "]");
        check(!viaBrute.isEmpty(), "the clustered soup really does contain intersections");
        long allPairs = (long) soup.length * (soup.length - 1) / 2;
        check(tree.narrowTests <= allPairs, "narrow phase never exceeds all pairs");
        check(tree.narrowTests > 0, "narrow phase actually ran");
        check(tree.nodeCount() >= 7, "26 primitives split into a real tree, nodes = " + tree.nodeCount());

        // --- cross-cluster pairs are excluded by the tree structure -------------
        for (long[] pair : viaBvh) {
            check(!(pair[0] < 20 && pair[1] >= 20), "cross cluster pair must not intersect");
        }

        // --- sparse grid: the tree must prune almost everything -----------------
        long[][][] scattered = new long[40][][];
        for (int i = 0; i < 40; i++) {
            long x = (i % 5) * 100;
            long y = ((i / 5) % 4) * 100;
            long z = (i / 20) * 100;
            scattered[i] = tri(x, y, z, x + 7, y + 2, z + 1, x + 3, y + 9, z + 4);
        }
        Bvh sparseTree = new Bvh(scattered);
        check(sparseTree.intersectingPairs().isEmpty(), "grid triangles are pairwise separated");
        long sparseAll = (long) scattered.length * (scattered.length - 1) / 2;
        check(sparseTree.boxTests * 2 < sparseAll,
                "pruning: " + sparseTree.boxTests + " box tests vs " + sparseAll + " pairs");
        check(brutePairs(scattered).isEmpty(), "brute force agrees the grid is separated");

        // --- rebuilding on a reordered soup finds the same pairs ----------------
        long[][][] reversed = new long[soup.length][][];
        for (int i = 0; i < soup.length; i++) reversed[i] = soup[soup.length - 1 - i];
        check(pairKeys(new Bvh(reversed).intersectingPairs()).equals(pairKeys(brutePairs(reversed))),
                "reordered soup still matches brute force");

        // --- box query oracle ----------------------------------------------------
        for (int trial = 0; trial < 12; trial++) {
            long[] qlo = {nextInt(120), nextInt(120), nextInt(120)};
            long[] qhi = {qlo[0] + nextInt(20), qlo[1] + nextInt(20), qlo[2] + nextInt(20)};
            List<Integer> got = tree.queryBox(qlo, qhi);
            List<Integer> want = new ArrayList<>();
            for (int i = 0; i < soup.length; i++) {
                long[] box = Tri.boxOf(soup[i]);
                if (Bvh.boxesOverlap(new long[]{box[0], box[1], box[2]},
                        new long[]{box[3], box[4], box[5]}, qlo, qhi)) {
                    want.add(i);
                }
            }
            check(got.equals(want), "box query mismatch in trial " + trial);
        }

        // --- determinism ---------------------------------------------------------
        Bvh again = new Bvh(soup);
        check(pairKeys(again.intersectingPairs()).equals(pairKeys(viaBvh)), "BVH result is stable");
        check(again.nodeCount() == tree.nodeCount(), "BVH shape is stable");

        System.out.println("triangles = " + soup.length + ", intersecting pairs = " + viaBvh.size()
                + ", bvh node pairs tested = " + tree.boxTests + " (all pairs = " + allPairs + ")"
                + ", narrow-phase tests = " + tree.narrowTests + ", nodes = " + tree.nodeCount());
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
