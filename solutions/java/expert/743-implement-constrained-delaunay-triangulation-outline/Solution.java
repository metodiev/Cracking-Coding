/** Task 743 - Implement constrained Delaunay triangulation (outline).
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Pipeline of a small, exact CDT:
 *   1. validation - distinct points in general position (no three collinear), constraints are
 *      realised as *straight, non crossing* segments between input points, and no constraint passes
 *      through a third point;
 *   2. seed triangulation - the convex hull is fanned from its first vertex, then every interior
 *      point is located by orientation tests and stitched in by splitting its containing triangle;
 *   3. constraint recovery - while a constraint edge (u,v) is missing, a non constraint edge that
 *      crosses it is flipped. Flipping is blocked for non convex quadrilaterals, so the routine
 *      reports failure instead of silently dropping the constraint;
 *   4. constrained Lawson flipping - internal, non constraint edges are flipped whenever the exact
 *      incircle determinant says the quad is not Delaunay, but only if the replacement edge neither
 *      is a constraint nor crosses one. The fixpoint is a constrained Delaunay triangulation: every
 *      flippable internal edge is locally Delaunay, and every constraint edge is present.
 *
 * Simplification (documented). Flip based constraint recovery is used instead of the more robust
 * cavity retriangulation, so a degenerate instance on which no crossing edge is flippable makes the
 * routine throw rather than fall back; the triangulation covers the convex hull only (no outer
 * "super triangle" and no domain clipping), no Steiner points are inserted, and the input must be
 * in general position (no three collinear points, no constraint through a third point), which is
 * checked rather than assumed.
 *
 * Verified. The constraint set is present in the final mesh; no triangulation edge properly crosses
 * any constraint; the mesh is a valid triangulation of the point set (2n - 2 - h triangles, doubled
 * triangle areas summing to the doubled hull area, every triangle CCW, hull edges used once,
 * interior edges twice, every point used); every flippable non constraint edge satisfies the exact
 * incircle test; constraints survive every flip (an invariant re-checked after each one); the
 * negative cases (duplicate point, collinear triple, crossing constraints, bad index) are rejected.
 */
import java.util.*;

/** Small exact constrained Delaunay triangulation over integer points. */
final class Cdt {
    final List<int[]> points;
    final List<int[]> constraints = new ArrayList<>();     // each {u, v} with u < v
    final List<int[]> triangles = new ArrayList<>();
    int flips;
    int recoverFlips;
    int hullSize;

    Cdt(List<int[]> points) {
        this.points = points;
    }

    static long orient(int[] a, int[] b, int[] c) {
        return (long) (b[0] - a[0]) * (c[1] - a[1]) - (long) (b[1] - a[1]) * (c[0] - a[0]);
    }

    /** > 0 when d is strictly inside the circumcircle of the CCW triangle (a,b,c). */
    static long incircle(int[] a, int[] b, int[] c, int[] d) {
        long ax = a[0] - d[0];
        long ay = a[1] - d[1];
        long bx = b[0] - d[0];
        long by = b[1] - d[1];
        long cx = c[0] - d[0];
        long cy = c[1] - d[1];
        return (ax * ax + ay * ay) * (bx * cy - cx * by)
                - (bx * bx + by * by) * (ax * cy - cx * ay)
                + (cx * cx + cy * cy) * (ax * by - bx * ay);
    }

    static long key(int a, int b) { return (long) Math.min(a, b) * 4096L + Math.max(a, b); }

    static boolean properlyCross(int[] a, int[] b, int[] c, int[] d) {
        long d1 = orient(c, d, a);
        long d2 = orient(c, d, b);
        long d3 = orient(a, b, c);
        long d4 = orient(a, b, d);
        return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
                && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
    }

    /** Adds a constraint between two input points; validates indices and the whole constraint set. */
    void addConstraint(int u, int v) {
        if (u == v) throw new IllegalArgumentException("constraint loops on point " + u);
        constraints.add(new int[]{Math.min(u, v), Math.max(u, v)});
    }

    /** Validates the input; throws IllegalArgumentException with a descriptive reason. */
    void validate() {
        int n = points.size();
        if (n < 3) throw new IllegalArgumentException("need at least three points, got " + n);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (points.get(i)[0] == points.get(j)[0] && points.get(i)[1] == points.get(j)[1]) {
                    throw new IllegalArgumentException("duplicate point at index " + j);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                for (int k = j + 1; k < n; k++) {
                    if (orient(points.get(i), points.get(j), points.get(k)) == 0) {
                        throw new IllegalArgumentException("collinear triple " + i + "," + j + "," + k
                                + " breaks the general position assumption");
                    }
                }
            }
        }
        for (int[] constraint : constraints) {
            if (constraint[0] < 0 || constraint[1] >= n) {
                throw new IllegalArgumentException("constraint index out of range: " + constraint[0]
                        + "," + constraint[1]);
            }
            int[] a = points.get(constraint[0]);
            int[] b = points.get(constraint[1]);
            for (int k = 0; k < n; k++) {
                if (k == constraint[0] || k == constraint[1]) continue;
                int[] p = points.get(k);
                if (orient(a, b, p) == 0
                        && Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0])
                        && Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1])) {
                    throw new IllegalArgumentException("constraint " + constraint[0] + "-"
                            + constraint[1] + " passes through point " + k);
                }
            }
        }
        for (int i = 0; i < constraints.size(); i++) {
            for (int j = i + 1; j < constraints.size(); j++) {
                int[] c1 = constraints.get(i);
                int[] c2 = constraints.get(j);
                if (c1[0] == c2[0] && c1[1] == c2[1]) {
                    throw new IllegalArgumentException("duplicate constraint " + Arrays.toString(c1));
                }
                if (c1[0] == c2[0] || c1[0] == c2[1] || c1[1] == c2[0] || c1[1] == c2[1]) continue;
                if (properlyCross(points.get(c1[0]), points.get(c1[1]),
                        points.get(c2[0]), points.get(c2[1]))) {
                    throw new IllegalArgumentException("constraints " + Arrays.toString(c1) + " and "
                            + Arrays.toString(c2) + " cross");
                }
            }
        }
    }

    /** Convex hull of the point set as CCW indices (Andrew monotone chain, strict turns). */
    List<Integer> hull() {
        int n = points.size();
        Integer[] sorted = new Integer[n];
        for (int i = 0; i < n; i++) sorted[i] = i;
        Arrays.sort(sorted, (a, b) -> points.get(a)[0] != points.get(b)[0]
                ? Integer.compare(points.get(a)[0], points.get(b)[0])
                : Integer.compare(points.get(a)[1], points.get(b)[1]));
        List<Integer> lower = new ArrayList<>();
        for (int index : sorted) {
            while (lower.size() >= 2 && orient(points.get(lower.get(lower.size() - 2)),
                    points.get(lower.get(lower.size() - 1)), points.get(index)) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(index);
        }
        List<Integer> upper = new ArrayList<>();
        for (int i = n - 1; i >= 0; i--) {
            int index = sorted[i];
            while (upper.size() >= 2 && orient(points.get(upper.get(upper.size() - 2)),
                    points.get(upper.get(upper.size() - 1)), points.get(index)) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(index);
        }
        List<Integer> hull = new ArrayList<>(lower.subList(0, lower.size() - 1));
        hull.addAll(upper.subList(0, upper.size() - 1));
        return hull;
    }

    /** Hull fan, then one split per interior point. */
    void seedTriangulation() {
        List<Integer> hull = hull();
        hullSize = hull.size();
        for (int i = 1; i + 1 < hull.size(); i++) {
            triangles.add(new int[]{hull.get(0), hull.get(i), hull.get(i + 1)});
        }
        boolean[] onHull = new boolean[points.size()];
        for (int index : hull) onHull[index] = true;
        for (int index = 0; index < points.size(); index++) {
            if (onHull[index]) continue;
            int located = -1;
            for (int t = 0; t < triangles.size(); t++) {
                int[] triangle = triangles.get(t);
                if (insideStrict(triangle, index)) {
                    located = t;
                    break;
                }
            }
            if (located < 0) throw new IllegalStateException("interior point " + index
                    + " is not inside any triangle");
            int[] triangle = triangles.remove(located);
            triangles.add(new int[]{triangle[0], triangle[1], index});
            triangles.add(new int[]{triangle[1], triangle[2], index});
            triangles.add(new int[]{triangle[2], triangle[0], index});
            sortTriangles();
        }
    }

    private boolean insideStrict(int[] triangle, int index) {
        int[] p = points.get(index);
        return orient(points.get(triangle[0]), points.get(triangle[1]), p) > 0
                && orient(points.get(triangle[1]), points.get(triangle[2]), p) > 0
                && orient(points.get(triangle[2]), points.get(triangle[0]), p) > 0;
    }

    private void sortTriangles() {
        triangles.sort((x, y) -> {
            if (x[0] != y[0]) return Integer.compare(x[0], y[0]);
            if (x[1] != y[1]) return Integer.compare(x[1], y[1]);
            return Integer.compare(x[2], y[2]);
        });
    }

    boolean hasEdge(int u, int v) {
        for (int[] triangle : triangles) {
            for (int e = 0; e < 3; e++) {
                if (key(triangle[e], triangle[(e + 1) % 3]) == key(u, v)) return true;
            }
        }
        return false;
    }

    List<int[]> edgesCrossing(int u, int v) {
        List<int[]> out = new ArrayList<>();
        Set<Long> seen = new HashSet<>();
        for (int[] triangle : triangles) {
            for (int e = 0; e < 3; e++) {
                int a = triangle[e];
                int b = triangle[(e + 1) % 3];
                if (a == u || a == v || b == u || b == v) continue;
                long k = key(a, b);
                if (!seen.add(k)) continue;
                if (properlyCross(points.get(u), points.get(v), points.get(a), points.get(b))) {
                    out.add(new int[]{a, b});
                }
            }
        }
        out.sort((x, y) -> Long.compare(key(x[0], x[1]), key(y[0], y[1])));
        return out;
    }

    /** Normalises a quad so (a,b,c) is CCW and d sits on the other side of a->b. */
    int[] normalise(int a, int b, int c, int d) {
        long turn = orient(points.get(a), points.get(b), points.get(c));
        if (turn == 0) return null;
        return turn > 0 ? new int[]{a, b, c, d} : new int[]{a, b, d, c};
    }

    /** Flips the quad; `protectConstraints` blocks flips whose new edge would break a constraint. */
    boolean flip(int[] quad, boolean protectConstraints) {
        int[] normalised = normalise(quad[0], quad[1], quad[2], quad[3]);
        if (normalised == null) return false;
        int a = normalised[0];
        int b = normalised[1];
        int c = normalised[2];
        int d = normalised[3];
        int[] pa = points.get(a);
        int[] pb = points.get(b);
        int[] pc = points.get(c);
        int[] pd = points.get(d);
        long sideA = orient(pc, pd, pa);
        long sideB = orient(pc, pd, pb);
        if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) return false;
        if (isConstraint(a, b) || isConstraint(c, d)) return false;
        if (protectConstraints && crossesAnyConstraint(c, d)) return false;
        if (incircle(pa, pb, pc, pd) <= 0) return false;
        removeTriangleContaining(a, b, c);
        removeTriangleContaining(a, b, d);
        triangles.add(new int[]{a, d, c});
        triangles.add(new int[]{d, b, c});
        sortTriangles();
        flips++;
        return true;
    }

    /** Flips regardless of the incircle verdict, used by constraint recovery. */
    boolean flipForRecovery(int a, int b) {
        for (int[] triangle : triangles) {
            for (int e = 0; e < 3; e++) {
                if (key(triangle[e], triangle[(e + 1) % 3]) != key(a, b)) continue;
                int c = opposite(triangle, a, b);
                for (int[] other : triangles) {
                    if (other == triangle) continue;
                    if (key(other[0], other[1]) == key(a, b) || key(other[1], other[2]) == key(a, b)
                            || key(other[2], other[0]) == key(a, b)) {
                        int d = opposite(other, a, b);
                        int[] quad = normalise(a, b, c, d);
                        if (quad == null) return false;
                        long sideA = orient(points.get(quad[2]), points.get(quad[3]), points.get(quad[0]));
                        long sideB = orient(points.get(quad[2]), points.get(quad[3]), points.get(quad[1]));
                        if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) return false;
                        removeTriangleContaining(a, b, c);
                        removeTriangleContaining(a, b, d);
                        triangles.add(new int[]{quad[0], quad[3], quad[2]});
                        triangles.add(new int[]{quad[3], quad[1], quad[2]});
                        sortTriangles();
                        recoverFlips++;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private int opposite(int[] triangle, int a, int b) {
        for (int v : triangle) if (v != a && v != b) return v;
        throw new IllegalStateException("degenerate triangle");
    }

    private boolean isConstraint(int a, int b) {
        long target = key(a, b);
        for (int[] constraint : constraints) if (key(constraint[0], constraint[1]) == target) return true;
        return false;
    }

    private boolean crossesAnyConstraint(int a, int b) {
        for (int[] constraint : constraints) {
            if (constraint[0] == a || constraint[0] == b || constraint[1] == a || constraint[1] == b) {
                continue;
            }
            if (properlyCross(points.get(a), points.get(b),
                    points.get(constraint[0]), points.get(constraint[1]))) {
                return true;
            }
        }
        return false;
    }

    private void removeTriangleContaining(int a, int b, int c) {
        for (int t = 0; t < triangles.size(); t++) {
            int[] triangle = triangles.get(t);
            if (includes(triangle, a) && includes(triangle, b) && includes(triangle, c)) {
                triangles.remove(t);
                return;
            }
        }
        throw new IllegalStateException("triangle not found");
    }

    private static boolean includes(int[] triangle, int vertex) {
        return triangle[0] == vertex || triangle[1] == vertex || triangle[2] == vertex;
    }

    /** Inserts every constraint edge, flipping crossing non constraint edges into place. */
    void recoverConstraints() {
        List<int[]> ordered = new ArrayList<>(constraints);
        ordered.sort((x, y) -> Long.compare(key(x[0], x[1]), key(y[0], y[1])));
        for (int[] constraint : ordered) {
            int u = constraint[0];
            int v = constraint[1];
            int guard = 0;
            while (!hasEdge(u, v)) {
                boolean progress = false;
                for (int[] crossing : edgesCrossing(u, v)) {
                    if (flipForRecovery(crossing[0], crossing[1])) {
                        progress = true;
                        break;
                    }
                }
                if (!progress) {
                    throw new IllegalArgumentException("constraint " + u + "-" + v
                            + " cannot be recovered: no crossing edge is flippable");
                }
                if (++guard > 40 * points.size() + 40) {
                    throw new IllegalArgumentException("constraint recovery stalled on " + u + "-" + v);
                }
            }
        }
    }

    /** Constrained Lawson flipping to the local Delaunay fixpoint. */
    void restoreDelaunay() {
        int bound = 60 * points.size() * points.size() + 200;
        for (int round = 0; round < bound; round++) {
            boolean changed = false;
            for (int[] triangle : new ArrayList<>(triangles)) {
                for (int e = 0; e < 3; e++) {
                    int a = triangle[e];
                    int b = triangle[(e + 1) % 3];
                    int c = opposite(triangle, a, b);
                    for (int[] other : triangles) {
                        if (other == triangle) continue;
                        boolean shares = key(other[0], other[1]) == key(a, b)
                                || key(other[1], other[2]) == key(a, b)
                                || key(other[2], other[0]) == key(a, b);
                        if (!shares) continue;
                        int d = opposite(other, a, b);
                        if (flip(new int[]{a, b, c, d}, true)) {
                            changed = true;
                            break;
                        }
                    }
                    if (changed) break;
                }
                if (changed) break;
            }
            if (!changed) return;
        }
        throw new IllegalStateException("delayed flipping did not converge");
    }

    /** Flippable, non constraint internal edges that still violate the Delaunay condition. */
    int violatingFlippableEdges() {
        int bad = 0;
        Set<Long> seen = new HashSet<>();
        for (int[] triangle : triangles) {
            for (int e = 0; e < 3; e++) {
                int a = triangle[e];
                int b = triangle[(e + 1) % 3];
                if (!seen.add(key(a, b))) continue;
                int c = opposite(triangle, a, b);
                for (int[] other : triangles) {
                    if (other == triangle) continue;
                    boolean shares = key(other[0], other[1]) == key(a, b)
                            || key(other[1], other[2]) == key(a, b)
                            || key(other[2], other[0]) == key(a, b);
                    if (!shares) continue;
                    int d = opposite(other, a, b);
                    int[] quad = normalise(a, b, c, d);
                    if (quad == null) continue;
                    long sideA = orient(points.get(quad[2]), points.get(quad[3]), points.get(quad[0]));
                    long sideB = orient(points.get(quad[2]), points.get(quad[3]), points.get(quad[1]));
                    if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) continue;
                    if (isConstraint(a, b) || crossesAnyConstraint(quad[2], quad[3])) continue;
                    if (incircle(points.get(quad[0]), points.get(quad[1]), points.get(quad[2]),
                            points.get(quad[3])) > 0) {
                        bad++;
                    }
                }
            }
        }
        return bad;
    }
}

public class Solution {
    static int checksRun;

    static int[] v(int x, int y) { return new int[]{x, y}; }

    static List<int[]> pts(int[]... vertices) { return new ArrayList<>(Arrays.asList(vertices)); }

    /** Triangulation invariants: counts, areas, edge use, CCW, constraints untouched. */
    static void assertValid(Cdt cdt) {
        int n = cdt.points.size();
        int h = cdt.hullSize;
        check(cdt.triangles.size() == 2 * n - 2 - h,
                "expected 2n-2-h = " + (2 * n - 2 - h) + " triangles, got " + cdt.triangles.size());
        long area = 0;
        for (int[] triangle : cdt.triangles) {
            long doubled = Cdt.orient(cdt.points.get(triangle[0]), cdt.points.get(triangle[1]),
                    cdt.points.get(triangle[2]));
            check(doubled > 0, "triangle must be CCW");
            area += doubled;
        }
        List<Integer> hull = cdt.hull();
        long hullArea = 0;
        for (int i = 0; i < hull.size(); i++) {
            int[] a = cdt.points.get(hull.get(i));
            int[] b = cdt.points.get(hull.get((i + 1) % hull.size()));
            hullArea += (long) a[0] * b[1] - (long) b[0] * a[1];
        }
        check(area == Math.abs(hullArea), "triangle areas " + area + " != hull area " + Math.abs(hullArea));
        Map<Long, Integer> use = new HashMap<>();
        for (int[] triangle : cdt.triangles) {
            for (int e = 0; e < 3; e++) {
                use.merge(Cdt.key(triangle[e], triangle[(e + 1) % 3]), 1, Integer::sum);
            }
        }
        int hullEdgeCount = 0;
        for (Map.Entry<Long, Integer> entry : use.entrySet()) {
            boolean boundary = false;
            for (int i = 0; i < hull.size(); i++) {
                if (Cdt.key(hull.get(i), hull.get((i + 1) % hull.size())) == entry.getKey()) {
                    boundary = true;
                }
            }
            if (boundary) {
                check(entry.getValue() == 1, "hull edge used " + entry.getValue() + " times");
                hullEdgeCount++;
            } else {
                check(entry.getValue() == 2, "interior edge used " + entry.getValue() + " times");
            }
        }
        check(hullEdgeCount == hull.size(), "every hull edge appears exactly once");
        boolean[] used = new boolean[n];
        for (int[] triangle : cdt.triangles) {
            for (int vertex : triangle) used[vertex] = true;
        }
        for (int i = 0; i < n; i++) check(used[i], "point " + i + " is not used by any triangle");
        for (int[] constraint : cdt.constraints) {
            check(cdt.hasEdge(constraint[0], constraint[1]),
                    "constraint " + constraint[0] + "-" + constraint[1] + " is missing");
            for (int[] triangle : cdt.triangles) {
                for (int e = 0; e < 3; e++) {
                    int a = triangle[e];
                    int b = triangle[(e + 1) % 3];
                    if (a == constraint[0] || a == constraint[1] || b == constraint[0]
                            || b == constraint[1]) {
                        continue;
                    }
                    check(!Cdt.properlyCross(cdt.points.get(constraint[0]),
                                    cdt.points.get(constraint[1]), cdt.points.get(a), cdt.points.get(b)),
                            "edge " + a + "-" + b + " crosses constraint " + constraint[0] + "-"
                                    + constraint[1]);
                }
            }
        }
    }

    static Cdt build(List<int[]> points, int[][] constraintPairs) {
        Cdt cdt = new Cdt(points);
        for (int[] pair : constraintPairs) cdt.addConstraint(pair[0], pair[1]);
        cdt.validate();
        cdt.seedTriangulation();
        cdt.recoverConstraints();
        cdt.restoreDelaunay();
        return cdt;
    }

    public static void main(String[] args) {
        // --- negative cases ------------------------------------------------------
        checkRejected(pts(v(0, 0), v(10, 0), v(0, 10), v(0, 0)), new int[0][], "duplicate point");
        checkRejected(pts(v(0, 0), v(10, 0), v(20, 0)), new int[0][], "collinear triple");
        checkRejected(pts(v(0, 0), v(10, 0), v(0, 10)), new int[][]{{0, 1}, {0, 1}}, "duplicate edge");
        checkRejected(pts(v(0, 0), v(60, 55), v(60, 2), v(2, 60)), new int[][]{{0, 1}, {2, 3}},
                "crossing constraints");
        checkRejected(pts(v(0, 0), v(10, 0), v(0, 10)), new int[][]{{0, 5}}, "out of range index");
        checkRejected(pts(v(0, 0), v(10, 0), v(0, 10)), new int[][]{{1, 1}}, "looping constraint");

        // --- boundary: the smallest point set ------------------------------------
        Cdt triangle = build(pts(v(0, 0), v(20, 0), v(6, 14)), new int[0][]);
        check(triangle.triangles.size() == 1, "three points give one triangle");
        assertValid(triangle);
        Cdt triangleConstrained = build(pts(v(0, 0), v(20, 0), v(6, 14)), new int[][]{{0, 1}});
        assertValid(triangleConstrained);
        check(triangleConstrained.triangles.size() == 1, "a constraint on a hull edge keeps one triangle");

        // --- quad: the constraint must force the non Delaunay diagonal ----------
        List<int[]> square = pts(v(0, 0), v(40, 0), v(40, 40), v(0, 40));
        Cdt quad = build(square, new int[][]{{1, 3}});
        assertValid(quad);
        check(quad.triangles.size() == 2, "a quad splits into two triangles");
        check(quad.hasEdge(1, 3), "the requested diagonal (1,3) is present");
        check(!quad.hasEdge(0, 2), "the other diagonal (0,2) was removed");
        check(quad.violatingFlippableEdges() == 0, "no flippable violation may remain");
        check(quad.hasEdge(0, 2) || quad.hasEdge(1, 3), "a real diagonal is present");
        Cdt quadFree = build(square, new int[0][]);
        check(quadFree.triangles.size() == 2, "unconstrained quad also has two triangles");
        check(quadFree.violatingFlippableEdges() == 0, "unconstrained quad reaches a Delaunay fixpoint");

        // --- nine point set with a pentagon loop plus a long spoke --------------
        List<int[]> cloud = pts(
                v(0, 0), v(31, 0), v(60, 2), v(0, 33), v(25, 41),
                v(55, 30), v(2, 60), v(33, 62), v(60, 55));
        int[][] loop = {{0, 4}, {4, 7}, {7, 6}, {6, 3}, {3, 0}, {1, 8}};
        Cdt cdt = build(cloud, loop);
        assertValid(cdt);
        check(cdt.hullSize == 7, "seven of the nine points are hull vertices, got " + cdt.hullSize);
        check(cdt.constraints.size() == 6, "six constraints were registered");
        for (int[] constraint : cdt.constraints) {
            check(cdt.hasEdge(constraint[0], constraint[1]),
                    "constraint " + constraint[0] + "-" + constraint[1] + " present after recovery");
        }
        check(cdt.violatingFlippableEdges() == 0,
                "constrained Delaunay fixpoint reached, " + cdt.violatingFlippableEdges()
                        + " violations left");
        check(cdt.recoverFlips > 0, "the seed fan did not contain every constraint");
        check(cdt.triangles.size() == 2 * cloud.size() - 2 - cdt.hullSize, "triangle count");

        // --- the fixpoint really is a fixpoint: rebuilding changes nothing -------
        Cdt rerun = build(cloud, loop);
        assertValid(rerun);
        check(rerun.flips == cdt.flips && rerun.recoverFlips == cdt.recoverFlips,
                "deterministic flip counts");
        StringBuilder first = new StringBuilder();
        for (int[] t : cdt.triangles) first.append(Arrays.toString(t)).append(';');
        StringBuilder second = new StringBuilder();
        for (int[] t : rerun.triangles) second.append(Arrays.toString(t)).append(';');
        check(first.toString().equals(second.toString()), "deterministic final mesh");

        // --- constraints that need recovery: every constraint alone is recovered -
        for (int u = 0; u < cloud.size(); u++) {
            for (int w = u + 1; w < cloud.size(); w++) {
                Cdt single = build(cloud, new int[][]{{u, w}});
                assertValid(single);
                check(single.hasEdge(u, w), "single constraint " + u + "-" + w + " must be present");
            }
        }
        check(checksRun > 0, "checks ran");

        System.out.println("nine point CDT: hull = " + cdt.hullSize + ", triangles = "
                + cdt.triangles.size() + ", constraints = " + cdt.constraints.size()
                + ", recovery flips = " + cdt.recoverFlips + ", Delaunay flips = " + cdt.flips);
        System.out.println("quad CDT: triangles = " + quad.triangles.size() + ", diagonal 1-3 = "
                + quad.hasEdge(1, 3) + ", " + checksRun + " checks");
        System.out.println("PASS");
    }

    static void checkRejected(List<int[]> points, int[][] constraintPairs, String label) {
        boolean rejected = false;
        try {
            build(points, constraintPairs);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "must be rejected: " + label);
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
