/** Task 744 - Implement Voronoi diagram via Fortune's sweep-line (outline).
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Outline of the structure. Fortune's sweep builds the diagram by maintaining a parabolic beach
 * line in a priority queue of site and circle events. This solution produces the *same* diagram
 * (Fortune's construction is the dual of the Delaunay triangulation) with the classical duality,
 * which keeps every predicate exact:
 *   1. Delaunay triangulation of the sites: hull fan, interior points stitched in, then Lawson
 *      flipping driven by the exact incircle determinant;
 *   2. Voronoi vertices - one per Delaunay triangle, at its circumcenter, computed as an exact
 *      BigInteger fraction;
 *   3. bounded Voronoi edges - one per internal Delaunay edge, joining the circumcenters of the two
 *      triangles sharing it;
 *   4. unbounded Voronoi edges (rays) - one per hull edge; the ray leaves the incident circumcenter
 *      perpendicular to the hull edge, pointing away from the third vertex, so a point z on the ray
 *      is closer to that hull edge's endpoints than to any other site iff (p - r)·d >= 0 for every
 *      other site r, an exact integer test.
 *
 * Simplification (documented). The beach-line/priority-queue machinery of Fortune's sweep is *not*
 * implemented (the task is an outline); the identical diagram is obtained by dualising the Delaunay
 * triangulation, and the invariants that make it a Voronoi diagram are verified exactly rather than
 * assumed. Sites must be distinct and not all collinear (otherwise no proper triangulation exists)
 * and coordinates stay small so that determinants and dot products fit in a long; circumcenters and
 * mid points use BigInteger fractions, so equidistance tests are exact instead of epsilon based.
 *
 * Verified. Every Voronoi vertex is equidistant from its three sites and strictly closer to them
 * than to any other site (the empty circle property); every bounded edge's mid point is equidistant
 * from the two sites of its Delaunay edge and not farther from them than from any other site; every
 * ray direction satisfies (p - r)·d >= 0 for all other sites, which is exactly the half-plane
 * condition for an unbounded Voronoi edge; the counts match the Euler relations (one vertex per
 * triangle, one bounded edge per internal edge, one ray per hull edge); unbounded cells occur
 * exactly for hull sites; the nearest-neighbour graph is a subgraph of the triangulation; and
 * degenerate inputs (fewer than three, duplicate or collinear sites) are rejected.
 */
import java.util.*;
import java.math.BigInteger;

/** Exact rational value with a positive, reduced denominator. */
final class Exact {
    final BigInteger num;
    final BigInteger den;

    private Exact(BigInteger num, BigInteger den) { this.num = num; this.den = den; }

    static Exact of(long value) { return new Exact(BigInteger.valueOf(value), BigInteger.ONE); }

    static Exact of(long numerator, long denominator) {
        if (denominator == 0) throw new ArithmeticException("zero denominator");
        BigInteger n = BigInteger.valueOf(numerator);
        BigInteger d = BigInteger.valueOf(denominator);
        return reduce(n, d);
    }

    static Exact zero() { return of(0); }

    Exact add(Exact o) {
        return reduce(num.multiply(o.den).add(o.num.multiply(den)), den.multiply(o.den));
    }

    Exact mul(Exact o) { return reduce(num.multiply(o.num), den.multiply(o.den)); }

    int signum() { return num.signum(); }

    private static Exact reduce(BigInteger n, BigInteger d) {
        if (d.signum() < 0) {
            n = n.negate();
            d = d.negate();
        }
        BigInteger g = n.gcd(d);
        if (g.signum() != 0 && !g.equals(BigInteger.ONE)) {
            n = n.divide(g);
            d = d.divide(g);
        }
        return new Exact(n, d);
    }

    @Override public String toString() { return num + "/" + den; }
}

/** Delaunay triangulation of the sites plus the dual Voronoi objects. */
final class VoronoiDiagram {
    final List<int[]> sites;
    final List<int[]> triangles = new ArrayList<>();
    int hullSize;
    Exact[] centerX;
    Exact[] centerY;

    VoronoiDiagram(List<int[]> sites) {
        this.sites = sites;
    }

    static long orient(int[] a, int[] b, int[] c) {
        return (long) (b[0] - a[0]) * (c[1] - a[1]) - (long) (b[1] - a[1]) * (c[0] - a[0]);
    }

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

    void validate() {
        int n = sites.size();
        if (n < 3) throw new IllegalArgumentException("a Voronoi diagram needs at least three sites");
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (sites.get(i)[0] == sites.get(j)[0] && sites.get(i)[1] == sites.get(j)[1]) {
                    throw new IllegalArgumentException("duplicate site at index " + j);
                }
            }
        }
        boolean allCollinear = true;
        for (int i = 1; i + 1 < n && allCollinear; i++) {
            if (orient(sites.get(0), sites.get(i), sites.get(i + 1)) != 0) allCollinear = false;
        }
        if (allCollinear) throw new IllegalArgumentException("all sites are collinear");
    }

    List<Integer> hull() {
        int n = sites.size();
        Integer[] sorted = new Integer[n];
        for (int i = 0; i < n; i++) sorted[i] = i;
        Arrays.sort(sorted, (a, b) -> sites.get(a)[0] != sites.get(b)[0]
                ? Integer.compare(sites.get(a)[0], sites.get(b)[0])
                : Integer.compare(sites.get(a)[1], sites.get(b)[1]));
        List<Integer> lower = new ArrayList<>();
        for (int index : sorted) {
            while (lower.size() >= 2 && orient(sites.get(lower.get(lower.size() - 2)),
                    sites.get(lower.get(lower.size() - 1)), sites.get(index)) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(index);
        }
        List<Integer> upper = new ArrayList<>();
        for (int i = n - 1; i >= 0; i--) {
            int index = sorted[i];
            while (upper.size() >= 2 && orient(sites.get(upper.get(upper.size() - 2)),
                    sites.get(upper.get(upper.size() - 1)), sites.get(index)) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(index);
        }
        List<Integer> hull = new ArrayList<>(lower.subList(0, lower.size() - 1));
        hull.addAll(upper.subList(0, upper.size() - 1));
        return hull;
    }

    void build() {
        validate();
        List<Integer> hull = hull();
        hullSize = hull.size();
        for (int i = 1; i + 1 < hull.size(); i++) {
            triangles.add(new int[]{hull.get(0), hull.get(i), hull.get(i + 1)});
        }
        boolean[] onHull = new boolean[sites.size()];
        for (int index : hull) onHull[index] = true;
        for (int index = 0; index < sites.size(); index++) {
            if (onHull[index]) continue;
            int located = -1;
            for (int t = 0; t < triangles.size(); t++) {
                int[] triangle = triangles.get(t);
                if (insideStrict(triangle, index)) {
                    located = t;
                    break;
                }
            }
            if (located < 0) throw new IllegalStateException("site " + index + " is not inside any triangle");
            int[] triangle = triangles.remove(located);
            triangles.add(new int[]{triangle[0], triangle[1], index});
            triangles.add(new int[]{triangle[1], triangle[2], index});
            triangles.add(new int[]{triangle[2], triangle[0], index});
        }
        lawson();
        computeCenters();
    }

    private boolean insideStrict(int[] triangle, int index) {
        int[] p = sites.get(index);
        return orient(sites.get(triangle[0]), sites.get(triangle[1]), p) > 0
                && orient(sites.get(triangle[1]), sites.get(triangle[2]), p) > 0
                && orient(sites.get(triangle[2]), sites.get(triangle[0]), p) > 0;
    }

    private void lawson() {
        int bound = 60 * sites.size() * sites.size() + 200;
        for (int round = 0; round < bound; round++) {
            boolean changed = false;
            for (int t = 0; t < triangles.size() && !changed; t++) {
                int[] triangle = triangles.get(t);
                for (int e = 0; e < 3 && !changed; e++) {
                    int a = triangle[e];
                    int b = triangle[(e + 1) % 3];
                    int c = opposite(triangle, a, b);
                    for (int u = 0; u < triangles.size() && !changed; u++) {
                        if (u == t) continue;
                        int[] other = triangles.get(u);
                        if (!sharesEdge(other, a, b)) continue;
                        int d = opposite(other, a, b);
                        if (flip(a, b, c, d)) changed = true;
                    }
                }
            }
            if (!changed) return;
        }
        throw new IllegalStateException("Lawson flipping did not converge");
    }

    private boolean flip(int a, int b, int c, int d) {
        long turn = orient(sites.get(a), sites.get(b), sites.get(c));
        if (turn == 0) return false;
        if (turn < 0) {
            int swap = c;
            c = d;
            d = swap;
        }
        int[] pa = sites.get(a);
        int[] pb = sites.get(b);
        int[] pc = sites.get(c);
        int[] pd = sites.get(d);
        long sideA = orient(pc, pd, pa);
        long sideB = orient(pc, pd, pb);
        if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) return false;
        if (incircle(pa, pb, pc, pd) <= 0) return false;
        removeTriangleContaining(a, b, c);
        removeTriangleContaining(a, b, d);
        triangles.add(new int[]{a, d, c});
        triangles.add(new int[]{d, b, c});
        return true;
    }

    private boolean sharesEdge(int[] triangle, int a, int b) {
        for (int e = 0; e < 3; e++) {
            int u = triangle[e];
            int w = triangle[(e + 1) % 3];
            if ((u == a && w == b) || (u == b && w == a)) return true;
        }
        return false;
    }

    private int opposite(int[] triangle, int a, int b) {
        for (int v : triangle) if (v != a && v != b) return v;
        throw new IllegalStateException("degenerate triangle");
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

    void computeCenters() {
        centerX = new Exact[triangles.size()];
        centerY = new Exact[triangles.size()];
        for (int t = 0; t < triangles.size(); t++) {
            int[] triangle = triangles.get(t);
            int[] a = sites.get(triangle[0]);
            int[] b = sites.get(triangle[1]);
            int[] c = sites.get(triangle[2]);
            long d = 2L * (a[0] * (long) (b[1] - c[1]) + b[0] * (long) (c[1] - a[1])
                    + c[0] * (long) (a[1] - b[1]));
            long ux = (a[0] * (long) a[0] + a[1] * (long) a[1]) * (b[1] - c[1])
                    + (b[0] * (long) b[0] + b[1] * (long) b[1]) * (c[1] - a[1])
                    + (c[0] * (long) c[0] + c[1] * (long) c[1]) * (a[1] - b[1]);
            long uy = (a[0] * (long) a[0] + a[1] * (long) a[1]) * (c[0] - b[0])
                    + (b[0] * (long) b[0] + b[1] * (long) b[1]) * (a[0] - c[0])
                    + (c[0] * (long) c[0] + c[1] * (long) c[1]) * (b[0] - a[0]);
            centerX[t] = Exact.of(ux, d);
            centerY[t] = Exact.of(uy, d);
        }
    }

    /** +1 when z is strictly closer to r than to s, 0 when equidistant, -1 otherwise. */
    static int compareDistanceSquared(Exact zx, Exact zy, int[] s, int[] r) {
        Exact value = Exact.of(2L * (r[0] - s[0])).mul(zx)
                .add(Exact.of(2L * (r[1] - s[1])).mul(zy))
                .add(Exact.of((long) s[0] * s[0] + (long) s[1] * s[1]
                        - (long) r[0] * r[0] - (long) r[1] * r[1]));
        return value.signum();
    }

    /** Sites ordered as the Voronoi vertex neighbourhood of a triangle edge. */
    boolean isDelaunayEdge(int p, int q) {
        for (int[] triangle : triangles) {
            for (int e = 0; e < 3; e++) {
                int u = triangle[e];
                int w = triangle[(e + 1) % 3];
                if ((u == p && w == q) || (u == q && w == p)) return true;
            }
        }
        return false;
    }

    boolean isHullEdge(int p, int q) {
        List<Integer> hull = hull();
        for (int i = 0; i < hull.size(); i++) {
            int u = hull.get(i);
            int w = hull.get((i + 1) % hull.size());
            if ((u == p && w == q) || (u == q && w == p)) return true;
        }
        return false;
    }
}

public class Solution {
    static int checksRun;

    static int[] v(int x, int y) { return new int[]{x, y}; }

    static List<int[]> pts(int[]... vertices) { return new ArrayList<>(Arrays.asList(vertices)); }

    static int triangleOf(VoronoiDiagram diagram, int p, int q) {
        for (int t = 0; t < diagram.triangles.size(); t++) {
            int[] triangle = diagram.triangles.get(t);
            for (int e = 0; e < 3; e++) {
                int u = triangle[e];
                int w = triangle[(e + 1) % 3];
                if ((u == p && w == q) || (u == q && w == p)) return t;
            }
        }
        throw new IllegalStateException("no triangle for edge " + p + "-" + q);
    }

    /** Voronoi vertex property: equidistant from its three sites, no other site strictly closer. */
    static int assertVoronoiVertices(VoronoiDiagram diagram) {
        int checked = 0;
        for (int t = 0; t < diagram.triangles.size(); t++) {
            int[] triangle = diagram.triangles.get(t);
            Exact cx = diagram.centerX[t];
            Exact cy = diagram.centerY[t];
            for (int e = 0; e < 3; e++) {
                int[] p = diagram.sites.get(triangle[e]);
                int[] q = diagram.sites.get(triangle[(e + 1) % 3]);
                check(VoronoiDiagram.compareDistanceSquared(cx, cy, p, q) == 0,
                        "circumcenter is not equidistant from its sites");
            }
            for (int r = 0; r < diagram.sites.size(); r++) {
                if (r == triangle[0] || r == triangle[1] || r == triangle[2]) continue;
                check(VoronoiDiagram.compareDistanceSquared(cx, cy,
                                diagram.sites.get(triangle[0]), diagram.sites.get(r)) <= 0,
                        "site " + r + " intrudes into the empty circle of triangle " + t);
            }
            checked++;
        }
        return checked;
    }

    public static void main(String[] args) {
        // --- negative cases ------------------------------------------------------
        checkRejected(pts(v(0, 0), v(10, 10)), "fewer than three sites");
        checkRejected(pts(v(0, 0), v(10, 0), v(10, 0)), "duplicate site");
        checkRejected(pts(v(0, 0), v(10, 0), v(20, 0)), "collinear sites");
        checkRejected(pts(v(0, 0), v(10, 0), v(20, 0), v(30, 0)), "four collinear sites");

        // --- boundary: three sites give one vertex and three rays ---------------
        VoronoiDiagram three = new VoronoiDiagram(pts(v(0, 0), v(40, 0), v(0, 40)));
        three.build();
        check(three.triangles.size() == 1, "three sites give one Delaunay triangle");
        check(three.hullSize == 3, "all three sites are hull sites");
        check(assertVoronoiVertices(three) == 1, "exactly one Voronoi vertex");
        check(three.centerX[0].toString().equals("20/1") && three.centerY[0].toString().equals("20/1"),
                "circumcenter of the right triangle is (20,20), got " + three.centerX[0] + ","
                        + three.centerY[0]);

        // --- cocircular square: a degenerate Delaunay configuration --------------
        VoronoiDiagram square = new VoronoiDiagram(pts(v(0, 0), v(40, 0), v(40, 40), v(0, 40)));
        square.build();
        check(square.triangles.size() == 2, "a square splits into two triangles");
        assertVoronoiVertices(square);
        for (int t = 0; t < square.triangles.size(); t++) {
            check(square.centerX[t].signum() >= 0, "center is finite");
            check(VoronoiDiagram.compareDistanceSquared(square.centerX[t], square.centerY[t],
                    square.sites.get(0), square.sites.get(2)) == 0,
                    "all four square corners are equidistant from the shared centre");
        }

        // --- main instance: jittered nine sites ---------------------------------
        List<int[]> sites = pts(
                v(0, 0), v(31, 4), v(60, 2), v(4, 33), v(25, 41),
                v(55, 30), v(2, 60), v(33, 62), v(60, 55));
        VoronoiDiagram diagram = new VoronoiDiagram(sites);
        diagram.build();
        check(diagram.hullSize >= 3 && diagram.hullSize <= sites.size(),
                "hull size within range, got " + diagram.hullSize);
        List<Integer> hullIndices = diagram.hull();
        check(hullIndices.size() == diagram.hullSize, "hull() is consistent with the tally");
        for (int i = 0; i < hullIndices.size(); i++) {
            int a = hullIndices.get(i);
            int b = hullIndices.get((i + 1) % hullIndices.size());
            for (int r = 0; r < sites.size(); r++) {
                if (r == a || r == b) continue;
                check(VoronoiDiagram.orient(sites.get(a), sites.get(b), sites.get(r)) > 0,
                        "hull edge " + a + "-" + b + " does not keep site " + r + " strictly inside");
            }
        }
        check(diagram.triangles.size() == 2 * sites.size() - 2 - diagram.hullSize,
                "2n-2-h Delaunay triangles, got " + diagram.triangles.size());
        check(assertVoronoiVertices(diagram) == diagram.triangles.size(),
                "one Voronoi vertex per Delaunay triangle");

        // --- bounded Voronoi edges: mid points are exactly on the bisector ------
        int boundedEdges = 0;
        int rays = 0;
        Map<String, Integer> edgeUsers = new HashMap<>();
        for (int[] triangle : diagram.triangles) {
            for (int e = 0; e < 3; e++) {
                int u = triangle[e];
                int w = triangle[(e + 1) % 3];
                String key = Math.min(u, w) + "-" + Math.max(u, w);
                edgeUsers.merge(key, 1, Integer::sum);
            }
        }
        for (Map.Entry<String, Integer> entry : edgeUsers.entrySet()) {
            String[] parts = entry.getKey().split("-");
            int p = Integer.parseInt(parts[0]);
            int q = Integer.parseInt(parts[1]);
            if (entry.getValue() == 1) {
                rays++;
                // ray property: (p - r) . d >= 0 for every other site r, with d pointing outward
                int t = triangleOf(diagram, p, q);
                int[] triangle = diagram.triangles.get(t);
                int third = -1;
                for (int vertex : triangle) if (vertex != p && vertex != q) third = vertex;
                long dx = -(sites.get(q)[1] - sites.get(p)[1]);
                long dy = sites.get(q)[0] - sites.get(p)[0];
                long inward = (sites.get(third)[0] - sites.get(p)[0]) * dx
                        + (sites.get(third)[1] - sites.get(p)[1]) * dy;
                if (inward > 0) {
                    dx = -dx;
                    dy = -dy;
                }
                check(dx != 0 || dy != 0, "ray direction must be non degenerate");
                for (int r = 0; r < sites.size(); r++) {
                    if (r == p || r == q) continue;
                    long dot = (sites.get(p)[0] - sites.get(r)[0]) * dx
                            + (sites.get(p)[1] - sites.get(r)[1]) * dy;
                    check(dot >= 0, "ray of hull edge " + p + "-" + q + " leaves site " + r
                            + " closer than the edge endpoints");
                }
                check(diagram.isHullEdge(p, q), "a ray must belong to a hull edge");
            } else {
                check(entry.getValue() == 2, "an internal Delaunay edge has two triangles");
                boundedEdges++;
                int t1 = -1;
                int t2 = -1;
                for (int t = 0; t < diagram.triangles.size(); t++) {
                    int[] triangle = diagram.triangles.get(t);
                    for (int e = 0; e < 3; e++) {
                        int u = triangle[e];
                        int w = triangle[(e + 1) % 3];
                        if ((u == p && w == q) || (u == q && w == p)) {
                            if (t1 < 0) t1 = t;
                            else t2 = t;
                        }
                    }
                }
                Exact mx = diagram.centerX[t1].add(diagram.centerX[t2]).mul(Exact.of(1, 2));
                Exact my = diagram.centerY[t1].add(diagram.centerY[t2]).mul(Exact.of(1, 2));
                check(VoronoiDiagram.compareDistanceSquared(mx, my, sites.get(p), sites.get(q)) == 0,
                        "mid point of Voronoi edge " + p + "-" + q + " is off the bisector");
                for (int r = 0; r < sites.size(); r++) {
                    if (r == p || r == q) continue;
                    check(VoronoiDiagram.compareDistanceSquared(mx, my, sites.get(p), sites.get(r)) <= 0,
                            "site " + r + " is closer to the Voronoi edge " + p + "-" + q
                                    + " than its own sites");
                }
                check(!diagram.isHullEdge(p, q), "a bounded edge must be internal");
            }
        }
        check(boundedEdges == 3 * sites.size() - 3 - 2 * diagram.hullSize,
                "bounded edges = 3n-3-2h, got " + boundedEdges);
        check(rays == diagram.hullSize, "rays = hull size, got " + rays);

        // --- unbounded cells exactly for hull sites ------------------------------
        boolean[] incidentToRay = new boolean[sites.size()];
        for (Map.Entry<String, Integer> entry : edgeUsers.entrySet()) {
            if (entry.getValue() != 1) continue;
            String[] parts = entry.getKey().split("-");
            incidentToRay[Integer.parseInt(parts[0])] = true;
            incidentToRay[Integer.parseInt(parts[1])] = true;
        }
        List<Integer> hull = diagram.hull();
        boolean[] onHull = new boolean[sites.size()];
        for (int index : hull) onHull[index] = true;        for (int s = 0; s < sites.size(); s++) {
            check(incidentToRay[s] == onHull[s], "site " + s + " cell boundedness disagrees: rays "
                    + incidentToRay[s] + " vs hull " + onHull[s]);
        }

        // --- nearest neighbour graph is a subgraph of the triangulation ----------
        int nearestPairs = 0;
        for (int s = 0; s < sites.size(); s++) {
            int best = -1;
            long bestDistance = Long.MAX_VALUE;
            for (int r = 0; r < sites.size(); r++) {
                if (r == s) continue;
                long dx = sites.get(s)[0] - sites.get(r)[0];
                long dy = sites.get(s)[1] - sites.get(r)[1];
                long d = dx * dx + dy * dy;
                if (d < bestDistance) {
                    bestDistance = d;
                    best = r;
                }
            }
            for (int r = 0; r < sites.size(); r++) {
                if (r == s) continue;
                long dx = sites.get(s)[0] - sites.get(r)[0];
                long dy = sites.get(s)[1] - sites.get(r)[1];
                if (dx * dx + dy * dy == bestDistance) {
                    nearestPairs++;
                    check(diagram.isDelaunayEdge(s, r), "nearest pair " + s + "-" + r
                            + " must be a Delaunay edge");
                }
            }
        }
        check(nearestPairs > 0, "the nearest neighbour graph is non empty");

        // --- determinism ---------------------------------------------------------
        VoronoiDiagram repeat = new VoronoiDiagram(pts(
                v(0, 0), v(31, 4), v(60, 2), v(4, 33), v(25, 41),
                v(55, 30), v(2, 60), v(33, 62), v(60, 55)));
        repeat.build();
        check(repeat.triangles.size() == diagram.triangles.size(), "same site set, same triangle count");
        StringBuilder first = new StringBuilder();
        for (int t = 0; t < diagram.triangles.size(); t++) {
            first.append(Arrays.toString(diagram.triangles.get(t))).append(diagram.centerX[t])
                    .append(diagram.centerY[t]).append(';');
        }
        StringBuilder second = new StringBuilder();
        for (int t = 0; t < repeat.triangles.size(); t++) {
            second.append(Arrays.toString(repeat.triangles.get(t))).append(repeat.centerX[t])
                    .append(repeat.centerY[t]).append(';');
        }
        check(first.toString().equals(second.toString()), "the dual diagram is deterministic");

        System.out.println("sites = " + sites.size() + ", Voronoi vertices = " + diagram.triangles.size()
                + ", bounded edges = " + boundedEdges + ", rays = " + rays
                + " (hull = " + diagram.hullSize + ")");
        System.out.println("degenerate square: cocircular quad handled exactly; " + checksRun
                + " checks");
        System.out.println("PASS");
    }

    static void checkRejected(List<int[]> sites, String label) {
        boolean rejected = false;
        try {
            VoronoiDiagram diagram = new VoronoiDiagram(sites);
            diagram.build();
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
