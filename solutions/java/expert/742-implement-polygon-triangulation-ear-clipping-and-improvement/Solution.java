/** Task 742 - Implement polygon triangulation (ear clipping) and improvement.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Part 1 - ear clipping. A vertex of a CCW simple polygon is an *ear* when its triangle turns left
 * (orient > 0) and no other polygon vertex lies inside that triangle. Clipping ears one at a time
 * (always keeping the polygon simple) yields exactly n-2 triangles and n-3 diagonals in O(n^2).
 *
 * Part 2 - improvement by Lawson flipping. For every internal diagonal (a,b) shared by the
 * triangles (a,b,c) and (b,a,d): if the quadrilateral a-c-b-d is strictly convex (a and b lie on
 * opposite sides of cd) and d lies strictly inside the circumcircle of (a,b,c) - the exact
 * incircle determinant - then the diagonal is flipped to (c,d). Each flip locally maximises the
 * minimum angle of that quadrilateral, so the global minimum angle is non-decreasing and the loop
 * terminates in a locally Delaunay triangulation, which is the standard max-min-angle improvement.
 *
 * Simplification (documented). Ear clipping is the plain O(n^2) version (no priority queue of ears,
 * no monotone decomposition), the improvement is one Lawson pass to a *locally* Delaunay state
 * (not a globally optimal minimum-weight triangulation, which is dynamic programming and out of
 * scope), only strictly convex flippable quads are flipped, and the input must be a simple CCW
 * polygon with distinct, non-collinear vertex triples and small integer coordinates (so the
 * incircle determinant fits exactly in a long).
 *
 * Verified. Combinatorial invariants after *every* flip (n-2 triangles, n-3 diagonals, each polygon
 * edge used once, each diagonal used twice, areas summing to the polygon area); every diagonal is
 * tested to lie inside the polygon (no polygon edge crossing, mid point strictly inside); triangle
 * interiors are sampled and each sample must be inside the polygon; the flipping loop leaves no
 * Delaunay-violating internal diagonal (checked by an independent scan); the minimum angle is
 * non-decreasing and the improvement is deterministic.
 */
import java.util.*;

/** Ear clipping followed by Lawson flipping to a locally Delaunay triangulation. */
final class Mesh {
    final List<int[]> polygon;
    final List<int[]> triangles = new ArrayList<>();
    int flips;

    Mesh(List<int[]> polygon) {
        this.polygon = polygon;
    }

    static long orient(int[] a, int[] b, int[] c) {
        return (long) (b[0] - a[0]) * (c[1] - a[1]) - (long) (b[1] - a[1]) * (c[0] - a[0]);
    }

    static long doubledArea(List<int[]> poly) {
        long sum = 0;
        for (int i = 0; i < poly.size(); i++) {
            int[] a = poly.get(i);
            int[] b = poly.get((i + 1) % poly.size());
            sum += (long) a[0] * b[1] - (long) b[0] * a[1];
        }
        return sum;
    }

    /** Determinant of the circle through a, b, c evaluated at d; > 0 means d is strictly inside. */
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

    static boolean insideOrOn(int[] a, int[] b, int[] c, int[] p) {
        return orient(a, b, p) >= 0 && orient(b, c, p) >= 0 && orient(c, a, p) >= 0;
    }

    static boolean segmentsIntersect(int[] a, int[] b, int[] c, int[] d) {
        long d1 = orient(c, d, a);
        long d2 = orient(c, d, b);
        long d3 = orient(a, b, c);
        long d4 = orient(a, b, d);
        if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
                && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
            return true;
        }
        return (d1 == 0 && onSegment(c, d, a)) || (d2 == 0 && onSegment(c, d, b))
                || (d3 == 0 && onSegment(a, b, c)) || (d4 == 0 && onSegment(a, b, d));
    }

    private static boolean onSegment(int[] a, int[] b, int[] p) {
        return Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0])
                && Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
    }

    /** Rejects non simple input before any ear can be clipped. */
    static void requireSimple(List<int[]> polygon) {
        int n = polygon.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (j == i + 1 || (i == 0 && j == n - 1)) continue;
                if (segmentsIntersect(polygon.get(i), polygon.get((i + 1) % n),
                        polygon.get(j), polygon.get((j + 1) % n))) {
                    throw new IllegalArgumentException("polygon is not simple");
                }
            }
        }
    }

    void earClip() {
        int n = polygon.size();
        if (n < 3) throw new IllegalArgumentException("need at least three vertices, got " + n);
        requireSimple(polygon);
        List<Integer> remaining = new ArrayList<>();
        for (int i = 0; i < n; i++) remaining.add(i);
        if (doubledArea(polygon) < 0) Collections.reverse(remaining);
        int safety = 0;
        while (remaining.size() > 3) {
            boolean clipped = false;
            for (int i = 0; i < remaining.size(); i++) {
                int a = remaining.get((i - 1 + remaining.size()) % remaining.size());
                int b = remaining.get(i);
                int c = remaining.get((i + 1) % remaining.size());
                if (isEar(remaining, a, b, c)) {
                    triangles.add(new int[]{a, b, c});
                    remaining.remove(i);
                    clipped = true;
                    break;
                }
            }
            if (!clipped) throw new IllegalArgumentException("no ear found: degenerate polygon");
            if (++safety > 4 * n + 8) throw new IllegalArgumentException("ear clipping stalled");
        }
        triangles.add(new int[]{remaining.get(0), remaining.get(1), remaining.get(2)});
    }

    private boolean isEar(List<Integer> remaining, int a, int b, int c) {
        int[] pa = polygon.get(a);
        int[] pb = polygon.get(b);
        int[] pc = polygon.get(c);
        if (orient(pa, pb, pc) <= 0) return false;
        for (int v : remaining) {
            if (v == a || v == b || v == c) continue;
            if (insideOrOn(pa, pb, pc, polygon.get(v))) return false;
        }
        return true;
    }

    /** edge key -> the (one or two) triangle indices using that edge. */
    Map<Long, List<Integer>> edgeIndex() {
        Map<Long, List<Integer>> map = new LinkedHashMap<>();
        for (int t = 0; t < triangles.size(); t++) {
            int[] triangle = triangles.get(t);
            for (int e = 0; e < 3; e++) {
                long key = key(triangle[e], triangle[(e + 1) % 3]);
                map.computeIfAbsent(key, k -> new ArrayList<>()).add(t);
            }
        }
        return map;
    }

    static long key(int a, int b) {
        return (long) Math.min(a, b) * 1000 + Math.max(a, b);
    }

    /** Internal diagonals with their two opposite vertices, in a deterministic order. */
    List<int[]> internalDiagonals() {
        List<int[]> out = new ArrayList<>();
        Map<Long, List<Integer>> map = edgeIndex();
        List<Long> keys = new ArrayList<>(map.keySet());
        Collections.sort(keys);
        for (long k : keys) {
            List<Integer> users = map.get(k);
            if (users.size() != 2) continue;         // polygon edge (1) or a broken mesh
            int a = (int) (k / 1000);
            int b = (int) (k % 1000);
            int c = oppositeVertex(users.get(0), a, b);
            int d = oppositeVertex(users.get(1), a, b);
            out.add(new int[]{a, b, c, d});
        }
        return out;
    }

    private int oppositeVertex(int triangleIndex, int a, int b) {
        int[] triangle = triangles.get(triangleIndex);
        for (int v : triangle) if (v != a && v != b) return v;
        throw new IllegalStateException("degenerate triangle");
    }

    /**
     * Normalises a diagonal {a, b, c, d} so that the triangle (a, b, c) is CCW (c on the left of
     * a->b) and d on the other side, which is the orientation the flip formula assumes. Returns
     * null for a degenerate (collinear) configuration.
     */
    private int[] normalisedDiagonal(int[] diagonal) {
        int a = diagonal[0];
        int b = diagonal[1];
        int c = diagonal[2];
        int d = diagonal[3];
        long turn = orient(polygon.get(a), polygon.get(b), polygon.get(c));
        if (turn == 0) return null;
        return turn > 0 ? new int[]{a, b, c, d} : new int[]{a, b, d, c};
    }

    /** Lawson flip of a normalised quad; returns true when the mesh changed. */
    boolean tryFlip(int[] quad) {
        int[] normalised = normalisedDiagonal(quad);
        if (normalised == null) return false;
        int a = normalised[0];
        int b = normalised[1];
        int c = normalised[2];
        int d = normalised[3];
        int[] pa = polygon.get(a);
        int[] pb = polygon.get(b);
        int[] pc = polygon.get(c);
        int[] pd = polygon.get(d);
        long sideA = orient(pc, pd, pa);
        long sideB = orient(pc, pd, pb);
        if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) return false;  // not convex
        if (incircle(pa, pb, pc, pd) <= 0) return false;                            // already Delaunay
        // the quad in CCW order is a -> d -> b -> c, so the flipped triangles are (a,d,c), (d,b,c)
        removeTriangle(a, b, c);
        removeTriangle(a, b, d);
        triangles.add(new int[]{a, d, c});
        triangles.add(new int[]{d, b, c});
        Collections.sort(triangles, (x, y) -> {
            if (x[0] != y[0]) return Integer.compare(x[0], y[0]);
            if (x[1] != y[1]) return Integer.compare(x[1], y[1]);
            return Integer.compare(x[2], y[2]);
        });
        flips++;
        return true;
    }

    private void removeTriangle(int a, int b, int c) {
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

    /** Runs flips until no internal diagonal violates the Delaunay condition. */
    void improve() {
        int bound = 40 * polygon.size() * polygon.size() + 100;
        int rounds = 0;
        while (rounds++ < bound) {
            boolean changed = false;
            for (int[] diagonal : internalDiagonals()) {
                if (tryFlip(diagonal)) {
                    changed = true;
                    break;
                }
            }
            if (!changed) return;
        }
        throw new IllegalStateException("flip loop did not converge");
    }

    /**
     * Number of *flippable* (strictly convex) quadrilaterals whose diagonal still violates the
     * Delaunay condition. Non convex quads are skipped: no flip can change them, so the guarantee
     * the flipping loop provides is exactly "no convex quad is Delaunay-violating".
     */
    int violatingDiagonalCount() {
        int bad = 0;
        for (int[] diagonal : internalDiagonals()) {
            int[] quad = normalisedDiagonal(diagonal);
            if (quad == null) continue;
            int[] pa = polygon.get(quad[0]);
            int[] pb = polygon.get(quad[1]);
            int[] pc = polygon.get(quad[2]);
            int[] pd = polygon.get(quad[3]);
            long sideA = orient(pc, pd, pa);
            long sideB = orient(pc, pd, pb);
            if (!((sideA > 0 && sideB < 0) || (sideA < 0 && sideB > 0))) continue;   // not convex
            if (incircle(pa, pb, pc, pd) > 0) bad++;
        }
        return bad;
    }

    /** Smallest angle over all triangles, in degrees. */
    double minimumAngle() {
        double best = 180;
        for (int[] triangle : triangles) {
            double[] angles = anglesOf(triangle);
            for (double angle : angles) best = Math.min(best, angle);
        }
        return best;
    }

    double[] anglesOf(int[] triangle) {
        double[] result = new double[3];
        for (int i = 0; i < 3; i++) {
            int[] a = polygon.get(triangle[i]);
            int[] b = polygon.get(triangle[(i + 1) % 3]);
            int[] c = polygon.get(triangle[(i + 2) % 3]);
            double v1x = b[0] - a[0];
            double v1y = b[1] - a[1];
            double v2x = c[0] - a[0];
            double v2y = c[1] - a[1];
            double cos = (v1x * v2x + v1y * v2y)
                    / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y));
            cos = Math.max(-1, Math.min(1, cos));
            result[i] = Math.toDegrees(Math.acos(cos));
        }
        return result;
    }

    /** Inside-or-on test for a rational point given in scaled integer coordinates. */
    boolean insidePolygonScaled(long px, long py, long scale) {
        boolean inside = false;
        for (int i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            long xi = polygon.get(i)[0] * scale;
            long yi = polygon.get(i)[1] * scale;
            long xj = polygon.get(j)[0] * scale;
            long yj = polygon.get(j)[1] * scale;
            if ((yi > py) != (yj > py)) {
                long lhs = (px - xi) * (yj - yi);
                long rhs = (py - yi) * (xj - xi);
                if (yj > yi ? lhs < rhs : lhs > rhs) inside = !inside;
            }
        }
        return inside;
    }
}

public class Solution {
    private static long seed = 0x082EFA98EC4E6C89L;
    static int checksRun;

    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    static List<int[]> poly(int[]... vertices) { return new ArrayList<>(Arrays.asList(vertices)); }

    static int[] v(int x, int y) { return new int[]{x, y}; }

    /** Combinatorial + geometric invariants that must hold before and after every flip. */
    static void assertMeshValid(Mesh mesh, String stage) {
        int n = mesh.polygon.size();
        check(mesh.triangles.size() == n - 2, stage + ": expected n-2 triangles, got "
                + mesh.triangles.size());
        long area = 0;
        for (int[] triangle : mesh.triangles) {
            long doubled = Mesh.orient(mesh.polygon.get(triangle[0]), mesh.polygon.get(triangle[1]),
                    mesh.polygon.get(triangle[2]));
            check(doubled > 0, stage + ": every triangle must stay CCW");
            area += doubled;
        }
        check(area == Math.abs(Mesh.doubledArea(mesh.polygon)),
                stage + ": areas " + area + " != polygon area " + Math.abs(Mesh.doubledArea(mesh.polygon)));
        int polygonEdges = 0;
        int diagonals = 0;
        for (Map.Entry<Long, List<Integer>> entry : mesh.edgeIndex().entrySet()) {
            int a = (int) (entry.getKey() / 1000);
            int b = (int) (entry.getKey() % 1000);
            boolean boundary = (a + 1) % n == b || (b + 1) % n == a;
            if (boundary) {
                check(entry.getValue().size() == 1, stage + ": boundary edge used twice");
                polygonEdges++;
            } else {
                check(entry.getValue().size() == 2, stage + ": diagonal used "
                        + entry.getValue().size() + " times");
                diagonals++;
            }
        }
        check(polygonEdges == n, stage + ": not every polygon edge appears");
        check(diagonals == n - 3, stage + ": expected n-3 diagonals, got " + diagonals);
    }

    /** Every diagonal must stay inside the polygon: no boundary crossing, mid point inside. */
    static void assertDiagonalsInside(Mesh mesh) {
        int n = mesh.polygon.size();
        for (Map.Entry<Long, List<Integer>> entry : mesh.edgeIndex().entrySet()) {
            int a = (int) (entry.getKey() / 1000);
            int b = (int) (entry.getKey() % 1000);
            boolean boundary = (a + 1) % n == b || (b + 1) % n == a;
            if (boundary) continue;
            int[] pa = mesh.polygon.get(a);
            int[] pb = mesh.polygon.get(b);
            for (int i = 0; i < n; i++) {
                int[] q = mesh.polygon.get(i);
                int[] r = mesh.polygon.get((i + 1) % n);
                if (i == a || i == b || (i + 1) % n == a || (i + 1) % n == b) continue;
                check(!Mesh.segmentsIntersect(pa, pb, q, r), "diagonal crosses a polygon edge");
            }
            long midX = pa[0] + pb[0];
            long midY = pa[1] + pb[1];
            check(mesh.insidePolygonScaled(midX, midY, 2), "diagonal mid point is outside the polygon");
        }
    }

    /** Samples lattice points inside every triangle; all of them must be inside the polygon. */
    static int assertTrianglesInside(Mesh mesh, int perTriangle) {
        int tested = 0;
        for (int[] triangle : mesh.triangles) {
            int[] a = mesh.polygon.get(triangle[0]);
            int[] b = mesh.polygon.get(triangle[1]);
            int[] c = mesh.polygon.get(triangle[2]);
            int xMin = Math.min(a[0], Math.min(b[0], c[0]));
            int xMax = Math.max(a[0], Math.max(b[0], c[0]));
            int yMin = Math.min(a[1], Math.min(b[1], c[1]));
            int yMax = Math.max(a[1], Math.max(b[1], c[1]));
            boolean roomy = (xMax - xMin + 1) * (yMax - yMin + 1) >= 30;
            int found = 0;
            for (int attempt = 0; attempt < 400 && found < perTriangle; attempt++) {
                int px = xMin + nextInt(xMax - xMin + 1);
                int py = yMin + nextInt(yMax - yMin + 1);
                if (!Mesh.insideOrOn(a, b, c, new int[]{px, py})) continue;
                if (Mesh.orient(a, b, new int[]{px, py}) == 0
                        || Mesh.orient(b, c, new int[]{px, py}) == 0
                        || Mesh.orient(c, a, new int[]{px, py}) == 0) {
                    continue;   // stay away from the boundary for a clean interior test
                }
                found++;
                tested++;
                check(mesh.insidePolygonScaled(px, py, 1),
                        "triangle sample " + px + "," + py + " lies outside the polygon");
            }
            check(!roomy || found >= 1, "no interior sample found for a roomy triangle");
        }
        check(tested >= mesh.triangles.size(), "too few interior samples: " + tested);
        return tested;
    }

    public static void main(String[] args) {
        // --- rejected input ------------------------------------------------------
        boolean tooSmall = false;
        try {
            Mesh bad = new Mesh(poly(v(0, 0), v(5, 5)));
            bad.earClip();
        } catch (IllegalArgumentException expected) {
            tooSmall = true;
        }
        check(tooSmall, "fewer than three vertices must be rejected");

        boolean bowtie = false;
        try {
            Mesh bad = new Mesh(poly(v(0, 0), v(10, 10), v(10, 0), v(0, 10)));
            bad.earClip();
        } catch (IllegalArgumentException expected) {
            bowtie = true;
        }
        check(bowtie, "a self-intersecting polygon must be rejected");

        // --- boundary: a single triangle -----------------------------------------
        Mesh triangle = new Mesh(poly(v(0, 0), v(20, 0), v(6, 14)));
        triangle.earClip();
        assertMeshValid(triangle, "triangle");
        check(triangle.internalDiagonals().isEmpty(), "a triangle has no internal diagonal");
        check(triangle.minimumAngle() > 0 && triangle.minimumAngle() <= 60.0001,
                "triangle minimum angle in range");
        triangle.improve();
        check(triangle.flips == 0, "a single triangle cannot be flipped");
        assertMeshValid(triangle, "triangle after improve");

        // --- square: exactly one flippable diagonal ------------------------------
        Mesh square = new Mesh(poly(v(0, 0), v(40, 0), v(40, 40), v(0, 40)));
        square.earClip();
        assertMeshValid(square, "square");
        check(square.internalDiagonals().size() == 1, "a quad has exactly one diagonal");
        square.improve();
        assertMeshValid(square, "square after improve");
        check(square.violatingDiagonalCount() == 0, "the square must end locally Delaunay");

        // --- convex hexagon ------------------------------------------------------
        Mesh hexagon = new Mesh(poly(v(0, 0), v(30, 0), v(45, 26), v(30, 52), v(0, 52), v(-15, 26)));
        hexagon.earClip();
        assertMeshValid(hexagon, "hexagon");
        assertDiagonalsInside(hexagon);
        assertTrianglesInside(hexagon, 3);
        double beforeHexagon = hexagon.minimumAngle();
        hexagon.improve();
        assertMeshValid(hexagon, "hexagon after improve");
        assertDiagonalsInside(hexagon);
        check(hexagon.violatingDiagonalCount() == 0, "hexagon must end locally Delaunay");
        check(hexagon.minimumAngle() >= beforeHexagon - 1e-9,
                "min angle must not shrink: " + beforeHexagon + " -> " + hexagon.minimumAngle());

        // --- non convex comb: the interesting case -------------------------------
        List<int[]> comb = poly(
                v(0, 0), v(60, 0), v(60, 10), v(50, 20), v(40, 10),
                v(30, 20), v(20, 10), v(10, 20), v(0, 10));
        Mesh combMesh = new Mesh(comb);
        combMesh.earClip();
        assertMeshValid(combMesh, "comb");
        assertDiagonalsInside(combMesh);
        int combSamples = assertTrianglesInside(combMesh, 4);
        double beforeComb = combMesh.minimumAngle();
        int violationsBefore = combMesh.violatingDiagonalCount();
        combMesh.improve();
        assertMeshValid(combMesh, "comb after improve");
        assertDiagonalsInside(combMesh);
        assertTrianglesInside(combMesh, 2);
        check(combMesh.violatingDiagonalCount() == 0,
                "comb must end locally Delaunay, violations left "
                        + combMesh.violatingDiagonalCount());
        check(combMesh.minimumAngle() >= beforeComb - 1e-9,
                "comb min angle must not shrink: " + beforeComb + " -> " + combMesh.minimumAngle());
        check(combMesh.flips > 0, "the comb triangulation needed improving");
        check(combMesh.minimumAngle() > 0, "no degenerate triangles after flipping");
        check(combMesh.triangles.size() == comb.size() - 2, "comb triangle count after flips");

        // --- invariant across the whole flip sequence ----------------------------
        Mesh serpentine = new Mesh(poly(
                v(0, 0), v(50, 0), v(50, 10), v(10, 10), v(10, 20), v(50, 20),
                v(50, 30), v(10, 30), v(10, 40), v(50, 40), v(50, 50), v(0, 50)));
        serpentine.earClip();
        assertMeshValid(serpentine, "serpentine");
        assertDiagonalsInside(serpentine);
        int observedFlips = 0;
        double previousAngle = serpentine.minimumAngle();
        while (true) {
            boolean changed = false;
            for (int[] diagonal : serpentine.internalDiagonals()) {
                if (serpentine.tryFlip(diagonal)) {
                    changed = true;
                    observedFlips++;
                    assertMeshValid(serpentine, "after flip " + observedFlips);
                    assertDiagonalsInside(serpentine);
                    check(serpentine.minimumAngle() >= previousAngle - 1e-9,
                            "min angle decreased during flip " + observedFlips);
                    previousAngle = serpentine.minimumAngle();
                    break;
                }
            }
            if (!changed) break;
        }
        check(serpentine.flips == observedFlips, "flip counter matches the observed flips");
        check(serpentine.violatingDiagonalCount() == 0, "serpentine ends locally Delaunay");
        assertTrianglesInside(serpentine, 2);
        assertMeshValid(serpentine, "serpentine final");
        check(observedFlips > 0, "the serpentine needed at least one flip");

        // --- determinism and repeatability ---------------------------------------
        Mesh repeat = new Mesh(poly(
                v(0, 0), v(50, 0), v(50, 10), v(10, 10), v(10, 20), v(50, 20),
                v(50, 30), v(10, 30), v(10, 40), v(50, 40), v(50, 50), v(0, 50)));
        repeat.earClip();
        repeat.improve();
        check(repeat.flips == serpentine.flips, "same polygon, same flip count");
        StringBuilder first = new StringBuilder();
        for (int[] cell : serpentine.triangles) first.append(Arrays.toString(cell)).append(';');
        StringBuilder second = new StringBuilder();
        for (int[] cell : repeat.triangles) second.append(Arrays.toString(cell)).append(';');
        check(first.toString().equals(second.toString()), "same polygon, same final mesh");
        check(combSamples > 0, "triangle sampling actually probed points");

        System.out.println("hexagon: flips = " + hexagon.flips + ", min angle " + round2(beforeHexagon)
                + " -> " + round2(hexagon.minimumAngle()));
        System.out.println("comb: flips = " + combMesh.flips + ", min angle " + round2(beforeComb)
                + " -> " + round2(combMesh.minimumAngle()) + ", violations before = "
                + violationsBefore);
        System.out.println("serpentine: n = " + serpentine.polygon.size() + ", flips = "
                + serpentine.flips + ", min angle " + round2(serpentine.minimumAngle())
                + ", " + checksRun + " checks");
        System.out.println("PASS");
    }

    static double round2(double value) { return Math.round(value * 100) / 100.0; }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
