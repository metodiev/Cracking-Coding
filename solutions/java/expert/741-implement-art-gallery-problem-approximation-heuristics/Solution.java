/** Task 741 - Implement Art Gallery problem approximation heuristics.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Pipeline of the classic approximation:
 *   1. ear clipping triangulates the simple polygon into n-2 triangles (an ear is a convex vertex
 *      whose triangle contains no other polygon vertex; clipping ears one by one is O(n^2));
 *   2. guards are placed by a greedy set cover on the *dual* of that triangulation: every candidate
 *      is a polygon vertex, and it covers exactly the triangles incident to it. Because the
 *      triangles partition the polygon and every triangle is incident to its own guard, "cover all
 *      triangles" is equivalent to "see the whole polygon" - the standard reduction behind the
 *      art gallery theorem.
 * All arithmetic is exact integer arithmetic: ear tests use orientation signs, and point/coverage
 * tests are orientation tests on integer or scaled-rational coordinates.
 *
 * Simplification (documented). The task asks for heuristics, so this is the greedy triangulation
 * cover - not the exact minimum (which is NP-hard for general polygons); the optimality gap is
 * *measured* against an exhaustive search on small instances rather than assumed. Polygons are
 * simple, CCW, have distinct integer vertices and contain no collinear triples; guards live at
 * polygon vertices (vertex-guards only, no edge/point guards) and reflex-vertex localisation is
 * left to the greedy. Non-simple input (a self-intersecting bowtie) is detected and rejected.
 *
 * Verified. The triangulation covers the polygon exactly (n-2 triangles, triangle areas sum to the
 * polygon's doubled shoelace area, every triangle CCW); the greedy result is a genuine cover (every
 * triangle has at least one guard vertex, checked independently); coverage is re-verified
 * geometrically by sampling lattice points strictly inside the polygon, locating the containing
 * triangle and additionally sampling the guard-to-sample segment to confirm it never leaves the
 * polygon; the guard count is compared with the exhaustive optimum (a lower bound that the greedy
 * must not undercut) and with a provable case (a convex polygon needs exactly one guard).
 */
import java.util.*;

/** Ear clipping triangulation of a simple polygon given as integer vertices. */
final class EarClipper {
    static long orient(int[] a, int[] b, int[] c) {
        return (long) (b[0] - a[0]) * (c[1] - a[1]) - (long) (b[1] - a[1]) * (c[0] - a[0]);
    }

    static long doubledArea(List<int[]> polygon) {
        long sum = 0;
        for (int i = 0; i < polygon.size(); i++) {
            int[] a = polygon.get(i);
            int[] b = polygon.get((i + 1) % polygon.size());
            sum += (long) a[0] * b[1] - (long) b[0] * a[1];
        }
        return sum;
    }

    /** Triangle indices of the triangulation; throws when the input is not an ear-clippable polygon. */
    static List<int[]> triangulateIndices(List<int[]> polygon) {
        int n = polygon.size();
        if (n < 3) throw new IllegalArgumentException("a polygon needs at least 3 vertices, got " + n);
        requireSimple(polygon);
        List<Integer> remaining = new ArrayList<>();
        for (int i = 0; i < n; i++) remaining.add(i);
        if (doubledArea(polygon) < 0) Collections.reverse(remaining);   // normalise to CCW
        List<int[]> triangles = new ArrayList<>();
        int safety = 0;
        while (remaining.size() > 3) {
            boolean clipped = false;
            for (int i = 0; i < remaining.size(); i++) {
                int a = remaining.get((i - 1 + remaining.size()) % remaining.size());
                int b = remaining.get(i);
                int c = remaining.get((i + 1) % remaining.size());
                if (isEar(polygon, remaining, a, b, c)) {
                    triangles.add(new int[]{a, b, c});
                    remaining.remove(i);
                    clipped = true;
                    break;
                }
            }
            if (!clipped) {
                throw new IllegalArgumentException("no ear found: polygon is not simple or degenerates");
            }
            if (++safety > 4 * n + 8) throw new IllegalArgumentException("ear clipping did not converge");
        }
        triangles.add(new int[]{remaining.get(0), remaining.get(1), remaining.get(2)});
        return triangles;
    }

    static boolean isEar(List<int[]> polygon, List<Integer> remaining, int a, int b, int c) {
        int[] pa = polygon.get(a);
        int[] pb = polygon.get(b);
        int[] pc = polygon.get(c);
        if (orient(pa, pb, pc) <= 0) return false;          // reflex or collinear: not an ear
        for (int v : remaining) {
            if (v == a || v == b || v == c) continue;
            if (insideOrOn(pa, pb, pc, polygon.get(v))) return false;
        }
        return true;
    }

    static boolean insideOrOn(int[] a, int[] b, int[] c, int[] p) {
        return orient(a, b, p) >= 0 && orient(b, c, p) >= 0 && orient(c, a, p) >= 0;
    }

    /** Rejects polygons whose non adjacent edges touch or cross: the input must be simple. */
    static void requireSimple(List<int[]> polygon) {
        int n = polygon.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (j == i + 1 || (i == 0 && j == n - 1)) continue;   // adjacent edges share a vertex
                if (segmentsIntersect(polygon.get(i), polygon.get((i + 1) % n),
                        polygon.get(j), polygon.get((j + 1) % n))) {
                    throw new IllegalArgumentException("polygon is not simple: edges " + i + " and "
                            + j + " intersect");
                }
            }
        }
    }

    /** Closed segment intersection, exact in integers; collinear overlap counts as intersecting. */
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
}

/** Greedy vertex guards plus an exhaustive optimum for cross-checking. */
final class ArtGallery {
    final List<int[]> polygon;
    final List<int[]> triangles;
    List<Integer> guards = new ArrayList<>();

    ArtGallery(List<int[]> polygon) {
        this.polygon = polygon;
        this.triangles = EarClipper.triangulateIndices(polygon);
    }

    static boolean contains(int[] triangle, int vertex) {
        return triangle[0] == vertex || triangle[1] == vertex || triangle[2] == vertex;
    }

    /** Greedy set cover over the dual of the triangulation; deterministic tie breaking. */
    void greedyGuards() {
        boolean[] covered = new boolean[triangles.size()];
        guards = new ArrayList<>();
        while (true) {
            int bestVertex = -1;
            int bestCount = 0;
            for (int v = 0; v < polygon.size(); v++) {
                int count = 0;
                for (int t = 0; t < triangles.size(); t++) {
                    if (!covered[t] && contains(triangles.get(t), v)) count++;
                }
                if (count > bestCount) {
                    bestCount = count;
                    bestVertex = v;
                }
            }
            if (bestVertex < 0) break;
            guards.add(bestVertex);
            for (int t = 0; t < triangles.size(); t++) {
                if (contains(triangles.get(t), bestVertex)) covered[t] = true;
            }
        }
    }

    boolean coversEverything() {
        for (int[] triangle : triangles) {
            boolean hit = false;
            for (int guard : guards) if (contains(triangle, guard)) hit = true;
            if (!hit) return false;
        }
        return true;
    }

    /** Exhaustive minimum number of vertex guards; only for tiny polygons. */
    int exhaustiveOptimum() {
        int n = polygon.size();
        int[] combination = new int[n];
        for (int k = 1; k <= n; k++) {
            if (search(0, 0, k, combination)) return k;
        }
        return n;
    }

    private boolean search(int start, int depth, int k, int[] combination) {
        if (depth == k) {
            List<Integer> candidate = new ArrayList<>();
            for (int i = 0; i < k; i++) candidate.add(combination[i]);
            List<Integer> saved = guards;
            guards = candidate;
            boolean ok = coversEverything();
            guards = saved;
            return ok;
        }
        for (int v = start; v < polygon.size(); v++) {
            combination[depth] = v;
            if (search(v + 1, depth + 1, k, combination)) return true;
        }
        return false;
    }

    /** Exact integer ray casting for a point strictly inside a simple polygon. */
    static boolean strictlyInside(List<int[]> polygon, long px, long py) {
        boolean inside = false;
        for (int i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            long xi = polygon.get(i)[0];
            long yi = polygon.get(i)[1];
            long xj = polygon.get(j)[0];
            long yj = polygon.get(j)[1];
            if ((yi > py) != (yj > py)) {
                long lhs = (px - xi) * (yj - yi);
                long rhs = (py - yi) * (xj - xi);
                if (yj > yi ? lhs < rhs : lhs > rhs) inside = !inside;
            }
        }
        return inside;
    }

    /** Inside-or-on test for a rational point given in scaled integer coordinates. */
    static boolean insideClosure2D(List<int[]> polygon, long px, long py, long scale) {
        if (strictlyInsideScaled(polygon, px, py, scale)) return true;
        for (int i = 0; i < polygon.size(); i++) {
            int[] a = polygon.get(i);
            int[] b = polygon.get((i + 1) % polygon.size());
            long ax = a[0] * scale;
            long ay = a[1] * scale;
            long bx = b[0] * scale;
            long by = b[1] * scale;
            long cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
            if (cross != 0) continue;
            if (Math.min(ax, bx) <= px && px <= Math.max(ax, bx)
                    && Math.min(ay, by) <= py && py <= Math.max(ay, by)) {
                return true;
            }
        }
        return false;
    }

    private static boolean strictlyInsideScaled(List<int[]> polygon, long px, long py, long scale) {
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
    private static long seed = 0xA4093822299F31D0L;
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

    /** Combinatorial check of the triangulation: count, CCW, area, shared edges. */
    static void assertTriangulation(ArtGallery gallery) {
        List<int[]> polygon = gallery.polygon;
        List<int[]> triangles = gallery.triangles;
        check(triangles.size() == polygon.size() - 2,
                "ear clipping must produce n-2 triangles, got " + triangles.size());
        long triangleArea = 0;
        for (int[] t : triangles) {
            long doubled = EarClipper.orient(polygon.get(t[0]), polygon.get(t[1]), polygon.get(t[2]));
            check(doubled > 0, "every triangle must be CCW");
            triangleArea += doubled;
        }
        check(triangleArea == Math.abs(EarClipper.doubledArea(polygon)),
                "triangle areas " + triangleArea + " must sum to the polygon area "
                        + Math.abs(EarClipper.doubledArea(polygon)));
        Map<String, Integer> edgeUse = new HashMap<>();
        for (int[] t : triangles) {
            for (int e = 0; e < 3; e++) {
                int a = t[e];
                int b = t[(e + 1) % 3];
                String key = Math.min(a, b) + "-" + Math.max(a, b);
                edgeUse.merge(key, 1, Integer::sum);
            }
        }
        int polygonEdges = 0;
        int diagonals = 0;
        for (Map.Entry<String, Integer> entry : edgeUse.entrySet()) {
            String[] parts = entry.getKey().split("-");
            int a = Integer.parseInt(parts[0]);
            int b = Integer.parseInt(parts[1]);
            boolean adjacent = (a + 1) % polygon.size() == b || (b + 1) % polygon.size() == a;
            if (adjacent) {
                check(entry.getValue() == 1, "a polygon edge must belong to exactly one triangle");
                polygonEdges++;
            } else {
                check(entry.getValue() == 2, "an internal diagonal must be shared by two triangles");
                diagonals++;
            }
        }
        check(polygonEdges == polygon.size(), "all polygon edges appear in the triangulation");
        check(diagonals == polygon.size() - 3, "a triangulation has n-3 diagonals");
    }

    /** Geometric coverage oracle: samples inside the polygon must see a guard. */
    static int sampledCoverage(ArtGallery gallery, int samples, int segmentSteps) {
        List<int[]> polygon = gallery.polygon;
        int xMin = Integer.MAX_VALUE;
        int xMax = Integer.MIN_VALUE;
        int yMin = Integer.MAX_VALUE;
        int yMax = Integer.MIN_VALUE;
        for (int[] p : polygon) {
            xMin = Math.min(xMin, p[0]);
            xMax = Math.max(xMax, p[0]);
            yMin = Math.min(yMin, p[1]);
            yMax = Math.max(yMax, p[1]);
        }
        int tested = 0;
        int visible = 0;
        for (int attempt = 0; attempt < samples * 20 && tested < samples; attempt++) {
            int px = xMin + nextInt(xMax - xMin + 1);
            int py = yMin + nextInt(yMax - yMin + 1);
            if (!ArtGallery.strictlyInside(polygon, px, py)) continue;
            tested++;
            // locate the triangle containing the sample and read its guards off the triangulation
            List<Integer> guardsOnTriangle = new ArrayList<>();
            for (int t = 0; t < gallery.triangles.size(); t++) {
                int[] tri = gallery.triangles.get(t);
                if (EarClipper.insideOrOn(polygon.get(tri[0]), polygon.get(tri[1]),
                        polygon.get(tri[2]), new int[]{px, py})) {
                    for (int guard : gallery.guards) {
                        if (ArtGallery.contains(tri, guard)) guardsOnTriangle.add(guard);
                    }
                }
            }
            check(!guardsOnTriangle.isEmpty(),
                    "sample " + px + "," + py + " lies in an unguarded triangle");
            // independent visibility check: the guard-to-sample segment must stay in the polygon
            int guard = guardsOnTriangle.get(0);
            int[] g = polygon.get(guard);
            boolean staysInside = true;
            for (int i = 1; i < segmentSteps; i++) {
                long sx = (long) g[0] * segmentSteps + (long) i * (px - g[0]);
                long sy = (long) g[1] * segmentSteps + (long) i * (py - g[1]);
                if (!ArtGallery.insideClosure2D(polygon, sx, sy, segmentSteps)) staysInside = false;
            }
            if (staysInside) visible++;
        }
        check(tested == samples, "the sampler found only " + tested + " of " + samples
                + " interior lattice points");
        check(visible == samples, "only " + visible + " of " + samples
                + " samples are actually visible from a guard");
        return visible;
    }

    public static void main(String[] args) {
        // --- rejected input: too few vertices, and a self-intersecting bowtie -----
        boolean tooSmall = false;
        try {
            new ArtGallery(poly(v(0, 0), v(10, 10)));
        } catch (IllegalArgumentException expected) {
            tooSmall = true;
        }
        check(tooSmall, "fewer than three vertices must be rejected");

        boolean bowtie = false;
        try {
            ArtGallery broken = new ArtGallery(poly(v(0, 0), v(10, 10), v(10, 0), v(0, 10)));
            broken.greedyGuards();
            broken.coversEverything();
        } catch (IllegalArgumentException expected) {
            bowtie = true;
        }
        check(bowtie, "a self-intersecting polygon must be rejected");

        // --- boundary: a single triangle needs exactly one guard ----------------
        ArtGallery triangle = new ArtGallery(poly(v(0, 0), v(20, 0), v(6, 14)));
        triangle.greedyGuards();
        assertTriangulation(triangle);
        check(triangle.triangles.size() == 1, "a triangle is its own triangulation");
        check(triangle.guards.size() == 1, "a triangle needs one guard");
        check(triangle.coversEverything(), "the single guard covers the triangle");
        check(triangle.exhaustiveOptimum() == 1, "exhaustive optimum for a triangle is 1");

        // --- convex polygon: one guard at any vertex is optimal ------------------
        ArtGallery convex = new ArtGallery(poly(v(0, 0), v(40, 0), v(60, 25), v(40, 50), v(0, 50)));
        convex.greedyGuards();
        assertTriangulation(convex);
        check(convex.guards.size() == 1, "a convex polygon needs exactly one guard");
        check(convex.exhaustiveOptimum() == 1, "exhaustive optimum for a convex polygon is 1");
        check(sampledCoverage(convex, 120, 24) == 120, "convex coverage verified by sampling");

        // --- comb polygon: several guards, compared with the exhaustive optimum --
        List<int[]> comb = poly(
                v(0, 0), v(60, 0), v(60, 10), v(50, 20), v(40, 10),
                v(30, 20), v(20, 10), v(10, 20), v(0, 10));
        ArtGallery gallery = new ArtGallery(comb);
        gallery.greedyGuards();
        assertTriangulation(gallery);
        check(gallery.guards.size() >= 1, "the comb needs at least one guard");
        check(gallery.coversEverything(), "the greedy cover is complete");
        int optimum = gallery.exhaustiveOptimum();
        check(gallery.guards.size() >= optimum, "the greedy result cannot beat the exact optimum");
        check(gallery.guards.size() <= optimum + 1,
                "greedy " + gallery.guards.size() + " should stay within one guard of the optimum "
                        + optimum);
        check(optimum >= 2, "the comb cannot be watched by a single guard");
        check(sampledCoverage(gallery, 150, 32) == 150, "comb coverage verified by sampling");

        // --- monotone greedy invariant: coverage never decreases ------------------
        // A serpentine (x-monotone corridor) polygon with five arms and two reflex pockets.
        List<int[]> spiral = poly(
                v(0, 0), v(50, 0), v(50, 10), v(10, 10), v(10, 20), v(50, 20),
                v(50, 30), v(10, 30), v(10, 40), v(50, 40), v(50, 50), v(0, 50));
        ArtGallery spiralGallery = new ArtGallery(spiral);
        assertTriangulation(spiralGallery);
        boolean[] covered = new boolean[spiralGallery.triangles.size()];
        int previous = 0;
        spiralGallery.guards = new ArrayList<>();
        while (true) {
            int bestVertex = -1;
            int bestCount = 0;
            for (int vertex = 0; vertex < spiral.size(); vertex++) {
                int count = 0;
                for (int t = 0; t < spiralGallery.triangles.size(); t++) {
                    if (!covered[t] && ArtGallery.contains(spiralGallery.triangles.get(t), vertex)) {
                        count++;
                    }
                }
                if (count > bestCount) {
                    bestCount = count;
                    bestVertex = vertex;
                }
            }
            if (bestVertex < 0) break;
            spiralGallery.guards.add(bestVertex);
            for (int t = 0; t < spiralGallery.triangles.size(); t++) {
                if (ArtGallery.contains(spiralGallery.triangles.get(t), bestVertex)) covered[t] = true;
            }
            int now = 0;
            for (boolean flag : covered) if (flag) now++;
            check(now > previous, "each greedy step must strictly reduce the uncovered set");
            check(now <= spiralGallery.triangles.size(), "coverage cannot exceed the triangle count");
            previous = now;
        }
        check(spiralGallery.coversEverything(), "the spiral is fully covered");
        int spiralOptimum = spiralGallery.exhaustiveOptimum();
        check(spiralGallery.guards.size() + 1 >= spiralOptimum, "spiral greedy is within one guard");
        check(sampledCoverage(spiralGallery, 120, 32) == 120, "spiral coverage verified by sampling");

        // --- determinism and guard placement --------------------------------------
        ArtGallery repeat = new ArtGallery(comb);
        repeat.greedyGuards();
        check(repeat.guards.equals(gallery.guards), "greedy guard set is deterministic");
        for (int guard : gallery.guards) {
            check(guard >= 0 && guard < comb.size(), "guard index inside the vertex range");
        }
        check(new HashSet<>(gallery.guards).size() == gallery.guards.size(),
                "guards are distinct vertices");

        System.out.println("triangle guards = " + triangle.guards.size()
                + ", convex guards = " + convex.guards.size()
                + ", comb guards = " + gallery.guards.size() + " (optimum " + optimum + " of n = "
                + comb.size() + ")"
                + ", spiral guards = " + spiralGallery.guards.size() + " (optimum " + spiralOptimum
                + ")");
        System.out.println("sampled interior points covered: 120 + 150 + 120, " + checksRun
                + " checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
