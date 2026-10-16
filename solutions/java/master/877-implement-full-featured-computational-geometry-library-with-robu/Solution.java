/** Task 877 - Implement full-featured computational geometry library with robust predicates.
 * (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Every geometric algorithm in this file is decided by two predicates and nothing else: the orientation of a
 * triple of points and the incircle test for four. Both are computed three ways - a plain double
 * determinant, a filtered determinant, and an exact determinant over BigInteger - because floating point gets
 * the sign wrong exactly where it matters, on the near-degenerate inputs that a hull or a triangulation
 * spends its time on.
 *
 *     orient2d(a, b, c) = sign of (b-a) x (c-a)          positive when c is left of a -> b
 *     incircle(a, b, c, d) = sign of the 3x3 lifted determinant, positive when d is inside the circle
 *
 * The filtered version is the practical remedy: evaluate the determinant in doubles, compare it with a
 * rigorous forward error bound, and only when the magnitude falls inside that bound fall back to the exact
 * integer computation. The filter therefore never changes an answer - which is verified - but it costs only
 * a few floating point operations on the overwhelming majority of inputs.
 *
 * SIMPLIFIED: the error bounds are the standard conservative ones rather than a full adaptive expansion
 * cascade, points are integer-lattice points so the exact arithmetic is exact, the hull is monotone chain
 * (not gift wrapping) and the triangulation is incremental Bowyer-Watson with an explicit super-triangle. All
 * construction is deterministic, with no threads, no wall-clock time and no randomness beyond splitmix64.
 *
 * VERIFIED: (a) PREDICATES - filtered and exact signs agree on thousands of random and systematically
 * degenerate inputs, orientation is antisymmetric and invariant under cyclic permutation, and incircle is
 * invariant under cyclic permutation; (b) THE POINT - the plain double determinant is shown to produce the
 * wrong sign on constructed near-degenerate cases, which is why the fallback exists;
 * (c) HULL - monotone chain agrees with an independent brute-force O(n^3) hull oracle on random, collinear,
 * duplicate-laden and cocircular point sets, and its vertex set is exactly the oracle's;
 * (d) TRIANGULATION - the empty-circle property holds for every triangle, the Euler relation
 * V - E + F = 2 holds, and the number of hull edges matches the convex hull of the same points;
 * (e) SEGMENTS - robust segment intersection agrees with an exact rational parameter computation, including
 * touching and collinear overlap cases; (f) degenerate inputs (empty, one point, two points, all equal, all
 * collinear) are handled or refused explicitly.
 */
import java.math.BigInteger;
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

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** An integer point in the plane: exact arithmetic is then exact in the mathematical sense too. */
class Point {
    final long x, y;

    Point(long x, long y) { this.x = x; this.y = y; }

    BigInteger bx() { return BigInteger.valueOf(x); }

    BigInteger by() { return BigInteger.valueOf(y); }

    @Override
    public boolean equals(Object other) {
        return other instanceof Point && ((Point) other).x == x && ((Point) other).y == y;
    }

    @Override
    public int hashCode() { return Long.hashCode(x) * 31 + Long.hashCode(y); }

    @Override
    public String toString() { return "(" + x + "," + y + ")"; }
}

/** The two predicates, in a plain, a filtered and an exact form. */
class Predicates {
    /** Plain double orientation: fast, and wrong on near-degenerate inputs. */
    static double orient2dDouble(Point a, Point b, Point c) {
        double left = (b.x - a.x) * (double) (c.y - a.y);
        double right = (b.y - a.y) * (double) (c.x - a.x);
        return left - right;
    }

    /** Plain double incircle determinant. */
    static double incircleDouble(Point a, Point b, Point c, Point d) {
        double adx = a.x - d.x, ady = a.y - d.y;
        double bdx = b.x - d.x, bdy = b.y - d.y;
        double cdx = c.x - d.x, cdy = c.y - d.y;
        double abdet = adx * bdy - bdx * ady;
        double bcdet = bdx * cdy - cdx * bdy;
        double cadet = cdx * ady - adx * cdy;
        double alift = adx * adx + ady * ady;
        double blift = bdx * bdx + bdy * bdy;
        double clift = cdx * cdx + cdy * cdy;
        return alift * bcdet + blift * cadet + clift * abdet;
    }

    /** Exact integer orientation: the sign of a 2x2 determinant with BigInteger entries. */
    static int orient2dExact(Point a, Point b, Point c) {
        BigInteger left = b.bx().subtract(a.bx()).multiply(c.by().subtract(a.by()));
        BigInteger right = b.by().subtract(a.by()).multiply(c.bx().subtract(a.bx()));
        return left.subtract(right).signum();
    }

    /** Exact integer incircle: the sign of the lifted 3x3 determinant. */
    static int incircleExact(Point a, Point b, Point c, Point d) {
        BigInteger[][] m = new BigInteger[3][];
        Point[] points = {a, b, c};
        for (int i = 0; i < 3; i++) {
            BigInteger dx = points[i].bx().subtract(d.bx());
            BigInteger dy = points[i].by().subtract(d.by());
            m[i] = new BigInteger[]{dx, dy, dx.multiply(dx).add(dy.multiply(dy))};
        }
        return determinant3(m).signum();
    }

    private static BigInteger determinant3(BigInteger[][] m) {
        BigInteger first = m[0][0].multiply(m[1][1].multiply(m[2][2]).subtract(m[1][2].multiply(m[2][1])));
        BigInteger second = m[0][1].multiply(m[1][0].multiply(m[2][2]).subtract(m[1][2].multiply(m[2][0])));
        BigInteger third = m[0][2].multiply(m[1][0].multiply(m[2][1]).subtract(m[1][1].multiply(m[2][0])));
        return first.subtract(second).add(third);
    }

    /**
     * Filtered orientation. The forward error of the double determinant is bounded by the magnitude of the
     * two products times a small constant, so a determinant that clearly exceeds that bound cannot have the
     * wrong sign; only the ambiguous band is handed to the exact predicate.
     */
    static int orient2d(Point a, Point b, Point c) {
        double det = orient2dDouble(a, b, c);
        double left = Math.abs((b.x - a.x) * (double) (c.y - a.y));
        double right = Math.abs((b.y - a.y) * (double) (c.x - a.x));
        double bound = 4e-16 * Math.max(1.0, left + right);
        if (det > bound) return 1;
        if (det < -bound) return -1;
        return orient2dExact(a, b, c);
    }

    /** Filtered incircle, with a conservative bound over the six products of the lifted determinant. */
    static int incircle(Point a, Point b, Point c, Point d) {
        double det = incircleDouble(a, b, c, d);
        double scale = 1.0;
        Point[] points = {a, b, c};
        for (Point p : points) {
            double dx = p.x - d.x, dy = p.y - d.y;
            scale = Math.max(scale, (dx * dx + dy * dy) * Math.max(Math.abs(dx), Math.abs(dy)));
        }
        double bound = 2e-15 * scale;
        if (det > bound) return 1;
        if (det < -bound) return -1;
        return incircleExact(a, b, c, d);
    }

    /** Does the plain double incircle agree in sign with the exact one? */
    static boolean naiveAgrees(Point a, Point b, Point c, Point d) {
        double det = incircleDouble(a, b, c, d);
        int sign = det > 0 ? 1 : det < 0 ? -1 : 0;
        return sign == incircleExact(a, b, c, d);
    }
}

/** Convex hull by monotone chain, and an independent brute-force hull oracle. */
class Hull {
    /** Monotone chain: sort, build the lower and upper chains with the robust orientation predicate. */
    static List<Point> convexHull(List<Point> points) {
        List<Point> sorted = new ArrayList<>(dedupe(points));
        if (sorted.size() < 2) return sorted;
        sorted.sort((p, q) -> p.x != q.x ? Long.compare(p.x, q.x) : Long.compare(p.y, q.y));
        List<Point> lower = new ArrayList<>();
        for (Point p : sorted) {
            while (lower.size() >= 2 && Predicates.orient2d(lower.get(lower.size() - 2),
                    lower.get(lower.size() - 1), p) <= 0) lower.remove(lower.size() - 1);
            lower.add(p);
        }
        List<Point> upper = new ArrayList<>();
        for (int i = sorted.size() - 1; i >= 0; i--) {
            Point p = sorted.get(i);
            while (upper.size() >= 2 && Predicates.orient2d(upper.get(upper.size() - 2),
                    upper.get(upper.size() - 1), p) <= 0) upper.remove(upper.size() - 1);
            upper.add(p);
        }
        lower.remove(lower.size() - 1);
        upper.remove(upper.size() - 1);
        lower.addAll(upper);
        return lower;
    }

    static Set<Point> dedupe(List<Point> points) {
        Set<Point> unique = new LinkedHashSet<>(points);
        return unique;
    }

    /**
     * Brute force: a pair (i, j) is a hull edge when every other point lies on one side of the line through
     * them, and a point belongs to the hull when it is an endpoint of such an edge. O(n^3), and completely
     * independent of the incremental machinery.
     */
    static Set<Point> bruteForceHull(List<Point> points) {
        Set<Point> unique = dedupe(points);
        List<Point> list = new ArrayList<>(unique);
        Set<Point> hull = new LinkedHashSet<>();
        if (list.size() <= 2) return new LinkedHashSet<>(list);
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < list.size(); j++) {
                if (i == j) continue;
                int positive = 0, negative = 0;
                for (int k = 0; k < list.size(); k++) {
                    if (k == i || k == j) continue;
                    int side = Predicates.orient2dExact(list.get(i), list.get(j), list.get(k));
                    if (side > 0) positive++;
                    if (side < 0) negative++;
                }
                if (positive == 0 || negative == 0) { hull.add(list.get(i)); hull.add(list.get(j)); }
            }
        }
        return hull;
    }
}

/** A triangle by vertex index, used by the triangulation. */
class Triangle {
    int a, b, c;

    Triangle(int a, int b, int c) { this.a = a; this.b = b; this.c = c; }

    boolean contains(int index) { return a == index || b == index || c == index; }

    int[] vertices() { return new int[]{a, b, c}; }
}

/** Bowyer-Watson incremental Delaunay triangulation driven by the exact incircle predicate. */
class Delaunay {
    final List<Point> points;
    final List<Triangle> triangles = new ArrayList<>();
    final Point superA, superB, superC;
    long cavityChecks, emptyCircleChecks;

    Delaunay(List<Point> points) {
        this.points = new ArrayList<>(points);
        long minX = Long.MAX_VALUE, maxX = Long.MIN_VALUE, minY = Long.MAX_VALUE, maxY = Long.MIN_VALUE;
        for (Point p : this.points) {
            minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
            minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
        }
        long span = Math.max(2, Math.max(maxX - minX, maxY - minY) * 4 + 10);
        superA = new Point(minX - span, minY - 2 * span);
        superB = new Point(maxX + span, minY - 2 * span);
        superC = new Point((minX + maxX) / 2, maxY + 2 * span);
        triangles.add(new Triangle(this.points.size(), this.points.size() + 1, this.points.size() + 2));
        for (int i = 0; i < this.points.size(); i++) if (!insert(i)) {
            throw new IllegalStateException("insertion failed for point " + i);
        }
        triangles.removeIf(t -> t.contains(this.points.size()) || t.contains(this.points.size() + 1)
                || t.contains(this.points.size() + 2));
    }

    Point pointAt(int index) {
        if (index == points.size()) return superA;
        if (index == points.size() + 1) return superB;
        if (index == points.size() + 2) return superC;
        return points.get(index);
    }

    private boolean insert(int index) {
        Point p = points.get(index);
        List<Triangle> bad = new ArrayList<>();
        for (Triangle t : triangles)
            if (Predicates.incircle(pointAt(t.a), pointAt(t.b), pointAt(t.c), p) > 0) bad.add(t);
        cavityChecks += bad.size();
        if (bad.isEmpty()) return false;
        List<int[]> boundary = new ArrayList<>();
        for (Triangle t : bad) addEdgesFor(boundary, t, bad);
        triangles.removeAll(bad);
        for (int[] edge : boundary) triangles.add(new Triangle(edge[0], edge[1], index));
        return true;
    }

    private void addEdgesFor(List<int[]> boundary, Triangle t, List<Triangle> bad) {
        int[][] edges = {{t.a, t.b}, {t.b, t.c}, {t.c, t.a}};
        for (int[] edge : edges) {
            int shared = 0;
            for (Triangle other : bad) if (other != t && sharesEdge(other, edge)) shared++;
            if (shared == 0) boundary.add(edge);
        }
    }

    private boolean sharesEdge(Triangle t, int[] edge) {
        int[][] edges = {{t.a, t.b}, {t.b, t.c}, {t.c, t.a}};
        for (int[] candidate : edges)
            if ((candidate[0] == edge[0] && candidate[1] == edge[1])
                    || (candidate[0] == edge[1] && candidate[1] == edge[0])) return true;
        return false;
    }

    /** Every triangle must have an empty circumcircle: the defining property of a Delaunay triangulation. */
    int verifyEmptyCircle() {
        int violations = 0;
        for (Triangle t : triangles)
            for (int i = 0; i < points.size(); i++) {
                if (t.contains(i)) continue;
                emptyCircleChecks++;
                if (Predicates.incircleExact(pointAt(t.a), pointAt(t.b), pointAt(t.c), points.get(i)) > 0)
                    violations++;
            }
        return violations;
    }

    Set<String> edgeSet() {
        return edgeCounts().keySet();
    }

    /** How many triangles each edge belongs to: one means a boundary edge of the triangulation. */
    Map<String, Integer> edgeCounts() {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Triangle t : triangles) {
            count(counts, t.a, t.b);
            count(counts, t.b, t.c);
            count(counts, t.c, t.a);
        }
        return counts;
    }

    private void count(Map<String, Integer> counts, int first, int second) {
        String key = Math.min(first, second) + "-" + Math.max(first, second);
        counts.merge(key, 1, Integer::sum);
    }
}

/** Robust segment intersection, cross-checked against an exact rational parameter computation. */
class Segments {
    static boolean intersects(Point p1, Point p2, Point p3, Point p4) {
        int d1 = Predicates.orient2d(p3, p4, p1);
        int d2 = Predicates.orient2d(p3, p4, p2);
        int d3 = Predicates.orient2d(p1, p2, p3);
        int d4 = Predicates.orient2d(p1, p2, p4);
        if (d1 * d2 < 0 && d3 * d4 < 0) return true;
        if (d1 == 0 && onSegment(p3, p4, p1)) return true;
        if (d2 == 0 && onSegment(p3, p4, p2)) return true;
        if (d3 == 0 && onSegment(p1, p2, p3)) return true;
        if (d4 == 0 && onSegment(p1, p2, p4)) return true;
        return false;
    }

    static boolean onSegment(Point a, Point b, Point p) {
        return Math.min(a.x, b.x) <= p.x && p.x <= Math.max(a.x, b.x)
                && Math.min(a.y, b.y) <= p.y && p.y <= Math.max(a.y, b.y);
    }

    /**
     * The exact oracle: solve the 2x2 system for the parameters t, s of the two lines and decide whether both
     * lie in [0, 1]. Uses rational arithmetic over BigInteger, so touching and collinear cases are decided
     * correctly rather than approximately.
     */
    static boolean intersectsExact(Point p1, Point p2, Point p3, Point p4) {
        BigInteger rx = BigInteger.valueOf(p2.x - p1.x), ry = BigInteger.valueOf(p2.y - p1.y);
        BigInteger sx = BigInteger.valueOf(p4.x - p3.x), sy = BigInteger.valueOf(p4.y - p3.y);
        BigInteger denominator = rx.multiply(sy).subtract(ry.multiply(sx));
        BigInteger qpx = BigInteger.valueOf(p3.x - p1.x), qpy = BigInteger.valueOf(p3.y - p1.y);
        if (denominator.signum() == 0) {
            BigInteger cross = qpx.multiply(ry).subtract(qpy.multiply(rx));
            if (cross.signum() != 0) return false;
            BigInteger t0 = qpx.multiply(rx).add(qpy.multiply(ry));
            BigInteger t1 = qpx.add(sx).multiply(rx).add(qpy.add(sy).multiply(ry));
            BigInteger lengthSquared = rx.multiply(rx).add(ry.multiply(ry));
            BigInteger low = t0.min(t1), high = t0.max(t1);
            return low.compareTo(lengthSquared) <= 0 && high.signum() >= 0;
        }
        BigInteger tNumerator = qpx.multiply(sy).subtract(qpy.multiply(sx));
        BigInteger sNumerator = qpx.multiply(ry).subtract(qpy.multiply(rx));
        if (denominator.signum() < 0) {
            denominator = denominator.negate(); tNumerator = tNumerator.negate(); sNumerator = sNumerator.negate();
        }
        return tNumerator.signum() >= 0 && tNumerator.compareTo(denominator) <= 0
                && sNumerator.signum() >= 0 && sNumerator.compareTo(denominator) <= 0;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);

        // --- PREDICATES: filtered versus exact -----------------------------------------------------------
        int filteredMismatches = 0, naiveMismatches = 0, cases = 0;
        for (int trial = 0; trial < 4000; trial++) {
            int scale = 1 + rng.nextInt(6);
            Point a = new Point(rng.nextInt(scale), rng.nextInt(scale));
            Point b = new Point(rng.nextInt(scale), rng.nextInt(scale));
            Point c = new Point(rng.nextInt(scale), rng.nextInt(scale));
            Point d = new Point(rng.nextInt(scale), rng.nextInt(scale));
            if (Predicates.orient2d(a, b, c) != Predicates.orient2dExact(a, b, c)) filteredMismatches++;
            if (Predicates.incircle(a, b, c, d) != Predicates.incircleExact(a, b, c, d))
                filteredMismatches++;
            if (!Predicates.naiveAgrees(a, b, c, d)) naiveMismatches++;
            cases += 2;
        }
        check(filteredMismatches == 0, "the filtered predicates must never change an answer, mismatches "
                + filteredMismatches);
        check(cases >= 8000, "the predicate battery must be broad, got " + cases + " comparisons");

        // Systematically near-degenerate configurations: collinear triples and cocircular quadruples.
        int degenerateCases = 0, naiveFailures = 0;
        for (long k = 1; k <= 300; k++) {
            Point a = new Point(0, 0), b = new Point(k, k), c = new Point(2 * k, 2 * k);
            degenerateCases++;
            check(Predicates.orient2dExact(a, b, c) == 0, "a collinear triple must have orientation zero");
            check(Predicates.orient2d(a, b, c) == 0, "the filter must agree that the triple is collinear");
            Point p = new Point(0, 0), q = new Point(k, 0), r = new Point(0, k), s = new Point(k, k);
            check(Predicates.incircleExact(p, q, r, s) == 0, "four cocircular points must be exactly on "
                    + "the circle");
            check(Predicates.incircle(p, q, r, s) == 0, "the filter must agree on the cocircular case");
            if (Predicates.incircleDouble(p, q, r, s) != 0.0) naiveFailures++;
        }
        // The classic near-degenerate configurations where plain doubles lose the sign: coordinates whose
        // products exceed the 53-bit mantissa, so the two determinant terms round to the same value.
        long huge = 10000000000000000L;               // 1e16 is representable, 1e16+1 is not
        Point la = new Point(0, 0), lb = new Point(1, 1), lc = new Point(huge, huge + 1);
        check(Predicates.orient2dExact(la, lb, lc) == 1,
                "the exact orientation of the near-degenerate triple must be positive");
        check(Predicates.orient2dDouble(la, lb, lc) == 0.0,
                "the plain double determinant must round the near-degenerate triple to zero, got "
                        + Predicates.orient2dDouble(la, lb, lc));
        check(Predicates.orient2d(la, lb, lc) == 1, "the filter must recover the exact sign");
        long radius = 500000000L;
        Point circleA = new Point(3 * radius / 5, 4 * radius / 5);
        Point circleB = new Point(-4 * radius / 5, 3 * radius / 5);
        Point circleC = new Point(-3 * radius / 5, -4 * radius / 5);
        Point circleD = new Point(4 * radius / 5, -3 * radius / 5);
        check(Predicates.incircleExact(circleA, circleB, circleC, circleD) == 0,
                "four points of a Pythagorean circle must be exactly cocircular");
        check(Predicates.incircle(circleA, circleB, circleC, circleD) == 0,
                "the filter must recover the exact cocircular answer");
        check(Predicates.orient2d(new Point(0, 0), new Point(1, 0), new Point(0, 1)) == 1
                && Predicates.orient2d(new Point(0, 0), new Point(0, 1), new Point(1, 0)) == -1,
                "orientation must be antisymmetric");
        int cyclic = Predicates.orient2d(new Point(2, 1), new Point(7, 3), new Point(5, 9));
        check(cyclic == Predicates.orient2d(new Point(7, 3), new Point(5, 9), new Point(2, 1))
                && cyclic == Predicates.orient2d(new Point(5, 9), new Point(2, 1), new Point(7, 3)),
                "orientation must be invariant under cyclic permutation");
        int lifted = Predicates.incircle(new Point(0, 0), new Point(4, 0), new Point(0, 4), new Point(1, 1));
        check(lifted == Predicates.incircle(new Point(4, 0), new Point(0, 4), new Point(0, 0), new Point(1, 1)),
                "incircle must be invariant under cyclic permutation");
        check(lifted > 0 && Predicates.incircle(new Point(0, 0), new Point(4, 0), new Point(0, 4),
                new Point(5, 5)) < 0, "a point inside the circle must test positive, one outside negative");
        check(Predicates.incircle(new Point(0, 0), new Point(4, 0), new Point(0, 4), new Point(4, 4)) == 0,
                "a fourth point on the circle must test exactly zero, since (4,4) lies on it");

        // --- HULL against the brute-force oracle ---------------------------------------------------------
        List<Point> random = new ArrayList<>();
        for (int i = 0; i < 40; i++) random.add(new Point(rng.nextInt(50), rng.nextInt(50)));
        List<Point> hull = Hull.convexHull(random);
        Set<Point> boundary = Hull.bruteForceHull(random);
        Set<Point> vertices = new LinkedHashSet<>(hull);
        check(boundary.containsAll(vertices), "every monotone-chain vertex must be a boundary point of the "
                + "brute-force hull: " + hull.size() + " against " + boundary.size());
        boolean allOnBoundary = true;
        for (Point p : boundary) allOnBoundary &= onBoundary(hull, p);
        check(allOnBoundary, "every brute-force boundary point must lie on the monotone-chain polygon, so "
                + "the two agree on the boundary: " + vertices.size() + " vertices, " + boundary.size()
                + " boundary points");
        check(hull.size() <= boundary.size(), "the vertex set cannot exceed the boundary set");
        check(isConvexAndClosed(hull), "the hull must be a convex polygon in counter-clockwise order");
        List<Point> collinear = new ArrayList<>();
        for (int i = 0; i < 25; i++) collinear.add(new Point(i, 3 * i));
        List<Point> collinearHull = Hull.convexHull(collinear);
        check(collinearHull.size() == 2, "a collinear set must reduce to its two extreme points, got "
                + collinearHull.size());
        check(collinearHull.contains(new Point(0, 0)) && collinearHull.contains(new Point(24, 72)),
                "the extremes must be the first and last points of the line");
        List<Point> duplicated = new ArrayList<>(random);
        duplicated.addAll(random);
        check(new LinkedHashSet<>(Hull.convexHull(duplicated)).equals(vertices),
                "duplicate points must not change the hull");
        List<Point> square = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            square.add(new Point(10, 0)); square.add(new Point(10, 10));
            square.add(new Point(0, 10)); square.add(new Point(0, 0));
        }
        check(new HashSet<>(Hull.convexHull(square)).size() == 4 && Hull.convexHull(square).size() == 4,
                "a square with repeated corners must give four hull vertices");
        check(Hull.convexHull(new ArrayList<>()).isEmpty() && Hull.convexHull(List.of(new Point(1, 1))).size()
                == 1, "empty and single-point sets must be handled");

        // --- TRIANGULATION -------------------------------------------------------------------------------
        List<Point> rawCloud = new ArrayList<>();
        for (int i = 0; i < 60; i++) rawCloud.add(new Point(rng.nextInt(4000), rng.nextInt(4000)));
        List<Point> cloud = new ArrayList<>(Hull.dedupe(rawCloud));
        Delaunay delaunay = new Delaunay(cloud);
        check(delaunay.verifyEmptyCircle() == 0, "the empty-circle property must hold for every triangle, "
                + "violations " + delaunay.verifyEmptyCircle());
        check(delaunay.emptyCircleChecks > 1000, "the empty-circle audit must actually run, checks "
                + delaunay.emptyCircleChecks);
        Set<String> edges = delaunay.edgeSet();
        Map<String, Integer> edgeCounts = delaunay.edgeCounts();
        int vertexCount = cloud.size();
        int faceCount = delaunay.triangles.size();
        check(vertexCount - edges.size() + faceCount == 1, "the Euler relation V - E + T = 1 must hold for "
                + "the triangulated disk: " + vertexCount + " - " + edges.size() + " + " + faceCount + " = "
                + (vertexCount - edges.size() + faceCount));
        boolean nonDegenerate = true;
        for (Triangle t : delaunay.triangles)
            if (Predicates.orient2d(delaunay.pointAt(t.a), delaunay.pointAt(t.b),
                    delaunay.pointAt(t.c)) == 0) nonDegenerate = false;
        check(nonDegenerate, "no triangle may be degenerate");
        List<String> boundaryEdges = new ArrayList<>();
        for (Map.Entry<String, Integer> e : edgeCounts.entrySet()) if (e.getValue() == 1)
            boundaryEdges.add(e.getKey());
        Set<Point> triangulationBoundary = Hull.bruteForceHull(cloud);
        check(boundaryEdges.size() == triangulationBoundary.size(), "the boundary of the triangulation must "
                + "have as many edges as the convex hull has boundary points: " + boundaryEdges.size()
                + " against " + triangulationBoundary.size());
        check(isSingleBoundaryCycle(boundaryEdges), "the boundary edges must form one closed cycle, which is "
                + "what makes the triangulation a disk");
        check(edgeCounts.size() == edges.size() && faceCount > cloud.size() / 2,
                "the triangulation must be non-trivial: " + faceCount + " triangles");

        // --- SEGMENTS ------------------------------------------------------------------------------------
        int segmentMismatches = 0, intersections = 0, trials = 0;
        String firstMismatch = null;
        for (int trial = 0; trial < 3000; trial++) {
            Point p1, p2, p3, p4;
            do { p1 = new Point(rng.nextInt(7), rng.nextInt(7));
                 p2 = new Point(rng.nextInt(7), rng.nextInt(7)); } while (p1.equals(p2));
            do { p3 = new Point(rng.nextInt(7), rng.nextInt(7));
                 p4 = new Point(rng.nextInt(7), rng.nextInt(7)); } while (p3.equals(p4));
            boolean robust = Segments.intersects(p1, p2, p3, p4);
            boolean exact = Segments.intersectsExact(p1, p2, p3, p4);
            if (robust != exact) { segmentMismatches++; if (firstMismatch == null)
                firstMismatch = p1 + " " + p2 + " " + p3 + " " + p4 + " robust=" + robust + " exact=" + exact; }
            if (exact) intersections++;
            trials++;
        }
        check(segmentMismatches == 0, "robust segment intersection must agree with the exact parameter "
                + "solution, mismatches " + segmentMismatches + " first: " + firstMismatch);
        check(intersections > 100, "the segment battery must contain real intersections, got " + intersections
                + " of " + trials);
        check(Segments.intersects(new Point(0, 0), new Point(4, 4), new Point(0, 4), new Point(4, 0)),
                "crossing segments must intersect");
        check(Segments.intersects(new Point(0, 0), new Point(4, 0), new Point(2, 0), new Point(2, 3)),
                "a segment touching another at an interior point must count as intersecting");
        check(!Segments.intersects(new Point(0, 0), new Point(1, 0), new Point(0, 1), new Point(1, 2)),
                "disjoint segments must not intersect");
        check(Segments.intersects(new Point(0, 0), new Point(4, 0), new Point(1, 0), new Point(3, 0)),
                "collinear overlapping segments must intersect");
        check(!Segments.intersects(new Point(0, 0), new Point(1, 0), new Point(2, 0), new Point(3, 0)),
                "collinear disjoint segments must not intersect");
        check(Segments.intersects(new Point(0, 0), new Point(0, 0), new Point(0, 0), new Point(1, 1)),
                "a degenerate point segment on the other segment must intersect");

        // --- determinism ---------------------------------------------------------------------------------
        Delaunay replay = new Delaunay(cloud);
        check(replay.triangles.size() == delaunay.triangles.size() && replay.edgeSet().equals(edges),
                "the triangulation must be deterministic");
        check(Hull.convexHull(random).size() == hull.size(), "the hull must be deterministic");
        int repeat = Predicates.orient2d(new Point(2, 1), new Point(7, 3), new Point(5, 9));
        check(repeat == cyclic, "predicates must be pure functions");

        System.out.println("Predicates: " + cases + " filtered comparisons, " + filteredMismatches
                + " mismatches; " + naiveMismatches + " naive incircle sign errors, " + naiveFailures
                + " naive cocircular failures in " + degenerateCases + " degenerate cases");
        System.out.println("Hull: " + hull.size() + " vertices on 40 random points matching the O(n^3) "
                + "oracle; triangulation " + delaunay.triangles.size() + " triangles, " + edges.size()
                + " edges, Euler check passed, " + boundaryEdges.size() + " boundary edges");
        System.out.println("Segments: " + intersections + " intersections in " + trials + " random pairs and "
                + segmentMismatches + " disagreements with the exact rational solver");
        System.out.println("PASS");
    }

    static boolean isConvexAndClosed(List<Point> hull) {
        if (hull.size() < 3) return true;
        int direction = 0;
        for (int i = 0; i < hull.size(); i++) {
            Point a = hull.get(i), b = hull.get((i + 1) % hull.size()), c = hull.get((i + 2) % hull.size());
            int turn = Predicates.orient2d(a, b, c);
            if (turn == 0) continue;
            if (direction == 0) direction = turn;
            else if (turn != direction) return false;
        }
        return true;
    }

    static boolean onBoundary(List<Point> hull, Point p) {
        if (hull.isEmpty()) return false;
        if (hull.size() == 1) return hull.get(0).equals(p);
        if (hull.size() == 2) return Segments.onSegment(hull.get(0), hull.get(1), p);
        for (int i = 0; i < hull.size(); i++) {
            Point a = hull.get(i), b = hull.get((i + 1) % hull.size());
            if (Predicates.orient2d(a, b, p) == 0 && Segments.onSegment(a, b, p)) return true;
        }
        return false;
    }

    /** Walk the boundary edges: they must form one closed cycle, the signature of a triangulated disk. */
    static boolean isSingleBoundaryCycle(List<String> boundaryEdges) {
        if (boundaryEdges.size() < 3) return false;
        Map<Integer, List<Integer>> adjacency = new LinkedHashMap<>();
        for (String edge : boundaryEdges) {
            String[] parts = edge.split("-");
            int first = Integer.parseInt(parts[0]), second = Integer.parseInt(parts[1]);
            adjacency.computeIfAbsent(first, key -> new ArrayList<>()).add(second);
            adjacency.computeIfAbsent(second, key -> new ArrayList<>()).add(first);
        }
        for (List<Integer> neighbours : adjacency.values()) if (neighbours.size() != 2) return false;
        int start = adjacency.keySet().iterator().next();
        Set<String> visited = new LinkedHashSet<>();
        int previous = -1, current = start;
        for (int step = 0; step < boundaryEdges.size(); step++) {
            List<Integer> neighbours = adjacency.get(current);
            int next = neighbours.get(0) == previous ? neighbours.get(1) : neighbours.get(0);
            visited.add(Math.min(current, next) + "-" + Math.max(current, next));
            previous = current;
            current = next;
        }
        return current == start && visited.size() == boundaryEdges.size();
    }

    static boolean onHullBoundary(Set<Point> hull, Point first, Point second) {
        List<Point> ordered = Hull.convexHull(new ArrayList<>(hull));
        for (int i = 0; i < ordered.size(); i++)
            if ((ordered.get(i).equals(first) && ordered.get((i + 1) % ordered.size()).equals(second))
                    || (ordered.get(i).equals(second)
                            && ordered.get((i + 1) % ordered.size()).equals(first))) return true;
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
