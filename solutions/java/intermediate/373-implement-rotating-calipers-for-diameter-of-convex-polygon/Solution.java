/** Task 373 - Diameter of a convex polygon by rotating calipers. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

/**
 * The diameter of a point set is its farthest pair; the farthest pair of a set always
 * lies on the convex hull, so the problem reduces to the diameter of a convex polygon.
 * Rotating calipers walk an antipodal vertex j along the hull for each directed edge
 * (i, i+1): j advances while the triangle area (i, i+1, j+1) exceeds (i, i+1, j).
 * Every diameter pair is antipodal to some edge, so checking the distance from both
 * edge endpoints to j for all i finds the exact maximum.  All arithmetic is integer:
 * the result is the exact SQUARED distance; a float square root is taken only to
 * compare against the classic square = side*sqrt(2) expectation (relative tolerance
 * 1e-12, the standard double-sqrt tolerance).
 *
 * The hull is built by Andrew's monotone chain (duplicate points removed, collinear
 * points dropped).  Degenerate cases are documented: empty/one point -> diameter 0;
 * two distinct points or a fully collinear set -> hull is the two extreme points.
 * Cross-checked against the brute-force O(h^2) scan over the hull.
 *
 * Independent oracle: brute force over ALL input points (not just the hull) - the
 * farthest pair of any point set is a hull pair, so all three numbers must agree.
 */

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

public class Solution {
    /** Andrew's monotone chain: counter-clockwise hull without collinear points. */
    static int[][] convexHull(int[][] points) {
        TreeSet<Long> unique = new TreeSet<>();
        for (int[] p : points) unique.add(((long) p[0] << 32) ^ (p[1] & 0xFFFFFFFFL));
        List<int[]> sorted = new ArrayList<>();
        for (long key : unique) sorted.add(new int[]{(int) (key >> 32), (int) key});
        if (sorted.size() <= 2) return sorted.toArray(new int[0][]);

        List<int[]> lower = new ArrayList<>();
        for (int[] p : sorted) {
            while (lower.size() >= 2
                    && cross(lower.get(lower.size() - 2), lower.get(lower.size() - 1), p) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(p);
        }
        List<int[]> upper = new ArrayList<>();
        for (int i = sorted.size() - 1; i >= 0; i--) {
            int[] p = sorted.get(i);
            while (upper.size() >= 2
                    && cross(upper.get(upper.size() - 2), upper.get(upper.size() - 1), p) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(p);
        }
        List<int[]> hull = new ArrayList<>();
        for (int i = 0; i + 1 < lower.size(); i++) hull.add(lower.get(i));
        for (int i = 0; i + 1 < upper.size(); i++) hull.add(upper.get(i));
        return hull.toArray(new int[0][]);
    }

    static long cross(int[] o, int[] a, int[] b) {
        return (long) (a[0] - o[0]) * (b[1] - o[1]) - (long) (a[1] - o[1]) * (b[0] - o[0]);
    }

    static long distSquared(int[] a, int[] b) {
        long dx = a[0] - b[0], dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    static long area2(int[] a, int[] b, int[] c) {
        return (long) (b[0] - a[0]) * (c[1] - a[1]) - (long) (b[1] - a[1]) * (c[0] - a[0]);
    }

    /** Rotating calipers over a CCW convex hull; exact integer result. */
    static long hullDiameterSquared(int[][] hull) {
        int h = hull.length;
        if (h <= 1) return 0;
        if (h == 2) return distSquared(hull[0], hull[1]);
        long best = 0;
        int j = 1;
        for (int i = 0; i < h; i++) {
            int ni = (i + 1) % h;
            int steps = 0;
            while (steps <= h) {                                   // guard against degenerate ties
                int nj = (j + 1) % h;
                if (area2(hull[i], hull[ni], hull[nj]) > area2(hull[i], hull[ni], hull[j])) {
                    j = nj;
                    steps++;
                } else {
                    break;
                }
            }
            best = Math.max(best, distSquared(hull[i], hull[j]));
            best = Math.max(best, distSquared(hull[ni], hull[j]));
        }
        return best;
    }

    static long bruteHullDiameterSquared(int[][] hull) {
        long best = 0;
        for (int i = 0; i < hull.length; i++) {
            for (int k = i + 1; k < hull.length; k++) {
                best = Math.max(best, distSquared(hull[i], hull[k]));
            }
        }
        return best;
    }

    static long bruteAllPointsDiameterSquared(int[][] points) {
        long best = 0;
        for (int i = 0; i < points.length; i++) {
            for (int k = i + 1; k < points.length; k++) {
                best = Math.max(best, distSquared(points[i], points[k]));
            }
        }
        return best;
    }

    static int[][] lcgPoints(int count, Lcg lcg, int extent) {
        int[][] points = new int[count][];
        for (int i = 0; i < count; i++) {
            points[i] = new int[]{lcg.next(extent), lcg.next(extent)};
        }
        return points;
    }

    public static void main(String[] args) {
        // classic square: side 3, diagonal 3*sqrt(2)
        int side = 3;
        int[][] square = {{0, 0}, {side, 0}, {side, side}, {0, side}};
        int[][] hull = convexHull(square);
        long diameter2 = hullDiameterSquared(hull);
        check(diameter2 == 2L * side * side && diameter2 == 18, "square diameter^2 = 18");
        check(hull.length == 4, "square hull has 4 vertices");
        check(Math.abs(Math.sqrt(diameter2) - side * Math.sqrt(2)) <= 1e-12 * side * Math.sqrt(2),
                "square diagonal is side*sqrt(2)");

        // hand-checked small cases
        check(hullDiameterSquared(convexHull(new int[][]{{7, 7}})) == 0, "single point");
        check(hullDiameterSquared(convexHull(new int[][]{{1, 1}, {4, 5}})) == 25, "two points");
        int[][] triangle = {{0, 0}, {4, 0}, {0, 3}};
        hull = convexHull(triangle);
        check(hull.length == 3 && hullDiameterSquared(hull) == 25, "3-4-5 triangle");
        check(hullDiameterSquared(hull) == bruteHullDiameterSquared(hull), "triangle vs oracle");
        int[][] rectangle = {{0, 0}, {2, 0}, {2, 5}, {0, 5}};
        check(hullDiameterSquared(convexHull(rectangle)) == 29, "rectangle diagonal");
        int[][] collinear = {{0, 0}, {1, 1}, {3, 3}};
        hull = convexHull(collinear);
        check(hull.length == 2 && hull[0][0] == 0 && hull[1][0] == 3, "collinear hull degenerates");
        check(hullDiameterSquared(hull) == 18, "collinear diameter");
        int[][] withInterior = {{0, 0}, {3, 0}, {3, 3}, {0, 3}, {1, 1}};
        hull = convexHull(withInterior);
        check(hull.length == 4 && hullDiameterSquared(hull) == 18, "interior point ignored");

        // LCG cross-checks: calipers == brute force on the hull == brute force on all points
        Lcg lcg = new Lcg(37313);
        for (int trial = 0; trial < 60; trial++) {
            int count = 3 + lcg.next(10);
            int[][] points = lcgPoints(count, lcg, 50);
            hull = convexHull(points);
            long calipers = hullDiameterSquared(hull);
            check(calipers == bruteHullDiameterSquared(hull), "calipers vs hull oracle " + trial);
            check(calipers == bruteAllPointsDiameterSquared(points), "calipers vs all-points oracle " + trial);
        }

        // larger deterministic set for a wider sweep
        int[][] points = lcgPoints(120, lcg, 100);
        hull = convexHull(points);
        check(hull.length >= 3, "large hull size");
        check(hullDiameterSquared(hull) == bruteAllPointsDiameterSquared(points), "large set diameter");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
