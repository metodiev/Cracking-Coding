/** Task 372 - Polygon area via the shoelace formula (+ orientation). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */

/**
 * The shoelace (Gauss) formula sums the cross products x_i*y_{i+1} - x_{i+1}*y_i over
 * the edges of the closed polygon.  Half of that sum is the SIGNED area: positive when
 * the vertices are listed counter-clockwise, negative when clockwise, and its absolute
 * value is the area for any *simple* polygon.  Because the vertices are integers the
 * doubled area is an exact integer and the area is an exact multiple of 0.5, so all
 * comparisons below are exact (no tolerance) except for the deliberately tolerant
 * rasterization cross-check of a non-axis-aligned triangle.
 *
 * Self-intersecting / self-touching (figure-8, "bowtie") polygons: the formula is not
 * an area at all.  It returns the sum of the lobe areas weighted by their traversal
 * orientation, so a bowtie whose two lobes cancel reports 0 (or the signed difference
 * for unequal lobes).  abs() does not repair that - only simple polygons are
 * guaranteed to yield the true area.  Two bowties below document this behaviour.
 *
 * Independent oracle: rasterizedArea counts unit cells (or finer) of the bounding box
 * whose centres lie inside the polygon via ray casting.  For polygons whose edges are
 * axis-parallel with integer coordinates every cell lies entirely inside or outside,
 * so the count is EXACT and asserted equal to |shoelace area|.  For a general
 * triangle the comparison uses a documented tolerance (cells crossed by the boundary
 * contribute at most ~perimeter/samplesPerUnit).
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
    /** Twice the signed area (an exact integer for integer vertices). */
    static long signedDoubleArea(int[][] points) {
        long total = 0;
        for (int i = 0; i < points.length; i++) {
            int[] p = points[i];
            int[] q = points[(i + 1) % points.length];
            total += (long) p[0] * q[1] - (long) q[0] * p[1];
        }
        return total;
    }

    /** Signed area; exact (a multiple of 0.5) because vertices are integers. */
    static double polygonArea(int[][] points) {
        return signedDoubleArea(points) / 2.0;
    }

    /** +1 counter-clockwise, -1 clockwise, 0 degenerate (all collinear). */
    static int orientation(int[][] points) {
        long doubleArea = signedDoubleArea(points);
        return doubleArea > 0 ? 1 : (doubleArea < 0 ? -1 : 0);
    }

    /** Ray casting; sample points used below never lie on an edge. */
    static boolean pointInPolygon(double x, double y, int[][] points) {
        boolean inside = false;
        for (int i = 0; i < points.length; i++) {
            int[] p = points[i];
            int[] q = points[(i + 1) % points.length];
            if ((p[1] > y) != (q[1] > y)) {
                double xCross = p[0] + (y - p[1]) * (q[0] - p[0]) / (double) (q[1] - p[1]);
                if (xCross > x) inside = !inside;
            }
        }
        return inside;
    }

    /** Count cell centres inside the polygon; exact for integer axis-aligned edges. */
    static double rasterizedArea(int[][] points, int samplesPerUnit) {
        if (points.length < 3) return 0.0;
        int minX = points[0][0], maxX = points[0][0], minY = points[0][1], maxY = points[0][1];
        for (int[] p : points) {
            minX = Math.min(minX, p[0]);
            maxX = Math.max(maxX, p[0]);
            minY = Math.min(minY, p[1]);
            maxY = Math.max(maxY, p[1]);
        }
        double epsilon = 1e-6;                 // nudges samples off exact integer/diagonal lines
        long inside = 0;
        for (int i = minX * samplesPerUnit; i < maxX * samplesPerUnit; i++) {
            double x = (i + 0.5) / samplesPerUnit + epsilon;
            for (int j = minY * samplesPerUnit; j < maxY * samplesPerUnit; j++) {
                double y = (j + 0.5) / samplesPerUnit + epsilon;
                if (pointInPolygon(x, y, points)) inside++;
            }
        }
        return inside / (double) (samplesPerUnit * samplesPerUnit);
    }

    static int[][] lcgRectangle(Lcg lcg) {
        int width = 1 + lcg.next(6);
        int height = 1 + lcg.next(6);
        return new int[][]{{0, 0}, {width, 0}, {width, height}, {0, height}};
    }

    static int[][] lcgLShape(Lcg lcg, int[] expectedOut) {
        int a = 2 + lcg.next(7);
        int b = 1 + lcg.next(4);
        int c = 1 + lcg.next(a - 1);
        int d = b + 1 + lcg.next(5);
        expectedOut[0] = a * b + c * (d - b);
        return new int[][]{{0, 0}, {a, 0}, {a, b}, {c, b}, {c, d}, {0, d}};
    }

    static int[][] reversed(int[][] points) {
        int[][] result = new int[points.length][];
        for (int i = 0; i < points.length; i++) result[i] = points[points.length - 1 - i];
        return result;
    }

    public static void main(String[] args) {
        // --- hand-checked polygons -------------------------------------------
        int[][] square = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        check(signedDoubleArea(square) == 8, "square doubled area");
        check(polygonArea(square) == 4.0, "square area 4");
        check(orientation(square) == 1, "square is CCW");
        check(rasterizedArea(square, 1) == 4.0, "square rasterized exactly");

        int[][] smallTriangle = {{0, 0}, {1, 0}, {0, 1}};
        check(polygonArea(smallTriangle) == 0.5, "unit triangle 0.5");
        check(orientation(smallTriangle) == 1, "unit triangle CCW");

        int[][] bigTriangle = {{0, 0}, {2, 0}, {0, 2}};
        check(polygonArea(bigTriangle) == 2.0, "leg-2 triangle 2");

        int[][] clockwise = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
        check(signedDoubleArea(clockwise) == -8, "clockwise doubled area");
        check(polygonArea(clockwise) == -4.0, "clockwise negative area");
        check(Math.abs(polygonArea(clockwise)) == 4.0, "clockwise absolute area 4");
        check(orientation(clockwise) == -1, "clockwise orientation");

        int[][] lShape = {{0, 0}, {3, 0}, {3, 1}, {1, 1}, {1, 3}, {0, 3}};
        check(polygonArea(lShape) == 5.0, "L-shape area 5");
        check(rasterizedArea(lShape, 1) == 5.0, "L-shape rasterized exactly");

        // general triangle with diagonal edges: rasterized within the documented tolerance
        int[][] general = {{0, 0}, {5, 1}, {2, 4}};
        check(polygonArea(general) == 9.0, "general triangle 9");
        double estimate = rasterizedArea(general, 32);
        check(Math.abs(estimate - 9.0) <= 0.5, "general triangle rasterized within 0.5, got " + estimate);

        // collinear vertices do not change the area
        int[][] collinearExtra = {{0, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}};
        check(polygonArea(collinearExtra) == 4.0, "collinear extra vertex");
        check(orientation(new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}, {0, 0}}) == 1, "repeated vertex");

        // --- figure-8 (bowtie) documented behaviour --------------------------
        int[][] bowtie = {{0, 0}, {2, 2}, {2, 0}, {0, 2}};            // lobes +1 and -1
        check(signedDoubleArea(bowtie) == 0, "bowtie lobes cancel");
        check(polygonArea(bowtie) == 0.0, "bowtie signed area 0");
        int[][] skewedBowtie = {{0, 0}, {2, 4}, {2, 0}, {0, 2}};      // 8/3 - 2/3 = 2
        check(signedDoubleArea(skewedBowtie) == -4, "skewed bowtie doubled area");
        check(polygonArea(skewedBowtie) == -2.0, "skewed bowtie -2");

        // --- LCG cross-checks against the exact rasterization oracle ---------
        Lcg lcg = new Lcg(37211);
        for (int trial = 0; trial < 30; trial++) {
            int[][] rectangle = lcgRectangle(lcg);
            int width = 0, height = 0;
            for (int[] p : rectangle) {
                width = Math.max(width, p[0]);
                height = Math.max(height, p[1]);
            }
            check(polygonArea(rectangle) == (double) (width * height), "rect area " + trial);
            check(rasterizedArea(rectangle, 1) == (double) (width * height), "rect raster " + trial);
            check(polygonArea(reversed(rectangle)) == -polygonArea(rectangle), "rect reversed " + trial);
            check(orientation(reversed(rectangle)) == -orientation(rectangle), "rect reversed orientation " + trial);
        }
        for (int trial = 0; trial < 30; trial++) {
            int[] expected = new int[1];
            int[][] shape = lcgLShape(lcg, expected);
            check(signedDoubleArea(shape) == 2L * expected[0], "L doubled area " + trial);
            check(polygonArea(shape) == (double) expected[0], "L area " + trial);
            check(rasterizedArea(shape, 1) == (double) expected[0], "L raster " + trial);
            check(orientation(shape) == 1, "L orientation " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
