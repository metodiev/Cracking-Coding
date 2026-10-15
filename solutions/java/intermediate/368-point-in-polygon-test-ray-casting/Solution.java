/** Task 368 - Point-in-polygon test (ray casting). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

public class Solution {
    private static int seed = 31415;

    static long crossAt(int[] o, int[] a, int[] b) {
        return (long) (a[0] - o[0]) * (b[1] - o[1]) - (long) (a[1] - o[1]) * (b[0] - o[0]);
    }

    /** Exact test for p lying on the closed segment a-b. */
    static boolean onSegment(int[] a, int[] b, int[] p) {
        if (crossAt(a, b, p) != 0) return false;
        return Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0]) &&
                Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1]);
    }

    static boolean onBoundary(int[][] polygon, int[] p) {
        for (int i = 0; i < polygon.length; i++) {
            if (onSegment(polygon[i], polygon[(i + 1) % polygon.length], p)) return true;
        }
        return false;
    }

    /**
     * Even-odd ray casting with the half-open edge rule: an edge counts only when exactly
     * one endpoint has y > p.y, so horizontal edges never cross and a vertex shared by two
     * edges is counted once.  The intersection test is exact integer arithmetic: instead of
     * computing x_intersection, it compares (x1-px)*dy + (py-y1)*(x2-x1) against zero with
     * the sign of dy.
     */
    static boolean rayCastInside(int[][] polygon, int[] p) {
        int px = p[0], py = p[1];
        boolean inside = false;
        for (int i = 0; i < polygon.length; i++) {
            int x1 = polygon[i][0], y1 = polygon[i][1];
            int x2 = polygon[(i + 1) % polygon.length][0], y2 = polygon[(i + 1) % polygon.length][1];
            if ((y1 > py) == (y2 > py)) continue;               // ignore horizontal edges
            long dy = y2 - y1;
            long lhs = (long) (x1 - px) * dy + (long) (py - y1) * (x2 - x1);
            if ((lhs > 0) == (dy > 0)) inside = !inside;
        }
        return inside;
    }

    /** Winding number alternative: +1 per CCW crossing above p, -1 per clockwise one. */
    static int windingNumber(int[][] polygon, int[] p) {
        int px = p[0], py = p[1];
        int winding = 0;
        for (int i = 0; i < polygon.length; i++) {
            int[] a = polygon[i];
            int[] b = polygon[(i + 1) % polygon.length];
            if (a[1] <= py) {
                if (b[1] > py && crossAt(a, b, p) > 0) winding++;
            } else {
                if (b[1] <= py && crossAt(a, b, p) < 0) winding--;
            }
        }
        return winding;
    }

    /** Convex hull (CCW) used to build test polygons from LCG points. */
    static int[][] monotoneChain(int[][] points) {
        int[][] copy = points.clone();
        Arrays.sort(copy, Comparator.<int[]>comparingInt(p -> p[0]).thenComparingInt(p -> p[1]));
        List<int[]> pts = new ArrayList<>();
        for (int[] p : copy) {
            if (pts.isEmpty() || pts.get(pts.size() - 1)[0] != p[0] ||
                    pts.get(pts.size() - 1)[1] != p[1]) {
                pts.add(p.clone());
            }
        }
        if (pts.size() <= 2) return pts.toArray(new int[0][]);
        List<int[]> lower = new ArrayList<>();
        for (int[] p : pts) {
            while (lower.size() >= 2 &&
                    crossAt(lower.get(lower.size() - 2), lower.get(lower.size() - 1), p) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(p);
        }
        List<int[]> upper = new ArrayList<>();
        for (int i = pts.size() - 1; i >= 0; i--) {
            int[] p = pts.get(i);
            while (upper.size() >= 2 &&
                    crossAt(upper.get(upper.size() - 2), upper.get(upper.size() - 1), p) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(p);
        }
        List<int[]> hull = new ArrayList<>(lower.subList(0, lower.size() - 1));
        hull.addAll(upper.subList(0, upper.size() - 1));
        return hull.toArray(new int[0][]);
    }

    public static void main(String[] args) {
        int[][] square = {{0, 0}, {4, 0}, {4, 4}, {0, 4}};
        check(!onBoundary(square, new int[]{2, 2}) && rayCastInside(square, new int[]{2, 2}), "square inside");
        check(!onBoundary(square, new int[]{5, 2}) && !rayCastInside(square, new int[]{5, 2}), "square right outside");
        check(!rayCastInside(square, new int[]{-1, 2}), "square left outside");
        check(!rayCastInside(square, new int[]{2, 5}), "square above outside");
        check(onBoundary(square, new int[]{4, 4}), "square vertex boundary");
        check(onBoundary(square, new int[]{0, 2}), "square left edge boundary");
        check(onBoundary(square, new int[]{2, 0}), "square bottom edge boundary");
        check(onBoundary(square, new int[]{0, 0}), "square corner boundary");
        // (-1, 0): the horizontal ray passes straight through vertices (0,0) and (4,0);
        // the half-open rule counts both vertical edges exactly once -> outside.
        check(!rayCastInside(square, new int[]{-1, 0}) && windingNumber(square, new int[]{-1, 0}) == 0,
                "ray through two vertices");

        int[][] triangle = {{0, 0}, {6, 0}, {0, 6}};
        check(!onBoundary(triangle, new int[]{1, 1}) && rayCastInside(triangle, new int[]{1, 1}), "triangle inside");
        check(onBoundary(triangle, new int[]{3, 3}), "triangle hypotenuse boundary");
        check(!rayCastInside(triangle, new int[]{5, 2}), "triangle outside above hypotenuse");
        check(!rayCastInside(triangle, new int[]{7, 0}), "triangle outside right of x = 6");
        check(onBoundary(triangle, new int[]{0, 3}), "triangle left edge boundary");

        // Concave "arrow" polygon: a square with a triangular notch cut from the top.
        int[][] arrow = {{0, 0}, {6, 0}, {6, 6}, {3, 3}, {0, 6}};
        check(!onBoundary(arrow, new int[]{3, 1}) && rayCastInside(arrow, new int[]{3, 1}), "arrow inside bottom");
        check(!rayCastInside(arrow, new int[]{3, 4}), "arrow notch outside");
        check(!onBoundary(arrow, new int[]{1, 4}) && rayCastInside(arrow, new int[]{1, 4}), "arrow inside left");
        check(!onBoundary(arrow, new int[]{5, 4}) && rayCastInside(arrow, new int[]{5, 4}), "arrow inside right");
        check(onBoundary(arrow, new int[]{3, 3}), "arrow notch vertex boundary");
        check(onBoundary(arrow, new int[]{5, 5}), "arrow on edge y = x");
        check(!rayCastInside(arrow, new int[]{2, 6}), "arrow above notch outside");
        check(onBoundary(arrow, new int[]{0, 3}), "arrow left edge boundary");
        // (1, 3): the ray y = 3 passes through the notch vertex (3, 3) and exits at x = 6;
        // the vertex contributes two crossings that cancel, then one real crossing -> inside.
        check(!onBoundary(arrow, new int[]{1, 3}), "point (1,3) not on boundary");
        check(rayCastInside(arrow, new int[]{1, 3}), "ray through notch vertex");
        check(windingNumber(arrow, new int[]{1, 3}) == 1, "winding number of (1,3)");

        // Both algorithms must agree everywhere except on the boundary.
        int[][][] polygons = {square, triangle, arrow};
        for (int[][] polygon : polygons) {
            for (int x = -2; x <= 9; x++) {
                for (int y = -2; y <= 9; y++) {
                    int[] p = {x, y};
                    if (onBoundary(polygon, p)) continue;
                    check(rayCastInside(polygon, p) == (windingNumber(polygon, p) != 0),
                            "agreement at " + x + "," + y);
                }
            }
        }

        // LCG convex polygons from hulls: ray casting and winding number must agree.
        for (int trial = 0; trial < 30; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 3 + seed % 20;
            int[][] points = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % 12;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % 12;
                points[i] = new int[]{x, y};
            }
            int[][] polygon = monotoneChain(points);
            if (polygon.length < 3) continue;
            for (int x = -2; x <= 14; x++) {
                for (int y = -2; y <= 14; y++) {
                    int[] p = {x, y};
                    if (onBoundary(polygon, p)) continue;    // boundary is documented as undefined
                    check(rayCastInside(polygon, p) == (windingNumber(polygon, p) != 0),
                            "LCG agreement trial " + trial);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
