/** Task 366 - Compute convex hull (Graham scan). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

public class Solution {
    private static int seed = 15731;

    /** Cross product (a - o) x (b - o); > 0 means o -> a -> b turns left. */
    static long cross(int[] o, int[] a, int[] b) {
        return (long) (a[0] - o[0]) * (b[1] - o[1]) - (long) (a[1] - o[1]) * (b[0] - o[0]);
    }

    private static long dist2(int[] a, int[] b) {
        long dx = a[0] - b[0];
        long dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    private static int[][] uniqueSortedPoints(int[][] points) {
        int[][] copy = points.clone();
        Arrays.sort(copy, Comparator.<int[]>comparingInt(p -> p[0]).thenComparingInt(p -> p[1]));
        List<int[]> unique = new ArrayList<>();
        for (int[] p : copy) {
            if (unique.isEmpty() ||
                    unique.get(unique.size() - 1)[0] != p[0] ||
                    unique.get(unique.size() - 1)[1] != p[1]) {
                unique.add(p.clone());
            }
        }
        return unique.toArray(new int[0][]);
    }

    /** Convex hull, counter-clockwise, starting at the lexicographically smallest point. */
    static int[][] grahamScan(int[][] points) {
        int[][] pts = uniqueSortedPoints(points);
        if (pts.length <= 2) return pts.clone();
        final int[] pivot = pts[0];
        Integer[] order = new Integer[pts.length - 1];
        for (int i = 0; i < order.length; i++) order[i] = i + 1;
        Arrays.sort(order, (i, j) -> {
            long turn = cross(pivot, pts[i], pts[j]);
            if (turn > 0) return -1;
            if (turn < 0) return 1;
            return Long.compare(dist2(pivot, pts[i]), dist2(pivot, pts[j]));
        });
        List<int[]> hull = new ArrayList<>();
        hull.add(pivot);
        for (int idx : order) {
            int[] point = pts[idx];
            while (hull.size() >= 2 &&
                    cross(hull.get(hull.size() - 2), hull.get(hull.size() - 1), point) <= 0) {
                hull.remove(hull.size() - 1);
            }
            hull.add(point);
        }
        return hull.toArray(new int[0][]);
    }

    /** Independent gift wrapping oracle with the same conventions as grahamScan. */
    static int[][] jarvisMarch(int[][] points) {
        int[][] pts = uniqueSortedPoints(points);
        if (pts.length <= 2) return pts.clone();
        List<int[]> hull = new ArrayList<>();
        int[] start = pts[0];
        int[] current = start;
        hull.add(start);
        while (true) {
            int[] candidate = null;
            for (int[] p : pts) {
                if (Arrays.equals(p, current)) continue;
                if (candidate == null) {
                    candidate = p;
                    continue;
                }
                long turn = cross(current, candidate, p);
                if (turn < 0) {
                    candidate = p;
                } else if (turn == 0) {
                    boolean sameDirection = ((long) (p[0] - current[0]) * (candidate[0] - current[0]) +
                            (long) (p[1] - current[1]) * (candidate[1] - current[1])) > 0;
                    if (sameDirection && dist2(current, p) > dist2(current, candidate)) {
                        candidate = p;
                    }
                }
            }
            if (Arrays.equals(candidate, start)) break;
            hull.add(candidate);
            current = candidate;
            if (hull.size() > pts.length) throw new AssertionError("degenerate input");
        }
        return hull.toArray(new int[0][]);
    }

    /** Twice the signed area (shoelace); positive for counter-clockwise order. */
    static long hullArea2(int[][] hull) {
        long total = 0;
        for (int i = 0; i < hull.length; i++) {
            int[] a = hull[i];
            int[] b = hull[(i + 1) % hull.length];
            total += (long) a[0] * b[1] - (long) b[0] * a[1];
        }
        return total;
    }

    private static boolean sameHull(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        }
        return true;
    }

    private static String show(int[][] hull) {
        StringBuilder builder = new StringBuilder();
        for (int[] p : hull) builder.append(Arrays.toString(p));
        return builder.toString();
    }

    public static void main(String[] args) {
        int[][] square = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        int[][] squareHull = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        check(sameHull(grahamScan(square), squareHull), "square");
        check(sameHull(jarvisMarch(square), squareHull), "square jarvis");
        check(hullArea2(grahamScan(square)) == 8, "square twice-area 8");

        int[][] inside = {{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}};
        check(sameHull(grahamScan(inside), squareHull), "interior point excluded");
        check(sameHull(jarvisMarch(inside), squareHull), "interior point excluded jarvis");

        int[][] collinear = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
        check(sameHull(grahamScan(collinear), new int[][]{{0, 0}, {3, 0}}), "collinear run");
        check(sameHull(jarvisMarch(collinear), new int[][]{{0, 0}, {3, 0}}), "collinear jarvis");

        int[][] mixed = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {0, 3}};
        check(sameHull(grahamScan(mixed), new int[][]{{0, 0}, {3, 3}, {0, 3}}), "diagonal run");
        check(sameHull(jarvisMarch(mixed), grahamScan(mixed)), "diagonal jarvis");

        int[][] duplicates = {{0, 0}, {0, 0}, {2, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 2}};
        check(sameHull(grahamScan(duplicates), squareHull), "duplicates");
        check(sameHull(jarvisMarch(duplicates), squareHull), "duplicates jarvis");

        check(sameHull(grahamScan(new int[][]{{0, 0}, {4, 0}, {2, 3}}),
                new int[][]{{0, 0}, {4, 0}, {2, 3}}), "triangle");
        check(sameHull(grahamScan(new int[][]{{5, 7}}), new int[][]{{5, 7}}), "single point");
        check(sameHull(grahamScan(new int[][]{{0, 0}, {1, 1}}),
                new int[][]{{0, 0}, {1, 1}}), "two points");
        check(grahamScan(new int[0][]).length == 0, "no points");

        // LCG cross-check: hull size, doubled area and the exact point list vs Jarvis.
        for (int trial = 0; trial < 60; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 40;
            int modulus = 6 + seed % 20;                       // small ranges force collinearity
            int[][] points = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % modulus;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % modulus;
                points[i] = new int[]{x, y};
            }
            int[][] scan = grahamScan(points);
            int[][] march = jarvisMarch(points);
            check(scan.length == march.length, "random size trial " + trial);
            check(hullArea2(scan) == hullArea2(march), "random area trial " + trial);
            check(sameHull(scan, march), "random hull trial " + trial);
            for (int i = 0; i < scan.length && scan.length > 2; i++) {
                int[] a = scan[(i + scan.length - 1) % scan.length];
                int[] b = scan[i];
                int[] c = scan[(i + 1) % scan.length];
                check(cross(a, b, c) > 0,
                        "convex at " + show(scan) + " from " + show(points));
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
