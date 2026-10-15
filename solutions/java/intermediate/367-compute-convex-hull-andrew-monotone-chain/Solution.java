/** Task 367 - Compute convex hull (Andrew monotone chain). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

public class Solution {
    private static int seed = 90210;

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

    /** Convex hull (CCW, starting at the lexicographically smallest point); collinear dropped. */
    static int[][] monotoneChain(int[][] points) {
        int[][] pts = uniqueSortedPoints(points);
        if (pts.length <= 2) return pts.clone();
        List<int[]> lower = new ArrayList<>();
        for (int[] p : pts) {
            while (lower.size() >= 2 &&
                    cross(lower.get(lower.size() - 2), lower.get(lower.size() - 1), p) <= 0) {
                lower.remove(lower.size() - 1);
            }
            lower.add(p);
        }
        List<int[]> upper = new ArrayList<>();
        for (int i = pts.length - 1; i >= 0; i--) {
            int[] p = pts[i];
            while (upper.size() >= 2 &&
                    cross(upper.get(upper.size() - 2), upper.get(upper.size() - 1), p) <= 0) {
                upper.remove(upper.size() - 1);
            }
            upper.add(p);
        }
        List<int[]> hull = new ArrayList<>(lower.subList(0, lower.size() - 1));
        hull.addAll(upper.subList(0, upper.size() - 1));
        return hull.toArray(new int[0][]);
    }

    /** The task 366 Graham scan, kept here so both hulls can be compared directly. */
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

    static long signedArea2(int[][] hull) {
        long total = 0;
        for (int i = 0; i < hull.length; i++) {
            int[] a = hull[i];
            int[] b = hull[(i + 1) % hull.length];
            total += (long) a[0] * b[1] - (long) b[0] * a[1];
        }
        return total;
    }

    /** Counter-clockwise order starting at the lexicographically smallest point. */
    static int[][] normalize(int[][] hull) {
        if (hull.length <= 1) return hull.clone();
        List<int[]> pts = new ArrayList<>(Arrays.asList(hull));
        if (signedArea2(hull) < 0) Collections.reverse(pts);
        int start = 0;
        for (int i = 1; i < pts.size(); i++) {
            int[] p = pts.get(i);
            int[] best = pts.get(start);
            if (p[0] < best[0] || (p[0] == best[0] && p[1] < best[1])) start = i;
        }
        List<int[]> rotated = new ArrayList<>(pts.subList(start, pts.size()));
        rotated.addAll(pts.subList(0, start));
        return rotated.toArray(new int[0][]);
    }

    private static boolean sameHull(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        // Same hand-checked sets as task 366; both algorithms must return the same hull.
        Object[][] cases = {
            {"square", new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}},
                    new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}}},
            {"inside", new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1}},
                    new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}}},
            {"collinear", new int[][]{{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                    new int[][]{{0, 0}, {3, 0}}},
            {"mixed", new int[][]{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {0, 3}},
                    new int[][]{{0, 0}, {3, 3}, {0, 3}}},
            {"duplicates", new int[][]{{0, 0}, {0, 0}, {2, 0}, {2, 0}, {2, 2}, {0, 2}, {2, 2}},
                    new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}}},
            {"triangle", new int[][]{{0, 0}, {4, 0}, {2, 3}},
                    new int[][]{{0, 0}, {4, 0}, {2, 3}}},
        };
        for (Object[] testCase : cases) {
            String name = (String) testCase[0];
            int[][] points = (int[][]) testCase[1];
            int[][] expected = (int[][]) testCase[2];
            check(sameHull(monotoneChain(points), expected), name + " monotone chain");
            check(sameHull(grahamScan(points), expected), name + " graham scan");
            check(sameHull(normalize(monotoneChain(points)), normalize(grahamScan(points))), name);
        }
        check(sameHull(normalize(monotoneChain(new int[][]{{5, 7}})), normalize(grahamScan(new int[][]{{5, 7}}))), "single");
        check(sameHull(normalize(monotoneChain(new int[][]{{0, 0}, {1, 1}})),
                normalize(grahamScan(new int[][]{{0, 0}, {1, 1}}))), "pair");
        check(monotoneChain(new int[0][]).length == 0, "empty");
        check(signedArea2(monotoneChain(new int[][]{{0, 0}, {2, 0}, {2, 2}, {0, 2}})) == 8,
                "square twice-area 8");

        // Normalization really canonicalizes: reversed clockwise input flips back.
        int[][] clockwise = {{0, 0}, {0, 2}, {2, 2}, {2, 0}};
        int[][] canonical = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        check(signedArea2(clockwise) < 0, "clockwise signed area negative");
        check(sameHull(normalize(clockwise), canonical), "normalize flips clockwise");
        int[][] rotated = {{2, 2}, {0, 2}, {0, 0}, {2, 0}};
        check(sameHull(normalize(rotated), canonical), "normalize rotates to smallest");

        // LCG cross-check: monotone chain and Graham scan agree after normalization.
        for (int trial = 0; trial < 80; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 45;
            int modulus = 5 + seed % 25;                       // small ranges force collinearity
            int[][] points = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x = seed % modulus;
                seed = (seed * 75 + 74) % 65537;
                int y = seed % modulus;
                points[i] = new int[]{x, y};
            }
            int[][] chain = normalize(monotoneChain(points));
            int[][] scan = normalize(grahamScan(points));
            check(sameHull(chain, scan), "random trial " + trial);
            check(signedArea2(chain) >= 0, "random CCW trial " + trial);
            for (int i = 0; i < chain.length && chain.length > 2; i++) {
                int[] a = chain[(i + chain.length - 1) % chain.length];
                int[] b = chain[i];
                int[] c = chain[(i + 1) % chain.length];
                check(cross(a, b, c) > 0, "convex trial " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
