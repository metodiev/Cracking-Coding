/** Task 735 - Implement advanced geometric data structures: dynamic convex hull maintenance.
 * (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Structure. The live point set is mirrored by two ordered chains, `upper` and `lower`, each of
 * which holds the current hull vertices sorted by abscissa. Both chains are kept *strictly*
 * convex: consecutive triples of the upper chain turn right (orient < 0), consecutive triples of
 * the lower chain turn left (orient > 0). Inserting p = (x, y) into one chain:
 *   1. if p lies on or below (resp. above) the chord spanned by its two x-neighbours in the chain,
 *      p cannot be a vertex of that chain and the chain is left untouched;
 *   2. otherwise p is linked in and the neighbours whose incident turn has become concave are
 *      peeled away - first walking left, then walking right - which is the classic incremental
 *      gift-wrapping repair, O(k + log n) for k vertices removed (k may be 0).
 * The two chains therefore always describe exactly the boundary of the current hull, and every
 * query (containment, extreme point, area, diameter) is answered from them.
 *
 * Simplification (documented). The task is the *dynamic maintenance*; deletion is deliberately not
 * handled kinetically - delete() drops the point and rebuilds both chains with a monotone sweep
 * (O(n log n)), which is what production libraries fall back to for adversarial delete sequences.
 * Containment / extreme-point / diameter queries scan the O(h) boundary instead of using a tangent
 * search on a balanced BST, and the point set must have pairwise distinct abscissae (a duplicate x
 * is rejected with IllegalArgumentException). All arithmetic is exact long arithmetic - no
 * epsilons - so degenerate (collinear) inputs are classified exactly rather than "almost".
 *
 * Verified. After *every* insertion the maintained chains equal the hull of the points inserted so
 * far, cross-checked against an independently written monotone-chain hull (vertex-set equality),
 * against the convexity/extremality of every reported vertex (all other points on the inner side
 * of both incident edges), and against a shoelace area. Containment is cross-checked against a
 * brute-force "all cross products non-negative on the brute hull" oracle, extreme-point queries
 * against a scan over the entire point set, and deletion against a from-scratch rebuild.
 */
import java.util.*;

/** Dynamic convex hull over points with distinct abscissae, exact integer arithmetic. */
final class DynamicHull {
    private final TreeMap<Long, Long> all = new TreeMap<>();    // whole live point set, x -> y
    private final TreeMap<Long, Long> upper = new TreeMap<>();  // upper chain, x -> y (right turns)
    private final TreeMap<Long, Long> lower = new TreeMap<>();  // lower chain, x -> y (left turns)
    private long rebuilds;                                      // rebuilds caused by deletion

    /** Twice the signed area of triangle (a, b, c); positive when a->b->c turns left. */
    static long orient(long ax, long ay, long bx, long by, long cx, long cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    /** True when the middle vertex of a triple breaks strict convexity of the given chain. */
    private static boolean violates(long ax, long ay, long bx, long by, long cx, long cy, boolean up) {
        long turn = orient(ax, ay, bx, by, cx, cy);
        return up ? turn >= 0 : turn <= 0;
    }

    private void insertInto(TreeMap<Long, Long> chain, long x, long y, boolean up) {
        Long left = chain.lowerKey(x);
        Long right = chain.higherKey(x);
        if (left != null && right != null
                && violates(left, chain.get(left), x, y, right, chain.get(right), up)) {
            return;   // p hides under the chord of its neighbours: not a vertex of this chain
        }
        chain.put(x, y);
        while (true) {   // peel concave neighbours towards smaller x
            Long prev = chain.lowerKey(x);
            if (prev == null) break;
            Long prev2 = chain.lowerKey(prev);
            if (prev2 == null) break;
            if (violates(prev2, chain.get(prev2), prev, chain.get(prev), x, y, up)) chain.remove(prev);
            else break;
        }
        while (true) {   // peel concave neighbours towards larger x
            Long next = chain.higherKey(x);
            if (next == null) break;
            Long next2 = chain.higherKey(next);
            if (next2 == null) break;
            if (violates(x, y, next, chain.get(next), next2, chain.get(next2), up)) chain.remove(next);
            else break;
        }
    }

    /** Adds a point; the abscissa must be new (documented general-position assumption). */
    void insert(long x, long y) {
        if (all.containsKey(x)) {
            throw new IllegalArgumentException("duplicate abscissa " + x);
        }
        all.put(x, y);
        insertInto(upper, x, y, true);
        insertInto(lower, x, y, false);
    }

    /** Removes a point; returns false when it was not present. Rebuilds both chains. */
    boolean delete(long x, long y) {
        Long current = all.get(x);
        if (current == null || current.longValue() != y) return false;
        all.remove(x);
        rebuild();
        return true;
    }

    /** Monotone sweep rebuilding both chains; used by delete() and as the brute-force oracle. */
    private void rebuild() {
        upper.clear();
        lower.clear();
        rebuilds++;
        List<long[]> up = new ArrayList<>();
        List<long[]> low = new ArrayList<>();
        for (Map.Entry<Long, Long> e : all.entrySet()) {   // TreeMap iteration is ascending in x
            long x = e.getKey();
            long y = e.getValue();
            while (up.size() >= 2 && violates(up.get(up.size() - 2)[0], up.get(up.size() - 2)[1],
                    up.get(up.size() - 1)[0], up.get(up.size() - 1)[1], x, y, true)) {
                up.remove(up.size() - 1);
            }
            up.add(new long[]{x, y});
            while (low.size() >= 2 && violates(low.get(low.size() - 2)[0], low.get(low.size() - 2)[1],
                    low.get(low.size() - 1)[0], low.get(low.size() - 1)[1], x, y, false)) {
                low.remove(low.size() - 1);
            }
            low.add(new long[]{x, y});
        }
        for (long[] p : up) upper.put(p[0], p[1]);
        for (long[] p : low) lower.put(p[0], p[1]);
    }

    /** Boundary vertices in counter-clockwise order (lower chain, then upper chain backwards). */
    List<long[]> hullCCW() {
        List<long[]> poly = new ArrayList<>();
        for (long[] p : chainPoints(lower, false)) appendDistinct(poly, p);
        for (long[] p : chainPoints(upper, true)) appendDistinct(poly, p);
        if (poly.size() > 1) {
            long[] first = poly.get(0);
            long[] last = poly.get(poly.size() - 1);
            if (first[0] == last[0] && first[1] == last[1]) poly.remove(poly.size() - 1);
        }
        return poly;
    }

    private static void appendDistinct(List<long[]> poly, long[] p) {
        if (poly.isEmpty()) { poly.add(p); return; }
        long[] last = poly.get(poly.size() - 1);
        if (last[0] != p[0] || last[1] != p[1]) poly.add(p);
    }

    private static List<long[]> chainPoints(TreeMap<Long, Long> chain, boolean descending) {
        List<long[]> out = new ArrayList<>();
        for (Map.Entry<Long, Long> e : (descending ? chain.descendingMap() : chain).entrySet()) {
            out.add(new long[]{e.getKey(), e.getValue()});
        }
        return out;
    }

    int pointCount() { return all.size(); }
    int size() { return upper.size() + lower.size(); }
    long rebuildCount() { return rebuilds; }

    /** Inside-or-on test for a convex CCW polygon; degenerate (2-vertex) hulls become segments. */
    boolean contains(long qx, long qy) {
        List<long[]> poly = hullCCW();
        if (poly.isEmpty()) return false;
        if (poly.size() == 1) return poly.get(0)[0] == qx && poly.get(0)[1] == qy;
        int n = poly.size();
        for (int i = 0; i < n; i++) {
            long[] a = poly.get(i);
            long[] b = poly.get((i + 1) % n);
            if (orient(a[0], a[1], b[0], b[1], qx, qy) < 0) return false;
        }
        return true;
    }

    /** Vertex maximizing the dot product with (dx, dy), or null for an empty hull. */
    long[] extreme(long dx, long dy) {
        List<long[]> poly = hullCCW();
        long[] best = null;
        long bestValue = 0;
        for (long[] p : poly) {
            long value = p[0] * dx + p[1] * dy;
            if (best == null || value > bestValue) {
                best = p;
                bestValue = value;
            }
        }
        return best;
    }

    /** Twice the hull area (kept doubled so the result stays an exact integer). */
    long doubledArea() {
        List<long[]> poly = hullCCW();
        if (poly.size() < 3) return 0;
        long sum = 0;
        for (int i = 0; i < poly.size(); i++) {
            long[] a = poly.get(i);
            long[] b = poly.get((i + 1) % poly.size());
            sum += a[0] * b[1] - b[0] * a[1];
        }
        return Math.abs(sum);
    }

    /** Farthest pair by squared distance, found by an exhaustive scan of the boundary. */
    long diameterSquared() {
        List<long[]> poly = hullCCW();
        long best = 0;
        for (int i = 0; i < poly.size(); i++) {
            for (int j = i + 1; j < poly.size(); j++) {
                long dx = poly.get(i)[0] - poly.get(j)[0];
                long dy = poly.get(i)[1] - poly.get(j)[1];
                best = Math.max(best, dx * dx + dy * dy);
            }
        }
        return best;
    }
}

public class Solution {
    private static long seed = 0x243F6A8885A308D3L;

    /** splitmix64 - deterministic; Math.random is never used. */
    static long nextLong() {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    /**
     * Independent hull oracle: Andrew's monotone chain over a copy of the points, written with two
     * separate chains (lower over ascending x, upper over descending x) so that it shares no code
     * with DynamicHull's chain repair. CCW, collinear vertices removed, strict turns only.
     */
    static List<long[]> bruteHull(List<long[]> points) {
        List<long[]> pts = new ArrayList<>(points);
        pts.sort((p, q) -> p[0] != q[0] ? Long.compare(p[0], q[0]) : Long.compare(p[1], q[1]));
        if (pts.size() <= 2) return pts;
        List<long[]> lower = new ArrayList<>();
        for (long[] p : pts) {
            while (lower.size() >= 2 && !leftTurn(lower.get(lower.size() - 2),
                    lower.get(lower.size() - 1), p)) {
                lower.remove(lower.size() - 1);
            }
            lower.add(p);
        }
        List<long[]> upper = new ArrayList<>();
        for (int i = pts.size() - 1; i >= 0; i--) {
            long[] p = pts.get(i);
            while (upper.size() >= 2 && !leftTurn(upper.get(upper.size() - 2),
                    upper.get(upper.size() - 1), p)) {
                upper.remove(upper.size() - 1);
            }
            upper.add(p);
        }
        List<long[]> hull = new ArrayList<>(lower.subList(0, lower.size() - 1));
        hull.addAll(upper.subList(0, upper.size() - 1));
        return hull;
    }

    private static boolean leftTurn(long[] a, long[] b, long[] c) {
        return DynamicHull.orient(a[0], a[1], b[0], b[1], c[0], c[1]) > 0;
    }

    static String vertexKey(long[] p) { return p[0] + "," + p[1]; }

    static List<String> vertexKeys(List<long[]> poly) {
        List<String> keys = new ArrayList<>();
        for (long[] p : poly) keys.add(vertexKey(p));
        Collections.sort(keys);
        return keys;
    }

    /** Every polygon vertex must have all points strictly on the inner side of both its edges. */
    static boolean isExtremalPolygon(List<long[]> poly, List<long[]> points) {
        int n = poly.size();
        if (n < 3) return true;
        for (int i = 0; i < n; i++) {
            long[] a = poly.get(i);
            long[] b = poly.get((i + 1) % n);
            for (int j = 0; j < n; j++) {
                if (j == i || j == (i + 1) % n) continue;
                if (DynamicHull.orient(a[0], a[1], b[0], b[1], poly.get(j)[0], poly.get(j)[1]) < 0) {
                    return false;   // polygon not convex / not CCW
                }
            }
            for (long[] p : points) {
                if (DynamicHull.orient(a[0], a[1], b[0], b[1], p[0], p[1]) < 0) return false;
            }
        }
        return true;
    }

    static boolean insideOracle(List<long[]> brutePoly, long qx, long qy) {
        int n = brutePoly.size();
        if (n == 1) return brutePoly.get(0)[0] == qx && brutePoly.get(0)[1] == qy;
        for (int i = 0; i < n; i++) {
            long[] a = brutePoly.get(i);
            long[] b = brutePoly.get((i + 1) % n);
            if (DynamicHull.orient(a[0], a[1], b[0], b[1], qx, qy) < 0) return false;
        }
        return true;
    }

    /** Random points with pairwise distinct abscissae, built from a shuffled x-prefix. */
    static List<long[]> randomPoints(int count, int span) {
        List<Long> xs = new ArrayList<>();
        for (long x = 0; x < span; x++) xs.add(x);
        for (int i = xs.size() - 1; i > 0; i--) {   // deterministic Fisher-Yates
            int j = nextInt(i + 1);
            Collections.swap(xs, i, j);
        }
        List<long[]> pts = new ArrayList<>();
        for (int i = 0; i < count; i++) pts.add(new long[]{xs.get(i), nextInt(span)});
        return pts;
    }

    public static void main(String[] args) {
        // --- degenerate inputs --------------------------------------------------
        DynamicHull empty = new DynamicHull();
        check(empty.pointCount() == 0, "empty hull has no points");
        check(empty.hullCCW().isEmpty(), "empty hull has no vertices");
        check(empty.doubledArea() == 0, "empty hull area is zero");
        check(!empty.contains(0, 0), "empty hull contains nothing");
        check(empty.extreme(1, 1) == null, "empty hull has no extreme point");

        DynamicHull single = new DynamicHull();
        single.insert(5, -3);
        check(single.contains(5, -3) && !single.contains(5, -2), "single point hull");
        check(single.extreme(1, 0)[0] == 5 && single.extreme(-1, 0)[0] == 5, "single extreme point");
        check(single.size() == 2 && single.doubledArea() == 0,
                "single point: upper and lower chain both hold the shared vertex");

        // --- rejected input (negative check) ------------------------------------
        boolean rejected = false;
        try {
            single.insert(5, 9);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "duplicate abscissa must be rejected");
        check(single.pointCount() == 1, "rejected insert left the structure untouched");

        // --- collinear input ----------------------------------------------------
        List<long[]> line = new ArrayList<>(Arrays.asList(
                new long[]{0, 0}, new long[]{1, 1}, new long[]{2, 2}, new long[]{3, 3}, new long[]{4, 4}));
        DynamicHull degenerate = new DynamicHull();
        for (long[] p : line) degenerate.insert(p[0], p[1]);
        check(degenerate.hullCCW().size() == 2, "collinear hull collapses to two endpoints");
        check(degenerate.doubledArea() == 0, "collinear hull has zero area");
        check(degenerate.contains(2, 2) && !degenerate.contains(2, 3), "collinear hull is a segment");
        check(degenerate.size() == 4, "collinear chains both keep the two endpoints");

        // --- incremental insertion checked after every single insert -------------
        List<long[]> cloud = randomPoints(44, 60);
        DynamicHull dynamic = new DynamicHull();
        List<long[]> inserted = new ArrayList<>();
        for (int step = 0; step < cloud.size(); step++) {
            long[] p = cloud.get(step);
            dynamic.insert(p[0], p[1]);
            inserted.add(p);
            List<long[]> expected = bruteHull(inserted);
            check(vertexKeys(dynamic.hullCCW()).equals(vertexKeys(expected)),
                    "hull mismatch after insert #" + step + " of " + p[0] + "," + p[1]);
            check(isExtremalPolygon(dynamic.hullCCW(), inserted),
                    "maintained hull is not convex/extremal after insert #" + step);
            check(dynamic.doubledArea() == shoelace(expected), "area mismatch after insert #" + step);
            check(dynamic.size() == dynamic.hullCCW().size() + (dynamic.hullCCW().size() >= 2 ? 2 : 1),
                    "chain bookkeeping wrong after insert #" + step);
        }
        check(dynamic.pointCount() == 44, "all points present");
        check(dynamic.size() >= dynamic.hullCCW().size() && dynamic.size() <= 2 * 44,
                "chain sizes stay sane");

        // --- containment oracle on pseudo random probes --------------------------
        List<long[]> finalBrute = bruteHull(inserted);
        int inside = 0;
        for (int i = 0; i < 200; i++) {
            int qx = nextInt(70) - 5;
            int qy = nextInt(70) - 5;
            boolean got = dynamic.contains(qx, qy);
            boolean want = insideOracle(finalBrute, qx, qy);
            check(got == want, "containment differs at " + qx + "," + qy + ": " + got + " vs " + want);
            if (got) inside++;
        }
        check(inside > 0 && inside < 200, "probe mix actually covers inside and outside");

        // --- extreme point and diameter oracles ---------------------------------
        long[][] directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-3, 2}, {7, -5}};
        for (long[] d : directions) {
            long[] got = dynamic.extreme(d[0], d[1]);
            long bestValue = Long.MIN_VALUE;
            for (long[] p : inserted) bestValue = Math.max(bestValue, p[0] * d[0] + p[1] * d[1]);
            check(got[0] * d[0] + got[1] * d[1] == bestValue,
                    "extreme point for " + Arrays.toString(d));
        }
        long bruteDiameter = 0;
        for (int i = 0; i < inserted.size(); i++) {
            for (int j = i + 1; j < inserted.size(); j++) {
                long dx = inserted.get(i)[0] - inserted.get(j)[0];
                long dy = inserted.get(i)[1] - inserted.get(j)[1];
                bruteDiameter = Math.max(bruteDiameter, dx * dx + dy * dy);
            }
        }
        check(dynamic.diameterSquared() == bruteDiameter, "diameter matches exhaustive point scan");

        // --- insertion order must not matter ------------------------------------
        List<long[]> ascending = new ArrayList<>(inserted);
        ascending.sort((a, b) -> Long.compare(a[0], b[0]));
        DynamicHull sorted = new DynamicHull();
        for (long[] p : ascending) sorted.insert(p[0], p[1]);
        check(vertexKeys(sorted.hullCCW()).equals(vertexKeys(dynamic.hullCCW())),
                "sorted insertion yields the same hull");
        check(sorted.doubledArea() == dynamic.doubledArea(), "sorted insertion yields the same area");

        // --- deletion rebuilds and stays exact ----------------------------------
        int alive = dynamic.pointCount();
        long rebuildsBefore = dynamic.rebuildCount();
        boolean removed = dynamic.delete(cloud.get(0)[0], cloud.get(0)[1]);
        check(removed, "delete reported success");
        check(dynamic.rebuildCount() == rebuildsBefore + 1, "delete triggered exactly one rebuild");
        inserted.remove(0);
        alive--;
        check(dynamic.pointCount() == alive, "point count after delete");
        check(vertexKeys(dynamic.hullCCW()).equals(vertexKeys(bruteHull(inserted))),
                "hull after delete matches brute force");
        check(!dynamic.delete(cloud.get(0)[0], cloud.get(0)[1]), "second delete is a no-op");
        check(dynamic.delete(cloud.get(1)[0], cloud.get(1)[1]), "second point deletable");
        inserted.remove(0);
        check(vertexKeys(dynamic.hullCCW()).equals(vertexKeys(bruteHull(inserted))),
                "hull after two deletes matches brute force");

        // --- delete everything: back to a degenerate structure -------------------
        DynamicHull draining = new DynamicHull();
        for (long[] p : cloud) draining.insert(p[0], p[1]);
        for (long[] p : cloud) check(draining.delete(p[0], p[1]), "draining delete " + p[0]);
        check(draining.pointCount() == 0 && draining.hullCCW().isEmpty(), "drained hull is empty");
        check(draining.doubledArea() == 0 && !draining.contains(0, 0), "drained hull is inert");

        // --- larger stress: monotone chains stay correct over many inserts -------
        DynamicHull stress = new DynamicHull();
        List<long[]> stressPoints = randomPoints(300, 400);
        int maxHull = 0;
        for (int i = 0; i < stressPoints.size(); i++) {
            stress.insert(stressPoints.get(i)[0], stressPoints.get(i)[1]);
            if (i % 25 == 24 || i == stressPoints.size() - 1) {
                List<long[]> want = bruteHull(stressPoints.subList(0, i + 1));
                check(vertexKeys(stress.hullCCW()).equals(vertexKeys(want)),
                        "stress hull mismatch at " + i);
            }
            maxHull = Math.max(maxHull, stress.size());
        }
        check(stress.doubledArea() == shoelace(bruteHull(stressPoints)), "stress area");
        check(maxHull <= 2 * 300, "chain sizes never exceed twice the point count");
        check(stress.hullCCW().size() < stress.pointCount(), "random cloud leaves interior points");

        System.out.println("hull vertices (dynamic) = " + dynamic.hullCCW().size()
                + ", stress hull vertices = " + stress.hullCCW().size()
                + ", rebuilds = " + draining.rebuildCount() + dynamic.rebuildCount());
        System.out.println("PASS");
    }

    /** Shoelace on a CCW polygon: twice the area, matching DynamicHull.doubledArea(). */
    static long shoelace(List<long[]> poly) {
        if (poly.size() < 3) return 0;
        long sum = 0;
        for (int i = 0; i < poly.size(); i++) {
            long[] a = poly.get(i);
            long[] b = poly.get((i + 1) % poly.size());
            sum += a[0] * b[1] - b[0] * a[1];
        }
        return Math.abs(sum);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
