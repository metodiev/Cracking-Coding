/** Task 371 - Line-segment intersection (Bentley-Ottmann sweep outline). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

/**
 * Detect all intersecting pairs among n segments.  The full Bentley-Ottmann algorithm
 * runs in O((n + k) log n) for k intersection points; this task implements the sweep
 * *skeleton* (event queue + status structure) and handles the small cases this repo
 * tests (n <= ~12) exactly, plus a plain O(n^2) fallback used as the oracle.
 *
 * Bentley-Ottmann invariant (the full algorithm this skeleton approximates):
 *   1. Events: every segment endpoint, ordered lexicographically by (x, y).  A left
 *      endpoint inserts the segment into the status structure; a right endpoint
 *      removes it.  Intersection points found during the sweep are also pushed as
 *      events so that the order of the status structure is restored after a crossing.
 *   2. Status structure: the segments currently crossing the vertical sweep line,
 *      kept ordered by their y coordinate at the current sweep x.  In general
 *      position no two of them cross between events, so the order is well defined
 *      and stable there.
 *   3. Crossing detection: only neighbours in the status can become adjacent-and-
 *      swapped between events, so a crossing is reported when two segments are
 *      adjacent in the status (on insertion/removal, and after processing a found
 *      intersection event) and their exact intersection test succeeds.  Each found
 *      intersection is reported once, at the first event x where the two segments
 *      become adjacent.
 *   4. Degeneracies (documented but only partially handled here): vertical segments
 *      (processed entirely at one event x), several events with equal x, endpoint
 *      touches, collinear overlaps and overlapping collinear bundles.
 *
 * Small-case exact handling used here: at every event x, after all left endpoints of
 * that x have been inserted and finished segments dropped, every *pair* of segments
 * in the status structure is tested with the exact integer predicate.  An
 * intersecting pair always spans the event x = max(left_x1, left_x2) (the later
 * segment's left endpoint), so every intersecting pair is seen at least once,
 * duplicate reports collapse into a set, and the count matches the O(n^2) oracle.
 *
 * All predicates use exact integer arithmetic (cross products); no floating point.
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
    /** Twice the signed area of (p, q, r): >0 CCW, <0 CW, 0 collinear. */
    static long orient(int px, int py, int qx, int qy, int rx, int ry) {
        return (long) (qx - px) * (ry - py) - (long) (qy - py) * (rx - px);
    }

    /** True when collinear q lies inside segment pr (endpoints included). */
    static boolean onSegment(int px, int py, int qx, int qy, int rx, int ry) {
        return Math.min(px, rx) <= qx && qx <= Math.max(px, rx)
                && Math.min(py, ry) <= qy && qy <= Math.max(py, ry);
    }

    /** Exact closed-segment intersection test (touches and overlaps count). */
    static boolean segmentsIntersect(int[] s, int[] t) {
        long o1 = orient(s[0], s[1], s[2], s[3], t[0], t[1]);
        long o2 = orient(s[0], s[1], s[2], s[3], t[2], t[3]);
        long o3 = orient(t[0], t[1], t[2], t[3], s[0], s[1]);
        long o4 = orient(t[0], t[1], t[2], t[3], s[2], s[3]);
        if (o1 * o2 < 0 && o3 * o4 < 0) return true;                    // proper crossing
        if (o1 == 0 && onSegment(s[0], s[1], t[0], t[1], s[2], s[3])) return true;
        if (o2 == 0 && onSegment(s[0], s[1], t[2], t[3], s[2], s[3])) return true;
        if (o3 == 0 && onSegment(t[0], t[1], s[0], s[1], t[2], t[3])) return true;
        if (o4 == 0 && onSegment(t[0], t[1], s[2], s[3], t[2], t[3])) return true;
        return false;
    }

    /** Normalise {x1,y1,x2,y2} so that (x1,y1) <= (x2,y2) lexicographically. */
    static int[][] normalized(int[][] segments) {
        int[][] result = new int[segments.length][];
        for (int i = 0; i < segments.length; i++) {
            int[] s = segments[i].clone();
            if (s[0] > s[2] || (s[0] == s[2] && s[1] > s[3])) {
                int tmp = s[0]; s[0] = s[2]; s[2] = tmp;
                tmp = s[1]; s[1] = s[3]; s[3] = tmp;
            }
            result[i] = s;
        }
        return result;
    }

    /** -1/0/1 comparing the segments' exact y values at the sweep line x. */
    static int compareY(int[] a, int[] b, int x) {
        long na = (long) a[1] * (a[2] - a[0]) + (long) (a[3] - a[1]) * (x - a[0]);
        long da = a[2] - a[0];
        long nb = (long) b[1] * (b[2] - b[0]) + (long) (b[3] - b[1]) * (x - b[0]);
        long db = b[2] - b[0];
        if (a[2] == a[0]) { na = a[1]; da = 1; }
        if (b[2] == b[0]) { nb = b[1]; db = 1; }
        long left = na * db, right = nb * da;
        return left < right ? -1 : (left > right ? 1 : 0);
    }

    /** Sweep skeleton: event queue + status structure; returns the intersecting pairs. */
    static TreeSet<String> sweepIntersectionPairs(int[][] rawSegments) {
        int[][] segments = normalized(rawSegments);
        List<int[]> events = new ArrayList<>();
        for (int idx = 0; idx < segments.length; idx++) {
            events.add(new int[]{segments[idx][0], segments[idx][1], idx, 0});   // left
            events.add(new int[]{segments[idx][2], segments[idx][3], idx, 1});   // right
        }
        events.sort((u, v) -> {
            if (u[0] != v[0]) return Integer.compare(u[0], v[0]);
            if (u[1] != v[1]) return Integer.compare(u[1], v[1]);
            if (u[2] != v[2]) return Integer.compare(u[2], v[2]);
            return Integer.compare(u[3], v[3]);
        });

        ArrayList<Integer> status = new ArrayList<>();          // active ids, y-ordered
        TreeSet<String> found = new TreeSet<>();
        int i = 0;
        while (i < events.size()) {
            int x = events.get(i)[0];
            int j = i;
            while (j < events.size() && events.get(j)[0] == x) {
                int[] event = events.get(j);
                if (event[3] == 0) {                            // insert, keeping y-order
                    int pos = 0;
                    while (pos < status.size()
                            && compareY(segments[status.get(pos)], segments[event[2]], x) <= 0) {
                        pos++;
                    }
                    status.add(pos, event[2]);
                }
                j++;
            }
            status.removeIf(k -> segments[k][2] < x);           // finish finished segments
            for (int a = 0; a < status.size(); a++) {
                for (int b = a + 1; b < status.size(); b++) {
                    if (segmentsIntersect(segments[status.get(a)], segments[status.get(b)])) {
                        int low = Math.min(status.get(a), status.get(b));
                        int high = Math.max(status.get(a), status.get(b));
                        found.add(low + ":" + high);
                    }
                }
            }
            i = j;
        }
        return found;
    }

    /** O(n^2) oracle: test every pair with the same exact predicate. */
    static TreeSet<String> bruteForceIntersectionPairs(int[][] segments) {
        TreeSet<String> found = new TreeSet<>();
        for (int i = 0; i < segments.length; i++) {
            for (int j = i + 1; j < segments.length; j++) {
                if (segmentsIntersect(segments[i], segments[j])) found.add(i + ":" + j);
            }
        }
        return found;
    }

    static int[][] lcgSegments(int count, Lcg lcg) {
        int[][] segments = new int[count][];
        for (int i = 0; i < count; i++) {
            int x1, y1, x2, y2;
            do {
                x1 = lcg.next(41); y1 = lcg.next(41);
                x2 = lcg.next(41); y2 = lcg.next(41);
            } while (x1 == x2 && y1 == y2);
            segments[i] = new int[]{x1, y1, x2, y2};
        }
        return segments;
    }

    public static void main(String[] args) {
        // --- hand predicates ------------------------------------------------
        check(segmentsIntersect(new int[]{0, 0, 4, 4}, new int[]{0, 4, 4, 0}), "proper crossing");
        check(segmentsIntersect(new int[]{0, 0, 2, 2}, new int[]{2, 2, 4, 0}), "shared endpoint");
        check(segmentsIntersect(new int[]{0, 0, 4, 0}, new int[]{2, 0, 6, 0}), "collinear overlap");
        check(!segmentsIntersect(new int[]{0, 0, 2, 0}, new int[]{0, 1, 2, 1}), "parallel disjoint");
        check(segmentsIntersect(new int[]{0, 0, 4, 0}, new int[]{2, 0, 2, 3}), "T-touch");
        check(!segmentsIntersect(new int[]{0, 0, 1, 1}, new int[]{3, 0, 4, 1}), "separated");
        check(segmentsIntersect(new int[]{0, 0, 1, 1}, new int[]{1, 1, 2, 0}), "endpoint meet");

        // --- hand sweep set: 4 intersecting pairs ---------------------------
        int[][] hand = {{0, 0, 4, 4}, {0, 4, 4, 0}, {2, 2, 6, 2}, {6, 0, 6, 6}, {8, 8, 9, 9}};
        TreeSet<String> expected = new TreeSet<>(Arrays.asList("0:1", "0:2", "1:2", "2:3"));
        check(sweepIntersectionPairs(hand).equals(expected), "hand sweep set");
        check(bruteForceIntersectionPairs(hand).equals(expected), "hand brute force");
        check(expected.size() == 4, "hand count is 4");

        // --- collinear / touch family ----------------------------------------
        int[][] collinear = {{0, 0, 4, 0}, {2, 0, 6, 0}, {0, 1, 6, 1}, {0, 0, 0, 1}};
        TreeSet<String> collinearExpected = new TreeSet<>(Arrays.asList("0:1", "0:3", "2:3"));
        check(sweepIntersectionPairs(collinear).equals(collinearExpected), "collinear family");
        check(sweepIntersectionPairs(collinear).equals(bruteForceIntersectionPairs(collinear)),
                "collinear family vs oracle");

        int[][] disjoint = {{0, 0, 1, 1}, {3, 0, 4, 1}, {5, 5, 6, 6}};
        check(sweepIntersectionPairs(disjoint).isEmpty(), "disjoint set");
        check(sweepIntersectionPairs(new int[0][]).isEmpty(), "empty input");

        // --- LCG cross-checks against the O(n^2) oracle -----------------------
        Lcg lcg = new Lcg(37113);
        for (int trial = 0; trial < 40; trial++) {
            int count = 5 + lcg.next(8);
            int[][] segments = lcgSegments(count, lcg);
            check(sweepIntersectionPairs(segments).equals(bruteForceIntersectionPairs(segments)),
                    "random sweep vs brute force " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
