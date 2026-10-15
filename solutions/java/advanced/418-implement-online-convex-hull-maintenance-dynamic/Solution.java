/** Task 418 - Implement online convex hull maintenance (dynamic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Online convex hull trick supporting BOTH signs and BOTH insertion orders:
 *   * mode 'max' maintains the upper envelope, mode 'min' the lower envelope,
 *   * order 'inc' inserts lines with non-decreasing slopes, order 'dec' with non-increasing
 *     ones — both are the orders produced by sorting candidate lines by slope.
 * The hull is kept as arrays (ms, bs) of lines whose breakpoints increase. When appending
 * (m, b) after (m1, b1), (m2, b2) the middle line m2 is unnecessary exactly when
 *     sign * [(b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1)] >= 0,
 * where sign = +1 for (max, inc) / (min, dec) and -1 for (max, dec) / (min, inc); the
 * condition compares the two intersection abscissas without ever dividing. Duplicate
 * slopes keep only the best intercept (larger for max, smaller for min). Queries are
 * unimodal over the hull, so an O(log n) adjacent-pair binary search finds the peak
 * (for max) or the valley (for min). Deletions are out of scope: this is add-only.
 * Everything is exact integer arithmetic — no floating point intersections.
 */
class ConvexHullTrick {
    private final boolean forMax;
    private final long sign;
    private final long[] ms = new long[1024];
    private final long[] bs = new long[1024];
    private int count = 0;

    ConvexHullTrick(boolean forMax, boolean increasing) {
        this.forMax = forMax;
        this.sign = (forMax ? 1 : -1) * (increasing ? 1 : -1);
    }

    int size() {
        return count;
    }

    long mAt(int i) {
        return ms[i];
    }

    long bAt(int i) {
        return bs[i];
    }

    void addLine(long m, long b) {
        if (count > 0 && m == ms[count - 1]) {
            boolean better = forMax ? b > bs[count - 1] : b < bs[count - 1];
            if (!better) return;
            count--;
        }
        while (count >= 2) {
            long m1 = ms[count - 2], b1 = bs[count - 2];
            long m2 = ms[count - 1], b2 = bs[count - 1];
            long expr = (b1 - b2) * (m - m2) - (b2 - b) * (m2 - m1);
            if (sign * expr >= 0) count--;
            else break;
        }
        ms[count] = m;
        bs[count] = b;
        count++;
    }

    long query(long x) {
        if (count == 0) throw new IllegalStateException("query on an empty hull");
        int lo = 0, hi = count - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            long vMid = ms[mid] * x + bs[mid];
            long vNext = ms[mid + 1] * x + bs[mid + 1];
            if (forMax) {
                if (vMid <= vNext) lo = mid + 1;
                else hi = mid;
            } else {
                if (vMid >= vNext) lo = mid + 1;
                else hi = mid;
            }
        }
        return ms[lo] * x + bs[lo];
    }
}

public class Solution {
    private static int seed;

    private static int nextRand() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static long bruteBest(long[][] lines, long x, boolean forMax) {
        long best = forMax ? Long.MIN_VALUE : Long.MAX_VALUE;
        for (long[] line : lines) {
            long value = line[0] * x + line[1];
            best = forMax ? Math.max(best, value) : Math.min(best, value);
        }
        return best;
    }

    static long[][] lcgLines(int count) {
        long[][] lines = new long[count][];
        for (int k = 0; k < count; k++) {
            long m = nextRand() % 141 - 70;
            long b = nextRand() % 4001 - 2000;
            lines[k] = new long[]{m, b};
        }
        return lines;
    }

    static void sortBySlope(long[][] lines, boolean increasing) {
        java.util.Arrays.sort(lines, (p, q) -> increasing
                ? Long.compare(p[0], q[0])
                : Long.compare(q[0], p[0]));
    }

    public static void main(String[] args) {
        // hand case: lines y = x, y = 2x + 3, y = 4x + 5 (max): 5 at 0, 45 at 10, -10 at -10
        ConvexHullTrick hull = new ConvexHullTrick(true, true);
        for (long[] line : new long[][]{{1, 0}, {2, 3}, {4, 5}}) hull.addLine(line[0], line[1]);
        check(hull.query(0) == 5, "hand 0");
        check(hull.query(10) == 45, "hand 10");
        check(hull.query(-10) == -10, "hand -10");
        // the same lines inserted from the other end of the slope order agree everywhere
        ConvexHullTrick dec = new ConvexHullTrick(true, false);
        for (long[] line : new long[][]{{4, 5}, {2, 3}, {1, 0}}) dec.addLine(line[0], line[1]);
        for (long x = -10; x <= 10; x++) check(dec.query(x) == hull.query(x), "order agreement " + x);

        // hand case for min: y = 2x + 5, y = x + 1: 1 at 0, 11 at 10, -15 at -10
        ConvexHullTrick mn = new ConvexHullTrick(false, true);
        mn.addLine(2, 5);
        mn.addLine(1, 1);
        check(mn.query(0) == 1 && mn.query(10) == 11 && mn.query(-10) == -15, "min hand");
        ConvexHullTrick mnDec = new ConvexHullTrick(false, false);
        mnDec.addLine(1, 1);
        mnDec.addLine(2, 5);
        for (long x = -10; x <= 10; x++) check(mnDec.query(x) == mn.query(x), "min order agreement " + x);

        // duplicate slopes keep only the best intercept: max keeps b = -1 (value 20 at x = 7)
        ConvexHullTrick dup = new ConvexHullTrick(true, true);
        for (int i = 0; i < 4; i++) dup.addLine(3, -1);
        check(dup.query(7) == 20 && dup.size() == 1, "dup max");
        ConvexHullTrick dup2 = new ConvexHullTrick(false, true);
        dup2.addLine(3, 10);
        dup2.addLine(3, 5);
        check(dup2.query(7) == 26 && dup2.size() == 1, "dup min");
        ConvexHullTrick dup3 = new ConvexHullTrick(true, true);
        dup3.addLine(3, -1);
        dup3.addLine(3, -2);
        check(dup3.size() == 1 && dup3.query(7) == 20, "dup worse discarded");

        // fixed-line oracle in all four mode/order combinations
        long[][] fixedSrc = {{1, 0}, {-1, 3}, {0, -2}, {5, -10}, {-4, 6}};
        for (boolean forMax : new boolean[]{true, false}) {
            for (boolean increasing : new boolean[]{true, false}) {
                long[][] fixed = fixedSrc.clone();
                sortBySlope(fixed, increasing);
                ConvexHullTrick probe = new ConvexHullTrick(forMax, increasing);
                for (long[] line : fixed) probe.addLine(line[0], line[1]);
                for (long x = -20; x <= 20; x++) {
                    check(probe.query(x) == bruteBest(fixed, x, forMax),
                            "fixed " + forMax + " " + increasing + " " + x);
                }
            }
        }

        // interleaved add/query scripts from the deterministic LCG, all four combinations
        int index = 0;
        for (boolean forMax : new boolean[]{true, false}) {
            for (boolean increasing : new boolean[]{true, false}) {
                seed = 4180 + index++;
                long[][] raw = lcgLines(60);
                sortBySlope(raw, increasing);
                ConvexHullTrick h = new ConvexHullTrick(forMax, increasing);
                java.util.List<long[]> added = new java.util.ArrayList<>();
                for (int k = 0; k < raw.length; k++) {
                    h.addLine(raw[k][0], raw[k][1]);
                    added.add(raw[k]);
                    check(h.size() <= added.size(), "hull size bound");
                    for (int t = 0; t < h.size(); t++) {
                        boolean present = false;
                        for (long[] line : added) {
                            if (line[0] == h.mAt(t) && line[1] == h.bAt(t)) present = true;
                        }
                        check(present, "hull line present");
                    }
                    if (k % 3 == 2 || k == raw.length - 1) {
                        for (int q = 0; q < 3; q++) {
                            long x = nextRand() % 2001 - 1000;
                            check(h.query(x) == bruteBest(added.toArray(new long[0][]), x, forMax),
                                    "script query " + index + " " + k + " " + x);
                        }
                    }
                }
                for (long x = -200; x <= 200; x++) {
                    check(h.query(x) == bruteBest(raw, x, forMax), "sweep " + index + " " + x);
                }
                java.util.Set<Long> distinct = new java.util.HashSet<>();
                for (long[] line : raw) distinct.add(line[0]);
                check(h.size() <= distinct.size(), "distinct slope bound");
            }
        }

        // querying before the first insertion is rejected
        boolean threw = false;
        try {
            new ConvexHullTrick(true, true).query(3);
        } catch (IllegalStateException e) {
            threw = true;
        }
        check(threw, "empty hull rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
