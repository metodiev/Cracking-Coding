/** Task 395 - Implement the dynamic convex hull trick for max queries (Li Chao tree). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class LiChao {
    private final int xLo, xHi;
    private final boolean[] has;
    private final long[] m, b;

    LiChao(int xLo, int xHi) {
        this.xLo = xLo;
        this.xHi = xHi;
        int size = 4 * (xHi - xLo + 1);
        has = new boolean[size];
        m = new long[size];
        b = new long[size];
    }

    void addLine(long slope, long intercept) {
        add(1, xLo, xHi, slope, intercept);
    }

    private void add(int node, int lo, int hi, long slope, long intercept) {
        if (!has[node]) {
            has[node] = true;
            m[node] = slope;
            b[node] = intercept;
            return;
        }
        int mid = (lo + hi) >> 1;   // floor division; / truncates toward zero and breaks negative intervals
        if (slope * mid + intercept > m[node] * mid + b[node]) {
            long tmpM = m[node]; m[node] = slope; slope = tmpM;
            long tmpB = b[node]; b[node] = intercept; intercept = tmpB;
        }
        if (lo == hi) return;
        if (slope * lo + intercept > m[node] * lo + b[node]) {
            add(2 * node, lo, mid, slope, intercept);
        } else if (slope * hi + intercept > m[node] * hi + b[node]) {
            add(2 * node + 1, mid + 1, hi, slope, intercept);
        }
    }

    long query(int x) {
        int node = 1, lo = xLo, hi = xHi;
        Long best = null;
        while (true) {
            if (has[node]) {
                long value = m[node] * x + b[node];
                if (best == null || value > best) best = value;
            }
            if (lo == hi) break;
            int mid = (lo + hi) >> 1;   // floor division; / truncates toward zero and breaks negative intervals
            if (x <= mid) {
                node = 2 * node;
                hi = mid;
            } else {
                node = 2 * node + 1;
                lo = mid + 1;
            }
        }
        if (best == null) throw new IllegalStateException("query on an empty tree");
        return best;
    }
}

public class Solution {
    static long bruteForce(long[][] lines, int x) {
        long best = Long.MIN_VALUE;
        for (long[] line : lines) best = Math.max(best, line[0] * x + line[1]);
        return best;
    }

    private static int seed = 4242;

    private static long[][] lcgLines(int count, int mRange, int bRange) {
        long[][] lines = new long[count][];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            long slope = seed % (2 * mRange + 1) - mRange;
            seed = (seed * 75 + 74) % 65537;
            long intercept = seed % (2 * bRange + 1) - bRange;
            lines[k] = new long[]{slope, intercept};
        }
        return lines;
    }

    public static void main(String[] args) {
        LiChao tree = new LiChao(-500, 500);
        for (long[] line : new long[][]{{1, 0}, {-1, 0}, {0, 0}}) tree.addLine(line[0], line[1]);
        check(tree.query(5) == 5, "v shape right");
        check(tree.query(-5) == 5, "v shape left");
        check(tree.query(0) == 0, "v shape middle");
        check(tree.query(3) == 3, "v shape 3");

        LiChao cross = new LiChao(-500, 500);
        cross.addLine(2, 1);
        cross.addLine(-3, 10);
        check(cross.query(2) == 5, "crossing at 2");
        check(cross.query(1) == 7, "crossing at 1");
        check(cross.query(0) == 10, "crossing at 0");
        check(cross.query(5) == 11, "crossing at 5");

        long[][] fixed = {{1, 0}, {-1, 3}, {0, -2}, {5, -10}, {-4, 6}};
        LiChao checker = new LiChao(-20, 20);
        for (long[] line : fixed) checker.addLine(line[0], line[1]);
        for (int x = -20; x <= 20; x++) {
            check(checker.query(x) == bruteForce(fixed, x), "fixed line oracle " + x);
        }

        long[][] lines = lcgLines(200, 50, 1000);
        LiChao big = new LiChao(-500, 500);
        for (long[] line : lines) big.addLine(line[0], line[1]);
        for (int step = 0; step <= 100; step++) {
            int x = -500 + step * 10;
            check(big.query(x) == bruteForce(lines, x), "random oracle " + x);
        }

        LiChao duplicate = new LiChao(-10, 10);
        for (int i = 0; i < 4; i++) duplicate.addLine(3, -1);
        check(duplicate.query(7) == 20, "duplicate lines");

        boolean threw = false;
        try { new LiChao(-10, 10).query(0); }
        catch (IllegalStateException e) { threw = true; }
        check(threw, "empty tree rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
