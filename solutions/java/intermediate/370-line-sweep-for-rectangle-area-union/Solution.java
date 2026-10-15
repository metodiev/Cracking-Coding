/** Task 370 - Line sweep for rectangle area union. (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

class CoverLengthTree {
    // Segment tree over compressed y edges.  count == how many active rectangles fully
    // cover the node, length == covered length of the node's span.  Counts (not booleans)
    // are what make overlaps of three or more rectangles and exact touching edges work.
    private final int[] ys;
    private final int m;
    private final int[] count;
    private final int[] length;

    CoverLengthTree(int[] sortedYs) {
        ys = sortedYs;
        m = ys.length - 1;
        count = new int[4 * Math.max(m, 1)];
        length = new int[4 * Math.max(m, 1)];
    }

    void rangeAdd(int yLo, int yHi, int delta) {
        if (m == 0 || yLo >= yHi) return;
        update(1, 0, m, yLo, yHi, delta);
    }

    private void update(int node, int lo, int hi, int ql, int qr, int delta) {
        if (qr <= lo || hi <= ql) return;
        if (ql <= lo && hi <= qr) {
            count[node] += delta;
        } else {
            int mid = (lo + hi) / 2;
            update(2 * node, lo, mid, ql, qr, delta);
            update(2 * node + 1, mid, hi, ql, qr, delta);
        }
        if (count[node] > 0) {
            length[node] = ys[hi] - ys[lo];
        } else if (hi - lo == 1) {
            length[node] = 0;
        } else {
            length[node] = length[2 * node] + length[2 * node + 1];
        }
    }

    int measure() {
        return m == 0 ? 0 : length[1];
    }
}

public class Solution {
    private static int seed = 1618;

    /** Area of the union of axis-aligned rectangles [x1, y1, x2, y2]. */
    static long unionArea(int[][] rectangles) {
        List<int[]> events = new ArrayList<>();
        TreeSet<Integer> ySet = new TreeSet<>();
        for (int[] r : rectangles) {
            if (r[0] >= r[2] || r[1] >= r[3]) continue;       // zero/negative area: ignored
            events.add(new int[]{r[0], 1, r[1], r[3]});
            events.add(new int[]{r[2], -1, r[1], r[3]});
            ySet.add(r[1]);
            ySet.add(r[3]);
        }
        if (events.isEmpty()) return 0;
        int[] ys = new int[ySet.size()];
        int at = 0;
        Map<Integer, Integer> index = new HashMap<>();
        for (int y : ySet) {
            index.put(y, at);
            ys[at++] = y;
        }
        events.sort(Comparator.<int[]>comparingInt(e -> e[0]).thenComparingInt(e -> e[1]));
        CoverLengthTree tree = new CoverLengthTree(ys);
        long area = 0;
        int prevX = events.get(0)[0];
        int i = 0;
        while (i < events.size()) {
            int x = events.get(i)[0];
            area += (long) tree.measure() * (x - prevX);
            while (i < events.size() && events.get(i)[0] == x) {
                int[] event = events.get(i);
                tree.rangeAdd(index.get(event[2]), index.get(event[3]), event[1]);
                i++;
            }
            prevX = x;
        }
        return area;
    }

    /** Independent oracle: sum the fully covered cells of the compressed coordinate grid. */
    static long unionAreaOracle(int[][] rectangles) {
        List<int[]> rects = new ArrayList<>();
        TreeSet<Integer> xsSet = new TreeSet<>();
        TreeSet<Integer> ysSet = new TreeSet<>();
        for (int[] r : rectangles) {
            if (r[0] < r[2] && r[1] < r[3]) {
                rects.add(r);
                xsSet.add(r[0]);
                xsSet.add(r[2]);
                ysSet.add(r[1]);
                ysSet.add(r[3]);
            }
        }
        if (rects.isEmpty()) return 0;
        int[] xs = new int[xsSet.size()];
        int[] ys = new int[ysSet.size()];
        int at = 0;
        for (int x : xsSet) xs[at++] = x;
        at = 0;
        for (int y : ysSet) ys[at++] = y;
        long area = 0;
        for (int i = 0; i + 1 < xs.length; i++) {
            for (int j = 0; j + 1 < ys.length; j++) {
                for (int[] r : rects) {
                    if (r[0] <= xs[i] && r[2] >= xs[i + 1] &&
                            r[1] <= ys[j] && r[3] >= ys[j + 1]) {
                        area += (long) (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]);
                        break;
                    }
                }
            }
        }
        return area;
    }

    private static boolean sameArea(long a, long b, String label) {
        check(a == b, label + " got " + a + " want " + b);
        return true;
    }

    public static void main(String[] args) {
        // Canonical example: [0,0,2,2] and [1,1,3,3] overlap in a 1x1 square -> 7.
        int[][] canonical = {{0, 0, 2, 2}, {1, 1, 3, 3}};
        sameArea(unionArea(canonical), 7, "canonical");
        sameArea(unionAreaOracle(canonical), 7, "canonical oracle");

        sameArea(unionArea(new int[][]{{0, 0, 1, 1}, {2, 2, 3, 3}}), 2, "disjoint");
        sameArea(unionArea(new int[][]{{0, 0, 10, 10}, {2, 2, 4, 4}}), 100, "nested");
        sameArea(unionArea(new int[][]{{1, 1, 3, 3}, {1, 1, 3, 3}}), 4, "identical");
        sameArea(unionArea(new int[][]{{0, 0, 2, 2}, {2, 0, 4, 2}}), 8, "touching edge");
        sameArea(unionArea(new int[][]{{0, 0, 1, 1}, {1, 1, 2, 2}}), 2, "touching corner");
        sameArea(unionArea(new int[][]{{5, 5, 5, 9}}), 0, "zero width");
        sameArea(unionArea(new int[][]{{5, 5, 5, 5}}), 0, "point rectangle");
        sameArea(unionArea(new int[][]{{0, 0, 3, 3}, {1, 1, 1, 4}}), 9, "zero-area among valid");
        sameArea(unionArea(new int[0][]), 0, "empty");
        sameArea(unionArea(new int[][]{{2, 3, 7, 9}}), 30, "single rectangle");
        sameArea(unionArea(new int[][]{{0, 0, 4, 4}, {0, 0, 4, 4}, {1, 1, 3, 3}}), 16, "repeats");

        // Triple overlap: y in [0,1) spans x 0..4 (length 4), y in [1,3) also 4, y in [3,4) 3.
        int[][] triple = {{0, 0, 3, 3}, {1, 0, 4, 3}, {0, 1, 3, 4}};
        sameArea(unionArea(triple), 15, "triple union");
        sameArea(unionAreaOracle(triple), 15, "triple union oracle");

        // LCG cross-check, including degenerate rectangles and negative coordinates.
        for (int trial = 0; trial < 40; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 8;
            int modulus = 8 + seed % 10;
            int[][] rectangles = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int x1 = seed % modulus;
                seed = (seed * 75 + 74) % 65537;
                int width = seed % (modulus / 2);
                seed = (seed * 75 + 74) % 65537;
                int y1 = seed % modulus;
                seed = (seed * 75 + 74) % 65537;
                int height = seed % (modulus / 2);
                rectangles[i] = new int[]{x1, y1, x1 + width, y1 + height};
            }
            seed = (seed * 75 + 74) % 65537;
            int offset = -(seed % 7);
            int[][] shifted = new int[count][];
            for (int i = 0; i < count; i++) {
                int[] r = rectangles[i];
                shifted[i] = new int[]{r[0] + offset, r[1] + offset, r[2] + offset, r[3] + offset};
            }
            sameArea(unionArea(shifted), unionAreaOracle(shifted), "random trial " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
