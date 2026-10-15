/** Task 361 - Implement interval stabbing queries with segment tree. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class StabbingSegmentTree {
    // All query points are known up front, so they are coordinate-compressed to indices
    // and the segment tree stores cover-count adds; a point query walks leaf -> root.
    private final int[] points;
    private final int n;
    private final int[] tree;

    StabbingSegmentTree(int[] rawPoints) {
        int[] sorted = rawPoints.clone();
        Arrays.sort(sorted);
        int unique = 0;
        for (int i = 0; i < sorted.length; i++) {
            if (i == 0 || sorted[i] != sorted[i - 1]) sorted[unique++] = sorted[i];
        }
        this.points = Arrays.copyOf(sorted, unique);
        this.n = unique;
        this.tree = new int[2 * Math.max(n, 1)];
    }

    void addInterval(int left, int right) {
        if (n == 0) return;
        int lo = lowerBound(points, n, left);
        int hi = upperBound(points, n, right);
        int l = lo + n;
        int r = hi + n;
        while (l < r) {
            if ((l & 1) == 1) tree[l++]++;
            if ((r & 1) == 1) tree[--r]++;
            l >>= 1;
            r >>= 1;
        }
    }

    int coverCount(int point) {
        int index = lowerBound(points, n, point);
        if (index >= n || points[index] != point) {
            throw new IllegalArgumentException("point was not registered");
        }
        int total = 0;
        for (int i = index + n; i >= 1; i >>= 1) total += tree[i];
        return total;
    }

    private static int lowerBound(int[] values, int length, int target) {
        int lo = 0, hi = length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (values[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int upperBound(int[] values, int length, int target) {
        int lo = 0, hi = length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (values[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

public class Solution {
    private static int seed = 20261;

    private static int naiveCoverCount(int[][] intervals, int point) {
        int count = 0;
        for (int[] interval : intervals) {
            if (interval[0] <= point && point <= interval[1]) count++;
        }
        return count;
    }

    public static void main(String[] args) {
        // Hand-built example from the task description: intervals [1,4], [3,6], [5,8].
        int[][] intervals = {{1, 4}, {3, 6}, {5, 8}};
        int[] points = {0, 2, 4, 6, 7, 9};
        int[] expected = {0, 1, 2, 2, 1, 0};
        StabbingSegmentTree tree = new StabbingSegmentTree(points);
        for (int[] interval : intervals) tree.addInterval(interval[0], interval[1]);
        for (int k = 0; k < points.length; k++) {
            check(tree.coverCount(points[k]) == expected[k],
                    "hand example point " + points[k] + " -> " + expected[k]);
            check(naiveCoverCount(intervals, points[k]) == expected[k],
                    "naive agrees at " + points[k]);
        }

        // Adding the same interval twice doubles the counts.
        StabbingSegmentTree doubled = new StabbingSegmentTree(points);
        for (int[] interval : intervals) {
            doubled.addInterval(interval[0], interval[1]);
            doubled.addInterval(interval[0], interval[1]);
        }
        for (int k = 0; k < points.length; k++) {
            check(doubled.coverCount(points[k]) == 2 * expected[k], "doubled at " + points[k]);
        }

        // Duplicate query points collapse; single point works; no points is a no-op.
        StabbingSegmentTree dup = new StabbingSegmentTree(new int[]{3, 3, 5, 3});
        dup.addInterval(1, 4);
        check(dup.coverCount(3) == 1 && dup.coverCount(5) == 0, "duplicate query points");
        StabbingSegmentTree solo = new StabbingSegmentTree(new int[]{7});
        solo.addInterval(7, 7);
        solo.addInterval(8, 9);
        check(solo.coverCount(7) == 1, "single registered point");
        new StabbingSegmentTree(new int[0]).addInterval(-3, 12);   // must not fail

        // LCG cross-check against the naive cover count.
        int probeSize = 48;
        int[] probe = new int[probeSize];
        for (int i = 0; i < probeSize; i++) probe[i] = i - 2;
        for (int trial = 0; trial < 25; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 40;
            int[][] randomIntervals = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int left = seed % 30;
                seed = (seed * 75 + 74) % 65537;
                int width = 1 + seed % 10;
                randomIntervals[i] = new int[]{left, left + width};
            }
            StabbingSegmentTree randomTree = new StabbingSegmentTree(probe);
            for (int[] interval : randomIntervals) {
                randomTree.addInterval(interval[0], interval[1]);
            }
            for (int p : probe) {
                check(randomTree.coverCount(p) == naiveCoverCount(randomIntervals, p),
                        "random trial " + trial + " point " + p);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
