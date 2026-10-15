/** Task 335 - Implement data structure for skyline queries (segment tree variant). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class SkylineTree {
    // Each node keeps `tag` (a height pending for its whole range) and `best` (tallest point
    // inside it). Updates only raise heights, so applying max is monotone and tag only ever
    // grows, which makes push-down unnecessary for point/range-max queries.
    private final int n;
    private final int size;
    private final int[] tag;
    private final int[] best;

    SkylineTree(int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1");
        int power = 1;
        while (power < n) power <<= 1;
        this.n = n;
        this.size = power;
        this.tag = new int[2 * power];
        this.best = new int[2 * power];
    }

    /** Raises the skyline to at least `height` on the half-open range [left, right). */
    void update(int left, int right, int height) {
        if (left < 0 || left >= right || right > n) throw new IllegalArgumentException("invalid range");
        update(1, 0, size, left, right, height);
    }

    /** Skyline height at `index`. */
    int point(int index) {
        if (index < 0 || index >= n) throw new IllegalArgumentException("index out of range");
        int node = 1, lo = 0, hi = size, height = 0;
        while (true) {
            height = Math.max(height, tag[node]);
            if (hi - lo == 1) return height;
            int mid = (lo + hi) / 2;
            if (index < mid) {
                node = 2 * node;
                hi = mid;
            } else {
                node = 2 * node + 1;
                lo = mid;
            }
        }
    }

    /** Tallest point in [left, right); 0 when the range is empty. */
    int rangeMax(int left, int right) {
        if (left < 0 || left > right || right > n) throw new IllegalArgumentException("invalid range");
        if (left == right) return 0;
        return rangeMax(1, 0, size, left, right);
    }

    int[] skyline() {
        int[] heights = new int[n];
        for (int i = 0; i < n; i++) heights[i] = point(i);
        return heights;
    }

    private void update(int node, int lo, int hi, int left, int right, int height) {
        if (right <= lo || hi <= left) return;
        if (left <= lo && hi <= right) {
            tag[node] = Math.max(tag[node], height);
            best[node] = Math.max(best[node], height);
            return;
        }
        int mid = (lo + hi) / 2;
        update(2 * node, lo, mid, left, right, height);
        update(2 * node + 1, mid, hi, left, right, height);
        best[node] = Math.max(tag[node], Math.max(best[2 * node], best[2 * node + 1]));
    }

    private int rangeMax(int node, int lo, int hi, int left, int right) {
        if (left >= right || right <= lo || hi <= left) return 0;
        if (left <= lo && hi <= right) return best[node];
        int mid = (lo + hi) / 2;
        return Math.max(tag[node], Math.max(rangeMax(2 * node, lo, mid, left, right),
                rangeMax(2 * node + 1, mid, hi, left, right)));
    }
}

public class Solution {
    /** Oracle: keep a height per position and apply each update directly. */
    static int[] bruteSchedule(int n, List<int[]> updates) {
        int[] heights = new int[n];
        for (int[] update : updates) {
            for (int i = update[0]; i < update[1]; i++) heights[i] = Math.max(heights[i], update[2]);
        }
        return heights;
    }

    private static int seed = 7788;

    private static List<int[]> lcgUpdates(int count, int n) {
        List<int[]> updates = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            int left = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int right = left + 1 + seed % (n - left);
            seed = (seed * 75 + 74) % 65537;
            updates.add(new int[]{left, right, seed % 100});
        }
        return updates;
    }

    private static String join(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        // Hand-checked: [1,4)->5, [2,3)->7, [4,6)->2 over 7 positions.
        SkylineTree tree = new SkylineTree(7);
        int[][] initial = {{1, 4, 5}, {2, 3, 7}, {4, 6, 2}};
        for (int[] update : initial) tree.update(update[0], update[1], update[2]);
        check(join(tree.skyline()).equals("0,5,7,5,2,2,0"), "hand skyline");
        check(tree.point(2) == 7 && tree.point(6) == 0 && tree.point(0) == 0, "hand points");
        check(tree.rangeMax(0, 3) == 7, "range 0..3");
        check(tree.rangeMax(2, 5) == 7, "range 2..5");
        check(tree.rangeMax(3, 6) == 5, "range 3..6");
        check(tree.rangeMax(0, 7) == 7, "range 0..7");
        check(tree.rangeMax(5, 5) == 0, "empty range");
        check(tree.rangeMax(6, 7) == 0, "untouched tail");

        tree.update(2, 3, 4);
        check(tree.point(2) == 7, "smaller update is a no-op");
        tree.update(2, 3, 9);
        check(tree.point(2) == 9 && tree.rangeMax(1, 3) == 9, "bigger update raises");

        SkylineTree tiny = new SkylineTree(1);
        check(tiny.point(0) == 0 && tiny.rangeMax(0, 1) == 0, "tiny empty");
        tiny.update(0, 1, 42);
        check(tiny.point(0) == 42 && tiny.rangeMax(0, 1) == 42, "tiny updated");

        SkylineTree boundary = new SkylineTree(4);
        int[][] wide = {{0, 4, 1}, {0, 1, 2}, {3, 4, 3}, {1, 3, 4}};
        for (int[] update : wide) boundary.update(update[0], update[1], update[2]);
        check(join(boundary.skyline()).equals("2,4,4,3"), "boundary skyline");
        check(boundary.rangeMax(0, 4) == 4, "boundary full");
        check(boundary.rangeMax(0, 2) == 4, "boundary left");
        check(boundary.rangeMax(2, 4) == 4, "boundary right");
        check(boundary.rangeMax(3, 4) == 3, "boundary tail");

        for (int trial = 0; trial < 20; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 40;
            List<int[]> updates = lcgUpdates(1 + seed % 60, n);
            SkylineTree randomTree = new SkylineTree(n);
            List<int[]> applied = new ArrayList<>();
            for (int[] update : updates) {
                randomTree.update(update[0], update[1], update[2]);
                applied.add(update);
            }
            int[] expected = bruteSchedule(n, applied);
            check(join(randomTree.skyline()).equals(join(expected)), "random skyline " + trial);
            for (int left = 0; left < n; left++) {
                int bestSoFar = 0;
                for (int right = left + 1; right <= n; right++) {
                    bestSoFar = Math.max(bestSoFar, expected[right - 1]);
                    check(randomTree.rangeMax(left, right) == bestSoFar, "random range " + trial);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
