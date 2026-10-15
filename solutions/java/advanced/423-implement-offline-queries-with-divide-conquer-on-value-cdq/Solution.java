/** Task 423 - Implement offline queries with divide & conquer on value (CDQ). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class CdqCounter {
    private final int[] xs;
    private final int[] ys;
    private final int[] ranks;
    private final int[] bit;
    private final int rankCount;
    private long total;

    CdqCounter(int[][] points) {
        int n = points.length;
        xs = new int[n];
        ys = new int[n];
        for (int i = 0; i < n; i++) {
            xs[i] = points[i][0];
            ys[i] = points[i][1];
        }
        int[] sortedY = ys.clone();
        Arrays.sort(sortedY);
        Map<Integer, Integer> rankOf = new HashMap<>();
        int next = 0;
        for (int value : sortedY) {
            if (!rankOf.containsKey(value)) rankOf.put(value, next++);
        }
        ranks = new int[n];
        for (int i = 0; i < n; i++) ranks[i] = rankOf.get(ys[i]);
        rankCount = next;
        bit = new int[rankCount + 1];
    }

    long count() {
        if (xs.length < 2) return 0;
        cdq(0, xs.length);
        return total;
    }

    private void bitAdd(int rank, int delta) {
        int i = rank + 1;
        while (i <= rankCount) {
            bit[i] += delta;
            i += i & -i;
        }
    }

    private int bitPrefix(int count) {          // active values with rank < count
        int result = 0;
        int i = count;
        while (i > 0) {
            result += bit[i];
            i -= i & -i;
        }
        return result;
    }

    private int[] cdq(int lo, int hi) {
        // returns indices [lo, hi) sorted by x descending; adds cross pairs to total
        if (hi - lo <= 1) return new int[]{lo};
        int mid = (lo + hi) / 2;
        int[] left = cdq(lo, mid);
        int[] right = cdq(mid, hi);
        int[] touched = new int[left.length];
        int touchedCount = 0;
        int pointer = 0;
        for (int j : right) {                   // right is sorted by x descending too
            while (pointer < left.length && xs[left[pointer]] > xs[j]) {
                int rank = ranks[left[pointer]];
                bitAdd(rank, 1);
                touched[touchedCount++] = rank;
                pointer++;
            }
            total += pointer - bitPrefix(ranks[j] + 1);
        }
        for (int i = 0; i < touchedCount; i++) bitAdd(touched[i], -1);
        int[] merged = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        while (i < left.length && j < right.length) {
            if (xs[left[i]] >= xs[right[j]]) merged[k++] = left[i++];
            else merged[k++] = right[j++];
        }
        while (i < left.length) merged[k++] = left[i++];
        while (j < right.length) merged[k++] = right[j++];
        return merged;
    }
}

public class Solution {
    static long bruteCount(int[][] points) {
        long total = 0;
        for (int i = 0; i < points.length; i++) {
            for (int j = i + 1; j < points.length; j++) {
                if (points[i][0] > points[j][0] && points[i][1] > points[j][1]) total++;
            }
        }
        return total;
    }

    public static void main(String[] args) {
        check(new CdqCounter(new int[][]{}).count() == 0, "empty");
        check(new CdqCounter(new int[][]{{1, 1}}).count() == 0, "single");
        check(new CdqCounter(new int[][]{{3, 1}, {2, 2}}).count() == 0, "first fails y");
        check(new CdqCounter(new int[][]{{2, 2}, {3, 1}}).count() == 0, "x order");
        check(new CdqCounter(new int[][]{{5, 5}, {3, 4}, {1, 2}}).count() == 3, "decreasing chain");
        check(new CdqCounter(new int[][]{{1, 1}, {2, 2}, {3, 3}}).count() == 0, "increasing chain");
        check(new CdqCounter(new int[][]{{3, 3}, {3, 2}}).count() == 0, "equal x not counted");
        check(new CdqCounter(new int[][]{{4, 4}, {3, 4}}).count() == 0, "equal y not counted");
        int[][] decreasing = new int[5][];
        for (int i = 0; i < 5; i++) decreasing[i] = new int[]{7 - i, 9 - i};
        check(new CdqCounter(decreasing).count() == 10, "all pairs of a decreasing chain");
        check(new CdqCounter(new int[][]{{5, 1}, {4, 4}, {6, 2}, {3, 3}}).count() == 1, "mixed points");

        int local = 424242;
        for (int trial = 0; trial < 40; trial++) {
            local = (local * 75 + 74) % 65537;
            int n = 2 + local % 59;
            int[][] points = new int[n][];
            for (int i = 0; i < n; i++) {
                local = (local * 75 + 74) % 65537;
                int x = local % 11 - 5;
                local = (local * 75 + 74) % 65537;
                points[i] = new int[]{x, local % 11 - 5};
            }
            long got = new CdqCounter(points).count();
            long expected = bruteCount(points);
            check(got == expected, "lcg trial " + trial + " got=" + got + " want=" + expected);
        }

        int[][] equal = new int[12][];
        for (int i = 0; i < 12; i++) equal[i] = new int[]{7, 7};
        check(new CdqCounter(equal).count() == 0, "all-equal points");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
