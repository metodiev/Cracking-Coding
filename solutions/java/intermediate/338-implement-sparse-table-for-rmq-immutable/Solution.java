/** Task 338 - Implement sparse table for RMQ (immutable). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class SparseTable {
    // Row j holds the minimum of every window of length 2^j; a query combines the two
    // overlapping power-of-two windows that cover [left, right] in O(1).
    private final int n;
    private final int[] values;
    private final int[] logs;
    private final int[][] table;

    SparseTable(int[] values) {
        if (values.length == 0) throw new IllegalArgumentException("values must not be empty");
        this.n = values.length;
        this.values = values.clone();
        this.logs = new int[n + 1];
        for (int i = 2; i <= n; i++) logs[i] = logs[i / 2] + 1;
        int levels = logs[n] + 1;
        this.table = new int[levels][];
        table[0] = this.values;
        for (int level = 1; level < levels; level++) {
            int half = 1 << (level - 1);
            int width = n - (1 << level) + 1;
            table[level] = new int[width];
            for (int i = 0; i < width; i++) {
                table[level][i] = Math.min(table[level - 1][i], table[level - 1][i + half]);
            }
        }
    }

    /** Minimum of values[left..right], both ends included. */
    int query(int left, int right) {
        if (left < 0 || left > right || right >= n) throw new IllegalArgumentException("invalid range");
        int level = logs[right - left + 1];
        return Math.min(table[level][left], table[level][right - (1 << level) + 1]);
    }
}

public class Solution {
    private static int seed = 8383;

    private static int naiveMin(int[] values, int left, int right) {
        int best = values[left];
        for (int i = left + 1; i <= right; i++) best = Math.min(best, values[i]);
        return best;
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
        // Hand-checked array; every expectation below is verified by eye.
        int[] values = {4, 2, 7, 1, 3, 9, 0, 5};
        SparseTable table = new SparseTable(values);
        check(table.query(0, 7) == 0, "full range");
        check(table.query(1, 3) == 1, "1..3");
        check(table.query(0, 0) == 4, "0..0");
        check(table.query(1, 1) == 2, "1..1");
        check(table.query(2, 2) == 7, "2..2");
        check(table.query(3, 7) == 0, "3..7");
        check(table.query(4, 6) == 0, "4..6");
        check(table.query(2, 5) == 1, "2..5");
        check(table.query(5, 7) == 0, "5..7");
        check(table.query(0, 2) == 2, "0..2");
        check(table.query(6, 6) == 0, "6..6");

        SparseTable dup = new SparseTable(new int[]{5, 5, 5, 5});
        check(dup.query(0, 3) == 5 && dup.query(1, 2) == 5, "duplicates");
        SparseTable mixed = new SparseTable(new int[]{3, -1, 4, -1, 5, -9, 2, 6, -9, 7});
        check(mixed.query(0, 9) == -9 && mixed.query(5, 8) == -9 && mixed.query(0, 3) == -1,
                "negatives and ties");
        SparseTable single = new SparseTable(new int[]{42});
        check(single.query(0, 0) == 42, "single element");

        for (int trial = 0; trial < 8; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = 1 + seed % 64;
            int[] randomValues = new int[size];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                randomValues[i] = seed % 1000;
            }
            SparseTable randomTable = new SparseTable(randomValues);
            for (int left = 0; left < size; left++) {
                for (int right = left; right < size; right++) {
                    check(randomTable.query(left, right) == naiveMin(randomValues, left, right),
                            "random " + trial + " [" + left + "," + right + "]");
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
