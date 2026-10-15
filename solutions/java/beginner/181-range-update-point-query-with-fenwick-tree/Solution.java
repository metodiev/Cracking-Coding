/** Task 181 - Range update, point query with Fenwick tree. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /**
     * Dual of task 180's Fenwick tree: the tree stores a *difference* array, so
     * rangeAdd costs two point updates and pointQuery is a prefix sum (all O(log n)).
     */
    static class RangeUpdateFenwick {
        private final int n;
        private final long[] tree;

        RangeUpdateFenwick(int size) {
            n = size;
            tree = new long[size + 1];
        }

        private void add(int i, long delta) {
            while (i <= n) {
                tree[i] += delta;
                i += i & -i;
            }
        }

        void rangeAdd(int lo, int hi, long value) {
            if (lo < 0 || hi >= n || lo > hi) throw new IndexOutOfBoundsException("range out of bounds");
            add(lo + 1, value);
            add(hi + 2, -value);      // hi + 2 > n is a silent no-op suffix update
        }

        long pointQuery(int index) {
            if (index < 0 || index >= n) throw new IndexOutOfBoundsException("index out of range");
            long total = 0;
            for (int i = index + 1; i > 0; i -= i & -i) total += tree[i];
            return total;
        }
    }

    static long[] snapshot(RangeUpdateFenwick fw, int size) {
        long[] values = new long[size];
        for (int i = 0; i < size; i++) values[i] = fw.pointQuery(i);
        return values;
    }

    public static void main(String[] args) {
        RangeUpdateFenwick fw = new RangeUpdateFenwick(6);
        fw.rangeAdd(1, 3, 5);
        check(Arrays.equals(snapshot(fw, 6), new long[] {0, 5, 5, 5, 0, 0}), "add [1,3] +5");
        fw.rangeAdd(0, 5, -2);
        check(Arrays.equals(snapshot(fw, 6), new long[] {-2, 3, 3, 3, -2, -2}), "add [0,5] -2");
        fw.rangeAdd(2, 2, 10);
        check(Arrays.equals(snapshot(fw, 6), new long[] {-2, 3, 13, 3, -2, -2}), "single position");
        fw.rangeAdd(5, 5, 7);
        check(fw.pointQuery(5) == 5, "range ending at n - 1");
        fw.rangeAdd(0, 0, -3);
        check(fw.pointQuery(0) == -5, "range starting at 0");

        boolean threw = false;
        try { fw.rangeAdd(3, 2, 1); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "lo > hi rejected");
        threw = false;
        try { fw.rangeAdd(0, 6, 1); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "hi >= n rejected");
        threw = false;
        try { fw.pointQuery(-1); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "negative index rejected");
        threw = false;
        try { new RangeUpdateFenwick(0).pointQuery(0); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "empty structure rejected");

        Random rng = new Random(11);
        int size = 12;
        long[] values = new long[size];
        RangeUpdateFenwick fw2 = new RangeUpdateFenwick(size);
        for (int t = 0; t < 80; t++) {
            int lo = rng.nextInt(size);
            int hi = lo + rng.nextInt(size - lo);
            long value = rng.nextInt(19) - 9;
            for (int k = lo; k <= hi; k++) values[k] += value;
            fw2.rangeAdd(lo, hi, value);
            for (int i = 0; i < size; i++) {
                check(fw2.pointQuery(i) == values[i], "cross-check position " + i);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
