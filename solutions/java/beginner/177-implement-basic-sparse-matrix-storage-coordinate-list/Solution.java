/** Task 177 - Implement basic sparse matrix storage (coordinate list). (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    static class SparseMatrix {
        final int rows, cols;
        final Map<Long, Integer> data = new HashMap<>();   // packed key (row * cols + col)

        SparseMatrix(int rows, int cols) {
            if (rows < 0 || cols < 0) throw new IllegalArgumentException("dimensions must be non-negative");
            this.rows = rows;
            this.cols = cols;
        }

        private long key(int row, int col) { return (long) row * cols + col; }

        private void check(int row, int col) {
            if (row < 0 || row >= rows || col < 0 || col >= cols)
                throw new IndexOutOfBoundsException("index out of range: (" + row + ", " + col + ")");
        }

        void set(int row, int col, int value) {
            check(row, col);
            if (value == 0) data.remove(key(row, col));
            else data.put(key(row, col), value);
        }

        int get(int row, int col) {
            check(row, col);
            return data.getOrDefault(key(row, col), 0);
        }

        int nnz() { return data.size(); }

        int[][] toDense() {
            int[][] dense = new int[rows][cols];
            for (Map.Entry<Long, Integer> e : data.entrySet()) {
                long k = e.getKey();
                dense[(int) (k / cols)][(int) (k % cols)] = e.getValue();
            }
            return dense;
        }
    }

    public static void main(String[] args) {
        SparseMatrix m = new SparseMatrix(3, 4);
        m.set(0, 0, 5);
        m.set(1, 2, 7);
        m.set(2, 3, -1);
        check(m.nnz() == 3, "three stored");
        check(m.get(0, 0) == 5 && m.get(0, 1) == 0 && m.get(2, 3) == -1, "get");
        int[][] dense = m.toDense();
        check(dense[0][0] == 5 && dense[1][2] == 7 && dense[2][3] == -1 && dense[1][1] == 0, "dense");

        m.set(1, 2, 0);
        check(m.nnz() == 2 && m.get(1, 2) == 0, "zero removes");

        SparseMatrix empty = new SparseMatrix(2, 2);
        check(empty.nnz() == 0 && empty.toDense()[1][1] == 0, "empty matrix");

        boolean threw = false;
        try { m.get(5, 0); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "bounds checked");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
