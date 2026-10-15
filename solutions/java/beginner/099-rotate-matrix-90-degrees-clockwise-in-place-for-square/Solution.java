/** Task 099 - Rotate matrix 90 degrees clockwise (in-place for square). (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static void rotateMatrix90Clockwise(int[][] m) {
        int n = m.length;
        for (int[] row : m) {
            if (row.length != n) throw new IllegalArgumentException("matrix must be square");
        }
        // Transpose across the main diagonal.
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int tmp = m[i][j];
                m[i][j] = m[j][i];
                m[j][i] = tmp;
            }
        }
        // Reverse every row.
        for (int[] row : m) {
            for (int lo = 0, hi = row.length - 1; lo < hi; lo++, hi--) {
                int tmp = row[lo];
                row[lo] = row[hi];
                row[hi] = tmp;
            }
        }
    }

    public static void main(String[] args) {
        int[][] m1 = {{1}};
        rotateMatrix90Clockwise(m1);
        check(deepEquals(m1, new int[][]{{1}}), "1x1");

        int[][] m2 = {{1, 2}, {3, 4}};
        rotateMatrix90Clockwise(m2);
        check(deepEquals(m2, new int[][]{{3, 1}, {4, 2}}), "2x2");

        int[][] m3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        rotateMatrix90Clockwise(m3);
        check(deepEquals(m3, new int[][]{{7, 4, 1}, {8, 5, 2}, {9, 6, 3}}), "3x3");

        int[][] m4 = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        rotateMatrix90Clockwise(m4);
        check(deepEquals(m4, new int[][]{
                {13, 9, 5, 1},
                {14, 10, 6, 2},
                {15, 11, 7, 3},
                {16, 12, 8, 4}}), "4x4");

        boolean threw = false;
        try {
            rotateMatrix90Clockwise(new int[][]{{1, 2, 3}, {4, 5, 6}});
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "non-square throws");
        System.out.println("PASS");
    }

    static boolean deepEquals(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (!Arrays.equals(a[i], b[i])) return false;
        }
        return true;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
