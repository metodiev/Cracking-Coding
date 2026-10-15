/** Task 100 - Check if matrix is symmetric. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static boolean isSymmetric(int[][] m) {
        int n = m.length;
        for (int[] row : m) {
            if (row.length != n) return false;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (m[i][j] != m[j][i]) return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        check(isSymmetric(new int[][]{{1, 2}, {2, 1}}), "2x2 symmetric");
        check(!isSymmetric(new int[][]{{1, 2}, {3, 4}}), "2x2 asymmetric");
        check(isSymmetric(new int[][]{{5}}), "1x1");
        check(isSymmetric(new int[0][]), "empty");
        check(!isSymmetric(new int[][]{{}}), "one empty row");
        check(isSymmetric(new int[][]{{1, 2, 3}, {2, 4, 5}, {3, 5, 6}}), "3x3 symmetric");
        check(!isSymmetric(new int[][]{{1, 2, 3}, {2, 4, 5}, {3, 6, 6}}), "3x3 asymmetric");
        check(!isSymmetric(new int[][]{{1, 2, 3}, {4, 5, 6}}), "non-square");
        check(!isSymmetric(new int[][]{{1, 2}, {2}}), "ragged");
        check(isSymmetric(new int[][]{{0, -1}, {-1, 2}}), "negatives symmetric");
        check(!isSymmetric(new int[][]{{0, -1}, {1, 2}}), "negatives asymmetric");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
