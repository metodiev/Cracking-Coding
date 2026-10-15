/** Task 098 - Transpose a 2D matrix. (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static int[][] transpose(int[][] matrix) {
        if (matrix.length == 0) return new int[0][];
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] result = new int[cols][rows];
        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                result[c][r] = matrix[r][c];
            }
        }
        return result;
    }

    public static void main(String[] args) {
        check(deepEquals(transpose(new int[][]{{1, 2, 3}, {4, 5, 6}}), new int[][]{{1, 4}, {2, 5}, {3, 6}}), "2x3");
        check(deepEquals(transpose(new int[][]{{1, 2}, {3, 4}}), new int[][]{{1, 3}, {2, 4}}), "2x2");
        check(deepEquals(transpose(new int[][]{{1, 2, 3}}), new int[][]{{1}, {2}, {3}}), "1x3");
        check(deepEquals(transpose(new int[][]{{1}, {2}, {3}}), new int[][]{{1, 2, 3}}), "3x1");
        check(deepEquals(transpose(new int[][]{{1}}), new int[][]{{1}}), "1x1");
        check(deepEquals(transpose(new int[0][]), new int[0][]), "empty");
        check(deepEquals(transpose(new int[][]{{}}), new int[0][]), "one empty row");
        check(deepEquals(transpose(new int[][]{{1, 2}, {3, 4}, {5, 6}}), new int[][]{{1, 3, 5}, {2, 4, 6}}), "3x2");

        int[][] m = {{1, 2}, {3, 4}};
        int[][] t = transpose(m);
        check(t != m, "returns a new matrix");
        check(deepEquals(m, new int[][]{{1, 2}, {3, 4}}), "input unchanged");
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
