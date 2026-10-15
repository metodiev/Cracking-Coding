/** Task 097 - Flatten a 2D array (row-major). (Tier: Beginner, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static int[] flatten2D(int[][] matrix) {
        int total = 0;
        for (int[] row : matrix) total += row.length;
        int[] result = new int[total];
        int out = 0;
        for (int[] row : matrix) {
            for (int value : row) result[out++] = value;
        }
        return result;
    }

    public static void main(String[] args) {
        check(Arrays.equals(flatten2D(new int[][]{{1, 2}, {3, 4}, {5}}), new int[]{1, 2, 3, 4, 5}), "ragged");
        check(Arrays.equals(flatten2D(new int[][]{{1}, {2, 3}, {4, 5, 6}}), new int[]{1, 2, 3, 4, 5, 6}), "growing rows");
        check(Arrays.equals(flatten2D(new int[][]{{1, 2, 3}, {4, 5, 6}}), new int[]{1, 2, 3, 4, 5, 6}), "rectangular");
        check(Arrays.equals(flatten2D(new int[][]{{1, 2}, {3, 4}}), new int[]{1, 2, 3, 4}), "2x2");
        check(Arrays.equals(flatten2D(new int[][]{{}, {1, 2}, {}}), new int[]{1, 2}), "empty rows");
        check(Arrays.equals(flatten2D(new int[][]{{1}}), new int[]{1}), "single");
        check(Arrays.equals(flatten2D(new int[0][]), new int[0]), "empty matrix");
        check(Arrays.equals(flatten2D(new int[][]{{}}), new int[0]), "one empty row");
        check(Arrays.equals(flatten2D(new int[][]{{-1, 0}, {0, -1}}), new int[]{-1, 0, 0, -1}), "negatives");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
