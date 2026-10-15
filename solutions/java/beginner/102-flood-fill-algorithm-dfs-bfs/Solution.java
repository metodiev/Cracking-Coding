/** Task 102 - Flood fill algorithm (DFS/BFS). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image.length == 0 || image[0].length == 0) return image;
        int rows = image.length, cols = image[0].length;
        int startColor = image[sr][sc];
        if (startColor == newColor) return image;
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        image[sr][sc] = newColor;
        queue.add(new int[] {sr, sc});
        int[] dr = {1, -1, 0, 0}, dc = {0, 0, 1, -1};
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            for (int k = 0; k < 4; k++) {
                int nr = cell[0] + dr[k], nc = cell[1] + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && image[nr][nc] == startColor) {
                    image[nr][nc] = newColor;
                    queue.add(new int[] {nr, nc});
                }
            }
        }
        return image;
    }

    public static void main(String[] args) {
        int[][] classic = {{1, 1, 1}, {1, 1, 0}, {1, 0, 1}};
        check(Arrays.deepEquals(floodFill(classic, 1, 1, 2), new int[][] {{2, 2, 2}, {2, 2, 0}, {2, 0, 1}}), "classic fill");
        int[][] noop = {{0, 0}, {0, 0}};
        check(Arrays.deepEquals(floodFill(noop, 0, 0, 0), new int[][] {{0, 0}, {0, 0}}), "same color no-op");
        int[][] patch = {{1, 1, 0}, {0, 1, 0}};
        check(Arrays.deepEquals(floodFill(patch, 0, 0, 5), new int[][] {{5, 5, 0}, {0, 5, 0}}), "patch");
        int[][] single = {{7}};
        check(Arrays.deepEquals(floodFill(single, 0, 0, 9), new int[][] {{9}}), "single cell");
        check(floodFill(new int[0][], 0, 0, 1).length == 0, "empty image");
        int[][] mix = {{1, 1}, {1, 0}};
        check(Arrays.deepEquals(floodFill(mix, 0, 0, 0), new int[][] {{0, 0}, {0, 0}}), "fill with zero");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
