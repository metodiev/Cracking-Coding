/** Task 101 - Count islands in boolean grid (DFS). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static int countIslands(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return 0;
        int rows = grid.length, cols = grid[0].length;
        int[][] seen = new int[rows][];
        for (int r = 0; r < rows; r++) seen[r] = grid[r].clone();
        int islands = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (seen[r][c] == 1) {
                    islands++;
                    flood(seen, r, c);
                }
            }
        }
        return islands;
    }

    static void flood(int[][] grid, int sr, int sc) {
        int rows = grid.length, cols = grid[0].length;
        int[] stack = new int[rows * cols];
        int top = 0;
        stack[top++] = sr * cols + sc;
        grid[sr][sc] = 0;
        int[] dr = {1, -1, 0, 0}, dc = {0, 0, 1, -1};
        while (top > 0) {
            int cell = stack[--top];
            int r = cell / cols, c = cell % cols;
            for (int k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1) {
                    grid[nr][nc] = 0;
                    stack[top++] = nr * cols + nc;
                }
            }
        }
    }

    public static void main(String[] args) {
        check(countIslands(new int[0][0]) == 0, "empty grid");
        check(countIslands(new int[][] {{}}) == 0, "empty row");
        check(countIslands(new int[][] {{0, 0}, {0, 0}}) == 0, "all water");
        check(countIslands(new int[][] {{1, 1}, {1, 1}}) == 1, "all land");
        check(countIslands(new int[][] {{1}}) == 1, "single land");
        check(countIslands(new int[][] {{0}}) == 0, "single water");
        check(countIslands(new int[][] {{1, 0}, {0, 1}}) == 2, "diagonal not connected");
        int[][] mixed = {
            {1, 1, 0, 0, 0},
            {1, 1, 0, 0, 1},
            {0, 0, 0, 1, 1},
            {0, 0, 0, 0, 0},
            {1, 0, 1, 0, 1},
        };
        check(countIslands(mixed) == 5, "multiple islands");
        check(mixed[0][0] == 1, "input not mutated");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
