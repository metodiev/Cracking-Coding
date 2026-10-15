/** Task 103 - Shortest path in unweighted grid (BFS). (Tier: Beginner, CTCI Ch 11) */
import java.util.*;

public class Solution {
    static int shortestPath(int[][] grid, int[] start, int[] goal) {
        if (grid.length == 0 || grid[0].length == 0) return -1;
        int rows = grid.length, cols = grid[0].length;
        int sr = start[0], sc = start[1], gr = goal[0], gc = goal[1];
        if (grid[sr][sc] == 1 || grid[gr][gc] == 1) return -1;
        if (sr == gr && sc == gc) return 0;
        int[][] dist = new int[rows][cols];
        for (int[] row : dist) Arrays.fill(row, -1);
        dist[sr][sc] = 0;
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        queue.add(new int[] {sr, sc});
        int[] dr = {1, -1, 0, 0}, dc = {0, 0, 1, -1};
        while (!queue.isEmpty()) {
            int[] cell = queue.poll();
            for (int k = 0; k < 4; k++) {
                int nr = cell[0] + dr[k], nc = cell[1] + dc[k];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 0 && dist[nr][nc] == -1) {
                    dist[nr][nc] = dist[cell[0]][cell[1]] + 1;
                    if (nr == gr && nc == gc) return dist[nr][nc];
                    queue.add(new int[] {nr, nc});
                }
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        int[][] open3 = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        check(shortestPath(open3, new int[] {0, 0}, new int[] {2, 2}) == 4, "open grid");
        check(shortestPath(open3, new int[] {0, 0}, new int[] {0, 0}) == 0, "start == goal");
        check(shortestPath(new int[][] {{0}}, new int[] {0, 0}, new int[] {0, 0}) == 0, "single cell");
        int[][] detour = {{0, 0, 0}, {1, 1, 0}, {0, 0, 0}};
        check(shortestPath(detour, new int[] {0, 0}, new int[] {2, 0}) == 6, "forced detour");
        int[][] split = {{0, 1}, {1, 0}};
        check(shortestPath(split, new int[] {0, 0}, new int[] {1, 1}) == -1, "diagonal unreachable");
        int[][] blocked = {{0, 1}, {0, 0}};
        check(shortestPath(blocked, new int[] {0, 0}, new int[] {0, 1}) == -1, "blocked goal");
        check(shortestPath(blocked, new int[] {0, 1}, new int[] {0, 0}) == -1, "blocked start");
        check(shortestPath(new int[0][], new int[] {0, 0}, new int[] {0, 0}) == -1, "empty grid");
        int[][] direct = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
        check(shortestPath(direct, new int[] {0, 0}, new int[] {3, 3}) == 6, "manhattan distance");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
