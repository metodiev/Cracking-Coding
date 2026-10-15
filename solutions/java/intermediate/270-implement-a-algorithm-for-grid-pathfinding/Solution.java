/** Task 270 - Implement A* algorithm for grid pathfinding. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    private static final int[][] DIRS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    private static boolean free(int[][] grid, int r, int c) {
        return r >= 0 && r < grid.length && c >= 0 && c < grid[0].length && grid[r][c] == 0;
    }

    static int manhattan(int[] a, int[] b) {
        return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
    }

    /**
     * A* on a 4-neighbour grid with unit move cost and the Manhattan heuristic, which is
     * admissible (one move changes the distance to the goal by at most 1) and consistent.
     * grid[r][c] == 1 is a wall. Returns the minimum-move path including both endpoints,
     * or null when no path exists (also when start or goal lies on a wall).
     * Deterministic tie-break: the open set is ordered by (f, h, row-major cell id), so
     * with equal f the cell closer to the goal and then the smaller index expands first.
     * Closed-set lazy deletion (stale queue entries are skipped) keeps it simple.
     */
    static List<int[]> astar(int[][] grid, int[] start, int[] goal) {
        int rows = grid.length, cols = rows > 0 ? grid[0].length : 0;
        if (!free(grid, start[0], start[1]) || !free(grid, goal[0], goal[1])) return null;
        int sr = start[0], sc = start[1], gr = goal[0], gc = goal[1];
        if (sr == gr && sc == gc) {
            List<int[]> single = new ArrayList<>();
            single.add(new int[]{sr, sc});
            return single;
        }
        int[][] g = new int[rows][cols];
        for (int[] row : g) Arrays.fill(row, Integer.MAX_VALUE);
        g[sr][sc] = 0;
        boolean[][] closed = new boolean[rows][cols];
        int[][] parentR = new int[rows][cols], parentC = new int[rows][cols];
        PriorityQueue<int[]> pq = new PriorityQueue<>((p, q) -> {
            if (p[0] != q[0]) return Integer.compare(p[0], q[0]);   // f
            if (p[1] != q[1]) return Integer.compare(p[1], q[1]);   // h
            return Integer.compare(p[2], q[2]);                     // row-major cell id
        });
        int h0 = manhattan(start, goal);
        pq.add(new int[]{h0, h0, sr * cols + sc, sr, sc});
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int r = cur[3], c = cur[4];
            if (closed[r][c]) continue;
            closed[r][c] = true;
            if (r == gr && c == gc) {
                LinkedList<int[]> path = new LinkedList<>();
                int pr = r, pc = c;
                while (pr != sr || pc != sc) {
                    path.addFirst(new int[]{pr, pc});
                    int nr = parentR[pr][pc], nc = parentC[pr][pc];
                    pr = nr;
                    pc = nc;
                }
                path.addFirst(new int[]{sr, sc});
                return path;
            }
            for (int[] d : DIRS) {
                int nr = r + d[0], nc = c + d[1];
                if (!free(grid, nr, nc) || closed[nr][nc]) continue;
                int tentative = g[r][c] + 1;
                if (tentative < g[nr][nc]) {
                    g[nr][nc] = tentative;
                    parentR[nr][nc] = r;
                    parentC[nr][nc] = c;
                    int h = Math.abs(nr - gr) + Math.abs(nc - gc);
                    pq.add(new int[]{tentative + h, h, nr * cols + nc, nr, nc});
                }
            }
        }
        return null;
    }

    static Integer astarCost(int[][] grid, int[] start, int[] goal) {
        List<int[]> path = astar(grid, start, goal);
        return path == null ? null : path.size() - 1;
    }

    /** Exact distance from every free cell to goal (-1 when unreachable/walled). */
    static int[][] bfsDistancesFrom(int[][] grid, int[] goal) {
        int rows = grid.length, cols = rows > 0 ? grid[0].length : 0;
        int[][] dist = new int[rows][cols];
        for (int[] row : dist) Arrays.fill(row, -1);
        if (!free(grid, goal[0], goal[1])) return dist;
        dist[goal[0]][goal[1]] = 0;
        Deque<int[]> q = new ArrayDeque<>();
        q.add(new int[]{goal[0], goal[1]});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            for (int[] d : DIRS) {
                int nr = cur[0] + d[0], nc = cur[1] + d[1];
                if (free(grid, nr, nc) && dist[nr][nc] == -1) {
                    dist[nr][nc] = dist[cur[0]][cur[1]] + 1;
                    q.add(new int[]{nr, nc});
                }
            }
        }
        return dist;
    }

    static int bfsCost(int[][] grid, int[] start, int[] goal) {
        int[][] dist = bfsDistancesFrom(grid, goal);
        if (dist.length == 0 || dist[0].length == 0) return -1;
        if (start[0] < 0 || start[0] >= dist.length || start[1] < 0 || start[1] >= dist[0].length) return -1;
        return dist[start[0]][start[1]];
    }

    static boolean validPath(int[][] grid, List<int[]> path, int[] start, int[] goal) {
        if (path == null) return false;
        int[] first = path.get(0), last = path.get(path.size() - 1);
        if (first[0] != start[0] || first[1] != start[1]) return false;
        if (last[0] != goal[0] || last[1] != goal[1]) return false;
        for (int[] cell : path)
            if (!(cell[0] >= 0 && cell[0] < grid.length && cell[1] >= 0 && cell[1] < grid[0].length
                    && grid[cell[0]][cell[1]] == 0)) return false;
        for (int i = 1; i < path.size(); i++) {
            int step = Math.abs(path.get(i)[0] - path.get(i - 1)[0]) + Math.abs(path.get(i)[1] - path.get(i - 1)[1]);
            if (step != 1) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        int[][] serpentine = {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0},
        };
        int[][] detour = {
            {0, 0, 0, 0},
            {1, 1, 1, 0},
            {0, 0, 0, 0},
        };
        int[][] blocked = {
            {0, 1, 0},
            {0, 1, 0},
            {1, 1, 0},
        };
        int[][] openGrid = new int[4][6];

        check(astarCost(serpentine, new int[]{0, 0}, new int[]{4, 4}) == 8, "serpentine = 8");
        check(bfsCost(serpentine, new int[]{0, 0}, new int[]{4, 4}) == 8, "bfs serpentine = 8");
        Integer detourCost = astarCost(detour, new int[]{0, 0}, new int[]{2, 0});
        check(detourCost != null && detourCost == 8, "detour = 8 (vs Manhattan 2)");
        check(bfsCost(detour, new int[]{0, 0}, new int[]{2, 0}) == 8, "bfs detour = 8");
        check(astarCost(detour, new int[]{2, 0}, new int[]{0, 0}) == 8, "detour reversed = 8");
        check(astarCost(openGrid, new int[]{0, 0}, new int[]{3, 5}) == 8, "open grid = Manhattan 8");
        check(astarCost(blocked, new int[]{0, 0}, new int[]{0, 2}) == null, "no path -> null");
        check(bfsCost(blocked, new int[]{0, 0}, new int[]{0, 2}) == -1, "bfs no path -> -1");

        List<int[]> self = astar(serpentine, new int[]{2, 2}, new int[]{2, 2});
        check(self != null && self.size() == 1 && self.get(0)[0] == 2 && self.get(0)[1] == 2,
                "start == goal -> single cell");
        check(astar(serpentine, new int[]{1, 1}, new int[]{0, 0}) == null, "start on a wall");
        check(astar(serpentine, new int[]{0, 0}, new int[]{1, 1}) == null, "goal on a wall");

        int[][][] grids = {serpentine, detour, detour, openGrid, serpentine};
        int[][] starts = {{0, 0}, {0, 0}, {2, 0}, {0, 0}, {2, 2}};
        int[][] goals = {{4, 4}, {2, 0}, {0, 0}, {3, 5}, {2, 2}};
        for (int i = 0; i < grids.length; i++) {
            List<int[]> path = astar(grids[i], starts[i], goals[i]);
            check(validPath(grids[i], path, starts[i], goals[i]), "valid path #" + i);
            if (grids[i] == openGrid) {                 // on an open grid Manhattan is exact
                int[][] dist = bfsDistancesFrom(openGrid, goals[i]);
                for (int r = 0; r < openGrid.length; r++)
                    for (int c = 0; c < openGrid[0].length; c++)
                        check(Math.abs(r - goals[i][0]) + Math.abs(c - goals[i][1]) == dist[r][c],
                                "open grid heuristic exact");
            }
        }

        int[][][] admissibilityGrids = {serpentine, detour, blocked};
        int[][] admissibilityGoals = {{4, 4}, {2, 0}, {0, 2}};
        for (int i = 0; i < admissibilityGrids.length; i++) {
            int[][] dist = bfsDistancesFrom(admissibilityGrids[i], admissibilityGoals[i]);
            for (int r = 0; r < admissibilityGrids[i].length; r++)
                for (int c = 0; c < admissibilityGrids[i][0].length; c++)
                    if (dist[r][c] >= 0)
                        check(Math.abs(r - admissibilityGoals[i][0]) + Math.abs(c - admissibilityGoals[i][1]) <= dist[r][c],
                                "admissible on reachable cells");
        }

        Random rng = new Random(270);
        for (int test = 0; test < 300; test++) {        // A* cost == BFS cost + valid path
            int rows = 2 + rng.nextInt(5), cols = 2 + rng.nextInt(5);
            int[][] grid = new int[rows][cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) grid[r][c] = rng.nextDouble() < 0.3 ? 1 : 0;
            int[] start = {rng.nextInt(rows), rng.nextInt(cols)};
            int[] goal = {rng.nextInt(rows), rng.nextInt(cols)};
            Integer got = astarCost(grid, start, goal);
            int want = bfsCost(grid, start, goal);
            check((got == null ? -1 : got) == want, "random cost agreement");
            if (got != null) {
                check(validPath(grid, astar(grid, start, goal), start, goal), "random path validity");
                if (want > 0) {
                    int[][] dist = bfsDistancesFrom(grid, goal);
                    check(Math.abs(start[0] - goal[0]) + Math.abs(start[1] - goal[1]) <= dist[start[0]][start[1]],
                            "random admissibility");
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
