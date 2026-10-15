/** Task 320 - Count distinct island shapes in a grid (normalised traversal). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    /** Number of 4-connected islands of 1s (shapes merged or not). */
    static int countIslands(int[][] grid) {
        int rows = grid.length;
        int cols = rows > 0 ? grid[0].length : 0;
        boolean[][] seen = new boolean[rows][cols];
        int total = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 1 && !seen[r][c]) {
                    total++;
                    ArrayDeque<int[]> stack = new ArrayDeque<>();
                    stack.push(new int[]{r, c});
                    seen[r][c] = true;
                    while (!stack.isEmpty()) {
                        int[] cell = stack.pop();
                        for (int[] step : STEPS) {
                            int nr = cell[0] + step[0], nc = cell[1] + step[1];
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1 && !seen[nr][nc]) {
                                seen[nr][nc] = true;
                                stack.push(new int[]{nr, nc});
                            }
                        }
                    }
                }
            }
        }
        return total;
    }

    static final int[][] STEPS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    /**
     * Shape keys (sets of normalised "dr,dc" cells) for 4-connected islands; only translations
     * are identified, rotations and reflections are different shapes.
     */
    static Set<Set<String>> collectIslandShapes(int[][] grid) {
        int rows = grid.length;
        int cols = rows > 0 ? grid[0].length : 0;
        boolean[][] seen = new boolean[rows][cols];
        Set<Set<String>> shapes = new HashSet<>();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 1 && !seen[r][c]) {
                    List<int[]> cells = new ArrayList<>();
                    ArrayDeque<int[]> stack = new ArrayDeque<>();
                    stack.push(new int[]{r, c});
                    seen[r][c] = true;
                    while (!stack.isEmpty()) {
                        int[] cell = stack.pop();
                        cells.add(cell);
                        for (int[] step : STEPS) {
                            int nr = cell[0] + step[0], nc = cell[1] + step[1];
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1 && !seen[nr][nc]) {
                                seen[nr][nc] = true;
                                stack.push(new int[]{nr, nc});
                            }
                        }
                    }
                    int minR = Integer.MAX_VALUE, minC = Integer.MAX_VALUE;
                    for (int[] cell : cells) {
                        minR = Math.min(minR, cell[0]);
                        minC = Math.min(minC, cell[1]);
                    }
                    Set<String> shape = new HashSet<>();
                    for (int[] cell : cells) shape.add((cell[0] - minR) + "," + (cell[1] - minC));
                    shapes.add(shape);
                }
            }
        }
        return shapes;
    }

    static int countDistinctIslands(int[][] grid) {
        return collectIslandShapes(grid).size();
    }

    /** Independent oracle: recursive flood fill + sorted tuple shape keys. */
    static Set<String> islandShapesSortedTuples(int[][] grid) {
        int rows = grid.length;
        int cols = rows > 0 ? grid[0].length : 0;
        boolean[][] seen = new boolean[rows][cols];
        Set<String> shapes = new HashSet<>();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (grid[r][c] == 1 && !seen[r][c]) {
                    List<int[]> cells = new ArrayList<>();
                    flood(grid, seen, r, c, cells);
                    int minR = Integer.MAX_VALUE, minC = Integer.MAX_VALUE;
                    for (int[] cell : cells) {
                        minR = Math.min(minR, cell[0]);
                        minC = Math.min(minC, cell[1]);
                    }
                    List<String> keys = new ArrayList<>();
                    for (int[] cell : cells) keys.add((cell[0] - minR) + "," + (cell[1] - minC));
                    Collections.sort(keys);
                    shapes.add(String.join(";", keys));
                }
            }
        }
        return shapes;
    }

    private static void flood(int[][] grid, boolean[][] seen, int r, int c, List<int[]> cells) {
        int rows = grid.length, cols = grid[0].length;
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != 1 || seen[r][c]) return;
        seen[r][c] = true;
        cells.add(new int[]{r, c});
        flood(grid, seen, r + 1, c, cells);
        flood(grid, seen, r - 1, c, cells);
        flood(grid, seen, r, c + 1, cells);
        flood(grid, seen, r, c - 1, cells);
    }

    /** Canonical string form of the frozenset-style shape keys, for a direct set comparison. */
    static Set<String> canonicalize(Set<Set<String>> shapes) {
        Set<String> out = new HashSet<>();
        for (Set<String> shape : shapes) {
            List<String> cells = new ArrayList<>(shape);
            Collections.sort(cells);
            out.add(String.join(";", cells));
        }
        return out;
    }

    static boolean isConnectedShape(Set<String> shape) {
        if (shape.isEmpty()) return false;
        ArrayDeque<String> stack = new ArrayDeque<>();
        Set<String> reached = new HashSet<>();
        String start = shape.iterator().next();
        stack.push(start);
        reached.add(start);
        while (!stack.isEmpty()) {
            String[] parts = stack.pop().split(",");
            int r = Integer.parseInt(parts[0]), c = Integer.parseInt(parts[1]);
            for (int[] step : STEPS) {
                String next = (r + step[0]) + "," + (c + step[1]);
                if (shape.contains(next) && reached.add(next)) stack.push(next);
            }
        }
        return reached.equals(shape);
    }

    private static int seed = 3200;

    private static int[][] lcgGrid(int rows, int cols) {
        int[][] grid = new int[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 3 == 0 ? 1 : 0;
            }
        }
        return grid;
    }

    public static void main(String[] args) {
        int[][] classic = {
            {1, 1, 0, 0, 0},
            {1, 1, 0, 0, 0},
            {0, 0, 0, 1, 1},
            {0, 0, 0, 0, 0},
            {1, 1, 1, 1, 1},
        };
        check(countDistinctIslands(classic) == 3, "classic");
        Set<String> expected = new HashSet<>(Arrays.asList(
                "0,0;0,1;1,0;1,1",                                  // 2x2 square
                "0,0;0,1",                                          // horizontal bar
                "0,0;0,1;0,2;0,3;0,4"));                            // 1x5 line
        check(canonicalize(collectIslandShapes(classic)).equals(expected), "classic shapes");

        // Rotations are NOT identified: a horizontal bar and a vertical bar are different shapes
        int[][] rotated = {
            {1, 1, 0},
            {0, 0, 0},
            {0, 0, 1},
            {0, 0, 1},
        };
        check(countDistinctIslands(rotated) == 2, "rotation not equal");
        check(canonicalize(collectIslandShapes(rotated)).equals(new HashSet<>(Arrays.asList("0,0;0,1", "0,0;1,0"))), "rotation keys");

        // Identical islands (even far apart) count once
        check(countDistinctIslands(new int[][]{{1, 1, 0, 0}, {1, 1, 0, 0}, {0, 0, 1, 1}, {0, 0, 1, 1}}) == 1, "identical");
        // ... but an L-shape and its mirror are two shapes
        check(countDistinctIslands(new int[][]{{1, 0, 0}, {1, 1, 0}, {0, 0, 1}, {0, 0, 1}}) == 2, "mirror");
        check(countDistinctIslands(new int[][]{{1}}) == 1, "single cell");
        check(countDistinctIslands(new int[][]{{0, 0}, {0, 0}}) == 0, "all zeros");
        check(countDistinctIslands(new int[][]{}) == 0, "empty grid");

        // Diagonal cells are not 4-connected: two separate islands, but both are single cells,
        // so they still share one shape. The 3x3 grid below distinguishes the two situations:
        // 4-connectivity gives {(0,0)} and {(0,0),(0,1)} -> 2 shapes, diagonal merging only 1.
        check(countIslands(new int[][]{{1, 0}, {0, 1}}) == 2, "diagonal islands");
        check(countDistinctIslands(new int[][]{{1, 0}, {0, 1}}) == 1, "diagonal shapes");
        check(countDistinctIslands(new int[][]{{1, 0, 0}, {0, 1, 1}, {0, 0, 0}}) == 2, "diagonal shapes 2");
        check(countDistinctIslands(new int[][]{{1, 1, 1, 1}}) == 1, "single row");
        check(countDistinctIslands(new int[][]{{1}, {1}, {1}}) == 1, "single column");

        // Zero padding on the top/left must not change the answer (shapes are translation invariant)
        int[][] padded = new int[9][7];
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++) padded[r + 2][c + 2] = classic[r][c];
        check(countDistinctIslands(padded) == 3, "padded");

        List<int[][]> grids = new ArrayList<>();
        for (int i = 0; i < 40; i++) grids.add(lcgGrid(8, 8));
        for (int i = 0; i < 20; i++) grids.add(lcgGrid(6, 11));
        for (int i = 0; i < 20; i++) grids.add(lcgGrid(4, 4));

        for (int[][] grid : grids) {
            Set<Set<String>> shapes = collectIslandShapes(grid);
            Set<String> oracle = islandShapesSortedTuples(grid);
            check(canonicalize(shapes).equals(oracle), "cross-check");
            check(countDistinctIslands(grid) == oracle.size(), "count");
            for (Set<String> shape : shapes) {
                int minR = Integer.MAX_VALUE, minC = Integer.MAX_VALUE;
                for (String key : shape) {
                    String[] parts = key.split(",");
                    minR = Math.min(minR, Integer.parseInt(parts[0]));
                    minC = Math.min(minC, Integer.parseInt(parts[1]));
                }
                check(minR == 0 && minC == 0, "normalised");                 // keys are normalised ...
                check(isConnectedShape(shape), "connected");                 // ... and 4-connected
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
