// Task 320 - Count distinct island shapes in a grid (normalised traversal). (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static readonly int[][] Steps =
    {
        new[] { 1, 0 }, new[] { -1, 0 }, new[] { 0, 1 }, new[] { 0, -1 },
    };

    /// <summary>Number of 4-connected islands of 1s (shapes merged or not).</summary>
    public static int CountIslands(int[][] grid)
    {
        int rows = grid.Length;
        int cols = rows > 0 ? grid[0].Length : 0;
        var seen = new bool[rows, cols];
        int total = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (grid[r][c] == 1 && !seen[r, c])
                {
                    total++;
                    var stack = new Stack<int[]>();
                    stack.Push(new[] { r, c });
                    seen[r, c] = true;
                    while (stack.Count > 0)
                    {
                        int[] cell = stack.Pop();
                        foreach (int[] step in Steps)
                        {
                            int nr = cell[0] + step[0], nc = cell[1] + step[1];
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1 && !seen[nr, nc])
                            {
                                seen[nr, nc] = true;
                                stack.Push(new[] { nr, nc });
                            }
                        }
                    }
                }
            }
        }
        return total;
    }

    /// <summary>
    /// One sorted array of normalised "dr,dc" cells per island (translation only; rotations and
    /// reflections are different shapes).
    /// </summary>
    public static List<List<string>> CollectIslandShapes(int[][] grid)
    {
        int rows = grid.Length;
        int cols = rows > 0 ? grid[0].Length : 0;
        var seen = new bool[rows, cols];
        var shapes = new List<List<string>>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (grid[r][c] == 1 && !seen[r, c])
                {
                    var stack = new Stack<int[]>();
                    stack.Push(new[] { r, c });
                    seen[r, c] = true;
                    var cells = new List<int[]>();
                    while (stack.Count > 0)
                    {
                        int[] cell = stack.Pop();
                        cells.Add(cell);
                        foreach (int[] step in Steps)
                        {
                            int nr = cell[0] + step[0], nc = cell[1] + step[1];
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == 1 && !seen[nr, nc])
                            {
                                seen[nr, nc] = true;
                                stack.Push(new[] { nr, nc });
                            }
                        }
                    }
                    int minR = cells.Min(cell => cell[0]);
                    int minC = cells.Min(cell => cell[1]);
                    var shape = cells.Select(cell => (cell[0] - minR) + "," + (cell[1] - minC)).ToList();
                    shape.Sort(StringComparer.Ordinal);
                    shapes.Add(shape);
                }
            }
        }
        return shapes;
    }

    public static HashSet<string> CollectIslandShapeKeys(int[][] grid)
    {
        return new HashSet<string>(CollectIslandShapes(grid).Select(shape => string.Join(";", shape)));
    }

    public static int CountDistinctIslands(int[][] grid) => CollectIslandShapeKeys(grid).Count;

    /// <summary>Independent oracle: recursive flood fill, returning a set of canonical joined keys.</summary>
    public static HashSet<string> IslandShapesSortedTuples(int[][] grid)
    {
        int rows = grid.Length;
        int cols = rows > 0 ? grid[0].Length : 0;
        var seen = new bool[rows, cols];
        var shapes = new HashSet<string>();

        void Flood(int r, int c, List<int[]> cells)
        {
            if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != 1 || seen[r, c]) return;
            seen[r, c] = true;
            cells.Add(new[] { r, c });
            Flood(r + 1, c, cells);
            Flood(r - 1, c, cells);
            Flood(r, c + 1, cells);
            Flood(r, c - 1, cells);
        }

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (grid[r][c] == 1 && !seen[r, c])
                {
                    var cells = new List<int[]>();
                    Flood(r, c, cells);
                    int minR = cells.Min(cell => cell[0]);
                    int minC = cells.Min(cell => cell[1]);
                    var keys = cells.Select(cell => (cell[0] - minR) + "," + (cell[1] - minC)).OrderBy(key => key, StringComparer.Ordinal);
                    shapes.Add(string.Join(";", keys));
                }
            }
        }
        return shapes;
    }

    public static bool IsConnectedShape(List<string> shape)
    {
        if (shape.Count == 0) return false;
        var cellSet = new HashSet<string>(shape);
        var stack = new Stack<string>();
        var reached = new HashSet<string>();
        stack.Push(shape[0]);
        reached.Add(shape[0]);
        while (stack.Count > 0)
        {
            string[] parts = stack.Pop().Split(',');
            int r = int.Parse(parts[0]), c = int.Parse(parts[1]);
            foreach (int[] step in Steps)
            {
                string next = (r + step[0]) + "," + (c + step[1]);
                if (cellSet.Contains(next) && reached.Add(next)) stack.Push(next);
            }
        }
        return reached.Count == cellSet.Count;
    }

    private static int seed = 3200;

    private static int[][] LcgGrid(int rows, int cols)
    {
        var grid = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            grid[r] = new int[cols];
            for (int c = 0; c < cols; c++)
            {
                seed = (seed * 75 + 74) % 65537;
                grid[r][c] = seed % 3 == 0 ? 1 : 0;
            }
        }
        return grid;
    }

    public static void Main()
    {
        var classic = new[]
        {
            new[] { 1, 1, 0, 0, 0 },
            new[] { 1, 1, 0, 0, 0 },
            new[] { 0, 0, 0, 1, 1 },
            new[] { 0, 0, 0, 0, 0 },
            new[] { 1, 1, 1, 1, 1 },
        };
        Check(CountDistinctIslands(classic) == 3, "classic");
        Check(CollectIslandShapeKeys(classic).SetEquals(new[]
        {
            "0,0;0,1;1,0;1,1",                                  // 2x2 square
            "0,0;0,1",                                          // horizontal bar
            "0,0;0,1;0,2;0,3;0,4",                              // 1x5 line
        }), "classic shapes");

        // Rotations are NOT identified: a horizontal bar and a vertical bar are different shapes
        var rotated = new[]
        {
            new[] { 1, 1, 0 },
            new[] { 0, 0, 0 },
            new[] { 0, 0, 1 },
            new[] { 0, 0, 1 },
        };
        Check(CountDistinctIslands(rotated) == 2, "rotation not equal");
        Check(CollectIslandShapeKeys(rotated).SetEquals(new[] { "0,0;0,1", "0,0;1,0" }), "rotation keys");

        // Identical islands (even far apart) count once
        Check(CountDistinctIslands(new[] { new[] { 1, 1, 0, 0 }, new[] { 1, 1, 0, 0 }, new[] { 0, 0, 1, 1 }, new[] { 0, 0, 1, 1 } }) == 1, "identical");
        // ... but an L-shape and its mirror are two shapes
        Check(CountDistinctIslands(new[] { new[] { 1, 0, 0 }, new[] { 1, 1, 0 }, new[] { 0, 0, 1 }, new[] { 0, 0, 1 } }) == 2, "mirror");
        Check(CountDistinctIslands(new[] { new[] { 1 } }) == 1, "single cell");
        Check(CountDistinctIslands(new[] { new[] { 0, 0 }, new[] { 0, 0 } }) == 0, "all zeros");
        Check(CountDistinctIslands(Array.Empty<int[]>()) == 0, "empty grid");

        // Diagonal cells are not 4-connected: two separate islands, but both are single cells,
        // so they still share one shape. The 3x3 grid below distinguishes the two situations:
        // 4-connectivity gives {(0,0)} and {(0,0),(0,1)} -> 2 shapes, diagonal merging only 1.
        Check(CountIslands(new[] { new[] { 1, 0 }, new[] { 0, 1 } }) == 2, "diagonal islands");
        Check(CountDistinctIslands(new[] { new[] { 1, 0 }, new[] { 0, 1 } }) == 1, "diagonal shapes");
        Check(CountDistinctIslands(new[] { new[] { 1, 0, 0 }, new[] { 0, 1, 1 }, new[] { 0, 0, 0 } }) == 2, "diagonal shapes 2");
        Check(CountDistinctIslands(new[] { new[] { 1, 1, 1, 1 } }) == 1, "single row");
        Check(CountDistinctIslands(new[] { new[] { 1 }, new[] { 1 }, new[] { 1 } }) == 1, "single column");

        // Zero padding on the top/left must not change the answer (shapes are translation invariant)
        var padded = new int[9][];
        for (int r = 0; r < 9; r++) padded[r] = new int[7];
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++) padded[r + 2][c + 2] = classic[r][c];
        Check(CountDistinctIslands(padded) == 3, "padded");

        var grids = new List<int[][]>();
        for (int i = 0; i < 40; i++) grids.Add(LcgGrid(8, 8));
        for (int i = 0; i < 20; i++) grids.Add(LcgGrid(6, 11));
        for (int i = 0; i < 20; i++) grids.Add(LcgGrid(4, 4));

        foreach (int[][] grid in grids)
        {
            var keys = CollectIslandShapeKeys(grid);
            var oracle = IslandShapesSortedTuples(grid);
            Check(keys.SetEquals(oracle), "cross-check");
            Check(CountDistinctIslands(grid) == oracle.Count, "count");
            foreach (var shape in CollectIslandShapes(grid))
            {
                int minR = shape.Min(key => int.Parse(key.Split(',')[0]));
                int minC = shape.Min(key => int.Parse(key.Split(',')[1]));
                Check(minR == 0 && minC == 0, "normalised");                 // keys are normalised ...
                Check(IsConnectedShape(shape), "connected");                 // ... and 4-connected
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
