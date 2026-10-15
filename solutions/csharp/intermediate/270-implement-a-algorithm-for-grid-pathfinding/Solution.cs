// Task 270 - Implement A* algorithm for grid pathfinding. (Tier: Intermediate, CTCI Ch 4)
using System;
using System.Collections.Generic;

public static class Solution
{
    private static readonly int[][] Dirs = { new[] { 1, 0 }, new[] { -1, 0 }, new[] { 0, 1 }, new[] { 0, -1 } };

    private static bool Free(int[][] grid, int r, int c) =>
        r >= 0 && r < grid.Length && c >= 0 && c < grid[0].Length && grid[r][c] == 0;

    public static int Manhattan(int[] a, int[] b) => Math.Abs(a[0] - b[0]) + Math.Abs(a[1] - b[1]);

    /// <summary>
    /// A* on a 4-neighbour grid with unit move cost and the Manhattan heuristic, which is
    /// admissible (one move changes the distance to the goal by at most 1) and consistent.
    /// grid[r][c] == 1 is a wall. Returns the minimum-move path including both endpoints,
    /// or null when no path exists (also when start or goal lies on a wall).
    /// Deterministic tie-break: priority (f, h, row-major cell id) means with equal f the
    /// cell closer to the goal and then the smaller index expands first.
    /// </summary>
    public static List<int[]>? Astar(int[][] grid, int[] start, int[] goal)
    {
        int rows = grid.Length, cols = rows > 0 ? grid[0].Length : 0;
        if (!Free(grid, start[0], start[1]) || !Free(grid, goal[0], goal[1])) return null;
        int sr = start[0], sc = start[1], gr = goal[0], gc = goal[1];
        if (sr == gr && sc == gc) return new List<int[]> { new[] { sr, sc } };
        var g = new int[rows, cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) g[r, c] = int.MaxValue;
        g[sr, sc] = 0;
        var closed = new bool[rows, cols];
        var parentR = new int[rows, cols];
        var parentC = new int[rows, cols];
        var pq = new PriorityQueue<int[], (int F, int H, int Id)>();
        int h0 = Manhattan(start, goal);
        pq.Enqueue(new[] { h0, h0, sr * cols + sc, sr, sc }, (h0, h0, sr * cols + sc));
        while (pq.Count > 0)
        {
            var cur = pq.Dequeue();
            int r = cur[3], c = cur[4];
            if (closed[r, c]) continue;
            closed[r, c] = true;
            if (r == gr && c == gc)
            {
                var path = new LinkedList<int[]>();
                int pr = r, pc = c;
                while (pr != sr || pc != sc)
                {
                    path.AddFirst(new[] { pr, pc });
                    int nr = parentR[pr, pc], nc = parentC[pr, pc];
                    pr = nr;
                    pc = nc;
                }
                path.AddFirst(new[] { sr, sc });
                return new List<int[]>(path);
            }
            foreach (var d in Dirs)
            {
                int nr = r + d[0], nc = c + d[1];
                if (!Free(grid, nr, nc) || closed[nr, nc]) continue;
                int tentative = g[r, c] + 1;
                if (tentative < g[nr, nc])
                {
                    g[nr, nc] = tentative;
                    parentR[nr, nc] = r;
                    parentC[nr, nc] = c;
                    int h = Math.Abs(nr - gr) + Math.Abs(nc - gc);
                    pq.Enqueue(new[] { tentative + h, h, nr * cols + nc, nr, nc }, (tentative + h, h, nr * cols + nc));
                }
            }
        }
        return null;
    }

    public static int? AstarCost(int[][] grid, int[] start, int[] goal)
    {
        var path = Astar(grid, start, goal);
        return path == null ? null : path.Count - 1;
    }

    /// <summary>Exact distance from every free cell to goal (-1 when unreachable/walled).</summary>
    public static int[][] BfsDistancesFrom(int[][] grid, int[] goal)
    {
        int rows = grid.Length, cols = rows > 0 ? grid[0].Length : 0;
        var dist = new int[rows][];
        for (int r = 0; r < rows; r++)
        {
            dist[r] = new int[cols];
            for (int c = 0; c < cols; c++) dist[r][c] = -1;
        }
        if (!Free(grid, goal[0], goal[1])) return dist;
        dist[goal[0]][goal[1]] = 0;
        var q = new Queue<int[]>();
        q.Enqueue(new[] { goal[0], goal[1] });
        while (q.Count > 0)
        {
            var cur = q.Dequeue();
            foreach (var d in Dirs)
            {
                int nr = cur[0] + d[0], nc = cur[1] + d[1];
                if (Free(grid, nr, nc) && dist[nr][nc] == -1)
                {
                    dist[nr][nc] = dist[cur[0]][cur[1]] + 1;
                    q.Enqueue(new[] { nr, nc });
                }
            }
        }
        return dist;
    }

    public static int BfsCost(int[][] grid, int[] start, int[] goal)
    {
        var dist = BfsDistancesFrom(grid, goal);
        if (dist.Length == 0 || dist[0].Length == 0) return -1;
        if (start[0] < 0 || start[0] >= dist.Length || start[1] < 0 || start[1] >= dist[0].Length) return -1;
        return dist[start[0]][start[1]];
    }

    public static bool ValidPath(int[][] grid, List<int[]>? path, int[] start, int[] goal)
    {
        if (path == null) return false;
        if (path[0][0] != start[0] || path[0][1] != start[1]) return false;
        var last = path[path.Count - 1];
        if (last[0] != goal[0] || last[1] != goal[1]) return false;
        foreach (var cell in path)
            if (!(cell[0] >= 0 && cell[0] < grid.Length && cell[1] >= 0 && cell[1] < grid[0].Length
                    && grid[cell[0]][cell[1]] == 0)) return false;
        for (int i = 1; i < path.Count; i++)
            if (Math.Abs(path[i][0] - path[i - 1][0]) + Math.Abs(path[i][1] - path[i - 1][1]) != 1) return false;
        return true;
    }

    public static void Main()
    {
        var serpentine = new[]
        {
            new[] { 0, 0, 0, 0, 0 },
            new[] { 0, 1, 1, 1, 0 },
            new[] { 0, 0, 0, 0, 0 },
            new[] { 0, 1, 1, 1, 0 },
            new[] { 0, 0, 0, 0, 0 },
        };
        var detour = new[]
        {
            new[] { 0, 0, 0, 0 },
            new[] { 1, 1, 1, 0 },
            new[] { 0, 0, 0, 0 },
        };
        var blocked = new[]
        {
            new[] { 0, 1, 0 },
            new[] { 0, 1, 0 },
            new[] { 1, 1, 0 },
        };
        var openGrid = new int[4][];
        for (int r = 0; r < 4; r++) openGrid[r] = new int[6];

        Check(AstarCost(serpentine, new[] { 0, 0 }, new[] { 4, 4 }) == 8, "serpentine = 8");
        Check(BfsCost(serpentine, new[] { 0, 0 }, new[] { 4, 4 }) == 8, "bfs serpentine = 8");
        Check(AstarCost(detour, new[] { 0, 0 }, new[] { 2, 0 }) == 8, "detour = 8 (vs Manhattan 2)");
        Check(BfsCost(detour, new[] { 0, 0 }, new[] { 2, 0 }) == 8, "bfs detour = 8");
        Check(AstarCost(detour, new[] { 2, 0 }, new[] { 0, 0 }) == 8, "detour reversed = 8");
        Check(AstarCost(openGrid, new[] { 0, 0 }, new[] { 3, 5 }) == 8, "open grid = Manhattan 8");
        Check(AstarCost(blocked, new[] { 0, 0 }, new[] { 0, 2 }) == null, "no path -> null");
        Check(BfsCost(blocked, new[] { 0, 0 }, new[] { 0, 2 }) == -1, "bfs no path -> -1");

        var self = Astar(serpentine, new[] { 2, 2 }, new[] { 2, 2 });
        Check(self != null && self.Count == 1 && self[0][0] == 2 && self[0][1] == 2, "start == goal -> single cell");
        Check(Astar(serpentine, new[] { 1, 1 }, new[] { 0, 0 }) == null, "start on a wall");
        Check(Astar(serpentine, new[] { 0, 0 }, new[] { 1, 1 }) == null, "goal on a wall");

        var cases = new (int[][] Grid, int[] Start, int[] Goal)[]
        {
            (serpentine, new[] { 0, 0 }, new[] { 4, 4 }),
            (detour, new[] { 0, 0 }, new[] { 2, 0 }),
            (detour, new[] { 2, 0 }, new[] { 0, 0 }),
            (openGrid, new[] { 0, 0 }, new[] { 3, 5 }),
            (serpentine, new[] { 2, 2 }, new[] { 2, 2 }),
        };
        foreach (var (grid, start, goal) in cases)
        {
            Check(ValidPath(grid, Astar(grid, start, goal), start, goal), "valid path");
            if (grid == openGrid)                   // on an open grid Manhattan is exact
            {
                var dist = BfsDistancesFrom(openGrid, goal);
                for (int r = 0; r < openGrid.Length; r++)
                    for (int c = 0; c < openGrid[0].Length; c++)
                        Check(Math.Abs(r - goal[0]) + Math.Abs(c - goal[1]) == dist[r][c], "heuristic exact");
            }
        }

        foreach (var (grid, goal) in new[] { (serpentine, new[] { 4, 4 }), (detour, new[] { 2, 0 }), (blocked, new[] { 0, 2 }) })
        {
            var dist = BfsDistancesFrom(grid, goal);
            for (int r = 0; r < grid.Length; r++)   // admissibility on every reachable cell
                for (int c = 0; c < grid[0].Length; c++)
                    if (dist[r][c] >= 0)
                        Check(Math.Abs(r - goal[0]) + Math.Abs(c - goal[1]) <= dist[r][c], "admissible");
        }

        var rng = new Random(270);
        for (int test = 0; test < 300; test++)      // A* cost == BFS cost + valid path
        {
            int rows = 2 + rng.Next(5), cols = 2 + rng.Next(5);
            var grid = new int[rows][];
            for (int r = 0; r < rows; r++)
            {
                grid[r] = new int[cols];
                for (int c = 0; c < cols; c++) grid[r][c] = rng.NextDouble() < 0.3 ? 1 : 0;
            }
            var start = new[] { rng.Next(rows), rng.Next(cols) };
            var goal = new[] { rng.Next(rows), rng.Next(cols) };
            int? got = AstarCost(grid, start, goal);
            int want = BfsCost(grid, start, goal);
            Check((got ?? -1) == want, "random cost agreement");
            if (got != null)
            {
                Check(ValidPath(grid, Astar(grid, start, goal), start, goal), "random path validity");
                if (want > 0)
                {
                    var dist = BfsDistancesFrom(grid, goal);
                    Check(Math.Abs(start[0] - goal[0]) + Math.Abs(start[1] - goal[1]) <= dist[start[0]][start[1]],
                        "random admissibility");
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
