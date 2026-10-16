// Task 922 - Implement multi-robot coordination with conflict resolution and task allocation. (Tier: Master, Beyond — Graphics, Geometry & Robotics)
//
// Warehouse robots on a 6x6 grid with static blocked cells. Task allocation is exact: the cost matrix holds grid
// shortest-path lengths (unreachable pairs are priced out), and the assignment minimises the total cost by
// enumerating the permutations. Coordination is a compact conflict-based search (CBS): each robot is planned with a
// time-expanded A* whose state is (cell, time), whose actions are the four neighbours plus waiting, whose step cost
// is 1 and whose heuristic is the grid Manhattan distance; cells and edges already reserved by higher-priority
// robots are forbidden. The first conflict of the joint plan becomes two child nodes, each forbidding one of the two
// robots from occupying the conflicting cell at that time; nodes are expanded in order until a conflict-free joint
// plan appears, and every path is padded by waiting at the goal so all robots share one timeline.
//
// Simplifications: static obstacles, unit-speed holonomic robots on a shared clock, cell occupancy as the whole
// collision model (no radii, no priority deadlock recovery beyond CBS branching), small instances, no replanning
// after the initial plan. Verified: allocation equals an independently enumerated optimum and is a permutation, the
// joint plan has no vertex or swap conflicts under independent replay, each path is adjacent-step valid between its
// own start and goal, the uncoordinated shortest paths are detected as conflicting, a crafted head-on swap is
// reported as an edge conflict, a robot starting on its goal stays put, a solo robot gets the BFS shortest path,
// the makespan stays within the individual optimum plus slack, and the plan is reproducible.
using System;
using System.Collections.Generic;

internal sealed class Warehouse
{
    internal readonly int W, H;
    internal readonly bool[] Blocked;
    internal Warehouse(int w, int h) { W = w; H = h; Blocked = new bool[w * h]; }
    internal void Block(int c, int r) { Blocked[r * W + c] = true; }
    internal bool Free(int cell) => cell >= 0 && cell < W * H && !Blocked[cell];
    internal int Manhattan(int a, int b) => Math.Abs(a % W - b % W) + Math.Abs(a / W - b / W);
    internal List<int> Neighbours(int cell)
    {
        var list = new List<int>();
        int c = cell % W, r = cell / W;
        if (c > 0 && Free(cell - 1)) list.Add(cell - 1);
        if (c < W - 1 && Free(cell + 1)) list.Add(cell + 1);
        if (r > 0 && Free(cell - W)) list.Add(cell - W);
        if (r < H - 1 && Free(cell + W)) list.Add(cell + W);
        return list;
    }
    /** Breadth-first shortest path as a cell sequence; null when unreachable. */
    internal List<int> BfsPath(int from, int to)
    {
        var parent = new int[W * H];
        for (int i = 0; i < parent.Length; i++) parent[i] = -2;
        parent[from] = -1;
        var queue = new Queue<int>(); queue.Enqueue(from);
        while (queue.Count > 0)
        {
            int cell = queue.Dequeue();
            if (cell == to) break;
            foreach (int next in Neighbours(cell))
                if (parent[next] == -2) { parent[next] = cell; queue.Enqueue(next); }
        }
        if (parent[to] == -2) return null;
        var path = new List<int>();
        for (int cell = to; cell >= 0; cell = parent[cell]) path.Add(cell);
        path.Reverse();
        return path;
    }
    internal int ShortestPath(int from, int to)
    {
        List<int> path = BfsPath(from, to);
        return path == null ? -1 : path.Count - 1;
    }
}

/** Conflict-based search over time-expanded A* plans. */
internal sealed class Coordinator
{
    internal readonly Warehouse Grid;
    internal readonly int[] Start, Goal;
    internal readonly int Horizon = 26;
    internal int Expanded;
    private readonly int _cells;
    internal Coordinator(Warehouse grid, int[] start, int[] goal)
    { Grid = grid; Start = start; Goal = goal; _cells = grid.W * grid.H; }
    /** A* over (cell, time) honouring reserved cells/edges and the node's extra constraints. */
    private List<int> PlanOne(int robot, HashSet<long> cells, HashSet<long> edges, List<int[]> constraints)
    {
        int states = _cells * Horizon;
        var g = new double[states]; var parent = new int[states];
        for (int s = 0; s < states; s++) { g[s] = double.PositiveInfinity; parent[s] = -1; }
        var open = new List<int>();
        int root = Start[robot];
        g[root] = 0; open.Add(root);
        while (open.Count > 0)
        {
            int bestAt = 0;
            for (int i = 1; i < open.Count; i++)
                if (g[open[i]] + Grid.Manhattan(open[i] % _cells, Goal[robot]) < g[open[bestAt]] + Grid.Manhattan(open[bestAt] % _cells, Goal[robot])) bestAt = i;
            int state = open[bestAt]; open.RemoveAt(bestAt);
            int cell = state % _cells, time = state / _cells;
            if (cell == Goal[robot])
            {
                var path = new List<int>();
                for (int s = state; s >= 0; s = parent[s]) path.Add(s % _cells);
                path.Reverse();
                return path;
            }
            if (time + 1 >= Horizon) continue;
            var options = Grid.Neighbours(cell); options.Add(cell);
            foreach (int next in options)
            {
                long cellKey = next * 100L + (time + 1), edgeKey = ((long)cell * _cells + next) * 100L + time;
                if (cells.Contains(cellKey) || edges.Contains(edgeKey)) continue;
                bool forbidden = false;
                foreach (int[] c in constraints) if (c[0] == robot && c[1] == next && c[2] == time + 1) forbidden = true;
                if (forbidden) continue;
                int nextState = (time + 1) * _cells + next;
                if (g[state] + 1.0 < g[nextState]) { g[nextState] = g[state] + 1.0; parent[nextState] = state; open.Add(nextState); }
            }
        }
        return null;
    }
    /** Plans every robot for one CBS node; reports the first conflict when the node is not yet conflict free. */
    internal bool Attempt(List<int[]> constraints, out List<List<int>> plans, out int[] conflict)
    {
        var cells = new HashSet<long>(); var edges = new HashSet<long>();
        plans = new List<List<int>>(); conflict = null;
        for (int r = 0; r < Start.Length; r++)
        {
            List<int> path = PlanOne(r, cells, edges, constraints);
            if (path == null) { plans = null; return false; }
            plans.Add(path);
            for (int t = 0; t < path.Count; t++)
            {
                cells.Add(path[t] * 100L + t);
                if (t + 1 < path.Count) edges.Add(((long)path[t] * _cells + path[t + 1]) * 100L + t);
            }
        }
        conflict = FirstConflict(plans);
        return conflict == null;
    }
    /** First conflict as {robotA, robotB, cellForA, cellForB, time}: a vertex clash or a head-on edge swap. */
    internal static int[] FirstConflict(List<List<int>> plans)
    {
        int span = 0;
        foreach (List<int> p in plans) span = Math.Max(span, p.Count);
        for (int t = 0; t < span; t++)
            for (int a = 0; a < plans.Count; a++)
                for (int b = a + 1; b < plans.Count; b++)
                {
                    int ca = plans[a][Math.Min(t, plans[a].Count - 1)], cb = plans[b][Math.Min(t, plans[b].Count - 1)];
                    if (ca == cb) return new[] { a, b, ca, cb, t };
                    if (t + 1 < span)
                    {
                        int na = plans[a][Math.Min(t + 1, plans[a].Count - 1)], nb = plans[b][Math.Min(t + 1, plans[b].Count - 1)];
                        if (ca == nb && cb == na && ca != cb) return new[] { a, b, na, nb, t + 1 };   // head-on swap
                    }
                }
        return null;
    }
    /** CBS: expand the conflict tree breadth first, forbidding one of the two cells at the clash time per child. */
    internal List<List<int>> Solve(int nodeLimit)
    {
        var queue = new Queue<List<int[]>>();
        queue.Enqueue(new List<int[]>());
        while (queue.Count > 0 && Expanded < nodeLimit)
        {
            List<int[]> node = queue.Dequeue(); Expanded++;
            if (Attempt(node, out List<List<int>> plans, out int[] conflict)) return plans;
            if (plans == null || conflict == null || queue.Count > 2 * nodeLimit) continue;
            queue.Enqueue(new List<int[]>(node) { new[] { conflict[0], conflict[2], conflict[4] } });
            queue.Enqueue(new List<int[]>(node) { new[] { conflict[1], conflict[3], conflict[4] } });
        }
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static Warehouse Build()
    {
        var grid = new Warehouse(6, 6);
        foreach (int cell in new[] { 8, 14, 20 }) grid.Block(cell % 6, cell / 6);   // 3-cell wall in column 2
        return grid;
    }
    /** Exact assignment by enumerating permutations (recursive swap walk). */
    private static int[] Assign(int[] cost, int n)
    {
        var order = new int[n]; for (int i = 0; i < n; i++) order[i] = i;
        var best = new int[n]; double bestCost = double.PositiveInfinity;
        Action<int> walk = null;
        walk = depth =>
        {
            if (depth == n)
            {
                double total = 0;
                for (int r = 0; r < n; r++) total += cost[r * n + order[r]];
                if (total < bestCost) { bestCost = total; Array.Copy(order, best, n); }
                return;
            }
            for (int i = depth; i < n; i++)
            {
                int swap = order[depth]; order[depth] = order[i]; order[i] = swap;
                walk(depth + 1);
                swap = order[depth]; order[depth] = order[i]; order[i] = swap;
            }
        };
        walk(0);
        return best;
    }
    /** Independent oracle for the three-robot assignment: nested loops with an injectivity filter. */
    private static double OracleAssign(int[] c, int n)
    {
        double best = double.PositiveInfinity;
        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) for (int d = 0; d < n; d++)
            if (a != b && b != d && a != d) best = Math.Min(best, c[a] + c[n + b] + c[2 * n + d]);
        return best;
    }
    /** Independent replay: [vertex conflicts, swap conflicts, invalid paths, makespan]. */
    private static int[] Replay(Warehouse grid, int[] start, int[] goal, List<List<int>> plans)
    {
        int vertex = 0, swap = 0, invalid = 0, span = 0;
        foreach (List<int> p in plans) span = Math.Max(span, p.Count);
        for (int r = 0; r < plans.Count; r++)
        {
            if (plans[r][0] != start[r] || plans[r][plans[r].Count - 1] != goal[r]) invalid++;
            for (int t = 0; t + 1 < plans[r].Count; t++)
                if (plans[r][t] != plans[r][t + 1] && !grid.Neighbours(plans[r][t]).Contains(plans[r][t + 1])) invalid++;
        }
        for (int t = 0; t < span; t++)
            for (int a = 0; a < plans.Count; a++)
                for (int b = a + 1; b < plans.Count; b++)
                {
                    int ca = plans[a][Math.Min(t, plans[a].Count - 1)], cb = plans[b][Math.Min(t, plans[b].Count - 1)];
                    if (ca == cb) vertex++;
                    if (t + 1 < span)
                    {
                        int na = plans[a][Math.Min(t + 1, plans[a].Count - 1)], nb = plans[b][Math.Min(t + 1, plans[b].Count - 1)];
                        if (ca == nb && cb == na && ca != cb) swap++;
                    }
                }
        return new[] { vertex, swap, invalid, span };
    }
    private static List<int> Pad(List<int> path, int goal, int length)
    {
        var padded = new List<int>(path);
        while (padded.Count < length) padded.Add(goal);
        return padded;
    }
    public static void Main()
    {
        Warehouse grid = Build();
        int[] starts = { 0, 5, 30 };
        int[] tasks = { 35, 12, 25 };

        // 1. task allocation against the independent enumeration oracle
        var cost = new int[9];
        for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++)
        {
            int length = grid.ShortestPath(starts[r], tasks[k]);
            cost[r * 3 + k] = length < 0 ? 1000 : length;
        }
        int[] assignment = Assign(cost, 3);
        double total = 0;
        for (int r = 0; r < 3; r++) total += cost[r * 3 + assignment[r]];
        Check(Math.Abs(total - OracleAssign(cost, 3)) < 1e-12, "assignment cost " + total + " equals the enumerated optimum " + OracleAssign(cost, 3));
        Check(assignment[0] != assignment[1] && assignment[1] != assignment[2] && assignment[0] != assignment[2], "the assignment is a permutation of the tasks");
        Check(total <= cost[0] + cost[4] + cost[8], "the optimum is no worse than the identity assignment");

        int[] goals = { tasks[assignment[0]], tasks[assignment[1]], tasks[assignment[2]] };
        // 2. uncoordinated shortest paths are valid on their own but not necessarily conflict free
        var naive = new List<List<int>>();
        for (int r = 0; r < 3; r++) naive.Add(Pad(grid.BfsPath(starts[r], goals[r]), goals[r], 30));
        int[] naiveReport = Replay(grid, starts, goals, naive);
        Check(naiveReport[2] == 0, "each uncoordinated shortest path is valid and reaches its assigned goal");
        // 3. corridor swap: the two naive paths collide head-on, CBS-lite resolves it with a detour
        var corridor = new Warehouse(3, 2);
        var swapCase = new Coordinator(corridor, new[] { 0, 1 }, new[] { 1, 0 });
        List<List<int>> swapPlan = swapCase.Solve(nodeLimit: 20);
        Check(swapPlan != null, "CBS-lite resolves the head-on swap (expanded " + swapCase.Expanded + " nodes)");
        int[] swapReport = Replay(corridor, new[] { 0, 1 }, new[] { 1, 0 }, swapPlan);
        Check(swapReport[0] == 0 && swapReport[1] == 0 && swapReport[2] == 0, "the resolved plan passes the independent replay");
        Check(swapReport[3] > 2, "resolution costs the detouring robot extra time (makespan " + swapReport[3] + " > 2)");
        var naiveSwap = new List<List<int>> { Pad(corridor.BfsPath(0, 1), 1, 6), Pad(corridor.BfsPath(1, 0), 0, 6) };
        int[] naiveSwapReport = Replay(corridor, new[] { 0, 1 }, new[] { 1, 0 }, naiveSwap);
        Check(naiveSwapReport[1] >= 1, "the uncoordinated corridor plan is detected as a head-on swap (" + naiveSwapReport[1] + " swaps)");

        // 4. the main three-robot instance under CBS-lite
        var coordinator = new Coordinator(grid, starts, goals);
        List<List<int>> plan = coordinator.Solve(nodeLimit: 40);
        Check(plan != null, "CBS-lite finds a conflict-free joint plan (expanded " + coordinator.Expanded + " nodes)");
        int[] report = Replay(grid, starts, goals, plan);
        Check(report[0] == 0 && report[1] == 0, "independent replay finds no vertex or swap conflict");
        Check(report[2] == 0, "every path is adjacent-step valid and joins its own start to its goal");
        Check(Coordinator.FirstConflict(plan) == null, "the coordinator's conflict test agrees with the replay");

        // 5. the makespan bound and a crafted head-on swap through the conflict predicate
        int longest = 0;
        for (int r = 0; r < 3; r++) longest = Math.Max(longest, grid.ShortestPath(starts[r], goals[r]));
        Check(report[3] <= longest + 4, "makespan " + report[3] + " stays within the longest individual path " + longest + " + 4");
        var swapPlans = new List<List<int>> { new List<int> { 0, 1, 1 }, new List<int> { 1, 0, 0 } };
        int[] swapConflict = Coordinator.FirstConflict(swapPlans);
        Check(swapConflict != null && swapConflict[0] == 0 && swapConflict[2] == 1 && swapConflict[4] == 1, "a crafted head-on swap is reported as an edge conflict at cell 1, time 1");

        // 4. boundary cases: a robot already on its goal, and a lone robot on free grid
        var idle = new Coordinator(grid, new[] { 4, 34 }, new[] { 4, 33 });
        List<List<int>> idlePlan = idle.Solve(nodeLimit: 20);
        Check(idlePlan != null && idlePlan[0].TrueForAll(cell => cell == 4), "a robot starting on its goal keeps a single-cell plan");
        var solo = new Coordinator(grid, new[] { 0 }, new[] { 35 });
        List<List<int>> soloPlan = solo.Solve(nodeLimit: 20);
        Check(soloPlan != null && soloPlan[0].Count - 1 == grid.ShortestPath(0, 35), "a lone robot gets the BFS shortest path (" + (soloPlan[0].Count - 1) + " steps)");

        // 5. reproducibility
        var repeat = new Coordinator(grid, starts, goals);
        List<List<int>> again = repeat.Solve(nodeLimit: 40);
        bool identical = again != null && again.Count == plan.Count;
        for (int r = 0; r < plan.Count && identical; r++) identical = string.Join(",", again[r]) == string.Join(",", plan[r]);
        Check(identical, "the coordinated plan is bit-for-bit reproducible");

        Console.WriteLine("allocation: " + string.Join(",", assignment) + " total " + total + " (oracle " + OracleAssign(cost, 3) + "); naive plan: " + naiveReport[0] + " vertex / " + naiveReport[1] + " swap conflicts");
        Console.WriteLine("coordination: CBS-lite expanded " + coordinator.Expanded + " nodes to makespan " + report[3] + " (individual optimum " + longest + "), 0 conflicts; idle robot plan " + idlePlan[0].Count + " cell");
        Console.WriteLine("PASS");
    }
}
