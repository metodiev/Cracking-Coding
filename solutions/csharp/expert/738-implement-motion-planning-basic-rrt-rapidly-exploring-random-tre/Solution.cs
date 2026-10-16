// Task 738 - Implement motion planning basic RRT (rapidly exploring random tree). (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// Rapidly-exploring random trees in the plane. The configuration space is the axis-aligned box
// [0,W]x[0,H] minus a list of axis-aligned rectangular obstacles (a small, exactly testable world). The tree
// starts at the start configuration; each iteration samples a configuration (uniform in the box, or the goal
// itself with 10% probability = the classic goal bias), finds the nearest tree node by a linear scan, steers a
// fixed step length towards the sample, and inserts the steered configuration only when the whole step is
// collision free. The planner stops as soon as the newest node can reach the goal in one more step.
// Invariants verified: every tree node lies in the free space, every parent-child edge is collision free when
// re-tested with an *independent* oracle (400 sample points along the edge, a completely different mechanism
// from the Liang-Barsky segment/rectangle clipping used for steering), each node is at most `step` away from
// its parent, and the returned path is a genuine parent chain that starts at the start configuration and whose
// last leg reaches the goal. Reachability claims are cross-checked against a grid flood fill, so "no path
// found" is a fact about the world and not merely about the planner's luck.
// Simplifications of this outline: a linear nearest-neighbour scan instead of a kd-tree, no path smoothing, no
// rewiring (this is the plain RRT, not RRT*), a single fixed step size, and double arithmetic with a
// conservative segment/rectangle test that treats grazing a corner as a collision. All sampling comes from a
// hand-written splitmix64 stream, so a run is byte-for-byte reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix
{
    internal static ulong Next(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);

    /// <summary>Uniform double in [0, 1).</summary>
    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);
}

internal sealed class Rect
{
    internal readonly double X0, Y0, X1, Y1;

    internal Rect(double x0, double y0, double x1, double y1)
    {
        X0 = Math.Min(x0, x1); X1 = Math.Max(x0, x1);
        Y0 = Math.Min(y0, y1); Y1 = Math.Max(y0, y1);
    }

    internal bool Contains(double x, double y) => x >= X0 && x <= X1 && y >= Y0 && y <= Y1;

    internal bool StrictlyInside(double x, double y) => x > X0 && x < X1 && y > Y0 && y < Y1;
}

internal sealed class World
{
    private readonly List<Rect> _obstacles;

    internal World(double width, double height, List<Rect> obstacles)
    {
        Width = width;
        Height = height;
        _obstacles = obstacles;
    }

    internal double Width { get; }
    internal double Height { get; }
    internal IReadOnlyList<Rect> Obstacles => _obstacles;

    internal bool Free(double x, double y)
    {
        if (x < 0 || x > Width || y < 0 || y > Height) return false;
        foreach (Rect r in _obstacles) if (r.Contains(x, y)) return false;
        return true;
    }

    internal bool StrictlyFree(double x, double y)
    {
        if (x < 0 || x > Width || y < 0 || y > Height) return false;
        foreach (Rect r in _obstacles) if (r.Contains(x, y)) return false;
        return true;
    }

    /// <summary>Liang-Barsky segment/rectangle clipping; touching a corner counts as a hit (conservative).</summary>
    internal bool SegmentFree(double x0, double y0, double x1, double y1)
    {
        if (!Free(x0, y0) || !Free(x1, y1)) return false;
        foreach (Rect r in _obstacles) if (Hits(r, x0, y0, x1, y1)) return false;
        return true;
    }

    private static bool Hits(Rect r, double x0, double y0, double x1, double y1)
    {
        double t0 = 0, t1 = 1;
        double dx = x1 - x0, dy = y1 - y0;
        if (!Clip(-dx, x0 - r.X0, ref t0, ref t1)) return false;
        if (!Clip(dx, r.X1 - x0, ref t0, ref t1)) return false;
        if (!Clip(-dy, y0 - r.Y0, ref t0, ref t1)) return false;
        if (!Clip(dy, r.Y1 - y0, ref t0, ref t1)) return false;
        return t0 <= t1;
    }

    private static bool Clip(double p, double q, ref double t0, ref double t1)
    {
        if (p == 0) return q >= 0;
        double t = q / p;
        if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
        else { if (t < t0) return false; if (t < t1) t1 = t; }
        return true;
    }

    /// <summary>Independent oracle: sample the segment densely and reject anything strictly inside an obstacle.</summary>
    internal bool SampledSegmentFree(double x0, double y0, double x1, double y1, int samples)
    {
        for (int i = 0; i <= samples; i++)
        {
            double t = (double)i / samples;
            double x = x0 + (x1 - x0) * t, y = y0 + (y1 - y0) * t;
            if (!StrictlyFree(x, y)) return false;
        }
        return true;
    }
}

internal sealed class Rrt
{
    private readonly World _world;
    private readonly double _step;
    private readonly List<double> _x = new List<double>();
    private readonly List<double> _y = new List<double>();
    private readonly List<int> _parent = new List<int>();
    private ulong _state;

    internal long Attempts, Rejected, Accepted, Iterations;
    internal int GoalIndex = -1;

    internal Rrt(World world, double step, ulong seed, double startX, double startY)
    {
        if (!world.Free(startX, startY)) throw new ArgumentException("the start configuration is not free", nameof(startX));
        _world = world;
        _step = step;
        _state = seed;
        _x.Add(startX);
        _y.Add(startY);
        _parent.Add(-1);
    }

    internal int NodeCount => _x.Count;
    internal double StepLength => _step;
    internal double X(int i) => _x[i];
    internal double Y(int i) => _y[i];
    internal int Parent(int i) => _parent[i];

    private int Nearest(double px, double py)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int i = 0; i < _x.Count; i++)
        {
            double dx = _x[i] - px, dy = _y[i] - py;
            double d = dx * dx + dy * dy;
            if (d < bestDistance) { bestDistance = d; best = i; }
        }
        return best;
    }

    private bool Extend(double goalX, double goalY)
    {
        Attempts++;
        double sampleX, sampleY;
        if (SplitMix.NextInt(ref _state, 10) == 0) { sampleX = goalX; sampleY = goalY; }
        else
        {
            sampleX = SplitMix.NextUnit(ref _state) * _world.Width;
            sampleY = SplitMix.NextUnit(ref _state) * _world.Height;
        }
        int near = Nearest(sampleX, sampleY);
        double dx = sampleX - _x[near], dy = sampleY - _y[near];
        double distance = Math.Sqrt(dx * dx + dy * dy);
        if (distance < 1e-12) { Rejected++; return false; }
        double scale = Math.Min(1.0, _step / distance);
        double newX = _x[near] + dx * scale, newY = _y[near] + dy * scale;
        if (!_world.SegmentFree(_x[near], _y[near], newX, newY)) { Rejected++; return false; }
        _x.Add(newX);
        _y.Add(newY);
        _parent.Add(near);
        Accepted++;
        return true;
    }

    internal bool Plan(double goalX, double goalY, int maxIterations)
    {
        if (!_world.Free(goalX, goalY)) throw new ArgumentException("the goal configuration is not free", nameof(goalX));
        if (_x.Count == 1 && Distance(0, goalX, goalY) < 1e-12) { GoalIndex = 0; return true; }
        for (int i = 0; i < maxIterations; i++)
        {
            Iterations++;
            Extend(goalX, goalY);
            int last = _x.Count - 1;
            if (Distance(last, goalX, goalY) <= _step && _world.SegmentFree(_x[last], _y[last], goalX, goalY))
            {
                _x.Add(goalX);
                _y.Add(goalY);
                _parent.Add(last);
                GoalIndex = _x.Count - 1;
                return true;
            }
        }
        return false;
    }

    private double Distance(int index, double px, double py)
    {
        double dx = _x[index] - px, dy = _y[index] - py;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>The goal-to-start parent chain, reversed so it starts at the tree root.</summary>
    internal List<int> Path()
    {
        var path = new List<int>();
        if (GoalIndex < 0) return path;
        for (int i = GoalIndex; i >= 0; i = _parent[i]) path.Add(i);
        path.Reverse();
        return path;
    }

    internal double PathLength()
    {
        List<int> path = Path();
        double total = 0;
        for (int i = 1; i < path.Count; i++)
        {
            double dx = _x[path[i]] - _x[path[i - 1]], dy = _y[path[i]] - _y[path[i - 1]];
            total += Math.Sqrt(dx * dx + dy * dy);
        }
        return total;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static World Corridor() => new World(100, 100, new List<Rect>
    {
        new Rect(20, 0, 40, 70),
        new Rect(60, 30, 80, 100)
    });

    private static World Open() => new World(100, 100, new List<Rect>());

    private static World Walled() => new World(100, 100, new List<Rect>
    {
        // A closed ring around (80, 80): four rectangles leaving the pocket (72,72)-(88,88) sealed off.
        new Rect(68, 68, 92, 71),
        new Rect(68, 89, 92, 92),
        new Rect(68, 71, 71, 89),
        new Rect(89, 71, 92, 89)
    });

    /// <summary>Independent reachability oracle: 8-connected flood fill over a 1x1 grid with corner checks.</summary>
    private static bool GridReachable(World world, double sx, double sy, double gx, double gy)
    {
        const int Size = 101;
        const double Cell = 1.0;
        bool[,] seen = new bool[Size, Size];
        var queue = new Queue<(int X, int Y)>();
        int startX = (int)Math.Round(sx / Cell), startY = (int)Math.Round(sy / Cell);
        int goalX = (int)Math.Round(gx / Cell), goalY = (int)Math.Round(gy / Cell);
        if (!world.Free(startX * Cell, startY * Cell) || !world.Free(goalX * Cell, goalY * Cell)) return false;
        seen[startX, startY] = true;
        queue.Enqueue((startX, startY));
        int[] dxs = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dys = { -1, -1, -1, 0, 0, 1, 1, 1 };
        while (queue.Count > 0)
        {
            var (cx, cy) = queue.Dequeue();
            if (cx == goalX && cy == goalY) return true;
            for (int k = 0; k < 8; k++)
            {
                int nx = cx + dxs[k], ny = cy + dys[k];
                if (nx < 0 || ny < 0 || nx >= Size || ny >= Size || seen[nx, ny]) continue;
                if (!world.Free(nx * Cell, ny * Cell)) continue;
                if (dxs[k] != 0 && dys[k] != 0
                    && (!world.Free((cx + dxs[k]) * Cell, cy * Cell) || !world.Free(cx * Cell, (cy + dys[k]) * Cell)))
                    continue;   // a diagonal step may not squeeze past a blocked corner
                seen[nx, ny] = true;
                queue.Enqueue((nx, ny));
            }
        }
        return false;
    }

    public static void Main()
    {
        // ---- boundary configurations ----------------------------------------------------------------
        var open = Open();
        var trivial = new Rrt(open, 5, 0x7381UL, 10, 10);
        Check(trivial.Plan(10, 10, 50) && trivial.Path().Count == 1 && trivial.PathLength() == 0,
            "start equal to goal yields a one-vertex path of length zero");

        bool threw = false;
        try { new Rrt(Corridor(), 5, 1, 30, 30); }
        catch (ArgumentException) { threw = true; }
        Check(threw, "a start configuration inside an obstacle is rejected instead of planned from");

        // ---- an empty world: one step towards the goal is enough ------------------------------------
        var easy = new Rrt(open, 5, 0x7382UL, 10, 10);
        double direct = Math.Sqrt(80 * 80 + 80 * 80);
        Check(easy.Plan(90, 90, 400) && easy.PathLength() >= direct - 1e-9
            && easy.PathLength() <= 1.5 * direct && easy.NodeCount <= 200,
            "in an empty world the plan stays within 1.5x of the straight line (" + Math.Round(easy.PathLength(), 3)
            + " vs " + Math.Round(direct, 3) + ") using " + easy.NodeCount + " nodes");

        // ---- the corridor instance -------------------------------------------------------------------
        var world = Corridor();
        var rrt = new Rrt(world, 4, 0x7383UL, 5, 5);
        bool solved = rrt.Plan(95, 95, 4000);
        Check(solved, "the two-obstacle corridor instance is solved within 4000 iterations");
        Check(GridReachable(world, 5, 5, 95, 95), "the independent flood fill agrees that the goal is reachable");

        List<int> path = rrt.Path();
        Check(path[0] == 0 && rrt.GoalIndex == path[path.Count - 1] && path.Count >= 2,
            "the path is a parent chain from the tree root to the goal node");
        bool chainOk = true;
        for (int i = 1; i < path.Count; i++) if (rrt.Parent(path[i]) != path[i - 1]) chainOk = false;
        Check(chainOk, "every path vertex is the child of the previous one");

        bool nodesFree = true, edgesFree = true, stepBounded = true;
        string? firstFailure = null;
        for (int i = 0; i < rrt.NodeCount; i++)
        {
            if (!world.Free(rrt.X(i), rrt.Y(i)))
            {
                nodesFree = false;
                firstFailure ??= "node " + i + " is not free";
            }
            if (i == 0) continue;
            int parent = rrt.Parent(i);
            double dx = rrt.X(i) - rrt.X(parent), dy = rrt.Y(i) - rrt.Y(parent);
            if (Math.Sqrt(dx * dx + dy * dy) > rrt.StepLength + 1e-9)
            {
                stepBounded = false;
                firstFailure ??= "edge " + parent + "->" + i + " is longer than one step";
            }
            if (!world.SampledSegmentFree(rrt.X(parent), rrt.Y(parent), rrt.X(i), rrt.Y(i), 400))
            {
                edgesFree = false;
                firstFailure ??= "edge " + parent + "->" + i + " crosses an obstacle";
            }
        }
        Check(nodesFree, "every tree node lies in the free configuration space: " + (firstFailure ?? "ok"));
        Check(edgesFree, "all " + (rrt.NodeCount - 1) + " tree edges are collision free under the sampling oracle");
        Check(stepBounded, "every tree edge is at most one step long");
        Check(world.SampledSegmentFree(rrt.X(rrt.Parent(rrt.GoalIndex)), rrt.Y(rrt.Parent(rrt.GoalIndex)), 95, 95, 400),
            "the final leg from the last tree node to the goal is collision free");

        double straight = Math.Sqrt(90 * 90 + 90 * 90);
        Check(rrt.PathLength() >= straight - 1e-9,
            "the planned path is never shorter than the straight-line distance (" + Math.Round(rrt.PathLength(), 3) + ")");
        Check(rrt.NodeCount == rrt.Accepted + 2 && rrt.Attempts == rrt.Iterations && rrt.Rejected > 0,
            "node bookkeeping is exact: " + rrt.NodeCount + " nodes = start + " + rrt.Accepted
            + " accepted steps + goal, " + rrt.Rejected + " rejected extensions");

        // ---- determinism ------------------------------------------------------------------------------
        var again = new Rrt(world, 4, 0x7383UL, 5, 5);
        again.Plan(95, 95, 4000);
        Check(again.NodeCount == rrt.NodeCount && Math.Abs(again.PathLength() - rrt.PathLength()) < 1e-12,
            "the same seed reproduces the same tree and the same path length");

        // ---- an enclosed goal: the planner must report failure and the oracle agree ---------------------
        var walled = Walled();
        var stuck = new Rrt(walled, 4, 0x7384UL, 5, 5);
        bool found = stuck.Plan(80, 80, 3000);
        Check(!found && stuck.Path().Count == 0, "the goal enclosed by an obstacle ring is not reached in 3000 iterations");
        Check(!GridReachable(walled, 5, 5, 80, 80),
            "the independent flood fill confirms the enclosed goal is genuinely unreachable");
        Check(GridReachable(new World(100, 100, new List<Rect> { new Rect(0, 40, 60, 60) }), 5, 5, 95, 95),
            "the same oracle does report reachability when a gap exists");

        Console.WriteLine("iterations=" + rrt.Iterations + " nodes=" + rrt.NodeCount
            + " accepted=" + rrt.Accepted + " rejected=" + rrt.Rejected
            + " pathVertices=" + rrt.Path().Count + " pathLength=" + Math.Round(rrt.PathLength(), 4)
            + " straightLine=" + Math.Round(straight, 4));
        Console.WriteLine("PASS");
    }
}
