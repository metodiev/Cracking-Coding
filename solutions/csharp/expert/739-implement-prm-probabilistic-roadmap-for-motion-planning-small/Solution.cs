// Task 739 - Implement PRM (probabilistic roadmap) for motion planning small. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// A probabilistic roadmap for the plane with axis-aligned rectangular obstacles. The roadmap is built once
// (deterministically, from a splitmix64 stream): milestones are sampled uniformly in the workspace and kept
// only when they are collision free, then every milestone is joined to its k nearest neighbours by a straight
// segment whenever that segment is collision free; the start and goal are inserted as extra nodes and joined
// the same way. Queries are cheap afterwards: Dijkstra over the roadmap returns the shortest path in the graph.
// Invariants verified: every milestone lies in the free space, every roadmap edge is collision free when
// re-tested with an independent sampling oracle (500 points along the edge instead of the Liang-Barsky
// segment/rectangle clipping used while building), every consecutive pair of the returned path is a real
// roadmap edge, the path length is at least the Euclidean start-goal distance, and the path can never use more
// than milestones+2 vertices. Two structural properties are checked independently: the connected-component
// structure computed by union-find must agree with Dijkstra's verdict for several random query pairs, and a
// larger k must never produce a longer shortest path because the k-nearest neighbour sets are nested.
// Simplifications of this outline: k-nearest neighbours by a linear scan (no kd-tree or grid), no lazy/smart
// collision checking, no roadmap repair or resampling, and plain double arithmetic. "No path" is corroborated
// by a flood fill on a 1x1 grid, so a failure is a property of the world rather than of the sampler. All
// randomness comes from the hand-written splitmix64 stream, so runs are reproducible.
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

    internal bool Free(double x, double y)
    {
        if (x < 0 || x > Width || y < 0 || y > Height) return false;
        foreach (Rect r in _obstacles) if (r.Contains(x, y)) return false;
        return true;
    }

    internal bool StrictlyFree(double x, double y)
    {
        if (x < 0 || x > Width || y < 0 || y > Height) return false;
        foreach (Rect r in _obstacles) if (r.StrictlyInside(x, y)) return false;
        return true;
    }

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

    /// <summary>Independent oracle: dense sampling, anything strictly inside an obstacle blocks the segment.</summary>
    internal bool SampledSegmentFree(double x0, double y0, double x1, double y1, int samples)
    {
        for (int i = 0; i <= samples; i++)
        {
            double t = (double)i / samples;
            if (!StrictlyFree(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)) return false;
        }
        return true;
    }
}

internal sealed class Roadmap
{
    private readonly World _world;
    private readonly List<double> _x = new List<double>();
    private readonly List<double> _y = new List<double>();
    private readonly List<List<(int Node, double Cost)>> _adjacency = new List<List<(int, double)>>();
    private readonly List<(int A, int B)> _edges = new List<(int, int)>();
    private readonly HashSet<(int, int)> _edgeSet = new HashSet<(int, int)>();
    private ulong _state;

    internal long SampleAttempts, RejectedSamples, RejectedEdges;

    internal Roadmap(World world, ulong seed)
    {
        _world = world;
        _state = seed;
    }

    internal int NodeCount => _x.Count;
    internal int EdgeCount => _edges.Count;
    internal long Milestones => SampleAttempts - RejectedSamples;
    internal double X(int i) => _x[i];
    internal double Y(int i) => _y[i];
    internal IReadOnlyList<(int A, int B)> Edges => _edges;

    /// <summary>Samples `count` milestones; attempts that land in an obstacle are counted and skipped.</summary>
    internal void SampleMilestones(int count)
    {
        for (int i = 0; i < count; i++)
        {
            SampleAttempts++;
            double x = SplitMix.NextUnit(ref _state) * _world.Width;
            double y = SplitMix.NextUnit(ref _state) * _world.Height;
            if (!_world.Free(x, y)) { RejectedSamples++; continue; }
            AddNode(x, y);
        }
    }

    internal int AddNode(double x, double y)
    {
        _x.Add(x);
        _y.Add(y);
        _adjacency.Add(new List<(int, double)>());
        return _x.Count - 1;
    }

    /// <summary>Joins every node to its k nearest neighbours whose connecting segment is collision free.</summary>
    internal void Connect(int k)
    {
        for (int i = 0; i < NodeCount; i++)
        {
            var candidates = new List<(double Distance, int Node)>();
            for (int j = 0; j < NodeCount; j++)
            {
                if (j == i) continue;
                candidates.Add((Distance(i, j), j));
            }
            candidates.Sort((a, b) => a.Distance != b.Distance ? a.Distance.CompareTo(b.Distance) : a.Node.CompareTo(b.Node));
            int limit = Math.Min(k, candidates.Count);
            for (int c = 0; c < limit; c++)
            {
                int j = candidates[c].Node;
                var key = i < j ? (i, j) : (j, i);
                if (_edgeSet.Contains(key)) continue;
                // Coincident nodes are already in the free space: join them with a zero-length edge.
                if (candidates[c].Distance > 1e-12 && !_world.SegmentFree(_x[i], _y[i], _x[j], _y[j]))
                {
                    RejectedEdges++;
                    continue;
                }
                _edgeSet.Add(key);
                _edges.Add(key);
                _adjacency[i].Add((j, candidates[c].Distance));
                _adjacency[j].Add((i, candidates[c].Distance));
            }
        }
    }

    private double Distance(int i, int j)
    {
        double dx = _x[i] - _x[j], dy = _y[i] - _y[j];
        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>Dijkstra over the roadmap; an empty list means the two nodes are in different components.</summary>
    internal List<int> ShortestPath(int from, int to)
    {
        var path = new List<int>();
        if (from == to) { path.Add(from); return path; }
        var distance = new double[NodeCount];
        var previous = new int[NodeCount];
        var done = new bool[NodeCount];
        for (int i = 0; i < NodeCount; i++) { distance[i] = double.MaxValue; previous[i] = -1; }
        distance[from] = 0;
        for (int step = 0; step < NodeCount; step++)
        {
            int best = -1;
            for (int i = 0; i < NodeCount; i++)
                if (!done[i] && (best < 0 || distance[i] < distance[best])) best = i;
            if (best < 0 || double.IsPositiveInfinity(distance[best]) || distance[best] == double.MaxValue) break;
            done[best] = true;
            foreach (var (node, cost) in _adjacency[best])
                if (distance[best] + cost < distance[node]) { distance[node] = distance[best] + cost; previous[node] = best; }
        }
        if (distance[to] == double.MaxValue) return path;
        for (int at = to; at >= 0; at = previous[at]) { path.Add(at); if (at == from) break; }
        if (path.Count == 0 || path[path.Count - 1] != from) return new List<int>();
        path.Reverse();
        return path;
    }

    internal double PathLength(List<int> path)
    {
        double total = 0;
        for (int i = 1; i < path.Count; i++) total += Distance(path[i], path[i - 1]);
        return total;
    }

    internal bool HasEdge(int a, int b) => _edgeSet.Contains(a < b ? (a, b) : (b, a));
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

    private static World Split() => new World(100, 100, new List<Rect>
    {
        new Rect(0, 45, 100, 55)          // a bar across the whole width: the workspace is disconnected
    });

    private static World Ring() => new World(100, 100, new List<Rect>
    {
        new Rect(68, 68, 92, 71),
        new Rect(68, 89, 92, 92),
        new Rect(68, 71, 71, 89),
        new Rect(89, 71, 92, 89)
    });

    /// <summary>Independent component oracle: union-find over the roadmap edges.</summary>
    private static int[] Components(Roadmap map)
    {
        var parent = new int[map.NodeCount];
        for (int i = 0; i < parent.Length; i++) parent[i] = i;
        int Find(int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
        foreach (var (a, b) in map.Edges)
        {
            int ra = Find(a), rb = Find(b);
            if (ra != rb) parent[ra] = rb;
        }
        for (int i = 0; i < parent.Length; i++) parent[i] = Find(i);
        return parent;
    }

    /// <summary>Independent reachability oracle: 1x1 flood fill with diagonal corner checks.</summary>
    private static bool GridReachable(World world, double sx, double sy, double gx, double gy)
    {
        const int Size = 101;
        bool[,] seen = new bool[Size, Size];
        var queue = new Queue<(int X, int Y)>();
        int startX = (int)Math.Round(sx), startY = (int)Math.Round(sy);
        int goalX = (int)Math.Round(gx), goalY = (int)Math.Round(gy);
        if (!world.Free(startX, startY) || !world.Free(goalX, goalY)) return false;
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
                if (nx < 0 || ny < 0 || nx >= Size || ny >= Size || seen[nx, ny] || !world.Free(nx, ny)) continue;
                if (dxs[k] != 0 && dys[k] != 0
                    && (!world.Free(cx + dxs[k], cy) || !world.Free(cx, cy + dys[k]))) continue;
                seen[nx, ny] = true;
                queue.Enqueue((nx, ny));
            }
        }
        return false;
    }

    private static Roadmap Build(World world, int milestones, int k, ulong seed, int startX, int startY, int goalX, int goalY)
    {
        var map = new Roadmap(world, seed);
        map.SampleMilestones(milestones);
        map.AddNode(startX, startY);
        map.AddNode(goalX, goalY);
        map.Connect(k);
        return map;
    }

    public static void Main()
    {
        // ---- zero milestones: the roadmap degenerates to the direct start-goal segment ------------------
        var open = new World(100, 100, new List<Rect>());
        var direct = Build(open, 0, 1, 0x7391UL, 5, 5, 95, 95);
        var directPath = direct.ShortestPath(0, 1);
        Check(direct.NodeCount == 2 && direct.EdgeCount == 1 && directPath.Count == 2
            && Math.Abs(direct.PathLength(directPath) - Math.Sqrt(90 * 90 + 90 * 90)) < 1e-9,
            "with no milestones the roadmap is a single edge whose length is exactly the Euclidean distance");

        var blocked = Build(Split(), 0, 1, 0x7391UL, 5, 5, 95, 95);
        Check(blocked.EdgeCount == 0 && blocked.ShortestPath(0, 1).Count == 0,
            "a full-width bar leaves the start and goal in different components");

        var same = Build(open, 0, 1, 0x7391UL, 40, 40, 40, 40);
        Check(same.NodeCount == 2 && same.ShortestPath(0, 1).Count == 2 && same.PathLength(same.ShortestPath(0, 1)) == 0,
            "start equal to goal is a zero-length path");

        // ---- the corridor instance with a real roadmap ---------------------------------------------------
        var world = Corridor();
        var map = Build(world, 200, 8, 0x7392UL, 5, 5, 95, 95);
        int start = map.NodeCount - 2, goal = map.NodeCount - 1;
        List<int> path = map.ShortestPath(start, goal);
        Check(path.Count > 0 && path[0] == start && path[path.Count - 1] == goal,
            "the roadmap connects the start to the goal through the corridor");
        Check(map.Milestones >= 100 && map.NodeCount == map.Milestones + 2 && map.SampleAttempts == 200,
            "milestone bookkeeping is exact: " + map.Milestones + " of 200 samples survived");

        bool milestonesFree = true, edgesFree = true, pathEdgesExist = true;
        string? firstFailure = null;
        for (int i = 0; i < map.NodeCount; i++)
            if (!world.Free(map.X(i), map.Y(i)))
            {
                milestonesFree = false;
                firstFailure ??= "node " + i + " is not free";
            }
        foreach (var (a, b) in map.Edges)
            if (!world.SampledSegmentFree(map.X(a), map.Y(a), map.X(b), map.Y(b), 500))
            {
                edgesFree = false;
                firstFailure ??= "edge " + a + "-" + b + " crosses an obstacle";
            }
        for (int i = 1; i < path.Count; i++)
            if (!map.HasEdge(path[i - 1], path[i]))
            {
                pathEdgesExist = false;
                firstFailure ??= "path step " + path[i - 1] + "->" + path[i] + " is not a roadmap edge";
            }
        Check(milestonesFree, "every milestone lies in the free space: " + (firstFailure ?? "ok"));
        Check(edgesFree, "all " + map.EdgeCount + " roadmap edges survive the sampling oracle");
        Check(pathEdgesExist, "every step of the returned path is a roadmap edge");

        double straight = Math.Sqrt(90 * 90 + 90 * 90);
        Check(map.PathLength(path) >= straight - 1e-9 && path.Count <= map.Milestones + 2,
            "the path is never shorter than the straight line (" + Math.Round(map.PathLength(path), 3)
            + ") and uses " + path.Count + " vertices");
        Check(map.EdgeCount <= 8L * map.NodeCount && map.RejectedEdges > 0,
            "every node contributed at most k=8 new edges: " + map.EdgeCount + " edges, "
            + map.RejectedEdges + " candidate edges rejected by the collision check");
        Check(GridReachable(world, 5, 5, 95, 95), "the flood fill agrees that the corridor instance is solvable");

        // ---- components (union-find) must agree with Dijkstra for random queries -------------------------
        int[] component = Components(map);
        ulong state = 0x7393UL;
        bool componentsAgree = true;
        for (int q = 0; q < 8; q++)
        {
            int a = SplitMix.NextInt(ref state, map.NodeCount / 2);
            int b = SplitMix.NextInt(ref state, map.NodeCount / 2);
            bool reachable = map.ShortestPath(a, b).Count > 0;
            if (reachable != (component[a] == component[b]))
            {
                componentsAgree = false;
                firstFailure ??= "query " + a + "->" + b + " disagrees with union-find";
            }
        }
        Check(componentsAgree, "union-find and Dijkstra agree on all 8 random queries: " + (firstFailure ?? "ok"));

        // ---- more neighbours can only help (an obstacle-free roadmap stays connected for every k) ----------
        var openWorld = new World(100, 100, new List<Rect>());
        var k4 = Build(openWorld, 80, 4, 0x7395UL, 5, 5, 95, 95);
        var k12 = Build(openWorld, 80, 12, 0x7395UL, 5, 5, 95, 95);
        var path4 = k4.ShortestPath(k4.NodeCount - 2, k4.NodeCount - 1);
        var path12 = k12.ShortestPath(k12.NodeCount - 2, k12.NodeCount - 1);
        bool nested = true;
        foreach (var (a, b) in k4.Edges) if (!k12.HasEdge(a, b)) nested = false;
        Check(nested && k12.EdgeCount >= k4.EdgeCount,
            "the k=12 roadmap keeps every k=4 edge (" + k4.EdgeCount + " -> " + k12.EdgeCount + " edges)");
        Check(path4.Count > 0 && path12.Count > 0 && k12.PathLength(path12) <= k4.PathLength(path4) + 1e-9,
            "k=12 yields a path no longer than k=4 ("
            + Math.Round(k12.PathLength(path12), 3) + " <= " + Math.Round(k4.PathLength(path4), 3) + ")");

        var repeat = Build(world, 200, 8, 0x7392UL, 5, 5, 95, 95);
        Check(repeat.EdgeCount == map.EdgeCount
            && Math.Abs(repeat.PathLength(repeat.ShortestPath(repeat.NodeCount - 2, repeat.NodeCount - 1)) - map.PathLength(path)) < 1e-12,
            "the same seed reproduces the same roadmap and the same path length");

        // ---- the sealed pocket: no path, corroborated by two independent oracles -------------------------
        var ring = Ring();
        var sealedMap = Build(ring, 200, 8, 0x7394UL, 5, 5, 80, 80);
        int sealedStart = sealedMap.NodeCount - 2, sealedGoal = sealedMap.NodeCount - 1;
        Check(sealedMap.ShortestPath(sealedStart, sealedGoal).Count == 0,
            "no roadmap path reaches a goal sealed inside an obstacle ring");
        Check(Components(sealedMap)[sealedStart] != Components(sealedMap)[sealedGoal],
            "union-find confirms the start and the sealed goal sit in different components");
        Check(!GridReachable(ring, 5, 5, 80, 80), "the flood fill confirms the sealed goal is genuinely unreachable");

        Console.WriteLine("milestones=" + map.Milestones + " rejectedSamples=" + map.RejectedSamples
            + " nodes=" + map.NodeCount + " edges=" + map.EdgeCount + " rejectedEdges=" + map.RejectedEdges
            + " pathVertices=" + path.Count + " pathLength=" + Math.Round(map.PathLength(path), 4)
            + " straightLine=" + Math.Round(straight, 4));
        Console.WriteLine("PASS");
    }
}
