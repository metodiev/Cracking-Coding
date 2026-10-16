// Task 740 - Implement visibility graph path planning for polygons. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// Shortest-path planning among polygonal obstacles. The scene is a rectangular boundary with rectangular
// holes; the graph nodes are the hole corners, the boundary corners and the start/goal configurations, and two
// nodes are joined exactly when the segment between them is a legal move: it must not properly cross any
// polygon edge, and its midpoint must be neither outside the boundary nor strictly inside a hole. With that
// predicate the graph contains the shortest path, because a taut path only bends at obstacle vertices, and a
// plain Dijkstra over the dense graph recovers it.
// Invariants verified: the visibility relation is symmetric and irreflexive, the midpoint guard really rules
// out segments that lie inside a hole (an independent oracle samples 500 points along every edge used, and
// every step of the returned path must be a graph edge), and the extrema: in a hole-free scene the graph is
// complete and the shortest path equals the Euclidean distance, while a scene with holes must detour and the
// number of graph edges must drop. The strongest cross-check is a totally different planner - Dijkstra on a
// 2x2 grid over the same workspace with diagonal corner checks - whose path is a legal path and therefore can
// never be shorter than the visibility-graph optimum; the visibility path must be at least the straight-line
// distance and, on the sampled scene, agrees with the grid path to within the grid's resolution.
// Simplifications of this outline: rectangles only (no general polygons, no curved obstacles), and the
// visibility predicate clips each candidate segment against the hole boxes - exact for integer coordinates -
// rejecting it only when a sub-segment of positive length lies strictly inside a hole, so that hugging a wall
// or grazing a corner stays legal while a hole's diagonal (which crosses no edge properly) is rejected. No
// rotational/translational sweeps, and double arithmetic for the lengths. All scenes are built from literal
// coordinates, so runs are deterministic.
using System;
using System.Collections.Generic;

internal sealed class Poly
{
    internal readonly List<(double X, double Y)> Points = new List<(double X, double Y)>();

    internal Poly(params double[] coordinates)
    {
        for (int i = 0; i + 1 < coordinates.Length; i += 2) Points.Add((coordinates[i], coordinates[i + 1]));
    }

    internal int Count => Points.Count;

    internal IEnumerable<((double X, double Y) A, (double X, double Y) B)> Edges()
    {
        for (int i = 0; i < Points.Count; i++) yield return (Points[i], Points[(i + 1) % Points.Count]);
    }
}

internal sealed class Scene
{
    internal readonly double Width, Height;
    private readonly Poly _boundary;
    private readonly List<Poly> _holes = new List<Poly>();
    internal readonly List<(double X, double Y)> Nodes = new List<(double X, double Y)>();

    internal Scene(double width, double height)
    {
        Width = width;
        Height = height;
        _boundary = new Poly(0, 0, width, 0, width, height, 0, height);
    }

    internal int HoleCount => _holes.Count;
    internal int EdgeCountOfScene => _boundary.Count + 4 * _holes.Count;

    internal void AddHole(Poly hole) => _holes.Add(hole);

    internal void AddNode(double x, double y)
    {
        double eps = 1e-9;
        // A vertex on a hole's boundary is a legal corridor configuration, one strictly inside is not.
        if (x < -eps || x > Width + eps || y < -eps || y > Height + eps || StrictlyInsideHole(x, y))
            throw new ArgumentException("a configuration must be inside the boundary and outside every hole");
        if (!Nodes.Exists(n => Math.Abs(n.X - x) < eps && Math.Abs(n.Y - y) < eps)) Nodes.Add((x, y));
    }

    internal void AddHoleCorners()
    {
        foreach (Poly hole in _holes) foreach (var p in hole.Points) AddNode(p.X, p.Y);
    }

    internal void AddBoundaryCorners()
    {
        foreach (var p in _boundary.Points) AddNode(p.X, p.Y);
    }

    private static bool HoleContains(Poly hole, double x, double y, bool strict)
    {
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var p in hole.Points)
        {
            minX = Math.Min(minX, p.X); maxX = Math.Max(maxX, p.X);
            minY = Math.Min(minY, p.Y); maxY = Math.Max(maxY, p.Y);
        }
        return strict ? x > minX && x < maxX && y > minY && y < maxY : x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    internal bool InsideBoundary(double x, double y) => x >= 0 && x <= Width && y >= 0 && y <= Height;

    internal bool StrictlyInsideHole(double x, double y)
    {
        foreach (Poly hole in _holes) if (HoleContains(hole, x, y, strict: true)) return true;
        return false;
    }

    internal bool Free(double x, double y) => InsideBoundary(x, y) && !StrictlyInsideHole(x, y);

    /// <summary>Independent oracle used to validate visibility: dense sampling of the segment.</summary>
    internal bool SampledFree((double X, double Y) a, (double X, double Y) b, int samples)
    {
        for (int i = 0; i <= samples; i++)
        {
            double t = (double)i / samples;
            if (!Free(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t)) return false;
        }
        return true;
    }

    /// <summary>Strict orientation determinant; sign only, computed in integers scaled to doubles.</summary>
    private static double Cross((double X, double Y) o, (double X, double Y) a, (double X, double Y) b)
        => (a.X - o.X) * (b.Y - o.Y) - (a.Y - o.Y) * (b.X - o.X);

    private static bool Clip(double p, double q, ref double t0, ref double t1)
    {
        if (p == 0) return q >= 0;
        double t = q / p;
        if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; }
        else { if (t < t0) return false; if (t < t1) t1 = t; }
        return true;
    }

    /// <summary>
    /// True when a sub-segment of positive length lies strictly inside the hole. Liang-Barsky clips the segment
    /// against the hole's closed box (exact for integer coordinates); an empty or single-point clip is only a
    /// graze, and a clipped sub-segment that hugs an edge has its midpoint exactly on the boundary, so both of
    /// those legal moves survive while the diagonal of a hole - which crosses no edge properly - is rejected.
    /// </summary>
    private static bool InteriorOverlap(Poly hole, (double X, double Y) a, (double X, double Y) b)
    {
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        foreach (var p in hole.Points)
        {
            minX = Math.Min(minX, p.X); maxX = Math.Max(maxX, p.X);
            minY = Math.Min(minY, p.Y); maxY = Math.Max(maxY, p.Y);
        }
        double t0 = 0, t1 = 1;
        double dx = b.X - a.X, dy = b.Y - a.Y;
        if (!Clip(-dx, a.X - minX, ref t0, ref t1)) return false;
        if (!Clip(dx, maxX - a.X, ref t0, ref t1)) return false;
        if (!Clip(-dy, a.Y - minY, ref t0, ref t1)) return false;
        if (!Clip(dy, maxY - a.Y, ref t0, ref t1)) return false;
        if (t0 >= t1) return false;
        double middle = (t0 + t1) / 2;
        return HoleContains(hole, a.X + dx * middle, a.Y + dy * middle, strict: true);
    }

    internal bool Visible(int u, int v)
    {
        if (u == v) return false;
        var a = Nodes[u];
        var b = Nodes[v];
        // Both nodes are legal configurations inside the boundary, so the segment only has to avoid the holes.
        foreach (Poly hole in _holes) if (InteriorOverlap(hole, a, b)) return false;
        return true;
    }
}

internal sealed class VisibilityGraph
{
    private readonly Scene _scene;
    private readonly bool[,] _visible;
    private readonly double[,] _cost;

    internal VisibilityGraph(Scene scene)
    {
        _scene = scene;
        int n = scene.Nodes.Count;
        _visible = new bool[n, n];
        _cost = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (scene.Visible(i, j))
                {
                    _visible[i, j] = _visible[j, i] = true;
                    var a = scene.Nodes[i];
                    var b = scene.Nodes[j];
                    _cost[i, j] = _cost[j, i] = Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
                }
    }

    internal int NodeCount => _scene.Nodes.Count;
    internal Scene Scene => _scene;

    internal bool Visible(int u, int v) => _visible[u, v];

    internal long EdgeCount()
    {
        long total = 0;
        for (int i = 0; i < NodeCount; i++)
            for (int j = i + 1; j < NodeCount; j++)
                if (_visible[i, j]) total++;
        return total;
    }

    /// <summary>Dijkstra over the dense visibility graph.</summary>
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
            if (best < 0 || distance[best] == double.MaxValue) break;
            done[best] = true;
            for (int j = 0; j < NodeCount; j++)
                if (!done[j] && _visible[best, j] && distance[best] + _cost[best, j] < distance[j])
                {
                    distance[j] = distance[best] + _cost[best, j];
                    previous[j] = best;
                }
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
        for (int i = 1; i < path.Count; i++)
        {
            var a = _scene.Nodes[path[i - 1]];
            var b = _scene.Nodes[path[i]];
            total += Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
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

    private static Scene OpenScene()
    {
        var scene = new Scene(100, 100);
        scene.AddBoundaryCorners();
        scene.AddNode(6, 6);
        scene.AddNode(94, 94);
        return scene;
    }

    private static Scene ObstacleScene()
    {
        var scene = new Scene(100, 100);
        scene.AddHole(new Poly(20, 20, 40, 20, 40, 40, 20, 40));
        scene.AddHole(new Poly(60, 40, 80, 40, 80, 60, 60, 60));
        scene.AddHole(new Poly(30, 65, 50, 65, 50, 85, 30, 85));
        scene.AddBoundaryCorners();
        scene.AddHoleCorners();
        scene.AddNode(6, 6);
        scene.AddNode(94, 94);
        return scene;
    }

    /// <summary>Independent planner: Dijkstra on a grid whose diagonal steps may not squeeze past a corner.</summary>
    private static double GridPathLength(Scene scene, (double X, double Y) start, (double X, double Y) goal)
    {
        const int Size = 51;
        const double Cell = 2.0;
        var distance = new double[Size, Size];
        var done = new bool[Size, Size];
        for (int i = 0; i < Size; i++)
            for (int j = 0; j < Size; j++)
                distance[i, j] = double.MaxValue;
        int sx = (int)Math.Round(start.X / Cell), sy = (int)Math.Round(start.Y / Cell);
        int gx = (int)Math.Round(goal.X / Cell), gy = (int)Math.Round(goal.Y / Cell);
        distance[sx, sy] = 0;
        int[] dxs = { -1, 0, 1, -1, 1, -1, 0, 1 };
        int[] dys = { -1, -1, -1, 0, 0, 1, 1, 1 };
        for (int step = 0; step < Size * Size; step++)
        {
            int bx = -1, by = -1;
            for (int x = 0; x < Size; x++)
                for (int y = 0; y < Size; y++)
                    if (!done[x, y] && (bx < 0 || distance[x, y] < distance[bx, by])) { bx = x; by = y; }
            if (bx < 0 || distance[bx, by] == double.MaxValue) break;
            done[bx, by] = true;
            for (int k = 0; k < 8; k++)
            {
                int nx = bx + dxs[k], ny = by + dys[k];
                if (nx < 0 || ny < 0 || nx >= Size || ny >= Size || done[nx, ny]) continue;
                if (!scene.Free(nx * Cell, ny * Cell)) continue;
                if (dxs[k] != 0 && dys[k] != 0
                    && (!scene.Free((bx + dxs[k]) * Cell, by * Cell) || !scene.Free(bx * Cell, (by + dys[k]) * Cell)))
                    continue;
                double length = dxs[k] != 0 && dys[k] != 0 ? Cell * Math.Sqrt(2) : Cell;
                if (distance[bx, by] + length < distance[nx, ny]) distance[nx, ny] = distance[bx, by] + length;
            }
        }
        return distance[gx, gy];
    }

    public static void Main()
    {
        // ---- a hole-free scene: the graph is complete and the answer is the straight line ---------------
        var openScene = OpenScene();
        var openGraph = new VisibilityGraph(openScene);
        int openStart = openScene.Nodes.Count - 2, openGoal = openScene.Nodes.Count - 1;
        long complete = (long)openGraph.NodeCount * (openGraph.NodeCount - 1) / 2;
        var openPath = openGraph.ShortestPath(openStart, openGoal);
        double direct = Math.Sqrt(88 * 88 + 88 * 88);
        Check(openGraph.EdgeCount() == complete, "without holes every pair of vertices is visible (" + complete + " edges)");
        Check(openPath.Count == 2 && Math.Abs(openGraph.PathLength(openPath) - direct) < 1e-9,
            "the shortest path in an empty polygon is the straight line of length " + Math.Round(direct, 4));

        // ---- the scene with three holes -----------------------------------------------------------------
        var scene = ObstacleScene();
        var graph = new VisibilityGraph(scene);
        int start = scene.Nodes.Count - 2, goal = scene.Nodes.Count - 1;
        List<int> path = graph.ShortestPath(start, goal);
        Check(path.Count >= 3 && path[0] == start && path[path.Count - 1] == goal,
            "the detour path visits " + path.Count + " vertices from the start to the goal");
        Check(graph.EdgeCount() < (long)graph.NodeCount * (graph.NodeCount - 1) / 2,
            "holes hide part of the visibility: " + graph.EdgeCount() + " of "
            + ((long)graph.NodeCount * (graph.NodeCount - 1) / 2) + " pairs are visible");

        bool symmetric = true, irreflexive = true, stepsAreEdges = true, stepsAreFree = true;
        string? firstFailure = null;
        for (int i = 0; i < graph.NodeCount; i++)
        {
            if (graph.Visible(i, i)) irreflexive = false;
            for (int j = i + 1; j < graph.NodeCount; j++)
                if (graph.Visible(i, j) != graph.Visible(j, i))
                {
                    symmetric = false;
                    firstFailure ??= "visibility of " + i + "," + j + " is not symmetric";
                }
        }
        for (int i = 1; i < path.Count; i++)
        {
            if (!graph.Visible(path[i - 1], path[i]))
            {
                stepsAreEdges = false;
                firstFailure ??= "step " + path[i - 1] + "->" + path[i] + " is not a graph edge";
            }
            if (!scene.SampledFree(scene.Nodes[path[i - 1]], scene.Nodes[path[i]], 500))
            {
                stepsAreFree = false;
                firstFailure ??= "step " + path[i - 1] + "->" + path[i] + " passes through an obstacle";
            }
        }
        Check(symmetric, "visibility is symmetric for all " + graph.NodeCount + " nodes: " + (firstFailure ?? "ok"));
        Check(irreflexive, "a vertex is never visible from itself");
        Check(stepsAreEdges, "every step of the path is an edge of the visibility graph");
        Check(stepsAreFree, "the independent sampling oracle confirms every path step is collision free");

        // ---- the visibility predicate against the sampling oracle on every pair -------------------------
        bool predicateAgrees = true;
        for (int i = 0; i < graph.NodeCount; i++)
            for (int j = i + 1; j < graph.NodeCount; j++)
                if (graph.Visible(i, j) != scene.SampledFree(scene.Nodes[i], scene.Nodes[j], 1000))
                {
                    predicateAgrees = false;
                    firstFailure ??= "pair " + i + "," + j + " disagrees with the sampling oracle";
                }
        Check(predicateAgrees, "the visibility predicate matches dense sampling on all "
            + (graph.NodeCount * (graph.NodeCount - 1) / 2) + " pairs: " + (firstFailure ?? "ok"));

        // ---- the independent grid planner can never be shorter, and is close ---------------------------
        double grid = GridPathLength(scene, scene.Nodes[start], scene.Nodes[goal]);
        double length = graph.PathLength(path);
        Check(grid >= length - 1e-9,
            "the grid planner's legal path (" + Math.Round(grid, 3) + ") is never shorter than the visibility optimum ("
            + Math.Round(length, 3) + ")");
        Check(grid <= length + 8.0,
            "the visibility optimum is within one grid cell diagonal per corner of the grid path: "
            + Math.Round(length, 3) + " vs " + Math.Round(grid, 3));
        Check(length > direct && length < 1.6 * direct,
            "the detour is longer than the straight line but stays within 1.6x of it");
        Check(new HashSet<int>(path).Count == path.Count, "the returned path is simple (no repeated vertex)");

        // ---- degenerate configurations -------------------------------------------------------------------
        var badStart = new Scene(100, 100);
        badStart.AddHole(new Poly(0, 0, 20, 0, 20, 20, 0, 20));
        bool threw = false;
        try { badStart.AddNode(10, 10); }
        catch (ArgumentException) { threw = true; }
        Check(threw, "a start configuration strictly inside a hole is rejected");

        var sameGraph = new VisibilityGraph(OpenScene());
        var samePath = sameGraph.ShortestPath(openStart, openStart);
        Check(samePath.Count == 1 && sameGraph.PathLength(samePath) == 0,
            "a start equal to the goal gives a zero-length path");

        Console.WriteLine("nodes=" + graph.NodeCount + " visiblePairs=" + graph.EdgeCount()
            + " pathVertices=" + path.Count + " visibilityLength=" + Math.Round(length, 4)
            + " gridLength=" + Math.Round(grid, 4) + " straightLine=" + Math.Round(direct, 4));
        Console.WriteLine("PASS");
    }
}
