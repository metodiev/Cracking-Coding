// Task 741 - Implement Art Gallery problem approximation heuristics. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// Guards are placed on the vertices of a simple polygon (the classic vertex-guard variant of the Art Gallery
// problem). Two heuristics are implemented and cross-checked:
//   * the theorem-based construction: triangulate the polygon by ear clipping, the dual graph of a polygon
//     triangulation is a tree, so a tree walk 3-colours the vertices; picking the smallest colour class gives a
//     valid guard set of at most floor(n/3) guards, because every triangle carries all three colours and a guard
//     standing on a triangle's vertex sees that whole triangle;
//   * a greedy set cover: the interior is represented by a grid of sample points, each vertex guards the samples
//     it can see (the segment must stay inside the polygon, checked by walking it step by step), and the vertex
//     covering the most still-uncovered samples is taken until nothing is uncovered.
// Invariants verified: the colour classes partition the vertices and each triangle holds one vertex of each
// colour, so the smallest class is a legal cover; the greedy cover really does cover every sample under a
// *denser, independent* walk of each guard-sample segment; the greedy never beats the exact optimum computed by
// exhaustive enumeration over vertex subsets (two variants: covering all triangles, and covering all samples);
// the classic bounds hold - optimum <= smallest colour class <= floor(n/3), and the set-cover greedy stays
// within the harmonic bound OPT*(1+ln m) - and the greedy provably needs one guard for a convex polygon.
// Simplifications of this outline: guards are restricted to vertices (the general problem allows any interior
// point, where the same floor(n/3) bound holds), coverage of the continuous interior is represented by a grid
// of samples with a step of 3 and visibility is decided by dense sampling of the segment rather than by an exact
// rational predicate, and exhaustive enumeration is only affordable for the small polygons used here. All inputs
// are literal integer coordinates, so every run is deterministic.
using System;
using System.Collections.Generic;

internal sealed class SimplePolygon
{
    internal readonly List<(long X, long Y)> Points = new List<(long X, long Y)>();

    internal SimplePolygon(IReadOnlyList<(long X, long Y)> input)
    {
        var points = new List<(long X, long Y)>(input);
        if (Shoelace(points) < 0) points.Reverse();
        Points = points;
    }

    internal int Count => Points.Count;
    internal (long X, long Y) this[int i] => Points[i];

    internal long Area2 => Math.Abs(Shoelace(Points));

    private static long Shoelace(List<(long X, long Y)> points)
    {
        long sum = 0;
        for (int i = 0; i < points.Count; i++)
        {
            var a = points[i];
            var b = points[(i + 1) % points.Count];
            sum += a.X * b.Y - b.X * a.Y;
        }
        return sum;
    }

    internal long Cross(int a, int b, int c)
        => (Points[b].X - Points[a].X) * (Points[c].Y - Points[a].Y)
         - (Points[b].Y - Points[a].Y) * (Points[c].X - Points[a].X);

    /// <summary>Boundary-inclusive point test for double coordinates (ray crossing with an explicit edge test).</summary>
    internal bool Contains(double px, double py)
    {
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            double ex = b.X - a.X, ey = b.Y - a.Y;
            double t = ((px - a.X) * ex + (py - a.Y) * ey) / (ex * ex + ey * ey);
            if (t >= 0 && t <= 1)
            {
                double dx = px - (a.X + ex * t), dy = py - (a.Y + ey * t);
                if (Math.Sqrt(dx * dx + dy * dy) <= 1e-9) return true;
            }
        }
        bool inside = false;
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            if ((a.Y > py) != (b.Y > py))
            {
                double x = a.X + (double)(b.X - a.X) * (py - a.Y) / (b.Y - a.Y);
                if (px < x) inside = !inside;
            }
        }
        return inside;
    }

    /// <summary>Strictly inside: inside the polygon and at least 1e-9 away from every edge.</summary>
    internal bool StrictlyContains(double px, double py)
    {
        for (int i = 0; i < Points.Count; i++)
        {
            var a = Points[i];
            var b = Points[(i + 1) % Points.Count];
            double ex = b.X - a.X, ey = b.Y - a.Y;
            double t = ((px - a.X) * ex + (py - a.Y) * ey) / (ex * ex + ey * ey);
            t = Math.Max(0, Math.Min(1, t));
            double dx = px - (a.X + ex * t), dy = py - (a.Y + ey * t);
            if (Math.Sqrt(dx * dx + dy * dy) < 1e-9) return false;
        }
        return Contains(px, py);
    }

    /// <summary>
    /// Independent visibility oracle used in the checks: the segment is walked in `steps` increments and every
    /// intermediate configuration must lie in the (boundary-inclusive) polygon.
    /// </summary>
    internal bool Sees(int vertex, double sx, double sy, int steps)
    {
        double x0 = Points[vertex].X, y0 = Points[vertex].Y;
        for (int i = 0; i <= steps; i++)
        {
            double t = (double)i / steps;
            if (!Contains(x0 + (sx - x0) * t, y0 + (sy - y0) * t)) return false;
        }
        return true;
    }
}

internal static class EarClip
{
    private static bool IsEar(SimplePolygon polygon, int[] prev, int[] next, int a, int b, int c)
    {
        if (polygon.Cross(a, b, c) <= 0) return false;
        long px = polygon[b].X, py = polygon[b].Y;
        for (int i = next[c]; i != a; i = next[i])
        {
            long qx = polygon[i].X, qy = polygon[i].Y;
            if (Side(polygon, a, b, qx, qy) >= 0 && Side(polygon, b, c, qx, qy) >= 0 && Side(polygon, c, a, qx, qy) >= 0)
                return false;
        }
        return true;
    }

    private static long Side(SimplePolygon polygon, int a, int b, long px, long py)
        => (polygon[b].X - polygon[a].X) * (py - polygon[a].Y) - (polygon[b].Y - polygon[a].Y) * (px - polygon[a].X);

    internal static List<(int A, int B, int C)> Triangulate(SimplePolygon polygon)
    {
        int n = polygon.Count;
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++) { prev[i] = (i - 1 + n) % n; next[i] = (i + 1) % n; }
        var triangles = new List<(int A, int B, int C)>();
        int remaining = n, current = 0, guard = 0;
        while (remaining > 3)
        {
            if (++guard > 4 * n * n) throw new InvalidOperationException("no ear found");
            int a = prev[current], b = current, c = next[current];
            if (IsEar(polygon, prev, next, a, b, c))
            {
                triangles.Add((a, b, c));
                next[a] = c;
                prev[c] = a;
                remaining--;
                current = c;
            }
            else current = next[current];
        }
        triangles.Add((prev[current], current, next[current]));
        return triangles;
    }

    internal static Dictionary<(int, int), List<int>> EdgeMap(List<(int A, int B, int C)> triangles)
    {
        var map = new Dictionary<(int, int), List<int>>();
        for (int t = 0; t < triangles.Count; t++)
        {
            var (a, b, c) = triangles[t];
            Add(map, a, b, t);
            Add(map, b, c, t);
            Add(map, c, a, t);
        }
        return map;
    }

    private static void Add(Dictionary<(int, int), List<int>> map, int a, int b, int t)
    {
        var key = a < b ? (a, b) : (b, a);
        if (!map.TryGetValue(key, out List<int>? list)) { list = new List<int>(); map[key] = list; }
        list.Add(t);
    }
}

internal static class Gallery
{
    /// <summary>Walks the dual tree of the triangulation and 3-colours the polygon's vertices.</summary>
    internal static int[] ThreeColour(SimplePolygon polygon, List<(int A, int B, int C)> triangles, out List<int>[] classes)
    {
        var colour = new int[polygon.Count];
        for (int i = 0; i < colour.Length; i++) colour[i] = -1;
        var map = EarClip.EdgeMap(triangles);
        // Each dual edge remembers which of the two shared vertices it joins, so the apex of a freshly reached
        // triangle is exactly the vertex that is not on the shared edge.
        var neighbours = new List<(int Triangle, int U, int V)>[triangles.Count];
        for (int t = 0; t < triangles.Count; t++) neighbours[t] = new List<(int, int, int)>();
        int interiorEdges = 0;
        foreach (var pair in map)
            if (pair.Value.Count == 2)
            {
                interiorEdges++;
                neighbours[pair.Value[0]].Add((pair.Value[1], pair.Key.Item1, pair.Key.Item2));
                neighbours[pair.Value[1]].Add((pair.Value[0], pair.Key.Item1, pair.Key.Item2));
            }
        LastDiagnostic = "edges=" + map.Count + " interior=" + interiorEdges;
        var visited = new bool[triangles.Count];
        var stack = new Stack<int>();
        var root = triangles[0];
        colour[root.A] = 0; colour[root.B] = 1; colour[root.C] = 2;
        visited[0] = true;
        stack.Push(0);
        while (stack.Count > 0)
        {
            int t = stack.Pop();
            foreach (var (child, u, v) in neighbours[t])
            {
                if (visited[child]) continue;
                visited[child] = true;
                var triangle = triangles[child];
                int apex = notIn(triangle, u, v);
                if (colour[apex] < 0)
                {
                    bool[] used = new bool[3];
                    used[colour[u] >= 0 ? colour[u] : 0] = true;
                    used[colour[v] >= 0 ? colour[v] : 0] = true;
                    for (int c = 0; c < 3; c++) if (!used[c]) { colour[apex] = c; break; }
                }
                stack.Push(child);
            }
        }
        classes = new[] { new List<int>(), new List<int>(), new List<int>() };
        int visitedTriangles = 0;
        foreach (bool seen in visited) if (seen) visitedTriangles++;
        LastDiagnostic += " trianglesVisited=" + visitedTriangles + "/" + triangles.Count
            + " colours=[" + string.Join(",", colour) + "]";
        for (int v = 0; v < colour.Length; v++) if (colour[v] >= 0) classes[colour[v]].Add(v);
        return colour;
    }

    private static int notIn((int A, int B, int C) triangle, int u, int v)
    {
        if (triangle.A != u && triangle.A != v) return triangle.A;
        if (triangle.B != u && triangle.B != v) return triangle.B;
        return triangle.C;
    }

    internal static string LastDiagnostic = string.Empty;

    internal static List<(double X, double Y)> Samples(SimplePolygon polygon, double step)
    {
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i < polygon.Count; i++)
        {
            minX = Math.Min(minX, polygon[i].X); maxX = Math.Max(maxX, polygon[i].X);
            minY = Math.Min(minY, polygon[i].Y); maxY = Math.Max(maxY, polygon[i].Y);
        }
        var samples = new List<(double X, double Y)>();
        for (double x = minX + step / 2; x < maxX; x += step)
            for (double y = minY + step / 2; y < maxY; y += step)
                if (polygon.StrictlyContains(x, y)) samples.Add((x, y));
        return samples;
    }

    /// <summary>Greedy set cover over the sample points; returns the chosen vertices in selection order.</summary>
    internal static List<int> Greedy(SimplePolygon polygon, List<(double X, double Y)> samples, bool[][] coverage)
    {
        var uncovered = new bool[samples.Count];
        for (int i = 0; i < uncovered.Length; i++) uncovered[i] = true;
        var chosen = new List<int>();
        int left = samples.Count;
        while (left > 0)
        {
            int best = -1, bestGain = 0;
            for (int v = 0; v < polygon.Count; v++)
            {
                int gain = 0;
                for (int s = 0; s < samples.Count; s++) if (uncovered[s] && coverage[v][s]) gain++;
                if (gain > bestGain) { bestGain = gain; best = v; }
            }
            if (best < 0) throw new InvalidOperationException("the samples cannot be covered by the vertices");
            chosen.Add(best);
            for (int s = 0; s < samples.Count; s++) if (uncovered[s] && coverage[best][s]) { uncovered[s] = false; left--; }
        }
        return chosen;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static SimplePolygon Triangle() => new SimplePolygon(new List<(long X, long Y)> { (0, 0), (20, 0), (6, 12) });
    private static SimplePolygon ConvexPentagon() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (12, 0), (16, 10), (6, 16), (-4, 10)
    });
    private static SimplePolygon Comb() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (0, 30), (20, 30), (20, 10), (30, 10), (30, 30), (50, 30),
        (50, 10), (60, 10), (60, 30), (80, 30), (80, 0)
    });
    private static SimplePolygon ZigZag() => new SimplePolygon(new List<(long X, long Y)>
    {
        (0, 0), (0, 20), (10, 20), (10, 8), (20, 8), (20, 20), (30, 20), (30, 0)
    });

    private static int OptimumByTriangles(SimplePolygon polygon, List<(int A, int B, int C)> triangles)
    {
        int n = polygon.Count, best = n;
        for (int mask = 1; mask < 1 << n; mask++)
        {
            int count = 0;
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) count++;
            if (count >= best) continue;
            bool covers = true;
            foreach (var (a, b, c) in triangles)
                if ((mask & (1 << a)) == 0 && (mask & (1 << b)) == 0 && (mask & (1 << c)) == 0) { covers = false; break; }
            if (covers) best = count;
        }
        return best;
    }

    private static int OptimumBySamples(int n, int sampleCount, bool[][] coverage)
    {
        int best = n;
        for (int mask = 1; mask < 1 << n; mask++)
        {
            int count = 0;
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) count++;
            if (count >= best) continue;
            bool covers = true;
            for (int s = 0; s < sampleCount && covers; s++)
            {
                bool seen = false;
                for (int v = 0; v < n && !seen; v++) if ((mask & (1 << v)) != 0 && coverage[v][s]) seen = true;
                if (!seen) covers = false;
            }
            if (covers) best = count;
        }
        return best;
    }

    private static bool[][] Coverage(SimplePolygon polygon, List<(double X, double Y)> samples, int steps)
    {
        var coverage = new bool[polygon.Count][];
        for (int v = 0; v < polygon.Count; v++)
        {
            coverage[v] = new bool[samples.Count];
            for (int s = 0; s < samples.Count; s++) coverage[v][s] = polygon.Sees(v, samples[s].X, samples[s].Y, steps);
        }
        return coverage;
    }

    public static void Main()
    {
        // ---- a triangle and a convex polygon: one guard is provably enough ------------------------------
        var triangle = Triangle();
        var triangleTriangles = EarClip.Triangulate(triangle);
        Gallery.ThreeColour(triangle, triangleTriangles, out List<int>[] triangleClasses);
        Check(Math.Min(triangleClasses[0].Count, Math.Min(triangleClasses[1].Count, triangleClasses[2].Count)) == 1
            && OptimumByTriangles(triangle, triangleTriangles) == 1,
            "a triangle needs exactly one guard, as the colour classes show");

        var pentagon = ConvexPentagon();
        var pentagonTriangles = EarClip.Triangulate(pentagon);
        var pentagonSamples = Gallery.Samples(pentagon, 3);
        var pentagonCoverage = Coverage(pentagon, pentagonSamples, 96);
        int pentagonGreedy = Gallery.Greedy(pentagon, pentagonSamples, pentagonCoverage).Count;
        Check(pentagonGreedy == 1 && OptimumBySamples(pentagon.Count, pentagonSamples.Count, pentagonCoverage) == 1
            && OptimumByTriangles(pentagon, pentagonTriangles) == 1,
            "one guard suffices for the convex pentagon: the greedy picks it and both brute forces confirm it");

        // ---- the comb: the theorem-based cover and the greedy cover --------------------------------------
        var comb = Comb();
        var triangles = EarClip.Triangulate(comb);
        int[] colour = Gallery.ThreeColour(comb, triangles, out List<int>[] classes);
        int minClass = Math.Min(classes[0].Count, Math.Min(classes[1].Count, classes[2].Count));
        Check(triangles.Count == comb.Count - 2 && classes[0].Count + classes[1].Count + classes[2].Count == comb.Count,
            "the colour classes partition the " + comb.Count + " vertices of the comb (got "
            + (classes[0].Count + classes[1].Count + classes[2].Count) + " coloured, "
            + triangles.Count + " triangles, " + Gallery.LastDiagnostic + ")");
        bool everyTriangleColoured = true, everyVertexColoured = true;
        foreach (var (a, b, c) in triangles)
            if (!(colour[a] >= 0 && colour[b] >= 0 && colour[c] >= 0 && colour[a] != colour[b] && colour[b] != colour[c] && colour[a] != colour[c]))
                everyTriangleColoured = false;
        foreach (int c in colour) if (c < 0) everyVertexColoured = false;
        Check(everyTriangleColoured && everyVertexColoured,
            "every triangle carries one vertex of each colour, so the smallest class is a legal cover");
        Check(minClass <= comb.Count / 3,
            "the smallest class has " + minClass + " guards, within the floor(n/3) = " + (comb.Count / 3) + " bound");
        Check(OptimumByTriangles(comb, triangles) <= minClass,
            "the exact optimum over vertices cannot beat the colour-class cover");

        var samples = Gallery.Samples(comb, 3);
        var coverage = Coverage(comb, samples, 96);
        List<int> greedy = Gallery.Greedy(comb, samples, coverage);
        Check(greedy.Count > 0 && samples.Count > 40, "the comb's interior is represented by " + samples.Count + " samples");
        Check(greedy.Count >= OptimumBySamples(comb.Count, samples.Count, coverage),
            "the greedy never uses fewer guards than the exact sample optimum");
        Check(greedy.Count <= minClass,
            "the greedy matches the floor(n/3) construction: " + greedy.Count + " <= " + minClass);
        Check(greedy.Count <= OptimumBySamples(comb.Count, samples.Count, coverage) * (1 + Math.Log(samples.Count)),
            "the set-cover greedy stays inside its harmonic approximation bound");

        // ---- the independent verification of the greedy cover --------------------------------------------
        bool covered = true;
        string? firstFailure = null;
        foreach (var (sx, sy) in samples)
        {
            bool seen = false;
            foreach (int v in greedy) if (comb.Sees(v, sx, sy, 400)) seen = true;   // denser walk than the greedy used
            if (!seen)
            {
                covered = false;
                firstFailure ??= "sample (" + sx + "," + sy + ") is not seen by any chosen guard";
            }
        }
        Check(covered, "all " + samples.Count + " samples are covered under the dense independent walk: " + (firstFailure ?? "ok"));
        Check(greedy.TrueForAll(v => v >= 0 && v < comb.Count) && new HashSet<int>(greedy).Count == greedy.Count,
            "the guards are distinct polygon vertices");

        // ---- greedy progress, determinism and a small non-convex instance --------------------------------
        var gains = new List<int>();
        {
            var uncovered = new bool[samples.Count];
            for (int i = 0; i < uncovered.Length; i++) uncovered[i] = true;
            foreach (int v in greedy)
            {
                int gain = 0;
                for (int s = 0; s < samples.Count; s++) if (uncovered[s] && coverage[v][s]) { uncovered[s] = false; gain++; }
                gains.Add(gain);
            }
        }
        bool decreasing = true;
        for (int i = 0; i < gains.Count; i++) if (gains[i] <= 0) decreasing = false;
        Check(decreasing, "every greedy step strictly reduced the uncovered set (" + string.Join(",", gains) + ")");

        var repeat = Gallery.Greedy(comb, samples, Coverage(comb, samples, 96));
        Check(repeat.Count == greedy.Count && repeat.TrueForAll(v => greedy.Contains(v)),
            "the greedy selection is deterministic for the same inputs");

        var zigzag = ZigZag();
        var zigzagTriangles = EarClip.Triangulate(zigzag);
        var zigzagSamples = Gallery.Samples(zigzag, 3);
        var zigzagCoverage = Coverage(zigzag, zigzagSamples, 96);
        int zigzagGreedy = Gallery.Greedy(zigzag, zigzagSamples, zigzagCoverage).Count;
        int zigzagOptimum = OptimumBySamples(zigzag.Count, zigzagSamples.Count, zigzagCoverage);
        Check(zigzagGreedy >= zigzagOptimum && zigzagGreedy <= Math.Max(2, zigzagOptimum + 1)
            && OptimumByTriangles(zigzag, zigzagTriangles) <= zigzag.Count / 3,
            "the non-convex zigzag needs " + zigzagGreedy + " guards (exact optimum over samples "
            + zigzagOptimum + ", over triangles " + OptimumByTriangles(zigzag, zigzagTriangles) + ")");

        Console.WriteLine("combVertices=" + comb.Count + " triangles=" + triangles.Count
            + " colourClasses=" + classes[0].Count + "/" + classes[1].Count + "/" + classes[2].Count
            + " samples=" + samples.Count + " greedyGuards=" + greedy.Count + " minClassGuards=" + minClass);
        Console.WriteLine("PASS");
    }
}
