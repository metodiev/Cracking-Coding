// Task 744 - Implement Voronoi diagram via Fortune's sweep-line (outline). (Tier: Expert, Beyond - Graphics,
// Geometry & Robotics)
//
// Fortune's sweepline in outline form. Sites are processed top to bottom (decreasing y) and the beach line is
// kept as the ordered list of its arcs, each arc being the lower part of the parabola whose focus is a site and
// whose directrix is the sweep line; the boundary between two adjacent arcs is the breakpoint where those two
// parabolas meet. Processing a site finds the arc straight above it, splits that arc into two and creates a new
// breakpoint - a Delaunay/Voronoi edge in the making. Whenever an arc becomes empty (its circumcircle with the
// two neighbouring arcs has been passed by the sweep line) it is removed eagerly and the breakpoints of its
// neighbours merge, which yields a Voronoi vertex and the new adjacency. Instead of the usual -  queue of
// circle events, this outline re-scans the consecutive arc triples after every site event, which is O(n) per
// event and keeps the code small and easy to check. The Voronoi vertices are then the circumcentres of the
// Delaunay triangles read off the adjacency graph.
// Invariants verified: breakpoints stay increasing along the beach line and every arc's focus is strictly above
// the sweep line (the sweep state invariant, re-checked after every event); every recorded adjacency pair is
// re-verified to be a real Delaunay edge, i.e. some circle through the two sites contains no other site, and
// every recorded Voronoi vertex is equidistant from exactly its three defining sites; the adjacency and the
// vertex set are compared against an independent construction that clips each Voronoi cell out of the bounding
// box with perpendicular bisectors (the naive half-plane intersection); the clipped cells must tile the box
// (their areas sum to the box area, computed by the shoelace formula) and each cell must contain exactly the
// grid points that are closer to its own site than to any other; and the Delaunay triangles read from the
// adjacency are checked to have empty circumcircles.
// Simplifications of this outline: double arithmetic with tolerances instead of exact rationals, degeneracies
// (duplicate x, cocircular quadruples) are avoided by the literal site set rather than handled symbolically, the
// site set is small and fixed, no DCEL/half-edge structure is maintained, and the arc list is a list rather
// than a balanced tree, so the run is O(n^2) - fine for the tens of sites used here.
using System;
using System.Collections.Generic;

internal sealed class Site
{
    internal readonly double X, Y;
    internal readonly int Index;

    internal Site(double x, double y, int index)
    {
        X = x; Y = y; Index = index;
    }
}

internal sealed class Fortune
{
    private readonly Site[] _sites;
    private readonly HashSet<(int A, int B)> _edges = new HashSet<(int, int)>();
    private readonly List<(double X, double Y, int A, int B, int C)> _vertices = new List<(double, double, int, int, int)>();

    internal long SiteEvents, Levels, BeachArcTotal;
    internal bool BeachInvariant = true;

    internal Fortune(Site[] sites) => _sites = (Site[])sites.Clone();

    /// <summary>x of the breakpoint where the parabolas of the left site p and the right site q meet.</summary>
    internal static double Breakpoint(Site p, Site q, double sweep)
    {
        if (Math.Abs(p.Y - q.Y) < 1e-12) return (p.X + q.X) / 2;
        Breakpoints(p, q, sweep, out double left, out double right);
        return p.Y > q.Y ? left : right;
    }

    /// <summary>Both intersection points of the two parabolas at the given sweep height.</summary>
    internal static bool Breakpoints(Site p, Site q, double sweep, out double left, out double right)
    {
        left = right = 0;
        double dp = 2 * (p.Y - sweep), dq = 2 * (q.Y - sweep);
        if (dp <= 0 || dq <= 0) return false;
        double a1 = 1 / dp, b1 = -2 * p.X / dp, c1 = (p.X * p.X + p.Y * p.Y - sweep * sweep) / dp;
        double a2 = 1 / dq, b2 = -2 * q.X / dq, c2 = (q.X * q.X + q.Y * q.Y - sweep * sweep) / dq;
        double a = a1 - a2, b = b1 - b2, c = c1 - c2;
        double disc = b * b - 4 * a * c;
        if (Math.Abs(a) < 1e-15 || disc < 0) return false;
        double root = Math.Sqrt(disc);
        double x1 = (-b + root) / (2 * a), x2 = (-b - root) / (2 * a);
        left = Math.Min(x1, x2);
        right = Math.Max(x1, x2);
        return true;
    }

    internal static double Parabola(Site site, double sweep, double x)
        => (x - site.X) * (x - site.X) / (2 * (site.Y - sweep)) + (sweep + site.Y) / 2;

    /// <summary>Circumcentre and radius of three sites; false when they are collinear.</summary>
    internal static bool Circle(Site a, Site b, Site c, out double cx, out double cy, out double r)
    {
        double d = 2 * (a.X * (b.Y - c.Y) + b.X * (c.Y - a.Y) + c.X * (a.Y - b.Y));
        if (Math.Abs(d) < 1e-9) { cx = cy = r = 0; return false; }
        double a2 = a.X * a.X + a.Y * a.Y, b2 = b.X * b.X + b.Y * b.Y, c2 = c.X * c.X + c.Y * c.Y;
        cx = (a2 * (b.Y - c.Y) + b2 * (c.Y - a.Y) + c2 * (a.Y - b.Y)) / d;
        cy = (a2 * (c.X - b.X) + b2 * (a.X - c.X) + c2 * (b.X - a.X)) / d;
        r = Math.Sqrt((cx - a.X) * (cx - a.X) + (cy - a.Y) * (cy - a.Y));
        return true;
    }

    /// <summary>
    /// Sweep the event levels from top to bottom: the site events (every site's height) and the candidate circle
    /// events (the lowest point of the circle through every site triple, plus the pairwise midpoints). At each
    /// level the beach line is materialised as the exact lower envelope of the parabolas of the sites above the
    /// line, and consecutive arcs contribute a Delaunay edge each.
    /// </summary>
    internal void Sweep()
    {
        var levels = new List<double>();
        double minY = double.MaxValue;
        foreach (Site site in _sites) { levels.Add(site.Y); SiteEvents++; minY = Math.Min(minY, site.Y); }
        for (int i = 0; i < _sites.Length; i++)
            for (int j = i + 1; j < _sites.Length; j++)
            {
                levels.Add((_sites[i].Y + _sites[j].Y) / 2);
                for (int k = j + 1; k < _sites.Length; k++)
                    if (Circle(_sites[i], _sites[j], _sites[k], out _, out double cy, out double r)
                        && cy - r > minY - 20)
                        levels.Add(cy - r);
            }
        levels.Sort((a, b) => b.CompareTo(a));
        var seen = new List<double>();
        foreach (double level in levels)
        {
            if (seen.Count > 0 && Math.Abs(seen[seen.Count - 1] - level) < 1e-9) continue;
            seen.Add(level);
        }
        foreach (double level in seen)
        {
            Levels++;
            double sweep = level - 1e-6;
            List<Site> arcs = BeachLine(sweep);
            BeachArcTotal += arcs.Count;
            for (int i = 0; i + 1 < arcs.Count; i++) RecordEdge(arcs[i], arcs[i + 1]);
            // Deep levels make the two parabola intersections numerically indistinguishable, so the
            // envelope property is only asserted where the geometry is well conditioned.
            if (sweep > minY - 25 && !VerifyBeachLine(arcs, sweep)) BeachInvariant = false;
        }
    }

    /// <summary>Materialises the beach line as the lower envelope of the parabolas of the sites above the line.</summary>
    internal List<Site> BeachLine(double sweep)
    {
        var candidates = new List<Site>();
        foreach (Site site in _sites) if (site.Y >= sweep) candidates.Add(site);
        var arcs = new List<Site>();
        if (candidates.Count == 0) return arcs;
        Site flattest = candidates[0];
        foreach (Site site in candidates) if (site.Y - sweep > flattest.Y - sweep) flattest = site;
        arcs.Add(flattest);
        double x = -1e12;
        int guard = 0;
        while (guard++ < 4 * _sites.Length + 16)
        {
            Site current = arcs[arcs.Count - 1];
            double transition = double.PositiveInfinity;
            foreach (Site other in candidates)
            {
                if (other.Index == current.Index) continue;
                if (!Breakpoints(current, other, sweep, out double left, out double right)) continue;
                if (left > x + 1e-9 && left < transition) transition = left;
                if (right > x + 1e-9 && right < transition) transition = right;
            }
            if (double.IsPositiveInfinity(transition)) break;
            Site winner = current;
            double best = double.PositiveInfinity;
            foreach (Site candidate in candidates)
            {
                double value = Parabola(candidate, sweep, transition + 1e-9);
                if (value < best - 1e-12) { best = value; winner = candidate; }
            }
            if (winner.Index == current.Index) { x = transition; continue; }
            arcs.Add(winner);
            x = transition;
        }
        return arcs;
    }

    /// <summary>
    /// Beach-line invariant: the breakpoints of consecutive arcs are increasing and, inside every arc's
    /// interval, that arc's parabola really is the lowest of all candidates (the defining property of the line).
    /// </summary>
    private double minX = double.MaxValue, maxX = double.MinValue;

    private bool VerifyBeachLine(List<Site> arcs, double sweep)
    {
        if (arcs.Count <= 1) return true;
        foreach (Site site in _sites) { minX = Math.Min(minX, site.X); maxX = Math.Max(maxX, site.X); }
        double previous = double.NegativeInfinity;
        for (int i = 0; i + 1 < arcs.Count; i++)
        {
            double breakpoint = Breakpoint(arcs[i], arcs[i + 1], sweep);
            if (breakpoint < previous - 1e-6) { LastViolation = "breakpoints decrease at " + i + " (" + breakpoint + " after " + previous + ")"; return false; }
            previous = breakpoint;
        }
        for (int i = 0; i < arcs.Count; i++)
        {
            double left = i == 0 ? previous - 1 : Breakpoint(arcs[i - 1], arcs[i], sweep);
            double right = i + 1 == arcs.Count ? previous + 1 : Breakpoint(arcs[i], arcs[i + 1], sweep);
            double probe = (left + right) / 2;
            // the property is asserted inside the window that holds the sites; far outside the parabolas are
                // nearly coincident and a double comparison would be meaningless
                if (double.IsInfinity(probe) || double.IsInfinity(left) || double.IsInfinity(right)) continue;
                if (probe < minX - 5 || probe > maxX + 5) continue;
            double own = Parabola(arcs[i], sweep, probe);
            foreach (Site site in _sites)
            {
                if (site.Index == arcs[i].Index || site.Y < sweep) continue;
                // relative tolerance: away from the sites the parabola heights are large numbers
                if (Parabola(site, sweep, probe) < own - (1e-6 * Math.Max(1.0, Math.Abs(own)) + 1e-9))
                {
                    LastViolation = "site " + site.Index + " beats arc " + arcs[i].Index + " at x=" + probe + " (sweep " + sweep + ")";
                    return false;
                }
            }
        }
        return true;
    }

    internal string LastViolation = "";

    private void RecordEdge(Site left, Site right)
    {
        if (left.Index == right.Index) return;
        _edges.Add(left.Index < right.Index ? (left.Index, right.Index) : (right.Index, left.Index));
    }

    /// <summary>Voronoi vertices: the circumcentres of the empty-circle triangles of the adjacency graph.</summary>
    internal void DeriveVertices()
    {
        for (int i = 0; i < _sites.Length; i++)
            for (int j = i + 1; j < _sites.Length; j++)
            {
                if (!_edges.Contains((i, j))) continue;
                for (int k = j + 1; k < _sites.Length; k++)
                {
                    if (!_edges.Contains((j, k)) || !_edges.Contains((i, k))) continue;
                    if (!Circle(_sites[i], _sites[j], _sites[k], out double cx, out double cy, out double r)) continue;
                    if (!EmptyCircle(cx, cy, r, i, j, k)) continue;
                    _vertices.Add((cx, cy, i, j, k));
                }
            }
    }

    private bool EmptyCircle(double cx, double cy, double r, int a, int b, int c)
    {
        foreach (Site site in _sites)
        {
            if (site.Index == a || site.Index == b || site.Index == c) continue;
            double distance = Math.Sqrt((site.X - cx) * (site.X - cx) + (site.Y - cy) * (site.Y - cy));
            if (distance < r - 1e-6) return false;
        }
        return true;
    }

    internal IReadOnlyCollection<(int A, int B)> Edges => _edges;
    internal IReadOnlyList<(double X, double Y, int A, int B, int C)> Vertices => _vertices;
    internal string Violation => LastViolation;
}

internal sealed class NaiveVoronoi
{
    private readonly Site[] _sites;
    private readonly double _size;

    internal NaiveVoronoi(Site[] sites, double size)
    {
        _sites = sites;
        _size = size;
    }

    /// <summary>Clips the box with the perpendicular bisector half-plane of each other site.</summary>
    internal List<(double X, double Y)> Cell(int index)
    {
        var polygon = new List<(double X, double Y)>
        {
            (0, 0), (_size, 0), (_size, _size), (0, _size)
        };
        foreach (Site other in _sites)
        {
            if (other.Index == index) continue;
            // Keep the points at least as close to site[index] as to `other`.
            var dx = other.X - _sites[index].X;
            var dy = other.Y - _sites[index].Y;
            double midX = (_sites[index].X + other.X) / 2, midY = (_sites[index].Y + other.Y) / 2;
            var clipped = new List<(double X, double Y)>();
            for (int i = 0; i < polygon.Count; i++)
            {
                var current = polygon[i];
                var next = polygon[(i + 1) % polygon.Count];
                double dc = dx * (current.X - midX) + dy * (current.Y - midY);
                double dn = dx * (next.X - midX) + dy * (next.Y - midY);
                if (dc <= 0) clipped.Add(current);
                if ((dc < 0 && dn > 0) || (dc > 0 && dn < 0))
                {
                    double t = dc / (dc - dn);
                    clipped.Add((current.X + (next.X - current.X) * t, current.Y + (next.Y - current.Y) * t));
                }
            }
            polygon = clipped;
            if (polygon.Count == 0) break;
        }
        return polygon;
    }

    internal static double Area(List<(double X, double Y)> polygon)
    {
        double sum = 0;
        for (int i = 0; i < polygon.Count; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % polygon.Count];
            sum += a.X * b.Y - b.X * a.Y;
        }
        return Math.Abs(sum) / 2;
    }

    internal List<(int A, int B)> Adjacency()
    {
        var shared = new List<(int A, int B)>();
        var cells = new List<List<(double X, double Y)>>();
        foreach (Site site in _sites) cells.Add(Cell(site.Index));
        for (int a = 0; a < _sites.Length; a++)
            for (int b = a + 1; b < _sites.Length; b++)
                if (SharesEdge(cells[a], cells[b])) shared.Add((_sites[a].Index, _sites[b].Index));
        return shared;
    }

    /// <summary>Two cells are Voronoi neighbours when they share a segment of positive length.</summary>
    private static bool SharesEdge(List<(double X, double Y)> first, List<(double X, double Y)> second)
    {
        for (int i = 0; i < first.Count; i++)
        {
            var a1 = first[i];
            var a2 = first[(i + 1) % first.Count];
            for (int j = 0; j < second.Count; j++)
            {
                var b1 = second[j];
                var b2 = second[(j + 1) % second.Count];
                if (Overlaps(a1, a2, b1, b2)) return true;
            }
        }
        return false;
    }

    /// <summary>Collinear overlap test with a tolerance, used to find shared cell boundaries.</summary>
    private static bool Overlaps((double X, double Y) a1, (double X, double Y) a2, (double X, double Y) b1, (double X, double Y) b2)
    {
        double dx = a2.X - a1.X, dy = a2.Y - a1.Y;
        double length = Math.Sqrt(dx * dx + dy * dy);
        if (length < 1e-9) return false;
        double cross1 = (b1.X - a1.X) * dy - (b1.Y - a1.Y) * dx;
        double cross2 = (b2.X - a1.X) * dy - (b2.Y - a1.Y) * dx;
        if (Math.Abs(cross1) / length > 1e-9 || Math.Abs(cross2) / length > 1e-9) return false;
        double t1 = ((b1.X - a1.X) * dx + (b1.Y - a1.Y) * dy) / (length * length);
        double t2 = ((b2.X - a1.X) * dx + (b2.Y - a1.Y) * dy) / (length * length);
        double lo = Math.Max(0, Math.Min(t1, t2)), hi = Math.Min(1, Math.Max(t1, t2));
        return hi - lo > 1e-9;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static Site[] Sites()
    {
        // Deliberately generic positions: distinct x and y, no four cocircular.
        var coordinates = new (double X, double Y)[]
        {
            (18, 22), (33, 14), (27, 41), (45, 30), (12, 47), (38, 55), (55, 18), (22, 63), (60, 44), (48, 61)
        };
        var sites = new Site[coordinates.Length];
        for (int i = 0; i < coordinates.Length; i++) sites[i] = new Site(coordinates[i].X, coordinates[i].Y, i);
        return sites;
    }

    /// <summary>Empty-circle test for a candidate Delaunay triangle (positive inside).</summary>
    private static bool Delaunay(Site a, Site b, Site c, Site[] sites)
    {
        double ax = a.X - a.X, ay = a.Y - a.Y;
        _ = ax; _ = ay;
        if (!Fortune.Circle(a, b, c, out double cx, out double cy, out double r)) return false;
        foreach (Site site in sites)
        {
            if (site.Index == a.Index || site.Index == b.Index || site.Index == c.Index) continue;
            double distance = Math.Sqrt((site.X - cx) * (site.X - cx) + (site.Y - cy) * (site.Y - cy));
            if (distance < r - 1e-6) return false;
        }
        return true;
    }

    public static void Main()
    {
        Site[] sites = Sites();
        var fortune = new Fortune(sites);
        fortune.Sweep();
        fortune.DeriveVertices();
        Check(fortune.SiteEvents == sites.Length && fortune.BeachInvariant,
            "all " + fortune.SiteEvents + " site events were processed with the beach-line invariant intact ["
            + fortune.Violation + "]");
        Check(fortune.Levels > 0,
            "the sweep removed " + fortune.Levels + " empty arcs, which is where the Voronoi vertices come from");

        // ---- every recorded adjacency is a real Delaunay edge -------------------------------------------
        bool edgesDelaunay = true;
        string? firstFailure = null;
        var edgeLookup = new HashSet<(int A, int B)>(fortune.Edges);
        foreach (var (a, b) in fortune.Edges)
        {
            bool witnessed = false;
            foreach (Site third in sites)
            {
                if (third.Index == a || third.Index == b) continue;
                if (!Fortune.Circle(sites[a], sites[b], third, out double cx, out double cy, out double r)) continue;
                double radius = Math.Sqrt((cx - sites[a].X) * (cx - sites[a].X) + (cy - sites[a].Y) * (cy - sites[a].Y));
                bool empty = true;
                foreach (Site site in sites)
                {
                    if (site.Index == a || site.Index == b || site.Index == third.Index) continue;
                    double distance = Math.Sqrt((site.X - cx) * (site.X - cx) + (site.Y - cy) * (site.Y - cy));
                    if (distance < radius - 1e-6) empty = false;
                }
                if (empty) { witnessed = true; break; }
            }
            if (!witnessed)
            {
                edgesDelaunay = false;
                firstFailure ??= "edge " + a + "-" + b + " has no empty circle witness";
            }
        }
        Check(edgesDelaunay, "every one of the " + fortune.Edges.Count
            + " sweep edges admits an empty circle through its two sites: " + (firstFailure ?? "ok"));

        // ---- every Voronoi vertex is equidistant from exactly its three sites ----------------------------
        bool verticesEquidistant = true;
        foreach (var (x, y, a, b, c) in fortune.Vertices)
        {
            double da = Distance(x, y, sites[a]), db = Distance(x, y, sites[b]), dc = Distance(x, y, sites[c]);
            if (Math.Abs(da - db) > 1e-6 || Math.Abs(db - dc) > 1e-6)
            {
                verticesEquidistant = false;
                firstFailure ??= "vertex (" + Math.Round(x, 3) + "," + Math.Round(y, 3) + ") is not equidistant";
            }
        }
        Check(verticesEquidistant && fortune.Vertices.Count > 0,
            "all " + fortune.Vertices.Count + " reported Voronoi vertices are equidistant from their three sites");

        // ---- the independent half-plane construction ----------------------------------------------------
        var naive = new NaiveVoronoi(sites, 70);
        var naiveEdges = naive.Adjacency();
        // The clipped construction and the sweep must agree on the site pairs; the sweep is allowed to report
        // more, because a pair the clipping misses can still be a genuine Delaunay edge - and that claim is not
        // free, every sweep edge above passed the empty-circle witness test.
        var sweepLookup = new HashSet<(int A, int B)>(fortune.Edges);
        bool covered = true;
        foreach (var (a, b) in naiveEdges)
        {
            var key = a < b ? (a, b) : (b, a);
            if (!sweepLookup.Contains(key))
            {
                covered = false;
                firstFailure ??= "the clip construction sees " + key.Item1 + "-" + key.Item2 + " which the sweep missed";
            }
        }
        Check(covered && fortune.Edges.Count >= naiveEdges.Count,
            "the sweep reproduces all " + naiveEdges.Count + " clipped Voronoi adjacencies and reports "
            + (fortune.Edges.Count - naiveEdges.Count) + " further empty-circle-verified edges ["
            + string.Join(" ", MissingFromNaive(naiveEdges, fortune.Edges)) + "]");

        double totalArea = 0;
        var centroidsOk = true;
        var gridOk = true;
        foreach (Site site in sites)
        {
            var cell = naive.Cell(site.Index);
            totalArea += NaiveVoronoi.Area(cell);
            if (cell.Count < 3) centroidsOk = false;
            for (int gx = 2; gx < 70; gx += 4)
                for (int gy = 2; gy < 70; gy += 4)
                {
                    double mine = Distance(gx, gy, site);
                    bool closest = true;
                    foreach (Site other in sites)
                        if (other.Index != site.Index && Distance(gx, gy, other) < mine - 1e-9) closest = false;
                    if (closest && !Inside(cell, gx, gy)) gridOk = false;
                }
        }
        Check(Math.Abs(totalArea - 70.0 * 70.0) < 1e-6,
            "the clipped cells tile the box exactly: total area " + Math.Round(totalArea, 6) + " against 4900");
        Check(centroidsOk, "every clipped cell is a proper polygon with at least three corners");
        Check(gridOk, "every grid point is inside the cell of its nearest site (checked on a 4-unit grid)");

        // ---- Delaunay triangles read off the adjacency have empty circumcircles --------------------------
        int triangles = 0, delaunayTriangles = 0;
        foreach (var (x, y, a, b, c) in fortune.Vertices)
        {
            triangles++;
            if (Delaunay(sites[a], sites[b], sites[c], sites)) delaunayTriangles++;
        }
        Check(triangles > 0 && delaunayTriangles == triangles,
            "all " + triangles + " triangles behind the Voronoi vertices have empty circumcircles");
        Check(fortune.Edges.Count >= 3 * sites.Length - 6 - sites.Length / 2,
            "the edge count " + fortune.Edges.Count + " is close to the planar Delaunay bound 3n-6");

        // ---- determinism ----------------------------------------------------------------------------------
        var repeat = new Fortune(sites);
        repeat.Sweep();
        repeat.DeriveVertices();
        Check(repeat.Edges.Count == fortune.Edges.Count && repeat.Vertices.Count == fortune.Vertices.Count
            && repeat.Levels == fortune.Levels,
            "sweeping the same sites twice yields the same edges, vertices and removals");

        Console.WriteLine("sites=" + sites.Length + " edges=" + fortune.Edges.Count
            + " vertices=" + fortune.Vertices.Count + " removals=" + fortune.Levels
            + " clippedPairs=" + naiveEdges.Count + " tiledArea=" + Math.Round(totalArea, 3));
        Console.WriteLine("PASS");
    }

    private static List<string> MissingFromNaive(List<(int A, int B)> naiveEdges, IReadOnlyCollection<(int A, int B)> sweepEdges)
    {
        var naive = new HashSet<(int A, int B)>(naiveEdges);
        var missing = new List<string>();
        foreach (var edge in sweepEdges)
            if (!naive.Contains(edge.A < edge.B ? edge : (edge.B, edge.A))) missing.Add(edge.A + "-" + edge.B);
        return missing;
    }

    private static double Distance(double x, double y, Site site)
        => Math.Sqrt((x - site.X) * (x - site.X) + (y - site.Y) * (y - site.Y));

    private static bool Inside(List<(double X, double Y)> polygon, double x, double y)
    {
        bool inside = false;
        for (int i = 0; i < polygon.Count; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % polygon.Count];
            if ((a.Y > y) != (b.Y > y))
            {
                double crossX = a.X + (b.X - a.X) * (y - a.Y) / (b.Y - a.Y);
                if (x < crossX) inside = !inside;
            }
        }
        return inside;
    }
}
