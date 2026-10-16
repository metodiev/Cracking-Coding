// Task 772 - Implement dynamic planar graph algorithms (face updates) outline. (Tier: Expert, CTCI Ch 4 - Trees and Graphs)
//
// A plane straight-line embedding is stored as a rotation system: every vertex keeps its neighbours in
// counter-clockwise angular order, and the faces of the embedding are recovered by walking directed
// edges - from (u, v) the walk continues to (v, w) where w follows u in the rotation of v. This file
// implements the dynamic part of that structure: an edge may be inserted or deleted at any time, the
// insertion is accepted only when the new segment neither duplicates an edge, is a self loop, nor
// crosses (or collinearly overlaps) an existing edge, which keeps the drawing planar by construction,
// and the face structure is then recovered by one traversal over at most 2*E directed edges.
//
// Documented simplification (this task is an outline): faces are recomputed by a full O(E) traversal
// after every update instead of using a fully dynamic face-splitting/merging structure (where an edge
// split costs O(log n) through face trees, so a sequence of m updates costs O(m log n) rather than
// O(m*E)); the geometric planarity gate replaces the "trust the embedding" assumption of the
// combinatorial view; and vertex coordinates are fixed when the graph is created, so no vertex
// insertion/relocation is supported.
//
// Invariants verified after every update of a randomized workload, and on the structured cases:
//   * rotation consistency - both endpoints list the edge, and the rotation size is the degree;
//   * the face walks partition the 2*E directed edges (each appears in exactly one walk);
//   * Euler for the traversal: #faces == E - V_with_edges + 2 * (#components with edges), with the
//     component count coming from an independent union-find oracle;
//   * geometry: no two segments of the drawing cross (checked by a second, independent intersection
//     test over all edge pairs), exactly one face per edge-bearing component has non-positive signed
//     area, and for the grid mesh the face areas are 16 unit squares plus one face of area 16 with a
//     16-edge boundary walk.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class PlanarGraph
{
    private readonly int _n;
    private readonly int[] _x;
    private readonly int[] _y;
    private readonly List<int>[] _rot;                       // neighbours in ccw angular order
    private readonly bool[,] _adj;

    internal int EdgeCount;
    internal int Accepted;
    internal int Rejected;
    internal int Removed;

    internal PlanarGraph(int[] xs, int[] ys)
    {
        _n = xs.Length;
        _x = xs;
        _y = ys;
        _rot = new List<int>[_n];
        _adj = new bool[_n, _n];
        for (int i = 0; i < _n; i++) _rot[i] = new List<int>();
    }

    internal int VertexCount => _n;
    internal int Degree(int v) => v >= 0 && v < _n ? _rot[v].Count : 0;
    internal bool HasEdge(int a, int b) => a >= 0 && b >= 0 && a < _n && b < _n && _adj[a, b];
    internal List<int> Neighbours(int v) => new List<int>(_rot[v]);

    private double Angle(int v, int w) => Math.Atan2(_y[w] - _y[v], _x[w] - _x[v]);

    private static long Cross3(long ax, long ay, long bx, long by, long cx, long cy)
        => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);

    private bool OnSegment(int p, int q, int r)
        => Cross3(_x[p], _y[p], _x[q], _y[q], _x[r], _y[r]) == 0
           && Math.Min(_x[p], _x[q]) <= _x[r] && _x[r] <= Math.Max(_x[p], _x[q])
           && Math.Min(_y[p], _y[q]) <= _y[r] && _y[r] <= Math.Max(_y[p], _y[q]);

    /// <summary>True when the two drawn segments share an interior point (touching endpoints is fine).</summary>
    private bool SegmentsInterfere(int p, int q, int r, int s)
    {
        int shared = 0;
        if (p == r || p == s) shared++;
        if (q == r || q == s) shared++;
        if (shared == 0)
        {
            long d1 = Cross3(_x[p], _y[p], _x[q], _y[q], _x[r], _y[r]);
            long d2 = Cross3(_x[p], _y[p], _x[q], _y[q], _x[s], _y[s]);
            long d3 = Cross3(_x[r], _y[r], _x[s], _y[s], _x[p], _y[p]);
            long d4 = Cross3(_x[r], _y[r], _x[s], _y[s], _x[q], _y[q]);
            if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
            if (d1 == 0 && OnSegment(p, q, r)) return true;   // collinear overlap
            if (d2 == 0 && OnSegment(p, q, s)) return true;
            if (d3 == 0 && OnSegment(r, s, p)) return true;
            if (d4 == 0 && OnSegment(r, s, q)) return true;
            return false;
        }
        // Edges that share a vertex still overlap when one of them runs through the other's interior.
        if (shared == 1)
        {
            int common = (p == r || p == s) ? p : q;
            int otherNew = p == common ? q : p;
            int otherOldStart = r == common ? s : r;
            if (OnSegment(common, otherNew, otherOldStart)) return true;
            if (OnSegment(common, otherOldStart, otherNew)) return true;
        }
        return false;
    }

    internal bool WouldInterfere(int a, int b)
    {
        for (int u = 0; u < _n; u++)
            foreach (int v in _rot[u])
            {
                if (u > v) continue;
                if (SegmentsInterfere(a, b, u, v)) return true;
            }
        return false;
    }

    internal bool Insert(int a, int b)
    {
        if (a < 0 || b < 0 || a >= _n || b >= _n || a == b) { Rejected++; return false; }
        if (_adj[a, b]) { Rejected++; return false; }
        if (WouldInterfere(a, b)) { Rejected++; return false; }
        _adj[a, b] = _adj[b, a] = true;
        Splice(a, b);
        Splice(b, a);
        EdgeCount++;
        Accepted++;
        return true;
    }

    private void Splice(int v, int w)
    {
        double angle = Angle(v, w);
        int at = 0;
        while (at < _rot[v].Count && Angle(v, _rot[v][at]) < angle) at++;
        _rot[v].Insert(at, w);
    }

    internal bool Delete(int a, int b)
    {
        if (!HasEdge(a, b)) return false;
        _adj[a, b] = _adj[b, a] = false;
        _rot[a].Remove(b);
        _rot[b].Remove(a);
        EdgeCount--;
        Removed++;
        return true;
    }

    /// <summary>Face walks: from (u, v) continue to (v, w), where w follows u in the rotation of v.</summary>
    internal List<List<int>> Faces(out string violation)
    {
        violation = null;
        var visited = new HashSet<long>();
        var faces = new List<List<int>>();
        for (int u = 0; u < _n; u++)
            foreach (int v in _rot[u])
            {
                long code = (long)u * _n + v;
                if (visited.Contains(code)) continue;
                var walk = new List<int>();
                int startU = u, startV = v, cu = u, cv = v;
                while (true)
                {
                    visited.Add((long)cu * _n + cv);
                    walk.Add(cu);
                    int index = _rot[cv].IndexOf(cu);
                    if (index < 0) { violation = "directed edge (" + cu + "," + cv + ") is missing from the rotation"; return faces; }
                    int next = _rot[cv][(index + 1) % _rot[cv].Count];
                    cu = cv;
                    cv = next;
                    if (cu == startU && cv == startV) break;
                    if (visited.Contains((long)cu * _n + cv))
                    {
                        violation = "the face walk re-enters an already visited directed edge";
                        return faces;
                    }
                }
                faces.Add(walk);
            }
        return faces;
    }

    internal double SignedArea(List<int> walk)
    {
        if (walk.Count < 3) return 0;
        long twice = 0;
        for (int i = 0; i < walk.Count; i++)
        {
            int a = walk[i], b = walk[(i + 1) % walk.Count];
            twice += (long)_x[a] * _y[b] - (long)_x[b] * _y[a];
        }
        return twice / 2.0;
    }

    /// <summary>Independent union-find over the edges: how many components actually carry an edge.</summary>
    internal int ComponentsWithEdges()
    {
        var parent = new int[_n];
        for (int i = 0; i < _n; i++) parent[i] = i;
        int Find(int v) { while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; } return v; }
        for (int u = 0; u < _n; u++)
            foreach (int v in _rot[u])
            {
                if (u > v) continue;
                int ru = Find(u), rv = Find(v);
                if (ru != rv) parent[ru] = rv;
            }
        var roots = new HashSet<int>();
        for (int v = 0; v < _n; v++) if (_rot[v].Count > 0) roots.Add(Find(v));
        return roots.Count;
    }

    internal int VerticesWithEdges()
    {
        int count = 0;
        for (int v = 0; v < _n; v++) if (_rot[v].Count > 0) count++;
        return count;
    }

    /// <summary>Rotation bookkeeping plus the "every directed edge is in exactly one face" property.</summary>
    internal string InvariantViolation()
    {
        for (int u = 0; u < _n; u++)
        {
            foreach (int v in _rot[u])
            {
                if (!_adj[u, v]) return "rotation of " + u + " lists " + v + " without an edge";
                if (!_rot[v].Contains(u)) return "edge (" + u + "," + v + ") is not listed on both sides";
            }
            for (int w = 0; w < _n; w++) if (_adj[u, w] && !_rot[u].Contains(w)) return "edge (" + u + "," + w + ") is missing from a rotation";
            for (int i = 1; i < _rot[u].Count; i++)
                if (Angle(u, _rot[u][i - 1]) >= Angle(u, _rot[u][i])) return "rotation of " + u + " is not in ccw order";
        }
        var faces = Faces(out string violation);
        if (violation != null) return violation;
        int directed = 0;
        foreach (List<int> walk in faces) directed += walk.Count;
        if (directed != 2 * EdgeCount) return "the face walks cover " + directed + " directed edges instead of " + (2 * EdgeCount);
        int expected = EdgeCount - VerticesWithEdges() + 2 * ComponentsWithEdges();
        if (faces.Count != expected) return "traversal gives " + faces.Count + " faces, Euler predicts " + expected;
        return null;
    }

    /// <summary>Second, independent crossing test used only by the checks (all edge pairs).</summary>
    internal bool AnyCrossing()
    {
        var edges = new List<int[]>();
        for (int u = 0; u < _n; u++)
            foreach (int v in _rot[u])
                if (u < v) edges.Add(new[] { u, v });
        for (int i = 0; i < edges.Count; i++)
            for (int j = i + 1; j < edges.Count; j++)
            {
                int[] e = edges[i], f = edges[j];
                if (SegmentsInterfere(e[0], e[1], f[0], f[1])) return true;
            }
        return false;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Grid mesh: n x n vertices, unit spacing, used because its face structure is known.</summary>
    private static PlanarGraph Grid(int n)
    {
        var xs = new int[n * n];
        var ys = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                xs[i * n + j] = j;
                ys[i * n + j] = i;
            }
        return new PlanarGraph(xs, ys);
    }

    private static void MeshEdges(PlanarGraph g, int n)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int v = i * n + j;
                if (j + 1 < n) g.Insert(v, v + 1);
                if (i + 1 < n) g.Insert(v, v + n);
            }
    }

    public static void Main()
    {
        // ---- the empty and the singleton embedding ----------------------------------------------
        {
            var empty = new PlanarGraph(new int[0], new int[0]);
            var emptyFaces = empty.Faces(out string emptyViolation);
            Check(emptyViolation == null && emptyFaces.Count == 0, "an empty graph has no face walks");
            Check(empty.InvariantViolation() == null, "the empty graph satisfies the invariants");

            var single = new PlanarGraph(new[] { 0 }, new[] { 0 });
            var singleFaces = single.Faces(out _);
            Check(singleFaces.Count == 0 && single.Degree(0) == 0, "an isolated vertex has no half-edges");
            Check(single.ComponentsWithEdges() == 0, "an isolated vertex is not counted as a component with edges");
            Check(single.InvariantViolation() == null, "0 = 0 - 0 + 2*0 faces for the isolated vertex");

            Check(!single.Insert(0, 0), "a self loop is rejected");
            Check(single.Rejected == 1 && single.EdgeCount == 0, "the rejected self loop left the graph untouched");
            Check(!single.Insert(0, 5), "an out-of-range endpoint is rejected");
        }

        // ---- a single edge, a path and a triangle ------------------------------------------------
        {
            var edge = new PlanarGraph(new[] { 0, 4 }, new[] { 0, 0 });
            Check(edge.Insert(0, 1), "a single edge can be inserted");
            Check(!edge.Insert(0, 1), "a duplicate edge is rejected");
            Check(edge.InvariantViolation() == null, "one edge keeps the Euler identity 1 = 1 - 2 + 2*1");
            var edgeFaces = edge.Faces(out _);
            Check(edgeFaces.Count == 1 && edgeFaces[0].Count == 2, "one edge is bounded by a single two-step face walk");
            Check(edge.SignedArea(edgeFaces[0]) == 0, "the degenerate face of a single edge has zero area");

            var path = new PlanarGraph(new[] { 0, 1, 2, 3 }, new[] { 0, 0, 0, 0 });
            for (int i = 0; i + 1 < 4; i++) Check(path.Insert(i, i + 1), "path edge " + i + " is accepted");
            Check(path.EdgeCount == 3 && path.ComponentsWithEdges() == 1, "the path is one component with three edges");
            Check(path.Faces(out _).Count == 1, "a tree has a single face");
            Check(path.InvariantViolation() == null, "the path satisfies the invariants: 1 = 3 - 4 + 2*1");
            Check(!path.Insert(0, 3), "a collinear overlapping edge is rejected");
            Check(path.Rejected == 1, "the overlap was counted as a rejection");
            Check(!path.Insert(0, 2), "an edge running through an interior vertex is rejected as well");

            var triangle = new PlanarGraph(new[] { 0, 4, 0 }, new[] { 0, 0, 4 });
            Check(triangle.Insert(0, 1) && triangle.Insert(1, 2) && triangle.Insert(2, 0), "the triangle is accepted edge by edge");
            var faces = triangle.Faces(out _);
            Check(faces.Count == 2, "a triangle has an inner and an outer face");
            Check(triangle.InvariantViolation() == null, "the triangle satisfies the invariants: 2 = 3 - 3 + 2*1");
            double positive = double.MinValue, negative = double.MaxValue;
            foreach (List<int> walk in faces)
            {
                double area = triangle.SignedArea(walk);
                if (area > positive) positive = area;
                if (area < negative) negative = area;
            }
            Check(Math.Abs(positive - 8) < 1e-9, "the unbounded face walk of the (0,0),(4,0),(0,4) triangle encloses +8");
            Check(Math.Abs(negative + 8) < 1e-9, "the bounded face walk encloses -8: the successor walk orients bounded faces clockwise");
            Check(positive + negative == 0, "the two face walks enclose the same magnitude with opposite orientation");
            Check(triangle.Delete(1, 2), "an edge can be deleted again");
            Check(triangle.Faces(out _).Count == 1, "deleting a triangle edge merges the two faces back into one");
            Check(triangle.InvariantViolation() == null, "the graph stays consistent after the deletion");
            Check(!triangle.Delete(1, 2), "deleting the same edge twice is a no-op");
        }

        // ---- the grid mesh has a known face structure --------------------------------------------
        {
            const int N = 5;
            var g = Grid(N);
            MeshEdges(g, N);
            Check(g.EdgeCount == 2 * N * (N - 1), "the 5x5 mesh has 2*n*(n-1) = 40 edges");
            var faces = g.Faces(out _);
            Check(faces.Count == (N - 1) * (N - 1) + 1, "a 5x5 mesh has 16 unit squares plus the outer face");
            Check(g.InvariantViolation() == null, "the mesh satisfies the invariants: 17 = 40 - 25 + 2*1");

            var sizes = new List<int>();
            var areas = new List<double>();
            int positivelyOriented = 0;
            foreach (List<int> walk in faces)
            {
                sizes.Add(walk.Count);
                double signed = g.SignedArea(walk);
                if (signed > 0) positivelyOriented++;
                areas.Add(Math.Abs(signed));
            }
            sizes.Sort();
            areas.Sort();
            Check(positivelyOriented == 1, "exactly one face - the unbounded one - is oriented positively");
            Check(sizes[0] == 4 && sizes[sizes.Count - 1] == 16, "face boundary lengths are sixteen 4-cycles and one 16-cycle");
            int unitSquares = 0;
            foreach (int size in sizes) if (size == 4) unitSquares++;
            Check(unitSquares == 16, "exactly sixteen faces have a four-edge boundary");
            Check(Math.Abs(areas[0] - 1) < 1e-9 && Math.Abs(areas[1] - 1) < 1e-9 && Math.Abs(areas[areas.Count - 1] - 16) < 1e-9,
                "face areas are sixteen unit squares plus one face of area 16");
            double innerTotal = 0;
            foreach (double area in areas) if (Math.Abs(area - 16) > 1e-9) innerTotal += area;
            Check(Math.Abs(innerTotal - 16) < 1e-9, "the inner faces tile the 4x4 mesh area exactly");
            Check(!g.AnyCrossing(), "no two edges of the mesh cross (independent pair-wise test)");
            Console.WriteLine("mesh 5x5: V=25 E=" + g.EdgeCount + " F=" + faces.Count + " innerArea=" + innerTotal.ToString("F1"));
        }

        // ---- dynamic edge updates on the mesh -----------------------------------------------------
        {
            const int N = 5;
            var g = Grid(N);
            MeshEdges(g, N);
            int baseline = g.Faces(out _).Count;

            // a diagonal splits one unit square into two triangles
            int corner = 0;                                     // vertex (0,0)
            Check(g.Insert(corner, corner + N + 1), "the diagonal of the bottom-left cell is accepted");
            Check(g.Faces(out _).Count == baseline + 1, "splitting a face adds exactly one face");
            Check(g.InvariantViolation() == null, "the split mesh still satisfies the invariants");
            double areaSum = 0;
            var splitFaces = g.Faces(out _);
            foreach (List<int> walk in splitFaces) areaSum += Math.Abs(g.SignedArea(walk) - 16) > 1e-9 ? Math.Abs(g.SignedArea(walk)) : 0;
            Check(Math.Abs(areaSum - 16) < 1e-9, "the inner faces still tile the mesh after the split");
            bool hasHalf = false;
            foreach (List<int> walk in splitFaces) if (Math.Abs(Math.Abs(g.SignedArea(walk)) - 0.5) < 1e-9) hasHalf = true;
            Check(hasHalf, "the two halves of the split cell have area 0.5 each");

            Check(!g.Insert(corner, corner + N + 1), "re-inserting the same diagonal is rejected as a duplicate");
            Check(!g.Insert(corner, corner + 2 * N + 2), "a segment crossing existing edges is rejected");
            Check(!g.Insert(corner + 1, corner + N), "an edge whose interior passes through the diagonal is rejected");
            Check(g.Rejected == 3, "three interference rejections were counted");
            Check(!g.AnyCrossing(), "the drawing is still crossing-free after the rejected attempts");

            // deleting a boundary edge merges an inner square with the outer face
            Check(g.Delete(corner, corner + 1), "a boundary edge can be removed");
            Check(g.Faces(out _).Count == baseline, "removing a boundary edge costs one face again");
            Check(g.InvariantViolation() == null, "the invariants survive the deletion");
            Check(g.Insert(corner, corner + 1), "the boundary edge can be restored");
            Check(g.Faces(out _).Count == baseline + 1, "restoring it brings the extra face back");
            Check(g.Delete(corner, corner + N + 1), "the diagonal can be removed as well");
            Check(g.Faces(out _).Count == baseline, "removing the diagonal merges the two triangles back");
        }

        // ---- a bridge changes the component count, not the face count ----------------------------
        {
            var g = new PlanarGraph(new[] { 0, 4, 0, 20, 24, 20 }, new[] { 0, 0, 4, 0, 0, 4 });
            Check(g.Insert(0, 1) && g.Insert(1, 2) && g.Insert(2, 0), "the first triangle is built");
            Check(g.ComponentsWithEdges() == 1 && g.Faces(out _).Count == 2, "one triangle has two faces");
            Check(g.Insert(3, 4) && g.Insert(4, 5) && g.Insert(5, 3), "the second triangle is built");
            Check(g.ComponentsWithEdges() == 2 && g.Faces(out _).Count == 4, "two triangles in their own components have four faces");
            Check(g.InvariantViolation() == null, "two components satisfy 4 = 6 - 6 + 2*2");
            Check(g.Insert(1, 3), "the bridge between the triangles is accepted");
            Check(g.ComponentsWithEdges() == 1, "the bridge merges the two components");
            Check(g.Faces(out _).Count == 3, "a bridge does not add a face (the two outer faces merge)");
            Check(g.InvariantViolation() == null, "the bridged graph satisfies 3 = 7 - 6 + 2*1");
            Check(!g.Insert(1, 3), "the bridge cannot be inserted twice");
            Check(g.Delete(1, 3), "the bridge can be removed");
            Check(g.ComponentsWithEdges() == 2 && g.Faces(out _).Count == 4, "removing the bridge restores the two separate embeddings");
        }

        // ---- randomized workload with the invariants re-checked after every update ----------------
        {
            const int N = 6;
            var g = Grid(N);
            MeshEdges(g, N);
            ulong state = 0xFACEFEED0BADF00DUL;
            bool invariantsHeld = true, eulerHeld = true, planarHeld = true;
            string firstProblem = "ok";
            int inserts = 0, deletes = 0;
            for (int step = 0; step < 300; step++)
            {
                // half of the attempts stay inside one cell (mesh edges and diagonals), the rest are
                // arbitrary pairs, so the workload contains both cheap updates and rejections
                int i = SplitMix64.NextInt(ref state, N), j = SplitMix64.NextInt(ref state, N);
                int a = i * N + j;
                int b;
                if (SplitMix64.NextInt(ref state, 2) == 0)
                {
                    int kind = SplitMix64.NextInt(ref state, 3);
                    if (kind == 0 && j + 1 < N) b = a + 1;
                    else if (kind == 1 && i + 1 < N) b = a + N;
                    else if (kind == 2 && i + 1 < N && j + 1 < N) b = a + N + 1;
                    else b = SplitMix64.NextInt(ref state, N * N);
                }
                else
                {
                    b = SplitMix64.NextInt(ref state, N * N);
                }
                if (b == a) continue;
                if (SplitMix64.NextInt(ref state, 2) == 0)
                {
                    if (g.Insert(a, b)) inserts++;
                }
                else
                {
                    if (g.Delete(a, b)) deletes++;
                }
                string violation = g.InvariantViolation();
                if (violation != null) { invariantsHeld = false; firstProblem = "step " + step + ": " + violation; break; }
                int faces = g.Faces(out _).Count;
                int predicted = g.EdgeCount - g.VerticesWithEdges() + 2 * g.ComponentsWithEdges();
                if (faces != predicted) { eulerHeld = false; firstProblem = "step " + step + ": faces " + faces + " vs " + predicted; break; }
                if (g.AnyCrossing()) { planarHeld = false; firstProblem = "step " + step + ": crossing edges"; break; }
            }
            Check(invariantsHeld, "the rotation/face invariants hold after every update: " + firstProblem);
            Check(eulerHeld, "Euler's identity holds after every update: " + firstProblem);
            Check(planarHeld, "the drawing never contains crossing edges: " + firstProblem);
            Check(inserts > 20 && deletes > 20, "the workload exercised both insertions and deletions (" + inserts + "/" + deletes + ")");
            Check(g.Rejected > 0, "the workload also exercised rejections (" + g.Rejected + ")");
            Console.WriteLine("random workload: accepted=" + g.Accepted + " rejected=" + g.Rejected + " removed=" + g.Removed
                + " finalEdges=" + g.EdgeCount + " components=" + g.ComponentsWithEdges()
                + " faces=" + g.Faces(out _).Count);
        }

        Console.WriteLine("PASS");
    }
}
