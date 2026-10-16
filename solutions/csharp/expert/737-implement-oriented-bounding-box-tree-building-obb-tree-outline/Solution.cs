// Task 737 - Implement oriented bounding box tree building (OBB-tree) outline. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// An OBB tree is built from a triangle mesh. Each node is fitted to the *distinct* vertices of its triangles
// (deduplicating matters: shared corners would otherwise be weighted once per incident triangle and tilt the
// covariance): the covariance of those points is eigen-decomposed by a cyclic Jacobi sweep, the eigenvectors
// become the node's orthonormal frame ordered by decreasing eigenvalue, the node centre is the point mean and
// each half-extent is the tight max |dot(p - centre, axis_k)|. A node with more than MaxLeaf triangles is split
// at the median of the triangle centroids projected on its longest half-extent axis; the split is repeated
// until leaves hold MaxLeaf triangles or the depth cap is reached.
// Invariants verified for every node: the frame is orthonormal, each stored (axis, eigenvalue) pair really
// solves Cov * axis = eigenvalue * axis for the node's own points, the half-extents are tight, every triangle
// of the node - and of all its descendants - lies inside the node box, the leaves partition the mesh, and
// NodeCount == 2*Leaves - 1. Quality is measured against the axis-aligned alternative: for a long box rotated
// by a 3-4-5 style orthogonal matrix the fitted box volume must be a small fraction of the AABB volume, and
// fitting the rotated and the unrotated mesh must yield the same volume (rotation invariance).
// Simplifications of this outline: the split is a median split (no surface-area heuristic sweep, no
// partitioning by triangle-box overlap), leaves are not refit after motion, and the fit uses plain double
// arithmetic with 1e-9 tolerances rather than exact predicates - the meshes are constructed so that these
// tolerances are never load-bearing.
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
}

internal struct Vec3
{
    internal double X, Y, Z;

    internal Vec3(double x, double y, double z)
    {
        X = x; Y = y; Z = z;
    }

    internal double this[int k] => k == 0 ? X : k == 1 ? Y : Z;

    public static Vec3 operator +(Vec3 a, Vec3 b) => new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    public static Vec3 operator -(Vec3 a, Vec3 b) => new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    public static Vec3 operator *(Vec3 a, double s) => new Vec3(a.X * s, a.Y * s, a.Z * s);

    internal double Dot(Vec3 o) => X * o.X + Y * o.Y + Z * o.Z;
    internal Vec3 Cross(Vec3 o) => new Vec3(Y * o.Z - Z * o.Y, Z * o.X - X * o.Z, X * o.Y - Y * o.X);
    internal double Length => Math.Sqrt(Dot(this));
    internal Vec3 Normalized
    {
        get
        {
            double length = Length;
            return length < 1e-15 ? new Vec3(1, 0, 0) : this * (1.0 / length);
        }
    }
}

internal sealed class Triangle
{
    internal readonly Vec3 A, B, C;

    internal Triangle(Vec3 a, Vec3 b, Vec3 c)
    {
        A = a; B = b; C = c;
    }

    internal Vec3 Centroid => (A + B + C) * (1.0 / 3.0);
}

internal sealed class ObbNode
{
    internal Vec3 Centre;
    internal readonly Vec3[] Axis = new Vec3[3];
    internal readonly double[] Half = new double[3];
    internal readonly double[] Eigen = new double[3];
    internal double[,] Cov = new double[3, 3];
    internal Vec3 AabbMin, AabbMax;
    internal int Start, Count, Left = -1, Right = -1;

    internal bool IsLeaf => Left < 0;

    internal double Volume => 8.0 * Half[0] * Half[1] * Half[2];

    internal double AabbVolume =>
        (AabbMax.X - AabbMin.X) * (AabbMax.Y - AabbMin.Y) * (AabbMax.Z - AabbMin.Z);

    /// <summary>Signed distance of a point from the box surface: negative inside, positive outside.</summary>
    internal double SignedOutside(Vec3 p)
    {
        double worst = double.NegativeInfinity;
        Vec3 d = p - Centre;
        for (int k = 0; k < 3; k++) worst = Math.Max(worst, Math.Abs(d.Dot(Axis[k])) - Half[k]);
        return worst;
    }
}

internal sealed class ObbTree
{
    internal const int MaxLeaf = 2;

    private readonly Triangle[] _triangles;
    private readonly int[] _order;
    private readonly List<ObbNode> _nodes = new List<ObbNode>();

    internal int MaxDepth;
    internal long Splits, JacobiSweeps;

    internal ObbTree(IList<Triangle> triangles)
    {
        _triangles = new Triangle[triangles.Count];
        for (int i = 0; i < triangles.Count; i++) _triangles[i] = triangles[i];
        _order = new int[_triangles.Length];
        for (int i = 0; i < _order.Length; i++) _order[i] = i;
        if (_triangles.Length > 0) Build(0, _triangles.Length, 0);
    }

    internal int NodeCount => _nodes.Count;
    internal ObbNode? Root => _nodes.Count > 0 ? _nodes[0] : null;
    internal Triangle Triangle(int index) => _triangles[index];
    internal int TriangleCount => _triangles.Length;

    internal List<ObbNode> Leaves()
    {
        var leaves = new List<ObbNode>();
        foreach (ObbNode node in _nodes) if (node.IsLeaf) leaves.Add(node);
        return leaves;
    }

    private int Build(int start, int count, int depth)
    {
        var node = new ObbNode { Start = start, Count = count };
        int me = _nodes.Count;
        _nodes.Add(node);
        Fit(node);
        if (depth > MaxDepth) MaxDepth = depth;
        if (count <= MaxLeaf || depth >= 8) return me;
        int axis = 0;
        if (node.Half[1] > node.Half[axis]) axis = 1;
        if (node.Half[2] > node.Half[axis]) axis = 2;
        Vec3 direction = node.Axis[axis];
        Array.Sort(_order, start, count,
            Comparer<int>.Create((a, b) => (_triangles[a].Centroid.Dot(direction)).CompareTo(_triangles[b].Centroid.Dot(direction))));
        Splits++;
        int mid = count / 2;
        node.Left = Build(start, mid, depth + 1);
        node.Right = Build(start + mid, count - mid, depth + 1);
        return me;
    }

    /// <summary>Distinct vertices of the triangles in [start, start+count).</summary>
    private List<Vec3> Points(int start, int count)
    {
        var seen = new HashSet<(double, double, double)>();
        var points = new List<Vec3>();
        for (int i = start; i < start + count; i++)
        {
            Triangle triangle = _triangles[_order[i]];
            foreach (Vec3 p in new[] { triangle.A, triangle.B, triangle.C })
                if (seen.Add((p.X, p.Y, p.Z))) points.Add(p);
        }
        return points;
    }

    private void Fit(ObbNode node)
    {
        List<Vec3> points = Points(node.Start, node.Count);
        Vec3 centre = new Vec3(0, 0, 0);
        foreach (Vec3 p in points) centre = centre + p;
        centre = centre * (1.0 / points.Count);
        node.Centre = centre;

        double[,] cov = new double[3, 3];
        foreach (Vec3 p in points)
        {
            Vec3 d = p - centre;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    cov[i, j] += d[i] * d[j];
        }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                cov[i, j] /= points.Count;
        node.Cov = cov;

        double[,] vectors = Identity();
        double[] values = Diagonalize(cov, vectors);
        int[] ranked = { 0, 1, 2 };
        Array.Sort(ranked, (a, b) => values[b].CompareTo(values[a]));
        for (int k = 0; k < 3; k++)
        {
            node.Eigen[k] = values[ranked[k]];
            node.Axis[k] = new Vec3(vectors[0, ranked[k]], vectors[1, ranked[k]], vectors[2, ranked[k]]).Normalized;
            double half = 0;
            foreach (Vec3 p in points) half = Math.Max(half, Math.Abs((p - centre).Dot(node.Axis[k])));
            node.Half[k] = half;
        }
        Aabb(node, points);
    }

    private void Aabb(ObbNode node, List<Vec3> points)
    {
        Vec3 low = points[0], high = points[0];
        for (int i = 1; i < points.Count; i++)
        {
            Vec3 p = points[i];
            low = new Vec3(Math.Min(low.X, p.X), Math.Min(low.Y, p.Y), Math.Min(low.Z, p.Z));
            high = new Vec3(Math.Max(high.X, p.X), Math.Max(high.Y, p.Y), Math.Max(high.Z, p.Z));
        }
        node.AabbMin = low;
        node.AabbMax = high;
    }

    /// <summary>Cyclic Jacobi sweep; returns the eigenvalues and overwrites `vectors` with the eigenvectors.</summary>
    private double[] Diagonalize(double[,] matrix, double[,] vectors)
    {
        var a = (double[,])matrix.Clone();
        for (int sweep = 0; sweep < 16; sweep++)
        {
            double off = Math.Abs(a[0, 1]) + Math.Abs(a[0, 2]) + Math.Abs(a[1, 2]);
            if (off < 1e-15) break;
            JacobiSweeps++;
            for (int p = 0; p < 2; p++)
            {
                for (int q = p + 1; q < 3; q++)
                {
                    if (Math.Abs(a[p, q]) < 1e-18) continue;
                    // J(p,q,phi) with J[p,q]=sin, J[q,p]=-sin zeroes the (p,q) entry of J^T A J exactly when
                    // tan(2*phi) = 2*a[p,q] / (a[q,q] - a[p,p]).
                    double phi = 0.5 * Math.Atan2(2 * a[p, q], a[q, q] - a[p, p]);
                    double c = Math.Cos(phi), s = Math.Sin(phi);
                    double[,] r = Identity();
                    r[p, p] = c; r[q, q] = c; r[p, q] = s; r[q, p] = -s;
                    Copy(a, Multiply(Multiply(Transpose(r), a), r));
                    Copy(vectors, Multiply(vectors, r));
                }
            }
        }
        return new[] { a[0, 0], a[1, 1], a[2, 2] };
    }

    private static double[,] Identity()
    {
        var m = new double[3, 3];
        for (int i = 0; i < 3; i++) m[i, i] = 1;
        return m;
    }

    private static double[,] Transpose(double[,] m)
    {
        var t = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                t[i, j] = m[j, i];
        return t;
    }

    private static double[,] Multiply(double[,] x, double[,] y)
    {
        var m = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    m[i, j] += x[i, k] * y[k, j];
        return m;
    }

    private static void Copy(double[,] target, double[,] source)
    {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                target[i, j] = source[i, j];
    }

    /// <summary>Full structural invariant; null when the tree is a consistent OBB tree of the mesh.</summary>
    internal string? InvariantViolation()
    {
        if (_triangles.Length == 0) return _nodes.Count == 0 ? null : "a mesh-less tree must have no node";
        var seen = new int[_triangles.Length];
        var ancestors = new List<ObbNode>();
        string? error = null;
        int visited = 0;
        int leaves = Walk(0, seen, ancestors, ref error, ref visited);
        if (error != null) return error;
        if (visited != _nodes.Count) return "unreachable nodes";
        for (int i = 0; i < seen.Length; i++) if (seen[i] != 1) return "triangle " + i + " appears " + seen[i] + " times";
        if (_nodes.Count != 2 * leaves - 1) return "node count " + _nodes.Count + " != 2*leaves-1";
        return null;
    }

    private int Walk(int index, int[] seen, List<ObbNode> ancestors, ref string? error, ref int visited)
    {
        visited++;
        ObbNode node = _nodes[index];
        CheckFrame(node, ref error);
        CheckFit(node, ref error);
        for (int k = 0; k < 2; k++)
            if (node.Eigen[k] < node.Eigen[k + 1] - 1e-12) error ??= "eigenvalues are not sorted descending";
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++)
                if (Math.Abs(node.Axis[i].Dot(node.Axis[j])) > 1e-9) error ??= "axes are not orthogonal";
        foreach (Vec3 p in Points(node.Start, node.Count))
        {
            if (node.SignedOutside(p) > 1e-9) error ??= "a vertex of this node lies outside its own box";
            foreach (ObbNode ancestor in ancestors)
                if (ancestor.SignedOutside(p) > 1e-9) error ??= "a vertex escapes an ancestor box";
        }
        ancestors.Add(node);
        int leaves;
        if (node.IsLeaf)
        {
            if (node.Count > MaxLeaf) error ??= "leaf holds " + node.Count + " triangles";
            for (int i = node.Start; i < node.Start + node.Count; i++)
            {
                int triangle = _order[i];
                if (triangle < 0 || triangle >= _triangles.Length) { error ??= "leaf index out of range"; continue; }
                seen[triangle]++;
            }
            leaves = 1;
        }
        else
        {
            if (_nodes[node.Left].Count + _nodes[node.Right].Count != node.Count) error ??= "children do not partition the parent";
            leaves = Walk(node.Left, seen, ancestors, ref error, ref visited)
                   + Walk(node.Right, seen, ancestors, ref error, ref visited);
        }
        ancestors.RemoveAt(ancestors.Count - 1);
        return leaves;
    }

    private static void CheckFrame(ObbNode node, ref string? error)
    {
        for (int k = 0; k < 3; k++)
        {
            if (Math.Abs(node.Axis[k].Length - 1) > 1e-9) error ??= "axis " + k + " is not unit length";
            if (double.IsNaN(node.Half[k]) || double.IsNaN(node.Centre.X)) error ??= "NaN in the fitted box";
        }
    }

    /// <summary>
    /// The three real invariants of the fit, all recomputed from the node's own points: the centre is their
    /// mean, each half-extent is the tight maximum of |dot(p - centre, axis_k)|, and the stored covariance and
    /// (axis, eigenvalue) pairs satisfy Cov * axis = eigenvalue * axis.
    /// </summary>
    private void CheckFit(ObbNode node, ref string? error)
    {
        List<Vec3> points = Points(node.Start, node.Count);
        Vec3 mean = new Vec3(0, 0, 0);
        foreach (Vec3 p in points) mean = mean + p;
        mean = mean * (1.0 / points.Count);
        if ((mean - node.Centre).Length > 1e-9) error ??= "centre is not the mean of the node points";

        var cov = new double[3, 3];
        Vec3 low = points[0], high = points[0];
        foreach (Vec3 p in points)
        {
            Vec3 d = p - node.Centre;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    cov[i, j] += d[i] * d[j];
            low = new Vec3(Math.Min(low.X, p.X), Math.Min(low.Y, p.Y), Math.Min(low.Z, p.Z));
            high = new Vec3(Math.Max(high.X, p.X), Math.Max(high.Y, p.Y), Math.Max(high.Z, p.Z));
        }
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                cov[i, j] /= points.Count;
                if (Math.Abs(cov[i, j] - node.Cov[i, j]) > 1e-9) error ??= "stored covariance is stale";
            }
        if ((low - node.AabbMin).Length > 1e-9 || (high - node.AabbMax).Length > 1e-9)
            error ??= "stored AABB is not the tight bounding box of the node points";

        for (int k = 0; k < 3; k++)
        {
            double tight = 0;
            foreach (Vec3 p in points) tight = Math.Max(tight, Math.Abs((p - node.Centre).Dot(node.Axis[k])));
            if (Math.Abs(tight - node.Half[k]) > 1e-9) error ??= "half-extent " + k + " is not tight";
            var product = new Vec3(
                node.Cov[0, 0] * node.Axis[k].X + node.Cov[0, 1] * node.Axis[k].Y + node.Cov[0, 2] * node.Axis[k].Z,
                node.Cov[1, 0] * node.Axis[k].X + node.Cov[1, 1] * node.Axis[k].Y + node.Cov[1, 2] * node.Axis[k].Z,
                node.Cov[2, 0] * node.Axis[k].X + node.Cov[2, 1] * node.Axis[k].Y + node.Cov[2, 2] * node.Axis[k].Z);
            Vec3 scaled = node.Axis[k] * node.Eigen[k];
            double residual = Math.Max(Math.Abs(product.X - scaled.X),
                Math.Max(Math.Abs(product.Y - scaled.Y), Math.Abs(product.Z - scaled.Z)));
            if (residual > 1e-9 * Math.Max(1.0, node.Eigen[k])) error ??= "axis " + k + " is not an eigenvector";
        }
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Orthogonal 3-4-5 style matrix (divided by 3) used to give the test mesh a skewed pose.</summary>
    private static Vec3 Rotate(Vec3 p)
    {
        const double third = 1.0 / 3.0;
        return new Vec3(
            (p.X + 2 * p.Y + 2 * p.Z) * third,
            (2 * p.X + p.Y - 2 * p.Z) * third,
            (2 * p.X - 2 * p.Y + p.Z) * third);
    }

    private static List<Triangle> BoxMesh(double hx, double hy, double hz, bool rotated)
    {
        var corner = new Vec3[8];
        for (int b = 0; b < 8; b++)
        {
            var p = new Vec3((b & 1) == 0 ? -hx : hx, (b & 2) == 0 ? -hy : hy, (b & 4) == 0 ? -hz : hz);
            corner[b] = rotated ? Rotate(p) : p;
        }
        int[][] faces =
        {
            new[] { 0, 2, 6, 4 }, new[] { 1, 5, 7, 3 }, new[] { 0, 4, 5, 1 },
            new[] { 2, 3, 7, 6 }, new[] { 0, 1, 3, 2 }, new[] { 4, 6, 7, 5 }
        };
        var mesh = new List<Triangle>();
        foreach (int[] face in faces)
        {
            mesh.Add(new Triangle(corner[face[0]], corner[face[1]], corner[face[2]]));
            mesh.Add(new Triangle(corner[face[0]], corner[face[2]], corner[face[3]]));
        }
        return mesh;
    }

    private static List<Triangle> FlatMesh()
    {
        var mesh = new List<Triangle>();
        for (int i = 0; i < 6; i++)
            mesh.Add(new Triangle(new Vec3(i, 0, 0), new Vec3(i + 1, 0, 0), new Vec3(i, 0, 1)));
        return mesh;
    }

    private static List<Triangle> CloudMesh(int count, ulong seed)
    {
        var mesh = new List<Triangle>();
        ulong state = seed;
        for (int i = 0; i < count; i++)
        {
            var a = new Vec3(SplitMix.NextInt(ref state, 40) - 20, SplitMix.NextInt(ref state, 40) - 20, SplitMix.NextInt(ref state, 40) - 20);
            var b = a + new Vec3(SplitMix.NextInt(ref state, 9) - 4, SplitMix.NextInt(ref state, 9) - 4, SplitMix.NextInt(ref state, 9) - 4);
            var c = a + new Vec3(SplitMix.NextInt(ref state, 9) - 4, SplitMix.NextInt(ref state, 9) - 4, SplitMix.NextInt(ref state, 9) - 4);
            mesh.Add(new Triangle(a, b, c));
        }
        return mesh;
    }

    private static int TotalLeafTriangles(ObbTree tree)
    {
        int total = 0;
        foreach (ObbNode leaf in tree.Leaves()) total += leaf.Count;
        return total;
    }

    public static void Main()
    {
        // ---- degenerate meshes -----------------------------------------------------------------------
        var none = new ObbTree(new List<Triangle>());
        Check(none.NodeCount == 0 && none.Root == null && none.InvariantViolation() == null,
            "an empty mesh produces no node and a trivially clean invariant");

        var single = new ObbTree(new List<Triangle> { new Triangle(new Vec3(0, 0, 0), new Vec3(4, 0, 0), new Vec3(0, 3, 0)) });
        Check(single.NodeCount == 1 && single.Root!.IsLeaf && single.InvariantViolation() == null,
            "one triangle produces a single leaf that contains it: " + (single.InvariantViolation() ?? "ok"));
        Check(single.Root!.Volume < 1e-9, "a flat primitive gets a flat box (zero volume)");

        var flat = new ObbTree(FlatMesh());
        Check(flat.InvariantViolation() == null && flat.Root!.Volume < 1e-9,
            "a plane of triangles fits a degenerate box of zero volume without breaking the invariant");
        Check(TotalLeafTriangles(flat) == flat.TriangleCount && flat.NodeCount == 2 * flat.Leaves().Count - 1,
            "leaves partition the flat mesh and the node count follows 2*leaves-1");

        var repeated = new ObbTree(new List<Triangle>
        {
            new Triangle(new Vec3(1, 1, 1), new Vec3(2, 1, 1), new Vec3(1, 2, 1)),
            new Triangle(new Vec3(1, 1, 1), new Vec3(2, 1, 1), new Vec3(1, 2, 1)),
            new Triangle(new Vec3(1, 1, 1), new Vec3(2, 1, 1), new Vec3(1, 2, 1))
        });
        Check(repeated.InvariantViolation() == null && repeated.Root!.Volume < 1e-9,
            "three copies of the same triangle fit a flat zero-volume box");

        // ---- the rotated long box: the OBB must beat the AABB ------------------------------------------
        var plain = new ObbTree(BoxMesh(3, 1, 0.5, rotated: false));
        var rotated = new ObbTree(BoxMesh(3, 1, 0.5, rotated: true));
        Check(rotated.InvariantViolation() == null, "the rotated box mesh yields a consistent tree: "
            + (rotated.InvariantViolation() ?? "ok"));
        Check(Math.Abs(rotated.Root!.Half[0] - 3) < 1e-6 && Math.Abs(rotated.Root!.Half[1] - 1) < 1e-6
            && Math.Abs(rotated.Root!.Half[2] - 0.5) < 1e-6,
            "the fitted box recovers the box's own half-extents 3, 1, 0.5 in decreasing eigenvalue order");
        Check(Math.Abs(rotated.Root!.Volume - 12) < 1e-6,
            "the fitted box volume equals the true box volume 12");
        Check(rotated.Root!.Volume < 0.2 * rotated.Root!.AabbVolume,
            "the OBB volume " + Math.Round(rotated.Root!.Volume, 4) + " is a small fraction of the AABB volume "
            + Math.Round(rotated.Root!.AabbVolume, 4));
        Check(Math.Abs(rotated.Root!.Volume - plain.Root!.Volume) < 1e-6,
            "fitting the rotated and the unrotated mesh yields the same volume (rotation invariance)");
        Check(TotalLeafTriangles(rotated) == rotated.TriangleCount && rotated.NodeCount == 2 * rotated.Leaves().Count - 1
            && rotated.MaxDepth >= 1,
            "the box tree is a real tree: " + rotated.NodeCount + " nodes, " + rotated.Leaves().Count + " leaves, depth " + rotated.MaxDepth);

        // ---- a randomised cloud: invariants only ------------------------------------------------------
        bool cloudsClean = true, leavesBounded = true, subdivisionCompact = true;
        string? firstFailure = null;
        double worstRatio = 0;
        for (int round = 0; round < 4; round++)
        {
            var tree = new ObbTree(CloudMesh(20 + 10 * round, 0x73700000UL + (ulong)round * 2027UL));
            if (tree.InvariantViolation() is string violation)
            {
                cloudsClean = false;
                firstFailure ??= "round " + round + ": " + violation;
            }
            if (TotalLeafTriangles(tree) != tree.TriangleCount || tree.NodeCount != 2 * tree.Leaves().Count - 1)
            {
                leavesBounded = false;
                firstFailure ??= "round " + round + ": leaf partition broken";
            }
            if (tree.MaxDepth < 2) leavesBounded = false;
            double leafVolume = 0;
            foreach (ObbNode leaf in tree.Leaves())
            {
                if (leaf.Count > ObbTree.MaxLeaf) leavesBounded = false;
                leafVolume += leaf.Volume;
            }
            // The subdivision must stay compact: the leaf boxes together cannot exceed the root AABB, which
            // is the trivial upper bound of any spatial subdivision of the same points.
            if (!(leafVolume < tree.Root!.AabbVolume)) subdivisionCompact = false;
            worstRatio = Math.Max(worstRatio, tree.Root!.Volume / tree.Root!.AabbVolume);
        }
        Check(cloudsClean, "all four random clouds satisfy the OBB invariant: " + (firstFailure ?? "ok"));
        Check(leavesBounded, "every random cloud partitions its triangles over leaves of at most "
            + ObbTree.MaxLeaf + " triangles and is at least two levels deep");
        Check(subdivisionCompact, "the leaf boxes together stay inside the root AABB volume (worst OBB/AABB "
            + Math.Round(worstRatio, 4) + ")");

        Console.WriteLine("nodes=" + rotated.NodeCount + " leaves=" + rotated.Leaves().Count
            + " depth=" + rotated.MaxDepth + " splits=" + rotated.Splits
            + " jacobiSweeps=" + rotated.JacobiSweeps
            + " rootVolume=" + Math.Round(rotated.Root!.Volume, 4)
            + " rootAabbVolume=" + Math.Round(rotated.Root!.AabbVolume, 4));
        Console.WriteLine("PASS");
    }
}
