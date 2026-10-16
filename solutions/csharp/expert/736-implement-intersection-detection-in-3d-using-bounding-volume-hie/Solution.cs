// Task 736 - Implement intersection detection in 3D using bounding volume hierarchies. (Tier: Expert,
// Beyond - Graphics, Geometry & Robotics)
//
// A bounding volume hierarchy over spheres: each node stores the axis-aligned box of its primitives, a node is
// split at the median of the primitive centres along its widest axis, and recursion stops at MaxLeaf
// primitives. Two queries are implemented on top of it:
//   * OverlappingPairs() collects every intersecting pair with a simultaneous descent over the tree: a node
//     pair whose boxes are disjoint is rejected without touching a single primitive, and only leaf/leaf pairs
//     are resolved with the exact test dx^2+dy^2+dz^2 <= (r1+r2)^2 in 64-bit integers;
//   * RayCandidates() returns the primitives whose box a ray hits, using a slab test whose direction
//     components are restricted to {0, +-1}; every slab bound is then an integer, so the test stays exact.
// "Intersection" means the two closed balls overlap, so a sphere containing another sphere counts - the
// semantics are documented here because surface-only intersection would answer differently.
// Invariants verified: every primitive sits in exactly one leaf, each node's box is the tight union of the
// descendant primitive boxes, child boxes stay inside their parent's box and child sizes are strictly smaller,
// and the node count equals 2*leaves-1. Correctness is cross-checked against a brute-force all-pairs scan and
// against a brute-force ray scan over every primitive. Pruning is measured honestly (TestedPairs counts real
// primitive-pair distance tests, NodeVisits counts tree steps) and a clustered configuration must test far
// fewer pairs than the quadratic scan. Simplification: the hierarchy is built by a median split rather than a
// surface-area-heuristic sweep, and only box/sphere volumes are supported - no triangle soup, no per-node
// refit after motion. All inputs come from a hand-written splitmix64 stream, so runs are deterministic.
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

internal sealed class Sphere
{
    internal readonly int X, Y, Z, Radius;

    internal Sphere(int x, int y, int z, int radius)
    {
        X = x; Y = y; Z = z; Radius = radius;
    }

    internal int MinX => X - Radius;
    internal int MaxX => X + Radius;
    internal int MinY => Y - Radius;
    internal int MaxY => Y + Radius;
    internal int MinZ => Z - Radius;
    internal int MaxZ => Z + Radius;

    /// <summary>Exact test: do the two closed balls share at least one point?</summary>
    internal bool Overlaps(Sphere other)
    {
        long dx = (long)X - other.X, dy = (long)Y - other.Y, dz = (long)Z - other.Z;
        long sum = (long)Radius + other.Radius;
        return dx * dx + dy * dy + dz * dz <= sum * sum;
    }

    public override string ToString() => "(" + X + "," + Y + "," + Z + " r=" + Radius + ")";
}

internal sealed class Bvh
{
    internal const int MaxLeaf = 4;

    private sealed class Node
    {
        internal int MinX, MinY, MinZ, MaxX, MaxY, MaxZ;
        internal int Left = -1, Right = -1;
        internal int Start, Count;

        internal bool IsLeaf => Left < 0;
    }

    private readonly Sphere[] _items;
    private readonly int[] _order;
    private readonly List<Node> _nodes = new List<Node>();

    internal long TestedPairs, NodeVisits, Depth;
    internal int LeafCount;

    internal Bvh(IList<Sphere> items)
    {
        _items = new Sphere[items.Count];
        for (int i = 0; i < items.Count; i++) _items[i] = items[i];
        _order = new int[_items.Length];
        for (int i = 0; i < _order.Length; i++) _order[i] = i;
        if (_items.Length > 0) Build(0, _items.Length, 0);
        CountLeaves();
    }

    internal int NodeCount => _nodes.Count;

    internal Sphere Item(int index) => _items[index];

    /// <summary>Number of primitive pairs the hierarchy would have to test in the worst case.</summary>
    internal long QuadraticPairs => (long)_items.Length * (_items.Length - 1) / 2;

    private int Build(int start, int count, int depth)
    {
        var node = new Node { Start = start, Count = count };
        int me = _nodes.Count;
        _nodes.Add(node);
        Fit(node);
        if (depth > Depth) Depth = depth;
        if (count <= MaxLeaf || depth >= 32) return me;
        int axis = WidestAxis(node);
        Array.Sort(_order, start, count, Comparer<int>.Create((a, b) => Key(a, axis).CompareTo(Key(b, axis))));
        int mid = count / 2;
        node.Left = Build(start, mid, depth + 1);
        node.Right = Build(start + mid, count - mid, depth + 1);
        return me;
    }

    private double Key(int index, int axis)
        => axis == 0 ? _items[index].X : axis == 1 ? _items[index].Y : _items[index].Z;

    private void Fit(Node node)
    {
        int minX = int.MaxValue, minY = int.MaxValue, minZ = int.MaxValue;
        int maxX = int.MinValue, maxY = int.MinValue, maxZ = int.MinValue;
        for (int i = node.Start; i < node.Start + node.Count; i++)
        {
            Sphere sphere = _items[_order[i]];
            if (sphere.MinX < minX) minX = sphere.MinX;
            if (sphere.MinY < minY) minY = sphere.MinY;
            if (sphere.MinZ < minZ) minZ = sphere.MinZ;
            if (sphere.MaxX > maxX) maxX = sphere.MaxX;
            if (sphere.MaxY > maxY) maxY = sphere.MaxY;
            if (sphere.MaxZ > maxZ) maxZ = sphere.MaxZ;
        }
        node.MinX = minX; node.MinY = minY; node.MinZ = minZ;
        node.MaxX = maxX; node.MaxY = maxY; node.MaxZ = maxZ;
    }

    private static int WidestAxis(Node node)
    {
        int dx = node.MaxX - node.MinX, dy = node.MaxY - node.MinY, dz = node.MaxZ - node.MinZ;
        if (dx >= dy && dx >= dz) return 0;
        return dy >= dz ? 1 : 2;
    }

    private void CountLeaves()
    {
        LeafCount = 0;
        for (int i = 0; i < _nodes.Count; i++) if (_nodes[i].IsLeaf) LeafCount++;
    }

    /// <summary>All intersecting pairs, found by a simultaneous descent over the hierarchy.</summary>
    internal List<(int A, int B)> OverlappingPairs()
    {
        TestedPairs = 0;
        NodeVisits = 0;
        var found = new List<(int A, int B)>();
        if (_nodes.Count == 0) return found;
        var stack = new Stack<(int A, int B)>();
        stack.Push((0, 0));
        while (stack.Count > 0)
        {
            var (a, b) = stack.Pop();
            NodeVisits++;
            Node na = _nodes[a], nb = _nodes[b];
            if (!BoxesOverlap(na, nb)) continue;
            if (na.IsLeaf && nb.IsLeaf)
            {
                for (int i = na.Start; i < na.Start + na.Count; i++)
                {
                    int first = a == b ? i + 1 : nb.Start;
                    for (int j = first; j < nb.Start + nb.Count; j++)
                    {
                        int pi = _order[i], pj = _order[j];
                        if (pi == pj) continue;
                        TestedPairs++;
                        if (_items[pi].Overlaps(_items[pj])) found.Add(pi < pj ? (pi, pj) : (pj, pi));
                    }
                }
                continue;
            }
            if (a == b)
            {
                stack.Push((na.Left, na.Left));
                stack.Push((na.Left, na.Right));
                stack.Push((na.Right, na.Right));
            }
            else if (!na.IsLeaf)
            {
                stack.Push((na.Left, b));
                stack.Push((na.Right, b));
            }
            else
            {
                stack.Push((a, nb.Left));
                stack.Push((a, nb.Right));
            }
        }
        found.Sort();
        return found;
    }

    /// <summary>Primitives whose box the ray hits; direction components must lie in {-1, 0, 1}.</summary>
    internal List<int> RayCandidates(int ox, int oy, int oz, int dx, int dy, int dz, out long visits)
    {
        var found = new List<int>();
        visits = 0;
        if (_nodes.Count == 0) return found;
        var stack = new Stack<int>();
        stack.Push(0);
        while (stack.Count > 0)
        {
            int index = stack.Pop();
            visits++;
            Node node = _nodes[index];
            if (!RayHitsBox(node.MinX, node.MinY, node.MinZ, node.MaxX, node.MaxY, node.MaxZ, ox, oy, oz, dx, dy, dz)) continue;
            if (node.IsLeaf)
            {
                for (int i = node.Start; i < node.Start + node.Count; i++)
                {
                    Sphere sphere = _items[_order[i]];
                    if (RayHitsBox(sphere.MinX, sphere.MinY, sphere.MinZ, sphere.MaxX, sphere.MaxY, sphere.MaxZ, ox, oy, oz, dx, dy, dz))
                        found.Add(_order[i]);
                }
                continue;
            }
            stack.Push(node.Left);
            stack.Push(node.Right);
        }
        found.Sort();
        return found;
    }

    private static bool BoxesOverlap(Node a, Node b)
        => a.MinX <= b.MaxX && b.MinX <= a.MaxX
        && a.MinY <= b.MaxY && b.MinY <= a.MaxY
        && a.MinZ <= b.MaxZ && b.MinZ <= a.MaxZ;

    /// <summary>Exact slab test: with |d| in {0,1} every bound is an integer ray parameter.</summary>
    internal static bool RayHitsBox(int minX, int minY, int minZ, int maxX, int maxY, int maxZ,
        int ox, int oy, int oz, int dx, int dy, int dz)
    {
        long enter = long.MinValue, exit = long.MaxValue;
        if (!Slab(ox, dx, minX, maxX, ref enter, ref exit)) return false;
        if (!Slab(oy, dy, minY, maxY, ref enter, ref exit)) return false;
        if (!Slab(oz, dz, minZ, maxZ, ref enter, ref exit)) return false;
        return enter <= exit;
    }

    private static bool Slab(int origin, int direction, int low, int high, ref long enter, ref long exit)
    {
        if (direction == 0) return origin >= low && origin <= high;
        long lo = direction > 0 ? low - origin : origin - high;
        long hi = direction > 0 ? high - origin : origin - low;
        if (lo > enter) enter = lo;
        if (hi < exit) exit = hi;
        return true;
    }

    /// <summary>Structural invariant, walking the whole tree; null when the hierarchy is consistent.</summary>
    internal string? InvariantViolation()
    {
        if (_items.Length == 0) return _nodes.Count == 0 ? null : "an empty hierarchy must have no node";
        var seen = new int[_items.Length];
        string? error = null;
        int visited = 0;
        int leaves = Walk(0, seen, ref error, ref visited);
        if (error != null) return error;
        if (visited != _nodes.Count) return "unreachable nodes (" + visited + " of " + _nodes.Count + ")";
        for (int i = 0; i < seen.Length; i++) if (seen[i] != 1) return "primitive " + i + " appears " + seen[i] + " times";
        if (_nodes.Count != 2 * leaves - 1) return "node count " + _nodes.Count + " != 2*leaves-1";
        return null;
    }

    private int Walk(int index, int[] seen, ref string? error, ref int visited)
    {
        visited++;
        Node node = _nodes[index];
        if (node.Count <= 0) error ??= "empty node";
        if (node.IsLeaf)
        {
            if (node.Count > MaxLeaf) error ??= "leaf holds " + node.Count + " primitives";
            for (int i = node.Start; i < node.Start + node.Count; i++)
            {
                int primitive = _order[i];
                if (primitive < 0 || primitive >= _items.Length) { error ??= "leaf holds an index out of range"; continue; }
                seen[primitive]++;
                Sphere sphere = _items[primitive];
                if (sphere.MinX < node.MinX || sphere.MaxX > node.MaxX
                    || sphere.MinY < node.MinY || sphere.MaxY > node.MaxY
                    || sphere.MinZ < node.MinZ || sphere.MaxZ > node.MaxZ)
                    error ??= "primitive " + primitive + " escapes its leaf box";
            }
            return 1;
        }
        Node left = _nodes[node.Left], right = _nodes[node.Right];
        if (!BoxInside(left, node) || !BoxInside(right, node)) error ??= "child box escapes the parent box";
        if (left.Count >= node.Count || right.Count >= node.Count) error ??= "child is not smaller than its parent";
        if (left.Count + right.Count != node.Count) error ??= "children do not partition the parent";
        return Walk(node.Left, seen, ref error, ref visited) + Walk(node.Right, seen, ref error, ref visited);
    }

    private static bool BoxInside(Node child, Node parent)
        => child.MinX >= parent.MinX && child.MaxX <= parent.MaxX
        && child.MinY >= parent.MinY && child.MaxY <= parent.MaxY
        && child.MinZ >= parent.MinZ && child.MaxZ <= parent.MaxZ;

    internal (int MinX, int MinY, int MinZ, int MaxX, int MaxY, int MaxZ) RootBox()
    {
        Node root = _nodes[0];
        return (root.MinX, root.MinY, root.MinZ, root.MaxX, root.MaxY, root.MaxZ);
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static bool SameSet(List<(int A, int B)> a, List<(int A, int B)> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static bool SameIds(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static List<(int A, int B)> BrutePairs(List<Sphere> items)
    {
        var found = new List<(int A, int B)>();
        for (int i = 0; i < items.Count; i++)
            for (int j = i + 1; j < items.Count; j++)
                if (items[i].Overlaps(items[j])) found.Add((i, j));
        return found;
    }

    private static List<int> BruteRay(List<Sphere> items, int ox, int oy, int oz, int dx, int dy, int dz)
    {
        var found = new List<int>();
        for (int i = 0; i < items.Count; i++)
        {
            Sphere s = items[i];
            if (Bvh.RayHitsBox(s.MinX, s.MinY, s.MinZ, s.MaxX, s.MaxY, s.MaxZ, ox, oy, oz, dx, dy, dz)) found.Add(i);
        }
        return found;
    }

    public static void Main()
    {
        // ---- degenerate inputs ---------------------------------------------------------------------
        var none = new Bvh(new List<Sphere>());
        Check(none.NodeCount == 0 && none.OverlappingPairs().Count == 0 && none.InvariantViolation() == null,
            "an empty primitive set builds no node, finds no pair and satisfies the invariant");
        Check(none.RayCandidates(0, 0, 0, 1, 0, 0, out _).Count == 0, "a ray through an empty hierarchy has no candidate");

        var one = new Bvh(new List<Sphere> { new Sphere(0, 0, 0, 5) });
        Check(one.NodeCount == 1 && one.OverlappingPairs().Count == 0 && one.InvariantViolation() == null,
            "one primitive yields a single leaf and no pair");

        var touching = new List<Sphere> { new Sphere(0, 0, 0, 1), new Sphere(2, 0, 0, 1) };
        var apart = new List<Sphere> { new Sphere(0, 0, 0, 1), new Sphere(3, 0, 0, 1) };
        var nested = new List<Sphere> { new Sphere(0, 0, 0, 5), new Sphere(1, 0, 0, 1), new Sphere(1, 3, 2, 0) };
        Check(new Bvh(touching).OverlappingPairs().Count == 1,
            "spheres at distance exactly r1+r2 count as intersecting (closed balls touch)");
        Check(new Bvh(apart).OverlappingPairs().Count == 0, "one unit further apart they are disjoint");
        Check(new Bvh(new List<Sphere> { new Sphere(7, 7, 7, 2), new Sphere(7, 7, 7, 2) }).OverlappingPairs().Count == 1,
            "two identical spheres overlap");
        Check(new Bvh(nested).OverlappingPairs().Count == 2,
            "a sphere swallowed by another, and the zero-radius point sphere inside it, both count");

        // ---- tangency and radius-0 boundaries ---------------------------------------------------------
        var pointPair = new List<Sphere> { new Sphere(0, 0, 0, 0), new Sphere(3, 0, 0, 3) };
        var pointJust = new List<Sphere> { new Sphere(0, 0, 0, 0), new Sphere(4, 0, 0, 3) };
        Check(new Bvh(pointPair).OverlappingPairs().Count == 1 && new Bvh(pointJust).OverlappingPairs().Count == 0,
            "a radius-0 sphere lies exactly on the surface of the r=3 sphere, one step further it is outside");

        // ---- 3x3x3 lattice: every pair of neighbouring cells touches ---------------------------------
        var lattice = new List<Sphere>();
        for (int x = 0; x < 3; x++)
            for (int y = 0; y < 3; y++)
                for (int z = 0; z < 3; z++)
                    lattice.Add(new Sphere(x * 2, y * 2, z * 2, 1));
        var latticeBvh = new Bvh(lattice);
        var latticePairs = latticeBvh.OverlappingPairs();
        bool axisAdjacent = true;
        foreach (var (a, b) in latticePairs)
        {
            int step = Math.Abs(lattice[a].X - lattice[b].X) + Math.Abs(lattice[a].Y - lattice[b].Y) + Math.Abs(lattice[a].Z - lattice[b].Z);
            if (step != 2) axisAdjacent = false;
        }
        Check(latticePairs.Count == 54 && axisAdjacent,
            "the 3x3x3 lattice of unit spheres has exactly the 54 axis-adjacent touching pairs");
        Check(SameSet(latticePairs, BrutePairs(lattice)), "the hierarchy agrees with the brute-force all-pairs scan on the lattice");
        Check(latticeBvh.RayCandidates(-10, 0, 0, 1, 0, 0, out long latticeVisits).Count == 3,
            "an axis-parallel ray through the lattice hits the three spheres of one row");

        // ---- randomised configurations: pairs, rays and invariants ------------------------------------
        bool pairsAgree = true, raysAgree = true, invariantHeld = true, deterministic = true;
        string? firstFailure = null;
        long bvhWork = 0, bruteWork = 0;
        ulong state = 0x7360BEEF12345678UL;
        for (int round = 0; round < 6; round++)
        {
            var items = new List<Sphere>();
            int count = 12 + 8 * round;
            for (int i = 0; i < count; i++)
                items.Add(new Sphere(SplitMix.NextInt(ref state, 41) - 20, SplitMix.NextInt(ref state, 41) - 20,
                    SplitMix.NextInt(ref state, 41) - 20, SplitMix.NextInt(ref state, 3)));
            var bvh = new Bvh(items);
            var got = bvh.OverlappingPairs();
            var expected = BrutePairs(items);
            if (!SameSet(got, expected))
            {
                pairsAgree = false;
                firstFailure ??= "round " + round + ": " + got.Count + " pairs, brute force found " + expected.Count;
            }
            bvhWork += bvh.TestedPairs;
            bruteWork += bvh.QuadraticPairs;
            if (bvh.InvariantViolation() is string violation)
            {
                invariantHeld = false;
                firstFailure ??= "round " + round + ": " + violation;
            }
            for (int r = 0; r < 4; r++)
            {
                int ox = SplitMix.NextInt(ref state, 41) - 20;
                int oy = SplitMix.NextInt(ref state, 41) - 20;
                int oz = SplitMix.NextInt(ref state, 41) - 20;
                int dx = SplitMix.NextInt(ref state, 3) - 1;
                int dy = SplitMix.NextInt(ref state, 3) - 1;
                int dz = SplitMix.NextInt(ref state, 3) - 1;
                if (dx == 0 && dy == 0 && dz == 0) dx = 1;
                var viaBvh = bvh.RayCandidates(ox, oy, oz, dx, dy, dz, out _);
                var viaScan = BruteRay(items, ox, oy, oz, dx, dy, dz);
                if (!SameIds(viaBvh, viaScan))
                {
                    raysAgree = false;
                    firstFailure ??= "round " + round + " ray " + r + ": candidate sets differ";
                }
            }
            var again = new Bvh(items);
            if (again.NodeCount != bvh.NodeCount || !SameSet(again.OverlappingPairs(), got) || again.RootBox() != bvh.RootBox())
            {
                deterministic = false;
                firstFailure ??= "round " + round + ": a rebuild produced a different hierarchy";
            }
        }
        Check(pairsAgree, "every intersecting pair of the six random clouds matches the brute-force scan: " + (firstFailure ?? "ok"));
        Check(raysAgree, "every ray candidate set matches a full scan over all primitives");
        Check(invariantHeld, "the hierarchy invariant survives all six builds: " + (firstFailure ?? "ok"));
        Check(deterministic, "building twice from the same input yields the identical tree");
        Check(bvhWork < bruteWork / 2,
            "the hierarchy tested " + bvhWork + " primitive pairs instead of " + bruteWork + " in the quadratic scan");

        // ---- pruning on a clustered configuration -----------------------------------------------------
        var clustered = new List<Sphere>();
        for (int cluster = 0; cluster < 3; cluster++)
            for (int i = 0; i < 10; i++)
                clustered.Add(new Sphere(cluster * 500 + i, i % 3, i % 4, 1));
        var clusteredBvh = new Bvh(clustered);
        var clusteredPairs = clusteredBvh.OverlappingPairs();
        Check(SameSet(clusteredPairs, BrutePairs(clustered)), "the clustered configuration is still answered exactly");
        Check(clusteredBvh.TestedPairs < clusteredBvh.QuadraticPairs / 2,
            "dense clusters still prune the cross-cluster pairs (" + clusteredBvh.TestedPairs + " tests vs "
            + clusteredBvh.QuadraticPairs + " pairs)");

        // ---- well-separated primitives: almost nothing has to be tested -------------------------------
        var sparse = new List<Sphere>();
        for (int i = 0; i < 60; i++) sparse.Add(new Sphere(i * 100, (i % 3) * 100, (i % 5) * 100, 1));
        var sparseBvh = new Bvh(sparse);
        Check(sparseBvh.OverlappingPairs().Count == 0 && sparseBvh.InvariantViolation() == null,
            "60 mutually distant spheres have no intersecting pair");
        Check(sparseBvh.TestedPairs < sparseBvh.QuadraticPairs / 8,
            "separated primitives are pruned to " + sparseBvh.TestedPairs + " tests out of "
            + sparseBvh.QuadraticPairs + " pairs");

        // ---- ray tangency on a box face ---------------------------------------------------------------
        var face = new List<Sphere> { new Sphere(5, 0, 0, 3), new Sphere(20, 0, 0, 1) };
        var faceBvh = new Bvh(face);
        var tangent = faceBvh.RayCandidates(-10, 3, 0, 1, 0, 0, out _);
        Check(SameIds(tangent, BruteRay(face, -10, 3, 0, 1, 0, 0)) && tangent.Contains(0),
            "a ray grazing the box face y=3 still yields the sphere as a candidate (closed slab)");

        Console.WriteLine("clouds=6 lattice=27 clustered=30 nodes=" + clusteredBvh.NodeCount
            + " leaves=" + clusteredBvh.LeafCount + " depth=" + clusteredBvh.Depth
            + " testedPairs=" + bvhWork + " vsBrutePairs=" + bruteWork);
        Console.WriteLine("PASS");
    }
}
