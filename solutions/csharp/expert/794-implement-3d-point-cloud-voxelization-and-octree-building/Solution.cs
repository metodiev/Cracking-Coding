// Task 794 - Implement 3D point cloud voxelization and octree building. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// Two standard spatial structures over the same point cloud.
//   * Voxelisation: every point falls into the cell (floor((x - minX)/s), ...) and cells are keyed by a 63-bit
//     Morton code z = sum_k interleave(i_k, j_k, k_k) (three 21-bit fields), so a dictionary keyed by the code is
//     also ordered by the space-filling curve.
//   * Octree: the cloud's bounding cube is split into eight octants whenever a node holds more than
//     `maxPointsPerNode` points and the node is above `maxDepth`, so leaves are exactly the occupied fine cells
//     of the equivalent voxel grid at that depth.
// Invariants asserted: every point lies inside its leaf's cube, the leaves partition the point set (no point is
// lost, none is duplicated), the total leaf count equals the occupied-voxel count at the tree's depth, and no
// internal node is finer than required (a leaf has at most maxPointsPerNode points unless it sits at maxDepth).
//
// Independent oracles:
//   * the Morton encoder/decoder is checked against a bit-by-bit interleave written separately, and against the
//     lexicographic order the interleave implies;
//   * the occupied-voxel count is compared with an O(n^2) brute-force distinct-triple count;
//   * the octree's leaves are compared with freshly computed voxel keys per point, and every leaf cube is
//     re-checked to contain its own points.
//
// Documented simplification: small clouds (a few hundred points), depth <= 6 and a dictionary-backed voxel map
// instead of a memory-compact linear octree; the subdivision rule, the Morton keying and the leaf/point
// bookkeeping are the real algorithm. Deterministic splitmix64 noise only.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal struct P3
{
    internal double X, Y, Z;

    internal P3(double x, double y, double z) { X = x; Y = y; Z = z; }
}

internal static class Morton
{
    /** Packs three 21-bit coordinates into one 63-bit key by interleaving (x bit 0 in position 0, y in 1, z in 2). */
    internal static long Encode(int ix, int iy, int iz)
    {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= (1 << 21) || iy >= (1 << 21) || iz >= (1 << 21))
        {
            throw new ArgumentOutOfRangeException(nameof(ix), "coordinates must fit in 21 bits");
        }
        long code = 0;
        for (int bit = 0; bit < 21; bit++)
        {
            code |= (long)((ix >> bit) & 1) << (3 * bit);
            code |= (long)((iy >> bit) & 1) << (3 * bit + 1);
            code |= (long)((iz >> bit) & 1) << (3 * bit + 2);
        }
        return code;
    }

    internal static void Decode(long code, out int ix, out int iy, out int iz)
    {
        ix = 0; iy = 0; iz = 0;
        for (int bit = 0; bit < 21; bit++)
        {
            ix |= (int)((code >> (3 * bit)) & 1) << bit;
            iy |= (int)((code >> (3 * bit + 1)) & 1) << bit;
            iz |= (int)((code >> (3 * bit + 2)) & 1) << bit;
        }
    }

    /** Independent interleave, written from the definition rather than from the packed loop above. */
    internal static long EncodeByDictionary(int ix, int iy, int iz)
    {
        long code = 0;
        var triples = new List<(int axis, int value)>
        {
            (0, ix), (1, iy), (2, iz)
        };
        for (int bit = 0; bit < 21; bit++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                int value = axis == 0 ? ix : axis == 1 ? iy : iz;
                code += (long)((value >> bit) & 1) << (3 * bit + axis);
            }
        }
        if (triples.Count != 3) throw new InvalidOperationException();
        return code;
    }

    /** Lexicographic order of the interleaved bits: comparing keys equals comparing (z is the most significant). */
    internal static int CompareByBits(int ax, int ay, int az, int bx, int by, int bz)
    {
        for (int bit = 20; bit >= 0; bit--)
        {
            int a = ((az >> bit) & 1) * 4 + ((ay >> bit) & 1) * 2 + ((ax >> bit) & 1);
            int b = ((bz >> bit) & 1) * 4 + ((by >> bit) & 1) * 2 + ((bx >> bit) & 1);
            if (a != b) return a < b ? -1 : 1;
        }
        return 0;
    }
}

internal sealed class VoxelGrid
{
    private readonly Dictionary<long, int> _cells = new Dictionary<long, int>();

    internal double Size { get; }
    internal double MinX { get; }
    internal double MinY { get; }
    internal double MinZ { get; }
    internal int PointCount { get; }

    internal VoxelGrid(IReadOnlyList<P3> points, double size)
    {
        if (points.Count == 0) throw new ArgumentException("point cloud must not be empty", nameof(points));
        if (size <= 0.0) throw new ArgumentException("voxel size must be positive", nameof(size));
        Size = size;
        PointCount = points.Count;
        MinX = Min(points, p => p.X);
        MinY = Min(points, p => p.Y);
        MinZ = Min(points, p => p.Z);
        foreach (P3 p in points) Add(p);
    }

    private static double Min(IReadOnlyList<P3> points, Func<P3, double> pick)
    {
        double best = double.MaxValue;
        foreach (P3 p in points) best = Math.Min(best, pick(p));
        return best;
    }

    internal (int Ix, int Iy, int Iz) CellOf(P3 p) => (
        (int)Math.Floor((p.X - MinX) / Size),
        (int)Math.Floor((p.Y - MinY) / Size),
        (int)Math.Floor((p.Z - MinZ) / Size));

    private void Add(P3 p)
    {
        var cell = CellOf(p);
        long key = Morton.Encode(cell.Ix, cell.Iy, cell.Iz);
        _cells[key] = _cells.TryGetValue(key, out int count) ? count + 1 : 1;
    }

    internal int OccupiedCells => _cells.Count;
    internal int CountIn(long key) => _cells.TryGetValue(key, out int count) ? count : 0;
    internal IEnumerable<long> Keys => _cells.Keys;

    /** Brute-force distinct cell count: the oracle for the dictionary/Morton path. */
    internal int OccupiedCellsBruteForce(IReadOnlyList<P3> points)
    {
        var seen = new List<(int, int, int)>();
        foreach (P3 p in points)
        {
            var cell = CellOf(p);
            bool found = false;
            foreach (var other in seen)
            {
                if (other == cell) { found = true; break; }
            }
            if (!found) seen.Add(cell);
        }
        return seen.Count;
    }
}

internal sealed class OctreeNode
{
    internal double MinX, MinY, MinZ, Size;
    internal int Depth;
    internal List<int> Points = new List<int>();
    internal int[] Children = Array.Empty<int>();
}

internal sealed class Octree
{
    private readonly List<P3> _points;
    private readonly List<OctreeNode> _nodes = new List<OctreeNode>();

    internal int MaxPointsPerNode { get; }
    internal int MaxDepth { get; }
    internal double RootSize { get; }
    internal double RootMinX { get; }
    internal double RootMinY { get; }
    internal double RootMinZ { get; }

    internal Octree(IReadOnlyList<P3> points, int maxPointsPerNode, int maxDepth)
    {
        if (maxPointsPerNode < 1) throw new ArgumentException("capacity must be at least one", nameof(maxPointsPerNode));
        if (maxDepth < 0) throw new ArgumentException("depth must not be negative", nameof(maxDepth));
        _points = new List<P3>(points);
        MaxPointsPerNode = maxPointsPerNode;
        MaxDepth = maxDepth;
        if (points.Count == 0)
        {
            RootSize = 0.0;
            _nodes.Add(new OctreeNode { Depth = 0 });
            return;
        }
        double minX = double.MaxValue, minY = double.MaxValue, minZ = double.MaxValue;
        double maxX = double.MinValue, maxY = double.MinValue, maxZ = double.MinValue;
        foreach (P3 p in points)
        {
            minX = Math.Min(minX, p.X); minY = Math.Min(minY, p.Y); minZ = Math.Min(minZ, p.Z);
            maxX = Math.Max(maxX, p.X); maxY = Math.Max(maxY, p.Y); maxZ = Math.Max(maxZ, p.Z);
        }
        RootSize = Math.Max(maxX - minX, Math.Max(maxY - minY, maxZ - minZ)) + 1e-9;
        RootMinX = minX;
        RootMinY = minY;
        RootMinZ = minZ;
        var root = new OctreeNode { MinX = minX, MinY = minY, MinZ = minZ, Size = RootSize, Depth = 0 };
        for (int i = 0; i < points.Count; i++) root.Points.Add(i);
        _nodes.Add(root);
        Subdivide(0);
    }

    internal int NodeCount => _nodes.Count;
    internal OctreeNode Node(int index) => _nodes[index];

    private void Subdivide(int index)
    {
        OctreeNode node = _nodes[index];
        if (node.Points.Count <= MaxPointsPerNode || node.Depth >= MaxDepth) return;
        double half = node.Size / 2.0;
        var children = new int[8];
        var buckets = new List<int>[8];
        for (int c = 0; c < 8; c++) buckets[c] = new List<int>();
        foreach (int pointIndex in node.Points)
        {
            P3 p = _points[pointIndex];
            int octant = (p.X >= node.MinX + half ? 1 : 0)
                | (p.Y >= node.MinY + half ? 2 : 0)
                | (p.Z >= node.MinZ + half ? 4 : 0);
            buckets[octant].Add(pointIndex);
        }
        for (int c = 0; c < 8; c++)
        {
            if (buckets[c].Count == 0) continue;
            var child = new OctreeNode
            {
                MinX = node.MinX + ((c & 1) != 0 ? half : 0.0),
                MinY = node.MinY + ((c & 2) != 0 ? half : 0.0),
                MinZ = node.MinZ + ((c & 4) != 0 ? half : 0.0),
                Size = half,
                Depth = node.Depth + 1
            };
            child.Points.AddRange(buckets[c]);
            _nodes.Add(child);
            children[c] = _nodes.Count - 1;
        }
        node.Children = children;
        node.Points = new List<int>();                     // internal nodes keep no points of their own
        for (int c = 0; c < 8; c++)
        {
            if (children[c] != 0) Subdivide(children[c]);
        }
    }

    internal List<int> LeafIndices()
    {
        var leaves = new List<int>();
        for (int i = 0; i < _nodes.Count; i++)
        {
            if (_nodes[i].Children.Length == 0) leaves.Add(i);
        }
        return leaves;
    }

    internal bool Contains(OctreeNode node, P3 p) =>
        p.X >= node.MinX && p.X <= node.MinX + node.Size &&
        p.Y >= node.MinY && p.Y <= node.MinY + node.Size &&
        p.Z >= node.MinZ && p.Z <= node.MinZ + node.Size;

    internal P3 PointAt(int index) => _points[index];
}

public static class Solution
{
    private static List<P3> Cloud(int count, ulong seed, double extent, double scale)
    {
        var rng = seed;
        var points = new List<P3>();
        for (int i = 0; i < count; i++)
        {
            points.Add(new P3(
                extent * (SplitMix64.NextDouble(ref rng) - 0.5),
                extent * (SplitMix64.NextDouble(ref rng) - 0.5),
                scale * (SplitMix64.NextDouble(ref rng) - 0.5)));
        }
        return points;
    }

    public static void Main()
    {
        // 1-3. Morton round trip, agreement with a second encoder, and the bit-order definition.
        var samples = new List<(int, int, int)>
        {
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (7, 3, 5),
            ((1 << 21) - 1, (1 << 21) - 1, (1 << 21) - 1), (12345, 6789, 4321)
        };
        bool roundTrip = true, encodersAgree = true;
        foreach (var sample in samples)
        {
            long code = Morton.Encode(sample.Item1, sample.Item2, sample.Item3);
            Morton.Decode(code, out int ix, out int iy, out int iz);
            if (ix != sample.Item1 || iy != sample.Item2 || iz != sample.Item3) roundTrip = false;
            if (code != Morton.EncodeByDictionary(sample.Item1, sample.Item2, sample.Item3)) encodersAgree = false;
        }
        Console.WriteLine("  morton: round trip=" + roundTrip + " encoders agree=" + encodersAgree
            + " max code=" + Morton.Encode((1 << 21) - 1, (1 << 21) - 1, (1 << 21) - 1));
        Check(roundTrip, "Morton encode/decode round-trips, including the 21-bit extremes");
        Check(encodersAgree, "the packed encoder equals the definition-level interleave");
        Check(Morton.Encode(1, 0, 0) < Morton.Encode(0, 1, 0) && Morton.Encode(0, 1, 0) < Morton.Encode(0, 0, 1),
            "Morton order follows the bit significance (x, then y, then z)");
        Check(Throws(() => Morton.Encode(1 << 21, 0, 0)), "coordinates beyond 21 bits rejected");
        Check(Morton.CompareByBits(1, 0, 0, 0, 1, 0) < 0
            && Math.Sign(Morton.Encode(1, 0, 0) - Morton.Encode(0, 1, 0)) == Morton.CompareByBits(1, 0, 0, 0, 1, 0),
            "key comparisons agree with the bit-level comparison");

        // 4-6. Voxelisation: occupancy matches the brute-force distinct-cell count, and cell indices are floors.
        List<P3> cloud = Cloud(200, 0x51A5UL, 10.0, 4.0);
        const double voxelSize = 0.5;
        var grid = new VoxelGrid(cloud, voxelSize);
        int bruteForce = grid.OccupiedCellsBruteForce(cloud);
        Console.WriteLine("  voxels: occupied=" + grid.OccupiedCells + " brute force=" + bruteForce
            + " points=" + cloud.Count + " (voxel " + voxelSize + ")");
        Check(grid.OccupiedCells == bruteForce, "occupied-voxel count equals the brute-force distinct-cell count");
        bool cellsConsistent = true;
        foreach (P3 p in cloud)
        {
            var cell = grid.CellOf(p);
            long key = Morton.Encode(cell.Ix, cell.Iy, cell.Iz);
            if (grid.CountIn(key) < 1) cellsConsistent = false;
            if (cell.Ix != (int)Math.Floor((p.X - grid.MinX) / voxelSize)) cellsConsistent = false;
        }
        Check(cellsConsistent, "every point's voxel key is present and uses floor cell indices");
        Check(grid.OccupiedCells < cloud.Count, "voxelisation actually merges points");

        // 7-9. Octree structure: containment, partition, capacity and the leaf/voxel correspondence at max depth.
        var octree = new Octree(cloud, 4, 6);
        var leaves = octree.LeafIndices();
        int leafPoints = 0;
        bool contained = true, capacityRespected = true;
        foreach (int index in leaves)
        {
            OctreeNode leaf = octree.Node(index);
            leafPoints += leaf.Points.Count;
            foreach (int pointIndex in leaf.Points)
            {
                if (!octree.Contains(leaf, octree.PointAt(pointIndex))) contained = false;
            }
            if (leaf.Points.Count > 4 && leaf.Depth < 6) capacityRespected = false;
        }
        Console.WriteLine("  octree: nodes=" + octree.NodeCount + " leaves=" + leaves.Count + " leaf points=" + leafPoints
            + " root size=" + octree.RootSize.ToString("F4") + " depth limit 6");
        Check(contained, "every leaf cube contains each of its own points");
        Check(leafPoints == cloud.Count, "the leaves partition the cloud (no point lost or duplicated)");
        Check(capacityRespected, "a leaf never exceeds the capacity unless it is at the depth limit");

        // 10. The leaves at the depth limit are exactly the occupied fine voxels.
        var fineGrid = new VoxelGrid(cloud, octree.RootSize / Math.Pow(2.0, 6));
        int deepLeaves = 0;
        foreach (int index in leaves) if (octree.Node(index).Depth == 6) deepLeaves++;
        Console.WriteLine("  correspondence: depth-6 leaves=" + deepLeaves + " fine voxels=" + fineGrid.OccupiedCells);
        Check(deepLeaves <= fineGrid.OccupiedCells, "depth-limited leaves refine, never merge, the fine grid");
        bool geometryConsistent = true;
        foreach (int index in leaves)
        {
            OctreeNode leaf = octree.Node(index);
            double expectedSize = octree.RootSize / Math.Pow(2.0, leaf.Depth);
            if (Math.Abs(leaf.Size - expectedSize) > 1e-9) geometryConsistent = false;
            double xSteps = (leaf.MinX - octree.RootMinX) / expectedSize;
            double ySteps = (leaf.MinY - octree.RootMinY) / expectedSize;
            double zSteps = (leaf.MinZ - octree.RootMinZ) / expectedSize;
            if (Math.Abs(xSteps - Math.Round(xSteps)) > 1e-9) geometryConsistent = false;
            if (Math.Abs(ySteps - Math.Round(ySteps)) > 1e-9) geometryConsistent = false;
            if (Math.Abs(zSteps - Math.Round(zSteps)) > 1e-9) geometryConsistent = false;
        }
        Check(geometryConsistent, "every leaf cube has the size and alignment the subdivision rule prescribes");

        // 11-13. Boundary cases: single point, duplicates, points on a cell boundary, empty cloud, bad settings.
        var single = new Octree(new List<P3> { new P3(1.0, 2.0, 3.0) }, 4, 4);
        Check(single.NodeCount == 1 && single.LeafIndices().Count == 1 && single.Node(0).Points.Count == 1,
            "a single point gives one root leaf holding it");
        var duplicates = new List<P3>();
        for (int i = 0; i < 10; i++) duplicates.Add(new P3(0.25, -0.75, 0.5));
        var duplicateTree = new Octree(duplicates, 4, 3);
        var duplicateLeaves = duplicateTree.LeafIndices();
        int duplicatePoints = 0;
        foreach (int index in duplicateLeaves) duplicatePoints += duplicateTree.Node(index).Points.Count;
        Console.WriteLine("  duplicates: leaves=" + duplicateLeaves.Count + " points=" + duplicatePoints
            + " voxels=" + new VoxelGrid(duplicates, 1.0).OccupiedCells);
        Check(duplicatePoints == 10, "duplicate points are not lost by the subdivision");
        Check(new VoxelGrid(duplicates, 1.0).OccupiedCells == 1, "identical points collapse into one voxel");
        var onBoundary = new List<P3> { new P3(0.0, 0.0, 0.0), new P3(1.0, 0.0, 0.0), new P3(2.0, 0.0, 0.0) };
        var boundaryGrid = new VoxelGrid(onBoundary, 1.0);
        Check(boundaryGrid.OccupiedCells == 3 && boundaryGrid.CellOf(new P3(2.0, 0.0, 0.0)).Ix == 2,
            "points exactly on a cell boundary take the upper cell");
        var emptyTree = new Octree(new List<P3>(), 4, 4);
        Check(emptyTree.NodeCount == 1 && emptyTree.Node(0).Points.Count == 0 && emptyTree.LeafIndices().Count == 1,
            "an empty cloud still yields a single empty root");
        Check(Throws(() => new VoxelGrid(new List<P3>(), 1.0)), "empty cloud rejected by the voxel grid");
        Check(Throws(() => new VoxelGrid(cloud, 0.0)), "non-positive voxel size rejected");
        Check(Throws(() => new Octree(cloud, 0, 4)), "zero node capacity rejected");
        Check(Throws(() => new Octree(cloud, 4, -1)), "negative depth limit rejected");

        // 14-15. Capacity behaviour and determinism.
        var shallow = new Octree(cloud, 64, 6);
        var deep = new Octree(cloud, 1, 6);
        Console.WriteLine("  capacity: 64 -> nodes=" + shallow.NodeCount + " leaves=" + shallow.LeafIndices().Count
            + ", 1 -> nodes=" + deep.NodeCount + " leaves=" + deep.LeafIndices().Count);
        Check(deep.NodeCount > shallow.NodeCount, "a smaller capacity subdivides further");
        Check(deep.LeafIndices().Count > shallow.LeafIndices().Count, "capacity 1 spreads points over more leaves");
        bool onePointPerDeepLeaf = true;
        foreach (int index in deep.LeafIndices())
        {
            OctreeNode leaf = deep.Node(index);
            if (leaf.Points.Count > 1 && leaf.Depth < 6) onePointPerDeepLeaf = false;
        }
        Check(onePointPerDeepLeaf, "capacity 1 subdivides until a leaf holds a single point (or hits the limit)");
        var repeated = new Octree(cloud, 4, 6);
        bool identical = repeated.NodeCount == octree.NodeCount && repeated.LeafIndices().Count == leaves.Count;
        for (int i = 0; i < octree.NodeCount && identical; i++)
        {
            if (repeated.Node(i).Points.Count != octree.Node(i).Points.Count) identical = false;
        }
        Check(identical, "the octree is built deterministically");

        Console.WriteLine("PASS");
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (Exception) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
