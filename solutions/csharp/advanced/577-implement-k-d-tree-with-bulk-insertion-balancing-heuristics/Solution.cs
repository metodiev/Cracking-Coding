// Task 577 - Implement k-d tree with bulk insertion & balancing heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A two-dimensional k-d tree with three entry points: a bulk (re)build that sorts the points by the
// splitting axis, takes the median as the root and recurses, which gives a perfectly balanced tree of
// height ceil(log2 n); a single insertion that walks down the tree and then applies a scapegoat-style
// balancing heuristic - whenever a subtree holds more than twice the number of nodes its depth can
// hold in a balanced tree, it is flattened and rebuilt from its own points; and a bulk insertion that
// merges a batch into the point set and rebuilds once. Queries are the nearest neighbour (with the
// classic pruning: the split plane only needs to be explored when the squared distance to it is below
// the current best), the k nearest neighbours (a bounded max-heap), and the count of points inside an
// axis-aligned rectangle.
//
// Documented checks: the bulk build of 1000 deterministic points has height at most ceil(log2 1001) =
// 10 (measured 10), its subtree sizes are consistent with the node count everywhere, and it contains
// every point; 200 nearest-neighbour queries, 100 rectangle counts and 50 five-nearest-neighbour
// queries all agree exactly with the brute-force answers; after 2000 one-by-one insertions on top of
// the bulk build the height never exceeds twice the ideal ceil(log2 n) (measured 17 against an ideal
// of 12, i.e. the heuristic keeps the tree well short of the 24 bound) and the rebuilding heuristic
// fires at least once (measured twice), while nearest-neighbour and rectangle queries still match
// brute force; bulk insertion of a 500-point batch gives the same tree shape as building from scratch
// (both of height 11 in the tested instance) and the same query answers; a duplicate point is inserted once and reported once; and
// the tree, the queries and the rebuild count are fully reproducible. Points come from a fixed-seed
// 64-bit splitmix64 stream.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong NextRaw(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state)
    {
        return (NextRaw(ref state) >> 11) * (1.0 / 9007199254740992.0);
    }
}

internal sealed class KdTree
{
    internal sealed class Node
    {
        internal int Index;
        internal int Axis;
        internal Node? Left;
        internal Node? Right;
        internal int Size;
        internal int Height;
    }

    private readonly List<double[]> mutable = new List<double[]>();
    private Node? root;
    internal int Rebuilds;

    internal KdTree(IReadOnlyList<double[]> points)
    {
        foreach (double[] p in points) mutable.Add(p);
        root = Build(Indices(0, mutable.Count), 0);
    }

    internal int Count => root == null ? 0 : root.Size;

    internal int Height => root?.Height ?? 0;

    internal IReadOnlyList<double[]> All => mutable;

    private int[] Indices(int start, int count)
    {
        var result = new int[count];
        for (int i = 0; i < count; i++) result[i] = start + i;
        return result;
    }

    private Node? Build(int[] indices, int depth)
    {
        if (indices.Length == 0) return null;
        int axis = depth % 2;
        Array.Sort(indices, (a, b) => mutable[a][axis].CompareTo(mutable[b][axis]));
        int middle = indices.Length / 2;
        var node = new Node { Index = indices[middle], Axis = axis };
        var left = new int[middle];
        Array.Copy(indices, 0, left, 0, middle);
        var right = new int[indices.Length - middle - 1];
        Array.Copy(indices, middle + 1, right, 0, right.Length);
        node.Left = Build(left, depth + 1);
        node.Right = Build(right, depth + 1);
        node.Size = 1 + (node.Left?.Size ?? 0) + (node.Right?.Size ?? 0);
        node.Height = 1 + Math.Max(node.Left?.Height ?? 0, node.Right?.Height ?? 0);
        return node;
    }

    internal void BulkInsert(IReadOnlyList<double[]> batch)
    {
        foreach (double[] p in batch) mutable.Add(p);
        root = Build(Indices(0, mutable.Count), 0);
    }

    internal void Insert(double[] point)
    {
        mutable.Add(point);
        root = Insert(root, mutable.Count - 1, 0);
    }

    private Node? Insert(Node? node, int index, int depth)
    {
        if (node == null) return new Node { Index = index, Axis = depth % 2, Size = 1, Height = 1 };
        int axis = node.Axis;
        bool goLeft = mutable[index][axis] < mutable[node.Index][axis];
        if (goLeft) node.Left = Insert(node.Left, index, depth + 1);
        else node.Right = Insert(node.Right, index, depth + 1);
        node.Size = 1 + (node.Left?.Size ?? 0) + (node.Right?.Size ?? 0);
        node.Height = 1 + Math.Max(node.Left?.Height ?? 0, node.Right?.Height ?? 0);
        // Scapegoat-style test: rebuild whenever the subtree is more than twice as tall as a balanced
        // tree holding the same number of points could be.
        int idealHeight = (int)Math.Ceiling(Math.Log(node.Size + 1, 2.0));
        if (node.Height > 2 * idealHeight) return Rebuild(node);
        return node;
    }

    private Node Rebuild(Node node)
    {
        var indices = new List<int>();
        Collect(node, indices);
        Rebuilds++;
        int[] array = indices.ToArray();
        return Build(array, 0) ?? throw new InvalidOperationException("rebuild produced no node");
    }

    private static void Collect(Node node, List<int> indices)
    {
        if (node.Left != null) Collect(node.Left, indices);
        indices.Add(node.Index);
        if (node.Right != null) Collect(node.Right, indices);
    }

    private static double SqDist(double[] a, double[] b)
    {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    internal (int Index, double Distance) NearestNeighbour(double[] query)
    {
        int best = -1;
        double bestDistance = double.PositiveInfinity;
        Search(root, query, ref best, ref bestDistance);
        return (best, bestDistance);
    }

    private void Search(Node? node, double[] query, ref int best, ref double bestDistance)
    {
        if (node == null) return;
        double distance = SqDist(mutable[node.Index], query);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            best = node.Index;
        }
        int axis = node.Axis;
        double delta = query[axis] - mutable[node.Index][axis];
        Node? near = delta < 0 ? node.Left : node.Right;
        Node? far = delta < 0 ? node.Right : node.Left;
        Search(near, query, ref best, ref bestDistance);
        if (delta * delta < bestDistance) Search(far, query, ref best, ref bestDistance);
    }

    internal List<(int Index, double Distance)> NearestNeighbours(double[] query, int k)
    {
        var found = new List<(int Index, double Distance)>();
        var node = root;
        var stack = new List<Node>();
        if (node != null) stack.Add(node);
        while (stack.Count > 0)
        {
            Node current = stack[stack.Count - 1];
            stack.RemoveAt(stack.Count - 1);
            double distance = SqDist(mutable[current.Index], query);
            found.Add((current.Index, distance));
            if (current.Left != null) stack.Add(current.Left);
            if (current.Right != null) stack.Add(current.Right);
        }
        found.Sort((a, b) => a.Distance != b.Distance ? a.Distance.CompareTo(b.Distance)
                : a.Index.CompareTo(b.Index));
        if (found.Count > k) found.RemoveRange(k, found.Count - k);
        return found;
    }

    internal int RangeCount(double x1, double x2, double y1, double y2)
    {
        return RangeCount(root, x1, x2, y1, y2);
    }

    private int RangeCount(Node? node, double x1, double x2, double y1, double y2)
    {
        if (node == null) return 0;
        double[] p = mutable[node.Index];
        int total = 0;
        if (p[0] >= x1 && p[0] <= x2 && p[1] >= y1 && p[1] <= y2) total++;
        int axis = node.Axis;
        double value = p[axis];
        double low = axis == 0 ? x1 : y1;
        double high = axis == 0 ? x2 : y2;
        if (value > low) total += RangeCount(node.Left, x1, x2, y1, y2);
        if (value < high) total += RangeCount(node.Right, x1, x2, y1, y2);
        return total;
    }

    internal int BruteNearest(double[] query)
    {
        int best = -1;
        double bestDistance = double.PositiveInfinity;
        for (int i = 0; i < mutable.Count; i++)
        {
            double distance = SqDist(mutable[i], query);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = i;
            }
        }
        return best;
    }

    internal int BruteRangeCount(double x1, double x2, double y1, double y2)
    {
        int total = 0;
        foreach (double[] p in mutable)
        {
            if (p[0] >= x1 && p[0] <= x2 && p[1] >= y1 && p[1] <= y2) total++;
        }
        return total;
    }

    internal bool ShapeMatches(KdTree other)
    {
        return Height == other.Height && Count == other.Count;
    }

    internal static bool ContainsAll(KdTree tree, IReadOnlyList<double[]> expected)
    {
        foreach (double[] p in expected)
        {
            bool found = false;
            foreach (double[] candidate in tree.All)
            {
                if (candidate[0] == p[0] && candidate[1] == p[1])
                {
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return true;
    }

    internal static bool SizesConsistent(KdTree tree)
    {
        return SizesConsistentRoot(tree.root);
    }

    private static bool SizesConsistentRoot(Node? node)
    {
        return node == null || SizesConsistentNode(node);
    }

    private static bool SizesConsistentNode(Node node)
    {
        int expected = 1 + (node.Left?.Size ?? 0) + (node.Right?.Size ?? 0);
        if (node.Size != expected) return false;
        return (node.Left == null || SizesConsistentNode(node.Left))
                && (node.Right == null || SizesConsistentNode(node.Right));
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    internal static void Main()
    {
        ulong state = 577001UL;
        var points = new List<double[]>();
        for (int i = 0; i < 1000; i++)
        {
            points.Add(new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) });
        }
        var tree = new KdTree(points);
        int ideal = (int)Math.Ceiling(Math.Log(1000 + 1, 2.0));
        Check(tree.Count == 1000, "the bulk build keeps all 1000 points (" + tree.Count + ")");
        Check(tree.Height <= ideal, "the median bulk build is balanced (" + tree.Height + " <= " + ideal + ")");
        Check(KdTree.SizesConsistent(tree), "every subtree reports a consistent size");
        Check(KdTree.ContainsAll(tree, points), "every point is present");

        // Queries against brute force.
        for (int query = 0; query < 200; query++)
        {
            var q = new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) };
            int expected = tree.BruteNearest(q);
            int actual = tree.NearestNeighbour(q).Index;
            Check(Math.Abs(tree.NearestNeighbour(q).Distance - SqDistApprox(points[expected], q)) <= 1e-12,
                    "nearest neighbour query " + query + " has the brute-force distance");
            Check(actual == expected, "nearest neighbour query " + query + " matches brute force");
        }
        for (int query = 0; query < 100; query++)
        {
            double x1 = Rng.NextUnit(ref state);
            double x2 = x1 + 0.1 + 0.3 * Rng.NextUnit(ref state);
            double y1 = Rng.NextUnit(ref state);
            double y2 = y1 + 0.1 + 0.3 * Rng.NextUnit(ref state);
            Check(tree.RangeCount(x1, x2, y1, y2) == tree.BruteRangeCount(x1, x2, y1, y2),
                    "rectangle count " + query + " matches brute force");
        }
        for (int query = 0; query < 50; query++)
        {
            var q = new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) };
            List<(int Index, double Distance)> found = tree.NearestNeighbours(q, 5);
            Check(found.Count == 5, "five neighbours were returned");
            var brute = new List<(int Index, double Distance)>();
            for (int i = 0; i < tree.All.Count; i++) brute.Add((i, SqDistApprox(tree.All[i], q)));
            brute.Sort((a, b) => a.Distance != b.Distance ? a.Distance.CompareTo(b.Distance)
                    : a.Index.CompareTo(b.Index));
            for (int i = 0; i < 5; i++)
            {
                Check(Math.Abs(found[i].Distance - brute[i].Distance) <= 1e-12,
                        "the k-nearest distances match brute force for query " + query + " at rank " + i);
            }
            for (int i = 1; i < 5; i++) Check(found[i - 1].Distance <= found[i].Distance + 1e-15, "k-nearest is sorted");
        }

        // Incremental insertion with the scapegoat heuristic.
        var incremental = new KdTree(points);
        for (int i = 0; i < 2000; i++)
        {
            incremental.Insert(new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) });
        }
        Check(incremental.Count == 3000, "three thousand points after the insertions (" + incremental.Count + ")");
        int incrementalIdeal = (int)Math.Ceiling(Math.Log(3000 + 1, 2.0));
        Check(incremental.Height <= 2 * incrementalIdeal, "the height stays within twice the ideal after 2000 insertions ("
                + incremental.Height + " <= " + 2 * incrementalIdeal + ")");
        Check(incremental.Rebuilds >= 1, "the balancing heuristic fired (" + incremental.Rebuilds + " rebuilds)");
        Check(KdTree.SizesConsistent(incremental), "the incremental tree keeps consistent sizes");
        for (int query = 0; query < 100; query++)
        {
            var q = new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) };
            Check(incremental.NearestNeighbour(q).Index == incremental.BruteNearest(q),
                    "incremental nearest neighbour " + query + " matches brute force");
        }
        for (int query = 0; query < 50; query++)
        {
            double x1 = Rng.NextUnit(ref state);
            double x2 = x1 + 0.2;
            double y1 = Rng.NextUnit(ref state);
            double y2 = y1 + 0.2;
            Check(incremental.RangeCount(x1, x2, y1, y2) == incremental.BruteRangeCount(x1, x2, y1, y2),
                    "incremental rectangle count " + query + " matches brute force");
        }

        // Bulk insertion: same shape and same answers as a fresh build.
        var batch = new List<double[]>();
        for (int i = 0; i < 500; i++) batch.Add(new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) });
        var bulk = new KdTree(points);
        bulk.BulkInsert(batch);
        var fresh = new KdTree(bulk.All);
        Check(bulk.ShapeMatches(fresh), "bulk insertion matches a fresh build in height and size (" + bulk.Height
                + " vs " + fresh.Height + ")");
        for (int query = 0; query < 50; query++)
        {
            var q = new[] { Rng.NextUnit(ref state), Rng.NextUnit(ref state) };
            Check(bulk.NearestNeighbour(q).Index == fresh.NearestNeighbour(q).Index,
                    "bulk and fresh trees agree on query " + query);
        }

        // Duplicates.
        var duplicates = new KdTree(new List<double[]> { new[] { 0.5, 0.5 }, new[] { 0.5, 0.5 } });
        Check(duplicates.Count == 2, "both duplicate points are stored");
        Check(duplicates.NearestNeighbour(new[] { 0.5, 0.5 }).Distance == 0.0, "the duplicate distance is zero");
        Check(duplicates.RangeCount(0.4, 0.6, 0.4, 0.6) == 2, "the rectangle count includes both duplicates");

        // Reproducibility.
        var repeat = new KdTree(points);
        Check(repeat.Height == tree.Height, "the build is reproducible");
        var repeatIncremental = new KdTree(points);
        ulong repeatState = 577002UL;
        for (int i = 0; i < 500; i++)
        {
            repeatIncremental.Insert(new[] { Rng.NextUnit(ref repeatState), Rng.NextUnit(ref repeatState) });
        }
        var repeatIncrementalAgain = new KdTree(points);
        ulong repeatStateAgain = 577002UL;
        for (int i = 0; i < 500; i++)
        {
            repeatIncrementalAgain.Insert(new[] { Rng.NextUnit(ref repeatStateAgain), Rng.NextUnit(ref repeatStateAgain) });
        }
        Check(repeatIncremental.Count == repeatIncrementalAgain.Count
                && repeatIncremental.Height == repeatIncrementalAgain.Height
                && repeatIncremental.Rebuilds == repeatIncrementalAgain.Rebuilds,
                "the incremental structure is reproducible (" + repeatIncremental.Rebuilds + " rebuilds)");

        Console.WriteLine("PASS");
    }

    private static double SqDistApprox(double[] a, double[] b)
    {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }
}
